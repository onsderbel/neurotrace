"""Module de prétraitement NeuroTrace.

Entrée  : un fichier image IRM (jpg, jpeg, png).
Sortie  : une image 224x224 prête pour l'IA, le tenseur normalisé, et un résumé des opérations.

IMPORTANT : le modèle v1.0.0 a été entraîné avec seulement « redimensionnement + normalisation ».
Les options crop / denoise / clahe sont donc DÉSACTIVÉES par défaut. Les activer sans réentraîner
le modèle avec le même prétraitement fait baisser la précision (voir le notebook de réentraînement).

Ce module n'a besoin ni de PyTorch ni de la blockchain : il utilise seulement NumPy, Pillow et OpenCV.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PREPROCESSING_VERSION = "1.0.0"
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")


class InvalidImageError(Exception):
    """Image refusée. `code` est un identifiant stable, utilisable par les agents et l'interface."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class PreprocessConfig:
    img_size: int = 224
    mean: tuple = (0.485, 0.456, 0.406)   # mêmes valeurs que pendant l'entraînement
    std: tuple = (0.229, 0.224, 0.225)
    # Contrôles de validité
    min_side: int = 64
    max_file_mb: float = 20.0
    min_contrast: float = 1.0             # écart-type minimal des niveaux de gris
    max_aspect_ratio: float = 4.0
    # Étapes optionnelles (désactivées par défaut, voir l'avertissement en haut du fichier)
    crop: bool = False
    crop_threshold: int = 15
    crop_margin: float = 0.03
    crop_min_area_ratio: float = 0.2
    denoise: bool = False
    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_tile: int = 8

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreprocessResult:
    image_rgb: np.ndarray   # uint8, (H, W, 3) : l'image prétraitée, à sauvegarder comme artefact
    tensor: np.ndarray      # float32, (3, H, W) : entrée du modèle
    summary: dict           # résumé des opérations (JSON-compatible)


# ---------------------------------------------------------------- chargement et validation

def load_image(path, config: PreprocessConfig | None = None):
    """Charge et valide un fichier image. Retourne (tableau RGB uint8, infos). Lève InvalidImageError."""
    cfg = config or PreprocessConfig()
    p = Path(path)

    if not p.is_file():
        raise InvalidImageError("FILE_NOT_FOUND", "Fichier introuvable.")
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise InvalidImageError("UNSUPPORTED_FORMAT",
                                f"Format non supporté. Formats acceptés : {', '.join(ALLOWED_EXTENSIONS)}.")
    size = p.stat().st_size
    if size == 0:
        raise InvalidImageError("EMPTY_FILE", "Le fichier est vide.")
    if size > cfg.max_file_mb * 1024 * 1024:
        raise InvalidImageError("FILE_TOO_LARGE", f"Fichier trop volumineux (max {cfg.max_file_mb} Mo).")

    try:
        with Image.open(p) as im:
            im.verify()                       # détecte les fichiers corrompus ou qui ne sont pas des images
        with Image.open(p) as im:
            fmt = im.format
            im.load()
            rgb = np.array(im.convert("RGB"))
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise InvalidImageError("CORRUPT_IMAGE", "Le fichier n'est pas une image valide ou est corrompu.") from None

    h, w = rgb.shape[:2]
    if min(h, w) < cfg.min_side:
        raise InvalidImageError("TOO_SMALL", f"Image trop petite ({w}x{h}, minimum {cfg.min_side} px de côté).")
    if max(h, w) / min(h, w) > cfg.max_aspect_ratio:
        raise InvalidImageError("BAD_ASPECT_RATIO", f"Proportions anormales ({w}x{h}).")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if float(gray.std()) < cfg.min_contrast:
        raise InvalidImageError("UNIFORM_IMAGE", "Image uniforme (aucun contenu exploitable).")

    return rgb, {"format": fmt, "width": w, "height": h, "file_size_bytes": size}


def quality_metrics(rgb: np.ndarray) -> dict:
    """Indicateurs simples de qualité, utilisés ensuite par le module de validation."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return {
        "mean_brightness": round(float(gray.mean()), 2),
        "contrast_std": round(float(gray.std()), 2),
        "sharpness_laplacian_var": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "dark_pixel_ratio": round(float((gray < 10).mean()), 4),
    }


# ---------------------------------------------------------------- étapes de traitement

def _crop_foreground(rgb: np.ndarray, cfg: PreprocessConfig):
    """Recadre autour de la zone non noire (la tête), avec une petite marge."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = (blur > cfg.crop_threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return rgb, {"applied": False, "reason": "aucune zone détectée"}
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    H, W = gray.shape
    if (w * h) / (W * H) < cfg.crop_min_area_ratio:
        return rgb, {"applied": False, "reason": "zone détectée trop petite"}
    m = int(cfg.crop_margin * max(W, H))
    x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
    return rgb[y0:y1, x0:x1], {"applied": True, "box": [x0, y0, x1, y1]}


def _denoise(rgb: np.ndarray) -> np.ndarray:
    """Réduction du bruit avec un filtre bilatéral (préserve les contours)."""
    return cv2.bilateralFilter(rgb, 5, 35, 35)


def _clahe(rgb: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Amélioration locale du contraste (CLAHE) sur la luminosité."""
    l, a, b = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB))
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_tile, cfg.clahe_tile))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)


def _resize(rgb: np.ndarray, size: int) -> np.ndarray:
    # Même méthode que torchvision.transforms.Resize pendant l'entraînement (PIL, bilinéaire)
    return np.array(Image.fromarray(rgb).resize((size, size), Image.BILINEAR))


def _normalize(image_rgb: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    arr = image_rgb.astype(np.float32) / 255.0
    mean = np.array(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(cfg.std, dtype=np.float32).reshape(1, 1, 3)
    return np.ascontiguousarray(((arr - mean) / std).transpose(2, 0, 1))


# ---------------------------------------------------------------- fonction principale

def preprocess_file(path, config: PreprocessConfig | None = None) -> PreprocessResult:
    """Pipeline complet. Lève InvalidImageError si l'image est invalide."""
    cfg = config or PreprocessConfig()
    rgb, info = load_image(path, cfg)
    quality = quality_metrics(rgb)

    steps = [{"name": "validate", "applied": True}]
    work = rgb

    if cfg.crop:
        work, crop_info = _crop_foreground(work, cfg)
        steps.append({"name": "crop", **crop_info})
    else:
        steps.append({"name": "crop", "applied": False, "reason": "désactivé"})

    if cfg.denoise:
        work = _denoise(work)
    steps.append({"name": "denoise", "applied": cfg.denoise})

    if cfg.clahe:
        work = _clahe(work, cfg)
    steps.append({"name": "clahe", "applied": cfg.clahe})

    image_rgb = _resize(work, cfg.img_size)
    steps.append({"name": "resize", "applied": True, "size": [cfg.img_size, cfg.img_size],
                  "interpolation": "bilinear"})

    tensor = _normalize(image_rgb, cfg)
    steps.append({"name": "normalize", "applied": True, "mean": list(cfg.mean), "std": list(cfg.std)})

    # Le nom du fichier n'est volontairement PAS enregistré (il pourrait contenir un identifiant de patient).
    summary = {
        "module": "preprocessing",
        "version": PREPROCESSING_VERSION,
        "status": "OK",
        "input": info,
        "quality": quality,
        "steps": steps,
        "output": {"width": cfg.img_size, "height": cfg.img_size, "channels": 3},
        "config": cfg.to_dict(),
    }
    return PreprocessResult(image_rgb=image_rgb, tensor=tensor, summary=summary)


def save_preprocessed_image(result: PreprocessResult, path) -> None:
    """Sauvegarde l'image prétraitée en PNG (sans perte : le hash SHA-256 reste stable)."""
    Image.fromarray(result.image_rgb).save(path, format="PNG")


if __name__ == "__main__":
    # Essai rapide : python -m ai.preprocessing image.png [--out sortie.png] [--crop] [--clahe] [--denoise]
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Prétraitement d'une image IRM")
    ap.add_argument("image")
    ap.add_argument("--out")
    ap.add_argument("--crop", action="store_true")
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--denoise", action="store_true")
    args = ap.parse_args()
    try:
        res = preprocess_file(args.image, PreprocessConfig(crop=args.crop, clahe=args.clahe, denoise=args.denoise))
    except InvalidImageError as e:
        print(json.dumps({"status": "REJECTED", **e.to_dict()}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    if args.out:
        save_preprocessed_image(res, args.out)
        print("Image prétraitée enregistrée :", args.out)
    print(json.dumps(res.summary, ensure_ascii=False, indent=2))
