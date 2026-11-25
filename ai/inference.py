"""Inférence : charge le modèle entraîné et analyse une image prétraitée.

Sortie de predict() = exactement le format du contrat de données (docs/CONTRAT_DONNEES.md).
Les paramètres de prétraitement (taille, moyenne, écart-type) sont lus DANS le fichier du modèle :
ainsi, l'analyse utilise toujours les mêmes réglages que l'entraînement (reproductibilité).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ai.explainability import DISCLAIMER, METHOD, GradCAM, summarize_cam
from ai.preprocessing import PreprocessConfig

DEFAULT_MODEL_PATH = Path("ai/models/neurotrace_model.pt")
REQUIRED_KEYS = ("model_id", "model_version", "arch", "class_names", "img_size", "mean", "std", "state_dict")


class ModelLoadError(Exception):
    """Le fichier du modèle est absent ou invalide."""


class NeuroTraceModel:
    def __init__(self, path=DEFAULT_MODEL_PATH, device: str = "cpu"):
        import torch
        from torchvision import models

        self._torch = torch
        self.device = torch.device(device)
        p = Path(path)
        if not p.is_file():
            raise ModelLoadError(f"Fichier modèle introuvable : {p}")
        try:
            ckpt = torch.load(p, map_location="cpu")
        except Exception as e:  # fichier corrompu, ou format refusé par la protection de torch.load
            raise ModelLoadError(f"Impossible de lire le modèle : {type(e).__name__}") from None
        missing = [k for k in REQUIRED_KEYS if k not in ckpt]
        if missing:
            raise ModelLoadError(f"Fichier modèle incomplet, clés manquantes : {missing}")
        if ckpt["arch"] != "resnet18":
            raise ModelLoadError(f"Architecture non supportée : {ckpt['arch']}")

        self.model_id = ckpt["model_id"]
        self.model_version = ckpt["model_version"]
        self.class_names = list(ckpt["class_names"])
        self.img_size = int(ckpt["img_size"])
        self.mean = tuple(ckpt["mean"])
        self.std = tuple(ckpt["std"])
        self.test_metrics = ckpt.get("test_metrics", {})
        self.training_info = ckpt.get("training", {})

        net = models.resnet18(weights=None)
        net.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(net.fc.in_features, len(self.class_names)))
        try:
            net.load_state_dict(ckpt["state_dict"])
        except RuntimeError:
            raise ModelLoadError("Les poids ne correspondent pas à l'architecture attendue.") from None
        self.net = net.to(self.device).eval()
        self._gradcam = GradCAM(self.net, self.net.layer4[-1])

    # ------------------------------------------------------------ prétraitement cohérent

    def preprocess_config(self, **options) -> PreprocessConfig:
        """Configuration de prétraitement alignée sur l'entraînement de CE modèle."""
        return PreprocessConfig(img_size=self.img_size, mean=self.mean, std=self.std, **options)

    # ------------------------------------------------------------ prédiction

    def _to_batch(self, tensor: np.ndarray):
        arr = np.asarray(tensor, dtype=np.float32)
        expected = (3, self.img_size, self.img_size)
        if arr.shape != expected:
            raise ValueError(f"Forme d'entrée invalide : {arr.shape}, attendu {expected}")
        return self._torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def predict(self, tensor: np.ndarray, case_id: str | None = None) -> dict:
        """Retourne la prédiction au format du contrat de données."""
        torch = self._torch
        x = self._to_batch(tensor)
        with torch.no_grad():
            probs = torch.softmax(self.net(x), dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return {
            "case_id": case_id,
            "predicted_class": self.class_names[idx],
            "class_scores": {c: round(float(p), 4) for c, p in zip(self.class_names, probs)},
            "confidence": round(float(probs[idx]), 4),
            "model_id": self.model_id,
            "model_version": self.model_version,
        }

    def explain(self, tensor: np.ndarray, target_class: str | None = None, prediction: dict | None = None):
        """Grad-CAM pour la classe prédite (ou pour `target_class`).

        Retourne (carte de chaleur float32 (H, W) entre 0 et 1, dictionnaire d'explication).
        """
        x = self._to_batch(tensor)
        if prediction is None:
            prediction = self.predict(tensor)
        target = target_class or prediction["predicted_class"]
        if target not in self.class_names:
            raise ValueError(f"Classe inconnue : {target}")
        cam = self._gradcam(x, self.class_names.index(target))
        info = {
            "method": METHOD,
            "target_layer": "layer4[-1]",
            "target_class": target,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "cam_summary": summarize_cam(cam),
            "disclaimer": DISCLAIMER,
        }
        return cam, info


def load_model(path=DEFAULT_MODEL_PATH, device: str = "cpu") -> NeuroTraceModel:
    return NeuroTraceModel(path, device)


if __name__ == "__main__":
    # Essai complet : python -m ai.inference image.jpg [--out-dir samples]
    import argparse
    import json

    from PIL import Image

    from ai.explainability import overlay_heatmap
    from ai.preprocessing import InvalidImageError, preprocess_file, save_preprocessed_image
    from ai.validation import validate

    ap = argparse.ArgumentParser(description="Analyse d'une image IRM : prétraitement, IA, Grad-CAM")
    ap.add_argument("image")
    ap.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    try:
        model = load_model(args.model)
        pre = preprocess_file(args.image, model.preprocess_config())
    except (ModelLoadError, InvalidImageError) as e:
        print(json.dumps({"status": "ERROR", "error": str(e)}, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    pred = model.predict(pre.tensor, case_id="CASE_TEST")
    cam, explanation = model.explain(pre.tensor, prediction=pred)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_preprocessed_image(pre, out / "preprocessed.png")
    Image.fromarray(overlay_heatmap(pre.image_rgb, cam)).save(out / "gradcam.png")
    validation = validate(pred, pre.summary, explanation)
    print(json.dumps({"prediction": pred, "explanation": explanation, "validation": validation},
                     ensure_ascii=False, indent=2))
    print(f"\nImages enregistrées dans {out}/ : preprocessed.png et gradcam.png")
    print(f"DÉCISION : {validation['status']}")
    for reason in validation["reasons"]:
        print(" -", reason)
