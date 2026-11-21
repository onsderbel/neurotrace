"""Tests du module de prétraitement. Lancer depuis la racine du dépôt : pytest"""
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from ai.preprocessing import (
    InvalidImageError,
    PreprocessConfig,
    preprocess_file,
    save_preprocessed_image,
)


def make_mri(path, size=300, seed=0):
    """Fausse IRM : fond noir, ellipse grise (la « tête ») et un peu de bruit."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), np.uint8)
    cv2.ellipse(img, (size // 2, size // 2), (size // 3, size // 2 - 20), 0, 0, 360, 140, -1)
    img = np.clip(img.astype(int) + rng.integers(0, 15, img.shape), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    return path


@pytest.fixture
def mri(tmp_path):
    return make_mri(tmp_path / "mri.png")


# ---------- cas valides

def test_image_valide(mri):
    r = preprocess_file(mri)
    assert r.image_rgb.shape == (224, 224, 3) and r.image_rgb.dtype == np.uint8
    assert r.tensor.shape == (3, 224, 224) and r.tensor.dtype == np.float32
    assert r.summary["status"] == "OK"


def test_jpeg_accepte(tmp_path):
    p = tmp_path / "scan.jpg"
    Image.open(make_mri(tmp_path / "tmp.png")).convert("RGB").save(p, "JPEG")
    assert preprocess_file(p).summary["input"]["format"] == "JPEG"


def test_rgba_accepte(tmp_path):
    arr = np.array(Image.open(make_mri(tmp_path / "tmp.png")).convert("RGBA"))
    p = tmp_path / "rgba.png"
    Image.fromarray(arr, "RGBA").save(p)
    assert preprocess_file(p).image_rgb.shape == (224, 224, 3)


def test_pretraitement_par_defaut_identique_a_l_entrainement(mri):
    # Par défaut : seulement redimensionnement bilinéaire PIL + normalisation (comme dans le notebook)
    attendu = np.array(Image.open(mri).convert("RGB").resize((224, 224), Image.BILINEAR))
    assert np.array_equal(preprocess_file(mri).image_rgb, attendu)


def test_normalisation_correcte(mri):
    r = preprocess_file(mri)
    cfg = PreprocessConfig()
    ref = (r.image_rgb.astype(np.float32) / 255 - np.array(cfg.mean, np.float32)) / np.array(cfg.std, np.float32)
    assert np.allclose(r.tensor, ref.transpose(2, 0, 1), atol=1e-6)


def test_resultat_deterministe(mri, tmp_path):
    a, b = preprocess_file(mri), preprocess_file(mri)
    assert np.array_equal(a.tensor, b.tensor)
    save_preprocessed_image(a, tmp_path / "a.png")
    save_preprocessed_image(b, tmp_path / "b.png")
    # mêmes octets => même hash SHA-256 (indispensable pour la vérification d'intégrité)
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes()


def test_png_sauvegarde_relu_identique(mri, tmp_path):
    r = preprocess_file(mri)
    save_preprocessed_image(r, tmp_path / "pre.png")
    assert np.array_equal(np.array(Image.open(tmp_path / "pre.png")), r.image_rgb)


def test_resume_json_et_sans_nom_de_fichier(mri):
    s = preprocess_file(mri).summary
    texte = json.dumps(s)
    assert "mri.png" not in texte                       # pas d'identifiant potentiel dans le résumé
    assert {"mean_brightness", "contrast_std", "sharpness_laplacian_var", "dark_pixel_ratio"} <= set(s["quality"])
    assert [st["name"] for st in s["steps"]] == ["validate", "crop", "denoise", "clahe", "resize", "normalize"]


# ---------- étapes optionnelles

def test_options_actives(mri):
    cfg = PreprocessConfig(crop=True, denoise=True, clahe=True)
    r = preprocess_file(mri, cfg)
    steps = {s["name"]: s for s in r.summary["steps"]}
    assert steps["crop"]["applied"] and steps["denoise"]["applied"] and steps["clahe"]["applied"]
    assert r.image_rgb.shape == (224, 224, 3)


def test_recadrage_trouve_la_tete(tmp_path):
    img = np.zeros((400, 400), np.uint8)
    cv2.ellipse(img, (150, 150), (100, 120), 0, 0, 360, 150, -1)
    p = tmp_path / "off.png"
    Image.fromarray(img).save(p)
    crop = {s["name"]: s for s in preprocess_file(p, PreprocessConfig(crop=True)).summary["steps"]}["crop"]
    assert crop["applied"]
    x0, y0, x1, y1 = crop["box"]
    assert x1 - x0 < 400 and y1 - y0 < 400


def test_recadrage_ignore_si_zone_trop_petite(tmp_path):
    img = np.zeros((300, 300), np.uint8)
    cv2.circle(img, (150, 150), 10, 200, -1)
    p = tmp_path / "small.png"
    Image.fromarray(img).save(p)
    r = preprocess_file(p, PreprocessConfig(crop=True))
    assert {s["name"]: s for s in r.summary["steps"]}["crop"]["applied"] is False


# ---------- images invalides : toujours une erreur claire, jamais un plantage

def code_of(path, cfg=None):
    with pytest.raises(InvalidImageError) as e:
        preprocess_file(path, cfg)
    return e.value.code


def test_fichier_introuvable(tmp_path):
    assert code_of(tmp_path / "nope.png") == "FILE_NOT_FOUND"


def test_format_non_supporte(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("bonjour")
    assert code_of(p) == "UNSUPPORTED_FORMAT"


def test_fichier_vide(tmp_path):
    p = tmp_path / "empty.png"
    p.write_bytes(b"")
    assert code_of(p) == "EMPTY_FILE"


def test_faux_png(tmp_path):
    p = tmp_path / "fake.png"
    p.write_text("ceci n'est pas une image")
    assert code_of(p) == "CORRUPT_IMAGE"


def test_png_tronque(tmp_path, mri):
    p = tmp_path / "cut.png"
    p.write_bytes(mri.read_bytes()[:200])
    assert code_of(p) == "CORRUPT_IMAGE"


def test_image_trop_petite(tmp_path):
    p = tmp_path / "tiny.png"
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (16, 16), dtype=np.uint8)).save(p)
    assert code_of(p) == "TOO_SMALL"


def test_image_uniforme(tmp_path):
    p = tmp_path / "black.png"
    Image.fromarray(np.zeros((200, 200), np.uint8)).save(p)
    assert code_of(p) == "UNIFORM_IMAGE"


def test_proportions_anormales(tmp_path):
    p = tmp_path / "strip.png"
    Image.fromarray(np.random.default_rng(2).integers(0, 255, (80, 800), dtype=np.uint8)).save(p)
    assert code_of(p) == "BAD_ASPECT_RATIO"


def test_fichier_trop_gros(mri):
    assert code_of(mri, PreprocessConfig(max_file_mb=0.0001)) == "FILE_TOO_LARGE"


def test_un_lot_avec_images_invalides_ne_plante_pas(tmp_path, mri):
    bad = tmp_path / "bad.png"
    bad.write_text("x")
    resultats = []
    for f in [mri, bad, tmp_path / "absent.png", mri]:
        try:
            resultats.append(preprocess_file(f).summary["status"])
        except InvalidImageError as e:
            resultats.append(e.code)
    assert resultats == ["OK", "CORRUPT_IMAGE", "FILE_NOT_FOUND", "OK"]
