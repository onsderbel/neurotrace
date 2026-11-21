"""Tests de l'affichage Grad-CAM (sans PyTorch)."""
import numpy as np
import pytest

from ai.explainability import overlay_heatmap, summarize_cam


def blob(h=224, w=224, cx=150, cy=60, sigma=15):
    ys, xs = np.mgrid[0:h, 0:w]
    cam = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2)).astype(np.float32)
    return cam / cam.max()


def img(h=224, w=224):
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_overlay_forme_et_type():
    out = overlay_heatmap(img(), blob())
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8


def test_overlay_alpha_zero_rend_l_image_d_origine():
    im = img()
    assert np.array_equal(overlay_heatmap(im, blob(), alpha=0.0), im)


def test_overlay_modifie_l_image_avec_alpha_positif():
    im = img()
    assert not np.array_equal(overlay_heatmap(im, blob(), alpha=0.5), im)


def test_overlay_redimensionne_une_carte_plus_petite():
    out = overlay_heatmap(img(224, 224), blob(7, 7, 3, 3, 1.5))
    assert out.shape == (224, 224, 3)


def test_overlay_alpha_invalide():
    with pytest.raises(ValueError):
        overlay_heatmap(img(), blob(), alpha=1.5)


def test_resume_localise_le_pic():
    s = summarize_cam(blob(cx=150, cy=60))
    assert not s["empty"]
    assert abs(s["peak_xy"][0] - 150 / 223) < 0.02 and abs(s["peak_xy"][1] - 60 / 223) < 0.02
    assert abs(s["center_of_mass_xy"][0] - 150 / 223) < 0.03
    assert 0 < s["high_activation_area_ratio"] < 0.2


def test_resume_carte_vide():
    s = summarize_cam(np.zeros((224, 224), np.float32))
    assert s["empty"] and s["peak_xy"] is None and s["high_activation_area_ratio"] == 0.0
