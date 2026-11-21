"""Tests de l'inférence et de Grad-CAM. Utilisent un petit modèle aléatoire créé à la volée,
plus (si présent) le vrai modèle ai/models/neurotrace_model.pt."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
models = pytest.importorskip("torchvision.models")

from ai.inference import ModelLoadError, NeuroTraceModel  # noqa: E402
from ai.preprocessing import preprocess_file  # noqa: E402

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
REAL_MODEL = Path("ai/models/neurotrace_model.pt")


def make_checkpoint(path, seed=0, **override):
    torch.manual_seed(seed)
    net = models.resnet18(weights=None)
    net.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(net.fc.in_features, len(CLASSES)))
    ckpt = {
        "model_id": "test_resnet18", "model_version": "0.0.1", "arch": "resnet18",
        "class_names": CLASSES, "img_size": 224,
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "state_dict": net.state_dict(),
    }
    ckpt.update(override)
    torch.save(ckpt, path)
    return path


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return NeuroTraceModel(make_checkpoint(tmp_path_factory.mktemp("m") / "model.pt"))


@pytest.fixture
def tensor():
    return np.random.default_rng(1).normal(size=(3, 224, 224)).astype(np.float32)


# ---------- chargement

def test_metadonnees_chargees(model):
    assert model.model_id == "test_resnet18" and model.model_version == "0.0.1"
    assert model.class_names == CLASSES and model.img_size == 224


def test_fichier_modele_absent(tmp_path):
    with pytest.raises(ModelLoadError, match="introuvable"):
        NeuroTraceModel(tmp_path / "absent.pt")


def test_fichier_modele_corrompu(tmp_path):
    p = tmp_path / "bad.pt"
    p.write_bytes(b"ceci n'est pas un modele")
    with pytest.raises(ModelLoadError):
        NeuroTraceModel(p)


def test_checkpoint_incomplet(tmp_path):
    p = tmp_path / "partial.pt"
    torch.save({"model_id": "x"}, p)
    with pytest.raises(ModelLoadError, match="incomplet"):
        NeuroTraceModel(p)


def test_poids_incompatibles(tmp_path):
    p = make_checkpoint(tmp_path / "w.pt", class_names=CLASSES + ["autre"])   # 5 classes pour 4 sorties
    with pytest.raises(ModelLoadError, match="architecture"):
        NeuroTraceModel(p)


def test_config_de_pretraitement_alignee(model):
    cfg = model.preprocess_config()
    assert cfg.img_size == 224 and cfg.mean == model.mean and cfg.std == model.std
    assert cfg.crop is False and cfg.clahe is False


# ---------- prédiction

def test_format_du_contrat(model, tensor):
    p = model.predict(tensor, case_id="CASE_001")
    assert set(p) == {"case_id", "predicted_class", "class_scores", "confidence", "model_id", "model_version"}
    assert p["case_id"] == "CASE_001" and p["model_version"] == "0.0.1"
    assert p["predicted_class"] in CLASSES and list(p["class_scores"]) == CLASSES


def test_scores_coherents(model, tensor):
    p = model.predict(tensor)
    assert abs(sum(p["class_scores"].values()) - 1.0) < 1e-3
    assert p["confidence"] == max(p["class_scores"].values())
    assert p["class_scores"][p["predicted_class"]] == p["confidence"]
    assert 0.0 <= p["confidence"] <= 1.0


def test_prediction_deterministe(model, tensor):
    assert model.predict(tensor) == model.predict(tensor)


def test_forme_d_entree_invalide(model):
    with pytest.raises(ValueError, match="Forme"):
        model.predict(np.zeros((3, 100, 100), np.float32))


# ---------- Grad-CAM

def test_gradcam_forme_et_bornes(model, tensor):
    cam, info = model.explain(tensor)
    assert cam.shape == (224, 224) and cam.dtype == np.float32
    assert 0.0 <= float(cam.min()) and float(cam.max()) <= 1.0
    assert info["method"] == "Grad-CAM" and "disclaimer" in info


def test_gradcam_utilise_la_classe_predite_par_defaut(model, tensor):
    pred = model.predict(tensor)
    _, info = model.explain(tensor, prediction=pred)
    assert info["target_class"] == pred["predicted_class"]


def test_gradcam_classe_choisie_et_classe_inconnue(model, tensor):
    _, info = model.explain(tensor, target_class="notumor")
    assert info["target_class"] == "notumor"
    with pytest.raises(ValueError, match="inconnue"):
        model.explain(tensor, target_class="inexistante")


def test_gradcam_ne_modifie_pas_le_modele(model, tensor):
    avant = model.predict(tensor)
    poids = {k: v.clone() for k, v in model.net.state_dict().items()}
    model.explain(tensor)
    assert not model.net.training
    assert all(torch.equal(poids[k], v) for k, v in model.net.state_dict().items())
    assert all(p.grad is None for p in model.net.parameters())
    assert model.predict(tensor) == avant


def test_gradcam_deterministe(model, tensor):
    a, _ = model.explain(tensor)
    b, _ = model.explain(tensor)
    assert np.allclose(a, b)


# ---------- chaîne complète avec le vrai fichier de modèle (ignoré s'il est absent)

@pytest.mark.skipif(not REAL_MODEL.exists(), reason="ai/models/neurotrace_model.pt absent")
def test_vrai_modele_charge_et_analyse(tmp_path):
    from PIL import Image
    m = NeuroTraceModel(REAL_MODEL)
    assert m.model_id == "resnet18_brain_mri" and m.class_names == CLASSES
    assert abs(m.test_metrics["accuracy"] - 0.9475) < 1e-6
    img = tmp_path / "x.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (300, 300), dtype=np.uint8)).save(img)
    pre = preprocess_file(img, m.preprocess_config())
    pred = m.predict(pre.tensor, case_id="CASE_X")
    assert pred["predicted_class"] in CLASSES and 0 <= pred["confidence"] <= 1
    cam, info = m.explain(pre.tensor, prediction=pred)
    assert cam.shape == (224, 224)


def test_gradcam_identique_a_un_calcul_independant(model, tensor):
    """Recalcule Grad-CAM autrement (torch.autograd.grad) et compare avec notre implémentation."""
    import torch.nn.functional as F
    net = model.net
    x = torch.from_numpy(tensor).unsqueeze(0).requires_grad_(True)
    feats = {}
    h = net.layer4[-1].register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    logits = net(x)
    h.remove()
    idx = int(logits.argmax())
    grad = torch.autograd.grad(logits[0, idx], feats["a"])[0]
    cam = F.relu((grad.mean(dim=(2, 3), keepdim=True) * feats["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    ours, _ = model.explain(tensor)
    assert np.allclose(ours, cam, atol=1e-5)
