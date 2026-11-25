"""Tests du module de validation, dont un test sur les VRAIES prédictions du modèle (ai/results/predictions.csv)."""
import json
from pathlib import Path

import pytest

from ai.validation import (ACCEPTED, REJECTED, REVIEW_REQUIRED, ValidationConfig, validate)

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def pred(cls="glioma", conf=0.9994, case="CASE_001"):
    rest = (1.0 - conf) / 3
    scores = {c: (conf if c == cls else rest) for c in CLASSES}
    return {"case_id": case, "predicted_class": cls, "class_scores": scores, "confidence": conf,
            "model_id": "resnet18_brain_mri", "model_version": "1.0.0"}


def pre(**quality):
    q = {"mean_brightness": 66.0, "contrast_std": 60.0, "sharpness_laplacian_var": 195.0, "dark_pixel_ratio": 0.35}
    q.update(quality)
    return {"status": "OK", "quality": q}


def expl(cls="glioma", empty=False, area=0.19):
    return {"target_class": cls, "cam_summary": {"empty": empty, "high_activation_area_ratio": area}}


def check(res, name):
    return next(c for c in res["checks"] if c["name"] == name)


# ---------- seuils de confiance

def test_confiance_elevee_acceptee():
    r = validate(pred(conf=0.9994), pre(), expl())
    assert r["status"] == ACCEPTED and r["case_id"] == "CASE_001"


def test_exemple_reel_du_projet_part_en_revision():
    # IRM de test (glioma, bien classée) avec confiance 0.8289 : correcte, mais en dessous du seuil
    r = validate(pred(conf=0.8289), pre(), expl())
    assert r["status"] == REVIEW_REQUIRED


def test_limites_exactes_des_seuils():
    assert validate(pred(conf=0.99))["status"] == ACCEPTED
    assert validate(pred(conf=0.9899))["status"] == REVIEW_REQUIRED
    assert validate(pred(conf=0.60))["status"] == REVIEW_REQUIRED
    assert validate(pred(conf=0.5999))["status"] == REJECTED


def test_seuil_renforce_pour_notumor():
    assert validate(pred("notumor", 0.995))["status"] == REVIEW_REQUIRED
    assert validate(pred("notumor", 0.9995))["status"] == ACCEPTED
    assert validate(pred("glioma", 0.995))["status"] == ACCEPTED
    assert "renforcé" in check(validate(pred("notumor", 0.995)), "confidence")["detail"]


def test_confiance_tres_basse_rejetee():
    r = validate(pred(conf=0.40))
    assert r["status"] == REJECTED and check(r, "confidence")["result"] == "FAIL"


# ---------- cohérence de la sortie de l'IA

@pytest.mark.parametrize("mutation", [
    lambda p: p.update(confidence=0.5),                                              # ne correspond pas au score
    lambda p: p.update(predicted_class="meningioma"),                                # pas le score maximal
    lambda p: p.update(predicted_class="inconnue"),
    lambda p: p["class_scores"].update(glioma=0.2),                                  # somme différente de 1
    lambda p: p.update(confidence=1.5),
    lambda p: p.update(confidence="beaucoup"),
    lambda p: p.pop("class_scores"),
    lambda p: p.update(class_scores={}),
])
def test_prediction_incoherente_rejetee(mutation):
    p = pred(conf=0.9994)
    mutation(p)
    r = validate(p, pre(), expl())
    assert r["status"] == REJECTED
    assert check(r, "prediction_coherence")["result"] == "FAIL"


@pytest.mark.parametrize("garbage", [None, "texte", 42, [], {}, {"confidence": 0.99}])
def test_ne_plante_jamais_sur_des_entrees_absurdes(garbage):
    assert validate(garbage, garbage, garbage)["status"] == REJECTED


# ---------- qualité d'image et explication (déclenchent une relecture)

def test_contraste_faible_declenche_la_relecture():
    r = validate(pred(conf=0.9999), pre(contrast_std=5.0), expl())
    assert r["status"] == REVIEW_REQUIRED and check(r, "image_quality")["result"] == "REVIEW"


def test_image_presque_noire_et_floue():
    assert validate(pred(), pre(dark_pixel_ratio=0.95), expl())["status"] == REVIEW_REQUIRED
    assert validate(pred(), pre(sharpness_laplacian_var=2.0), expl())["status"] == REVIEW_REQUIRED


def test_prétraitement_en_echec_rejete():
    assert validate(pred(), {"status": "ERROR"}, expl())["status"] == REJECTED


def test_explication_d_une_autre_classe_rejetee():
    assert validate(pred("glioma"), pre(), expl("pituitary"))["status"] == REJECTED


def test_carte_vide_ou_diffuse_declenche_la_relecture():
    assert validate(pred(), pre(), expl(empty=True))["status"] == REVIEW_REQUIRED
    assert validate(pred(), pre(), expl(area=0.8))["status"] == REVIEW_REQUIRED


def test_elements_optionnels_absents_sont_ignores_et_signales():
    r = validate(pred(conf=0.9994))
    assert r["status"] == ACCEPTED
    assert check(r, "image_quality")["result"] == "SKIPPED"
    assert check(r, "explanation_consistency")["result"] == "SKIPPED"


# ---------- intégrité des étapes précédentes et identité du modèle

def test_modification_detectee_en_amont_rejette_meme_avec_confiance_parfaite():
    r = validate(pred(conf=0.9999), pre(), expl(), previous_steps_verified=False)
    assert r["status"] == REJECTED and "Modification détectée" in " ".join(r["reasons"])


def test_etapes_precedentes_verifiees():
    r = validate(pred(conf=0.9999), pre(), expl(), previous_steps_verified=True)
    assert r["status"] == ACCEPTED and check(r, "previous_steps_integrity")["result"] == "PASS"


def test_identite_du_modele():
    ok = ValidationConfig(expected_model_id="resnet18_brain_mri", expected_model_version="1.0.0")
    bad = ValidationConfig(expected_model_version="2.0.0")
    assert validate(pred(), pre(), expl(), ok)["status"] == ACCEPTED
    assert validate(pred(), pre(), expl(), bad)["status"] == REJECTED


# ---------- configuration et format de sortie

def test_configuration_invalide():
    with pytest.raises(ValueError):
        ValidationConfig(accept_threshold=0.5, reject_threshold=0.6)
    with pytest.raises(ValueError):
        ValidationConfig(class_accept_thresholds={"notumor": 0.3})


def test_sortie_json_complete_et_seuils_enregistres():
    r = validate(pred(), pre(), expl())
    json.dumps(r)
    assert {"case_id", "status", "reasons", "checks", "thresholds", "validator_version"} <= set(r)
    assert r["thresholds"]["accept_threshold"] == 0.99 and r["thresholds"]["reject_threshold"] == 0.60
    assert r["reasons"]


def test_une_raison_est_donnee_quand_ce_n_est_pas_accepte():
    for conf in (0.8, 0.3):
        r = validate(pred(conf=conf))
        assert r["status"] != ACCEPTED and r["reasons"]


# ---------- les seuils tiennent sur les vraies données du modèle v1.0.0

CSV = Path("ai/results/predictions.csv")


@pytest.mark.skipif(not CSV.exists(), reason="ai/results/predictions.csv absent")
def test_seuils_sur_les_vraies_predictions():
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(CSV)

    def run(split):
        d = df[df.split == split]
        accepted_errors = accepted = 0
        for row in d.itertuples():
            p = {"case_id": row.file, "predicted_class": row.pred_class, "confidence": row.confidence,
                 "class_scores": {c: getattr(row, f"p_{c}") for c in CLASSES}}
            if validate(p)["status"] == ACCEPTED:
                accepted += 1
                accepted_errors += 1 - row.correct
        return len(d), accepted, accepted_errors

    n, acc, err = run("validation")      # groupe sur lequel les seuils ont été choisis
    assert err == 0 and acc / n > 0.85

    n, acc, err = run("test")            # groupe jamais utilisé pour le choix des seuils
    assert 1 - err / acc >= 0.99         # précision des cas acceptés >= 99 % (contre 94,75 % sans validation)
    assert acc / n >= 0.85               # au moins 85 % des cas sont traités automatiquement
