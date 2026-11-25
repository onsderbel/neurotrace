"""Module de validation : décide si le résultat de l'IA peut être accepté.

Trois états possibles :
  ACCEPTED         le résultat est cohérent et la confiance est élevée ;
  REVIEW_REQUIRED  un humain doit relire ce cas ;
  REJECTED         le résultat est inutilisable (incohérent, confiance trop basse, étape précédente modifiée...).

Les seuils de CONFIANCE ont été choisis en mesurant le modèle v1.0.0 sur le groupe « validation »,
puis vérifiés sur le groupe « test » (voir ai/results/predictions.csv et les tests).
Les seuils de QUALITÉ D'IMAGE et d'ATTENTION Grad-CAM sont PROVISOIRES : ils ne font que déclencher
une relecture (jamais un rejet) et doivent être calibrés sur le dataset.

Cette validation est un garde-fou, pas un diagnostic : elle ne détecte pas les erreurs commises avec
une confiance très élevée (voir la section « limites » du rapport).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

VALIDATOR_VERSION = "1.0.0"

ACCEPTED = "ACCEPTED"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
REJECTED = "REJECTED"

PASS, REVIEW, FAIL, SKIPPED = "PASS", "REVIEW", "FAIL", "SKIPPED"


@dataclass(frozen=True)
class ValidationConfig:
    # --- seuils de confiance (calibrés sur ai/results/predictions.csv)
    accept_threshold: float = 0.99        # >= : accepté
    reject_threshold: float = 0.60        # <  : rejeté (le modèle est proche du hasard)
    # Précaution : un faux « pas de tumeur » est plus grave qu'un faux « tumeur »
    class_accept_thresholds: dict = field(default_factory=lambda: {"notumor": 0.999})
    # --- seuils PROVISOIRES (déclenchent seulement une relecture)
    min_contrast_std: float = 20.0
    max_dark_pixel_ratio: float = 0.85
    min_sharpness: float = 10.0
    max_cam_area_ratio: float = 0.60
    # --- cohérence
    score_sum_tolerance: float = 0.01
    expected_model_id: str | None = None
    expected_model_version: str | None = None

    def __post_init__(self):
        if not 0.0 < self.reject_threshold < self.accept_threshold <= 1.0:
            raise ValueError("Il faut 0 < reject_threshold < accept_threshold <= 1")
        for cls, t in self.class_accept_thresholds.items():
            if not self.reject_threshold < t <= 1.0:
                raise ValueError(f"Seuil invalide pour la classe {cls}")

    def accept_threshold_for(self, class_name) -> float:
        return self.class_accept_thresholds.get(class_name, self.accept_threshold)

    def to_dict(self) -> dict:
        return asdict(self)


def _check_prediction(pred, tol: float) -> tuple[str, str]:
    """Cohérence interne de la sortie de l'IA (format du contrat de données)."""
    if not isinstance(pred, dict):
        return FAIL, "La prédiction est absente ou n'est pas un dictionnaire."
    missing = [k for k in ("predicted_class", "class_scores", "confidence") if k not in pred]
    if missing:
        return FAIL, f"Prédiction incomplète, champs manquants : {missing}."
    scores = pred["class_scores"]
    if not isinstance(scores, dict) or not scores:
        return FAIL, "class_scores est vide ou invalide."
    try:
        values = {k: float(v) for k, v in scores.items()}
        conf = float(pred["confidence"])
    except (TypeError, ValueError):
        return FAIL, "Les scores ou la confiance ne sont pas des nombres."
    if any(not 0.0 <= v <= 1.0 for v in values.values()) or not 0.0 <= conf <= 1.0:
        return FAIL, "Un score ou la confiance est hors de l'intervalle [0, 1]."
    if abs(sum(values.values()) - 1.0) > tol:
        return FAIL, f"La somme des scores vaut {sum(values.values()):.4f} au lieu de 1."
    cls = pred["predicted_class"]
    if cls not in values:
        return FAIL, f"La classe prédite « {cls} » ne figure pas dans les scores."
    if values[cls] < max(values.values()) - 1e-6:
        return FAIL, "La classe prédite n'est pas celle qui a le score le plus élevé."
    if abs(values[cls] - conf) > 1e-3:
        return FAIL, "La confiance ne correspond pas au score de la classe prédite."
    return PASS, "Sortie de l'IA cohérente (scores, classe prédite et confiance concordent)."


def _run(prediction, preprocess_summary, explanation, cfg, previous_steps_verified):
    checks = []

    def add(name, result, detail):
        checks.append({"name": name, "result": result, "detail": detail})

    # 1. Cohérence de la sortie de l'IA
    res, detail = _check_prediction(prediction, cfg.score_sum_tolerance)
    add("prediction_coherence", res, detail)
    pred_ok = res == PASS

    # 2. Identité du modèle (reproductibilité)
    if cfg.expected_model_id is None and cfg.expected_model_version is None:
        add("model_identity", SKIPPED, "Aucun modèle attendu n'est configuré.")
    elif pred_ok:
        ok = (cfg.expected_model_id in (None, prediction.get("model_id"))
              and cfg.expected_model_version in (None, prediction.get("model_version")))
        add("model_identity", PASS if ok else FAIL,
            "Le modèle utilisé est celui qui est attendu." if ok else
            f"Modèle inattendu : {prediction.get('model_id')} {prediction.get('model_version')}.")
    else:
        add("model_identity", SKIPPED, "Prédiction invalide.")

    # 3. Confiance
    if pred_ok:
        conf, cls = float(prediction["confidence"]), prediction["predicted_class"]
        t_acc, t_rej = cfg.accept_threshold_for(cls), cfg.reject_threshold
        if conf >= t_acc:
            add("confidence", PASS, f"Confiance {conf:.4f} >= seuil d'acceptation {t_acc:g} (classe « {cls} »).")
        elif conf >= t_rej:
            strict = " renforcé" if t_acc != cfg.accept_threshold else ""
            add("confidence", REVIEW,
                f"Confiance {conf:.4f} inférieure au seuil d'acceptation{strict} {t_acc:g} pour « {cls} » : relecture requise.")
        else:
            add("confidence", FAIL, f"Confiance {conf:.4f} inférieure au seuil de rejet {t_rej:g} : le modèle est trop incertain.")
    else:
        add("confidence", SKIPPED, "Prédiction invalide.")

    # 4. Prétraitement et qualité de l'image
    if preprocess_summary is None:
        add("preprocessing_status", SKIPPED, "Résumé de prétraitement non fourni.")
        add("image_quality", SKIPPED, "Résumé de prétraitement non fourni.")
    else:
        status = preprocess_summary.get("status") if isinstance(preprocess_summary, dict) else None
        if status != "OK":
            add("preprocessing_status", FAIL, f"Le prétraitement ne s'est pas terminé correctement (statut : {status}).")
            add("image_quality", SKIPPED, "Prétraitement non valide.")
        else:
            add("preprocessing_status", PASS, "Prétraitement terminé correctement.")
            q = preprocess_summary.get("quality") or {}
            problems = []
            if q.get("contrast_std", 1e9) < cfg.min_contrast_std:
                problems.append(f"contraste faible ({q['contrast_std']} < {cfg.min_contrast_std:g})")
            if q.get("dark_pixel_ratio", 0.0) > cfg.max_dark_pixel_ratio:
                problems.append(f"image presque entièrement noire ({q['dark_pixel_ratio']} > {cfg.max_dark_pixel_ratio:g})")
            if q.get("sharpness_laplacian_var", 1e9) < cfg.min_sharpness:
                problems.append(f"image floue ({q['sharpness_laplacian_var']} < {cfg.min_sharpness:g})")
            if problems:
                add("image_quality", REVIEW, "Qualité d'image douteuse : " + ", ".join(problems) + ".")
            else:
                add("image_quality", PASS, "Indicateurs de qualité d'image dans les limites attendues.")

    # 5. Cohérence de l'explication Grad-CAM
    if explanation is None:
        add("explanation_consistency", SKIPPED, "Explication non fournie.")
    elif not pred_ok:
        add("explanation_consistency", SKIPPED, "Prédiction invalide.")
    else:
        summary = (explanation.get("cam_summary") or {}) if isinstance(explanation, dict) else {}
        if explanation.get("target_class") != prediction["predicted_class"]:
            add("explanation_consistency", FAIL, "L'explication porte sur une autre classe que la prédiction.")
        elif summary.get("empty"):
            add("explanation_consistency", REVIEW, "La carte de chaleur est vide : l'explication n'est pas exploitable.")
        elif summary.get("high_activation_area_ratio", 0.0) > cfg.max_cam_area_ratio:
            add("explanation_consistency", REVIEW,
                f"Attention du modèle très diffuse ({summary['high_activation_area_ratio']:.0%} de l'image).")
        else:
            add("explanation_consistency", PASS, "Explication cohérente avec la prédiction.")

    # 6. Intégrité des étapes précédentes (vérifiée par le module blockchain)
    if previous_steps_verified is None:
        add("previous_steps_integrity", SKIPPED, "Intégrité des étapes précédentes non vérifiée.")
    elif previous_steps_verified:
        add("previous_steps_integrity", PASS, "Les étapes précédentes sont intègres.")
    else:
        add("previous_steps_integrity", FAIL, "Modification détectée sur une étape précédente.")
    return checks


def validate(prediction, preprocess_summary=None, explanation=None, config: ValidationConfig | None = None,
             previous_steps_verified: bool | None = None) -> dict:
    """Retourne le résultat de validation. Ne lève jamais d'exception : une erreur interne donne REJECTED."""
    cfg = config or ValidationConfig()
    try:
        checks = _run(prediction, preprocess_summary, explanation, cfg, previous_steps_verified)
    except Exception as e:  # garde-fou : la validation ne doit jamais interrompre le pipeline
        checks = [{"name": "internal_error", "result": FAIL, "detail": f"Erreur interne de validation : {type(e).__name__}."}]

    results = [c["result"] for c in checks]
    status = REJECTED if FAIL in results else REVIEW_REQUIRED if REVIEW in results else ACCEPTED
    reasons = [c["detail"] for c in checks if c["result"] in (FAIL, REVIEW)] or \
              [c["detail"] for c in checks if c["result"] == PASS]
    p = prediction if isinstance(prediction, dict) else {}
    return {
        "case_id": p.get("case_id"),
        "status": status,
        "reasons": reasons,
        "checks": checks,
        "model_id": p.get("model_id"),
        "model_version": p.get("model_version"),
        "validator_version": VALIDATOR_VERSION,
        "thresholds": cfg.to_dict(),
    }
