# Contrat de données entre modules

Ce document est la référence commune. Toute modification se discute à deux.

## 1. Sortie du module IA (ai/ -> backend/)

```json
{
  "case_id": "CASE_001",
  "predicted_class": "glioma",
  "class_scores": {"glioma": 0.92, "meningioma": 0.04, "pituitary": 0.03, "notumor": 0.01},
  "confidence": 0.92,
  "model_id": "mobilenetv2_brain_mri",
  "model_version": "1.0.0"
}
```

## 2. Sortie du module de validation

```json
{
  "case_id": "CASE_001",
  "status": "ACCEPTED",
  "reasons": ["confidence >= 0.90", "image quality OK"]
}
```
`status` : `ACCEPTED` | `REVIEW_REQUIRED` | `REJECTED`
Seuils initiaux (à ajuster) : >= 0.90 accepté, 0.60 à 0.90 révision, < 0.60 rejeté.

## 3. Artefact enregistré sur Fabric (backend/ -> chaincode)

```json
{
  "artifact_id": "ART_CASE_001_prediction",
  "case_id": "CASE_001",
  "artifact_type": "PREDICTION",
  "sha256": "e3b0c44298fc1c149afbf4c8996fb924...",
  "timestamp": "2026-11-21T14:30:00Z",
  "model_version": "1.0.0",
  "parent_artifact_id": "ART_CASE_001_preprocessed",
  "storage_ref": "storage/cases/CASE_001/prediction.json"
}
```

`artifact_type` : `ORIGINAL_IMAGE` | `PREPROCESSED_IMAGE` | `PREDICTION` | `VALIDATION` | `EXPLANATION` | `FINAL_RESULT`

Chaîne de provenance :
ORIGINAL_IMAGE -> PREPROCESSED_IMAGE -> PREDICTION -> VALIDATION -> EXPLANATION -> FINAL_RESULT

## 4. Fonctions du chaincode

| Fonction | Rôle |
|---|---|
| `RegisterArtifact(json)` | Enregistre un artefact |
| `GetArtifact(artifact_id)` | Lit un artefact |
| `GetCaseHistory(case_id)` | Tous les artefacts d'un cas |
| `VerifyHash(artifact_id, sha256)` | Compare un hash à celui enregistré |

## 5. Organisation du stockage hors chaîne

```
storage/cases/CASE_001/
  original.png
  preprocessed.png
  prediction.json
  validation.json
  gradcam.png
  final_result.json
```

## 6. Endpoints API (proposition)

| Méthode | Route | Rôle |
|---|---|---|
| POST | `/cases` | Upload + lance le pipeline |
| GET | `/cases/{id}` | Résultat du cas |
| GET | `/cases/{id}/history` | Historique blockchain |
| POST | `/cases/{id}/verify` | Vérifie l'intégrité |
