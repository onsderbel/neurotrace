# NeuroTrace

Détection de tumeurs cérébrales par IA + traçabilité et intégrité via Hyperledger Fabric.
Prototype pédagogique, année universitaire 2026-2027.

## Structure

| Dossier | Contenu | Responsable |
|---|---|---|
| `ai/` | Entraînement, prétraitement, validation, Grad-CAM | B |
| `blockchain/` | Réseau Fabric, chaincode | A |
| `backend/` | API FastAPI, orchestrateur, hash, vérification | A |
| `frontend/` | Interface web | B |
| `storage/` | Stockage hors chaîne (non versionné) | A |
| `docs/` | Contrat de données, rapport | Les deux |
| `scripts/` | Scripts utilitaires (check_env.sh) | A |

## Pipeline

Image IRM -> Prétraitement -> IA -> Validation -> Explication -> SHA-256 -> Fabric -> Vérification

## Démarrage

1. `bash scripts/check_env.sh` : vérifie l'environnement
2. Lire `docs/CONTRAT_DONNEES.md` avant de coder un module

## Règles Git

- Une branche par fonctionnalité : `feat/chaincode`, `feat/ia-training`...
- Pas de commit direct sur `main` : Pull Request puis relecture par l'autre
- Jamais d'images IRM, de modèles lourds ou de clés dans Git
