# Partie Blockchain : NeuroTrace

La blockchain ne stocke que des empreintes SHA-256 et des métadonnées : jamais d'image ni de donnée médicale.

## Architecture

    Agents / orchestrateur
            |  ArtifactRecorder.record(case_id, type, octets)
            v
    CaseStorage (storage/cases/CASE_001/...)  +  sha256_file
            |  BlockchainClient (HTTP + clé API)
            v
    Gateway Node.js (127.0.0.1:3000)
            |
            v
    Hyperledger Fabric (mychannel, chaincode neurotrace)

## Prérequis
- WSL (Ubuntu 24), Docker, Node.js 18+, Python 3.12 (+ `python3.12-venv`), jq
- Hyperledger Fabric 2.5.16 et CA 1.5.17 dans `~/fabric-samples` (`bash scripts/check_env.sh`)

## Installation (une seule fois)
    python3 -m venv .venv
    source .venv/bin/activate
    pip install pytest requests
    (cd blockchain/gateway && npm install)
    openssl rand -hex 16 > ~/.neurotrace_key && chmod 600 ~/.neurotrace_key

La clé API reste dans votre dossier personnel : ne la mettez jamais dans Git.

## Démarrage
1. Réseau + chaîne de code (arrête, relance, déploie et teste ; attendu : `11 OK, 0 KO`) :

        bash blockchain/scripts/reset_all.sh

2. Gateway, dans un terminal à part (à relancer après chaque `reset_all.sh`) :

        cd blockchain/gateway
        export GATEWAY_API_KEY="$(cat ~/.neurotrace_key)"
        node server.js

3. Dans chaque terminal Python :

        source .venv/bin/activate
        export GATEWAY_API_KEY="$(cat ~/.neurotrace_key)"

## Tests et démonstrations
    python -m pytest traceability -q               # attendu : 31 passed (gateway lancé)
    python scripts/simulate_pipeline.py            # cas simulé, client en mémoire
    python scripts/simulate_pipeline.py --real     # cas simulé sur le vrai Fabric
    python scripts/fraud_demo.py                   # scénario de fraude

La démonstration de fraude enregistre un cas de 6 artefacts, modifie `prediction.json` sur le disque, puis montre le passage de `INTEGRITY_VERIFIED` à `MODIFICATION_DETECTED`. Le fichier est restauré à la fin.

## Utilisation depuis le backend
    from traceability.recorder import ArtifactRecorder
    from traceability.blockchain_client import BlockchainClient
    from traceability.verify import verify_case

    rec = ArtifactRecorder(BlockchainClient())     # en développement : FakeBlockchainClient()
    rec.record("CASE_001", "PREDICTION", prediction_bytes, model_version="1.0.0")
    verify_case("CASE_001")                        # INTEGRITY_VERIFIED ou MODIFICATION_DETECTED

Le backend s'exécute depuis la racine du dépôt pour que `traceability` soit importable. `record` est rejouable sans risque après une panne.

## Règles
- Org1 enregistre, Org2 ne peut que lire.
- Un artefact enregistré n'est jamais modifié ni écrasé.
- Le timestamp est posé par le chaincode (heure de la transaction).
- Statuts de vérification : `INTEGRITY_VERIFIED` ou `MODIFICATION_DETECTED`.
- Aucune clé ni donnée médicale dans Git.

## Dépannage
- `401 : Clé API manquante ou invalide` : `GATEWAY_API_KEY` n'est pas exportée dans ce terminal.
- `Service blockchain injoignable` : le gateway ou le réseau Fabric est éteint.
- `Aucun artefact enregistré pour CASE_...` après un `reset_all.sh` : le registre a été vidé ; supprimer les anciens dossiers `storage/cases/CASE_*`.
