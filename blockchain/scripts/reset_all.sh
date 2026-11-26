#!/usr/bin/env bash
set -euo pipefail
TN="${FABRIC_TEST_NETWORK:-$HOME/fabric-samples/test-network}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo "[1/4] Arrêt du réseau";                    (cd "$TN" && ./network.sh down)
echo "[2/4] Démarrage du réseau et du canal";    (cd "$TN" && ./network.sh up createChannel -c mychannel -ca)
echo "[3/4] Déploiement du chaincode";           bash "$ROOT/blockchain/scripts/deploy.sh"
echo "[4/4] Tests du chaincode";                 bash "$ROOT/blockchain/scripts/test_chaincode.sh"
echo "Réseau prêt. Relance le gateway : cd blockchain/gateway && node server.js"
