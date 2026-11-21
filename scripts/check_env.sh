#!/usr/bin/env bash
# Vérifie l'environnement NeuroTrace (à lancer dans WSL)
GREEN='\033[0;32m'; RED='\033[0;31m'; YEL='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}   $1"; }
fail() { echo -e "${RED}[KO]${NC}   $1"; FAILED=1; }
warn() { echo -e "${YEL}[WARN]${NC} $1"; }
FAILED=0

check_cmd() {
  if command -v "$1" >/dev/null 2>&1; then ok "$1 : $($1 $2 2>&1 | head -n1)"
  else fail "$1 introuvable"; fi
}

echo "=== Système ==="
grep -qi microsoft /proc/version && ok "WSL détecté" || warn "Pas de WSL détecté"
case "$PWD" in /mnt/*) warn "Le projet est sur /mnt/c : déplace-le dans ~/ (beaucoup plus rapide et évite des bugs Fabric)";; *) ok "Projet sur le système de fichiers Linux";; esac

echo; echo "=== Outils de base ==="
check_cmd git --version
check_cmd curl --version
check_cmd jq --version
check_cmd python3 --version
check_cmd pip3 --version

echo; echo "=== Docker ==="
check_cmd docker --version
if docker info >/dev/null 2>&1; then ok "Le démon Docker répond"; else fail "Docker ne répond pas (lance Docker Desktop, active l'intégration WSL)"; fi
if docker compose version >/dev/null 2>&1; then ok "docker compose : $(docker compose version | head -n1)"; else fail "docker compose introuvable"; fi

echo; echo "=== Langages chaincode ==="
check_cmd node --version
check_cmd npm --version
check_cmd go version

echo; echo "=== Hyperledger Fabric ==="
FS=""
for p in "$HOME/fabric-samples" "$PWD/fabric-samples" "$PWD/../fabric-samples"; do
  [ -d "$p/test-network" ] && FS="$p" && break
done
if [ -n "$FS" ]; then
  ok "fabric-samples trouvé : $FS"
  if [ -x "$FS/bin/peer" ]; then ok "peer : $("$FS/bin/peer" version | grep -m1 Version)"; else fail "Binaires Fabric absents dans $FS/bin"; fi
  [ -x "$FS/bin/configtxgen" ] && ok "configtxgen présent" || fail "configtxgen absent"
else
  fail "fabric-samples introuvable (voir commande d'installation dans le chat)"
fi

echo; echo "=== Ports utilisés (doivent être libres) ==="
for port in 7050 7051 9051 7054 8000 3000; do
  if (ss -ltn 2>/dev/null | grep -q ":$port "); then warn "port $port déjà utilisé"; else ok "port $port libre"; fi
done

echo
if [ $FAILED -eq 0 ]; then echo -e "${GREEN}Environnement prêt.${NC}"; else echo -e "${RED}Des éléments sont à corriger.${NC}"; fi
