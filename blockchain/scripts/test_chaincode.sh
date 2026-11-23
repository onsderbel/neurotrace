#!/usr/bin/env bash
set -uo pipefail
TN="${FABRIC_TEST_NETWORK:-$HOME/fabric-samples/test-network}"
cd "$TN"
export PATH="$TN/../bin:$PATH"
export FABRIC_CFG_PATH="$TN/../config/"
export CORE_PEER_TLS_ENABLED=true
ORGS="$TN/organizations"
ORDERER_CA="$ORGS/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem"
ORG1_TLS="$ORGS/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
ORG2_TLS="$ORGS/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"

set_org() {  # $1 = numéro d'org, $2 = port du peer
  export CORE_PEER_LOCALMSPID="Org${1}MSP"
  export CORE_PEER_TLS_ROOTCERT_FILE="$ORGS/peerOrganizations/org${1}.example.com/peers/peer0.org${1}.example.com/tls/ca.crt"
  export CORE_PEER_MSPCONFIGPATH="$ORGS/peerOrganizations/org${1}.example.com/users/Admin@org${1}.example.com/msp"
  export CORE_PEER_ADDRESS="localhost:$2"
}
invoke() {  # $1 = fonction, $2.. = arguments
  local fn="$1"; shift
  local cmd; cmd=$(jq -cn --arg f "$fn" '{function:$f,Args:$ARGS.positional}' --args "$@")
  peer chaincode invoke -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
    --tls --cafile "$ORDERER_CA" -C mychannel -n neurotrace \
    --peerAddresses localhost:7051 --tlsRootCertFiles "$ORG1_TLS" \
    --peerAddresses localhost:9051 --tlsRootCertFiles "$ORG2_TLS" -c "$cmd" 2>&1
  sleep 3   # laisse le temps au bloc d'etre valide avant l'etape suivante
}
query() {
  local fn="$1"; shift
  peer chaincode query -C mychannel -n neurotrace \
    -c "$(jq -cn --arg f "$fn" '{function:$f,Args:$ARGS.positional}' --args "$@")" 2>&1
}
PASS=0; FAIL=0
check() {  # $1 = description, $2 = 0 si OK
  if [ "$2" -eq 0 ]; then echo "[OK]   $1"; PASS=$((PASS+1))
  else
    echo "[KO]   $1"; FAIL=$((FAIL+1))
    echo "$out" | tr -cd '\11\12\15\40-\176' | tail -4 | sed 's/^/         > /'
  fi
}

CASE="CASE_$(date +%s)"
H_ABC="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"   # sha256("abc")
H_EMPTY="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" # sha256("")
ORIG="ART_${CASE}_ORIGINAL_IMAGE"; PRE="ART_${CASE}_PREPROCESSED_IMAGE"
echo "Cas de test : $CASE"

set_org 1 7051
A1=$(jq -cn --arg id "$ORIG" --arg c "$CASE" --arg h "$H_ABC" \
  '{artifact_id:$id,case_id:$c,artifact_type:"ORIGINAL_IMAGE",sha256:$h,model_version:null,parent_artifact_id:null,storage_ref:"storage/cases/\($c)/original.png"}')
A2=$(jq -cn --arg id "$PRE" --arg p "$ORIG" --arg c "$CASE" --arg h "$H_EMPTY" \
  '{artifact_id:$id,case_id:$c,artifact_type:"PREPROCESSED_IMAGE",sha256:$h,model_version:null,parent_artifact_id:$p,storage_ref:"storage/cases/\($c)/preprocessed.png"}')

out=$(invoke RegisterArtifact "$A1"); echo "$out" | grep -q "status:200"; check "Org1 enregistre ORIGINAL_IMAGE" $?
out=$(invoke RegisterArtifact "$A2"); echo "$out" | grep -q "status:200"; check "Org1 enregistre PREPROCESSED_IMAGE (parent = original)" $?

out=$(query GetArtifact "$ORIG"); echo "$out" | jq -e --arg h "$H_ABC" '.sha256==$h' >/dev/null 2>&1; check "GetArtifact renvoie le bon hash" $?
out=$(query GetCaseHistory "$CASE"); echo "$out" | jq -e 'length==2' >/dev/null 2>&1; check "GetCaseHistory renvoie 2 artefacts" $?
out=$(query VerifyHash "$ORIG" "$H_ABC"); echo "$out" | grep -q INTEGRITY_VERIFIED; check "VerifyHash : hash identique -> INTEGRITY_VERIFIED" $?
out=$(query VerifyHash "$ORIG" "$H_EMPTY"); echo "$out" | grep -q MODIFICATION_DETECTED; check "VerifyHash : hash différent -> MODIFICATION_DETECTED" $?
out=$(query VerifyHash "$ORIG" "pas-un-hash"); echo "$out" | grep -qi "invalide"; check "VerifyHash : hash invalide -> erreur claire" $?
out=$(query GetArtifact "ART_INEXISTANT"); echo "$out" | grep -qi "introuvable"; check "GetArtifact : artefact inconnu -> erreur claire" $?
out=$(invoke RegisterArtifact "$A1"); echo "$out" | grep -qi "enregistr"; check "Écrasement d'un artefact refusé" $?

set_org 2 9051
A3=$(jq -cn --arg c "$CASE" --arg h "$H_ABC" --arg p "$PRE" \
  '{artifact_id:"ART_\($c)_PREDICTION",case_id:$c,artifact_type:"PREDICTION",sha256:$h,model_version:"v1.0.0",parent_artifact_id:$p,storage_ref:"storage/cases/\($c)/prediction.json"}')
out=$(invoke RegisterArtifact "$A3"); echo "$out" | grep -qi "non autoris"; check "Org2 (non autorisée) refusée à l'enregistrement" $?
out=$(query GetArtifact "$ORIG"); echo "$out" | jq -e '.artifact_id' >/dev/null 2>&1; check "Org2 peut lire un artefact" $?

echo "Résultat : $PASS OK, $FAIL KO"
[ "$FAIL" -eq 0 ]
