#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TN="${FABRIC_TEST_NETWORK:-$HOME/fabric-samples/test-network}"
cd "$TN"
rm -f neurotrace.tar.gz
./network.sh deployCC -ccn neurotrace -ccp "$ROOT/blockchain/chaincode" -ccl javascript
