import time
import pytest
import requests
from traceability.blockchain_client import BlockchainClient, BlockchainError

H_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
H_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_gateway_eteint_donne_une_erreur_claire():
    c = BlockchainClient(url="http://localhost:59999", timeout=2)
    with pytest.raises(BlockchainError, match="injoignable"):
        c.get_artifact("ART_X")


def _gateway_up():
    try:
        return requests.get("http://localhost:3000/health", timeout=2).ok
    except requests.RequestException:
        return False


@pytest.mark.skipif(not _gateway_up(), reason="gateway non démarré")
def test_integration_cycle_complet():
    c = BlockchainClient()
    case = f"CASE_{int(time.time())}"
    orig = f"ART_{case}_original"
    art = {"artifact_id": orig, "case_id": case, "artifact_type": "ORIGINAL_IMAGE",
           "sha256": H_ABC, "storage_ref": f"storage/cases/{case}/original.png"}
    assert c.register_artifact(art)["sha256"] == H_ABC
    assert c.verify_hash(orig, H_ABC)["status"] == "INTEGRITY_VERIFIED"
    assert c.verify_hash(orig, H_EMPTY)["status"] == "MODIFICATION_DETECTED"
    assert [a["artifact_id"] for a in c.get_case_history(case)] == [orig]
    with pytest.raises(BlockchainError, match="déjà enregistré"):
        c.register_artifact(art)
    with pytest.raises(BlockchainError, match="introuvable"):
        c.get_artifact("ART_INCONNU")
