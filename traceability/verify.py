"""Vérification d'intégrité d'un cas complet : fichiers du disque vs hashes de la blockchain."""
from pathlib import Path

from .hashing import sha256_file
from .storage import CaseStorage

STATUS_OK = "INTEGRITY_VERIFIED"
STATUS_KO = "MODIFICATION_DETECTED"


def verify_case(case_id: str, client=None, storage=None) -> dict:
    """Recalcule le SHA-256 de chaque fichier du cas et le compare à la blockchain."""
    if client is None:
        from .blockchain_client import BlockchainClient
        client = BlockchainClient()
    storage = storage or CaseStorage()

    history = client.get_case_history(case_id)
    if not history:
        raise LookupError(f"Aucun artefact enregistré pour {case_id}")

    results, all_ok = [], True
    for art in history:
        filename = Path(art["storage_ref"]).name
        path = storage.path(case_id, filename)
        entry = {"artifact_id": art["artifact_id"], "artifact_type": art["artifact_type"],
                 "file": filename, "expected_sha256": art["sha256"],
                 "actual_sha256": None, "reason": None}
        if not path.is_file():
            entry.update(status=STATUS_KO, reason="FILE_MISSING")
        else:
            entry["actual_sha256"] = sha256_file(path)
            # la comparaison est faite par le chaincode : la blockchain fait foi
            answer = client.verify_hash(art["artifact_id"], entry["actual_sha256"])
            entry["status"] = answer["status"]
            if answer["status"] != STATUS_OK:
                entry["reason"] = "HASH_MISMATCH"
        all_ok = all_ok and entry["status"] == STATUS_OK
        results.append(entry)

    return {"case_id": case_id,
            "status": STATUS_OK if all_ok else STATUS_KO,
            "artifacts": results}
