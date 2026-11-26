"""Faux client blockchain en mémoire : même interface et mêmes règles que le chaincode."""
import re
from datetime import datetime, timezone

from .blockchain_client import BlockchainError

TYPE_ORDER = ["ORIGINAL_IMAGE", "PREPROCESSED_IMAGE", "FEATURES",
              "PREDICTION", "VALIDATION", "EXPLANATION", "FINAL_RESULT"]
HASH_RE = re.compile(r"[0-9a-fA-F]{64}")


class FakeBlockchainClient:
    def __init__(self):
        self._arts = {}

    def register_artifact(self, artifact: dict) -> dict:
        a = dict(artifact)
        aid = a.get("artifact_id")
        if aid in self._arts:
            raise BlockchainError(f"409 : Artefact {aid} déjà enregistré : écrasement interdit")
        if a.get("artifact_type") not in TYPE_ORDER:
            raise BlockchainError("400 : artifact_type invalide")
        if not HASH_RE.fullmatch(a.get("sha256") or ""):
            raise BlockchainError("400 : sha256 invalide (64 caractères hexadécimaux)")
        parent = a.get("parent_artifact_id")
        if a["artifact_type"] == "ORIGINAL_IMAGE":
            if parent:
                raise BlockchainError("400 : ORIGINAL_IMAGE ne peut pas avoir de parent")
        else:
            if not parent:
                raise BlockchainError(f"400 : parent_artifact_id obligatoire pour {a['artifact_type']}")
            if parent not in self._arts:
                raise BlockchainError(f"404 : Artefact {parent} introuvable")
            if self._arts[parent]["case_id"] != a["case_id"]:
                raise BlockchainError(f"400 : Le parent {parent} appartient à un autre cas")
        a["sha256"] = a["sha256"].lower()
        a["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        a["model_version"] = a.get("model_version")
        a["parent_artifact_id"] = parent or None
        self._arts[aid] = a
        return dict(a)

    def get_artifact(self, artifact_id: str) -> dict:
        if artifact_id not in self._arts:
            raise BlockchainError(f"404 : Artefact {artifact_id} introuvable")
        return dict(self._arts[artifact_id])

    def get_case_history(self, case_id: str) -> list[dict]:
        arts = [dict(a) for a in self._arts.values() if a["case_id"] == case_id]
        return sorted(arts, key=lambda a: (TYPE_ORDER.index(a["artifact_type"]), a["timestamp"]))

    def verify_hash(self, artifact_id: str, sha256: str) -> dict:
        if not HASH_RE.fullmatch(sha256 or ""):
            raise BlockchainError("400 : Hash invalide : 64 caractères hexadécimaux attendus")
        art = self.get_artifact(artifact_id)
        status = "INTEGRITY_VERIFIED" if art["sha256"] == sha256.lower() else "MODIFICATION_DETECTED"
        return {"artifact_id": artifact_id, "status": status,
                "expected_sha256": art["sha256"], "provided_sha256": sha256.lower()}
