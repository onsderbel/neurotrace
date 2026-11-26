"""Enregistrement d'une étape du pipeline : rangement hors chaîne + SHA-256 + Fabric, en un appel."""
import hashlib

from .blockchain_client import BlockchainError
from .hashing import sha256_file
from .storage import CaseStorage

# (type d'artefact, suffixe de l'artifact_id, fichier) dans l'ordre de la chaîne de provenance
# du contrat de données (FEATURES n'y figure pas)
CHAIN = [
    ("ORIGINAL_IMAGE", "original", "original.png"),
    ("PREPROCESSED_IMAGE", "preprocessed", "preprocessed.png"),
    ("PREDICTION", "prediction", "prediction.json"),
    ("VALIDATION", "validation", "validation.json"),
    ("EXPLANATION", "explanation", "gradcam.png"),
    ("FINAL_RESULT", "final", "final_result.json"),
]
SPEC = {t: (suffix, filename) for t, suffix, filename in CHAIN}


def artifact_id_for(case_id: str, artifact_type: str) -> str:
    return f"ART_{case_id}_{SPEC[artifact_type][0]}"


def default_parent(case_id: str, artifact_type: str):
    types = [t for t, _, _ in CHAIN]
    i = types.index(artifact_type)
    return None if i == 0 else artifact_id_for(case_id, types[i - 1])


class ArtifactRecorder:
    def __init__(self, client, storage=None):
        self.client = client
        self.storage = storage or CaseStorage()

    def record(self, case_id, artifact_type, data: bytes, model_version=None,
               parent_artifact_id="auto") -> dict:
        """Range le fichier, calcule son SHA-256 et l'enregistre. Rejouable sans risque."""
        if artifact_type not in SPEC:
            raise ValueError(f"artifact_type inconnu : {artifact_type!r} (attendu : {', '.join(SPEC)})")
        _, filename = SPEC[artifact_type]
        parent = default_parent(case_id, artifact_type) if parent_artifact_id == "auto" else parent_artifact_id

        path = self.storage.path(case_id, filename)
        if path.exists():
            # reprise après panne : on accepte le fichier déjà rangé s'il est identique
            if sha256_file(path) != hashlib.sha256(data).hexdigest():
                raise FileExistsError(f"{case_id}/{filename} existe déjà avec un contenu différent")
            storage_ref = f"storage/cases/{case_id}/{filename}"
        else:
            storage_ref = self.storage.save(case_id, filename, data)

        artifact = {
            "artifact_id": artifact_id_for(case_id, artifact_type),
            "case_id": case_id,
            "artifact_type": artifact_type,
            "sha256": sha256_file(path),          # octets exacts du fichier sur le disque
            "model_version": model_version,
            "parent_artifact_id": parent,
            "storage_ref": storage_ref,
        }
        try:
            return self.client.register_artifact(artifact)
        except BlockchainError as e:
            if "déjà enregistré" not in str(e):
                raise
            existing = self.client.get_artifact(artifact["artifact_id"])
            if existing["sha256"] != artifact["sha256"]:
                raise                              # même id, contenu différent : refus
            return existing                        # déjà enregistré à l'identique
