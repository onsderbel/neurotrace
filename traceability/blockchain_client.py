"""Client Python du service gateway NeuroTrace (interface attendue par les agents)."""
import os

import requests

DEFAULT_URL = os.environ.get("NEUROTRACE_GATEWAY_URL", "http://localhost:3000")


class BlockchainError(Exception):
    """Erreur claire : réseau éteint, artefact introuvable, refus, hash invalide..."""


class BlockchainClient:
    def __init__(self, url: str = DEFAULT_URL, timeout: float = 70):
        self.url = url.rstrip("/")
        self.timeout = timeout   # une écriture attend la validation du bloc (jusqu'à plusieurs secondes)

    @staticmethod
    def _headers():
        key = os.environ.get("GATEWAY_API_KEY")
        return {"X-API-Key": key} if key else {}

    def _call(self, method: str, route: str, **kwargs):
        try:
            r = requests.request(method, self.url + route, timeout=self.timeout, headers=self._headers(), **kwargs)
        except requests.ConnectionError:
            raise BlockchainError(f"Service blockchain injoignable ({self.url}) : gateway ou réseau Fabric éteint ?")
        except requests.Timeout:
            raise BlockchainError(f"Délai dépassé en appelant {route}")
        try:
            body = r.json()
        except ValueError:
            raise BlockchainError(f"Réponse illisible ({r.status_code}) : {r.text[:200]}")
        if not r.ok:
            raise BlockchainError(f"{r.status_code} : {body.get('error', body)}")
        return body

    def register_artifact(self, artifact: dict) -> dict:
        return self._call("POST", "/artifacts", json=artifact)

    def get_artifact(self, artifact_id: str) -> dict:
        return self._call("GET", f"/artifacts/{artifact_id}")

    def get_case_history(self, case_id: str) -> list[dict]:
        return self._call("GET", f"/cases/{case_id}/history")

    def verify_hash(self, artifact_id: str, sha256: str) -> dict:
        return self._call("POST", f"/artifacts/{artifact_id}/verify", json={"sha256": sha256})
