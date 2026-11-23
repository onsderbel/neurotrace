"""Stockage hors chaîne des fichiers d'un cas : storage/cases/CASE_001/..."""
import re
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "storage" / "cases"
CASE_RE = re.compile(r"CASE_\d{3,}")
FILENAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class CaseStorage:
    def __init__(self, root=None):
        self.root = Path(root) if root else DEFAULT_ROOT

    def _check(self, case_id: str, filename: str) -> None:
        if not CASE_RE.fullmatch(case_id or ""):
            raise ValueError(f"case_id invalide : {case_id!r} (format attendu : CASE_001)")
        if not FILENAME_RE.fullmatch(filename or "") or ".." in filename:
            raise ValueError(f"nom de fichier invalide : {filename!r}")

    def path(self, case_id: str, filename: str) -> Path:
        """Chemin réel du fichier sur le disque."""
        self._check(case_id, filename)
        return self.root / case_id / filename

    def save(self, case_id: str, filename: str, data: bytes) -> str:
        """Écrit le fichier et renvoie le storage_ref (jamais d'écrasement)."""
        p = self.path(case_id, filename)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(p, "xb") as f:      # "x" : échoue si le fichier existe déjà
                f.write(data)
        except FileExistsError:
            raise FileExistsError(f"{case_id}/{filename} existe déjà : un artefact n'est jamais écrasé")
        return f"storage/cases/{case_id}/{filename}"
