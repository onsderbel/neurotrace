"""Calcul d'empreinte SHA-256 sur les octets exacts d'un fichier."""
import hashlib


def sha256_file(path) -> str:
    """Renvoie le SHA-256 (64 caractères hexadécimaux) du fichier, lu en binaire."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
