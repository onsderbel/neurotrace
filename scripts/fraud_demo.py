#!/usr/bin/env python3
"""Scénario de fraude : un fichier stocké hors chaîne est modifié, la vérification doit le détecter."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traceability.blockchain_client import BlockchainClient, BlockchainError
from traceability.hashing import sha256_file
from traceability.storage import CaseStorage
from traceability.verify import STATUS_KO, STATUS_OK, verify_case

STEPS = [  # (suffixe de l'id, type d'artefact, fichier) : dans l'ordre de la chaîne de provenance
    ("original", "ORIGINAL_IMAGE", "original.png"),
    ("preprocessed", "PREPROCESSED_IMAGE", "preprocessed.png"),
    ("prediction", "PREDICTION", "prediction.json"),
    ("validation", "VALIDATION", "validation.json"),
    ("explanation", "EXPLANATION", "gradcam.png"),
    ("final", "FINAL_RESULT", "final_result.json"),
]


def fake_content(case_id, filename):
    """Contenu synthétique (aucune donnée médicale)."""
    if filename.endswith(".png"):
        return b"\x89PNG\r\n\x1a\n" + f"DEMO-SYNTHETIC {filename} {case_id}".encode()
    docs = {
        "prediction.json": {"case_id": case_id, "predicted_class": "glioma", "confidence": 0.92,
                            "model_id": "mobilenetv2_brain_mri", "model_version": "1.0.0"},
        "validation.json": {"case_id": case_id, "status": "ACCEPTED", "reasons": ["confidence >= 0.90"]},
        "final_result.json": {"case_id": case_id, "predicted_class": "glioma", "status": "ACCEPTED"},
    }
    return json.dumps(docs[filename], indent=2).encode()


def create_case(case_id, client, storage):
    parent = None
    for suffix, art_type, filename in STEPS:
        ref = storage.save(case_id, filename, fake_content(case_id, filename))
        art = {"artifact_id": f"ART_{case_id}_{suffix}", "case_id": case_id, "artifact_type": art_type,
               "sha256": sha256_file(storage.path(case_id, filename)), "model_version": "1.0.0",
               "parent_artifact_id": parent, "storage_ref": ref}
        client.register_artifact(art)
        print(f"    {art_type:<20} {filename:<18} {art['sha256'][:12]}...")
        parent = art["artifact_id"]


def tamper(path):
    original = path.read_bytes()
    if b"glioma" in original:
        path.write_bytes(original.replace(b"glioma", b"notumor", 1))
        return original, "glioma -> notumor"
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
    return original, "1 octet modifié"


def show(report):
    for a in report["artifacts"]:
        if a["status"] != STATUS_OK:
            print(f"    - {a['artifact_type']} ({a['file']}) : {a['reason']}")
            print(f"      hash enregistré : {a['expected_sha256'][:16]}...")
            print(f"      hash recalculé  : {(a['actual_sha256'] or 'fichier absent')[:16]}...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="cas existant à attaquer (sinon : nouveau cas de démonstration)")
    ap.add_argument("--file", default="prediction.json", help="fichier à modifier")
    args = ap.parse_args()

    client, storage = BlockchainClient(), CaseStorage()
    case_id = args.case or f"CASE_{int(time.time())}"
    print(f"Cas : {case_id}")
    try:
        if not args.case:
            print("[1] Enregistrement de 6 artefacts sur Fabric (chaque écriture attend la validation du bloc)")
            create_case(case_id, client, storage)

        r1 = verify_case(case_id, client, storage)
        print(f"[2] Vérification initiale ............. {r1['status']}")

        target = storage.path(case_id, args.file)
        original, what = tamper(target)
        print(f"[3] Modification de {args.file} ({what})")
        try:
            r2 = verify_case(case_id, client, storage)
            print(f"[4] Vérification après modification ... {r2['status']}")
            show(r2)
        finally:
            target.write_bytes(original)          # restauration, même en cas d'erreur
        print("[5] Fichier restauré")

        r3 = verify_case(case_id, client, storage)
        print(f"[6] Vérification après restauration ... {r3['status']}")
    except BlockchainError as e:
        print(f"ERREUR blockchain : {e}")
        return 2

    ok = (r1["status"], r2["status"], r3["status"]) == (STATUS_OK, STATUS_KO, STATUS_OK)
    print("RÉSULTAT : démonstration réussie." if ok else "RÉSULTAT : comportement inattendu !")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
