#!/usr/bin/env python3
"""Simule l'orchestrateur : enregistre un cas complet avec des données synthétiques (contrat de données)."""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traceability.blockchain_client import BlockchainClient, BlockchainError
from traceability.fake_client import FakeBlockchainClient
from traceability.recorder import ArtifactRecorder
from traceability.storage import CaseStorage
from traceability.verify import verify_case

MODEL_VERSION = "1.0.0"


def png(label, case_id):
    return b"\x89PNG\r\n\x1a\n" + f"SYNTHETIC {label} {case_id}".encode()


def js(obj):
    return json.dumps(obj, indent=2).encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="utiliser le vrai gateway Fabric (port 3000)")
    ap.add_argument("--case", help="identifiant du cas (sinon généré)")
    args = ap.parse_args()

    if args.real:
        client, storage, mode = BlockchainClient(), CaseStorage(), "vrai gateway Fabric"
    else:
        client, storage, mode = FakeBlockchainClient(), CaseStorage(root=tempfile.mkdtemp(prefix="neurotrace_sim_")), "client simulé en mémoire"
    case_id = args.case or f"CASE_{int(time.time())}"
    rec = ArtifactRecorder(client, storage)
    print(f"Cas : {case_id}  ({mode})")

    steps = [
        ("ORIGINAL_IMAGE", png("original", case_id), None),
        ("PREPROCESSED_IMAGE", png("preprocessed", case_id), None),
        ("PREDICTION", js({"case_id": case_id, "predicted_class": "glioma",
                           "class_scores": {"glioma": 0.92, "meningioma": 0.04, "pituitary": 0.03, "notumor": 0.01},
                           "confidence": 0.92, "model_id": "mobilenetv2_brain_mri",
                           "model_version": MODEL_VERSION}), MODEL_VERSION),
        ("VALIDATION", js({"case_id": case_id, "status": "ACCEPTED",
                           "reasons": ["confidence >= 0.90", "image quality OK"]}), MODEL_VERSION),
        ("EXPLANATION", png("gradcam", case_id), MODEL_VERSION),
        ("FINAL_RESULT", js({"case_id": case_id, "predicted_class": "glioma", "status": "ACCEPTED"}), MODEL_VERSION),
    ]
    try:
        for art_type, data, model_version in steps:
            art = rec.record(case_id, art_type, data, model_version=model_version)
            print(f"  {art_type:<20} {art['sha256'][:12]}...  parent={art['parent_artifact_id']}")
        report = verify_case(case_id, client, storage)
    except BlockchainError as e:
        print(f"ERREUR blockchain : {e}")
        return 2
    print(f"Vérification : {report['status']} ({len(report['artifacts'])} artefacts)")
    return 0 if report["status"] == "INTEGRITY_VERIFIED" else 1


if __name__ == "__main__":
    sys.exit(main())
