import pytest
from traceability.hashing import sha256_file
from traceability.storage import CaseStorage
from traceability.verify import STATUS_KO, STATUS_OK, verify_case


class FakeClient:
    def __init__(self):
        self.arts = {}

    def register_artifact(self, a):
        self.arts[a["artifact_id"]] = dict(a)
        return a

    def get_case_history(self, case_id):
        return [a for a in self.arts.values() if a["case_id"] == case_id]

    def verify_hash(self, artifact_id, sha256):
        exp = self.arts[artifact_id]["sha256"]
        return {"artifact_id": artifact_id, "status": STATUS_OK if exp == sha256 else STATUS_KO,
                "expected_sha256": exp, "provided_sha256": sha256}


@pytest.fixture
def setup(tmp_path):
    storage, client = CaseStorage(root=tmp_path), FakeClient()
    parent = None
    for name, typ, fn, data in [("original", "ORIGINAL_IMAGE", "original.png", b"img"),
                                ("prediction", "PREDICTION", "prediction.json", b'{"c": "glioma"}')]:
        ref = storage.save("CASE_001", fn, data)
        art = {"artifact_id": f"ART_CASE_001_{name}", "case_id": "CASE_001", "artifact_type": typ,
               "sha256": sha256_file(storage.path("CASE_001", fn)), "storage_ref": ref,
               "parent_artifact_id": parent}
        client.register_artifact(art)
        parent = art["artifact_id"]
    return storage, client


def test_cas_intact(setup):
    storage, client = setup
    r = verify_case("CASE_001", client, storage)
    assert r["status"] == STATUS_OK
    assert all(a["status"] == STATUS_OK for a in r["artifacts"])


def test_fichier_modifie_est_detecte(setup):
    storage, client = setup
    storage.path("CASE_001", "prediction.json").write_bytes(b'{"c": "notumor"}')
    r = verify_case("CASE_001", client, storage)
    assert r["status"] == STATUS_KO
    by_id = {a["artifact_id"]: a for a in r["artifacts"]}
    assert by_id["ART_CASE_001_prediction"]["reason"] == "HASH_MISMATCH"
    assert by_id["ART_CASE_001_original"]["status"] == STATUS_OK


def test_fichier_supprime_est_detecte(setup):
    storage, client = setup
    storage.path("CASE_001", "original.png").unlink()
    r = verify_case("CASE_001", client, storage)
    assert r["status"] == STATUS_KO
    assert r["artifacts"][0]["reason"] == "FILE_MISSING"


def test_cas_inconnu(setup):
    storage, client = setup
    with pytest.raises(LookupError):
        verify_case("CASE_999", client, storage)
