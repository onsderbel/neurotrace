import pytest
from traceability.blockchain_client import BlockchainError
from traceability.fake_client import FakeBlockchainClient
from traceability.recorder import CHAIN, ArtifactRecorder, artifact_id_for, default_parent
from traceability.storage import CaseStorage
from traceability.verify import STATUS_KO, STATUS_OK, verify_case

CASE = "CASE_001"


def record_all(rec):
    for t, _, _ in CHAIN:
        rec.record(CASE, t, f"contenu {t}".encode())


@pytest.fixture
def env(tmp_path):
    client, storage = FakeBlockchainClient(), CaseStorage(root=tmp_path)
    return client, storage, ArtifactRecorder(client, storage)


def test_chaine_complete(env):
    client, storage, rec = env
    record_all(rec)
    history = client.get_case_history(CASE)
    assert [a["artifact_type"] for a in history] == [t for t, _, _ in CHAIN]
    assert history[0]["parent_artifact_id"] is None
    for prev, cur in zip(history, history[1:]):
        assert cur["parent_artifact_id"] == prev["artifact_id"]
    assert verify_case(CASE, client, storage)["status"] == STATUS_OK


def test_identifiants_selon_le_contrat():
    assert artifact_id_for(CASE, "PREDICTION") == "ART_CASE_001_prediction"
    assert default_parent(CASE, "PREDICTION") == "ART_CASE_001_preprocessed"
    assert default_parent(CASE, "ORIGINAL_IMAGE") is None


def test_meme_enregistrement_deux_fois_est_idempotent(env):
    client, _, rec = env
    a1 = rec.record(CASE, "ORIGINAL_IMAGE", b"img")
    a2 = rec.record(CASE, "ORIGINAL_IMAGE", b"img")
    assert a1["sha256"] == a2["sha256"]
    assert len(client.get_case_history(CASE)) == 1


def test_donnees_differentes_refusees(env):
    _, _, rec = env
    rec.record(CASE, "ORIGINAL_IMAGE", b"img")
    with pytest.raises(FileExistsError):
        rec.record(CASE, "ORIGINAL_IMAGE", b"autre image")


def test_reprise_apres_panne_blockchain(env):
    client, storage, rec = env
    real, calls = client.register_artifact, {"n": 0}

    def flaky(artifact):
        calls["n"] += 1
        if calls["n"] == 1:
            raise BlockchainError("Service blockchain injoignable")
        return real(artifact)

    client.register_artifact = flaky
    with pytest.raises(BlockchainError):
        rec.record(CASE, "ORIGINAL_IMAGE", b"img")
    assert storage.path(CASE, "original.png").exists()          # déjà rangé avant la panne
    art = rec.record(CASE, "ORIGINAL_IMAGE", b"img")             # nouvel essai : pas d'erreur de fichier
    assert art["artifact_id"] == "ART_CASE_001_original"


def test_modification_detectee_apres_enregistrement(env):
    client, storage, rec = env
    record_all(rec)
    storage.path(CASE, "prediction.json").write_bytes(b"truque")
    assert verify_case(CASE, client, storage)["status"] == STATUS_KO


def test_type_inconnu(env):
    _, _, rec = env
    with pytest.raises(ValueError):
        rec.record(CASE, "FICHIER_BIZARRE", b"x")


def test_fake_refuse_parent_inconnu():
    with pytest.raises(BlockchainError, match="introuvable"):
        FakeBlockchainClient().register_artifact(
            {"artifact_id": "A", "case_id": CASE, "artifact_type": "PREPROCESSED_IMAGE",
             "sha256": "0" * 64, "parent_artifact_id": "ART_X", "storage_ref": "r"})


def test_fake_refuse_ecrasement():
    c = FakeBlockchainClient()
    art = {"artifact_id": "A", "case_id": CASE, "artifact_type": "ORIGINAL_IMAGE",
           "sha256": "0" * 64, "storage_ref": "r"}
    c.register_artifact(art)
    with pytest.raises(BlockchainError, match="déjà enregistré"):
        c.register_artifact(art)
