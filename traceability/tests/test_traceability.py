import hashlib
import pytest
from traceability.hashing import sha256_file
from traceability.storage import CaseStorage

H_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
H_EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_sha256_abc(tmp_path):
    f = tmp_path / "a.txt"
    f.write_bytes(b"abc")
    assert sha256_file(f) == H_ABC


def test_sha256_fichier_vide(tmp_path):
    f = tmp_path / "vide.bin"
    f.write_bytes(b"")
    assert sha256_file(f) == H_EMPTY


def test_sha256_gros_fichier_lu_par_blocs(tmp_path):
    data = b"x" * (3 * 1024 * 1024 + 17)
    f = tmp_path / "gros.bin"
    f.write_bytes(data)
    assert sha256_file(f) == hashlib.sha256(data).hexdigest()


def test_un_octet_change_le_hash(tmp_path):
    f = tmp_path / "p.json"
    f.write_bytes(b'{"predicted_class": "glioma"}')
    avant = sha256_file(f)
    f.write_bytes(b'{"predicted_class": "gliomb"}')
    assert sha256_file(f) != avant


def test_save_renvoie_storage_ref_et_ecrit(tmp_path):
    s = CaseStorage(root=tmp_path)
    ref = s.save("CASE_001", "prediction.json", b"{}")
    assert ref == "storage/cases/CASE_001/prediction.json"
    assert s.path("CASE_001", "prediction.json").read_bytes() == b"{}"


def test_ecrasement_refuse(tmp_path):
    s = CaseStorage(root=tmp_path)
    s.save("CASE_001", "original.png", b"v1")
    with pytest.raises(FileExistsError):
        s.save("CASE_001", "original.png", b"v2")
    assert s.path("CASE_001", "original.png").read_bytes() == b"v1"


@pytest.mark.parametrize("case_id", ["Dupont", "CASE_1", "case_001", "../CASE_001", ""])
def test_case_id_invalide(tmp_path, case_id):
    with pytest.raises(ValueError):
        CaseStorage(root=tmp_path).save(case_id, "a.json", b"")


@pytest.mark.parametrize("name", ["../x.json", "a/b.json", "..", ".env", ""])
def test_nom_de_fichier_invalide(tmp_path, name):
    with pytest.raises(ValueError):
        CaseStorage(root=tmp_path).save("CASE_001", name, b"")
