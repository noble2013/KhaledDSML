"""Research provenance and file-integrity checks."""
import hashlib
import json
import pytest

from khaled_dsml.reproducibility import (
    reproducibility_manifest, save_reproducibility_manifest,
    sha256_file, software_versions, verify_reproducibility_manifest,
)


def test_sha256_is_streaming_and_matches_hashlib(tmp_path):
    data = b"KhaledDSML" * 250
    file = tmp_path / "file.bin"
    file.write_bytes(data)
    assert sha256_file(file, block_size=7) == hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError):
        sha256_file(file, block_size=0)


def test_reproducibility_manifest_is_deterministic_and_private(tmp_path):
    a = tmp_path / "dataset.csv"
    b = tmp_path / "answers.txt"
    a.write_text("value\n1\n2\n", encoding="utf-8")
    b.write_text("research", encoding="utf-8")
    manifest = reproducibility_manifest([a, b], seed=42, packages=["python", "numpy"])
    again = reproducibility_manifest([b, a], seed=42, packages=["python", "numpy"])
    assert manifest == again
    assert manifest["seed"] == 42
    assert manifest["software"]["numpy"] is not None
    assert all("/" not in item["filename"] for item in manifest["inputs"])
    assert str(tmp_path) not in str(manifest)
    assert "interpretation_en" in manifest and "interpretation_ar" in manifest
    assert "created_utc" not in manifest


def test_manifest_save_verify_and_detect_input_change(tmp_path):
    data = tmp_path / "original.csv"
    data.write_text("x\n5\n", encoding="utf-8")
    manifest = reproducibility_manifest(data, seed=2026)
    saved = save_reproducibility_manifest(manifest, tmp_path / "manifest.json")
    parsed = json.loads(saved.read_text(encoding="utf-8"))
    assert parsed["inputs"][0]["filename"] == "original.csv"
    assert verify_reproducibility_manifest(parsed, data)["inputs_match"]
    data.write_text("x\n6\n", encoding="utf-8")
    result = verify_reproducibility_manifest(parsed, data)
    assert not result["inputs_match"]
    assert isinstance(result["interpretation_ar"], str)
    assert isinstance(result["interpretation_en"], str)
    assert "not" in result["interpretation_en"] or "differ" in result["interpretation_en"]


def test_manifest_rejects_invalid_path_and_seed(tmp_path):
    path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        reproducibility_manifest(path)
    with pytest.raises(ValueError):
        reproducibility_manifest([])
    path.write_text("ok", encoding="utf-8")
    with pytest.raises(ValueError):
        reproducibility_manifest(path, seed=-1)
    with pytest.raises(ValueError):
        software_versions([""])


def test_timestamp_is_opt_in(tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("col\n1\n", encoding="utf-8")
    record = reproducibility_manifest(path, include_timestamp=True)
    assert "created_utc" in record
    assert record["created_utc"].endswith("+00:00")
