"""Guarantee structural completeness and reproducible source evidence for all 256 IDs."""
import csv
import importlib.util
from pathlib import Path

from khaled_dsml import henze_zirkler_normality_test, little_mcar_test


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "tools" / "catalogue_audit.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("khaled_catalogue_audit_tool", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_catalogue_contains_exactly_256_unique_items_and_12_domains():
    module = _load_script()
    rows = module.load_catalogue_trace()
    assert len(rows) == 256
    assert {int(r["id"]) for r in rows} == set(range(1, 257))
    assert len({r["domain"] for r in rows}) == 12


def test_source_trace_importability_and_evidence_are_not_signoff():
    module = _load_script()
    evidence = module.build_source_trace(module.load_catalogue_trace(), ROOT / "tests")
    assert len(evidence) == 256
    summary = module.evidence_summary(evidence)
    assert summary["total_catalogue_items"] == 256
    assert summary["source_api_candidate_rows"] >= 64
    assert summary["missing_candidate_names"] == 0
    assert summary["scientifically_signed_off"] is None
    assert all(row["review_status"] == "REQUIRES INDEPENDENT SCIENTIFIC SIGN-OFF"
               for row in evidence)
    # The source tracker is stale: Little MCAR became public in 0.9.
    assert "little_mcar_test" in evidence[1]["verified_importable_callables"]
    # Henze–Zirkler is newly available on the development branch.
    assert "henze_zirkler_normality_test" in evidence[21]["verified_importable_callables"]


def test_source_trace_cli_csv_is_readable_and_stays_unverified(tmp_path):
    module = _load_script()
    dest = tmp_path / "catalogue_source_evidence.csv"
    assert module.main(["--output", str(dest)]) == 0
    with dest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 256
    assert rows[0]["id"] == "1"
    assert rows[-1]["id"] == "256"
    assert all("SCIENTIFIC SIGN-OFF" in r["review_status"] for r in rows)
