"""Reproducible *source evidence* trace for the 256-row catalogue.

A public callable and a test mention are NOT scientific sign-off. This script
does not modify the historically annotated reference catalogue or overstate
completion: a compound catalogue row may remain incomplete despite an API.
"""
from __future__ import annotations

import argparse
import csv
import importlib
from collections import Counter
from pathlib import Path


CATALOGUE_PATH = Path(__file__).resolve().parents[1] / "docs" / "CATALOG_256_SOURCE_TRACE_V010.tsv"
REQUIRED_COLUMNS = {
    "id", "domain", "catalogue_term_excerpt", "historical_status",
    "exact_public_api_candidates", "triage_not_scientific_signoff",
    "original_tracker_record_verbatim",
}


def load_catalogue_trace(path=CATALOGUE_PATH):
    """Read all 256 rows, checking IDs and structural integrity."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or not REQUIRED_COLUMNS <= set(reader.fieldnames):
            raise ValueError("Catalogue trace has missing required columns")
        rows = list(reader)
    if len(rows) != 256:
        raise ValueError(f"Expected 256 catalogue items, got {len(rows)}")
    ids = [int(r["id"]) for r in rows]
    if ids != list(range(1, 257)):
        raise ValueError("Catalogue ids must be exactly 1..256 in order")
    if len({r["domain"] for r in rows}) != 12:
        raise ValueError("Expected exactly 12 research domains")
    if any(not r["catalogue_term_excerpt"] for r in rows):
        raise ValueError("Missing catalogue term")
    return rows


def _candidate_names(row):
    """Names copied from old tracker notes are hypotheses until checked."""
    return [
        v.split(" [")[0].strip()
        for v in row["exact_public_api_candidates"].split(";")
        if v.strip()
    ]


def build_source_trace(rows, test_dir, *, package=None):
    """Return row evidence without ever assigning a scientific completion label."""
    if package is None:
        package = importlib.import_module("khaled_dsml")
    test_dir = Path(test_dir)
    texts = {
        path.name: path.read_text(encoding="utf-8")
        for path in test_dir.glob("test_*.py")
    }
    evidence = []
    for row in rows:
        names = _candidate_names(row)
        found = [name for name in names if callable(getattr(package, name, None))]
        invalid = sorted(set(names) - set(found))
        related = [
            filename for filename, content in texts.items()
            if any(name in content for name in found)
        ]
        evidence.append({
            "id": int(row["id"]),
            "domain": row["domain"],
            "catalogue_term": row["catalogue_term_excerpt"],
            "historical_status": row["historical_status"],
            "candidate_api_names": "; ".join(names),
            "verified_importable_callables": "; ".join(found),
            "candidate_names_missing_from_api": "; ".join(invalid),
            "test_files_mentioning_candidate": "; ".join(sorted(related)),
            "review_status": "REQUIRES INDEPENDENT SCIENTIFIC SIGN-OFF",
        })
    return evidence


def evidence_summary(evidence):
    """Mechanical inventory counts, NOT method-completion counts."""
    return {
        "total_catalogue_items": len(evidence),
        "source_api_candidate_rows": sum(
            bool(row["verified_importable_callables"]) for row in evidence
        ),
        "test_mentioned_candidate_rows": sum(
            bool(row["test_files_mentioning_candidate"]) for row in evidence
        ),
        "missing_candidate_names": sum(
            len(row["candidate_names_missing_from_api"].split("; "))
            if row["candidate_names_missing_from_api"] else 0
            for row in evidence
        ),
        "domains": dict(sorted(Counter(row["domain"] for row in evidence).items())),
        "scientifically_signed_off": None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(CATALOGUE_PATH))
    parser.add_argument("--tests", default=str(CATALOGUE_PATH.parents[1] / "tests"))
    parser.add_argument("--output", help="Optional output CSV with source-level evidence")
    args = parser.parse_args(argv)
    evidence = build_source_trace(load_catalogue_trace(args.input), args.tests)
    summary = evidence_summary(evidence)
    for key, value in summary.items():
        print(f"{key}: {value}")
    if args.output:
        path = Path(args.output)
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(evidence[0]))
            writer.writeheader()
            writer.writerows(evidence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
