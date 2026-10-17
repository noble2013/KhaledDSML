"""Local, read-only KhaledDSML version / publication safety gate.

Runs on Python 3.10+ with standard library only. The 256-row TSV lists
source candidates, NEVER verified scientific completeness. Publishing is
kept separate from source checks and requires explicit GitHub controls.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = Path("docs/CATALOG_256_SOURCE_TRACE_V010.tsv")
MANIFEST = Path("MANIFEST.in")


def read_versions(root=ROOT):
    root=Path(root)
    toml=(root/"pyproject.toml").read_text(encoding="utf-8")
    package=(root/"src/khaled_dsml/__init__.py").read_text(encoding="utf-8")
    a=re.search(r'^version\s*=\s*"([^"]+)"\s*$',toml,re.MULTILINE)
    b=re.search(r'^__version__\s*=\s*"([^"]+)"\s*$',package,re.MULTILINE)
    if a is None or b is None:
        raise ValueError("Missing unambiguous project version or package __version__")
    if a.group(1)!=b.group(1):
        raise ValueError("pyproject.toml version disagrees with root package __version__")
    version=a.group(1)
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:a\d+|b\d+|rc\d+|\.dev\d+)?",version):
        raise ValueError("Version does not follow canonical project PEP 440 scheme")
    return version


def validate_trace(root=ROOT):
    root=Path(root)
    with (root/CATALOGUE).open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f,delimiter="\t"))
    if len(rows)!=256 or [r.get("id") for r in rows] != list(map(str,range(1,257))):
        raise ValueError("Expected original 256 source catalogue IDs in stable order")
    if any(not r.get("exact_public_api_candidates","").strip() for r in rows):
        raise ValueError("Source catalogue has items without linked public candidate API")
    # Require unusually prominent scope warnings for the two *substitutes*.
    row85,row117=rows[84],rows[116]
    if "post_study_precision_scenarios" not in row85["exact_public_api_candidates"]:
        raise ValueError("Row #85 must map to an honest precision alternative")
    if "NOT observed" not in row85["triage_not_scientific_signoff"]:
        raise ValueError("Row #85 must explicitly reject observed-power substitution")
    if "vuong_strictly_nonnested" not in row117["exact_public_api_candidates"]:
        raise ValueError("Row #117 must map only to a restricted nonnested method")
    if "NOT Vuong test for Poisson-vs-ZIP" not in row117["triage_not_scientific_signoff"]:
        raise ValueError("Row #117 must explicitly reject invalid Vuong inference")
    if "recursive-include docs *.md *.tsv" not in (root/MANIFEST).read_text(encoding="utf-8"):
        raise ValueError("Original 256-row catalogue TSV missing from sdist MANIFEST.in")
    return {
        "catalogue_rows":len(rows),
        "mapped_candidate_rows":sum(bool(r["exact_public_api_candidates"].strip()) for r in rows),
        "scientifically_signed_off":None,
        "restricted_substitute_ids":(85,117),
    }


def validate_target(version, target, *, tag=None, confirm=None, prerelease=None):
    """Prevent accidental stable PyPI publication from a dev/RC source."""
    if target not in ("source","testpypi","pypi"):
        raise ValueError("target must be source, testpypi or pypi")
    if target=="source":
        return "SOURCE VERIFIED — no external publish action"
    if target=="testpypi":
        if confirm!="TESTPYPI":
            raise ValueError('TestPyPI requires explicit confirm text "TESTPYPI"')
        if not re.search(r"(?:a|b|rc|\.dev)\d+$",version):
            raise ValueError("TestPyPI release-candidate workflow accepts development/prerelease versions only")
        return "TESTPYPI PRE-RELEASE AUTHORIZED BY EXPLICIT USER ACTION"
    if tag!=f"v{version}":
        raise ValueError("Production GitHub Release tag must exactly match package version")
    if not re.fullmatch(r"\d+\.\d+\.\d+",version):
        raise ValueError("Production stable PyPI workflow rejects all development/prerelease versions")
    if prerelease!="false":
        raise ValueError("Production GitHub Release must not be marked prerelease")
    return "STABLE VERSION TAG VERIFIED — GitHub protected environment still required"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target",choices=("source","testpypi","pypi"),
                        default="source")
    parser.add_argument("--tag")
    parser.add_argument("--prerelease")
    parser.add_argument("--confirm")
    args=parser.parse_args(argv)
    try:
        version=read_versions()
        evidence=validate_trace()
        message=validate_target(version,args.target,tag=args.tag,
                                confirm=args.confirm,prerelease=args.prerelease)
    except (ValueError,OSError) as exc:
        print("RELEASE GATE BLOCKED:",exc,file=sys.stderr)
        return 1
    print("KhaledDSML",version)
    print("Source catalogue candidate rows:",evidence["mapped_candidate_rows"],
          "/ 256; restricted substitutes: #85, #117")
    print("Independent scientific sign-off is NOT claimed.")
    print(message)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
