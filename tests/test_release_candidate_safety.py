"""Release-candidate integrity tests: source, publication and archives.

These checks avoid any network calls or external publishing. They validate
production guards separately from the scientific correctness of methods.
"""
from __future__ import annotations

from pathlib import Path
import csv
import importlib.util
import io
import tarfile
import zipfile
import pytest


ROOT=Path(__file__).resolve().parents[1]


def _module(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


release=_module("release_gate_tests","tools/release_gate.py")
distcheck=_module("check_dist_tests","tools/check_distributions.py")


def test_release_candidate_source_version_is_single_source_of_truth():
    assert release.read_versions(ROOT)=="0.10.0rc1"
    data=release.validate_trace(ROOT)
    assert data["catalogue_rows"]==256
    assert data["mapped_candidate_rows"]==256
    assert data["restricted_substitute_ids"]==(85,117)
    assert data["scientifically_signed_off"] is None
    assert release.validate_target("0.10.0rc1","source").startswith("SOURCE")


def test_source_version_mismatch_and_pep440_bad_versions_rejected(tmp_path):
    (tmp_path/"src/khaled_dsml").mkdir(parents=True)
    py=tmp_path/"pyproject.toml"
    api=tmp_path/"src/khaled_dsml/__init__.py"
    py.write_text('[project]\nname = "khaled-dsml"\nversion = "0.10.0rc1"\n',
                  encoding="utf-8")
    api.write_text('__version__ = "0.10.0rc2"\n',encoding="utf-8")
    with pytest.raises(ValueError,match="disagrees"):
        release.read_versions(tmp_path)
    api.write_text('__version__ = "0.10.0rc1"\n',encoding="utf-8")
    assert release.read_versions(tmp_path)=="0.10.0rc1"
    py.write_text('[project]\nversion = "v0.10"\n',encoding="utf-8")
    api.write_text('__version__ = "v0.10"\n',encoding="utf-8")
    with pytest.raises(ValueError,match="canonical"):
        release.read_versions(tmp_path)


@pytest.mark.parametrize("bad",[
    {"tag":"v0.10.0rc1","prerelease":"false"},
    {"tag":"v0.10.0","prerelease":"false"},
    {"tag":"v0.10.0rc1","prerelease":"true"},
    {"tag":None,"prerelease":"false"},
])
def test_production_rejects_every_release_candidate_even_with_stable_flag(bad):
    with pytest.raises(ValueError):
        release.validate_target("0.10.0rc1","pypi",**bad)


def test_production_only_accepts_exact_stable_release_not_flagged_prerelease():
    assert release.validate_target("0.10.0","pypi",
                                   tag="v0.10.0",prerelease="false").startswith("STABLE")
    for options in [
        {"tag":"v0.10.0","prerelease":"true"},
        {"tag":"v0.10.1","prerelease":"false"},
        {"tag":"0.10.0","prerelease":"false"},
    ]:
        with pytest.raises(ValueError):
            release.validate_target("0.10.0","pypi",**options)


def test_testpypi_only_accepts_manual_confirmed_prerelease():
    for version in ("0.10.0rc1","0.10.0b2","0.11.0.dev1"):
        assert release.validate_target(version,"testpypi",
                                       confirm="TESTPYPI").startswith("TESTPYPI")
    for confirm in (None,"Yes","pypi","TESTPYPI ","TESTPYPI\n"):
        with pytest.raises(ValueError):
            release.validate_target("0.10.0rc1","testpypi",confirm=confirm)
    with pytest.raises(ValueError):
        release.validate_target("0.10.0","testpypi",confirm="TESTPYPI")


def test_reject_missing_source_candidate_and_weak_substitute_warning(tmp_path):
    (tmp_path/"docs").mkdir()
    path=tmp_path/release.CATALOGUE
    rows=(ROOT/release.CATALOGUE).read_text(encoding="utf-8-sig").splitlines()
    (tmp_path/"MANIFEST.in").write_text(
        "recursive-include docs *.md *.tsv\n",encoding="utf-8")
    path.write_text("\n".join(rows)+"\n",encoding="utf-8")
    assert release.validate_trace(tmp_path)["mapped_candidate_rows"]==256
    col=rows[0].split("\t")
    header=rows[0].split("\t")
    index=header.index("exact_public_api_candidates")
    row1=rows[1].split("\t")
    row1[index]=""
    rows[1]="\t".join(row1)
    path.write_text("\n".join(rows)+"\n",encoding="utf-8")
    with pytest.raises(ValueError,match="without linked"):
        release.validate_trace(tmp_path)
    rows[1]=(ROOT/release.CATALOGUE).read_text(encoding="utf-8-sig").splitlines()[1]
    note_idx=header.index("triage_not_scientific_signoff")
    row85=rows[85].split("\t")
    row85[note_idx]="Misleading full coverage assertion"
    rows[85]="\t".join(row85)
    path.write_text("\n".join(rows)+"\n",encoding="utf-8")
    with pytest.raises(ValueError,match="reject observed-power"):
        release.validate_trace(tmp_path)


def _tiny_archives(tmp_path,version="0.10.0rc1",include_trace=True):
    wheel=tmp_path/f"khaled_dsml-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel,"w") as z:
        z.writestr("khaled_dsml/__init__.py",f'__version__ = "{version}"\n')
        z.writestr(f"khaled_dsml-{version}.dist-info/METADATA",
                   f"Metadata-Version: 2.3\nName: khaled-dsml\nVersion: {version}\n")
    archive=tmp_path/f"khaled_dsml-{version}.tar.gz"
    with tarfile.open(archive,"w:gz") as t:
        root=f"khaled_dsml-{version}/"
        needed=[
            "pyproject.toml","README.md",
            "docs/ZERO_INFLATION_AND_PRECISION_0_10_AR_EN.md",
            "src/khaled_dsml/__init__.py",
        ]
        if include_trace: needed.append(distcheck.TRACE)
        for file in needed:
            payload=("id\tname\n"+"\n".join(f"{j}\ttest" for j in range(1,257))+"\n"
                     if file==distcheck.TRACE else file+"\n").encode()
            inf=tarfile.TarInfo(root+file)
            inf.size=len(payload)
            t.addfile(inf,io.BytesIO(payload))


def test_distribution_inspection_detects_evidence_and_version_alignment(tmp_path):
    _tiny_archives(tmp_path)
    r=distcheck.inspect_dist(tmp_path,"0.10.0rc1")
    assert r["source_catalogue_rows"]==256
    assert r["wheel"].endswith(".whl")
    with pytest.raises(ValueError,match="Wheel version"):
        distcheck.inspect_dist(tmp_path,"0.10.0rc2")


def test_distribution_inspection_blocks_missing_catalogue_in_source_archive(tmp_path):
    _tiny_archives(tmp_path,include_trace=False)
    with pytest.raises(ValueError,match="missing"):
        distcheck.inspect_dist(tmp_path,"0.10.0rc1")


def test_source_publication_workflows_are_guarded():
    release_yaml=(ROOT/".github/workflows/release.yml").read_text(encoding="utf-8")
    test_yaml=(ROOT/".github/workflows/testpypi.yml").read_text(encoding="utf-8")
    ci_yaml=(ROOT/".github/workflows/python-tests.yml").read_text(encoding="utf-8")
    assert "--target pypi" in release_yaml
    assert "RELEASE_IS_PRERELEASE" in release_yaml
    assert "--target testpypi" in test_yaml
    assert "confirm_publish" in test_yaml
    assert "environment:" in release_yaml and "name: pypi" in release_yaml
    assert "name: testpypi" in test_yaml
    assert "python tools/check_distributions.py dist" in ci_yaml
    for x in (release_yaml,test_yaml):
        assert "exports,unitroot,quantiles" in x
