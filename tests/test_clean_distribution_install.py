"""Clean distribution installation gates: prevent source-tree false successes.

Actual wheel and sdist installations run on CI in separate real venvs.
Unit tests here check reject paths and inspect verification script boundaries.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location(
    "khaled_install_dist_smoke",ROOT/"tools/install_dist_smoke.py")
smoke=importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(smoke)


def test_verification_runs_in_isolated_python_outside_checkout():
    assert "source.is_relative_to(prefix)" in smoke.VERIFY
    assert "importlib.metadata" in smoke.VERIFY
    assert 'metadata.version("khaled-dsml")' in smoke.VERIFY
    assert 'dsml.poisson_excess_zero_score' in smoke.VERIFY
    assert 'dsml.post_study_precision_scenarios' in smoke.VERIFY


def test_both_archives_required_and_exactly_matched_version(tmp_path):
    version="0.10.0"
    wheel=tmp_path/f"khaled_dsml-{version}-py3-none-any.whl"
    wheel.write_bytes(b"illustrative fixture")
    with pytest.raises(ValueError,match="one matching wheel and sdist"):
        smoke.find_archives(tmp_path,version)
    sdist=tmp_path/f"khaled_dsml-{version}.tar.gz"
    sdist.write_bytes(b"illustrative fixture")
    result=smoke.find_archives(tmp_path,version)
    assert result["wheel"]==wheel
    assert result["sdist"]==sdist
    duplicate=tmp_path/f"khaled_dsml-{version}-py2-none-any.whl"
    duplicate.write_bytes(b"invalid extra wheel")
    with pytest.raises(ValueError,match="one matching wheel"):
        smoke.find_archives(tmp_path,version)


@pytest.mark.parametrize("modes",[
    (),("wheel","foreign"),("tar","sdist"),
])
def test_unrecognized_distribution_modes_rejected_before_install(tmp_path,modes):
    version="0.10.0"
    (tmp_path/f"khaled_dsml-{version}-py3-none-any.whl").write_bytes(b"x")
    (tmp_path/f"khaled_dsml-{version}.tar.gz").write_bytes(b"x")
    with pytest.raises(ValueError,match="mode"):
        smoke.run_install_checks(tmp_path,version,modes=modes)


def test_release_workflow_calls_both_clean_installs():
    ci=(ROOT/".github/workflows/python-tests.yml").read_text(encoding="utf-8")
    release=(ROOT/".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "python tools/install_dist_smoke.py dist" in ci
    assert "python tools/install_dist_smoke.py dist" in release
    assert "python tools/release_gate.py --target pypi --tag v0.10.0 --prerelease false" in ci
    assert "needs: [test, distributed-integration, notebook-smoke]" in ci


def test_stable_release_no_longer_dev_or_rc():
    import khaled_dsml
    assert khaled_dsml.__version__=="0.10.0"
