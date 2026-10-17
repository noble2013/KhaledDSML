"""Notebook executable-source safeguards and release pipeline regression tests.

The actual science notebooks are fully RUN under the separate CI notebook-smoke
job. These quick tests validate the runner's rejection of unsafe/invalid cells.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location(
    "khaled_notebook_smoke",ROOT/"tools/notebook_smoke.py")
smoke=importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(smoke)


def _write_nb(path,cells):
    obj={"nbformat":4,"nbformat_minor":5,"metadata":{},
         "cells":[
             {"cell_type":"code","metadata":{},"execution_count":None,
              "outputs":[],"source":[s]} for s in cells]}
    path.write_text(json.dumps(obj),encoding="utf-8")
    return path


@pytest.mark.parametrize("nb",smoke.NOTEBOOKS)
def test_modern_teaching_notebooks_have_multiple_compilable_research_cells(nb):
    cells=smoke.extract_cells(ROOT/"notebooks"/nb)
    assert len(cells)>=3
    for index,code in cells:
        assert "git+https" not in code
        compile(code,f"{nb}:cell{index}","exec")


def test_runner_skips_only_initial_standalone_pip(tmp_path):
    file=_write_nb(tmp_path/"notebook.ipynb",[
        "%pip install -q numpy","value=1","assert value==1"])
    result=smoke.extract_cells(file)
    assert len(result)==2
    assert result[0][1]=="value=1"


@pytest.mark.parametrize("cells",[
    ["value=2","%pip install -q numpy"],
    ["%pip install -q numpy\nimport os","value=1"],
    ["!rm -rf example","value=1"],
    ["%matplotlib inline","value=1"],
])
def test_runner_rejects_unsupported_magic_sequences(tmp_path,cells):
    file=_write_nb(tmp_path/"bad.ipynb",cells)
    with pytest.raises(ValueError,match="pip|magics|No executable"):
        smoke.extract_cells(file)


def test_deployment_workflows_must_run_research_smoke_and_guard_refs():
    ci=(ROOT/".github/workflows/python-tests.yml").read_text(encoding="utf-8")
    testpy=(ROOT/".github/workflows/testpypi.yml").read_text(encoding="utf-8")
    stable=(ROOT/".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "notebook-smoke:" in ci
    assert "needs: [test, distributed-integration, notebook-smoke]" in ci
    assert "python tools/notebook_smoke.py" in ci
    assert "python tools/notebook_smoke.py" in stable
    assert "python tools/notebook_smoke.py" in testpy
    assert "refs/heads/main" in testpy
    assert 'git merge-base --is-ancestor' in stable
    assert "environment:" in stable and "name: pypi" in stable

def test_entire_advanced_notebook_suite_is_mandatory():
    assert len(smoke.NOTEBOOKS)==10
    assert smoke.NOTEBOOKS[0].startswith("14_")
    assert smoke.NOTEBOOKS[-1].startswith("23_")
