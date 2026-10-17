"""Notebook references must not send students to deleted feature branches."""
from __future__ import annotations

from pathlib import Path
import json
import re
import pytest


ROOT=Path(__file__).resolve().parents[1]
NAMES=[
    "14_bayesian_binomial_studio.ipynb",
    "15_monitoring_meta_studio.ipynb",
    "16_research_integrity_studio.ipynb",
    "17_ml_validation_studio.ipynb",
    "18_general_linear_models_studio.ipynb",
    "19_final_research_studio.ipynb",
    "20_specialized_inference_and_exports.ipynb",
    "21_pp_grimmer_pcurve_studio.ipynb",
    "22_sem_shap_tdigest_bayes.ipynb",
    "23_zero_inflation_vuong_precision.ipynb",
]


@pytest.mark.parametrize("file_name",NAMES)
def test_recent_colab_notebooks_are_complete_and_use_active_source(file_name):
    text=(ROOT/"notebooks"/file_name).read_text(encoding="utf-8")
    notebook=json.loads(text)
    assert notebook["nbformat"]==4
    assert notebook["cells"]
    assert notebook["metadata"]["kernelspec"]["name"]=="python3"
    assert "@feature/v0.10-" not in text
    assert "0.10.0.dev0" not in text
    assert "0.10.0rc1" in text
    assert any(cell["cell_type"]=="code" for cell in notebook["cells"])
    for cell in notebook["cells"]:
        assert isinstance(cell["source"],list)
        assert all(isinstance(s,str) for s in cell["source"])
        if cell["cell_type"]=="code":
            assert cell["execution_count"] is None
            assert not cell["outputs"]
    remote_refs=re.findall(
        r'git\+https://github\.com/noble2013/KhaledDSML\.git@([^"\s]+)',
        text,
    )
    if remote_refs:
        assert set(remote_refs)=={"main"}, f"Invalid source install refs: {remote_refs}"


def test_release_docs_acknowledge_prerelease_and_scientific_limitations():
    readme=(ROOT/"README.md").read_text(encoding="utf-8")
    docs=(ROOT/"docs/RELEASE_CANDIDATE_0_10_AR_EN.md").read_text(encoding="utf-8")
    guide=(ROOT/"docs/PUBLISH_AR.md").read_text(encoding="utf-8")
    assert "0.10.0rc1" in readme and "0.9.0" in readme
    assert "256" in docs and "#85" in docs and "#117" in docs
    assert "scientific" in docs.lower()
    assert "TESTPYPI" in guide
    assert "Trusted" in guide
