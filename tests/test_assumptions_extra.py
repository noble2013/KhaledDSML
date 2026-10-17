"""Assumption diagnostics with reference values and bilingual descriptions."""
import numpy as np
import pandas as pd
import pytest

from khaled_dsml.assumptions_extra import (
    anderson_darling_normality, lilliefors_normality, runs_test,
    durbin_watson_statistic, kmo_sampling_adequacy,
    bartlett_sphericity_test, condition_number_vif,
)


def test_anderson_critical_threshold_no_fake_pvalue():
    rng = np.random.default_rng(123)
    r = anderson_darling_normality(rng.normal(size=80))
    assert r["p_value"] is None
    assert r["critical_value"] > 0
    assert "normality" in r["interpretation_en"]
    with pytest.raises(ValueError):
        anderson_darling_normality(rng.normal(size=30), alpha=.123)


def test_lilliefors_with_optional_statsmodels():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(44)
    r = lilliefors_normality(rng.normal(size=100))
    assert 0 <= r.p_value <= 1
    assert "normal" in r.null_hypothesis_en


def test_runs_test_signal_and_nonrandom():
    x = np.tile([0,1], 10)
    r = runs_test(x)
    assert r.p_value < .05
    assert "random" in r.null_hypothesis_en
    assert r.sample_size == 20
    with pytest.raises(ValueError):
        runs_test([1] * 15)


def test_durbin_watson_no_fake_pvalue():
    r = durbin_watson_statistic([1, -1, 1, -1, 1])
    assert 0 <= r["statistic"] <= 4
    assert r["p_value"] is None
    assert "Durbin" in r["interpretation_en"]
    with pytest.raises(ValueError):
        durbin_watson_statistic([0,0,0])


def test_kmo_and_bartlett_reference():
    rng = np.random.default_rng(17)
    latent = rng.normal(size=180)
    df = pd.DataFrame({
        "x": latent + rng.normal(scale=.3,size=180),
        "y": latent + rng.normal(scale=.3,size=180),
        "z": latent + rng.normal(scale=.3,size=180),
    })
    kmo = kmo_sampling_adequacy(df)
    assert 0 < kmo["kmo"] <= 1
    assert list(kmo["by_variable"].index) == ["x","y","z"]
    bart = bartlett_sphericity_test(df)
    assert bart.statistic > 0 and bart.p_value < .05
    assert "identity" in bart.null_hypothesis_en


def test_condition_number_and_vif():
    rng = np.random.default_rng(99)
    df = pd.DataFrame(rng.normal(size=(100, 3)), columns=["x","y","z"])
    r = condition_number_vif(df)
    assert all(r["vif"] >= 1 - 1e-9)
    assert r["condition_index"] >= 1
    df["z"] = df["x"] * 2
    with pytest.raises(ValueError, match="singular"):
        condition_number_vif(df)


def test_bad_data_rejected_for_kmo():
    with pytest.raises(ValueError):
        kmo_sampling_adequacy(pd.DataFrame({"x":[1,2],"y":[2,4]}))
