"""Independent numerical checks of drift and streaming analysis."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.spatial.distance import jensenshannon

from khaled_dsml import (
    distribution_drift_profile, welford_streaming_summary,
    sample_ratio_mismatch, render_analysis_dashboard,
)


def test_drift_psi_js_wasserstein_ks_match_separate_references():
    reference = np.repeat(np.arange(10, dtype=float), 20)
    current = np.r_[np.repeat(np.arange(10, dtype=float), 16),
                    np.repeat(np.arange(7, 10, dtype=float), 15)]
    result = distribution_drift_profile(reference, current, bins=5)
    table = result["table"]
    rp = table["reference_share_smoothed"].to_numpy()
    cp = table["current_share_smoothed"].to_numpy()
    assert result["population_stability_index"] == pytest.approx(
        np.sum((cp-rp)*np.log(cp/rp)), abs=1e-13
    )
    assert result["jensen_shannon_divergence_bits"] == pytest.approx(
        jensenshannon(rp, cp, base=2.)**2, abs=1e-13
    )
    assert result["wasserstein_distance"] == pytest.approx(
        stats.wasserstein_distance(reference, current)
    )
    ks = stats.ks_2samp(reference, current)
    assert result["ks_statistic"] == pytest.approx(ks.statistic)
    assert result["ks_p_value"] == pytest.approx(ks.pvalue)
    assert table["reference_count"].sum() == len(reference)
    assert table["current_count"].sum() == len(current)
    assert result["jensen_shannon_divergence_bits"] >= 0
    assert result["n_bins"] <= 5


def test_drift_unchanged_distributions_are_zero_even_with_repeated_values():
    rng = np.random.default_rng(22)
    reference = rng.normal(size=140)
    result = distribution_drift_profile(reference, reference.copy(), bins=10)
    assert result["population_stability_index"] == pytest.approx(0, abs=1e-14)
    assert result["jensen_shannon_divergence_bits"] == pytest.approx(0, abs=1e-14)
    assert result["wasserstein_distance"] == pytest.approx(0)
    assert result["ks_p_value"] == 1
    assert "p-value" in result["interpretation_en"]


def test_drift_out_of_reference_bounds_not_dropped_and_shift_detected():
    rng = np.random.default_rng(11)
    reference = rng.normal(size=320)
    current = rng.normal(loc=4, size=310)
    result = distribution_drift_profile(reference, current, bins=10)
    assert result["table"]["current_count"].sum() == len(current)
    assert result["population_stability_index"] > 0
    assert result["wasserstein_distance"] > 3.0
    assert result["ks_p_value"] < 1e-20


@pytest.mark.parametrize("ref,cur,params", [
    (np.ones(50), np.arange(50), {}),
    (np.arange(10), np.arange(30), {}),
    (np.r_[np.arange(30), np.nan], np.arange(35), {}),
    (np.arange(30), np.arange(30), {"bins": 1}),
    (np.arange(30), np.arange(30), {"smoothing": 0}),
    (np.arange(30), np.arange(30), {"alpha": True}),
])
def test_drift_rejects_invalid_inputs(ref, cur, params):
    with pytest.raises(ValueError):
        distribution_drift_profile(ref, cur, **params)


def test_welford_partition_invariance_and_ddof():
    rng = np.random.default_rng(13)
    sample = rng.normal(loc=100000., scale=.125, size=2000)
    chunks = [sample[:6], np.array([]), sample[6:9], sample[9:120], sample[120:]]
    result = welford_streaming_summary(iter(chunks))
    assert result["n_samples"] == 2000
    assert result["mean"] == pytest.approx(sample.mean(), abs=1e-9)
    assert result["variance_sample"] == pytest.approx(sample.var(ddof=1), rel=1e-10)
    assert result["std_sample"] == pytest.approx(sample.std(ddof=1), rel=1e-10)
    single = welford_streaming_summary([sample])
    assert single["variance_population"] == pytest.approx(sample.var(ddof=0))
    assert "p_value" not in result


@pytest.mark.parametrize("chunks", [
    [], [[1.]], [[np.nan, 2.]], [[1., np.inf]], [[[1., 2.], [3., 4.]]],
])
def test_welford_rejects_insufficient_or_invalid_data(chunks):
    with pytest.raises(ValueError):
        welford_streaming_summary(chunks)


def test_sample_ratio_mismatch_matches_scipy_chisquare():
    obs = [560, 440]
    r = sample_ratio_mismatch(obs, [.5, .5])
    ref = stats.chisquare(obs, f_exp=[500, 500])
    assert r["statistic"] == pytest.approx(ref.statistic)
    assert r["p_value"] == pytest.approx(ref.pvalue)
    assert r["reject_null"] is True
    assert r["degrees_of_freedom"] == 1
    assert r["table"]["expected"].sum() == pytest.approx(sum(obs))
    assert "p_value" in r


@pytest.mark.parametrize("obs,exp", [
    ([3, 4], [.5, .5]), ([100, 100], [0, 1]),
    ([50, 50], [.3, .3]), ([100, -1], [.5, .5]),
    ([10.0, 10.0], [.5, .5]), ([100, 200, 300], [.5, .5]),
])
def test_srm_rejects_unsupported_inputs(obs, exp):
    with pytest.raises(ValueError):
        sample_ratio_mismatch(obs, exp)


def test_monitoring_results_render_offline_bilingual():
    rng = np.random.default_rng(8)
    r = distribution_drift_profile(rng.normal(size=100), rng.normal(size=100))
    for result in (r, welford_streaming_summary([[1,2], [3,4]]),
                   sample_ratio_mismatch([60,40],[.5,.5])):
        html = render_analysis_dashboard(result, theme="aurora")
        assert 'dir="rtl"' in html
        assert result["interpretation_ar"] in html
        assert result["interpretation_en"] in html
        assert "<table" in html or result["test"] == "Welford streaming summary"
