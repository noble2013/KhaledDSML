"""Reference cases and input-validation checks for v0.3 time-series APIs."""
import numpy as np
import pytest
from scipy import stats

from khaled_dsml.timeseries_advanced import (
    arch_lm_test, box_pierce_test, forecast_accuracy_report,
    mann_kendall_trend_test, mean_absolute_scaled_error,
    pettitt_change_point_test,
)


def test_mann_kendall_monotonic_upward_and_downward():
    upward = mann_kendall_trend_test(list(range(20)))
    assert upward.reject_null
    assert upward.p_value < 0.001
    assert upward.statistic > 0
    assert "no monotonic trend" in upward.null_hypothesis_en.lower()
    downward = mann_kendall_trend_test(list(range(20, 0, -1)))
    assert downward.reject_null
    assert downward.statistic < 0
    assert "تنازلي" in downward.notes


def test_mann_kendall_ties_without_false_significance():
    values = [1, 2, 1, 3, 2, 1, 2, 3, 1, 2, 3, 2]
    r = mann_kendall_trend_test(values)
    assert 0 <= r.p_value <= 1
    assert np.isfinite(r.statistic)


def test_mann_kendall_rejects_constant_missing_and_too_short():
    for series in ([1] * 12, [1, 2, 3], [0] * 11 + [np.nan]):
        with pytest.raises(ValueError):
            mann_kendall_trend_test(series)


def test_pettitt_step_change_location_and_significance():
    values = [0] * 20 + [10] * 20
    r = pettitt_change_point_test(values)
    assert r["change_index"] == 19
    assert r["first_segment_size"] == 20
    assert r["reject_null"]
    assert r["p_value"] < 0.001
    assert isinstance(r["interpretation_ar"], str)
    assert isinstance(r["interpretation_en"], str)
    assert "independent observations" in r["interpretation_en"]


def test_pettitt_rejects_constant():
    with pytest.raises(ValueError):
        pettitt_change_point_test([3] * 12)


def test_box_pierce_matches_statsmodels_reference():
    pytest.importorskip("statsmodels")
    from statsmodels.stats.diagnostic import acorr_ljungbox
    x = np.tile([1.0, -1.0], 40)
    ours = box_pierce_test(x, lags=5)
    reference = acorr_ljungbox(x, lags=[5], boxpierce=True, return_df=True)
    assert ours.statistic == pytest.approx(reference["bp_stat"].iloc[0])
    assert ours.p_value == pytest.approx(reference["bp_pvalue"].iloc[0])
    assert ours.reject_null
    assert "autocorrelation" in ours.null_hypothesis_en


def test_arch_lm_matches_statsmodels_reference():
    pytest.importorskip("statsmodels")
    from statsmodels.stats.diagnostic import het_arch
    rng = np.random.default_rng(17)
    e = np.empty(350)
    e[0] = rng.normal()
    for t in range(1, len(e)):
        e[t] = rng.normal() * np.sqrt(0.2 + 0.6 * e[t - 1] ** 2)
    ours = arch_lm_test(e, lags=4)
    lm, p, _, _ = het_arch(e, nlags=4)
    assert ours.statistic == pytest.approx(lm)
    assert ours.p_value == pytest.approx(p)
    assert "ARCH" in ours.null_hypothesis_en


@pytest.mark.parametrize("lags", [0, -1, 1000, 1.5, True])
def test_box_pierce_rejects_invalid_lags(lags):
    with pytest.raises(ValueError):
        box_pierce_test(np.arange(40, dtype=float), lags=lags)


def test_mase_and_forecast_metrics_exact_case():
    training = [1, 2, 3, 4, 5]
    actual = [6, 7]
    forecast = [5.5, 7.5]
    assert mean_absolute_scaled_error(actual, forecast, training) == pytest.approx(0.5)
    report = forecast_accuracy_report(actual, forecast, training)
    assert report["mase"] == pytest.approx(0.5)
    assert report["rmse"] == pytest.approx(0.5)
    assert report["mae"] == pytest.approx(0.5)
    assert "MASE" in report["interpretation_ar"]
    assert "MASE" in report["interpretation_en"]


def test_mase_seasonal_naive_and_bad_baselines():
    actual, pred = [10, 20, 10], [10, 20, 12]
    training = [10, 20, 11, 21, 12, 22]
    assert mean_absolute_scaled_error(actual, pred, training, seasonality=2) == pytest.approx(2/3)
    with pytest.raises(ValueError, match="undefined"):
        mean_absolute_scaled_error([6], [5], [7, 7, 7])
    with pytest.raises(ValueError):
        mean_absolute_scaled_error([5, 6], [5], [1, 2, 3])
    with pytest.raises(ValueError):
        mean_absolute_scaled_error([5], [5], [1, 2], seasonality=2)


def test_smape_both_zero_defined_as_zero():
    report = forecast_accuracy_report([0, 2], [0, 3], [1, 2, 3])
    assert report["smape_percent"] == pytest.approx(20)
    assert np.isfinite(report["smape_percent"])
