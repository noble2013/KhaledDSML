"""Research reference checks for study-level meta-analysis routines."""
import numpy as np
import pytest
from scipy import stats
from html import unescape

from khaled_dsml import (
    meta_analysis_heterogeneity, meta_analysis_random_effects,
    egger_small_study_effects_test, render_analysis_dashboard,
)


def _reference_data():
    y = np.array([.04, .27, .42, .14, -.09, .63, .32, .19, .47, -.13, .35, .18])
    se = np.array([.09, .12, .10, .08, .11, .09, .14, .13, .10, .12, .09, .11])
    return y, se


def test_cochran_q_i2_dl_tau_squared_match_independent_hand_formula():
    y, se = _reference_data()
    r = meta_analysis_heterogeneity(y, se)
    w = 1 / se**2
    mu = (w * y).sum() / w.sum()
    q = (w * (y-mu)**2).sum()
    df = len(y) - 1
    c = w.sum() - (w**2).sum() / w.sum()
    tau2 = max(0, (q-df)/c)
    i2 = max(0, (q-df)/q) * 100
    assert r["fixed_effect_estimate"] == pytest.approx(mu)
    assert r["fixed_effect_standard_error"] == pytest.approx(np.sqrt(1/w.sum()))
    assert r["Q"] == pytest.approx(q)
    assert r["p_value"] == pytest.approx(stats.chi2.sf(q, df))
    assert r["I_squared_percent"] == pytest.approx(i2)
    assert r["tau_squared_dl"] == pytest.approx(tau2)
    assert r["tau_squared_dl"] > 0
    assert 0 <= r["I_squared_percent"] <= 100
    assert r["table"]["inverse_variance_weight"].sum() == pytest.approx(1)
    assert "p_value" in r


def test_dl_meta_analysis_matches_statsmodels_reference():
    meta = pytest.importorskip("statsmodels.stats.meta_analysis")
    y, se = _reference_data()
    r = meta_analysis_random_effects(y, se)
    ref = meta.combine_effects(y, se**2, method_re="chi2", use_t=False)
    assert r["tau_squared_dl"] == pytest.approx(ref.tau2, rel=1e-10)
    assert r["estimate"] == pytest.approx(ref.mean_effect_re, rel=1e-10)
    assert r["standard_error"] == pytest.approx(ref.sd_eff_w_re, rel=1e-10)
    assert r["Q"] == pytest.approx(ref.q)
    assert r["ci_lower"] == pytest.approx(r["estimate"] - stats.norm.ppf(.975)*r["standard_error"])
    assert r["ci_upper"] == pytest.approx(r["estimate"] + stats.norm.ppf(.975)*r["standard_error"])
    assert "p_value" not in r
    assert r["table"]["random_effect_weight"].sum() == pytest.approx(1)


def test_zero_tau_when_effects_identical_and_input_not_mutated():
    y = np.array([.2,.2,.2,.2])
    se = np.array([.15,.12,.18,.10])
    before = y.copy()
    r = meta_analysis_random_effects(y, se)
    assert r["tau_squared_dl"] == 0
    assert r["I_squared_percent"] == 0
    assert r["estimate"] == pytest.approx(.2)
    np.testing.assert_array_equal(y, before)


def test_egger_intercept_matches_independent_statsmodels_regression():
    sm = pytest.importorskip("statsmodels.api")
    y, se = _reference_data()
    r = egger_small_study_effects_test(y, se)
    model = sm.OLS(y / se, sm.add_constant(1/se)).fit()
    assert r["intercept"] == pytest.approx(model.params[0], rel=1e-9, abs=1e-9)
    assert r["precision_slope"] == pytest.approx(model.params[1], rel=1e-9)
    assert r["intercept_standard_error"] == pytest.approx(model.bse[0], rel=1e-9)
    assert r["statistic"] == pytest.approx(model.tvalues[0], rel=1e-9)
    assert r["p_value"] == pytest.approx(model.pvalues[0], rel=1e-9)
    assert r["degrees_of_freedom"] == 10
    assert "publication bias" in r["interpretation_en"]


@pytest.mark.parametrize("effects, se", [
    ([.2, .3], [.1]), ([.2], [.1]),
    ([.2, np.nan, .1], [.1, .2, .1]),
    ([.2, .3, .4], [.1, 0, .1]),
    ([.2, .3, .4], [.1, -0.1, .1]),
])
def test_heterogeneity_rejects_invalid_study_inputs(effects, se):
    with pytest.raises(ValueError):
        meta_analysis_heterogeneity(effects, se)


def test_egger_rejects_too_few_or_identical_precisions():
    with pytest.raises(ValueError):
        egger_small_study_effects_test(np.arange(9)*.1, np.arange(9)*.01+.1)
    with pytest.raises(ValueError, match="Distinct"):
        egger_small_study_effects_test(np.arange(12)*.1, np.ones(12)*.1)


@pytest.mark.parametrize("alpha", [0, 1, -1, True, np.nan])
def test_invalid_significance_levels_rejected(alpha):
    y, se = _reference_data()
    for fn in (meta_analysis_random_effects, meta_analysis_heterogeneity,
               egger_small_study_effects_test):
        with pytest.raises(ValueError):
            fn(y, se, alpha=alpha)


def test_meta_results_have_bilingual_studio_rendering():
    y, se = _reference_data()
    for r in (meta_analysis_heterogeneity(y, se),
              meta_analysis_random_effects(y, se),
              egger_small_study_effects_test(y, se)):
        html = render_analysis_dashboard(r, theme="ocean")
        assert 'dir="rtl"' in html
        assert r["interpretation_ar"] in html
        assert r["interpretation_en"] in unescape(html)
        assert "table" in html.lower()
