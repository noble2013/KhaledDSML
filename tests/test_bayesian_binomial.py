"""Independent references and boundary tests for Bayesian binomial methods."""
import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special, stats

from khaled_dsml import (
    beta_binomial_posterior, beta_binomial_group_comparison,
    render_analysis_dashboard,
)


def test_conjugate_posterior_matches_scipy_exact_distribution():
    r = beta_binomial_posterior(
        3, 10, prior_alpha=1, prior_beta=1,
        credible_level=.95, predictive_trials=8,
    )
    assert r["posterior_alpha"] == 4
    assert r["posterior_beta"] == 8
    assert r["posterior_mean"] == pytest.approx(1/3)
    assert r["posterior_variance"] == pytest.approx(stats.beta.var(4, 8))
    lo, hi = stats.beta.ppf([.025, .975], 4, 8)
    assert r["credible_lower"] == pytest.approx(lo)
    assert r["credible_upper"] == pytest.approx(hi)
    t = r["table"]
    np.testing.assert_allclose(
        t["posterior_predictive_probability"],
        stats.betabinom.pmf(np.arange(9), 8, 4, 8), rtol=1e-12,
    )
    assert t["posterior_predictive_probability"].sum() == pytest.approx(1)
    assert r["predictive_mean_successes"] == pytest.approx(8/3)
    assert "p_value" not in r


def test_jeffreys_prior_and_zero_future_trials_are_supported():
    r = beta_binomial_posterior(0, 3, prior_alpha=.5, prior_beta=.5, predictive_trials=0)
    assert r["posterior_alpha"] == .5
    assert r["posterior_beta"] == 3.5
    assert r["table"].to_dict("records") == [
        {"future_successes": 0, "posterior_predictive_probability": 1.0}
    ]


def test_model_comparison_matches_independent_beta_integral_and_bayes_factor():
    sa, na, sb, nb = 20, 35, 8, 40
    a, b = 2., 3.
    r = beta_binomial_group_comparison(
        sa, na, sb, nb, prior_alpha=a, prior_beta=b,
    )
    x1, y1 = a+sa, b+na-sa
    x2, y2 = a+sb, b+nb-sb
    reference = stats.beta(x1, y1).expect(
        lambda x: stats.beta.cdf(x, x2, y2), lb=0, ub=1
    )
    assert r["probability_a_greater_b_given_h1"] == pytest.approx(reference, abs=1e-8)
    independent_log_bf = (
        special.betaln(x1, y1) + special.betaln(x2, y2)
        - special.betaln(a+sa+sb, b+na+nb-sa-sb) - special.betaln(a, b)
    )
    assert r["log_bayes_factor_10"] == pytest.approx(independent_log_bf, rel=1e-12)
    assert np.log(r["bayes_factor_10"]) == pytest.approx(independent_log_bf)
    assert 0 < r["integration_absolute_error"] < 1e-7
    assert "p_value" not in r
    assert isinstance(r["table"], pd.DataFrame)
    assert list(r["table"]["group"]) == ["A", "B"]


def test_model_comparison_probability_symmetry_and_equality():
    r = beta_binomial_group_comparison(15, 35, 15, 35)
    assert r["probability_a_greater_b_given_h1"] == pytest.approx(.5, abs=1e-9)
    first = beta_binomial_group_comparison(30, 40, 5, 40)
    swap = beta_binomial_group_comparison(5, 40, 30, 40)
    assert first["probability_a_greater_b_given_h1"] > .99
    assert first["probability_a_greater_b_given_h1"] + swap[
        "probability_a_greater_b_given_h1"
    ] == pytest.approx(1, abs=1e-8)
    assert first["log_bayes_factor_10"] == pytest.approx(swap["log_bayes_factor_10"])


@pytest.mark.parametrize("kwargs", [
    {"successes": True, "trials": 20},
    {"successes": -1, "trials": 20},
    {"successes": 21, "trials": 20},
    {"successes": 1.5, "trials": 20},
    {"successes": 1, "trials": 0},
    {"successes": 1, "trials": 20, "prior_alpha": 0},
    {"successes": 1, "trials": 20, "prior_beta": np.nan},
    {"successes": 1, "trials": 20, "prior_alpha": True},
    {"successes": 1, "trials": 20, "credible_level": 1},
    {"successes": 1, "trials": 20, "predictive_trials": -1},
    {"successes": 1, "trials": 20, "predictive_trials": 2001},
])
def test_posterior_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        beta_binomial_posterior(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"successes_a": True, "trials_a": 5, "successes_b": 2, "trials_b": 5},
    {"successes_a": 8, "trials_a": 5, "successes_b": 2, "trials_b": 5},
    {"successes_a": 3, "trials_a": 5, "successes_b": 2, "trials_b": 0},
    {"successes_a": 3, "trials_a": 5, "successes_b": -1, "trials_b": 6},
    {"successes_a": 3, "trials_a": 5, "successes_b": 2, "trials_b": 6, "prior_beta": -1},
    {"successes_a": 3, "trials_a": 5, "successes_b": 2, "trials_b": 6, "credible_level": 0},
])
def test_group_comparison_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        beta_binomial_group_comparison(**kwargs)


def test_bayesian_results_use_bilingual_analytics_studio_without_false_p_value():
    a = beta_binomial_posterior(10, 25, predictive_trials=3)
    b = beta_binomial_group_comparison(14, 25, 8, 25)
    for item in (a, b):
        html = render_analysis_dashboard(item, theme="amethyst")
        assert 'dir="rtl"' in html
        assert "جدول نتائج التحليل" in html
        assert item["interpretation_ar"] in html
        assert item["interpretation_en"] in html
        assert "p_value" not in item


def test_bayesian_notebook_is_valid_bilingual_and_unexecuted():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "notebooks" / "14_bayesian_binomial_studio.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    text = "\n".join("".join(cell["source"]) for cell in nb["cells"])
    from khaled_dsml import __version__
    assert __version__ in text
    assert "beta_binomial_posterior" in text
    assert "beta_binomial_group_comparison" in text
    assert "التحليل البايزي" in text
    assert all(not cell.get("outputs") for cell in nb["cells"] if cell["cell_type"] == "code")
