"""Reference tests for common items from the 256-row academic catalogue."""
import numpy as np
import pytest

from khaled_dsml.catalog_tests import (
    one_sample_t_test, dagostino_normality_test, bartlett_test,
    fligner_test, kendall_test, fisher_exact_test, chi_square_gof_test,
    friedman_test, sign_test, adjust_pvalues, cohen_d, cliffs_delta,
    bootstrap_mean_ci, permutation_mean_test,
)


def test_one_sample_t_and_bilingual_hypotheses():
    r = one_sample_t_test([5, 6, 7, 8, 9], mu=0)
    assert r.reject_null
    assert r.statistic > 0 and r.p_value < .05
    assert "mean" in r.null_hypothesis_en
    with pytest.raises(ValueError):
        one_sample_t_test([1, 1, 1], mu=1)


def test_dagostino_normality():
    rng = np.random.default_rng(7)
    result = dagostino_normality_test(rng.normal(size=80))
    assert 0 <= result.p_value <= 1 and "normal" in result.null_hypothesis_en
    with pytest.raises(ValueError):
        dagostino_normality_test([1, 2, 3])


def test_bartlett_and_fligner():
    a, b = [1, 2, 3, 4, 5], [11, 12, 13, 14, 15]
    assert 0 <= bartlett_test(a, b).p_value <= 1
    assert 0 <= fligner_test(a, b).p_value <= 1


def test_kendall():
    r = kendall_test([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert r.statistic == pytest.approx(-1)
    assert r.sample_size == 5


def test_fisher_exact_known_case():
    r = fisher_exact_test([[1, 9], [11, 3]])
    assert r.p_value < .01
    with pytest.raises(ValueError):
        fisher_exact_test([[1, 2], [3, 4.5]])


def test_chi_gof_and_invalid_expected():
    r = chi_square_gof_test([10, 20, 30], [20, 20, 20])
    assert r.statistic == pytest.approx(10)
    assert r.p_value < .01
    with pytest.raises(ValueError):
        chi_square_gof_test([10, 20, 30], [10, 10, 10])


def test_friedman_and_sign():
    r = friedman_test([1, 3, 3, 4, 5], [2, 4, 5, 6, 7], [3, 5, 6, 7, 8])
    assert 0 <= r.p_value <= 1
    sign = sign_test([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7])
    assert sign.reject_null and sign.sample_size == 6


def test_pvalue_corrections():
    pytest.importorskip("statsmodels")
    r = adjust_pvalues([.001, .03, .2], method="holm")
    assert r["reject"] == [True, False, False]
    assert 0 <= min(r["adjusted_p_values"]) <= 1
    with pytest.raises(ValueError):
        adjust_pvalues([.5, np.nan])


def test_effect_sizes():
    assert cohen_d([2, 3, 4], [1, 2, 3]) > 0
    assert cohen_d([2, 3, 4], [1, 2, 3], hedges=True) < cohen_d([2, 3, 4], [1, 2, 3])
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == pytest.approx(1)
    assert cliffs_delta([1, 2, 3], [5, 6, 7]) == pytest.approx(-1)


def test_bootstrap_resampling():
    r = bootstrap_mean_ci([1, 2, 3, 4, 5, 6, 7, 8], n_resamples=200)
    assert r["lower"] <= 4.5 <= r["upper"]
    assert r["method"] == "BCa"


def test_permutation_test():
    r = permutation_mean_test([1, 2, 3, 4, 5], [10, 11, 12, 13, 14],
                              n_resamples=300)
    assert r.reject_null
    assert r.p_value < .05
    assert "exchangeable" in r.null_hypothesis_en
