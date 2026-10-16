import numpy as np
import pytest
from khaled_dsml import (normality_test, shapiro_test, jarque_bera_test,
    independent_t_test, paired_t_test, mann_whitney_test, wilcoxon_test,
    anova_test, kruskal_test, levene_test, chi_square_test,
    pearson_test, spearman_test, ks_two_sample_test, descriptive_summary)
import pandas as pd


def test_normality():
    rng = np.random.default_rng(1)
    r = normality_test(rng.normal(size=100))
    assert 0 <= r.p_value <= 1 and "الفرضية" in r.interpretation_ar
    assert r.sample_size == 100 and r.to_dict()["test"] == "Shapiro-Wilk"


def test_jarque_bera():
    assert 0 <= jarque_bera_test(np.random.default_rng(10).normal(size=100)).p_value <= 1


def test_invalid_alpha():
    with pytest.raises(ValueError):
        shapiro_test([1, 2, 3], alpha=1)


def test_independent_t_test_detects_difference():
    rng = np.random.default_rng(7)
    a, b = rng.normal(0, 1, 100), rng.normal(2, 1, 100)
    assert independent_t_test(a, b).reject_null


def test_paired_t_test():
    r = paired_t_test([1, 2, 3, 4, 5], [3, 2.9, 4, 7, 6])
    assert r.reject_null
    with pytest.raises(ValueError):
        paired_t_test([1, 2], [1])


def test_nonparametric():
    assert mann_whitney_test([1, 2, 3, 4, 5], [7, 8, 9, 10, 11]).reject_null
    assert wilcoxon_test([1, 2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8]).reject_null
    assert kruskal_test([1, 2, 3, 4], [10, 11, 12, 13]).reject_null


def test_anova_levene():
    assert anova_test([1, 2, 3, 2], [8, 9, 10, 9]).reject_null
    assert 0 <= levene_test([1, 2, 3, 4], [3, 3, 3, 5]).p_value <= 1


def test_chi_square_independence():
    r = chi_square_test([[30, 1], [1, 30]])
    assert r.reject_null and r.sample_size == 62
    with pytest.raises(ValueError):
        chi_square_test([[1, 0], [0, 0]])


def test_correlations():
    assert pearson_test([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]).statistic == pytest.approx(1)
    assert spearman_test([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]).statistic == pytest.approx(-1)


def test_ks_two_sample():
    assert ks_two_sample_test(range(50), range(100, 150)).reject_null


def test_descriptive():
    result = descriptive_summary(pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]}))
    assert result.loc["a", "mean"] == 2


def test_invalid_method():
    with pytest.raises(ValueError):
        normality_test([1, 2, 3], method="unknown")
