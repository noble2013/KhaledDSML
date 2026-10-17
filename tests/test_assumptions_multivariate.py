"""Independent checks for the expanded multivariate-assumption suite."""
import numpy as np
import pytest
from scipy import stats

from khaled_dsml import (
    mardia_multivariate_normality, box_m_covariance_equality,
    mauchly_sphericity_test, breusch_godfrey_test, levene_mean_test,
)


def test_mardia_skew_kurtosis_are_separate_valid_probabilities():
    rng = np.random.default_rng(101)
    sample = rng.normal(size=(120,3))
    result = mardia_multivariate_normality(sample)
    assert 0 <= result["skewness_p_value"] <= 1
    assert 0 <= result["kurtosis_p_value"] <= 1
    assert result["skewness_df"] == 10
    assert result["n_samples"] == 120
    assert "separate" in result["interpretation_en"]
    assert "p_value" not in result  # no fictional combined p-value


def test_mardia_skew_matches_independent_formula():
    rng = np.random.default_rng(102)
    x = rng.normal(size=(40,2))
    r = mardia_multivariate_normality(x)
    centered=x-x.mean(axis=0)
    cov=(centered.T@centered)/len(x)
    gram=centered@np.linalg.inv(cov)@centered.T
    skew=(gram**3).sum()/len(x)**2
    assert r["skewness"] == pytest.approx(skew,rel=1e-11)
    assert r["skewness_chi_square"] == pytest.approx(len(x)*skew/6)


def test_box_m_is_invariant_under_common_rescaling():
    rng = np.random.default_rng(29)
    x,y=rng.normal(size=(60,3)),rng.normal(size=(65,3))
    a=box_m_covariance_equality(x,y)
    b=box_m_covariance_equality(x*np.array([4,.5,2]),y*np.array([4,.5,2]))
    assert a.statistic == pytest.approx(b.statistic,rel=1e-8)
    assert a.p_value == pytest.approx(b.p_value,rel=1e-8)
    assert 0<=a.p_value<=1
    with pytest.raises(ValueError):
        box_m_covariance_equality(x,np.ones((60,3)))


def test_mauchly_w_detects_non_sphericity_and_is_unit_scale_invariant():
    rng=np.random.default_rng(15)
    base=rng.normal(size=(140,3))
    scale=base*np.array([1,1,8])
    r=mauchly_sphericity_test(scale)
    assert 0<r["W"]<1
    assert r["degrees_of_freedom"]==2
    assert r["p_value"]<.05
    same=mauchly_sphericity_test(scale*3)
    assert same["W"]==pytest.approx(r["W"],rel=1e-11)
    with pytest.raises(ValueError):
        mauchly_sphericity_test(base[:8])


def test_breusch_godfrey_matches_statsmodels_exact_statistic():
    sm=pytest.importorskip("statsmodels.api")
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey
    rng=np.random.default_rng(8)
    x=rng.normal(size=180)
    errors=np.zeros(len(x))
    for i in range(1,len(x)):
        errors[i]=.7*errors[i-1]+rng.normal()
    model=sm.OLS(1+2*x+errors,sm.add_constant(x)).fit()
    r=breusch_godfrey_test(model,nlags=2)
    lm,p,*_=acorr_breusch_godfrey(model,nlags=2)
    assert r.statistic==pytest.approx(lm)
    assert r.p_value==pytest.approx(p)


def test_levene_mean_is_not_brown_forsythe_median():
    a=[0,1,2,3,50,60]
    b=[3,4,5,6,7,8]
    r=levene_mean_test(a,b)
    ref=stats.levene(a,b,center="mean")
    assert r.statistic==pytest.approx(ref.statistic)
    assert r.p_value==pytest.approx(ref.pvalue)
    assert "mean" in r.test.lower()
