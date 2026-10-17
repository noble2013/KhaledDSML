"""Scientific references and invalid-input cases for extended inferential tests."""
import numpy as np
import pytest
from scipy import stats
from khaled_dsml import (
    welch_anova_test, mood_median_test, brunner_munzel_test,
    mcnemar_test, cochran_q_test, one_proportion_z_test,
    two_proportions_z_test, wilson_proportion_interval,
    point_biserial_test, partial_correlation_test, cramers_v,
    phi_coefficient, fisher_z_correlation_test, glass_delta,
    eta_squared, omega_squared, cohen_w, contingency_risk_measures,
)


def test_welch_anova_matches_statsmodels_reference():
    pytest.importorskip("statsmodels")
    from statsmodels.stats.oneway import anova_oneway
    rng=np.random.default_rng(19)
    groups=[rng.normal(0,1,25),rng.normal(1,3,35),rng.normal(2,2,30)]
    actual=welch_anova_test(*groups)
    ref=anova_oneway(groups,use_var="unequal",welch_correction=True)
    assert actual.statistic==pytest.approx(ref.statistic)
    assert actual.p_value==pytest.approx(ref.pvalue)
    assert actual.null_hypothesis_en


def test_nonparametric_matches_scipy():
    a=[1,3,2,5,8,4,5,4]
    b=[4,5,7,9,11,6,10,8]
    c=[8,9,10,12,12,10,11,13]
    mood=mood_median_test(a,b,c)
    assert mood.p_value==pytest.approx(stats.median_test(a,b,c,ties="ignore").pvalue)
    bm=brunner_munzel_test(a,b)
    assert bm.p_value==pytest.approx(stats.brunnermunzel(a,b,distribution="t").pvalue)


def test_mcnemar_exact_binomial_and_approximation():
    table=[[10,3],[10,20]]
    result=mcnemar_test(table)
    assert result.p_value==pytest.approx(stats.binomtest(3,13,.5).pvalue)
    chi=mcnemar_test(table,exact=False)
    assert chi.statistic==pytest.approx((abs(3-10)-1)**2/13)
    with pytest.raises(ValueError):
        mcnemar_test([[1,0],[0,2]])


def test_cochran_q_against_statsmodels():
    pytest.importorskip("statsmodels")
    from statsmodels.stats.contingency_tables import cochrans_q
    x=np.array([[1,0,0],[1,0,1],[0,0,1],[0,1,1],[1,0,0],[1,1,0]])
    result=cochran_q_test(x)
    reference=cochrans_q(x)
    assert result.statistic==pytest.approx(reference.statistic)
    assert result.p_value==pytest.approx(reference.pvalue)


def test_proportions_and_wilson_reference():
    from scipy.stats import norm
    p=one_proportion_z_test(58,100,null_proportion=.5)
    z=(.58-.5)/np.sqrt(.25/100)
    assert p.statistic==pytest.approx(z)
    assert p.p_value==pytest.approx(2*norm.sf(abs(z)))
    two=two_proportions_z_test(60,100,40,100)
    assert two.p_value<.01
    w=wilson_proportion_interval(0,30)
    assert w["lower"]==pytest.approx(0)
    assert 0 < w["upper"] < .2
    with pytest.raises(ValueError):
        one_proportion_z_test(1,10,null_proportion=.1)


def test_point_biserial_and_partial_reference():
    b=[0,0,0,1,1,1,1,0,1,0]
    y=[1,2,3,5,8,9,7,4,8,2]
    got=point_biserial_test(b,y)
    assert got.statistic==pytest.approx(stats.pointbiserialr(b,y).statistic)
    rng=np.random.default_rng(18)
    z=rng.normal(size=120)
    x=z+rng.normal(size=120)
    y2=z+rng.normal(size=120)
    part=partial_correlation_test(x,y2,z)
    assert part.null_hypothesis_en
    assert np.isfinite(part.statistic)


def test_contingency_associations():
    x=np.array([[30,10],[10,30]])
    assert phi_coefficient(x)==pytest.approx(.5)
    assert cramers_v(x,correction=False)==pytest.approx(.5)
    assert cramers_v(x,correction=True)<.5
    odds=contingency_risk_measures(x)
    assert odds["odds_ratio"]==pytest.approx(9)
    assert odds["relative_risk"]==pytest.approx(3)
    assert odds["or_ci"][0]<9<odds["or_ci"][1]
    with pytest.raises(ValueError):
        contingency_risk_measures([[0,20],[10,30]])


def test_effect_sizes_reference():
    a=[2,3,4,5,6]
    b=[8,9,10,11,12]
    assert glass_delta(b,a)==pytest.approx(6/np.std(a,ddof=1))
    groups=[a,b]
    assert eta_squared(groups)>.8
    assert 0<omega_squared(groups)<=eta_squared(groups)
    assert cohen_w([50,50],[.5,.5])==pytest.approx(0)
    assert cohen_w([70,30],[.5,.5])==pytest.approx(.4)


def test_fisher_z_independent_only():
    test=fisher_z_correlation_test(.6,80,.1,85)
    assert test.reject_null
    with pytest.raises(ValueError):
        fisher_z_correlation_test(1.,80,.1,85)


def test_bad_shapes_and_zero_variation_rejected():
    with pytest.raises(ValueError):
        phi_coefficient([[1,0],[0,0]])
    with pytest.raises(ValueError):
        partial_correlation_test([1]*20,list(range(20)),list(range(20)))
    with pytest.raises(ValueError):
        welch_anova_test([1]*20,[2]*20)
    with pytest.raises(ValueError):
        cochran_q_test([[1,2,0],[0,1,1],[1,0,0]])
