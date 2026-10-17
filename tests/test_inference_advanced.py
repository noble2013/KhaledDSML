"""Reference-based tests for extended categorical and equivalence inference."""
import numpy as np
import pytest
from scipy import stats
from khaled_dsml import (
    cochran_armitage_trend_test, g_test_independence,
    cochran_mantel_haenszel_test, newcombe_two_proportions_interval,
    welch_tost_equivalence, two_group_power_analysis,
)


def test_cochran_armitage_trend_matches_score_formula():
    x=np.array([20.,35.,50.])
    n=np.array([100.,100.,100.])
    r=cochran_armitage_trend_test(x,n)
    p=x.sum()/n.sum()
    scores=np.arange(3,dtype=float)
    z=np.dot(scores,x-p*n)/np.sqrt(p*(1-p)*(np.dot(n,scores**2)-np.dot(n,scores)**2/n.sum()))
    assert r.statistic==pytest.approx(z)
    assert r.p_value==pytest.approx(2*stats.norm.sf(abs(z)))
    assert r.reject_null
    with pytest.raises(ValueError):
        cochran_armitage_trend_test([1,2,3],[10,10,10])


def test_g_independence_matches_scipy():
    counts=np.array([[20,30,40],[40,30,20]])
    expected=stats.chi2_contingency(counts,correction=False,lambda_="log-likelihood")
    r=g_test_independence(counts)
    assert r.statistic==pytest.approx(expected.statistic)
    assert r.p_value==pytest.approx(expected.pvalue)
    with pytest.raises(ValueError):
        g_test_independence([[1,2],[2,1]])


def test_cmh_matches_statsmodels_stratified_reference():
    pytest.importorskip("statsmodels")
    from statsmodels.stats.contingency_tables import StratifiedTable
    t=np.array([[[15,6],[8,17]],[[24,10],[11,26]]],dtype=int)
    r=cochran_mantel_haenszel_test(t)
    ref=StratifiedTable([t[0],t[1]]).test_null_odds(correction=False)
    assert r.statistic==pytest.approx(ref.statistic)
    assert r.p_value==pytest.approx(ref.pvalue)


def test_newcombe_interval_matches_statsmodels_reference():
    pytest.importorskip("statsmodels")
    from statsmodels.stats.proportion import confint_proportions_2indep
    r=newcombe_two_proportions_interval(55,100,38,110)
    low,high=confint_proportions_2indep(55,100,38,110,
                                       method="newcomb",compare="diff")
    assert r["lower"]==pytest.approx(low)
    assert r["upper"]==pytest.approx(high)
    assert r["lower"]<r["difference"]<r["upper"]


def test_tost_supports_equivalence_only_when_margin_prespecified():
    rng=np.random.default_rng(10)
    a=rng.normal(10,.15,60)
    b=rng.normal(10,.15,60)
    r=welch_tost_equivalence(a,b,margin=.5)
    assert r["reject_null"]
    assert r["p_value"]<.05
    narrow=welch_tost_equivalence(a,b,margin=.0001)
    assert not narrow["reject_null"]
    with pytest.raises(ValueError):
        welch_tost_equivalence(a,b,margin=0)


def test_power_solver_reference_and_rejections():
    pytest.importorskip("statsmodels")
    from statsmodels.stats.power import TTestIndPower
    result=two_group_power_analysis(.5,power=.8)
    actual=TTestIndPower().power(effect_size=.5,nobs1=result["n_per_group"],alpha=.05)
    assert actual>=.8
    assert result["n_per_group"]>=30
    r=two_group_power_analysis(.5,n_per_group=64)
    assert r["power"]==pytest.approx(TTestIndPower().power(effect_size=.5,nobs1=64,alpha=.05))
    with pytest.raises(ValueError):
        two_group_power_analysis(.5,n_per_group=30,power=.8)
