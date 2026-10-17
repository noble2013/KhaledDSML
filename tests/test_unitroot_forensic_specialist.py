"""Independent numerical checks for PP unit roots, GRIMMER and p-curve."""
from decimal import Decimal, ROUND_HALF_UP
from html import escape
from itertools import product
import numpy as np
import pytest
from scipy import stats

from khaled_dsml import (
    phillips_perron_unit_root, grimmer_variance_consistency,
    pcurve_right_skew_screen, render_analysis_dashboard,
)


@pytest.mark.parametrize("test_type,trend,lags",[
    ("tau","c",8),("rho","c",5),("tau","ct",10),("tau","n",0),
])
def test_phillips_perron_matches_arch_reference(test_type,trend,lags):
    from arch.unitroot import PhillipsPerron
    rng=np.random.default_rng(14)
    x=np.cumsum(rng.normal(size=320))
    expected=PhillipsPerron(x,trend=trend,test_type=test_type,lags=lags)
    r=phillips_perron_unit_root(x,trend=trend,test_type=test_type,lags=lags)
    assert r["statistic"]==pytest.approx(float(expected.stat),rel=1e-11,abs=1e-11)
    assert r["p_value"]==pytest.approx(float(expected.pvalue),rel=1e-11,abs=1e-12)
    assert r["nw_lags"]==lags
    assert len(r["table"])>=3
    assert r["reject_unit_root"] == (r["p_value"]<.05)
    assert "unit root" in r["interpretation_en"]


def test_phillips_perron_auto_bandwidth_and_guards():
    from arch.unitroot import PhillipsPerron
    rng=np.random.default_rng(24)
    x=rng.normal(size=230)
    r=phillips_perron_unit_root(x)
    reference=PhillipsPerron(x)
    assert r["nw_lags"]==reference.lags
    assert r["p_value"]==pytest.approx(float(reference.pvalue))
    for kwargs in ({"trend":"X"},{"test_type":"bad"},{"lags":True},{"lags":300},{"alpha":True}):
        with pytest.raises(ValueError):
            phillips_perron_unit_root(x,**kwargs)
    with pytest.raises(ValueError):
        phillips_perron_unit_root(np.ones(300))
    with pytest.raises(ValueError):
        phillips_perron_unit_root([2.0]*20)


def _rounded_summary(values,mean_dp,sd_dp):
    n=len(values)
    a=Decimal(sum(values))/Decimal(n)
    m=a.quantize(Decimal(1).scaleb(-mean_dp),rounding=ROUND_HALF_UP)
    q=sum(v*v for v in values)
    unbiased=(Decimal(q)-Decimal(sum(values)**2)/Decimal(n))/Decimal(n-1)
    with __import__("decimal").localcontext() as ctx:
        ctx.prec=70
        std=unbiased.sqrt()
    s=std.quantize(Decimal(1).scaleb(-sd_dp),rounding=ROUND_HALF_UP)
    return str(m),str(s)


@pytest.mark.parametrize("n,bounds",[(2,(1,3)),(3,(1,4)),(4,(0,3))])
def test_grimmer_matches_exhaustive_integer_arrays_and_inconsistent_combinations(n,bounds):
    low,high=bounds
    attained=set()
    for values in product(range(low,high+1),repeat=n):
        attained.add(_rounded_summary(values,2,2))
    # Every attainable rounded pair must pass the exact DP oracle.
    for mu,sd in sorted(attained):
        r=grimmer_variance_consistency(mu,sd,n,minimum_score=low,maximum_score=high)
        assert r["arithmetically_consistent"],(n,mu,sd)
        assert r["witness_moments"] is not None
    # Try impossible SD and mean combinations, not only built-in positive examples.
    examples=[("0.00","0.00"),("1.50","0.01"),("2.75","0.04"),("3.10","1.01")]
    for mu,sd in examples:
        if Decimal(mu)<low or Decimal(mu)>high:continue
        actual=(mu,sd) in attained
        result=grimmer_variance_consistency(mu,sd,n,minimum_score=low,maximum_score=high)
        assert result["arithmetically_consistent"]==actual


def test_grimmer_known_sd_ddof_one_rounding_and_rejection():
    yes=grimmer_variance_consistency("2.00","1.41",2,
                                     minimum_score=1,maximum_score=3)
    assert yes["arithmetically_consistent"]
    no=grimmer_variance_consistency("2.00","1.42",2,
                                    minimum_score=1,maximum_score=3)
    assert not no["arithmetically_consistent"]
    d=grimmer_variance_consistency("2.50","1.29",4,
                                   minimum_score=1,maximum_score=4)
    assert d["arithmetically_consistent"]
    for kw in ({"sample_size":51},{"sample_size":1},
               {"sample_size":4,"decimals_sd":5},
               {"sample_size":4,"minimum_score":1,"maximum_score":15}):
        args=dict(sample_size=4)
        args.update(kw)
        with pytest.raises(ValueError):
            grimmer_variance_consistency("2.50","1.29",**args)


def test_pcurve_matches_scipy_binomial_with_exact_significant_p():
    values=[.001,.002,.01,.019,.022,.024,.025,.028,.045,.049]
    r=pcurve_right_skew_screen(values)
    assert r["n_p_below_midpoint"]==6
    assert r["n_p_above_or_equal_midpoint"]==4
    assert r["statistic"]==6
    assert r["p_value"]==pytest.approx(
        stats.binomtest(6,10,.5,alternative="greater").pvalue)
    assert not r["reject_uniform_significant_p"]
    shifted=pcurve_right_skew_screen(values,midpoint=.01)
    assert shifted["p_value"]==pytest.approx(
        stats.binomtest(2,10,.2,alternative="greater").pvalue)
    assert "power estimator" in r["interpretation_en"]


@pytest.mark.parametrize("values,kwargs",[
    ([.01,.02,.03],{}),([.001,.002,.03,.051,.01],{}),
    ([.01,.02,.03,.04,0],{}),([.01,.02,.03,.04,float("nan")],{}),
    ([.001,.002,.01,.03,.04],{"threshold":True}),
    ([.001,.002,.01,.03,.04],{"midpoint":.05}),
])
def test_pcurve_rejects_unusable_inference(values,kwargs):
    with pytest.raises(ValueError):
        pcurve_right_skew_screen(values,**kwargs)


def test_specialist_outputs_have_escaped_bilingual_dashboard():
    rng=np.random.default_rng(42)
    out=[
        phillips_perron_unit_root(rng.normal(size=135)),
        grimmer_variance_consistency("2.00","1.41",2,minimum_score=1,maximum_score=3),
        pcurve_right_skew_screen([.001,.002,.003,.004,.03,.04])
    ]
    for item in out:
        html=render_analysis_dashboard(item,theme="ocean")
        assert 'dir="rtl"' in html
        assert escape(item["interpretation_ar"]) in html
        assert escape(item["interpretation_en"]) in html
        assert "<table" in html or item["test"].startswith("GRIMMER")
