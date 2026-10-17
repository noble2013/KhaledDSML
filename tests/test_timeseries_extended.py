"""Extended time-series tests with statsmodels references and caveats."""
import numpy as np
import pytest
from scipy import stats
ts=pytest.importorskip("statsmodels.tsa.stattools")
from khaled_dsml import (
    kpss_stationarity_test, zivot_andrews_break_test,
    granger_predictivity_tests, engle_granger_cointegration_test,
    johansen_cointegration_test, diebold_mariano_test,
)


def test_kpss_null_is_stationarity_and_matches_reference():
    rng=np.random.default_rng(3)
    a=rng.normal(size=90)
    got=kpss_stationarity_test(a,nlags="auto")
    expected=ts.kpss(a,regression="c",nlags="auto")
    assert got.statistic==pytest.approx(expected[0])
    assert got.p_value==pytest.approx(expected[1])
    assert "stationary" in got.null_hypothesis_en


def test_zivot_andrews_break_reference():
    rng=np.random.default_rng(6)
    series=np.cumsum(rng.normal(size=100))
    got=zivot_andrews_break_test(series,maxlag=2)
    ref=ts.zivot_andrews(series,maxlag=2,regression="c",autolag="AIC")
    assert got["statistic"]==pytest.approx(ref[0])
    assert got["p_value"]==pytest.approx(ref[1])
    assert got["break_index"]==ref[4]


def test_granger_predictivity_is_not_causal_claim():
    rng=np.random.default_rng(7)
    x=rng.normal(size=110)
    y=np.zeros(110)
    for i in range(1,len(y)):
        y[i]=.7*x[i-1]+rng.normal(scale=.4)
    r=granger_predictivity_tests(y,x,maxlag=2)
    assert len(r["table"])==2
    assert r["table"]["p_value"].between(0,1).all()
    assert r["table"].loc[r["table"]["lag"]==1,"p_value"].iloc[0] < .01
    assert "not proof" in r["interpretation_en"]


def test_engle_granger_reference_and_same_length():
    rng=np.random.default_rng(4)
    x=np.cumsum(rng.normal(size=120))
    y=x+rng.normal(scale=.2,size=120)
    got=engle_granger_cointegration_test(x,y)
    ref=ts.coint(x,y,trend="c",autolag="aic")
    assert got.statistic==pytest.approx(ref[0])
    assert got.p_value==pytest.approx(ref[1])
    with pytest.raises(ValueError):
        engle_granger_cointegration_test(x,y[:-1])


def test_johansen_rank_reference():
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    rng=np.random.default_rng(11)
    x=np.cumsum(rng.normal(size=120))
    y=x+rng.normal(scale=.3,size=120)
    arr=np.column_stack([x,y])
    got=johansen_cointegration_test(arr)
    ref=coint_johansen(arr,0,1)
    assert got["table"]["trace_statistic"].to_numpy()==pytest.approx(ref.lr1)
    assert got["table"]["critical_value"].to_numpy()==pytest.approx(ref.cvt[:,1])


def test_diebold_mariano_zero_variance_rejected_and_large_difference():
    rng=np.random.default_rng(22)
    true=rng.normal(size=150)
    best=true+rng.normal(scale=.1,size=150)
    worst=true+rng.normal(scale=1.2,size=150)
    r=diebold_mariano_test(true,best,worst,horizon=1)
    assert r.p_value < .05
    assert "not" in r.notes_en.lower() or "same-horizon" in r.notes_en.lower()
    with pytest.raises(ValueError):
        diebold_mariano_test(true,true,true)
    with pytest.raises(ValueError):
        diebold_mariano_test(true,best,worst,horizon=0)
