"""Additional econometric time-series diagnostics with bilingual caveats.

Tests operate on chronologically ordered finite observations and require
statsmodels. Predictive Granger association must not be called causality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from ._results import make_result


def _series(data,minimum=24):
    a=np.asarray(data,dtype=float).reshape(-1)
    if len(a)<minimum or not np.isfinite(a).all():
        raise ValueError(f"Need >= {minimum} chronological, finite observations")
    if a.std()==0:
        raise ValueError("Constant time series are not valid for these tests")
    return a


def _sm():
    try:
        import statsmodels.tsa.stattools as ts
    except ImportError as exc:
        raise ImportError('Install khaled-dsml[econometrics] to run advanced time-series tests') from exc
    return ts


def _alpha(alpha):
    if not 0<alpha<1:
        raise ValueError("alpha must be in (0,1)")


def kpss_stationarity_test(values, *, regression="c", nlags="auto", alpha=.05):
    """KPSS: null is (level/trend) stationarity, unlike the ADF null."""
    _alpha(alpha)
    ts=_sm()
    if regression not in ("c","ct"):
        raise ValueError("regression must be c (level) or ct (trend)")
    x=_series(values)
    statistic,p,_,_=ts.kpss(x,regression=regression,nlags=nlags)
    return make_result(
        "KPSS stationarity",statistic,p,len(x),alpha,
        "السلسلة ساكنة حول المستوى أو الاتجاه المحدد.",
        "القيمة الاحتمالية قد تكون مقيدة بجدول KPSS؛ افحص التحذيرات عند الأطراف.",
        null_en="The series is level- or trend-stationary under the selected specification.",
        notes_en="KPSS table p-values may be capped at boundaries; inspect warnings.",
    )


def zivot_andrews_break_test(values, *, maxlag=None, regression="c", alpha=.05):
    """Zivot-Andrews unit root test with one endogenously selected break."""
    _alpha(alpha); ts=_sm(); x=_series(values,minimum=40)
    if regression not in ("c","t","ct"):
        raise ValueError("regression must be c, t, or ct")
    statistic,p,crit,lag,break_idx=ts.zivot_andrews(
        x,maxlag=maxlag,regression=regression,autolag="AIC")
    return {
        "test":"Zivot-Andrews endogenous break","statistic":float(statistic),
        "p_value":float(p),"alpha":alpha,"reject_null":bool(p<alpha),
        "lag":int(lag),"break_index":int(break_idx),
        "critical_values":dict(crit),"n_samples":len(x),
        "interpretation_ar":"الفرضية الصفرية جذر وحدة دون تحوّل؛ الرفض يدعم السكون مع تحول واحد، لا يثبت استقرارًا اقتصاديًا عامًا.",
        "interpretation_en":"The null is a unit root without a break; rejection supports break-stationarity, not general economic stability.",
    }


def granger_predictivity_tests(target, candidate, *, maxlag=3, alpha=.05) -> dict:
    """Test whether lagged candidate adds *predictive* value for target."""
    _alpha(alpha); ts=_sm()
    y=_series(target,minimum=30)
    x=_series(candidate,minimum=30)
    if len(x)!=len(y) or type(maxlag) is not int or not 1<=maxlag<=min(12,(len(y)-5)//3):
        raise ValueError("Need aligned series and maxlag small relative to n")
    sample=np.column_stack([y,x])
    from contextlib import redirect_stdout
    from io import StringIO
    with redirect_stdout(StringIO()):
        result=ts.grangercausalitytests(sample,maxlag=maxlag)
    rows=[]
    for lag in sorted(result):
        f,p,df,dfnum=result[lag][0]["ssr_ftest"]
        rows.append({"lag":lag,"f_statistic":float(f),"p_value":float(p),
                     "reject_null":bool(p<alpha)})
    return {
        "table":pd.DataFrame(rows),"n_samples":len(y),
        "interpretation_ar":"اختبار قدرة تنبؤية مشروطة بالإبطاءات؛ لا يعني وجود سببية واقعية أو تجريبية.",
        "interpretation_en":"Lagged-predictive association, not proof of real-world or intervention-based causation.",
    }


def engle_granger_cointegration_test(a,b, *, alpha=.05, trend="c"):
    """Residual-based Engle-Granger cointegration test for two I(1) series."""
    _alpha(alpha); ts=_sm()
    x=_series(a,minimum=40); y=_series(b,minimum=40)
    if len(x)!=len(y):
        raise ValueError("Series must be aligned and equal-length")
    if trend not in ("c","ct","ctt","n"):
        raise ValueError("Invalid trend specification")
    statistic,p,_=ts.coint(x,y,trend=trend,autolag="aic")
    return make_result(
        "Engle-Granger cointegration",statistic,p,len(x),alpha,
        "لا توجد علاقة تكامل مشترك بين السلسلتين.",
        "ينبغي أولًا تقييم رتبة تكامل كل سلسلة وتحديد اتجاه معادلة التكامل.",
        null_en="The pair does not cointegrate under the specified deterministic trend.",
        notes_en="Check each series is I(1) and specify regression orientation before interpreting.",
    )


def johansen_cointegration_test(data, *, det_order=0, k_ar_diff=1, significance=.95) -> dict:
    """Johansen trace-test table with documented critical-value comparison."""
    _sm()
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    x=np.asarray(data,dtype=float)
    if x.ndim!=2 or x.shape[1]<2 or x.shape[0]<40 or not np.isfinite(x).all():
        raise ValueError("Need n>=40 finite chronological observations on >=2 variables")
    if det_order not in (-1,0,1) or type(k_ar_diff) is not int or k_ar_diff<1:
        raise ValueError("det_order must be -1,0,1; k_ar_diff>=1")
    if significance not in (.90,.95,.99):
        raise ValueError("significance must be one of .90, .95, .99")
    result=coint_johansen(x,det_order,k_ar_diff)
    ix={.90:0,.95:1,.99:2}[significance]
    rows=[]
    for rank in range(x.shape[1]):
        statistic=float(result.lr1[rank])
        critical=float(result.cvt[rank,ix])
        if not np.isfinite(critical):
            raise ValueError("Johansen critical values unavailable for this dimension")
        rows.append({
            "null_rank_at_most":rank,"trace_statistic":statistic,
            "critical_value":critical,"reject_null":bool(statistic>critical),
        })
    return {
        "table":pd.DataFrame(rows),"n_samples":x.shape[0],
        "interpretation_ar":"اختبار أثر Johansen لرتبة التكامل المشترك؛ يتطلب متغيرات مناسبة، واختيار إبطاءات واتجاهات محددًا مسبقًا.",
        "interpretation_en":"Johansen trace cointegration ranks require suitable integration orders, deterministic terms and lag selection.",
    }


def diebold_mariano_test(actual, forecast_a, forecast_b, *, loss="squared", horizon=1, alpha=.05):
    """Two-sided asymptotic DM test with Bartlett HAC for overlapping forecasts.

    No small-sample Harvey correction; tests loss equality on same-horizon
    paired out-of-sample forecasts, assuming suitable loss-differential mixing.
    """
    _alpha(alpha)
    a=np.asarray(actual,dtype=float).reshape(-1)
    x=np.asarray(forecast_a,dtype=float).reshape(-1)
    y=np.asarray(forecast_b,dtype=float).reshape(-1)
    n=len(a)
    if n<30 or not (n==len(x)==len(y)) or not np.isfinite(a).all() or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Need >=30 aligned finite held-out forecast observations")
    if loss not in ("squared","absolute") or type(horizon) is not int or not 1<=horizon<=n//4:
        raise ValueError("loss must be squared/absolute and horizon within sample")
    d=((a-x)**2-(a-y)**2 if loss=="squared" else np.abs(a-x)-np.abs(a-y))
    demean=d-d.mean()
    gamma0=float(np.dot(demean,demean)/n)
    variance=gamma0
    for lag in range(1,horizon):
        gamma=float(np.dot(demean[lag:],demean[:-lag])/n)
        variance+=2*(1-lag/horizon)*gamma
    if variance<=0:
        raise ValueError("Loss differential has zero/nonpositive long-run variance")
    statistic=float(d.mean()/np.sqrt(variance/n))
    return make_result("Diebold-Mariano forecast loss",statistic,
        2*stats.norm.sf(abs(statistic)),n,alpha,
        "متوسط فرق خسارة التنبؤ بين النموذجين يساوي صفرًا.",
        "تقريب طبيعي بتغاير HAC؛ اختبر نفس أفق التنبؤ وبيانات اختبار خارج العينة.",
        null_en="Mean paired forecast loss difference between models is zero.",
        notes_en="Asymptotic Bartlett-HAC; use same-horizon out-of-sample forecasts, no small-sample correction.",
    )
