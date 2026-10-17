"""Specialist unit-root inference with an optional, independently tested backend.

The Phillips-Perron (PP) p-value is the arch package's MacKinnon regression
surface approximation; it is not a standard normal or simple Student-t test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def phillips_perron_unit_root(series, *, trend="c", test_type="tau",
                             lags=None, alpha=.05):
    """Phillips–Perron unit-root test, using arch.unitroot.PhillipsPerron.

    Null: the observed series has a unit root. Alternative: stationarity
    around chosen deterministic specification: none, constant or linear
    trend. Bandwidth lags control Newey–West long-run variance estimation,
    NOT an ADF-style number of lagged difference regressors. No handling of
    structural breaks, seasonality, multiple hypotheses or irregular time
    gaps is implied. Validate your sequence's sampling and pretest choices.
    """
    try:
        x=np.asarray(series,dtype=float)
    except (TypeError,ValueError) as exc:
        raise ValueError("series must be a numeric 1D time series") from exc
    if (x.ndim!=1 or len(x)<30 or len(x)>2_000_000
        or not np.isfinite(x).all() or np.std(x)<=1e-12):
        raise ValueError("Require 30..2,000,000 finite, varying sequential observations")
    if (trend not in ("n","c","ct") or test_type not in ("tau","rho")
        or lags is not None and (type(lags) is not int or not 0<=lags<len(x)//2)
        or isinstance(alpha,(bool,np.bool_)) or not np.isscalar(alpha)
        or not np.isfinite(alpha) or not 0<alpha<1):
        raise ValueError("trend n/c/ct, type tau/rho, valid NW bandwidth and alpha required")
    try:
        from arch.unitroot import PhillipsPerron
    except ImportError as exc:
        raise ImportError('Install with: pip install "khaled-dsml[unitroot]"') from exc
    model=PhillipsPerron(x,lags=lags,trend=trend,test_type=test_type)
    statistic=float(model.stat)
    p=float(model.pvalue)
    if not np.isfinite(statistic) or not np.isfinite(p) or not 0<=p<=1:
        raise ValueError("PP test was not estimable on the supplied data")
    critical=model.critical_values
    table=pd.DataFrame([
        {"level":str(name),"critical_value":float(value)}
        for name,value in critical.items()
    ])
    return {
        "test":"Phillips-Perron unit-root test",
        "n_samples":len(x),"trend":trend,"test_type":test_type,
        "nw_lags":int(model.lags),"statistic":statistic,
        "p_value":p,"alpha":float(alpha),
        "reject_unit_root":bool(p<alpha),
        "table":table,
        "null_hypothesis_ar":"تحتوي السلسلة على جذر وحدة وفق الصياغة الحتمية المختارة.",
        "null_hypothesis_en":"The time series has a unit root under the selected deterministic specification.",
        "interpretation_ar":(
            "اختبار فيليبس–بيرون لجذر الوحدة بمرجع arch، مع "
            "تصحيح الارتباط التسلسلي عبر تباين Newey–West طويل الأمد. "
            "الفرض الصفري وجود جذر وحدة؛ عدم رفضه لا يثبت يقينًا "
            "عدم الاستقرارية. الافتراضات تتأثر باتجاه الزمن والانقطاعات "
            "الهيكلية والموسمية وخيارات التأخر."
        ),
        "interpretation_en":(
            "Phillips–Perron unit-root test delegated to the documented "
            "arch implementation, using Newey–West long-run variance and "
            "MacKinnon approximate p-values. The null is a unit root; "
            "failure to reject does not prove nonstationarity. "
            "Trend choice, structural breaks and serial dependence matter."
        ),
    }
