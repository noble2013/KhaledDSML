"""Bounded-memory quantile sketches backed by the established tdigest package.

Quantiles are approximations: neither exact sorted order statistics nor
calibrated confidence intervals are implied. Streaming weights denote
frequency counts, not survey-design weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tdigest_quantile_summary(values, *, probabilities=(.05,.25,.5,.75,.95),
                             compression=.01, weights=None):
    """Approximate quantiles with TDigest centroids and optional frequencies.

    probabilities are fractions (0..1); TDigest.percentile takes 0..100.
    The algorithm is order-sensitive and stochastic in some implementations;
    exact quantile equality is not promised. No full data are retained
    beyond the in-memory iterable supplied to this convenience function.
    """
    if (isinstance(compression,(bool,np.bool_)) or not np.isscalar(compression)
        or not np.isfinite(compression) or not 0<compression<=.5):
        raise ValueError("compression must be numeric in (0,.5]")
    try:
        probs=np.asarray(probabilities,dtype=float)
    except (TypeError,ValueError) as exc:
        raise ValueError("probabilities must be numeric") from exc
    if (probs.ndim!=1 or not 1<=len(probs)<=50
        or not np.isfinite(probs).all() or np.any(probs<0) or np.any(probs>1)
        or np.any(np.diff(probs)<0)):
        raise ValueError("probabilities must be sorted finite fractions inside [0,1]")
    try:
        from tdigest import TDigest
    except ImportError as exc:
        raise ImportError('Install with: pip install "khaled-dsml[quantiles]"') from exc
    try:
        arr=np.asarray(values,dtype=float)
    except (TypeError,ValueError) as exc:
        raise ValueError("values must be a finite numeric iterable") from exc
    if arr.ndim!=1 or len(arr)<10 or not np.isfinite(arr).all():
        raise ValueError("Need at least 10 finite observations")
    if weights is None:
        ws=np.ones(len(arr),dtype=int)
    else:
        try: wf=np.asarray(weights,dtype=float)
        except (TypeError,ValueError) as exc:
            raise ValueError("weights must be finite positive integer frequencies") from exc
        if (wf.shape!=arr.shape or not np.isfinite(wf).all()
            or not np.array_equal(wf,np.floor(wf))
            or np.any(wf<=0) or np.any(wf>1_000_000)):
            raise ValueError("weights must align with values and be positive integer frequencies")
        ws=wf.astype(np.int64)
    if ws.sum()>20_000_000:
        raise ValueError("Weighted count exceeds safety cap")
    digest=TDigest(delta=float(compression))
    for v,w in zip(arr,ws):
        digest.update(float(v),int(w))
    quantiles=[float(digest.percentile(float(p)*100.)) for p in probs]
    if not np.isfinite(quantiles).all():
        raise ValueError("Nonfinite digest quantile returned")
    qtable=pd.DataFrame({"quantile_probability":probs,
                         "estimated_value":quantiles})
    return {
        "test":"t-digest approximate streaming quantiles",
        "n_samples":int(ws.sum()),"n_input_rows":len(arr),
        "compression":float(compression),
        "n_centroids":len(digest.centroids_to_list()),
        "mean_from_data":float(np.average(arr,weights=ws)),
        "table":qtable,
        "interpretation_ar":(
            "حساب تقريبي للكميات المئوية باستخدام t-digest من "
            "مكتبة خارجية مستقلة. النتائج ليست رتبًا دقيقة ولا "
            "فواصل ثقة، وتعتمد جودة التقريب على التوزيع وترتيب "
            "التحديث ومعامل الضغط. الأوزان تكرارات صحيحة "
            "وليست أوزان معاينة احتمالية."
        ),
        "interpretation_en":(
            "Approximate streaming quantiles using the third-party "
            "t-digest centroid algorithm. Percentiles are not exact "
            "order statistics or confidence intervals. Accuracy depends "
            "on compression, value distribution and update ordering. "
            "Weights are integer frequencies, not survey weights."
        ),
    }
