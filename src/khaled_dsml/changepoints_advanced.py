"""Sequential diagnostics with precisely stated calibration and limitations.

PELT is an EXACT minimizer of the penalized Gaussian mean-shift SSE objective,
not a p-value test. DDM is a heuristic rate monitor and gives no controlled
false-alarm probability, especially for nonstationary/serially dependent errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pelt_mean_shift(values, *, penalty, min_segment_size=5, max_observations=3000):
    """Pruned Exact Linear Time (PELT) under piecewise constant SSE.

    Minimizes sum_segment sum((x-mean_segment)**2) + penalty*#changes.
    The PELT exact-pruning condition holds for this loss with K=0:
    splitting a segment cannot increase within-segment SSE. Does NOT claim
    linear complexity on all inputs; worst-case search remains quadratic.
    Penalty is chosen by the analyst, not auto-calibrated to false alarms.
    Change indexes denote the first index of the new segment.
    """
    try:x=np.asarray(values,dtype=float)
    except (TypeError,ValueError) as exc:
        raise ValueError("Finite numeric change-point signal required") from exc
    if (x.ndim!=1 or len(x)<10 or not np.isfinite(x).all()
        or type(min_segment_size) is not int or min_segment_size<2
        or 2*min_segment_size>len(x)
        or type(max_observations) is not int or len(x)>max_observations
        or isinstance(penalty,(bool,np.bool_)) or not np.isscalar(penalty)
        or not np.isfinite(penalty) or penalty<=0):
        raise ValueError("Finite 1D signal, min segment >=2, n within bounded limit, positive penalty")
    n=len(x)
    prefix=np.r_[0,np.cumsum(x)]
    prefixsq=np.r_[0,np.cumsum(x*x)]
    def loss(a,b):
        total=prefix[b]-prefix[a]
        cost=float(prefixsq[b]-prefixsq[a]-total*total/(b-a))
        return max(0.,cost)
    dp=np.full(n+1,np.inf)
    dp[0]=-float(penalty)
    prev=np.full(n+1,-1,dtype=int)
    # Candidate start s enters at t=s+min_segment_size and can be
    # dropped only after its segment has become valid.
    candidates=[]
    evaluated=0
    for t in range(min_segment_size,n+1):
        s=t-min_segment_size
        if np.isfinite(dp[s]):candidates.append(s)
        if not candidates:continue
        costs=[]
        for start in candidates:
            cost=dp[start]+loss(start,t)+penalty
            costs.append((float(cost),int(start)))
            evaluated+=1
        best,split=min(costs)
        dp[t]=best;prev[t]=split
        # PELT pruning requires F(s)+C(s,t) <= F(t) for retaining s;
        # equality is retained for correctness under ties.
        candidates=[
            start for val,start in costs
            if val-penalty <= best+1e-10*max(1.,abs(best))
        ]
    if not np.isfinite(dp[n]) or prev[n]<0:
        raise ValueError("No permissible segmentation found")
    endpoints=[]
    current=n
    while current>0:
        endpoints.append(current)
        current=int(prev[current])
        if current<0:raise ValueError("Corrupt dynamic-programming path")
    endpoints.append(0)
    endpoints=endpoints[::-1]
    changes=endpoints[1:-1]
    table=pd.DataFrame([
        {"segment":i+1,"start_index":int(a),"end_exclusive":int(b),
         "n":int(b-a),"mean":float(x[a:b].mean()),
         "within_ss":loss(a,b)}
        for i,(a,b) in enumerate(zip(endpoints[:-1],endpoints[1:]))
    ])
    return {
        "test":"PELT penalized squared-loss mean shifts",
        "n_samples":n,"penalty":float(penalty),
        "min_segment_size":min_segment_size,
        "n_changes":len(changes),"change_indices":list(map(int,changes)),
        "penalized_objective":float(dp[n]),
        "n_evaluated_segments":evaluated,
        "table":table,
        "interpretation_ar":(
            "خوارزمية PELT تعطي تقسيمًا أمثل لدالة كلفة مربعات الأخطاء "
            "ضمن عقوبة محددة مسبقًا وطول أدنى للمقاطع. ليست اختبار "
            "فرضيات ولا تقدم قيمة p أو ضمان عدد تغيّرات حقيقي. "
            "يعتمد التقدير على المقياس والضوضاء والعقوبة المختارة."
        ),
        "interpretation_en":(
            "Exact PELT pruning for penalized within-segment squared loss "
            "with prespecified penalty and minimum segment length. "
            "No calibrated p-value or guaranteed linear runtime; "
            "changes reflect the selected objective, not causation."
        ),
    }


def ddm_binary_error_monitor(errors, *, min_instances=30, warning_level=2.,
                             drift_level=3., reset_after_drift=True):
    """DDM: Gama et al. drift detection in sequential binary error events.

    At each observation, p=observed error rate, s=sqrt(p(1-p)/n);
    minimum p+s is recorded after min_instances. Warning threshold
    p_min+warning_level*s_min and drift threshold p_min+drift_level*s_min.
    Monitoring presumes stationary, approximately iid Bernoulli error before
    drift. This implementation does NOT calibrate global false-alarm risk.
    When resetting, each detected drift starts a fresh segment on the NEXT
    event. Zero-variance early windows are particularly fragile.
    """
    arr=np.asarray(errors)
    if (arr.ndim!=1 or len(arr)<30 or arr.dtype.kind not in "biu"
        or not np.isin(arr,[0,1]).all()
        or type(min_instances) is not int or not 20<=min_instances<=len(arr)
        or not np.isscalar(warning_level) or not np.isscalar(drift_level)
        or isinstance(warning_level,(bool,np.bool_))
        or isinstance(drift_level,(bool,np.bool_))
        or not np.isfinite(warning_level) or not np.isfinite(drift_level)
        or not 0<warning_level<drift_level
        or not isinstance(reset_after_drift,bool)):
        raise ValueError("Binary error stream, min>=20, and 0<warning<drift levels required")
    n=0;err=0;p_min=np.inf;s_min=np.inf
    flags=[];records=[]
    for idx,obs in enumerate(arr):
        n+=1;err+=int(obs)
        p=err/n;s=float(np.sqrt(p*(1-p)/n))
        stage="warmup"
        if n>=min_instances:
            if p+s<p_min+s_min:
                p_min=p;s_min=s
            if p+s>p_min+drift_level*s_min:
                stage="drift";flags.append(idx)
            elif p+s>p_min+warning_level*s_min:
                stage="warning"
            else:
                stage="stable"
        records.append({
            "index":idx,"error":int(obs),"segment_n":n,
            "error_rate":p,"standard_error":s,"state":stage,
        })
        if stage=="drift" and reset_after_drift:
            n=0;err=0;p_min=np.inf;s_min=np.inf
    return {
        "test":"DDM binary prediction-error drift monitor",
        "n_samples":len(arr),"n_drift_alarms":len(flags),
        "drift_indices":list(map(int,flags)),
        "min_instances":min_instances,"reset_after_drift":reset_after_drift,
        "table":pd.DataFrame(records),
        "interpretation_ar":(
            "يراقب DDM معدل خطأ التنبؤ عبر الزمن وخطأه المعياري "
            "مقارنةً بأفضل وضع سابق بعد فترة انطلاق. التنبيه "
            "استكشافي غير معاير لمعدل الإنذارات الكاذبة، "
            "ويضعف مع الاعتماد الزمني أو تغير توزيع الصعوبة."
        ),
        "interpretation_en":(
            "DDM sequentially compares binary error rate plus binomial "
            "SE against its best previous reference after warm-up. "
            "Alarms are heuristic: no familywise false-alarm guarantee, "
            "and serial dependence or changing case mix harms validity."
        ),
    }
