"""Transparent, conservative uncompressed ADWIN-style bounded-stream monitor.

This is an adaptive window with Hoeffding cut tests, not a bucket-compressed
implementation of Bifet and Gavaldà's optimized ADWIN2. Its global union
bound assumes bounded independent observations within a stationary interval
and a prespecified finite monitoring horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def adwin_bounded_mean_monitor(values, *, delta=.01, min_subwindow=15,
                               max_observations=1500):
    """Adaptive-window detector with explicit familywise Hoeffding bounds.

    At each time inspect every admissible window split; if
    |mean(left)-mean(right)| > sqrt(.5*log(2*T*T/delta)*(1/nL+1/nR)),
    discard the older prefix (one cut per time-step). T=len(values) is
    fixed in advance; the union bound over <=T² total comparisons controls
    an overall false-positive upper bound <=delta *within independent
    bounded iid stationary segments*, without claiming canonical ADWIN2.
    """
    try:
        x=np.asarray(values,dtype=float)
    except (ValueError,TypeError) as exc:
        raise ValueError("Bounded 0..1 numeric series required") from exc
    if (x.ndim!=1 or len(x)<30 or not np.isfinite(x).all()
        or np.any(x<0) or np.any(x>1)
        or isinstance(delta,(bool,np.bool_)) or not np.isscalar(delta)
        or not np.isfinite(delta) or not 0<delta<1
        or type(min_subwindow) is not int or not 5<=min_subwindow<=len(x)//2
        or type(max_observations) is not int or len(x)>max_observations):
        raise ValueError("Require bounded finite stream, 0<delta<1 and valid subwindow minimum")
    logterm=float(np.log(2*len(x)*len(x)/delta))
    window_start=0
    alarms=[]
    trace=[]
    for time in range(len(x)):
        current=x[window_start:time+1]
        n=len(current)
        best=None
        if n>=2*min_subwindow:
            sums=np.r_[0.,current.cumsum()]
            for split in range(min_subwindow,n-min_subwindow+1):
                nl=split;nr=n-split
                discrepancy=abs(sums[split]/nl-
                                (sums[n]-sums[split])/nr)
                bound=np.sqrt(.5*logterm*(1/nl+1/nr))
                if discrepancy>bound:
                    strength=discrepancy-bound
                    if best is None or strength>best[0]:
                        best=(strength,split,discrepancy,bound)
        kind=""
        if best is not None:
            old_start=window_start
            new_start=old_start+best[1]
            alarms.append({
                "detection_index":time,
                "new_window_start":new_start,
                "mean_discrepancy":float(best[2]),
                "hoeffding_bound":float(best[3]),
            })
            window_start=new_start
            kind="adaptive_window_change"
        trace.append({
            "index":time,"value":float(x[time]),
            "window_start":window_start,
            "window_n":time-window_start+1,"alarm":kind,
        })
    return {
        "test":"Uncompressed ADWIN-style Hoeffding window",
        "n_samples":len(x),"delta_horizon_bound":float(delta),
        "min_subwindow":min_subwindow,
        "n_alarms":len(alarms),
        "alarm_indices":[int(a["detection_index"]) for a in alarms],
        "alarm_details":alarms,
        "table":pd.DataFrame(trace),
        "interpretation_ar":(
            "كاشف نافذة تكيفية غير مضغوطة باختبارات حدود Hoeffding "
            "التحفظية على جميع نقاط التقسيم. يعتمد ضبط الاحتمال "
            "على استقلال المشاهدات وقيم ضمن [0،1] وأفق مراقبة معلوم. "
            "ليس تنفيذ ADWIN2 المضغوط ولا اختبارًا سببيًا."
        ),
        "interpretation_en":(
            "Uncompressed ADWIN-style adaptive window using a conservative "
            "finite-horizon Hoeffding union bound for bounded iid values. "
            "Not the optimized bucket-based ADWIN2 algorithm. "
            "Serial dependence, unknown horizons or selection invalidate "
            "the nominal bound."
        ),
    }
