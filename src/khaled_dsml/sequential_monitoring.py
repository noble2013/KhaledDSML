"""Sequential change alarms; descriptive stream monitors, NOT p-value tests.

Page–Hinkley requires application-specific calibration of delta/threshold/
warmup to a baseline stream and acceptable false-alarm rate. Monitoring
dependent, seasonal or drifting streams without adjustment is unreliable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def page_hinkley_change_monitor(
    values, *, delta=0.01, threshold=10.0, min_instances=30,
    direction="both", reset_after_alarm=True,
):
    """Page–Hinkley detector with running mean and cumulative deviation.

    Warm-up observations update the online mean and cumulative statistic but
    are not eligible for alarms. The two monitored directions are positive
    (upward shift) and negative (downward shift). Alarms are zero-based row
    positions. When reset_after_alarm=True, the next observation starts a
    new segment; each segment gets a fresh warm-up. With reset disabled,
    only the first alarm per direction per continuing run is reported.

    This is an *un-calibrated* diagnostic: do not treat alarm locations as
    tests with type-I error control or as causal change-point estimates.
    """
    try:
        data = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("values must be finite numbers") from exc
    if data.ndim != 1 or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("Need at least 2 finite univariate observations")
    if (isinstance(delta, (bool, np.bool_)) or not np.isscalar(delta) or not np.isfinite(delta)
        or delta < 0 or isinstance(threshold, (bool, np.bool_)) or not np.isscalar(threshold)
        or not np.isfinite(threshold) or threshold <= 0
        or type(min_instances) is not int or not 2 <= min_instances <= len(data)
        or direction not in ("both", "up", "down")
        or not isinstance(reset_after_alarm, bool)):
        raise ValueError("Require finite delta>=0, threshold>0, min_instances>=2, supported direction")
    count, mean = 0, 0.0
    up, up_min, down, down_min = 0., 0., 0., 0.
    up_seen, down_seen = False, False
    alarm_rows = []
    directions = []
    trace = []
    for index, x in enumerate(data):
        count += 1
        mean += (float(x) - mean) / count
        up += float(x)-mean-delta
        down += mean-float(x)-delta
        up_min = min(up_min, up)
        down_min = min(down_min, down)
        up_score, down_score = up - up_min, down - down_min
        flag_up = (count >= min_instances and direction in ("both","up")
                   and not up_seen and up_score > threshold)
        flag_down = (count >= min_instances and direction in ("both","down")
                     and not down_seen and down_score > threshold)
        kind = "both" if flag_up and flag_down else ("up" if flag_up else ("down" if flag_down else ""))
        if kind:
            alarm_rows.append(index)
            directions.append(kind)
            if reset_after_alarm:
                count, mean = 0, 0.
                up, up_min, down, down_min = 0., 0., 0., 0.
                up_seen, down_seen = False, False
            else:
                if flag_up:
                    up_seen = True
                if flag_down:
                    down_seen = True
        trace.append({
            "index": index, "value": float(x),
            "upward_score": float(up_score), "downward_score": float(down_score),
            "alarm_direction": kind,
        })
    return {
        "test": "Page-Hinkley sequential change monitor",
        "n_samples": len(data), "n_alarms": len(alarm_rows),
        "alarm_indices": alarm_rows, "alarm_directions": directions,
        "delta": float(delta), "threshold": float(threshold),
        "min_instances": min_instances, "reset_after_alarm": reset_after_alarm,
        "table": pd.DataFrame(trace),
        "interpretation_ar": (
            "كاشف تغير تسلسلي وصفي؛ مواقع الإنذار تعتمد على معلمات العتبة "
            "والسماح والانطلاقة. ليست إنذاراته قيم p أو دليلاً سببيًا. "
            "يلزم ضبط العتبات على بيانات مرجعية وفق معدل الإنذار الكاذب المقبول."
        ),
        "interpretation_en": (
            "Sequential Page–Hinkley alarms are descriptive thresholds, "
            "not statistical p-values, controlled false-positive rates or "
            "causal change-point estimates. Calibrate on a representative "
            "reference stream, considering autocorrelation and seasonality."
        ),
    }
