"""Time-series diagnostics and forecast metrics for applied economic teaching.

This module never sorts observations, fills missing dates or silently drops NaNs:
a regular, chronological series must be prepared before testing. These tests
describe statistical associations and do not establish causal relations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._results import make_result


def _ordered_numeric(values, *, minimum=10) -> np.ndarray:
    """Validate ordered finite observations without destroying time alignment."""
    try:
        x = pd.to_numeric(pd.Series(values), errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Provide a chronological numeric series") from exc
    if len(x) < minimum or not np.all(np.isfinite(x)):
        raise ValueError(
            f"Need at least {minimum} finite, consecutive, time-ordered observations; "
            "missing values must be handled explicitly"
        )
    return x


def _validate_lags(lags, n):
    if type(lags) is not int or not 1 <= lags < n:
        raise ValueError("lags must be an integer with 1 <= lags < sample size")


def _validate_alpha(alpha):
    if not isinstance(alpha, (int, float)) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")


def mann_kendall_trend_test(values, *, alpha=0.05):
    """Two-sided, tie-corrected, asymptotic Mann–Kendall trend test.

    Uses a continuity-corrected normal approximation. Requires >=10 samples.
    Serial dependence and seasonal patterns invalidate naive p-values; use
    a prewhitened/seasonal method instead when appropriate.
    """
    _validate_alpha(alpha)
    x = _ordered_numeric(values, minimum=10)
    n = len(x)
    if np.all(x == x[0]):
        raise ValueError("A constant series has no identifiable trend variation")

    # S is positive if later observations tend to exceed earlier observations.
    s = int(sum(int(np.sign(x[i + 1:] - x[i]).sum()) for i in range(n - 1)))
    tie_sizes = np.unique(x, return_counts=True)[1]
    tie_correction = sum(int(t * (t - 1) * (2 * t + 5)) for t in tie_sizes if t > 1)
    variance = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18
    if variance <= 0:
        raise ValueError("Cannot estimate trend variance for this series")
    z = (s - np.sign(s)) / np.sqrt(variance) if s != 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))
    tau = float(stats.kendalltau(np.arange(n), x).statistic)
    direction_ar = "تصاعدي" if s > 0 else "تنازلي" if s < 0 else "دون اتجاه واضح"
    direction_en = "increasing" if s > 0 else "decreasing" if s < 0 else "no clear"
    return make_result(
        "Mann-Kendall trend (normal approximation)", z, p_value, n, alpha,
        "لا يوجد اتجاه رتيب منتظم في السلسلة الزمنية.",
        f"الاتجاه المشاهد: {direction_ar}؛ Kendall tau = {tau:.3f}؛ S = {s}. "
        "قد يؤدي الارتباط الذاتي أو الموسمية إلى قيمة احتمالية غير صالحة.",
        null_en="There is no monotonic trend in the chronological series.",
        notes_en=f"Observed trend: {direction_en}; Kendall tau = {tau:.3f}; S = {s}. "
                 "Serial correlation or seasonality can invalidate this p-value.",
    )


def pettitt_change_point_test(values, *, alpha=0.05) -> dict:
    """Approximate single-change-point Pettitt rank test.

    change_index identifies the last observation in the first segment; both
    segments must contain at least one observation. P-value is an approximation
    assuming independent observations; ties may reduce its accuracy.
    """
    _validate_alpha(alpha)
    x = _ordered_numeric(values, minimum=10)
    if np.all(x == x[0]):
        raise ValueError("Pettitt test is undefined for an entirely constant series")
    n = len(x)
    ranks = stats.rankdata(x, method="average")
    cuts = np.arange(1, n)
    u = 2 * np.cumsum(ranks)[:-1] - cuts * (n + 1)
    position = int(np.argmax(np.abs(u)))
    k = float(abs(u[position]))
    p = float(min(1.0, 2 * np.exp(-6 * k ** 2 / (n ** 3 + n ** 2))))
    reject = bool(p < alpha)
    return {
        "test": "Pettitt single change-point",
        "statistic": k,
        "p_value": p,
        "alpha": float(alpha),
        "reject_null": reject,
        "change_index": position,
        "first_segment_size": position + 1,
        "n_samples": n,
        "interpretation_ar": (
            f"أقوى نقطة مرشحة للتغير تقع بعد المشاهدة رقم {position + 1}. "
            + ("توجد قرينة إحصائية على تغير بالموقع." if reject
               else "لا نرفض فرضية غياب تغير بالموقع؛ لا يعني هذا إثبات الثبات.")
            + " الاختبار تقريبي ويفترض استقلال المشاهدات.",
        ),
        "interpretation_en": (
            f"The strongest candidate split is after observation {position + 1}. "
            + ("Evidence suggests a change in location." if reject
               else "No change in location is detected at the selected alpha; stability is not proven.")
            + " This is an approximation requiring independent observations.",
        ),
    }


def box_pierce_test(residuals, *, lags=10, alpha=0.05):
    """Box–Pierce white-noise diagnostic with a chi-square approximation.

    Use residuals from an appropriately fitted time-series model, not arbitrary
    shuffled observations. No automatic degrees-of-freedom correction is used.
    """
    _validate_alpha(alpha)
    x = _ordered_numeric(residuals, minimum=12)
    _validate_lags(lags, len(x))
    if np.ptp(x) == 0:
        raise ValueError("Residuals are constant")
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
    except ImportError as exc:
        raise ImportError('Install khaled-dsml[econometrics] for Box-Pierce') from exc
    table = acorr_ljungbox(x, lags=[lags], boxpierce=True, return_df=True)
    return make_result(
        "Box-Pierce", float(table["bp_stat"].iloc[0]),
        float(table["bp_pvalue"].iloc[0]), len(x), alpha,
        f"لا يوجد ارتباط ذاتي للبواقي حتى الإبطاء {lags}.",
        "القيمة الاحتمالية تقريبية؛ تعتمد على ملاءمة النموذج وتصحيح درجات الحرية.",
        null_en=f"The residuals have no autocorrelation through lag {lags}.",
        notes_en="Chi-square approximation; model fitting and degrees-of-freedom adjustments matter.",
    )


def arch_lm_test(residuals, *, lags=5, alpha=0.05):
    """Engle ARCH-LM test for conditional heteroskedasticity of residuals."""
    _validate_alpha(alpha)
    x = _ordered_numeric(residuals, minimum=12)
    _validate_lags(lags, len(x))
    if len(x) <= lags + 5 or np.ptp(x) == 0:
        raise ValueError("Require enough nonconstant residuals beyond the chosen lags")
    try:
        from statsmodels.stats.diagnostic import het_arch
    except ImportError as exc:
        raise ImportError('Install khaled-dsml[econometrics] for ARCH-LM') from exc
    statistic, p_value, _, _ = het_arch(x, nlags=lags)
    return make_result(
        "Engle ARCH-LM", statistic, p_value, len(x), alpha,
        f"لا يوجد أثر ARCH حتى الإبطاء {lags} في مربعات البواقي.",
        "تشخيص تباين شرطي، وليس دليلًا على وجود علاقة سببية.",
        null_en=f"No ARCH effects in squared residuals through lag {lags}.",
        notes_en="Tests conditional heteroskedasticity, not causation.",
    )


def mean_absolute_scaled_error(actual, prediction, training, *, seasonality=1) -> float:
    """MASE: forecast MAE divided by seasonal-naive training MAE.

    The training history MUST predate the predictions. Seasonality is an
    integer >=1 and smaller than the training sample; a zero denominator
    makes MASE undefined and raises ValueError.
    """
    if type(seasonality) is not int or seasonality < 1:
        raise ValueError("seasonality must be a positive integer")
    y = _ordered_numeric(actual, minimum=1)
    yhat = _ordered_numeric(prediction, minimum=1)
    history = _ordered_numeric(training, minimum=2)
    if len(y) != len(yhat):
        raise ValueError("actual and prediction must be aligned")
    if seasonality >= len(history):
        raise ValueError("Need more training observations than the seasonal period")
    denominator = np.mean(np.abs(history[seasonality:] - history[:-seasonality]))
    if denominator == 0:
        raise ValueError("MASE is undefined for a constant seasonal naive training baseline")
    return float(np.mean(np.abs(y - yhat)) / denominator)


def forecast_accuracy_report(actual, prediction, training, *, seasonality=1) -> dict:
    """Report MAE, RMSE, MASE and safe sMAPE with Arabic/English interpretation.

    sMAPE ignores terms with both actual and forecast equal to zero (contributes
    zero error). Such terms use a zero denominator limit by convention.
    """
    y = _ordered_numeric(actual, minimum=1)
    pred = _ordered_numeric(prediction, minimum=1)
    if len(y) != len(pred):
        raise ValueError("actual and prediction must be aligned")
    abs_error = np.abs(y - pred)
    scale = np.abs(y) + np.abs(pred)
    smape_terms = np.divide(
        2 * abs_error, scale, out=np.zeros_like(abs_error), where=scale != 0,
    )
    mase = mean_absolute_scaled_error(y, pred, training, seasonality=seasonality)
    return {
        "n_samples": len(y),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "mase": mase,
        "smape_percent": float(100 * smape_terms.mean()),
        "seasonality": seasonality,
        "interpretation_ar": (
            f"بلغ MASE القيمة {mase:.3f}. إذا كان أقل من 1، فإن متوسط خطأ التنبؤ "
            "أصغر من متوسط خطأ المرجع الموسمي الساذج على بيانات التدريب. "
            "المقارنة خارج العينة وبنفس الفترات ضرورية."
        ),
        "interpretation_en": (
            f"MASE = {mase:.3f}. If below 1, mean forecast absolute error is "
            "smaller than the seasonal-naive scale computed on the training history. "
            "Meaningful conclusions require out-of-sample, same-horizon evaluation."
        ),
    }
