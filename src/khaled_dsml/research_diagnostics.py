"""Held-out model diagnostics and research-output arithmetic checks.

All significance calculations here require their stated sampling designs.
No metric detects fabricated data, publication bias or causal effects.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import numpy as np
import pandas as pd
from scipy import stats


def _vector(values, name):
    a = np.asarray(values)
    if a.ndim != 1 or len(a) < 2:
        raise ValueError(f"{name} must be a 1D array of at least 2 observations")
    if bool(np.asarray(pd.isna(a)).any()):
        raise ValueError(f"{name} must not contain missing labels")
    return a


def mcnemar_paired_predictions(
    y_true, predictions_a, predictions_b, *, exact=True, continuity_correction=True, alpha=.05
):
    """McNemar comparison on the SAME held-out records for any class labels.

    H0: P(A correct, B wrong) = P(A wrong, B correct). Exact mode applies
    two-sided binomtest to discordant counts. Asymptotic mode optionally
    applies continuity correction. The test is for paired, independent cases,
    NOT for independent test sets, repeated resampling or predictions fitted
    on the evaluation records. It tests error-rate symmetry, not calibrated
    probabilities, severity of errors or model quality in general.
    """
    truth = _vector(y_true, "y_true")
    a = _vector(predictions_a, "predictions_a")
    b = _vector(predictions_b, "predictions_b")
    if len(a) != len(truth) or len(b) != len(truth):
        raise ValueError("Paired prediction arrays must share length and ordering")
    if not isinstance(exact, (bool, np.bool_)) or not isinstance(
        continuity_correction, (bool, np.bool_)
    ):
        raise ValueError("exact and continuity_correction must be booleans")
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    ca = truth == a
    cb = truth == b
    both = int(np.count_nonzero(ca & cb))
    only_a = int(np.count_nonzero(ca & ~cb))
    only_b = int(np.count_nonzero(~ca & cb))
    neither = int(np.count_nonzero(~ca & ~cb))
    discordant = only_a + only_b
    if discordant == 0:
        statistic, p = (0.0, 1.0)
    elif exact:
        statistic = float(min(only_a, only_b))
        p = float(stats.binomtest(only_a, discordant, .5, alternative="two-sided").pvalue)
    else:
        correction = 1.0 if continuity_correction else 0.0
        statistic = float(max(0.0, abs(only_a-only_b)-correction)**2 / discordant)
        p = float(stats.chi2.sf(statistic, 1))
    return {
        "test": "McNemar paired prediction comparison",
        "n_samples": len(truth), "both_correct": both, "only_a_correct": only_a,
        "only_b_correct": only_b, "both_incorrect": neither,
        "n_discordant": discordant, "statistic": statistic, "p_value": p,
        "alpha": float(alpha), "reject_null": bool(p < alpha),
        "method": "exact binomial" if exact else (
            "chi-square with continuity correction" if continuity_correction
            else "chi-square without continuity correction"
        ),
        "accuracy_a": float(ca.mean()), "accuracy_b": float(cb.mean()),
        "null_hypothesis_ar": "احتمال تفوق أ على ب في الحالات المختلفة يساوي احتمال العكس.",
        "null_hypothesis_en": "Paired discordant correctness probabilities are equal.",
        "table": pd.DataFrame({
            "": ["Model B correct", "Model B incorrect"],
            "Model A correct": [both, only_a],
            "Model A incorrect": [only_b, neither],
        }),
        "interpretation_ar": (
            "اختبار ماكنمار يقارن تنبؤات نموذجين على الحالات نفسها، مع استقلال "
            "الحالات وعدم تدريب النموذجين على مجموعة التقييم. الاختبار الدقيق "
            "مناسب عندما تكون الحالات المختلفة قليلة؛ لا يثبت سببية الفروق."
        ),
        "interpretation_en": (
            "McNemar compares predictions from two models on exactly the same "
            "independent held-out cases. Exact inference uses discordant "
            "binomial counts; asymptotic chi-square can be inaccurate for "
            "small discordance. This is not a causal test."
        ),
    }


def mean_absolute_percentage_error_strict(actual, predicted):
    """Percentage MAPE with explicitly rejected zero actuals.

    Use only for finite numeric values, a nonzero outcome scale, and data
    for which relative error is meaningful. MAPE can strongly favor
    underforecasting and behave poorly for outcomes near zero or negative.
    """
    try:
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("actual and predicted must be numeric") from exc
    if (actual.ndim != 1 or predicted.ndim != 1 or len(actual) < 2
        or len(actual) != len(predicted) or not np.isfinite(actual).all()
        or not np.isfinite(predicted).all() or np.any(actual == 0)):
        raise ValueError("Need equal-length finite vectors, >=2 items, and no zero actuals")
    errors = np.abs((actual - predicted) / actual)
    return {
        "test": "Strict MAPE forecast accuracy",
        "n_samples": len(actual), "mape_percent": float(100*errors.mean()),
        "median_ape_percent": float(100*np.median(errors)),
        "table": pd.DataFrame({
            "actual": actual, "predicted": predicted, "absolute_percent_error": 100*errors,
        }),
        "interpretation_ar": (
            "مقياس خطأ نسبي وصفي فقط، يرفض القيم الحقيقية الصفرية صراحةً "
            "بدل استخدام مقامات اصطناعية. قد يضلل قرب الصفر وعند القيم السالبة؛ "
            "يفضل النظر معه إلى MAE وRMSE حسب سياق المسألة."
        ),
        "interpretation_en": (
            "Descriptive mean absolute percentage error; actual zeros "
            "are rejected rather than silently replaced by epsilons. "
            "MAPE behaves poorly near zero and can be inappropriate for "
            "negative or arbitrary-origin target scales."
        ),
    }


def _decimal(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        x = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not x.is_finite():
        raise ValueError(f"{label} must be finite")
    return x


def grim_mean_consistency(
    reported_mean, sample_size, *, decimals=2, minimum_score=1, maximum_score=5,
):
    """GRIM arithmetic consistency of a rounded integer-scale sample mean.

    Uses ROUND_HALF_UP explicitly. A failure indicates an arithmetic
    inconsistency under the stated integer bounds, n and rounding rule,
    not research misconduct. Passing does NOT verify the raw data.
    """
    if (type(sample_size) is not int or not 1 <= sample_size <= 1_000_000
        or type(decimals) is not int or not 0 <= decimals <= 6
        or type(minimum_score) is not int or type(maximum_score) is not int
        or maximum_score < minimum_score or not -100000 <= minimum_score <= maximum_score <= 100000):
        raise ValueError("Supply integer n, 0..6 decimal places and ordered integer score bounds")
    mean = _decimal(reported_mean, "reported_mean")
    q = Decimal(1).scaleb(-decimals)
    if mean != mean.quantize(q):
        raise ValueError("The reported mean has more precision than the specified decimals")
    proposed = (mean * sample_size).to_integral_value(rounding=ROUND_HALF_UP)
    low, high = minimum_score*sample_size, maximum_score*sample_size
    nearest = max(low, min(high, int(proposed)))
    implied = Decimal(nearest)/Decimal(sample_size)
    rounded = implied.quantize(q, rounding=ROUND_HALF_UP)
    consistent = bool(rounded == mean)
    return {
        "test": "GRIM integer-mean arithmetic consistency",
        "n_samples": sample_size, "reported_mean": str(mean),
        "decimals": decimals, "min_integer_score": minimum_score,
        "max_integer_score": maximum_score, "nearest_integer_total": nearest,
        "nearest_possible_mean": float(implied), "nearest_rounded_mean": str(rounded),
        "arithmetically_consistent": consistent,
        "interpretation_ar": (
            "فحص GRIM يتحقق من إمكانية تمثيل المتوسط المبلغ عنه بأعداد صحيحة "
            "في المجال المحدد وفق التقريب نصف للأعلى. عدم الاتساق تنبيه "
            "للمراجعة وليس دليلًا على تلاعب؛ والاتساق لا يثبت صحة البيانات."
        ),
        "interpretation_en": (
            "GRIM checks whether a reported rounded mean is arithmetically "
            "possible for bounded integer scores and the specified ROUND_HALF_UP "
            "rule. An inconsistency is not proof of misconduct; consistency "
            "does not verify the observations."
        ),
    }


def recalculate_t_test_p(statistic, degrees_of_freedom, *, alternative="two-sided",
                         reported_p=None, decimals=3):
    """Recalculate Student-t p-value and optionally check a rounded report.

    This is a consistency *screen*, not a way to infer the test's correct
    pairing, variance assumptions, effect size or hypothesis. Decimal
    comparison uses ROUND_HALF_UP, and exact inequalities (e.g. p<.001)
    are intentionally not parsed as rounded numeric reports.
    """
    if (isinstance(statistic, (bool,np.bool_)) or not np.isscalar(statistic)
        or not np.isfinite(statistic) or isinstance(degrees_of_freedom, (bool,np.bool_))
        or not np.isscalar(degrees_of_freedom) or not np.isfinite(degrees_of_freedom)
        or degrees_of_freedom <= 0 or alternative not in ("two-sided","greater","less")
        or type(decimals) is not int or not 0 <= decimals <= 8):
        raise ValueError("Finite t, positive df, valid tail and 0..8 decimals required")
    t = float(statistic)
    df = float(degrees_of_freedom)
    if alternative == "two-sided":
        p = float(2*stats.t.sf(abs(t),df))
    elif alternative == "greater":
        p = float(stats.t.sf(t,df))
    else:
        p = float(stats.t.cdf(t,df))
    quantizer = Decimal(1).scaleb(-decimals)
    rounded = Decimal(str(p)).quantize(quantizer, rounding=ROUND_HALF_UP)
    output = {
        "test": "Student t reported-p recalculation",
        "statistic": t, "degrees_of_freedom": df, "alternative": alternative,
        "p_value": p, "p_rounded": str(rounded), "decimals": decimals,
        "interpretation_ar": (
            "إعادة حساب قيمة p رياضيًا وفق توزيع Student t واتجاه الفرضية. "
            "المطابقة لا تتحقق من صحة التصميم أو عدد المشاهدات أو افتراضات "
            "الاختبار؛ اختلاف التقريب يستدعي الرجوع إلى النص الأصلي."
        ),
        "interpretation_en": (
            "Recalculates a t-distribution p-value for the specified tail. "
            "A numerical match does not validate design, sampling, degrees "
            "of freedom or test assumptions; check the source report."
        ),
    }
    if reported_p is not None:
        reported = _decimal(reported_p, "reported_p")
        if not 0 <= reported <= 1 or reported != reported.quantize(quantizer):
            raise ValueError("reported_p must be a rounded numeric probability on the declared precision")
        output["reported_p"] = str(reported)
        output["rounded_report_consistent"] = bool(reported == rounded)
    return output
