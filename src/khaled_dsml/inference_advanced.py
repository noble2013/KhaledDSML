"""Additional categorical inference, equivalence, and proportion intervals.

Only declared statistical assumptions justify asymptotic p-values. Categorical
tables use counts, not percentages. Multiple comparisons require correction.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from ._results import make_result
from .inference_extended import wilson_proportion_interval


def _count_vector(successes, trials):
    x, n = np.asarray(successes, dtype=float), np.asarray(trials, dtype=float)
    if (x.ndim != 1 or n.ndim != 1 or len(x) != len(n) or len(x) < 3
        or not np.isfinite(x).all() or not np.isfinite(n).all()
        or (n <= 0).any() or (x < 0).any() or (x > n).any()
        or (n != np.floor(n)).any() or (x != np.floor(x)).any()):
        raise ValueError("Need three or more ordered groups with integer successes <= positive integer totals")
    return x, n


def cochran_armitage_trend_test(successes, trials, *, scores=None, alpha=.05):
    """Two-sided Cochran-Armitage score test for an ordered binary proportion."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    x, n = _count_vector(successes, trials)
    k = len(x)
    w = np.arange(k, dtype=float) if scores is None else np.asarray(scores, dtype=float)
    if w.shape != (k,) or not np.isfinite(w).all() or np.ptp(w) <= 0:
        raise ValueError("scores must be distinct in at least two finite ordered levels")
    overall = x.sum()/n.sum()
    if overall <= 0 or overall >= 1:
        raise ValueError("At least one event and one nonevent required")
    if min(overall*n.min(), (1-overall)*n.min()) < 5:
        raise ValueError("Expected successes and failures must each be >=5 per group")
    numerator = float(np.dot(w, x - n*overall))
    variance = overall*(1-overall)*(
        float(np.dot(n,w*w)) - float(np.dot(n,w))**2/float(n.sum())
    )
    if variance <= 0:
        raise ValueError("Trend variance must be positive")
    z = numerator/np.sqrt(variance)
    return make_result(
        "Cochran-Armitage ordered proportion trend", z,
        float(2*stats.norm.sf(abs(z))), int(n.sum()), alpha,
        "لا يوجد اتجاه خطي في نسبة الحدث عبر المستويات المرتبة.",
        "أوزان المستويات واتجاهها ينبغي تحديدهما قبل تحليل البيانات.",
        null_en="There is no linear trend in event probability across ordered groups.",
        notes_en="Group scores and order must be prespecified; independent binomial counts assumed.",
    )


def g_test_independence(table, *, alpha=.05):
    """Log-likelihood ratio (G²) chi-square contingency independence test."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    t = np.asarray(table, dtype=float)
    if (t.ndim != 2 or min(t.shape) < 2 or not np.isfinite(t).all()
        or (t < 0).any() or (t != np.floor(t)).any()
        or (t.sum(axis=0) <= 0).any() or (t.sum(axis=1) <= 0).any()):
        raise ValueError("Need 2D integer frequency table with all marginal totals positive")
    res = stats.chi2_contingency(t, correction=False, lambda_="log-likelihood")
    if np.any(res.expected_freq < 5):
        raise ValueError("At least five expected observations per cell required for this asymptotic implementation")
    return make_result(
        "G-test categorical independence", res.statistic, res.pvalue, int(t.sum()), alpha,
        "متغيرا الجدول مستقلان في المجتمع.",
        f"درجات الحرية={res.dof}؛ اختبار تقاربي على تكرارات لا نسب مئوية.",
        null_en="The row and column categorical variables are independent.",
        notes_en=f"Likelihood-ratio G-test, df={res.dof}; input must be cell counts.",
    )


def cochran_mantel_haenszel_test(tables, *, alpha=.05):
    """CMH stratified 2x2 association conditional on prespecified strata."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    stack = np.asarray(tables, dtype=float)
    if stack.ndim != 3 or stack.shape[1:] != (2,2) or stack.shape[0] < 2:
        raise ValueError("Expected k-by-2-by-2 contingency tables for k>=2 strata")
    if (not np.isfinite(stack).all() or (stack < 0).any()
        or (stack != np.floor(stack)).any()
        or np.any(stack.sum(axis=(1,2)) < 4)):
        raise ValueError("Expected integer positive stratified cell counts")
    try:
        from statsmodels.stats.contingency_tables import StratifiedTable
    except ImportError as exc:
        raise ImportError('Install khaled-dsml[econometrics] for CMH') from exc
    strata = [stack[i] for i in range(len(stack))]
    result = StratifiedTable(strata).test_null_odds(correction=False)
    if not np.isfinite([result.statistic,result.pvalue]).all():
        raise ValueError("Stratified variance undefined for these tables")
    return make_result(
        "Cochran-Mantel-Haenszel stratified association",
        result.statistic, result.pvalue, int(stack.sum()), alpha,
        "لا يوجد ارتباط بين التعرض والنتيجة بعد ضبط طبقات الدراسة.",
        "تحتاج دراسة تجانس الأرجحية عبر الطبقات قبل تفسير نسبة أرجحية مشتركة.",
        null_en="There is no conditional exposure-outcome association across strata.",
        notes_en="Assess common-odds homogeneity and sampling design; statistical association is not causality.",
    )


def newcombe_two_proportions_interval(
    successes_a, trials_a, successes_b, trials_b, *, confidence_level=.95,
):
    """Newcombe hybrid-score CI for the difference between independent rates."""
    a = wilson_proportion_interval(successes_a, trials_a, confidence_level=confidence_level)
    b = wilson_proportion_interval(successes_b, trials_b, confidence_level=confidence_level)
    p1, p2 = a["proportion"], b["proportion"]
    diff = p1 - p2
    lower = diff - np.sqrt((p1-a["lower"])**2+(b["upper"]-p2)**2)
    upper = diff + np.sqrt((a["upper"]-p1)**2+(p2-b["lower"])**2)
    return {
        "difference": float(diff), "lower": float(max(-1, lower)),
        "upper": float(min(1,upper)), "confidence_level":confidence_level,
        "n_samples":int(trials_a+trials_b),
        "interpretation_ar": "فاصل ثقة Newcombe لفرق نسبتين من عينتين مستقلتين؛ لا يعوض تصميم المعاينة.",
        "interpretation_en": "Newcombe hybrid-score interval for independent proportions; sampling design matters.",
    }


def welch_tost_equivalence(group_a, group_b, *, margin, alpha=.05):
    """Two one-sided Welch t tests for a prespecified mean-difference margin.

    H0: difference <= -margin OR >= margin. Rejection means equivalence
    within (-margin,+margin), not numerical equality.
    """
    if not 0 < alpha < 1 or not np.isfinite(margin) or margin <= 0:
        raise ValueError("Require alpha in (0,1) and a positive prespecified equivalence margin")
    x, y = np.asarray(group_a, dtype=float), np.asarray(group_b, dtype=float)
    if (x.ndim != 1 or y.ndim != 1 or min(len(x),len(y)) < 3
        or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Need >=3 finite observations in each independent group")
    variance = np.var(x,ddof=1)/len(x)+np.var(y,ddof=1)/len(y)
    if variance <= 0:
        raise ValueError("Both groups together require positive uncertainty")
    a = np.var(x,ddof=1)/len(x)
    b = np.var(y,ddof=1)/len(y)
    df = (a+b)**2/(a*a/(len(x)-1)+b*b/(len(y)-1))
    se = np.sqrt(variance)
    difference = float(np.mean(x)-np.mean(y))
    lower_t = (difference+margin)/se
    upper_t = (difference-margin)/se
    p_low = float(stats.t.sf(lower_t,df))
    p_high = float(stats.t.cdf(upper_t,df))
    p_tost = max(p_low,p_high)
    equivalent = p_tost < alpha
    return {
        "test":"Welch TOST equivalence",
        "statistic":float(min(lower_t,-upper_t)),
        "p_value":p_tost,"alpha":alpha,"reject_null":equivalent,
        "difference":difference,"equivalence_margin":float(margin),
        "p_lower":p_low,"p_upper":p_high,"degrees_of_freedom":float(df),
        "n_samples":len(x)+len(y),
        "interpretation_ar":(
            "توجد أدلة على التكافؤ ضمن الهامش المحدد مسبقًا، وليس التطابق التام."
            if equivalent else
            "لا تكفي الأدلة لإثبات التكافؤ ضمن الهامش المحدد مسبقًا."
        ),
        "interpretation_en":(
            "Equivalence is supported within the prespecified margin, not exact equality."
            if equivalent else
            "Insufficient evidence for equivalence within the prespecified margin."
        ),
    }


def two_group_power_analysis(effect_size, *, n_per_group=None, power=None,
                             alpha=.05, alternative="two-sided"):
    """Independent-groups t-test power or required per-group size, via statsmodels.

    Supply exactly one of n_per_group or power; equal-sized groups assumed.
    The effect size is an *assumed* Cohen's d, never estimated from the input.
    """
    if not 0 < alpha < 1 or not np.isfinite(effect_size) or effect_size == 0:
        raise ValueError("alpha in (0,1) and nonzero finite assumed Cohen d required")
    if (n_per_group is None) == (power is None):
        raise ValueError("Specify exactly one of n_per_group or target power")
    if alternative not in ("two-sided","larger","smaller"):
        raise ValueError("Invalid alternative")
    try:
        from statsmodels.stats.power import TTestIndPower
    except ImportError as exc:
        raise ImportError('Install khaled-dsml[econometrics] for power analysis') from exc
    obj = TTestIndPower()
    d=float(effect_size)
    if power is not None:
        if not .05 < power < 1:
            raise ValueError("Target power must lie in (.05,1)")
        n=float(obj.solve_power(effect_size=d, power=power,alpha=alpha,
                                alternative=alternative))
        if not np.isfinite(n):
            raise ValueError("Power solver failed to converge")
        return {
            "n_per_group":int(np.ceil(n)),"target_power":float(power),
            "assumed_effect_size":d,
            "interpretation_ar":"تقدير مسبق يعتمد على حجم أثر مفترض وعينتين مستقلتين متساويتين.",
            "interpretation_en":"A priori sample-size estimate using assumed effect size and equal independent groups.",
        }
    if type(n_per_group) is not int or n_per_group < 3:
        raise ValueError("n_per_group must be integer >=3")
    achieved=float(obj.power(effect_size=d, nobs1=n_per_group,
                            alpha=alpha, alternative=alternative))
    return {
        "n_per_group":n_per_group,"power":achieved,"assumed_effect_size":d,
        "interpretation_ar":"قوة إحصائية مفترضة مبنية على أثر محدد مسبقًا؛ ليست قوة لاحقة مأخوذة من p-value.",
        "interpretation_en":"Prospective power at an assumed effect size; not post-hoc observed power from a p-value.",
    }
