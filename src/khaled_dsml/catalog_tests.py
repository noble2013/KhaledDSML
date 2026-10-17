"""Additional catalogue-backed tests, statistical corrections and effect sizes.

Uses SciPy for hypothesis testing and statsmodels optionally for p-value
adjustments. All returned `TestResult` values include Arabic/English text.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._results import TestResult, make_result
from .statistics import _sample


def one_sample_t_test(values, mu=0.0, *, alpha=0.05) -> TestResult:
    data = _sample(values, minimum=2)
    if np.std(data, ddof=1) == 0:
        raise ValueError("One-sample t-test requires nonzero sample variance")
    result = stats.ttest_1samp(data, popmean=mu)
    return make_result("One-sample t-test", result.statistic, result.pvalue,
                       len(data), alpha, f"المتوسط السكاني يساوي {mu:g}.",
                       null_en=f"The population mean equals {mu:g}.")


def dagostino_normality_test(values, *, alpha=0.05) -> TestResult:
    data = _sample(values, minimum=8)
    if np.ptp(data) == 0:
        raise ValueError("Normality test is undefined for constant observations")
    result = stats.normaltest(data)
    return make_result("D'Agostino K-squared", result.statistic, result.pvalue,
                       len(data), alpha, "تتبع البيانات توزيعًا طبيعيًا.")


def bartlett_test(*groups, alpha=0.05) -> TestResult:
    if len(groups) < 2:
        raise ValueError("Need at least two groups")
    arrays = [_sample(x, minimum=2) for x in groups]
    if any(np.var(x, ddof=1) == 0 for x in arrays):
        raise ValueError("Bartlett test requires nonzero within-group variance")
    result = stats.bartlett(*arrays)
    return make_result("Bartlett", result.statistic, result.pvalue,
                       sum(map(len, arrays)), alpha, "تباينات المجموعات متساوية.",
                       notes="اختبار Bartlett حساس لانتهاك شرط الاعتدالية.",
                       notes_en="Bartlett's test is sensitive to non-normality.")


def fligner_test(*groups, alpha=0.05) -> TestResult:
    if len(groups) < 2:
        raise ValueError("Need at least two groups")
    arrays = [_sample(x, minimum=2) for x in groups]
    r = stats.fligner(*arrays)
    return make_result("Fligner-Killeen", r.statistic, r.pvalue,
                       sum(map(len, arrays)), alpha, "تباينات المجموعات متساوية.")


def kendall_test(a, b, *, alpha=0.05) -> TestResult:
    from .statistics import _paired
    x, y = _paired(a, b)
    r = stats.kendalltau(x, y)
    return make_result("Kendall tau", r.statistic, r.pvalue, len(x), alpha,
                       "معامل الارتباط الرتبي السكاني يساوي صفرًا.")


def fisher_exact_test(table, *, alpha=0.05) -> TestResult:
    data = np.asarray(table, dtype=float)
    if data.shape != (2, 2) or not np.all(np.isfinite(data)) or np.any(data < 0):
        raise ValueError("Fisher exact test requires a nonnegative 2x2 count table")
    if np.any(data != np.floor(data)):
        raise ValueError("Fisher exact counts must be integers")
    if data.sum() == 0 or np.any(data.sum(axis=0) == 0) or np.any(data.sum(axis=1) == 0):
        raise ValueError("The contingency table must have nonzero row and column margins")
    r = stats.fisher_exact(data.astype(int), alternative="two-sided")
    return make_result("Fisher exact", r.statistic, r.pvalue,
                       int(data.sum()), alpha, "المتغيران الفئويان مستقلان.")


def chi_square_gof_test(observed, expected=None, *, alpha=0.05) -> TestResult:
    o = np.asarray(observed, dtype=float)
    e = None if expected is None else np.asarray(expected, dtype=float)
    if o.ndim != 1 or len(o) < 2 or not np.all(np.isfinite(o)) or np.any(o < 0) or o.sum() <= 0:
        raise ValueError("observed must be nonnegative finite category counts")
    if e is not None and (e.shape != o.shape or np.any(e <= 0) or
                          not np.all(np.isfinite(e)) or not np.isclose(e.sum(), o.sum())):
        raise ValueError("expected counts must be positive and total the observed counts")
    r = stats.chisquare(f_obs=o, f_exp=e)
    return make_result("Chi-square goodness-of-fit", r.statistic, r.pvalue,
                       int(o.sum()), alpha, "التكرارات المرصودة تتوافق مع التكرارات المتوقعة.",
                       notes="راجع شرط التكرارات المتوقعة المناسبة للتقريب.",
                       notes_en="Check expected cell counts before relying on the chi-square approximation.")


def friedman_test(*groups, alpha=0.05) -> TestResult:
    if len(groups) < 3:
        raise ValueError("Friedman requires three or more repeated measures")
    a = [np.asarray(x, dtype=float) for x in groups]
    if any(v.ndim != 1 or len(v) != len(a[0]) for v in a):
        raise ValueError("All paired groups must be one-dimensional and of equal length")
    valid = np.logical_and.reduce([np.isfinite(x) for x in a])
    samples = [x[valid] for x in a]
    if len(samples[0]) < 2:
        raise ValueError("Need at least two complete subjects")
    if np.all(np.array(samples) == samples[0][0]):
        raise ValueError("Test undefined for constant identical data")
    r = stats.friedmanchisquare(*samples)
    return make_result("Friedman", r.statistic, r.pvalue,
                       len(samples[0]), alpha,
                       "لا تختلف توزيعات العلاجات المتكررة في موقعها ضمن فرضيات الاختبار.")


def sign_test(before, after, *, alpha=0.05) -> TestResult:
    x, y = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Sign test requires aligned paired observations")
    diffs = (y - x)
    diffs = diffs[np.isfinite(diffs)]
    changed = diffs[diffs != 0]
    if not len(changed):
        raise ValueError("No nonzero differences to test")
    r = stats.binomtest(int((changed > 0).sum()), n=len(changed), p=.5)
    return make_result("Sign test", (changed > 0).sum(), r.pvalue, len(changed),
                       alpha, "الإشارة الموجبة والسالبة متساويتان في الاحتمال.",
                       null_en="Positive and negative paired differences are equally likely.")


def adjust_pvalues(p_values, *, method="holm", alpha=0.05) -> dict:
    """Familywise or FDR-adjusted p-values (statsmodels optional).

    Allowed: bonferroni, holm, sidak, fdr_bh, fdr_by.
    """
    options = {"bonferroni", "holm", "sidak", "fdr_bh", "fdr_by"}
    if method not in options:
        raise ValueError(f"method must be one of {sorted(options)}")
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or len(p) == 0 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be a nonempty one-dimensional array in [0,1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError as exc:
        raise ImportError('Install khaled-dsml[econometrics] for multiple testing adjustments') from exc
    reject, corrected, _, _ = multipletests(p, method=method, alpha=alpha)
    return {"method": method, "adjusted_p_values": corrected.tolist(),
            "reject": reject.astype(bool).tolist(), "alpha": alpha}


def cohen_d(group_a, group_b, *, hedges=False) -> float:
    """Independent-group Cohen d with pooled SD; optional Hedges small-sample g."""
    a, b = _sample(group_a, minimum=2), _sample(group_b, minimum=2)
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1))
                     / (n1 + n2 - 2))
    if pooled == 0:
        raise ValueError("Pooled standard deviation is zero")
    d = float((a.mean() - b.mean()) / pooled)
    if hedges:
        correction = 1 - 3 / (4 * (n1 + n2) - 9)
        d *= correction
    return d


def cliffs_delta(group_a, group_b) -> float:
    """Nonparametric effect size, P(X>Y)-P(X<Y), in [-1,1]."""
    a, b = _sample(group_a), _sample(group_b)
    return float((np.count_nonzero(a[:, None] > b[None, :]) -
                  np.count_nonzero(a[:, None] < b[None, :])) / (len(a) * len(b)))


def bootstrap_mean_ci(values, *, confidence_level=.95, method="BCa",
                      n_resamples=5000, seed=42) -> dict:
    """Confidence interval for a mean using SciPy bootstrap (BCa/percentile/basic)."""
    x = _sample(values, minimum=3)
    if method not in ("BCa", "percentile", "basic") or not 0 < confidence_level < 1:
        raise ValueError("Invalid bootstrap method or confidence level")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    if np.ptp(x) == 0:
        raise ValueError("Bootstrap is undefined for constant observations in this helper")
    r = stats.bootstrap((x,), np.mean, method=method, confidence_level=confidence_level,
                        n_resamples=int(n_resamples), rng=np.random.default_rng(seed))
    return {"estimate": float(x.mean()),
            "lower": float(r.confidence_interval.low),
            "upper": float(r.confidence_interval.high),
            "method": method, "confidence_level": confidence_level}


def permutation_mean_test(a, b, *, alpha=.05, n_resamples=5000, seed=42) -> TestResult:
    """Two-sided permutation test for independent-sample mean differences."""
    x, y = _sample(a), _sample(b)
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    statistic = lambda a, b: np.mean(a) - np.mean(b)
    r = stats.permutation_test((x, y), statistic, vectorized=False,
                               alternative="two-sided", n_resamples=int(n_resamples),
                               rng=np.random.default_rng(seed))
    return make_result("Permutation mean difference", r.statistic, r.pvalue,
                       len(x) + len(y), alpha, "العينتان قابلتان للتبادل تحت الفرضية الصفرية.",
                       null_en="The group labels are exchangeable under the null hypothesis.")
