"""Introductory inferential statistics. Each test explicitly reports H0."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from ._results import TestResult, make_result


def _sample(x, minimum=2):
    s = pd.to_numeric(pd.Series(x), errors="raise").dropna().to_numpy(dtype=float)
    if s.size < minimum or not np.all(np.isfinite(s)):
        raise ValueError(f"Need at least {minimum} finite numeric observations")
    return s


def shapiro_test(x, alpha=0.05) -> TestResult:
    a = _sample(x, 3)
    if np.ptp(a) == 0:
        raise ValueError("Shapiro-Wilk is undefined for constant data")
    result = stats.shapiro(a)
    note = "قد تكون p-value غير دقيقة للعينات أكبر من 5000." if len(a) > 5000 else ""
    return make_result("Shapiro-Wilk", result.statistic, result.pvalue, len(a), alpha,
                       "تتبع البيانات توزيعًا طبيعيًا.", note)


def jarque_bera_test(x, alpha=0.05) -> TestResult:
    a = _sample(x, 3)
    if np.ptp(a) == 0:
        raise ValueError("Jarque-Bera is undefined for constant data")
    r = stats.jarque_bera(a)
    return make_result("Jarque-Bera", r.statistic, r.pvalue, len(a), alpha,
                       "تتفق الالتواء والتفرطح مع التوزيع الطبيعي.")


def normality_test(x, method="shapiro", alpha=0.05) -> TestResult:
    """method='shapiro' or 'jarque_bera'."""
    if method == "shapiro":
        return shapiro_test(x, alpha)
    if method == "jarque_bera":
        return jarque_bera_test(x, alpha)
    raise ValueError("method must be 'shapiro' or 'jarque_bera'")


def independent_t_test(a, b, alpha=0.05, equal_var=False) -> TestResult:
    x, y = _sample(a), _sample(b)
    r = stats.ttest_ind(x, y, equal_var=equal_var)
    name = "Student t-test" if equal_var else "Welch t-test"
    return make_result(name, r.statistic, r.pvalue, len(x)+len(y), alpha,
                       "متوسطا المجموعتين متساويان.")


def paired_t_test(a, b, alpha=0.05) -> TestResult:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Paired samples must have the same shape")
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 2:
        raise ValueError("Need at least two valid pairs")
    r = stats.ttest_rel(x, y)
    return make_result("Paired t-test", r.statistic, r.pvalue, len(x), alpha,
                       "متوسط الفروق بين القياسين يساوي صفرًا.")


def mann_whitney_test(a, b, alpha=0.05) -> TestResult:
    x, y = _sample(a), _sample(b)
    r = stats.mannwhitneyu(x, y, alternative="two-sided")
    return make_result("Mann-Whitney U", r.statistic, r.pvalue, len(x)+len(y), alpha,
                       "توزيعا المجموعتين متماثلان.",
                       "لا يُفسَّر تلقائيًا على أنه اختبار تساوي الوسيطين.")


def wilcoxon_test(a, b, alpha=0.05) -> TestResult:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Paired samples must have the same shape")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 2 or np.allclose(x-y, 0):
        raise ValueError("Need at least two pairs and a nonzero difference")
    r = stats.wilcoxon(x, y)
    return make_result("Wilcoxon signed-rank", r.statistic, r.pvalue, len(x), alpha,
                       "توزيع الفروق متناظر حول الصفر.")


def anova_test(*groups, alpha=0.05) -> TestResult:
    if len(groups) < 2:
        raise ValueError("Provide at least two groups")
    arrays = [_sample(group) for group in groups]
    r = stats.f_oneway(*arrays)
    return make_result("One-way ANOVA", r.statistic, r.pvalue,
                       sum(map(len, arrays)), alpha, "متوسطات كل المجموعات متساوية.")


def kruskal_test(*groups, alpha=0.05) -> TestResult:
    if len(groups) < 2:
        raise ValueError("Provide at least two groups")
    arrays = [_sample(group) for group in groups]
    r = stats.kruskal(*arrays)
    return make_result("Kruskal-Wallis", r.statistic, r.pvalue,
                       sum(map(len, arrays)), alpha, "توزيعات المجموعات متماثلة.")


def levene_test(*groups, alpha=0.05) -> TestResult:
    if len(groups) < 2:
        raise ValueError("Provide at least two groups")
    arrays = [_sample(group) for group in groups]
    r = stats.levene(*arrays, center="median")
    return make_result("Brown-Forsythe (Levene)", r.statistic, r.pvalue,
                       sum(map(len, arrays)), alpha, "تباينات المجموعات متساوية.")


def chi_square_test(contingency_table, alpha=0.05) -> TestResult:
    table = np.asarray(contingency_table, dtype=float)
    if table.ndim != 2 or min(table.shape) < 2 or not np.all(np.isfinite(table)):
        raise ValueError("Expected a finite 2D contingency table >= 2x2")
    if np.any(table < 0) or np.any(table.sum(axis=0) == 0) or np.any(table.sum(axis=1) == 0):
        raise ValueError("Table needs nonnegative counts and nonzero margins")
    r = stats.chi2_contingency(table, correction=False)
    notes = "راجع التكرارات المتوقعة؛ قد يلزم اختبار دقيق عند ندرتها."
    return make_result("Pearson Chi-square independence", r.statistic, r.pvalue,
                       int(table.sum()), alpha, "المتغيران الفئويان مستقلان.", notes)


def pearson_test(a, b, alpha=0.05) -> TestResult:
    x, y = _paired(a, b)
    r = stats.pearsonr(x, y)
    return make_result("Pearson correlation", r.statistic, r.pvalue,
                       len(x), alpha, "الارتباط الخطي السكاني يساوي صفرًا.")


def spearman_test(a, b, alpha=0.05) -> TestResult:
    x, y = _paired(a, b)
    r = stats.spearmanr(x, y)
    return make_result("Spearman correlation", r.statistic, r.pvalue,
                       len(x), alpha, "الارتباط الرتبي السكاني يساوي صفرًا.")


def _paired(a, b):
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Need aligned one-dimensional samples")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        raise ValueError("Need at least 3 variable paired observations")
    return x, y


def ks_two_sample_test(a, b, alpha=0.05) -> TestResult:
    x, y = _sample(a), _sample(b)
    r = stats.ks_2samp(x, y)
    return make_result("Two-sample Kolmogorov-Smirnov", r.statistic, r.pvalue,
                       len(x)+len(y), alpha, "المجموعتان تتبعان التوزيع نفسه.")


def descriptive_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for numeric columns only."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    return data.describe(include="number").T
