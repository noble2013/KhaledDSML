"""Extended inferential tests, proportions, effect sizes, and research safeguards.

Bilingual test results preserve KhaledDSML's conservative null-hypothesis
interpretation. No test here proves causality or the null hypothesis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._results import make_result


def _vector(values, *, min_n=2):
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size < min_n or not np.isfinite(x).all():
        raise ValueError(f"Need at least {min_n} finite observations; handle missing values explicitly")
    return x


def _alpha(alpha):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")


def welch_anova_test(*groups, alpha=0.05):
    """Welch unequal-variance one-way ANOVA; statsmodels is optional."""
    _alpha(alpha)
    if len(groups) < 2:
        raise ValueError("Need at least two independent groups")
    samples = [_vector(x) for x in groups]
    if any(np.var(x, ddof=1) <= 0 for x in samples):
        raise ValueError("Each group needs a strictly positive sample variance")
    try:
        from statsmodels.stats.oneway import anova_oneway
    except ImportError as exc:
        raise ImportError('Install khaled-dsml[econometrics] for Welch ANOVA') from exc
    result = anova_oneway(samples, use_var="unequal", welch_correction=True)
    return make_result(
        "Welch one-way ANOVA", result.statistic, result.pvalue,
        sum(map(len, samples)), alpha, "متوسطات المجموعات السكانية متساوية.",
        "الاختبار يفترض استقلال المشاهدات؛ لا يتطلب تجانس التباين.",
        null_en="All population group means are equal.",
        notes_en="Independent groups are required; equal variances are not assumed.",
    )


def mood_median_test(*groups, alpha=0.05):
    """Mood median test, excluding observations tied at pooled median."""
    _alpha(alpha)
    if len(groups) < 2:
        raise ValueError("Need at least two groups")
    samples = [_vector(x) for x in groups]
    res = stats.median_test(*samples, ties="ignore")
    return make_result(
        "Mood median", res.statistic, res.pvalue, sum(map(len, samples)), alpha,
        "تقع وسيطات المجموعات عند الموقع نفسه.",
        "اختبار يعتمد على تقسيم القيم عند الوسيط العام وقد تكون قوته محدودة.",
        null_en="All group populations share the same median under the test assumptions.",
        notes_en="Observations tied at the pooled median are excluded.",
    )


def brunner_munzel_test(group_a, group_b, *, alpha=0.05):
    """Two-sided Brunner-Munzel nonparametric test."""
    _alpha(alpha)
    x, y = _vector(group_a, min_n=3), _vector(group_b, min_n=3)
    r = stats.brunnermunzel(x, y, alternative="two-sided", distribution="t")
    return make_result(
        "Brunner-Munzel", r.statistic, r.pvalue, len(x) + len(y), alpha,
        "احتمال تفوق مشاهدة من إحدى المجموعتين على الأخرى يساوي النصف.",
        "لا يحتاج تساوي شكل التوزيعين أو التباينين، لكنه يفترض استقلال العينتين.",
        null_en="The probability of stochastic superiority equals one half.",
        notes_en="Independent groups; not an equality-of-medians test in general.",
    )


def mcnemar_test(table, *, exact=None, correction=True, alpha=0.05):
    """McNemar on a paired 2x2 contingency table (off-diagonal discordances)."""
    _alpha(alpha)
    x = np.asarray(table, dtype=float)
    if x.shape != (2, 2) or not np.isfinite(x).all() or (x < 0).any() or (x != np.floor(x)).any():
        raise ValueError("Expected a 2x2 table of nonnegative integer paired counts")
    b, c = int(x[0, 1]), int(x[1, 0])
    discordant = b + c
    if discordant == 0:
        raise ValueError("No discordant pairs; the test has no evidence to evaluate")
    use_exact = discordant < 25 if exact is None else bool(exact)
    if use_exact:
        p_value = float(stats.binomtest(min(b, c), discordant, .5).pvalue)
        statistic = float(min(b, c))
        method = "exact binomial"
    else:
        statistic = ((abs(b-c) - (1 if correction else 0)) ** 2) / discordant
        p_value = float(stats.chi2.sf(statistic, 1))
        method = "chi-square approximation"
    return make_result(
        "McNemar paired", statistic, p_value, int(x.sum()), alpha,
        "احتمال الانتقال من الفئة الأولى إلى الثانية يساوي احتمال الانتقال العكسي.",
        f"اعتمادًا على {discordant} زوجًا مختلفًا؛ الطريقة: {method}.",
        null_en="Probabilities of the two directions of discordance are equal.",
        notes_en=f"Discordant pairs: {discordant}; method: {method}.",
    )


def cochran_q_test(data, *, alpha=0.05):
    """Cochran Q test for repeated binary observations across >=3 conditions."""
    _alpha(alpha)
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 3 or not np.isfinite(x).all():
        raise ValueError("Need a finite n-by-k matrix, n>=3, k>=3")
    if not np.isin(x, [0, 1]).all():
        raise ValueError("Cochran Q requires paired binary responses (0 or 1)")
    try:
        from statsmodels.stats.contingency_tables import cochrans_q
    except ImportError as exc:
        raise ImportError('Install khaled-dsml[econometrics] for Cochran Q') from exc
    r = cochrans_q(x)
    if not np.isfinite(r.statistic) or not np.isfinite(r.pvalue):
        raise ValueError("Cochran Q is undefined for these response patterns")
    return make_result(
        "Cochran Q", r.statistic, r.pvalue, x.shape[0], alpha,
        "نسبة الاستجابة الإيجابية متساوية عبر الشروط المترابطة.",
        f"شروط مرتبطة: {x.shape[1]}.", null_en="Success probabilities are equal across matched conditions.",
        notes_en=f"Matched conditions: {x.shape[1]}.",
    )


def _success_counts(successes, trials):
    if any(isinstance(x, bool) for x in (successes, trials)):
        raise ValueError("Counts must be integers")
    if int(successes) != successes or int(trials) != trials or not 0 <= successes <= trials or trials <= 0:
        raise ValueError("Need 0 <= successes <= trials, and positive integer trials")
    return int(successes), int(trials)


def one_proportion_z_test(successes, trials, *, null_proportion=.5, alpha=.05):
    """One-sample proportion z-test with the null-model standard error."""
    _alpha(alpha)
    x, n = _success_counts(successes, trials)
    if not 0 < null_proportion < 1:
        raise ValueError("null_proportion must be inside (0, 1)")
    if min(n*null_proportion, n*(1-null_proportion)) < 5:
        raise ValueError("Null-model expected successes and failures must each be at least 5")
    z = (x/n-null_proportion) / np.sqrt(null_proportion*(1-null_proportion)/n)
    p = float(2*stats.norm.sf(abs(z)))
    return make_result(
        "One-sample proportion z", z, p, n, alpha,
        f"النسبة السكانية تساوي {null_proportion:g}.",
        "اختبار تقريبي؛ لفترات الثقة في العينات الصغيرة استخدم Wilson.",
        null_en=f"The population proportion equals {null_proportion:g}.",
        notes_en="Normal approximation with null standard error; Wilson intervals are preferred for small samples.",
    )


def two_proportions_z_test(successes_a, trials_a, successes_b, trials_b, *, alpha=.05):
    """Two independent proportions with pooled standard error under H0."""
    _alpha(alpha)
    x, nx = _success_counts(successes_a, trials_a)
    y, ny = _success_counts(successes_b, trials_b)
    pooled = (x+y)/(nx+ny)
    if min(nx*pooled, nx*(1-pooled), ny*pooled, ny*(1-pooled)) < 5:
        raise ValueError("Need at least five expected successes and failures per group")
    se = np.sqrt(pooled*(1-pooled)*(1/nx+1/ny))
    z = (x/nx-y/ny)/se
    return make_result(
        "Two independent proportions z", z, 2*stats.norm.sf(abs(z)), nx+ny, alpha,
        "النسبتان السكانيتان متساويتان.",
        "يشترط استقلال العينتين وكفاية التكرارات المتوقعة.",
        null_en="The two independent population proportions are equal.",
        notes_en="Independent samples and sufficiently large expected cell counts required.",
    )


def wilson_proportion_interval(successes, trials, *, confidence_level=.95) -> dict:
    """Wilson-score confidence interval; finite in all nonempty binomial samples."""
    x, n = _success_counts(successes, trials)
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be inside (0,1)")
    z = stats.norm.ppf((1+confidence_level)/2)
    p = x/n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return {
        "proportion": p, "lower": float(max(0, center-half)),
        "upper": float(min(1, center+half)), "n_samples": n,
        "confidence_level": confidence_level,
        "interpretation_ar": "فاصل ثقة Wilson للنسبة السكانية مع افتراض معاينة ثنائية مستقلة.",
        "interpretation_en": "Wilson interval for a population proportion, assuming independent binomial sampling.",
    }


def point_biserial_test(binary, continuous, *, alpha=.05):
    """Point-biserial association for paired binary and quantitative observations."""
    _alpha(alpha)
    b, y = _vector(binary, min_n=4), _vector(continuous, min_n=4)
    if len(b) != len(y) or np.unique(b).size != 2 or np.std(y) == 0:
        raise ValueError("Need aligned paired vectors, exactly two binary levels and variable measurements")
    r = stats.pointbiserialr(b, y)
    return make_result(
        "Point-biserial correlation", r.statistic, r.pvalue, len(b), alpha,
        "الارتباط الخطي النقطي–ثنائي في المجتمع يساوي صفرًا.",
        "تفسير العلاقة لا يستلزم علاقة سببية.",
        null_en="The population point-biserial correlation is zero.",
        notes_en="Association does not establish causation.",
    )


def partial_correlation_test(a, b, controls, *, alpha=.05):
    """Partial Pearson correlation using OLS residualization with t(df=n-k-2)."""
    _alpha(alpha)
    x, y = _vector(a, min_n=5), _vector(b, min_n=5)
    z = np.asarray(controls, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    if z.ndim != 2 or z.shape[0] != len(x) or len(y) != len(x) or not np.isfinite(z).all():
        raise ValueError("controls must be a finite, aligned n-by-k numeric matrix")
    n, k = z.shape
    if k < 1 or n <= k+3:
        raise ValueError("Need at least one control and n > k+3")
    design = np.column_stack([np.ones(n), z])
    if np.linalg.matrix_rank(design) != k+1:
        raise ValueError("Collinear controls are not supported")
    xr = x-design@np.linalg.lstsq(design, x, rcond=None)[0]
    yr = y-design@np.linalg.lstsq(design, y, rcond=None)[0]
    if np.std(xr) == 0 or np.std(yr) == 0:
        raise ValueError("Zero residual variation after controlling for covariates")
    rho = float(np.corrcoef(xr, yr)[0, 1])
    df = n-k-2
    test = rho*np.sqrt(df/max(1e-16, 1-rho*rho))
    return make_result(
        "Partial Pearson correlation", test, float(2*stats.t.sf(abs(test), df)),
        n, alpha, "الارتباط الخطي الجزئي بعد ضبط المتغيرات المساعدة يساوي صفرًا.",
        f"الارتباط الجزئي r={rho:.4f}؛ درجات الحرية={df}.",
        null_en="Partial population linear correlation conditional on controls is zero.",
        notes_en=f"Partial r={rho:.4f}; residual degrees of freedom={df}.",
    )


def cramers_v(table, *, correction=True):
    """Bias-corrected Cramer's V for a two-way contingency table."""
    values = np.asarray(table, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all() or (values < 0).any() or values.sum() == 0:
        raise ValueError("Need a nonnegative two-way table with >=2 rows and columns")
    if np.any(values.sum(axis=0) == 0) or np.any(values.sum(axis=1) == 0):
        raise ValueError("Empty margins make chi-square inference undefined")
    n = values.sum()
    chi = stats.chi2_contingency(values, correction=False).statistic
    phi2 = chi/n
    rows, cols = values.shape
    if correction:
        if n <= 1:
            raise ValueError("Need at least two observations")
        phi2 = max(0, phi2-(cols-1)*(rows-1)/(n-1))
        r = rows-(rows-1)**2/(n-1)
        c = cols-(cols-1)**2/(n-1)
        denom = min(r-1,c-1)
    else:
        denom = min(rows-1,cols-1)
    return float(np.sqrt(phi2/denom)) if denom > 0 else 0.0


def phi_coefficient(table):
    """Signed phi association for a 2x2 table."""
    x = np.asarray(table, dtype=float)
    if x.shape != (2,2) or not np.isfinite(x).all() or (x<0).any():
        raise ValueError("Need a 2x2 nonnegative table")
    a,b,c,d = x.flat
    den = np.sqrt((a+b)*(c+d)*(a+c)*(b+d))
    if den == 0:
        raise ValueError("All row and column margins must be positive")
    return float((a*d-b*c)/den)


def fisher_z_correlation_test(corr_a, size_a, corr_b, size_b, *, alpha=.05):
    """Compare correlations from independent samples using Fisher's transform."""
    _alpha(alpha)
    if not -1 < corr_a < 1 or not -1 < corr_b < 1 or size_a <= 3 or size_b <= 3:
        raise ValueError("Need |r|<1 and two independent samples with n>3")
    se = np.sqrt(1/(size_a-3)+1/(size_b-3))
    z = (np.arctanh(corr_a)-np.arctanh(corr_b))/se
    return make_result(
        "Fisher z independent correlations", z,
        2*stats.norm.sf(abs(z)), int(size_a+size_b), alpha,
        "معاملا الارتباط السكانيان في العينتين المستقلتين متساويان.",
        "هذه الصيغة لا تصلح للارتباطات التابعة ضمن نفس العينة.",
        null_en="The two independent-sample population correlations are equal.",
        notes_en="Not suitable for dependent correlations in the same sample.",
    )


def glass_delta(treatment, control):
    """Standardized mean difference using control-group sample SD."""
    t, c = _vector(treatment), _vector(control)
    sd = np.std(c, ddof=1)
    if sd <= 0:
        raise ValueError("Control-group standard deviation must be positive")
    return float((np.mean(t)-np.mean(c))/sd)


def eta_squared(groups):
    """Ordinary one-way ANOVA eta-squared from independent numeric groups."""
    if len(groups) < 2:
        raise ValueError("Need >=2 groups")
    g = [_vector(a) for a in groups]
    pooled = np.concatenate(g)
    ss_total = np.sum((pooled-pooled.mean())**2)
    if ss_total <= 0:
        raise ValueError("Outcome has no variation")
    ss_between = sum(len(a)*(np.mean(a)-pooled.mean())**2 for a in g)
    return float(ss_between/ss_total)


def omega_squared(groups):
    """One-way ANOVA omega-squared, clipped at zero for interpretation."""
    if len(groups) < 2:
        raise ValueError("Need >=2 groups")
    g = [_vector(a) for a in groups]
    n, k = sum(map(len,g)), len(g)
    if n <= k:
        raise ValueError("Need residual degrees of freedom")
    pooled = np.concatenate(g)
    sst = np.sum((pooled-pooled.mean())**2)
    ssb = sum(len(a)*(a.mean()-pooled.mean())**2 for a in g)
    ssw = sst-ssb
    if sst <= 0:
        raise ValueError("No total variation")
    mse = ssw/(n-k)
    return float(max(0, (ssb-(k-1)*mse)/(sst+mse))) if mse >= 0 else 0.0


def cohen_w(observed, expected_probabilities):
    """Goodness-of-fit Cohen's w using strictly positive expected probabilities."""
    observed = _vector(observed, min_n=2)
    p = _vector(expected_probabilities, min_n=2)
    if observed.size != p.size or (observed < 0).any() or observed.sum() <= 0 or (p <= 0).any() or not np.isclose(p.sum(),1):
        raise ValueError("Need observed nonnegative counts and strictly positive expected probabilities summing to 1")
    empirical = observed/observed.sum()
    return float(np.sqrt(np.sum((empirical-p)**2/p)))


def contingency_risk_measures(table, *, confidence_level=.95):
    """Odds ratio and risk ratio for [event,non-event] treatment/control rows.

    Reject zero cells instead of silently applying an unannounced correction.
    """
    a = np.asarray(table,dtype=float)
    if a.shape != (2,2) or not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError("All four cells must be positive in a 2x2 event/non-event table")
    if not 0 < confidence_level < 1:
        raise ValueError("Invalid confidence_level")
    e1, n1, e0, n0 = a.flat
    OR = (e1*n0)/(n1*e0)
    RR = (e1/(e1+n1))/(e0/(e0+n0))
    z = stats.norm.ppf((1+confidence_level)/2)
    se_or = np.sqrt(np.sum(1/a))
    se_rr = np.sqrt(1/e1-1/(e1+n1)+1/e0-1/(e0+n0))
    return {
        "odds_ratio": float(OR), "relative_risk": float(RR),
        "or_ci": tuple(map(float,np.exp(np.log(OR)+np.array([-1,1])*z*se_or))),
        "rr_ci": tuple(map(float,np.exp(np.log(RR)+np.array([-1,1])*z*se_rr))),
        "interpretation_ar": "مقياس ارتباط بين التعرض والحدث؛ لا يثبت السببية.",
        "interpretation_en": "Exposure-outcome association measures; not evidence of causality.",
    }
