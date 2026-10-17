"""Multivariate assumptions and independence diagnostics.

All asymptotic tests need independent samples under their designs. The
reported p-values are *not* claims that a model, distribution, or causal
interpretation has been proven. Optional statsmodels is imported on demand.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.linalg import helmert
from ._results import make_result


def _matrix(data, *, n_min=8, p_min=2):
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < n_min or x.shape[1] < p_min:
        raise ValueError(f"Expected n-by-p numeric data with n>={n_min} and p>={p_min}")
    if not np.isfinite(x).all():
        raise ValueError("Missing or infinite observations require explicit handling")
    if x.shape[0] <= x.shape[1] + 2:
        raise ValueError("Need substantially more observations than dimensions")
    if np.linalg.matrix_rank(np.cov(x, rowvar=False)) < x.shape[1]:
        raise ValueError("Multivariate covariance matrix must have full rank")
    return x


def mardia_multivariate_normality(data, *, alpha=0.05):
    """Mardia multivariate skewness/kurtosis (large-sample approximations).

    Mardia skewness uses ML covariance (divisor n), with chi-square(df =
    p(p+1)(p+2)/6); kurtosis is an asymptotic N(0,1) statistic.
    Test both components separately; no combined/fabricated p-value.
    Memory/time O(n²), so up to 2000 observations by design.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    x = _matrix(data, n_min=15)
    n, p = x.shape
    if n > 2000:
        raise ValueError("Mardia skewness requires O(n²) memory; subsample with a declared design")
    centered = x - x.mean(axis=0)
    sigma = centered.T @ centered / n
    if np.linalg.matrix_rank(sigma) < p:
        raise ValueError("Singular population-normalized covariance")
    inner = centered @ np.linalg.solve(sigma, centered.T)
    skew = float(np.sum(inner ** 3) / n ** 2)
    skew_chi = n * skew / 6
    skew_df = p * (p + 1) * (p + 2) // 6
    skew_p = float(stats.chi2.sf(skew_chi, skew_df))
    radial = np.diag(inner)
    kurt = float(np.mean(radial ** 2))
    expected = p * (p + 2)
    z = (kurt - expected) / np.sqrt(8 * p * (p + 2) / n)
    kurt_p = float(2 * stats.norm.sf(abs(z)))
    return {
        "test": "Mardia multivariate normality",
        "skewness": skew, "skewness_chi_square": float(skew_chi),
        "skewness_df": skew_df, "skewness_p_value": skew_p,
        "kurtosis": kurt, "kurtosis_z": float(z),
        "kurtosis_p_value": kurt_p, "alpha": alpha,
        "n_samples": n, "n_variables": p,
        "interpretation_ar": (
            "اختبار مارديا يعرض نتيجتين منفصلتين: الالتواء والتفلطح متعدد المتغيرات. "
            "لا تجمع القيمتين الاحتماليتين بلا تصحيح للمقارنات، وعدم الرفض لا يثبت الاعتدالية."
        ),
        "interpretation_en": (
            "Mardia reports separate asymptotic skewness and kurtosis tests; "
            "do not combine their p-values without multiple-testing control. "
            "Non-rejection does not establish multivariate normality."
        ),
    }


def box_m_covariance_equality(*groups, alpha=0.05):
    """Box's M chi-square approximation for equal covariance matrices.

    Strongly sensitive to non-normality. Groups must be independent,
    with n_i>p+1, and each sample covariance positive-definite.
    """
    if not 0 < alpha < 1 or len(groups) < 2:
        raise ValueError("Need >=2 groups and 0<alpha<1")
    matrices = [_matrix(g, n_min=8) for g in groups]
    p = matrices[0].shape[1]
    if any(g.shape[1] != p or len(g) <= p + 1 for g in matrices):
        raise ValueError("Groups must have equal dimension and n_i>p+1")
    n_i = np.array([len(g) for g in matrices], dtype=float)
    dfs = n_i - 1
    covs = [np.cov(g, rowvar=False, ddof=1) for g in matrices]
    pooled = sum(d*c for d, c in zip(dfs, covs)) / dfs.sum()
    logs = []
    for c in covs + [pooled]:
        sign, logdet = np.linalg.slogdet(c)
        if sign <= 0:
            raise ValueError("Covariances must be positive definite")
        logs.append(logdet)
    M = dfs.sum() * logs[-1] - sum(d*v for d, v in zip(dfs, logs[:-1]))
    g = len(groups)
    factor = ((2*p*p + 3*p - 1)/(6*(p+1)*(g-1))) * (
        sum(1/dfs) - 1/dfs.sum()
    )
    chi = float(max(0, (1-factor)*M))
    p_value = float(stats.chi2.sf(chi, (g-1)*p*(p+1)//2))
    return make_result(
        "Box M covariance homogeneity", chi, p_value, int(n_i.sum()),
        alpha, "مصفوفات التباين والتغاير للمجموعات المستقلة متساوية.",
        "تقريب كاي تربيع حساس لانتهاك الاعتدالية متعددة المتغيرات.",
        null_en="Independent groups share the same covariance matrix.",
        notes_en="Asymptotic chi-square, sensitive to multivariate nonnormality.",
    )


def mauchly_sphericity_test(repeated, *, alpha=0.05):
    """Mauchly repeated-measures sphericity from Helmert contrasts.

    Large-sample chi-square approximation without GG/HF corrections.
    Requires >=3 within-subject conditions and complete paired cases.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    x = _matrix(repeated, n_min=12, p_min=3)
    n, k = x.shape
    h = helmert(k, full=False)
    cov = np.cov(x @ h.T, rowvar=False, ddof=1)
    eig = np.linalg.eigvalsh(cov)
    if np.min(eig) <= 0:
        raise ValueError("Contrast covariance must be positive definite")
    d = k - 1
    log_w = float(np.sum(np.log(eig)) - d*np.log(np.mean(eig)))
    w = float(np.exp(log_w))
    correction = 1 - (2*d*d + d + 2) / (6*d*(n-1))
    if correction <= 0:
        raise ValueError("Sample too small for the Mauchly approximation")
    chi = max(0.0, -(n-1)*correction*log_w)
    df = d*(d+1)//2 - 1
    p_val = float(stats.chi2.sf(chi, df))
    return {
        "test": "Mauchly sphericity",
        "statistic": float(chi), "p_value": p_val, "W": min(1.0, w),
        "degrees_of_freedom": int(df), "n_samples": n, "alpha": alpha,
        "reject_null": bool(p_val < alpha),
        "interpretation_ar": (
            "فرضية الكروية: تساوي تباينات فروق الشروط المترابطة. "
            "عند رفضها يلزم تصحيح Greenhouse–Geisser أو طريقة ملائمة أخرى."
        ),
        "interpretation_en": (
            "Null: equal variances of within-person condition differences. "
            "If rejected, consider Greenhouse–Geisser correction or an appropriate model."
        ),
    }


def breusch_godfrey_test(fitted_ols, *, nlags=4, alpha=0.05):
    """Breusch–Godfrey LM serial-correlation test on a fitted regression."""
    if not 0 < alpha < 1 or type(nlags) is not int or nlags < 1:
        raise ValueError("alpha in (0,1), positive integer nlags required")
    try:
        from statsmodels.stats.diagnostic import acorr_breusch_godfrey
    except ImportError as exc:
        raise ImportError('Install khaled-dsml[econometrics] for Breusch-Godfrey') from exc
    result = acorr_breusch_godfrey(fitted_ols, nlags=nlags)
    stat, p = float(result[0]), float(result[1])
    return make_result(
        "Breusch-Godfrey residual autocorrelation", stat, p,
        int(fitted_ols.nobs), alpha,
        "لا يوجد ارتباط ذاتي للبواقي حتى رتبة الإبطاء المحددة.",
        f"اختبار LM على بواقي نموذج منحدر؛ عدد الإبطاءات {nlags}.",
        null_en=f"No serial correlation of the regression errors through lag {nlags}.",
        notes_en="LM test on fitted model residuals; model assumptions matter.",
    )


def levene_mean_test(*groups, alpha=0.05):
    """Mean-centered Levene, distinct from median-centered Brown–Forsythe."""
    if len(groups) < 2 or not 0 < alpha < 1:
        raise ValueError("Need 2+ groups and 0<alpha<1")
    samples = [np.asarray(g, dtype=float).reshape(-1) for g in groups]
    if any(len(g) < 3 or not np.isfinite(g).all() for g in samples):
        raise ValueError("Each group requires >=3 finite values")
    r = stats.levene(*samples, center="mean")
    return make_result(
        "Levene mean-centered", r.statistic, r.pvalue,
        sum(map(len, samples)), alpha,
        "تباينات المجموعات متساوية (ليفين المرتكز على المتوسط).",
        "ليفين المتمركز حول المتوسط أكثر تأثرًا بالشواذ من Brown–Forsythe.",
        null_en="Independent groups have equal population variances.",
        notes_en="Mean-centered Levene is more sensitive to outliers than median-centered Brown–Forsythe.",
    )
