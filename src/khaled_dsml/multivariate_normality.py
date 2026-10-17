"""Henze–Zirkler multivariate-normality diagnostic with Monte Carlo calibration.

The statistic is affine invariant when the maximum-likelihood sample covariance
is positive definite. Null calibration simulates independent Gaussian samples
and re-estimates their covariance *on every replicate*. No approximate lognormal
p-values or claims of proven normality are made.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist


def _henze_zirkler_statistic(data, beta):
    """Direct Henze–Zirkler Gaussian-kernel formula using ML whitening."""
    n, p = data.shape
    centered = data - data.mean(axis=0)
    covariance = centered.T @ centered / n
    try:
        whitened = np.linalg.solve(np.linalg.cholesky(covariance), centered.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("Multivariate covariance must be positive definite") from exc

    squared_radii = np.einsum("ij,ij->i", whitened, whitened)
    # pdist excludes the n zero diagonal distances; restore them explicitly.
    sum_kernel = n + 2.0 * np.exp(-0.5 * beta**2 * pdist(whitened, "sqeuclidean")).sum()
    term1 = sum_kernel / n**2
    term2 = (2.0 / (1.0 + beta**2)**(p / 2.0)) * np.mean(
        np.exp(-(beta**2 / (2.0 * (1.0 + beta**2))) * squared_radii)
    )
    term3 = (1.0 + 2.0 * beta**2)**(-p / 2.0)
    return float(max(0.0, n * (term1 - term2 + term3)))


def henze_zirkler_normality_test(
    data, *, alpha=0.05, n_resamples=199, random_state=42,
):
    """Test multivariate normality using a parametric Monte Carlo null.

    Null: independent complete observations come from a nonsingular multivariate
    Gaussian distribution. Beta is the HZ smoothing parameter; statistics are
    re-evaluated after refitting mean/covariance for each Gaussian null sample.
    The conservative finite Monte Carlo p-value is (1 + #null >= observed)/(B+1).
    Complexity is O((B+1)*n^2); n>700 requires deliberate subsampling by users.

    Rejection suggests departure from joint normality; non-rejection does not
    demonstrate Gaussianity, nor validate independence or downstream tests.
    """
    if isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if type(n_resamples) is not int or not 99 <= n_resamples <= 10000:
        raise ValueError("n_resamples must be an integer from 99 to 10000")
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError("Expected a numeric n-by-p matrix")
    n, p = x.shape
    if p < 2 or n < max(20, p + 5) or n > 700:
        raise ValueError("Expected p>=2 and max(20,p+5)<=n<=700")
    if not np.isfinite(x).all():
        raise ValueError("Data must contain only finite values; handle missingness explicitly")
    if not np.all(np.isfinite(np.var(x, axis=0))):
        raise ValueError("Invalid covariance")
    beta = (1 / np.sqrt(2)) * ((2 * p + 1) / 4)**(1 / (p + 4)) * n**(1 / (p + 4))
    observed = _henze_zirkler_statistic(x, beta)
    rng = np.random.default_rng(random_state)
    exceedances = 0
    for _ in range(n_resamples):
        null_sample = rng.standard_normal((n, p))
        reference = _henze_zirkler_statistic(null_sample, beta)
        exceedances += reference >= observed
    p_value = (1 + exceedances) / (n_resamples + 1)
    return {
        "test": "Henze-Zirkler multivariate normality (Monte Carlo)",
        "statistic": observed, "p_value": float(p_value),
        "beta": float(beta), "n_resamples": n_resamples,
        "monte_carlo_exceedances": int(exceedances),
        "n_samples": n, "n_variables": p, "alpha": float(alpha),
        "reject_null": bool(p_value < alpha),
        "null_hypothesis_ar": "المشاهدات مستقلة وتتبع توزيعًا طبيعيًا متعدد المتغيرات.",
        "null_hypothesis_en": "Independent observations follow a multivariate normal distribution.",
        "interpretation_ar": (
            "اختبار Henze–Zirkler بمعايرة مونت كارلو ومصفوفة تغاير معاد تقديرها "
            "لكل محاكاة. الرفض دليل ضد الاعتدالية المشتركة؛ عدم الرفض لا يثبتها. "
            "الدقة الاحتمالية محدودة بعدد المحاكاة، والاستقلالية شرط مستقل."
        ),
        "interpretation_en": (
            "Henze–Zirkler with Monte Carlo null calibration and covariance "
            "refitted for each replicate. Rejection suggests nonnormality; "
            "non-rejection does not establish normality. P-value resolution "
            "depends on the number of simulations; independence is required."
        ),
    }
