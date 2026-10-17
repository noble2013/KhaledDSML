"""Assumption diagnostics drawn from the KhaledDSML teaching catalogue.

Only explicitly defined tests return p-values; descriptive statistics do not.
Both Arabic and English descriptions are provided for interpretation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from ._results import make_result


def _numeric_sample(values, *, minimum=3):
    x = pd.to_numeric(pd.Series(values), errors="raise").dropna().to_numpy(dtype=float)
    if len(x) < minimum or not np.all(np.isfinite(x)):
        raise ValueError(f"Need at least {minimum} finite observed values")
    return x


def anderson_darling_normality(values, *, alpha=.05) -> dict:
    """Anderson-Darling normality test with tabulated critical value; no fake p-value."""
    x = _numeric_sample(values, minimum=8)
    if np.ptp(x) == 0:
        raise ValueError("Normality testing is not defined for a constant sample")
    r = stats.anderson(x, dist="norm")
    levels = np.asarray(r.significance_level, dtype=float) / 100
    indices = np.flatnonzero(np.isclose(levels, alpha, atol=1e-9, rtol=0))
    if len(indices) != 1:
        raise ValueError(f"alpha must match a reported significance level: {levels.tolist()}")
    critical = float(r.critical_values[int(indices[0])])
    rejected = bool(r.statistic > critical)
    return {
        "test": "Anderson-Darling normality",
        "statistic": float(r.statistic), "critical_value": critical,
        "alpha": alpha, "reject_null": rejected, "p_value": None,
        "interpretation_ar": (
            "ترفض فرضية الاعتدالية وفق القيمة الحرجة المدرجة."
            if rejected else "لا ترفض فرضية الاعتدالية وفق القيمة الحرجة المدرجة، وهذا لا يثبتها."
        ),
        "interpretation_en": (
            "Reject the normality null based on the tabulated critical value."
            if rejected else "Do not reject the normality null based on the tabulated critical value; normality is not proven."
        ),
    }


def lilliefors_normality(values, *, alpha=.05):
    """Lilliefors normality test (normal mean/variance estimated from data)."""
    x = _numeric_sample(values, minimum=5)
    if np.ptp(x) == 0:
        raise ValueError("Normality test is undefined for constant observations")
    try:
        from statsmodels.stats.diagnostic import lilliefors
    except ImportError as exc:
        raise ImportError('Install khaled-dsml[econometrics] for Lilliefors') from exc
    statistic, p_value = lilliefors(x, dist="norm", pvalmethod="table")
    return make_result(
        "Lilliefors", statistic, p_value, len(x), alpha,
        "البيانات تتبع توزيعًا طبيعيًا بمعلمات مقدرة من العينة.",
        null_en="Data follow a normal distribution whose parameters are estimated from the sample.",
    )


def runs_test(values, *, cutoff="median", alpha=.05):
    """Approximate two-sided Wald-Wolfowitz runs test for a binary split.

    A median cutoff drops ties. The normal approximation requires at least
    five observations in each of the two groups.
    """
    x = _numeric_sample(values, minimum=10)
    if cutoff == "median":
        pivot = float(np.median(x))
    elif isinstance(cutoff, (int, float)) and np.isfinite(cutoff):
        pivot = float(cutoff)
    else:
        raise ValueError("cutoff must be median or a finite numeric value")
    x = x[x != pivot]
    labels = x > pivot
    n1, n2 = int(labels.sum()), int((~labels).sum())
    if min(n1, n2) < 5:
        raise ValueError("Runs test normal approximation requires at least 5 observations in each group")
    runs = 1 + int(np.count_nonzero(labels[1:] != labels[:-1]))
    mean = 1 + 2 * n1 * n2 / (n1 + n2)
    variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) /
                ((n1 + n2) ** 2 * (n1 + n2 - 1)))
    z = float((runs - mean) / np.sqrt(variance))
    p_value = float(2 * stats.norm.sf(abs(z)))
    return make_result(
        "Wald-Wolfowitz runs", z, p_value, n1+n2, alpha,
        "ترتيب الإشارات أعلى وأسفل القطع عشوائي وفق نموذج الاستقلال.",
        "تقريب طبيعي؛ حُذفت حالات التساوي مع القطع.",
        null_en="The order of high and low observations is random under the independence model.",
        notes_en="Normal approximation; observations tied at the cutoff were excluded.",
    )


def durbin_watson_statistic(residuals) -> dict:
    """Compute DW in [0,4] without inventing a universal p-value."""
    e = _numeric_sample(residuals, minimum=3)
    denominator = float(np.sum(e ** 2))
    if denominator == 0:
        raise ValueError("Residuals are identically zero")
    dw = float(np.sum(np.diff(e) ** 2) / denominator)
    return {
        "statistic": dw, "n": len(e), "p_value": None,
        "interpretation_ar": (
            f"إحصائية دوربن–واتسون {dw:.3f}. اقترابها من 2 قد يتفق مع ضعف "
            "الارتباط الذاتي من الرتبة الأولى؛ يلزم تصميم الانحدار والقيم الحرجة للتفسير."
        ),
        "interpretation_en": (
            f"Durbin-Watson statistic: {dw:.3f}. Values near 2 may be consistent "
            "with weak first-order autocorrelation; model design and reference "
            "bounds are required for inference."
        ),
    }


def _validated_matrix(data):
    frame = data.select_dtypes(include="number") if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    frame = frame.dropna()
    x = frame.to_numpy(dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or x.shape[0] <= x.shape[1]:
        raise ValueError("Need more complete observations than at least two features")
    if not np.all(np.isfinite(x)) or np.any(np.std(x, axis=0, ddof=1) == 0):
        raise ValueError("Need finite nonconstant variables")
    corr = np.corrcoef(x, rowvar=False)
    if np.linalg.matrix_rank(corr) != x.shape[1]:
        raise ValueError("Correlation matrix is singular; remove duplicate/collinear columns")
    return frame, x, corr


def kmo_sampling_adequacy(data) -> dict:
    """Kaiser-Meyer-Olkin overall and individual measures from correlations."""
    frame, _, corr = _validated_matrix(data)
    precision = np.linalg.inv(corr)
    scale = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    partial = -precision / scale
    np.fill_diagonal(partial, 0.0)
    r2 = corr ** 2
    np.fill_diagonal(r2, 0.0)
    p2 = partial ** 2
    numerators = np.sum(r2, axis=0)
    denominators = numerators + np.sum(p2, axis=0)
    if np.any(denominators == 0) or np.sum(denominators) == 0:
        raise ValueError("KMO is undefined when correlations and partial correlations vanish")
    individual = numerators / denominators
    overall = float(numerators.sum() / denominators.sum())
    return {
        "kmo": overall,
        "by_variable": pd.Series(individual, index=frame.columns, name="kmo"),
        "interpretation_ar": (
            "يقيس KMO ملاءمة الترابط لتحليل العوامل، ولا يثبت صحة البنية العاملية."
        ),
        "interpretation_en": (
            "KMO measures correlation suitability for factor analysis; it does not validate a factor structure."
        ),
    }


def bartlett_sphericity_test(data, *, alpha=.05):
    """Bartlett test of identity population correlation matrix for factor analysis."""
    frame, x, corr = _validated_matrix(data)
    n, p = x.shape
    sign, logdet = np.linalg.slogdet(corr)
    if sign <= 0:
        raise ValueError("Positive-definite correlation matrix required")
    chi2 = -(n - 1 - (2*p + 5)/6) * logdet
    degrees = p*(p-1)//2
    p_value = float(stats.chi2.sf(chi2, degrees))
    return make_result(
        "Bartlett sphericity", chi2, p_value, n, alpha,
        "مصفوفة الارتباط السكاني تساوي مصفوفة الوحدة.",
        f"درجات الحرية = {degrees}.",
        null_en="The population correlation matrix is the identity matrix.",
        notes_en=f"Degrees of freedom: {degrees}.",
    )


def condition_number_vif(data) -> dict:
    """Condition number of standardized features and correlation-based VIFs."""
    frame, x, corr = _validated_matrix(data)
    eigenvalues = np.linalg.eigvalsh(corr)
    condition_index = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
    inv = np.linalg.inv(corr)
    vifs = pd.Series(np.diag(inv), index=frame.columns, name="vif")
    return {
        "condition_index": condition_index,
        "vif": vifs,
        "interpretation_ar": "ارتفاع VIF ورقم الحالة قد يشير إلى تعدد ارتباط خطي؛ ليست هذه وحدها حجة لحذف المتغيرات.",
        "interpretation_en": "High VIF or condition index can indicate multicollinearity; they alone do not justify deleting features.",
    }
