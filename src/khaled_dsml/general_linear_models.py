"""Classical ANCOVA, one-way MANOVA and one-factor repeated-measures ANOVA.

All three are *designed analyses*, not causal identification. Independent
cases, correctly specified contrasts, residual conditions and preregistered
hypotheses are the responsibility of the analyst. Statsmodels stays optional
and is imported only on invocation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _require_statsmodels():
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import AnovaRM, anova_lm
        from statsmodels.multivariate.manova import MANOVA
    except ImportError as exc:
        raise ImportError('Install with: pip install "khaled-dsml[econometrics]"') from exc
    return sm, smf, AnovaRM, anova_lm, MANOVA


def _alpha(alpha):
    if isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha):
        raise ValueError("alpha must be a number strictly in (0,1)")
    try:
        valid = np.isfinite(alpha) and 0 < alpha < 1
    except TypeError:
        valid = False
    if not valid:
        raise ValueError("alpha must be a number strictly in (0,1)")
    return float(alpha)


def _values(values, label, *, minimum=2):
    try:
        arr = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} must be finite numeric values") from exc
    if arr.ndim != 1 or len(arr) < minimum or not np.isfinite(arr).all():
        raise ValueError(f"{label} must be 1D and contain at least {minimum} finite numbers")
    return arr


def _groups(groups, n, minimum_per_group=4):
    try:
        series = pd.Series(groups)
    except (TypeError, ValueError) as exc:
        raise ValueError("groups must be a 1D sequence of labels") from exc
    if len(series) != n or series.isna().any() or not all(
        isinstance(v, str) and v.strip() for v in series
    ):
        raise ValueError("Use nonmissing, nonempty string group labels aligned to rows")
    unique = list(pd.unique(series))
    if len(unique) < 2 or len(unique) > 40:
        raise ValueError("Require between 2 and 40 distinct groups")
    counts = series.value_counts()
    if counts.min() < minimum_per_group:
        raise ValueError(f"Each group needs at least {minimum_per_group} cases")
    return series.reset_index(drop=True), unique


def ancova_one_factor(
    response, groups, covariate, *, alpha=0.05,
):
    """One-factor ANCOVA with one continuous covariate and common group slope.

    Main group effect is the added-variable partial F test comparing:
        H0: response ~ covariate
        H1: response ~ covariate + group
    A separate *diagnostic* partial F test checks the group-by-covariate
    interaction against the additive model; rejected parallel slopes
    invalidate straightforward common-slope adjusted-mean interpretation.
    Adjusted group predictions are evaluated at the pooled mean covariate.
    No causal claim or randomized-group assumption is inferred from model fit.
    """
    alpha = _alpha(alpha)
    y = _values(response, "response", minimum=16)
    x = _values(covariate, "covariate", minimum=16)
    if len(y) != len(x):
        raise ValueError("response and covariate must have the same length")
    group, levels = _groups(groups, len(y))
    sm, smf, _, anova_lm, _ = _require_statsmodels()
    data = pd.DataFrame({"_y": y, "_cov": x, "_group": group})
    reduced = smf.ols("_y ~ _cov", data=data).fit()
    full = smf.ols("_y ~ _cov + C(_group)", data=data).fit()
    interacted = smf.ols("_y ~ _cov * C(_group)", data=data).fit()
    if (full.df_resid < 4 or interacted.df_resid < 4 or
        np.linalg.matrix_rank(full.model.exog) != full.model.exog.shape[1] or
        np.linalg.matrix_rank(interacted.model.exog) != interacted.model.exog.shape[1] or
        full.ssr <= 1e-12 or interacted.ssr <= 1e-12):
        raise ValueError("Insufficient variation, full rank or positive residual df for ANCOVA")
    comparison = anova_lm(reduced, full)
    homog = anova_lm(full, interacted)
    f_group = float(comparison["F"].iloc[1])
    p_group = float(comparison["Pr(>F)"].iloc[1])
    f_homog = float(homog["F"].iloc[1])
    p_homog = float(homog["Pr(>F)"].iloc[1])
    if not all(np.isfinite(z) for z in (f_group, p_group, f_homog, p_homog)):
        raise ValueError("ANCOVA inferential statistics are undefined")
    avg_cov = float(x.mean())
    adjusted = pd.DataFrame({"group": levels, "_group": levels,
                             "_cov": [avg_cov] * len(levels)})
    adjusted["adjusted_mean"] = full.predict(adjusted)
    adjusted["n_group"] = [int((group == level).sum()) for level in levels]
    return {
        "test": "One-factor ANCOVA common-slope adjusted group test",
        "n_samples": len(y), "n_groups": len(levels), "alpha": alpha,
        "covariate_mean": avg_cov, "statistic": f_group, "p_value": p_group,
        "numerator_df": float(comparison["df_diff"].iloc[1]),
        "denominator_df": float(full.df_resid),
        "reject_equal_adjusted_means": bool(p_group < alpha),
        "slope_homogeneity_f": f_homog,
        "slope_homogeneity_p_value": p_homog,
        "slope_homogeneity_rejected": bool(p_homog < alpha),
        "partial_eta_squared": float((reduced.ssr - full.ssr) / reduced.ssr),
        "table": adjusted[["group", "n_group", "adjusted_mean"]],
        "null_hypothesis_ar": "متوسطات المجموعات المعدّلة عند مستوى التغاير نفسه متساوية ضمن نموذج الميل المشترك.",
        "null_hypothesis_en": "Adjusted group means are equal under a common covariate slope.",
        "interpretation_ar": (
            "تحليل ANCOVA بعامل تصنيفي وتغاير عددي واحد. قيمة p للمجموعات "
            "تقارن النموذج المعدّل مع نموذج التغاير وحده؛ يُفحص تفاعل "
            "المجموعة مع التغاير منفصلًا. رفض توازي الميل يجعل تفسير "
            "المتوسطات المعدّلة بميل مشترك غير ملائم. النتائج ارتباطية، "
            "وتتطلب استقلالية الحالات وصحة افتراضات البواقي."
        ),
        "interpretation_en": (
            "Classical ANCOVA partial F compares the additive group model "
            "to the covariate-only model. A separate interaction test "
            "examines parallel slopes. If interaction is detected, do "
            "not interpret common-slope adjusted means uncritically. "
            "Independent cases and valid residual assumptions are required; "
            "association is not causation."
        ),
    }


def manova_one_factor(outcomes, groups, *, alpha=0.05):
    """One-way multivariate ANOVA with Wilks, Pillai, Hotelling and Roy.

    Outcomes are complete continuous measurements on independent subjects.
    Uses statsmodels MANOVA multivariate tests for the one-factor group term.
    Requires a full-rank pooled *within-group residual* covariance and at
    least 4 observations per group (plus n > p+number_of_groups+2).
    Wilks p-value is a model-based approximation; multiple alternative
    test statistics in the table should not be interpreted as independent
    confirmatory hypothesis tests. Not a repeated-measures procedure.
    """
    alpha = _alpha(alpha)
    try:
        array = np.asarray(outcomes, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("outcomes must be a complete numeric matrix") from exc
    if (array.ndim != 2 or array.shape[1] < 2 or array.shape[1] > 15
        or array.shape[0] < 16 or not np.isfinite(array).all()):
        raise ValueError("Need finite 2D outcomes, >=2 dimensions, and >=16 cases")
    n, p = array.shape
    group, levels = _groups(groups, n, minimum_per_group=max(4, p+1))
    g = len(levels)
    if n <= p + g + 2:
        raise ValueError("More independent observations are required for MANOVA")
    residuals = array - pd.DataFrame(array).groupby(group).transform("mean").to_numpy()
    if np.linalg.matrix_rank(residuals) < p:
        raise ValueError("Pooled within-group outcome covariance must have full rank")
    _, _, _, _, MANOVA = _require_statsmodels()
    df = pd.DataFrame(array, columns=[f"_y{i}" for i in range(p)])
    df["_group"] = group
    formula = " + ".join(f"_y{i}" for i in range(p)) + " ~ C(_group)"
    stats_table = MANOVA.from_formula(formula, data=df).mv_test().results["C(_group)"]["stat"]
    keys = ["Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace",
            "Roy's greatest root"]
    rows = []
    for name in keys:
        row = stats_table.loc[name]
        values = [float(row[col]) for col in ("Value", "F Value", "Num DF", "Den DF", "Pr > F")]
        if not all(np.isfinite(v) for v in values):
            raise ValueError("MANOVA statistics are nonfinite")
        rows.append({
            "method": name, "statistic": values[0],
            "approx_f": values[1], "num_df": values[2], "den_df": values[3],
            "p_value": values[4],
        })
    table = pd.DataFrame(rows)
    p_wilks = float(rows[0]["p_value"])
    return {
        "test": "One-way MANOVA group effect",
        "n_samples": n, "n_groups": g, "n_outcomes": p,
        "alpha": alpha, "wilks_lambda": float(rows[0]["statistic"]),
        "pillai_trace": float(rows[1]["statistic"]),
        "p_value": p_wilks, "reject_equal_mean_vectors_wilks": bool(p_wilks < alpha),
        "table": table,
        "null_hypothesis_ar": "متجهات متوسطات المخرجات المتعددة متساوية في جميع المجموعات.",
        "null_hypothesis_en": "All groups share the same multivariate response mean vector.",
        "interpretation_ar": (
            "تحليل MANOVA لعامل واحد ومخرجات متعددة. تُعرض إحصاءات ويلكس "
            "وبيلّاي وهوتلينغ وروي كما يحسبها statsmodels، مع اعتماد قيمة "
            "ويلكس الافتراضية للقرار. لا يُعد اختبارًا لقياسات الشخص المتكررة؛ "
            "يتطلب استقلالية الحالات وتغايرًا كافي الرتبة وصحة افتراضات النموذج."
        ),
        "interpretation_en": (
            "One-way MANOVA reports Wilks, Pillai, Hotelling–Lawley and "
            "Roy statistics using statsmodels, with Wilks the designated "
            "default decision. Independent subjects, continuous outcomes, "
            "full-rank pooled residual covariance and valid multivariate "
            "model assumptions are required. Not repeated-measures ANOVA."
        ),
    }


def repeated_measures_one_factor(wide, *, alpha=0.05):
    """Balanced one-factor repeated-measures ANOVA on complete wide data.

    Each row is one subject; each column is a repeated condition/time point.
    statsmodels AnovaRM assumes sphericity for the classical F reference.
    No Greenhouse–Geisser or Huynh–Feldt correction is claimed. This is a
    descriptive-design classical test, not a mixed model for missing visits.
    """
    alpha = _alpha(alpha)
    if isinstance(wide, pd.DataFrame):
        if not wide.columns.is_unique:
            raise ValueError("Repeated-measure columns must be unique")
        names = list(map(str, wide.columns))
    else:
        names = None
    try:
        y = np.asarray(wide, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Repeated observations must be numeric") from exc
    if y.ndim != 2 or y.shape[0] < 6 or not 2 <= y.shape[1] <= 20 or not np.isfinite(y).all():
        raise ValueError("Need >=6 subjects, 2..20 repeated conditions and complete finite values")
    n, k = y.shape
    if names is None:
        names = [f"visit_{i + 1}" for i in range(k)]
    if len(set(names)) != k:
        raise ValueError("Repeated-measure names must remain unique as strings")
    if np.sum((y - y.mean(axis=1, keepdims=True)) ** 2) <= 1e-12:
        raise ValueError("Repeated-measure differences are all constant")
    _, _, AnovaRM, _, _ = _require_statsmodels()
    df = pd.DataFrame(y, columns=names)
    df["subject"] = np.arange(n)
    long = df.melt(id_vars=["subject"], var_name="occasion", value_name="observed")
    record = AnovaRM(long, "observed", "subject", within=["occasion"]).fit().anova_table.loc["occasion"]
    f = float(record["F Value"])
    numerator_df, denominator_df = float(record["Num DF"]), float(record["Den DF"])
    pval = float(record["Pr > F"])
    if not all(np.isfinite(x) for x in (f, numerator_df, denominator_df, pval)):
        raise ValueError("Repeated ANOVA F statistic is undefined")
    return {
        "test": "One-factor repeated-measures ANOVA uncorrected",
        "n_samples": n, "n_conditions": k,
        "statistic": f, "p_value": pval,
        "numerator_df": numerator_df, "denominator_df": denominator_df,
        "alpha": alpha, "reject_equal_condition_means": bool(pval < alpha),
        "sphericity_correction": "none; uncorrected classical F",
        "table": pd.DataFrame({
            "condition": names, "mean": y.mean(axis=0),
            "std": y.std(axis=0, ddof=1),
        }),
        "null_hypothesis_ar": "متوسطات شروط القياس المتكرر على الأفراد أنفسهم متساوية.",
        "null_hypothesis_en": "Population condition means are equal for repeated measurements on subjects.",
        "interpretation_ar": (
            "اختبار ANOVA أحادي العامل للقياسات المتكررة على المشاركين أنفسهم "
            "وبدون قيم ناقصة. تتطلب دلالة F التقليدية افتراض الكروية؛ "
            "لا تطبّق هذه الدالة تصحيح Greenhouse–Geisser أو Huynh–Feldt. "
            "لا تستبدل النموذج المختلط للزيارات المفقودة."
        ),
        "interpretation_en": (
            "Classical one-factor repeated-measures ANOVA on complete "
            "balanced subjects. The nominal F p-value assumes sphericity; "
            "neither Greenhouse–Geisser nor Huynh–Feldt corrections are "
            "applied. Not a mixed-effects model for incomplete visits."
        ),
    }
