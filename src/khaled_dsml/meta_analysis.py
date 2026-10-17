"""Independent-study meta-analysis diagnostics, with explicit inferential limits.

Inputs must be study-level effect estimates on a COMMON interpretable scale,
paired with their own sampling standard errors. Independence of studies and
correct variances are the analyst's responsibility; duplicate studies and
publication selection are not corrected here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _studies(estimates, standard_errors, *, minimum):
    try:
        y = np.asarray(estimates, dtype=float)
        se = np.asarray(standard_errors, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Numeric study-level effect estimates and standard errors required") from exc
    if (y.ndim != 1 or se.ndim != 1 or len(y) != len(se)
        or len(y) < minimum or len(y) > 100000
        or not np.isfinite(y).all() or not np.isfinite(se).all()
        or np.any(se <= 1e-12) or np.any(se > 1e12)):
        raise ValueError(f"Need >= {minimum} studies, finite effects, positive usable standard errors")
    return y, se


def _summary(y, se):
    weight = 1 / se**2
    total = weight.sum()
    pooled = float(np.dot(weight, y)/total)
    q = float(np.sum(weight*(y-pooled)**2))
    df = len(y) - 1
    correction = float(total - np.dot(weight, weight)/total)
    tau2 = max(0., (q-df)/correction)
    i2 = max(0., (q-df)/q) * 100 if q > 0 else 0.
    return weight, pooled, q, df, correction, tau2, i2


def meta_analysis_heterogeneity(estimates, standard_errors, *, alpha=0.05):
    """Fixed-inverse-variance summary with Cochran Q, I² and DL tau².

    Q p-values rely on a large-sample chi-square reference and may be
    underpowered for small study counts. I² and DL tau² are noisy with few
    studies. Fixed-effect confidence intervals use normal approximation.
    """
    if isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be (0,1)")
    y, se = _studies(estimates, standard_errors, minimum=2)
    weight, pooled, q, df, _, tau2, i2 = _summary(y, se)
    p = float(stats.chi2.sf(q, df))
    z = float(stats.norm.ppf(1-alpha/2))
    fixed_se = float(np.sqrt(1/weight.sum()))
    return {
        "test": "Cochran Q and I-squared heterogeneity",
        "n_studies": len(y), "fixed_effect_estimate": pooled,
        "fixed_effect_standard_error": fixed_se,
        "fixed_ci_lower": pooled-z*fixed_se,
        "fixed_ci_upper": pooled+z*fixed_se,
        "Q": q, "degrees_of_freedom": df, "p_value": p,
        "I_squared_percent": i2, "tau_squared_dl": tau2,
        "alpha": float(alpha), "reject_homogeneity": bool(p < alpha),
        "table": pd.DataFrame({
            "study": np.arange(1, len(y)+1), "effect": y,
            "standard_error": se, "inverse_variance_weight": weight/weight.sum()
        }),
        "null_hypothesis_ar": "تتوافق جميع التأثيرات الحقيقية للدراسات مع تأثير مشترك.",
        "null_hypothesis_en": "Study effects are compatible with one common true effect.",
        "interpretation_ar": (
            "إحصائية Q للتغاير بين الدراسات وI² ونسبة التغاير التقديرية "
            "وtau² بطريقة DerSimonian–Laird. قيم p تقريبية وحساسة لحجم "
            "الدراسات، وI² غير ثابتة خصوصًا عندما يكون عدد الدراسات صغيرًا."
        ),
        "interpretation_en": (
            "Cochran Q tests between-study heterogeneity approximately; "
            "I² is descriptive and DL tau² is a moment estimate. Small "
            "study counts make Q underpowered and I²/tau² unstable. "
            "All effects must be comparable and independent."
        ),
    }


def meta_analysis_random_effects(estimates, standard_errors, *, alpha=0.05):
    """DerSimonian–Laird random-effects pooled estimate with normal interval.

    NOT Hartung-Knapp/Sidik-Jonkman; z-based intervals may undercover for
    small numbers of studies or uncertain heterogeneity. Report tau², I²
    and heterogeneity Q alongside the estimate; no causal claim.
    """
    if isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be (0,1)")
    y, se = _studies(estimates, standard_errors, minimum=3)
    _, _, q, df, _, tau2, i2 = _summary(y, se)
    random_weights = 1/(se**2 + tau2)
    pooled = float(np.dot(random_weights, y)/random_weights.sum())
    std = float(np.sqrt(1/random_weights.sum()))
    z = float(stats.norm.ppf(1-alpha/2))
    return {
        "test": "DerSimonian-Laird random-effects meta-analysis",
        "n_studies": len(y), "estimate": pooled,
        "standard_error": std, "ci_lower": pooled-z*std,
        "ci_upper": pooled+z*std, "alpha": float(alpha),
        "tau_squared_dl": float(tau2), "I_squared_percent": float(i2),
        "Q": float(q), "Q_p_value": float(stats.chi2.sf(q, df)),
        "degrees_of_freedom_Q": df,
        "method": "DerSimonian-Laird (z-based interval)",
        "table": pd.DataFrame({
            "study": np.arange(1, len(y)+1), "effect": y, "standard_error": se,
            "random_effect_weight": random_weights/random_weights.sum(),
        }),
        "interpretation_ar": (
            "متوسط تأثيرات عشوائية بطريقة DerSimonian–Laird وتباين tau² "
            "المقدر بلحظات العينة؛ فاصل الثقة مبني على تقريب z وقد يكون "
            "مفرط التفاؤل مع عدد قليل من الدراسات. لا يعتمد Hartung–Knapp."
        ),
        "interpretation_en": (
            "DerSimonian–Laird random-effects pooled effect with a normal-z "
            "confidence interval, not Hartung–Knapp. It may undercover "
            "with few studies or uncertain tau²; independent comparable "
            "study effects and accurate standard errors are required."
        ),
    }


def egger_small_study_effects_test(estimates, standard_errors, *, alpha=0.05):
    """Egger unweighted precision-regression intercept for k>=10 studies.

    OLS of (effect/SE) on 1/SE with an intercept, df=k-2. Nonzero
    intercept suggests small-study effects, NOT proof of publication bias.
    Study heterogeneity can also produce asymmetry.
    """
    if isinstance(alpha, (bool, np.bool_)) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be (0,1)")
    y, se = _studies(estimates, standard_errors, minimum=10)
    precision = 1/se
    standardized = y/se
    design = np.column_stack((np.ones(len(y)), precision))
    if np.linalg.matrix_rank(design) != 2:
        raise ValueError("Distinct study precisions are required for Egger regression")
    coef, _, _, _ = np.linalg.lstsq(design, standardized, rcond=None)
    residual = standardized - design @ coef
    df = len(y) - 2
    mse = float(residual@residual / df)
    cov = mse * np.linalg.inv(design.T@design)
    if cov[0,0] <= 0:
        raise ValueError("Egger intercept uncertainty is degenerate")
    se_intercept = float(np.sqrt(cov[0,0]))
    t = float(coef[0] / se_intercept)
    p = float(2*stats.t.sf(abs(t), df))
    return {
        "test": "Egger small-study effects regression",
        "n_studies": len(y), "intercept": float(coef[0]),
        "intercept_standard_error": se_intercept,
        "precision_slope": float(coef[1]), "statistic": t,
        "p_value": p, "degrees_of_freedom": df,
        "alpha": float(alpha), "reject_zero_intercept": bool(p < alpha),
        "null_hypothesis_ar": "المقطع الاعتراضي لانحدار إيغر يساوي صفرًا.",
        "null_hypothesis_en": "Egger precision-regression intercept equals zero.",
        "interpretation_ar": (
            "اختبار استكشافي لانحياز مرتبط بحجم الدراسة عبر مقطع انحدار إيغر. "
            "لا يثبت تحيز النشر، إذ يمكن أن ينتج عن عدم التجانس أو "
            "اختلاف جودة الدراسات؛ يُستخدم هنا لعشر دراسات مستقلة على الأقل."
        ),
        "interpretation_en": (
            "Exploratory Egger regression for small-study effects with >=10 "
            "independent studies. Intercept asymmetry is not proof of "
            "publication bias and may reflect heterogeneity or study design."
        ),
    }
