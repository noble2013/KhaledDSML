"""Mediation and moderation diagnostics for observational research.

These are *statistical associations*, not evidence of natural/direct causal
effects. The analysis requires a preregistered design and domain scrutiny of
unmeasured confounding, linearity, temporal ordering and sampling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _sm():
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError('Install khaled-dsml[econometrics] for mediation/moderation') from exc
    return sm


def _alpha(alpha):
    if isinstance(alpha, bool) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")


def _data(x, m, y, controls=None, *, n_min=30):
    arrays = []
    for name, v in [("exposure", x), ("mediator/moderator", m), ("outcome", y)]:
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 1 or len(arr) < n_min or not np.isfinite(arr).all():
            raise ValueError(f"{name} requires >= {n_min} finite aligned observations")
        if arr.std() <= 0:
            raise ValueError(f"{name} has zero variation")
        arrays.append(arr)
    if len(set(map(len, arrays))) != 1:
        raise ValueError("All arrays must be aligned and have equal length")
    n = len(arrays[0])
    if controls is None:
        covariates = np.empty((n, 0), dtype=float)
    else:
        covariates = np.asarray(controls, dtype=float)
        if covariates.ndim == 1:
            covariates = covariates[:, None]
        if (covariates.ndim != 2 or len(covariates) != n
            or covariates.shape[1] < 1 or not np.isfinite(covariates).all()):
            raise ValueError("controls must have shape (n,) or (n, k), all finite")
    if n <= covariates.shape[1] + 8:
        raise ValueError("Not enough observations compared with the covariates")
    return (*arrays, covariates)


def _fit(outcome, design):
    sm = _sm()
    matrix = sm.add_constant(design, has_constant="add")
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError("Regression design must have full column rank")
    return sm.OLS(outcome, matrix).fit()


def mediation_paths(exposure, mediator, outcome, *, controls=None):
    """OLS product-of-coefficients statistical mediation path summary.

    X→M gives a. Y~X+M gives b,c'. Y~X gives c. Identity c≈c'+a*b
    requires same rows, intercept and exactly matched covariate adjustment.
    """
    x, m, y, covariates = _data(exposure, mediator, outcome, controls)
    with_x = np.column_stack([x, covariates])
    with_xm = np.column_stack([x, m, covariates])
    mediator_fit = _fit(m, with_x)
    outcome_fit = _fit(y, with_xm)
    total_fit = _fit(y, with_x)
    a = float(mediator_fit.params[1])
    b = float(outcome_fit.params[2])
    direct = float(outcome_fit.params[1])
    total = float(total_fit.params[1])
    indirect = a*b
    return {
        "test": "OLS product-of-coefficients mediation",
        "a_path": a, "b_path": b, "direct_effect": direct,
        "indirect_effect": float(indirect), "total_effect": total,
        "decomposition_residual": float(total-(direct+indirect)),
        "a_se": float(mediator_fit.bse[1]),
        "b_se": float(outcome_fit.bse[2]),
        "n_samples": len(x), "n_covariates": covariates.shape[1],
        "interpretation_ar": (
            "تحليل وساطة خطية وصفية؛ حاصل ضرب a×b أثر غير مباشر إحصائي، "
            "ولا يثبت سببية الوساطة أو غياب المربكات غير المقاسة."
        ),
        "interpretation_en": (
            "OLS mediation path decomposition is associational, not evidence "
            "of causal mediation or absence of unmeasured confounding."
        ),
    }


def sobel_mediation_test(exposure, mediator, outcome, *, controls=None, alpha=.05):
    """Two-sided Sobel delta-method z test for a*b=0.

    Neglects cross-equation covariance and may be unreliable because the
    product has an asymmetric finite-sample distribution; prefer bootstrap.
    """
    _alpha(alpha)
    r = mediation_paths(exposure, mediator, outcome, controls=controls)
    a, b = r["a_path"], r["b_path"]
    se = np.sqrt(b*b*r["a_se"]**2 + a*a*r["b_se"]**2)
    if not np.isfinite(se) or se<=0:
        raise ValueError("Sobel standard error undefined for this sample")
    z = float((a*b)/se)
    p = float(2*stats.norm.sf(abs(z)))
    return {
        **r, "test": "Sobel indirect-effect z approximation",
        "statistic": z, "p_value": p, "alpha":alpha,
        "reject_null": bool(p<alpha), "indirect_se":float(se),
        "null_hypothesis_ar": "حاصل ضرب مساري الوساطة a×b يساوي صفرًا.",
        "null_hypothesis_en": "The product of regression paths a and b equals zero.",
        "interpretation_ar": (
            "اختبار Sobel التقريبي لحاصل ضرب المسارين؛ توزيعه غير طبيعي غالبًا "
            "خاصة في العينات المحدودة، ولا يمثل إثباتًا لسببية الوساطة."
        ),
        "interpretation_en": (
            "Asymptotic Sobel z on the regression-path product; "
            "the finite-sample product is often non-normal, and mediation is not proven causal."
        ),
    }


def bootstrap_mediation_interval(
    exposure, mediator, outcome, *, controls=None,
    repetitions=1000, confidence_level=.95, seed=42,
):
    """Paired case-resampling percentile CI of OLS indirect association.

    Each resample keeps X,M,Y,covariates together. Returns an interval but
    never invents a p-value; resampling assumes independent observation units.
    """
    if (type(repetitions) is not int or not 100<=repetitions<=10000
        or not np.isfinite(confidence_level) or not 0<confidence_level<1):
        raise ValueError("repetitions must be 100-10000 and confidence level in (0,1)")
    x, m, y, covariates = _data(exposure, mediator, outcome, controls)
    reference = mediation_paths(x, m, y, controls=covariates if covariates.size else None)
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=float)
    n=len(x)
    for i in range(repetitions):
        idx=rng.integers(0,n,size=n)
        design_m=np.column_stack([np.ones(n),x[idx],covariates[idx]])
        design_y=np.column_stack([np.ones(n),x[idx],m[idx],covariates[idx]])
        if (np.linalg.matrix_rank(design_m)<design_m.shape[1]
            or np.linalg.matrix_rank(design_y)<design_y.shape[1]):
            raise ValueError("Rank-deficient bootstrap sample; increase study size or revise design")
        a = np.linalg.lstsq(design_m, m[idx], rcond=None)[0][1]
        b = np.linalg.lstsq(design_y, y[idx], rcond=None)[0][2]
        samples[i] = a*b
    tail=(1-confidence_level)/2
    lower, upper = np.quantile(samples,[tail,1-tail])
    return {
        "test": "Case bootstrap mediation indirect effect",
        "indirect_effect":reference["indirect_effect"],
        "ci_lower":float(lower), "ci_upper":float(upper),
        "confidence_level":confidence_level,
        "n_samples":n,"replications":repetitions,
        "interpretation_ar": (
            "فاصل Bootstrap المئيني لحاصل ضرب مساري الوساطة مع سحب الحالات كاملة؛ "
            "لا يمثل قيمة p ولا يثبت السببـية."
        ),
        "interpretation_en": (
            "Paired-case percentile bootstrap of the OLS indirect association; "
            "not a hypothesis-test p-value or evidence of causal mediation."
        ),
    }


def _moderator_models(x, moderator, y, covariates):
    base = _fit(y, np.column_stack([x, moderator, covariates]))
    full = _fit(y, np.column_stack([x, moderator, x*moderator, covariates]))
    return base, full


def moderation_interaction_test(
    exposure, moderator, outcome, *, controls=None, alpha=.05, robust="HC3",
):
    """OLS moderation via X*W; nested R² increase and interaction significance.

    Nested F test assumes classical homoscedastic errors. The interaction
    t-test supports HC3 robust covariance by default; both are reported
    distinctly to avoid interpreting the F as robust.
    """
    _alpha(alpha)
    if robust not in (None, "HC3"):
        raise ValueError("robust must be HC3 or None")
    x,w,y,covariates=_data(exposure,moderator,outcome,controls)
    reduced, full = _moderator_models(x,w,y,covariates)
    fitted=full.get_robustcov_results(cov_type=robust) if robust else full
    idx=3
    statistic=float(fitted.tvalues[idx])
    p=float(fitted.pvalues[idx])
    f, f_p, _=full.compare_f_test(reduced)
    return {
        "test": "OLS interaction moderation",
        "interaction_coefficient":float(full.params[idx]),
        "interaction_se":float(fitted.bse[idx]),
        "statistic":statistic,"p_value":p,"alpha":alpha,
        "reject_null":bool(p<alpha),
        "r2_main_effects":float(reduced.rsquared),
        "r2_interaction":float(full.rsquared),
        "delta_r2":float(full.rsquared-reduced.rsquared),
        "nested_f":float(f),"nested_f_p_value":float(f_p),
        "covariance": robust or "nonrobust",
        "n_samples":len(x),
        "interpretation_ar": (
            "معامل التفاعل يقيس تغيّر ميل X بتغيّر W في نموذج خطي. "
            "اختبار التفاعل يستخدم HC3 إذا طُلب؛ اختبار F الإضافي يفترض تباينًا ثابتًا."
        ),
        "interpretation_en": (
            "The X×W coefficient describes a varying linear association. "
            "Interaction t uses HC3 if selected; the nested-F for ΔR² assumes "
            "homoscedastic independent regression errors."
        ),
    }


def simple_slopes_table(
    exposure, moderator, outcome, *, moderator_values=None,
    controls=None, alpha=.05, robust="HC3",
):
    """Conditional slope of X at chosen W values with covariance-based CIs."""
    _alpha(alpha)
    if robust not in (None,"HC3"):
        raise ValueError("robust must be HC3 or None")
    x,w,y,covariates=_data(exposure,moderator,outcome,controls)
    _,full=_moderator_models(x,w,y,covariates)
    fitted=full.get_robustcov_results(cov_type=robust) if robust else full
    if moderator_values is None:
        moderator_values=[float(np.mean(w)-np.std(w,ddof=1)),
                          float(np.mean(w)),
                          float(np.mean(w)+np.std(w,ddof=1))]
    vals=np.asarray(moderator_values,dtype=float)
    if vals.ndim!=1 or len(vals)<1 or len(vals)>100 or not np.isfinite(vals).all():
        raise ValueError("Supply 1 to 100 finite moderator levels")
    cov=np.asarray(fitted.cov_params())
    df=float(full.df_resid)
    crit=float(stats.t.ppf(1-alpha/2,df))
    table=[]
    for value in vals:
        slope=float(full.params[1]+full.params[3]*value)
        variance=float(cov[1,1]+2*value*cov[1,3]+value*value*cov[3,3])
        if variance<=0:
            raise ValueError("Conditional slope covariance is nonpositive")
        se=np.sqrt(variance)
        t=slope/se
        p=float(2*stats.t.sf(abs(t),df))
        table.append({
            "moderator_value":float(value), "slope":slope,
            "standard_error":float(se),
            "ci_lower":float(slope-crit*se),
            "ci_upper":float(slope+crit*se),
            "p_value":p,"reject_null":bool(p<alpha),
        })
    return {
        "test": "OLS conditional simple slopes",
        "table":pd.DataFrame(table),
        "alpha":alpha,"n_samples":len(x),
        "covariance":robust or "nonrobust",
        "interpretation_ar": (
            "ميول مشروطة لمتغير X عند قيم محددة مسبقًا للمتغير المعدِّل W؛ "
            "القيم خارج نطاق W المرصود تمثل استقراءً غير مضمون."
        ),
        "interpretation_en": (
            "Conditional X slopes at specified W values; values outside the "
            "observed moderator range are extrapolation, not empirical evidence."
        ),
    }


def johnson_neyman_region(
    exposure, moderator, outcome, *, controls=None,
    alpha=.05, robust="HC3",
):
    """Johnson–Neyman boundaries for conditional X slope over observed W range.

    Solves (b_x+b_xw*w)^2=t_critical^2*Var(b_x+b_xw*w) analytically.
    Only boundaries within the observed W range are reported.
    """
    _alpha(alpha)
    if robust not in (None,"HC3"):
        raise ValueError("robust must be HC3 or None")
    x,w,y,covariates=_data(exposure,moderator,outcome,controls)
    _,model=_moderator_models(x,w,y,covariates)
    fitted=model.get_robustcov_results(cov_type=robust) if robust else model
    cov=np.asarray(fitted.cov_params())
    t2=float(stats.t.ppf(1-alpha/2,model.df_resid)**2)
    bx,bxw=float(model.params[1]),float(model.params[3])
    A=bxw**2-t2*cov[3,3]
    B=2*(bx*bxw-t2*cov[1,3])
    C=bx**2-t2*cov[1,1]
    eps=1e-12*max(1,abs(A),abs(B),abs(C))
    roots=[]
    if abs(A)<eps:
        if abs(B)>=eps:
            roots=[-C/B]
    else:
        roots=[complex(z) for z in np.roots([A,B,C])]
    minimum, maximum=float(w.min()),float(w.max())
    within=sorted({
        float(np.real(z)) for z in roots
        if abs(np.imag(z))<1e-8 and minimum<=float(np.real(z))<=maximum
    })
    boundaries=[minimum,*within,maximum]
    intervals=[]
    for a,b in zip(boundaries[:-1],boundaries[1:]):
        middle=(a+b)/2
        slope=bx+bxw*middle
        variance=cov[1,1]+2*middle*cov[1,3]+middle**2*cov[3,3]
        sig=bool(variance>0 and slope*slope>t2*variance)
        intervals.append({
            "moderator_lower":a, "moderator_upper":b,
            "slope_significant":sig,
        })
    return {
        "test":"Johnson-Neyman conditional significance",
        "table":pd.DataFrame(intervals),
        "boundaries":within,"moderator_min":minimum,"moderator_max":maximum,
        "alpha":alpha,"n_samples":len(x),
        "covariance":robust or "nonrobust",
        "interpretation_ar": (
            "حدود Johnson–Neyman محصورة بنطاق المعدل المرصود وتحدد دلالة الميل "
            "الإحصائية، لا تأثيرًا سببيًا أو دلالة عملية مؤكدة."
        ),
        "interpretation_en": (
            "Johnson–Neyman roots are restricted to observed W; intervals "
            "describe conditional statistical significance, not causality or practical importance."
        ),
    }
