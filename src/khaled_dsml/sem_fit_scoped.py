"""Scientifically scoped SEM fit and nested measurement invariance summaries.

This module only examines numbers from already fitted covariance-based models.
It does NOT fit SEM/CFA models or ensure ML, independent cases or measurement
identification. Indices and chi-square comparisons must use consistent input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _finite(value, label, *, minimum=None, strict=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{label} must be a finite numeric scalar")
    try: v=float(value)
    except (ValueError,TypeError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not np.isfinite(v) or (minimum is not None and
        (v<=minimum if strict else v<minimum)):
        raise ValueError(f"{label} has invalid value")
    return v


def sem_fit_indices(chi_square, degrees_of_freedom, baseline_chi_square,
                    baseline_degrees_of_freedom, n_samples, *,
                    observed_covariance=None, implied_covariance=None):
    """Compute conventional ML SEM CFI, TLI, RMSEA and optionally SRMR.

    Uses CFI=1-max(chi-df,0)/max(chi-df,chi0-df0,0) with baseline >0,
    TLI=(chi0/df0-chi/df)/(chi0/df0-1), RMSEA=sqrt(max((chi-df)/
    (df*(n-1)),0)). Returns raw CFI and TLI, not silently clipping them.

    SRMR is optional and computed from standardized covariance residuals
    with mean squares over the upper triangular covariance matrix (including
    diagonal), each residual standardized by sqrt(diag(S)_i*diag(S)_j).
    SRMR conventions vary across SEM packages; this is one declared form.

    Requires previously estimated models using the same observations,
    estimator and reference chi-square conventions; do not mix robust
    corrections/estimators and standard ML metrics.
    """
    chi=_finite(chi_square,"chi_square",minimum=0)
    df=_finite(degrees_of_freedom,"degrees_of_freedom",minimum=0,strict=True)
    chi0=_finite(baseline_chi_square,"baseline_chi_square",minimum=0)
    df0=_finite(baseline_degrees_of_freedom,"baseline_degrees_of_freedom",minimum=0,strict=True)
    if type(n_samples) is not int or n_samples<10:
        raise ValueError("n_samples must be an integer >=10")
    base_ratio=chi0/df0
    if base_ratio<=1+1e-10:
        raise ValueError("Independence baseline chi-square / df must exceed 1 for stable TLI")
    numerator=max(chi-df,0.)
    denominator=max(chi-df,chi0-df0,0.)
    if denominator<=1e-12:
        raise ValueError("Degenerate CFI numerator/denominator")
    cfi=1-numerator/denominator
    tli=(base_ratio-chi/df)/(base_ratio-1)
    rmsea=float(np.sqrt(max((chi-df)/(df*(n_samples-1)),0)))
    srmr=None
    if (observed_covariance is None)!=(implied_covariance is None):
        raise ValueError("SRMR requires BOTH observed and implied covariance matrices")
    if observed_covariance is not None:
        try:
            observed=np.asarray(observed_covariance,dtype=float)
            implied=np.asarray(implied_covariance,dtype=float)
        except (TypeError,ValueError) as exc:
            raise ValueError("Covariance inputs must be numeric square matrices") from exc
        if (observed.ndim!=2 or observed.shape!=implied.shape or
            observed.shape[0]!=observed.shape[1] or
            not 2<=len(observed)<=100 or
            not np.isfinite(observed).all() or not np.isfinite(implied).all()
            or not np.allclose(observed,observed.T,rtol=1e-8,atol=1e-8)
            or not np.allclose(implied,implied.T,rtol=1e-8,atol=1e-8)
            or np.any(np.diag(observed)<=0)
            or np.linalg.eigvalsh(observed).min()<=0
            or np.linalg.eigvalsh(implied).min()<=0):
            raise ValueError("Observed and implied matrices must be same shape symmetric positive-definite")
        denom=np.sqrt(np.outer(np.diag(observed),np.diag(observed)))
        residual=(observed-implied)/denom
        index=np.triu_indices(len(observed))
        srmr=float(np.sqrt(np.mean(residual[index]**2)))
    return {
        "test":"Conventional covariance-based SEM fit indices",
        "n_samples":n_samples,"chi_square":chi,"degrees_of_freedom":df,
        "baseline_chi_square":chi0,"baseline_degrees_of_freedom":df0,
        "cfi":float(cfi),"tli":float(tli),"rmsea":rmsea,"srmr":srmr,
        "srmr_convention":"standardized covariance upper triangle including diagonal",
        "interpretation_ar":(
            "مؤشرات المطابقة CFI وTLI وRMSEA وSRMR الوصفية من نتائج "
            "نموذج SEM مقدّر مسبقًا. لا تُقدّر الدالة المعادلات البنائية "
            "ولا تثبت صلاحية نموذج القياس. ينبغي توحيد تقدير النموذج "
            "والنموذج المرجعي والعينة؛ لا تُخلط إحصاءات ML "
            "الاعتيادية مع تصحيحات robust."
        ),
        "interpretation_en":(
            "Descriptive conventional covariance-based SEM fit indices from "
            "a previously fitted model and matching independence baseline. "
            "This does not fit or validate SEM. Use consistent sample and "
            "unscaled ML chi-square definitions; do not mix robust-scaled "
            "statistics. SRMR uses the documented covariance convention."
        ),
    }


def measurement_invariance_nested_ml(less_restricted_chi_square,
                                     less_restricted_df,
                                     more_restricted_chi_square,
                                     more_restricted_df, *,
                                     alpha=.05, level="metric"):
    """Naive nested ML chi-square difference for pre-fitted CFA invariance.

    The two models must be genuinely nested with IDENTICAL observed data,
    fitted via the same conventional ML estimator. Ordinal WLSMV, robust
    corrections, different missing-data patterns, clusters and bootstraps
    require a method-specific scaled difference procedure, NOT this one.
    Levels are descriptive input labels, NOT identified by this function.
    """
    chi0=_finite(less_restricted_chi_square,"less_restricted_chi_square",minimum=0)
    df0=_finite(less_restricted_df,"less_restricted_df",minimum=0,strict=True)
    chi1=_finite(more_restricted_chi_square,"more_restricted_chi_square",minimum=0)
    df1=_finite(more_restricted_df,"more_restricted_df",minimum=0,strict=True)
    alpha=_finite(alpha,"alpha",minimum=0,strict=True)
    if alpha>=1 or not isinstance(level,str) or level not in ("metric","scalar","strict","other"):
        raise ValueError("alpha in (0,1) and one of four declared comparison labels required")
    difference=chi1-chi0
    ddf=df1-df0
    if ddf<=0 or difference< -1e-8:
        raise ValueError("Require increased df and no improved chi-square for nested ML model")
    difference=max(0.,difference)
    p=float(stats.chi2.sf(difference,ddf))
    return {
        "test":"Nested conventional-ML measurement invariance screen",
        "invariance_level":level,"chi_square_difference":float(difference),
        "degrees_of_freedom_difference":float(ddf),
        "statistic":float(difference),"p_value":p,"alpha":float(alpha),
        "reject_added_invariance_constraints":bool(p<alpha),
        "table":pd.DataFrame({
            "model":["less restricted","more restricted"],
            "chi_square":[chi0,chi1],"df":[df0,df1],
        }),
        "interpretation_ar":(
            "فحص فروق كاي تربيع لنموذجَي CFA متداخلين مقدرين مسبقًا "
            "باستخدام ML التقليدي وعلى الحالات نفسها. لا تنشئ الدالة "
            "قيود التكافؤ ولا تتحقق من تعريفها؛ يلزم تصحيح خاص مع "
            "WLSMV أو الاختبارات المتينة أو البيانات غير المستقلة. "
            "عدم الرفض لا يثبت التكافؤ بصورة قاطعة."
        ),
        "interpretation_en":(
            "Ordinary nested ML chi-square difference for pre-fitted "
            "measurement invariance CFA models with genuinely nested "
            "restrictions and identical cases. No invariance fitting "
            "or verification of identification is performed. Scaled/"
            "robust and ordinal estimators need dedicated difference tests; "
            "failure to reject cannot prove invariance."
        ),
    }
