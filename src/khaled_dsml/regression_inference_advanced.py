"""Transparent OLS nested-model inference and model-size diagnostics.

Reference models must be specified *before* inspecting tests. No automatic
search/selection or causal inference. Inference is classical and iid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def nested_ols_wald_lr_score(y, reduced_design, full_design, *, alpha=.05):
    """Three related asymptotic tests of additional linear regressors.

    Wald uses homoskedastic unrestricted OLS residual variance and chi2(q)
    approximation. LR uses Gaussian normal-profiled log likelihood;
    LM/Score uses restricted-model residual variance. The exact finite-sample
    nested-model F statistic is also given as a cross-check. Collinearity,
    heteroskedasticity, dependent errors or model search invalidate p-values.
    """
    try:
        y=np.asarray(y,dtype=float)
        r=np.asarray(reduced_design,dtype=float)
        f=np.asarray(full_design,dtype=float)
    except (TypeError,ValueError) as exc:
        raise ValueError("Need finite numeric outcome and design matrices") from exc
    if (y.ndim!=1 or r.ndim!=2 or f.ndim!=2 or len(y)<20
        or len(r)!=len(y) or len(f)!=len(y) or r.shape[1]<1
        or f.shape[1]<=r.shape[1] or len(y)-f.shape[1]<5
        or not np.isfinite(y).all() or not np.isfinite(r).all()
        or not np.isfinite(f).all() or isinstance(alpha,(bool,np.bool_))
        or not np.isscalar(alpha) or not np.isfinite(alpha) or not 0<alpha<1):
        raise ValueError("Full-rank, aligned nested finite OLS matrices and alpha (0,1) required")
    n,k=len(y),f.shape[1];q=k-r.shape[1]
    if (np.linalg.matrix_rank(r)!=r.shape[1] or np.linalg.matrix_rank(f)!=k
        or np.linalg.norm(r-f@np.linalg.lstsq(f,r,rcond=None)[0]) >
        1e-8*max(1,np.linalg.norm(r))):
        raise ValueError("Reduced columns must lie in full design column space")
    re=y-r@np.linalg.lstsq(r,y,rcond=None)[0]
    fu=y-f@np.linalg.lstsq(f,y,rcond=None)[0]
    s0=float(re@re);s1=float(fu@fu)
    if s0<=1e-12 or s1<=1e-12 or s0+1e-8<s1:
        raise ValueError("Positive residual variance and nested model order required")
    improvement=max(0.,s0-s1)
    wald=improvement/(s1/(n-k))
    lr=n*np.log(s0/s1)
    score=n*improvement/s0
    exact_f=wald/q
    return {
        "test":"Nested OLS Wald LR and Score tests",
        "n_samples":n,"n_parameters_full":k,"n_added_parameters":q,
        "wald_chisquare":float(wald),"wald_p_value":float(stats.chi2.sf(wald,q)),
        "likelihood_ratio":float(lr),"lr_p_value":float(stats.chi2.sf(lr,q)),
        "score_lm":float(score),"score_p_value":float(stats.chi2.sf(score,q)),
        "exact_nested_f":float(exact_f),
        "exact_f_p_value":float(stats.f.sf(exact_f,q,n-k)),
        "alpha":float(alpha),
        "table":pd.DataFrame({
            "method":["Wald chi2","Likelihood ratio chi2","Score/LM chi2","Exact nested OLS F"],
            "statistic":[wald,lr,score,exact_f],
            "degrees_of_freedom_num":[q,q,q,q],
            "degrees_of_freedom_den":[np.nan,np.nan,np.nan,n-k],
            "p_value":[stats.chi2.sf(wald,q),stats.chi2.sf(lr,q),stats.chi2.sf(score,q),
                       stats.f.sf(exact_f,q,n-k)],
        }),
        "null_hypothesis_ar":"المعاملات الإضافية في نموذج OLS الكامل تساوي صفرًا.",
        "null_hypothesis_en":"All regressors added in the full nested OLS model have zero coefficients.",
        "interpretation_ar":(
            "تعرض اختبارات Wald وLR وLM المرجعية لمعاملات خطية إضافية، "
            "إلى جانب اختبار F الدقيق تحت شروط OLS الكلاسيكية. اختبارات "
            "كاي تربيع تقريبية، ولا يجوز استخدامها بعد اختيار المتغيرات "
            "وفق النتائج نفسها دون تصحيح التصميم."
        ),
        "interpretation_en":(
            "Joint Wald, Gaussian LR and restricted Score/LM chi-square "
            "approximations plus classical exact nested OLS F. Validity "
            "requires prespecified nested full-rank designs, iid errors, "
            "appropriate linear conditional mean and homoskedasticity."
        ),
    }


def regression_pseudo_r_squared(log_likelihood, null_log_likelihood, n_samples):
    """McFadden, Cox–Snell and Nagelkerke likelihood-based pseudo-R2.

    Log likelihoods must be for the *same observations* and maximum-likelihood
    models, with null log-likelihood <=0 and full model >= null; comparison
    has meaning only within the same outcome likelihood family.
    """
    try:
        ll=float(log_likelihood);ll0=float(null_log_likelihood)
    except (TypeError,ValueError) as exc:
        raise ValueError("Log likelihoods must be finite") from exc
    if (type(n_samples) is not int or n_samples<3 or not np.isfinite(ll)
        or not np.isfinite(ll0) or ll0>=0 or ll<ll0-1e-8 or ll>0
        or isinstance(log_likelihood,(bool,np.bool_))
        or isinstance(null_log_likelihood,(bool,np.bool_))):
        raise ValueError("n>=3 and finite ll0<=ll<=0, ll0<0 required")
    m=1-ll/ll0
    cs=-np.expm1(2*(ll0-ll)/n_samples)
    maxcs=-np.expm1(2*ll0/n_samples)
    if maxcs<=0:
        raise ValueError("Null model likelihood scale unsuitable")
    return {
        "test":"Likelihood-based pseudo-R-squared",
        "n_samples":n_samples,"log_likelihood":ll,
        "null_log_likelihood":ll0,
        "mcfadden_r2":float(m),"cox_snell_r2":float(cs),
        "nagelkerke_r2":float(cs/maxcs),
        "interpretation_ar":(
            "مؤشرات McFadden وCox–Snell وNagelkerke مستندة إلى لوغاريتم "
            "الاحتمال لعينات ونماذج متوافقة. لا تساوي R² للانحدار الخطي "
            "ولا تمثل نسبة سببية للتباين المفسر أو اختبارًا إحصائيًا."
        ),
        "interpretation_en":(
            "McFadden, Cox–Snell and Nagelkerke pseudo-R2 from compatible "
            "maximum log likelihoods on the same cases. Not ordinary OLS "
            "variance explained, causal attribution or a hypothesis test."
        ),
    }


def cohen_f_squared_nested_models(r_squared_full, r_squared_reduced):
    """Cohen f2 effect size for prespecified nested linear R2 values."""
    try:full=float(r_squared_full);red=float(r_squared_reduced)
    except (ValueError,TypeError) as exc:
        raise ValueError("Nested R2 values must be numeric") from exc
    if (not np.isfinite(full) or not np.isfinite(red)
        or isinstance(r_squared_full,(bool,np.bool_))
        or isinstance(r_squared_reduced,(bool,np.bool_))
        or not 0<=red<=full<1):
        raise ValueError("Require 0 <= reduced R2 <= full R2 < 1")
    f2=(full-red)/(1-full)
    return {
        "test":"Nested regression Cohen f-squared",
        "r_squared_full":full,"r_squared_reduced":red,
        "f_squared":float(f2),
        "interpretation_ar":(
            "حجم أثر f² = (R² الكامل − R² المقيد)/(1 − R² الكامل) "
            "لمقارنة نموذجين خطيين متداخلين وفق نفس العينة. "
            "مؤشر وصفي لا يثبت دلالة أو سببية."
        ),
        "interpretation_en":(
            "Cohen f² is a descriptive incremental-effect size for "
            "prespecified nested R2 fits on the same sample. "
            "It is neither a p-value nor causal evidence."
        ),
    }


def two_sample_power_sensitivity(n_per_group, *, alpha=.05, target_power=.8,
                                 allocation_ratio=1.0):
    """Invert classical two-sided independent t-test noncentral-t power.

    Returns minimum absolute standardized Cohen d detectable at target power
    under the assumed variance equality and sample allocations. It does not
    estimate observed/posthoc power from a reported p-value.
    """
    if (type(n_per_group) is not int or n_per_group<5
        or any(isinstance(v,(bool,np.bool_)) or not np.isscalar(v)
               or not np.isfinite(v) for v in (alpha,target_power,allocation_ratio))
        or not 0<alpha<1 or not alpha<target_power<1 or not .1<=allocation_ratio<=10):
        raise ValueError("n>=5, 0<alpha<target_power<1, ratio in [.1,10] required")
    from statsmodels.stats.power import TTestIndPower
    effect=TTestIndPower().solve_power(
        effect_size=None,nobs1=n_per_group,ratio=allocation_ratio,
        alpha=alpha,power=target_power,alternative="two-sided",
    )
    return {
        "test":"Independent two-sample t power sensitivity",
        "n_per_group":n_per_group,"allocation_ratio":float(allocation_ratio),
        "target_power":float(target_power),"alpha":float(alpha),
        "minimum_detectable_cohen_d":float(effect),
        "interpretation_ar":(
            "تحليل حساسية القدرة: أصغر حجم أثر مطلق Cohen d تكفي له "
            "الأحجام المفترضة لبلوغ قدرة اختبار محددة وفق نموذج t المستقل "
            "ثنائي الطرف ومتساوي التباين. ليس قدرة لاحقة مشتقة من قيمة p."
        ),
        "interpretation_en":(
            "Prospective sensitivity inversion of the noncentral two-sided "
            "independent equal-variance t-test to find minimum detectable "
            "absolute Cohen d. Not observed post-hoc power."
        ),
    }
