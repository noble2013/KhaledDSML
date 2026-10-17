"""Exact small-sample categorical Bayesian point-null comparisons.

This is a narrow Dirichlet-multinomial analytical Bayes factor, not a
universal Bayesian t-test, model-average, optional stopping guarantee, or
causal analysis. A prior must be chosen before examining observed counts.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def dirichlet_multinomial_bayes_factor(counts, null_probabilities, *,
                                       prior_concentration=None):
    """Exact BF10: Dirichlet-multinomial alternative versus multinomial H0.

    Compare H0: fixed categorical event probabilities p0 with H1:
    unknown multinomial proportions p~Dirichlet(alpha). The multinomial
    combinatorial coefficient cancels in the Bayes factor.
    Handles small finite count samples exactly on the log-gamma scale,
    with all categories present in model (zero observed counts allowed).
    """
    try:
        cnt=np.asarray(counts,dtype=float)
        p=np.asarray(null_probabilities,dtype=float)
        alpha=np.ones(len(cnt),dtype=float) if prior_concentration is None else np.asarray(prior_concentration,dtype=float)
    except (TypeError,ValueError) as exc:
        raise ValueError("counts, null probabilities and prior must be numeric vectors") from exc
    if (cnt.ndim!=1 or p.ndim!=1 or alpha.ndim!=1 or
        len(cnt)<2 or len(cnt)>50 or cnt.shape!=p.shape or cnt.shape!=alpha.shape
        or not np.isfinite(cnt).all() or not np.isfinite(p).all()
        or not np.isfinite(alpha).all() or not np.array_equal(cnt,np.floor(cnt))
        or np.any(cnt<0) or cnt.sum()<2 or cnt.sum()>100_000
        or np.any(p<=0) or not np.isclose(p.sum(),1,rtol=1e-9,atol=1e-12)
        or np.any(alpha<=0)):
        raise ValueError("Require >=2 categorical bins with integer counts, proper p0 and positive Dirichlet prior")
    n=int(cnt.sum())
    log_m1=(gammaln(alpha.sum())-gammaln(alpha.sum()+n)
            +np.sum(gammaln(alpha+cnt)-gammaln(alpha)))
    log_m0=float(np.dot(cnt,np.log(p)))
    logbf=float(log_m1-log_m0)
    return {
        "test":"Exact Dirichlet-multinomial point-null Bayes factor",
        "n_samples":n,"n_categories":len(cnt),
        "log_bayes_factor_10":logbf,
        "bayes_factor_10":float(np.exp(logbf)) if logbf<700 else float("inf"),
        "posterior_mean_probabilities":list(map(float,(cnt+alpha)/(n+alpha.sum()))),
        "prior_concentration":list(map(float,alpha)),
        "null_probabilities":list(map(float,p)),
        "counts":list(map(int,cnt)),
        "interpretation_ar":(
            "عامل بايز BF10 دقيق لمقارنة توزيـع احتمالات فئوية ثابتة "
            "بتوزيع ديريشليه بديل محدد مسبقًا. يعتمد على اختيار "
            "التركيز القبلي وفئات العينة، ولا يجوز تعميمه على "
            "كل اختبارات بايز أو اعتباره احتمال صدق الفرضية دون "
            "احتمالات قبلية للنماذج."
        ),
        "interpretation_en":(
            "Exact log-gamma Dirichlet-multinomial Bayes factor for "
            "a fixed categorical probability null against a declared "
            "Dirichlet prior alternative. Highly prior-sensitive in "
            "small samples; BF10 is not a posterior probability of H1 "
            "without model prior odds, nor a universal Bayes test."
        ),
    }
