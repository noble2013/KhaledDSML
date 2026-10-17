"""Conjugate Bayesian analyses for independent Bernoulli/binomial outcomes.

Every posterior statement is *conditional on the chosen Beta prior and the
binomial sampling model*. Credible intervals are not frequentist confidence
intervals, and Bayes factors compare precisely specified prior models.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import integrate, special, stats


def _count(value, label, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer, not a boolean")
    value = int(value)
    if value < (1 if positive else 0) or value > 100000:
        raise ValueError(f"{label} must lie within supported counting bounds")
    return value


def _prior(prior_alpha, prior_beta, credible_level):
    for name, x in (("prior_alpha", prior_alpha), ("prior_beta", prior_beta)):
        if isinstance(x, (bool, np.bool_)) or not np.isscalar(x):
            raise ValueError(f"{name} must be a strictly positive finite number")
        try:
            valid = np.isfinite(x) and x > 0
        except TypeError as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not valid:
            raise ValueError(f"{name} must be a strictly positive finite number")
    if (isinstance(credible_level, (bool, np.bool_)) or not np.isscalar(credible_level)
        or not np.isfinite(credible_level) or not 0 < credible_level < 1):
        raise ValueError("credible_level must lie strictly between 0 and 1")
    return float(prior_alpha), float(prior_beta)


def beta_binomial_posterior(
    successes, trials, *, prior_alpha=1.0, prior_beta=1.0,
    credible_level=0.95, predictive_trials=10,
):
    """Closed-form Beta-binomial posterior and exact predictive distribution.

    Assumptions: independent Bernoulli trials and homogeneous event probability
    conditional on p, with p~Beta(a,b). Posterior p|data~Beta(a+s,b+n-s).
    The predictive distribution for new trials is Beta-binomial, *not* the
    plug-in binomial using the posterior mean.
    """
    s = _count(successes, "successes")
    n = _count(trials, "trials", positive=True)
    k = _count(predictive_trials, "predictive_trials")
    if s > n or k > 2000:
        raise ValueError("Need 0 <= successes <= trials and predictive_trials <= 2000")
    a, b = _prior(prior_alpha, prior_beta, credible_level)
    pa, pb = a + s, b + n - s
    tail = (1.0 - credible_level) / 2.0
    low, high = stats.beta.ppf([tail, 1.0 - tail], pa, pb)
    counts = np.arange(k + 1)
    probs = stats.betabinom.pmf(counts, k, pa, pb)
    if not np.isfinite(probs).all() or abs(probs.sum() - 1.0) > 1e-9:
        raise ValueError("Predictive probabilities did not normalize")
    return {
        "test": "Bayesian Beta-binomial posterior",
        "successes": s, "trials": n,
        "prior_alpha": a, "prior_beta": b,
        "posterior_alpha": pa, "posterior_beta": pb,
        "posterior_mean": float(pa / (pa + pb)),
        "posterior_variance": float(pa * pb / ((pa + pb) ** 2 * (pa + pb + 1))),
        "posterior_median": float(stats.beta.median(pa, pb)),
        "credible_level": float(credible_level),
        "credible_lower": float(low), "credible_upper": float(high),
        "predictive_trials": k,
        "predictive_mean_successes": float(k * pa / (pa + pb)),
        "table": pd.DataFrame({
            "future_successes": counts, "posterior_predictive_probability": probs
        }),
        "interpretation_ar": (
            "توزيع لاحق Beta واحتمالات تنبؤية Beta-binomial تحت استقلالية "
            "التجارب وثبات معدل الحدث وافتراضات التوزيع القبلي المُعلَن. "
            "فاصل المصداقية بايزي وليس فاصل ثقة تكراريًا؛ لا تُستنتج السببية."
        ),
        "interpretation_en": (
            "Exact conjugate Beta posterior and Beta-binomial predictive "
            "probabilities conditional on the declared prior, independent "
            "trials, and a common Bernoulli rate. A credible interval is "
            "not a frequentist confidence interval and is not causal evidence."
        ),
    }


def beta_binomial_group_comparison(
    successes_a, trials_a, successes_b, trials_b, *,
    prior_alpha=1.0, prior_beta=1.0, credible_level=0.95,
):
    """Bayesian comparison of two independent binomial groups.

    H0: groups have ONE shared rate with a single Beta(a,b) prior.
    H1: groups have TWO independent rates, each Beta(a,b).
    BF10 is the marginal-likelihood ratio H1/H0. Its value is
    prior-sensitive and is not a posterior probability of H1.
    Under H1, Pr(rate_A > rate_B | data, H1) is obtained by quadrature.
    This is not appropriate for paired outcomes, clustered trials or
    uncontrolled group comparisons presented as causal evidence.
    """
    sa = _count(successes_a, "successes_a")
    na = _count(trials_a, "trials_a", positive=True)
    sb = _count(successes_b, "successes_b")
    nb = _count(trials_b, "trials_b", positive=True)
    if sa > na or sb > nb:
        raise ValueError("Successes may not exceed trials")
    a, b = _prior(prior_alpha, prior_beta, credible_level)
    a1, b1 = a + sa, b + na - sa
    a2, b2 = a + sb, b + nb - sb
    prior_log_beta = special.betaln(a, b)
    log_bf10 = float(
        special.betaln(a1, b1) + special.betaln(a2, b2)
        - special.betaln(a + sa + sb, b + na + nb - sa - sb)
        - prior_log_beta
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", integrate.IntegrationWarning)
        try:
            probability, error = integrate.quad(
                lambda x: stats.beta.pdf(x, a1, b1) * stats.beta.cdf(x, a2, b2),
                0.0, 1.0, epsabs=1e-10, epsrel=1e-10, limit=350
            )
        except integrate.IntegrationWarning as exc:
            raise ValueError("Posterior quadrature did not converge reliably") from exc
    if (not np.isfinite(probability) or not np.isfinite(error) or error > 1e-7
        or probability < -1e-8 or probability > 1 + 1e-8):
        raise ValueError("Posterior quadrature inaccurate; review extreme counts")
    interval_a = stats.beta.ppf([(1-credible_level)/2, (1+credible_level)/2], a1, b1)
    interval_b = stats.beta.ppf([(1-credible_level)/2, (1+credible_level)/2], a2, b2)
    return {
        "test": "Bayesian independent-binomial model comparison",
        "successes_a": sa, "trials_a": na, "successes_b": sb, "trials_b": nb,
        "prior_alpha": a, "prior_beta": b,
        "posterior_mean_a": float(a1/(a1+b1)),
        "posterior_mean_b": float(a2/(a2+b2)),
        "probability_a_greater_b_given_h1": float(np.clip(probability, 0, 1)),
        "integration_absolute_error": float(error),
        "log_bayes_factor_10": log_bf10,
        "bayes_factor_10": float(np.exp(log_bf10)) if log_bf10 < 700 else None,
        "credible_level": float(credible_level),
        "table": pd.DataFrame({
            "group": ["A", "B"],
            "posterior_mean": [a1/(a1+b1), a2/(a2+b2)],
            "credible_lower": [float(interval_a[0]), float(interval_b[0])],
            "credible_upper": [float(interval_a[1]), float(interval_b[1])],
        }),
        "model_h0_en": "One common Bernoulli rate p with Beta(a,b) prior.",
        "model_h1_en": "Independent rates p_A and p_B, each with Beta(a,b) prior.",
        "interpretation_ar": (
            "احتمال تفوق معدل المجموعة أ محسوب شرطيًا على نموذج المعدلين المنفصلين. "
            "عامل بايز BF10 يقارن نموذج معدلين مستقلين مع معدل مشترك، ويتغير "
            "بتغير التوزيع القبلي. ليس قيمة p أو احتمالًا لاحقًا لصحة النموذج، "
            "ولا يصلح تلقائيًا للبيانات المزدوجة أو الادعاءات السببية."
        ),
        "interpretation_en": (
            "The probability that rate A exceeds B is conditional on the "
            "two-rate model. BF10 compares distinct-rate versus shared-rate "
            "models under the specified Beta priors; it is prior-sensitive "
            "and neither a p-value nor the posterior probability of H1. "
            "Independent unclustered trials are required, not paired data."
        ),
    }
