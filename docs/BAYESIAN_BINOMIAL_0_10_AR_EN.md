# Bayesian Beta-binomial laboratory — v0.10.0.dev0
# مختبر التحليل البايزي للبيانات الثنائية

**Status | الحالة:** development only — not published on PyPI; official stable release remains 0.9.0.

## 1. Conjugate Beta-binomial inference | الاستدلال البايزي المترافق

**English:** For independent Bernoulli outcomes with a homogeneous event rate \(p\), choose \(p\sim\mathrm{Beta}(a,b)\). After observing \(s\) successes in \(n\) trials, the posterior is \(\mathrm{Beta}(a+s,b+n-s)\), so the posterior mean is \((a+s)/(a+b+n)\). Equal-tailed Bayesian credible intervals are exact Beta quantiles **conditional on the prior and model**, not frequentist confidence intervals. For \(m\) new trials, the posterior predictive number of successes is Beta-binomial, integrating parameter uncertainty.

**العربية:** عند افتراض تجارب مستقلة بمعدل نجاح ثابت وتوزيع قبلي Beta، يكون التوزيع البعدي Beta مع تحديث معلمتي النجاح والإخفاق. يُحسب فاصل المصداقية من كميات التوزيع البعدي، ولا يجوز وصفه بفاصل ثقة تكراري أو تفسيره على أنه دليل سببي.

```python
from khaled_dsml import beta_binomial_posterior, display_analysis_dashboard

posterior = beta_binomial_posterior(
    successes=18, trials=50, prior_alpha=1, prior_beta=1,
    credible_level=.95, predictive_trials=10,
)
display_analysis_dashboard(posterior, theme="aurora")
```

## 2. Independent group model comparison | مقارنة نموذجين لمجموعتين مستقلتين

**English:** `beta_binomial_group_comparison` compares H0 (one common event probability with Beta(a,b) prior) against H1 (separate probabilities, independently Beta(a,b)). The exact *marginal-likelihood ratio* (Bayes factor BF10) is computed in log space via `scipy.special.betaln`:

\[
\log BF_{10} = \log B(a+s_A,b+n_A-s_A)+\log B(a+s_B,b+n_B-s_B)-
\log B(a+s_A+s_B,b+n_A+n_B-s_A-s_B)-\log B(a,b).
\]

The reported \(P(p_A>p_B\mid D,H_1)\) is evaluated by numerical integration of the product of a Beta density and a Beta CDF. The integration error is reported. The Bayes factor is **not** the posterior probability of H1: model prior odds would be needed for that. It is **sensitive to choice of Beta prior**. Do not use for paired/clustered outcomes.

**العربية:** يُقارن عامل بايز الدليل الهامشي لنموذج المعدلين المستقلين بالدليل لنموذج المعدل المشترك، تحت التوزيع القبلي المعلن. احتمال تفوق إحدى المجموعتين مشروط بصحة النموذج البديل، ولا يساوي احتمال صحة النموذج نفسه. لا يُعد عامل بايز قيمة احتمالية p، ولا يجوز تفسير الفرق سببيًا بغير تصميم تجريبي مناسب.

```python
from khaled_dsml import beta_binomial_group_comparison, display_analysis_dashboard

comparison = beta_binomial_group_comparison(
    successes_a=30, trials_a=60,
    successes_b=19, trials_b=60,
    prior_alpha=1, prior_beta=1
)
display_analysis_dashboard(comparison, theme="amethyst")
```

## 3. Methodology and scientific limitations | الفروض والقيود العلمية

- A pre-declared prior matters; report its parameters in the study.
- Trials must be independent, and observations in a group should share an underlying event rate.
- The credible interval represents posterior probability conditional on prior, model, and observed counts.
- For credible intervals, report the level; for predictive probabilities, report the future trial horizon.
- Group comparison uses H0 one common rate, H1 two independent rates. H1 posterior group superiority does not automatically imply Bayes factor support for H1.
- Extreme counts may challenge numerical quadrature and are rejected if error control fails.
- No fabricated frequentist p-values or confidence intervals.

**التوثيق المرجعي:** Beta–binomial conjugacy, integrated Beta marginal likelihoods, and exact Beta posterior predictive formulas. Implementation references are independently tested against `scipy.stats.beta`, `scipy.stats.betabinom`, and `scipy.special.betaln` and an independent Beta expectation integral in `tests/test_bayesian_binomial.py`.
