# KhaledDSML 0.10 — Final source-trace closure with scientific guardrails
## إغلاق فجوات الكتالوج دون ادعاءات استدلالية غير صحيحة

**Development version:** `0.10.0.dev0` (GitHub, not published to PyPI). **Published stable:** `0.9.0`.

> The 256 original catalogue rows now have at least one linked PUBLIC API *candidate*. This means **source-trace coverage** only; it does NOT mean 256 methods are completely implemented, statistically equivalent to the original named methods, validated for every sampling design, or professionally certified. The final two rows are explicitly **limited replacements / constrained use cases**.

## 1 — Zero inflation: do NOT misuse Vuong | التضخم الصفري

Wilson (2015) demonstrated why the classical nonnested Vuong z-test is not a valid direct zero-inflation test for Poisson against ZIP: Poisson is a boundary/overlap case within ZIP. **No classical Vuong p-value is supplied for that comparison.**

### Safe diagnostic: van den Broek (1995)

`poisson_excess_zero_score(counts, alpha=.05)` tests whether an **iid intercept-only Poisson count distribution** displays unexpectedly many zeros. It requires at least 50 independent nonnegative integer observations, a positive estimated mean and sufficient expected zeros and nonzeros. It uses the efficient-variance score:

```
lambda_hat = mean(counts)
p0 = exp(-lambda_hat)
z = (n_zeros - n*p0) / sqrt(n*p0*(1-p0) - n*lambda_hat*p0**2)
p_one_sided_excess = 1 - Phi(z)
chi_square_2sided = z**2 ; p_2sided = Pr(ChiSquare(df=1) >= chi_square_2sided)
```

The **directional upper-tail p-value** is the public `p_value`; the ordinary two-sided `chi²` p-value is separately labeled. A negative z is a **deficit of zeros**, not positive evidence for zero inflation. This test **cannot distinguish structural zero inflation from omitted heterogeneity, a negative-binomial alternative, serial dependence or overdispersion**. It does not fit a ZIP count regression with covariates.

Reference: van den Broek, J. (1995). *A score test for zero inflation in a Poisson distribution*. *Biometrics*, 51, 738–743. DOI [10.2307/2532959](https://doi.org/10.2307/2532959). Independent R reference: [vcdExtra zero.test](https://search.r-project.org/CRAN/refmans/vcdExtra/html/zero.test.html).

### Proper Vuong, genuinely nonnested models ONLY

`vuong_strictly_nonnested(loglik_model_a, loglik_model_b, model_a_family="...", model_b_family="...", relationship="strictly_nonnested")` computes the standard asymptotic two-sided z-test from same-row **individual log-likelihood contributions**:

```
d_i = log f_A(y_i) - log f_B(y_i)
z = sqrt(n) * mean(d_i) / sample_sd(d_i)
p_two_sided = 2 * (1 - Phi(abs(z)))
```

Researchers must independently establish genuinely **strictly nonnested, distinguishable** models, iid observations, comparable likelihood scale and prespecified model-selection protocol. A declaration cannot verify nonnestedness; the function **explicitly rejects Poisson and ZIP labels used together** and zero-variance paired differences. Not a valid nested Poisson-vs-ZIP test, not corrected for model search or overlapping hypotheses.

Reference: Wilson, P. (2015). *The misuse of the Vuong test for non-nested models to test for zero-inflation*. *Economics Letters*, 127, 51–53. DOI [10.1016/j.econlet.2014.12.029](https://doi.org/10.1016/j.econlet.2014.12.029).

## 2 — Post-study reporting without observed-power fallacy | ما بعد الدراسة

Calculating achieved/observed power by inserting the study's **observed effect size or p-value** into a power function is an analytically redundant transformation of the same finding, not independent evidence that a real effect does/does not exist. We intentionally DO NOT expose that value.

`post_study_precision_scenarios(estimate, standard_error, externally_specified_effects=[...], alpha=.05)` instead provides:

1. A normal-approximation **effect-size confidence interval** `estimate ± z_(1-alpha/2)*SE`.
2. A sensitivity table of **hypothetical two-sided normal power**, conditional on externally proposed true effects `delta` and **reusing** the reported SE:
   `Pr(|Normal(delta/SE, 1)| > zcrit)`.
3. `observed_power_computed=False`, with Arabic and English cautions.

The `externally_specified_effects` MUST be chosen from scientific goals/prior literature before consulting this sample's effect estimate, not reverse-engineered from the resulting p-value. The retrospective SE is a **rough planning surrogate** and must be reestimated for a future study's sample size and design. These scenarios provide no extra inference about the finished experiment. The function does not compute p-values.

Reference: Hoenig & Heisey (2001). *The Abuse of Power: The Pervasive Fallacy of Power Calculations for Data Analysis*. *The American Statistician*, 55, 19–24. DOI [10.1198/000313001300339897](https://doi.org/10.1198/000313001300339897).

## 3 — Reproducible bilingual examples | أمثلة عربية وإنجليزية

```python
import numpy as np
from scipy import stats
from khaled_dsml import (
    poisson_excess_zero_score, vuong_strictly_nonnested,
    post_study_precision_scenarios, display_analysis_dashboard,
)
rng=np.random.default_rng(2026)

# Intercept-only iid Poisson zero-excess screen
counts=np.where(rng.random(700)<.30,0,rng.poisson(1.8,size=700))
display_analysis_dashboard(poisson_excess_zero_score(counts),theme="ocean")

# Distinct strictly nonnested *lognormal vs exponential* density families
sample=rng.lognormal(.2,.55,300)
a=stats.lognorm.logpdf(sample,s=.55,scale=np.exp(.2))
b=stats.expon.logpdf(sample,scale=sample.mean())
result=vuong_strictly_nonnested(
    a,b,model_a_family="lognormal",model_b_family="exponential",
    relationship="strictly_nonnested")
display_analysis_dashboard(result,theme="amethyst")

# Use externally defined scientifically interesting true-effect scenarios
precision=post_study_precision_scenarios(
    estimate=.25,standard_error=.10,
    externally_specified_effects=[0.,.1,.3,.5])
display_analysis_dashboard(precision,theme="aurora")
```

## 4 — Evidence and completeness | تحقق منهجي لا مجرد تغطية

**Testing file:** `tests/test_final_two_scientific_guardrails.py` cross-checks van den Broek efficient-variance formula with the independent R reference, SciPy normal/chi-square distributions and sign-sensitive synthetic count data. It compares Vuong to separately calculated per-case lognormal versus exponential log-densities and guards the invalid ZIP/Poisson comparison. Hypothetical sensitivity probabilities are independently tested against SciPy `norm.sf/cdf` and are unchanged when the **observed estimate changes** while externally specified effects and study precision remain fixed.

**Original catalogue row #85:** linked to `post_study_precision_scenarios` as a **scientifically safer alternative**, NOT equivalent implementation of observed-power-from-observed-effect. **Row #117:** linked to a strictly nonnested Vuong method AND a distinct van den Broek Poisson-excess-zero method, NOT an invalid Poisson-vs-ZIP Vuong test.

**256 of 256 = API candidate links with scope notes; 0 guarantee of blanket scientific certification.** Audit notes must retain methodological mismatch flags and future expert validation tasks.

## 5 — PowerShell, build and installation

```powershell
git checkout main
git pull origin main
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,econometrics,exports,unitroot,quantiles]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m build
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps ".\dist\khaled_dsml-0.10.0.dev0-py3-none-any.whl"
.\.venv\Scripts\python.exe -c "import khaled_dsml; print(khaled_dsml.__version__)"
```

**Do not release stable 0.10 to PyPI based only on candidate mappings or CI pass**; distribution publishing and complete domain-science audit remain separate gates.
