# KhaledDSML 0.10 — Advanced SEM, SHAP stability, t-digest and Bayesian evidence
## دليل المؤشرات البنائية وثبات التفسير والتقدير التقريبي وقيود SPRITE

**Development only:** `0.10.0.dev0` in `feature/v0.10-sem-shap-tdigest`; the official PyPI distribution is still **0.9.0**. This PR should be merged only after Python 3.10–3.13, genuine Dask/PySpark, wheel/sdist and independent reference verification.

### 1. SEM fit indices | مؤشرات ملاءمة النماذج البنائية

`sem_fit_indices(chi_square, degrees_of_freedom, baseline_chi_square, baseline_degrees_of_freedom, n_samples, observed_covariance=None, implied_covariance=None)` calculates **CFI, TLI, RMSEA** and optionally **SRMR** from conventional unscaled covariance-based *previously fitted* SEM/CFA results.

Formulas:
- `CFI = 1 - max(chi² - df,0)/max(chi² - df, chi0² - df0, 0)`;
- `TLI = (chi0²/df0 - chi²/df)/(chi0²/df0 - 1)`;
- `RMSEA = sqrt(max((chi²-df)/(df*(n-1)),0))`;
- Optional `SRMR`: root average squared standardized covariance residual across upper-triangle INCLUDING diagonal cells, with scaling `(S - Sigma)/sqrt(diag(S)_i diag(S)_j)`. **Other SEM packages use different conventions.**

**العربية:** تتطلب الدالة مخرجات صالحة للنموذج الأصلي والاستقلالي من العينة نفسها ومقدّر ملائم. لا تُقدّر نموذج SEM داخليًا ولا تُدقق صلاحية عامل كامن، ولا تمزج قيم كاي تربيع المصححة robust بالإحصاءات التقليدية. لا تُستخرج السببية من مؤشرات المطابقة.

### 2. Nested measurement invariance | فحص تكافؤ القياس

`measurement_invariance_nested_ml(less_restricted_chi_square,less_restricted_df,more_restricted_chi_square,more_restricted_df,level="metric")` returns ordinary chi-square difference `Δχ² = χ²_more - χ²_less`, `Δdf = df_more - df_less` and the chi-square tail probability for **genuinely nested** measurement invariance CFA models.

**العربية:** هذا فحص لمخرجات نماذج ML مقدرة مسبقًا، لا يقوم بملاءمة التكافؤ المتري أو القياسي، ولا يُستخدم تلقائيًا لنماذج WLSMV أو التصحيحات المتينة أو مجموعات تعتمد مشاهداتها بعضها على بعض. عدم الرفض لا يثبت التكافؤ.

### 3. SHAP stability | ثبات تفسير SHAP

`shap_explanation_stability(shap_runs,feature_names=...)` accepts a three-dimensional tensor `(repeated_runs, same_holdout_rows, same_features)`. It compares **mean absolute precomputed SHAP** importances among repeated model runs, with pairwise Spearman rank correlations and feature-wise across-run coefficient of variation.

**العربية:** يجب الحفاظ على الصفوف نفسها، ونظام السمات نفسه، وتعريف المخرجات والخلفية المرجعية نفسها. الدالة لا تحسب SHAP من الصفر ولا تعد بثبات النموذج على بيانات جديدة أو علاقة سببية. خلوّ الصفات من تفاوت الأهمية يفضي إلى ارتباط غير معرف ترفضه الدالة.

### 4. t-digest | الكميات المئوية التقريبية

`tdigest_quantile_summary(values,probabilities=(.05,.25,.5,.75,.95),compression=.01,weights=None)` delegates to the independent `tdigest` Python package, offering approximate quantiles from a centroid sketch. Install `khaled-dsml[quantiles]`. Probabilities are fractions from 0 to 1; third-party backend percentiles expect 0 to 100. Optional weights are **positive integer frequencies**, not survey sampling weights. Reference API: <https://github.com/CamDavidsonPilon/tdigest>.

**العربية:** لا تُعدّ الكميات المئوية مضمونة الدقة، ولا يضمن ضبط الضغط هوامش خطأ احتمالية مناسبة لكل توزيع. يُقارن التقدير في الاختبارات مع تنفيذ المكتبة الأصلية والقيم الدقيقة للعينة ضمن هامش معلوم.

### 5. Exact categorical small-sample Bayes factor | عامل بايز للفئات

`dirichlet_multinomial_bayes_factor(counts,null_probabilities,prior_concentration)` evaluates an analytical BF10 of `H0: p=p0` against `H1: p~Dirichlet(alpha)`, via log-gamma evidence. The multinomial coefficient cancels from the ratio; a separate exact SciPy `dirichlet_multinomial.pmf / multinomial.pmf` comparison validates the formula.

**العربية:** الفحص مناسب فقط لفئات متعددة مع أعداد مشاهدات صحيحة وتركيز قبلي محدد؛ ليس اختبارات بايز العامة للمتوسطات وANOVA. قيمة BF لا تمثل احتمال صحة النموذج البديل دون احتمالات قبلية للنماذج.

### 6. SPRITE-inspired synthetic witness | بناء مجموعة درجات ممكنة

`sprite_integer_constraint_witness(reported_mean,reported_sd,n_samples,...,fixed_counts=None)` searches **finite integer histograms** over at most six score categories and `n<=18`. It produces one *synthetic possible histogram* that matches the declared ROUND_HALF_UP mean and sample SD and any fixed category frequencies. This is **NOT the full stochastic SPRITE protocol**, does not reconstruct the real source dataset, and is not evidence of data manipulation.

**العربية:** الغرض تعليم كيفية اختبار إمكان وجود بيانات مطابقة لملخص إحصائي وفق شروط صريحة. لا يجوز تفسير العينة التي أنتجتها الدالة على أنها البيانات الخام المفقودة للبحث.

### Runnable example | مثال تطبيقي

```python
import numpy as np
from khaled_dsml import (
    sem_fit_indices, measurement_invariance_nested_ml,
    shap_explanation_stability, tdigest_quantile_summary,
    dirichlet_multinomial_bayes_factor,
    sprite_integer_constraint_witness, display_analysis_dashboard,
)
rng=np.random.default_rng(83)
display_analysis_dashboard(sem_fit_indices(60,40,250,60,180),theme="amethyst")
display_analysis_dashboard(measurement_invariance_nested_ml(110,90,119,94),theme="ocean")
shap_runs=np.stack([rng.normal(size=(25,4))*[.1,.3,.6,1.] for _ in range(3)])
display_analysis_dashboard(shap_explanation_stability(shap_runs),theme="aurora")
display_analysis_dashboard(tdigest_quantile_summary(rng.normal(size=800)),theme="ocean")
display_analysis_dashboard(dirichlet_multinomial_bayes_factor([4,2,1],[.3,.3,.4]),theme="amethyst")
display_analysis_dashboard(sprite_integer_constraint_witness("2.00","1.41",2,
  minimum_score=1,maximum_score=3),theme="aurora")
```

### Science and catalogue audit | ما بقي من الكتالوج

**254/256 original catalogue rows have source-level public API candidates.** Many candidates implement only the explicitly documented restricted method; this is NOT 254 scientifically signed-off fully implemented studies.

The only two unmapped rows remain:
- **#85 Observed post-hoc power** from a realized effect/p-value. This is mathematically dependent on the observed statistic and commonly adds no independent inferential evidence; prospective power and sensitivity calculations already exist.
- **#117 Zero-inflated Vuong test.** The ordinary non-nested Vuong normal reference is **invalid as a general test of zero inflation** when comparing Poisson with zero-inflated Poisson; see Wilson (2015), *Economics Letters*, DOI 10.1016/j.econlet.2014.12.029 and <https://pmc.ncbi.nlm.nih.gov/articles/PMC6345607/>. A scientifically justified boundary-aware or tailored score/bootstrap test is necessary instead; do not generate a misleading Vuong p-value.

**Verification:** tests check formula identities, SciPy nested chi-square and exact Dirichlet evidence, per-pair SHAP Spearman reference, direct third-party TDigest algorithm output, and brute-force integer-response moment reconstruction. Optional `tdigest` is installed during full CI.

**Release:** source remains `0.10.0.dev0`; do not announce an official published stable 0.10.0 until the separate release gate and domain expert signoff.
