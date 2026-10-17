# KhaledDSML 0.10 — Integrated scientific gapfill | الدفعة العلمية الشاملة

**Status / الحالة:** `0.10.0.dev0` in source development, NOT yet an official PyPI release. Last officially published distribution: **0.9.0**. This work is proposed in `feature/v0.10-final-scientific-gapfill`.

## Research scope | النطاق العلمي

This integrated batch introduces **21 new public functions** (some functions map several original catalogue rows). They have Arabic RTL / English LTR interpretations and work with the offline Analytics Studio, subject to CI and independent review. They belong to five thematic groups:

### 1. Experiments / التجارب وقرار الأعمال
- `bernoulli_sprt` — prespecified, independent Bernoulli sequential probability ratio test for simple hypotheses `p0 < p1`, with noncrossing reported as inconclusive. No p-value is fabricated; nominal Wald error targets are approximations.
- `cuped_randomized_effect` — pre-experiment covariate adjustment, adjusted treatment mean difference and *approximate* Welch inference; the nuisance coefficient comes from pooled outcome data, so confirmatory analysis may require cross-fitting or regression-design review. Causal interpretation **requires valid randomization**.
- `shard_distribution_consistency` — numeric shard vs disjoint remainder KS scores with Holm correction across shards, requiring **independent rows and mutually disjoint shards**.
- `practical_proportion_difference` — ordinary two-sample large-sample z-test plus prespecified practical effect threshold. Practical threshold on point estimate is **not equivalence/noninferiority proof**.

### 2. Regression inference / الاستدلال في الانحدار
- `nested_ols_wald_lr_score` — classical Gaussian iid OLS added-regressor **joint Wald**, likelihood ratio, restricted Score/LM approximations and the exact nested-model F. These three chi-square p-values are **asymptotic**, not generic for all fitted regressions or post-selection feature searches.
- `regression_pseudo_r_squared` — likelihood-based McFadden, Cox–Snell, Nagelkerke pseudo-R² for same-sample compatible likelihoods; not ordinary variance R².
- `cohen_f_squared_nested_models` — incremental Cohen f² for preregistered nested linear models.
- `two_sample_power_sensitivity` — prospective minimum detectable Cohen d for equal-variance independent-sample t based on noncentral t; **not observed power** computed from the resulting p-value.
- `box_tidwell_logistic_screen` — one-positive-predictor logistic Box–Tidwell interaction screen; separation and misspecification invalidate the Wald approximation.

### 3. ML evaluation / تقييم النماذج
- `leave_one_out_cv_evaluation` — one iid observation held out per fold; single-sample R² and AUC are deliberately rejected.
- `hopkins_cluster_tendency` — exploratory Hopkins statistic using an axis-aligned uniform artificial reference and KNN; strongly sensitive to scaling and dimension.
- `holdout_noise_robustness` — perturbation of numeric independently held-out features by Gaussian noise without refitting the model.
- `corrected_resampled_t` — Nadeau–Bengio approximate t for paired repeated holdout scores and a supplied, known train/test ratio.
- `five_by_two_cv_comparison` — Dietterich 5×2cv paired t and Alpaydin combined F **only for actual paired 5×2cv fold measurements**. The function cannot reconstruct a valid split design from scores alone.
- `prediction_seed_stability` — independently refit a supplied estimator under multiple specified random seeds and compare held-out predictions. No claims about causal robustness.
- `gap_statistic_kmeans` — KMeans Gap(k) against a uniformly sampled bounding box, with the one-standard-error selection rule; results depend on scales and synthetic reference.

### 4. Data validation / جودة البيانات وتمثيلها
- `sample_population_comparison` — two-sample KS for numeric and chi-square GOF for reference categorical shares, with Holm over columns. Cannot establish population representativeness from a failure to reject.
- `early_late_nonresponse_screen` — Welch early vs late respondent comparison; later respondents are **not nonrespondents**.
- `scientific_report_alerts` — metadata-only warnings such as small n, missing bilingual interpretations, ANCOVA slope interaction flag and uncorrected sphericity. No automatic assumption validation.

### 5. Probabilistic sketches / خوارزميات التخمين التقريبي
- `hyperloglog_cardinality` — deterministic bounded-memory HyperLogLog estimator of distinct token counts, with low-cardinality linear-counting correction and an *approximate nominal* standard error.
- `minhash_lsh_candidates` — stable MinHash signatures with banded locality-sensitive hashing for likely near-duplicate token sets. LSH may miss similar pairs; verify final candidate matches with exact Jaccard.

## Example / مثال شامل

```python
from khaled_dsml import (
    bernoulli_sprt, nested_ols_wald_lr_score,
    hyperloglog_cardinality, scientific_report_alerts,
    display_analysis_dashboard,
)
import numpy as np

rng = np.random.default_rng(130)
sprt = bernoulli_sprt([1,1,0,1,1,1,0,1,1,1],p0=.3,p1=.7)
display_analysis_dashboard(sprt,theme="amethyst")

x = rng.normal(size=80)
z = rng.normal(size=80)
y = 1 + .5*x + .6*z + rng.normal(size=80)
restricted = np.column_stack([np.ones(80),x])
unrestricted = np.column_stack([np.ones(80),x,z])
ols = nested_ols_wald_lr_score(y,restricted,unrestricted)
display_analysis_dashboard(ols,theme="ocean")

sketch = hyperloglog_cardinality(list(range(500))*3)
display_analysis_dashboard(sketch,theme="aurora")

alerts = scientific_report_alerts({
    "test":"Exploratory analysis",
    "n_samples":12,
    "interpretation_ar":"نتيجة استكشافية",
    "interpretation_en":"Exploratory result",
})
display_analysis_dashboard(alerts,theme="amethyst")
```

## Verification | المرجعيات والتأكد العلمي

`tests/test_final_scientific_gapfill.py` checks SciPy independent SPRT/KS/t/OLS references, CUPED adjustment against Welch, corrected multiple-testing adjustments, sklearn leave-one-out/Hopkins/noise behavior, fixed-seed deterministic sketches and arithmetic counterexamples. `tests/test_resampling_and_model_completion.py` verifies independent Nadeau–Bengio / Dietterich / Alpaydin formulas, logistic Box–Tidwell against statsmodels GLM, and deterministic KMeans inertia and seed metrics. Final CI must test Python 3.10–3.13, genuine Dask/PySpark integration, wheel/sdist build/install.

## Catalogue evidence versus completion | كتالوج 256 بندًا

The original 256 catalogue items remain intact. **237 rows are now linked to candidate callable source-level evidence and 19 have no mapped public callable.** These are NOT counts of full scientific method coverage or approval. Some mapped rows correspond only to explicitly restricted model families or exploratory screens.

**Remaining unlinked IDs requiring specialist review:** #66 polychoric, #68 dependent-correlation Steiger, #74 Conover, #83 generic small-sample Bayes factors beyond the documented Beta-binomial case, #85 post-hoc power (observed power computed from a p-value is not a reliable extra inferential finding), #92 generic effect-size r, #117 Vuong inflated-zero comparisons, #123 CFI/TLI/RMSEA/SRMR, #124 measurement invariance, #188 SHAP stability, #198 Phillips–Perron, #208 PELT, #217 ADWIN, #218 DDM, #229 t-digest, #240 GRIMMER, #241 SPRITE, #243 p-curve, #255 DOCX/Excel export.

These should **not** be silently filled with placeholder p-values, inaccurate names, or thin renaming of unrelated tests. They need specialized implementations, genuinely installed optional dependencies, reference publications and independently reproducible checks.

**Editorial policy / سياسة الاعتماد:** Passing CI, importing an API or finding its name in a catalogue does not constitute scientific certification. The v0.10 stable release should not be advertised as "all 256 scientifically complete".
