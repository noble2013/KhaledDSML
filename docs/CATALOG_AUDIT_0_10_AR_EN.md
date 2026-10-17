# تدقيق كتالوج KhaledDSML (256 بندًا) | Catalogue coverage audit

**Baseline / نقطة الانطلاق:** `main` release `0.9.0`, snapshot 17 October 2026.
**Branch / فرع العمل:** source baseline after merged PR #9–14; specialist source methods on `feature/v0.10-specialized-inference-monitoring` (`0.10.0.dev0`, unpublished).
**Provenance / المصدر:** `KhaledDSML_Catalog_Verified_Phase2_2026-10-17.xlsx` saved in the author's Library. The reference workbook, **not** the number of pytest tests, defines the 256 units of work.

## Scope and historical numbers | النطاق والأرقام التاريخية

The reference tracker contains exactly **256 items across 12 domains**. Its historical field values are **42 complete, 34 partial, 180 not started**; those were recorded *before* the latest source-code audit and **must not be represented as the verified coverage of v0.9.0**. For example, its item #2 still says Little's MCAR is absent, while `little_mcar_test` is exported and tested in v0.9.0. The source snapshot therefore contradicts the tracker.

يضم ملف التتبع المرجعي 256 بندًا، لكنه وثيقة تاريخية غير مُزامَنة مع الدوال الأخيرة؛ لا يجوز تحويل 42/34/180 إلى نسبة إنجاز حقيقية دون فحص كل بند مستقلًا.

| Domain | العدد / Items | What still needs review |
|---|---:|---|
| inference | 67 | Mapping each classical/robust/posthoc test to current APIs and independent numerical tests |
| ml_eval | 51 | Metrics, calibration, fairness, clustering validation and split hygiene |
| regression | 27 | Regression, GLM, diagnostics, econometric extensions |
| bigdata | 22 | Spark/Dask functionality, actual distributed-engine integration versus wrapper presence |
| psychometrics | 20 | Reliability, dimensionality, validity and CFA/SEM gaps |
| assumptions | 19 | Normality/independence tests; Henze–Zirkler is newly added on the development branch |
| timeseries | 16 | Unit root, cointegration and forecast comparison checks |
| audit | 11 | Missingness item #2 is stale; item #1 still mixes exact/fuzzy duplicates |
| forensics | 8 | Define legitimate null hypotheses and limitations before claiming coverage |
| dimension | 6 | Dimension reduction and independent reference coverage |
| nlp | 5 | Corpus-level vs individual-item coverage and multilingual behavior |
| report | 4 | Actual HTML/print, RTL/LTR, exports, usability and accessibility |
| **Total** | **256** | **No verified completion claim yet** |

## 256-row trace | جدول مطابقة جميع البنود

A machine-assisted and manually enriched **source-trace of all 256 rows** is preserved in [CATALOG_256_SOURCE_TRACE_V010.tsv](CATALOG_256_SOURCE_TRACE_V010.tsv). It identifies **245 rows with candidate public API names** after source review (initial tracker notes plus 118 manually mapped catalogue rows); **11 rows currently have no public API-name candidate**. **These figures are not completion/absence totals:** the original notes are stale and a compound row may contain both implemented and missing submethods. The TSV preserves each original tracker line verbatim for subsequent independent scientific sign-off. Importability and test-file references can be recomputed with `python tools/catalogue_audit.py --output catalogue_evidence.csv`; this does not assign completion labels.

أعددنا مطابقة نصية لجميع البنود الـ256: **245 بندًا لها مرشحات أسماء دوال عامة مرتبطة**، و**11 بندًا لا يحمل اسم دالة عامة مرشحًا**؛ وهذا ليس حكمًا بأنها غير منفذة ولا نسبة إنجاز. يجب ربط كل بند بالأدلة وباختبارات مرجعية قبل اعتماد حالته.

## Specialized inferential and export mappings / الاستدلال المتخصص والتقارير

New source-level callable mappings for **#66, #68, #74, #92, #208, #217, #218 and #255** introduce ordinal polychoric latent MLE, Williams–Steiger paired overlap, tie-free Conover–Iman, justified z-based r, exact minimum-length PELT SSE segmentation, uncompressed ADWIN-style bounded monitoring, DDM binary error monitoring, and real editable Word/Excel bilingual exports. **245 of 256 rows have callable candidates; 11 remain without matching public functions.** Do not count these as scientific signoffs, especially where methods are limited to a defined statistical design.

## Integrated research gapfill | إكمال موحد قابل للتدقيق

This development batch links **25 additional catalogue rows** (new 16 initial APIs plus 5 further resampling/model-stability methods) to source candidates, resulting in **237 of 256 rows with candidate callable mappings** and **19 without**. Some APIs cover several rows and some catalogue rows describe broader procedures than the new API; this is **source evidence**, not verified scientific completeness.

The remaining 19 unlinked items have exact IDs, limitations and suggested specialist verification requirements in [FINAL_SCIENTIFIC_GAPFILL_0_10_AR_EN.md](FINAL_SCIENTIFIC_GAPFILL_0_10_AR_EN.md). Do not substitute placeholder results for methods requiring specialist SEM software, drift libraries or factorial resampling designs.

## General linear models mapping | ربط النماذج الخطية

IDs **#37 repeated-measures ANOVA**, **#38 ANCOVA**, **#39 one-way MANOVA** now map to new explicit APIs in `general_linear_models.py`, with independent arithmetic checks, a bilingual methodology guide and Colab 18. This is three source-level callable candidates, **not** complete scientific approval or a license to disregard sphericity, parallel slopes, independence, covariance rank or conditional model assumptions.

## Model validation audit and recovered mappings | تدقيق تقييم نماذج التعلّم

Mapping candidates added for #167 nested CV, #177 predictive label permutation significance, #178 the shared predictive Y-randomization scope, #179 Dummy benchmark, #187 heldout permutation importance, and #193 learning curve. Existing implementations reconciled on #26/#200 Ljung–Box, #98 Breusch–Pagan, #125 Cronbach alpha, #196 ADF, #207 regression *residual* CUSUM (not generic stream detection), #227 p-value multiple-testing adjustment (not a comprehensive sequential design). This yields 209 candidate-matched versus 47 unmapped items. All are **source-level candidates**, not counts of complete scientific implementations.

## Integrity and reproducibility research mapping / ربط أدوات نزاهة البحث

Six extra candidate rows (#158, #170, #219, #222, #239, #242) link to strict MAPE, paired McNemar, Page–Hinkley, schema contracts, GRIM arithmetic and numerical t p-value checks. These are *presence and test-evidence mappings only*, not complete scientific acceptance of compound catalogue items. Main source remains v0.10 development, not an official PyPI release.

ستة بنود إضافية ارتبطت بمرشحات دوال مع فحوص مستقلة للمرجع الرياضي والنتائج العربية، دون إعلان اكتمال أي بند مركب دون مراجعة تفاصيله.

## Additional verifiable callable links | ربط مرشحات وظائف جديدة

Eight original rows (#214, #215, #216, #224, #228, #244, #245, #246) now carry candidates from the monitoring and study meta-analysis modules. These are **source mapping candidates**, not definitive scientific completion. A row may require other algorithms or end-to-end verification before it is marked complete. The mapping trace and CLI re-evaluate importability and test references after checkout.

ثمانية بنود أصبحت مرتبطة بدوال مرشحة جديدة، مع بقاء حالة الاعتماد العلمي لكل بند منفصلة عن وجود الدالة أو نجاح اختبار تقني.

## Verified source-level changes relative to the tracker | اختلافات مثبتة بالكود

- **Item #2, audit:** `missing_pattern_report` + `little_mcar_test` exist in v0.9.0, with tests in `tests/test_missing_data_advanced.py`. This establishes presence and some tests, **not** blanket validation of missingness mechanisms.
- **Item #22, assumptions (Henze–Zirkler):** marked “not started” in the reference workbook. Implemented for the first time in `0.10.0.dev0` as `henze_zirkler_normality_test`, with independently computed kernel-formula tests. **Only a development candidate until CI and review pass.**
- **Missingness MNAR:** `mnar_delta_sensitivity` adds explicitly hypothetical delta shifts for completed-data means, not an identified MNAR probability model.
- **Dask / PySpark:** published adapters already exist in `distributed.py` and `distributed_analytics.py`; do not classify this as “not started” merely because old trackers predate v0.7.0.
- **Post-hoc / ROC:** `posthoc_comparisons.py` and `roc_comparison.py` exist since v0.6.0; the tracker must be checked at the level of each test and assumptions.
- **Mediation / moderation:** `mediation_moderation.py` exists since v0.8.0. Association is not causation and is not automatically SEM.
- **CFA / SEM:** no dedicated public implementation was found in the v0.9.0 package inventory; treat as a candidate gap rather than as implemented.
- **Bayesian model comparison:** no public Beta-binomial comparison in v0.9.0. New development functions `beta_binomial_posterior` and `beta_binomial_group_comparison` now provide prior-conditional binomial tools with independently verified SciPy references; **not full coverage of Bayesian model classes or all Bayesian catalogue sub-items**.

## Acceptance protocol / بروتوكول اعتماد بند

For each catalogue row, record: **ID; bilingual method name; fully implemented / partially implemented / missing / requires validation; exact importable function(s); numerical reference; edge-case tests; doc/notebook evidence; Python 3.10–3.13 CI; reviewer decision**. A name match is not enough. Compound catalogue rows must only be “complete” when **all** their components are covered. The audit table stays open until 256 such signed-off rows exist.

لكل بند نتحقق من الدالة الفعلية، وصحة الصيغ، والفروض العلمية، والحالات الحدّية، واختبارات مرجعية مستقلّة، والتوثيق؛ ولا تكفي الإشارة إلى اسم دالة أو عدد الاختبارات الكلي.

**Current audit disposition:** Source inventory and a documented gap review have begun; item-by-item scientific sign-off of the 256 rows is **not complete**. Historical 42/34/180 values are deliberately not updated as definitive v0.9.0 totals.
