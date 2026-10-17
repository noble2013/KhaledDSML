# تدقيق كتالوج KhaledDSML (256 بندًا) | Catalogue coverage audit

**Baseline / نقطة الانطلاق:** `main` release `0.9.0`, snapshot 17 October 2026.
**Branch / فرع العمل:** `feature/v0.10-catalog-henze-zirkler` (`0.10.0.dev0`, unpublished).
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

A machine-assisted **source-trace of all 256 rows** is preserved in [CATALOG_256_SOURCE_TRACE_V010.tsv](CATALOG_256_SOURCE_TRACE_V010.tsv). It conservatively identifies **64 rows with exact public API-name candidates** from their tracker notes (plus explicit v0.9/v0.10 differences); **192 rows have no exact match recorded**. **These figures are not completion/absence totals:** the original notes are stale and a compound row may contain both implemented and missing submethods. The TSV preserves each original tracker line verbatim for subsequent independent scientific sign-off.

أعددنا مطابقة نصية لجميع البنود الـ256: **64 بندًا لها مرشحات أسماء دوال عامة**، و**192 بندًا لا تحمل تطابق اسمًا صريحًا في التعليقات القديمة**؛ وهذا ليس حكمًا بأنها غير منفذة ولا نسبة إنجاز. يجب ربط كل بند بالأدلة وباختبارات مرجعية قبل اعتماد حالته.

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
