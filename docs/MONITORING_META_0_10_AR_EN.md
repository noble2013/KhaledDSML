# KhaledDSML 0.10 — Data monitoring and meta-analysis | مراقبة البيانات والتحليل البعدي

**Status:** development branch `feature/v0.10-research-gapfill`. Official PyPI release remains `0.9.0`. All examples below use synthetic data.

## 1. Data drift | انجراف البيانات

`distribution_drift_profile(reference, current, bins=10)` computes, from independent complete numeric samples:

- **PSI / Population Stability Index:** `Σ(current_share - reference_share) * ln(current_share / reference_share)`, using strictly positive additive probability smoothing. Quantile-bin cutoffs are estimated **only on the reference** and are made open-ended so out-of-range new observations cannot disappear.
- **Jensen–Shannon divergence** in **bits**: squared Jensen–Shannon distance, between 0 and 1, using those same discretized probabilities. Sensitive to bin boundaries.
- **Wasserstein distance** of the original, unbinned numeric empirical samples, in measurement units. A shift in units changes the score.
- **KS statistic and p-value:** two-sample Kolmogorov–Smirnov test. Its p-value assumes independent representative samples, is not adjusted for multiple scans and does not establish cause.

**العربية:** المؤشرات الثلاثة الأولى وصفية وليست اختبارات دلالة إحصائية. لا توجد عتبة عالمية صالحة لكل التطبيقات لتفسير PSI. قد تتأثر مؤشرات التجزئة بعدد الفئات وتوزيع المرجع. لا تُعد KS صالحة تلقائيًا للمراقبة المتكررة على النوافذ المتداخلة.

```python
from khaled_dsml import distribution_drift_profile, display_analysis_dashboard
import numpy as np
rng = np.random.default_rng(12)
reference = rng.normal(100, 15, 1000)
current = rng.normal(107, 15, 900)
result = distribution_drift_profile(reference, current, bins=10)
display_analysis_dashboard(result, theme="ocean")
```

## 2. Welford aggregation | المتوسط والتباين المتدفقان

`welford_streaming_summary(chunks)` merges per-chunk means and second central moments without materializing the complete dataset. Returns sample variance (ddof=1) and population variance (ddof=0). It does not silently drop nonfinite data.

**العربية:** تتحقق ميزة المعالجة المتدفقة عند استعمال مولد يقرأ أجزاء صغيرة من الملف. يتطلب التعامل مع البيانات المفقودة سياسة صريحة قبل الاستدعاء.

## 3. Sample Ratio Mismatch | انحراف نسب التخصيص التجريبي

`sample_ratio_mismatch(observed_counts, expected_proportions)` reports a chi-square statistic and p-value for pre-registered, mutually exclusive, independent experimental arms. All expected counts must be >=5. Repeated unadjusted sequential looks can invalidate nominal type-I error.

```python
from khaled_dsml import sample_ratio_mismatch
srm = sample_ratio_mismatch([560, 440], [0.5, 0.5])
```

## 4. Meta-analysis | التحليل البعدي للدراسات

These functions assume **comparable study effect scales**, **accurate sampling standard errors**, and **independent studies**, not repeated effect sizes from one study.

- `meta_analysis_heterogeneity(y,se)` computes inverse-variance fixed-effect pooling, Cochran Q chi-square approximation, I², and the DerSimonian–Laird moment estimate `tau²=max(0,(Q-df)/(Σw - Σw²/Σw))`. Q has low power with few studies.
- `meta_analysis_random_effects(y,se)` uses weights `1/(se²+tau²)` and **normal-z** confidence intervals. These are *not* Hartung–Knapp intervals and may undercover with few studies, particularly under uncertainty in tau².
- `egger_small_study_effects_test(y,se)` uses unweighted OLS regression of standardized effects `y/se` on study precision `1/se`, with a t-test on the intercept and at least 10 studies. A nonzero intercept signals *small-study effects*, **not proven publication bias**.

**العربية:** هذه الأدوات لا تثبت سببية النتائج، ولا تصح للبيانات المزدوجة أو مخرجات دراسات غير مستقلة. يجب توثيق اختيار نوع التأثير، وطريقة استخلاص الخطأ المعياري، وتصحيح الاعتماد بين الدراسات عند الحاجة. يشترط انحدار إيغر 10 دراسات على الأقل لأغراض الاستعمال الاستكشافي، ولا يكفي وحده لاستنتاج تحيز النشر.

```python
from khaled_dsml import (
    meta_analysis_heterogeneity, meta_analysis_random_effects,
    egger_small_study_effects_test, display_analysis_dashboard,
)
effects = [.04, .27, .42, .14, -.09, .63, .32, .19, .47, -.13, .35, .18]
ses = [.09, .12, .10, .08, .11, .09, .14, .13, .10, .12, .09, .11]
display_analysis_dashboard(meta_analysis_random_effects(effects, ses), theme="amethyst")
```

## Scientific verification | منهجية التحقق

References in `tests/test_monitoring_advanced.py` include `scipy.stats.ks_2samp`, `scipy.stats.wasserstein_distance`, `scipy.spatial.distance.jensenshannon`, `scipy.stats.chisquare`, and independent PSI and streaming variance formulas. `tests/test_meta_analysis.py` includes manually computed Cochran Q/I²/DL weights, `statsmodels.stats.meta_analysis.combine_effects`, and independent `statsmodels.api.OLS` Egger intercept tests. Negative inputs, missing values and RTL/LTR report rendering are tested.

**Catalogue accounting:** newly traced candidate methods connect to the original rows #214, #215, #216, #224, #228, #244, #245 and #246. Row #213 (χ²) remains a broader composite item: SRM's chi-square is not evidence of general-purpose big-data chi-square functionality. No 256-row sign-off is implied.
