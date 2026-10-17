# KhaledDSML 0.10 — Research diagnostics & data integrity
## دليل عربي/إنجليزي لسلامة التحليل وجودة البيانات

**Branch / الفرع:** `feature/v0.10-diagnostics-integrity` • **Status:** development only. No PyPI publication. The stable distribution stays 0.9.0.

## 1 — Paired ML comparisons / اختبار McNemar للمقارنة المزدوجة

`mcnemar_paired_predictions(y_true, predictions_a, predictions_b)` compares the correctness of two models **on the same, ordered, independently sampled held-out cases**. The null is **equal probability of each discordant direction** (A correct/B wrong versus A wrong/B correct), NOT equivalence of classifiers. By default the exact two-sided `scipy.stats.binomtest` is used for the discordant counts. Optional chi-square mode uses `(|b-c|-1)²/(b+c)` with a nonnegative correction, or the uncorrected statistic.

**عربي:** يجب أن تكون التنبؤات من نموذجين على الأشخاص أنفسهم، دون تدريب على مجموعة الاختبار. لا يجوز تطبيق الاختبار على تقييمين غير مزدوجين، ولا تجاهل الاعتماد بين الحالات أو تكرار المقارنات.

```python
from khaled_dsml import mcnemar_paired_predictions
truth = [1, 0, 1, 0, 1, 0]
model_a = [1, 0, 1, 1, 1, 1]
model_b = [0, 0, 1, 0, 0, 0]
result = mcnemar_paired_predictions(truth, model_a, model_b)
```

## 2 — Strict MAPE / حساب MAPE بضوابط واضحة

`mean_absolute_percentage_error_strict(actual,predicted)` computes `100 * mean(abs((actual-predicted)/actual))`. **Zero actuals are rejected** rather than hidden using epsilon. This metric may still be misleading for actual values near zero or negative: use MAE, RMSE and suitable scales for additional context. It is descriptive, not a hypothesis test.

## 3 — GRIM / الفحص الحسابي للمتوسط

`grim_mean_consistency(reported_mean, sample_size, decimals=2, minimum_score=1, maximum_score=5)` checks whether an integer bounded item scale could produce the *reported rounded mean* at that n. The rounding policy is **ROUND_HALF_UP** and must match the article's reporting convention.

**عربي:** عدم الاتساق الحسابي لا يعني إثبات التزوير؛ قد تكون هناك أوزان، قيم مفقودة، متوسطات مركبة، قاعدة تقريب مختلفة، أو عدد فعلي مختلف. النجاح كذلك لا يثبت صحة بيانات الدراسة.

```python
from khaled_dsml import grim_mean_consistency
r = grim_mean_consistency("3.21", 7, decimals=2, minimum_score=1, maximum_score=5)
print(r["arithmetically_consistent"])
```

## 4 — Recompute a reported t-test p-value / مراجعة قيمة p من إحصائية t

`recalculate_t_test_p(t,df,alternative="two-sided",reported_p="0.023",decimals=3)` uses the Student t distribution for two-sided or selected one-sided tails, then compares **rounded** numeric reports using ROUND_HALF_UP. An inequality such as `p < .001` is **not parsed** as a numeric rounded statement.

This checks arithmetic **only**, not whether the degrees of freedom, assumptions, paired design, or the sample size were correctly reported.

## 5 — Schema contracts / عقد بنية البيانات

`validate_dataframe_contract(frame, schema, allow_extra_columns=True)` audits declared columns, type `kind` (numeric/integer/string/boolean/datetime/category), missingness, uniqueness, bounds and allowed values. It leaves the original DataFrame unchanged and returns **aggregate violation counts**, not raw row values. Invalid schema declarations fail explicitly.

```python
from khaled_dsml import validate_dataframe_contract
contract = {
    "id": {"kind": "integer", "nullable": False, "unique": True},
    "income": {"kind": "numeric", "min": 0, "nullable": False},
    "group": {"kind": "string", "allowed_values": ["A", "B"]},
}
# Replace synthetic_dataframe with your own DataFrame.
# report = validate_dataframe_contract(synthetic_dataframe, contract)
```

**عربي:** يجب تحديد العقد قبل تقييم البيانات لتجنب تكييف المعايير بعد رؤية النتائج. تجاوز الاختبارات لا ينفي الأخطاء المنهجية أو تحيز القياس.

## 6 — Page–Hinkley / إنذارات التغير المتتابع

`page_hinkley_change_monitor(values,delta=.01,threshold=10,min_instances=30,direction="both",reset_after_alarm=True)` estimates a running mean and cumulative upward/downward deviations, issuing zero-based indexed alarms after a warm-up. Threshold and delta are **user-calibrated** for the domain, false alarm costs, autocorrelation and seasonality. This returns **no p-values** or guaranteed false positive rate.

**عربي:** تغير الموقع الزمني للإشارة ليس إثباتًا للسببية ولا تقديرًا مضمونًا لنقطة تغير حقيقية. يُنصح باختبار معدلات الإنذارات الكاذبة على بيانات مرجعية مستقلة.

## Rendering, verification and evidence

All six methods provide bilingual Arabic/English interpretations to the offline **Analytics Studio**. Arabic titles render RTL and English titles LTR. Unit/reference tests in `tests/test_integrity_diagnostics.py` independently check McNemar versus SciPy binomial and chi-square distributions, MAPE versus scikit-learn, GRIM against exhaustive integer-total enumeration for small n, t-tail p-values versus SciPy, exact DataFrame-contract violation counts, Page–Hinkley shift/no-shift behavior, and Studio markup.

The original 256-item catalogue gains **six mapped public callable candidates** (#158, #170, #219, #222, #239, #242). This is **not** scientific sign-off of compound items nor an assertion that all advanced methods are complete.
