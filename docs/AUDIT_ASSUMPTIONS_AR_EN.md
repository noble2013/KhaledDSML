# تدقيق جودة البيانات واختبارات الافتراضات | Data Audit & Statistical Assumptions

**المؤلف / Author:** الأستاذ الدكتور خالد علي — Dr. Khaled Ali  
**جامعة الشهيد حمه لخضر – الوادي، الجزائر / University of El Oued**  
**كلية العلوم الاقتصادية والتجارية وعلوم التسيير**  
**البريد المهني / Institutional email:** ali-khaled@univ-eloued.dz

## حالة النسخة / Release status

هذه الوظائف من التطوير الجاري **0.2.0** على فرع `feature/rtl-reporting-v0.2.0`، وليست جزءًا من الإصدار الرسمي `0.1.0` المنشور على PyPI. جرّب الفرع في بيئة منفصلة قبل استخدامه في بحث.

**In English:** These functions are available on the development branch only. The latest officially released PyPI version is still 0.1.0.

### Google Colab

```python
%pip install --no-cache-dir --no-deps --force-reinstall "git+https://github.com/noble2013/KhaledDSML.git@feature/rtl-reporting-v0.2.0"
```

أعد تشغيل بيئة Colab بعد التثبيت / Restart the runtime before importing.

## التدقيق المتكامل | One-call dataset audit

```python
import pandas as pd
from khaled_dsml import audit_dataset, render_audit_report, save_audit_report

df = pd.DataFrame({
    "income": [1000, 1200, 1200, None, 1300, 1300],
    "wilaya": ["الوادي", "الجزائر", "الجزائر", "بسكرة", "الوادي", "الوادي"],
})
audit = audit_dataset(df, missing_alert=0.15)
print(audit["interpretation_ar"])
print(audit["interpretation_en"])
display(audit["column_diagnostics"])
save_audit_report(audit, "data_audit_ar_en.html")
```

لا تحذف `audit_dataset` أي بيانات. إذا رُصدت مشكلات، يوصى بمراجعتها بشريًا.

## وظائف التدقيق / Audit methods

| الوظيفة | شرح بالعربية | English explanation |
|---|---|---|
| `duplicate_rows_report` | الصفوف المتطابقة تمامًا | Exact duplicate rows |
| `missing_pattern_report` | أنماط الفقد ونسبته | Descriptive missingness patterns |
| `column_diagnostics` | الأعمدة الثابتة وشبه الثابتة والقيم خارج المدى المحدد | Constant/near-constant columns and range checks |
| `modified_z_outliers` | القيم الشاذة باستخدام MAD | Robust modified Z-scores |
| `mahalanobis_outliers` | المسافة التربيعية متعددة المتغيرات وقيمة قطع كاي تربيع | Classical Mahalanobis distance and chi-square screening |
| `ml_outliers` | Isolation Forest أو LOF | Unsupervised Isolation Forest/LOF flags |
| `survey_straightlining` | تكرار إجابة واحدة في استبيان متعدد البنود | Possible survey straight-lining |
| `benford_first_digit_test` | اختبار توزيع الرقم الأول | Benford first-digit diagnostic |
| `terminal_digit_test` | اختبار انتظام الرقم الأخير للأعداد الصحيحة | Uniform final-digit test |
| `train_test_overlap` | كشف تكرار الصفوف الكاملة بين التدريب والاختبار | Full-row overlap across train and test |
| `audit_dataset` | ملخص قابل للطباعة | Bilingual consolidated quality report |

**حدود الصلاحية / Interpretation limits:**
- فحص نمط الفقد **ليس اختبار Little's MCAR**، وما زال الأخير متبقيًا. / Missing patterns do not prove MCAR; Little's test remains pending.
- فحص التكرار يكتشف **التكرارات التامة** فقط؛ تكرار شبه مطابق يحتاج مطابقة إضافية ومراجعة الهوية. / Exact duplicates are covered, not fuzzy entity matching.
- مسافة Mahalanobis مبنية على متوسط وتغاير كلاسيكيين، وتفترض مصفوفة غير منفردة. / Classical covariance is not robust to contamination.
- Benford ليس ملائمًا لبيانات عشوائية صغيرة المدى أو الأسعار المُسقفة؛ فشل الاختبار **ليس دليلًا على الاحتيال**. / Rejection is not evidence of fraud.
- Straight-lining لا يثبت سوء الإجابة. / Repeated responses do not prove respondent misconduct.
- صفوف متطابقة بين التدريب والاختبار مؤشر تداخل؛ **تسرب الهدف** يحتاج مراجعة الخصائص المتاحة وقت التنبؤ. / Row overlap does not detect all forms of target leakage.

## اختبارات الافتراضات / Assumption checks

```python
import numpy as np
import pandas as pd
from khaled_dsml import (
    anderson_darling_normality, lilliefors_normality, runs_test,
    durbin_watson_statistic, kmo_sampling_adequacy,
    bartlett_sphericity_test, condition_number_vif,
)

rng = np.random.default_rng(42)
sample = rng.normal(size=80)
ad = anderson_darling_normality(sample)
print(ad["interpretation_ar"])
print(ad["interpretation_en"])
print("Lilliefors p:", lilliefors_normality(sample).p_value)

survey = pd.DataFrame({
    "item1": rng.normal(size=100),
    "item2": rng.normal(size=100),
    "item3": rng.normal(size=100),
})
print(kmo_sampling_adequacy(survey)["kmo"])
print(bartlett_sphericity_test(survey).interpretation_en)
print(condition_number_vif(survey)["vif"])
```

- `anderson_darling_normality`: القرار بناءً على **قيمة حرجة** من SciPy ولا يُخترَع لها `p-value`. / Uses tabulated critical values; no fabricated p-value.
- `lilliefors_normality`: اختبار الاعتدالية مع تقدير معلمات التوزيع؛ يتطلب `statsmodels`. / Normality with fitted parameters.
- `runs_test`: تقريب طبيعي لاختبار عشوائية ترتيب القيم؛ يتطلب 5 قيم على الأقل في كل فئة بعد القطع. / Binary runs-test normal approximation.
- `durbin_watson_statistic`: إحصائية فقط؛ لا توجد قيمة احتمالية موحّدة بلا نموذج انحدار وفروض ملائمة. / A diagnostic statistic, not a universal p-value.
- `kmo_sampling_adequacy`: الملاءمة العامة ولدى كل متغير لتحليل العوامل. / Overall and per-variable sampling adequacy.
- `bartlett_sphericity_test`: اختبار مصفوفة الارتباط مقارنة بمصفوفة الوحدة. / Correlation identity test.
- `condition_number_vif`: رقم حالة ومعامل تضخم التباين لمتغيرات رقمية؛ ارتفاعهما مؤشر، لا قاعدة آلية لحذف المتغيرات. / Collinearity indicators are not a feature-deletion rule.

## الخطوة التالية / Remaining catalogue work

يحتوي كتالوج المستخدم على 256 بندًا في 12 وحدة. هذه المرحلة تغطي إضافات أساسية للتدقيق والافتراضات فقط؛ وتبقى: اختبار Little MCAR، والتحقق من استجابة الاستبيان السريعة، والمطابقة التقريبية، وBox's M، وMauchly، وMardia، وHenze-Zirkler وغيرها. ينبغي التحقق من كل خوارزمية متقدمة أمام نتائج مرجعية قبل وصفها بأنها صالحة للنشر العلمي.

## قواعد البحث / Research safeguards

تجنب الاستنتاج بالسببية من اختبارات وصفية، ولا تُفسّر عدم رفض فرضية العدم على أنه إثبات لها. وثّق قيمة `alpha`، وتصميم العينة، وفروض التوزيع، واستقلالية المشاهدات والتعديلات على المقارنات المتعددة. / Avoid causal inference from descriptive diagnostics, and do not treat failure to reject H0 as proof. Document assumptions and sampling design.
