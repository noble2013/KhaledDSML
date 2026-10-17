# KhaledDSML 0.5.0 — Advanced Academic Methods | المناهج البحثية المتقدمة

**Development status / الحالة:** نسخة تطويرية غير منشورة. النسخة الرسمية على PyPI: **0.4.0**.  
**Created for:** Dr. Khaled Ali — University of El Oued.

توسعة منهجية تستهدف بنودًا لم تُنفذ في كتالوج الاختبارات، مع الاحتفاظ بواجهة **Analytics Studio** الإبداعية ذات الاتجاه الصحيح: العربية RTL بمحاذاة اليمين، والإنجليزية LTR بمحاذاة اليسار. الأدوات الحالية لا تمثل جميع بنود الكتالوج الـ256 ولا تثبت صلاحية نموذج بحثي دون مراجعة مستقلة.

## ما الجديد | Implemented APIs

| مجال | APIs | ملاحظات أساسية |
|---|---|---|
| الافتراضات المتعددة | `mardia_multivariate_normality`, `box_m_covariance_equality`, `mauchly_sphericity_test` | الالتواء والتفلطح في Mardia اختباران مستقلان؛ Box M حساس لعدم الاعتدالية |
| الاستقلال والتباين | `breusch_godfrey_test`, `levene_mean_test` | يعمل Breusch–Godfrey على انحدار مُلائم؛ Levene mean ليس Brown–Forsythe |
| اختبارات فئوية | `cochran_armitage_trend_test`, `g_test_independence`, `cochran_mantel_haenszel_test` | تعديلات المقارنات المتعددة والطبقات شرط لتفسير ملائم |
| النسب والتكافؤ | `newcombe_two_proportions_interval`, `welch_tost_equivalence`, `two_group_power_analysis` | TOST يحتاج هامش تكافؤ مسبقًا؛ القوة تعتمد على أثر مفترض |
| القياس النفسي | `omega_single_factor`, `intraclass_correlation`, `krippendorff_nominal_alpha` | أوميغا عامل واحد ليست أوميغا هرمية؛ ICC بنموذجين وبدون CI مختلق |
| التمثيلات والعناقيد | `trustworthiness_embedding`, `distance_correlation`, `bootstrap_cluster_stability` | مؤشرات تشخيصية وصفية لا تثبت معنوية العناقيد |
| النصوص والمدونات | `corpus_tokens`, `lexical_diversity`, `zipf_frequency_analysis`, `dunning_corpus_keyness`, `paired_bootstrap_score_difference` | دعم Unicode للعربية؛ Zipf وTTR وصفيان؛ Bootstrap يقدم فاصلًا وليس p-value |

**Dependencies:** `numpy`, `pandas`, `scipy`, `scikit-learn` أساسية؛ `statsmodels` اختياري لبعض التحليلات المتقدمة.

## التثبيت التجريبي في Colab

```python
%pip install -U statsmodels
%pip install --force-reinstall --no-cache-dir --no-deps "git+https://github.com/noble2013/KhaledDSML.git@feature/v0.5-methods-completion"
```

ثم أعد تشغيل الجلسة **Runtime → Restart session** قبل استيراد المكتبة.

## 1. اختبار Box M وتفسير ثنائي اللغة بالواجهة الجميلة

```python
import numpy as np
from khaled_dsml import (
    box_m_covariance_equality, display_analysis_dashboard
)

rng = np.random.default_rng(42)
group1 = rng.normal(size=(75, 3))
group2 = rng.normal(size=(80, 3))
result = box_m_covariance_equality(group1, group2)

display_analysis_dashboard(
    result,
    title_ar="فحص تجانس مصفوفات التباين والتغاير",
    title_en="MULTIVARIATE ASSUMPTIONS · BOX M",
    theme="aurora",
    font_size=22,
)
```

**Caution:** Non-rejection does not prove equality of covariance matrices. Box M is sensitive to departures from multivariate normality.

## 2. اختبار تكافؤ متوسطين مع هامش محدد مسبقًا

```python
from khaled_dsml import welch_tost_equivalence, display_analysis_dashboard

rng = np.random.default_rng(123)
a = rng.normal(10, .18, size=90)
b = rng.normal(10, .18, size=90)
equivalence = welch_tost_equivalence(a, b, margin=.5)
display_analysis_dashboard(
    equivalence, title_ar="اختبار تكافؤ متوسطين",
    title_en="EQUIVALENCE · WELCH TOST",
    theme="ocean",
)
```

قيمة p في TOST تختبر فرضية **عدم التكافؤ** عند هامش محدد مسبقًا؛ لا تعني أن المتوسطين متطابقان.

## 3. ثبات المحكمين وتقييم التضمين

```python
from khaled_dsml import (
    krippendorff_nominal_alpha, intraclass_correlation,
    trustworthiness_embedding, display_analysis_dashboard,
)
import numpy as np

agreement = krippendorff_nominal_alpha([
    ["نعم", "نعم", None],
    ["لا", "لا", "لا"],
    ["نعم", "لا", "نعم"],
    ["لا", None, "لا"],
])
rng = np.random.default_rng(41)
original = rng.normal(size=(60, 3))
embedded = original[:, :2]
trust = trustworthiness_embedding(original, embedded)
display_analysis_dashboard([agreement, trust], theme="amethyst")
```

## 4. تحليل مدونة لغوية عربية

```python
from khaled_dsml import (
    lexical_diversity, zipf_frequency_analysis,
    display_analysis_dashboard,
)

text = "الاقتصاد والتنمية في الجزائر " * 30 + "البيانات في الاقتصاد " * 10
lexicon = lexical_diversity(text)
zipf = zipf_frequency_analysis(text, min_types=5)

display_analysis_dashboard(
    [lexicon, zipf],
    title_ar="لوحة تحليل المدونة العربية",
    title_en="ARABIC LANGUAGE CORPUS · LEXICAL INSIGHTS",
    theme="aurora",
)
```

يُعرض جدول التردد الحقيقي ضمن Analytics Studio عند توافره. تجزئة الكلمات هذه بسيطة وليست تحليلًا صرفيًا كاملًا للغة العربية.

## 5. فاصل ثقة للفرق في مقياس أداء نموذجين

```python
import numpy as np
from khaled_dsml import paired_bootstrap_score_difference, display_analysis_dashboard

reference = np.arange(30)
model_a = reference.copy()
model_b = reference + 1
def score(ref, prediction):
    return -float(np.mean((np.asarray(ref)-prediction)**2))

result = paired_bootstrap_score_difference(
    reference, model_a, model_b, score, n_resamples=500, seed=42
)
display_analysis_dashboard(result, theme="ocean")
```

**Interpretation:** Paired bootstrap percentile intervals should not be presented as exact p-values, and text documents must be resampled in aligned pairs.

## البنود التي لا تزال تحتاج تنفيذًا خاصًا

- Little's MCAR مع معالجة دقيقة لأنماط الفقد.
- Henze–Zirkler، وSEM/CFA/measurement invariance بمراجع متعددة.
- اختبارات DeLong والمقارنات النموذجية المتقدمة غير المغطاة بعد.
- الاختبارات البايزية، ووثائق APA/Word/Excel المتكاملة.
- بيانات التدريب الضخمة جدًا وعمليات Spark/Dask المتخصصة التي لم تضف لها وظائف صريحة بعد.

لكل اختبار جديد اختبارات `pytest` مرجعية وفحوص للمدخلات؛ لا توضع علامة «منجز» إلا عند تحقق التنفيذ الصحيح ونجاح CI.
