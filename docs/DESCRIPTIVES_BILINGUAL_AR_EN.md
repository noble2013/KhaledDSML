# الإحصاء الوصفي والتفسير الثنائي اللغة | Descriptive statistics & bilingual interpretations

**المؤلف / Author:** الأستاذ الدكتور خالد علي — Dr. Khaled Ali  
**الانتماء / Affiliation:** جامعة الشهيد حمه لخضر، الوادي / University of El Oued  
**الكلية / Faculty:** كلية العلوم الاقتصادية والتجارية وعلوم التسيير  
**البريد المهني / Contact:** ali-khaled@univ-eloued.dz

> هذه الوظائف قيد التطوير على فرع `feature/rtl-reporting-v0.2.0`. **الإصدار الرسمي على PyPI ما زال 0.1.0**. لا تستورد الوظائف الجديدة من الإصدار القديم.

## التثبيت في Google Colab | Colab installation

```python
%pip install --no-cache-dir --force-reinstall --no-deps "git+https://github.com/noble2013/KhaledDSML.git@feature/rtl-reporting-v0.2.0"
```

أعد تشغيل الجلسة ثم نفذ المثال. / Restart runtime before importing the changed package.

## 1. الإحصاء الوصفي الشامل | Comprehensive descriptive statistics

```python
from khaled_dsml import (
    describe_numeric, frequency_table, descriptive_table,
    display_descriptive_report, save_descriptive_report,
)

sample = [10, 12, 12, 14, 18, 20, None]
stats = describe_numeric(sample)
print(stats["mean"], stats["median"], stats["std"], stats["skewness"])
display_descriptive_report(stats)
save_descriptive_report(stats, "descriptive_ar_en.html")
```

تحتوي `describe_numeric` على: العدد الكلي والناقص والصالح، المتوسط، الوسيط، المنوال/المنوالات، المجموع، الصغرى والكبرى والمدى، الربيعات والوسيط وملخص القيم الخمس، القيم المئوية الشائعة، التباين والانحراف المعياري والخطأ المعياري للمتوسط، معامل الاختلاف، المدى الربيعي، انحراف المتوسط والوسيط المطلق، الالتواء، التفرطح الزائد، المتوسط المشذب والهندسي والتوافقي، منتصف المدى، قاعدة القيم الشاذة 1.5×IQR ومعامل جيني للقيم غير السالبة.

`describe_numeric` includes total/valid/missing observations; arithmetic mean, median and modes; sum, range, extrema, quartiles, common percentiles and five-number summary; variance, standard deviation, standard error, CV, IQR, mean and median absolute deviations, bias-corrected skewness and excess kurtosis; trimmed, geometric and harmonic means; midrange, IQR outlier count, and Gini for nonnegative values.

**الاصطلاحات / Conventions:**
- التباين والانحراف المعياري افتراضيًا للعينة `ddof=1`، ويمكن استعمال `ddof=0` للمجتمع.
- الدوال تتجاهل `NaN` في المتغيرات الرقمية، لكن لا تتجاهل `Infinity`: ترفضه.
- `geometric_mean` و`harmonic_mean` غير معرّفين للبيانات غير الموجبة؛ تعرض `None`.
- `cv_percent` غير معرّف عندما يقترب المتوسط الحسابي من الصفر؛ تعرض `NaN`.
- `skewness` و`excess_kurtosis` قد يكونان غير معرّفين للعينات الصغيرة أو الثابتة.
- لا تُفسَّر القيمة الشاذة تلقائيًا على أنها خطأ، ولا يثبت الإحصاء الوصفي السببية.

## 2. الجداول التكرارية والفئات | Categorical and frequency analysis

```python
from khaled_dsml import describe_categorical, frequency_table
categories = ["الجزائر", "الوادي", "الوادي", "بسكرة", None]
print(describe_categorical(categories))
display(frequency_table(categories))
```

تُعرض التكرارات، والنسب النسبية والمئوية، والتكرارات التراكمية، والعدد الفريد، والمنوال، وShannon entropy وGini impurity.

## 3. الجداول متعددة المتغيرات | Multi-variable tables

```python
import pandas as pd
from khaled_dsml import descriptive_table, covariance_correlation, weighted_mean

df = pd.DataFrame({
    "sales": [100, 140, 160, 180],
    "cost": [80, 95, 122, 145],
    "wilaya": ["El Oued", "Algiers", "El Oued", "Algiers"],
})
display(descriptive_table(df))
matrix = covariance_correlation(df)
display(matrix["correlation"])
print(weighted_mean([10, 20, 30], [1, 2, 1]))
```

## 4. اختبارات جديدة من الكتالوج | Additional inference methods

```python
from khaled_dsml import (
    one_sample_t_test, dagostino_normality_test, kendall_test,
    adjust_pvalues, cohen_d, bootstrap_mean_ci,
)

r = one_sample_t_test([20, 22, 24, 28, 30], mu=20)
print(r.interpretation_ar)
print(r.interpretation_en)

print(adjust_pvalues([0.001, 0.02, 0.15], method="holm"))
print(cohen_d([10, 11, 12, 13], [7, 8, 9, 10]))
print(bootstrap_mean_ci([1, 2, 3, 4, 5, 6], n_resamples=500))
```

أضيفت كذلك `bartlett_test`, `fligner_test`, `fisher_exact_test`, `chi_square_gof_test`, `friedman_test`, `sign_test`, `cliffs_delta`, `permutation_mean_test`.

**English:** Each `TestResult` now provides `interpretation_ar`, `interpretation_en`, `null_hypothesis_ar`, `null_hypothesis_en`. Arabic-English HTML reporting is supported. Non-rejection never proves that the null is true. Bootstrap and permutation inference require assumptions about sampling and exchangeability.

## 5. حدود التغطية / Catalogue coverage

الملف المرفق يحوي **256 بندًا في 12 وحدة** ويصف خريطة تنفيذ لعدة إصدارات (v0.1 وv0.2 وv0.3 ولاحقًا). ليس مرجعًا لمعيار وزاري نهائي، وبعض البنود متداخلة أو تعني مسارات تحليل كاملة وليست اختبارًا واحدًا.

**Implemented and tested in the new branch:** comprehensive descriptive statistics, bilingual inferential interpretations, reusable HTML reports, the additional core inference functions above, and binary ML evaluation.

**Not yet implemented as certified APIs:** Little's MCAR, multivariate normality/Box's M, factorial ANOVA/ANCOVA/MANOVA, advanced psychometrics including EFA/CFA/SEM, advanced resampling comparisons (e.g. DeLong), all streaming and distributed drift detectors, NLP keyness, and research-forensics tests. They will be prioritized separately and compared with trusted software/reference cases before release.

لا ينبغي استخدام الدوال لأبحاث أو قرارات حساسة دون التحقق من اختيار الاختبار وفروضه، وحجم الأثر، وفواصل الثقة، وطريقة العينة.

## 6. Documentation in English

Data can be described with `describe_numeric`, `describe_categorical`, `frequency_table`, `descriptive_table`, `weighted_mean`, `covariance_correlation` and `gini_coefficient`. Their display functions create local, print-ready HTML with Arabic right-to-left paragraphs and English left-to-right explanations. Advanced catalogue entries are staged rather than falsely labelled as implemented.
