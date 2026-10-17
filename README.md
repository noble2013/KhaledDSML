# KhaledDSML — علم البيانات وتعلّم الآلة وتحليل البيانات الضخمة

**الإعداد والتطوير: الأستاذ الدكتور خالد علي (Khaled Ali)**  
**الجامعة:** جامعة الشهيد حمه لخضر – الوادي، الجزائر (University of El Oued, Algeria)  
**الكلية:** كلية العلوم الاقتصادية والتجارية وعلوم التسيير (Faculty of Economic, Commercial and Management Sciences)  
**البريد الإلكتروني المهني:** [ali-khaled@univ-eloued.dz](mailto:ali-khaled@univ-eloued.dz)  
**الإصدار الرسمي المنشور على PyPI: 0.9.0 (17 أكتوبر 2026)**  
**فرع التطوير الجاري: 0.10.0.dev0 — لم يُنشر بعد**

حزمة Python تعليمية للمقاييس التالية:

- علم البيانات وتعلّم الآلة — السنة الثالثة ليسانس اقتصاد وتسيير المؤسسات.
- تحليل البيانات الضخمة — السنة الثانية ماستر اقتصاد كمي.

> تنبيه: تم تصميم المشروع وفق البرامج البيداغوجية التي أعدّها المؤلف؛ لا يُدَّعى تطابقه مع بطاقة وزارية رسمية دون مراجعتها. البيانات الاصطناعية للأغراض التعليمية فقط. الإحصاء الوصفي لا يثبت السببية، وعدم رفض الفرضية الصفرية لا يثبت صحتها.

## لماذا هذه المكتبة؟

- اختبارات فرضيات شائعة مع `statistic`، و`p_value`، وفرضية صفرية، وتفسير عربي.
- فحص جودة البيانات واكتشاف القيم الشاذة وتحسين الذاكرة.
- أنابيب تدريب آمنة من تسرب البيانات، وتقييم الانحدار والتصنيف والتجميع.
- معالجة CSV بالأجزاء، وتجميع شهري وجغرافي، وتوليد مليون معاملة اصطناعية إذا طُلب ذلك.
- 24 سؤال اختيار متعدد للتقويم التكويني مع التصحيح والتفسير.
- دعم Windows وJupyterLab وGoogle Colab بعد تثبيت الاعتمادات.

## التثبيت من المجلد المصدر — Windows PowerShell

افتح PowerShell من داخل المجلد الذي يحتوي `pyproject.toml`:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[all,dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\khaled-dsml.exe demo --rows 1000
.\.venv\Scripts\khaled-dsml.exe quiz --course big_data --count 5
```

بعض بيئات Windows تستعمل `py` بدل `py -3.13` عند عدم توفر الفرع 3.13.

*الإصدار الرسمي الحالي على PyPI:*

```powershell
py -m pip install khaled-dsml
```

### KhaledDSML 0.9.0 — البيانات المفقودة والاستكمال المتعدد (منشور)

أربع أدوات أكاديمية إضافية: **Little MCAR** مع تقدير EM، وفحص ارتباط غياب المتغيرات المرصودة وتصحيح Holm، وتعويض متعدد باحتمالات مع عزل بيانات التدريب عن الاختبار، ودمج تقديرات Rubin.

نتائج جمالية في **Analytics Studio** بالاتجاه العربي RTL والإنجليزي LTR، مع تنبيه واضح إلى أن رفض MCAR لا يحدد تلقائيًا MAR أو MNAR.

[الدليل العلمي للبيانات المفقودة 0.9 بالعربية والإنجليزية](docs/MISSING_DATA_0_9_AR_EN.md) · [دفتر Colab 12 بلوحات إحصائية إبداعية](notebooks/12_missing_data_studio.ipynb).

**0.9.0 منشور رسميًا** على [PyPI](https://pypi.org/project/khaled-dsml/0.9.0/).

### KhaledDSML 0.8.0 — الوساطة والتعديل الإحصائي (منشور)

إضافة **6 وظائف بحثية**: مسارات الوساطة `a/b/c/c′`، واختبار **Sobel**، وفواصل **Bootstrap** للارتباط غير المباشر، واختبار التفاعل `X×W` مع `ΔR²`، والميول المشروطة **Simple Slopes**، وحدود **Johnson–Neyman**.

تدعم Analytics Studio جداول ثنائية اللغة: **خط عربي أثقل ومحاذاة يمين، والإنجليزية يسار**، مع ثيمات Aurora/Ocean/Amethyst وتقرير HTML أنيق دون اتصال.

[الدليل العلمي 0.8 بالعربية والإنجليزية](docs/MEDIATION_MODERATION_0_8_AR_EN.md) · [دفتر Colab 11 التفاعلي](notebooks/11_mediation_moderation_studio.ipynb)

`0.8.0` إصدار منشور سابق. دلالات نماذج الوساطة *ارتباطية* ولا تثبت سببًا.

### KhaledDSML 0.7.0 — PySpark وDask للتحليل الموزع (منشور)

**PySpark مُضافة منذ إصدار 0.2.0** ضمن الإضافة الاختيارية `[spark]`، ولا تتطلب النسخة الأساسية تثبيتها. يوسّع 0.7 استخدام Spark وDask ليشمل: فحص NULL وNaN والسلاسل الفارغة، قياس التكرارات، جداول إحصائية رقمية، تقسيمًا زمنيًا لا يفقد التواريخ المعطوبة، وتدريب نموذج تصنيف لوجستي وتقييم ROC-AUC/PR-AUC موزّعًا عبر Spark ML.

```powershell
py -3.13 -m pip install --upgrade "khaled-dsml[distributed]==0.9.0"
```

وظائف 0.7 متاحة في الإصدار الرسمي 0.9.0. للاطلاع على الواجهات: [دليل PySpark وDask](docs/PYSPARK_DASK_0_7_AR_EN.md) · [دفتر Colab بالتصميم الإبداعي](notebooks/10_pyspark_dask_analytics.ipynb). تبقى العربية RTL يمينًا والإنجليزية LTR يسارًا في تقارير Analytics Studio.

### KhaledDSML 0.6.0 — المقارنات البعدية ومقارنة ROC-AUC (منشور)

إضافة خمس اختبارات مقارنة بعدية: **Tukey HSD وGames–Howell وDunn وDunnett وNemenyi–Friedman**، مع اختبار **DeLong المترابط** لمقارنة ROC-AUC لنموذجين على الأشخاص أنفسهم، وBootstrap مزدوج بفاصل ثقة.

تظهر النتائج في جداول تفاعلية بالتمرير ضمن **Analytics Studio**، مع تدرجات Aurora/Ocean/Amethyst، ومحاذاة العربية يمينًا والإنجليزية يسارًا، وتصدير HTML ملائم للطباعة.

```python
from khaled_dsml import (
    games_howell_comparisons, display_analysis_dashboard
)
import numpy as np
rng=np.random.default_rng(19)
groups={
    "الوادي":rng.normal(50,10,30),
    "بسكرة":rng.normal(56,14,30),
    "الجزائر":rng.normal(60,11,30),
}
display_analysis_dashboard(
    games_howell_comparisons(groups), theme="aurora"
)
```

[دفتر Colab الأكاديمي 09](notebooks/09_posthoc_roc_studio.ipynb) · [دليل 0.6 بالعربية والإنجليزية](docs/POSTHOC_ROC_0_6_AR_EN.md)

0.6.0 إصدار منشور سابق، ولا يمثل إنهاءً لكل بنود الكتالوج الـ256.

### KhaledDSML 0.5.0 — استكمال اختبارات البحث العلمي (تطوير)

تم توسيع المكتبة في فرع التطوير باختبارات **Mardia، Box's M، Mauchly، Breusch–Godfrey، Cochran–Armitage، CMH، G-test، TOST**، وفواصل Newcombe، وتحليل القوة، وثبات **Omega/ICC/Krippendorff**، وTrustworthiness وBootstrap للعناقيد، وتحليل النصوص العربية **Zipf وTTR وKeyness**.

تستخدم النتائج واجهة **Analytics Studio** المبتكرة مع محاذاة العربية يمينًا والإنجليزية يسارًا. وظائف 0.5.0 منشورة، ولا يعني ذلك اكتمال البنود الـ256.

[دليل التطوير 0.5 بالعربية والإنجليزية](docs/METHODS_0_5_AR_EN.md) · [دفتر Colab جاهز باختبارات متعددة ولوحات أنيقة](notebooks/08_advanced_methods.ipynb)

```python
from khaled_dsml import (
    box_m_covariance_equality, display_analysis_dashboard
)
import numpy as np
rng = np.random.default_rng(5)
result = box_m_covariance_equality(
    rng.normal(size=(60, 3)), rng.normal(size=(70, 3))
)
display_analysis_dashboard(result, theme="aurora")
```

### Academic Analysis Suite 0.4.0 — تحديث شامل للاختبارات البحثية

**مرحلة 0.4.0 منشورة سابقًا على PyPI**. أضيفت اختبارات الاستدلال والنسب، وتقييم النماذج ومعايرتها، وموثوقية الاستبيانات، والتحليل العاملي الاستكشافي، وتشخيص الانحدار، والسلاسل الزمنية المتقدمة. تبقى التقارير الإبداعية **Analytics Studio** باللغتين العربية يمينًا والإنجليزية يسارًا.

```python
from khaled_dsml import (
    mcnemar_test, wilson_proportion_interval,
    classification_diagnostics, display_analysis_dashboard,
)

test = mcnemar_test([[20, 5], [15, 30]])
display_analysis_dashboard(test, theme="aurora")

interval = wilson_proportion_interval(65, 100)
display_analysis_dashboard(interval, theme="amethyst")
```

الوثائق العلمية للأسماء والفرضيات وشروط الاستخدام: [Academic Suite 0.4 AR/EN](docs/ACADEMIC_SUITE_0_4_AR_EN.md). لا ادعاء بإتمام كل كتالوج 256 بندًا قبل تغطيته المرجعية والتحقق الكامل.

### Analytics Studio — واجهة تحليل إحصائي إبداعية (تطوير 0.3.0)

الواجهة الجديدة **افتراضيًا** تعرض رأسًا متدرجًا، وبطاقات مؤشرات محسوبة فعليًا، وقرار الاختبار عند توافره، وتفسيرًا عربيًا من اليمين وإنجليزيًا من اليسار. تدعم الرسم SVG دون إنترنت، وثلاثة ألوان: `aurora` و`ocean` و`amethyst`. يمكن العودة إلى الشكل النصي القديم بـ `style="plain"`.

```python
from khaled_dsml import (
    mann_kendall_trend_test, display_bilingual_results,
    display_analysis_dashboard, save_analysis_dashboard,
)

data = [100, 108, 113, 121, 125, 138, 142, 154, 163, 178, 185, 197]
result = mann_kendall_trend_test(data)

# النمط الافتراضي الآن تصميم احترافي
display_bilingual_results(result, theme="aurora", chart_data=data)

# تقرير موسع للطباعة
display_analysis_dashboard(result, theme="ocean", chart_data=data)
save_analysis_dashboard(result, "report_ar_en.html", chart_data=data)
```

**[دفتر Colab جاهز بتصميم مبهر](notebooks/06_analytics_studio.ipynb)** · **[دليل الاستخدام بالعربية والإنجليزية](docs/ANALYTICS_STUDIO_AR_EN.md)**. أصبحت هذه الوظائف ضمن الإصدارات المنشورة بدءًا من `0.3.0`.

### التطوير 0.3.0: السلاسل الزمنية وقابلية إعادة البحث

الإضافات التالية **موجودة في فرع التطوير فقط** `feature/v0.3-timeseries-reproducibility`، ولم تُنشر بعد على PyPI:

```python
from khaled_dsml import (
    mann_kendall_trend_test, pettitt_change_point_test,
    forecast_accuracy_report, reproducibility_manifest,
    display_bilingual_result, display_bilingual_results,
)
trend = mann_kendall_trend_test([1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11, 12])
display_bilingual_result(trend)  # العربية يمين، الإنجليزية يسار، خط واضح
```


**دفتر جاهز للتجربة:** [05_bilingual_display.ipynb](notebooks/05_bilingual_display.ipynb) — استيراد الاختبارات وعرض النتائج العربية والإنجليزية بنفس شكل Colab، وحفظ تقرير HTML للطباعة.

**النمط النصي الكلاسيكي لدفاتر Google Colab وJupyter (تطوير 0.3.0):** استخدم `style="plain"` للحصول على تنسيق بلا إطارات، العربية RTL بمحاذاة اليمين، والإنجليزية LTR بمحاذاة اليسار. أصبح نمط `studio` الإبداعي هو الافتراضي:

```python
from khaled_dsml import (
    display_bilingual_result, display_bilingual_results,
    save_bilingual_results, pettitt_change_point_test,
)

change = pettitt_change_point_test([10] * 20 + [30] * 20)
display_bilingual_result(trend, style="plain")   # النمط الكلاسيكي
display_bilingual_results([trend, change], font_size=24, style="plain")  # عدة نتائج
save_bilingual_results([trend, change], "statistics_ar_en.html", style="plain")
```

تتضمن كذلك Box–Pierce وARCH-LM وMASE والتوثيق بالبصمة SHA-256. التفاصيل في [دليل السلاسل الزمنية وإعادة إنتاج البحث](docs/TIMESERIES_REPRODUCIBILITY_AR_EN.md).

## أمثلة عملية

### اختبار اعتدالية

```python
from khaled_dsml import normality_test
result = normality_test([12, 13, 15, 17, 16, 14, 13, 18])
print(result.interpretation_ar)
print(result.p_value)
```

### تقارير عربية بتنسيق RTL — منذ 0.2.0

```python
from khaled_dsml import normality_test, display_test_report, save_test_report

result = normality_test([12, 13, 15, 17, 16, 14, 13, 18])
display_test_report(result)  # عرض منسق داخل Jupyter أو Google Colab
save_test_report(result, "report.html")  # تقرير قابل للطباعة دون اتصال
```

**هذه الوظائف متاحة رسميًا منذ الإصدار 0.2.0**. راجع [التوثيق العربي](docs/RTL_REPORTS_AR.md).

### تقييم التصنيف الثنائي وROC-AUC — منذ 0.2.0

```python
from khaled_dsml import binary_classification_metrics, display_classification_report

metrics = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
display_classification_report(metrics)
```

لرسم ROC ومقارنة نماذج الانحدار اللوجستي وأشجار القرار دون تسرّب بيانات التدريب، راجع [دليل تقييم التصنيف العربي](docs/ML_EVALUATION_AR.md).

### إحصاءات وصفية شاملة + شرح عربي وإنجليزي (منذ 0.2.0)

```python
from khaled_dsml import describe_numeric, display_descriptive_report

summary = describe_numeric([10, 12, 12, 14, 18, 20, None])
display_descriptive_report(summary)  # Arabic RTL + English LTR
```

تشمل النزعة المركزية، والتشتت، والجداول التكرارية، والربيعيات والقيم المئوية، والالتواء والتفرطح، والمتوسط المشذب والهندسي والتوافقي، والارتباط والتغاير، ومعامل جيني، والتقارير الثنائية اللغة.

**[الدليل الأكاديمي عربي–إنجليزي](docs/DESCRIPTIVES_BILINGUAL_AR_EN.md)** | **[كتالوج المستخدم: متابعة 256 بندًا](docs/DESCRIPTIVES_BILINGUAL_AR_EN.md#5-حدود-التغطية--catalogue-coverage)**

### تدقيق جودة البيانات والافتراضات | Data audit and assumption checks (0.2.0)

```python
import pandas as pd
from IPython.display import HTML, display
from khaled_dsml import audit_dataset, render_audit_report, anderson_darling_normality

df = pd.DataFrame({"price": [100, 110, None, 110], "sales": [5, 7, 8, 7]})
audit = audit_dataset(df)
display(HTML(render_audit_report(audit)))
print(anderson_darling_normality([1.1, 1.2, 1.5, 1.8, 2.1, 2.4, 2.6, 3.0]))
```

توفر وحدة `audit` كشف التكرارات التامة، أنماط الفقد، الأعمدة شبه الثابتة، القيم الشاذة، بنفورد، الأرقام النهائية، وتداخل التدريب والاختبار. تضم `assumptions_extra` اختبارات Anderson–Darling وLilliefors وRuns واختبار Bartlett للكروية وKMO وDurbin–Watson وVIF.

**[التوثيق الأكاديمي بالعربية والإنجليزية](docs/AUDIT_ASSUMPTIONS_AR_EN.md)**. وظائف هذا القسم منشورة رسميًا منذ 0.2.0 على PyPI.

### كشف مشاكل البيانات

```python
from khaled_dsml import generate_transactions, quality_report, missingness_report
transactions = generate_transactions(100_000)
print(quality_report(transactions))
print(missingness_report(transactions))
```

### تعلم الآلة (منع تسرب البيانات)

```python
from sklearn.datasets import make_regression
from khaled_dsml import split_data, make_regressor, regression_metrics
X, y = make_regression(n_samples=200, n_features=3, noise=3, random_state=42)
X_train, X_test, y_train, y_test = split_data(X, y)
model = make_regressor(X_train, "ridge", alpha=1.0)
model.fit(X_train, y_train)
print(regression_metrics(y_test, model.predict(X_test)))
```

### Dask وApache Spark لتحليل البيانات الضخمة (منذ 0.2.0)

المكتبة تدعم تشغيل المحركين محليًا باعتماديات اختيارية. يقوم Dask بمعالجة الملفات على أجزاء وينفذ التجميع عند الطلب، بينما يعيد Spark نتائج موزعة دون تحويل الملفات الكبيرة إلى pandas.

```python
from khaled_dsml import dask_group_summary, dask_missingness_report
# تنفيذ مؤجل؛ تسترجع الجداول الصغيرة فقط إلى الذاكرة
summary = dask_group_summary("transactions.csv", group_by="wilaya",
                             amount_column="montant_da", compute=True)
print(summary)
print(dask_missingness_report("transactions.csv"))
```

```python
from khaled_dsml import create_spark_session, spark_read_csv, spark_group_summary
spark = create_spark_session(master="local[2]")
df = spark_read_csv(spark, "transactions.csv")
spark_group_summary(df).show()
spark.stop()
```

**<a href="docs/DASK_SPARK_AR_EN.md">دليل Dask وSpark بالعربية والإنجليزية</a>** — أمثلة للتجميع الشهري والمفقودات، وطريقة التثبيت في Colab وWindows. يجب تثبيت الإضافات الاختيارية أولًا، وSpark يحتاج Java. هذه الوظائف الموسعة متاحة رسميًا منذ 0.2.0 على PyPI.

### مليون معاملة والتجميع بالأجزاء

```python
from khaled_dsml import generate_transactions, csv_chunk_summary, csv_chunk_monthly
transactions = generate_transactions(1_000_000)
transactions.to_csv("transactions.csv", index=False)
print(csv_chunk_summary("transactions.csv", group_by="wilaya"))
print(csv_chunk_monthly("transactions.csv"))
```

`read_csv(chunksize=...)` لا يحتاج تحميل الملف بكامله. **لكن** توليد مليون صف في `generate_transactions` يحتفظ بها في الذاكرة قبل الحفظ؛ اختر عدد صفوف أصغر لجهاز بذاكرة محدودة.

### أسئلة تصحيح آلي

```python
from khaled_dsml import get_questions, grade_quiz
questions = get_questions("machine_learning", limit=3, seed=3)
for question in questions:
    print(question.id, question.question)
answers = {q.id: "A" for q in questions}  # إجابات افتراضية
print(grade_quiz(questions, answers))
```

## واجهات المكتبة

| الوحدة | أمثلة الدوال |
|---|---|
| `reports` (التطوير 0.2.0) | `render_test_report`, `display_test_report`, `save_test_report` (تقارير عربية RTL) |
| `statistics` | `normality_test`, `independent_t_test`, `paired_t_test`, `anova_test`, `kruskal_test`, `chi_square_test`, `pearson_test`, `spearman_test`, `levene_test`, `ks_two_sample_test` |
| `quality` | `quality_report`, `missingness_report`, `iqr_outliers`, `fill_missing_numeric`, `optimize_memory`, `distribution_drift_test` |
| `audit` (0.2.0) | `audit_dataset`, `missing_pattern_report`, `duplicate_rows_report`, `modified_z_outliers`, `mahalanobis_outliers`, `ml_outliers` |
| `assumptions_extra` (0.2.0) | `anderson_darling_normality`, `lilliefors_normality`, `runs_test`, `kmo_sampling_adequacy`, `condition_number_vif` |
| `descriptive` (0.2.0) | `describe_numeric`, `descriptive_table`, `frequency_table`, `describe_categorical`, `display_descriptive_report` |
| `catalog_tests` (0.2.0) | `one_sample_t_test`, `fisher_exact_test`, `dagostino_normality_test`, `bootstrap_mean_ci`, `adjust_pvalues`, `cohen_d` |
| `ml_evaluation` (0.2.0) | `binary_classification_metrics`, `roc_curve_table`, `compare_classifiers`, `render_classification_report`, `display_classification_report` |
| `ml` | `make_regressor`, `make_classifier`, `cross_validation_summary`, `classification_metrics`, `regression_metrics`, `cluster_data`, `pca_summary` |
| `bigdata` | `generate_transactions`, `csv_chunk_summary`, `csv_chunk_missing`, `csv_chunk_monthly`, `map_reduce_word_count`, `json_documents_to_table` |
| `sqltools` | `sqlite_import_frame`, `sqlite_group_summary` (SQLite محلي لتجارب SQL) |
| `timeseries` | `lagged_features` (خصائص مؤخرة دون تسرب للمستقبل) |
| `distributed` | `dask_read_csv`, `dask_group_summary`, `dask_missingness_report`, `dask_monthly_summary`, `spark_read_csv`, `spark_group_summary`, `spark_missingness_report`, `spark_monthly_summary` (تثبيت اختياري `[dask]` أو `[spark]` أو `[distributed]`) |
| `econometrics` | `adf_test`, `vif_table`, `breusch_pagan_test`, `ljung_box_test`, `cronbach_alpha` (تثبيت `econometrics`) |
| `plots` | `plot_missingness`, `plot_monthly`, `plot_confusion` (تثبيت `plots`) |
| `quizzes` | `get_questions`, `grade_quiz` |

لخريطة المقياسين والتغطية التفصيلية، راجع [docs/COURSE_MAPPING_AR.md](docs/COURSE_MAPPING_AR.md). للحصول على أمثلة مكتملة راجع `examples/`.

**لإنشاء المستودع ونشر الحزمة بأمان عبر GitHub Actions وTrusted Publishing (بدون حفظ مفتاح API)، راجع [دليل النشر العربي](docs/PUBLISH_AR.md).**

## الإصدار الرسمي الحالي والتطوير القادم

- **الإصدار المنشور:** [KhaledDSML 0.9.0 على PyPI](https://pypi.org/project/khaled-dsml/0.9.0/).
- **التطوير القادم:** `0.10.0.dev0` على فرع [feature/v0.10-catalog-henze-zirkler](https://github.com/noble2013/KhaledDSML/tree/feature/v0.10-catalog-henze-zirkler)؛ **غير منشور**، ولا يُدمج قبل نجاح GitHub Actions.
- **التدقيق:** [مراجعة كتالوج 256 بندًا](docs/CATALOG_AUDIT_0_10_AR_EN.md). لا يعني عدد اختبارات pytest اكتمال الكتالوج.
- **وظائف تجريبية:** Henze–Zirkler بمعايرة Monte Carlo، وتحليل حساسية MNAR بعوامل delta افتراضية؛ انظر [الدليل العلمي](docs/RESEARCH_METHODS_0_10_AR_EN.md). [دفتر Colab 13](notebooks/13_multivariate_mnar_studio.ipynb).

```powershell
py -3.13 -m pip install --upgrade "khaled-dsml[econometrics,distributed]==0.9.0"
```

## Notes (English)

KhaledDSML is an **educational** Python toolkit; it wraps proven NumPy, pandas, SciPy and scikit-learn primitives rather than reinventing statistical algorithms. Optional statistics are backed by statsmodels. It is not a distributed processing engine: the MapReduce example runs locally; Optional Dask and Spark adapters are included, and require compatible engines; real local Spark and Dask integration tests were executed for the published v0.9.0. Do not fit preprocessing on test data. Hypothesis-test interpretations should be checked against sampling design and assumptions. An MIT license covers this original wrapper code, not a claim of authorship over third-party algorithms.

## حقوق الاستخدام

الترخيص MIT. اسم التوزيع المنشور على PyPI هو `khaled-dsml`. كل الإضافات يمكن مراجعتها عبر اختبارات `pytest` قبل النشر. حقوق المنهجيات والخوارزميات الأساسية محفوظة لمؤلفيها ومشروعاتها الأصلية.
