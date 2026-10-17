# KhaledDSML — علم البيانات وتعلّم الآلة وتحليل البيانات الضخمة

**الإعداد والتطوير: الأستاذ الدكتور خالد علي (Khaled Ali)**  
**الجامعة:** جامعة الشهيد حمه لخضر – الوادي، الجزائر (University of El Oued, Algeria)  
**الكلية:** كلية العلوم الاقتصادية والتجارية وعلوم التسيير (Faculty of Economic, Commercial and Management Sciences)  
**البريد الإلكتروني المهني:** [ali-khaled@univ-eloued.dz](mailto:ali-khaled@univ-eloued.dz)  
**الإصدار المستقر المنشور: 0.1.0 — نسخة التطوير القادمة: 0.2.0**

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

## أمثلة عملية

### اختبار اعتدالية

```python
from khaled_dsml import normality_test
result = normality_test([12, 13, 15, 17, 16, 14, 13, 18])
print(result.interpretation_ar)
print(result.p_value)
```

### تقارير عربية بتنسيق RTL — تطوير 0.2.0

```python
from khaled_dsml import normality_test, display_test_report, save_test_report

result = normality_test([12, 13, 15, 17, 16, 14, 13, 18])
display_test_report(result)  # عرض منسق داخل Jupyter أو Google Colab
save_test_report(result, "report.html")  # تقرير قابل للطباعة دون اتصال
```

**تنبيه:** هذه الدوال ضمن الفرع التجريبي `feature/rtl-reporting-v0.2.0`، ولم تصل بعد إلى إصدار PyPI المنشور `0.1.0`. تعليمات تجربة الفرع في [التوثيق العربي](docs/RTL_REPORTS_AR.md).

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
| `reports` (التطوير 0.2.0) | `render_test_report`, `display_test_report`, `save_test_report` (تقارير عربية RTL) |\n| `statistics` | `normality_test`, `independent_t_test`, `paired_t_test`, `anova_test`, `kruskal_test`, `chi_square_test`, `pearson_test`, `spearman_test`, `levene_test`, `ks_two_sample_test` |
| `quality` | `quality_report`, `missingness_report`, `iqr_outliers`, `fill_missing_numeric`, `optimize_memory`, `distribution_drift_test` |
| `ml` | `make_regressor`, `make_classifier`, `cross_validation_summary`, `classification_metrics`, `regression_metrics`, `cluster_data`, `pca_summary` |
| `bigdata` | `generate_transactions`, `csv_chunk_summary`, `csv_chunk_missing`, `csv_chunk_monthly`, `map_reduce_word_count`, `json_documents_to_table` |
| `sqltools` | `sqlite_import_frame`, `sqlite_group_summary` (SQLite محلي لتجارب SQL) |
| `timeseries` | `lagged_features` (خصائص مؤخرة دون تسرب للمستقبل) |
| `distributed` | `dask_csv_summary`, `spark_group_summary` (تثبيت `[dask]` أو `[spark]` واستخدام محرك فعلي) |
| `econometrics` | `adf_test`, `vif_table`, `breusch_pagan_test`, `ljung_box_test`, `cronbach_alpha` (تثبيت `econometrics`) |
| `plots` | `plot_missingness`, `plot_monthly`, `plot_confusion` (تثبيت `plots`) |
| `quizzes` | `get_questions`, `grade_quiz` |

لخريطة المقياسين والتغطية التفصيلية، راجع [docs/COURSE_MAPPING_AR.md](docs/COURSE_MAPPING_AR.md). للحصول على أمثلة مكتملة راجع `examples/`.

**لإنشاء المستودع ونشر الحزمة بأمان عبر GitHub Actions وTrusted Publishing (بدون حفظ مفتاح API)، راجع [دليل النشر العربي](docs/PUBLISH_AR.md).**

## الإصدار الرسمي الحالي والتطوير القادم

- **PyPI الحالي:** [KhaledDSML 0.1.0](https://pypi.org/project/khaled-dsml/0.1.0/) — مستقر للاستخدام.
- **التطوير 0.2.0:** عرض الاختبارات بالعربية RTL؛ قيد الاختبارات والمراجعة على فرع مستقل.
- **المنهجية:** لا يُنشر الإصدار 0.2.0 رسميًا قبل نجاح GitHub Actions ومراجعة التغييرات. راجع [دليل النشر](docs/PUBLISH_AR.md).

## Notes (English)

KhaledDSML is an **educational** Python toolkit; it wraps proven NumPy, pandas, SciPy and scikit-learn primitives rather than reinventing statistical algorithms. Optional statistics are backed by statsmodels. It is not a distributed processing engine: the MapReduce example runs locally; Optional Dask and Spark adapters are included, but need real engines and are not validated on a Spark cluster in the published version 0.1.0. Do not fit preprocessing on test data. Hypothesis-test interpretations should be checked against sampling design and assumptions. An MIT license covers this original wrapper code, not a claim of authorship over third-party algorithms.

## حقوق الاستخدام

الترخيص MIT. اسم التوزيع المنشور على PyPI هو `khaled-dsml`. كل الإضافات يمكن مراجعتها عبر اختبارات `pytest` قبل النشر. حقوق المنهجيات والخوارزميات الأساسية محفوظة لمؤلفيها ومشروعاتها الأصلية.
