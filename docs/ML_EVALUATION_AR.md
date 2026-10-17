# تقييم نماذج التصنيف الثنائية — KhaledDSML 0.2.0

**المؤلف:** الأستاذ الدكتور خالد علي — جامعة الشهيد حمه لخضر، الوادي، الجزائر  
**كلية العلوم الاقتصادية والتجارية وعلوم التسيير**

هذا الجزء يضيف الاختبارات العملية الخاصة بتقييم أداء نماذج تعلم الآلة في مقياس علم البيانات وتعلم الآلة وتطبيقات البيانات الضخمة.

> الإصدار المنشور في PyPI الآن 0.1.0. هذه الإضافات موجودة **فقط** على فرع التطوير `feature/rtl-reporting-v0.2.0`، ولم تصدر رسميًا بعد.

## 1. تثبيت إصدار التطوير في Google Colab

```python
%pip install --no-deps --force-reinstall "git+https://github.com/noble2013/KhaledDSML.git@feature/rtl-reporting-v0.2.0"
```

بعد تثبيت النسخة الجديدة أعد تشغيل جلسة Colab قبل الاستيراد.

## 2. تقييم احتمالات التصنيف (مثال مباشر)

```python
from khaled_dsml import binary_classification_metrics, roc_curve_table
from khaled_dsml import display_classification_report

y_true = [0, 0, 1, 1]
y_probability = [0.1, 0.4, 0.35, 0.8]

metrics = binary_classification_metrics(y_true, y_probability, positive_label=1)
print(metrics)
display_classification_report(metrics)
print(roc_curve_table(y_true, y_probability))
```

**النتيجة المرجعية لهذا المثال:** ROC-AUC = `0.75`، الدقة = `0.75`، والاسترجاع عند العتبة `0.5` هو `0.5`.

## 3. رسم منحنى ROC

```python
from khaled_dsml.plots import plot_roc_curve

fig, ax = plot_roc_curve(y_true, y_probability)
```

يتطلب Matplotlib؛ ثبّته عند الحاجة بـ `%pip install matplotlib`.

## 4. مقارنة الانحدار اللوجستي وشجرة القرار

```python
from sklearn.datasets import make_classification
from khaled_dsml import make_classifier, split_data, compare_classifiers

X, y = make_classification(
    n_samples=300, n_features=6, n_informative=4,
    n_redundant=0, random_state=42
)
X_train, X_test, y_train, y_test = split_data(
    X, y, stratify=True, random_state=42
)

models = {
    "Logistic Regression": make_classifier(X_train, "logistic"),
    "Decision Tree": make_classifier(X_train, "decision_tree", max_depth=4),
}

comparison = compare_classifiers(
    models, X_train, y_train, X_test, y_test, positive_label=1
)
display(comparison)
```

كل نموذج يُستنسخ ويُدرّب على **بيانات التدريب فقط**. التنبؤات الاحتمالية تُحتسب على بيانات الاختبار فقط. تُرتب النماذج حسب ROC-AUC ثم F1، لكن الأفضلية على عينة واحدة لا تعني تفوقًا مضمونًا خارج العينة.

## 5. حفظ تقرير عربي HTML

```python
from khaled_dsml import save_classification_report

save_classification_report(
    metrics, "classification_ar.html",
    title="تقرير نموذج التنبؤ بالتعثر في السداد"
)
```

لفتح التقرير من Colab نزّل الملف:

```python
from google.colab import files
files.download("classification_ar.html")
```

## 6. المعايير الإحصائية وحدودها

| المقياس | معنى مبسط |
|---|---|
| Accuracy | نسبة التنبؤات الصحيحة إجمالًا |
| Balanced Accuracy | متوسط الاسترجاع لكل فئة؛ مفيد عند عدم توازن الفئات |
| Precision | نسبة التنبؤات الموجبة الصحيحة من جميع التنبؤات الموجبة |
| Recall | نسبة الحالات الموجبة التي اكتشفها النموذج |
| Specificity | نسبة الحالات السالبة التي اكتشفها النموذج |
| F1 | المتوسط التوافقي لـ Precision وRecall |
| ROC-AUC | قدرة النموذج على ترتيب الحالة الموجبة فوق السالبة عبر العتبات |
| Average Precision | ملخص منحنى Precision-Recall، يتأثر بانتشار الفئة الموجبة |

**تنبيه منهجي:** يجب إدخال **احتمال الفئة الموجبة** وليس التصنيف 0/1 في `binary_classification_metrics`. ويتطلب ROC-AUC وجود كلا الصنفين في بيانات التقييم. التصنيف يعتمد على عتبة قابلة للتغيير؛ ROC-AUC لا تعتمد على العتبة. لا تعني ROC-AUC المرتفعة أن الاحتمالات معايرة جيدًا. دقة النموذج على بيانات اصطناعية لا تثبت صلاحيته لاتخاذ قرارات ائتمانية حقيقية.

يُستخدم `compare_classifiers` للتصنيف **الثنائي فقط** ويتطلب أن تدعم النماذج `predict_proba`. عند التعامل مع بيانات زمنية، ينبغي التقسيم حسب الزمن وعدم تطبيق التقسيم العشوائي.
