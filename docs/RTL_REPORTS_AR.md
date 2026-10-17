# التقارير العربية RTL — KhaledDSML 0.2.0 (نسخة تطوير)

**المؤلف:** أ.د. خالد علي — جامعة الشهيد حمه لخضر، الوادي، الجزائر  
**كلية العلوم الاقتصادية والتجارية وعلوم التسيير**  
**البريد المهني:** ali-khaled@univ-eloued.dz

تُعرض النتائج الرقمية باستعمال الاتجاه `ltr` داخل قالب عربي `rtl`، لتفادي اختلاط الأرقام والحروف في الطرفيات. لا يحتاج التقرير HTML الناتج إلى إنترنت ولا إلى JavaScript.

> **مهم:** هذا التطوير على فرع `feature/rtl-reporting-v0.2.0`، ولم يُنشر على PyPI الرسمي بعد. الإصدار المتاح للجميع في PyPI هو `0.1.0`.

## 1. الاستخدام في Google Colab

لتجربة نسخة التطوير الحالية في خلية Colab (تحتاج الإنترنت ووجود git):

```python
%pip install "git+https://github.com/noble2013/KhaledDSML.git@feature/rtl-reporting-v0.2.0"
```

ثم في خلية جديدة:

```python
from khaled_dsml import normality_test, display_test_report

result = normality_test([12, 13, 15, 17, 16, 14, 13, 18])
display_test_report(result)
```

## 2. إصدار تقرير HTML قابل للطباعة

```python
from khaled_dsml import (
    normality_test, pearson_test, save_test_report,
)

results = [
    normality_test([12, 13, 15, 17, 16, 14, 13, 18]),
    pearson_test([10, 20, 30, 40, 50], [12, 22, 32, 44, 51]),
]

path = save_test_report(results, "rapport_tests.html", title="تقرير التحليل الإحصائي")
print(path)
```

افتح `rapport_tests.html` في المتصفح أو اطبعه إلى PDF من المتصفح. يبقى نص الاستنتاج مرتبطًا بفرضيات الاختبار وحدوده.

## 3. توليد HTML لاستخدامه في تطبيق آخر

```python
from khaled_dsml import normality_test, render_test_report

report_html = render_test_report(normality_test([1, 2, 3, 4, 5]), digits=4)
```

**وظائف الوحدة `reports`:**

- `render_test_report(result_or_results, title=..., digits=4)` يعيد مستند HTML كاملًا مشفّرًا UTF-8.
- `display_test_report(result_or_results, ...)` يعرض النتيجة في Colab/Jupyter، ويتطلب IPython.
- `save_test_report(result_or_results, path, ...)` يحفظ الملف محليًا دون الاتصال بالإنترنت.

يمكن تمرير نتيجة اختبار واحدة، أو مجموعة نتائج في قائمة. القيم الاحتمالية الصغيرة تعرض كمتباينة تقريبية لا كصفر، وتُعقّم النصوص المضافة إلى HTML لمنع تفسيرها كشيفرة.

## 4. حدود التفسير

- `p-value` لا تمثل احتمال صحة الفرضية الصفرية.
- عدم رفض `H0` لا يثبت صحتها.
- التقرير لا يُجري الاختبارات تلقائيًا ولا يتحقق من ملاءمة نموذج السحب أو استقلالية المشاهدات؛ هذه مسؤولية الباحث.
- التقارير تحتوي نصوص النتائج الرقمية والاستنتاجات فقط، ولا تتضمن بيانات العينة الأصلية إلا إن أدخلها المطور في حقول النصوص.
