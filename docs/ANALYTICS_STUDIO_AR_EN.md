# KhaledDSML Analytics Studio — تقارير إحصائية إبداعية ثنائية اللغة

**الإصدار:** 0.3.0 قيد التطوير / Unreleased 0.3.0  
**الانتماء:** جامعة الشهيد حمه لخضر، الوادي — University of El Oued

## فلسفة التصميم | Design principles

صُممت هذه الواجهة للأستاذ والطالب والباحث: هدوء بصري مع حضور علمي قوي. ليست مجرد طباعة نصوص؛ إنها لوحة تقارير بحثية تعرض مؤشرات محسوبة فعلًا، وعناوين واضحة، ومخرجات عربية RTL وإنجليزية LTR لكل تحليل.

The dashboard is a **self-contained HTML document**. It needs no external fonts, CDN, JavaScript, subscription, or cloud service. All dynamic user text is HTML-escaped. Its styles are isolated from the wider notebook; charts use inline SVG built only from the observations explicitly provided by the researcher.

### الأنماط المتاحة / Styles

| النمط / Style | العرض / Appearance | الاستخدام / Usage |
|---|---|---|
| `studio` **(افتراضي / Default)** | رأس متدرج + بطاقات قيم + شرح مزدوج | نتائج التدريس والبحث |
| `plain` | عرض نصي بلا بطاقات | الأسلوب السابق حسب الصورة |
| `theme="aurora"` | أزرق بترولي وفيروزي | النمط الموصى به |
| `theme="ocean"` | أزرق بحري هادئ | تحليلات اقتصادية |
| `theme="amethyst"` | أرجواني أكاديمي | العروض الجامعية |

## التثبيت التجريبي في Colab / Development setup

```python
%pip install --upgrade statsmodels
%pip install --no-cache-dir --force-reinstall --no-deps "git+https://github.com/noble2013/KhaledDSML.git@feature/v0.3-timeseries-reproducibility"
```

أعد تشغيل الجلسة: **Runtime → Restart session**، ثم أعد تشغيل خلايا الاستيراد والتحليل فقط.

## 1. لوحة إبداعية بنتائج متعددة

```python
from khaled_dsml import (
    mann_kendall_trend_test, pettitt_change_point_test,
    forecast_accuracy_report, display_bilingual_results,
)

series = [100, 106, 111, 119, 127, 136, 141, 153, 160, 172, 181, 196]
trend = mann_kendall_trend_test(series)
change = pettitt_change_point_test([12] * 20 + [30] * 20)
forecast = forecast_accuracy_report(
    [205, 215, 225], [203, 218, 222],
    [100, 110, 120, 130, 140, 150],
)

display_bilingual_results(
    [trend, change, forecast],
    style="studio",
    theme="aurora",
    font_size=21,
    chart_data=series,
)
```

**معنى الرسم / Meaning of the chart:** الرسم يُظهر القيم المرصودة حسب ترتيب إدخالها فقط. لا يستنتج اتجاهًا أو سببية تلقائيًا. نتائج الاختبارات مأخوذة من وظائف التحليل الإحصائي.

## 2. التقرير الأكاديمي المفصل

```python
from khaled_dsml import (
    render_analysis_dashboard, display_analysis_dashboard,
    save_analysis_dashboard,
)

display_analysis_dashboard(
    [trend, change, forecast],
    title_ar="دراسة تغير المبيعات الشهرية",
    title_en="MONTHLY SALES · EVIDENCE & INSIGHTS",
    theme="ocean",
    chart_data=series,
)

path = save_analysis_dashboard(
    [trend, change, forecast],
    "economic_analysis_dashboard.html",
    title_ar="التقرير الإحصائي للمبيعات",
    theme="aurora",
    chart_data=series,
)
print(path)
```

من المتصفح يمكنك فتح ملف HTML ثم **طباعة → حفظ كـ PDF**، ويُطبق نمط طباعة أخف لتوفير الحبر. لا يُنشأ ملف PDF مباشرة بهذه الدوال.

## 3. مقارنة الألوان / Compare themes

```python
from khaled_dsml import display_analysis_dashboard

for theme in ("aurora", "ocean", "amethyst"):
    display_analysis_dashboard(trend, theme=theme, chart_data=series)
```

## 4. الرجوع إلى التنسيق القديم / Classic look

```python
from khaled_dsml import display_bilingual_results

display_bilingual_results([trend, change, forecast], style="plain", font_size=24)
```

يبقى النمط القديم متاحًا لتسهيل الاستخدام على الأجهزة محدودة المساحة أو عند الرغبة في عرض نصي فقط.

## 5. الإحصاء الوصفي وتعلم الآلة

```python
from khaled_dsml import (
    describe_numeric, binary_classification_metrics, display_analysis_dashboard
)

descriptive = describe_numeric([12, 14, 14, 18, 22, 25, 30])
classification = binary_classification_metrics(
    [0, 0, 1, 1], [.10, .40, .35, .80]
)
display_analysis_dashboard([descriptive, classification], theme="amethyst")
```

لا تُعرض القيم غير الموجودة في نتيجة الاختبار على أنها قياسات. ولا تُستخدم ألوان النجاح أو التحذير بوصفها حكمًا مستقلًا على صحة نموذج علمي. A metric is shown only if the algorithm actually returned it. Rejecting or not rejecting a hypothesis does not imply causality or proof of the null.

## 6. دفتر Colab جاهز

[06_analytics_studio.ipynb](../notebooks/06_analytics_studio.ipynb)

## ملاحظات منهجية مهمة | Scientific caveats

- `p-value` لا تمثل احتمال صحة فرضية العدم.
- `MASE` يحتاج سلسلة تدريب سابقة على فترة التنبؤ، ومقياسًا ساذجًا غير صفري.
- قياسات ROC-AUC تخص ترتيب التصنيف الثنائي، ولا تقيس المعايرة.
- الرسوم المصغّرة توضح القيم المدخلة دون إضافة تقديرات مصطنعة.
- التحاليل تحتاج فحص افتراضات الاختبارات، وحجم العينة، وتصميم التقسيم الزمني.
