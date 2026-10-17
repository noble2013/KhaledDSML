# السلاسل الزمنية وقابلية إعادة إنتاج البحث | Time-series & Reproducibility

**KhaledDSML 0.3.0 — نسخة تطويرية لم تُنشر بعد / unreleased development version**

**Author / المؤلف:** الأستاذ الدكتور خالد علي (Dr. Khaled Ali), University of El Oued, Algeria.  
**Faculty / الكلية:** العلوم الاقتصادية والتجارية وعلوم التسيير.

> الإصدار الرسمي المنشور على PyPI هو `khaled-dsml==0.2.0`. الوظائف أدناه توجد فقط على فرع `feature/v0.3-timeseries-reproducibility`. لا تكتب `pip install khaled-dsml==0.3.0` قبل صدور الإصدار رسميًا.

## التثبيت التجريبي في Google Colab | Development installation

```python
%pip install --upgrade statsmodels
%pip install --no-deps --force-reinstall --no-cache-dir "git+https://github.com/noble2013/KhaledDSML.git@feature/v0.3-timeseries-reproducibility"
```

ثم **Runtime → Restart session**. لا تستعمل دفترًا قديماً يحوي خلية تعيد تثبيت الإصدار 0.2.0.

## اختبارات الاتجاه والانقطاع | Trend & change detection

```python
from khaled_dsml import (
    mann_kendall_trend_test, pettitt_change_point_test,
    display_bilingual_result,
)

monthly_sales = [95, 101, 103, 105, 112, 117, 122, 120, 127, 131, 137, 145]
trend = mann_kendall_trend_test(monthly_sales)
print(trend.interpretation_ar)
print(trend.interpretation_en)

series_with_shift = [10] * 20 + [25] * 20
change = pettitt_change_point_test(series_with_shift)
print(change["change_index"], change["p_value"])
print(change["interpretation_ar"])
print(change["interpretation_en"])
```

- **Mann–Kendall:** اختبار اتجاه رتيب عبر الزمن، باستخدام تقريب طبيعي وتصحيح التكرارات (ties). لا يصح افتراض استقلال القيم في وجود موسمية أو ارتباط ذاتي. / A monotonic-trend test using a tie-adjusted asymptotic approximation, not valid with unaddressed serial dependence or seasonality.
- **Pettitt:** يحدد أقوى تقسيم تقريبي لسلسلة إلى جزأين. `change_index` فهرس يبدأ من **صفر** ويدل على *آخر مشاهدة قبل نقطة التحوّل*. القيمة الاحتمالية تقريبية وتفترض الاستقلال. / The change index is zero-based and marks the final point of the earlier segment, not a calendar timestamp.

## اختبار بواقي النماذج | Model residual diagnostics

```python
import numpy as np
from khaled_dsml import box_pierce_test, arch_lm_test

rng = np.random.default_rng(21)
residuals = rng.normal(size=160)
print(box_pierce_test(residuals, lags=8).interpretation_en)
print(arch_lm_test(residuals, lags=5).interpretation_ar)
```

هاتان الدالتان تحتاجان `statsmodels` (الخيار `khaled-dsml[econometrics]`). **Box–Pierce** يقيس الارتباط الذاتي عبر الإبطاءات المحددة، بينما **ARCH-LM** يختبر نمط الاعتماد في مربعات البواقي؛ يجب أن تنشأ المدخلات من نموذج زمني ملائم. / Both tests use asymptotic p-values and depend on appropriate model residuals. They do **not** demonstrate causality.

## تقييم جودة التنبؤ دون تسرب | Leakage-safe forecasting evaluation

```python
from khaled_dsml import (
    mean_absolute_scaled_error, forecast_accuracy_report,
    display_bilingual_result,
)

train = [100, 110, 120, 130, 140, 150, 160]
actual = [170, 180, 190]
predicted = [168, 182, 189]

mase = mean_absolute_scaled_error(actual, predicted, train, seasonality=1)
print(mase)
report = forecast_accuracy_report(actual, predicted, train)
print(report["interpretation_ar"])
print(report["interpretation_en"])
```

`MASE` يقسّم متوسط الخطأ المطلق للتنبؤ على متوسط خطأ التنبؤ الساذج من **عينة التدريب فقط**؛ إذا كانت تلك القيمة صفرًا فالمقياس غير معرّف ويظهر خطأ واضح. يجب حفظ الترتيب الزمني وتساوي أفق التنبؤ في المقارنة. / Baseline is computed **only from pre-forecast training history**. If the baseline denominator vanishes, MASE is undefined.

`sMAPE` هنا يستخدم الصيغة `200 * abs(y-pred)/(abs(y)+abs(pred))` بالمئة؛ إذا كانت القيمتان صفرًا يُحسب الحد بصفر وفق اصطلاح الدالة. هذا الاصطلاح ينبغي التصريح به عند نشر النتائج. / sMAPE is convention-sensitive; the zero/zero term contributes zero in this API.

## اتجاه الخطوط في Jupyter وColab | Readable Arabic / English display

`print()` يعرض العربية باتجاه الطرفية وقد يجعل الخط صغيرًا والمحاذاة من اليسار. لذلك استخدم `display_bilingual_result()` لجميع اختبارات المكتبة التي ترجع `TestResult` أو قاموسًا يحوي التفسير بالعربية والإنجليزية. العربية RTL بمحاذاة اليمين بخط عريض، والإنجليزية LTR بمحاذاة اليسار.

The renderer escapes untrusted text, isolates each language's direction and uses a larger font in notebook cells. It changes **only** the report panel, not the entire Colab page.

```python
from khaled_dsml import display_bilingual_result, save_bilingual_result

# Compatible with both TestResult objects and bilingual dictionaries
display_bilingual_result(trend, font_size=23)
display_bilingual_result(change)
display_bilingual_result(report, title="تقييم التنبؤ | Forecast evaluation")

# Portable HTML report, suitable for browser printing
save_bilingual_result(trend, "trend_ar_en.html")
```

استخدم `font_size=21` افتراضيًا، أو `font_size=23` للعرض في قاعة الدرس. / Default font size is 21 px; 23 px is suitable for classroom presentation.

## تتبع البيانات والبيئة | Dataset integrity and software provenance

```python
from khaled_dsml import (
    reproducibility_manifest,
    save_reproducibility_manifest,
    verify_reproducibility_manifest,
)

with open("sales_data.csv", "w", encoding="utf-8") as f:
    f.write("month,sales\n2026-01,100\n2026-02,110\n")

manifest = reproducibility_manifest(["sales_data.csv"], seed=42)
save_reproducibility_manifest(manifest, "research_manifest.json")
print(verify_reproducibility_manifest(manifest, ["sales_data.csv"]))
```

يحتفظ ملف JSON ببصمة SHA-256، وحجم كل ملف، واسمه دون مسارات المستخدم المطلقة، وإصدارات Python والمكتبات المهمة، مع رقم `seed` اختياري. يتجنب الطابع الزمني افتراضيًا حتى تبقى المخرجات متطابقة عند إعادة التشغيل في البيئة نفسها.

**تنبيه مهم / Important:** التطابق في SHA-256 يثبت أن البايتات نفسها بقيت كما هي في الظروف الطبيعية، لكنه لا يثبت صحة البحث أو نزاهة المصدر، ولا يحمي من التزوير إذا أمكن تعديل الملف وبصمته معًا. / File hashes check byte identity, not the statistical validity or authenticity of research. A trusted signature or independent archive is needed for tamper-evidence.

## خريطة التغطية | Catalogue roadmap

هذه المرحلة تغطي مجموعة من البنود غير المنفذة في `timeseries` و`forensics` من كتالوج 256 بندًا. لا نزعم تنفيذ اختبارات Granger أو Johansen أو DeLong أو PELT أو SEM أو الاختبارات الجنائية المتقدمة. تظل باقي البنود ضمن خريطة الطريق حتى تنال اختبارات مرجعية مستقلة.

## متطلبات الاستخدام | Quality requirements

- لا تزيل القيم المفقودة من السلاسل الزمنية بصمت؛ اتبع معالجة تحافظ على الترتيب.
- لا تفسر `p_value` على أنها احتمال صحة الفرضية الصفرية.
- وثّق الموسمية والارتباط الذاتي، وتقسيم التدريب والاختبار، وعدد الإبطاءات ومستوى الدلالة.
- استخدم أمثلة اصطناعية للتدريب؛ لا تعتمد عليها لاتخاذ قرارات اقتصادية فعلية.
