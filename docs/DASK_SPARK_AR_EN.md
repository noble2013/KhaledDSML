# KhaledDSML — Dask & Apache Spark | تحليل البيانات الضخمة

**الإعداد / Author:** الأستاذ الدكتور خالد علي (Dr. Khaled Ali)  
**University of El Oued — Faculty of Economic, Commercial and Management Sciences**

**النسخة:** 0.2.0 التطويرية فقط، فرع `feature/rtl-reporting-v0.2.0`. إصدار PyPI الرسمي هو `0.1.0`.  
**Version:** Development branch only; not yet published as 0.2.0 on PyPI.

## لماذا محركان؟ / Why two engines?

| Dask | Apache Spark |
|---|---|
| قريب من pandas ويناسب الدروس على حاسوب واحد أو عنقود | إطار عمل للبيانات الموزعة عبر JVM مع Python |
| Lazy DataFrame / تنفيذ مؤجل | Spark DataFrame / تنفيذ مؤجل |
| قراءة CSV بالأجزاء / Partitioned CSV | معالجة CSV وParquet وتوزيع SQL |
| لا يحتاج Java | يحتاج Java متوافقًا مع إصدار PySpark |
| مجاني ومفتوح المصدر | مجاني ومفتوح المصدر |

**Important:** Neither engine requires a paid cloud service. A local computer or Google Colab can run small exercises. Distributed processing does not guarantee that every dataset will fit in RAM; avoid collecting huge raw datasets into pandas.

## التثبيت / Install

### في Google Colab — تطوير 0.2.0

```python
%pip install "dask[dataframe]" pyspark
%pip install --no-deps --no-cache-dir --force-reinstall "git+https://github.com/noble2013/KhaledDSML.git@feature/rtl-reporting-v0.2.0"
```

**أعد تشغيل Runtime بعد التثبيت.** Colab may supply compatible Java; starting Spark requires a Java runtime supported by the installed PySpark version. Spark startup depends on platform settings and is not guaranteed in every hosted session.

### في Windows PowerShell — Python 3.13

```powershell
py -3.13 -m pip install "dask[dataframe]" pyspark
py -3.13 -m pip install --upgrade git+https://github.com/noble2013/KhaledDSML.git@feature/rtl-reporting-v0.2.0
```

ثبت Java الملائم لإصدار PySpark وتحقق من `java -version`. استخدم `JAVA_HOME` عند الحاجة. **Dask لا يتطلب Java.**

### تثبيت اختياري من ملف المشروع بعد دمج الإصدار

```powershell
py -3.13 -m pip install ".[dask]"
py -3.13 -m pip install ".[spark]"
py -3.13 -m pip install ".[distributed]"
```

هذه الصيغة تتطلب وجود `pyproject.toml` في المجلد الحالي.

## Dask — مثال اقتصادي / Economic example

```python
import pandas as pd
from khaled_dsml import (
    dask_read_csv, dask_group_summary, dask_missingness_report, dask_monthly_summary,
)

transactions = pd.DataFrame({
    "wilaya": ["El Oued", "Algiers", "El Oued", "Oran"],
    "montant_da": [1000.0, 2000.0, 1800.0, None],
    "date": ["2026-01-04", "2026-01-16", "2026-02-08", "2026-02-14"],
})
transactions.to_csv("transactions_small.csv", index=False)

frame = dask_read_csv("transactions_small.csv", blocksize="16MB")
lazy_by_wilaya = dask_group_summary(frame)   # Dask DataFrame, not pandas
print(lazy_by_wilaya.compute())               # Small aggregate only

display(dask_missingness_report(frame))
display(dask_monthly_summary(frame, compute=True))
```

**التفسير / Interpretation:** `count` counts valid non-null amounts, `sum` is the total, and `mean` is the arithmetic mean. `dask_group_summary` and `dask_monthly_summary` default to lazy execution. `dask_csv_summary` is kept for backwards compatibility and returns a computed small pandas group table.

يمكنك تحديد `blocksize` بحسب ذاكرة الجهاز؛ تجنّب `.compute()` للبيانات الخام الكبيرة. Invalid dates are omitted from monthly grouping; use appropriate timezone and date format for real sources.

## Apache Spark — المثال نفسه / Same analysis in Spark

```python
from khaled_dsml import (
    create_spark_session, spark_read_csv, spark_group_summary,
    spark_missingness_report, spark_monthly_summary,
)

spark = create_spark_session(app_name="KhaledDSML-Classroom", master="local[2]")
sdf = spark_read_csv(spark, "transactions_small.csv")

by_wilaya = spark_group_summary(sdf)
by_wilaya.show()                              # Displays aggregate only
display(spark_missingness_report(sdf))       # Collects compact counts only
spark_monthly_summary(sdf).show()

spark.stop()
```

**التفسير / Interpretation:** Spark keeps the raw DataFrame distributed; `spark_group_summary` and `spark_monthly_summary` return Spark DataFrames and do not call `toPandas()`. For real production CSV, supply an explicit Spark schema rather than repeatedly scanning data for type inference. `spark_missingness_report` counts SQL NULL (not NaN or empty strings); clean such values explicitly when needed.

## ضبط الأداء والحدود / Performance and boundaries

- **Dask:** القراءة من CSV مقسمة، وتجميع lazily؛ `compute=True` يعيد *الجداول المجمعة الصغيرة فقط* إلى pandas.
- **Spark:** التشغيل المحلي `local[2]` لا يعني وجود عنقود متعدد الأجهزة؛ لا تدّع أن التجربة موزعة على خوادم.
- **Schema:** المبلغ يجب أن يكون رقمياً في كلا المحركين؛ أدخل تنظيف تحويل الأنواع قبل التجميع. Numeric amount missing values are excluded from `count`.
- **Time:** تواريخ Dask تجمع حسب UTC في هذا المثال، وSpark يجمع تواريخ `yyyy-MM-dd` ما لم تغير `date_format`. Normalize timestamps in real projects.
- **Nulls:** `dask_missingness_report` يستخدم `isna`، بينما `spark_missingness_report` يعدّ SQL NULL فقط. لذلك قد تختلف نتائج القيم `NaN` وعبارات النص الفارغة.
- **Resources:** Spark يحتاج Java، وقد يستهلك ذاكرة/وقت بدء أكبر؛ Dask مناسب للمقارنة التعليمية مع pandas.
- **Validation:** اختبارات GitHub تشمل تشغيل Dask وSpark الحقيقيين على CSV صغير، وتحقق تجميع حسب الشهر والولاية وقيم الفقد. ليست اختبار أداء على عنقود واسع النطاق.

## وظائف API الجديدة / API reference

| Dask | Spark |
|---|---|
| `dask_read_csv` | `create_spark_session`, `spark_read_csv` |
| `dask_group_summary` | `spark_group_summary` |
| `dask_missingness_report` | `spark_missingness_report` |
| `dask_monthly_summary` | `spark_monthly_summary` |
| `dask_csv_summary` (متوافق مع القديم) | `spark_csv_summary` |

مصادر علمية / Official references: [Dask DataFrame](https://docs.dask.org/en/stable/dataframe.html), [PySpark SQL](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/index.html).
