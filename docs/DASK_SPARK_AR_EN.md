# Dask وApache Spark مع KhaledDSML | Bilingual Guide

**المؤلف / Author:** الأستاذ الدكتور خالد علي (Dr. Khaled Ali)  
**الجامعة / University:** جامعة الشهيد حمه لخضر – الوادي، الجزائر / University of El Oued, Algeria  
**البريد المهني / Institutional email:** ali-khaled@univ-eloued.dz

> هذه الوظائف في فرع تطوير `feature/rtl-reporting-v0.2.0`. إصدار PyPI الرسمي لا يزال `0.1.0`. لا تنشر الفرع أو تمزجه مع `main` قبل موافقة المؤلف.

## 1. الفرق بين Dask وSpark | What is the difference?

- **Dask**: يعمل على ملفات CSV أكبر من الذاكرة بتقسيم البيانات إلى أجزاء وينفذ العمليات مؤجلًا. `compute=True` يحوّل **الناتج المجمع الصغير فقط** إلى pandas، وقد يكبر الناتج إذا كثرت المجموعات.
- **Spark**: محرك معالجة موزعة باستخدام JVM. تظل نواتج `spark_group_summary` و`spark_monthly_summary` في Spark DataFrame. لا تجرِ `toPandas()` على البيانات الخام الكبيرة.
- **Dask** partitions work and can execute lazily across cores or a cluster. **Spark** uses a JVM-backed execution engine and can work locally or on a configured distributed cluster. These wrappers do not provision or benchmark a production cluster.

## 2. تثبيت Windows / Local Python

يُفضَّل استخدام بيئة افتراضية. Python 3.13 مدعوم في CI، ولا يلزم تثبيت المحركين إذا استُخدم واحد فقط.

```powershell
py -3.13 -m venv "$HOME\KhaledDSML-bigdata"
& "$HOME\KhaledDSML-bigdata\Scripts\python.exe" -m pip install --upgrade pip
& "$HOME\KhaledDSML-bigdata\Scripts\python.exe" -m pip install "git+https://github.com/noble2013/KhaledDSML.git@feature/rtl-reporting-v0.2.0"
& "$HOME\KhaledDSML-bigdata\Scripts\python.exe" -m pip install "dask[dataframe]" pyspark
```

لتشغيل Spark يجب تثبيت Java متوافقة مع نسخة PySpark وضبط JAVA_HOME عند الحاجة. استُخدمت Java 17 في اختبار CI على Linux. قد تختلف متطلبات Java وفق إصدار Spark وWindows.

**English:** Install the development package and optional engines in a dedicated virtual environment. PySpark needs a supported Java runtime.

## 3. تثبيت Google Colab

شغّل الخلايا التالية أولًا:

```python
%pip install --no-cache-dir --force-reinstall --no-deps "git+https://github.com/noble2013/KhaledDSML.git@feature/rtl-reporting-v0.2.0"
%pip install "dask[dataframe]" pyspark
```

أعد تشغيل جلسة Colab ثم اختبر:

```python
import dask, pyspark, khaled_dsml
print("KhaledDSML:", khaled_dsml.__version__)
print("Dask:", dask.__version__)
print("Spark:", pyspark.__version__)
```

يمكن أن تتطلب بعض بيئات Colab ضبط إصدار Java المتاح إلى إصدار متوافق. **The runtime must provide a compatible Java runtime for PySpark.**

## 4. بيانات اقتصادية تعليمية | Economic sample dataset

```python
import pandas as pd

sample = pd.DataFrame({
    "wilaya": ["El Oued", "Algiers", "El Oued", "Oran", "El Oued"],
    "montant_da": [100.0, 60.0, 200.0, 90.0, None],
    "date": ["2026-01-02", "2026-01-05", "2026-02-05",
             "2026-02-09", "2026-02-12"]
})
sample.to_csv("transactions.csv", index=False)
```

## 5. تحليل Dask

```python
from khaled_dsml import (
    dask_read_csv, dask_group_summary,
    dask_missingness_report, dask_monthly_summary,
)

df = dask_read_csv("transactions.csv", blocksize="64MB")
lazy_result = dask_group_summary(df, group_by="wilaya", amount_column="montant_da")
print(type(lazy_result))  # dask.dataframe.DataFrame
display(lazy_result.compute())  # grouped summary only
display(dask_missingness_report(df))
display(dask_monthly_summary(df, date_column="date", compute=True))
```

يمكن اختيار `blocksize` أصغر حسب ذاكرة الجهاز. `dask_read_csv` لا يجعل قراءة كل أنواع بيانات CSV آمنة تلقائيًا؛ يجب ضبط `dtype` والتواريخ عند الحاجة.

**English:** `dask_group_summary` and `dask_monthly_summary` return lazy Dask DataFrames by default. Setting `compute=True` materializes only the grouped output.

## 6. Apache Spark

```python
from khaled_dsml import (
    create_spark_session, spark_read_csv, spark_group_summary,
    spark_missingness_report, spark_monthly_summary,
)

spark = create_spark_session(app_name="KhaledDSML-Teaching", master="local[2]")
try:
    df = spark_read_csv(spark, "transactions.csv", header=True, infer_schema=True)
    spark_group_summary(df, group_by="wilaya", amount_column="montant_da").show()
    display(spark_missingness_report(df))
    spark_monthly_summary(df, date_column="date").show()
finally:
    spark.stop()
```

ناتج `spark_group_summary` و`spark_monthly_summary` هو **Spark DataFrame**، والتقييم الفعلي يحدث عند تنفيذ إجراء مثل `.show()`. أما `spark_missingness_report` فيرجع جدول pandas صغيرًا يُجمع بتجميع واحد على Spark.

**English:** The Spark group and monthly functions remain in Spark. Only the aggregate missingness report collects a compact result.

## 7. سلامة البيانات والتفسير | Data integrity and interpretation

- `count` في التجميع يعني عدد قيم `montant_da` غير المفقودة، وليس بالضرورة عدد جميع الصفوف.
- التواريخ المفقودة أو غير القابلة للتحويل تُستبعد من التجميع الشهري.
- في Spark: `NULL` يُحسب ضمن المفقودات، لكن `NaN` والسلسلة الفارغة قد تحتاجان معالجة مستقلة.
- يجب تطبيع المنطقة الزمنية وتعريف `schema` قبل تطبيق العمليات الإنتاجية.
- التجميع وحده وصفي ولا يثبت علاقة سببية.
- الاختبارات داخل CI شغّلت المحركين على بيانات محلية صغيرة؛ **لا تعني اعتمادًا على عنقود متعدد الأجهزة أو اختبار أداء بملايين السجلات**.

**English:** Counts exclude null monetary values. Null-date rows are omitted from monthly aggregation. Correct time zones and an explicit schema are important for production. A successful local integration test does not certify production-scale cluster performance.
