# KhaledDSML 0.7 — PySpark + Dask Distributed Analytics

**Status / الحالة:** In development; not yet published on PyPI.  
**Published stable version:** `0.6.0`.  
**Author:** Dr. Khaled Ali.

## Is PySpark part of the library? | هل أُضيف PySpark؟

**نعم.** PySpark optional extra has been supported since 0.2.0, and this 0.7 development expands it with distributed data quality, chronological splits and Spark ML. Base installation does **not** install PySpark, Java, Spark or Dask.

| Install | Functionality |
|---|---|
| `pip install "khaled-dsml[spark]"` | PySpark engine; Java required |
| `pip install "khaled-dsml[dask]"` | Dask lazy dataframes |
| `pip install "khaled-dsml[distributed]"` | Both Spark and Dask |
| `pip install "khaled-dsml"` | Lightweight core; no Spark/Dask |

**Production rule:** Spark transformations return Spark DataFrames without collecting raw rows. Small aggregate reports, training model objects, and evaluator scores require computing Spark actions. Spark may process a very large dataset distributed across executors, depending on the installation, partitioning, hardware, cluster configuration and network.

## New 0.7 functions | وظائف 0.7 الجديدة

| API | Input → output | Notes |
|---|---|---|
| `spark_missingness_detailed(frame)` | Spark DF → compact pandas DataFrame | SQL NULL, floating NaN, blank text counted separately |
| `spark_duplicate_summary(frame, subset=None)` | Spark DF → small dictionary | Two Spark actions, no raw collect |
| `spark_numeric_profile(frame,columns=None)` | Spark DF → compact pandas DataFrame | Mean, standard deviation, min, max, counts; skip NULL/NaN/infinity |
| `spark_time_holdout(frame,date_column,cutoff)` | Spark DF → 3 lazy Spark DFs | Train before cutoff, test at/after, malformed dates preserved in invalid |
| `spark_fit_logistic_classifier(train_df,...)` | Train Spark DF → fitted PipelineModel | Logistic regression only on train; no implicit shuffle |
| `spark_binary_classification_metrics(pred_df)` | Predicted Spark DF → compact dictionary | ROC-AUC, PR-AUC and confusion-derived metrics |
| `dask_duplicate_summary(source)` | CSV or Dask DF → dictionary | Exact duplicate count |
| `dask_numeric_profile(source)` | CSV or Dask DF → compact pandas DataFrame | Aggregated numeric stats |
| `dask_time_holdout(source, cutoff)` | CSV or Dask DF → 3 lazy Dask DFs | Chronological split and invalid-date partition |

## Google Colab

In Google Colab, install from the development branch while 0.7 is still unreleased:

```python
%pip install --upgrade "khaled-dsml[distributed]==0.6.0"
%pip install --no-deps --force-reinstall --no-cache-dir "git+https://github.com/noble2013/KhaledDSML.git@feature/v0.7-distributed-analytics"
```

Restart runtime, and ensure the Colab runtime provides a Java installation compatible with PySpark. For local Windows Python, use the same `[spark]` extra on the **published** version until 0.7 is officially released.

## Spark data quality and reporting

```python
from khaled_dsml import (
    spark_missingness_detailed, spark_numeric_profile,
    spark_duplicate_summary, display_analysis_dashboard
)
from khaled_dsml.distributed import create_spark_session

spark=create_spark_session(master="local[2]")
df=spark.createDataFrame([
    ("الوادي",100.,"2026-01-10"),
    ("الوادي",100.,"2026-01-10"),
    ("بسكرة",225.,"2026-02-11")
],["wilaya","montant_da","date"])
missing=spark_missingness_detailed(df)
numeric=spark_numeric_profile(df,columns=["montant_da"])
duplicates=spark_duplicate_summary(df)
display(missing,numeric)
display_analysis_dashboard(
    duplicates,
    title_ar="تقرير جودة السجلات المالية",
    title_en="BIG DATA · DUPLICATE RECORDS",
    theme="aurora",font_size=22
)
```

## Chronological split / فصل زمني دون تسرب

```python
from khaled_dsml import spark_time_holdout

partitions=spark_time_holdout(df,date_column="date",cutoff="2026-02-01")
train=partitions["train"]
test=partitions["test"]
invalid=partitions["invalid"]
print(train.count(),test.count(),invalid.count())
```

The user must not fit preprocessors/statistics on combined train+test; prevent information leakage by fitting transforms on train alone.

## Distributed Spark ML model evaluation

```python
import numpy as np
from khaled_dsml import (
    spark_fit_logistic_classifier,
    spark_binary_classification_metrics,
    display_analysis_dashboard,
)

rng=np.random.default_rng(24)
rows=[(i, float(x),float(x>0)) for i,x in enumerate(rng.normal(size=200))]
records=spark.createDataFrame(rows,["record_id","predictor","label"])
train=records.where("record_id < 140")
test=records.where("record_id >= 140")  # illustration: IDs follow known ordering
model=spark_fit_logistic_classifier(train,feature_columns=["predictor"])
prediction=model.transform(test)
metrics=spark_binary_classification_metrics(prediction)
display_analysis_dashboard(
    metrics,
    title_ar="نتائج نموذج تصنيف موزّع",
    title_en="SPARK ML · BINARY VALIDATION",
    theme="ocean",font_size=21
)
```

**In actual studies**, do not use `limit` plus `subtract` as a validation design. Create an explicit patient/student/transaction ID or chronological cutoff. Never report a training score as held-out performance. Binary ROC/PR scores require both actual classes in the test data.

## Dask datasets

```python
from khaled_dsml import (
    dask_duplicate_summary,
    dask_numeric_profile,
    dask_time_holdout
)

print(dask_duplicate_summary("transactions.csv"))
display(dask_numeric_profile("transactions.csv"))
parts=dask_time_holdout("transactions.csv",date_column="date",cutoff="2026-06-01")
print(parts["train"].shape[0].compute())
```

## Scientific and operational safeguards

- Arabic interpretation: RTL and right-aligned. English interpretation: LTR and left-aligned in **Analytics Studio**.
- No download of full Spark/Dask DataFrames into local pandas occurs in profiling functions.
- Spark numeric reports explicitly ignore infinite/NaN/NULL numeric values; always report the valid n.
- Spark duplicate counts can require a full distributed shuffle; no promise of zero resource cost.
- There is no automatic imputation, dropping of observations, or hidden removal of invalid dates.
- The exported HTML report is offline-capable and supports the established Aurora, Ocean and Amethyst styles.
- 0.7 extends a documented subset; it is **not** proof that all 256 catalog items are complete.
