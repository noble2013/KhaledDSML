"""Scalable, optional PySpark and Dask analysis and ML without raw-data collection.

Spark operations run on Spark DataFrames, and Dask operations on lazy
Dask frames. Only compact statistics/models may be materialized locally.
No data is uploaded to third-party services or storage by these helpers.
"""
from __future__ import annotations

from datetime import date
import math

from .distributed import _spark, _dask, _as_dask_frame, _as_spark_frame, _require_columns


def _names(frame, requested):
    if len(frame.columns) != len(set(frame.columns)):
        raise ValueError("Column names must be unique")
    if requested is None:
        return list(frame.columns)
    cols = [requested] if isinstance(requested, str) else list(requested)
    if not cols or len(set(cols)) != len(cols):
        raise ValueError("Specify one or more unique columns")
    _require_columns(frame, cols)
    return cols


def _date_cutoff(cutoff):
    if not isinstance(cutoff, str):
        raise TypeError("cutoff must be ISO-formatted YYYY-MM-DD")
    try:
        value = date.fromisoformat(cutoff)
    except ValueError as exc:
        raise ValueError("cutoff must be a valid ISO YYYY-MM-DD date") from exc
    if value.isoformat() != cutoff:
        raise ValueError("cutoff must use exact YYYY-MM-DD format")
    return cutoff


def spark_missingness_detailed(spark_df):
    """One compact aggregation for SQL null, floating NaN and blank text.

    Counts each category separately and reports missing_count as their sum;
    categories are disjoint. Does not pull individual Spark rows into Python.
    """
    import pandas as pd
    F = _as_spark_frame(spark_df)
    from pyspark.sql.types import FloatType, DoubleType, StringType
    names = _names(spark_df, None)
    if not names:
        raise ValueError("Need at least one column")
    expressions = [F.count(F.lit(1)).alias("_rows")]
    for i, name in enumerate(names):
        col = F.col(name)
        dtype = spark_df.schema[name].dataType
        expressions.append(
            F.sum(F.when(col.isNull(), 1).otherwise(0)).alias(f"_null_{i}")
        )
        if isinstance(dtype, (FloatType, DoubleType)):
            expressions.append(
                F.sum(F.when(F.isnan(col), 1).otherwise(0)).alias(f"_nan_{i}")
            )
        if isinstance(dtype, StringType):
            expressions.append(
                F.sum(F.when(F.trim(col) == "", 1).otherwise(0)).alias(f"_blank_{i}")
            )
    record = spark_df.agg(*expressions).first().asDict()
    total = int(record["_rows"])
    records = []
    for i, name in enumerate(names):
        nulls = int(record[f"_null_{i}"] or 0)
        nans = int(record.get(f"_nan_{i}") or 0)
        blanks = int(record.get(f"_blank_{i}") or 0)
        missing = nulls + nans + blanks
        records.append({
            "column": name, "dtype": str(spark_df.schema[name].dataType),
            "null_count": nulls, "nan_count": nans, "blank_count": blanks,
            "missing_count": missing,
            "missing_percent": 100.*missing/total if total else 0.,
        })
    return pd.DataFrame.from_records(records).set_index("column")


def spark_duplicate_summary(spark_df, *, subset=None):
    """Exact duplicate count on selected columns; Spark computes distributed.

    Triggers two Spark actions but never collects individual records.
    Duplicate rows = total rows - distinct tuples of selected columns.
    """
    _as_spark_frame(spark_df)
    names = _names(spark_df, subset)
    if not names:
        raise ValueError("DataFrame must have columns")
    rows = spark_df.count()
    distinct = spark_df.select(*names).distinct().count()
    return {
        "n_samples": int(rows), "n_unique": int(distinct),
        "duplicate_rows": int(rows-distinct), "columns": names,
        "interpretation_ar": "عدد الصفوف المتطابقة وفق الأعمدة المحددة دون نسخ البيانات الأصلية للذاكرة.",
        "interpretation_en": "Exact duplicate rows across selected columns, without collecting raw records.",
    }


def spark_numeric_profile(spark_df, *, columns=None):
    """Compute a compact numeric profile with one Spark aggregation action.

    Null, NaN and positive/negative infinity are excluded from descriptive
    statistics. Each result row corresponds to an input numeric column.
    """
    import pandas as pd
    F = _as_spark_frame(spark_df)
    from pyspark.sql.types import NumericType, FloatType, DoubleType
    names = _names(spark_df, columns)
    names = [n for n in names if isinstance(spark_df.schema[n].dataType, NumericType)]
    if not names:
        raise ValueError("No numeric fields to summarize")
    exprs = [F.count(F.lit(1)).alias("_rows")]
    for i, name in enumerate(names):
        col = F.col(name)
        dtype = spark_df.schema[name].dataType
        valid = col.isNotNull()
        if isinstance(dtype, (FloatType, DoubleType)):
            valid = valid & (~F.isnan(col)) & (col != float("inf")) & (col != -float("inf"))
        safe = F.when(valid, col.cast("double"))
        exprs.extend([
            F.count(safe).alias(f"_n_{i}"),
            F.avg(safe).alias(f"_avg_{i}"),
            F.stddev_samp(safe).alias(f"_std_{i}"),
            F.min(safe).alias(f"_min_{i}"),
            F.max(safe).alias(f"_max_{i}"),
        ])
    values = spark_df.agg(*exprs).first().asDict()
    out=[]
    for i, name in enumerate(names):
        out.append({
            "column": name, "n_samples": int(values["_rows"]),
            "n_valid": int(values[f"_n_{i}"]),
            "mean": float(values[f"_avg_{i}"]) if values[f"_avg_{i}"] is not None else math.nan,
            "std": float(values[f"_std_{i}"]) if values[f"_std_{i}"] is not None else math.nan,
            "min": float(values[f"_min_{i}"]) if values[f"_min_{i}"] is not None else math.nan,
            "max": float(values[f"_max_{i}"]) if values[f"_max_{i}"] is not None else math.nan,
        })
    return pd.DataFrame.from_records(out).set_index("column")


def spark_time_holdout(spark_df, *, date_column="date", cutoff):
    """Lazy, chronological Spark train/test/invalid-date partitions.

    Dates before cutoff are train; dates on/after are test. Invalid/missing
    dates stay in an explicit third DataFrame rather than being silently lost.
    No random shuffle and no raw row collection.
    """
    F = _as_spark_frame(spark_df)
    _require_columns(spark_df, [date_column])
    _date_cutoff(cutoff)
    date_value = F.to_date(F.col(date_column))
    threshold = F.lit(cutoff).cast("date")
    return {
        "train": spark_df.where(date_value < threshold),
        "test": spark_df.where(date_value >= threshold),
        "invalid": spark_df.where(date_value.isNull()),
        "interpretation_ar": "فصل زمني بلا خلط؛ تواريخ غير صالحة في مجموعة مستقلة.",
        "interpretation_en": "Chronological non-shuffled split; malformed/missing dates are retained separately.",
    }


def spark_fit_logistic_classifier(
    train_df, *, feature_columns, label_column="label",
    max_iter=100, reg_param=0., elastic_net_param=0., seed=42,
):
    """Fit Spark ML logistic regression only on the provided training frame.

    Returns fitted PipelineModel, not collected predictions or a claim of
    out-of-sample accuracy. Caller must reserve a separate test frame first.
    """
    _as_spark_frame(train_df)
    from pyspark.sql import functions as F
    from pyspark.sql.types import NumericType
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import LogisticRegression
    fields = _names(train_df, feature_columns)
    if label_column in fields or not fields:
        raise ValueError("Feature fields must be nonempty and exclude target label")
    _require_columns(train_df, [label_column])
    if any(not isinstance(train_df.schema[c].dataType, NumericType) for c in fields+[label_column]):
        raise TypeError("All features and binary target must have numeric Spark types")
    if (type(max_iter) is not int or max_iter < 1 or
        not 0 <= reg_param <= 100 or not 0 <= elastic_net_param <= 1):
        raise ValueError("Invalid Spark logistic regression parameters")
    label_values = [r[0] for r in train_df.select(label_column).distinct().limit(3).collect()]
    if len(label_values) != 2 or set(label_values) != {0, 1}:
        raise ValueError("Training target requires both numeric classes 0 and 1")
    assembler=VectorAssembler(
        inputCols=fields, outputCol="__kdsml_features", handleInvalid="error"
    )
    classifier=LogisticRegression(
        labelCol=label_column, featuresCol="__kdsml_features",
        maxIter=max_iter, regParam=float(reg_param),
        elasticNetParam=float(elastic_net_param),
    )
    model=Pipeline(stages=[assembler,classifier]).fit(train_df)
    return model


def spark_binary_classification_metrics(
    predictions_df, *, label_column="label", prediction_column="prediction",
    raw_prediction_column="rawPrediction",
):
    """Distributed ROC-AUC, PR-AUC, accuracy, precision, recall and F1.

    Counts/confusion matrix are collected as one compact aggregate; ROC/PR
    require Spark evaluators and retain scores on the cluster. No toPandas.
    """
    F = _as_spark_frame(predictions_df)
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    _require_columns(predictions_df, [
        label_column, prediction_column, raw_prediction_column
    ])
    label, pred = F.col(label_column), F.col(prediction_column)
    counts = predictions_df.agg(
        F.count(F.lit(1)).alias("total"),
        F.sum(F.when((label == 1) & (pred == 1),1).otherwise(0)).alias("tp"),
        F.sum(F.when((label == 0) & (pred == 0),1).otherwise(0)).alias("tn"),
        F.sum(F.when((label == 0) & (pred == 1),1).otherwise(0)).alias("fp"),
        F.sum(F.when((label == 1) & (pred == 0),1).otherwise(0)).alias("fn"),
        F.sum(F.when(label.isNull() | pred.isNull() |
                      (~label.isin(0,1)) | (~pred.isin(0,1)),1).otherwise(0)
              ).alias("invalid"),
    ).first().asDict()
    if not counts["total"] or counts["invalid"]:
        raise ValueError("Need nonempty held-out data with 0/1 labels and predictions")
    tp,tn,fp,fn=[int(counts[k] or 0) for k in ("tp","tn","fp","fn")]
    if tp+fn == 0 or tn+fp == 0:
        raise ValueError("Both actual classes are required for ROC-AUC")
    evaluator = BinaryClassificationEvaluator(
        labelCol=label_column, rawPredictionCol=raw_prediction_column,
    )
    auc=float(evaluator.evaluate(predictions_df, {evaluator.metricName:"areaUnderROC"}))
    ap=float(evaluator.evaluate(predictions_df, {evaluator.metricName:"areaUnderPR"}))
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else 0.
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    return {
        "n_samples": int(counts["total"]), "tp":tp,"tn":tn,"fp":fp,"fn":fn,
        "accuracy": (tp+tn)/counts["total"],"precision":precision,
        "recall":recall,"f1":f1,
        "roc_auc":auc,"pr_auc":ap,
        "interpretation_ar": "مقاييس مصنّف ثنائي محسوبة على Spark؛ صلاحية تعميمها تتطلب مجموعة اختبار مستقلة.",
        "interpretation_en": "Binary metrics computed on Spark; generalization requires truly held-out test data.",
    }


def dask_duplicate_summary(source, *, subset=None, blocksize="64MB"):
    """Compute compact exact duplicate counts with Dask; no raw .compute()."""
    frame = _as_dask_frame(source, blocksize)
    names = _names(frame, subset)
    if not names:
        raise ValueError("DataFrame must have at least one column")
    rows = int(frame.shape[0].compute())
    distinct = int(frame[names].drop_duplicates().shape[0].compute())
    return {
        "n_samples":rows,"n_unique":distinct,
        "duplicate_rows":rows-distinct,"columns":names,
        "interpretation_ar": "حساب التكرارات باستخدام Dask، دون تحويل الجدول الكامل إلى pandas.",
        "interpretation_en": "Dask duplicate counts without materializing the entire original DataFrame.",
    }


def dask_numeric_profile(source, *, columns=None, blocksize="64MB"):
    """Compact per-column Dask numeric statistics, streaming aggregate tasks."""
    import pandas as pd
    frame = _as_dask_frame(source, blocksize)
    names = _names(frame, columns)
    numeric=[n for n in names if pd.api.types.is_numeric_dtype(frame[n].dtype)]
    if not numeric:
        raise ValueError("No numeric fields to summarize")
    summary=frame[numeric].agg(["count","mean","std","min","max"]).compute().T
    summary = summary.rename(columns={"count":"n_valid"})
    summary["n_samples"]=int(frame.shape[0].compute())
    return summary[["n_samples","n_valid","mean","std","min","max"]]


def dask_time_holdout(
    source, *, date_column="date", cutoff, blocksize="64MB",
):
    """Lazy Dask chronological split with explicit invalid/missing-date partition."""
    import pandas as pd
    dd=_dask()
    frame=_as_dask_frame(source,blocksize)
    _require_columns(frame,[date_column])
    _date_cutoff(cutoff)
    stamps=dd.to_datetime(frame[date_column],errors="coerce",utc=True)
    date_limit=pd.Timestamp(cutoff).tz_localize("UTC")
    return {
        "train":frame[stamps<date_limit],
        "test":frame[stamps>=date_limit],
        "invalid":frame[stamps.isna()],
        "interpretation_ar": "تقسيم زمني كسول مع مجموعة مخصصة للتواريخ غير الصالحة.",
        "interpretation_en": "Lazy chronological split with an explicit malformed/missing dates partition.",
    }
