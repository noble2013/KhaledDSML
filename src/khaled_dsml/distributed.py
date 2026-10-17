"""Optional big-data integrations for Dask and Apache Spark.

The base KhaledDSML installation does not depend on Dask, PySpark or Java.
Dask functions produce lazy distributed results unless ``compute=True``.
Spark functions operate on Spark DataFrames and never call toPandas/collect
on raw datasets. Aggregating a small quality report is an explicit exception.
"""
from __future__ import annotations

from pathlib import Path


def _dask():
    try:
        import dask.dataframe as dd
    except ImportError as exc:
        raise ImportError(
            'Dask is optional: pip install "khaled-dsml[dask]"'
        ) from exc
    return dd


def _spark():
    try:
        from pyspark.sql import DataFrame as SparkDataFrame, functions as F
    except ImportError as exc:
        raise ImportError(
            'Spark is optional: pip install "khaled-dsml[spark]"; Java is also required'
        ) from exc
    return SparkDataFrame, F


def _require_columns(frame, columns):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")


def dask_read_csv(path, *, blocksize="64MB", **kwargs):
    """Read a CSV lazily; no full pandas DataFrame is created.

    Additional kwargs go to dask.dataframe.read_csv, e.g. dtype or assume_missing.
    The caller can tune blocksize to fit available RAM.
    """
    return _dask().read_csv(str(path), blocksize=blocksize, **kwargs)


def _as_dask_frame(source, blocksize):
    dd = _dask()
    if isinstance(source, dd.DataFrame):
        return source
    if isinstance(source, (str, Path)):
        return dask_read_csv(source, blocksize=blocksize)
    raise TypeError("Expected a Dask DataFrame or CSV file path")


def dask_group_summary(
    source, *, group_by="wilaya", amount_column="montant_da",
    blocksize="64MB", compute=False,
):
    """Aggregate numeric amounts with Dask; return lazy Dask or small pandas table.

    Count refers to nonmissing amount values, not necessarily all group rows.
    Input amounts must be numeric; nonnumeric columns raise a type error.
    """
    import pandas as pd
    frame = _as_dask_frame(source, blocksize)
    _require_columns(frame, [group_by, amount_column])
    if not pd.api.types.is_numeric_dtype(frame[amount_column].dtype):
        raise TypeError(f"{amount_column} must be numeric; provide dtype/read_csv options")
    result = frame.groupby(group_by)[amount_column].agg(["count", "sum", "mean"])
    if compute:
        return result.compute().sort_values("sum", ascending=False)
    return result


def dask_csv_summary(path, *, group_by="wilaya", amount_column="montant_da",
                     blocksize="64MB"):
    """Backward-compatible Dask CSV group summary returning a small pandas table."""
    return dask_group_summary(
        path, group_by=group_by, amount_column=amount_column,
        blocksize=blocksize, compute=True,
    )


def dask_missingness_report(source, *, blocksize="64MB"):
    """Compute only compact row/column missingness statistics, never raw rows."""
    import pandas as pd
    frame = _as_dask_frame(source, blocksize)
    rows = int(frame.shape[0].compute())
    counts = frame.isna().sum().compute().astype("int64")
    return pd.DataFrame({
        "missing_count": counts,
        "missing_percent": counts * (100.0 / rows if rows else 0.0),
    })


def dask_monthly_summary(
    source, *, date_column="date", amount_column="montant_da",
    blocksize="64MB", date_format=None, compute=False,
):
    """Aggregate CSV/Dask rows by YYYY-MM; invalid/missing dates are excluded.

    The returned lazy result can be computed explicitly when small. Month is
    derived in UTC; source timestamps should use compatible timezone semantics.
    """
    import pandas as pd
    frame = _as_dask_frame(source, blocksize)
    _require_columns(frame, [date_column, amount_column])
    if not pd.api.types.is_numeric_dtype(frame[amount_column].dtype):
        raise TypeError(f"{amount_column} must be numeric")
    dd = _dask()
    working = frame.assign(
        month=dd.to_datetime(
            frame[date_column], format=date_format, errors="coerce", utc=True
        ).dt.strftime("%Y-%m")
    )
    result = working.groupby("month")[amount_column].agg(["count", "sum", "mean"])
    if compute:
        return result.compute().sort_index()
    return result


def create_spark_session(*, app_name="KhaledDSML", master="local[2]"):
    """Create/reuse a SparkSession, using two local workers unless overridden."""
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise ImportError(
            'Spark is optional: pip install "khaled-dsml[spark]"; Java is required'
        ) from exc
    return SparkSession.builder.master(master).appName(app_name).getOrCreate()


def spark_read_csv(spark, path, *, header=True, infer_schema=True, schema=None, **options):
    """Read CSV as a Spark DataFrame.

    Schema inference can scan the input; pass an explicit schema for production.
    Additional CSV reader options (delimiter, timestampFormat, etc.) are accepted.
    """
    _spark()
    if not hasattr(spark, "read"):
        raise TypeError("spark must be an active SparkSession")
    reader = spark.read.option("header", bool(header))
    if schema is None:
        reader = reader.option("inferSchema", bool(infer_schema))
    else:
        reader = reader.schema(schema)
    for key, value in options.items():
        reader = reader.option(key, value)
    return reader.csv(str(path))


def _as_spark_frame(spark_df):
    SparkDataFrame, F = _spark()
    if not isinstance(spark_df, SparkDataFrame):
        raise TypeError("Expected a pyspark.sql.DataFrame")
    return F


def spark_group_summary(spark_df, *, group_by="wilaya", amount_column="montant_da"):
    """Return a Spark DataFrame; all operations stay inside the Spark engine.

    Count is of nonmissing numeric amounts, not all records of each group.
    """
    F = _as_spark_frame(spark_df)
    _require_columns(spark_df, [group_by, amount_column])
    from pyspark.sql.types import NumericType
    field = spark_df.schema[amount_column]
    if not isinstance(field.dataType, NumericType):
        raise TypeError(f"{amount_column} must be numeric; cast/clean it explicitly first")
    return spark_df.groupBy(F.col(group_by)).agg(
        F.count(F.col(amount_column)).alias("count"),
        F.sum(F.col(amount_column)).alias("sum"),
        F.avg(F.col(amount_column)).alias("mean"),
    ).orderBy(F.col("sum").desc())


def spark_csv_summary(
    spark, path, *, group_by="wilaya", amount_column="montant_da",
    schema=None,
):
    """Read and aggregate CSV using Spark; the returned result is still lazy."""
    frame = spark_read_csv(spark, path, schema=schema)
    return spark_group_summary(
        frame, group_by=group_by, amount_column=amount_column,
    )


def spark_missingness_report(spark_df):
    """Compute one compact missingness table via a Spark aggregation.

    This collects a single aggregate row containing counts, never the dataset.
    Null is counted as missing; empty strings and NaN need separate cleaning.
    """
    import pandas as pd
    F = _as_spark_frame(spark_df)
    names = list(spark_df.columns)
    if not names:
        raise ValueError("Spark DataFrame must have at least one column")
    if len(set(names)) != len(names):
        raise ValueError("Spark DataFrame columns must be unique")
    expressions = [F.count(F.lit(1)).alias("__rows")]
    expressions.extend(
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"__missing_{i}")
        for i, c in enumerate(names)
    )
    # count() is 0 on empty frames, but sums of missing flags can be NULL.
    result = spark_df.agg(*expressions).first().asDict()
    rows = int(result["__rows"])
    counts = [int(result[f"__missing_{i}"] or 0) for i in range(len(names))]
    return pd.DataFrame({
        "missing_count": counts,
        "missing_percent": [100 * n / rows if rows else 0.0 for n in counts],
    }, index=names)


def spark_monthly_summary(
    spark_df, *, date_column="date", amount_column="montant_da",
    date_format="yyyy-MM-dd",
):
    """Return a Spark DataFrame aggregated by month; bad dates are omitted.

    Parsing uses Spark's to_date with date_format. Timestamps/timezones require
    an explicit upstream normalization to the intended calendar timezone.
    """
    F = _as_spark_frame(spark_df)
    _require_columns(spark_df, [date_column, amount_column])
    from pyspark.sql.types import NumericType
    if not isinstance(spark_df.schema[amount_column].dataType, NumericType):
        raise TypeError(f"{amount_column} must be numeric")
    month = F.date_format(\n        F.try_to_timestamp(F.col(date_column), F.lit(date_format)), "yyyy-MM"\n    )
    working = spark_df.withColumn("__kdsml_month", month)
    return working.where(F.col("__kdsml_month").isNotNull()).groupBy(
        F.col("__kdsml_month")
    ).agg(
        F.count(F.col(amount_column)).alias("count"),
        F.sum(F.col(amount_column)).alias("sum"),
        F.avg(F.col(amount_column)).alias("mean"),
    ).withColumnRenamed("__kdsml_month", "month").orderBy("month")
