"""Optional adapters for Dask and Spark; real engines required when used."""
from __future__ import annotations


def dask_csv_summary(path, *, group_by="wilaya", amount_column="montant_da",
                     blocksize="64MB"):
    """Lazy distributed CSV computation; returns a small pandas result table."""
    try:
        import dask.dataframe as dd
    except ImportError as exc:
        raise ImportError('Install with: pip install "khaled-dsml[dask]"') from exc
    frame = dd.read_csv(str(path), blocksize=blocksize)
    result = frame.groupby(group_by)[amount_column].agg(["count", "sum", "mean"])
    return result.compute().sort_values("sum", ascending=False)


def spark_group_summary(spark_df, *, group_by="wilaya", amount_column="montant_da"):
    """Return aggregated *Spark DataFrame* without collecting it to driver RAM."""
    try:
        from pyspark.sql import functions as F
        from pyspark.sql import DataFrame as SparkDataFrame
    except ImportError as exc:
        raise ImportError('Install with: pip install "khaled-dsml[spark]"; Java is also required') from exc
    if not isinstance(spark_df, SparkDataFrame):
        raise TypeError("spark_df must be a pyspark.sql.DataFrame")
    if group_by not in spark_df.columns or amount_column not in spark_df.columns:
        raise KeyError("Requested columns are absent")
    return spark_df.groupBy(group_by).agg(
        F.count(amount_column).alias("count"),
        F.sum(amount_column).alias("sum"),
        F.avg(amount_column).alias("mean"),
    ).orderBy(F.col("sum").desc())
