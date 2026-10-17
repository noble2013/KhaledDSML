"""Dask and Spark integration tests; skipped unless optional engines are installed.

GitHub CI installs the engines in its separate distributed-integration job.
These tests use tiny local CSVs but execute real Dask and Spark operations.
"""
import pandas as pd
import pytest

from khaled_dsml.distributed import (
    create_spark_session, dask_csv_summary, dask_group_summary,
    dask_missingness_report, dask_monthly_summary, dask_read_csv,
    spark_csv_summary, spark_group_summary, spark_missingness_report,
    spark_monthly_summary, spark_read_csv,
)


@pytest.fixture
def transactions_csv(tmp_path):
    data = pd.DataFrame({
        "wilaya": ["El Oued", "Algiers", "El Oued", "Oran", "El Oued"],
        "montant_da": [100.0, 60.0, 200.0, 90.0, None],
        "date": ["2026-01-02", "2026-01-05", "2026-02-05",
                 "2026-02-09", "2026-02-12"],
    })
    filename = tmp_path / "transactions.csv"
    data.to_csv(filename, index=False)
    return filename


def test_dask_lazy_group_summary(transactions_csv):
    dd = pytest.importorskip("dask.dataframe")
    frame = dask_read_csv(transactions_csv, blocksize=None)
    assert isinstance(frame, dd.DataFrame)
    lazy = dask_group_summary(frame)
    assert isinstance(lazy, dd.DataFrame)
    actual = lazy.compute()
    assert actual.loc["El Oued", "sum"] == pytest.approx(300)
    assert actual.loc["El Oued", "count"] == 2
    assert actual.loc["Algiers", "mean"] == pytest.approx(60)
    old_api = dask_csv_summary(transactions_csv, blocksize=None)
    assert isinstance(old_api, pd.DataFrame)
    assert old_api.index[0] == "El Oued"


def test_dask_compact_quality_and_monthly(transactions_csv):
    dd = pytest.importorskip("dask.dataframe")
    quality = dask_missingness_report(transactions_csv, blocksize=None)
    assert quality.loc["montant_da", "missing_count"] == 1
    assert quality.loc["montant_da", "missing_percent"] == pytest.approx(20)
    monthly = dask_monthly_summary(transactions_csv, blocksize=None)
    assert isinstance(monthly, dd.DataFrame)
    computed = monthly.compute()
    assert computed.loc["2026-01", "sum"] == pytest.approx(160)
    assert computed.loc["2026-02", "sum"] == pytest.approx(290)
    assert dask_monthly_summary(transactions_csv, blocksize=None, compute=True).shape[0] == 2


def test_dask_invalid_column_or_non_numeric(transactions_csv):
    pytest.importorskip("dask.dataframe")
    with pytest.raises(KeyError):
        dask_group_summary(transactions_csv, amount_column="missing")
    with pytest.raises(TypeError):
        dask_group_summary(transactions_csv, amount_column="wilaya")


@pytest.fixture(scope="module")
def local_spark():
    pytest.importorskip("pyspark")
    try:
        spark = create_spark_session(app_name="KhaledDSML-Integration-Test",
                                     master="local[2]")
    except Exception as exc:
        pytest.fail(f"Unable to start local Spark. Verify Java/SPARK_LOCAL_IP: {exc}")
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


def test_spark_read_group_summary(local_spark, transactions_csv):
    from pyspark.sql import DataFrame as SparkDataFrame
    frame = spark_read_csv(local_spark, transactions_csv)
    assert isinstance(frame, SparkDataFrame)
    result = spark_group_summary(frame)
    assert isinstance(result, SparkDataFrame)
    amounts = {r["wilaya"]: r for r in result.collect()}
    assert amounts["El Oued"]["sum"] == pytest.approx(300)
    assert amounts["El Oued"]["count"] == 2
    from_csv = spark_csv_summary(local_spark, transactions_csv)
    assert isinstance(from_csv, SparkDataFrame)
    assert from_csv.count() == 3


def test_spark_missingness_and_monthly(local_spark, transactions_csv):
    frame = spark_read_csv(local_spark, transactions_csv)
    missing = spark_missingness_report(frame)
    assert missing.loc["montant_da", "missing_count"] == 1
    assert missing.loc["montant_da", "missing_percent"] == pytest.approx(20)
    monthly = spark_monthly_summary(frame)
    rows = {r["month"]: r for r in monthly.collect()}
    assert rows["2026-01"]["sum"] == pytest.approx(160)
    assert rows["2026-02"]["sum"] == pytest.approx(290)
    assert monthly.columns == ["month", "count", "sum", "mean"]


def test_spark_invalid_column_and_type(local_spark, transactions_csv):
    frame = spark_read_csv(local_spark, transactions_csv)
    with pytest.raises(KeyError):
        spark_group_summary(frame, amount_column="unknown")
    with pytest.raises(TypeError):
        spark_group_summary(frame, amount_column="wilaya")


def test_empty_spark_missingness(local_spark):
    from pyspark.sql.types import DoubleType, StructField, StructType
    schema = StructType([StructField("amount", DoubleType(), nullable=True)])
    frame = local_spark.createDataFrame([], schema=schema)
    quality = spark_missingness_report(frame)
    assert quality.loc["amount", "missing_count"] == 0
    assert quality.loc["amount", "missing_percent"] == 0
