"""Real PySpark + Dask 0.7 integration. Runs in distributed-integration CI.

All tests are on tiny fixtures but execute real distributed APIs, protecting
against accidentally collecting entire datasets into pandas.
"""
import math
import numpy as np
import pandas as pd
import pytest

from khaled_dsml.distributed_analytics import (
    spark_missingness_detailed, spark_duplicate_summary, spark_numeric_profile,
    spark_time_holdout, spark_fit_logistic_classifier,
    spark_binary_classification_metrics,
    dask_duplicate_summary, dask_numeric_profile, dask_time_holdout,
)


@pytest.fixture(scope="module")
def spark_session():
    pytest.importorskip("pyspark")
    from khaled_dsml.distributed import create_spark_session
    spark = create_spark_session(app_name="KhaledDSML-v07-Test",master="local[2]")
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()


@pytest.fixture
def spark_data(spark_session):
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType
    schema = StructType([
        StructField("wilaya",StringType()),
        StructField("amount",DoubleType()),
        StructField("note",StringType()),
        StructField("date",StringType()),
    ])
    rows=[
        ("El Oued",100.,"A","2026-01-01"),
        ("El Oued",100.,"A","2026-01-01"),
        ("Algiers",20.,"","2026-02-01"),
        ("Oran",float("nan"),None,"bad"),
        ("Oran",None,"B",None),
        ("Setif",float("inf"),"C","2026-03-01"),
    ]
    return spark_session.createDataFrame(rows,schema)


def test_spark_exact_duplicate_and_missing_categories(spark_data):
    d=spark_duplicate_summary(spark_data)
    assert d["n_samples"]==6
    assert d["duplicate_rows"]==1
    quality=spark_missingness_detailed(spark_data)
    assert quality.loc["amount","null_count"]==1
    assert quality.loc["amount","nan_count"]==1
    assert quality.loc["note","blank_count"]==1
    assert quality.loc["note","null_count"]==1
    assert quality.loc["note","missing_count"]==2
    assert quality.loc["amount","missing_percent"]==pytest.approx(100/3)


def test_spark_numeric_profile_excludes_nan_inf_and_null(spark_data):
    r=spark_numeric_profile(spark_data,columns=["amount"])
    assert r.loc["amount","n_samples"]==6
    assert r.loc["amount","n_valid"]==3
    assert r.loc["amount","mean"]==pytest.approx(220/3)
    assert r.loc["amount","min"]==pytest.approx(20)
    assert r.loc["amount","max"]==pytest.approx(100)
    with pytest.raises(ValueError):
        spark_numeric_profile(spark_data,columns=["wilaya"])


def test_spark_time_split_retains_malformed_rows(spark_data):
    parts=spark_time_holdout(spark_data,date_column="date",cutoff="2026-02-01")
    assert parts["train"].count()==2
    assert parts["test"].count()==2
    assert parts["invalid"].count()==2
    with pytest.raises(ValueError):
        spark_time_holdout(spark_data,date_column="date",cutoff="2026-99-99")
    with pytest.raises(KeyError):
        spark_time_holdout(spark_data,date_column="not_a_date",cutoff="2026-02-01")


def test_spark_logistic_fit_evaluation_is_distributed(spark_session):
    from pyspark.sql import DataFrame as SparkDataFrame
    rows=[]
    for i in range(100):
        x=float(i%20-10)
        y=1. if x>0 else 0.
        rows.append((float(x),float(i%7)/7.,y))
    data=spark_session.createDataFrame(rows,["x","x2","label"])
    train=data.filter("x2 < 0.65")
    test=data.filter("x2 >= 0.65")
    model=spark_fit_logistic_classifier(
        train,feature_columns=["x","x2"],max_iter=30,reg_param=.1
    )
    predicted=model.transform(test)
    assert isinstance(predicted,SparkDataFrame)
    result=spark_binary_classification_metrics(predicted)
    assert 0<=result["roc_auc"]<=1
    assert 0<=result["pr_auc"]<=1
    assert result["n_samples"]==test.count()
    assert result["accuracy"]>.8
    assert "held-out" in result["interpretation_en"]
    with pytest.raises(ValueError):
        spark_fit_logistic_classifier(train,feature_columns=["x","label"])


@pytest.fixture
def dask_data(tmp_path):
    pytest.importorskip("dask.dataframe")
    data=pd.DataFrame({
        "wilaya":["El Oued","El Oued","Algiers","Oran","Oran"],
        "amount":[100.,100.,60.,80.,np.nan],
        "date":["2026-01-01","2026-01-01","2026-02-01","bad",None],
    })
    path=tmp_path/"transactions.csv"
    data.to_csv(path,index=False)
    return path


def test_dask_duplicates_profile_against_pandas(dask_data):
    profile=dask_numeric_profile(dask_data)
    assert profile.loc["amount","n_valid"]==4
    assert profile.loc["amount","mean"]==pytest.approx(85)
    duplicate=dask_duplicate_summary(dask_data)
    assert duplicate["duplicate_rows"]==1
    assert duplicate["n_samples"]==5


def test_dask_chronological_partitions_remain_lazy(dask_data):
    import dask.dataframe as dd
    frames=dask_time_holdout(dask_data,cutoff="2026-02-01")
    assert isinstance(frames["train"],dd.DataFrame)
    assert frames["train"].shape[0].compute()==2
    assert frames["test"].shape[0].compute()==1
    assert frames["invalid"].shape[0].compute()==2
    with pytest.raises(ValueError):
        dask_time_holdout(dask_data,cutoff="2026-02")


def test_empty_spark_numeric_profile_and_duplicate(spark_session):
    from pyspark.sql.types import StructType, StructField, DoubleType
    df=spark_session.createDataFrame(
        [],StructType([StructField("x",DoubleType(),True)])
    )
    assert spark_duplicate_summary(df)["n_samples"]==0
    profile=spark_numeric_profile(df)
    assert profile.loc["x","n_valid"]==0
    assert math.isnan(profile.loc["x","mean"])
    quality=spark_missingness_detailed(df)
    assert quality.loc["x","missing_count"]==0
