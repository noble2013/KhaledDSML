import numpy as np
import pandas as pd
import pytest
from khaled_dsml import (quality_report, missingness_report, iqr_outliers,
    zscore_outliers, fill_missing_numeric, optimize_memory, memory_report,
    distribution_drift_test)


def test_quality_report():
    df = pd.DataFrame({"x": [1, None, 1], "cat": ["a", "b", "a"]})
    report = quality_report(df)
    assert report["rows"] == 3 and report["columns"] == 2
    assert report["missing_cells"] == 1 and report["duplicate_rows"] == 1
    assert missingness_report(df).loc["x", "percent"] == pytest.approx(100 / 3)


def test_iqr_outlier():
    df = pd.DataFrame({"x": [1, 2, 2, 3, 50, None]})
    found = iqr_outliers(df, "x")
    assert list(found) == [False, False, False, False, True, False]


def test_zscore_no_variance():
    assert zscore_outliers(pd.DataFrame({"a": [2, 2, 2]}), "a").sum() == 0


def test_fill_missing_nonmutating():
    df = pd.DataFrame({"a": [1., 3., None], "b": ["a", None, "b"]})
    filled = fill_missing_numeric(df)
    assert filled.loc[2, "a"] == 2
    assert pd.isna(df.loc[2, "a"]) and pd.isna(filled.loc[1, "b"])


def test_optimize_memory():
    df = pd.DataFrame({"a": np.arange(1000, dtype="int64"),
                       "b": ["sale"] * 1000, "c": np.arange(1000, dtype="float64")})
    out = optimize_memory(df)
    assert str(out.b.dtype) == "category"
    assert out.a.dtype.itemsize < df.a.dtype.itemsize
    assert out.c.dtype == df.c.dtype
    assert memory_report(out)["bytes"].sum() < memory_report(df)["bytes"].sum()


def test_distribution_drift_result():
    result = distribution_drift_test(range(100), range(100, 200))
    assert result.reject_null and result.p_value < .05


def test_empty_frame():
    assert quality_report(pd.DataFrame())["rows"] == 0
    assert missingness_report(pd.DataFrame({"x": []})).loc["x", "percent"] == 0


def test_invalid_dataframe_type():
    with pytest.raises(TypeError):
        quality_report([1, 2])
