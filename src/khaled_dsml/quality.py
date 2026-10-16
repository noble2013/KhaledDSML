"""Data inspection, cleaning helpers and simple distribution drift diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from ._results import make_result, TestResult


def quality_report(df: pd.DataFrame) -> dict:
    """Return shape, missing cells, duplicate rows, dtypes and memory usage."""
    _require_frame(df)
    return {
        "rows": len(df), "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_by_column": df.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1e6, 4),
    }


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    _require_frame(df)
    result = pd.DataFrame({"count": df.isna().sum()})
    result["percent"] = (100 * result["count"] / len(df)) if len(df) else 0.0
    return result


def _require_frame(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")


def iqr_outliers(data: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.Series:
    """Boolean mask of outliers; NaNs are False. Returns original row index."""
    _require_frame(data)
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    x = pd.to_numeric(data[column], errors="raise")
    if x.notna().sum() == 0:
        raise ValueError("Column has no observed values")
    q1, q3 = x.quantile([.25, .75])
    spread = q3 - q1
    return ((x < q1 - multiplier * spread) | (x > q3 + multiplier * spread)).fillna(False)


def zscore_outliers(data: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.Series:
    _require_frame(data)
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    x = pd.to_numeric(data[column], errors="raise")
    std = x.std(ddof=0)
    if not pd.notna(std) or std == 0:
        return pd.Series(False, index=data.index)
    return ((x - x.mean()).abs() / std > threshold).fillna(False)


def fill_missing_numeric(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Fill numeric missing values; return a copy. For ML use train-only imputation."""
    _require_frame(df)
    if strategy not in ("median", "mean", "zero"):
        raise ValueError("strategy must be median, mean or zero")
    out = df.copy()
    for c in out.select_dtypes(include="number").columns:
        if out[c].notna().sum() == 0:
            continue
        value = {"median": out[c].median, "mean": out[c].mean, "zero": lambda: 0}[strategy]()
        out[c] = out[c].fillna(value)
    return out


def memory_report(df: pd.DataFrame) -> pd.DataFrame:
    _require_frame(df)
    usage = df.memory_usage(deep=True, index=False)
    return pd.DataFrame({"bytes": usage, "megabytes": usage / 1e6}).sort_values("bytes", ascending=False)


def optimize_memory(df: pd.DataFrame, *, category_ratio: float = .5,
                    allow_float_downcast: bool = False) -> pd.DataFrame:
    """Downcast ints, optionally float64, and low-cardinality text columns.

    Float downcasting can be lossy and is disabled by default.
    Original dataframe is not changed.
    """
    _require_frame(df)
    if not 0 < category_ratio <= 1:
        raise ValueError("category_ratio must be in (0, 1]")
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_integer_dtype(s.dtype):
            out[c] = pd.to_numeric(s, downcast="integer")
        elif allow_float_downcast and pd.api.types.is_float_dtype(s.dtype):
            out[c] = pd.to_numeric(s, downcast="float")
        elif (pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype)):
            observed = s.dropna()
            if len(observed) and observed.nunique() / len(observed) <= category_ratio:
                out[c] = s.astype("category")
    return out


def distribution_drift_test(reference, current, alpha=.05) -> TestResult:
    from .statistics import ks_two_sample_test
    result = ks_two_sample_test(reference, current, alpha=alpha)
    return TestResult("Distribution drift / KS", result.statistic, result.p_value,
                      result.alpha, result.reject_null, result.interpretation_ar,
                      result.null_hypothesis_ar, result.sample_size,
                      "اختبار اختلاف التوزيعات وليس اختبارًا للسببية.")
