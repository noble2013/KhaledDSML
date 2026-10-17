"""Declarative, non-mutating pandas schema and quality-contract audit.

A validation report describes observed violations; it is not proof that the
measurement process or the underlying data are correct. No row-level values,
identifiers or sensitive observations are included in the default report.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import pandas as pd


_ALLOWED = {"numeric", "integer", "string", "boolean", "datetime", "category"}
_RULES = {"kind", "nullable", "unique", "min", "max", "allowed_values"}


def validate_dataframe_contract(frame, schema, *, allow_extra_columns=True):
    """Check column presence, declared dtype, missingness and value constraints.

    The schema is a mapping {column_name: {kind, nullable, unique, min,
    max, allowed_values}}. Numeric bounds only apply to numeric/integer
    columns; allowed_values use exact pandas membership, not implicit
    category conversion. Extra columns can be explicitly forbidden.
    Null constraints and uniqueness are checked separately.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not isinstance(schema, Mapping) or not schema or any(
        not isinstance(k, str) or not k for k in schema
    ):
        raise ValueError("schema must be a nonempty mapping from column names to rules")
    if not isinstance(allow_extra_columns, bool):
        raise ValueError("allow_extra_columns must be boolean")
    if not frame.columns.is_unique or not all(isinstance(x, str) for x in frame.columns):
        raise ValueError("DataFrame columns must be unique strings")
    report = []
    for col, rule in schema.items():
        if not isinstance(rule, Mapping) or not set(rule) <= _RULES:
            raise ValueError(f"Invalid rule keys for {col!r}")
        kind = rule.get("kind")
        if kind is not None and kind not in _ALLOWED:
            raise ValueError(f"Invalid kind for {col!r}")
        nullable, unique = rule.get("nullable", True), rule.get("unique", False)
        if not isinstance(nullable, bool) or not isinstance(unique, bool):
            raise ValueError("nullable and unique must be booleans")
        if ("min" in rule or "max" in rule) and kind not in ("numeric", "integer"):
            raise ValueError("min/max bounds require numeric or integer kind")
        if "min" in rule and "max" in rule and rule["min"] > rule["max"]:
            raise ValueError("Minimum bound exceeds maximum bound")
        allowed = rule.get("allowed_values")
        if allowed is not None:
            if not isinstance(allowed, (list, tuple, set, frozenset)) or not allowed:
                raise ValueError("allowed_values must be a nonempty collection")
            if any(pd.isna(v) for v in allowed):
                raise ValueError("allowed_values may not contain missing values")
        if col not in frame:
            report.append({"column": col, "rule": "present", "passed": False,
                           "violations": len(frame), "detail": "Missing column"})
            continue
        series = frame[col]
        type_checks = {
            "numeric": pd.api.types.is_numeric_dtype(series.dtype) and
                       not pd.api.types.is_bool_dtype(series.dtype),
            "integer": pd.api.types.is_integer_dtype(series.dtype),
            "string": pd.api.types.is_string_dtype(series.dtype) or
                      pd.api.types.is_object_dtype(series.dtype),
            "boolean": pd.api.types.is_bool_dtype(series.dtype),
            "datetime": pd.api.types.is_datetime64_any_dtype(series.dtype),
            "category": isinstance(series.dtype, pd.CategoricalDtype),
        }
        if kind is not None:
            ok = bool(type_checks[kind])
            report.append({"column": col, "rule": "kind", "passed": ok,
                           "violations": 0 if ok else len(series),
                           "detail": f"Declared {kind}; observed {series.dtype}"})
        if not nullable:
            n = int(series.isna().sum())
            report.append({"column": col, "rule": "not_null", "passed": n == 0,
                           "violations": n, "detail": "Missing entries"})
        if unique:
            n = int(series.dropna().duplicated(keep="first").sum())
            report.append({"column": col, "rule": "unique", "passed": n == 0,
                           "violations": n, "detail": "Repeated non-null entries"})
        nonnull = series.dropna()
        if "min" in rule or "max" in rule:
            if not type_checks["numeric"] and not type_checks["integer"]:
                for bound in ("min", "max"):
                    if bound in rule:
                        report.append({"column": col, "rule": bound, "passed": False,
                                       "violations": len(nonnull),
                                       "detail": "Numeric bound applied to nonnumeric dtype"})
            else:
                for bound in ("min", "max"):
                    if bound in rule:
                        threshold = rule[bound]
                        if isinstance(threshold, bool) or not np.isscalar(threshold) or not np.isfinite(threshold):
                            raise ValueError("Numeric bounds must be finite numbers")
                        count = int(((nonnull < threshold) if bound == "min" else
                                     (nonnull > threshold)).sum())
                        report.append({"column": col, "rule": bound, "passed": count == 0,
                                       "violations": count, "detail": "Outside declared numeric bounds"})
        if allowed is not None:
            count = int((~nonnull.isin(allowed)).sum())
            report.append({"column": col, "rule": "allowed_values", "passed": count == 0,
                           "violations": count, "detail": "Outside declared values"})
    if not allow_extra_columns:
        extra = sorted(set(frame.columns)-set(schema))
        report.append({"column": "*", "rule": "no_extra_columns",
                       "passed": len(extra) == 0, "violations": len(extra),
                       "detail": "Undeclared columns counted (names deliberately omitted)"})
    results = pd.DataFrame(report, columns=["column", "rule", "passed", "violations", "detail"])
    return {
        "test": "DataFrame schema contract validation",
        "n_rows": int(len(frame)), "n_columns": int(len(frame.columns)),
        "n_rules": int(len(results)), "n_failed_rules": int((~results["passed"]).sum()),
        "passed": bool(results["passed"].all()),
        "table": results,
        "interpretation_ar": (
            "فحص تصريح بنية البيانات وقيودها دون تعديل البيانات. يعرض أعداد "
            "المخالفات لا سجلات المستخدمين. النجاح لا يعني صحة المصدر أو "
            "خلوه من التحيز؛ يجب تحديد العقد قبل تقييم البيانات."
        ),
        "interpretation_en": (
            "Non-mutating structural contract check; reports aggregate "
            "violations without exposing raw row values. Passing a contract "
            "does not establish data provenance, validity or absence of bias."
        ),
    }
