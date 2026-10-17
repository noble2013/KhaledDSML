"""Descriptive pattern-mixture delta sensitivity for numeric missing values.

A delta shift represents an explicitly hypothetical MNAR departure. It does not
identify a missingness mechanism, calibrate an MNAR model, or justify inferential
p-values or confidence intervals.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def mnar_delta_sensitivity(original, imputations, *, column, deltas):
    """Compare completed-data column means under specified imputed-only shifts.

    original is a numeric pandas DataFrame *before* imputation. imputations is
    a sequence of >=2 fully completed DataFrames with identical columns/index.
    Only cells originally missing in 'column' are shifted; all observed cells
    remain untouched. Outputs are descriptive means, not Rubin inference.
    """
    if not isinstance(original, pd.DataFrame) or original.empty:
        raise TypeError("original must be a nonempty pandas DataFrame")
    if not original.columns.is_unique or column not in original.columns:
        raise ValueError("Column must exist and all column labels must be unique")
    if any(not pd.api.types.is_numeric_dtype(t) for t in original.dtypes):
        raise TypeError("All columns must be numeric")
    raw = original.to_numpy(dtype=float)
    if np.isinf(raw).any():
        raise ValueError("Infinite values must be treated explicitly")
    if not isinstance(imputations, Sequence) or isinstance(imputations, (str, bytes)):
        raise TypeError("imputations must be a sequence of DataFrames")
    if len(imputations) < 2:
        raise ValueError("At least two imputed datasets are needed")
    if not isinstance(deltas, Sequence) or isinstance(deltas, (str, bytes)):
        raise TypeError("deltas must be a sequence of finite numeric shifts")
    try:
        shifts = np.asarray(deltas, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("All delta values must be finite numeric shifts") from exc
    if shifts.ndim != 1 or len(shifts) < 2 or not np.isfinite(shifts).all():
        raise ValueError("Supply at least two finite delta values")
    location = original.columns.get_loc(column)
    original_values = raw[:, location]
    missing = np.isnan(original_values)
    if not missing.any() or missing.all():
        raise ValueError("Target column must contain both observed and missing cells")
    samples = []
    observed = np.isfinite(raw)
    for completed in imputations:
        if (not isinstance(completed, pd.DataFrame)
            or not completed.index.equals(original.index)
            or not completed.columns.equals(original.columns)):
            raise ValueError("Imputation index and column labels/order must match original")
        x = completed.to_numpy(dtype=float)
        if not np.isfinite(x).all() or not np.array_equal(x[observed], raw[observed]):
            raise ValueError("Imputations must be finite and preserve every observed value")
        samples.append(x[:, location])
    baseline_means = np.array([v.mean() for v in samples])
    baseline = float(baseline_means.mean())
    fraction = missing.sum() / len(original)
    rows = []
    for delta in shifts:
        # Algebraically equivalent to modifying only originally missing cells.
        means = baseline_means + fraction * delta
        rows.append({
            "delta": float(delta), "n_shifted": int(missing.sum()),
            "mean_across_imputations": float(means.mean()),
            "min_imputation_mean": float(means.min()),
            "max_imputation_mean": float(means.max()),
            "change_from_zero_delta": float(means.mean() - baseline),
        })
    return {
        "test": "MNAR delta-shift sensitivity (descriptive)",
        "column": column, "n_samples": len(original),
        "n_imputations": len(imputations), "n_missing": int(missing.sum()),
        "n_scenarios": len(shifts), "baseline_mean": baseline,
        "table": pd.DataFrame(rows),
        "interpretation_ar": (
            "تحليل حساسية افتراضي: تُزاح القيم المستكملة فقط بمقادير دلتا محددة. "
            "النتائج وصفية ولا تثبت أن الفقد MNAR، ولا تتضمن قيمة p أو فواصل ثقة "
            "أو ادعاء بتوافق نموذج التعويض مع نموذج التحليل."
        ),
        "interpretation_en": (
            "Hypothetical pattern-mixture delta shifts affect imputed cells "
            "only. Means are descriptive; shifts do not establish MNAR or "
            "deliver inferential p-values, confidence intervals, or valid "
            "Rubin pooling without a separate analysis model."
        ),
    }
