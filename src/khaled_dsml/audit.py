"""Dataset audit functions aligned with the KhaledDSML academic test catalogue.

These diagnostics flag observations for human review, not for automatic deletion.
Audit results include Arabic and English cautions and never claim to detect fraud.
"""
from __future__ import annotations

from collections.abc import Mapping
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ._results import make_result


def _frame(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    if not data.columns.is_unique:
        raise ValueError("Duplicate column names are ambiguous")
    return data


def duplicate_rows_report(data, *, columns=None) -> dict:
    """Count exact duplicate rows, including missing values, without modifying data."""
    frame = _frame(data)
    subset = list(frame.columns) if columns is None else list(columns)
    if not subset or any(c not in frame.columns for c in subset):
        raise ValueError("Select one or more existing columns")
    mask = frame.duplicated(subset=subset, keep="first")
    return {
        "row_count": len(frame),
        "duplicate_count": int(mask.sum()),
        "duplicate_share": float(mask.mean()) if len(frame) else 0.0,
        "duplicate_mask": mask,
        "columns": subset,
        "interpretation_ar": "تطابق الصفوف لا يثبت أنها مكررة خطأً؛ راجع المعرّفات وتصميم الجمع.",
        "interpretation_en": "Exact duplicate rows do not by themselves imply erroneous duplication; check record identifiers and collection design.",
    }


def missing_pattern_report(data) -> dict:
    """Missing-cell totals and distinct missingness patterns (1 means missing)."""
    frame = _frame(data)
    counts = frame.isna().sum().astype(int)
    percent = (counts / len(frame) * 100) if len(frame) else counts.astype(float)
    per_col = pd.DataFrame({"missing_count": counts, "missing_percent": percent})
    patterns = frame.isna().apply(
        lambda row: "".join("1" if value else "0" for value in row), axis=1
    ).value_counts().rename_axis("pattern").reset_index(name="count")
    patterns["percentage"] = patterns["count"] * (100 / len(frame) if len(frame) else 0)
    return {
        "columns": list(frame.columns),
        "by_column": per_col,
        "patterns": patterns,
        "total_missing_cells": int(counts.sum()),
        "complete_rows": int(frame.notna().all(axis=1).sum()),
        "interpretation_ar": "نمط القيم المفقودة وصفي ولا يثبت MCAR؛ اختبار Little يحتاج تحليلًا منفصلًا.",
        "interpretation_en": "Missingness patterns are descriptive and do not establish MCAR; Little's test requires separate inference.",
    }


def column_diagnostics(data, *, dominant_share=0.95, allowed_ranges=None) -> pd.DataFrame:
    """Detect constant/near-constant columns and optional out-of-range values."""
    frame = _frame(data)
    if not 0 < dominant_share <= 1:
        raise ValueError("dominant_share must be in (0, 1]")
    allowed = {} if allowed_ranges is None else dict(allowed_ranges)
    if any(c not in frame.columns for c in allowed):
        raise KeyError("allowed_ranges refers to an unknown column")
    records = []
    for name in frame.columns:
        observed = frame[name].dropna()
        counts = observed.value_counts(dropna=True)
        largest = float(counts.iloc[0] / len(observed)) if len(observed) else float("nan")
        invalid = 0
        if name in allowed:
            low, high = allowed[name]
            if low > high:
                raise ValueError("Each lower range bound must be <= upper bound")
            numbers = pd.to_numeric(observed, errors="raise")
            invalid = int(((numbers < low) | (numbers > high)).sum())
        records.append({
            "column": name, "dtype": str(frame[name].dtype),
            "n_valid": int(len(observed)), "n_missing": int(frame[name].isna().sum()),
            "n_unique": int(observed.nunique()),
            "dominant_share": largest,
            "is_constant": bool(len(observed) and observed.nunique() == 1),
            "is_near_constant": bool(len(observed) and largest >= dominant_share),
            "out_of_range_count": invalid,
        })
    return pd.DataFrame.from_records(records).set_index("column")


def modified_z_outliers(values, *, threshold=3.5) -> pd.Series:
    """Robust MAD-based mask, preserving Series index; missing rows are False.

    A zero MAD cannot support modified z-scores; a clear error is raised rather
    than silently claiming that observations are non-outliers.
    """
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be strictly positive")
    sample = pd.to_numeric(pd.Series(values), errors="raise")
    observed = sample.dropna()
    if len(observed) < 3 or not np.isfinite(observed.to_numpy(dtype=float)).all():
        raise ValueError("Need at least three finite observed values")
    center = float(observed.median())
    mad = float((observed - center).abs().median())
    if mad == 0:
        raise ValueError("MAD is zero; modified z-scores are undefined; use another method")
    z = .67448975 * (sample - center).abs() / mad
    return (z > threshold).fillna(False)


def mahalanobis_outliers(data, *, columns=None, alpha=.05) -> pd.DataFrame:
    """Classical squared Mahalanobis distances using chi-square reference cutoff.

    Assumes full-rank covariance; the chi-square cutoff is a screening heuristic,
    not a robust multivariate normality test.
    """
    frame = _frame(data)
    selected = list(frame.select_dtypes(include="number").columns) if columns is None else list(columns)
    if len(selected) < 2 or not 0 < alpha < 1:
        raise ValueError("Need at least two features and alpha in (0,1)")
    subset = frame[selected].dropna()
    x = subset.to_numpy(dtype=float)
    n, p = x.shape
    if n <= p or not np.isfinite(x).all():
        raise ValueError("Need more complete rows than features and finite values")
    covariance = np.cov(x, rowvar=False)
    if np.linalg.matrix_rank(covariance) != p:
        raise ValueError("Covariance is singular; remove collinear features")
    diffs = x - x.mean(axis=0)
    squared = np.einsum("ij,jk,ik->i", diffs, np.linalg.inv(covariance), diffs)
    cutoff = float(stats.chi2.ppf(1 - alpha, p))
    output = pd.DataFrame({
        "distance_squared": squared, "chi2_cutoff": cutoff,
        "is_outlier": squared > cutoff,
    }, index=subset.index)
    return output.reindex(frame.index)


def ml_outliers(data, *, method="isolation_forest", columns=None,
                contamination=.05, random_state=42, n_neighbors=20) -> pd.Series:
    """Optional Isolation Forest or LOF flags; no automatic deletion or fraud inference."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    frame = _frame(data)
    selected = list(frame.select_dtypes(include="number").columns) if columns is None else list(columns)
    if not selected or not 0 < contamination <= .5:
        raise ValueError("Need numeric features and contamination in (0,.5]")
    subset = frame[selected].dropna()
    x = subset.to_numpy(dtype=float)
    if len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("Need at least 3 finite complete rows")
    if method == "isolation_forest":
        model = IsolationForest(contamination=contamination, random_state=random_state,
                                n_estimators=100)
    elif method == "lof":
        if not 1 <= n_neighbors < len(x):
            raise ValueError("n_neighbors must be between 1 and n_rows - 1")
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    else:
        raise ValueError("method must be isolation_forest or lof")
    result = pd.Series(False, index=frame.index)
    result.loc[subset.index] = model.fit_predict(x) == -1
    return result


def survey_straightlining(data, *, min_items=3, threshold=.95) -> pd.DataFrame:
    """Flag identical survey answers; a flag is NOT proof of careless response."""
    frame = _frame(data)
    if frame.shape[1] < min_items or not 0 < threshold <= 1:
        raise ValueError("Need enough survey items and threshold in (0,1]")
    valid = frame.notna().sum(axis=1)
    max_share = frame.apply(
        lambda r: float(r.dropna().value_counts().iloc[0] / r.notna().sum())
        if r.notna().sum() else float("nan"), axis=1,
    )
    return pd.DataFrame({
        "answered_items": valid,
        "dominant_answer_share": max_share,
        "flag_for_review": (valid >= min_items) & (max_share >= threshold),
    }, index=frame.index)


def benford_first_digit_test(values, *, alpha=.05) :
    """Chi-square diagnostic for Benford first digits; suitability is contextual."""
    s = pd.to_numeric(pd.Series(values), errors="raise").dropna().to_numpy(dtype=float)
    if not np.isfinite(s).all() or len(s) < 20:
        raise ValueError("Need at least 20 finite numbers")
    x = np.abs(s[s != 0])
    if len(x) < 20:
        raise ValueError("Need at least 20 nonzero observations")
    digits = [int(f"{v:.15e}".lstrip("0.")[0]) for v in x]
    counts = np.bincount(digits, minlength=10)[1:]
    expectation = len(x) * np.log10(1 + 1 / np.arange(1, 10))
    r = stats.chisquare(counts, f_exp=expectation)
    return make_result(
        "Benford first digits", r.statistic, r.pvalue, len(x), alpha,
        "تتبع الأرقام الأولى توزيع بنفورد.",
        "لا تصلح القاعدة لكل البيانات، ورفضها ليس دليل احتيال.",
        null_en="First significant digits follow Benford's distribution.",
        notes_en="Benford's law is unsuitable for many datasets; rejection is not evidence of fraud.",
    )


def terminal_digit_test(values, *, alpha=.05):
    """Test uniformity of final digits for integer observations."""
    a = pd.to_numeric(pd.Series(values), errors="raise").dropna().to_numpy(dtype=float)
    if len(a) < 20 or not np.isfinite(a).all() or not np.all(a == np.floor(a)):
        raise ValueError("Need at least 20 finite integer observations")
    observed = np.bincount((np.abs(a).astype(np.int64) % 10), minlength=10)
    r = stats.chisquare(observed)
    return make_result("Terminal digit uniformity", r.statistic, r.pvalue, len(a),
                       alpha, "الأرقام النهائية من 0 إلى 9 متساوية الاحتمال.",
                       "يلزم أولًا التأكد من ملاءمة التوزيع المنتظم لهذه البيانات.",
                       null_en="Final digits 0 to 9 are equally likely.",
                       notes_en="Check whether terminal-digit uniformity is an appropriate benchmark.")


def train_test_overlap(train, test) -> dict:
    """Find exact full-row overlap: risk indicator, not proof of target leakage."""
    a, b = _frame(train), _frame(test)
    if list(a.columns) != list(b.columns):
        raise ValueError("Train/test feature names and order must match")
    def keys(df):
        return set(pd.util.hash_pandas_object(df, index=False).to_numpy(dtype=np.uint64))
    shared = keys(a) & keys(b)
    return {
        "shared_unique_rows": len(shared),
        "test_rows_overlapping": int(np.isin(
            pd.util.hash_pandas_object(b, index=False).to_numpy(dtype=np.uint64),
            list(shared),
        ).sum()),
        "interpretation_ar": "وجود صفوف مشتركة قد يشير إلى تسرب أو تكرار؛ راجع المعرفات ومجموعات الأفراد.",
        "interpretation_en": "Shared rows may indicate leakage or duplication; check entity identifiers and group splits.",
    }


def audit_dataset(data, *, missing_alert=.2, near_constant=.95) -> dict:
    """Conservative bilingual dataset summary: it never deletes records."""
    frame = _frame(data)
    if not 0 <= missing_alert <= 1:
        raise ValueError("missing_alert must be in [0,1]")
    duplicates = duplicate_rows_report(frame)
    missing = missing_pattern_report(frame)
    columns = column_diagnostics(frame, dominant_share=near_constant)
    missing_fraction = float(frame.isna().mean().max()) if len(frame) and len(frame.columns) else 0
    flagged = [k for k, v in {
        "duplicates": duplicates["duplicate_count"] > 0,
        "missingness": missing_fraction > missing_alert,
        "near_constant": bool(columns["is_near_constant"].any()) if len(columns) else False,
    }.items() if v]
    return {
        "n_rows": len(frame), "n_columns": len(frame.columns),
        "duplicate_rows": duplicates["duplicate_count"],
        "missing_cells": missing["total_missing_cells"],
        "flagged_issues": flagged,
        "column_diagnostics": columns,
        "missing_patterns": missing["patterns"],
        "interpretation_ar": (
            "تم تحديد مؤشرات للمراجعة، ولا تعني وجود تزوير أو خلل مؤكد."
            if flagged else "لم تُكتشف مؤشرات بالمقاييس المحددة؛ لا يُعد هذا ضمانًا لجودة البيانات."
        ),
        "interpretation_en": (
            "Potential issues were flagged for review, not confirmed as fraud or errors."
            if flagged else "No selected warning criteria fired; this does not guarantee data quality."
        ),
    }


def render_audit_report(summary: Mapping, *, title="تدقيق البيانات | Data quality audit") -> str:
    """Simple bilingual print-ready HTML from audit_dataset output."""
    required = ("n_rows", "n_columns", "duplicate_rows", "missing_cells",
                "interpretation_ar", "interpretation_en")
    if any(key not in summary for key in required):
        raise ValueError("Pass a full audit_dataset summary")
    title_safe = escape(str(title))
    table = [
        ("عدد الصفوف / Rows", "n_rows"),
        ("عدد الأعمدة / Columns", "n_columns"),
        ("الصفوف المتكررة / Duplicates", "duplicate_rows"),
        ("الخلايا المفقودة / Missing cells", "missing_cells"),
    ]
    trs = "".join(
        f"<tr><th>{escape(label)}</th><td>{int(summary[key])}</td></tr>"
        for label, key in table
    )
    return (
        '<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">'
        '<style>main{max-width:780px;margin:2rem auto;font:600 19px Tahoma,Arial;'
        'line-height:1.9;text-align:right}table{border-collapse:collapse;width:100%}'
        'td,th{padding:.4rem;border-bottom:1px solid #ddd}'
        '</style></head><body><main>'
        f'<h1>{title_safe}</h1><table>{trs}</table>'
        f'<p>{escape(str(summary["interpretation_ar"]))}</p>'
        f'<p lang="en" dir="ltr" style="text-align:left">'
        f'{escape(str(summary["interpretation_en"]))}</p>'
        '</main></body></html>'
    )


def save_audit_report(summary: Mapping, path: str | Path) -> Path:
    destination = Path(path)
    destination.write_text(render_audit_report(summary), encoding="utf-8")
    return destination
