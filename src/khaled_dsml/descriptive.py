"""Comprehensive descriptive statistics for teaching and applied data analysis.

These functions are descriptive only: they do not establish causality.
Undefined statistics are represented by NaN or None and explained in both
Arabic and English. NaN values are omitted from numeric calculations.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_PERCENTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _numeric_sample(values):
    series = pd.Series(values)
    try:
        numbers = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Expected numeric data; encode categories separately") from exc
    if np.isinf(numbers).any():
        raise ValueError("Infinity is not a valid observation")
    clean = numbers[~np.isnan(numbers)]
    if not len(clean):
        raise ValueError("At least one observed numeric value is required")
    return numbers, clean


def weighted_mean(values, weights) -> float:
    """Weighted arithmetic mean, rejecting negative/nonfinite weights.

    Missing value/weight pairs are removed together. Zero-weight rows are
    permitted but the total remaining weight must be strictly positive.
    """
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 1 or w.shape != x.shape:
        raise ValueError("values and weights must be aligned one-dimensional arrays")
    if np.isinf(x).any() or np.isinf(w).any():
        raise ValueError("Infinity is not allowed")
    valid = ~(np.isnan(x) | np.isnan(w))
    x, w = x[valid], w[valid]
    if not len(x) or np.any(w < 0) or not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError("Need finite nonnegative weights summing to a positive value")
    return float(np.average(x, weights=w))


def gini_coefficient(values) -> float:
    """Gini inequality coefficient; defined here for nonnegative observations."""
    _, x = _numeric_sample(values)
    if np.any(x < 0):
        raise ValueError("Gini requires nonnegative values")
    total = np.sum(x)
    if total == 0:
        return 0.0
    sorted_values = np.sort(x)
    n = len(x)
    return float((2 * np.dot(np.arange(1, n + 1), sorted_values)
                  / (n * total)) - (n + 1) / n)


def describe_numeric(values, *, ddof=1, percentiles=DEFAULT_PERCENTILES,
                     trim_fraction=0.10) -> dict:
    """Detailed univariate descriptive statistics (sample SD by default).

    Skewness and excess kurtosis use bias correction when sample sizes allow.
    Geometric/harmonic means are undefined for nonpositive observations.
    Gini is undefined for negative observations. Coefficient of variation is
    reported as a fraction, not a percentage; CV of near-zero mean is undefined.
    """
    if type(ddof) is not int or ddof not in (0, 1):
        raise ValueError("ddof must be 0 (population) or 1 (sample)")
    if not (0 <= trim_fraction < 0.5):
        raise ValueError("trim_fraction must be in [0, 0.5)")
    quantiles = tuple(float(v) for v in percentiles)
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in quantiles):
        raise ValueError("percentiles must lie between 0 and 1")
    original, x = _numeric_sample(values)
    n = len(x)
    ordered = np.sort(x)
    q1, med, q3 = np.quantile(x, [0.25, 0.50, 0.75])
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=ddof)) if n > ddof else float("nan")
    var = float(std ** 2)
    median_ad = float(stats.median_abs_deviation(x, scale=1.0))
    counts = pd.Series(x).value_counts()
    max_freq = int(counts.iloc[0])
    modes = sorted(float(v) for v in counts[counts == max_freq].index) if max_freq > 1 else []
    iqr = float(q3 - q1)
    nonconstant = not np.all(x == x[0])
    result = {
        "n_total": int(len(original)),
        "n_valid": int(n),
        "n_missing": int(len(original) - n),
        "n_unique": int(len(np.unique(x))),
        "sum": float(np.sum(x)),
        "mean": mean,
        "median": float(med),
        "mode": modes,
        "mode_frequency": max_freq if modes else 0,
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "range": float(ordered[-1] - ordered[0]),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": iqr,
        "five_number_summary": [float(ordered[0]), float(q1), float(med),
                                float(q3), float(ordered[-1])],
        "variance": var,
        "std": std,
        "sem": float(std / np.sqrt(n)),
        "cv": float(std / abs(mean)) if abs(mean) > 1e-12 else float("nan"),
        "cv_percent": float(100 * std / abs(mean)) if abs(mean) > 1e-12 else float("nan"),
        "mean_absolute_deviation": float(np.mean(np.abs(x - mean))),
        "median_absolute_deviation": median_ad,
        "skewness": float(stats.skew(x, bias=False)) if n >= 3 and nonconstant else float("nan"),
        "excess_kurtosis": float(stats.kurtosis(x, fisher=True, bias=False))
                           if n >= 4 and nonconstant else float("nan"),
        "trimmed_mean": float(stats.trim_mean(x, proportiontocut=trim_fraction)),
        "geometric_mean": float(stats.gmean(x)) if np.all(x > 0) else None,
        "harmonic_mean": float(stats.hmean(x)) if np.all(x > 0) else None,
        "midrange": float((ordered[0] + ordered[-1]) / 2),
        "outliers_iqr_1_5": int(np.count_nonzero((x < q1 - 1.5 * iqr) |
                                                 (x > q3 + 1.5 * iqr))),
        "gini": gini_coefficient(x) if np.all(x >= 0) else None,
        "percentiles": {f"p{int(round(p * 100)):02d}": float(np.quantile(x, p))
                        for p in quantiles},
        "ddof": ddof,
    }
    return result


def describe_categorical(values, *, include_missing=False) -> dict:
    """Frequency structure, Shannon entropy, Gini impurity and category modes."""
    series = pd.Series(values)
    missing = int(series.isna().sum())
    cleaned = series if include_missing else series.dropna()
    n = int(len(cleaned))
    frequencies = cleaned.value_counts(dropna=not include_missing)
    if not n:
        return {"n_total": int(len(series)), "n_valid": 0, "n_missing": missing,
                "n_categories": 0, "mode": [], "dominant_share": float("nan"),
                "entropy_bits": float("nan"), "gini_impurity": float("nan")}
    probs = frequencies.to_numpy(dtype=float) / n
    most_common = frequencies[frequencies == frequencies.max()].index.tolist()
    return {
        "n_total": int(len(series)), "n_valid": int(series.notna().sum()),
        "n_missing": missing, "n_categories": int(len(frequencies)),
        "mode": most_common, "dominant_share": float(probs.max()),
        "entropy_bits": float(-np.sum(probs * np.log2(probs))),
        "gini_impurity": float(1 - np.sum(probs ** 2)),
    }


def frequency_table(values, *, include_missing=False, sort_by="frequency") -> pd.DataFrame:
    """Return counts, relative/cumulative frequencies and percentages."""
    if sort_by not in ("frequency", "value"):
        raise ValueError("sort_by must be frequency or value")
    counts = pd.Series(values).value_counts(dropna=not include_missing, sort=sort_by=="frequency")
    if sort_by == "value":
        counts = counts.sort_index()
    n = counts.sum()
    fractions = counts / n if n else counts.astype(float)
    return pd.DataFrame({
        "count": counts.astype(int),
        "relative_frequency": fractions,
        "percentage": fractions * 100,
        "cumulative_count": counts.cumsum(),
        "cumulative_percentage": 100 * fractions.cumsum(),
    }).rename_axis("value").reset_index()


def descriptive_table(data: pd.DataFrame, *, columns=None, ddof=1) -> pd.DataFrame:
    """Numeric summaries by DataFrame column; metrics appear as table columns."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    selected = list(data.select_dtypes(include="number").columns) if columns is None else list(columns)
    if not selected:
        raise ValueError("No numeric columns selected")
    rows = {}
    for col in selected:
        if col not in data.columns:
            raise KeyError(col)
        try:
            rows[col] = describe_numeric(data[col], ddof=ddof)
        except ValueError as exc:
            if data[col].isna().all():
                rows[col] = {"n_total": len(data), "n_valid": 0,
                             "n_missing": len(data), "mean": float("nan")}
            else:
                raise ValueError(f"{col}: {exc}") from exc
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "variable"
    return result


def covariance_correlation(data: pd.DataFrame, *, method="pearson") -> dict:
    """Pairwise covariance and correlation matrices on numeric columns."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    frame = data.select_dtypes(include="number")
    if frame.shape[1] < 2:
        raise ValueError("Need at least two numeric variables")
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError("method must be pearson, spearman or kendall")
    return {"covariance": frame.cov(min_periods=2), "correlation": frame.corr(method=method,
                                                   min_periods=2)}


METRIC_LABELS = {
    "n_valid": ("عدد المشاهدات الصالحة", "Valid observations"),
    "n_missing": ("القيم المفقودة", "Missing observations"),
    "mean": ("المتوسط الحسابي", "Arithmetic mean"),
    "median": ("الوسيط", "Median"),
    "mode": ("المنوال", "Mode(s)"),
    "min": ("القيمة الصغرى", "Minimum"),
    "max": ("القيمة العظمى", "Maximum"),
    "range": ("المدى", "Range"),
    "variance": ("التباين", "Variance"),
    "std": ("الانحراف المعياري", "Standard deviation"),
    "sem": ("الخطأ المعياري للمتوسط", "Standard error of the mean"),
    "cv_percent": ("معامل الاختلاف (%)", "Coefficient of variation (%)"),
    "q1": ("الربيع الأول", "First quartile"),
    "q3": ("الربيع الثالث", "Third quartile"),
    "iqr": ("المدى الربيعي", "Interquartile range"),
    "skewness": ("معامل الالتواء", "Skewness"),
    "excess_kurtosis": ("التفرطح الزائد", "Excess kurtosis"),
    "median_absolute_deviation": ("الانحراف المطلق الوسيطي", "Median absolute deviation"),
    "mean_absolute_deviation": ("متوسط الانحرافات المطلقة", "Mean absolute deviation"),
    "trimmed_mean": ("المتوسط المشذب", "Trimmed mean"),
    "geometric_mean": ("المتوسط الهندسي", "Geometric mean"),
    "harmonic_mean": ("المتوسط التوافقي", "Harmonic mean"),
    "gini": ("معامل جيني", "Gini coefficient"),
    "outliers_iqr_1_5": ("القيم الشاذة وفق IQR", "IQR-rule outliers"),
}


def interpret_descriptive(summary: Mapping) -> dict:
    """Return cautious explanatory sentences in Arabic and English."""
    skew = summary.get("skewness", float("nan"))
    if np.isfinite(skew):
        ar_skew = ("انحراف نحو اليمين" if skew > 0.5 else
                   "انحراف نحو اليسار" if skew < -0.5 else "التواء محدود")
        en_skew = ("right-skewed" if skew > 0.5 else
                   "left-skewed" if skew < -0.5 else "little skew")
    else:
        ar_skew, en_skew = "لا يمكن تقدير الالتواء", "skewness is undefined"
    cv = summary.get("cv_percent", float("nan"))
    ar_cv = (f"معامل الاختلاف {cv:.2f}% وصف نسبي للتشتت."
             if np.isfinite(cv) else "معامل الاختلاف غير معرّف للمتوسط القريب من الصفر.")
    en_cv = (f"The coefficient of variation is {cv:.2f}%, a relative dispersion measure."
             if np.isfinite(cv) else "Coefficient of variation is undefined near zero mean.")
    ar = (f"العينة الصالحة تضم {summary.get('n_valid')} مشاهدة؛ "
          f"المتوسط {summary.get('mean', float('nan')):.3f} "
          f"والوسيط {summary.get('median', float('nan')):.3f}. "
          f"التوزيع يُظهر {ar_skew}. {ar_cv} "
          "لا تثبت الإحصاءات الوصفية الدلالة الإحصائية أو السببية.")
    en = (f"There are {summary.get('n_valid')} valid observations; "
          f"mean {summary.get('mean', float('nan')):.3f}, "
          f"median {summary.get('median', float('nan')):.3f}. "
          f"The distribution shows {en_skew}. {en_cv} "
          "Descriptive statistics do not demonstrate statistical significance or causality.")
    return {"ar": ar, "en": en}


def render_descriptive_report(summary: Mapping, *, title="الإحصاءات الوصفية | Descriptive statistics") -> str:
    """Standalone print-ready bilingual HTML; labels and text safely escaped."""
    interpretation = interpret_descriptive(summary)
    table = []
    for key, (label_ar, label_en) in METRIC_LABELS.items():
        value = summary.get(key, None)
        if value is None:
            shown = "N/A"
        elif isinstance(value, (int, np.integer)):
            shown = str(value)
        elif isinstance(value, (float, np.floating)):
            shown = f"{float(value):.4f}" if np.isfinite(value) else "N/A"
        else:
            shown = str(value)
        table.append(f'<tr><th>{escape(label_ar)} / {escape(label_en)}</th>'
                     f'<td dir="ltr">{escape(shown)}</td></tr>')
    rows = "\n".join(table)
    heading = escape(str(title))
    return (
        '<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{heading}</title>'
        '<style>.kdsml-desc{font:600 19px Tahoma,Arial,sans-serif;line-height:1.9;'
        'max-width:850px;margin:2rem auto;padding:1rem;direction:rtl;text-align:right}'
        '.kdsml-desc table{border-collapse:collapse;width:100%}'
        '.kdsml-desc td,.kdsml-desc th{padding:.3rem .6rem;border-bottom:1px solid #ddd}'
        '.kdsml-desc td{text-align:left} .kdsml-desc .english{direction:ltr;text-align:left;font-weight:600}'
        '</style></head><body><main class="kdsml-desc">'
        f'<h1>{heading}</h1><table><tbody>{rows}</tbody></table>'
        f'<h2>التفسير بالعربية</h2><p>{escape(interpretation["ar"])}</p>'
        f'<h2>English interpretation</h2><p class="english" lang="en">'
        f'{escape(interpretation["en"])}</p></main></body></html>'
    )


def save_descriptive_report(summary: Mapping, path: str | Path, *, title="الإحصاءات الوصفية | Descriptive statistics") -> Path:
    target = Path(path)
    target.write_text(render_descriptive_report(summary, title=title), encoding="utf-8")
    return target


def display_descriptive_report(summary: Mapping, *, title="الإحصاءات الوصفية | Descriptive statistics") -> None:
    """Show bilingual report in Jupyter/Colab (IPython optional)."""
    try:
        from IPython.display import display, HTML
    except ImportError as exc:
        raise ImportError('Install khaled-dsml[notebooks] for notebook display') from exc
    display(HTML(render_descriptive_report(summary, title=title)))
