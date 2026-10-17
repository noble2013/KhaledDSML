"""Catalogue-backed tests for dataset quality and audit diagnostics."""
import numpy as np
import pandas as pd
import pytest

from khaled_dsml.audit import (
    duplicate_rows_report, missing_pattern_report, column_diagnostics,
    modified_z_outliers, mahalanobis_outliers, ml_outliers,
    survey_straightlining, benford_first_digit_test, terminal_digit_test,
    train_test_overlap, audit_dataset, render_audit_report, save_audit_report,
)


def test_exact_duplicates_preserves_index_and_missing():
    df = pd.DataFrame({"a": [1, 1, 2, np.nan, np.nan],
                       "b": ["x", "x", "y", "z", "z"]}, index=[3, 4, 5, 6, 7])
    report = duplicate_rows_report(df)
    assert report["duplicate_count"] == 2
    assert list(report["duplicate_mask"][report["duplicate_mask"]].index) == [4, 7]
    assert "fraud" not in report["interpretation_en"]


def test_missing_patterns_and_counts():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
    result = missing_pattern_report(df)
    assert result["total_missing_cells"] == 2
    assert result["complete_rows"] == 1
    assert sorted(result["patterns"]["count"].tolist()) == [1, 1, 1]
    assert result["by_column"].loc["a", "missing_count"] == 1
    assert "MCAR" in result["interpretation_en"]


def test_column_diagnostics_near_constant_and_range():
    df = pd.DataFrame({"gender": ["A"] * 19 + ["B"], "age": list(range(20))})
    result = column_diagnostics(df, dominant_share=.95,
                                allowed_ranges={"age": (3, 17)})
    assert result.loc["gender", "is_near_constant"]
    assert not result.loc["age", "is_constant"]
    assert result.loc["age", "out_of_range_count"] == 6


def test_modified_z_flags_extreme_and_nan():
    x = pd.Series([1, 2, 3, 4, 5, 100, np.nan], index=range(10, 17))
    mask = modified_z_outliers(x)
    assert mask.loc[15] and not mask.loc[16]
    with pytest.raises(ValueError, match="MAD is zero"):
        modified_z_outliers([0, 0, 0, 1])


def test_mahalanobis_keeps_missing_index():
    rng = np.random.default_rng(88)
    data = pd.DataFrame(rng.normal(size=(80, 2)), columns=["x", "y"])
    data.loc[1, "x"] = None
    result = mahalanobis_outliers(data, alpha=.05)
    assert result.index.equals(data.index)
    assert pd.isna(result.loc[1, "is_outlier"])
    assert np.isfinite(result.loc[0, "distance_squared"])
    with pytest.raises(ValueError, match="singular"):
        mahalanobis_outliers(pd.DataFrame({"a": [1,2,3,4], "b":[2,4,6,8]}))


def test_unsupervised_outliers_in_both_methods():
    rng = np.random.default_rng(7)
    d = pd.DataFrame(rng.normal(size=(35, 2)), columns=["x", "y"])
    for method in ("isolation_forest", "lof"):
        result = ml_outliers(d, method=method, n_neighbors=10, contamination=.1)
        assert result.dtype == bool
        assert 1 <= result.sum() <= 8
    with pytest.raises(ValueError):
        ml_outliers(d, method="nope")


def test_straightlining_only_flags_for_review():
    survey = pd.DataFrame([[2,2,2,2], [1,2,3,4], [np.nan,np.nan,1,np.nan]])
    r = survey_straightlining(survey)
    assert r["flag_for_review"].tolist() == [True, False, False]


def test_benford_math_without_fraud_claims():
    rng = np.random.default_rng(123)
    first = rng.choice(np.arange(1,10), size=900, p=np.log10(1+1/np.arange(1,10)))
    numbers = first * (10 ** rng.integers(0,4,size=900)) + .2
    result = benford_first_digit_test(numbers)
    assert 0 <= result.p_value <= 1
    assert result.sample_size == 900
    assert "not evidence of fraud" in result.notes_en
    with pytest.raises(ValueError):
        benford_first_digit_test([1,2,3])


def test_terminal_digit_test():
    integers = list(range(200))
    r = terminal_digit_test(integers)
    assert r.statistic == pytest.approx(0)
    assert r.p_value == pytest.approx(1)
    with pytest.raises(ValueError):
        terminal_digit_test([1.5] * 100)


def test_train_test_overlap():
    train = pd.DataFrame({"x": [1,2,3], "z":["a","b","c"]})
    test = pd.DataFrame({"x": [2,2,4], "z":["b","b","d"]})
    r = train_test_overlap(train, test)
    assert r["shared_unique_rows"] == 1
    assert r["test_rows_overlapping"] == 2
    assert "leakage" in r["interpretation_en"]


def test_audit_bilingual_html_safely_escapes(tmp_path):
    df = pd.DataFrame({"a":[1,1,1,None], "b":[3,3,3,4]})
    summary = audit_dataset(df)
    assert {"duplicates", "near_constant"} & set(summary["flagged_issues"])
    html = render_audit_report(summary, title="<script>alert(1)</script>")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "lang=\"en\"" in html and "dir=\"rtl\"" in html
    f = save_audit_report(summary, tmp_path / "audit.html")
    assert "تدقيق" in f.read_text(encoding="utf8")


def test_audit_rejects_duplicate_columns():
    data = pd.DataFrame([[1,2]], columns=["x", "x"])
    with pytest.raises(ValueError):
        duplicate_rows_report(data)


def test_audit_empty_dataframes():
    df = pd.DataFrame({"a": pd.Series(dtype="float64")})
    s = audit_dataset(df)
    assert s["n_rows"] == 0 and s["missing_cells"] == 0
