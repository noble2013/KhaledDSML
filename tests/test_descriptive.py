"""Reference checks for descriptive statistics and bilingual HTML reports."""
import numpy as np
import pandas as pd
import pytest

from khaled_dsml.descriptive import (
    describe_numeric, describe_categorical, descriptive_table,
    frequency_table, weighted_mean, gini_coefficient, covariance_correlation,
    interpret_descriptive, render_descriptive_report, save_descriptive_report,
)
from khaled_dsml import normality_test
from khaled_dsml.reports import render_test_report


def test_center_dispersion_and_missing():
    r = describe_numeric([1, 2, 2, 3, np.nan])
    assert r["n_total"] == 5 and r["n_valid"] == 4 and r["n_missing"] == 1
    assert r["mean"] == pytest.approx(2)
    assert r["median"] == pytest.approx(2)
    assert r["mode"] == [2.]
    assert r["range"] == 2
    assert r["variance"] == pytest.approx(2 / 3)
    assert r["std"] == pytest.approx(np.sqrt(2/3))
    assert r["sem"] == pytest.approx(np.sqrt(2/3) / 2)
    assert r["iqr"] == pytest.approx(.5)
    assert r["min"] == 1 and r["max"] == 3


def test_percentiles_and_no_mode_with_singletons():
    r = describe_numeric([1, 2, 3, 4], percentiles=[.25, .75])
    assert r["mode"] == []
    assert r["percentiles"]["p25"] == pytest.approx(1.75)
    assert r["percentiles"]["p75"] == pytest.approx(3.25)
    assert r["skewness"] == pytest.approx(0)
    assert r["excess_kurtosis"] == pytest.approx(-1.2)
    assert r["geometric_mean"] == pytest.approx(24 ** .25)
    assert r["harmonic_mean"] == pytest.approx(4 / (1 + .5 + 1/3 + .25))


def test_population_and_constant_cases():
    r = describe_numeric([4, 4, 4], ddof=0)
    assert r["variance"] == 0 and r["std"] == 0
    assert np.isnan(r["skewness"])
    r2 = describe_numeric([5])
    assert np.isnan(r2["std"])
    assert np.isnan(r2["excess_kurtosis"])


def test_robust_and_inequality():
    r = describe_numeric([1, 2, 3, 4, 100])
    assert r["outliers_iqr_1_5"] == 1
    assert r["median_absolute_deviation"] == 1
    assert gini_coefficient([0, 0, 0]) == pytest.approx(0)
    assert gini_coefficient([0, 0, 10]) == pytest.approx(2/3)
    with pytest.raises(ValueError):
        gini_coefficient([-1, 2])
    assert describe_numeric([-1, 2])["geometric_mean"] is None


def test_weighted_average_and_missing_rows():
    assert weighted_mean([10, 20, 30], [1, 2, 1]) == pytest.approx(20)
    assert weighted_mean([10, np.nan, 20], [1, 1, 3]) == pytest.approx(17.5)
    for weight in [[-1, 2], [0, 0]]:
        with pytest.raises(ValueError):
            weighted_mean([10, 20], weight)


def test_categorical_and_frequencies():
    r = describe_categorical(["A", "A", "B", None])
    assert r["n_categories"] == 2 and r["mode"] == ["A"]
    assert r["n_missing"] == 1 and r["dominant_share"] == pytest.approx(2/3)
    assert r["entropy_bits"] > 0
    t = frequency_table(["a", "a", "b"])
    assert t["count"].sum() == 3
    assert t["percentage"].sum() == pytest.approx(100)
    assert t["cumulative_percentage"].iloc[-1] == pytest.approx(100)


def test_dataframe_summary_and_correlation():
    df = pd.DataFrame({
        "x": [1, 2, 3, 4],
        "y": [2, 4, 6, 8],
        "cat": ["a", "a", "b", "b"],
    })
    table = descriptive_table(df)
    assert set(table.index) == {"x", "y"}
    assert table.loc["x", "mean"] == pytest.approx(2.5)
    both = covariance_correlation(df)
    assert both["correlation"].loc["x", "y"] == pytest.approx(1)
    assert both["covariance"].loc["x", "y"] == pytest.approx(10/3)


def test_bilingual_description_and_safety(tmp_path):
    summary = describe_numeric([1, 2, 2, 3, 9])
    interpretation = interpret_descriptive(summary)
    assert "ar" in interpretation and "en" in interpretation
    assert "المتوسط" in interpretation["ar"] and "mean" in interpretation["en"]
    html = render_descriptive_report(summary, title="<script>alert(1)</script>")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert 'dir="rtl"' in html and 'lang="en"' in html
    saved = save_descriptive_report(summary, tmp_path / "bilingual.html")
    assert saved.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_inferential_english_backward_compatibility():
    result = normality_test([12, 13, 15, 17, 16, 14, 13, 18])
    assert "null hypothesis" in result.interpretation_en.lower()
    assert "normal distribution" in result.null_hypothesis_en
    html = render_test_report(result)
    assert "Interpretation:" in html and "الفرضية الصفرية" in html
    assert result.interpretation_en in html


@pytest.mark.parametrize("bad", [[np.nan], [np.inf], ["abc"]])
def test_reject_undefined_data(bad):
    with pytest.raises(ValueError):
        describe_numeric(bad)


def test_reject_invalid_configurations():
    with pytest.raises(ValueError):
        describe_numeric([1, 2], ddof=2)
    with pytest.raises(ValueError):
        describe_numeric([1, 2], trim_fraction=.5)
    with pytest.raises(ValueError):
        describe_numeric([1, 2], percentiles=[-1])
    with pytest.raises(ValueError):
        descriptive_table(pd.DataFrame({"cat": ["a"]}))
