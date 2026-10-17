"""Checks for Arabic RTL reports, correct decisions and HTML injection safety."""
from dataclasses import replace

import pytest

from khaled_dsml import normality_test
from khaled_dsml._results import TestResult as ResultRecord
from khaled_dsml.reports import render_test_report, save_test_report


def sample_result(p_value=0.01):
    return ResultRecord(
        test="Shapiro-Wilk", statistic=0.91, p_value=p_value,
        alpha=0.05, reject_null=p_value < 0.05,
        interpretation_ar="نرفض الفرضية الصفرية." if p_value < 0.05
        else "لا نرفض الفرضية الصفرية.",
        null_hypothesis_ar="العينة تتبع التوزيع الطبيعي.",
        sample_size=25, notes="ملاحظة تعليمية",
    )


def test_basic_rtl_html_report():
    output = render_test_report(sample_result())
    assert '<html lang="ar" dir="rtl">' in output
    assert '<meta charset="utf-8">' in output
    assert '<div class="khaled-dsml-report" dir="rtl" lang="ar">' in output
    assert '.khaled-dsml-report h1{' in output
    assert "Shapiro-Wilk" in output
    assert "p = 0.0100" in output
    assert "رفض الفرضية الصفرية" in output
    assert "25" in output


def test_accept_multiple_results():
    output = render_test_report([sample_result(0.001), sample_result(0.6)])
    assert output.count('<article class="result">') == 2
    assert "عدم رفض الفرضية الصفرية" in output
    assert "p = 0.6000" in output


def test_small_p_value_is_not_rendered_as_zero():
    output = render_test_report(sample_result(0.0))
    assert "p &lt; 0.0001" in output
    assert "p = 0.0000" not in output


def test_user_text_is_escaped():
    test = replace(
        sample_result(),
        test='<script>alert("bad")</script>',
        interpretation_ar="<img src=x onerror=alert(1)>",
        notes="<b>not trusted</b>",
    )
    output = render_test_report(test, title="<b>Title</b>")
    assert "<script>" not in output
    assert "<img " not in output
    assert "<b>Title</b>" not in output
    assert "&lt;script&gt;" in output
    assert "&lt;img " in output
    assert "&lt;b&gt;Title&lt;/b&gt;" in output


@pytest.mark.parametrize("digits", [-1, 0, 11, True, 3.1, "4"])
def test_digits_validation(digits):
    with pytest.raises(ValueError):
        render_test_report(sample_result(), digits=digits)


@pytest.mark.parametrize("items,error", [
    ([], ValueError), (["bad"], TypeError), ("bad", TypeError), (123, TypeError),
])
def test_result_validation(items, error):
    with pytest.raises(error):
        render_test_report(items)


def test_save_utf8_report(tmp_path):
    target = tmp_path / "report.html"
    returned = save_test_report(sample_result(), target, title="تقرير اختبار الطبيعية")
    assert returned == target
    assert "تقرير اختبار الطبيعية" in target.read_text(encoding="utf-8")
    assert "<html" in target.read_text(encoding="utf-8")


def test_example_real_statistical_result():
    result = normality_test([12, 13, 15, 17, 16, 14, 13, 18])
    html = render_test_report(result)
    assert "Shapiro-Wilk" in html
    assert "p = " in html
    assert result.interpretation_ar in html
