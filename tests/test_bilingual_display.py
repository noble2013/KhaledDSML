"""Notebook bilingual text has correct direction, weight, and HTML safety."""
import pytest

from khaled_dsml import (
    normality_test, mann_kendall_trend_test, pettitt_change_point_test,
    forecast_accuracy_report,
)
from khaled_dsml.bilingual_display import (
    render_bilingual_result, display_bilingual_result, save_bilingual_result,
)


def test_testresult_arabic_right_english_left():
    test = mann_kendall_trend_test([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    html = render_bilingual_result(test)
    assert 'class="kdsml-ar" dir="rtl" lang="ar"' in html
    assert 'class="kdsml-en" dir="ltr" lang="en"' in html
    assert "text-align:right !important" in html
    assert "text-align:left !important" in html
    assert "font-size:21px;font-weight:700" in html
    assert "font-size:21px;font-weight:600" in html
    assert test.interpretation_ar in html
    assert test.interpretation_en in html


def test_dictionary_and_other_bilingual_shapes():
    change = pettitt_change_point_test([1] * 20 + [5] * 20)
    forecast = forecast_accuracy_report(
        [6, 7, 8], [7, 7, 8], [1, 2, 3, 4, 5],
    )
    for item in [change, forecast, {"ar": "مرحبًا", "en": "Hello"}]:
        html = render_bilingual_result(item)
        assert 'dir="rtl"' in html
        assert 'dir="ltr"' in html


def test_existing_normality_results_work_without_breaking_old_api():
    test = normality_test([12, 13, 15, 17, 16, 14, 13, 18])
    html = render_bilingual_result(test)
    assert test.interpretation_en in html


def test_escape_untrusted_interpretations_and_title():
    payload = {
        "interpretation_ar": '<img src=x onerror=alert(1)>',
        "interpretation_en": '<script>alert(1)</script>',
    }
    html = render_bilingual_result(payload, title="<b>unsafe</b>")
    assert "<img " not in html
    assert "<script>" not in html
    assert "<b>unsafe</b>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img " in html
    assert "&lt;b&gt;unsafe&lt;/b&gt;" in html


@pytest.mark.parametrize("size", [13, 37, 22.5, True, "21"])
def test_font_size_validation(size):
    with pytest.raises(ValueError):
        render_bilingual_result({"ar": "أ", "en": "A"}, font_size=size)


@pytest.mark.parametrize("data,error", [
    ({}, ValueError),
    ({"ar": "", "en": "ok"}, ValueError),
    ({"ar": "ok"}, ValueError),
    (["ok", "ok"], TypeError),
])
def test_invalid_bilingual_inputs(data, error):
    with pytest.raises(error):
        render_bilingual_result(data)


def test_html_saved_with_utf8_encoding(tmp_path):
    path = tmp_path / "result.html"
    saved = save_bilingual_result(
        {"ar": "هذه نتيجة عربية.", "en": "This is an English result."}, path,
    )
    assert saved == path
    text = saved.read_text(encoding="utf-8")
    assert "<meta charset=\"utf-8\">" in text
    assert "هذه نتيجة عربية." in text


def test_ipython_display_is_available_with_optional_extra(monkeypatch):
    ipython = pytest.importorskip("IPython")
    import IPython.display
    recorded = []
    monkeypatch.setattr(IPython.display, "display", recorded.append)
    display_bilingual_result({"ar": "تم", "en": "Done"})
    assert len(recorded) == 1
    assert "text-align:right" in recorded[0].data
