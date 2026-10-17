"""Notebook bilingual text has correct direction, weight, and HTML safety."""
import pytest

from khaled_dsml import (
    normality_test, mann_kendall_trend_test, pettitt_change_point_test,
    forecast_accuracy_report,
)
from khaled_dsml.bilingual_display import (
    render_bilingual_result, display_bilingual_result, save_bilingual_result,
    render_bilingual_results, display_bilingual_results, save_bilingual_results,
)


def test_testresult_arabic_right_english_left():
    test = mann_kendall_trend_test([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    html = render_bilingual_result(test)
    assert 'class="kdsml-ar" dir="rtl" lang="ar"' in html
    assert 'class="kdsml-en" dir="ltr" lang="en"' in html
    assert "text-align:right !important" in html
    assert "text-align:left !important" in html
    assert "font-size:23px;font-weight:700" in html
    assert "font-size:23px;font-weight:700" in html
    assert "border:1px" not in html
    assert "<h2" not in html  # default must match screenshot without a heading
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


def test_plain_multi_result_layout_matches_colab_screenshot():
    from html import unescape
    trend = mann_kendall_trend_test([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    change = pettitt_change_point_test([1] * 20 + [5] * 20)
    forecast = forecast_accuracy_report([6, 7], [5, 7], [1, 2, 3, 4, 5])
    html = render_bilingual_results([trend, change, forecast], font_size=24)
    assert html.count('class="kdsml-pair"') == 3
    assert html.count('class="kdsml-ar" lang="ar" dir="rtl"') == 3
    assert html.count('class="kdsml-en" lang="en" dir="ltr"') == 3
    assert 'font-size:24px;font-weight:700' in html
    assert 'text-align:right !important' in html
    assert 'text-align:left !important' in html
    assert 'border:1px' not in html
    assert '<hr ' not in html
    assert '<h2' not in html
    assert html.index(trend.interpretation_ar) < html.index(change["interpretation_ar"])
    assert html.index(change["interpretation_ar"]) < html.index(forecast["interpretation_ar"])


def test_bilingual_batch_accepts_generator_and_rejects_invalid():
    values = iter([{"ar": "النتيجة الأولى", "en": "First result"},
                   {"ar": "النتيجة الثانية", "en": "Second result"}])
    text = render_bilingual_results(values)
    assert text.count('class="kdsml-pair"') == 2
    with pytest.raises(ValueError):
        render_bilingual_results([])
    with pytest.raises(TypeError):
        render_bilingual_results("ar and en")
    with pytest.raises(ValueError):
        render_bilingual_results([{"ar": "نعم", "en": "Yes"}, {"ar": ""}])
    with pytest.raises(TypeError):
        render_bilingual_results([123])


def test_batches_escape_all_text_and_support_explicit_title():
    html = render_bilingual_results(
        [{"ar": "<script>alert(1)</script>", "en": "safe"},
         {"ar": "آمن", "en": "<img src=x onerror=alert(1)>"}],
        title="<title>unsafe</title>",
    )
    assert "<script>alert" not in html
    assert "<img " not in html
    assert "<title>unsafe</title>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img " in html
    assert "&lt;title&gt;unsafe&lt;/title&gt;" in html
    assert '<h2 lang="ar" dir="rtl"' in html


def test_save_multiple_html_in_utf8_and_display(monkeypatch, tmp_path):
    records = [{"ar": "التفسير", "en": "Interpretation"},
               {"ar": "المتوسط", "en": "Mean"}]
    path = tmp_path / "group.html"
    save_bilingual_results(records, path)
    assert path.read_text(encoding="utf-8").count('class="kdsml-pair"') == 2
    pytest.importorskip("IPython")
    import IPython.display
    rendered = []
    monkeypatch.setattr(IPython.display, "display", rendered.append)
    display_bilingual_results(records)
    assert len(rendered) == 1
    assert rendered[0].data.count('class="kdsml-en"') == 2
