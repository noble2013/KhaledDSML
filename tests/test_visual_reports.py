"""Reference tests for beautiful and scientifically faithful analysis dashboards."""
from html import unescape

import numpy as np
import pytest

from khaled_dsml import (
    describe_numeric, normality_test, mann_kendall_trend_test,
    pettitt_change_point_test, forecast_accuracy_report,
    binary_classification_metrics,
    render_analysis_dashboard, save_analysis_dashboard,
    render_bilingual_results,
)


def _examples():
    return [
        mann_kendall_trend_test([12, 13, 15, 17, 16, 19, 20, 23, 25, 27, 29, 32]),
        pettitt_change_point_test([10] * 20 + [30] * 20),
        forecast_accuracy_report([100, 110, 120], [102, 108, 119],
                                 [50, 60, 70, 80, 90]),
    ]


def test_studio_layout_has_contrasting_hero_metric_cards_and_language_directions():
    html = render_analysis_dashboard(_examples())
    assert '<main class="kdsml-studio"' in html
    assert 'KHALED DSML · ANALYTICS STUDIO' in html
    assert 'linear-gradient' in html
    assert 'class="kdsml-metrics"' in html
    assert html.count('class="kdsml-result"') == 3
    assert html.count('class="kdsml-ar" dir="rtl" lang="ar"') == 3
    assert html.count('class="kdsml-en" dir="ltr" lang="en"') == 3
    assert 'text-align:right !important' in html
    assert 'text-align:left !important' in html
    assert 'font-weight:700' in html
    assert '@media print' in html
    assert '@media(max-width:680px)' in html


def test_statistical_metrics_are_true_inputs_and_pvalue_not_proof():
    result = normality_test([12, 13, 15, 17, 16, 14, 13, 18])
    html = render_analysis_dashboard(result)
    assert result.interpretation_ar in html
    assert result.interpretation_en in html
    assert 'Sample size' in html
    assert 'p-value' in html
    assert 'Do not reject H' in html or 'Reject H' in html
    assert 'do not establish causality' in html


def test_description_and_ml_metrics_are_understood_without_fake_values():
    descriptive = describe_numeric([2, 4, 6, 8, 10])
    classification = binary_classification_metrics([0, 0, 1, 1],
                                                   [.1, .4, .35, .8])
    output = render_analysis_dashboard([descriptive, classification])
    assert output.count('class="kdsml-result"') == 2
    assert 'Median' in output
    assert 'ROC-AUC' in output
    assert 'العينة الصالحة' in output
    assert 'valid observations' in output


def test_optional_real_series_chart_is_svg_offline_and_no_js():
    raw = [10, 12, 16, 14, 19, 24, 22, 30]
    html = render_analysis_dashboard(_examples(), chart_data=raw)
    assert '<polyline points="' in html
    assert 'Observed series' in html
    assert '<svg viewBox="0 0 620 112"' in html
    assert 'Min 10' in html and 'Max 30' in html
    assert 'http://' in html  # fixed, passive SVG namespace only
    assert '<script' not in html
    assert 'cdn.' not in html


def test_big_series_downsamples_without_giant_svg():
    html = render_analysis_dashboard({"ar": "السلسلة", "en": "Series"},
                                     chart_data=list(range(4000)))
    start = html.index('<polyline points="') + len('<polyline points="')
    end = html.index('" fill="none"', start)
    assert len(html[start:end].split()) == 120


def test_escape_titles_labels_notes_and_interpretations():
    payload = {
        "test": '<img src=x onerror=alert(1)>',
        "notes": "<b>unsafe</b>",
        "interpretation_ar": "<script>alert(1)</script>",
        "interpretation_en": "<svg onload=alert(1)>",
        "statistic": 1.25,
    }
    output = render_analysis_dashboard(
        payload, title_ar="<script>unsafe</script>",
        title_en="<img src=x>",
    )
    for snippet in ["<script>", "<img ", "<svg onload", "<b>unsafe</b>"]:
        assert snippet not in output
    assert "&lt;script&gt;" in output
    assert "&lt;img " in output
    assert "&lt;svg " in output
    assert "&lt;b&gt;" in output


@pytest.mark.parametrize("palette", ["aurora", "ocean", "amethyst"])
def test_themes_are_actual_scoped_palettes(palette):
    html = render_analysis_dashboard({"ar": "نتيجة", "en": "Result"}, theme=palette)
    assert '--kdsml-mid:' in html
    assert '--kdsml-accent:' in html


def test_bad_colors_sizes_charts_inputs_rejected():
    result = {"ar": "مقبول", "en": "Valid"}
    with pytest.raises(ValueError):
        render_analysis_dashboard(result, theme="unsafe")
    for size in [0, 13, 37, True, 21.1]:
        with pytest.raises(ValueError):
            render_analysis_dashboard(result, font_size=size)
    for observations in [[10], [1, np.nan], [1, np.inf], ["bad", "values"]]:
        with pytest.raises(ValueError):
            render_analysis_dashboard(result, chart_data=observations)
    with pytest.raises(TypeError):
        render_analysis_dashboard(result, chart_data="10,20")
    with pytest.raises(ValueError):
        render_analysis_dashboard([])


def test_save_report_utf8_and_legacy_plain_option(tmp_path):
    record = {"interpretation_ar": "بيانات واضحة", "interpretation_en": "Clear data"}
    path = tmp_path / "arabic_english.html"
    save_analysis_dashboard(record, path, title_ar="نتائج جميلة")
    text = path.read_text(encoding="utf-8")
    assert 'نتائج جميلة' in text
    assert '<meta charset="utf-8">' in text
    assert 'class="kdsml-studio"' in text
    assert 'KHALED DSML' in render_bilingual_results(record)  # new default
    assert 'kdsml-pair' in render_bilingual_results(record, style="plain")
