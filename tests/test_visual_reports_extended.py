"""Verify new extended metrics and tables render safely inside Analytics Studio."""
import pandas as pd
from khaled_dsml import render_analysis_dashboard, classification_diagnostics
from khaled_dsml.ml_extended import calibration_summary


def test_visualizes_real_ml_metrics_without_guessing():
    r=classification_diagnostics([0,0,1,1,0,1],[.1,.25,.45,.8,.2,.99],n_bins=4)
    html=render_analysis_dashboard(r,theme="amethyst")
    assert "MCC" in html
    assert "ROC-AUC" in html
    assert "KHALED DSML" in html
    assert 'dir="rtl"' in html
    assert 'dir="ltr"' in html


def test_calibration_bins_render_in_accessible_bilingual_table():
    r=calibration_summary([0,0,1,1],[.1,.2,.85,.9],n_bins=4)
    html=render_analysis_dashboard(r)
    assert '<section class="kdsml-table-panel">' in html
    assert 'جدول نتائج التحليل' in html
    assert 'ANALYTICAL RESULTS TABLE' in html
    assert 'mean_predicted' in html
    assert "kdsml-table-scroll" in html


def test_unsafe_dataframe_cell_is_escaped():
    payload={
        "interpretation_ar":"مؤشرات", "interpretation_en":"Indicators",
        "table":pd.DataFrame({"metric":["<script>alert(1)</script>"],
                              "value":[.48]}),
    }
    html=render_analysis_dashboard(payload)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_table_truncation_is_explicit_not_silent():
    payload={
        "ar":"نتائج تجريبية","en":"Experimental results",
        "table":pd.DataFrame({"x":range(30),"y":range(30)})
    }
    html=render_analysis_dashboard(payload)
    assert "Preview truncated to 25 rows" in html
    assert html.count("<tbody><tr>")==1
    assert html.count("<tr>")>=26


def test_new_scientific_metrics_use_correct_bilingual_labels():
    report = {
        "krippendorff_alpha": 0.82, "icc": 0.74,
        "ttr": 0.62, "n_raters": 3,
        "interpretation_ar": "مؤشرات الثبات والتنوع المعجمي",
        "interpretation_en": "Reliability and lexical diversity",
    }
    html = render_analysis_dashboard(report,theme="ocean")
    assert "Krippendorff α" in html
    assert "ألفا كريبندورف" in html
    assert "Type-token ratio" in html
    assert "مستوى الدلالة" not in html
    assert 'dir="rtl"' in html and 'dir="ltr"' in html
