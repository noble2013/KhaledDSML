"""Binary model evaluation: correct class probabilities, ROC, and reports."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from khaled_dsml import make_classifier, split_data
from khaled_dsml.ml_evaluation import (
    binary_classification_metrics, compare_classifiers,
    roc_curve_table, render_classification_report,
    save_classification_report,
)


def test_metrics_known_example():
    r = binary_classification_metrics([0, 0, 1, 1], [.1, .4, .35, .8])
    assert r["roc_auc"] == pytest.approx(.75)
    assert r["accuracy"] == pytest.approx(.75)
    assert r["average_precision"] == pytest.approx(5 / 6)
    assert (r["tn"], r["fp"], r["fn"], r["tp"]) == (2, 0, 1, 1)
    assert r["recall"] == pytest.approx(.5)
    assert r["specificity"] == pytest.approx(1.0)


def test_scores_do_not_depend_on_threshold_but_predictions_do():
    a = binary_classification_metrics([0, 0, 1, 1], [.1, .4, .35, .8], threshold=.3)
    b = binary_classification_metrics([0, 0, 1, 1], [.1, .4, .35, .8], threshold=.6)
    assert a["roc_auc"] == pytest.approx(b["roc_auc"])
    assert a["average_precision"] == pytest.approx(b["average_precision"])
    assert a["recall"] != b["recall"]


def test_arbitrary_positive_label():
    r = binary_classification_metrics(
        ["yes", "no", "yes", "no"], [.9, .2, .8, .1],
        positive_label="yes",
    )
    assert r["roc_auc"] == pytest.approx(1.0)
    assert r["tp"] == 2 and r["tn"] == 2
    inverse = binary_classification_metrics(
        ["yes", "no", "yes", "no"], [.1, .8, .2, .9],
        positive_label="no",
    )
    assert inverse["roc_auc"] == pytest.approx(1.0)


@pytest.mark.parametrize("scores", [
    [.1, .5, np.nan, .7], [.1, .5, np.inf, .7],
    [.1, .5, 1.2, .7], [-.1, .5, .8, .7], [[.1], [.3], [.8], [.9]],
    [.1, .4, .9],
])
def test_invalid_scores(scores):
    with pytest.raises(ValueError):
        binary_classification_metrics([0, 0, 1, 1], scores)


@pytest.mark.parametrize("labels", [[0, 0, 0, 0], [0, 1, 2, 1], [0, 1, None, 1]])
def test_invalid_targets(labels):
    with pytest.raises(ValueError):
        binary_classification_metrics(labels, [.1, .2, .3, .4])


@pytest.mark.parametrize("threshold", [-1, 0, 1, 1.1, np.nan, "0.5"])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        binary_classification_metrics([0, 0, 1, 1], [.1, .4, .5, .8], threshold=threshold)


def test_curve_table():
    table = roc_curve_table([0, 0, 1, 1], [.1, .4, .35, .8])
    assert list(table.columns) == ["fpr", "tpr", "threshold"]
    assert table.iloc[0]["fpr"] == 0 and table.iloc[0]["tpr"] == 0
    assert table.iloc[-1]["fpr"] == 1 and table.iloc[-1]["tpr"] == 1
    assert table.fpr.is_monotonic_increasing
    assert table.tpr.is_monotonic_increasing


def test_classifiers_compare_pipelines_without_fitting_originals():
    X, y = make_classification(
        n_samples=140, n_features=5, n_informative=3, n_redundant=0, random_state=29
    )
    xt, xv, yt, yv = split_data(X, y, stratify=True, random_state=42)
    candidates = {
        "logistic": make_classifier(xt, "logistic"),
        "tree": make_classifier(xt, "decision_tree", max_depth=3),
    }
    table = compare_classifiers(candidates, xt, yt, xv, yv)
    assert list(table.model) == list(table.sort_values(
        ["roc_auc", "f1"], ascending=False, kind="stable"
    ).model)
    assert set(table.model) == {"logistic", "tree"}
    assert table.roc_auc.between(0, 1).all()
    assert table.f1.between(0, 1).all()
    assert table.n_samples.tolist() == [len(yv), len(yv)]
    for candidate in candidates.values():
        with pytest.raises(NotFittedError):
            check_is_fitted(candidate)


def test_invalid_comparison_models():
    with pytest.raises(ValueError):
        compare_classifiers({}, [[0], [1]], [0, 1], [[0], [1]], [0, 1])


def test_html_report_is_rtl_and_escapes_title(tmp_path):
    r = binary_classification_metrics([0, 0, 1, 1], [.1, .4, .35, .8])
    content = render_classification_report(r, title="<script>bad</script>")
    assert 'dir="rtl"' in content
    assert "ROC-AUC" in content
    assert "0.7500" in content
    assert "&lt;script&gt;bad&lt;/script&gt;" in content
    assert "<script>bad</script>" not in content
    path = save_classification_report(r, tmp_path / "report.html", title="تقرير التصنيف")
    assert "تقرير التصنيف" in path.read_text(encoding="utf-8")


def test_html_requires_complete_metrics():
    with pytest.raises(ValueError):
        render_classification_report({"roc_auc": .7})


def test_plot_roc_curve_optional_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    from khaled_dsml.plots import plot_roc_curve

    fig, ax = plot_roc_curve([0, 0, 1, 1], [.1, .4, .35, .8])
    try:
        assert len(ax.lines) == 2
        assert "AUC=0.750" in ax.lines[0].get_label()
        assert ax.get_xlabel() == "False positive rate"
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
