"""Binary classification evaluation for teaching and economic case studies.

Scores are probabilities for the explicitly selected positive class. Keep train
and test samples separated: the comparison helper clones and fits each pipeline
only on its training data.
"""
from __future__ import annotations

from collections.abc import Mapping
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve,
)


def _binary_targets(y_true, positive_label):
    labels = np.asarray(y_true)
    if labels.ndim != 1 or len(labels) < 2:
        raise ValueError("y_true must be a one-dimensional array with at least two observations")
    if bool(np.any(pd.isna(labels))):
        raise ValueError("y_true cannot include missing labels")
    unique = pd.unique(labels)
    if len(unique) != 2 or not any(label == positive_label for label in unique):
        raise ValueError("Exactly two classes, including positive_label, are required")
    return (labels == positive_label).astype(int)


def _validated_inputs(y_true, y_probability, positive_label):
    target = _binary_targets(y_true, positive_label)
    try:
        scores = np.asarray(y_probability, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_probability must contain numeric probabilities") from exc
    if scores.ndim != 1 or len(scores) != len(target):
        raise ValueError("y_probability must be 1D and aligned with y_true")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Probabilities must be finite and between 0 and 1")
    return target, scores


def binary_classification_metrics(
    y_true, y_probability, *, threshold=0.5, positive_label=1
) -> dict:
    """Compute positive-class metrics from held-out probabilities.

    ROC-AUC and average precision are independent of threshold. Every returned
    count refers to the positive label supplied by the caller.
    """
    if not isinstance(threshold, (int, float, np.number)) or not (
        np.isfinite(threshold) and 0 < threshold < 1
    ):
        raise ValueError("threshold must be a finite number strictly between 0 and 1")
    target, scores = _validated_inputs(y_true, y_probability, positive_label)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    return {
        "n_samples": int(len(target)),
        "prevalence": float(target.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(target, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, scores)),
        "average_precision": float(average_precision_score(target, scores)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def roc_curve_table(y_true, y_probability, *, positive_label=1) -> pd.DataFrame:
    """Return false/true-positive rates and threshold values for an ROC curve."""
    target, scores = _validated_inputs(y_true, y_probability, positive_label)
    fpr, tpr, thresholds = roc_curve(target, scores, pos_label=1, drop_intermediate=False)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def compare_classifiers(
    models: Mapping[str, object], X_train, y_train, X_test, y_test, *,
    positive_label=1, threshold=0.5,
) -> pd.DataFrame:
    """Clone, train and compare probabilistic binary classifiers on held-out data.

    `models` values must be unfitted or fitted sklearn-compatible estimators
    supporting clone, fit, predict_proba and classes_. The originals are not
    changed, so preprocessing within Pipelines is fitted on X_train only.
    """
    if not isinstance(models, Mapping) or not models:
        raise ValueError("models must be a nonempty mapping of names to estimators")
    _binary_targets(y_train, positive_label)
    _binary_targets(y_test, positive_label)
    rows = []
    for name, estimator in models.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Every model must have a nonempty string name")
        model = clone(estimator)
        model.fit(X_train, y_train)
        if not hasattr(model, "predict_proba"):
            raise ValueError(f"{name}: model must support predict_proba for ROC-AUC")
        classes = np.asarray(model.classes_)
        indices = np.flatnonzero(classes == positive_label)
        if len(indices) != 1:
            raise ValueError(f"{name}: positive label not found in fitted model")
        probability = model.predict_proba(X_test)[:, int(indices[0])]
        summary = binary_classification_metrics(
            y_test, probability, threshold=threshold, positive_label=positive_label
        )
        rows.append({"model": name, **summary})
    return pd.DataFrame(rows).sort_values(
        ["roc_auc", "f1"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)


_AR_METRIC_NAMES = (
    ("n_samples", "حجم العينة"),
    ("prevalence", "نسبة الفئة الموجبة"),
    ("threshold", "عتبة التصنيف"),
    ("accuracy", "الدقة الكلية Accuracy"),
    ("balanced_accuracy", "الدقة المتوازنة Balanced Accuracy"),
    ("precision", "الإحكام Precision"),
    ("recall", "الاسترجاع Recall / Sensitivity"),
    ("specificity", "النوعية Specificity"),
    ("f1", "مؤشر F1"),
    ("roc_auc", "المساحة تحت منحنى ROC-AUC"),
    ("average_precision", "متوسط الإحكام Average Precision"),
    ("tn", "سلبي صحيح TN"),
    ("fp", "إيجابي كاذب FP"),
    ("fn", "سلبي كاذب FN"),
    ("tp", "إيجابي صحيح TP"),
)


def interpret_classification_metrics(metrics: Mapping) -> dict:
    """Interpret binary performance cautiously, in Arabic and English.

    No rating such as 'excellent' is assigned to AUC: validity depends on
    domain, baseline, class balance, calibration and test set design.
    """
    needed = ("roc_auc", "f1", "recall", "specificity", "prevalence", "threshold",
              "fp", "fn")
    if not isinstance(metrics, Mapping) or any(key not in metrics for key in needed):
        raise ValueError("Pass metrics from binary_classification_metrics")
    auc = float(metrics["roc_auc"])
    f1 = float(metrics["f1"])
    recall = float(metrics["recall"])
    spec = float(metrics["specificity"])
    prevalence = float(metrics["prevalence"])
    threshold = float(metrics["threshold"])
    return {
        "ar": (
            f"تبلغ ROC-AUC القيمة {auc:.3f}، وهي تلخص قدرة النموذج على ترتيب "
            "الحالات الموجبة فوق السالبة عبر العتبات، ولا تثبت معايرة الاحتمالات. "
            f"عند العتبة {threshold:.2f} يبلغ الاسترجاع {recall:.3f} "
            f"والنوعية {spec:.3f} ومؤشر F1 {f1:.3f}. "
            f"انتشار الفئة الموجبة في عينة التقييم {prevalence:.1%}؛ "
            f"الإيجابيات الكاذبة {int(metrics['fp'])} والسلبيات الكاذبة {int(metrics['fn'])}. "
            "فسّر النتائج في ضوء تكاليف الأخطاء، وتوازن الفئات، وتقسيم البيانات."
        ),
        "en": (
            f"ROC-AUC is {auc:.3f}; it summarizes ranking of positive above "
            "negative cases over thresholds, not probability calibration. "
            f"At threshold {threshold:.2f}, recall is {recall:.3f}, "
            f"specificity {spec:.3f}, and F1 {f1:.3f}. "
            f"Positive-class prevalence in this test set is {prevalence:.1%}; "
            f"there are {int(metrics['fp'])} false positives and "
            f"{int(metrics['fn'])} false negatives. Interpret performance in "
            "the context of error costs, class balance, and data splitting."
        ),
    }


def render_classification_report(metrics: Mapping, *, title="تقرير تقييم نموذج التصنيف") -> str:
    """Render a printable, escaped Arabic RTL report from binary metrics."""
    if not isinstance(metrics, Mapping) or any(k not in metrics for k, _ in _AR_METRIC_NAMES):
        raise ValueError("Pass a complete result from binary_classification_metrics")
    rows = []
    counts = {"n_samples", "tn", "fp", "fn", "tp"}
    for key, label in _AR_METRIC_NAMES:
        value = metrics[key]
        if key in counts:
            shown = str(int(value))
        else:
            shown = f"{float(value):.4f}"
        rows.append(
            f"<tr><th scope=\"row\">{escape(label)}</th>"
            f"<td dir=\"ltr\">{escape(shown)}</td></tr>"
        )
    heading = escape(str(title))
    body = "\n".join(rows)
    return (
        '<!doctype html>\n<html lang="ar" dir="rtl">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>{heading}</title>\n"
        '<style>.kdsml-classifier{font-family:Tahoma,Arial,sans-serif;'
        'direction:rtl;text-align:right;max-width:820px;margin:2rem auto;'
        'padding:1rem;line-height:1.8;background:white;color:#24243a}'
        '.kdsml-classifier table{border-collapse:collapse;width:100%}'
        '.kdsml-classifier th,.kdsml-classifier td{padding:.4rem .6rem;'
        'border-bottom:1px solid #ddd}.kdsml-classifier td{text-align:left}'
        '</style>\n</head><body>\n'
        '<section class="kdsml-classifier" dir="rtl" lang="ar">\n'
        f"<h1>{heading}</h1>\n"
        "<p>الأرقام محسوبة على بيانات التقييم؛ لا تثبت السببية أو صلاحية النموذج "
        "خارج هذه العينة. يجب تحديد الفئة الموجبة ومراجعة اختلال توازن الفئات.</p>\n"
        f"<table><tbody>{body}</tbody></table>\n"
        "<p>ROC-AUC يقيس قدرة ترتيب الحالات، ولا يقيس معايرة الاحتمالات. "
        "Average Precision يختلف عن المساحة الهندسية تحت منحنى Precision-Recall.</p>"
        "\n</section>\n</body></html>\n"
    )


def save_classification_report(metrics: Mapping, path: str | Path, *,
                               title="تقرير تقييم نموذج التصنيف") -> Path:
    """Save the Arabic model-evaluation report as a portable UTF-8 HTML file."""
    destination = Path(path)
    destination.write_text(render_classification_report(metrics, title=title), encoding="utf-8")
    return destination


def display_classification_report(metrics: Mapping, *, title="تقرير تقييم نموذج التصنيف") -> None:
    """Display the Arabic report directly inside Jupyter or Google Colab."""
    try:
        from IPython.display import HTML, display
    except ImportError as exc:
        raise ImportError(
            'display_classification_report needs IPython; install khaled-dsml[notebooks]'
        ) from exc
    display(HTML(render_classification_report(metrics, title=title)))
