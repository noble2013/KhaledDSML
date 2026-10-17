"""Optional classroom figures; matplotlib is installed through [plots]."""
from __future__ import annotations


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError('Install with: pip install "khaled-dsml[plots]"') from exc
    return plt


def plot_missingness(df):
    plt = _plt()
    fig, ax = plt.subplots()
    (100 * df.isna().mean()).plot.bar(ax=ax)
    ax.set(ylabel="Missing values (%)", title="Missingness by column")
    fig.tight_layout()
    return fig, ax


def plot_monthly(monthly):
    plt = _plt()
    fig, ax = plt.subplots()
    monthly.plot(ax=ax)
    ax.set(ylabel="Total (DA)", title="Monthly transaction amounts")
    fig.tight_layout()
    return fig, ax


def plot_confusion(confusion):
    import numpy as np
    plt = _plt()
    cm = np.asarray(confusion)
    fig, ax = plt.subplots()
    ax.imshow(cm)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center")
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix")
    fig.tight_layout()
    return fig, ax


def plot_roc_curve(y_true, y_probability, *, positive_label=1, ax=None):
    """Plot a binary ROC curve from held-out positive-class probabilities.

    Returns (figure, axes). Install the optional [plots] extra for matplotlib.
    The diagonal is the performance of a random ranking, not a fitted model.
    """
    from .ml_evaluation import binary_classification_metrics, roc_curve_table

    plt = _plt()
    points = roc_curve_table(y_true, y_probability, positive_label=positive_label)
    auc = binary_classification_metrics(
        y_true, y_probability, positive_label=positive_label
    )["roc_auc"]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(points["fpr"], points["tpr"], label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random ranking")
    ax.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Binary ROC curve",
        xlim=(0, 1), ylim=(0, 1),
    )
    ax.legend()
    fig.tight_layout()
    return fig, ax
