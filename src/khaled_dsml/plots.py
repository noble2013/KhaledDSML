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
