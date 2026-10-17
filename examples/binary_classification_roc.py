"""Binary economic risk case study with ROC-AUC, comparison and an Arabic report.

Data are synthetic; NOT a real credit-risk tool or policy recommendation.
"""
from sklearn.datasets import make_classification

from khaled_dsml import (
    binary_classification_metrics, compare_classifiers, make_classifier,
    save_classification_report, split_data,
)
from khaled_dsml.plots import plot_roc_curve


def main():
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=4,
        n_redundant=0, random_state=14,
    )
    xt, xv, yt, yv = split_data(X, y, stratify=True, random_state=42)
    classifiers = {
        "logistic": make_classifier(xt, "logistic"),
        "tree": make_classifier(xt, "decision_tree", max_depth=4),
    }
    print(compare_classifiers(classifiers, xt, yt, xv, yv))
    model = make_classifier(xt, "logistic").fit(xt, yt)
    classes = list(model.classes_)
    probability = model.predict_proba(xv)[:, classes.index(1)]
    result = binary_classification_metrics(yv, probability)
    print("ROC-AUC:", result["roc_auc"])
    report = save_classification_report(result, "classification_ar.html")
    print("Arabic report saved:", report)
    fig, _ = plot_roc_curve(yv, probability)
    fig.savefig("roc_curve.png", dpi=160, bbox_inches="tight")
    print("ROC plot saved: roc_curve.png")


if __name__ == "__main__":
    main()
