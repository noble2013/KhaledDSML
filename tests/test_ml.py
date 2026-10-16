import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from khaled_dsml import (split_data, make_classifier, make_regressor,
    classification_metrics, regression_metrics, cross_validation_summary,
    cluster_data, pca_summary)


def test_regression_pipeline():
    X, y = make_regression(n_samples=120, n_features=4, noise=2, random_state=4)
    xt, xv, yt, yv = split_data(X, y)
    estimator = make_regressor(xt, "ridge", alpha=1)
    estimator.fit(xt, yt)
    metrics = regression_metrics(yv, estimator.predict(xv))
    assert metrics["r2"] > .9 and metrics["rmse"] >= 0


def test_regression_dataframe_categoricals_and_missing():
    df = pd.DataFrame({"amount": [1, 2, None, 4, 5, 6, 7, 8],
                       "wilaya": ["Alger", "Oran", "Alger", "Oran"]*2})
    model = make_regressor(df, "linear").fit(df, [1, 2, 3, 4, 5, 6, 7, 8])
    pred = model.predict(pd.DataFrame({"amount": [3], "wilaya": ["Setif"]}))
    assert len(pred) == 1


def test_classification():
    X, y = make_classification(n_samples=160, n_features=5, n_informative=3,
                               n_redundant=0, random_state=3)
    xt, xv, yt, yv = split_data(X, y, stratify=True)
    model = make_classifier(xt, "logistic").fit(xt, yt)
    metrics = classification_metrics(yv, model.predict(xv))
    assert 0 <= metrics["f1_macro"] <= 1
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == len(yv)


def test_cross_validation():
    X, y = make_classification(n_samples=100, n_features=4, n_redundant=0, random_state=42)
    clf = make_classifier(X, "decision_tree", max_depth=3)
    result = cross_validation_summary(clf, X, y, task="classification", folds=4)
    assert len(result["fold_scores"]) == 4
    assert result["scoring"] == "f1_macro"


def test_regressor_methods():
    X = np.arange(20).reshape(10, 2)
    for algorithm in ["linear", "ridge", "lasso", "decision_tree", "random_forest"]:
        model = make_regressor(X, algorithm, n_estimators=4) if algorithm == "random_forest" else make_regressor(X, algorithm)
        assert len(model.fit(X, np.arange(10)).predict(X)) == 10


def test_classifier_methods():
    X = np.arange(40).reshape(20, 2)
    y = [0, 1] * 10
    for algorithm in ["logistic", "knn", "decision_tree", "random_forest"]:
        model = make_classifier(X, algorithm, n_estimators=5) if algorithm == "random_forest" else make_classifier(X, algorithm)
        assert len(model.fit(X, y).predict(X)) == 20


def test_clusters():
    rng = np.random.default_rng(20)
    X = np.vstack([rng.normal(-5, .1, (20, 2)), rng.normal(5, .1, (20, 2))])
    for method in ["kmeans", "hierarchical"]:
        result = cluster_data(X, method=method, n_clusters=2)
        assert result.silhouette is not None and result.silhouette > .8
        assert len(result.labels) == len(X)


def test_pca():
    rng = np.random.default_rng(10)
    result = pca_summary(rng.normal(size=(30, 4)), n_components=2)
    assert result["scores"].shape == (30, 2)
    assert sum(result["explained_variance_ratio"]) <= 1.01


def test_invalid_methods():
    with pytest.raises(ValueError):
        make_classifier([[1]], "not_a_model")
    with pytest.raises(ValueError):
        cluster_data([[1, 2], [3, 4]], n_clusters=3)
