"""Leakage-aware ML workflows and metrics for classroom use."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, recall_score,
    r2_score, silhouette_score, davies_bouldin_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def split_data(X, y, *, test_size=.2, random_state=42, stratify=False):
    """Always split before computing any preprocessing statistics."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state,
                            stratify=y if stratify else None)


def _preprocessor(X):
    if isinstance(X, pd.DataFrame):
        numeric = X.select_dtypes(include="number").columns.tolist()
        categorical = [c for c in X.columns if c not in numeric]
        transformers = []
        if numeric:
            transformers.append(("numeric", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric))
        if categorical:
            transformers.append(("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical))
        if not transformers:
            raise ValueError("X must contain at least one feature")
        return ColumnTransformer(transformers)
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])


def make_regressor(X, algorithm="linear", random_state=42, **kwargs) -> Pipeline:
    """Unfitted pipeline. Fit on training data only."""
    algorithms = {
        "linear": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "decision_tree": DecisionTreeRegressor,
        "random_forest": RandomForestRegressor,
    }
    if algorithm not in algorithms:
        raise ValueError("algorithm must be: " + ", ".join(algorithms))
    cls = algorithms[algorithm]
    if algorithm in ("decision_tree", "random_forest"):
        kwargs.setdefault("random_state", random_state)
    return Pipeline([("preprocess", _preprocessor(X)), ("model", cls(**kwargs))])


def make_classifier(X, algorithm="logistic", random_state=42, **kwargs) -> Pipeline:
    """Unfitted pipeline with imputation and categoricals inside CV folds."""
    algorithms = {
        "logistic": LogisticRegression,
        "knn": KNeighborsClassifier,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
    }
    if algorithm not in algorithms:
        raise ValueError("algorithm must be: " + ", ".join(algorithms))
    cls = algorithms[algorithm]
    if algorithm in ("logistic", "decision_tree", "random_forest"):
        kwargs.setdefault("random_state", random_state)
    if algorithm == "logistic":
        kwargs.setdefault("max_iter", 1000)
    return Pipeline([("preprocess", _preprocessor(X)), ("model", cls(**kwargs))])


def regression_metrics(y_true, y_pred) -> dict:
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred))}


def classification_metrics(y_true, y_pred) -> dict:
    """Macro averaging treats every class equally, including minority classes."""
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()}


def cross_validation_summary(estimator, X, y, *, task="regression", folds=5,
                             random_state=42, scoring=None) -> dict:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if task not in ("regression", "classification"):
        raise ValueError("task must be regression or classification")
    cv = (KFold(n_splits=folds, shuffle=True, random_state=random_state)
          if task == "regression" else
          StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state))
    score = scoring or ("r2" if task == "regression" else "f1_macro")
    values = cross_val_score(estimator, X, y, cv=cv, scoring=score)
    return {"scoring": score, "fold_scores": values.tolist(),
            "mean": float(np.mean(values)), "std": float(np.std(values))}


@dataclass
class ClusterResult:
    labels: np.ndarray
    silhouette: float | None
    davies_bouldin: float | None
    model: object


def cluster_data(X, *, method="kmeans", n_clusters=3, random_state=42,
                 scale=True) -> ClusterResult:
    x = np.asarray(X, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("Clustering requires a finite numeric 2D matrix")
    if not 2 <= n_clusters <= len(x):
        raise ValueError("Need 2 <= n_clusters <= number of rows")
    data = StandardScaler().fit_transform(x) if scale else x
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError("method must be kmeans or hierarchical")
    labels = model.fit_predict(data)
    n_unique = len(np.unique(labels))
    silhouette = (float(silhouette_score(data, labels))
                  if 2 <= n_unique < len(x) else None)
    db = float(davies_bouldin_score(data, labels)) if 2 <= n_unique < len(x) else None
    return ClusterResult(labels, silhouette, db, model)


def pca_summary(X, n_components=2, *, scale=True) -> dict:
    x = np.asarray(X, dtype=float)
    if x.ndim != 2 or not np.all(np.isfinite(x)):
        raise ValueError("PCA needs complete finite numeric 2D data")
    data = StandardScaler().fit_transform(x) if scale else x
    model = PCA(n_components=n_components)
    components = model.fit_transform(data)
    return {"scores": components, "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
            "components": model.components_, "model": model}
