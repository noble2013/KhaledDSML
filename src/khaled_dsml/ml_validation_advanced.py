"""Leakage-aware iid model validation helpers with reproducible references.

Independent rows are assumed. These helpers are NOT suitable unchanged for
grouped subjects, repeated measurements or time-series. Put scaling, imputation
and feature selection inside an sklearn Pipeline to avoid cross-fold leakage.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import pandas as pd
from sklearn.base import clone, is_classifier
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GridSearchCV, KFold, StratifiedKFold, cross_val_score,
    learning_curve, permutation_test_score,
)
from sklearn.utils.validation import check_consistent_length


def _xy(X, y, *, min_rows=12):
    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError("X must be a 2D DataFrame or array")
    if isinstance(X, pd.DataFrame):
        if not X.columns.is_unique:
            raise ValueError("Feature names must be unique")
    n, p = X.shape
    if n < min_rows or p < 1:
        raise ValueError(f"At least {min_rows} rows and one feature are required")
    a = np.asarray(y)
    if a.ndim != 1:
        raise ValueError("y must be one-dimensional")
    check_consistent_length(X, y)
    if pd.isna(a).any():
        raise ValueError("Target labels cannot be missing")
    return n, p, a


def _folds(cv, *, minimum=2, maximum=10):
    if type(cv) is not int or not minimum <= cv <= maximum:
        raise ValueError(f"cv must be an integer between {minimum} and {maximum}")
    return cv


def _scoring(scoring):
    if not isinstance(scoring, str) or not scoring.strip():
        raise ValueError("scoring must be an explicit sklearn scorer name")
    return get_scorer(scoring)


def _splitter(y, *, cv, classifier, seed):
    if classifier:
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or counts.min() < cv:
            raise ValueError("Stratified CV requires >=2 classes with >=cv rows in each")
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    return KFold(n_splits=cv, shuffle=True, random_state=seed)


def _slice(data, indices):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[indices]
    return np.asarray(data)[indices]


def nested_cv_model_evaluation(
    estimator, X, y, param_grid, *, outer_cv=3, inner_cv=3,
    scoring="accuracy", random_state=42,
):
    """Refit hyperparameter search exclusively on each outer training fold.

    Returns outer held-out scores and inner training-only best scores.
    The final model is intentionally NOT fitted or returned. A new final
    model may be fitted after reporting the unbiased outer score estimate.
    Outer fold scores are correlated and no p-value or CI is fabricated.
    """
    n, p, labels = _xy(X, y, min_rows=24)
    _folds(outer_cv)
    _folds(inner_cv)
    scorer = _scoring(scoring)
    if not isinstance(random_state, (int, np.integer)) or isinstance(random_state, (bool, np.bool_)):
        raise ValueError("random_state must be an integer")
    if not isinstance(param_grid, (Mapping, list)) or not param_grid:
        raise ValueError("param_grid must contain at least one hyperparameter configuration")
    classifier = is_classifier(estimator)
    outer = _splitter(labels, cv=outer_cv, classifier=classifier, seed=int(random_state))
    records = []
    for fold, (train, test) in enumerate(outer.split(X, labels if classifier else None), start=1):
        ytrain = _slice(y, train)
        inner = _splitter(np.asarray(ytrain), cv=inner_cv, classifier=classifier,
                          seed=int(random_state) + fold)
        search = GridSearchCV(
            clone(estimator), param_grid=param_grid, scoring=scoring,
            cv=inner, refit=True, error_score="raise", n_jobs=1,
        )
        search.fit(_slice(X, train), ytrain)
        outer_score = float(scorer(search.best_estimator_, _slice(X, test), _slice(y, test)))
        records.append({
            "fold": fold, "n_train": len(train), "n_test": len(test),
            "inner_best_cv_score": float(search.best_score_),
            "outer_test_score": outer_score,
            "best_params": repr(search.best_params_),
        })
    table = pd.DataFrame(records)
    if not np.isfinite(table["outer_test_score"]).all():
        raise ValueError("Outer scores must be finite")
    return {
        "test": "Nested cross-validation evaluation",
        "n_samples": n, "n_features": p,
        "outer_cv": outer_cv, "inner_cv": inner_cv,
        "scoring": scoring,
        "outer_score_mean": float(table["outer_test_score"].mean()),
        "outer_score_std": float(table["outer_test_score"].std(ddof=1)),
        "table": table,
        "interpretation_ar": (
            "فُصل اختيار المعلمات في الطيات الداخلية عن التقييم على الطيات "
            "الخارجية المستقلة عنها. تتطلب الطريقة مشاهدات مستقلة وسلسلة "
            "Pipeline تضم جميع خطوات التجهيز؛ درجات الطيات مترابطة ولا تشكل "
            "قيم p أو فواصل ثقة. لا يُدرّب نموذج نهائي تلقائيًا."
        ),
        "interpretation_en": (
            "Hyperparameter selection is confined to inner training folds, "
            "with outer held-out scoring. Requires iid rows and an "
            "end-to-end Pipeline for learned preprocessing. Correlated "
            "fold scores are descriptive, not p-values or confidence intervals."
        ),
    }


def permutation_score_significance(
    estimator, X, y, *, scoring="accuracy", cv=3,
    n_permutations=99, random_state=42,
):
    """CV score against label permutations under exchangeability.

    The permutation p-value is (1+null scores >= observed)/(B+1), as
    computed by sklearn. It is valid only when labels are exchangeable under
    H0, rows are iid, no global preprocessing/selection has leaked, and the
    whole model/pipeline is refitted for each permutation. It does not
    validate causal claims or adjust for multiple hypotheses.
    """
    n, p, labels = _xy(X, y, min_rows=18)
    _folds(cv)
    _scoring(scoring)
    if type(n_permutations) is not int or not 19 <= n_permutations <= 10000:
        raise ValueError("n_permutations must be an integer between 19 and 10000")
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer)):
        raise ValueError("random_state must be integer")
    classifier = is_classifier(estimator)
    splitter = _splitter(labels, cv=cv, classifier=classifier, seed=int(random_state))
    observed, scores, pval = permutation_test_score(
        clone(estimator), X, y, cv=splitter, scoring=scoring,
        n_permutations=n_permutations, random_state=int(random_state),
        n_jobs=1,
    )
    if not np.isfinite(scores).all():
        raise ValueError("Permutation scores must be finite")
    return {
        "test": "Permutation CV score significance",
        "n_samples": n, "n_features": p, "scoring": scoring,
        "n_permutations": n_permutations, "cv": cv,
        "observed_cv_score": float(observed),
        "null_score_mean": float(np.mean(scores)),
        "p_value": float(pval),
        "monte_carlo_resolution": 1/(n_permutations+1),
        "table": pd.DataFrame({
            "permutation": np.arange(1, n_permutations+1),
            "null_cv_score": scores,
        }),
        "interpretation_ar": (
            "قيمة احتمالية لاختبار تبديل الملصقات بشرط قابلية تبادلها تحت "
            "فرض العدم، واستقلالية الحالات، وإعادة تدريب النموذج داخل كل "
            "محاكاة. لا تُصحح تعدد المقارنات، وقرار اختيار النموذج قبل "
            "الاختبار قد يغيّر صلاحية الدلالة."
        ),
        "interpretation_en": (
            "CV permutation p-value conditional on label exchangeability "
            "under the null and iid rows, with a newly fitted estimator "
            "inside each permutation. No multiple-testing correction or "
            "causal inference is implied; avoid data leakage in preprocessing."
        ),
    }


def holdout_permutation_importance(
    fitted_estimator, X_holdout, y_holdout, *, scoring="accuracy",
    n_repeats=10, random_state=42,
):
    """Permutation importance on an independently reserved holdout set.

    The estimator must already be fitted without access to holdout labels,
    and all preprocessing must be part of its pipeline. Correlated features
    may dilute importance. Negative estimates are possible. Returned SD
    quantifies random permutations, not population inference or causal effects.
    """
    n, p, _ = _xy(X_holdout, y_holdout, min_rows=12)
    scorer = _scoring(scoring)
    if type(n_repeats) is not int or not 2 <= n_repeats <= 1000:
        raise ValueError("n_repeats must be an integer between 2 and 1000")
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer)):
        raise ValueError("random_state must be an integer")
    base = float(scorer(fitted_estimator, X_holdout, y_holdout))
    r = permutation_importance(
        fitted_estimator, X_holdout, y_holdout, scoring=scoring,
        n_repeats=n_repeats, random_state=int(random_state), n_jobs=1,
    )
    features = (list(map(str, X_holdout.columns)) if isinstance(X_holdout, pd.DataFrame)
                else [f"feature_{i}" for i in range(p)])
    table = pd.DataFrame({
        "feature": features,
        "mean_score_decrease": r.importances_mean,
        "sd_permutation_decrease": r.importances_std,
    }).sort_values("mean_score_decrease", ascending=False, kind="mergesort").reset_index(drop=True)
    return {
        "test": "Holdout permutation feature importance",
        "n_samples": n, "n_features": p, "n_repeats": n_repeats,
        "scoring": scoring, "baseline_holdout_score": base,
        "top_mean_score_decrease": float(table["mean_score_decrease"].max()),
        "table": table,
        "interpretation_ar": (
            "أهمية السمات مع إفساد قيم كل سمة على مجموعة اختبار لم تُستخدم "
            "في التدريب أو الاختيار. أثر السمات المترابطة قد يتوزع بينها. "
            "الانحراف المعروض ناتج عن إعادة التبديل وليس فاصل ثقة أو أثرًا سببيًا."
        ),
        "interpretation_en": (
            "Feature importance from repeated permutations on a separately "
            "reserved holdout, fitted estimator unchanged. Correlated "
            "features may share importance; variation over permutations "
            "is not a population confidence interval or causal evidence."
        ),
    }


def learning_curve_diagnostic(
    estimator, X, y, *, cv=3, scoring="accuracy",
    train_sizes=(0.5, 0.75, 1.0), random_state=42,
):
    """Descriptive cross-validation learning curve, not an external test.

    Every fold clones/fits estimator with its own training subset.
    All data-dependent preparation belongs inside the supplied Pipeline.
    """
    n, p, labels = _xy(X, y, min_rows=24)
    _folds(cv)
    _scoring(scoring)
    sizes = np.asarray(train_sizes, dtype=float)
    if (sizes.ndim != 1 or not 2 <= len(sizes) <= 12 or
        not np.isfinite(sizes).all() or
        np.any(sizes <= 0) or np.any(sizes > 1) or
        np.any(np.diff(sizes) <= 0)):
        raise ValueError("train_sizes must be strictly increasing fractions in (0,1]")
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer)):
        raise ValueError("random_state must be an integer")
    classifier = is_classifier(estimator)
    splitter = _splitter(labels, cv=cv, classifier=classifier, seed=int(random_state))
    sizes_abs, train_scores, validation_scores = learning_curve(
        clone(estimator), X, y, train_sizes=sizes, cv=splitter,
        scoring=scoring, shuffle=True, random_state=int(random_state),
        n_jobs=1, error_score="raise",
    )
    if len(np.unique(sizes_abs)) != len(sizes_abs):
        raise ValueError("train_sizes collapse to repeated integer sample counts")
    table = pd.DataFrame({
        "n_train": sizes_abs,
        "train_score_mean": train_scores.mean(axis=1),
        "train_score_std": train_scores.std(axis=1, ddof=1),
        "validation_score_mean": validation_scores.mean(axis=1),
        "validation_score_std": validation_scores.std(axis=1, ddof=1),
    })
    return {
        "test": "Cross-validation learning curve",
        "n_samples": n, "n_features": p, "cv": cv, "scoring": scoring,
        "n_train_sizes": len(table), "last_validation_score": float(table["validation_score_mean"].iloc[-1]),
        "table": table,
        "interpretation_ar": (
            "منحنى تعلّم وصفي يقارن أداء التدريب والتحقق عبر أحجام عينات "
            "متصاعدة. ينبغي وضع كل خطوات التجهيز داخل Pipeline؛ "
            "لا يمثل نقاط اختبار خارجي مستقلة أو قيمة p، وقد لا يلائم "
            "البيانات الزمنية أو المجموعات المعتمدة."
        ),
        "interpretation_en": (
            "Descriptive CV learning curve across growing training sizes. "
            "Place every learned preprocessing step within a Pipeline. "
            "Fold scores are correlated and are not external validation, "
            "p-values, or evidence of causal effects."
        ),
    }


def dummy_baseline_cv(X, y, *, task="classification", scoring=None, cv=3,
                      random_state=42):
    """Fold-wise honest dummy baseline to check whether a model adds value.

    In classification, most-frequent dummy; in regression, train-mean
    dummy. Training target statistics are computed within each fold only.
    """
    n, p, labels = _xy(X, y, min_rows=18)
    _folds(cv)
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'")
    if scoring is None:
        scoring = "accuracy" if task == "classification" else "neg_mean_absolute_error"
    _scoring(scoring)
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, (int, np.integer)):
        raise ValueError("random_state must be an integer")
    classifier = task == "classification"
    splitter = _splitter(labels, cv=cv, classifier=classifier, seed=int(random_state))
    model = (DummyClassifier(strategy="most_frequent") if classifier
             else DummyRegressor(strategy="mean"))
    scores = cross_val_score(model, X, y, scoring=scoring, cv=splitter,
                             error_score="raise", n_jobs=1)
    return {
        "test": "Cross-validation dummy baseline",
        "n_samples": n, "n_features": p, "task": task,
        "scoring": scoring, "cv": cv,
        "mean_cv_score": float(scores.mean()), "sd_cv_score": float(scores.std(ddof=1)),
        "table": pd.DataFrame({"fold": np.arange(1, cv+1), "validation_score": scores}),
        "interpretation_ar": (
            "خط أساس ساذج يقيس أداء التنبؤ بالفئة الأكثر تكرارًا أو متوسط "
            "متغير الاستجابة داخل التدريب في كل طية. لا يُستخدم تسرب "
            "لبيانات التحقق؛ لا تفسر درجات الطيات كاختبار فرضيات."
        ),
        "interpretation_en": (
            "Training-fold-only dummy model: most frequent class or "
            "mean response. A basic descriptive benchmark, not a hypothesis "
            "test; use grouped/time-series validation if observations are dependent."
        ),
    }
