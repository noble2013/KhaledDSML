"""Reference checks of leakage-aware iid machine-learning validation."""
import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GridSearchCV, KFold, StratifiedKFold, cross_val_score,
    learning_curve, permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from khaled_dsml import (
    nested_cv_model_evaluation, permutation_score_significance,
    holdout_permutation_importance, learning_curve_diagnostic,
    dummy_baseline_cv, render_analysis_dashboard,
)


def _classification(n=90):
    X, y = make_classification(
        n_samples=n, n_features=4, n_informative=3,
        n_redundant=0, class_sep=1.4, random_state=11,
    )
    return pd.DataFrame(X, columns=["signal", "noise_a", "noise_b", "noise_c"]), pd.Series(y)


def _pipeline():
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=350, solver="liblinear"))
    ])


def test_nested_cv_matches_manual_external_inner_search_scores():
    X, y = _classification(75)
    estimator = _pipeline()
    grid = {"model__C": [.1, 1.]}
    r = nested_cv_model_evaluation(estimator, X, y, grid,
                                    outer_cv=3, inner_cv=2,
                                    scoring="accuracy", random_state=24)
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=24)
    scores = []
    for fold, (tr, te) in enumerate(outer.split(X, y), 1):
        inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=24+fold)
        search = GridSearchCV(clone(estimator), grid, cv=inner,
                              scoring="accuracy", error_score="raise", n_jobs=1)
        search.fit(X.iloc[tr], y.iloc[tr])
        scores.append(search.score(X.iloc[te], y.iloc[te]))
    np.testing.assert_allclose(r["table"]["outer_test_score"], scores)
    assert r["outer_score_mean"] == pytest.approx(np.mean(scores))
    assert r["table"]["n_test"].sum() == len(X)
    assert "p_value" not in r
    # Original estimator must not be fitted or modified in nested CV.
    assert not hasattr(estimator["model"], "coef_")


def test_permutation_cv_matches_sklearn_paired_random_seed():
    X, y = _classification(60)
    model = _pipeline()
    r = permutation_score_significance(
        model, X, y, scoring="accuracy", cv=3,
        n_permutations=19, random_state=9,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=9)
    score, null, p = permutation_test_score(
        clone(model), X, y, cv=cv, scoring="accuracy",
        n_permutations=19, random_state=9, n_jobs=1,
    )
    assert r["observed_cv_score"] == pytest.approx(score)
    np.testing.assert_allclose(r["table"]["null_cv_score"], null)
    assert r["p_value"] == pytest.approx(p)
    assert r["monte_carlo_resolution"] == pytest.approx(.05)
    assert not hasattr(model["model"], "coef_")


def test_holdout_permutation_importance_matches_sklearn_and_feature_labels():
    X, y = make_regression(n_samples=110, n_features=3, n_informative=1,
                           noise=1.0, random_state=31)
    names = ["first", "second", "third"]
    train = pd.DataFrame(X[:65], columns=names)
    holdout = pd.DataFrame(X[65:], columns=names)
    model = LinearRegression().fit(train, y[:65])
    before_coef = model.coef_.copy()
    r = holdout_permutation_importance(
        model, holdout, y[65:], scoring="r2", n_repeats=6, random_state=33,
    )
    ref = permutation_importance(
        model, holdout, y[65:], scoring="r2", n_repeats=6,
        random_state=33, n_jobs=1,
    )
    expected = dict(zip(names, ref.importances_mean))
    for row in r["table"].to_dict("records"):
        assert row["mean_score_decrease"] == pytest.approx(expected[row["feature"]])
    assert sorted(r["table"]["feature"]) == sorted(names)
    assert r["baseline_holdout_score"] == pytest.approx(model.score(holdout,y[65:]))
    np.testing.assert_allclose(model.coef_, before_coef)
    assert "p_value" not in r


def test_learning_curve_matches_fixed_sklearn_reference_and_is_deterministic():
    X,y=_classification(90)
    model=_pipeline()
    r=learning_curve_diagnostic(
        model,X,y,cv=3,scoring="accuracy",
        train_sizes=(.5,.75,1.),random_state=8)
    splitter=StratifiedKFold(n_splits=3,shuffle=True,random_state=8)
    sizes,tr,va=learning_curve(
        clone(model),X,y,train_sizes=np.asarray([.5,.75,1.]),cv=splitter,
        scoring="accuracy",shuffle=True,random_state=8,
        n_jobs=1,error_score="raise")
    np.testing.assert_array_equal(r["table"]["n_train"],sizes)
    np.testing.assert_allclose(r["table"]["train_score_mean"],tr.mean(axis=1))
    np.testing.assert_allclose(r["table"]["validation_score_mean"],va.mean(axis=1))
    assert r["last_validation_score"] == pytest.approx(va.mean(axis=1)[-1])
    assert "p_value" not in r


def test_dummy_baselines_match_fold_only_sklearn_classification_and_regression():
    X,y=_classification(90)
    r=dummy_baseline_cv(X,y,task="classification",scoring="accuracy",cv=3,
                        random_state=19)
    split=StratifiedKFold(n_splits=3,shuffle=True,random_state=19)
    ref=cross_val_score(DummyClassifier(strategy="most_frequent"),X,y,
                        scoring="accuracy",cv=split,n_jobs=1)
    np.testing.assert_allclose(r["table"]["validation_score"],ref)
    assert r["mean_cv_score"] == pytest.approx(ref.mean())
    x,yy=make_regression(n_samples=72,n_features=3,random_state=5)
    reg=dummy_baseline_cv(x,yy,task="regression",scoring="neg_mean_absolute_error",
                          cv=3,random_state=7)
    refreg=cross_val_score(DummyRegressor(strategy="mean"),x,yy,
                            scoring="neg_mean_absolute_error",
                            cv=KFold(n_splits=3,shuffle=True,random_state=7))
    np.testing.assert_allclose(reg["table"]["validation_score"],refreg)


@pytest.mark.parametrize("kwargs",[
    {"outer_cv":1},{"inner_cv":11},{"scoring":"nonexistent_scorer"},
    {"param_grid":{}},{"param_grid":None},{"random_state":True},
])
def test_nested_rejects_invalid_configurations(kwargs):
    X,y=_classification(50)
    base=dict(param_grid={"model__C":[.1,1.]})
    base.update(kwargs)
    with pytest.raises((ValueError,TypeError)):
        nested_cv_model_evaluation(_pipeline(),X,y,**base)


@pytest.mark.parametrize("kwargs",[
    {"n_permutations":0},{"n_permutations":18},
    {"n_permutations":True},{"cv":1},{"scoring":"unknown"},
])
def test_permutation_test_rejects_invalid_configurations(kwargs):
    X,y=_classification(50)
    with pytest.raises((ValueError,TypeError)):
        permutation_score_significance(_pipeline(),X,y,**kwargs)


def test_stratification_rejects_too_few_minority_cases():
    rng=np.random.default_rng(1)
    x=rng.normal(size=(25,2))
    y=np.array([0]*23+[1]*2)
    with pytest.raises(ValueError,match="Stratified"):
        dummy_baseline_cv(x,y,cv=3)
    with pytest.raises(ValueError,match="Stratified"):
        nested_cv_model_evaluation(
            LogisticRegression(max_iter=300),x,y,{"C":[1.]},outer_cv=3,inner_cv=2)


@pytest.mark.parametrize("kwargs",[
    {"n_repeats":1},{"n_repeats":True},{"n_repeats":1001},
    {"scoring":"nonexistent"},{"random_state":True},
])
def test_importance_rejects_invalid_parameters(kwargs):
    X,y=_classification(35)
    trained=_pipeline().fit(X.iloc[:20],y.iloc[:20])
    with pytest.raises((ValueError,TypeError)):
        holdout_permutation_importance(trained,X.iloc[20:],y.iloc[20:],**kwargs)


@pytest.mark.parametrize("sizes",[
    (0.0,.5,1.),(.5,.5,1.),(.7,.4,1.),
    (.5,1.2),(.5,),(.5,float("nan")),
])
def test_learning_curve_rejects_invalid_train_sizes(sizes):
    X,y=_classification(50)
    with pytest.raises(ValueError):
        learning_curve_diagnostic(_pipeline(),X,y,train_sizes=sizes)


def test_bilingual_analytics_studio_with_positive_and_negative_signed_scores():
    X,y=_classification(45)
    nested=nested_cv_model_evaluation(_pipeline(),X,y,
                                      {"model__C":[1.]},outer_cv=3,inner_cv=2)
    fitted=_pipeline().fit(X.iloc[:30],y.iloc[:30])
    important=holdout_permutation_importance(fitted,X.iloc[30:],y.iloc[30:],
                                              n_repeats=2)
    dummy=dummy_baseline_cv(X,y)
    r=learning_curve_diagnostic(_pipeline(),X,y)
    for obj in (nested,important,dummy,r):
        html=render_analysis_dashboard(obj,theme="aurora")
        assert 'dir="rtl"' in html
        assert obj["interpretation_ar"] in html
        assert obj["interpretation_en"] in html
        assert "<table" in html
