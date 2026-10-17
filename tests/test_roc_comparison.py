"""Correlated ROC area comparisons tested against empirical rank references."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from khaled_dsml import (
    delong_auc_comparison, stratified_paired_auc_bootstrap,
    render_analysis_dashboard,
)


@pytest.fixture
def pair():
    rng=np.random.default_rng(42)
    y=np.array([0]*80+[1]*80)
    a=np.concatenate([rng.normal(0,1,80),rng.normal(.9,1,80)])
    b=np.concatenate([rng.normal(0,1,80),rng.normal(.5,1,80)])
    return y,a,b


def test_delong_auc_equals_sklearn_for_each_model(pair):
    y,a,b=pair
    r=delong_auc_comparison(y,a,b)
    assert r["roc_auc_a"]==pytest.approx(roc_auc_score(y,a))
    assert r["roc_auc_b"]==pytest.approx(roc_auc_score(y,b))
    assert r["auc_difference"]==pytest.approx(roc_auc_score(y,a)-roc_auc_score(y,b))
    assert r["variance_difference"]>0
    assert 0<=r["p_value"]<=1
    assert r["ci_lower"]<=r["auc_difference"]<=r["ci_upper"]
    assert "DeLong" in render_analysis_dashboard(r)


def test_delong_symmetry_and_monotonic_transformation(pair):
    y,a,b=pair
    r=delong_auc_comparison(y,a,b)
    reverse=delong_auc_comparison(y,b,a)
    assert r["auc_difference"]==pytest.approx(-reverse["auc_difference"])
    assert r["p_value"]==pytest.approx(reverse["p_value"])
    transformed=delong_auc_comparison(y,a*10+5,b*2+7)
    assert r["roc_auc_a"]==pytest.approx(transformed["roc_auc_a"])
    assert r["p_value"]==pytest.approx(transformed["p_value"])


def test_delong_ties_receive_half_credit():
    y=np.array([0,0,0,0,0,1,1,1,1,1])
    a=np.array([.3,.3,.5,.6,.1,.3,.3,.5,.9,.8])
    b=np.array([.2,.1,.5,.5,.3,.8,.6,.7,.6,.5])
    r=delong_auc_comparison(y,a,b)
    assert r["roc_auc_a"]==pytest.approx(roc_auc_score(y,a))
    assert r["roc_auc_b"]==pytest.approx(roc_auc_score(y,b))


def test_delong_identical_scores_cannot_invent_p_value(pair):
    y,a,_=pair
    with pytest.raises(ValueError,match="variance is zero"):
        delong_auc_comparison(y,a,a)
    with pytest.raises(ValueError):
        delong_auc_comparison(y,a[:20],a[:20])
    with pytest.raises(ValueError):
        delong_auc_comparison([0]*5+[1]*2,list(range(7)),list(range(7)))


def test_paired_stratified_bootstrap_deterministic_and_not_hypothesis_test(pair):
    y,a,b=pair
    r=stratified_paired_auc_bootstrap(y,a,b,repetitions=150,seed=10)
    s=stratified_paired_auc_bootstrap(y,a,b,repetitions=150,seed=10)
    assert r==s
    assert "p_value" not in r
    assert -1<=r["ci_lower"]<=r["ci_upper"]<=1
    assert r["n_samples"]==len(y)
    assert r["auc_difference"]==pytest.approx(roc_auc_score(y,a)-roc_auc_score(y,b))


def test_auc_bootstrap_invalid_iteration_count(pair):
    y,a,b=pair
    with pytest.raises(ValueError):
        stratified_paired_auc_bootstrap(y,a,b,repetitions=20)
    with pytest.raises(ValueError):
        delong_auc_comparison(y,a,b,alpha=2)
