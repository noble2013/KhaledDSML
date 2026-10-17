"""Model evaluation indicators tested against scikit-learn references."""
import numpy as np
import pytest
from sklearn import metrics
from khaled_dsml import (
    classification_diagnostics, calibration_summary, detailed_regression_metrics,
    cluster_validation_metrics, binary_group_fairness, time_series_split_indices,
    repeated_cv_indices, model_overfit_gap,
)


def test_classifier_matches_sklearn_reference():
    y=[0,0,0,1,1,1]
    p=[.1,.2,.6,.4,.8,.9]
    out=classification_diagnostics(y,p,n_bins=3)
    predicted=(np.array(p)>=.5).astype(int)
    assert out["mcc"]==pytest.approx(metrics.matthews_corrcoef(y,predicted))
    assert out["cohen_kappa"]==pytest.approx(metrics.cohen_kappa_score(y,predicted))
    assert out["log_loss"]==pytest.approx(metrics.log_loss(y,p))
    assert out["brier"]==pytest.approx(metrics.brier_score_loss(y,p))
    precision,recall,_=metrics.precision_recall_curve(y,p)
    assert out["pr_auc"]==pytest.approx(metrics.auc(recall,precision))
    assert out["average_precision"]==pytest.approx(metrics.average_precision_score(y,p))
    assert "out-of-sample" in out["interpretation_en"]


def test_calibration_ece_is_weighted_and_handles_empty_bins():
    y=[0,0,1,1]
    p=[.1,.1,.9,.9]
    r=calibration_summary(y,p,n_bins=10)
    assert r["expected_calibration_error"]==pytest.approx(.1)
    assert r["bins"]["count"].sum()==4
    assert len(r["bins"])==2


def test_regression_mape_zero_and_adjusted_r2():
    y=[2,4,6,8,10,12]
    p=[1,5,6,8,11,12]
    r=detailed_regression_metrics(y,p,n_predictors=2)
    assert r["mae"]==pytest.approx(np.mean(np.abs(np.array(y)-p)))
    assert r["r2"]==pytest.approx(metrics.r2_score(y,p))
    assert r["adjusted_r2"]==pytest.approx(1-(1-r["r2"])*5/3)
    no_mape=detailed_regression_metrics([0,1,2],[0,1,3])
    assert no_mape["mape_percent"] is None
    assert no_mape["smape_percent"]>=0


def test_cluster_scores_reference_and_external_agreement():
    from sklearn.datasets import make_blobs
    X,y=make_blobs(n_samples=120,centers=3,random_state=42)
    r=cluster_validation_metrics(X,y,reference=y)
    assert r["calinski_harabasz"]==pytest.approx(metrics.calinski_harabasz_score(X,y))
    assert r["adjusted_rand"]==pytest.approx(1)
    assert r["normalized_mutual_information"]==pytest.approx(1)
    with pytest.raises(ValueError):
        cluster_validation_metrics(X,[1]*len(y))


def test_group_fairness_gap_and_80pct_rule_context():
    y=[1,0,1,0,1,0,1,0]
    p=[1,0,0,0,1,1,1,0]
    groups=["A"]*4+["B"]*4
    out=binary_group_fairness(y,p,groups)
    assert out["group_rates"].shape==(2,5)
    assert out["demographic_parity_gap"]==pytest.approx(.5)
    assert out["equal_opportunity_gap"]==pytest.approx(.5)
    assert out["selection_rate_ratio"]==pytest.approx(1/3)
    assert "not a legal" in out["interpretation_en"]
    with pytest.raises(ValueError):
        binary_group_fairness([1,0,1,0],[1,0,1,0],["A","A","A","A"])


def test_chronological_cv_gap_no_lookahead():
    splits=time_series_split_indices(40,n_splits=4,gap=2,test_size=5)
    assert len(splits)==4
    for train,test in splits:
        assert max(train)+2<min(test)
        assert not set(train)&set(test)
    with pytest.raises(ValueError):
        time_series_split_indices(8,n_splits=4,gap=4,test_size=2)


def test_repeated_stratified_split_reproducible():
    y=np.tile([0,1],20)
    a=repeated_cv_indices(y,folds=5,repeats=2,seed=1)
    b=repeated_cv_indices(y,folds=5,repeats=2,seed=1)
    assert len(a)==10
    for (tr,te),(tr2,te2) in zip(a,b):
        assert np.array_equal(tr,tr2)
        assert np.array_equal(te,te2)
        assert set(tr).isdisjoint(te)


def test_overfit_gap_is_not_a_claim():
    r=model_overfit_gap(.99,.72)
    assert r["optimism_gap"]==pytest.approx(.27)
    assert "not definitive" in r["interpretation_en"]
