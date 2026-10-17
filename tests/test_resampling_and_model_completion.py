"""Numerical references for corrected resampling, logit form and clusters."""
import numpy as np
import pytest
from scipy import stats
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from khaled_dsml import (
    corrected_resampled_t, five_by_two_cv_comparison,
    box_tidwell_logistic_screen, prediction_seed_stability,
    gap_statistic_kmeans, render_analysis_dashboard,
)


def test_corrected_resampled_t_agrees_with_nadeau_bengio_formula():
    d=np.array([.03,.07,.01,-.01,.08,.04,.02,.05,.01,.03])
    r=corrected_resampled_t(d,n_train=80,n_test=20)
    variance=np.var(d,ddof=1)
    t=np.mean(d)/np.sqrt((1/len(d)+20/80)*variance)
    assert r["statistic"]==pytest.approx(t)
    assert r["p_value"]==pytest.approx(2*stats.t.sf(abs(t),len(d)-1))
    assert r["n_resamples"]==len(d)
    with pytest.raises(ValueError):
        corrected_resampled_t(np.ones(5),n_train=80,n_test=20)


def test_both_five_by_two_cv_tests_match_reference_definitions():
    arr=np.array([[.05,-.01],[.09,.02],[.04,-.02],
                  [.06,.00],[.08,-.01]])
    centered=arr-arr.mean(axis=1,keepdims=True)
    sumvar=float((centered**2).sum())
    t=arr[0,0]/np.sqrt(sumvar/5)
    f=(arr**2).sum()/(2*sumvar)
    a=five_by_two_cv_comparison(arr,method="paired_t")
    b=five_by_two_cv_comparison(arr,method="combined_f")
    assert a["statistic"]==pytest.approx(t)
    assert a["p_value"]==pytest.approx(2*stats.t.sf(abs(t),5))
    assert b["statistic"]==pytest.approx(f)
    assert b["p_value"]==pytest.approx(stats.f.sf(f,10,5))
    assert len(a["table"])==10
    for invalid in (np.ones((5,2)),np.ones((4,2)),np.ones((10,))):
        with pytest.raises(ValueError):
            five_by_two_cv_comparison(invalid)


def test_box_tidwell_coefficient_matches_statsmodels_glm():
    import statsmodels.api as sm
    rng=np.random.default_rng(42)
    x=np.exp(rng.normal(1,.5,160))
    probabilities=expit(-1+.9*x-.24*x*np.log(x))
    y=rng.binomial(1,probabilities)
    result=box_tidwell_logistic_screen(y,x)
    design=np.column_stack([np.ones(len(x)),x,x*np.log(x)])
    reference=sm.GLM(y,design,family=sm.families.Binomial()).fit()
    assert result["interaction_coefficient"]==pytest.approx(reference.params[2])
    assert result["statistic"]==pytest.approx(reference.tvalues[2])
    assert result["p_value"]==pytest.approx(
        2*stats.norm.sf(abs(reference.tvalues[2])))
    with pytest.raises(ValueError):
        box_tidwell_logistic_screen(y,np.r_[0,x[1:]])


def test_prediction_seed_stability_fixed_holdout_with_cloned_models():
    X,y=make_classification(n_samples=90,n_features=4,n_redundant=0,
                             random_state=22)
    model=RandomForestClassifier(n_estimators=12,max_depth=3,random_state=1)
    r=prediction_seed_stability(model,X[:60],y[:60],X[60:],
                                seeds=(1,5,9),task="classification")
    assert len(r["table"])==30
    assert .333333<=r["mean_stability_metric"]<=1
    assert not hasattr(model,"estimators_")
    repeat=prediction_seed_stability(model,X[:60],y[:60],X[60:],
                                     seeds=(1,5,9))
    assert r["mean_stability_metric"]==repeat["mean_stability_metric"]
    with pytest.raises(ValueError):
        prediction_seed_stability(model,X[:60],y[:60],X[60:],
                                  seeds=(1,1,2))


def test_gap_statistic_matches_manual_kmeans_and_box_reference():
    rng=np.random.default_rng(19)
    X=np.r_[rng.normal([-2,-2],.35,(40,2)),
            rng.normal([2,2],.35,(40,2))]
    a=gap_statistic_kmeans(X,max_clusters=3,n_reference=5,random_state=13)
    b=gap_statistic_kmeans(X,max_clusters=3,n_reference=5,random_state=13)
    assert a["selected_k"]==b["selected_k"]
    assert np.allclose(a["table"]["gap"],b["table"]["gap"])
    km=KMeans(n_clusters=2,n_init=10,random_state=13).fit(X)
    assert a["table"].loc[1,"observed_inertia"]==pytest.approx(km.inertia_)
    assert 1<=a["selected_k"]<=3
    assert "p_value" not in a
    with pytest.raises(ValueError):
        gap_statistic_kmeans(X,max_clusters=1)


def test_added_methods_render_arabic_and_english_without_external_fonts():
    rng=np.random.default_rng(7)
    x=np.exp(rng.normal(size=100))
    y=rng.binomial(1,expit(-1+1.2*x))
    items=[
        corrected_resampled_t(np.arange(8)*.03+.01,n_train=50,n_test=20),
        five_by_two_cv_comparison(
            np.array([[.05,-.01],[.04,.01],[.02,-.04],[.06,.03],[.08,-.02]])),
        box_tidwell_logistic_screen(y,x),
        gap_statistic_kmeans(rng.normal(size=(35,2)),max_clusters=2,n_reference=5),
    ]
    for result in items:
        html=render_analysis_dashboard(result,theme="amethyst")
        assert 'dir="rtl"' in html
        assert result["interpretation_ar"] in html
        assert result["interpretation_en"] in html
