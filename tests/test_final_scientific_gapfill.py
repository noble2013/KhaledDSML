"""Independent numerical and adversarial tests for final v0.10 gapfill."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import get_scorer

from khaled_dsml import (
    bernoulli_sprt, cuped_randomized_effect,
    shard_distribution_consistency, practical_proportion_difference,
    nested_ols_wald_lr_score, regression_pseudo_r_squared,
    cohen_f_squared_nested_models, two_sample_power_sensitivity,
    leave_one_out_cv_evaluation, hopkins_cluster_tendency,
    holdout_noise_robustness, sample_population_comparison,
    early_late_nonresponse_screen, scientific_report_alerts,
    hyperloglog_cardinality, minhash_lsh_candidates,
    render_analysis_dashboard,
)


def test_sprt_first_crossing_and_indeterminate_trace():
    r=bernoulli_sprt([1]*100,p0=.3,p1=.7)
    llr=np.cumsum(np.repeat(np.log(.7/.3),100))
    boundary=np.log(.8/.05)
    expected=int(np.flatnonzero(llr>=boundary)[0])
    assert r["stopping_index"]==expected
    assert r["decision"]=="favor_H1"
    assert r["log_likelihood_ratio"]==pytest.approx(llr[expected])
    assert len(r["table"])==expected+1
    undecided=bernoulli_sprt([1],p0=.49,p1=.51)
    assert undecided["decision"]=="inconclusive"
    r0=bernoulli_sprt([0]*100,p0=.3,p1=.7)
    assert r0["decision"]=="favor_H0"


@pytest.mark.parametrize("x,kwargs",[
    ([0,2],dict(p0=.2,p1=.6)),
    ([0.,1.],dict(p0=.2,p1=.6)),
    ([1,0],dict(p0=.8,p1=.2)),
    ([1,0],dict(p0=.2,p1=.5,alpha=True)),
    ([],dict(p0=.2,p1=.6)),
])
def test_sprt_rejects_invalid_design(x,kwargs):
    with pytest.raises(ValueError):bernoulli_sprt(x,**kwargs)


def _cuped():
    rng=np.random.default_rng(1)
    n=80
    t=np.r_[np.zeros(n,dtype=int),np.ones(n,dtype=int)]
    cov=rng.normal(size=2*n)
    y=10+.8*cov+1.2*t+rng.normal(0,.5,2*n)
    return y,t,cov


def test_cuped_matches_manual_ols_control_variate_welch():
    y,t,x=_cuped()
    r=cuped_randomized_effect(y,t,x)
    theta=np.cov(y,x,ddof=1)[0,1]/np.var(x,ddof=1)
    z=y-theta*(x-x.mean())
    expected=stats.ttest_ind(z[t==1],z[t==0],equal_var=False)
    assert r["theta"]==pytest.approx(theta)
    assert r["adjusted_difference"]==pytest.approx(z[t==1].mean()-z[t==0].mean())
    assert r["statistic"]==pytest.approx(expected.statistic)
    assert r["p_value"]==pytest.approx(expected.pvalue)
    assert r["table"]["n"].sum()==len(y)
    assert r["ci_lower"]<r["ci_upper"]


def test_cuped_rejects_observational_or_invalid_coding():
    y,t,x=_cuped()
    with pytest.raises(ValueError):cuped_randomized_effect(y,t.astype(float),x)
    with pytest.raises(ValueError):cuped_randomized_effect(y,t,np.ones(len(y)))
    with pytest.raises(ValueError):cuped_randomized_effect(y,t,x,alpha=True)


def test_shard_holm_and_ks_match_scipy_reference():
    rng=np.random.default_rng(13)
    shards=[rng.normal(size=55),rng.normal(size=57),rng.normal(1.5,1,size=60)]
    r=shard_distribution_consistency(shards)
    raw=[]
    for i,a in enumerate(shards):
        complement=np.concatenate(shards[:i]+shards[i+1:])
        raw.append(stats.ks_2samp(a,complement).pvalue)
    from statsmodels.stats.multitest import multipletests
    np.testing.assert_allclose(r["table"]["raw_p"],raw)
    np.testing.assert_allclose(r["table"]["holm_p"],multipletests(raw,method="holm")[1])
    assert r["n_flagged"]>=1


def test_practical_effect_separates_p_and_threshold():
    r=practical_proportion_difference(5100,10000,5000,10000,
                                       minimum_relevant_difference=.05)
    p_pool=10100/20000
    z=(.51-.50)/np.sqrt(p_pool*(1-p_pool)*(2/10000))
    assert r["statistic"]==pytest.approx(z)
    assert r["p_value"]==pytest.approx(2*stats.norm.sf(abs(z)))
    assert r["practically_relevant_point_estimate"] is False
    assert "p_value" in r


def _ols_data():
    rng=np.random.default_rng(2)
    n=90
    x=rng.normal(size=n)
    z=rng.normal(size=n)
    y=1+.6*x+.9*z+rng.normal(size=n)
    red=np.column_stack([np.ones(n),x])
    full=np.column_stack([np.ones(n),x,z])
    return y,red,full


def test_three_nested_ols_tests_match_independent_likelihood_and_f():
    y,x0,x1=_ols_data()
    r=nested_ols_wald_lr_score(y,x0,x1)
    sr=np.sum((y-x0@np.linalg.lstsq(x0,y,rcond=None)[0])**2)
    sf=np.sum((y-x1@np.linalg.lstsq(x1,y,rcond=None)[0])**2)
    n,k=x1.shape
    assert r["wald_chisquare"]==pytest.approx((sr-sf)/(sf/(n-k)))
    assert r["likelihood_ratio"]==pytest.approx(n*np.log(sr/sf))
    assert r["score_lm"]==pytest.approx(n*(sr-sf)/sr)
    assert r["exact_f_p_value"]==pytest.approx(stats.f.sf(r["exact_nested_f"],1,n-k))
    assert r["table"]["p_value"].between(0,1).all()


def test_nested_ols_rejects_unrelated_and_rank_deficient_models():
    y,x0,x1=_ols_data()
    with pytest.raises(ValueError):
        nested_ols_wald_lr_score(y,x0,np.column_stack((np.ones(len(y)),x1[:,1],x1[:,1])))
    with pytest.raises(ValueError):
        nested_ols_wald_lr_score(y,x0,np.column_stack((np.ones(len(y)),x1[:,2],x1[:,2]**2)))


def test_pseudo_r_squared_matches_standard_loglikelihood_algebra():
    r=regression_pseudo_r_squared(-85,-100,150)
    assert r["mcfadden_r2"]==pytest.approx(.15)
    assert r["cox_snell_r2"]==pytest.approx(1-np.exp(-30/150))
    assert r["nagelkerke_r2"]==pytest.approx((1-np.exp(-30/150))/(1-np.exp(-200/150)))
    assert "p_value" not in r
    with pytest.raises(ValueError):regression_pseudo_r_squared(-110,-100,150)


def test_cohen_f_squared_and_power_inversion():
    r=cohen_f_squared_nested_models(.65,.50)
    assert r["f_squared"]==pytest.approx(.15/.35)
    with pytest.raises(ValueError):cohen_f_squared_nested_models(.2,.5)
    p=two_sample_power_sensitivity(50,target_power=.8)
    from statsmodels.stats.power import TTestIndPower
    actual=TTestIndPower().power(effect_size=p["minimum_detectable_cohen_d"],
                                 nobs1=50,alpha=.05,ratio=1)
    assert actual==pytest.approx(.8,rel=1e-5)
    assert "p_value" not in p


def test_loo_refits_each_case_with_sklearn_exact_reference():
    rng=np.random.default_rng(8)
    X=rng.normal(size=(30,2))
    y=np.array([0,1]*15)
    model=DummyClassifier(strategy="most_frequent")
    r=leave_one_out_cv_evaluation(model,X,y)
    scores=cross_val_score(model,X,y,cv=LeaveOneOut(),scoring="accuracy")
    np.testing.assert_allclose(r["table"]["heldout_score"],scores)
    assert r["mean_cv_score"]==pytest.approx(scores.mean())
    with pytest.raises(ValueError):
        leave_one_out_cv_evaluation(model,X,y,scoring="r2")


def test_hopkins_matches_direct_neighbor_reference_and_is_deterministic():
    rng=np.random.default_rng(5)
    X=rng.uniform(size=(130,3))
    r=hopkins_cluster_tendency(X,sample_size=35,random_state=11)
    r2=hopkins_cluster_tendency(X,sample_size=35,random_state=11)
    assert r["hopkins"]==r2["hopkins"]
    assert 0<r["hopkins"]<1
    from sklearn.neighbors import NearestNeighbors
    seed=np.random.default_rng(11)
    ids=seed.choice(len(X),size=35,replace=False)
    u=seed.uniform(X.min(axis=0),X.max(axis=0),size=(35,3))
    model=NearestNeighbors(n_neighbors=2).fit(X)
    ua=model.kneighbors(u,n_neighbors=1)[0][:,0]
    wa=model.kneighbors(X[ids],n_neighbors=2)[0][:,1]
    assert r["hopkins"]==pytest.approx(ua.sum()/(ua.sum()+wa.sum()))


def test_noise_robustness_fixed_fit_and_reproducibility():
    rng=np.random.default_rng(5)
    X=pd.DataFrame(rng.normal(size=(90,2)),columns=["x","z"])
    y=2*X["x"]+X["z"]+.1*rng.normal(size=90)
    model=LinearRegression().fit(X.iloc[:60],y.iloc[:60])
    coef=model.coef_.copy()
    kwargs=dict(scoring="r2",noise_sd=.2,n_repeats=6,random_state=7)
    a=holdout_noise_robustness(model,X.iloc[60:],y.iloc[60:],**kwargs)
    b=holdout_noise_robustness(model,X.iloc[60:],y.iloc[60:],**kwargs)
    pd.testing.assert_frame_equal(a["table"],b["table"])
    np.testing.assert_allclose(model.coef_,coef)
    assert a["table"]["score_change"].mean()==pytest.approx(a["mean_score_change"])


def test_population_sample_numeric_and_categorical_references():
    rng=np.random.default_rng(9)
    population=pd.DataFrame({"metric":rng.normal(size=120),"category":["A"]*60+["B"]*60})
    sample=pd.DataFrame({"metric":rng.normal(1,size=100),"category":["A"]*55+["B"]*45})
    r=sample_population_comparison(population,sample,numeric=["metric"],
                                   categorical=["category"])
    table=r["table"].set_index("column")
    assert table.loc["metric","statistic"]==pytest.approx(
        stats.ks_2samp(population.metric,sample.metric).statistic)
    assert table.loc["category","statistic"]==pytest.approx(
        stats.chisquare([55,45],f_exp=[50,50]).statistic)
    assert table["p_value_holm"].between(0,1).all()


def test_early_late_screen_not_nonresponse_inference():
    rng=np.random.default_rng(5)
    a=rng.normal(size=30);b=rng.normal(.25,size=35)
    r=early_late_nonresponse_screen(a,b)
    ref=stats.ttest_ind(a,b,equal_var=False)
    assert r["statistic"]==pytest.approx(ref.statistic)
    assert r["p_value"]==pytest.approx(ref.pvalue)
    assert "not nonresponse" in r["interpretation_en"]


def test_scientific_alerts_no_false_claims():
    from khaled_dsml import repeated_measures_one_factor
    rng=np.random.default_rng(33)
    wide=rng.normal(size=(15,3))
    result=repeated_measures_one_factor(wide)
    screen=scientific_report_alerts(result,min_n=30)
    assert "small_sample" in set(screen["table"]["flag"])
    assert "uncorrected_sphericity" in set(screen["table"]["flag"])
    assert screen["passed_metadata_screen"] is False
    clean=scientific_report_alerts({
        "n_samples":50,"interpretation_ar":"نتيجة وصفية",
        "interpretation_en":"Descriptive result"})
    assert clean["n_alerts"]==0
    assert "cannot verify" in clean["interpretation_en"]


def test_hll_distinct_estimator_and_duplicates():
    empty=hyperloglog_cardinality([])
    assert empty["estimate_distinct"]==0
    r=hyperloglog_cardinality(list(range(1000))*5,precision=12)
    assert r["n_items_seen"]==5000
    assert abs(r["estimate_distinct"]-1000)/1000<.10
    with pytest.raises(ValueError):
        hyperloglog_cardinality(["hi",object()])


def test_minhash_lsh_identical_sets_and_jaccard_reference():
    docs=[["a","b","c","d"],["d","c","b","a"],["x","y","z"],["a","b","c"]]
    r=minhash_lsh_candidates(docs,num_perm=64,bands=8)
    rows=r["table"]
    match=rows.loc[(rows.left_document==0)&(rows.right_document==1)]
    assert len(match)==1
    assert match["minhash_jaccard_estimate"].iloc[0]==pytest.approx(1.0)
    assert r["n_candidate_pairs"]>=1
    assert r["n_documents"]==4
    with pytest.raises(ValueError):minhash_lsh_candidates([["a"],[]])


def test_results_render_arabic_rtl_and_english_ltr():
    y,t,x=_cuped()
    a,b,c=_ols_data()
    objects=[
        bernoulli_sprt([1]*10,p0=.3,p1=.7),
        cuped_randomized_effect(y,t,x),
        nested_ols_wald_lr_score(a,b,c),
        cohen_f_squared_nested_models(.3,.1),
        hopkins_cluster_tendency(np.random.default_rng(7).normal(size=(40,2))),
        hyperloglog_cardinality(list(range(90))),
        minhash_lsh_candidates([["a","b"],["a","b"]]),
        scientific_report_alerts({"n_samples":6,"interpretation_ar":"عربي",
                                  "interpretation_en":"English"}),
    ]
    for item in objects:
        html=render_analysis_dashboard(item,theme="ocean")
        assert 'dir="rtl"' in html
        assert item["interpretation_ar"] in html
        assert item["interpretation_en"] in html
