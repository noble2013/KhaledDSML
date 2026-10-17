"""Independent statsmodels references for the mediation/moderation suite.

Simulated cases are not causal demonstrations. Checks use analytical OLS
identities and covariance calculations, plus strict scientific guardrails.
"""
import numpy as np
import pytest
from scipy import stats
sm=pytest.importorskip("statsmodels.api")

from khaled_dsml import (
    mediation_paths, sobel_mediation_test, bootstrap_mediation_interval,
    moderation_interaction_test, simple_slopes_table, johnson_neyman_region,
    render_analysis_dashboard,
)


@pytest.fixture
def mediation_data():
    rng=np.random.default_rng(502)
    n=160
    x=rng.normal(size=n)
    c=rng.normal(size=n)
    m=.7*x+.25*c+rng.normal(size=n)
    y=.4*x+1.1*m+.2*c+rng.normal(size=n)
    return x,m,y,c


@pytest.fixture
def moderation_data():
    rng=np.random.default_rng(44)
    n=200
    x=rng.normal(size=n)
    w=rng.uniform(-2,2,n)
    y=1+.5*x+1.1*w+.9*x*w+rng.normal(scale=.4,size=n)
    return x,w,y


def test_mediation_paths_match_statsmodels_and_exact_decomposition(mediation_data):
    x,m,y,c=mediation_data
    r=mediation_paths(x,m,y,controls=c)
    mediator=sm.OLS(m,sm.add_constant(np.column_stack([x,c]))).fit()
    outcome=sm.OLS(y,sm.add_constant(np.column_stack([x,m,c]))).fit()
    total=sm.OLS(y,sm.add_constant(np.column_stack([x,c]))).fit()
    assert r["a_path"]==pytest.approx(mediator.params[1])
    assert r["a_se"]==pytest.approx(mediator.bse[1])
    assert r["b_path"]==pytest.approx(outcome.params[2])
    assert r["b_se"]==pytest.approx(outcome.bse[2])
    assert r["total_effect"]==pytest.approx(total.params[1])
    assert r["direct_effect"]==pytest.approx(outcome.params[1])
    assert r["decomposition_residual"]==pytest.approx(0,abs=1e-11)
    assert "not evidence" in r["interpretation_en"]


def test_sobel_delta_method_reference(mediation_data):
    x,m,y,c=mediation_data
    r=sobel_mediation_test(x,m,y,controls=c)
    a,b,sa,sb=(r[k] for k in ("a_path","b_path","a_se","b_se"))
    z=a*b/np.sqrt(b*b*sa*sa+a*a*sb*sb)
    assert r["statistic"]==pytest.approx(z)
    assert r["p_value"]==pytest.approx(2*stats.norm.sf(abs(z)))
    assert r["p_value"]<.01
    assert r["reject_null"]


def test_case_bootstrap_reproduces_and_has_no_fake_p_value(mediation_data):
    x,m,y,c=mediation_data
    r=bootstrap_mediation_interval(x,m,y,controls=c,repetitions=250,seed=17)
    again=bootstrap_mediation_interval(x,m,y,controls=c,repetitions=250,seed=17)
    assert r==again
    assert "p_value" not in r
    assert r["ci_lower"] < r["indirect_effect"] < r["ci_upper"]
    assert r["ci_lower"]>0
    assert r["replications"]==250


def test_moderation_interaction_matches_statsmodels_nested_f_and_hc3(moderation_data):
    x,w,y=moderation_data
    r=moderation_interaction_test(x,w,y)
    reduced=sm.OLS(y,sm.add_constant(np.column_stack([x,w]))).fit()
    full=sm.OLS(y,sm.add_constant(np.column_stack([x,w,x*w]))).fit()
    robust=full.get_robustcov_results(cov_type="HC3")
    F,p,_=full.compare_f_test(reduced)
    assert r["interaction_coefficient"]==pytest.approx(full.params[3])
    assert r["p_value"]==pytest.approx(robust.pvalues[3])
    assert r["statistic"]==pytest.approx(robust.tvalues[3])
    assert r["delta_r2"]==pytest.approx(full.rsquared-reduced.rsquared)
    assert r["nested_f"]==pytest.approx(F)
    assert r["nested_f_p_value"]==pytest.approx(p)
    assert r["interaction_coefficient"]>.5


def test_simple_slopes_covariance_matches_manual_hc3(moderation_data):
    x,w,y=moderation_data
    r=simple_slopes_table(x,w,y,moderator_values=[-1.,0.,1.])
    model=sm.OLS(y,sm.add_constant(np.column_stack([x,w,x*w]))).fit()
    cov=model.get_robustcov_results(cov_type="HC3").cov_params()
    for i,value in enumerate([-1.,0.,1.]):
        row=r["table"].iloc[i]
        expected=model.params[1]+model.params[3]*value
        expected_var=cov[1,1]+2*value*cov[1,3]+value*value*cov[3,3]
        assert row["slope"]==pytest.approx(expected)
        assert row["standard_error"]==pytest.approx(np.sqrt(expected_var))
        assert row["ci_lower"]<row["ci_upper"]
    assert r["table"]["p_value"].between(0,1).all()


def test_johnson_neyman_root_boundaries_against_conditional_t(moderation_data):
    x,w,y=moderation_data
    r=johnson_neyman_region(x,w,y)
    assert r["moderator_min"]==pytest.approx(w.min())
    assert r["moderator_max"]==pytest.approx(w.max())
    assert 0<=len(r["boundaries"])<=2
    assert len(r["table"])==len(r["boundaries"])+1
    model=sm.OLS(y,sm.add_constant(np.column_stack([x,w,x*w]))).fit()
    cov=model.get_robustcov_results(cov_type="HC3").cov_params()
    crit=stats.t.ppf(.975,model.df_resid)
    for z in r["boundaries"]:
        slope=model.params[1]+model.params[3]*z
        var=cov[1,1]+2*z*cov[1,3]+z*z*cov[3,3]
        assert abs(slope)/np.sqrt(var)==pytest.approx(crit,rel=1e-6)


def test_bilingual_dashboard_shows_real_conditional_results(moderation_data):
    x,w,y=moderation_data
    r=simple_slopes_table(x,w,y)
    html=render_analysis_dashboard(r,theme="aurora")
    assert "جدول نتائج التحليل" in html
    assert "ANALYTICAL RESULTS TABLE" in html
    assert 'dir="rtl"' in html and 'dir="ltr"' in html
    assert "moderator_value" in html


def test_invalid_missing_variance_design_and_categorical_inputs(mediation_data):
    x,m,y,c=mediation_data
    with pytest.raises(ValueError):
        mediation_paths(x,m[:-1],y)
    with pytest.raises(ValueError):
        mediation_paths(x,np.ones_like(m),y)
    with pytest.raises(ValueError):
        mediation_paths(x,m,y,controls=np.column_stack([c,c]))
    with pytest.raises(ValueError):
        sobel_mediation_test(x,m,y,alpha=1)
    with pytest.raises(ValueError):
        bootstrap_mediation_interval(x,m,y,repetitions=12)
    with pytest.raises(ValueError):
        simple_slopes_table(x,m,y,moderator_values=[float("nan")])
    with pytest.raises(ValueError):
        johnson_neyman_region(x,m,y,robust="HC0")


def test_causal_caveats_present_in_results(mediation_data,moderation_data):
    x,m,y,_=mediation_data
    w,x2,y2=moderation_data
    outputs=[
        mediation_paths(x,m,y),
        sobel_mediation_test(x,m,y),
        bootstrap_mediation_interval(x,m,y,repetitions=100),
        moderation_interaction_test(w,x2,y2),
        simple_slopes_table(w,x2,y2),
        johnson_neyman_region(w,x2,y2),
    ]
    for result in outputs:
        assert result["interpretation_ar"].strip()
        assert result["interpretation_en"].strip()
    assert "causal" in outputs[0]["interpretation_en"]
    assert "p_value" not in outputs[2]
