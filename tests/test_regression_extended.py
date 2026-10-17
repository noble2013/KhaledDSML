"""Regression inference and model-design reference checks (statsmodels optional)."""
import numpy as np
import pandas as pd
import pytest

sm=pytest.importorskip("statsmodels.api")

from khaled_dsml import (
    regression_white_test, regression_goldfeld_quandt_test,
    regression_reset_test, regression_rainbow_test, regression_influence_table,
    chow_structural_break_test, regression_cusum_test, model_information_criteria,
    logistic_hosmer_lemeshow, glm_overdispersion_diagnostic, two_way_anova,
)


@pytest.fixture
def sample_model():
    rng=np.random.default_rng(33)
    n=180
    x=np.linspace(-2,2,n)
    y=2+3*x+rng.normal(scale=.5,size=n)
    X=sm.add_constant(x)
    return y,X,sm.OLS(y,X).fit()


def test_white_matches_statsmodels(sample_model):
    from statsmodels.stats.diagnostic import het_white
    _,X,model=sample_model
    r=regression_white_test(model.resid,X)
    stat,p,_,_=het_white(model.resid,X)
    assert r.statistic==pytest.approx(stat)
    assert r.p_value==pytest.approx(p)


def test_goldfeld_quandt_reference(sample_model):
    from statsmodels.stats.diagnostic import het_goldfeldquandt
    y,X,_=sample_model
    got=regression_goldfeld_quandt_test(y,X)
    stat,p,_=het_goldfeldquandt(y,X,split=.5,alternative="two-sided")
    assert got.statistic==pytest.approx(stat)
    assert got.p_value==pytest.approx(p)


def test_reset_and_rainbow(sample_model):
    _,_,model=sample_model
    a=regression_reset_test(model)
    b=regression_rainbow_test(model)
    assert 0<=a.p_value<=1
    assert 0<=b.p_value<=1
    assert "does not identify" in a.notes_en.lower()


def test_influence_metrics_match_reference(sample_model):
    _,_,model=sample_model
    influence=model.get_influence()
    result=regression_influence_table(model)
    assert len(result)==int(model.nobs)
    assert result["leverage"].to_numpy()==pytest.approx(influence.hat_matrix_diag)
    assert result["cooks_d"].to_numpy()==pytest.approx(influence.cooks_distance[0])
    assert "automatic deletion" in result["interpretation_en"].iloc[0]


def test_chow_structural_change_detected_with_prespecified_index():
    rng=np.random.default_rng(4)
    x=np.linspace(0,1,120)
    X=sm.add_constant(x)
    y=2+x*3+rng.normal(scale=.03,size=120)
    y[60:]+=2
    result=chow_structural_break_test(y,X,break_index=60)
    assert result.p_value<.001
    with pytest.raises(ValueError):
        chow_structural_break_test(y,X,break_index=1)


def test_cusum_and_information_criteria(sample_model):
    y,X,m=sample_model
    r=regression_cusum_test(m)
    assert 0<=r.p_value<=1
    second=sm.OLS(y,np.column_stack([X,X[:,1]**2])).fit()
    table=model_information_criteria({"linear":m,"quadratic":second})
    assert set(["aic","bic","n_samples"])<=set(table.columns)
    assert table.shape[0]==2


def test_hosmer_lemeshow_calibration():
    rng=np.random.default_rng(8)
    p=np.linspace(.1,.9,150)
    y=rng.binomial(1,p)
    res=logistic_hosmer_lemeshow(y,p,groups=8)
    assert 0<=res.p_value<=1
    with pytest.raises(ValueError):
        logistic_hosmer_lemeshow([0,1]*15,[.5]*30,groups=20)


def test_glm_dispersion_and_two_way_anova():
    rng=np.random.default_rng(7)
    n=100
    x=rng.normal(size=n)
    counts=rng.poisson(np.exp(.2+.3*x))
    fit=sm.GLM(counts,sm.add_constant(x),family=sm.families.Poisson()).fit()
    r=glm_overdispersion_diagnostic(fit)
    assert r["pearson_dispersion"]==pytest.approx(fit.pearson_chi2/fit.df_resid)
    group_a=np.tile(["control","treatment"],50)
    group_b=np.repeat(["first","second"],50)
    outcome=(np.array(group_a)=="treatment")*.5 + (np.array(group_b)=="second")*.8 + rng.normal(size=n)
    df=pd.DataFrame({"score":outcome,"a":group_a,"b":group_b})
    table=two_way_anova(df,outcome="score",factor_a="a",factor_b="b")
    assert "PR(>F)" in table.columns
    assert table["PR(>F)"].notna().any()
