"""Research reference and adversarial tests for missing-data methods."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from khaled_dsml import (
    little_mcar_test, missingness_association_screen,
    multiple_impute_train_test_numeric, rubin_pool_scalar,
    render_analysis_dashboard,
)


@pytest.fixture
def mcar_frame():
    rng=np.random.default_rng(206)
    cov=np.array([[1,.3,.15],[.3,1,-.2],[.15,-.2,1.]])
    x=rng.multivariate_normal([2,4,6],cov,size=220)
    mask=rng.random(x.shape)<.18
    mask[np.arange(20),:]=False
    x[mask]=np.nan
    return pd.DataFrame(x,columns=["الدخل","التعلم","الاستهلاك"])


def test_little_statistic_formula_and_distinct_patterns(mcar_frame):
    r=little_mcar_test(mcar_frame)
    assert r["test"].startswith("Little")
    assert r["n_samples"]==len(mcar_frame)
    assert r["n_patterns"]==len(r["table"])
    assert r["degrees_of_freedom"]==int(
        (r["table"]["n_observed_variables"]).sum()-mcar_frame.shape[1]
    )
    assert r["p_value"]==pytest.approx(
        stats.chi2.sf(r["statistic"],r["degrees_of_freedom"])
    )
    assert 0<=r["p_value"]<=1
    assert r["em_iterations"]>0
    assert "does not prove" in r["interpretation_en"]
    assert "جدول نتائج التحليل" in render_analysis_dashboard(r,theme="amethyst")


def test_little_likely_non_mcar_shift():
    rng=np.random.default_rng(16)
    x=rng.normal(size=(320,3))
    # Response status of y depends on fully observed predictor x0.
    x[x[:,0]>0,1]=np.nan
    frame=pd.DataFrame(x,columns=list("abc"))
    r=little_mcar_test(frame)
    assert r["p_value"]<.01


def test_mcar_absence_invalid_inputs_and_singular_patterns():
    rng=np.random.default_rng(7)
    complete=pd.DataFrame(rng.normal(size=(80,3)))
    with pytest.raises(ValueError,match="some but not complete"):
        little_mcar_test(complete)
    wrong=complete.copy()
    wrong.iloc[0,0]=np.inf
    with pytest.raises(ValueError,match="Infinite"):
        little_mcar_test(wrong)
    with pytest.raises(ValueError):
        little_mcar_test(complete.iloc[:4].copy())
    bad=complete.copy()
    bad.columns=["a","a","b"]
    with pytest.raises(ValueError):
        little_mcar_test(bad)
    nothing=complete.copy()
    nothing.iloc[0,:]=np.nan
    with pytest.raises(ValueError,match="fully missing rows"):
        little_mcar_test(nothing)
    singular=complete.copy()
    singular.iloc[:,2]=singular.iloc[:,1]
    singular.iloc[0,0]=np.nan
    with pytest.raises(ValueError):
        little_mcar_test(singular)


def test_missingness_screen_holm_no_mcar_or_mar_claim(mcar_frame):
    r=missingness_association_screen(mcar_frame)
    t=r["table"]
    assert len(t)>0
    assert (t["holm_p_value"]>=t["raw_p_value"]).all()
    assert t["holm_p_value"].between(0,1).all()
    assert t["n_used"].le(len(mcar_frame)).all()
    assert "neither prove MAR" in r["interpretation_en"].lower() or (
        "neither prove MAR" in r["interpretation_en"]
    )
    html=render_analysis_dashboard(r)
    assert "ANALYTICAL RESULTS TABLE" in html
    assert "missing_target" in html


def test_missingness_screen_detects_observed_predictor_association():
    rng=np.random.default_rng(35)
    z=rng.normal(size=200)
    y=rng.normal(size=200)
    y[z>0]=np.nan
    x=pd.DataFrame({"z":z,"y":y})
    r=missingness_association_screen(x)
    row=r["table"]
    row=row[(row["missing_target"]=="y")&(row["observed_predictor"]=="z")].iloc[0]
    assert row["point_biserial_r"]>0
    assert row["holm_p_value"]<.001
    assert r["significant_comparisons"]>=1


def test_imputation_fits_train_only_preserves_observed_values():
    rng=np.random.default_rng(2)
    train=pd.DataFrame(rng.normal(size=(80,3)),columns=list("abc"))
    test=pd.DataFrame(rng.normal(size=(10,3)),columns=list("abc"))
    train.iloc[::5,1]=np.nan
    test.iloc[::2,2]=np.nan
    res=multiple_impute_train_test_numeric(
        train,test,n_imputations=3,max_iter=3,seed=42
    )
    assert len(res["imputations"])==3
    assert res["n_train"]==80 and res["n_test"]==10
    observed_train=np.isfinite(train.to_numpy())
    observed_test=np.isfinite(test.to_numpy())
    for completed_train,completed_test in res["imputations"]:
        assert completed_train.shape==train.shape
        assert completed_test.shape==test.shape
        assert np.isfinite(completed_train.to_numpy()).all()
        assert np.isfinite(completed_test.to_numpy()).all()
        np.testing.assert_allclose(
            completed_train.to_numpy()[observed_train],
            train.to_numpy()[observed_train],
        )
        np.testing.assert_allclose(
            completed_test.to_numpy()[observed_test],
            test.to_numpy()[observed_test],
        )
    assert not np.array_equal(
        res["imputations"][0][0].to_numpy(),
        res["imputations"][1][0].to_numpy(),
    )
    modified_test=test.copy()
    modified_test.loc[modified_test["a"].notna(),"a"]+=1000
    second=multiple_impute_train_test_numeric(
        train,modified_test,n_imputations=3,max_iter=3,seed=42
    )
    np.testing.assert_allclose(
        second["imputations"][0][0].to_numpy(),
        res["imputations"][0][0].to_numpy(),
    )


def test_imputation_one_row_holdout_with_all_missing_column():
    rng=np.random.default_rng(23)
    train=pd.DataFrame(rng.normal(size=(55,2)),columns=["a","b"])
    train.iloc[::6,1]=np.nan
    test=pd.DataFrame({"a":[2.],"b":[np.nan]})
    r=multiple_impute_train_test_numeric(
        train,test,n_imputations=2,max_iter=2,seed=4
    )
    assert np.isfinite(r["imputations"][0][1].to_numpy()).all()


def test_imputation_rejects_unsupported_inputs():
    rng=np.random.default_rng(25)
    tr=pd.DataFrame(rng.normal(size=(60,2)),columns=["a","b"])
    te=tr.iloc[:3].copy()
    with pytest.raises(ValueError,match="no missing"):
        multiple_impute_train_test_numeric(tr,te)
    tr.iloc[0,0]=np.nan
    with pytest.raises(ValueError):
        multiple_impute_train_test_numeric(tr,te[["b","a"]])
    with pytest.raises(ValueError):
        multiple_impute_train_test_numeric(tr,te,n_imputations=1)


def test_rubin_pool_matches_manual_t_reference():
    estimates=[.2,.5,.45,.3,.6]
    variances=[.015,.02,.03,.025,.02]
    result=rubin_pool_scalar(estimates,variances)
    within=np.mean(variances)
    between=np.var(estimates,ddof=1)
    total=within+(1+1/5)*between
    df=(5-1)*(1+within/((1+1/5)*between))**2
    critical=stats.t.ppf(.975,df)
    assert result["estimate"]==pytest.approx(np.mean(estimates))
    assert result["total_variance"]==pytest.approx(total)
    assert result["standard_error"]==pytest.approx(np.sqrt(total))
    assert result["degrees_of_freedom"]==pytest.approx(df)
    assert result["ci_lower"]==pytest.approx(
        np.mean(estimates)-critical*np.sqrt(total)
    )
    assert "p_value" not in result


def test_rubin_zero_between_uses_normal_and_degenerate_fails():
    r=rubin_pool_scalar([1.,1.,1.],[.04,.04,.04])
    assert np.isinf(r["degrees_of_freedom"])
    assert r["ci_upper"]==pytest.approx(1+stats.norm.ppf(.975)*.2)
    with pytest.raises(ValueError):
        rubin_pool_scalar([1.,1.],[0.,0.])
    with pytest.raises(ValueError):
        rubin_pool_scalar([1.,2.],[.2,-.3])
