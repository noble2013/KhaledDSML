"""Numerical reference tests of balanced classical GLM methods."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from khaled_dsml import (
    ancova_one_factor, manova_one_factor, repeated_measures_one_factor,
    render_analysis_dashboard,
)


def _ancova_data(n=30, seed=13):
    rng = np.random.default_rng(seed)
    group = ["A"] * n + ["B"] * n + ["C"] * n
    cov = rng.normal(3, 1, size=3*n)
    offset = np.repeat([-0.7, .4, 1.2], n)
    y = 5 + 1.8*cov + offset + rng.normal(scale=.5, size=3*n)
    return y, group, cov


def test_ancova_partial_f_matches_manual_nested_least_squares():
    y, group, x = _ancova_data()
    result = ancova_one_factor(y, group, x)
    n = len(y)
    red = np.column_stack((np.ones(n), x))
    full = np.column_stack((np.ones(n), x,
                            np.asarray(group) == "B", np.asarray(group) == "C"))
    pred_red = red @ np.linalg.lstsq(red, y, rcond=None)[0]
    pred_full = full @ np.linalg.lstsq(full, y, rcond=None)[0]
    ssr_red = np.sum((y-pred_red)**2)
    ssr_full = np.sum((y-pred_full)**2)
    f = ((ssr_red-ssr_full)/2) / (ssr_full/(n-4))
    assert result["statistic"] == pytest.approx(f, rel=1e-10)
    assert result["p_value"] == pytest.approx(stats.f.sf(f, 2, n-4), rel=1e-10)
    assert result["partial_eta_squared"] == pytest.approx((ssr_red-ssr_full)/ssr_red)
    assert result["numerator_df"] == 2
    assert result["denominator_df"] == n-4
    assert len(result["table"]) == 3
    assert result["table"]["n_group"].sum() == n
    assert result["reject_equal_adjusted_means"]


def test_ancova_adjusted_means_are_at_global_covariate_mean():
    y, group, x = _ancova_data()
    r = ancova_one_factor(y, group, x)
    import statsmodels.formula.api as smf
    model = smf.ols(
        "outcome ~ cov + C(group)",
        data=pd.DataFrame({"outcome": y, "group": group, "cov": x}),
    ).fit()
    m = np.mean(x)
    for row in r["table"].itertuples():
        predicted = model.predict(pd.DataFrame({"group": [row.group], "cov": [m]}))[0]
        assert row.adjusted_mean == pytest.approx(predicted)


def test_ancova_slope_homogeneity_detects_heterogeneous_group_slopes():
    rng = np.random.default_rng(23)
    x = np.tile(np.linspace(-2, 2, 40), 2)
    groups = ["A"]*40+["B"]*40
    y = 5 + np.where(np.array(groups)=="A", .4*x, 3.2*x)
    y = y + rng.normal(0, .2, len(x))
    r = ancova_one_factor(y, groups, x)
    assert r["slope_homogeneity_rejected"] is True
    assert r["slope_homogeneity_p_value"] < .01
    assert "common-slope" in r["interpretation_en"]


def test_ancova_rejects_perfect_fit_or_rank_deficiency():
    n=20
    groups=["A"]*n+["B"]*n
    x=np.r_[np.ones(n), np.ones(n)*2.]
    y=np.arange(2*n,dtype=float)
    with pytest.raises(ValueError):
        ancova_one_factor(y,groups,x)
    with pytest.raises(ValueError):
        ancova_one_factor([5.]*(2*n),groups,np.tile(np.arange(n),2))


@pytest.mark.parametrize("case",[
    "bad_groups", "short", "nan", "alpha", "misaligned",
])
def test_ancova_input_guardrails(case):
    y,g,x = _ancova_data()
    if case == "bad_groups": g[0] = None
    if case == "short": y=y[:9];g=g[:9];x=x[:9]
    if case == "nan": x[0] = np.nan
    if case == "misaligned": x=x[:-1]
    kwargs = {"alpha":True} if case == "alpha" else {}
    with pytest.raises(ValueError):
        ancova_one_factor(y,g,x,**kwargs)


def _manova_data():
    rng=np.random.default_rng(6)
    group=["A"]*32+["B"]*32+["C"]*32
    baseline=rng.multivariate_normal([1.,2.,-1.],
                                     [[1.,.3,0.],[.3,1.,.2],[0.,.2,1.]],size=96)
    offsets=np.repeat([[0,0,0],[.35,-.2,.25],[.9,.4,-.1]],32,axis=0)
    return baseline+offsets,group


def test_manova_wilks_lambda_matches_independent_e_h_determinants():
    Y,g=_manova_data()
    out=manova_one_factor(Y,g)
    grand=Y.mean(axis=0)
    E=np.zeros((3,3));H=np.zeros((3,3))
    for level in sorted(set(g)):
        sub=Y[np.asarray(g)==level]
        diff=sub-sub.mean(axis=0)
        E+=diff.T@diff
        d=sub.mean(axis=0)-grand
        H+=len(sub)*np.outer(d,d)
    wilks=np.linalg.det(E)/np.linalg.det(E+H)
    assert out["wilks_lambda"] == pytest.approx(wilks, rel=1e-10)
    assert out["p_value"] == pytest.approx(
        out["table"].loc[out["table"]["method"]=="Wilks' lambda","p_value"].iloc[0]
    )
    assert list(out["table"]["method"]) == [
        "Wilks' lambda", "Pillai's trace", "Hotelling-Lawley trace",
        "Roy's greatest root",
    ]
    assert out["n_groups"] == 3
    assert out["n_outcomes"] == 3


def test_manova_all_methods_match_statsmodels_reference():
    from statsmodels.multivariate.manova import MANOVA
    Y,g=_manova_data()
    res=manova_one_factor(Y,g)
    data=pd.DataFrame(Y,columns=["v0","v1","v2"])
    data["grp"]=g
    ref=MANOVA.from_formula("v0+v1+v2 ~ C(grp)",data=data).mv_test().results["C(grp)"]["stat"]
    for row in res["table"].itertuples():
        item=ref.loc[row.method]
        assert row.statistic == pytest.approx(float(item["Value"]))
        assert row.approx_f == pytest.approx(float(item["F Value"]))
        assert row.num_df == pytest.approx(float(item["Num DF"]))
        assert row.den_df == pytest.approx(float(item["Den DF"]))
        assert row.p_value == pytest.approx(float(item["Pr > F"]))


@pytest.mark.parametrize("kind",["one_outcome","duplicate","missing","small","misaligned","boolalpha"])
def test_manova_rejects_unsupported_inputs(kind):
    Y,g=_manova_data()
    alpha=.05
    if kind=="one_outcome": Y=Y[:,0:1]
    if kind=="duplicate": Y[:,2]=Y[:,0]
    if kind=="missing": Y[0,0]=np.nan
    if kind=="small": Y=Y[:10];g=g[:10]
    if kind=="misaligned":g=g[:-1]
    if kind=="boolalpha":alpha=True
    with pytest.raises(ValueError):
        manova_one_factor(Y,g,alpha=alpha)


def _repeated_data():
    rng=np.random.default_rng(33)
    n=25
    subject=rng.normal(size=(n,1))
    y=subject + rng.normal(0,.28,size=(n,3)) + np.array([0.,.4,1.1])
    return pd.DataFrame(y,columns=["baseline","middle","final"])


def test_repeated_anova_matches_independent_two_way_sums_of_squares():
    wide=_repeated_data()
    arr=wide.to_numpy()
    r=repeated_measures_one_factor(wide)
    n,k=arr.shape
    grand=arr.mean()
    ss_condition=n*np.sum((arr.mean(axis=0)-grand)**2)
    resid=(arr-arr.mean(axis=1,keepdims=True)
           -arr.mean(axis=0,keepdims=True)+grand)
    ss_error=np.sum(resid**2)
    f=(ss_condition/(k-1))/(ss_error/((n-1)*(k-1)))
    assert r["statistic"] == pytest.approx(f,rel=1e-10)
    assert r["p_value"] == pytest.approx(stats.f.sf(f,k-1,(n-1)*(k-1)))
    assert r["numerator_df"] == k-1
    assert r["denominator_df"] == (n-1)*(k-1)
    assert list(r["table"]["condition"]) == list(wide.columns)
    assert r["reject_equal_condition_means"]
    assert r["sphericity_correction"].startswith("none")


def test_repeated_matches_anovarm_reference():
    from statsmodels.stats.anova import AnovaRM
    wide=_repeated_data()
    long=wide.assign(subject=range(len(wide))).melt(
        id_vars="subject",var_name="occasion",value_name="y")
    ref=AnovaRM(long,"y","subject",within=["occasion"]).fit().anova_table.loc["occasion"]
    r=repeated_measures_one_factor(wide)
    assert r["statistic"] == pytest.approx(ref["F Value"],rel=1e-10)
    assert r["p_value"] == pytest.approx(ref["Pr > F"],rel=1e-10)


@pytest.mark.parametrize("case",["missing","one_condition","few_people","duplicate","constant","badalpha"])
def test_repeated_input_guardrails(case):
    wide=_repeated_data()
    kwargs={}
    if case=="missing": wide.iloc[2,1] = np.nan
    if case=="one_condition": wide=wide.iloc[:,0:1]
    if case=="few_people":wide=wide.iloc[:5]
    if case=="duplicate":wide.columns=["same","same","other"]
    if case=="constant":wide.loc[:,:]=3
    if case=="badalpha":kwargs["alpha"]=True
    with pytest.raises(ValueError):
        repeated_measures_one_factor(wide,**kwargs)


def test_glm_methods_render_arabic_rtl_and_english_ltr():
    y,g,x=_ancova_data()
    Y,G=_manova_data()
    for r in (
        ancova_one_factor(y,g,x),
        manova_one_factor(Y,G),
        repeated_measures_one_factor(_repeated_data()),
    ):
        html=render_analysis_dashboard(r,theme="amethyst")
        assert 'dir="rtl"' in html
        assert r["interpretation_ar"] in html
        assert r["interpretation_en"] in html
        assert "<table" in html
