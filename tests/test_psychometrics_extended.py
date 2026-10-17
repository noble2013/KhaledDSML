"""Reference and safety tests for psychometric indicators."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score
from khaled_dsml import (
    item_total_statistics, split_half_reliability, test_retest_correlation,
    cohen_kappa_agreement, fleiss_kappa_agreement, factor_scree_table,
    parallel_analysis, exploratory_factor_analysis, convergent_validity,
    htmt_ratio, fornell_larcker_matrix, harman_single_factor,
)


@pytest.fixture
def items():
    rng=np.random.default_rng(42)
    latent=rng.normal(size=120)
    return pd.DataFrame({
        f"q{j}":latent+rng.normal(scale=.3,size=len(latent))
        for j in range(1,7)
    })


def test_corrected_item_total_alpha_if_deleted(items):
    r=item_total_statistics(items)
    assert set(["corrected_item_total_r","alpha_if_deleted"])<=set(r.columns)
    assert len(r)==6 and (r["corrected_item_total_r"]>.5).all()
    assert (r["alpha_if_deleted"]<1).all()


def test_split_half_and_test_retest(items):
    result=split_half_reliability(items)
    assert .7 < result["spearman_brown"] <=1
    other=split_half_reliability(items,method="first_second")
    assert other["half_correlation"]>.7
    score=items.sum(axis=1)
    r=test_retest_correlation(score,score+.1*np.arange(len(items)))
    assert -1<=r["correlation"]<=1 and 0<=r["p_value"]<=1


def test_kappa_reference_and_invalid():
    a=[0,0,1,1,0,1]
    b=[0,1,1,1,0,1]
    k=cohen_kappa_agreement(a,b)
    assert k["kappa"]==pytest.approx(cohen_kappa_score(a,b))
    with pytest.raises(ValueError):
        cohen_kappa_agreement([0,1],[1])


def test_fleiss_kappa_reference():
    pytest.importorskip("statsmodels")
    from statsmodels.stats.inter_rater import fleiss_kappa
    votes=np.array([[3,0],[2,1],[1,2],[0,3],[2,1]],dtype=int)
    output=fleiss_kappa_agreement(votes)
    assert output["kappa"]==pytest.approx(fleiss_kappa(votes,method="fleiss"))


def test_factor_scree_and_parallel_reproducible(items):
    s=factor_scree_table(items)
    assert s["eigenvalue"].sum()==pytest.approx(6)
    assert s["cumulative_explained_variance"].iloc[-1]==pytest.approx(1)
    p=parallel_analysis(items,n_iter=30,seed=9)
    q=parallel_analysis(items,n_iter=30,seed=9)
    pd.testing.assert_frame_equal(p,q)
    assert p["retain"].iloc[0]


def test_efa_components_and_bilingual_caution(items):
    r=exploratory_factor_analysis(items,n_factors=1)
    assert r["loadings"].shape==(6,1)
    assert r["uniqueness"].shape[0]==6
    assert "confirmatory" in r["interpretation_en"]


def test_standardized_ave_cr():
    r=convergent_validity([.8,.7,.9])
    expected_ave=np.mean(np.array([.8,.7,.9])**2)
    assert r["ave"]==pytest.approx(expected_ave)
    assert .7<r["composite_reliability"]<=1
    with pytest.raises(ValueError):
        convergent_validity([1.2,.5])


def test_htmt_constructs(items):
    v=htmt_ratio(items,[["q1","q2","q3"],["q4","q5","q6"]])
    assert v["htmt"]>0
    with pytest.raises(ValueError):
        htmt_ratio(items,[["q1","q2"],["q2","q3"]])


def test_fornell_larcker_and_harman(items):
    corr=pd.DataFrame([[1,.3],[.3,1]],index=["A","B"],columns=["A","B"])
    out=fornell_larcker_matrix(corr,{"A":.64,"B":.81})
    assert out.loc["A","A"]==pytest.approx(.8)
    assert out.loc["A","B"]==pytest.approx(.3)
    harman=harman_single_factor(items)
    assert harman["first_component_variance_ratio"]>.5
    assert "not" in harman["interpretation_en"]


def test_missing_and_constant_items_rejected(items):
    data=items.copy()
    data.loc[0,"q1"]=np.nan
    with pytest.raises(ValueError):
        item_total_statistics(data)
    data=items.copy()
    data["q1"]=5
    with pytest.raises(ValueError):
        exploratory_factor_analysis(data)
