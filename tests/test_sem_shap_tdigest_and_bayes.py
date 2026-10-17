"""Independent numerical reference, exact combinatorial and input tests."""
from decimal import Decimal, ROUND_HALF_UP
from html import escape
from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import gammaln
from sklearn.datasets import make_blobs

from khaled_dsml import (
    sem_fit_indices, measurement_invariance_nested_ml,
    shap_explanation_stability, tdigest_quantile_summary,
    dirichlet_multinomial_bayes_factor,
    sprite_integer_constraint_witness, render_analysis_dashboard,
)


def test_sem_fit_independent_chi_square_formulas_and_covariance_srmr():
    observed=np.array([[2.,.5],[.5,1.5]])
    implied=np.array([[1.8,.4],[.4,1.6]])
    r=sem_fit_indices(60,40,250,60,180,
                      observed_covariance=observed,
                      implied_covariance=implied)
    # Standard ML conventional (unscaled) indices.
    assert r["cfi"]==pytest.approx(1-(60-40)/max(60-40,250-60,0))
    assert r["tli"]==pytest.approx(((250/60)-(60/40))/((250/60)-1))
    assert r["rmsea"]==pytest.approx(np.sqrt((60-40)/(40*(180-1))))
    e=(observed-implied)/np.sqrt(np.outer(np.diag(observed),np.diag(observed)))
    expected=np.sqrt(np.mean(e[np.triu_indices(2)]**2))
    assert r["srmr"]==pytest.approx(expected)
    assert "p_value" not in r


def test_sem_fit_boundary_noncentrality_and_input_rejections():
    low=sem_fit_indices(20,30,180,50,240)
    assert low["rmsea"]==0
    assert low["cfi"]==1
    assert low["srmr"] is None
    for kw in (
        {"degrees_of_freedom":0},
        {"baseline_chi_square":20},
        {"n_samples":5},
        {"observed_covariance":np.eye(2)},
        {"observed_covariance":np.eye(2),
         "implied_covariance":np.array([[1.,4.],[4.,1.]])},
    ):
        d=dict(chi_square=60,degrees_of_freedom=40,
               baseline_chi_square=250,baseline_degrees_of_freedom=60,
               n_samples=180)
        d.update(kw)
        with pytest.raises(ValueError):
            sem_fit_indices(**d)


def test_nested_invariance_matches_reference_chi_square_survival():
    r=measurement_invariance_nested_ml(110,90,119,94,level="metric")
    assert r["statistic"]==9
    assert r["degrees_of_freedom_difference"]==4
    assert r["p_value"]==pytest.approx(stats.chi2.sf(9,4))
    assert r["table"]["df"].tolist()==[90,94]
    assert "WLSMV" in r["interpretation_en"]
    with pytest.raises(ValueError):
        measurement_invariance_nested_ml(110,90,109,94)
    with pytest.raises(ValueError):
        measurement_invariance_nested_ml(110,90,119,90)


def test_shap_stability_independent_spearman_pairs():
    rng=np.random.default_rng(21)
    importance_base=np.array([.15,.35,.75,1.2,2.2])
    arrays=np.stack([
        rng.normal(0,1,size=(30,5))*importance_base
        for _ in range(4)
    ])
    r=shap_explanation_stability(arrays,
        feature_names=["a","b","c","d","e"])
    meanabs=np.abs(arrays).mean(axis=1)
    expected=[stats.spearmanr(meanabs[a],meanabs[b]).statistic
              for a in range(4) for b in range(a+1,4)]
    assert r["mean_pairwise_spearman"]==pytest.approx(np.mean(expected))
    assert r["min_pairwise_spearman"]==pytest.approx(min(expected))
    assert r["n_pairwise_comparisons"]==6
    assert r["table"].feature.tolist()==["a","b","c","d","e"]
    assert r["table"].mean_absolute_shap.to_numpy()==pytest.approx(meanabs.mean(axis=0))


def test_shap_stability_rejects_unaligned_missing_flat():
    good=np.tile(np.arange(1,6,dtype=float),(3,20,1))
    assert shap_explanation_stability(good)["mean_pairwise_spearman"]==pytest.approx(1)
    with pytest.raises(ValueError):
        shap_explanation_stability(good[:2])
    with pytest.raises(ValueError):
        shap_explanation_stability(np.ones_like(good))
    with pytest.raises(ValueError):
        shap_explanation_stability(np.where(good==1,np.nan,good))
    with pytest.raises(ValueError):
        shap_explanation_stability(good,feature_names=["a"]*5)


def test_tdigest_matches_independent_direct_backend_and_approx_exact():
    from tdigest import TDigest
    rng=np.random.default_rng(44)
    values=rng.normal(size=800)
    probabilities=(.01,.25,.5,.75,.99)
    result=tdigest_quantile_summary(values,probabilities=probabilities)
    reference=TDigest(delta=.01)
    for value in values:reference.update(float(value),1)
    observed=result["table"].estimated_value.to_numpy()
    np.testing.assert_allclose(observed,[reference.percentile(p*100) for p in probabilities])
    np.testing.assert_allclose(observed,np.percentile(values,np.array(probabilities)*100),atol=.20)
    assert result["n_samples"]==len(values)
    assert result["n_centroids"]>0


def test_tdigest_weighted_and_guardrails():
    from tdigest import TDigest
    values=[1.,2.,3.,4.,5.,6.,7.,8.,9.,10.]
    weights=[1,2,1,2,1,2,1,2,1,2]
    result=tdigest_quantile_summary(values,weights=weights,
                                    probabilities=(.2,.5,.8))
    ref=TDigest(delta=.01)
    for v,w in zip(values,weights):
        ref.update(v,w)
    np.testing.assert_allclose(result["table"].estimated_value,
                               [ref.percentile(100*p) for p in (.2,.5,.8)])
    assert result["n_samples"]==sum(weights)
    for kw in (
        {"compression":True},{"weights":[-1]*10},
        {"probabilities":(.8,.2)},{"weights":[1.2]*10},
    ):
        with pytest.raises(ValueError):
            tdigest_quantile_summary(values,**kw)


def test_dirichlet_bf_matches_scipy_direct_predictive_ratio():
    from scipy.stats import dirichlet_multinomial, multinomial
    count=np.array([3,1,2,0])
    null=np.array([.1,.2,.35,.35])
    alpha=np.array([2.,1.,1.,3.])
    r=dirichlet_multinomial_bayes_factor(count,null,prior_concentration=alpha)
    alternate=dirichlet_multinomial.pmf(count,alpha,n=int(count.sum()))
    reference=multinomial.pmf(count,n=int(count.sum()),p=null)
    assert r["bayes_factor_10"]==pytest.approx(alternate/reference)
    assert r["log_bayes_factor_10"]==pytest.approx(np.log(alternate/reference))
    assert sum(r["posterior_mean_probabilities"])==pytest.approx(1)
    assert r["n_samples"]==6


@pytest.mark.parametrize("counts,null,alpha",[
    ([2,1],[.8,.1],None),([2,-1],[.5,.5],None),
    ([1,1],[1.,0.],None),([2,1],[.5,.5],[0.,2.]),
])
def test_dirichlet_bf_invalid_designs(counts,null,alpha):
    with pytest.raises(ValueError):
        dirichlet_multinomial_bayes_factor(
            counts,null,prior_concentration=alpha)


def _rounded_mean_sd(xs):
    n=len(xs)
    mean=(Decimal(sum(xs))/Decimal(n)).quantize(
        Decimal(".01"),rounding=ROUND_HALF_UP)
    sq=sum(v*v for v in xs)
    variance=(Decimal(sq)-Decimal(sum(xs)**2)/Decimal(n))/(n-1)
    s=variance.sqrt().quantize(Decimal(".01"),rounding=ROUND_HALF_UP)
    return str(mean),str(s)


def test_sprite_constructs_witness_matching_moments_and_fixed_counts():
    values=[1,1,2,3,3]
    mean,sd=_rounded_mean_sd(values)
    r=sprite_integer_constraint_witness(mean,sd,5,
        minimum_score=1,maximum_score=3,fixed_counts={1:2})
    assert r["one_feasible_reconstruction_found"] is True
    witness=r["synthetic_scores"]
    assert len(witness)==5
    assert witness.count(1)>=2
    assert _rounded_mean_sd(witness)==(mean,sd)
    assert sum(r["table"].frequency)==5
    assert "original" in r["interpretation_en"]


def test_sprite_matches_bruteforce_feasibility_grid():
    all_vectors=list(product(range(1,4),repeat=4))
    realizable={_rounded_mean_sd(v) for v in all_vectors}
    for mean,sd in sorted(realizable)[:16]:
        r=sprite_integer_constraint_witness(mean,sd,4,
                                             minimum_score=1,maximum_score=3)
        assert r["one_feasible_reconstruction_found"] is True
        assert _rounded_mean_sd(r["synthetic_scores"])==(mean,sd)
    impossible=sprite_integer_constraint_witness("2.00","0.01",4,
                                                  minimum_score=1,maximum_score=3)
    assert impossible["one_feasible_reconstruction_found"] is False
    assert not impossible["synthetic_scores"]
    with pytest.raises(ValueError):
        sprite_integer_constraint_witness("2.00","0.01",4,
            minimum_score=1,maximum_score=3,fixed_counts={2:5})


def test_all_research_reports_bilingual_offline():
    rng=np.random.default_rng(8)
    runs=np.stack([rng.normal(size=(15,4))*[.1,.4,.8,1.2]
                   for _ in range(3)])
    result=[
        sem_fit_indices(60,40,250,60,170),
        measurement_invariance_nested_ml(100,80,109,85),
        shap_explanation_stability(runs),
        tdigest_quantile_summary(rng.normal(size=80)),
        dirichlet_multinomial_bayes_factor([3,2,1],[.3,.3,.4]),
        sprite_integer_constraint_witness("2.00","1.41",2,
                                          minimum_score=1,maximum_score=3),
    ]
    for item in result:
        html=render_analysis_dashboard(item,theme="amethyst")
        assert 'dir="rtl"' in html
        assert escape(item["interpretation_ar"]) in html
        assert escape(item["interpretation_en"]) in html
