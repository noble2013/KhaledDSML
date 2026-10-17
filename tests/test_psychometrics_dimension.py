"""Reference and limits for omega, agreement and dimension diagnostics."""
import numpy as np
import pytest

from khaled_dsml import (
    omega_single_factor, intraclass_correlation, krippendorff_nominal_alpha,
    trustworthiness_embedding, distance_correlation, bootstrap_cluster_stability,
)


def test_one_factor_omega_on_single_latent():
    rng=np.random.default_rng(19)
    latent=rng.normal(size=150)
    x=np.stack([latent+rng.normal(scale=.3,size=150) for _ in range(5)],axis=1)
    r=omega_single_factor(x)
    assert .8<r["omega_total_one_factor"]<=1
    assert r["n_items"]==5
    assert "not prove" in r["interpretation_en"]


def test_icc_perfect_agreement_and_consistency():
    scores=np.array([1,2,4,3,7,8,9,12],dtype=float)
    x=np.column_stack([scores,scores])
    r=intraclass_correlation(x,model="ICC2")
    s=intraclass_correlation(x,model="ICC3")
    assert r["icc"]==pytest.approx(1)
    assert s["icc"]==pytest.approx(1)
    x[:,1]+=2
    shifted=intraclass_correlation(x,model="ICC2")
    consistent=intraclass_correlation(x,model="ICC3")
    assert shifted["icc"]<consistent["icc"]
    assert consistent["icc"]==pytest.approx(1)


def test_krippendorff_alpha_perfect_and_missing_data():
    clean=[["yes","yes"],["no","no"],["yes","yes"],["no","no"]]
    r=krippendorff_nominal_alpha(clean)
    assert r["krippendorff_alpha"]==pytest.approx(1)
    with_missing=[["yes","yes",None],["no","no","no"],
                  ["yes","no",None],["no",None,"no"]]
    r=krippendorff_nominal_alpha(with_missing)
    assert np.isfinite(r["krippendorff_alpha"])
    assert r["n_pairable_units"]==4
    with pytest.raises(ValueError):
        krippendorff_nominal_alpha([[None,None],["yes",None],["no",None]])


def test_embedded_trustworthiness_reference():
    rng=np.random.default_rng(41)
    x=rng.normal(size=(50,3))
    from sklearn.manifold import trustworthiness
    r=trustworthiness_embedding(x,x,n_neighbors=5)
    assert r["trustworthiness"]==pytest.approx(1)
    embedded=x[:,:2]
    out=trustworthiness_embedding(x,embedded,n_neighbors=5)
    assert out["trustworthiness"]==pytest.approx(trustworthiness(x,embedded,n_neighbors=5))
    with pytest.raises(ValueError):
        trustworthiness_embedding(x,embedded,n_neighbors=30)


def test_distance_correlation_identical_and_independent_samples():
    rng=np.random.default_rng(17)
    x=rng.normal(size=80)
    assert distance_correlation(x,x)==pytest.approx(1)
    independent=distance_correlation(x,rng.normal(size=80))
    assert 0<=independent<=1
    with pytest.raises(ValueError):
        distance_correlation([2]*10,np.arange(10))


def test_reproducible_cluster_stability():
    from sklearn.datasets import make_blobs
    X,_=make_blobs(n_samples=90,centers=3,cluster_std=.25,random_state=11)
    a=bootstrap_cluster_stability(X,n_clusters=3,repetitions=10,seed=5)
    b=bootstrap_cluster_stability(X,n_clusters=3,repetitions=10,seed=5)
    assert a["scores"]==b["scores"]
    assert a["mean_ari"]>.9
    assert -1<=a["lower_ari"]<=a["upper_ari"]<=1
