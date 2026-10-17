"""Independent numerical and adversarial tests for v0.10 research methods."""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from khaled_dsml import (
    henze_zirkler_normality_test, mnar_delta_sensitivity,
    render_analysis_dashboard,
)


def test_hz_matches_independent_pairwise_kernel_formula():
    rng = np.random.default_rng(18)
    x = rng.normal(size=(36, 3))
    result = henze_zirkler_normality_test(x, n_resamples=99, random_state=7)
    n, p = x.shape
    z = x - x.mean(axis=0)
    z = z @ np.linalg.inv(np.linalg.cholesky(z.T @ z / n).T)
    beta = (1 / np.sqrt(2)) * ((2 * p + 1) / 4)**(1 / (p + 4)) * n**(1 / (p + 4))
    dsq = cdist(z, z, metric="sqeuclidean")
    radii = np.sum(z**2, axis=1)
    independent = n * (
        np.exp(-beta**2 * dsq / 2).mean()
        - 2 * np.mean(np.exp(-beta**2 * radii / (2 * (1 + beta**2))))
        / (1 + beta**2)**(p / 2)
        + (1 + 2 * beta**2)**(-p / 2)
    )
    assert result["statistic"] == pytest.approx(independent, abs=1e-10)
    assert result["p_value"] == (1 + result["monte_carlo_exceedances"]) / 100
    assert result["n_resamples"] == 99


def test_hz_affine_invariance_reproducibility_and_rejection():
    rng = np.random.default_rng(212)
    x = rng.normal(size=(75, 3))
    y = x @ np.array([[2, 0, 1], [0, -3, 0], [1, 0, 2]]) + [15, -5, 50]
    a = henze_zirkler_normality_test(x, n_resamples=99, random_state=9)
    b = henze_zirkler_normality_test(y, n_resamples=99, random_state=9)
    assert a["statistic"] == pytest.approx(b["statistic"], abs=1e-10)
    assert a["p_value"] == b["p_value"]
    nonnormal = rng.exponential(size=(90, 3))
    c = henze_zirkler_normality_test(nonnormal, n_resamples=199, random_state=9)
    assert c["p_value"] <= .05
    assert "عدم الرفض" in a["interpretation_ar"]
    html = render_analysis_dashboard(a, theme="aurora")
    assert "هينز" in html and 'dir="rtl"' in html


@pytest.mark.parametrize("bad", [
    np.ones((30, 3)),
    np.array([[np.nan, 1], [2, 3]]),
    np.ones((6, 2)),
    np.ones(25),
])
def test_hz_rejects_bad_shape_missingness_or_singularity(bad):
    with pytest.raises(ValueError):
        henze_zirkler_normality_test(bad, n_resamples=99)


@pytest.mark.parametrize("kwargs", [
    {"alpha": 1}, {"alpha": True},
    {"n_resamples": 19}, {"n_resamples": True}, {"n_resamples": 99.5},
])
def test_hz_validates_parameters(kwargs):
    with pytest.raises(ValueError):
        henze_zirkler_normality_test(np.random.default_rng(2).normal(size=(30, 3)), **kwargs)


def test_mnar_delta_only_changes_missing_cells_and_does_not_infer():
    original = pd.DataFrame({"الدخل": [1., 2., np.nan, 4., np.nan, 6.], "x": range(6)})
    v1 = original.fillna({"الدخل": 3.})
    v2 = original.fillna({"الدخل": 5.})
    a = mnar_delta_sensitivity(
        original, [v1, v2], column="الدخل", deltas=[-3., 0., 6.]
    )
    t = a["table"]
    baseline = (v1["الدخل"].mean() + v2["الدخل"].mean()) / 2
    assert a["baseline_mean"] == pytest.approx(baseline)
    np.testing.assert_allclose(t["mean_across_imputations"], baseline + (2 / 6) * np.array([-3, 0, 6]))
    np.testing.assert_allclose(t["change_from_zero_delta"], [-1, 0, 2])
    assert "p_value" not in a and "ci_lower" not in a
    assert "MNAR" in a["interpretation_en"]
    assert "RTL" in render_analysis_dashboard(a, theme="amethyst").upper()


def test_mnar_sensitivity_rejects_leakage_and_misalignments():
    orig = pd.DataFrame({"y": [1., np.nan, 3., 4.], "x": [1., 2., 3., 4.]})
    imputed = orig.fillna({"y": 2.})
    with pytest.raises(ValueError):
        mnar_delta_sensitivity(orig, [imputed], column="y", deltas=[0, 1])
    with pytest.raises(ValueError):
        mnar_delta_sensitivity(orig, [imputed, imputed.iloc[::-1]], column="y", deltas=[0, 1])
    bad = imputed.copy()
    bad.loc[0, "y"] = 100
    with pytest.raises(ValueError):
        mnar_delta_sensitivity(orig, [imputed, bad], column="y", deltas=[0, 1])
    with pytest.raises(ValueError):
        mnar_delta_sensitivity(orig, [imputed, imputed], column="y", deltas=[np.inf, 0])
    with pytest.raises(ValueError):
        mnar_delta_sensitivity(orig.fillna({"y": 2}), [imputed, imputed], column="y", deltas=[0, 1])
