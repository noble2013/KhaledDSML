import numpy as np
import pandas as pd
import pytest
from khaled_dsml.econometrics import (adf_test, vif_table, breusch_pagan_test,
                                         ljung_box_test, cronbach_alpha)


def test_adf_stationary():
    r = adf_test(np.random.default_rng(22).normal(size=200))
    assert r.reject_null


def test_ljung_box():
    r = ljung_box_test(np.random.default_rng(10).normal(size=100), lags=5)
    assert 0 <= r.p_value <= 1


def test_cronbach_alpha():
    rng = np.random.default_rng(5)
    latent = rng.normal(size=100)
    items = pd.DataFrame({"a": latent + rng.normal(scale=.1, size=100),
                          "b": latent + rng.normal(scale=.1, size=100),
                          "c": latent + rng.normal(scale=.1, size=100)})
    assert cronbach_alpha(items) > .9


def test_breusch_pagan():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(200, 2))
    residuals = rng.normal(size=200)
    assert 0 <= breusch_pagan_test(residuals, x).p_value <= 1


def test_vif():
    rng = np.random.default_rng(2)
    result = vif_table(pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"]))
    assert len(result) == 3 and result["vif"].min() >= 1 - 1e-6
