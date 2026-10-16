import numpy as np
import pandas as pd
import pytest
from khaled_dsml.timeseries import lagged_features


def test_lagged_features_are_past_only():
    s = pd.Series([10, 20, 30, 40, 100])
    out = lagged_features(s, lags=(1, 2), rolling_window=2)
    assert np.isnan(out.loc[0, "lag_1"])
    assert out.loc[4, "lag_1"] == 40
    assert out.loc[4, "lag_2"] == 30
    assert out.loc[4, "rolling_mean_2"] == 35


def test_invalid_lag():
    with pytest.raises(ValueError):
        lagged_features([1, 2, 3], lags=(0,))
