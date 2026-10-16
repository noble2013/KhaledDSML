"""Time-safe lagged features for classroom forecasting exercises."""
from __future__ import annotations
import pandas as pd


def lagged_features(series, *, lags=(1, 2, 3), rolling_window=None) -> pd.DataFrame:
    """Create past-only features: lag t-n and optional mean up to t-1.

    Input must already be sorted in ascending time order. No random splitting
    should be used to validate temporal forecasts.
    """
    values = pd.Series(series)
    if len(lags) < 1 or any(not isinstance(lag, int) or lag < 1 for lag in lags):
        raise ValueError("lags must be positive integers")
    out = pd.DataFrame(index=values.index)
    for lag in lags:
        out[f"lag_{lag}"] = values.shift(lag)
    if rolling_window is not None:
        if not isinstance(rolling_window, int) or rolling_window < 2:
            raise ValueError("rolling_window must be an integer >= 2")
        out[f"rolling_mean_{rolling_window}"] = values.shift(1).rolling(rolling_window).mean()
    return out
