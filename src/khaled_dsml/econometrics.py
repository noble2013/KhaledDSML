"""Optional statsmodels diagnostics for quantitative economics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from ._results import make_result, TestResult


def _require_statsmodels():
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ImportError('Install with: pip install "khaled-dsml[econometrics]"') from exc
    return sm


def adf_test(series, alpha=0.05) -> TestResult:
    sm = _require_statsmodels()
    from statsmodels.tsa.stattools import adfuller
    x = pd.to_numeric(pd.Series(series), errors="raise").dropna().to_numpy(dtype=float)
    if len(x) < 10:
        raise ValueError("Need at least 10 observations for this educational ADF check")
    stat, p, *_ = adfuller(x, autolag="AIC")
    return make_result("Augmented Dickey-Fuller", stat, p, len(x), alpha,
                       "توجد جذور وحدة (السلسلة غير مستقرة بالمعنى المختبر).",
                       "اختيار المكون الحتمي وطول الإبطاء يؤثر في الاستنتاج.")


def vif_table(frame: pd.DataFrame) -> pd.DataFrame:
    sm = _require_statsmodels()
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    if not isinstance(frame, pd.DataFrame) or frame.shape[1] < 2:
        raise ValueError("Provide numeric DataFrame with at least 2 features")
    arr = frame.to_numpy(dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("VIF requires complete finite numeric data")
    design = sm.add_constant(arr, has_constant="add")
    values = [float(variance_inflation_factor(design, i + 1)) for i in range(arr.shape[1])]
    return pd.DataFrame({"feature": frame.columns.tolist(), "vif": values})


def breusch_pagan_test(residuals, features, alpha=.05) -> TestResult:
    sm = _require_statsmodels()
    from statsmodels.stats.diagnostic import het_breuschpagan
    residuals = np.asarray(residuals, dtype=float)
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or len(residuals) != len(x) or not np.all(np.isfinite(x)):
        raise ValueError("Features must be a finite 2D matrix aligned with residuals")
    if x.shape[1] < 1 or len(x) <= x.shape[1] + 1:
        raise ValueError("Insufficient observations for Breusch-Pagan test")
    design = sm.add_constant(x, has_constant="add")
    stat, p, *_ = het_breuschpagan(residuals, design)
    return make_result("Breusch-Pagan", stat, p, len(residuals), alpha,
                       "تباين الأخطاء ثابت.")


def ljung_box_test(series, lags=10, alpha=0.05) -> TestResult:
    sm = _require_statsmodels()
    from statsmodels.stats.diagnostic import acorr_ljungbox
    x = pd.to_numeric(pd.Series(series), errors="raise").dropna().to_numpy(dtype=float)
    if lags < 1 or len(x) <= lags:
        raise ValueError("Need 1 <= lags < number of observations")
    r = acorr_ljungbox(x, lags=[lags], return_df=True)
    return make_result("Ljung-Box", r["lb_stat"].iloc[0], r["lb_pvalue"].iloc[0],
                       len(x), alpha, "لا يوجد ارتباط ذاتي حتى عدد الإبطاءات المختار.")


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Internal-consistency coefficient; no claim of unidimensionality."""
    if not isinstance(items, pd.DataFrame) or items.shape[1] < 2:
        raise ValueError("Need >=2 item columns")
    x = items.astype(float).dropna().to_numpy()
    if len(x) < 2:
        raise ValueError("Need >=2 complete cases")
    k = x.shape[1]
    total = np.var(x.sum(axis=1), ddof=1)
    if total <= 0:
        raise ValueError("Total-score variance is zero")
    return float((k / (k - 1)) * (1 - np.var(x, axis=0, ddof=1).sum() / total))
