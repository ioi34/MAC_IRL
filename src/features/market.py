from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_log_ratio(current: pd.Series, lagged: pd.Series) -> pd.Series:
    return np.log(current.astype(float) / lagged.astype(float))


def momentum(df: pd.DataFrame, window: int) -> pd.Series:
    return _safe_log_ratio(df["price"], df["price"].shift(window))


def volatility(df: pd.DataFrame, window: int) -> pd.Series:
    returns = _safe_log_ratio(df["price"], df["price"].shift(1))
    return returns.rolling(window=window, min_periods=window).std(ddof=0)
