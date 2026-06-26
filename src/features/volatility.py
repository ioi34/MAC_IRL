from __future__ import annotations

import numpy as np
import pandas as pd


def close_to_close_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    returns = np.log(df["price"].astype(float) / df["price"].shift(1).astype(float))
    return returns.rolling(window=window, min_periods=window).std(ddof=0)


def parkinson_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    log_hl = np.log(df["high"].astype(float) / df["low"].astype(float))
    rv = (log_hl ** 2) / (4 * np.log(2))
    return rv.rolling(window=window, min_periods=window).mean().apply(np.sqrt)


def build_volatility(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    params = config["features"]["params"]["volatility"]
    method = params.get("method", "close_to_close")
    if method == "parkinson":
        base = parkinson_volatility(df, params["window"])
    else:
        base = close_to_close_volatility(df, params["window"])
    if params.get("magnitude", False):
        return base * abs(action)
    return base * action
