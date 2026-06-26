from __future__ import annotations

import numpy as np
import pandas as pd


def parkinson_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    log_hl = np.log(df["high"].astype(float) / df["low"].astype(float))
    rv = (log_hl ** 2) / (4 * np.log(2))
    return rv.rolling(window=window, min_periods=window).mean().apply(np.sqrt)


def build_volatility(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    params = config["features"]["params"]["volatility"]
    base = parkinson_volatility(df, params["window"])
    if params.get("magnitude", False):
        return base * abs(action)
    return base * action
