from __future__ import annotations

import numpy as np
import pandas as pd


def volatility(df: pd.DataFrame, window: int) -> pd.Series:
    returns = np.log(df["price"].astype(float) / df["price"].shift(1).astype(float))
    return returns.rolling(window=window, min_periods=window).std(ddof=0)


def build_volatility(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    params = config["features"]["params"]["volatility"]
    base = volatility(df, params["window"])
    if params.get("magnitude", False):
        return base * abs(action)
    return base * action
