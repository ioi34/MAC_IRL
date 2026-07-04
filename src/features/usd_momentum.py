from __future__ import annotations

import numpy as np
import pandas as pd


def usd_momentum(df: pd.DataFrame, config: dict, window: int) -> pd.Series:
    price = pd.to_numeric(df["price"], errors="raise")
    usdkrw = pd.to_numeric(df[config["columns"]["usdkrw"]], errors="raise")
    return np.log(price / price.shift(window)) - np.log(usdkrw / usdkrw.shift(window))


def build_usd_momentum(
    df: pd.DataFrame,
    config: dict,
    investor: str,
    action: int,
) -> pd.Series:
    window = int(config["features"]["params"]["usd_momentum"]["window"])
    return usd_momentum(df, config, window) * action
