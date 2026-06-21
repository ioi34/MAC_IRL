from __future__ import annotations

import numpy as np
import pandas as pd


def momentum(df: pd.DataFrame, window: int) -> pd.Series:
    return np.log(df["price"].astype(float) / df["price"].shift(window).astype(float))


def build_momentum(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    params = config["features"]["params"]["momentum"]
    return momentum(df, params["window"]) * action
