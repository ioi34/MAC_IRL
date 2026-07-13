from __future__ import annotations

import numpy as np
import pandas as pd


def momentum(df: pd.DataFrame, window: int) -> pd.Series:
    return np.log(df["price"].astype(float) / df["price"].shift(window).astype(float))


def shortmom_orth(df: pd.DataFrame, short_window: int, long_window: int) -> pd.Series:
    short_return = momentum(df, short_window)
    long_return = momentum(df, long_window)
    return short_return - (short_window / long_window) * long_return


def build_momentum(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    params = config["features"]["params"]["momentum"]
    return momentum(df, params["window"]) * action


def build_shortmom_orth(
    df: pd.DataFrame,
    config: dict,
    investor: str,
    action: int,
) -> pd.Series:
    params = config["features"]["params"]["shortmom_orth"]
    return shortmom_orth(df, params["short_window"], params["long_window"]) * action
