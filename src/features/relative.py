from __future__ import annotations

import numpy as np
import pandas as pd


def relative(df: pd.DataFrame, window: int) -> pd.Series:
    samsung_log_ret = np.log(df["price"].astype(float) / df["price"].shift(window).astype(float))
    kospi_log_ret = np.log1p(df["kospi200_return"].astype(float)).rolling(window).sum()
    return samsung_log_ret - kospi_log_ret


def build_relative(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    params = config["features"]["params"]["relative"]
    return relative(df, params["window"]) * action
