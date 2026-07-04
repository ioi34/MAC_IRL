from __future__ import annotations

import numpy as np
import pandas as pd


def fx_return_1d(df: pd.DataFrame, config: dict) -> pd.Series:
    column = config["columns"]["usdkrw"]
    usdkrw = pd.to_numeric(df[column], errors="raise")
    return np.log(usdkrw / usdkrw.shift(1))


def build_fx_sensitivity_1d(
    df: pd.DataFrame,
    config: dict,
    investor: str,
    action: int,
) -> pd.Series:
    return -fx_return_1d(df, config) * action
