from __future__ import annotations

import numpy as np
import pandas as pd


def turnover_zscore(df: pd.DataFrame, trading_value_column: str, window: int) -> pd.Series:
    log_turnover = np.log(pd.to_numeric(df[trading_value_column], errors="raise"))
    rolling = log_turnover.rolling(window=window, min_periods=window)
    standard_deviation = rolling.std(ddof=0).replace(0, np.nan)
    return (log_turnover - rolling.mean()) / standard_deviation


def _build_turnover(
    df: pd.DataFrame,
    config: dict,
    action: int,
    feature_name: str,
) -> pd.Series:
    window = int(config["features"]["params"][feature_name]["window"])
    trading_value_column = config["columns"]["trading_value"]
    return turnover_zscore(df, trading_value_column, window) * action


def build_turnover_20(
    df: pd.DataFrame,
    config: dict,
    investor: str,
    action: int,
) -> pd.Series:
    return _build_turnover(df, config, action, "turnover_20")


def build_turnover_60(
    df: pd.DataFrame,
    config: dict,
    investor: str,
    action: int,
) -> pd.Series:
    return _build_turnover(df, config, action, "turnover_60")


def build_turnover_120(
    df: pd.DataFrame,
    config: dict,
    investor: str,
    action: int,
) -> pd.Series:
    return _build_turnover(df, config, action, "turnover_120")
