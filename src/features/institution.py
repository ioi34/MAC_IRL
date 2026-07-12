from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.labels import investor_trade_imbalance


def execution_persistence(
    df: pd.DataFrame,
    config: dict,
    investor: str,
    span: int,
) -> pd.Series:
    """End-of-day trade imbalance smoothed over recent days."""
    imbalance = investor_trade_imbalance(df, config, investor)
    return imbalance.ewm(span=span, adjust=False, min_periods=span).mean()


def residual_return(df: pd.DataFrame, config: dict, window: int) -> pd.Series:
    """Stock log return minus the matched KOSPI200 log return."""
    price = pd.to_numeric(df["price"], errors="raise")
    kospi = pd.to_numeric(df[config["columns"]["kospi200_return"]], errors="raise")
    stock_return = np.log(price / price.shift(window))
    market_return = np.log1p(kospi).rolling(window=window, min_periods=window).sum()
    return stock_return - market_return


def build_execution_persistence_3(
    df: pd.DataFrame,
    config: dict,
    investor: str,
    action: int,
) -> pd.Series:
    span = int(config["features"]["params"]["execution_persistence_3"].get("span", 3))
    return execution_persistence(df, config, investor, span) * action


def build_short_residual_return_1(
    df: pd.DataFrame,
    config: dict,
    investor: str,
    action: int,
) -> pd.Series:
    del investor
    return residual_return(df, config, window=1) * action


def build_short_residual_return_5(
    df: pd.DataFrame,
    config: dict,
    investor: str,
    action: int,
) -> pd.Series:
    del investor
    return residual_return(df, config, window=5) * action


def build_benchmark_drift_20(
    df: pd.DataFrame,
    config: dict,
    investor: str,
    action: int,
) -> pd.Series:
    del investor
    window = int(config["features"]["params"]["benchmark_drift_20"].get("window", 20))
    return residual_return(df, config, window=window) * action
