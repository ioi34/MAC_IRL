from __future__ import annotations

import numpy as np
import pandas as pd


def ma_loss_gap(df: pd.DataFrame, reference_window: int) -> pd.Series:
    reference = df["price"].rolling(window=reference_window, min_periods=reference_window).mean()
    return ((reference - df["price"]) / reference).clip(lower=0)


def build_loss_aversion(
    df: pd.DataFrame,
    config: dict,
    investor: str,
    action: int,
) -> pd.Series:
    params = config["features"]["params"]["loss_aversion"]
    gap = ma_loss_gap(df, params["reference_window"])
    return gap if action < 0 else gap * 0.0


def rolling_max_loss_gap(df: pd.DataFrame, reference_window: int) -> pd.Series:
    reference = df["price"].rolling(window=reference_window, min_periods=reference_window).max()
    return ((reference - df["price"]) / reference).clip(lower=0)


def average_cost_proxy_loss_gap(
    df: pd.DataFrame,
    investor: str,
    net_buy_volume_col: str,
    rho: float = 0.98,
) -> pd.Series:
    """Proxy investor average-cost loss gap.

    This optional feature requires share-volume net-buy data. It is included as an
    extension point but is not selected by the default config.
    """
    if net_buy_volume_col not in df.columns:
        raise KeyError(f"Missing net-buy volume column for {investor}: {net_buy_volume_col}")

    price = df["price"].to_numpy(dtype=float)
    nb_vol = df[net_buy_volume_col].to_numpy(dtype=float)
    quantity = 0.0
    avg_cost = np.nan
    gaps = np.full(len(df), np.nan, dtype=float)

    for idx in range(len(df)):
        if np.isfinite(avg_cost) and avg_cost > 0:
            gaps[idx] = max(0.0, (avg_cost - price[idx]) / avg_cost)
        if nb_vol[idx] > 0:
            decayed_quantity = rho * quantity
            denom = decayed_quantity + nb_vol[idx]
            avg_cost = (
                (decayed_quantity * avg_cost if np.isfinite(avg_cost) else 0.0)
                + nb_vol[idx] * price[idx]
            ) / denom
            quantity = denom
        else:
            quantity = rho * quantity

    return pd.Series(gaps, index=df.index, name=f"loss_proxy_{investor}")
