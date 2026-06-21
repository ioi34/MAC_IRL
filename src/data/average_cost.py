from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_average_cost_state(
    buy_quantity: pd.Series,
    sell_quantity: pd.Series,
    buy_value: pd.Series,
    sell_value: pd.Series,
    rho: float,
) -> pd.DataFrame:
    if not 0 < rho <= 1:
        raise ValueError("average_cost_rho must be in (0, 1]")

    values = pd.DataFrame(
        {
            "buy_quantity": pd.to_numeric(buy_quantity, errors="raise"),
            "sell_quantity": pd.to_numeric(sell_quantity, errors="raise"),
            "buy_value": pd.to_numeric(buy_value, errors="raise"),
            "sell_value": pd.to_numeric(sell_value, errors="raise"),
        },
        index=buy_quantity.index,
    ).astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Gross trade columns must contain only finite values")
    if (values < 0).any().any():
        raise ValueError("Gross trade quantities and values must be non-negative")
    if ((values["buy_quantity"] == 0) & (values["buy_value"] != 0)).any():
        raise ValueError("buy_value must be zero when buy_quantity is zero")
    if ((values["sell_quantity"] == 0) & (values["sell_value"] != 0)).any():
        raise ValueError("sell_value must be zero when sell_quantity is zero")
    if ((values["buy_quantity"] > 0) & (values["buy_value"] <= 0)).any():
        raise ValueError("buy_value must be positive when buy_quantity is positive")
    if ((values["sell_quantity"] > 0) & (values["sell_value"] <= 0)).any():
        raise ValueError("sell_value must be positive when sell_quantity is positive")

    buy_vwap = values["buy_value"].div(values["buy_quantity"].replace(0, np.nan))
    sell_vwap = values["sell_value"].div(values["sell_quantity"].replace(0, np.nan))
    total_quantity = values["buy_quantity"] + values["sell_quantity"]
    trade_vwap = (values["buy_value"] + values["sell_value"]).div(
        total_quantity.replace(0, np.nan)
    )

    holdings = np.zeros(len(values), dtype=float)
    average_costs = np.full(len(values), np.nan, dtype=float)
    previous_quantity = 0.0
    previous_cost = np.nan

    for idx, row in enumerate(values.itertuples(index=False)):
        effective_previous = rho * previous_quantity
        quantity = max(0.0, effective_previous + row.buy_quantity - row.sell_quantity)
        remaining_old = max(0.0, effective_previous - row.sell_quantity)
        remaining_new = quantity - remaining_old

        if quantity == 0:
            average_cost = np.nan
        elif remaining_old == 0:
            average_cost = buy_vwap.iloc[idx]
        elif remaining_new == 0:
            average_cost = previous_cost
        else:
            average_cost = (
                remaining_old * previous_cost + remaining_new * buy_vwap.iloc[idx]
            ) / quantity

        holdings[idx] = quantity
        average_costs[idx] = average_cost
        previous_quantity = quantity
        previous_cost = average_cost

    average_cost = pd.Series(average_costs, index=values.index)
    underwater_gap = ((average_cost - trade_vwap) / average_cost).clip(lower=0)
    return pd.DataFrame(
        {
            "buy_vwap": buy_vwap,
            "sell_vwap": sell_vwap,
            "trade_vwap": trade_vwap,
            "holding_proxy": holdings,
            "average_cost_proxy": average_cost,
            "underwater_gap": underwater_gap,
        },
        index=values.index,
    )


def add_average_cost_states(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()
    rho = float(config["preprocess"]["average_cost_rho"])
    for investor, columns in config["columns"]["investor_trades"].items():
        state = calculate_average_cost_state(
            out[columns["buy_quantity"]],
            out[columns["sell_quantity"]],
            out[columns["buy_value"]],
            out[columns["sell_value"]],
            rho,
        )
        for name in state.columns:
            out[f"{name}_{investor}"] = state[name]
    return out
