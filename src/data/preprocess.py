from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.average_cost import add_average_cost_states


def resolve_price_column(df: pd.DataFrame, columns: dict) -> str:
    price_col = columns["price"]
    fallback = columns.get("fallback_price")
    if price_col in df.columns:
        return price_col
    if fallback and fallback in df.columns:
        return fallback
    raise KeyError(f"Neither price column '{price_col}' nor fallback '{fallback}' exists")


def prepare_daily_frame(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    columns = config["columns"]
    date_col = columns["date"]
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    ticker = config.get("preprocess", {}).get("ticker")
    ticker_col = columns.get("ticker")
    if ticker and ticker_col in out.columns:
        out = out.loc[out[ticker_col] == ticker].copy()

    price_col = resolve_price_column(out, columns)
    if price_col != "price":
        out["price"] = out[price_col].astype(float)

    trade_columns = [
        column
        for investor_columns in columns["investor_trades"].values()
        for column in investor_columns.values()
    ]
    required = [date_col, "price", columns["trading_value"], *trade_columns]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise KeyError(
            "Missing required columns for gross-trade average-cost proxies: "
            f"{missing}"
        )

    out = out.sort_values(date_col).drop_duplicates(date_col, keep="last")
    trading_value = pd.to_numeric(out[columns["trading_value"]], errors="raise").replace(0, np.nan)
    for investor, investor_columns in columns["investor_trades"].items():
        buy_value = pd.to_numeric(out[investor_columns["buy_value"]], errors="raise")
        sell_value = pd.to_numeric(out[investor_columns["sell_value"]], errors="raise")
        out[f"net_buy_{investor}"] = buy_value - sell_value
        out[f"u_{investor}"] = out[f"net_buy_{investor}"] / trading_value

    if config.get("preprocess", {}).get("drop_missing_required", True):
        out = out.dropna(subset=required + [f"u_{i}" for i in config["investors"]])

    return add_average_cost_states(out.reset_index(drop=True), config)
