from __future__ import annotations

import numpy as np
import pandas as pd


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

    required = [
        date_col,
        "price",
        columns["trading_value"],
        *columns["net_buy"].values(),
    ]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    out = out.sort_values(date_col).drop_duplicates(date_col, keep="last")
    for investor, col in columns["net_buy"].items():
        out[f"u_{investor}"] = out[col].astype(float) / out[columns["trading_value"]].replace(0, np.nan)

    if config.get("preprocess", {}).get("drop_missing_required", True):
        out = out.dropna(subset=required + [f"u_{i}" for i in config["investors"]])

    return out.reset_index(drop=True)
