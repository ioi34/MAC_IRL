from __future__ import annotations

import pandas as pd


def _herd_from(source: str, df: pd.DataFrame, investor: str, action: int) -> pd.Series:
    if investor == source:
        return pd.Series(0.0, index=df.index)
    return df[f"u_{source}"].shift(1) * action


def build_herd_from_foreign(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    return _herd_from("foreign", df, investor, action)


def build_herd_from_institution(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    return _herd_from("institution", df, investor, action)


def build_herd_from_retail(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    return _herd_from("retail", df, investor, action)
