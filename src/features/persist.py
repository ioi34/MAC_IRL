from __future__ import annotations

import pandas as pd


def build_persist(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    return df[f"u_{investor}"].shift(1) * action
