from __future__ import annotations

import pandas as pd


def build_underwater(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    return df[f"underwater_gap_{investor}"] * action
