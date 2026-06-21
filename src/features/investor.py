from __future__ import annotations

import pandas as pd


def herd_flow(df: pd.DataFrame, investor: str, investors: list[str], window: int) -> pd.Series:
    other_investors = [name for name in investors if name != investor]
    shifted = []
    for other in other_investors:
        flow = df[f"u_{other}"].shift(1).rolling(window=window, min_periods=window).mean()
        shifted.append(flow)
    return sum(shifted) / len(shifted)

