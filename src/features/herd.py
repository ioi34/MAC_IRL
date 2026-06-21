from __future__ import annotations

import pandas as pd


def herd_flow(df: pd.DataFrame, investor: str, investors: list[str], window: int) -> pd.Series:
    other_investors = [name for name in investors if name != investor]
    lagged_flows = [
        df[f"u_{other}"].shift(1).rolling(window=window, min_periods=window).mean()
        for other in other_investors
    ]
    return sum(lagged_flows) / len(lagged_flows)


def build_herd(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    params = config["features"]["params"]["herd"]
    base = herd_flow(df, investor, config["investors"], params["window"])
    return base * action
