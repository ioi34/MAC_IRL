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


def _other_investors(investor: str, investors: list[str]) -> list[str]:
    return [inv for inv in investors if inv != investor]


def build_herd_a(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    source = _other_investors(investor, config["investors"])[0]
    return df[f"u_{source}"].shift(1) * action


def build_herd_b(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    source = _other_investors(investor, config["investors"])[1]
    return df[f"u_{source}"].shift(1) * action


def get_herd_source_map(investors: list[str]) -> list[dict]:
    rows = []
    for inv in investors:
        others = _other_investors(inv, investors)
        rows.append({"investor": inv, "herd_a": others[0], "herd_b": others[1]})
    return rows
