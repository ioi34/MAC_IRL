from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.labels import investor_trade_imbalance


def add_continuous_action_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add next-period continuous net-buy imbalance labels in [-1, 1]."""
    out = df.copy()
    label_shift = int(config.get("labeling", {}).get("label_shift", 1))
    for investor in config["investors"]:
        action = investor_trade_imbalance(out, config, investor)
        out[f"continuous_action_{investor}"] = action.shift(-label_shift)
    return out


def continuous_actions_array(df: pd.DataFrame, investors: list[str]) -> np.ndarray:
    cols = [f"continuous_action_{investor}" for investor in investors]
    return df[cols].to_numpy(dtype=np.float32)
