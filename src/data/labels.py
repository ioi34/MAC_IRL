from __future__ import annotations

import numpy as np
import pandas as pd


ACTION_TO_INDEX = {-1: 0, 1: 1}
INDEX_TO_ACTION = {v: k for k, v in ACTION_TO_INDEX.items()}


def rolling_binary_action_labels(
    flow: pd.Series,
    window: int = 252,
    quantile: float = 0.50,
) -> pd.Series:
    threshold = flow.shift(1).rolling(window=window, min_periods=window).quantile(quantile)
    labels = pd.Series(np.nan, index=flow.index, dtype="float64")
    labels = labels.mask(flow < threshold, -1)
    labels = labels.mask(flow >= threshold, 1)
    return labels


def add_action_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = df.copy()
    labeling = config["labeling"]
    for investor in config["investors"]:
        label = rolling_binary_action_labels(
            out[f"u_{investor}"],
            window=labeling["rolling_window"],
            quantile=labeling["quantile"],
        )
        shifted = label.shift(-int(labeling.get("label_shift", 1)))
        out[f"action_{investor}"] = shifted
        out[f"action_idx_{investor}"] = shifted.map(ACTION_TO_INDEX)
    return out


def labels_array(df: pd.DataFrame, investors: list[str]) -> np.ndarray:
    cols = [f"action_idx_{investor}" for investor in investors]
    return df[cols].to_numpy(dtype=np.int64)
