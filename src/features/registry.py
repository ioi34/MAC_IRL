from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.features.herd import build_herd
from src.features.momentum import build_momentum
from src.features.persist import build_persist
from src.features.underwater import build_underwater
from src.features.volatility import build_volatility

FeatureBuilder = Callable[[pd.DataFrame, dict, str, int], pd.Series]


FEATURE_REGISTRY: dict[str, FeatureBuilder] = {
    "underwater": build_underwater,
    "herd": build_herd,
    "momentum": build_momentum,
    "volatility": build_volatility,
    "persist": build_persist,
}


def build_feature_tensor(df: pd.DataFrame, config: dict) -> tuple[np.ndarray, list[str]]:
    feature_names = list(config["features"]["selected"])
    investors = list(config["investors"])
    actions = list(config["action_values"])
    tensor = np.empty((len(df), len(investors), len(actions), len(feature_names)), dtype=np.float32)

    for f_idx, name in enumerate(feature_names):
        if name not in FEATURE_REGISTRY:
            raise KeyError(f"Unknown feature '{name}'. Available: {sorted(FEATURE_REGISTRY)}")
        builder = FEATURE_REGISTRY[name]
        for i_idx, investor in enumerate(investors):
            for a_idx, action in enumerate(actions):
                tensor[:, i_idx, a_idx, f_idx] = builder(df, config, investor, int(action)).to_numpy(
                    dtype=np.float32
                )

    return tensor, feature_names


def valid_rows_for_model(df: pd.DataFrame, features: np.ndarray, investors: list[str]) -> np.ndarray:
    label_cols = [f"action_idx_{investor}" for investor in investors]
    labels_ok = df[label_cols].notna().all(axis=1).to_numpy()
    features_ok = np.isfinite(features).all(axis=(1, 2, 3))
    return labels_ok & features_ok
