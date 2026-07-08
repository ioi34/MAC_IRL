from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.registry import FEATURE_REGISTRY


def build_state_feature_tensor(df: pd.DataFrame, config: dict) -> tuple[np.ndarray, list[str]]:
    """Build action-free state features for the continuous-action policy."""
    feature_names = list(config["features"]["selected"])
    investors = list(config["investors"])
    tensor = np.empty((len(df), len(investors), len(feature_names)), dtype=np.float32)

    for f_idx, name in enumerate(feature_names):
        if name not in FEATURE_REGISTRY:
            raise KeyError(f"Unknown feature '{name}'. Available: {sorted(FEATURE_REGISTRY)}")
        builder = FEATURE_REGISTRY[name]
        for i_idx, investor in enumerate(investors):
            tensor[:, i_idx, f_idx] = builder(df, config, investor, action=1).to_numpy(
                dtype=np.float32
            )

    return tensor, feature_names


def valid_rows_for_continuous_model(
    df: pd.DataFrame,
    features: np.ndarray,
    investors: list[str],
    contexts: np.ndarray | None = None,
) -> np.ndarray:
    label_cols = [f"continuous_action_{investor}" for investor in investors]
    labels_ok = df[label_cols].notna().all(axis=1).to_numpy()
    features_ok = np.isfinite(features).all(axis=(1, 2))
    contexts_ok = np.ones(len(df), dtype=bool)
    if contexts is not None:
        contexts_ok = np.isfinite(contexts).all(axis=1)
    return labels_ok & features_ok & contexts_ok
