from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.features.investor import herd_flow
from src.features.loss_aversion import ma_loss_gap
from src.features.market import momentum, volatility

FeatureBuilder = Callable[[pd.DataFrame, dict, str, int], pd.Series]


def _action_signed(base: pd.Series, action: int) -> pd.Series:
    return base * action


def build_loss_aversion(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    params = config["features"]["params"]["loss_aversion"]
    gap = ma_loss_gap(df, params["reference_window"])
    return gap if action < 0 else gap * 0.0


def build_herd(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    params = config["features"]["params"]["herd"]
    base = herd_flow(df, investor, config["investors"], params["window"])
    return _action_signed(base, action)


def build_momentum(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    params = config["features"]["params"]["momentum"]
    return _action_signed(momentum(df, params["window"]), action)


def build_volatility(df: pd.DataFrame, config: dict, investor: str, action: int) -> pd.Series:
    params = config["features"]["params"]["volatility"]
    return _action_signed(-volatility(df, params["window"]), action)


FEATURE_REGISTRY: dict[str, FeatureBuilder] = {
    "loss_aversion": build_loss_aversion,
    "herd": build_herd,
    "momentum": build_momentum,
    "volatility": build_volatility,
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
