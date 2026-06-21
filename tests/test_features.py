import numpy as np
import pandas as pd

from src.data.labels import add_action_labels
from src.data.preprocess import prepare_daily_frame
from src.features.registry import build_feature_tensor


def _config():
    return {
        "columns": {
            "date": "date",
            "ticker": "ticker",
            "price": "adjusted_close",
            "fallback_price": "close",
            "trading_value": "trading_value",
            "net_buy": {
                "foreign": "foreign_net_buy",
                "institution": "institution_net_buy",
                "retail": "retail_net_buy",
            },
        },
        "investors": ["foreign", "institution", "retail"],
        "action_values": [-1, 1],
        "labeling": {"rolling_window": 3, "quantile": 0.50, "label_shift": 1},
        "preprocess": {"ticker": None, "drop_missing_required": True},
        "features": {
            "selected": ["loss_aversion", "herd", "momentum", "volatility"],
            "params": {
                "loss_aversion": {"reference_window": 3},
                "herd": {"window": 2},
                "momentum": {"window": 2},
                "volatility": {"window": 3},
            },
        },
    }


def test_feature_tensor_has_binary_actions_and_four_reward_features():
    n = 12
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n),
            "adjusted_close": np.linspace(100, 111, n),
            "trading_value": np.linspace(1000, 1110, n),
            "foreign_net_buy": np.arange(n),
            "institution_net_buy": np.arange(n) * -1,
            "retail_net_buy": np.ones(n),
        }
    )
    config = _config()
    prepared = add_action_labels(prepare_daily_frame(frame, config), config)
    features, names = build_feature_tensor(prepared, config)

    assert names == ["loss_aversion", "herd", "momentum", "volatility"]
    assert features.shape == (n, 3, 2, 4)


def test_herd_feature_uses_only_lagged_flows():
    config = _config()
    n = 12
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n),
            "adjusted_close": np.linspace(100, 111, n),
            "trading_value": np.ones(n) * 1000,
            "foreign_net_buy": np.zeros(n),
            "institution_net_buy": np.arange(n),
            "retail_net_buy": np.arange(n),
        }
    )
    prepared = add_action_labels(prepare_daily_frame(frame, config), config)
    original, _ = build_feature_tensor(prepared, config)
    prepared.loc[8, "u_institution"] = 999.0
    changed, _ = build_feature_tensor(prepared, config)
    herd_idx = config["features"]["selected"].index("herd")

    assert np.array_equal(original[8, 0, :, herd_idx], changed[8, 0, :, herd_idx])
