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
            "investor_trades": {
                investor: {
                    "buy_quantity": f"{investor}_buy_quantity",
                    "sell_quantity": f"{investor}_sell_quantity",
                    "buy_value": f"{investor}_buy_value",
                    "sell_value": f"{investor}_sell_value",
                }
                for investor in ["foreign", "institution", "retail"]
            },
        },
        "investors": ["foreign", "institution", "retail"],
        "action_values": [-2, -1, 1, 2],
        "labeling": {"rolling_window": 3, "quantiles": [0.25, 0.50, 0.75], "label_shift": 1},
        "preprocess": {
            "ticker": None,
            "drop_missing_required": True,
            "average_cost_rho": 0.98,
        },
        "features": {
            "selected": ["underwater", "herd", "momentum", "volatility"],
            "params": {
                "underwater": {},
                "herd": {"window": 2},
                "momentum": {"window": 2},
                "volatility": {"window": 3},
            },
        },
    }


def _gross_trade_columns(n: int) -> dict[str, np.ndarray]:
    columns = {}
    for offset, investor in enumerate(["foreign", "institution", "retail"]):
        buy_quantity = np.arange(n, dtype=float) + 10 + offset
        sell_quantity = np.arange(n, dtype=float) + 5 + 2 * offset
        columns[f"{investor}_buy_quantity"] = buy_quantity
        columns[f"{investor}_sell_quantity"] = sell_quantity
        columns[f"{investor}_buy_value"] = buy_quantity * np.linspace(100, 111, n)
        columns[f"{investor}_sell_value"] = sell_quantity * np.linspace(99, 110, n)
    return columns


def test_feature_tensor_has_four_actions_and_four_reward_features():
    n = 12
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n),
            "adjusted_close": np.linspace(100, 111, n),
            "high": np.linspace(101, 112, n),
            "low": np.linspace(99, 110, n),
            "trading_value": np.linspace(1000, 1110, n),
            **_gross_trade_columns(n),
        }
    )
    config = _config()
    prepared = add_action_labels(prepare_daily_frame(frame, config), config)
    features, names = build_feature_tensor(prepared, config)

    assert names == ["underwater", "herd", "momentum", "volatility"]
    assert features.shape == (n, 3, 4, 4)


def test_herd_feature_uses_only_lagged_flows():
    config = _config()
    n = 12
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n),
            "adjusted_close": np.linspace(100, 111, n),
            "high": np.linspace(101, 112, n),
            "low": np.linspace(99, 110, n),
            "trading_value": np.ones(n) * 1000,
            **_gross_trade_columns(n),
        }
    )
    prepared = add_action_labels(prepare_daily_frame(frame, config), config)
    original, _ = build_feature_tensor(prepared, config)
    prepared.loc[8, "u_institution"] = 999.0
    changed, _ = build_feature_tensor(prepared, config)
    herd_idx = config["features"]["selected"].index("herd")

    assert np.array_equal(original[8, 0, :, herd_idx], changed[8, 0, :, herd_idx])
