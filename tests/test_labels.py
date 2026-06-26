import numpy as np
import pandas as pd

from src.data.labels import add_action_labels, investor_trade_imbalance, rolling_binary_action_labels


def test_binary_labels_use_past_window_only():
    flow = pd.Series([0.0, 1.0, 2.0, 100.0, -100.0])
    labels = rolling_binary_action_labels(flow, window=3, quantile=0.50)

    assert labels.iloc[:3].isna().all()
    assert labels.iloc[3] == 1
    assert labels.iloc[4] == -1


def test_features_at_t_target_action_at_t_plus_one():
    frame = pd.DataFrame(
        {
            "foreign_buy_value": [10.0, 20.0, 30.0, 40.0, 10.0],
            "foreign_sell_value": [20.0, 20.0, 20.0, 20.0, 30.0],
        }
    )
    config = {
        "investors": ["foreign"],
        "columns": {
            "investor_trades": {
                "foreign": {
                    "buy_value": "foreign_buy_value",
                    "sell_value": "foreign_sell_value",
                }
            }
        },
        "labeling": {"rolling_window": 2, "quantile": 0.5, "label_shift": 1},
    }
    labeled = add_action_labels(frame, config)

    assert labeled.loc[2, "action_foreign"] == 1
    assert labeled.loc[3, "action_foreign"] == -1


def test_investor_trade_imbalance_uses_investor_gross_value():
    frame = pd.DataFrame(
        {
            "foreign_buy_value": [300.0],
            "foreign_sell_value": [100.0],
        }
    )
    config = {
        "columns": {
            "investor_trades": {
                "foreign": {
                    "buy_value": "foreign_buy_value",
                    "sell_value": "foreign_sell_value",
                }
            }
        }
    }

    imbalance = investor_trade_imbalance(frame, config, "foreign")

    assert np.isclose(imbalance.iloc[0], 0.5)
