import numpy as np
import pandas as pd

from src.data.labels import add_action_labels, investor_trade_imbalance, rolling_quartile_action_labels


def test_quartile_labels_use_past_window_only():
    flow = pd.Series([-0.6, -0.2, 0.2, 0.6, 0.9, -0.9])
    labels = rolling_quartile_action_labels(flow, window=4)

    assert labels.iloc[:4].isna().all()
    assert labels.iloc[4] == 2
    assert labels.iloc[5] == -2


def test_features_at_t_target_action_at_t_plus_one():
    flow = np.array([-0.6, -0.2, 0.2, 0.6, 0.9, -0.9])
    frame = pd.DataFrame({"foreign_buy_value": 1 + flow, "foreign_sell_value": 1 - flow})
    config = {
        "investors": ["foreign"],
        "action_values": [-2, -1, 1, 2],
        "columns": {
            "investor_trades": {
                "foreign": {
                    "buy_value": "foreign_buy_value",
                    "sell_value": "foreign_sell_value",
                }
            }
        },
        "labeling": {"rolling_window": 4, "quantiles": [0.25, 0.5, 0.75], "label_shift": 1},
    }
    labeled = add_action_labels(frame, config)

    assert labeled.loc[3, "action_foreign"] == 2
    assert labeled.loc[4, "action_foreign"] == -2
    assert labeled.loc[3, "action_idx_foreign"] == 3
    assert labeled.loc[4, "action_idx_foreign"] == 0


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
