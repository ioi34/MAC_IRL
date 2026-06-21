import pandas as pd

from src.data.labels import add_action_labels, rolling_binary_action_labels


def test_binary_labels_use_past_window_only():
    flow = pd.Series([0.0, 1.0, 2.0, 100.0, -100.0])
    labels = rolling_binary_action_labels(flow, window=3, quantile=0.50)

    assert labels.iloc[:3].isna().all()
    assert labels.iloc[3] == 1
    assert labels.iloc[4] == -1


def test_features_at_t_target_action_at_t_plus_one():
    frame = pd.DataFrame({"u_foreign": [0.0, 1.0, 2.0, 3.0, -1.0]})
    config = {
        "investors": ["foreign"],
        "labeling": {"rolling_window": 2, "quantile": 0.5, "label_shift": 1},
    }
    labeled = add_action_labels(frame, config)

    assert labeled.loc[2, "action_foreign"] == 1
    assert labeled.loc[3, "action_foreign"] == -1
