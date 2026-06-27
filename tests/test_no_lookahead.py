import numpy as np
import pandas as pd

from src.data.average_cost import calculate_average_cost_state
from src.features.scaling import fit_context_scaler, fit_feature_scaler


def test_standard_scaler_fit_uses_train_indices_only():
    features = np.ones((4, 1, 2, 1), dtype=np.float32)
    features[2:] = 100.0
    scaler = fit_feature_scaler(features, np.array([0, 1]))

    assert scaler.mean_.tolist() == [1.0]
    assert scaler.scale_.tolist() == [1.0]


def test_context_scaler_fit_uses_train_indices_only():
    contexts = np.array([[1.0, 2.0], [1.0, 2.0], [100.0, 200.0]], dtype=np.float32)
    scaler = fit_context_scaler(contexts, np.array([0, 1]))

    assert scaler.mean_.tolist() == [1.0, 2.0]
    assert scaler.scale_.tolist() == [1.0, 1.0]


def test_future_trades_do_not_change_current_average_cost_state():
    buy_quantity = pd.Series([10.0, 2.0, 3.0])
    sell_quantity = pd.Series([0.0, 1.0, 1.0])
    buy_value = pd.Series([1000.0, 180.0, 240.0])
    sell_value = pd.Series([0.0, 90.0, 80.0])
    original = calculate_average_cost_state(
        buy_quantity, sell_quantity, buy_value, sell_value, rho=0.98
    )

    buy_quantity.iloc[2] = 1000.0
    buy_value.iloc[2] = 50000.0
    changed = calculate_average_cost_state(
        buy_quantity, sell_quantity, buy_value, sell_value, rho=0.98
    )

    pd.testing.assert_frame_equal(original.iloc[:2], changed.iloc[:2])
