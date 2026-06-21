import numpy as np
import pandas as pd
import pytest

from src.data.average_cost import calculate_average_cost_state
from src.features.underwater import build_underwater


def test_average_cost_state_updates_inventory_and_uses_trade_vwap():
    state = calculate_average_cost_state(
        buy_quantity=pd.Series([10.0, 0.0, 2.0]),
        sell_quantity=pd.Series([0.0, 4.0, 0.0]),
        buy_value=pd.Series([1000.0, 0.0, 140.0]),
        sell_value=pd.Series([0.0, 320.0, 0.0]),
        rho=1.0,
    )

    assert state["holding_proxy"].tolist() == [10.0, 6.0, 8.0]
    assert state.loc[0, "average_cost_proxy"] == pytest.approx(100.0)
    assert state.loc[1, "average_cost_proxy"] == pytest.approx(100.0)
    assert state.loc[2, "average_cost_proxy"] == pytest.approx(92.5)
    assert state.loc[1, "underwater_gap"] == pytest.approx(0.2)
    assert state.loc[2, "underwater_gap"] == pytest.approx((92.5 - 70.0) / 92.5)


def test_sales_beyond_old_inventory_leave_new_buys_at_buy_vwap():
    state = calculate_average_cost_state(
        buy_quantity=pd.Series([10.0, 5.0]),
        sell_quantity=pd.Series([0.0, 12.0]),
        buy_value=pd.Series([1000.0, 400.0]),
        sell_value=pd.Series([0.0, 1080.0]),
        rho=1.0,
    )

    assert state.loc[1, "holding_proxy"] == pytest.approx(3.0)
    assert state.loc[1, "average_cost_proxy"] == pytest.approx(80.0)


def test_underwater_feature_is_one_identified_buy_sell_contrast():
    df = pd.DataFrame({"underwater_gap_foreign": [0.0, 0.2, 0.4]})

    buy = build_underwater(df, {}, "foreign", 1)
    sell = build_underwater(df, {}, "foreign", -1)

    np.testing.assert_allclose(buy, [0.0, 0.2, 0.4])
    np.testing.assert_allclose(sell, [0.0, -0.2, -0.4])


def test_average_cost_state_rejects_value_without_quantity():
    with pytest.raises(ValueError, match="buy_value must be zero"):
        calculate_average_cost_state(
            buy_quantity=pd.Series([0.0]),
            sell_quantity=pd.Series([0.0]),
            buy_value=pd.Series([100.0]),
            sell_value=pd.Series([0.0]),
            rho=0.98,
        )
