import numpy as np
import pandas as pd

from src.features.usd_momentum import build_usd_momentum, usd_momentum


def _config() -> dict:
    return {
        "columns": {"usdkrw": "usdkrw"},
        "features": {"params": {"usd_momentum": {"window": 2}}},
    }


def test_usd_momentum_subtracts_fx_return_from_krw_stock_return():
    frame = pd.DataFrame(
        {
            "price": [100.0, 110.0, 120.0],
            "usdkrw": [1000.0, 1000.0, 1100.0],
        }
    )

    result = usd_momentum(frame, _config(), window=2)

    expected = np.log(120.0 / 100.0) - np.log(1100.0 / 1000.0)
    assert np.isclose(result.iloc[2], expected)


def test_usd_momentum_buy_and_sell_features_are_opposites():
    frame = pd.DataFrame(
        {
            "price": [100.0, 110.0, 120.0],
            "usdkrw": [1000.0, 1000.0, 1100.0],
        }
    )

    buy = build_usd_momentum(frame, _config(), "foreign", action=1)
    sell = build_usd_momentum(frame, _config(), "foreign", action=-1)

    np.testing.assert_allclose(buy.iloc[2:], -sell.iloc[2:])


def test_future_values_do_not_change_usd_momentum_prefix():
    frame = pd.DataFrame(
        {
            "price": np.linspace(100.0, 130.0, 30),
            "usdkrw": np.linspace(1000.0, 1030.0, 30),
        }
    )
    original = usd_momentum(frame, _config(), window=2)
    changed = frame.copy()
    changed.loc[25:, "price"] = 500.0
    changed.loc[25:, "usdkrw"] = 2000.0

    changed_result = usd_momentum(changed, _config(), window=2)

    np.testing.assert_allclose(original.iloc[:25], changed_result.iloc[:25], equal_nan=True)
