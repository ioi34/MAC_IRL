import numpy as np
import pandas as pd

from src.features.institution import execution_persistence, residual_return


def _config() -> dict:
    return {
        "columns": {
            "kospi200_return": "kospi200_return",
            "investor_trades": {
                "institution": {
                    "buy_value": "institution_buy_value",
                    "sell_value": "institution_sell_value",
                }
            },
        }
    }


def test_execution_persistence_uses_current_and_past_imbalances_only():
    frame = pd.DataFrame(
        {
            "institution_buy_value": [3.0, 1.0, 4.0, 2.0],
            "institution_sell_value": [1.0, 3.0, 0.0, 2.0],
        }
    )

    result = execution_persistence(frame, _config(), "institution", span=3)
    changed = frame.copy()
    changed.loc[3, "institution_buy_value"] = 100.0
    changed_result = execution_persistence(changed, _config(), "institution", span=3)

    assert result.iloc[:2].isna().all()
    np.testing.assert_allclose(result.iloc[:3], changed_result.iloc[:3], equal_nan=True)


def test_residual_return_subtracts_matched_market_horizon():
    frame = pd.DataFrame(
        {
            "price": [100.0, 102.0, 104.04],
            "kospi200_return": [0.0, 0.01, 0.01],
        }
    )

    result = residual_return(frame, _config(), window=2)

    assert np.isclose(result.iloc[2], 2 * (np.log(1.02) - np.log(1.01)))
