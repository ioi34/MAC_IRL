import numpy as np
import pandas as pd

from src.data.contexts import build_context_matrix


def _config() -> dict:
    return {
        "columns": {
            "usdkrw": "usdkrw",
            "kospi200_return": "kospi200_return",
        },
        "contexts": {
            "selected": [
                "fx_return_1d",
                "fx_return_5d",
                "kospi_return_1d",
                "kospi_return_20d",
            ]
        },
    }


def test_context_formulas():
    n = 25
    frame = pd.DataFrame(
        {
            "usdkrw": 1000.0 * np.exp(np.arange(n) * 0.01),
            "kospi200_return": np.full(n, 0.01),
        }
    )

    contexts, names = build_context_matrix(frame, _config())

    assert names == [
        "fx_return_1d",
        "fx_return_5d",
        "kospi_return_1d",
        "kospi_return_20d",
    ]
    assert np.isclose(contexts[20, 0], 0.01)
    assert np.isclose(contexts[20, 1], 0.05)
    assert np.isclose(contexts[20, 2], 0.01)
    assert np.isclose(contexts[20, 3], 20 * np.log1p(0.01))


def test_future_values_do_not_change_context_prefix():
    frame = pd.DataFrame(
        {
            "usdkrw": np.linspace(1000.0, 1030.0, 30),
            "kospi200_return": np.linspace(-0.01, 0.01, 30),
        }
    )
    original, _ = build_context_matrix(frame, _config())
    changed_frame = frame.copy()
    changed_frame.loc[25:, "usdkrw"] = 5000.0
    changed_frame.loc[25:, "kospi200_return"] = 0.5
    changed, _ = build_context_matrix(changed_frame, _config())

    np.testing.assert_allclose(original[:25], changed[:25], equal_nan=True)
