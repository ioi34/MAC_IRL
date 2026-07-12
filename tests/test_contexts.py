import numpy as np
import pandas as pd

from src.data.contexts import (
    build_context_matrix,
    fx_level_zscore,
    liquidity_capacity,
    quarter_end_intensity,
)


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


def test_fx_level_zscore_compares_current_level_with_prior_history():
    log_fx = np.arange(260, dtype=float) * 0.01
    fx = pd.Series(np.exp(log_fx))

    result = fx_level_zscore(fx, window=252)

    expected = (log_fx[252] - log_fx[:252].mean()) / log_fx[:252].std(ddof=0)
    assert result.iloc[:252].isna().all()
    assert np.isclose(result.iloc[252], expected)


def test_future_values_do_not_change_fx_level_prefix():
    fx = pd.Series(1000.0 * np.exp(np.arange(300) * 0.001))
    original = fx_level_zscore(fx, window=252)
    changed = fx.copy()
    changed.iloc[280:] = 5000.0

    changed_result = fx_level_zscore(changed, window=252)

    np.testing.assert_allclose(original.iloc[:280], changed_result.iloc[:280], equal_nan=True)


def test_quarter_end_intensity_uses_known_calendar_distance():
    dates = pd.Series(pd.to_datetime(["2025-03-23", "2025-03-27", "2025-03-31"]))

    result = quarter_end_intensity(dates, window_days=7)

    np.testing.assert_allclose(result, [0.0, 3.0 / 7.0, 1.0])


def test_liquidity_capacity_increases_with_trading_value():
    price = pd.Series([100.0, 101.0, 102.0])
    low = liquidity_capacity(price, pd.Series([100.0, 100.0, 100.0]), window=1)
    high = liquidity_capacity(price, pd.Series([1000.0, 1000.0, 1000.0]), window=1)

    assert (high.iloc[1:] > low.iloc[1:]).all()
