import numpy as np
import pandas as pd

from src.features.fx import build_fx_sensitivity_1d


def test_fx_sensitivity_uses_negative_usdkrw_return_times_action():
    frame = pd.DataFrame({"usdkrw": [1000.0, 1010.0, 1000.0]})
    config = {"columns": {"usdkrw": "usdkrw"}}

    buy = build_fx_sensitivity_1d(frame, config, "foreign", action=1)
    sell = build_fx_sensitivity_1d(frame, config, "foreign", action=-1)

    assert np.isclose(buy.iloc[1], -np.log(1.01))
    np.testing.assert_allclose(buy.iloc[1:], -sell.iloc[1:])
