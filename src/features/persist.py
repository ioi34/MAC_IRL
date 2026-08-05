from __future__ import annotations

import pandas as pd

from src.data.labels import investor_trade_imbalance


def build_persist(
    df: pd.DataFrame,
    config: dict,
    investor: str,
    action: int,
) -> pd.Series:
    """Lagged own action: the investor's own prior trade imbalance.

    Uses the same definition as the modeled action (own gross-value denominator,
    ``investor_trade_imbalance``), so the feature is literally a past value of
    the action itself and the coefficient is an AR term on the action.

    The lag is configurable via ``features.params.persist.lag`` (default 1,
    preserving prior behavior). The target is ``a_{t+1}`` (label_shift: 1), so
    ``lag: 1`` yields ``a_{t-1}`` (an AR(2) gap skipping ``a_t``) while
    ``lag: 0`` yields ``a_t`` (the documented "yesterday's action" AR(1)).

    Corresponds to the "following own lag trades" component in the herding
    decomposition of Sias (2002). We do not label it herding: aggregate
    type-level flow cannot separate own-following from cross-following.
    """
    lag = int(config["features"]["params"].get("persist", {}).get("lag", 1))
    return investor_trade_imbalance(df, config, investor).shift(lag) * action
