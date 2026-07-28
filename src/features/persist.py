from __future__ import annotations

import pandas as pd

from src.data.labels import investor_trade_imbalance


def build_persist(
    df: pd.DataFrame,
    config: dict,
    investor: str,
    action: int,
) -> pd.Series:
    """Lagged own action: the investor's own prior-day trade imbalance.

    Uses the same definition as the modeled action (own gross-value denominator,
    ``investor_trade_imbalance``), so the feature is literally "yesterday's
    action" and the coefficient is an AR(1) on the action itself.

    Corresponds to the "following own lag trades" component in the herding
    decomposition of Sias (2002). We do not label it herding: aggregate
    type-level flow cannot separate own-following from cross-following.
    """
    return investor_trade_imbalance(df, config, investor).shift(1) * action
