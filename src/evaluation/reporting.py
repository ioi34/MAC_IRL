from __future__ import annotations

import pandas as pd


def summarize_cv_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = ["accuracy", "macro_f1", "nll"]
    long = metrics.melt(
        id_vars=["investor"],
        value_vars=value_columns,
        var_name="metric",
        value_name="value",
    )
    return (
        long.groupby(["investor", "metric"], sort=False)["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
