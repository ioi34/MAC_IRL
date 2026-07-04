from __future__ import annotations

import numpy as np
import pandas as pd


def fx_level_zscore(fx: pd.Series, window: int) -> pd.Series:
    log_fx = np.log(fx)
    history = log_fx.shift(1).rolling(window=window, min_periods=window)
    history_std = history.std(ddof=0).replace(0, np.nan)
    return (log_fx - history.mean()) / history_std


def build_context_matrix(df: pd.DataFrame, config: dict) -> tuple[np.ndarray, list[str]]:
    context_names = list(config["contexts"]["selected"])
    columns = config["columns"]
    fx = pd.to_numeric(df[columns["usdkrw"]], errors="raise")
    kospi_return = pd.to_numeric(df[columns["kospi200_return"]], errors="raise")

    available: dict[str, pd.Series] = {
        "fx_return_1d": np.log(fx / fx.shift(1)),
        "fx_return_5d": np.log(fx / fx.shift(5)),
        "fx_level_z_252": fx_level_zscore(fx, window=252),
        "kospi_return_1d": kospi_return,
        "kospi_return_20d": np.log1p(kospi_return).rolling(
            window=20, min_periods=20
        ).sum(),
    }

    if columns.get("vkospi") and columns["vkospi"] in df.columns:
        vkospi = pd.to_numeric(df[columns["vkospi"]], errors="raise")
        available["vkospi_1d"] = vkospi
        available["vkospi_return_1d"] = np.log(vkospi / vkospi.shift(1))
        available["vkospi_return_5d"] = np.log(vkospi / vkospi.shift(5))

    unknown = sorted(set(context_names).difference(available))
    if unknown:
        raise KeyError(f"Unknown contexts {unknown}. Available: {sorted(available)}")

    matrix = np.column_stack([available[name].to_numpy(dtype=np.float32) for name in context_names])
    return matrix.astype(np.float32), context_names
