from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Data file not found: {source}")
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    raise ValueError(f"Unsupported data format: {source.suffix}")


def load_daily_frame(config: dict) -> pd.DataFrame:
    return read_table(config["paths"]["raw_daily"])

