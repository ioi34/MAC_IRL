from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config import load_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display saved CPCV summaries")
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_configs(args.experiment_config)
    output_dir = Path(config["experiment"]["output_dir"])
    metrics = pd.read_csv(output_dir / "cv_metrics_summary.csv")
    weights = pd.read_csv(output_dir / "reward_weights_summary.csv")
    print("\nInvestor performance (CPCV mean/std)\n")
    print(metrics.to_string(index=False))
    print("\nReward weights (mean/std/direction consistency)\n")
    print(weights.to_string(index=False))


if __name__ == "__main__":
    main()
