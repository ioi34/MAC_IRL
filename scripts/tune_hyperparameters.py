from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

import pandas as pd
import yaml


GRIDS = {
    "coarse": {
        "batch_sizes": [256],
        "learning_rates": [0.001, 0.003, 0.01, 0.03],
        "l1_values": [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01],
        "epochs": [50],
    },
    "schedule": {
        "batch_sizes": [256],
        "learning_rates": [0.0005, 0.00075, 0.001, 0.0015, 0.002],
        "l1_values": [0.0003],
        "epochs": [25, 50, 75, 100],
    },
    "long_l1": {
        "batch_sizes": [256],
        "learning_rates": [0.001],
        "l1_values": [0.0, 0.0001, 0.001, 0.003, 0.01],
        "epochs": [100],
    },
    "short_l1": {
        "batch_sizes": [256],
        "learning_rates": [0.0005, 0.001],
        "l1_values": [0.0, 0.0001, 0.001, 0.003, 0.01],
        "epochs": [25],
    },
    "long_batch": {
        "batch_sizes": [64, 128, 256, 512, 2048],
        "learning_rates": [0.001],
        "l1_values": [0.0],
        "epochs": [100],
    },
    "short_batch": {
        "batch_sizes": [64, 128, 256, 512, 2048],
        "learning_rates": [0.0005, 0.001],
        "l1_values": [0.0003, 0.01],
        "epochs": [25],
    },
    "institution_precision": {
        "batch_sizes": [2048],
        "learning_rates": [0.0003, 0.0005, 0.00075, 0.001, 0.00125],
        "l1_values": [0.01],
        "epochs": [15, 20, 25, 30, 35],
    },
    "retail_precision": {
        "batch_sizes": [512],
        "learning_rates": [0.0005, 0.00075, 0.001, 0.00125, 0.0015],
        "l1_values": [0.0003],
        "epochs": [15, 20, 25, 30, 35],
    },
    "institution_micro": {
        "batch_sizes": [2048],
        "learning_rates": [0.0004, 0.0005, 0.0006],
        "l1_values": [0.01],
        "epochs": [5, 8, 10, 12, 15, 18],
    },
    "retail_micro": {
        "batch_sizes": [512],
        "learning_rates": [0.0013, 0.0015, 0.0017, 0.002],
        "l1_values": [0.0003],
        "epochs": [8, 10, 12, 15, 18],
    },
}

EXPERIMENT_DIR = Path("experiments/2026-07-01/hyperparameter_tuning")
RUN_ROOT = Path("runs/hyperparameter_tuning")


def value_name(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=GRIDS, default="coarse")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = GRIDS[args.stage]
    stage_dir = EXPERIMENT_DIR if args.stage == "coarse" else EXPERIMENT_DIR / args.stage
    config_dir = stage_dir / "configs"
    run_root = RUN_ROOT if args.stage == "coarse" else RUN_ROOT / args.stage
    config_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    result_frames = []

    combinations = product(
        grid["epochs"],
        grid["batch_sizes"],
        grid["learning_rates"],
        grid["l1_values"],
    )
    for epochs, batch_size, learning_rate, lambda_l1 in combinations:
        name = (
            f"ep_{epochs}_bs_{batch_size}_lr_{value_name(learning_rate)}_"
            f"l1_{value_name(lambda_l1)}"
        )
        if args.stage == "coarse":
            name = f"lr_{value_name(learning_rate)}_l1_{value_name(lambda_l1)}"
        output_dir = run_root / name
        config_path = config_dir / f"{name}.yaml"
        override = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss": {"lambda_l1": lambda_l1},
            "experiment": {
                "name": name,
                "output_dir": str(output_dir),
                "save_checkpoints": True,
            },
            "model": {
                "reward_type": "contextual_linear",
                "context_names": ["kospi_return_1d"],
            },
        }
        config_path.write_text(
            yaml.safe_dump(override, sort_keys=False),
            encoding="utf-8",
        )

        metrics_path = output_dir / "cv_metrics.csv"
        if not metrics_path.exists():
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.train",
                    "--data-config",
                    "configs/data_vkospi.yaml",
                    "--features-config",
                    "configs/features_vkospi.yaml",
                    "--model-config",
                    "configs/model.yaml",
                    "--train-config",
                    "configs/train.yaml",
                    "--experiment-config",
                    str(config_path),
                ],
                check=True,
            )

        metrics = pd.read_csv(metrics_path)
        metrics.insert(0, "lambda_l1", lambda_l1)
        metrics.insert(0, "learning_rate", learning_rate)
        metrics.insert(0, "batch_size", batch_size)
        metrics.insert(0, "epochs", epochs)
        metrics.insert(0, "experiment", name)
        result_frames.append(metrics)

    all_results = pd.concat(result_frames, ignore_index=True)
    all_results.to_csv(stage_dir / "all_cv_metrics.csv", index=False)

    summary = (
        all_results.groupby(
            [
                "experiment",
                "epochs",
                "batch_size",
                "learning_rate",
                "lambda_l1",
                "investor",
            ],
            as_index=False,
        )
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            nll_mean=("nll", "mean"),
            nll_std=("nll", "std"),
        )
    )
    summary["accuracy_rank"] = summary.groupby("investor")["accuracy_mean"].rank(
        method="min",
        ascending=False,
    )
    summary["nll_rank"] = summary.groupby("investor")["nll_mean"].rank(
        method="min",
        ascending=True,
    )
    summary = summary.sort_values(
        ["investor", "accuracy_rank", "nll_rank"],
        ignore_index=True,
    )
    summary.to_csv(stage_dir / "tuning_summary.csv", index=False)

    best = summary.groupby("investor", sort=False).head(1)
    best.to_csv(stage_dir / "best_by_investor.csv", index=False)
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
