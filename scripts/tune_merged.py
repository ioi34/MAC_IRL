"""Per-investor HP grid search with fixed context configuration.

Each investor is swept independently while the other two are held at their
current best values. Pass --contexts to specify which contexts the model uses.

Usage:
    python -m scripts.tune_merged --contexts kospi_return_1d fx_level_z_252
    python -m scripts.tune_merged --contexts kospi_return_1d
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

# Current best HP baseline (hold fixed when not being swept)
BASE = {
    "foreign":     {"epochs": 100, "batch_size": 256,  "learning_rate": 0.001,  "lambda_l1": 0.0},
    "institution": {"epochs": 15,  "batch_size": 2048, "learning_rate": 0.0005, "lambda_l1": 0.01},
    "retail":      {"epochs": 15,  "batch_size": 512,  "learning_rate": 0.0015, "lambda_l1": 0.0003},
}

SWEEPS = {
    "foreign":     {"epochs": [75, 100, 150],     "learning_rate": [0.0005, 0.001]},
    "institution": {"epochs": [10, 15, 20, 25],   "learning_rate": [0.0003, 0.0005, 0.00075]},
    "retail":      {"epochs": [10, 15, 20, 25],   "learning_rate": [0.001, 0.0015, 0.002]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-investor HP sweep with fixed contexts")
    parser.add_argument(
        "--contexts",
        nargs="+",
        default=["kospi_return_1d", "fx_level_z_252"],
        metavar="CTX",
        help="Context names to pass to the model (default: kospi_return_1d fx_level_z_252)",
    )
    parser.add_argument("--data-config",     default="configs/data_vkospi.yaml")
    parser.add_argument("--features-config", default="configs/features_vkospi.yaml")
    parser.add_argument("--model-config",    default="configs/model.yaml")
    parser.add_argument("--train-config",    default="configs/train.yaml")
    return parser.parse_args()


def build_override(target: str, epochs: int, lr: float) -> dict:
    overrides = {}
    for inv, hp in BASE.items():
        if inv == target:
            overrides[inv] = {
                "epochs": epochs,
                "batch_size": hp["batch_size"],
                "learning_rate": lr,
                "loss": {"lambda_l1": hp["lambda_l1"]},
            }
        else:
            overrides[inv] = {
                "epochs": hp["epochs"],
                "batch_size": hp["batch_size"],
                "learning_rate": hp["learning_rate"],
                "loss": {"lambda_l1": hp["lambda_l1"]},
            }
    return overrides


def vname(v: float) -> str:
    return f"{v:g}".replace(".", "p")


def main() -> None:
    args = parse_args()
    ctx_tag = "_".join(args.contexts)
    from datetime import date
    today = date.today().strftime("%Y-%m-%d")
    run_root   = Path(f"runs/tune_merged/{ctx_tag}")
    config_dir = Path(f"experiments/{today}/tune_merged_{ctx_tag}/configs")
    result_dir = Path(f"experiments/{today}/tune_merged_{ctx_tag}")
    config_dir.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    results = []
    for target, sweep in SWEEPS.items():
        for epochs, lr in product(sweep["epochs"], sweep["learning_rate"]):
            name = f"{target}_ep{epochs}_lr{vname(lr)}"
            output_dir  = run_root / name
            config_path = config_dir / f"{name}.yaml"

            config = {
                "model": {
                    "reward_type": "contextual_linear",
                    "context_names": args.contexts,
                },
                "investor_overrides": build_override(target, epochs, lr),
                "experiment": {
                    "name": name,
                    "output_dir": str(output_dir),
                    "save_checkpoints": False,
                },
            }
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

            metrics_path = output_dir / "cv_metrics_summary.csv"
            if not metrics_path.exists():
                subprocess.run(
                    [
                        sys.executable, "-m", "scripts.train",
                        "--data-config",       args.data_config,
                        "--features-config",   args.features_config,
                        "--model-config",      args.model_config,
                        "--train-config",      args.train_config,
                        "--experiment-config", str(config_path),
                    ],
                    check=True,
                )

            df = pd.read_csv(metrics_path)
            df.insert(0, "target_investor", target)
            df.insert(0, "lr", lr)
            df.insert(0, "epochs", epochs)
            df.insert(0, "name", name)
            results.append(df)
            print(f"[done] {name}")

    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(result_dir / "all_results.csv", index=False)

    acc = all_results[all_results.metric == "accuracy"].copy()
    for target in SWEEPS:
        sub = acc[(acc.target_investor == target) & (acc.investor == target)]
        best = sub.sort_values("mean", ascending=False).iloc[0]
        print(f"\n[best for {target}]  epochs={int(best.epochs)}  lr={best.lr}  acc={best['mean']:.4f}")


if __name__ == "__main__":
    main()
