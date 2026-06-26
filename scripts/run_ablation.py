from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from src.evaluation.ablation import remove_features
from src.utils.config import dump_yaml, load_configs


ABLATIONS = {
    "remove_underwater": {"underwater"},
    "remove_herd": {"herd"},
    "remove_momentum": {"momentum"},
    "remove_volatility": {"volatility"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature-removal CPCV ablations")
    parser.add_argument("--base-data-config", default="configs/data_extended.yaml")
    parser.add_argument("--base-features-config", default="configs/features_underwater.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--experiment-config", default="configs/experiment_underwater.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_configs(args.base_data_config, args.base_features_config, args.experiment_config)
    ablation_dir = Path("configs/ablations")
    ablation_dir.mkdir(parents=True, exist_ok=True)

    for name, removed in ABLATIONS.items():
        config = remove_features(base, removed)
        data_config = deepcopy({"paths": config["paths"], "columns": config["columns"]})
        data_config.update(
            {
                "investors": config["investors"],
                "action_values": config["action_values"],
                "action_names": config["action_names"],
                "labeling": config["labeling"],
                "sample": config["sample"],
                "preprocess": config["preprocess"],
            }
        )
        data_config["paths"]["processed_dataset"] = f"data/processed/{name}.npz"
        data_config["paths"]["processed_metadata"] = f"data/processed/{name}_metadata.json"
        config["experiment"]["name"] = name
        config["experiment"]["output_dir"] = f"runs/{name}"

        data_path = ablation_dir / f"{name}_data.yaml"
        feature_path = ablation_dir / f"{name}_features.yaml"
        experiment_path = ablation_dir / f"{name}_experiment.yaml"
        dump_yaml(data_config, data_path)
        dump_yaml({"features": config["features"], "scaling": config["scaling"]}, feature_path)
        dump_yaml({"experiment": config["experiment"]}, experiment_path)

        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.prepare_data",
                "--data-config",
                str(data_path),
                "--features-config",
                str(feature_path),
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.train",
                "--data-config",
                str(data_path),
                "--features-config",
                str(feature_path),
                "--model-config",
                args.model_config,
                "--train-config",
                args.train_config,
                "--experiment-config",
                str(experiment_path),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
