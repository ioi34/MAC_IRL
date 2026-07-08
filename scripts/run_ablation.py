from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from src.evaluation.ablation import build_ablation_variants, remove_features
from src.utils.config import dump_yaml, load_configs


GROUP_ABLATIONS = {
    "remove_behavioral_group": {"underwater", "herd_a", "herd_b"},
    "remove_traditional_group": {"momentum", "relative"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run merged_e2 feature-removal CPCV ablations")
    parser.add_argument("--base-data-config", default="configs/data_vkospi.yaml")
    parser.add_argument("--base-features-config", default="configs/features_vkospi.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--experiment-config", default="configs/experiment_merged_e2.yaml")
    parser.add_argument(
        "--config-dir",
        default="experiments/2026-07-06/configs/merged_e2_ablation",
    )
    parser.add_argument("--output-root", default="runs/ablation_merged_e2")
    return parser.parse_args()


def _variant_configs(
    base: dict,
    name: str,
    removed: set[str],
    output_root: Path,
) -> tuple[dict, dict, dict]:
    config = remove_features(base, removed)
    data_config = deepcopy(
        {
            "paths": config["paths"],
            "columns": config["columns"],
            "investors": config["investors"],
            "action_values": config["action_values"],
            "action_names": config["action_names"],
            "labeling": config["labeling"],
            "sample": config["sample"],
            "preprocess": config["preprocess"],
        }
    )
    data_config["paths"]["processed_dataset"] = str(
        Path("data/processed/ablation_merged_e2") / f"{name}.npz"
    )
    data_config["paths"]["processed_metadata"] = str(
        Path("data/processed/ablation_merged_e2") / f"{name}_metadata.json"
    )
    feature_config = {
        "paths": deepcopy(data_config["paths"]),
        "features": deepcopy(config["features"]),
        "contexts": deepcopy(config["contexts"]),
        "scaling": deepcopy(config["scaling"]),
    }
    experiment_config = {
        "model": deepcopy(config["model"]),
        "investor_overrides": deepcopy(config.get("investor_overrides", {})),
        "experiment": deepcopy(config["experiment"]),
    }
    experiment_config["experiment"]["name"] = f"ablation_merged_e2_{name}"
    experiment_config["experiment"]["output_dir"] = str(output_root / name)
    return data_config, feature_config, experiment_config


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    base = load_configs(
        args.base_data_config,
        args.base_features_config,
        args.model_config,
        args.train_config,
        args.experiment_config,
    )
    feature_names = list(base["features"]["selected"])
    variants = {"baseline": set()}
    variants.update(build_ablation_variants(feature_names, GROUP_ABLATIONS))
    config_dir = Path(args.config_dir)
    output_root = Path(args.output_root)
    config_dir.mkdir(parents=True, exist_ok=True)

    for name, removed in variants.items():
        data_config, feature_config, experiment_config = _variant_configs(
            base, name, removed, output_root
        )
        data_path = config_dir / f"{name}_data.yaml"
        feature_path = config_dir / f"{name}_features.yaml"
        experiment_path = config_dir / f"{name}_experiment.yaml"
        dump_yaml(data_config, data_path)
        dump_yaml(feature_config, feature_path)
        dump_yaml(experiment_config, experiment_path)

        _run(
            [
                sys.executable,
                "-m",
                "scripts.prepare_data",
                "--data-config",
                str(data_path),
                "--features-config",
                str(feature_path),
            ]
        )
        _run(
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
            ]
        )


if __name__ == "__main__":
    main()
