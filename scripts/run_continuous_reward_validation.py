from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_continuous import (
    resolve_investor_config,
    resolve_training_investors,
    validate_processed_schema,
)
from src.features.scaling import (
    fit_context_scaler,
    fit_feature_scaler,
    transform_context_matrix,
    transform_feature_tensor,
)
from src.models.continuous import ContinuousInvestorIRLModel, build_context_mask
from src.training.continuous_trainer import train_continuous_investor_model
from src.utils.config import dump_yaml, load_configs, load_yaml
from src.utils.seed import set_seed


BEHAVIORAL_FEATURES = {"herd", "underwater"}
TRADITIONAL_FEATURES = {"momentum", "shortmom_orth", "relative"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run continuous reward weight-stability and ablation validation"
    )
    parser.add_argument("--data-config", default="configs/data_continuous.yaml")
    parser.add_argument(
        "--features-config", default="configs/features_continuous_reward5.yaml"
    )
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument(
        "--experiment-config", default="configs/experiment_continuous.yaml"
    )
    parser.add_argument(
        "--output-root", default="runs/continuous_reward_validation"
    )
    parser.add_argument(
        "--config-dir",
        default="experiments/2026-07-13/configs/continuous_reward_validation",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument(
        "--ridge-strengths",
        nargs="+",
        type=float,
        default=[
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            1e-2,
            3e-2,
            1e-1,
            3e-1,
            1.0,
            3.0,
            10.0,
            30.0,
            100.0,
        ],
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _slug_strength(value: float) -> str:
    return f"{value:.0e}".replace("+", "")


def _write_variant_configs(
    feature_template: dict,
    experiment_template: dict,
    *,
    feature_names: list[str],
    context_names: list[str],
    output_dir: Path,
    config_dir: Path,
    config_name: str,
    ridge_strength: float | None = None,
    investors: list[str],
) -> tuple[Path, Path]:
    feature_config = deepcopy(feature_template)
    feature_config["features"]["selected"] = feature_names
    experiment_config = deepcopy(experiment_template)
    experiment_config.setdefault("model", {})["context_names"] = context_names
    interactions = experiment_config["model"].get("context_interactions")
    if interactions is not None:
        experiment_config["model"]["context_interactions"] = {
            name: values for name, values in interactions.items() if name in context_names
        }
    experiment_config.setdefault("experiment", {})["name"] = config_name
    experiment_config["experiment"]["output_dir"] = str(output_dir)
    if ridge_strength is not None:
        overrides = experiment_config.setdefault("investor_overrides", {})
        for investor in investors:
            override = overrides.setdefault(investor, {})
            override.setdefault("loss", {})["lambda_l1"] = 0.0
            override["weight_decay"] = ridge_strength

    feature_path = config_dir / f"{config_name}_features.yaml"
    experiment_path = config_dir / f"{config_name}_experiment.yaml"
    dump_yaml(feature_config, feature_path)
    dump_yaml(experiment_config, experiment_path)
    return feature_path, experiment_path


def _cpcv_complete(output_dir: Path) -> bool:
    return all(
        (output_dir / filename).exists()
        for filename in (
            "cv_summary.json",
            "predictions.csv",
            "reward_weights.csv",
            "reward_weights_summary.csv",
        )
    )


def _run_cpcv(
    args: argparse.Namespace,
    feature_path: Path,
    experiment_path: Path,
    output_dir: Path,
) -> None:
    if _cpcv_complete(output_dir) and not args.force:
        print(f"skip completed CPCV: {output_dir}", flush=True)
        return
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_continuous",
            "--data-config",
            args.data_config,
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


def _run_walk_forward(
    args: argparse.Namespace,
    feature_path: Path,
    experiment_path: Path,
    output_dir: Path,
) -> None:
    required = {
        "walk_forward_metrics.csv",
        "reward_weights.csv",
        "reward_weights_summary.csv",
    }
    if all((output_dir / filename).exists() for filename in required) and not args.force:
        print(f"skip completed walk-forward: {output_dir}", flush=True)
        return
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.train_continuous_walk_forward",
            "--data-config",
            args.data_config,
            "--features-config",
            str(feature_path),
            "--model-config",
            args.model_config,
            "--train-config",
            args.train_config,
            "--experiment-config",
            str(experiment_path),
            "--output-dir",
            str(output_dir),
            "--test-years",
            "2023",
            "2024",
            "2025",
        ],
        check=True,
    )


def _weight_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group["weight"].to_numpy(dtype=float)
        mean = float(values.mean())
        dominant_sign = 1 if mean > 0 else -1 if mean < 0 else 0
        signs = np.sign(values)
        lower, upper = np.quantile(values, [0.025, 0.975])
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "n_resamples": int(group["resample"].nunique()),
                "mean": mean,
                "std": float(values.std(ddof=1)),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "positive_rate": float((signs > 0).mean()),
                "negative_rate": float((signs < 0).mean()),
                "dominant_direction": {
                    1: "positive",
                    -1: "negative",
                    0: "zero",
                }[dominant_sign],
                "direction_consistency": float((signs == dominant_sign).mean()),
                "ci_excludes_zero": bool(lower > 0 or upper < 0),
                "passes_sign_90": bool((signs == dominant_sign).mean() >= 0.9),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _run_monthly_block_bootstrap(
    args: argparse.Namespace,
    config: dict,
    output_dir: Path,
) -> None:
    raw_weights_path = output_dir / "bootstrap_reward_weights.csv"
    if raw_weights_path.exists() and not args.force:
        existing = pd.read_csv(raw_weights_path, usecols=["resample"])
        if existing["resample"].nunique() == args.bootstrap_resamples:
            print(f"skip completed bootstrap: {output_dir}", flush=True)
            return

    data = np.load(config["paths"]["processed_dataset"], allow_pickle=False)
    validate_processed_schema(data, config)
    saved_features = data["feature_names"].astype(str).tolist()
    feature_names = list(config["features"]["selected"])
    feature_indices = [saved_features.index(name) for name in feature_names]
    features = data["features"][..., feature_indices]
    actions = data["actions"]
    investors = list(config["investors"])
    training_investors = resolve_training_investors(config, investors)
    context_names = list(config["model"].get("context_names", []))
    contexts = None
    if context_names:
        saved_contexts = data["context_names"].astype(str).tolist()
        context_indices = [saved_contexts.index(name) for name in context_names]
        contexts = data["contexts"][:, context_indices]
    context_mask = build_context_mask(
        feature_names,
        context_names,
        config.get("model", {}).get("context_interactions"),
    )

    dates = pd.to_datetime(data["dates"].astype(str))
    month_values = dates.to_period("M").astype(str).to_numpy()
    months = pd.unique(month_values)
    month_indices = {month: np.flatnonzero(month_values == month) for month in months}
    rng = np.random.default_rng(int(config["seed"]) + 91_000)
    reward_rows = []
    context_rows = []
    draw_rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(config, output_dir / "config_snapshot.yaml")

    with tempfile.TemporaryDirectory(prefix="macirl_weight_bootstrap_") as temp_dir:
        checkpoint_path = Path(temp_dir) / "model.pt"
        for resample in range(args.bootstrap_resamples):
            selected_months = rng.choice(months, size=len(months), replace=True)
            sample_indices = np.concatenate(
                [month_indices[month] for month in selected_months]
            )
            for draw, month in enumerate(selected_months):
                draw_rows.append(
                    {
                        "resample": resample,
                        "draw": draw,
                        "month": month,
                        "n_observations": len(month_indices[month]),
                    }
                )

            sampled_contexts = None
            if contexts is not None:
                context_scaler = fit_context_scaler(contexts, sample_indices)
                sampled_contexts = transform_context_matrix(
                    contexts, context_scaler
                )[sample_indices]

            for investor_idx, investor in training_investors:
                seed = int(config["seed"]) + 100_000 + resample * len(investors) + investor_idx
                set_seed(seed)
                investor_config = resolve_investor_config(config, investor)
                investor_features = features[:, investor_idx]
                feature_scaler = fit_feature_scaler(investor_features, sample_indices)
                sampled_features = transform_feature_tensor(
                    investor_features, feature_scaler
                )[sample_indices]
                tensors = [torch.as_tensor(sampled_features, dtype=torch.float32)]
                if sampled_contexts is not None:
                    tensors.append(
                        torch.as_tensor(sampled_contexts, dtype=torch.float32)
                    )
                tensors.append(
                    torch.as_tensor(
                        actions[sample_indices, investor_idx], dtype=torch.float32
                    )
                )
                loader = DataLoader(
                    TensorDataset(*tensors),
                    batch_size=int(investor_config["batch_size"]),
                    shuffle=True,
                    generator=torch.Generator().manual_seed(seed),
                )
                model = ContinuousInvestorIRLModel(
                    num_features=len(feature_names),
                    num_contexts=len(context_names),
                    context_mask=context_mask,
                )
                train_continuous_investor_model(
                    model,
                    loader,
                    investor_config,
                    checkpoint_path,
                    feature_names,
                    context_names,
                )
                for feature, weight in zip(
                    feature_names,
                    model.beta.detach().cpu().tolist(),
                    strict=True,
                ):
                    reward_rows.append(
                        {
                            "resample": resample,
                            "investor": investor,
                            "feature": feature,
                            "weight": float(weight),
                        }
                    )
                if context_names:
                    values = model.effective_context_weights().detach().cpu().numpy()
                    for feature_idx, feature in enumerate(feature_names):
                        for context_idx, context in enumerate(context_names):
                            context_rows.append(
                                {
                                    "resample": resample,
                                    "investor": investor,
                                    "feature": feature,
                                    "context": context,
                                    "weight": float(values[feature_idx, context_idx]),
                                }
                            )
            if (resample + 1) % 20 == 0 or resample + 1 == args.bootstrap_resamples:
                print(
                    f"completed monthly block bootstrap {resample + 1}/"
                    f"{args.bootstrap_resamples}",
                    flush=True,
                )

    reward_weights = pd.DataFrame(reward_rows)
    reward_weights.to_csv(raw_weights_path, index=False)
    _weight_summary(
        reward_weights, ["investor", "feature"]
    ).to_csv(output_dir / "bootstrap_reward_weights_summary.csv", index=False)
    if context_rows:
        context_weights = pd.DataFrame(context_rows)
        context_weights.to_csv(
            output_dir / "bootstrap_context_weights.csv", index=False
        )
        _weight_summary(
            context_weights, ["investor", "feature", "context"]
        ).to_csv(
            output_dir / "bootstrap_context_weights_summary.csv", index=False
        )
    pd.DataFrame(draw_rows).to_csv(
        output_dir / "bootstrap_month_draws.csv", index=False
    )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    config_dir = Path(args.config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    feature_template = load_yaml(args.features_config)
    experiment_template = load_yaml(args.experiment_config)
    config = load_configs(
        args.data_config,
        args.features_config,
        args.model_config,
        args.train_config,
        args.experiment_config,
    )
    feature_names = list(config["features"]["selected"])
    context_names = list(config["model"].get("context_names", []))
    investors = list(config["investors"])

    variants: dict[str, tuple[list[str], list[str]]] = {
        "baseline": (feature_names, context_names),
    }
    for feature in feature_names:
        variants[f"remove_{feature}"] = (
            [name for name in feature_names if name != feature],
            context_names,
        )
    variants["remove_behavioral_group"] = (
        [name for name in feature_names if name not in BEHAVIORAL_FEATURES],
        context_names,
    )
    variants["remove_traditional_group"] = (
        [name for name in feature_names if name not in TRADITIONAL_FEATURES],
        context_names,
    )
    for context in context_names:
        variants[f"remove_context_{context}"] = (
            feature_names,
            [name for name in context_names if name != context],
        )

    variant_config_paths: dict[str, tuple[Path, Path]] = {}
    for name, (selected_features, selected_contexts) in variants.items():
        output_dir = output_root / "ablation" / name
        paths = _write_variant_configs(
            feature_template,
            experiment_template,
            feature_names=selected_features,
            context_names=selected_contexts,
            output_dir=output_dir,
            config_dir=config_dir,
            config_name=f"ablation_{name}",
            investors=investors,
        )
        variant_config_paths[name] = paths
        _run_cpcv(args, *paths, output_dir)

    baseline_feature_path, baseline_experiment_path = variant_config_paths["baseline"]
    _run_walk_forward(
        args,
        baseline_feature_path,
        baseline_experiment_path,
        output_root / "walk_forward",
    )

    for strength in args.ridge_strengths:
        name = f"ridge_{_slug_strength(strength)}"
        output_dir = output_root / "regularization" / name
        paths = _write_variant_configs(
            feature_template,
            experiment_template,
            feature_names=feature_names,
            context_names=context_names,
            output_dir=output_dir,
            config_dir=config_dir,
            config_name=name,
            ridge_strength=strength,
            investors=investors,
        )
        _run_cpcv(args, *paths, output_dir)

    manifest = {
        "baseline": "continuous 5-feature contextual model",
        "ablation_variants": {
            name: {
                "features": selected_features,
                "contexts": selected_contexts,
            }
            for name, (selected_features, selected_contexts) in variants.items()
        },
        "bootstrap": {
            "unit": "calendar_month",
            "resamples": args.bootstrap_resamples,
        },
        "walk_forward_test_years": [2023, 2024, 2025],
        "ridge_strengths": args.ridge_strengths,
    }
    dump_yaml(manifest, output_root / "validation_manifest.yaml")
    _run_monthly_block_bootstrap(
        args,
        config,
        output_root / "weight_bootstrap",
    )


if __name__ == "__main__":
    main()
