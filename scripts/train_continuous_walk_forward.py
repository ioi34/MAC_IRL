from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_continuous import (
    _context_weights_frame,
    _evaluate_model,
    _reward_weights_frame,
    resolve_investor_config,
    resolve_training_investors,
    validate_processed_schema,
)
from src.evaluation.interpret import summarize_context_weights, summarize_reward_weights
from src.features.scaling import (
    fit_context_scaler,
    fit_feature_scaler,
    save_feature_scaler,
    transform_context_matrix,
    transform_feature_tensor,
)
from src.models.continuous import ContinuousInvestorIRLModel, build_context_mask
from src.training.continuous_trainer import train_continuous_investor_model
from src.utils.config import dump_yaml, load_configs
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expanding walk-forward continuous evaluation")
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--features-config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-years", nargs="+", type=int, default=[2023, 2024, 2025])
    return parser.parse_args()


def extra_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    train_mean: float,
) -> dict[str, float]:
    baseline_rmse = float(np.sqrt(np.mean(np.square(actual - train_mean))))
    rmse = float(np.sqrt(np.mean(np.square(actual - predicted))))
    actual_direction = actual >= 0
    predicted_direction = predicted >= 0
    tail = np.abs(actual) >= np.quantile(np.abs(actual), 0.75)
    return {
        "nrmse": rmse / float(np.std(actual)),
        "rmse_skill": 1.0 - rmse / baseline_rmse,
        "balanced_accuracy": float(
            balanced_accuracy_score(actual_direction, predicted_direction)
        ),
        "macro_f1": float(
            f1_score(
                actual_direction,
                predicted_direction,
                average="macro",
                labels=[False, True],
                zero_division=0,
            )
        ),
        "std_ratio": float(np.std(predicted) / np.std(actual)),
        "tail_direction_accuracy": float(
            np.mean(actual_direction[tail] == predicted_direction[tail])
        ),
    }


def main() -> None:
    args = parse_args()
    config = load_configs(
        args.data_config,
        args.features_config,
        args.model_config,
        args.train_config,
        args.experiment_config,
    )
    config["experiment"]["output_dir"] = args.output_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(config, output_dir / "config_snapshot.yaml")

    data = np.load(config["paths"]["processed_dataset"], allow_pickle=False)
    validate_processed_schema(data, config)
    saved_feature_names = data["feature_names"].astype(str).tolist()
    feature_names = list(config["features"]["selected"])
    feature_indices = [saved_feature_names.index(name) for name in feature_names]
    features = data["features"][..., feature_indices]
    actions = data["actions"]
    dates = pd.to_datetime(data["dates"].astype(str))
    investors = list(config["investors"])
    training_investors = resolve_training_investors(config, investors)
    context_names = list(config["model"].get("context_names", []))
    contexts = None
    if context_names:
        saved_context_names = data["context_names"].astype(str).tolist()
        context_indices = [saved_context_names.index(name) for name in context_names]
        contexts = data["contexts"][:, context_indices]
    context_mask = build_context_mask(
        feature_names,
        context_names,
        config.get("model", {}).get("context_interactions"),
    )

    metric_rows = []
    prediction_frames = []
    weight_frames = []
    context_weight_frames = []
    purge = int(config["cross_validation"].get("purged_size", 0))
    base_seed = int(config["seed"])
    for test_year in args.test_years:
        train_indices = np.flatnonzero(dates.year < test_year)
        test_indices = np.flatnonzero(dates.year == test_year)
        if purge:
            train_indices = train_indices[:-purge]
        if len(train_indices) == 0 or len(test_indices) == 0:
            raise ValueError(f"Empty train or test window for {test_year}")
        year_dir = output_dir / f"year_{test_year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            year_dir / "indices.npz",
            train_indices=train_indices,
            test_indices=test_indices,
            train_dates=data["dates"][train_indices],
            test_dates=data["dates"][test_indices],
        )

        train_contexts = None
        test_contexts = None
        if contexts is not None:
            context_scaler = fit_context_scaler(contexts, train_indices)
            save_feature_scaler(context_scaler, year_dir / "context_scaler.joblib")
            scaled_contexts = transform_context_matrix(contexts, context_scaler)
            train_contexts = scaled_contexts[train_indices]
            test_contexts = scaled_contexts[test_indices]

        for investor_idx, investor in training_investors:
            seed = base_seed + test_year * len(investors) + investor_idx
            set_seed(seed)
            investor_config = resolve_investor_config(config, investor)
            investor_features = features[:, investor_idx]
            feature_scaler = fit_feature_scaler(investor_features, train_indices)
            save_feature_scaler(feature_scaler, year_dir / f"{investor}_scaler.joblib")
            scaled_features = transform_feature_tensor(investor_features, feature_scaler)
            tensors = [
                torch.as_tensor(scaled_features[train_indices], dtype=torch.float32)
            ]
            if train_contexts is not None:
                tensors.append(torch.as_tensor(train_contexts, dtype=torch.float32))
            tensors.append(
                torch.as_tensor(actions[train_indices, investor_idx], dtype=torch.float32)
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
                year_dir / f"{investor}.pt",
                feature_names,
                context_names,
            )
            evaluated = _evaluate_model(
                model,
                scaled_features[test_indices],
                actions[test_indices, investor_idx],
                test_contexts,
            )
            predicted = evaluated.pop("predicted")
            evaluated.pop("state_score")
            actual = actions[test_indices, investor_idx]
            metric_rows.append(
                {
                    "test_year": test_year,
                    "investor": investor,
                    "train_size": len(train_indices),
                    "test_size": len(test_indices),
                    **evaluated,
                    **extra_metrics(actual, predicted, actions[train_indices, investor_idx].mean()),
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "test_year": test_year,
                        "observation_index": test_indices,
                        "date": data["dates"][test_indices].astype(str),
                        "investor": investor,
                        "actual_action": actual,
                        "predicted_action": predicted,
                    }
                )
            )
            weight_frames.append(
                _reward_weights_frame(model, investor, feature_names, test_year)
            )
            if context_names:
                context_weight_frames.append(
                    _context_weights_frame(
                        model,
                        investor,
                        feature_names,
                        context_names,
                        test_year,
                    )
                )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    weights = pd.concat(weight_frames, ignore_index=True)
    metrics.to_csv(output_dir / "walk_forward_metrics.csv", index=False)
    predictions.to_csv(output_dir / "walk_forward_predictions.csv", index=False)
    weights.to_csv(output_dir / "reward_weights.csv", index=False)
    summarize_reward_weights(weights).to_csv(
        output_dir / "reward_weights_summary.csv",
        index=False,
    )
    if context_weight_frames:
        context_weights = pd.concat(context_weight_frames, ignore_index=True)
        context_weights.to_csv(output_dir / "context_weights.csv", index=False)
        summarize_context_weights(context_weights).to_csv(
            output_dir / "context_weights_summary.csv",
            index=False,
        )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
