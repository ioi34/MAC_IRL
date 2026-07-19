from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.splits import build_cpcv, combine_test_folds
from src.evaluation.continuous_metrics import evaluate_continuous_actions
from src.evaluation.interpret import (
    summarize_context_main_weights,
    summarize_context_weights,
    summarize_reward_weights,
)
from src.features.scaling import (
    fit_context_scaler,
    fit_feature_scaler,
    save_feature_scaler,
    transform_context_matrix,
    transform_feature_tensor,
)
from src.models.continuous import ContinuousInvestorIRLModel, build_context_mask
from src.training.continuous_trainer import train_continuous_investor_model
from src.utils.config import deep_merge, dump_yaml, load_configs
from src.utils.logging import configure_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train continuous-action investor models with CPCV")
    parser.add_argument("--data-config", default="configs/data_continuous.yaml")
    parser.add_argument("--features-config", default="configs/features_continuous.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--experiment-config", default="configs/experiment_continuous.yaml")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def validate_processed_schema(data, config: dict) -> None:
    required = {"features", "actions", "dates", "feature_names", "investors"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(
            f"Processed continuous dataset is stale or incomplete; missing arrays: {missing}. "
            "Run `python -m scripts.prepare_continuous_data` with the current configs."
        )

    expected_features = list(config["features"]["selected"])
    saved_features = data["feature_names"].astype(str).tolist()
    expected_investors = list(config["investors"])
    saved_investors = data["investors"].astype(str).tolist()
    missing_features = sorted(set(expected_features).difference(saved_features))
    if missing_features:
        raise ValueError(
            f"Processed features {saved_features} do not match config {expected_features}; "
            f"missing {missing_features}. "
            "Run `python -m scripts.prepare_continuous_data` again."
        )
    if saved_investors != expected_investors:
        raise ValueError(
            f"Processed investors {saved_investors} do not match config {expected_investors}."
        )

    selected_contexts = list(config.get("model", {}).get("context_names", []))
    if selected_contexts:
        context_required = {"contexts", "context_names"}
        context_missing = sorted(context_required.difference(data))
        if context_missing:
            raise ValueError(
                f"Processed dataset is missing contextual arrays: {context_missing}. "
                "Run `python -m scripts.prepare_continuous_data` with the context configs."
            )
        saved_contexts = data["context_names"].astype(str).tolist()
        unknown_contexts = sorted(set(selected_contexts).difference(saved_contexts))
        if unknown_contexts:
            raise ValueError(
                f"Selected contexts {unknown_contexts} are absent from processed contexts "
                f"{saved_contexts}."
            )


def resolve_investor_config(config: dict, investor: str) -> dict:
    override = config.get("investor_overrides", {}).get(investor, {})
    return deep_merge(config, override)


def resolve_training_investors(config: dict, investors: list[str]) -> list[tuple[int, str]]:
    selected = list(config.get("experiment", {}).get("investors", investors))
    if len(selected) != len(set(selected)):
        raise ValueError(f"Duplicate experiment investors are not allowed: {selected}")
    unknown = sorted(set(selected).difference(investors))
    if unknown:
        raise ValueError(f"Unknown experiment investors {unknown}; available: {investors}")
    return [(investors.index(investor), investor) for investor in selected]


def _evaluate_model(
    model,
    features: np.ndarray,
    actions: np.ndarray,
    contexts: np.ndarray | None = None,
) -> dict:
    model.to("cpu")
    model.eval()
    with torch.no_grad():
        context_tensor = None
        if contexts is not None:
            context_tensor = torch.as_tensor(contexts, dtype=torch.float32)
        output = model(torch.as_tensor(features, dtype=torch.float32), context_tensor)
    predicted = output["action"].detach().cpu().numpy()
    state_score = output["state_score"].detach().cpu().numpy()
    metrics = evaluate_continuous_actions(actions, predicted)
    metrics["predicted"] = predicted
    metrics["state_score"] = state_score
    return metrics


def _reward_weights_frame(
    model,
    investor: str,
    feature_names: list[str],
    split: int,
) -> pd.DataFrame:
    beta = model.beta.detach().cpu().numpy()
    return pd.DataFrame(
        {
            "split": split,
            "investor": investor,
            "feature": feature_names,
            "weight": beta.astype(float),
        }
    )


def _context_weights_frame(
    model,
    investor: str,
    feature_names: list[str],
    context_names: list[str],
    split: int,
) -> pd.DataFrame:
    if model.context_weights is None:
        return pd.DataFrame()
    values = model.effective_context_weights().detach().cpu().numpy()
    rows = []
    for feature_idx, feature in enumerate(feature_names):
        for context_idx, context in enumerate(context_names):
            rows.append(
                {
                    "split": split,
                    "investor": investor,
                    "feature": feature,
                    "context": context,
                    "weight": float(values[feature_idx, context_idx]),
                }
            )
    return pd.DataFrame(rows)


def _context_main_weights_frame(
    model,
    investor: str,
    context_names: list[str],
    split: int,
) -> pd.DataFrame:
    if model.context_main is None:
        return pd.DataFrame()
    values = model.context_main.detach().cpu().numpy()
    return pd.DataFrame(
        {
            "split": split,
            "investor": investor,
            "context": context_names,
            "weight": values.astype(float),
        }
    )


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "mae",
        "rmse",
        "mse",
        "direction_accuracy",
        "correlation",
        "saturation_rate",
    ]
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


def main() -> None:
    args = parse_args()
    config = load_configs(
        args.data_config,
        args.features_config,
        args.model_config,
        args.train_config,
        args.experiment_config,
    )
    if args.seed is not None:
        config["seed"] = args.seed
    if args.output_dir is not None:
        config["experiment"]["output_dir"] = args.output_dir
    data = np.load(config["paths"]["processed_dataset"], allow_pickle=False)
    validate_processed_schema(data, config)
    saved_feature_names = data["feature_names"].astype(str).tolist()
    selected_feature_indices = [saved_feature_names.index(name) for name in config["features"]["selected"]]
    features = data["features"][..., selected_feature_indices]
    actions = data["actions"]
    dates = data["dates"]
    investors = list(config["investors"])
    training_investors = resolve_training_investors(config, investors)
    feature_names = list(config["features"]["selected"])
    selected_context_names = list(config["model"].get("context_names", []))
    contexts = None
    if selected_context_names:
        saved_context_names = data["context_names"].astype(str).tolist()
        selected_context_indices = [saved_context_names.index(name) for name in selected_context_names]
        contexts = data["contexts"][:, selected_context_indices]
    context_mask = build_context_mask(
        feature_names,
        selected_context_names,
        config.get("model", {}).get("context_interactions"),
    )
    context_main_effect = bool(config.get("model", {}).get("context_main_effect", False))

    cv = build_cpcv(config)
    base_seed = int(config["seed"])
    output_dir = Path(config["experiment"]["output_dir"])
    logger = configure_logging(output_dir)
    dump_yaml(config, output_dir / "config_snapshot.yaml")

    metric_rows = []
    weight_frames = []
    context_weight_frames = []
    context_main_frames = []
    prediction_frames = []
    split_records = []
    split_input = np.arange(len(features)).reshape(-1, 1)

    for split_id, (train_indices, test_folds) in enumerate(cv.split(split_input)):
        test_indices = combine_test_folds(test_folds)
        if np.intersect1d(train_indices, test_indices).size:
            raise RuntimeError(f"CPCV split {split_id} contains train/test overlap")

        split_dir = output_dir / f"split_{split_id:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)
        excluded_indices = np.setdiff1d(
            np.arange(len(features)),
            np.union1d(train_indices, test_indices),
        )
        np.savez_compressed(
            split_dir / "indices.npz",
            train_indices=train_indices,
            test_indices=test_indices,
            excluded_indices=excluded_indices,
            train_dates=dates[train_indices],
            test_dates=dates[test_indices],
        )
        split_records.append(
            {
                "split": split_id,
                "train_size": len(train_indices),
                "test_size": len(test_indices),
                "purged_embargoed_size": len(excluded_indices),
            }
        )

        train_contexts = None
        test_contexts = None
        if contexts is not None:
            context_scaler = fit_context_scaler(contexts, train_indices)
            save_feature_scaler(context_scaler, split_dir / "context_scaler.joblib")
            scaled_contexts = transform_context_matrix(contexts, context_scaler)
            train_contexts = scaled_contexts[train_indices]
            test_contexts = scaled_contexts[test_indices]

        for investor_idx, investor in training_investors:
            seed = base_seed + split_id * len(investors) + investor_idx
            set_seed(seed)
            investor_config = resolve_investor_config(config, investor)
            investor_features = features[:, investor_idx]
            scaler = fit_feature_scaler(investor_features, train_indices)
            save_feature_scaler(scaler, split_dir / f"{investor}_scaler.joblib")
            scaled_features = transform_feature_tensor(investor_features, scaler)
            train_features = scaled_features[train_indices]
            test_features = scaled_features[test_indices]
            train_tensors = [torch.as_tensor(train_features, dtype=torch.float32)]
            if train_contexts is not None:
                train_tensors.append(torch.as_tensor(train_contexts, dtype=torch.float32))
            train_tensors.append(
                torch.as_tensor(actions[train_indices, investor_idx], dtype=torch.float32)
            )
            train_dataset = TensorDataset(*train_tensors)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(investor_config["batch_size"]),
                shuffle=True,
                generator=generator,
            )
            model = ContinuousInvestorIRLModel(
                num_features=features.shape[-1],
                num_contexts=len(selected_context_names),
                context_mask=context_mask,
                context_main_effect=context_main_effect,
            )
            train_metrics = train_continuous_investor_model(
                model,
                train_loader,
                investor_config,
                split_dir / f"{investor}.pt",
                feature_names,
                selected_context_names,
            )
            test_metrics = _evaluate_model(
                model,
                test_features,
                actions[test_indices, investor_idx],
                test_contexts,
            )
            predicted = test_metrics.pop("predicted")
            state_score = test_metrics.pop("state_score")
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "split": split_id,
                        "observation_index": test_indices,
                        "date": dates[test_indices].astype(str),
                        "investor": investor,
                        "actual_action": actions[test_indices, investor_idx],
                        "predicted_action": predicted,
                        "state_score": state_score,
                    }
                )
            )
            metric_rows.append(
                {
                    "split": split_id,
                    "investor": investor,
                    "train_size": len(train_indices),
                    "test_size": len(test_indices),
                    **train_metrics,
                    **test_metrics,
                }
            )
            weight_frames.append(_reward_weights_frame(model, investor, feature_names, split_id))
            if selected_context_names:
                context_weight_frames.append(
                    _context_weights_frame(
                        model,
                        investor,
                        feature_names,
                        selected_context_names,
                        split_id,
                    )
                )
            if context_main_effect:
                context_main_frames.append(
                    _context_main_weights_frame(
                        model,
                        investor,
                        selected_context_names,
                        split_id,
                    )
                )

        logger.info("completed continuous CPCV split %d/%d", split_id + 1, cv.get_n_splits())

    metrics = pd.DataFrame(metric_rows)
    weights = pd.concat(weight_frames, ignore_index=True)
    metrics_summary = _summarize_metrics(metrics)
    weights_summary = summarize_reward_weights(weights)
    context_weights = None
    context_weights_summary = None
    if context_weight_frames:
        context_weights = pd.concat(context_weight_frames, ignore_index=True)
        context_weights_summary = summarize_context_weights(context_weights)
    context_main_weights = None
    context_main_weights_summary = None
    if context_main_frames:
        context_main_weights = pd.concat(context_main_frames, ignore_index=True)
        context_main_weights_summary = summarize_context_main_weights(context_main_weights)
    split_summary = pd.DataFrame(split_records)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    metrics.to_csv(output_dir / "cv_metrics.csv", index=False)
    metrics_summary.to_csv(output_dir / "cv_metrics_summary.csv", index=False)
    weights.to_csv(output_dir / "reward_weights.csv", index=False)
    weights_summary.to_csv(output_dir / "reward_weights_summary.csv", index=False)
    if context_weights is not None and context_weights_summary is not None:
        context_weights.to_csv(output_dir / "context_weights.csv", index=False)
        context_weights_summary.to_csv(
            output_dir / "context_weights_summary.csv", index=False
        )
    if context_main_weights is not None and context_main_weights_summary is not None:
        context_main_weights.to_csv(output_dir / "context_main_weights.csv", index=False)
        context_main_weights_summary.to_csv(
            output_dir / "context_main_weights_summary.csv", index=False
        )
    split_summary.to_csv(output_dir / "cv_splits.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary = {
        "n_observations": len(features),
        "n_splits": int(cv.get_n_splits()),
        "n_test_paths": int(cv.n_test_paths),
        "versions": {
            "skfolio": version("skfolio"),
            "scikit-learn": version("scikit-learn"),
            "torch": version("torch"),
        },
        "metrics": metrics_summary.to_dict(orient="records"),
        "reward_weights": weights_summary.to_dict(orient="records"),
        "context_weights": (
            context_weights_summary.to_dict(orient="records")
            if context_weights_summary is not None
            else []
        ),
        "context_main_weights": (
            context_main_weights_summary.to_dict(orient="records")
            if context_main_weights_summary is not None
            else []
        ),
    }
    with (output_dir / "cv_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("continuous CPCV complete: %d independent investor fits", len(metric_rows))


if __name__ == "__main__":
    main()
