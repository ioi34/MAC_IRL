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
from src.evaluation.interpret import (
    context_weights_frame,
    reward_weights_frame,
    summarize_context_weights,
    summarize_reward_weights,
)
from src.evaluation.metrics import evaluate_logits
from src.evaluation.reporting import summarize_cv_metrics
from src.features.scaling import (
    fit_context_scaler,
    fit_feature_scaler,
    save_feature_scaler,
    transform_context_matrix,
    transform_feature_tensor,
)
from src.models.mac_irl import InvestorIRLModel
from src.training.trainer import train_investor_model
from src.utils.config import deep_merge, dump_yaml, load_configs
from src.utils.logging import configure_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train independent investor models with CPCV")
    parser.add_argument("--data-config", default="configs/data_context.yaml")
    parser.add_argument("--features-config", default="configs/features_final.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--experiment-config", default="configs/experiment_final.yaml")
    return parser.parse_args()


def _evaluate_model(
    model,
    features: np.ndarray,
    labels: np.ndarray,
    contexts: np.ndarray | None = None,
) -> dict:
    model.to("cpu")
    model.eval()
    with torch.no_grad():
        context_tensor = None
        if contexts is not None:
            context_tensor = torch.as_tensor(contexts, dtype=torch.float32)
        logits = model(
            torch.as_tensor(features, dtype=torch.float32),
            context_tensor,
        )["logits"]
    metrics = evaluate_logits(logits, torch.as_tensor(labels, dtype=torch.long))
    probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    metrics["probabilities"] = probabilities
    metrics["predicted"] = probabilities.argmax(axis=1)
    return metrics


def validate_processed_schema(data, config: dict) -> None:
    required = {"features", "labels", "dates", "action_values", "feature_names", "investors"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(
            f"Processed dataset is stale or incomplete; missing arrays: {missing}. "
            "Run `python -m scripts.prepare_data` with the current configs."
        )

    expected_features = list(config["features"]["selected"])
    saved_features = data["feature_names"].astype(str).tolist()
    expected_investors = list(config["investors"])
    saved_investors = data["investors"].astype(str).tolist()
    expected_actions = list(config["action_values"])
    saved_actions = data["action_values"].astype(int).tolist()
    if saved_features != expected_features:
        raise ValueError(
            f"Processed features {saved_features} do not match config {expected_features}. "
            "Run `python -m scripts.prepare_data` again."
        )
    if saved_investors != expected_investors:
        raise ValueError(
            f"Processed investors {saved_investors} do not match config {expected_investors}."
        )
    if saved_actions != expected_actions:
        raise ValueError(f"Processed actions {saved_actions} do not match config {expected_actions}.")

    selected_contexts = list(config.get("model", {}).get("context_names", []))
    if selected_contexts:
        context_required = {"contexts", "context_names"}
        context_missing = sorted(context_required.difference(data))
        if context_missing:
            raise ValueError(
                f"Processed dataset is missing contextual arrays: {context_missing}. "
                "Run `python -m scripts.prepare_data` with the context configs."
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


def main() -> None:
    args = parse_args()
    config = load_configs(
        args.data_config,
        args.features_config,
        args.model_config,
        args.train_config,
        args.experiment_config,
    )
    data = np.load(config["paths"]["processed_dataset"], allow_pickle=False)
    validate_processed_schema(data, config)
    features = data["features"]
    labels = data["labels"]
    dates = data["dates"]
    investors = list(config["investors"])
    feature_names = list(config["features"]["selected"])
    selected_context_names = list(config["model"].get("context_names", []))
    contexts = None
    if selected_context_names:
        saved_context_names = data["context_names"].astype(str).tolist()
        selected_context_indices = [saved_context_names.index(name) for name in selected_context_names]
        contexts = data["contexts"][:, selected_context_indices]
    cv = build_cpcv(config)
    base_seed = int(config["seed"])
    output_dir = Path(config["experiment"]["output_dir"])
    logger = configure_logging(output_dir)
    dump_yaml(config, output_dir / "config_snapshot.yaml")
    if any(f in feature_names for f in ("herd_a", "herd_b")):
        from src.features.herd import get_herd_source_map
        pd.DataFrame(get_herd_source_map(investors)).to_csv(
            output_dir / "herd_source_map.csv", index=False
        )

    metric_rows = []
    weight_frames = []
    context_weight_frames = []
    confusion_records = []
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

        for investor_idx, investor in enumerate(investors):
            seed = base_seed + split_id * len(investors) + investor_idx
            set_seed(seed)
            investor_config = resolve_investor_config(config, investor)
            investor_features = features[:, investor_idx]
            scaler = fit_feature_scaler(investor_features, train_indices)
            save_feature_scaler(scaler, split_dir / f"{investor}_scaler.joblib")
            train_features = transform_feature_tensor(investor_features[train_indices], scaler)
            test_features = transform_feature_tensor(investor_features[test_indices], scaler)
            train_tensors = [torch.as_tensor(train_features, dtype=torch.float32)]
            if train_contexts is not None:
                train_tensors.append(torch.as_tensor(train_contexts, dtype=torch.float32))
            train_tensors.append(
                torch.as_tensor(labels[train_indices, investor_idx], dtype=torch.long)
            )
            train_dataset = TensorDataset(*train_tensors)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(investor_config["batch_size"]),
                shuffle=True,
                generator=generator,
            )
            model = InvestorIRLModel(
                num_features=features.shape[-1],
                tau=float(config["model"]["tau"]),
                num_contexts=len(selected_context_names),
            )
            train_metrics = train_investor_model(
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
                labels[test_indices, investor_idx],
                test_contexts,
            )
            confusion_records.append(
                {
                    "split": split_id,
                    "investor": investor,
                    "matrix": test_metrics.pop("confusion_matrix"),
                }
            )
            probabilities = test_metrics.pop("probabilities")
            predicted = test_metrics.pop("predicted")
            prediction_data = {
                "split": split_id,
                "observation_index": test_indices,
                "date": dates[test_indices].astype(str),
                "investor": investor,
                "y_true": labels[test_indices, investor_idx],
                "y_pred": predicted,
            }
            for action_idx in range(probabilities.shape[1]):
                prediction_data[f"probability_{action_idx}"] = probabilities[:, action_idx]
            prediction_frames.append(pd.DataFrame(prediction_data))
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
            weight_frames.append(reward_weights_frame(model, investor, feature_names, split_id))
            if selected_context_names:
                context_weight_frames.append(
                    context_weights_frame(
                        model,
                        investor,
                        feature_names,
                        selected_context_names,
                        split_id,
                    )
                )

        logger.info("completed CPCV split %d/%d", split_id + 1, cv.get_n_splits())

    metrics = pd.DataFrame(metric_rows)
    weights = pd.concat(weight_frames, ignore_index=True)
    metrics_summary = summarize_cv_metrics(metrics)
    weights_summary = summarize_reward_weights(weights)
    context_weights = None
    context_weights_summary = None
    if context_weight_frames:
        context_weights = pd.concat(context_weight_frames, ignore_index=True)
        context_weights_summary = summarize_context_weights(context_weights)
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
    split_summary.to_csv(output_dir / "cv_splits.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "confusion_matrices.json").open("w", encoding="utf-8") as f:
        json.dump(confusion_records, f, ensure_ascii=False, indent=2)
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
    }
    with (output_dir / "cv_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("CPCV complete: %d independent investor fits", len(metric_rows))


if __name__ == "__main__":
    main()
