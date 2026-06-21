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
from src.evaluation.interpret import reward_weights_frame, summarize_reward_weights
from src.evaluation.metrics import evaluate_logits
from src.evaluation.reporting import summarize_cv_metrics
from src.features.scaling import fit_feature_scaler, save_feature_scaler, transform_feature_tensor
from src.models.mac_irl import InvestorIRLModel
from src.training.trainer import train_investor_model
from src.utils.config import dump_yaml, load_configs
from src.utils.logging import configure_logging
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train independent investor models with CPCV")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--features-config", default="configs/features.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    return parser.parse_args()


def _evaluate_model(model, features: np.ndarray, labels: np.ndarray) -> dict:
    model.to("cpu")
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(features, dtype=torch.float32))["logits"]
    return evaluate_logits(logits, torch.as_tensor(labels, dtype=torch.long))


def main() -> None:
    args = parse_args()
    config = load_configs(
        args.data_config,
        args.features_config,
        args.model_config,
        args.train_config,
        args.experiment_config,
    )
    output_dir = Path(config["experiment"]["output_dir"])
    logger = configure_logging(output_dir)
    dump_yaml(config, output_dir / "config_snapshot.yaml")

    data = np.load(config["paths"]["processed_dataset"], allow_pickle=False)
    features = data["features"]
    labels = data["labels"]
    dates = data["dates"]
    investors = list(config["investors"])
    feature_names = list(config["features"]["selected"])
    cv = build_cpcv(config)
    base_seed = int(config["seed"])

    metric_rows = []
    weight_frames = []
    confusion_records = []
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

        for investor_idx, investor in enumerate(investors):
            seed = base_seed + split_id * len(investors) + investor_idx
            set_seed(seed)
            investor_features = features[:, investor_idx]
            scaler = fit_feature_scaler(investor_features, train_indices)
            save_feature_scaler(scaler, split_dir / f"{investor}_scaler.joblib")
            train_features = transform_feature_tensor(investor_features[train_indices], scaler)
            test_features = transform_feature_tensor(investor_features[test_indices], scaler)
            train_dataset = TensorDataset(
                torch.as_tensor(train_features, dtype=torch.float32),
                torch.as_tensor(labels[train_indices, investor_idx], dtype=torch.long),
            )
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=int(config["batch_size"]),
                shuffle=True,
                generator=generator,
            )
            model = InvestorIRLModel(
                num_features=features.shape[-1],
                tau=float(config["model"]["tau"]),
            )
            train_metrics = train_investor_model(
                model,
                train_loader,
                config,
                split_dir / f"{investor}.pt",
            )
            test_metrics = _evaluate_model(
                model,
                test_features,
                labels[test_indices, investor_idx],
            )
            confusion_records.append(
                {
                    "split": split_id,
                    "investor": investor,
                    "matrix": test_metrics.pop("confusion_matrix"),
                }
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
            weight_frames.append(reward_weights_frame(model, investor, feature_names, split_id))

        logger.info("completed CPCV split %d/%d", split_id + 1, cv.get_n_splits())

    metrics = pd.DataFrame(metric_rows)
    weights = pd.concat(weight_frames, ignore_index=True)
    metrics_summary = summarize_cv_metrics(metrics)
    weights_summary = summarize_reward_weights(weights)
    split_summary = pd.DataFrame(split_records)

    metrics.to_csv(output_dir / "cv_metrics.csv", index=False)
    metrics_summary.to_csv(output_dir / "cv_metrics_summary.csv", index=False)
    weights.to_csv(output_dir / "reward_weights.csv", index=False)
    weights_summary.to_csv(output_dir / "reward_weights_summary.csv", index=False)
    split_summary.to_csv(output_dir / "cv_splits.csv", index=False)
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
    }
    with (output_dir / "cv_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("CPCV complete: %d independent investor fits", len(metric_rows))


if __name__ == "__main__":
    main()
