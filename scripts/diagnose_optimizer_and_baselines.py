"""Diagnostics for the Adam-vs-exact-solver question and the missing baselines.

Three questions:

1. How far is the Adam solution from the exact optimum of its own objective?
   Adam starts at w = 0 and takes (epochs x batches_per_epoch) steps, so the
   reachable parameter norm is bounded by roughly (steps x learning_rate).

2. Does the model beat a trivial baseline?
   persistence: predict a_{t+1} with the previous day's own action a_t.

3. Do the reward features carry information beyond that baseline?
   Nested design: persistence only / reward features only / both.

Everything reuses the CPCV splits and per-split scaling of the main pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression

from src.data.splits import build_cpcv, combine_test_folds
from src.evaluation.continuous_metrics import evaluate_continuous_actions
from src.features.scaling import (
    fit_context_scaler,
    fit_feature_scaler,
    transform_context_matrix,
    transform_feature_tensor,
)
from src.utils.config import load_configs

LAMBDA_GRID = np.concatenate([[0.0], np.logspace(-6, -1, 26)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimizer and baseline diagnostics")
    parser.add_argument("--data-config", default="configs/data_continuous.yaml")
    parser.add_argument(
        "--features-config", default="configs/features_continuous_reward3_vkospi.yaml"
    )
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument(
        "--experiment-config",
        default="configs/experiment_continuous_reward3_ctxmain_default.yaml",
    )
    parser.add_argument("--output-dir", default="runs/diagnostics_exact_vs_adam")
    return parser.parse_args()


def fit_exact(design: np.ndarray, target: np.ndarray, lambda_l1: float) -> np.ndarray:
    if lambda_l1 <= 0.0:
        model = LinearRegression(fit_intercept=False)
    else:
        model = Lasso(alpha=lambda_l1 / 2.0, fit_intercept=False, max_iter=200_000, tol=1e-10)
    model.fit(design, target)
    return np.asarray(model.coef_, dtype=float).ravel()


def select_lambda(
    design: np.ndarray,
    target: np.ndarray,
    objective: str,
    *,
    n_inner: int = 5,
    purge: int = 5,
) -> float:
    """Forward-chaining inner CV on the training block only."""
    n = len(design)
    bounds = np.linspace(0, n, n_inner + 1).astype(int)
    scores = np.zeros(len(LAMBDA_GRID))
    counts = 0
    for i in range(1, n_inner):
        train_end = bounds[i]
        val_start = min(train_end + purge, n)
        val_end = bounds[i + 1]
        if val_end - val_start < 10 or train_end < 30:
            continue
        for j, lam in enumerate(LAMBDA_GRID):
            coef = fit_exact(design[:train_end], target[:train_end], float(lam))
            pred = np.clip(design[val_start:val_end] @ coef, -1.0, 1.0)
            truth = target[val_start:val_end]
            if objective == "mse":
                scores[j] -= float(np.mean((pred - truth) ** 2))
            else:
                scores[j] += float((np.sign(truth) == np.sign(pred)).mean())
        counts += 1
    if counts == 0:
        return 0.0
    return float(LAMBDA_GRID[int(np.argmax(scores))])


def build_design(features, contexts, context_main_effect: bool) -> np.ndarray:
    blocks = [features]
    if contexts is not None:
        n_f, n_c = features.shape[1], contexts.shape[1]
        inter = np.empty((len(features), n_f * n_c))
        col = 0
        for f in range(n_f):
            for k in range(n_c):
                inter[:, col] = features[:, f] * contexts[:, k]
                col += 1
        blocks.append(inter)
        if context_main_effect:
            blocks.append(contexts)
    return np.concatenate(blocks, axis=1).astype(np.float64)


def adam_reachable_bound(investor_config: dict, train_size: int) -> dict:
    """Adam from a zero start moves each coordinate by at most ~lr per step."""
    batch_size = int(investor_config["batch_size"])
    epochs = int(investor_config["epochs"])
    lr = float(investor_config["learning_rate"])
    batches_per_epoch = int(np.ceil(train_size / batch_size))
    steps = epochs * batches_per_epoch
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "batches_per_epoch": batches_per_epoch,
        "total_steps": steps,
        "max_abs_coefficient": steps * lr,
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(config["paths"]["processed_dataset"], allow_pickle=False)
    saved_features = data["feature_names"].astype(str).tolist()
    feature_names = list(config["features"]["selected"])
    features = data["features"][..., [saved_features.index(n) for n in feature_names]]
    actions = data["actions"]
    investors = list(config["investors"])
    context_names = list(config["model"]["context_names"])
    saved_contexts = data["context_names"].astype(str).tolist()
    contexts = data["contexts"][:, [saved_contexts.index(c) for c in context_names]]

    lagged_action = np.vstack([np.zeros((1, actions.shape[1])), actions[:-1]])

    cv = build_cpcv(config)
    split_input = np.arange(len(features)).reshape(-1, 1)

    rows: list[dict] = []
    coef_rows: list[dict] = []
    bound_rows: list[dict] = []

    for split_id, (train_indices, test_folds) in enumerate(cv.split(split_input)):
        test_indices = combine_test_folds(test_folds)
        context_scaler = fit_context_scaler(contexts, train_indices)
        scaled_contexts = transform_context_matrix(contexts, context_scaler)

        for investor_idx, investor in enumerate(investors):
            override = config.get("investor_overrides", {}).get(investor, {})
            investor_config = {**config, **override}
            if split_id == 0:
                bound_rows.append(
                    {
                        "investor": investor,
                        **adam_reachable_bound(investor_config, len(train_indices)),
                    }
                )

            scaler = fit_feature_scaler(features[:, investor_idx], train_indices)
            scaled = transform_feature_tensor(features[:, investor_idx], scaler)

            reward_design = build_design(
                scaled.astype(np.float64), scaled_contexts.astype(np.float64), True
            )
            lag = lagged_action[:, investor_idx : investor_idx + 1].astype(np.float64)
            y = actions[:, investor_idx].astype(np.float64)

            variants = {
                "persistence_only": lag,
                "reward_features_only": reward_design,
                "reward_plus_persistence": np.concatenate([reward_design, lag], axis=1),
            }
            for variant, design in variants.items():
                d_tr, y_tr = design[train_indices], y[train_indices]
                d_te, y_te = design[test_indices], y[test_indices]
                for objective in ("mse", "direction"):
                    lam = select_lambda(d_tr, y_tr, objective)
                    coef = fit_exact(d_tr, y_tr, lam)
                    pred = np.clip(d_te @ coef, -1.0, 1.0)
                    metrics = evaluate_continuous_actions(y_te, pred)
                    denominator = float(np.mean((y_te - y_tr.mean()) ** 2))
                    rows.append(
                        {
                            "split": split_id,
                            "investor": investor,
                            "variant": variant,
                            "lambda_objective": objective,
                            "lambda_l1": lam,
                            "direction_accuracy": metrics["direction_accuracy"],
                            "correlation": metrics["correlation"],
                            "mae": metrics["mae"],
                            "mse": metrics["mse"],
                            "oos_r2": 1.0 - metrics["mse"] / denominator if denominator else np.nan,
                        }
                    )
                    if variant == "reward_plus_persistence" and objective == "direction":
                        names = (
                            feature_names
                            + [f"{f}:{c}" for f in feature_names for c in context_names]
                            + [f"alpha:{c}" for c in context_names]
                            + ["lagged_own_action"]
                        )
                        for name, value in zip(names, coef, strict=True):
                            coef_rows.append(
                                {
                                    "split": split_id,
                                    "investor": investor,
                                    "term": name,
                                    "weight": float(value),
                                }
                            )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "variant_metrics.csv", index=False)
    summary = (
        results.groupby(["investor", "variant", "lambda_objective"])[
            ["direction_accuracy", "correlation", "mae", "oos_r2"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.to_csv(output_dir / "variant_summary.csv")

    coefs = pd.DataFrame(coef_rows)
    coefs.to_csv(output_dir / "augmented_coefficients.csv", index=False)
    coef_summary = (
        coefs.groupby(["investor", "term"])["weight"]
        .agg(
            mean="mean",
            std="std",
            zero_rate=lambda s: float((s.abs() < 1e-12).mean()),
            positive_rate=lambda s: float((s > 0).mean()),
        )
        .round(5)
        .reset_index()
    )
    coef_summary.to_csv(output_dir / "augmented_coefficients_summary.csv", index=False)

    bounds = pd.DataFrame(bound_rows)
    bounds.to_csv(output_dir / "adam_reachable_bounds.csv", index=False)

    print(summary.to_string())
    print()
    print(bounds.to_string(index=False))


if __name__ == "__main__":
    main()
