"""Exact-solver replacement for the Adam-trained continuous IRL model.

The training objective in `src/training/continuous_trainer.py` is

    (1/n) ||a - Xw||^2 + lambda * ||w||_1

with w = (beta, B, alpha) and no intercept. Because the saturation rate is 0 in
every split, clip(q, -1, 1) == q on the whole sample, so this objective is a
LASSO problem: convex, with a unique global optimum and an exact coordinate
descent solver. Adam only approximates that optimum, and the approximation
depends on epochs / learning rate / batch size / seed.

This script solves the same objective exactly and reports the same artifacts as
`scripts/train_continuous.py`, plus naive baselines.

Design matrix column order (matches the model's parameter layout):
    [ x_1..x_F, (x_f * c_k) for f, k, c_1..c_K ]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression

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
    transform_context_matrix,
    transform_feature_tensor,
)
from src.utils.config import deep_merge, dump_yaml, load_configs


def build_context_mask(feature_names, context_names, interactions):
    """numpy re-implementation of src.models.continuous.build_context_mask."""
    if interactions is None:
        return None
    feature_to_index = {name: i for i, name in enumerate(feature_names)}
    context_to_index = {name: i for i, name in enumerate(context_names)}
    if sorted(interactions) != sorted(context_names):
        raise ValueError("context_interactions must define exactly the selected contexts")
    mask = np.zeros((len(feature_names), len(context_names)), dtype=float)
    for context, feats in interactions.items():
        for feature in feats:
            mask[feature_to_index[feature], context_to_index[context]] = 1.0
    return mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the continuous IRL objective with an exact LASSO solver"
    )
    parser.add_argument("--data-config", default="configs/data_continuous.yaml")
    parser.add_argument("--features-config", default="configs/features_continuous_reward3_vkospi.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument(
        "--experiment-config",
        default="configs/experiment_continuous_reward3_ctxmain_default.yaml",
    )
    parser.add_argument(
        "--lambda-mode",
        choices=["config", "inner-cv", "inner-cv-direction"],
        default="config",
        help=(
            "config: reuse the per-investor lambda_l1 from the experiment config. "
            "inner-cv: select lambda per split on a purged inner split of the training data."
        ),
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


def build_design(
    features: np.ndarray,
    contexts: np.ndarray | None,
    context_mask: np.ndarray | None,
    context_main_effect: bool,
) -> np.ndarray:
    """[x, x (x) c, c] with the interaction mask applied."""
    blocks = [features]
    if contexts is not None:
        n_features = features.shape[1]
        n_contexts = contexts.shape[1]
        interactions = np.empty((len(features), n_features * n_contexts), dtype=np.float64)
        col = 0
        for f in range(n_features):
            for k in range(n_contexts):
                active = 1.0 if context_mask is None else float(context_mask[f, k])
                interactions[:, col] = features[:, f] * contexts[:, k] * active
                col += 1
        blocks.append(interactions)
        if context_main_effect:
            blocks.append(contexts)
    return np.concatenate(blocks, axis=1).astype(np.float64)


def fit_exact(design: np.ndarray, target: np.ndarray, lambda_l1: float) -> np.ndarray:
    """Minimise (1/n)||y - Xw||^2 + lambda*||w||_1 with no intercept.

    sklearn's Lasso objective is (1/(2n))||y - Xw||^2 + alpha*||w||_1,
    hence alpha = lambda / 2.
    """
    if lambda_l1 <= 0.0:
        model = LinearRegression(fit_intercept=False)
    else:
        model = Lasso(
            alpha=lambda_l1 / 2.0,
            fit_intercept=False,
            max_iter=200_000,
            tol=1e-10,
        )
    model.fit(design, target)
    return np.asarray(model.coef_, dtype=float).ravel()


LAMBDA_GRID = np.concatenate([[0.0], np.logspace(-6, -1, 26)])


def select_lambda_inner_cv(
    design: np.ndarray,
    target: np.ndarray,
    *,
    objective: str = "mse",
    n_inner: int = 5,
    purge: int = 5,
) -> float:
    """Forward-chaining inner CV on the training block only (no test leakage)."""
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
        d_tr, y_tr = design[:train_end], target[:train_end]
        d_va, y_va = design[val_start:val_end], target[val_start:val_end]
        for j, lam in enumerate(LAMBDA_GRID):
            coef = fit_exact(d_tr, y_tr, float(lam))
            pred = np.clip(d_va @ coef, -1.0, 1.0)
            if objective == "mse":
                scores[j] -= float(np.mean((pred - y_va) ** 2))
            else:
                scores[j] += float((np.sign(y_va) == np.sign(pred)).mean())
        counts += 1
    if counts == 0:
        return 0.0
    return float(LAMBDA_GRID[int(np.argmax(scores))])


def unpack(
    coef: np.ndarray,
    n_features: int,
    n_contexts: int,
    context_main_effect: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    beta = coef[:n_features]
    context_weights = None
    context_main = None
    cursor = n_features
    if n_contexts:
        size = n_features * n_contexts
        context_weights = coef[cursor : cursor + size].reshape(n_features, n_contexts)
        cursor += size
        if context_main_effect:
            context_main = coef[cursor : cursor + n_contexts]
    return beta, context_weights, context_main


def baseline_predictions(actions: np.ndarray, investor_idx: int) -> dict[str, np.ndarray]:
    """Naive references evaluated on the same observation grid.

    persistence: predict a_{t+1} with the previous observed action (a_t).
    The processed dataset is already aligned so that row t holds the state at t
    and the action at t+1, hence the previous row's action is a_t.
    """
    series = actions[:, investor_idx]
    previous = np.concatenate([[0.0], series[:-1]])
    return {"persistence": previous}


def main() -> None:
    args = parse_args()
    config = load_configs(
        args.data_config,
        args.features_config,
        args.model_config,
        args.train_config,
        args.experiment_config,
    )
    output_dir = Path(
        args.output_dir or f"{config['experiment']['output_dir']}_exact_{args.lambda_mode}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(config["paths"]["processed_dataset"], allow_pickle=False)
    saved_feature_names = data["feature_names"].astype(str).tolist()
    feature_names = list(config["features"]["selected"])
    feature_indices = [saved_feature_names.index(name) for name in feature_names]
    features = data["features"][..., feature_indices]
    actions = data["actions"]
    dates = data["dates"]
    investors = list(config["investors"])

    context_names = list(config["model"].get("context_names", []))
    contexts = None
    if context_names:
        saved_contexts = data["context_names"].astype(str).tolist()
        contexts = data["contexts"][:, [saved_contexts.index(c) for c in context_names]]
    context_mask = build_context_mask(
        feature_names, context_names, config.get("model", {}).get("context_interactions")
    )
    context_main_effect = bool(config.get("model", {}).get("context_main_effect", False))

    cv = build_cpcv(config)
    split_input = np.arange(len(features)).reshape(-1, 1)

    metric_rows: list[dict] = []
    weight_rows: list[dict] = []
    context_weight_rows: list[dict] = []
    context_main_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    lambda_rows: list[dict] = []

    for split_id, (train_indices, test_folds) in enumerate(cv.split(split_input)):
        test_indices = combine_test_folds(test_folds)
        train_contexts = test_contexts = None
        if contexts is not None:
            context_scaler = fit_context_scaler(contexts, train_indices)
            scaled_contexts = transform_context_matrix(contexts, context_scaler)
            train_contexts = scaled_contexts[train_indices]
            test_contexts = scaled_contexts[test_indices]

        for investor_idx, investor in enumerate(investors):
            investor_config = deep_merge(
                config, config.get("investor_overrides", {}).get(investor, {})
            )
            investor_features = features[:, investor_idx]
            scaler = fit_feature_scaler(investor_features, train_indices)
            scaled_features = transform_feature_tensor(investor_features, scaler)

            design_train = build_design(
                scaled_features[train_indices].astype(np.float64),
                None if train_contexts is None else train_contexts.astype(np.float64),
                context_mask,
                context_main_effect,
            )
            design_test = build_design(
                scaled_features[test_indices].astype(np.float64),
                None if test_contexts is None else test_contexts.astype(np.float64),
                context_mask,
                context_main_effect,
            )
            y_train = actions[train_indices, investor_idx].astype(np.float64)
            y_test = actions[test_indices, investor_idx].astype(np.float64)

            if args.lambda_mode.startswith("inner-cv"):
                objective = "direction" if args.lambda_mode.endswith("direction") else "mse"
                lambda_l1 = select_lambda_inner_cv(design_train, y_train, objective=objective)
            else:
                lambda_l1 = float(investor_config["loss"]["lambda_l1"])
            lambda_rows.append(
                {"split": split_id, "investor": investor, "lambda_l1": lambda_l1}
            )

            coef = fit_exact(design_train, y_train, lambda_l1)
            state_score = design_test @ coef
            predicted = np.clip(state_score, -1.0, 1.0)
            test_metrics = evaluate_continuous_actions(y_test, predicted)

            train_pred = np.clip(design_train @ coef, -1.0, 1.0)
            train_mse = float(np.mean((train_pred - y_train) ** 2))

            beta, context_weights, context_main = unpack(
                coef, len(feature_names), len(context_names), context_main_effect
            )
            for name, value in zip(feature_names, beta, strict=True):
                weight_rows.append(
                    {
                        "split": split_id,
                        "investor": investor,
                        "feature": name,
                        "weight": float(value),
                    }
                )
            if context_weights is not None:
                for f_idx, f_name in enumerate(feature_names):
                    for c_idx, c_name in enumerate(context_names):
                        context_weight_rows.append(
                            {
                                "split": split_id,
                                "investor": investor,
                                "feature": f_name,
                                "context": c_name,
                                "weight": float(context_weights[f_idx, c_idx]),
                            }
                        )
            if context_main is not None:
                for c_name, value in zip(context_names, context_main, strict=True):
                    context_main_rows.append(
                        {
                            "split": split_id,
                            "investor": investor,
                            "context": c_name,
                            "weight": float(value),
                        }
                    )

            row = {
                "split": split_id,
                "investor": investor,
                "train_size": len(train_indices),
                "test_size": len(test_indices),
                "lambda_l1": lambda_l1,
                "train_mse": train_mse,
                "train_rmse": train_mse**0.5,
                **test_metrics,
            }
            for base_name, base_series in baseline_predictions(actions, investor_idx).items():
                base_pred = base_series[test_indices]
                base_metrics = evaluate_continuous_actions(y_test, base_pred)
                row[f"{base_name}_direction_accuracy"] = base_metrics["direction_accuracy"]
                row[f"{base_name}_correlation"] = base_metrics["correlation"]
                row[f"{base_name}_mae"] = base_metrics["mae"]
            row["mean_direction_accuracy"] = float(
                (np.sign(y_test) == np.sign(y_train.mean())).mean()
            )
            denominator = float(np.mean((y_test - y_train.mean()) ** 2))
            row["oos_r2"] = float(1.0 - test_metrics["mse"] / denominator) if denominator else np.nan

            metric_rows.append(row)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "split": split_id,
                        "observation_index": test_indices,
                        "date": dates[test_indices].astype(str),
                        "investor": investor,
                        "actual_action": y_test,
                        "predicted_action": predicted,
                        "state_score": state_score,
                    }
                )
            )

    metrics = pd.DataFrame(metric_rows)
    weights = pd.DataFrame(weight_rows)
    value_columns = [c for c in metrics.columns if c not in {"split", "investor", "train_size", "test_size"}]
    metrics_summary = (
        metrics.melt(id_vars=["investor"], value_vars=value_columns, var_name="metric", value_name="value")
        .groupby(["investor", "metric"], sort=False)["value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    dump_yaml(config, output_dir / "config_snapshot.yaml")
    metrics.to_csv(output_dir / "cv_metrics.csv", index=False)
    metrics_summary.to_csv(output_dir / "cv_metrics_summary.csv", index=False)
    weights.to_csv(output_dir / "reward_weights.csv", index=False)
    summarize_reward_weights(weights).to_csv(output_dir / "reward_weights_summary.csv", index=False)
    pd.DataFrame(lambda_rows).to_csv(output_dir / "selected_lambda.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    if context_weight_rows:
        cw = pd.DataFrame(context_weight_rows)
        cw.to_csv(output_dir / "context_weights.csv", index=False)
        summarize_context_weights(cw).to_csv(output_dir / "context_weights_summary.csv", index=False)
    if context_main_rows:
        cm = pd.DataFrame(context_main_rows)
        cm.to_csv(output_dir / "context_main_weights.csv", index=False)
        summarize_context_main_weights(cm).to_csv(
            output_dir / "context_main_weights_summary.csv", index=False
        )
    (output_dir / "cv_summary.json").write_text(
        json.dumps(
            {
                "solver": "coordinate_descent_exact",
                "lambda_mode": args.lambda_mode,
                "n_splits": int(metrics["split"].nunique()),
                "n_observations": int(len(features)),
            },
            indent=2,
        )
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
