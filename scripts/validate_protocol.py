"""Stability protocol for the recovered rewards (paper section 3.5 / 4.5).

Produces the three validation artifacts the paper claims:

1. calendar-month block bootstrap  -> coefficient confidence intervals
2. expanding walk-forward           -> out-of-sample stability over time
3. regularization swap (L1 -> L2 / none) -> sign robustness to the penalty

All three solve the closed form of the same objective. Under the myopic concave
reward of section 3, the estimator reduces exactly to an L1-penalised regression
of next-day net buying on state features, so the closed form is the estimator,
not an approximation of it.

A fourth output records the agreement between the stochastic-gradient solution
used for the main tables and this closed form, which is the convergence evidence
for the institutional null.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge

from src.data.splits import build_cpcv, combine_test_folds
from src.evaluation.continuous_metrics import evaluate_continuous_actions
from src.features.scaling import (
    fit_context_scaler,
    fit_feature_scaler,
    transform_context_matrix,
    transform_feature_tensor,
)
from src.utils.config import load_configs

LAMBDA = {"foreign": 0.0, "institution": 0.01, "retail": 0.0003}
TERMS = None  # filled in main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward stability protocol")
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
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--output-dir", default="runs/protocol_validation")
    return parser.parse_args()


def design_matrix(features: np.ndarray, contexts: np.ndarray) -> np.ndarray:
    n_f, n_c = features.shape[1], contexts.shape[1]
    inter = np.column_stack(
        [features[:, f] * contexts[:, k] for f in range(n_f) for k in range(n_c)]
    )
    return np.column_stack([features, inter, contexts]).astype(np.float64)


def fit(design: np.ndarray, target: np.ndarray, penalty: str, strength: float) -> np.ndarray:
    if penalty == "none" or strength <= 0.0:
        model = LinearRegression(fit_intercept=False)
    elif penalty == "l1":
        model = Lasso(alpha=strength / 2.0, fit_intercept=False, max_iter=200_000, tol=1e-10)
    elif penalty == "l2":
        model = Ridge(alpha=strength * len(design) / 2.0, fit_intercept=False)
    else:
        raise ValueError(penalty)
    model.fit(design, target)
    return np.asarray(model.coef_, dtype=float).ravel()


def prepare(config) -> dict:
    data = np.load(config["paths"]["processed_dataset"], allow_pickle=False)
    feature_names = list(config["features"]["selected"])
    saved_features = data["feature_names"].astype(str).tolist()
    features = data["features"][..., [saved_features.index(n) for n in feature_names]]
    context_names = list(config["model"]["context_names"])
    saved_contexts = data["context_names"].astype(str).tolist()
    contexts = data["contexts"][:, [saved_contexts.index(c) for c in context_names]]
    return {
        "features": features,
        "contexts": contexts,
        "actions": data["actions"],
        "dates": pd.to_datetime(pd.Series(data["dates"].astype(str))),
        "investors": list(config["investors"]),
        "feature_names": feature_names,
        "context_names": context_names,
    }


def term_names(feature_names: list[str], context_names: list[str]) -> list[str]:
    return (
        [f"beta:{f}" for f in feature_names]
        + [f"B:{f}x{c}" for f in feature_names for c in context_names]
        + [f"alpha:{c}" for c in context_names]
    )


def calendar_month_bootstrap(d: dict, n_bootstrap: int, seed: int = 42) -> pd.DataFrame:
    """Resample calendar months with replacement, refit, record coefficients."""
    rng = np.random.default_rng(seed)
    months = d["dates"].dt.to_period("M").to_numpy()
    unique_months = np.unique(months)
    month_rows = {m: np.flatnonzero(months == m) for m in unique_months}
    names = term_names(d["feature_names"], d["context_names"])

    rows = []
    for investor_idx, investor in enumerate(d["investors"]):
        scaler = fit_feature_scaler(d["features"][:, investor_idx], np.arange(len(d["features"])))
        scaled = transform_feature_tensor(d["features"][:, investor_idx], scaler)
        context_scaler = fit_context_scaler(d["contexts"], np.arange(len(d["contexts"])))
        scaled_contexts = transform_context_matrix(d["contexts"], context_scaler)
        full_design = design_matrix(scaled.astype(np.float64), scaled_contexts.astype(np.float64))
        y = d["actions"][:, investor_idx].astype(np.float64)

        draws = np.empty((n_bootstrap, full_design.shape[1]))
        for b in range(n_bootstrap):
            picked = rng.choice(unique_months, size=len(unique_months), replace=True)
            idx = np.concatenate([month_rows[m] for m in picked])
            draws[b] = fit(full_design[idx], y[idx], "l1", LAMBDA[investor])
        for j, name in enumerate(names):
            column = draws[:, j]
            rows.append(
                {
                    "investor": investor,
                    "term": name,
                    "mean": float(column.mean()),
                    "ci_lower": float(np.percentile(column, 2.5)),
                    "ci_upper": float(np.percentile(column, 97.5)),
                    "sign_consistency": float(max((column > 0).mean(), (column < 0).mean())),
                    "excludes_zero": bool(
                        np.percentile(column, 2.5) > 0 or np.percentile(column, 97.5) < 0
                    ),
                }
            )
    return pd.DataFrame(rows)


def expanding_walk_forward(
    d: dict, *, initial: int = 400, step: int = 60, purge: int = 5
) -> pd.DataFrame:
    names = term_names(d["feature_names"], d["context_names"])
    n = len(d["features"])
    rows = []
    for investor_idx, investor in enumerate(d["investors"]):
        y = d["actions"][:, investor_idx].astype(np.float64)
        origin = 0
        while initial + purge + step <= n:
            train_end = initial + origin * step
            test_start = train_end + purge
            test_end = min(test_start + step, n)
            if test_end - test_start < 10:
                break
            train_idx = np.arange(train_end)
            test_idx = np.arange(test_start, test_end)

            scaler = fit_feature_scaler(d["features"][:, investor_idx], train_idx)
            scaled = transform_feature_tensor(d["features"][:, investor_idx], scaler)
            context_scaler = fit_context_scaler(d["contexts"], train_idx)
            scaled_contexts = transform_context_matrix(d["contexts"], context_scaler)
            design = design_matrix(
                scaled.astype(np.float64), scaled_contexts.astype(np.float64)
            )
            coef = fit(design[train_idx], y[train_idx], "l1", LAMBDA[investor])
            pred = np.clip(design[test_idx] @ coef, -1.0, 1.0)
            metrics = evaluate_continuous_actions(y[test_idx], pred)
            row = {
                "investor": investor,
                "window": origin,
                "train_end_date": str(d["dates"].iloc[train_end - 1].date()),
                "test_start_date": str(d["dates"].iloc[test_start].date()),
                "test_end_date": str(d["dates"].iloc[test_end - 1].date()),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "direction_accuracy": metrics["direction_accuracy"],
                "correlation": metrics["correlation"],
                "mae": metrics["mae"],
            }
            row.update({name: float(value) for name, value in zip(names, coef, strict=True)})
            rows.append(row)
            origin += 1
            if initial + origin * step + purge >= n:
                break
    return pd.DataFrame(rows)


def regularization_swap(d: dict, config) -> pd.DataFrame:
    """Re-estimate the CPCV protocol under L1 / L2 / no penalty."""
    cv = build_cpcv(config)
    split_input = np.arange(len(d["features"])).reshape(-1, 1)
    names = term_names(d["feature_names"], d["context_names"])
    rows = []
    for split_id, (train_idx, test_folds) in enumerate(cv.split(split_input)):
        test_idx = combine_test_folds(test_folds)
        context_scaler = fit_context_scaler(d["contexts"], train_idx)
        scaled_contexts = transform_context_matrix(d["contexts"], context_scaler)
        for investor_idx, investor in enumerate(d["investors"]):
            scaler = fit_feature_scaler(d["features"][:, investor_idx], train_idx)
            scaled = transform_feature_tensor(d["features"][:, investor_idx], scaler)
            design = design_matrix(
                scaled.astype(np.float64), scaled_contexts.astype(np.float64)
            )
            y = d["actions"][:, investor_idx].astype(np.float64)
            for penalty, strength in (
                ("l1", LAMBDA[investor]),
                ("l2", max(LAMBDA[investor], 0.001)),
                ("none", 0.0),
            ):
                coef = fit(design[train_idx], y[train_idx], penalty, strength)
                pred = np.clip(design[test_idx] @ coef, -1.0, 1.0)
                metrics = evaluate_continuous_actions(y[test_idx], pred)
                row = {
                    "split": split_id,
                    "investor": investor,
                    "penalty": penalty,
                    "direction_accuracy": metrics["direction_accuracy"],
                    "correlation": metrics["correlation"],
                }
                row.update({n_: float(v) for n_, v in zip(names, coef, strict=True)})
                rows.append(row)
    return pd.DataFrame(rows)


def convergence_check(d: dict, config, main_run: Path) -> pd.DataFrame:
    """Compare the reported stochastic-gradient solution with the closed form."""
    beta = pd.read_csv(main_run / "reward_weights.csv")
    bmat = pd.read_csv(main_run / "context_weights.csv")
    alpha = pd.read_csv(main_run / "context_main_weights.csv")
    cv = build_cpcv(config)
    split_input = np.arange(len(d["features"])).reshape(-1, 1)
    rows = []
    for split_id, (train_idx, test_folds) in enumerate(cv.split(split_input)):
        combine_test_folds(test_folds)
        context_scaler = fit_context_scaler(d["contexts"], train_idx)
        scaled_contexts = transform_context_matrix(d["contexts"], context_scaler)
        for investor_idx, investor in enumerate(d["investors"]):
            scaler = fit_feature_scaler(d["features"][:, investor_idx], train_idx)
            scaled = transform_feature_tensor(d["features"][:, investor_idx], scaler)
            design = design_matrix(
                scaled.astype(np.float64), scaled_contexts.astype(np.float64)
            )
            y = d["actions"][:, investor_idx].astype(np.float64)
            b = (
                beta[(beta.split == split_id) & (beta.investor == investor)]
                .set_index("feature")
                .loc[d["feature_names"], "weight"]
                .to_numpy()
            )
            bm = bmat[(bmat.split == split_id) & (bmat.investor == investor)].set_index(
                ["feature", "context"]
            )
            bv = np.array(
                [
                    bm.loc[(f, c), "weight"]
                    for f in d["feature_names"]
                    for c in d["context_names"]
                ]
            )
            a = (
                alpha[(alpha.split == split_id) & (alpha.investor == investor)]
                .set_index("context")
                .loc[d["context_names"], "weight"]
                .to_numpy()
            )
            w_sgd = np.concatenate([b, bv, a])
            w_closed = fit(design[train_idx], y[train_idx], "l1", LAMBDA[investor])
            lam = LAMBDA[investor]

            def objective(w):
                pred = np.clip(design[train_idx] @ w, -1.0, 1.0)
                return float(np.mean((pred - y[train_idx]) ** 2) + lam * np.abs(w).sum())

            rows.append(
                {
                    "split": split_id,
                    "investor": investor,
                    "objective_sgd": objective(w_sgd),
                    "objective_closed_form": objective(w_closed),
                    "relative_gap_pct": 100
                    * (objective(w_sgd) - objective(w_closed))
                    / objective(w_closed),
                    "cosine_similarity": float(
                        w_sgd @ w_closed / (np.linalg.norm(w_sgd) * np.linalg.norm(w_closed))
                    ),
                }
            )
    return pd.DataFrame(rows)


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
    d = prepare(config)

    boot = calendar_month_bootstrap(d, args.n_bootstrap)
    boot.to_csv(output_dir / "calendar_month_bootstrap.csv", index=False)

    wf = expanding_walk_forward(d)
    wf.to_csv(output_dir / "expanding_walk_forward.csv", index=False)

    swap = regularization_swap(d, config)
    swap.to_csv(output_dir / "regularization_swap.csv", index=False)

    conv = convergence_check(d, config, Path("runs/continuous_reward3_ctxmain_default"))
    conv.to_csv(output_dir / "convergence_check.csv", index=False)

    print("== calendar-month bootstrap (95% CI excludes zero) ==")
    print(boot[boot.excludes_zero][["investor", "term", "mean", "ci_lower", "ci_upper"]].to_string(index=False))
    print()
    print("== expanding walk-forward ==")
    print(
        wf.groupby("investor")[["direction_accuracy", "correlation", "mae"]]
        .agg(["mean", "std"])
        .round(4)
        .to_string()
    )
    print()
    print("== regularization swap ==")
    print(
        swap.groupby(["investor", "penalty"])[
            ["direction_accuracy", "beta:momentum", "beta:herd", "beta:underwater"]
        ]
        .mean()
        .round(4)
        .to_string()
    )
    print()
    print("== convergence of the reported solution ==")
    print(
        conv.groupby("investor")[["relative_gap_pct", "cosine_similarity"]]
        .agg(["mean", "max", "min"])
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
