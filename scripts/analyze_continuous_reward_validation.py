from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.ablation import benjamini_hochberg, feature_vif
from src.evaluation.continuous_validation import (
    CONTINUOUS_METRICS,
    aggregate_continuous_predictions,
    continuous_degradation_delta,
    continuous_metric_value,
    moving_block_bootstrap_continuous_deltas,
)
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze continuous reward validation outputs"
    )
    parser.add_argument(
        "--run-root", default="runs/continuous_reward_validation"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _paired_predictions(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    investor: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline_i = baseline.loc[baseline["investor"] == investor].sort_values(
        "observation_index"
    )
    candidate_i = candidate.loc[candidate["investor"] == investor].sort_values(
        "observation_index"
    )
    if not np.array_equal(
        baseline_i["observation_index"].to_numpy(),
        candidate_i["observation_index"].to_numpy(),
    ):
        raise ValueError(f"Prediction observations differ for {investor}")
    if not np.allclose(
        baseline_i["actual_action"].to_numpy(),
        candidate_i["actual_action"].to_numpy(),
    ):
        raise ValueError(f"Actual actions differ for {investor}")
    return (
        baseline_i["actual_action"].to_numpy(dtype=float),
        baseline_i["predicted_action"].to_numpy(dtype=float),
        candidate_i["predicted_action"].to_numpy(dtype=float),
    )


def _analyze_ablations(
    run_root: Path,
    output_dir: Path,
    *,
    block_length: int,
    n_resamples: int,
    seed: int,
) -> None:
    ablation_root = run_root / "ablation"
    run_dirs = sorted(
        path
        for path in ablation_root.iterdir()
        if path.is_dir() and (path / "predictions.csv").exists()
    )
    baseline_dir = ablation_root / "baseline"
    if baseline_dir not in run_dirs:
        raise FileNotFoundError(f"Missing ablation baseline: {baseline_dir}")
    aggregated = {
        path.name: aggregate_continuous_predictions(
            pd.read_csv(path / "predictions.csv")
        )
        for path in run_dirs
    }
    investors = aggregated["baseline"]["investor"].drop_duplicates().tolist()

    performance_rows = []
    for variant, frame in aggregated.items():
        for investor in investors:
            subset = frame.loc[frame["investor"] == investor]
            actual = subset["actual_action"].to_numpy(dtype=float)
            predicted = subset["predicted_action"].to_numpy(dtype=float)
            for metric in CONTINUOUS_METRICS:
                performance_rows.append(
                    {
                        "variant": variant,
                        "investor": investor,
                        "metric": metric,
                        "value": continuous_metric_value(metric, actual, predicted),
                    }
                )
    pd.DataFrame(performance_rows).to_csv(
        output_dir / "ablation_oos_performance.csv", index=False
    )

    comparison_rows = []
    for variant, frame in aggregated.items():
        if variant == "baseline":
            continue
        for investor_idx, investor in enumerate(investors):
            actual, baseline_predicted, candidate_predicted = _paired_predictions(
                aggregated["baseline"], frame, investor
            )
            for metric_idx, metric in enumerate(CONTINUOUS_METRICS):
                bootstrap = moving_block_bootstrap_continuous_deltas(
                    metric,
                    actual,
                    baseline_predicted,
                    candidate_predicted,
                    block_length=block_length,
                    n_resamples=n_resamples,
                    seed=seed + investor_idx * 100 + metric_idx,
                )
                finite = bootstrap[np.isfinite(bootstrap)]
                observed = continuous_degradation_delta(
                    metric, actual, baseline_predicted, candidate_predicted
                )
                comparison_rows.append(
                    {
                        "variant": variant,
                        "investor": investor,
                        "metric": metric,
                        "baseline_value": continuous_metric_value(
                            metric, actual, baseline_predicted
                        ),
                        "variant_value": continuous_metric_value(
                            metric, actual, candidate_predicted
                        ),
                        "degradation_delta": observed,
                        "ci_lower": float(np.quantile(finite, 0.025)),
                        "ci_upper": float(np.quantile(finite, 0.975)),
                        "p_value": float(
                            (np.count_nonzero(finite <= 0) + 1) / (len(finite) + 1)
                        ),
                        "p_improvement": float(
                            (np.count_nonzero(finite >= 0) + 1) / (len(finite) + 1)
                        ),
                        "block_length": block_length,
                        "n_resamples": len(finite),
                    }
                )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["q_value"] = comparisons.groupby(
        ["investor", "metric"], sort=False
    )["p_value"].transform(benjamini_hochberg)
    comparisons["q_improvement"] = comparisons.groupby(
        ["investor", "metric"], sort=False
    )["p_improvement"].transform(benjamini_hochberg)
    comparisons["significant_degradation"] = (
        (comparisons["ci_lower"] > 0) & (comparisons["q_value"] < 0.05)
    )
    comparisons["significant_improvement"] = (
        (comparisons["ci_upper"] < 0)
        & (comparisons["q_improvement"] < 0.05)
    )
    comparisons.to_csv(output_dir / "ablation_paired_bootstrap.csv", index=False)

    for filename in ("reward_weights_summary.csv", "context_weights_summary.csv"):
        frames = []
        for path in run_dirs:
            source = path / filename
            if source.exists():
                frame = pd.read_csv(source)
                frame.insert(0, "variant", path.name)
                frames.append(frame)
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(
                output_dir / f"ablation_{filename}", index=False
            )


def _analyze_vif(run_root: Path, output_dir: Path) -> None:
    config = load_yaml(run_root / "ablation" / "baseline" / "config_snapshot.yaml")
    data = np.load(config["paths"]["processed_dataset"], allow_pickle=False)
    saved_features = data["feature_names"].astype(str).tolist()
    feature_names = list(config["features"]["selected"])
    selected = [saved_features.index(name) for name in feature_names]
    investors = data["investors"].astype(str).tolist()
    vif_frames = []
    correlation_rows = []
    for investor_idx, investor in enumerate(investors):
        values = data["features"][:, investor_idx, selected].astype(float)
        vif = feature_vif(values, feature_names)
        vif.insert(0, "investor", investor)
        vif_frames.append(vif)
        correlation = np.corrcoef(values, rowvar=False)
        for left_idx, left in enumerate(feature_names):
            for right_idx in range(left_idx + 1, len(feature_names)):
                correlation_rows.append(
                    {
                        "investor": investor,
                        "feature_a": left,
                        "feature_b": feature_names[right_idx],
                        "correlation": float(correlation[left_idx, right_idx]),
                        "absolute_correlation": float(
                            abs(correlation[left_idx, right_idx])
                        ),
                    }
                )
    pd.concat(vif_frames, ignore_index=True).to_csv(
        output_dir / "feature_vif.csv", index=False
    )
    pd.DataFrame(correlation_rows).to_csv(
        output_dir / "feature_correlations.csv", index=False
    )


def _window_summary(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = group["weight"].to_numpy(dtype=float)
        mean = float(values.mean())
        signs = np.sign(values)
        dominant_sign = 1 if mean > 0 else -1 if mean < 0 else 0
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "n_windows": len(values),
                "mean": mean,
                "std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
                "positive_rate": float((signs > 0).mean()),
                "negative_rate": float((signs < 0).mean()),
                "direction_consistency": float((signs == dominant_sign).mean()),
                "sign_reversal": bool((signs > 0).any() and (signs < 0).any()),
                "passes_no_reversal": bool(
                    not ((signs > 0).any() and (signs < 0).any())
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _analyze_walk_forward(run_root: Path, output_dir: Path) -> None:
    walk_forward = run_root / "walk_forward"
    reward_weights = pd.read_csv(walk_forward / "reward_weights.csv")
    _window_summary(reward_weights, ["investor", "feature"]).to_csv(
        output_dir / "walk_forward_reward_stability.csv", index=False
    )
    context_path = walk_forward / "context_weights.csv"
    if context_path.exists():
        contexts = pd.read_csv(context_path)
        _window_summary(contexts, ["investor", "feature", "context"]).to_csv(
            output_dir / "walk_forward_context_stability.csv", index=False
        )
    pd.read_csv(walk_forward / "walk_forward_metrics.csv").to_csv(
        output_dir / "walk_forward_metrics.csv", index=False
    )


def _ridge_strength(config: dict, investor: str) -> float:
    return float(config["investor_overrides"][investor]["weight_decay"])


def _analyze_regularization(run_root: Path, output_dir: Path) -> None:
    regularization_root = run_root / "regularization"
    candidates = sorted(
        path
        for path in regularization_root.iterdir()
        if path.is_dir() and (path / "cv_metrics.csv").exists()
    )
    grid_rows = []
    for path in candidates:
        config = load_yaml(path / "config_snapshot.yaml")
        metrics = pd.read_csv(path / "cv_metrics.csv")
        for investor, group in metrics.groupby("investor", sort=False):
            grid_rows.append(
                {
                    "run": path.name,
                    "investor": investor,
                    "ridge_strength": _ridge_strength(config, investor),
                    "rmse_mean": float(group["rmse"].mean()),
                    "direction_accuracy_mean": float(
                        group["direction_accuracy"].mean()
                    ),
                    "correlation_mean": float(group["correlation"].mean()),
                }
            )
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(output_dir / "ridge_grid_metrics.csv", index=False)
    selected = grid.loc[grid.groupby("investor")["rmse_mean"].idxmin()].copy()
    selected.to_csv(output_dir / "ridge_selected.csv", index=False)

    baseline_dir = run_root / "ablation" / "baseline"
    lasso_rewards = pd.read_csv(baseline_dir / "reward_weights_summary.csv")
    lasso_contexts_path = baseline_dir / "context_weights_summary.csv"
    ridge_reward_frames = []
    ridge_context_frames = []
    for row in selected.itertuples(index=False):
        run_dir = regularization_root / row.run
        rewards = pd.read_csv(run_dir / "reward_weights_summary.csv")
        rewards = rewards.loc[rewards["investor"] == row.investor].copy()
        rewards["ridge_strength"] = row.ridge_strength
        ridge_reward_frames.append(rewards)
        context_path = run_dir / "context_weights_summary.csv"
        if context_path.exists():
            contexts = pd.read_csv(context_path)
            contexts = contexts.loc[contexts["investor"] == row.investor].copy()
            contexts["ridge_strength"] = row.ridge_strength
            ridge_context_frames.append(contexts)

    ridge_rewards = pd.concat(ridge_reward_frames, ignore_index=True)
    _regularization_comparison(
        lasso_rewards,
        ridge_rewards,
        ["investor", "feature"],
    ).to_csv(output_dir / "ridge_lasso_reward_comparison.csv", index=False)
    if lasso_contexts_path.exists() and ridge_context_frames:
        _regularization_comparison(
            pd.read_csv(lasso_contexts_path),
            pd.concat(ridge_context_frames, ignore_index=True),
            ["investor", "feature", "context"],
        ).to_csv(output_dir / "ridge_lasso_context_comparison.csv", index=False)


def _regularization_comparison(
    lasso: pd.DataFrame,
    ridge: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    lasso_columns = keys + [
        "mean",
        "std",
        "direction_consistency",
        "dominant_direction",
    ]
    ridge_columns = lasso_columns + ["ridge_strength"]
    merged = lasso[lasso_columns].merge(
        ridge[ridge_columns],
        on=keys,
        suffixes=("_lasso", "_ridge"),
        validate="one_to_one",
    )
    merged["sign_agreement"] = (
        np.sign(merged["mean_lasso"]) == np.sign(merged["mean_ridge"])
    )
    merged["lasso_exact_zero"] = np.isclose(
        merged["mean_lasso"], 0.0, atol=1e-8
    )
    merged["ridge_to_lasso_abs_ratio"] = np.divide(
        merged["mean_ridge"].abs(),
        merged["mean_lasso"].abs(),
        out=np.full(len(merged), np.nan),
        where=merged["mean_lasso"].abs() > 1e-12,
    )
    return merged


def _analyze_correlated_pair(run_root: Path, output_dir: Path) -> None:
    ablation_root = run_root / "ablation"
    baseline_rewards = pd.read_csv(
        ablation_root / "baseline" / "reward_weights_summary.csv"
    )
    rows = []
    for removed, remaining in (("momentum", "relative"), ("relative", "momentum")):
        separated = pd.read_csv(
            ablation_root / f"remove_{removed}" / "reward_weights_summary.csv"
        )
        base = baseline_rewards.loc[
            baseline_rewards["feature"] == remaining,
            ["investor", "mean", "std", "direction_consistency"],
        ]
        reduced = separated.loc[
            separated["feature"] == remaining,
            ["investor", "mean", "std", "direction_consistency"],
        ]
        merged = base.merge(
            reduced,
            on="investor",
            suffixes=("_baseline", "_separated"),
            validate="one_to_one",
        )
        merged.insert(0, "remaining_feature", remaining)
        merged.insert(0, "removed_feature", removed)
        merged["sign_flip"] = (
            np.sign(merged["mean_baseline"]) != np.sign(merged["mean_separated"])
        )
        merged["absolute_magnitude_ratio"] = np.divide(
            merged["mean_separated"].abs(),
            merged["mean_baseline"].abs(),
            out=np.full(len(merged), np.nan),
            where=merged["mean_baseline"].abs() > 1e-12,
        )
        rows.append(merged)
    pd.concat(rows, ignore_index=True).to_csv(
        output_dir / "correlated_pair_separation.csv", index=False
    )


def _stability_method_summary(run_root: Path, output_dir: Path) -> None:
    rows = []

    def add_grouped(
        frame: pd.DataFrame,
        *,
        method: str,
        parameter: str,
        passed: pd.Series,
        criterion: str,
    ) -> None:
        counted = frame.assign(_passed=passed).groupby("investor", sort=False)[
            "_passed"
        ]
        for investor, values in counted:
            rows.append(
                {
                    "method": method,
                    "parameter": parameter,
                    "investor": investor,
                    "passed": int(values.sum()),
                    "total": int(len(values)),
                    "pass_rate": float(values.mean()),
                    "criterion": criterion,
                }
            )

    baseline = run_root / "ablation" / "baseline"
    for parameter, filename in (
        ("beta", "reward_weights_summary.csv"),
        ("B", "context_weights_summary.csv"),
    ):
        path = baseline / filename
        if path.exists():
            frame = pd.read_csv(path)
            add_grouped(
                frame,
                method="CPCV",
                parameter=parameter,
                passed=frame["direction_consistency"] >= 0.9,
                criterion="45개 split 부호 일관성 >= 90%",
            )

    bootstrap_root = run_root / "weight_bootstrap"
    for parameter, filename in (
        ("beta", "bootstrap_reward_weights_summary.csv"),
        ("B", "bootstrap_context_weights_summary.csv"),
    ):
        path = bootstrap_root / filename
        if path.exists():
            frame = pd.read_csv(path)
            add_grouped(
                frame,
                method="monthly_block_bootstrap",
                parameter=parameter,
                passed=frame["passes_sign_90"] & frame["ci_excludes_zero"],
                criterion="200회 부호 일관성 >= 90% 및 95% CI가 0 제외",
            )

    for parameter, filename in (
        ("beta", "walk_forward_reward_stability.csv"),
        ("B", "walk_forward_context_stability.csv"),
    ):
        path = output_dir / filename
        if path.exists():
            frame = pd.read_csv(path)
            add_grouped(
                frame,
                method="expanding_walk_forward",
                parameter=parameter,
                passed=frame["passes_no_reversal"],
                criterion="2023~2025 3개 구간 부호 반전 없음",
            )

    for parameter, filename in (
        ("beta", "ridge_lasso_reward_comparison.csv"),
        ("B", "ridge_lasso_context_comparison.csv"),
    ):
        path = output_dir / filename
        if path.exists():
            frame = pd.read_csv(path)
            add_grouped(
                frame,
                method="ridge_vs_lasso",
                parameter=parameter,
                passed=frame["sign_agreement"],
                criterion="CPCV-RMSE 선택 Ridge와 현재 Lasso의 평균 부호 일치",
            )

    separated = pd.read_csv(output_dir / "correlated_pair_separation.csv")
    add_grouped(
        separated,
        method="momentum_relative_separation",
        parameter="beta",
        passed=~separated["sign_flip"],
        criterion="momentum/relative를 각각 제거했을 때 잔존 계수 부호 유지",
    )
    pd.DataFrame(rows).to_csv(
        output_dir / "stability_method_summary.csv", index=False
    )


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    _analyze_ablations(
        run_root,
        output_dir,
        block_length=args.block_length,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    _analyze_vif(run_root, output_dir)
    _analyze_walk_forward(run_root, output_dir)
    _analyze_regularization(run_root, output_dir)
    _analyze_correlated_pair(run_root, output_dir)
    _stability_method_summary(run_root, output_dir)


if __name__ == "__main__":
    main()
