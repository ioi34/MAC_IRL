from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.ablation import (
    aggregate_predictions,
    benjamini_hochberg,
    degradation_delta,
    feature_vif,
    moving_block_bootstrap_deltas,
)


INDIVIDUAL_VARIANTS = {
    "remove_underwater",
    "remove_momentum",
    "remove_relative",
    "remove_volatility",
    "remove_herd_a",
    "remove_herd_b",
}
METRICS = ("accuracy", "macro_f1", "nll")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze merged_e2 ablation outputs")
    parser.add_argument("--run-root", default="runs/ablation_merged_e2")
    parser.add_argument(
        "--baseline-data",
        default="data/processed/ablation_merged_e2/baseline.npz",
    )
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        (column for column in frame if column.startswith("probability_")),
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )


def _paired_predictions(
    baseline: pd.DataFrame,
    ablation: pd.DataFrame,
    investor: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline_i = baseline.loc[baseline["investor"] == investor].sort_values(
        "observation_index"
    )
    ablation_i = ablation.loc[ablation["investor"] == investor].sort_values(
        "observation_index"
    )
    if not np.array_equal(
        baseline_i["observation_index"].to_numpy(),
        ablation_i["observation_index"].to_numpy(),
    ):
        raise ValueError(f"Prediction observations differ for {investor}")
    if not np.array_equal(baseline_i["y_true"].to_numpy(), ablation_i["y_true"].to_numpy()):
        raise ValueError(f"Prediction labels differ for {investor}")
    probability_columns = _probability_columns(baseline_i)
    return (
        baseline_i["y_true"].to_numpy(dtype=int),
        baseline_i[probability_columns].to_numpy(dtype=float),
        ablation_i[probability_columns].to_numpy(dtype=float),
    )


def _split_comparison(
    baseline_metrics: pd.DataFrame,
    ablation_metrics: pd.DataFrame,
    investor: str,
    metric: str,
) -> tuple[float, float]:
    columns = ["split", "investor", metric]
    paired = baseline_metrics[columns].merge(
        ablation_metrics[columns],
        on=["split", "investor"],
        suffixes=("_baseline", "_ablation"),
        validate="one_to_one",
    )
    paired = paired.loc[paired["investor"] == investor]
    if metric == "nll":
        deltas = paired[f"{metric}_ablation"] - paired[f"{metric}_baseline"]
    else:
        deltas = paired[f"{metric}_baseline"] - paired[f"{metric}_ablation"]
    return float(deltas.mean()), float((deltas > 0).mean())


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    variant_dirs = sorted(path for path in run_root.iterdir() if path.is_dir())
    baseline_dir = run_root / "baseline"
    baseline_predictions = aggregate_predictions(
        pd.read_csv(baseline_dir / "predictions.csv")
    )
    baseline_metrics = pd.read_csv(baseline_dir / "cv_metrics.csv")
    investors = baseline_predictions["investor"].drop_duplicates().tolist()
    comparison_rows = []
    metric_summary_frames = []
    reward_weight_frames = []
    context_weight_frames = []

    for variant_dir in variant_dirs:
        variant = variant_dir.name
        summary = pd.read_csv(variant_dir / "cv_metrics_summary.csv")
        summary.insert(0, "variant", variant)
        metric_summary_frames.append(summary)
        reward_weights = pd.read_csv(variant_dir / "reward_weights_summary.csv")
        reward_weights.insert(0, "variant", variant)
        reward_weight_frames.append(reward_weights)
        context_weights = pd.read_csv(variant_dir / "context_weights_summary.csv")
        context_weights.insert(0, "variant", variant)
        context_weight_frames.append(context_weights)
        if variant == "baseline":
            continue

        ablation_predictions = aggregate_predictions(
            pd.read_csv(variant_dir / "predictions.csv")
        )
        ablation_metrics = pd.read_csv(variant_dir / "cv_metrics.csv")
        for investor in investors:
            y_true, baseline_probabilities, ablation_probabilities = _paired_predictions(
                baseline_predictions, ablation_predictions, investor
            )
            for metric in METRICS:
                observed_delta = degradation_delta(
                    metric, y_true, baseline_probabilities, ablation_probabilities
                )
                bootstrap = moving_block_bootstrap_deltas(
                    metric,
                    y_true,
                    baseline_probabilities,
                    ablation_probabilities,
                    block_length=args.block_length,
                    n_resamples=args.n_resamples,
                    seed=args.seed,
                )
                split_delta, split_win_rate = _split_comparison(
                    baseline_metrics, ablation_metrics, investor, metric
                )
                comparison_rows.append(
                    {
                        "variant": variant,
                        "investor": investor,
                        "metric": metric,
                        "oof_degradation": observed_delta,
                        "ci_lower": float(np.quantile(bootstrap, 0.025)),
                        "ci_upper": float(np.quantile(bootstrap, 0.975)),
                        "p_value": float((np.count_nonzero(bootstrap <= 0) + 1) / (len(bootstrap) + 1)),
                        "split_mean_degradation": split_delta,
                        "split_win_rate": split_win_rate,
                        "block_length": args.block_length,
                        "n_resamples": args.n_resamples,
                    }
                )

    comparison = pd.DataFrame(comparison_rows)
    comparison["q_value"] = np.nan
    individual = comparison["variant"].isin(INDIVIDUAL_VARIANTS)
    for _, indices in comparison.loc[individual].groupby(["investor", "metric"]).groups.items():
        comparison.loc[indices, "q_value"] = benjamini_hochberg(
            comparison.loc[indices, "p_value"]
        )
    comparison["significant"] = (
        individual
        & (comparison["ci_lower"] > 0)
        & (comparison["q_value"] < 0.05)
    )
    comparison.to_csv(run_root / "metric_comparison.csv", index=False)
    pd.concat(metric_summary_frames, ignore_index=True).to_csv(
        run_root / "model_metrics.csv", index=False
    )
    pd.concat(reward_weight_frames, ignore_index=True).to_csv(
        run_root / "reward_weights_summary.csv", index=False
    )
    pd.concat(context_weight_frames, ignore_index=True).to_csv(
        run_root / "context_weights_summary.csv", index=False
    )

    data = np.load(args.baseline_data, allow_pickle=False)
    feature_names = data["feature_names"].astype(str).tolist()
    investors = data["investors"].astype(str).tolist()
    action_values = data["action_values"].astype(int).tolist()
    buy_idx = action_values.index(1)
    vif_frames = []
    correlation_rows = []
    for investor_idx, investor in enumerate(investors):
        state_features = data["features"][:, investor_idx, buy_idx, :].astype(float)
        vif = feature_vif(state_features, feature_names)
        vif.insert(0, "investor", investor)
        vif_frames.append(vif)
        correlation = np.corrcoef(state_features, rowvar=False)
        for left_idx, left in enumerate(feature_names):
            for right_idx in range(left_idx + 1, len(feature_names)):
                correlation_rows.append(
                    {
                        "investor": investor,
                        "feature_a": left,
                        "feature_b": feature_names[right_idx],
                        "correlation": float(correlation[left_idx, right_idx]),
                    }
                )
    pd.concat(vif_frames, ignore_index=True).to_csv(run_root / "feature_vif.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(
        run_root / "feature_correlations.csv", index=False
    )


if __name__ == "__main__":
    main()
