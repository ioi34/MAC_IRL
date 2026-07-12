from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_VARIANTS = [
    "e0_baseline",
    "e1_persistence",
    "e7_persistence_liquidity",
]

LABELS = {
    "e0_baseline": "E0 baseline",
    "e1_persistence": "E1 persistence",
    "e7_persistence_liquidity": "E7 persistence x liquidity",
    "t2_ep50_lr0p001_l10p01": "T2: 50 epochs, lr=0.001, L1=0.01",
    "t6_ep50_lr0p001_l10": "T6: 50 epochs, lr=0.001, L1=0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot continuous institution predictions against actual actions"
    )
    parser.add_argument(
        "--runs-root",
        default="runs/continuous_institution_ablation",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/2026-07-10/institution_prediction_graphs",
    )
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        help="Explicit run directories; overrides --runs-root and --variants",
    )
    parser.add_argument("--rolling-window", type=int, default=20)
    return parser.parse_args()


def load_variant_predictions(run_dir: Path, variant: str) -> pd.DataFrame:
    path = run_dir / "predictions.csv"
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame.loc[frame["investor"] == "institution"].copy()
    grouped = (
        frame.groupby(["observation_index", "date"], as_index=False)
        .agg(
            actual_action=("actual_action", "first"),
            predicted_action=("predicted_action", "mean"),
            predicted_std=("predicted_action", "std"),
            n_test_paths=("predicted_action", "size"),
        )
        .sort_values(["date", "observation_index"])
    )
    grouped["variant"] = variant
    grouped["label"] = LABELS.get(variant, variant)
    return grouped


def balanced_accuracy(actual_direction: np.ndarray, predicted_direction: np.ndarray) -> float:
    positive = actual_direction
    negative = ~actual_direction
    tpr = np.mean(predicted_direction[positive] == actual_direction[positive])
    tnr = np.mean(predicted_direction[negative] == actual_direction[negative])
    return float((tpr + tnr) / 2.0)


def summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in frame.groupby("variant", sort=False):
        actual = group["actual_action"].to_numpy(dtype=float)
        predicted = group["predicted_action"].to_numpy(dtype=float)
        actual_direction = actual >= 0
        predicted_direction = predicted >= 0
        rows.append(
            {
                "variant": variant,
                "label": group["label"].iloc[0],
                "n_dates": len(group),
                "rmse": float(np.sqrt(np.mean(np.square(predicted - actual)))),
                "mae": float(np.mean(np.abs(predicted - actual))),
                "pearson": float(np.corrcoef(actual, predicted)[0, 1]),
                "direction_accuracy": float(np.mean(actual_direction == predicted_direction)),
                "balanced_accuracy": balanced_accuracy(actual_direction, predicted_direction),
                "actual_std": float(np.std(actual)),
                "predicted_std": float(np.std(predicted)),
                "std_ratio": float(np.std(predicted) / np.std(actual)),
                "actual_min": float(np.min(actual)),
                "actual_max": float(np.max(actual)),
                "predicted_min": float(np.min(predicted)),
                "predicted_max": float(np.max(predicted)),
            }
        )
    return pd.DataFrame(rows)


def plot_raw_time_series(frame: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    first = frame.loc[frame["variant"] == frame["variant"].iloc[0]]
    ax.plot(
        first["date"],
        first["actual_action"],
        color="0.25",
        linewidth=0.9,
        alpha=0.65,
        label="Actual institution net-buy action",
    )
    for _, group in frame.groupby("variant", sort=False):
        ax.plot(
            group["date"],
            group["predicted_action"],
            linewidth=1.4,
            label=group["label"].iloc[0],
        )
    ax.axhline(0.0, color="0.15", linewidth=0.8, alpha=0.5)
    ax.set_title("Institution continuous action: actual vs. CPCV test prediction")
    ax.set_ylabel("Net-buy action")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_rolling_time_series(
    frame: pd.DataFrame, output_path: Path, rolling_window: int
) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    first = frame.loc[frame["variant"] == frame["variant"].iloc[0]].copy()
    ax.plot(
        first["date"],
        first["actual_action"].rolling(rolling_window, min_periods=1).mean(),
        color="0.2",
        linewidth=2.2,
        label=f"Actual ({rolling_window}d rolling mean)",
    )
    for _, group in frame.groupby("variant", sort=False):
        smoothed = group["predicted_action"].rolling(rolling_window, min_periods=1).mean()
        ax.plot(
            group["date"],
            smoothed,
            linewidth=1.8,
            label=f"{group['label'].iloc[0]} ({rolling_window}d)",
        )
    ax.axhline(0.0, color="0.15", linewidth=0.8, alpha=0.5)
    ax.set_title(
        f"Institution continuous action: actual vs. prediction ({rolling_window}d rolling mean)"
    )
    ax.set_ylabel("Net-buy action")
    ax.set_xlabel("Date")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_scatter(frame: pd.DataFrame, output_path: Path) -> None:
    variants = list(frame["variant"].drop_duplicates())
    fig, axes = plt.subplots(1, len(variants), figsize=(5.2 * len(variants), 5), sharex=True, sharey=True)
    if len(variants) == 1:
        axes = [axes]
    lower = min(frame["actual_action"].min(), frame["predicted_action"].min())
    upper = max(frame["actual_action"].max(), frame["predicted_action"].max())
    padding = (upper - lower) * 0.05
    lower -= padding
    upper += padding
    for ax, variant in zip(axes, variants, strict=True):
        group = frame.loc[frame["variant"] == variant]
        ax.scatter(
            group["actual_action"],
            group["predicted_action"],
            s=13,
            alpha=0.38,
            edgecolors="none",
        )
        ax.plot([lower, upper], [lower, upper], color="0.2", linewidth=1.0, alpha=0.65)
        ax.axhline(0.0, color="0.65", linewidth=0.8)
        ax.axvline(0.0, color="0.65", linewidth=0.8)
        ax.set_title(group["label"].iloc[0])
        ax.set_xlabel("Actual action")
        ax.grid(True, alpha=0.22)
    axes[0].set_ylabel("Predicted action")
    for ax in axes:
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Institution actual vs. predicted action by date", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_dirs:
        run_dirs = [Path(path) for path in args.run_dirs]
        variants = [path.name for path in run_dirs]
    else:
        variants = args.variants
        run_dirs = [runs_root / variant for variant in variants]
    frames = [
        load_variant_predictions(run_dir, variant)
        for run_dir, variant in zip(run_dirs, variants, strict=True)
    ]
    frame = pd.concat(frames, ignore_index=True)
    metrics = summarize_metrics(frame)

    frame.to_csv(output_dir / "institution_actual_vs_prediction_aggregated.csv", index=False)
    metrics.to_csv(output_dir / "institution_actual_vs_prediction_metrics.csv", index=False)

    plot_raw_time_series(frame, output_dir / "institution_actual_vs_prediction_raw.png")
    plot_rolling_time_series(
        frame,
        output_dir / f"institution_actual_vs_prediction_rolling_{args.rolling_window}d.png",
        args.rolling_window,
    )
    plot_scatter(frame, output_dir / "institution_actual_vs_prediction_scatter.png")

    print(metrics.to_string(index=False))
    print(f"Saved plots and tables to {output_dir}")


if __name__ == "__main__":
    main()
