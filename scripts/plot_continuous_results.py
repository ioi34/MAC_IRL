from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot continuous experiment results")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rolling-window", type=int, default=20)
    return parser.parse_args()


def plot_reward_weights(run_dir: Path, output_dir: Path) -> None:
    weights = pd.read_csv(run_dir / "reward_weights_summary.csv")
    investors = weights["investor"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(investors), figsize=(5.2 * len(investors), 4.8), sharey=True)
    if len(investors) == 1:
        axes = [axes]
    for ax, investor in zip(axes, investors, strict=True):
        group = weights.loc[weights["investor"] == investor]
        colors = ["#2878B5" if value >= 0 else "#D9534F" for value in group["mean"]]
        ax.bar(group["feature"], group["mean"], yerr=group["std"], color=colors, alpha=0.85)
        ax.axhline(0.0, color="0.2", linewidth=0.8)
        ax.set_title(investor)
        ax.tick_params(axis="x", rotation=40)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Beta weight (CPCV mean ± std)")
    fig.suptitle("Continuous reward weights")
    fig.tight_layout()
    fig.savefig(output_dir / "reward_weights.png", dpi=180)
    plt.close(fig)


def plot_predictions(run_dir: Path, output_dir: Path, rolling_window: int) -> None:
    predictions = pd.read_csv(run_dir / "predictions.csv", parse_dates=["date"])
    aggregated = (
        predictions.groupby(["investor", "observation_index", "date"], as_index=False)
        .agg(actual_action=("actual_action", "first"), predicted_action=("predicted_action", "mean"))
        .sort_values(["investor", "date"])
    )
    investors = aggregated["investor"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(investors), 1, figsize=(14, 3.7 * len(investors)), sharex=True)
    if len(investors) == 1:
        axes = [axes]
    for ax, investor in zip(axes, investors, strict=True):
        group = aggregated.loc[aggregated["investor"] == investor]
        actual = group["actual_action"].rolling(rolling_window, min_periods=1).mean()
        predicted = group["predicted_action"].rolling(rolling_window, min_periods=1).mean()
        ax.plot(group["date"], actual, color="0.25", linewidth=1.8, label="Actual")
        ax.plot(group["date"], predicted, color="#2878B5", linewidth=1.8, label="Predicted")
        ax.axhline(0.0, color="0.2", linewidth=0.8, alpha=0.6)
        ax.set_title(investor)
        ax.set_ylabel("Net-buy action")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Actual vs. CPCV prediction ({rolling_window}-day rolling mean)")
    fig.tight_layout()
    fig.savefig(output_dir / f"actual_vs_prediction_rolling_{rolling_window}d.png", dpi=180)
    plt.close(fig)
    aggregated.to_csv(output_dir / "predictions_aggregated.csv", index=False)


def plot_training_loss(run_dir: Path, output_dir: Path) -> None:
    frames = []
    for path in sorted(run_dir.glob("split_*/*_loss_history.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "investor", path.stem.removesuffix("_loss_history"))
        frame.insert(0, "split", path.parent.name)
        frames.append(frame)
    history = pd.concat(frames, ignore_index=True)
    investors = history["investor"].drop_duplicates().tolist()
    fig, axes = plt.subplots(1, len(investors), figsize=(5.2 * len(investors), 4.8))
    if len(investors) == 1:
        axes = [axes]
    summary_frames = []
    for ax, investor in zip(axes, investors, strict=True):
        group = history.loc[history["investor"] == investor]
        for _, split in group.groupby("split", sort=False):
            ax.plot(split["epoch"], split["train_mse"], color="0.45", alpha=0.1, linewidth=0.7)
        summary = (
            group.groupby("epoch", as_index=False)
            .agg(
                train_mse_mean=("train_mse", "mean"),
                train_mse_std=("train_mse", "std"),
                train_mse_min=("train_mse", "min"),
                train_mse_max=("train_mse", "max"),
                total_loss_mean=("total_loss", "mean"),
                total_loss_std=("total_loss", "std"),
            )
        )
        summary.insert(0, "investor", investor)
        summary_frames.append(summary)
        lower = summary["train_mse_mean"] - summary["train_mse_std"]
        upper = summary["train_mse_mean"] + summary["train_mse_std"]
        ax.fill_between(summary["epoch"], lower, upper, color="#2878B5", alpha=0.2)
        ax.plot(
            summary["epoch"],
            summary["train_mse_mean"],
            color="#2878B5",
            linewidth=2.2,
            label="Train MSE mean ± 1 std",
        )
        if not summary["total_loss_mean"].equals(summary["train_mse_mean"]):
            ax.plot(
                summary["epoch"],
                summary["total_loss_mean"],
                color="#F28E2B",
                linewidth=1.6,
                linestyle="--",
                label="Total loss mean",
            )
        initial = summary["train_mse_mean"].iloc[0]
        final = summary["train_mse_mean"].iloc[-1]
        reduction = 100.0 * (initial - final) / initial
        ax.set_title(f"{investor} ({reduction:.1f}% reduction)")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
    axes[0].set_ylabel("Training loss")
    fig.suptitle("Continuous model training convergence across 45 CPCV splits")
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss_curves.png", dpi=180)
    plt.close(fig)
    pd.concat(summary_frames, ignore_index=True).to_csv(
        output_dir / "training_loss_summary.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_reward_weights(run_dir, output_dir)
    plot_predictions(run_dir, output_dir, args.rolling_window)
    plot_training_loss(run_dir, output_dir)
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
