"""Compare 3-feature context variants: none / default / vkospi / default+vkospi."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/Users/yumin/Desktop/MAC_IRL/MAC_IRL")
OUT = ROOT / "experiments/2026-07-19/1720_continuous_reward3_vkospi_graphs"

RUNS = {
    "no context": ROOT / "runs/continuous_reward3_no_context",
    "default (kospi+fx)": ROOT / "runs/continuous_reward3_default_context",
    "vkospi only": ROOT / "runs/continuous_reward3_vkospi_context",
    "default + vkospi": ROOT / "runs/continuous_reward3_default_plus_vkospi",
}
COLORS = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

INVESTORS = ["foreign", "institution", "retail"]
FEATURES = ["momentum", "herd", "underwater"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
    }
)


def style_axis(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def annotate_bars(ax, bars, values, stds, run_idx, fmt):
    stagger = 12 if run_idx % 2 else 0
    for bar, value, std in zip(bars, values, stds, strict=True):
        anchor = value + std if value >= 0 else value - std
        ax.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2, anchor),
            textcoords="offset points",
            xytext=(0, 4 + stagger if value >= 0 else -11 - stagger),
            ha="center",
            fontsize=7.0,
            color=INK,
        )


def plot_metric_comparison() -> None:
    frames = []
    for run_name, run_dir in RUNS.items():
        frame = pd.read_csv(run_dir / "cv_metrics_summary.csv")
        frame.insert(0, "run", run_name)
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(OUT / "comparison_metrics_long.csv", index=False)

    metric_specs = [
        ("direction_accuracy", "Direction accuracy"),
        ("correlation", "Correlation"),
        ("mae", "MAE"),
        ("rmse", "RMSE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    width = 0.19
    for ax, (metric, title) in zip(axes.ravel(), metric_specs, strict=True):
        for run_idx, run_name in enumerate(RUNS):
            rows = (
                metrics.query("run == @run_name and metric == @metric")
                .set_index("investor")
                .reindex(INVESTORS)
            )
            positions = [i + (run_idx - 1.5) * width for i in range(len(INVESTORS))]
            bars = ax.bar(
                positions,
                rows["mean"],
                width=width * 0.9,
                yerr=rows["std"],
                color=COLORS[run_idx],
                error_kw={"ecolor": MUTED, "linewidth": 1.0, "capsize": 2},
                label=run_name,
            )
            annotate_bars(ax, bars, rows["mean"], rows["std"], run_idx, "{:.3f}")
        ax.axhline(0.0, color=BASELINE, linewidth=0.8)
        if metric == "direction_accuracy":
            ax.axhline(0.5, color=MUTED, linewidth=0.9, linestyle=":")
        ax.set_xticks(range(len(INVESTORS)))
        ax.set_xticklabels(INVESTORS)
        ax.set_title(title, fontsize=11)
        style_axis(ax)
    axes[0, 0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("3-feature model: OOS metrics by context variant", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "comparison_context_variants_oos.png", dpi=180)
    plt.close(fig)


def plot_vkospi_interactions() -> None:
    c = pd.read_csv(RUNS["vkospi only"] / "context_weights_summary.csv")
    c = c[c.context == "vkospi_1d"]
    c.to_csv(OUT / "vkospi_interaction_terms.csv", index=False)
    fig, axes = plt.subplots(1, len(INVESTORS), figsize=(13, 4.4))
    for ax, investor in zip(axes, INVESTORS, strict=True):
        rows = c[c.investor == investor].set_index("feature").reindex(FEATURES)
        bars = ax.bar(
            FEATURES,
            rows["mean"],
            width=0.55,
            yerr=rows["std"],
            color="#2a78d6",
            error_kw={"ecolor": MUTED, "linewidth": 1.0, "capsize": 3},
        )
        for bar, value, cons in zip(bars, rows["mean"], rows["direction_consistency"], strict=True):
            ax.annotate(
                f"{value:+.4f}\n({cons:.0%})",
                (bar.get_x() + bar.get_width() / 2, 0),
                textcoords="offset points",
                xytext=(0, -26),
                ha="center",
                fontsize=7.5,
                color=INK,
            )
        ax.axhline(0.0, color=BASELINE, linewidth=0.8)
        ax.set_title(investor, fontsize=11)
        ax.set_xticks(range(len(FEATURES)))
        ax.set_xticklabels(FEATURES)
        ax.tick_params(axis="x", pad=30)
        style_axis(ax)
    axes[0].set_ylabel("B (feature × vkospi_1d), CPCV mean ± std")
    fig.suptitle("vkospi interaction terms — value (sign consistency)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "vkospi_interaction_terms.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_metric_comparison()
    plot_vkospi_interactions()
    print("saved", OUT)


if __name__ == "__main__":
    main()
