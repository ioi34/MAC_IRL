"""Generate the two English figures used in the four-page SPR manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUNS = Path("runs/revision_20260727")
PREDICTIONS = Path(
    "runs/continuous_reward3_ctxmain_default_exact_config/predictions.csv"
)
OUT = Path("paper-icaif26/figures")
OUT.mkdir(parents=True, exist_ok=True)

INVESTORS = ["foreign", "institution", "retail"]
INVESTOR_LABEL = {
    "foreign": "Foreign",
    "institution": "Institutional",
    "retail": "Individual",
}
COLOR = {
    "foreign": "#0072B2",
    "institution": "#E69F00",
    "retail": "#009E73",
}
TERM_LABEL = {
    "beta:momentum": "Momentum",
    "beta:herd": "Lagged cross-flow",
    "beta:underwater": "Underwater position",
    "alpha:kospi_return_1d": "KOSPI 200 return",
    "alpha:fx_level_z_252": "USD/KRW level",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 7.3,
        "axes.axisbelow": True,
        "axes.edgecolor": "#4B5563",
        "axes.grid": True,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "grid.alpha": 0.28,
        "grid.color": "#9CA3AF",
        "grid.linewidth": 0.45,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    }
)


def coefficient_stability_figure() -> None:
    coefficients = pd.read_csv(RUNS / "task1_cpcv_coefficients.csv")
    coefficients = coefficients[coefficients.spec == "feat3"]
    bootstrap = pd.read_csv(
        "runs/protocol_validation/calendar_month_bootstrap.csv"
    )

    panels = [
        (
            r"A. Baseline feature weights $\beta$",
            ["beta:momentum", "beta:herd", "beta:underwater"],
        ),
        (
            r"B. Direct context effects $\alpha$",
            ["alpha:kospi_return_1d", "alpha:fx_level_z_252"],
        ),
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.1, 2.7),
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )
    rng = np.random.default_rng(20260728)

    for ax, (title, terms) in zip(axes, panels):
        rows: list[tuple[str, str]] = []
        positions: list[float] = []
        position = 0.0
        for investor in INVESTORS:
            for term in terms:
                rows.append((term, investor))
                positions.append(position)
                position += 1.0
            position += 0.55

        labels: list[str] = []
        for y, (term, investor) in zip(positions, rows):
            split_values = coefficients[
                (coefficients.investor == investor)
                & (coefficients.term == term)
            ].weight.to_numpy()
            interval = bootstrap[
                (bootstrap.investor == investor) & (bootstrap.term == term)
            ].iloc[0]
            jitter = rng.normal(0.0, 0.045, len(split_values))
            ax.scatter(
                split_values,
                np.full_like(split_values, y) + jitter,
                s=4.5,
                color="#A3A3A3",
                alpha=0.48,
                zorder=2,
                linewidths=0,
            )
            ax.plot(
                [interval.ci_lower, interval.ci_upper],
                [y, y],
                color=COLOR[investor],
                lw=1.15,
                alpha=0.48,
                zorder=3,
            )
            mean = split_values.mean()
            sd = split_values.std(ddof=0)
            ax.plot(
                [mean - sd, mean + sd],
                [y, y],
                color=COLOR[investor],
                lw=2.35,
                zorder=4,
            )
            filled = bool(interval.excludes_zero)
            ax.scatter(
                [mean],
                [y],
                s=31,
                zorder=5,
                color=COLOR[investor] if filled else "white",
                edgecolor=COLOR[investor],
                linewidths=1.25,
            )
            ax.text(
                0.98,
                y,
                f"{mean:+.4f}",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=6.6,
                color=COLOR[investor],
                clip_on=False,
            )
            labels.append(
                f"{INVESTOR_LABEL[investor]}: {TERM_LABEL[term]}"
            )

        ax.axvline(0, color="#374151", lw=0.8, zorder=0)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=6.3)
        ax.set_title(title, fontsize=8.4, loc="left", pad=4)
        ax.set_xlabel("Coefficient", fontsize=7.2)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_ylim(position - 0.85, -0.6)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.16)
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax + 0.22 * (xmax - xmin))

    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            color="#A3A3A3",
            markersize=3,
            label="Individual CPCV split",
        ),
        plt.Line2D(
            [],
            [],
            color="#4B5563",
            lw=2.2,
            label=r"Split mean $\pm$ SD",
        ),
        plt.Line2D(
            [],
            [],
            color="#4B5563",
            lw=1.0,
            alpha=0.5,
            label="Calendar-month bootstrap 95% CI",
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc="white",
            mec="#4B5563",
            mew=1.2,
            markersize=4.5,
            label="Hollow marker: CI includes zero",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=5.8,
        bbox_to_anchor=(0.5, 0.002),
        columnspacing=1.0,
        handletextpad=0.45,
    )
    fig.suptitle(
        "Sampling Stability of Coefficients by Investor Type",
        fontsize=9.2,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.20, right=0.99, top=0.82, bottom=0.29, wspace=0.90
    )
    for extension in ("pdf", "png"):
        fig.savefig(
            OUT / f"fig4p_forest_no_v5_en.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def prediction_figure() -> None:
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    daily = (
        predictions.groupby(
            ["investor", "observation_index", "date"], as_index=False
        )
        .agg(
            actual=("actual_action", "first"),
            predicted=("predicted_action", "mean"),
        )
        .sort_values(["investor", "date"])
    )
    metrics = pd.read_csv(RUNS / "task1_cpcv_metrics_summary.csv")
    metrics = metrics[metrics.spec == "feat3"].set_index("investor")

    window = 20
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 2.25), sharex=True)
    for ax, investor in zip(axes, INVESTORS):
        series = daily[daily.investor == investor].set_index("date")
        actual = series.actual.rolling(window, min_periods=window).mean()
        predicted = series.predicted.rolling(
            window, min_periods=window
        ).mean()
        oos_r2 = float(metrics.loc[investor, "oos_r2"])
        ax.axhline(0, color="#6B7280", lw=0.6, zorder=0)
        ax.plot(
            actual.index,
            actual,
            color="#333333",
            lw=0.72,
            label="Realized behavior",
        )
        ax.plot(
            predicted.index,
            predicted,
            color=COLOR[investor],
            lw=0.92,
            label="OOS prediction",
        )
        ax.set_title(
            f"{INVESTOR_LABEL[investor]}  "
            rf"(daily OOS $R^2$ {oos_r2:.3f})",
            fontsize=5.9,
            loc="left",
            pad=1,
        )
        ax.set_ylabel("Net-buying\nbehavior", fontsize=5.1)
        ax.tick_params(axis="both", labelsize=4.8, length=2.3)
        ax.legend(
            loc="lower left",
            ncol=2,
            fontsize=4.8,
            handlelength=1.8,
            columnspacing=1.0,
        )
        ax.grid(axis="y", visible=False)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].set_xlabel("Date", fontsize=5.5)
    fig.suptitle(
        "Realized Behavior and CPCV Out-of-Sample Predictions",
        fontsize=7.2,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.085, right=0.997, top=0.88, bottom=0.15, hspace=0.38
    )
    for extension in ("pdf", "png"):
        fig.savefig(
            OUT / f"fig4p_prediction_en.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    coefficient_stability_figure()
    prediction_figure()
    print("Wrote English four-page manuscript figures.")
