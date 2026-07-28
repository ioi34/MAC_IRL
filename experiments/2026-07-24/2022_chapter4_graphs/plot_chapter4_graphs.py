"""Create four Chapter 4 figures from existing continuous-model run outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent

MAIN_RUN = ROOT / "runs/continuous_reward3_ctxmain_default"
COMPARISON_RUNS = {
    "컨텍스트 없음": ROOT / "runs/continuous_reward3_no_context",
    "상호작용 B만": ROOT / "runs/continuous_reward3_default_context",
}

INVESTORS = ["foreign", "institution", "retail"]
INVESTOR_LABELS = {
    "foreign": "외국인",
    "institution": "기관",
    "retail": "개인",
}
FEATURES = ["momentum", "herd", "underwater"]
FEATURE_LABELS = {
    "momentum": "모멘텀",
    "herd": "군집",
    "underwater": "손실구간",
}
CONTEXTS = ["kospi_return_1d", "fx_level_z_252"]
CONTEXT_LABELS = {
    "kospi_return_1d": "KOSPI200 1일 수익률",
    "fx_level_z_252": "USD/KRW 252일 수준",
}

FEATURE_COLORS = {
    "momentum": "#0072B2",
    "herd": "#D55E00",
    "underwater": "#009E73",
}
INVESTOR_COLORS = {
    "foreign": "#0072B2",
    "institution": "#E69F00",
    "retail": "#009E73",
}
COMPARISON_COLORS = {
    "컨텍스트 없음": "#7A7A7A",
    "상호작용 B만": "#CC79A7",
}


def configure_style() -> None:
    font_path = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = family

    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D7D7D7",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "lines.linewidth": 1.8,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def validate_sources() -> None:
    required = [
        MAIN_RUN / "reward_weights.csv",
        MAIN_RUN / "context_main_weights.csv",
        MAIN_RUN / "context_weights.csv",
        MAIN_RUN / "cv_metrics.csv",
        MAIN_RUN / "predictions.csv",
    ]
    required.extend(path / "cv_metrics.csv" for path in COMPARISON_RUNS.values())
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")

    main_metrics = pd.read_csv(MAIN_RUN / "cv_metrics.csv")
    key_frame = main_metrics[["split", "investor"]]
    if len(main_metrics) != 45 * 3:
        raise ValueError("Main run must contain 45 CPCV splits for three investors")
    for run_path in COMPARISON_RUNS.values():
        comparison = pd.read_csv(run_path / "cv_metrics.csv")
        if not key_frame.equals(comparison[["split", "investor"]]):
            raise ValueError(f"CPCV split keys do not align: {run_path}")


def plot_effective_weights() -> None:
    beta = pd.read_csv(MAIN_RUN / "reward_weights.csv").rename(
        columns={"weight": "beta"}
    )
    interaction = pd.read_csv(MAIN_RUN / "context_weights.csv").rename(
        columns={"weight": "interaction"}
    )
    merged = interaction.merge(
        beta,
        on=["split", "investor", "feature"],
        how="inner",
        validate="many_to_one",
    )

    z_grid = np.linspace(-2.0, 2.0, 81)
    expanded = merged.merge(pd.DataFrame({"context_z": z_grid}), how="cross")
    expanded["effective_weight"] = (
        expanded["beta"] + expanded["interaction"] * expanded["context_z"]
    )
    summary = (
        expanded.groupby(
            ["investor", "context", "feature", "context_z"], as_index=False
        )
        .agg(
            mean=("effective_weight", "mean"),
            p10=("effective_weight", lambda values: values.quantile(0.10)),
            p90=("effective_weight", lambda values: values.quantile(0.90)),
        )
        .sort_values(["investor", "context", "feature", "context_z"])
    )
    summary.to_csv(OUT / "01_context_effective_weights_data.csv", index=False)

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(13.2, 10.4),
        sharex=True,
        sharey="row",
    )
    for row, investor in enumerate(INVESTORS):
        for col, context in enumerate(CONTEXTS):
            ax = axes[row, col]
            for feature in FEATURES:
                rows = summary[
                    (summary["investor"] == investor)
                    & (summary["context"] == context)
                    & (summary["feature"] == feature)
                ]
                x = rows["context_z"].to_numpy(dtype=float)
                mean = rows["mean"].to_numpy(dtype=float)
                p10 = rows["p10"].to_numpy(dtype=float)
                p90 = rows["p90"].to_numpy(dtype=float)
                ax.fill_between(
                    x,
                    p10,
                    p90,
                    color=FEATURE_COLORS[feature],
                    alpha=0.13,
                    linewidth=0,
                )
                ax.plot(
                    x,
                    mean,
                    color=FEATURE_COLORS[feature],
                    label=FEATURE_LABELS[feature],
                )
            ax.axhline(0, color="#555555", linewidth=0.8)
            ax.axvline(0, color="#999999", linewidth=0.7, linestyle=":")
            ax.set_title(
                f"{INVESTOR_LABELS[investor]} · {CONTEXT_LABELS[context]}"
            )
            if col == 0:
                ax.set_ylabel("유효 보상가중치  β + B·C")
            if row == 2:
                ax.set_xlabel("컨텍스트 수준 (학습표준편차)")
            ax.set_xlim(-2, 2)
            ax.set_xticks([-2, -1, 0, 1, 2])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle(
        "시장 컨텍스트에 따른 투자자별 유효 보상가중치 변화",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "실선은 45개 CPCV 분할 평균, 음영은 분할 간 10–90% 범위다. 다른 컨텍스트는 0으로 고정했다.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save_figure(fig, "01_context_effective_weights")


def plot_actual_vs_prediction() -> None:
    predictions = pd.read_csv(
        MAIN_RUN / "predictions.csv", parse_dates=["date"]
    )
    aggregated = (
        predictions.groupby(
            ["investor", "observation_index", "date"], as_index=False
        )
        .agg(
            actual_action=("actual_action", "first"),
            predicted_action=("predicted_action", "mean"),
            prediction_count=("predicted_action", "size"),
        )
        .sort_values(["investor", "date"])
    )
    if not (aggregated["prediction_count"] == 9).all():
        raise ValueError("Each observation must have nine CPCV test predictions")

    aggregated["actual_rolling20"] = aggregated.groupby("investor")[
        "actual_action"
    ].transform(lambda values: values.rolling(20, min_periods=20).mean())
    aggregated["predicted_rolling20"] = aggregated.groupby("investor")[
        "predicted_action"
    ].transform(lambda values: values.rolling(20, min_periods=20).mean())
    aggregated.to_csv(OUT / "02_actual_vs_prediction_rolling20d_data.csv", index=False)

    fig, axes = plt.subplots(3, 1, figsize=(13.2, 9.3), sharex=True)
    for ax, investor in zip(axes, INVESTORS, strict=True):
        rows = aggregated[aggregated["investor"] == investor]
        ax.plot(
            rows["date"],
            rows["actual_rolling20"],
            color="#3E3E3E",
            linewidth=1.8,
            label="실제 행동",
        )
        ax.plot(
            rows["date"],
            rows["predicted_rolling20"],
            color=INVESTOR_COLORS[investor],
            linewidth=2.0,
            label="주모형 예측",
        )
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_title(INVESTOR_LABELS[investor], loc="left", fontweight="normal")
        ax.set_ylabel("순매수 행동")
        ax.legend(loc="upper right", frameon=False, ncol=2)

    axes[-1].set_xlabel("날짜")
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.suptitle(
        "투자자별 실제 행동과 CPCV 외표본 예측 행동",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "각 날짜의 9개 CPCV 시험예측을 먼저 평균한 뒤 실제값과 예측값에 20거래일 이동평균을 적용했다.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save_figure(fig, "02_actual_vs_prediction_rolling20d")


def add_forest_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    category_column: str,
    category_order: list[str],
    category_labels: dict[str, str],
    title: str,
) -> None:
    rng = np.random.default_rng(42)
    positions: list[float] = []
    labels: list[str] = []
    position_lookup: dict[tuple[str, str], float] = {}
    cursor = 0.0
    for investor in INVESTORS:
        for category in category_order:
            positions.append(cursor)
            labels.append(
                f"{INVESTOR_LABELS[investor]} · {category_labels[category]}"
            )
            position_lookup[(investor, category)] = cursor
            cursor += 1.0
        cursor += 0.65

    for (investor, category), rows in frame.groupby(
        ["investor", category_column], sort=False
    ):
        y = position_lookup[(investor, category)]
        weights = rows["weight"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.13, 0.13, size=len(weights))
        ax.scatter(
            weights,
            y + jitter,
            s=11,
            color="#A8A8A8",
            alpha=0.45,
            linewidths=0,
            zorder=1,
        )
        mean = float(np.mean(weights))
        std = float(np.std(weights, ddof=1))
        ax.errorbar(
            mean,
            y,
            xerr=std,
            fmt="o",
            markersize=6,
            color=INVESTOR_COLORS[investor],
            ecolor=INVESTOR_COLORS[investor],
            elinewidth=1.8,
            capsize=3,
            zorder=3,
        )
        offset = 7 if mean >= 0 else -7
        align = "left" if mean >= 0 else "right"
        ax.annotate(
            f"{mean:+.4f}",
            xy=(mean, y),
            xytext=(offset, -1),
            textcoords="offset points",
            ha=align,
            va="center",
            fontsize=8.5,
            color="#333333",
        )

    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("계수값")


def plot_beta_alpha_forest() -> None:
    beta = pd.read_csv(MAIN_RUN / "reward_weights.csv")
    alpha = pd.read_csv(MAIN_RUN / "context_main_weights.csv")

    beta_export = beta.assign(
        parameter="beta",
        term=beta["feature"],
    )[["split", "investor", "parameter", "term", "weight"]]
    alpha_export = alpha.assign(
        parameter="alpha",
        term=alpha["context"],
    )[["split", "investor", "parameter", "term", "weight"]]
    pd.concat([beta_export, alpha_export], ignore_index=True).to_csv(
        OUT / "03_beta_alpha_forest_data.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.2))
    add_forest_panel(
        axes[0],
        beta,
        category_column="feature",
        category_order=FEATURES,
        category_labels=FEATURE_LABELS,
        title="A. 기본 보상가중치 β",
    )
    add_forest_panel(
        axes[1],
        alpha,
        category_column="context",
        category_order=CONTEXTS,
        category_labels=CONTEXT_LABELS,
        title="B. 컨텍스트 직접 주효과 α",
    )
    fig.suptitle(
        "투자자별 보상가중치와 컨텍스트 주효과의 분할 간 안정성",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "회색 점은 개별 CPCV 분할, 색 점과 오차선은 각각 45개 분할 평균과 분할 간 표준편차다.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), w_pad=3.0)
    save_figure(fig, "03_beta_alpha_forest")


def plot_paired_ablation_delta() -> None:
    main = pd.read_csv(MAIN_RUN / "cv_metrics.csv")
    records: list[pd.DataFrame] = []
    for comparison_name, run_path in COMPARISON_RUNS.items():
        comparison = pd.read_csv(run_path / "cv_metrics.csv")
        merged = main.merge(
            comparison,
            on=["split", "investor"],
            suffixes=("_main", "_comparison"),
            validate="one_to_one",
        )
        for metric in ["direction_accuracy", "correlation", "mae"]:
            if metric == "mae":
                delta = (
                    merged[f"{metric}_comparison"] - merged[f"{metric}_main"]
                )
            else:
                delta = (
                    merged[f"{metric}_main"] - merged[f"{metric}_comparison"]
                )
            if metric in {"direction_accuracy", "correlation"}:
                delta = delta * 100.0
            records.append(
                pd.DataFrame(
                    {
                        "split": merged["split"],
                        "investor": merged["investor"],
                        "comparison": comparison_name,
                        "metric": metric,
                        "improvement": delta,
                    }
                )
            )
    deltas = pd.concat(records, ignore_index=True)
    deltas.to_csv(OUT / "04_paired_ablation_delta_data.csv", index=False)

    metric_specs = [
        ("direction_accuracy", "A. 방향 정확도", "개선량 (%p)"),
        ("correlation", "B. 상관계수", "개선량 (%p)"),
        ("mae", "C. MAE 감소", "개선량 (행동 단위)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.4))
    base_positions = np.arange(len(INVESTORS), dtype=float)
    offsets = {
        "컨텍스트 없음": -0.17,
        "상호작용 B만": 0.17,
    }
    rng = np.random.default_rng(42)

    for ax, (metric, title, ylabel) in zip(axes, metric_specs, strict=True):
        metric_rows = deltas[deltas["metric"] == metric]
        for comparison_name in COMPARISON_RUNS:
            values_by_investor = []
            positions = []
            for investor_index, investor in enumerate(INVESTORS):
                rows = metric_rows[
                    (metric_rows["investor"] == investor)
                    & (metric_rows["comparison"] == comparison_name)
                ]
                values = rows["improvement"].to_numpy(dtype=float)
                position = base_positions[investor_index] + offsets[comparison_name]
                values_by_investor.append(values)
                positions.append(position)
                jitter = rng.uniform(-0.055, 0.055, size=len(values))
                ax.scatter(
                    position + jitter,
                    values,
                    s=12,
                    color=COMPARISON_COLORS[comparison_name],
                    alpha=0.32,
                    linewidths=0,
                    zorder=1,
                )
                mean = float(np.mean(values))
                ax.scatter(
                    position,
                    mean,
                    marker="D",
                    s=34,
                    color=COMPARISON_COLORS[comparison_name],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=4,
                )
                ax.annotate(
                    f"{mean:+.2f}" if metric != "mae" else f"{mean:+.4f}",
                    xy=(position, mean),
                    xytext=(0, 7 if mean >= 0 else -11),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if mean >= 0 else "top",
                    fontsize=8.0,
                    color="#333333",
                )

            box = ax.boxplot(
                values_by_investor,
                positions=positions,
                widths=0.23,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#333333", "linewidth": 1.1},
                whiskerprops={"color": COMPARISON_COLORS[comparison_name]},
                capprops={"color": COMPARISON_COLORS[comparison_name]},
                boxprops={
                    "facecolor": COMPARISON_COLORS[comparison_name],
                    "edgecolor": COMPARISON_COLORS[comparison_name],
                    "alpha": 0.20,
                },
                zorder=2,
            )
            for patch in box["boxes"]:
                patch.set_alpha(0.20)

        ax.axhline(0, color="#555555", linewidth=0.9)
        ax.set_xticks(
            base_positions,
            [INVESTOR_LABELS[investor] for investor in INVESTORS],
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=8,
            markerfacecolor=COMPARISON_COLORS[name],
            markeredgecolor="none",
            label=name,
        )
        for name in COMPARISON_RUNS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.suptitle(
        "컨텍스트 주효과 포함 주모형의 CPCV 분할별 외표본 성능 개선",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "모든 패널에서 양수는 주모형 개선을 뜻한다. 점은 45개 대응 분할, 상자는 분할 분포, ◆는 평균이다.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.89), w_pad=2.5)
    save_figure(fig, "04_paired_ablation_delta")


def main() -> None:
    configure_style()
    validate_sources()
    plot_effective_weights()
    plot_actual_vs_prediction()
    plot_beta_alpha_forest()
    plot_paired_ablation_delta()
    print(f"Saved Chapter 4 figures to {OUT}")


if __name__ == "__main__":
    main()
