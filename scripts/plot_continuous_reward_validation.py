from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter


INVESTORS = ["foreign", "institution", "retail"]
INVESTOR_LABELS = {
    "foreign": "외국인",
    "institution": "기관",
    "retail": "개인",
}
FEATURES = ["momentum", "shortmom_orth", "relative", "herd", "underwater"]
FEATURE_COLORS = {
    "momentum": "#4C78A8",
    "shortmom_orth": "#72B7B2",
    "relative": "#F2CF5B",
    "herd": "#E45756",
    "underwater": "#B279A2",
}
VARIANT_LABELS = {
    "remove_momentum": "momentum 제거",
    "remove_shortmom_orth": "shortmom_orth 제거",
    "remove_relative": "relative 제거",
    "remove_herd": "herd 제거",
    "remove_underwater": "underwater 제거",
    "remove_behavioral_group": "행동재무 그룹 제거",
    "remove_traditional_group": "전통 특징 그룹 제거",
    "remove_context_kospi_return_1d": "KOSPI 컨텍스트 제거",
    "remove_context_fx_level_z_252": "FX 컨텍스트 제거",
}
VARIANT_ORDER = list(VARIANT_LABELS)
METHOD_LABELS = {
    "CPCV": "CPCV 부호",
    "monthly_block_bootstrap": "월 bootstrap",
    "expanding_walk_forward": "walk-forward",
    "ridge_vs_lasso": "Ridge–L1",
    "momentum_relative_separation": "상관 특징 분리",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot continuous reward stability and ablation validation"
    )
    parser.add_argument(
        "--run-root", default="runs/continuous_reward_validation"
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/2026-07-13/1609_reward_validation_graphs",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Apple SD Gothic Neo",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_stability_scorecard(run_root: Path, output_dir: Path) -> None:
    frame = pd.read_csv(run_root / "analysis" / "stability_method_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), constrained_layout=True)
    for ax, parameter, title in zip(
        axes,
        ("beta", "B"),
        ("기본 가중치 β", "컨텍스트 가중치 B"),
        strict=True,
    ):
        subset = frame.loc[frame["parameter"] == parameter].copy()
        methods = [
            method
            for method in METHOD_LABELS
            if method in subset["method"].unique()
        ]
        matrix = np.full((len(methods), len(INVESTORS)), np.nan)
        labels = np.full(matrix.shape, "", dtype=object)
        for row_idx, method in enumerate(methods):
            for col_idx, investor in enumerate(INVESTORS):
                matched = subset.loc[
                    (subset["method"] == method)
                    & (subset["investor"] == investor)
                ]
                if matched.empty:
                    continue
                row = matched.iloc[0]
                matrix[row_idx, col_idx] = row["pass_rate"]
                labels[row_idx, col_idx] = f"{int(row['passed'])}/{int(row['total'])}"
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(INVESTORS)), [INVESTOR_LABELS[x] for x in INVESTORS])
        ax.set_yticks(range(len(methods)), [METHOD_LABELS[x] for x in methods])
        ax.set_title(title, pad=12)
        for row_idx in range(len(methods)):
            for col_idx in range(len(INVESTORS)):
                if labels[row_idx, col_idx]:
                    color = "white" if matrix[row_idx, col_idx] < 0.35 else "black"
                    ax.text(
                        col_idx,
                        row_idx,
                        labels[row_idx, col_idx],
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        color=color,
                    )
        ax.set_xticks(np.arange(-0.5, len(INVESTORS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(methods), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
    colorbar.set_label("통과 비율")
    fig.suptitle(
        "가중치 안정성 점검표\n초록일수록 여러 검증에서 안정적",
        fontsize=17,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "01_stability_scorecard.png")


def plot_beta_bootstrap_forest(run_root: Path, output_dir: Path) -> None:
    baseline = pd.read_csv(
        run_root / "ablation" / "baseline" / "reward_weights_summary.csv"
    )
    bootstrap = pd.read_csv(
        run_root / "weight_bootstrap" / "bootstrap_reward_weights_summary.csv"
    )
    ridge = pd.read_csv(
        run_root / "analysis" / "ridge_lasso_reward_comparison.csv"
    )
    merged = (
        baseline[["investor", "feature", "mean"]]
        .rename(columns={"mean": "cpcv_mean"})
        .merge(
            bootstrap[
                [
                    "investor",
                    "feature",
                    "mean",
                    "ci_lower",
                    "ci_upper",
                    "passes_sign_90",
                    "ci_excludes_zero",
                ]
            ].rename(columns={"mean": "bootstrap_mean"}),
            on=["investor", "feature"],
        )
        .merge(
            ridge[["investor", "feature", "mean_ridge"]],
            on=["investor", "feature"],
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), sharey=True)
    y = np.arange(len(FEATURES))[::-1]
    for ax, investor in zip(axes, INVESTORS, strict=True):
        group = merged.loc[merged["investor"] == investor].set_index("feature").loc[FEATURES]
        lower_error = group["bootstrap_mean"] - group["ci_lower"]
        upper_error = group["ci_upper"] - group["bootstrap_mean"]
        passed = group["passes_sign_90"] & group["ci_excludes_zero"]
        ax.errorbar(
            group["bootstrap_mean"],
            y,
            xerr=np.vstack([lower_error, upper_error]),
            fmt="o",
            color="#222222",
            ecolor="#777777",
            elinewidth=2,
            capsize=4,
            label="월 bootstrap 평균·95% CI",
        )
        ax.scatter(
            group["cpcv_mean"], y + 0.13, marker="s", color="#4C78A8", s=48,
            label="CPCV 평균", zorder=3,
        )
        ax.scatter(
            group["mean_ridge"], y - 0.13, marker="^", color="#F58518", s=55,
            label="선택 Ridge 평균", zorder=3,
        )
        for y_value, is_passed in zip(y, passed, strict=True):
            ax.text(
                0.98,
                y_value,
                "통과" if is_passed else "실패",
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9,
                color="#2E7D32" if is_passed else "#C62828",
                fontweight="bold",
            )
        ax.axvline(0, color="#444444", linewidth=1, linestyle="--")
        ax.set_title(
            f"{INVESTOR_LABELS[investor]}  ·  bootstrap 통과 {int(passed.sum())}/5"
        )
        ax.set_xlabel("가중치")
        ax.grid(axis="x", alpha=0.22)
    axes[0].set_yticks(y, FEATURES)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "β 가중치 안정성: CPCV · 월 bootstrap · Ridge 비교\nCI가 0을 가로지르면 bootstrap 실패",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.87])
    save_figure(fig, output_dir / "02_beta_bootstrap_forest.png")


def plot_walk_forward_beta(run_root: Path, output_dir: Path) -> None:
    frame = pd.read_csv(run_root / "walk_forward" / "reward_weights.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharex=True)
    for ax, investor in zip(axes, INVESTORS, strict=True):
        subset = frame.loc[frame["investor"] == investor]
        reversals = 0
        for feature in FEATURES:
            group = subset.loc[subset["feature"] == feature].sort_values("split")
            values = group["weight"].to_numpy()
            reversal = (values > 0).any() and (values < 0).any()
            reversals += int(reversal)
            ax.plot(
                group["split"],
                values,
                marker="o",
                linewidth=2.0,
                markersize=5.5,
                color=FEATURE_COLORS[feature],
                linestyle="--" if reversal else "-",
                label=feature,
            )
        ax.axhline(0, color="#333333", linewidth=1, linestyle="--")
        ax.set_title(f"{INVESTOR_LABELS[investor]}  ·  부호 반전 {reversals}/5")
        ax.set_xticks([2023, 2024, 2025])
        ax.set_xlabel("테스트 연도")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("β 가중치")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "2023–2025 expanding walk-forward β\n점선은 기간 중 부호가 반전된 특징",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.87])
    save_figure(fig, output_dir / "03_walk_forward_beta.png")


def plot_ablation_heatmap(
    run_root: Path,
    output_dir: Path,
    *,
    investors: list[str] = INVESTORS,
    output_name: str = "04_ablation_impact_heatmap.png",
) -> None:
    frame = pd.read_csv(run_root / "analysis" / "ablation_paired_bootstrap.csv")
    figure_width = 17 if len(investors) == 3 else 14
    fig, axes = plt.subplots(
        1, 3, figsize=(figure_width, 8.3), constrained_layout=True
    )
    metric_titles = {
        "direction_accuracy": "방향정확도 저하",
        "rmse": "RMSE 증가",
        "correlation": "상관 감소",
    }
    for ax, metric in zip(axes, metric_titles, strict=True):
        subset = frame.loc[frame["metric"] == metric].set_index(
            ["variant", "investor"]
        )
        matrix = np.array(
            [
                [subset.loc[(variant, investor), "degradation_delta"] for investor in investors]
                for variant in VARIANT_ORDER
            ]
        )
        vmax = float(np.nanmax(np.abs(matrix)))
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(investors)), [INVESTOR_LABELS[x] for x in investors])
        ax.set_yticks(
            range(len(VARIANT_ORDER)),
            [VARIANT_LABELS[x] for x in VARIANT_ORDER],
        )
        ax.set_title(metric_titles[metric], pad=10)
        for row_idx, variant in enumerate(VARIANT_ORDER):
            for col_idx, investor in enumerate(investors):
                row = subset.loc[(variant, investor)]
                significant = bool(
                    row["significant_degradation"] or row["significant_improvement"]
                )
                value = row["degradation_delta"]
                color = "white" if abs(value) > vmax * 0.52 else "black"
                decimals = 4 if metric == "rmse" else 3
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:+.{decimals}f}{'*' if significant else ''}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold" if significant else "normal",
                    color=color,
                )
        ax.set_xticks(np.arange(-0.5, len(investors), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(VARIANT_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        colorbar = fig.colorbar(image, ax=ax, shrink=0.65, pad=0.02)
        colorbar.set_label("제거에 따른 저하량")
    subjects = "·".join(INVESTOR_LABELS[investor] for investor in investors)
    fig.suptitle(
        f"Ablation 영향 ({subjects}): 빨강은 제거 시 악화, 파랑은 제거 시 개선\n"
        "* 20일 block bootstrap + BH 보정 q<0.05",
        fontsize=17,
        fontweight="bold",
    )
    save_figure(fig, output_dir / output_name)


def plot_ablation_bars_without_institution(
    run_root: Path,
    output_dir: Path,
) -> None:
    frame = pd.read_csv(run_root / "analysis" / "ablation_paired_bootstrap.csv")
    investors = ["foreign", "retail"]
    metric_titles = {
        "direction_accuracy": "방향정확도 저하",
        "rmse": "RMSE 증가",
        "correlation": "상관 감소",
    }
    colors = {"foreign": "#4C78A8", "retail": "#54A24B"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 8.5), sharey=True)
    y = np.arange(len(VARIANT_ORDER))
    bar_height = 0.34
    for ax, metric in zip(axes, metric_titles, strict=True):
        subset = frame.loc[frame["metric"] == metric].set_index(
            ["variant", "investor"]
        )
        values_by_investor = {}
        for investor_idx, investor in enumerate(investors):
            values = np.array(
                [
                    subset.loc[(variant, investor), "degradation_delta"]
                    for variant in VARIANT_ORDER
                ]
            )
            values_by_investor[investor] = values
            positions = y + (investor_idx - 0.5) * bar_height
            ax.barh(
                positions,
                values,
                height=bar_height,
                color=colors[investor],
                alpha=0.88,
                label=INVESTOR_LABELS[investor],
            )
        limit = max(
            np.max(np.abs(values)) for values in values_by_investor.values()
        ) * 1.28
        ax.set_xlim(-limit, limit)
        for investor_idx, investor in enumerate(investors):
            positions = y + (investor_idx - 0.5) * bar_height
            for position, variant, value in zip(
                positions,
                VARIANT_ORDER,
                values_by_investor[investor],
                strict=True,
            ):
                row = subset.loc[(variant, investor)]
                significant = bool(
                    row["significant_degradation"] or row["significant_improvement"]
                )
                decimals = 4 if metric == "rmse" else 3
                offset = limit * 0.025 if value >= 0 else -limit * 0.025
                ax.text(
                    value + offset,
                    position,
                    f"{value:+.{decimals}f}{'*' if significant else ''}",
                    ha="left" if value >= 0 else "right",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold" if significant else "normal",
                )
        ax.axvline(0, color="#333333", linewidth=1.1)
        ax.set_title(metric_titles[metric])
        ax.grid(axis="x", alpha=0.22)
        ax.set_xlabel("← 제거 시 개선       제거 시 악화 →")
    axes[0].set_yticks(
        y,
        [VARIANT_LABELS[variant] for variant in VARIANT_ORDER],
    )
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Ablation 성능 영향 막대그래프 (기관 제외)\n"
        "0보다 크면 제거 시 악화, 0보다 작으면 제거 시 개선 · * BH q<0.05",
        fontsize=17,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.9])
    save_figure(fig, output_dir / "07_ablation_impact_bars_without_institution.png")


def _correlation_matrix(frame: pd.DataFrame, investor: str) -> np.ndarray:
    matrix = np.eye(len(FEATURES))
    index = {feature: idx for idx, feature in enumerate(FEATURES)}
    for row in frame.loc[frame["investor"] == investor].itertuples(index=False):
        left = index[row.feature_a]
        right = index[row.feature_b]
        matrix[left, right] = row.correlation
        matrix[right, left] = row.correlation
    return matrix


def plot_multicollinearity(run_root: Path, output_dir: Path) -> None:
    correlations = pd.read_csv(
        run_root / "analysis" / "feature_correlations.csv"
    )
    vif = pd.read_csv(run_root / "analysis" / "feature_vif.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    images = []
    for ax, investor in zip(axes.flat[:3], INVESTORS, strict=True):
        matrix = _correlation_matrix(correlations, investor)
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        images.append(image)
        ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=40, ha="right")
        ax.set_yticks(range(len(FEATURES)), FEATURES)
        ax.set_title(f"{INVESTOR_LABELS[investor]} 피처 상관계수")
        for row_idx in range(len(FEATURES)):
            for col_idx in range(len(FEATURES)):
                value = matrix[row_idx, col_idx]
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if abs(value) > 0.6 else "black",
                )
    fig.colorbar(images[0], ax=list(axes.flat[:3]), shrink=0.6, label="상관계수")

    ax = axes.flat[3]
    x = np.arange(len(FEATURES))
    width = 0.24
    investor_colors = ["#4C78A8", "#F58518", "#54A24B"]
    for idx, (investor, color) in enumerate(
        zip(INVESTORS, investor_colors, strict=True)
    ):
        values = (
            vif.loc[vif["investor"] == investor]
            .set_index("feature")
            .loc[FEATURES, "vif"]
        )
        ax.bar(
            x + (idx - 1) * width,
            values,
            width,
            color=color,
            label=INVESTOR_LABELS[investor],
        )
    ax.axhline(5, color="#C62828", linestyle="--", linewidth=1.5, label="경고 기준 VIF=5")
    ax.set_xticks(x, FEATURES, rotation=40, ha="right")
    ax.set_ylabel("VIF")
    ax.set_ylim(0, 5.5)
    ax.set_title("다중공선성 VIF: 모두 경고 기준 미만")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.22)
    fig.suptitle(
        "피처 상관성과 다중공선성\nmomentum–relative 상관 0.67, 최대 VIF 2.44",
        fontsize=17,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "05_correlation_and_vif.png")


def plot_regularization_and_absorption(run_root: Path, output_dir: Path) -> None:
    ridge = pd.read_csv(run_root / "analysis" / "ridge_grid_metrics.csv")
    selected = pd.read_csv(run_root / "analysis" / "ridge_selected.csv")
    separation = pd.read_csv(
        run_root / "analysis" / "correlated_pair_separation.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)

    ax = axes[0]
    investor_colors = ["#4C78A8", "#F58518", "#54A24B"]
    for investor, color in zip(INVESTORS, investor_colors, strict=True):
        group = ridge.loc[ridge["investor"] == investor].sort_values("ridge_strength")
        chosen = selected.loc[selected["investor"] == investor].iloc[0]
        ax.plot(
            group["ridge_strength"],
            group["rmse_mean"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            label=INVESTOR_LABELS[investor],
        )
        ax.scatter(
            chosen["ridge_strength"],
            chosen["rmse_mean"],
            marker="*",
            s=180,
            color=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=4,
        )
        ax.annotate(
            f"λ={chosen['ridge_strength']:g}",
            (chosen["ridge_strength"], chosen["rmse_mean"]),
            xytext=(4, 7),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("L2 규제 강도 λ")
    ax.set_ylabel("CPCV 평균 RMSE")
    ax.set_title("Ridge 강도 탐색  ·  ★ 선택값")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)

    ax = axes[1]
    directions = [
        ("momentum", "relative", "momentum 제거 → relative"),
        ("relative", "momentum", "relative 제거 → momentum"),
    ]
    x = np.arange(len(directions))
    width = 0.24
    for idx, (investor, color) in enumerate(
        zip(INVESTORS, investor_colors, strict=True)
    ):
        values = []
        for removed, remaining, _ in directions:
            row = separation.loc[
                (separation["investor"] == investor)
                & (separation["removed_feature"] == removed)
                & (separation["remaining_feature"] == remaining)
            ].iloc[0]
            values.append(row["absolute_magnitude_ratio"])
        ax.bar(
            x + (idx - 1) * width,
            values,
            width,
            color=color,
            label=INVESTOR_LABELS[investor],
        )
        for x_value, value in zip(x + (idx - 1) * width, values, strict=True):
            ax.text(x_value, value + 0.04, f"{value:.2f}×", ha="center", fontsize=9)
    ax.axhline(1, color="#333333", linestyle="--", linewidth=1.2)
    ax.set_xticks(x, [label for _, _, label in directions])
    ax.set_ylabel("잔존 β 절댓값 / 기준선 절댓값")
    ax.set_title("momentum–relative 대체 효과")
    ax.set_ylim(0, max(2.55, separation["absolute_magnitude_ratio"].max() * 1.18))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.suptitle(
        "정규화와 상관 특징 분리 진단",
        fontsize=17,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "06_regularization_and_absorption.png")


def main() -> None:
    args = parse_args()
    configure_style()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_stability_scorecard(run_root, output_dir)
    plot_beta_bootstrap_forest(run_root, output_dir)
    plot_walk_forward_beta(run_root, output_dir)
    plot_ablation_heatmap(run_root, output_dir)
    plot_ablation_heatmap(
        run_root,
        output_dir,
        investors=["foreign", "retail"],
        output_name="04b_ablation_impact_heatmap_without_institution.png",
    )
    plot_ablation_bars_without_institution(run_root, output_dir)
    plot_multicollinearity(run_root, output_dir)
    plot_regularization_and_absorption(run_root, output_dir)
    print(f"Saved reward-validation plots to {output_dir}")


if __name__ == "__main__":
    main()
