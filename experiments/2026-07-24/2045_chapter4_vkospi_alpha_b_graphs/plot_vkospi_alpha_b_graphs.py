"""Create three Chapter 4 candidate figures from the VKOSPI + alpha + B run."""

from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/continuous_reward3_ctxmain_plus_vkospi"

INVESTORS = ["foreign", "institution", "retail"]
INVESTOR_LABELS = {
    "foreign": "외국인",
    "institution": "기관",
    "retail": "개인",
}
INVESTOR_COLORS = {
    "foreign": "#0072B2",
    "institution": "#E69F00",
    "retail": "#009E73",
}
FEATURES = ["momentum", "herd", "underwater"]
FEATURE_LABELS = {
    "momentum": "모멘텀",
    "herd": "군집",
    "underwater": "손실구간",
}
CONTEXTS = ["kospi_return_1d", "fx_level_z_252", "vkospi_1d"]
CONTEXT_LABELS = {
    "kospi_return_1d": "KOSPI200 수익률",
    "fx_level_z_252": "USD/KRW 수준",
    "vkospi_1d": "VKOSPI 변동성",
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
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
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


def validate_run() -> None:
    required = [
        RUN / "reward_weights.csv",
        RUN / "context_main_weights.csv",
        RUN / "context_weights.csv",
    ]
    required.extend(
        RUN / f"split_{split_id:02d}" / f"{investor}_loss_history.csv"
        for split_id in range(45)
        for investor in INVESTORS
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing[:5]}")

    beta = pd.read_csv(RUN / "reward_weights.csv")
    alpha = pd.read_csv(RUN / "context_main_weights.csv")
    interaction = pd.read_csv(RUN / "context_weights.csv")
    expected = {
        "beta": 45 * len(INVESTORS) * len(FEATURES),
        "alpha": 45 * len(INVESTORS) * len(CONTEXTS),
        "interaction": (
            45 * len(INVESTORS) * len(FEATURES) * len(CONTEXTS)
        ),
    }
    actual = {
        "beta": len(beta),
        "alpha": len(alpha),
        "interaction": len(interaction),
    }
    if actual != expected:
        raise ValueError(f"Unexpected coefficient row counts: {actual}")


def add_summary_columns(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    summary = (
        frame.groupby(group_columns, as_index=False)
        .agg(
            mean=("weight", "mean"),
            std=("weight", "std"),
            positive_rate=("weight", lambda values: (values > 0).mean()),
            negative_rate=("weight", lambda values: (values < 0).mean()),
            split_count=("split", "nunique"),
        )
    )
    summary["direction_consistency"] = summary[
        ["positive_rate", "negative_rate"]
    ].max(axis=1)
    return frame.merge(summary, on=group_columns, validate="many_to_one")


def load_stability_data() -> dict[str, pd.DataFrame]:
    beta = pd.read_csv(RUN / "reward_weights.csv")
    beta["parameter_type"] = "β 보상가중치"
    beta["parameter"] = beta["feature"].map(FEATURE_LABELS)
    beta = add_summary_columns(beta, ["investor", "feature"])

    alpha = pd.read_csv(RUN / "context_main_weights.csv")
    alpha["parameter_type"] = "α 컨텍스트 주효과"
    alpha["parameter"] = alpha["context"].map(CONTEXT_LABELS)
    alpha = add_summary_columns(alpha, ["investor", "context"])

    interaction = pd.read_csv(RUN / "context_weights.csv")
    interaction["parameter_type"] = "B 상호작용"
    interaction["parameter"] = (
        interaction["feature"].map(FEATURE_LABELS)
        + " × "
        + interaction["context"].map(CONTEXT_LABELS)
    )
    interaction = add_summary_columns(
        interaction,
        ["investor", "feature", "context"],
    )

    export_frames = []
    for key, frame in {
        "beta": beta,
        "alpha": alpha,
        "B": interaction,
    }.items():
        exported = frame.copy()
        exported.insert(0, "coefficient", key)
        export_frames.append(exported)
    pd.concat(export_frames, ignore_index=True, sort=False).to_csv(
        OUT / "01_beta_alpha_b_stability_data.csv",
        index=False,
    )
    return {
        "β 보상가중치": beta,
        "α 컨텍스트 주효과": alpha,
        "B 상호작용": interaction,
    }


def ordered_parameters(parameter_type: str) -> list[str]:
    if parameter_type == "β 보상가중치":
        return [FEATURE_LABELS[feature] for feature in FEATURES]
    if parameter_type == "α 컨텍스트 주효과":
        return [CONTEXT_LABELS[context] for context in CONTEXTS]
    return [
        f"{FEATURE_LABELS[feature]} × {CONTEXT_LABELS[context]}"
        for feature in FEATURES
        for context in CONTEXTS
    ]


def plot_stability() -> None:
    frames = load_stability_data()
    types = list(frames)
    limits = {}
    for parameter_type, frame in frames.items():
        bound = float(np.abs(frame["weight"]).max())
        limits[parameter_type] = max(bound * 1.18, 0.001)

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(17.2, 13.8),
        sharex="col",
    )
    for row, investor in enumerate(INVESTORS):
        color = INVESTOR_COLORS[investor]
        for col, parameter_type in enumerate(types):
            ax = axes[row, col]
            frame = frames[parameter_type]
            frame = frame[frame["investor"] == investor]
            parameters = ordered_parameters(parameter_type)
            positions = np.arange(len(parameters))
            split_jitter = np.linspace(-0.17, 0.17, 45)

            for position, parameter in zip(positions, parameters, strict=True):
                rows = frame[frame["parameter"] == parameter].sort_values("split")
                if len(rows) != 45:
                    raise ValueError(
                        f"{investor} {parameter_type} {parameter}: "
                        f"expected 45 splits, found {len(rows)}"
                    )
                mean = float(rows["mean"].iloc[0])
                std = float(rows["std"].iloc[0])
                ax.scatter(
                    rows["weight"],
                    position + split_jitter,
                    s=9,
                    color="#9A9A9A",
                    alpha=0.36,
                    linewidths=0,
                    zorder=2,
                )
                ax.errorbar(
                    mean,
                    position,
                    xerr=std,
                    fmt="o",
                    markersize=5.8,
                    color=color,
                    ecolor=color,
                    elinewidth=2.0,
                    capsize=3,
                    zorder=4,
                )
                offset = limits[parameter_type] * 0.035
                ax.text(
                    mean + (offset if mean >= 0 else -offset),
                    position - 0.24,
                    f"{mean:+.4f}",
                    color=color,
                    fontsize=7.8,
                    ha="left" if mean >= 0 else "right",
                    va="center",
                )

            ax.axvline(0, color="#454545", linewidth=0.9, zorder=1)
            ax.set_xlim(
                -limits[parameter_type],
                limits[parameter_type],
            )
            ax.set_yticks(positions)
            ax.set_yticklabels(parameters)
            ax.invert_yaxis()
            ax.grid(axis="y", visible=False)
            ax.set_title(
                f"{INVESTOR_LABELS[investor]} · {parameter_type}",
                color=color if col == 0 else "#222222",
                fontweight="semibold" if col == 0 else "normal",
            )
            if row == 2:
                ax.set_xlabel("추정 계수")

    fig.suptitle(
        "투자자별 보상가중치와 컨텍스트 효과의 분할 간 안정성",
        fontsize=16,
        y=0.997,
    )
    fig.text(
        0.5,
        0.012,
        "회색 점은 45개 CPCV 분할의 개별 추정치, 색 점과 가로선은 평균 ± 1 표준편차다. "
        "B는 보상특징 × 컨텍스트 상호작용 계수다.",
        ha="center",
        fontsize=9.7,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.975), h_pad=2.2, w_pad=1.8)
    save_figure(fig, "01_beta_alpha_b_stability")


def plot_reward_weights() -> None:
    summary = pd.read_csv(RUN / "reward_weights_summary.csv")
    summary["feature"] = pd.Categorical(
        summary["feature"],
        categories=FEATURES,
        ordered=True,
    )
    summary = summary.sort_values(["investor", "feature"])
    summary.to_csv(OUT / "02_reward_weights_bar_data.csv", index=False)

    y_bound = float((summary["mean"].abs() + summary["std"]).max()) * 1.23
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.8, 4.9),
        sharey=True,
    )
    for ax, investor in zip(axes, INVESTORS, strict=True):
        rows = summary[summary["investor"] == investor]
        means = rows["mean"].to_numpy(dtype=float)
        stds = rows["std"].to_numpy(dtype=float)
        x = np.arange(len(FEATURES))
        colors = np.where(means >= 0, "#4C92C3", "#D96A68")
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            width=0.68,
            color=colors,
            edgecolor="none",
            error_kw={
                "ecolor": "#222222",
                "elinewidth": 1.5,
                "capsize": 0,
            },
        )
        ax.axhline(0, color="#555555", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [FEATURE_LABELS[feature] for feature in FEATURES],
            rotation=32,
            ha="right",
        )
        ax.set_title(INVESTOR_LABELS[investor])
        ax.set_ylim(-y_bound, y_bound)
        ax.grid(axis="x", visible=False)
        for bar, mean, std in zip(bars, means, stds, strict=True):
            offset = y_bound * 0.025
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + (offset if mean >= 0 else -offset),
                f"{mean:+.4f}",
                ha="center",
                va="bottom" if mean >= 0 else "top",
                fontsize=8.4,
                color="#333333",
            )

    axes[0].set_ylabel("β 보상가중치 (CPCV 평균 ± 표준편차)")
    fig.suptitle(
        "연속형 모형의 투자자별 보상가중치",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "파란색은 양(+)의 평균 계수, 붉은색은 음의 평균 계수다. 오차막대는 45개 분할의 1 표준편차다.",
        ha="center",
        fontsize=9.3,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94), w_pad=1.2)
    save_figure(fig, "02_reward_weights_bar")


def load_loss_history() -> pd.DataFrame:
    frames = []
    for split_id in range(45):
        for investor in INVESTORS:
            path = RUN / f"split_{split_id:02d}" / f"{investor}_loss_history.csv"
            frame = pd.read_csv(path)
            frame.insert(0, "investor", investor)
            frame.insert(0, "split", split_id)
            frames.append(frame)
    history = pd.concat(frames, ignore_index=True)
    if not np.isfinite(
        history[["train_mse", "total_loss"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite training loss encountered")
    return history


def plot_training_convergence() -> None:
    history = load_loss_history()
    summary = (
        history.groupby(["investor", "epoch"], as_index=False)
        .agg(
            train_mse_mean=("train_mse", "mean"),
            train_mse_std=("train_mse", "std"),
            total_loss_mean=("total_loss", "mean"),
            total_loss_std=("total_loss", "std"),
            split_count=("split", "nunique"),
        )
        .sort_values(["investor", "epoch"])
    )
    if not (summary["split_count"] == 45).all():
        raise ValueError("Every epoch must contain all 45 CPCV splits")

    reductions = []
    for investor in INVESTORS:
        rows = summary[summary["investor"] == investor]
        first = float(rows["train_mse_mean"].iloc[0])
        last = float(rows["train_mse_mean"].iloc[-1])
        reductions.append(
            {
                "investor": investor,
                "first_train_mse_mean": first,
                "last_train_mse_mean": last,
                "reduction_percent": (first - last) / first * 100,
                "epochs": int(rows["epoch"].max()),
            }
        )
    reduction_frame = pd.DataFrame(reductions)
    summary = summary.merge(
        reduction_frame[["investor", "reduction_percent"]],
        on="investor",
        validate="many_to_one",
    )
    summary.to_csv(OUT / "03_training_convergence_data.csv", index=False)
    reduction_frame.to_csv(
        OUT / "03_training_convergence_reduction.csv",
        index=False,
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.2, 5.0),
    )
    for ax, investor in zip(axes, INVESTORS, strict=True):
        investor_history = history[history["investor"] == investor]
        rows = summary[summary["investor"] == investor]
        for split_id in range(45):
            split_rows = investor_history[
                investor_history["split"] == split_id
            ]
            ax.plot(
                split_rows["epoch"],
                split_rows["train_mse"],
                color="#8A8A8A",
                alpha=0.13,
                linewidth=0.8,
                zorder=1,
            )

        epoch = rows["epoch"].to_numpy(dtype=float)
        mean = rows["train_mse_mean"].to_numpy(dtype=float)
        std = rows["train_mse_std"].to_numpy(dtype=float)
        ax.fill_between(
            epoch,
            mean - std,
            mean + std,
            color="#4C92C3",
            alpha=0.24,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            epoch,
            mean,
            color="#0072B2",
            linewidth=2.2,
            label="훈련 MSE 평균 ± 1 표준편차",
            zorder=3,
        )
        ax.plot(
            epoch,
            rows["total_loss_mean"],
            color="#E67E22",
            linestyle="--",
            linewidth=1.7,
            label="총손실 평균",
            zorder=3,
        )
        reduction = float(rows["reduction_percent"].iloc[0])
        ax.set_title(
            f"{INVESTOR_LABELS[investor]} ({reduction:.1f}% 감소)"
        )
        ax.set_xlabel("에포크")
        ax.set_xlim(float(epoch.min()), float(epoch.max()))
        ax.legend(frameon=True, loc="best")

    axes[0].set_ylabel("훈련 손실")
    fig.suptitle(
        "45개 CPCV 분할에서의 연속형 모형 학습 수렴",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        "옅은 회색선은 개별 분할의 훈련 MSE이며, 제목의 감소율은 첫 에포크 대비 마지막 에포크 평균 MSE 변화다.",
        ha="center",
        fontsize=9.3,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94), w_pad=1.4)
    save_figure(fig, "03_training_convergence")


def main() -> None:
    configure_style()
    validate_run()
    plot_stability()
    plot_reward_weights()
    plot_training_convergence()
    print(f"Created figures and source CSVs in {OUT}")


if __name__ == "__main__":
    main()
