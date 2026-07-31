"""ICAIF '26 원고 그림 생성.

단단(single-column) 판형용. 기존 전폭 그림 3개를 2개로 줄인다:
  fig_actual_vs_pred_1col.png   (70mm)  — 실현 행동 vs CPCV 표본외 예측
  fig_weights_1col.png          (90mm)  — beta(위) + alpha(아래) 2패널, 기존 Fig 3+4 병합

식별 표시는 V2(부트스트랩 95% 구간의 0 배제) 하나만 쓴다.
V3는 식별 판정에 쓰지 않으므로 'boundary case' 마커를 두지 않는다.

사용:
  python3 scripts/make_paper_figures.py --out-dir ../acmart-primary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

CANONICAL = Path("runs/continuous_reward3_persist_epochs75")
VALIDATION = Path("runs/continuous_reward3_persist_epochs75_validation")

COL_W = 3.35  # acmart sigconf 단단 폭 (inch)
INVESTORS = ["foreign", "institution", "retail"]
INVESTOR_LABEL = {"foreign": "Foreign", "institution": "Institutional", "retail": "Retail"}
COLOR = {"foreign": "#1f77b4", "institution": "#7f7f7f", "retail": "#d95f02"}
FEATURE_LABEL = {"momentum": "Momentum", "persist": "Flow persistence", "underwater": "Loss region"}
FEATURES = ["momentum", "persist", "underwater"]
CONTEXT_LABEL = {"kospi_return_1d": "KOSPI 200 return", "fx_level_z_252": "FX level"}
CONTEXTS = ["kospi_return_1d", "fx_level_z_252"]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "savefig.facecolor": "white",
        }
    )


def _marker(excludes_zero: bool) -> dict:
    """V2 통과 여부만으로 채움을 정한다."""
    return dict(marker="o", markersize=3.2, markerfacecolor=None) if excludes_zero else dict(
        marker="o", markersize=3.2, markerfacecolor="white"
    )


def fig_weights(out_path: Path) -> None:
    beta = pd.read_csv(VALIDATION / "ablation/baseline/reward_weights_summary.csv")
    beta_v2 = pd.read_csv(VALIDATION / "weight_bootstrap/bootstrap_reward_weights_summary.csv")
    beta = beta.merge(beta_v2[["investor", "feature", "ci_excludes_zero"]], on=["investor", "feature"])
    alpha = pd.read_csv(VALIDATION / "weight_bootstrap/bootstrap_context_main_weights_summary.csv")

    fig, (ax_b, ax_a) = plt.subplots(
        2, 1, figsize=(COL_W, 3.54), height_ratios=[9, 6], constrained_layout=True
    )

    # --- beta: 유형 3 x 특징 3 = 9행 (위에서 아래로) ---
    rows, labels = [], []
    for inv in INVESTORS:
        for feat in FEATURES:
            r = beta[(beta.investor == inv) & (beta.feature == feat)].iloc[0]
            rows.append((inv, r["mean"], r["std"], bool(r["ci_excludes_zero"])))
            labels.append(FEATURE_LABEL[feat])
    for y, (inv, m, s, ok) in enumerate(reversed(rows)):
        c = COLOR[inv]
        mk = _marker(ok)
        ax_b.errorbar(
            m, y, xerr=s, color=c, ecolor=c, elinewidth=0.7, capsize=1.5,
            markeredgecolor=c, markeredgewidth=0.7,
            **{**mk, "markerfacecolor": mk["markerfacecolor"] or c},
        )
    ax_b.set_yticks(range(len(rows)))
    ax_b.set_yticklabels(list(reversed(labels)))
    ax_b.set_xlabel(r"Reward weight $\beta$ (CPCV mean $\pm$ 1 SD)", labelpad=1)
    for i, inv in enumerate(INVESTORS):
        ax_b.text(
            1.015, len(rows) - 2 - 3 * i, INVESTOR_LABEL[inv],
            transform=ax_b.get_yaxis_transform(), ha="center", va="center",
            rotation=270, fontsize=6, fontweight="bold", color=COLOR[inv],
        )
    for split in (2.5, 5.5):
        ax_b.axhline(split, color="0.85", lw=0.5)

    # --- alpha: 유형 3 x 맥락 2 = 6행 ---
    rows, labels = [], []
    for inv in INVESTORS:
        for ctx in CONTEXTS:
            r = alpha[(alpha.investor == inv) & (alpha.context == ctx)].iloc[0]
            rows.append((inv, r["mean"], r["ci_lower"], r["ci_upper"], bool(r["ci_excludes_zero"])))
            labels.append(CONTEXT_LABEL[ctx])
    for y, (inv, m, lo, hi, ok) in enumerate(reversed(rows)):
        c = COLOR[inv]
        mk = _marker(ok)
        ax_a.errorbar(
            m, y, xerr=[[m - lo], [hi - m]], color=c, ecolor=c, elinewidth=0.7, capsize=1.5,
            markeredgecolor=c, markeredgewidth=0.7,
            **{**mk, "markerfacecolor": mk["markerfacecolor"] or c},
        )
    ax_a.set_yticks(range(len(rows)))
    ax_a.set_yticklabels(list(reversed(labels)))
    ax_a.set_xlabel(r"Direct context effect $\alpha$ (bootstrap mean and 95\% CI)".replace("\\%", "%"), labelpad=1)
    # 2행 그룹이라 세로 공간이 좁다. 축약 라벨을 쓴다(색은 beta 패널과 동일).
    short = {"foreign": "Foreign", "institution": "Inst.", "retail": "Retail"}
    for i, inv in enumerate(INVESTORS):
        ax_a.text(
            1.015, len(rows) - 1.5 - 2 * i, short[inv],
            transform=ax_a.get_yaxis_transform(), ha="center", va="center",
            rotation=270, fontsize=5.5, fontweight="bold", color=COLOR[inv],
        )
    ax_a.axhline(1.5, color="0.85", lw=0.5)
    ax_a.axhline(3.5, color="0.85", lw=0.5)

    for ax in (ax_b, ax_a):
        ax.axvline(0, color="0.2", lw=0.6)
        ax.grid(axis="x", color="0.9", lw=0.4)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(y=0.08)

    handles = [
        plt.Line2D([], [], color="0.2", marker="o", markersize=3.2, lw=0.7, label="V2 interval excludes zero"),
        plt.Line2D([], [], color="0.2", marker="o", markersize=3.2, lw=0.7,
                   markerfacecolor="white", label="V2 interval includes zero"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.035))
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def fig_actual_vs_pred(out_path: Path, window: int = 20) -> None:
    df = pd.read_csv(CANONICAL / "predictions.csv", parse_dates=["date"])
    daily = df.groupby(["investor", "date"])[["actual_action", "predicted_action"]].mean().reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(COL_W, 2.76), sharex=True, constrained_layout=True)
    for ax, inv in zip(axes, INVESTORS):
        d = daily[daily.investor == inv].sort_values("date").set_index("date")
        roll = d[["actual_action", "predicted_action"]].rolling(window).mean()
        ax.plot(roll.index, roll["actual_action"], color="0.25", lw=0.7, label="Realized")
        ax.plot(roll.index, roll["predicted_action"], color=COLOR[inv], lw=0.7, label="Predicted")
        ax.axhline(0, color="0.8", lw=0.4)
        ax.set_ylabel(INVESTOR_LABEL[inv], fontsize=6)
        ax.set_ylim(-0.55, 0.55)
        ax.set_yticks([-0.4, 0, 0.4])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(loc="upper right", ncol=2, frameon=False, fontsize=5.5, handlelength=1.2)
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="../acmart-primary")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _style()
    fig_weights(out / "fig_weights_1col.png")
    fig_actual_vs_pred(out / "fig_actual_vs_pred_1col.png")
    print(f"wrote {out}/fig_weights_1col.png, {out}/fig_actual_vs_pred_1col.png")


if __name__ == "__main__":
    main()
