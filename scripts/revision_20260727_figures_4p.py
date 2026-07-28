"""4쪽 재구성본의 그림 2개를 정확해(좌표하강 Lasso) 결과에서 생성.

그림 1: 계수 안정성 forest plot (beta 3항 + alpha 2항, 3유형).
        회색 점 = 45개 CPCV 분할, 굵은 선 = 분할 간 표준편차,
        옅은 선 = 달력월 블록 부트스트랩 95% CI, 속 빈 표식 = CI가 0 포함.
        회색 음영 = 아무도 특징에 반응하지 않는 V5a 무선호 대역.
        보라 음영 = 외국인은 관측 반응, 개인 직접 반응은 0인 V5b 조건부 청산 대역
                    (개인 momentum·alpha 두 항).
그림 2: 실제 행동과 CPCV 표본외 예측의 20거래일 이동평균.

입력: runs/revision_20260727/task1_cpcv_coefficients.csv (spec=feat3)
      runs/protocol_validation/calendar_month_bootstrap.csv
      runs/revision_20260727/task2_null_sim_summary.csv (spec=feat3)
      runs/revision_20260727/task2_conditional_null_summary.csv
      runs/continuous_reward3_ctxmain_default_exact_config/predictions.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

RUNS = Path("runs/revision_20260727")
PREDICTIONS = Path("runs/continuous_reward3_ctxmain_default_exact_config/predictions.csv")
OUT = Path("paper-icaif26/figures")
OUT.mkdir(parents=True, exist_ok=True)

INV = ["foreign", "institution", "retail"]
KO = {"foreign": "외국인", "institution": "기관", "retail": "개인"}
COLOR = {"foreign": "#0072B2", "institution": "#E69F00", "retail": "#009E73"}
TERM_KO = {
    "beta:momentum": "모멘텀",
    "beta:herd": "시차 교차",
    "beta:underwater": "손실구간",
    "alpha:kospi_return_1d": "KOSPI200 수익률",
    "alpha:fx_level_z_252": "USD/KRW 수준",
}

for font_path in (
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
):
    if Path(font_path).exists():
        try:
            font_manager.fontManager.addfont(font_path)
        except Exception:
            pass
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
font_family = next(
    (
        name
        for name in ("Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans CJK JP")
        if name in available_fonts
    ),
    "sans-serif",
)
plt.rcParams.update({
    "font.family": font_family,
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
})


def figure1():
    c = pd.read_csv(RUNS / "task1_cpcv_coefficients.csv")
    c = c[c.spec == "feat3"]
    b = pd.read_csv("runs/protocol_validation/calendar_month_bootstrap.csv")
    nul = pd.read_csv(RUNS / "task2_null_sim_summary.csv")
    nul = nul[nul.spec == "feat3"]
    conditional = pd.read_csv(RUNS / "task2_conditional_null_summary.csv")
    conditional_terms = {
        "beta:momentum", "alpha:kospi_return_1d", "alpha:fx_level_z_252",
    }

    panels = [("A. 기본 특징 가중치 $\\beta$",
               ["beta:momentum", "beta:herd", "beta:underwater"]),
              ("B. 컨텍스트 직접 주효과 $\\alpha$",
               ["alpha:kospi_return_1d", "alpha:fx_level_z_252"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9),
                             gridspec_kw={"width_ratios": [1.08, 1.0]})
    rng = np.random.default_rng(20260727)
    for ax, (title, terms) in zip(axes, panels):
        rows, ypos = [], []
        row = 0.0
        for inv in INV:
            for term in terms:
                rows.append((term, inv))
                ypos.append(row)
                row += 1.0
            row += 0.55
        labels = []
        for y, (term, inv) in zip(ypos, rows):
            g = c[(c.investor == inv) & (c.term == term)].weight.to_numpy()
            bb = b[(b.investor == inv) & (b.term == term)].iloc[0]
            use_conditional = inv == "retail" and term in conditional_terms
            band = (
                conditional[(conditional.investor == inv) & (conditional.term == term)].iloc[0]
                if use_conditional
                else nul[(nul.investor == inv) & (nul.term == term)].iloc[0]
            )
            jitter = rng.normal(0.0, 0.045, len(g))
            ax.plot([band.q2_5, band.q97_5], [y, y],
                    color="#D8D5F2" if use_conditional else "#E5E7EB", lw=7.0,
                    solid_capstyle="butt", zorder=1)
            ax.scatter(g, np.full_like(g, y) + jitter, s=4.5, color="#A3A3A3",
                       alpha=0.48, zorder=2, linewidths=0)
            ax.plot([bb.ci_lower, bb.ci_upper], [y, y], color=COLOR[inv], lw=1.15,
                    alpha=0.48, zorder=3)
            m, sd = g.mean(), g.std(ddof=0)
            ax.plot([m - sd, m + sd], [y, y], color=COLOR[inv], lw=2.35, zorder=4)
            filled = bool(bb.excludes_zero)
            ax.scatter([m], [y], s=31, zorder=5, color=COLOR[inv] if filled else "white",
                       edgecolor=COLOR[inv], linewidths=1.25, marker="o")
            ax.text(0.98, y, f"{m:+.4f}", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=6.6, color=COLOR[inv],
                    clip_on=False)
            labels.append(f"{KO[inv]} · {TERM_KO[term]}")
        ax.axvline(0, color="#374151", lw=0.8, zorder=0)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=6.7)
        ax.set_title(title, fontsize=8.4, loc="left", pad=4)
        ax.set_xlabel("계수값", fontsize=7.2)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_ylim(row - 0.85, -0.6)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.16)
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin, xmax + 0.22 * (xmax - xmin))
    handles = [
        plt.Line2D([], [], color="#E5E7EB", lw=6, label="V5a 무선호 95% 대역"),
        plt.Line2D([], [], color="#D8D5F2", lw=6, label="V5b 개인 조건부 청산 95% 대역"),
        plt.Line2D([], [], marker="o", ls="", color="#A3A3A3", markersize=3,
                   label="개별 CPCV 분할"),
        plt.Line2D([], [], color="#4B5563", lw=2.2, label="분할 평균 $\\pm$ 표준편차"),
        plt.Line2D([], [], color="#4B5563", lw=1.0, alpha=0.5,
                   label="월 블록 부트스트랩 95% CI"),
        plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#4B5563",
                   mew=1.2, markersize=4.5, label="빈 표식: CI가 0 포함"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=5.9, bbox_to_anchor=(0.5, 0.002),
               columnspacing=1.2, handletextpad=0.5)
    fig.suptitle("투자자별 계수의 표본 안정성과 두 합성 대조",
                 fontsize=9.2, y=0.995)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.83, bottom=0.34, wspace=0.60)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig4p_forest.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure2():
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    daily = (
        predictions.groupby(["investor", "observation_index", "date"], as_index=False)
        .agg(actual=("actual_action", "first"), predicted=("predicted_action", "mean"))
        .sort_values(["investor", "date"])
    )
    metrics = pd.read_csv(RUNS / "task1_cpcv_metrics_summary.csv")
    metrics = metrics[metrics.spec == "feat3"].set_index("investor")
    window = 20
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 2.25), sharex=True)
    for ax, inv in zip(axes, INV):
        g = daily[daily.investor == inv].set_index("date")
        actual = g.actual.rolling(window, min_periods=window).mean()
        predicted = g.predicted.rolling(window, min_periods=window).mean()
        oos_r2 = float(metrics.loc[inv, "oos_r2"])
        ax.axhline(0, color="#6B7280", lw=0.6, zorder=0)
        ax.plot(actual.index, actual, color="#333333", lw=0.72, label="실제 행동")
        ax.plot(predicted.index, predicted, color=COLOR[inv], lw=0.92, label="표본외 예측")
        ax.set_title(f"{KO[inv]}  (일별 표본외 $R^2$ {oos_r2:.3f})",
                     fontsize=5.9, loc="left", pad=1)
        ax.set_ylabel("순매수 행동", fontsize=5.3)
        ax.tick_params(axis="both", labelsize=4.8, length=2.3)
        ax.legend(loc="lower left", ncol=2, fontsize=4.8,
                  handlelength=1.8, columnspacing=1.0)
        ax.grid(axis="y", visible=False)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].set_xlabel("날짜", fontsize=5.5)
    fig.suptitle("투자자별 실제 행동과 CPCV 표본외 예측 행동",
                 fontsize=7.2, y=0.995)
    fig.subplots_adjust(left=0.075, right=0.997, top=0.88, bottom=0.15, hspace=0.38)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig4p_prediction.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure1()
    figure2()
    print("wrote", sorted(p.name for p in OUT.glob("fig4p_*")))
