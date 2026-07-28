"""Generate the paper figures from the recorded experiment artifacts.

Every figure is built from the same run, the two-context context-main-effect
model in ``runs/continuous_reward3_ctxmain_default``, so that no figure in the
paper is drawn from a different specification.

Main figures
  fig1_reward_weights_bar     recovered beta weights per investor
  fig2_actual_vs_prediction   actual and out-of-sample predicted net buying
  fig3_beta_alpha_forest      split-level stability with month-block CIs
  fig4_training_convergence   training loss over epochs across the 45 splits
  fig5_oos_performance        split distributions against the persistence rule
  fig6_walk_forward           coefficient paths over expanding windows
  fig7_institution_null       magnitude and penalty sensitivity

Appendix figures
  figA1_no_saturation, figA2_feature_correlation, figA3_cross_investor_actions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

MAIN_RUN = Path("runs/continuous_reward3_ctxmain_default")
EXACT_RUN = Path("runs/continuous_reward3_ctxmain_default_exact_config")
PROTOCOL = Path("runs/protocol_validation")
DATASET = Path("data/processed/dataset_continuous_reward3_vkospi.npz")
N_SPLITS = 45

INVESTORS = ["foreign", "institution", "retail"]
KO = {"foreign": "외국인", "institution": "기관", "retail": "개인"}
COLOR = {"foreign": "#0072B2", "institution": "#E69F00", "retail": "#009E73"}
POSITIVE, NEGATIVE = "#4C92C3", "#D96A68"

FEATURES = ["momentum", "herd", "underwater"]
FEATURE_KO = {"momentum": "모멘텀", "herd": "군집", "underwater": "손실구간"}
CONTEXTS = ["kospi_return_1d", "fx_level_z_252"]
CONTEXT_KO = {
    "kospi_return_1d": "KOSPI200 1일 수익률",
    "fx_level_z_252": "USD/KRW 252일 수준",
}


def setup_style() -> None:
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ):
        if Path(path).exists():
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                pass
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in (
        "Apple SD Gothic Neo",
        "Noto Sans CJK KR",
        "NanumGothic",
        "Noto Sans CJK JP",
        "Noto Serif CJK JP",
    ):
        if candidate in available:
            mpl.rcParams["font.family"] = candidate
            break
    mpl.rcParams.update(
        {
            "axes.axisbelow": True,
            "axes.edgecolor": "#444444",
            "axes.grid": True,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titlesize": 11,
            "axes.unicode_minus": False,
            "figure.dpi": 200,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "savefig.bbox": "tight",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def save(fig: plt.Figure, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png")
    plt.close(fig)


def load_terms() -> pd.DataFrame:
    beta = pd.read_csv(MAIN_RUN / "reward_weights.csv")
    beta["term"] = "beta:" + beta["feature"]
    alpha = pd.read_csv(MAIN_RUN / "context_main_weights.csv")
    alpha["term"] = "alpha:" + alpha["context"]
    inter = pd.read_csv(MAIN_RUN / "context_weights.csv")
    inter["term"] = "B:" + inter["feature"] + "x" + inter["context"]
    return pd.concat(
        [df[["split", "investor", "term", "weight"]] for df in (beta, alpha, inter)],
        ignore_index=True,
    )


# --------------------------------------------------------------------------- #
# 1. recovered reward weights, per investor
# --------------------------------------------------------------------------- #
def fig1_reward_weights_bar(out: Path) -> None:
    summary = pd.read_csv(MAIN_RUN / "reward_weights_summary.csv")
    summary["feature"] = pd.Categorical(summary["feature"], categories=FEATURES, ordered=True)
    summary = summary.sort_values(["investor", "feature"])

    bound = float((summary["mean"].abs() + summary["std"]).max()) * 1.30
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9), sharey=True)
    for ax, investor in zip(axes, INVESTORS, strict=True):
        rows = summary[summary.investor == investor]
        means = rows["mean"].to_numpy(float)
        stds = rows["std"].to_numpy(float)
        consistency = rows["direction_consistency"].to_numpy(float)
        x = np.arange(len(FEATURES))
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            width=0.66,
            color=np.where(means >= 0, POSITIVE, NEGATIVE),
            edgecolor="none",
            error_kw={"ecolor": "#222222", "elinewidth": 1.4, "capsize": 0},
        )
        ax.axhline(0, color="#555555", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([FEATURE_KO[f] for f in FEATURES])
        ax.set_title(KO[investor])
        ax.set_ylim(-bound, bound)
        ax.grid(axis="x", visible=False)
        for bar, mean, std, cons in zip(bars, means, stds, consistency, strict=True):
            offset = bound * 0.055
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + (offset if mean >= 0 else -offset),
                f"{mean:+.4f}\n({cons*100:.0f}%)",
                ha="center",
                va="bottom" if mean >= 0 else "top",
                fontsize=8.0,
                color="#333333",
                linespacing=1.25,
            )
    axes[0].set_ylabel("β 보상가중치 (CPCV 평균 ± 표준편차)")
    fig.suptitle("투자자별로 복원된 기본 보상가중치", fontsize=13, y=1.0)
    fig.text(
        0.5,
        -0.06,
        "파란색은 양(+), 붉은색은 음(−)의 평균 계수다. 오차막대는 45개 분할의 1 표준편차이며, "
        "괄호 안은 부호 일관성이다.",
        ha="center",
        fontsize=8.8,
        color="#555555",
    )
    fig.tight_layout()
    save(fig, out, "fig1_reward_weights_bar")


# --------------------------------------------------------------------------- #
# 2. actual vs out-of-sample prediction
# --------------------------------------------------------------------------- #
def fig2_actual_vs_prediction(out: Path, window: int = 20) -> None:
    preds = pd.read_csv(MAIN_RUN / "predictions.csv", parse_dates=["date"])
    daily = (
        preds.groupby(["investor", "date"], as_index=False)
        .agg(actual=("actual_action", "mean"), predicted=("predicted_action", "mean"))
        .sort_values(["investor", "date"])
    )
    fig, axes = plt.subplots(3, 1, figsize=(9.4, 6.4), sharex=True)
    for ax, investor in zip(axes, INVESTORS, strict=True):
        s = daily[daily.investor == investor].set_index("date")
        actual = s["actual"].rolling(window, min_periods=window).mean()
        predicted = s["predicted"].rolling(window, min_periods=window).mean()
        ax.axhline(0, color="#888888", lw=0.7)
        ax.plot(actual.index, actual, color="#333333", lw=1.5, label="실제 행동")
        ax.plot(
            predicted.index,
            predicted,
            color=COLOR[investor],
            lw=1.8,
            label="주모형 예측",
        )
        correlation = np.corrcoef(
            *[v.dropna().to_numpy() for v in (actual, predicted)]
        )[0, 1] if actual.dropna().size else np.nan
        ax.set_title(f"{KO[investor]}   (이동평균 상관 {correlation:.2f})", fontsize=10, loc="left")
        ax.set_ylabel("순매수 행동")
        ax.legend(loc="lower left", ncol=2, fontsize=8.5, borderaxespad=0.3)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].set_xlabel("날짜")
    fig.suptitle("투자자별 실제 행동과 CPCV 표본외 예측 행동", fontsize=13)
    fig.text(
        0.5,
        -0.02,
        f"각 날짜의 9개 CPCV 시험예측을 먼저 평균한 뒤 실제값과 예측값에 {window}거래일 "
        "이동평균을 적용했다. 예측 진폭이 실제보다 작은 것은 L1 정규화에 의한 축소 때문이다. 괄호 안 상관은 이동평균 계열 간 값이며, 일별 표본외 상관(표 2)보다 높다.",
        ha="center",
        fontsize=8.8,
        color="#555555",
    )
    fig.tight_layout()
    save(fig, out, "fig2_actual_vs_prediction")


# --------------------------------------------------------------------------- #
# 3. beta / alpha forest with bootstrap intervals
# --------------------------------------------------------------------------- #
def fig3_beta_alpha_forest(out: Path) -> None:
    terms = load_terms()
    boot = pd.read_csv(PROTOCOL / "calendar_month_bootstrap.csv")
    rng = np.random.default_rng(0)

    panels = [
        (
            "A. 기본 보상가중치 β",
            [(f"beta:{f}", FEATURE_KO[f]) for f in FEATURES],
        ),
        (
            "B. 컨텍스트 직접 주효과 α",
            [(f"alpha:{c}", CONTEXT_KO[c]) for c in CONTEXTS],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    for ax, (title, entries) in zip(axes, panels, strict=True):
        labels, positions = [], []
        row = 0.0
        for investor in INVESTORS:
            for term, term_label in entries:
                splits = terms[(terms.term == term) & (terms.investor == investor)]["weight"]
                b = boot[(boot.term == term) & (boot.investor == investor)].iloc[0]
                ax.scatter(
                    splits,
                    np.full(len(splits), row) + rng.normal(0, 0.05, len(splits)),
                    s=8,
                    color="#B0B0B0",
                    alpha=0.55,
                    linewidths=0,
                    zorder=2,
                )
                ax.plot(
                    [b.ci_lower, b.ci_upper],
                    [row, row],
                    color=COLOR[investor],
                    lw=1.4,
                    alpha=0.55,
                    zorder=3,
                )
                ax.errorbar(
                    splits.mean(),
                    row,
                    xerr=splits.std(),
                    fmt="o",
                    ms=7,
                    color=COLOR[investor],
                    mfc=COLOR[investor] if b.excludes_zero else "white",
                    mec=COLOR[investor],
                    mew=1.6,
                    ecolor=COLOR[investor],
                    elinewidth=2.4,
                    capsize=0,
                    zorder=4,
                )
                labels.append(f"{KO[investor]} · {term_label}")
                positions.append(row)
                row -= 1.0
            row -= 0.4
        ax.axvline(0, color="#555555", lw=0.9, zorder=1)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.set_xlabel("계수값")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.16)
    # annotate point estimates on the right edge of each panel
    for ax, (_, entries) in zip(axes, panels, strict=True):
        xmax = ax.get_xlim()[1]
        row = 0.0
        for investor in INVESTORS:
            for term, _ in entries:
                mean = terms[(terms.term == term) & (terms.investor == investor)]["weight"].mean()
                ax.text(
                    xmax * 0.985,
                    row,
                    f"{mean:+.4f}",
                    ha="right",
                    va="center",
                    fontsize=8.0,
                    color=COLOR[investor],
                )
                row -= 1.0
            row -= 0.4

    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#555555", label="95% CI가 0 제외"),
        plt.Line2D(
            [], [], marker="o", ls="", mfc="white", mec="#555555", mew=1.6, label="95% CI가 0 포함"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.8, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("투자자별 보상가중치와 컨텍스트 주효과의 안정성", fontsize=13)
    fig.text(
        0.5,
        -0.09,
        "회색 점은 개별 CPCV 분할이다. 굵은 오차선은 45개 분할의 표준편차, 옅은 가로선은 "
        "달력월 블록 부트스트랩 95% 신뢰구간이다.",
        ha="center",
        fontsize=8.8,
        color="#555555",
    )
    fig.tight_layout()
    save(fig, out, "fig3_beta_alpha_forest")


# --------------------------------------------------------------------------- #
# 4. training convergence
# --------------------------------------------------------------------------- #
def load_loss_history() -> pd.DataFrame:
    frames = []
    for split_id in range(N_SPLITS):
        for investor in INVESTORS:
            path = MAIN_RUN / f"split_{split_id:02d}" / f"{investor}_loss_history.csv"
            frame = pd.read_csv(path)
            frame.insert(0, "investor", investor)
            frame.insert(0, "split", split_id)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def fig4_training_convergence(out: Path) -> None:
    history = load_loss_history()
    convergence = pd.read_csv(PROTOCOL / "convergence_check.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8))
    for ax, investor in zip(axes, INVESTORS, strict=True):
        rows = history[history.investor == investor]
        for _, group in rows.groupby("split"):
            ax.plot(group["epoch"], group["train_mse"], color="#CCCCCC", lw=0.5, alpha=0.6)
        summary = rows.groupby("epoch")["train_mse"].agg(["mean", "std"])
        ax.fill_between(
            summary.index,
            summary["mean"] - summary["std"],
            summary["mean"] + summary["std"],
            color=COLOR[investor],
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(summary.index, summary["mean"], color=COLOR[investor], lw=2.0)
        reduction = (summary["mean"].iloc[0] - summary["mean"].iloc[-1]) / summary["mean"].iloc[0]
        gap = convergence[convergence.investor == investor]["relative_gap_pct"].mean()
        ax.set_title(
            f"{KO[investor]}  (MSE {reduction*100:.1f}% 감소)", fontsize=10.5
        )
        ax.set_xlabel("에포크")
        ax.annotate(
            f"폐쇄형 최적해까지\n남은 여지 {gap:.2f}%",
            xy=(summary.index[-1], summary["mean"].iloc[-1]),
            xytext=(0.53, 0.72),
            textcoords="axes fraction",
            fontsize=8.4,
            color="#333333",
            ha="left",
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.9),
        )
    axes[0].set_ylabel("훈련 MSE")
    fig.suptitle("45개 CPCV 분할에서의 학습 수렴", fontsize=13)
    fig.text(
        0.5,
        -0.04,
        "옅은 회색선은 개별 분할, 굵은 선과 띠는 45개 분할의 평균 ± 1 표준편차다. "
        "주석은 도달한 해가 동일 목적함수의 폐쇄형 최적해 대비 남긴 상대 격차다.",
        ha="center",
        fontsize=8.8,
        color="#555555",
    )
    fig.tight_layout()
    save(fig, out, "fig4_training_convergence")


# --------------------------------------------------------------------------- #
# 5. out-of-sample performance against the persistence rule
# --------------------------------------------------------------------------- #
def fig5_oos_performance(out: Path) -> None:
    metrics = pd.read_csv(MAIN_RUN / "cv_metrics.csv")
    baseline = pd.read_csv(EXACT_RUN / "cv_metrics.csv")
    panels = [("direction_accuracy", "방향 정확도", 0.5), ("correlation", "예측–실제 상관계수", 0.0)]
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, (metric, label, reference) in zip(axes, panels, strict=True):
        for pos, investor in enumerate(INVESTORS):
            values = metrics[metrics.investor == investor][metric].to_numpy()
            parts = ax.violinplot([values], positions=[pos], widths=0.66, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(COLOR[investor])
                body.set_alpha(0.20)
                body.set_edgecolor("none")
            ax.scatter(
                np.full(len(values), pos) + rng.normal(0, 0.055, len(values)),
                values,
                s=7,
                color=COLOR[investor],
                alpha=0.55,
                linewidths=0,
            )
            ax.hlines(values.mean(), pos - 0.3, pos + 0.3, color=COLOR[investor], lw=2.4)
            base = baseline[baseline.investor == investor][f"persistence_{metric}"].mean()
            ax.hlines(base, pos - 0.34, pos + 0.34, color="#333333", lw=1.5, ls=(0, (4, 2)))
        ax.axhline(reference, color="#999999", lw=0.8, ls=":")
        ax.set_xticks(range(3))
        ax.set_xticklabels([KO[i] for i in INVESTORS])
        ax.set_title(label, fontsize=10.5)
    axes[0].set_ylabel("45개 CPCV 분할")
    handles = [
        plt.Line2D([], [], color="#333333", lw=2.4, label="주모형 평균"),
        plt.Line2D([], [], color="#333333", lw=1.5, ls=(0, (4, 2)), label="지속성 기준선"),
    ]
    axes[1].legend(handles=handles, loc="upper right", fontsize=8.5)
    fig.suptitle("표본외 성능의 분할 간 분포와 자명한 기준선", fontsize=13)
    fig.tight_layout()
    save(fig, out, "fig5_oos_performance")


# --------------------------------------------------------------------------- #
# 6. expanding walk-forward
# --------------------------------------------------------------------------- #
def fig6_walk_forward(out: Path) -> None:
    wf = pd.read_csv(PROTOCOL / "expanding_walk_forward.csv")
    entries = [
        ("beta:momentum", "β 모멘텀"),
        ("alpha:kospi_return_1d", "α KOSPI200"),
        ("alpha:fx_level_z_252", "α USD/KRW"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.0))
    for ax, (term, label) in zip(axes[:3], entries, strict=True):
        ax.axhline(0, color="#555555", lw=0.9)
        for investor in INVESTORS:
            s = wf[wf.investor == investor].sort_values("window")
            ax.plot(s["window"], s[term], marker="o", ms=3.6, lw=1.6, color=COLOR[investor])
        ax.set_title(label, fontsize=10.5)
        ax.set_xlabel("확장 창")
        ax.set_xticks(range(0, 10, 3))
    axes[0].set_ylabel("계수")
    ax = axes[3]
    for investor in INVESTORS:
        s = wf[wf.investor == investor].sort_values("window")
        ax.plot(
            s["window"],
            s["direction_accuracy"],
            marker="o",
            ms=3.6,
            lw=1.6,
            color=COLOR[investor],
        )
    ax.axhline(0.5, color="#999999", lw=0.8, ls=":")
    ax.set_title("방향 정확도", fontsize=10.5)
    ax.set_xlabel("확장 창")
    ax.set_xticks(range(0, 10, 3))
    handles = [plt.Line2D([], [], marker="o", ms=3.6, color=COLOR[i], label=KO[i]) for i in INVESTORS]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Expanding walk-forward: 성능은 창마다 흔들리지만 계수 부호는 유지된다", fontsize=12.5)
    fig.tight_layout()
    save(fig, out, "fig6_walk_forward")


# --------------------------------------------------------------------------- #
# 7. institutional null
# --------------------------------------------------------------------------- #
def fig7_institution_null(out: Path) -> None:
    swap = pd.read_csv(PROTOCOL / "regularization_swap.csv")
    terms = load_terms()
    entries = [(f"beta:{f}", FEATURE_KO[f]) for f in FEATURES] + [
        (f"alpha:{c}", "α " + ("KOSPI200" if c == "kospi_return_1d" else "USD/KRW"))
        for c in CONTEXTS
    ]
    fig = plt.figure(figsize=(11.0, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.6, 1, 1, 1], wspace=0.45)

    ax = fig.add_subplot(gs[0, 0])
    width = 0.26
    for offset, investor in zip((-width, 0, width), INVESTORS, strict=True):
        magnitudes = [
            abs(terms[(terms.term == t) & (terms.investor == investor)]["weight"].mean())
            for t, _ in entries
        ]
        ax.barh(
            np.arange(len(entries)) + offset,
            magnitudes,
            height=width,
            color=COLOR[investor],
            label=KO[investor],
        )
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([label for _, label in entries])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(3e-4, 2e-1)
    ax.set_xlabel("계수 절대값 (로그 축)")
    ax.set_title("복원된 선호의 크기", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8)

    penalties = ["l1", "l2", "none"]
    penalty_ko = {"l1": "L1", "l2": "L2", "none": "무벌점"}
    keys = [(f"beta:{f}", FEATURE_KO[f]) for f in FEATURES]
    for col, investor in enumerate(INVESTORS):
        ax = fig.add_subplot(gs[0, col + 1])
        ax.axhline(0, color="#555555", lw=0.9)
        for marker, (key, klabel) in zip(("o", "s", "^"), keys, strict=True):
            values = [
                swap[(swap.investor == investor) & (swap.penalty == p)][key].mean()
                for p in penalties
            ]
            ax.plot(range(3), values, marker=marker, ms=4.5, lw=1.6, color=COLOR[investor], label=klabel)
        ax.set_xticks(range(3))
        ax.set_xticklabels([penalty_ko[p] for p in penalties])
        ax.set_title(KO[investor], fontsize=10.5)
        ax.margins(y=0.35)
        if col == 0:
            ax.set_ylabel("β 계수")
        if col == 2:
            ax.legend(fontsize=7.5, loc="best")
    fig.suptitle("기관의 수렴된 null: 작은 크기와 벌점 의존성 (우측 세 패널은 독립 y축)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out, "fig7_institution_null")


# --------------------------------------------------------------------------- #
# appendix
# --------------------------------------------------------------------------- #
def figA1_no_saturation(out: Path) -> None:
    preds = pd.read_csv(MAIN_RUN / "predictions.csv")
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    bins = np.linspace(-1.05, 1.05, 121)
    for investor in INVESTORS:
        scores = preds[preds.investor == investor]["state_score"]
        ax.hist(
            bins[:-1],
            bins=bins,
            weights=np.histogram(scores, bins=bins, density=True)[0],
            histtype="step",
            lw=1.6,
            color=COLOR[investor],
            label=f"{KO[investor]}  (|q|max = {scores.abs().max():.3f})",
        )
    for bound in (-1, 1):
        ax.axvline(bound, color="#333333", lw=1.2, ls="--")
    ax.set_xlabel("잠재 행동점수 q")
    ax.set_ylabel("밀도")
    ax.set_xlim(-1.1, 1.1)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("포화 부재: |q| ≤ 1이므로 명제 1의 전제가 표본 전체에서 성립", fontsize=10, loc="left")
    fig.tight_layout()
    save(fig, out, "figA1_no_saturation")


def figA2_feature_correlation(out: Path) -> None:
    data = np.load(DATASET, allow_pickle=False)
    feature_names = data["feature_names"].astype(str).tolist()
    matrix = np.column_stack([data["features"][:, 0, :], data["contexts"][:, :2]])
    labels = [FEATURE_KO[f] for f in feature_names] + ["KOSPI200", "USD/KRW"]
    corr = np.corrcoef(matrix.T)
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{corr[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(corr[i, j]) > 0.55 else "#222222",
            )
    fig.colorbar(im, ax=ax, shrink=0.8, label="상관계수")
    ax.set_title("보상 특징과 컨텍스트의 상관", fontsize=10)
    fig.tight_layout()
    save(fig, out, "figA2_feature_correlation")


def figA3_cross_investor_actions(out: Path) -> None:
    data = np.load(DATASET, allow_pickle=False)
    actions = data["actions"]
    pairs = [(0, 2, "외국인", "개인"), (1, 2, "기관", "개인"), (0, 1, "외국인", "기관")]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
    for ax, (i, j, li, lj) in zip(axes, pairs, strict=True):
        r = np.corrcoef(actions[:, i], actions[:, j])[0, 1]
        ax.scatter(actions[:, i], actions[:, j], s=5, alpha=0.28, color="#0072B2", linewidths=0)
        ax.plot([-1.02, 1.02], [1.02, -1.02], color="#D96A68", lw=1.1, ls="--")
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.02, 1.02)
        ax.set_xlabel(li)
        ax.set_ylabel(lj)
        ax.set_title(f"r = {r:.3f}", fontsize=10)
    fig.suptitle("동일 거래일 유형 간 행동 (점선은 완전 상쇄선)", fontsize=12)
    fig.tight_layout()
    save(fig, out, "figA3_cross_investor_actions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper figures")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    setup_style()
    fig1_reward_weights_bar(out)
    fig2_actual_vs_prediction(out)
    fig3_beta_alpha_forest(out)
    fig4_training_convergence(out)
    fig5_oos_performance(out)
    fig6_walk_forward(out)
    fig7_institution_null(out)
    figA1_no_saturation(out)
    figA2_feature_correlation(out)
    figA3_cross_investor_actions(out)
    print(f"wrote figures to {out.resolve()}")


if __name__ == "__main__":
    main()
