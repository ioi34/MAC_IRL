"""개정본 LaTeX 표 본문을 결과 CSV에서 직접 생성 (전사 오류 방지).

산출: paper-icaif26/sections/generated/*.tex
- tab3_body.tex        본문 표 3 (3특징 vs 4특징)
- tabA1_body.tex       부록 A1 (Adam vs 정확해, 33항 전부)
- tabA2_body.tex       부록 A2 (청산 유도 성분 vs 관측)
- tabA2b_body.tex      부록 A2b (합성 시뮬레이션 95% 대역)
- tabA3_body.tex       부록 A3 (통일 λ)
- tabA4_body.tex       부록 A4 (ρ × 모멘텀 창 민감도)
- tab1_rows.tex        표 1에 추가할 4특징 행
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RUNS = Path("runs/revision_20260727")
OUT = Path("paper-icaif26/sections/generated")
OUT.mkdir(parents=True, exist_ok=True)

KO_INV = {"foreign": "외국인", "institution": "기관", "retail": "개인"}
KO_TERM = {
    "beta:momentum": "$\\beta^{mom}$",
    "beta:herd": "$\\beta^{herd}$",
    "beta:underwater": "$\\beta^{uw}$",
    "beta:lag_own": "$\\beta^{lag}$",
    "alpha:kospi_return_1d": "$\\alpha^{KOSPI}$",
    "alpha:fx_level_z_252": "$\\alpha^{FX}$",
    "B:momentumxkospi_return_1d": "$B$ mom$\\times$KOSPI",
    "B:momentumxfx_level_z_252": "$B$ mom$\\times$FX",
    "B:herdxkospi_return_1d": "$B$ herd$\\times$KOSPI",
    "B:herdxfx_level_z_252": "$B$ herd$\\times$FX",
    "B:underwaterxkospi_return_1d": "$B$ uw$\\times$KOSPI",
    "B:underwaterxfx_level_z_252": "$B$ uw$\\times$FX",
}


def f4(x):
    return f"${x:+.4f}$"


def pct(x):
    return f"{100 * x:.1f}"


def ci(lo, hi, star=True):
    mark = "$^{*}$" if (star and (lo > 0 or hi < 0)) else ""
    return f"$[{lo:+.4f},\\,{hi:+.4f}]${mark}"


# ---------------------------------------------------------------- 표 3
def table3():
    s = pd.read_csv(RUNS / "task1_cpcv_coefficients_summary.csv")
    b4 = pd.read_csv(RUNS / "task1_month_bootstrap.csv")
    b3 = pd.read_csv("runs/protocol_validation/calendar_month_bootstrap.csv")

    def row3(inv, term):
        r = s[(s.spec == "feat3") & (s.investor == inv) & (s.term == term)]
        if r.empty:
            return None
        c = b3[(b3.investor == inv) & (b3.term == term)]
        return float(r["mean"].iloc[0]), float(r.sign_consistency.iloc[0]), \
            (float(c.ci_lower.iloc[0]), float(c.ci_upper.iloc[0])) if len(c) else None

    def row4(inv, term):
        r = s[(s.spec == "feat4_main") & (s.investor == inv) & (s.term == term)]
        c = b4[(b4.spec == "feat4_main") & (b4.investor == inv) & (b4.term == term)]
        return float(r["mean"].iloc[0]), float(r.sign_consistency.iloc[0]), \
            (float(c.ci_lower.iloc[0]), float(c.ci_upper.iloc[0]))

    order = [
        ("foreign", "beta:momentum", "생존"),
        ("retail", "beta:momentum", "생존"),
        ("foreign", "alpha:fx_level_z_252", "생존"),
        ("retail", "alpha:fx_level_z_252", "생존"),
        ("foreign", "alpha:kospi_return_1d", "축소, CI 탈락"),
        ("retail", "alpha:kospi_return_1d", "\\textbf{부호 반전}"),
        ("institution", "alpha:kospi_return_1d", "강화, CI 통과"),
        ("foreign", "beta:herd", "생존"),
        ("institution", "beta:herd", "축소"),
        ("retail", "beta:herd", "생존"),
        ("foreign", "beta:underwater", "CI 미통과 유지"),
        ("retail", "beta:underwater", "약화"),
        ("institution", "beta:underwater", "0으로 소거"),
        ("foreign", "beta:lag_own", "(통제항)"),
        ("institution", "beta:lag_own", "(통제항)"),
        ("retail", "beta:lag_own", "(통제항)"),
    ]
    lines = []
    for inv, term, verdict in order:
        r3 = row3(inv, term)
        m4, sc4, c4 = row4(inv, term)
        if r3 is None:
            c3s = "---"
            m3s = "---"
        else:
            m3, sc3, c3 = r3
            m3s = f"{f4(m3)} ({pct(sc3)}\\%)"
            c3s = ci(*c3)
        lines.append(
            f"{KO_INV[inv]} & {KO_TERM[term]} & {m3s} & {c3s} & "
            f"{f4(m4)} ({pct(sc4)}\\%) & {ci(*c4)} & {verdict} \\\\"
        )
    (OUT / "tab3_body.tex").write_text("\n".join(lines) + "%\n", encoding="utf-8")


# ---------------------------------------------------------------- 표 1 추가 행
def table1_rows():
    m = pd.read_csv(RUNS / "task1_cpcv_metrics_summary.csv")
    invs = ["foreign", "institution", "retail"]

    def get(spec, col):
        return [float(m[(m.spec == spec) & (m.investor == i)][col].iloc[0]) for i in invs]

    da = get("feat4_main", "direction_accuracy")
    ds = get("feat4_main", "direction_accuracy_std")
    co = get("feat4_main", "correlation")
    cs = get("feat4_main", "correlation_std")
    ma = get("feat4_main", "mae")
    sat = get("feat4_main", "saturation_rate")
    lines = [
        "방향 정확도 & " + " & ".join(f"${d:.3f}\\pm{s:.3f}$" for d, s in zip(da, ds)) + " \\\\",
        "상관계수 & " + " & ".join(f"${c:.3f}\\pm{s:.3f}$" for c, s in zip(co, cs)) + " \\\\",
        "MAE & " + " & ".join(f"${x:.3f}$" for x in ma) + " \\\\",
        "포화 비율 & " + " & ".join(f"${x:.3f}$" for x in sat) + " \\\\",
    ]
    (OUT / "tab1_rows.tex").write_text("\n".join(lines) + "%\n", encoding="utf-8")


# ---------------------------------------------------------------- 부록 A1
def tableA1():
    t = pd.read_csv(RUNS / "task0_adam_vs_exact_by_term.csv")
    order_inv = ["foreign", "institution", "retail"]
    order_term = ["beta:momentum", "beta:herd", "beta:underwater",
                  "B:momentumxkospi_return_1d", "B:momentumxfx_level_z_252",
                  "B:herdxkospi_return_1d", "B:herdxfx_level_z_252",
                  "B:underwaterxkospi_return_1d", "B:underwaterxfx_level_z_252",
                  "alpha:kospi_return_1d", "alpha:fx_level_z_252"]
    lines = []
    for inv in order_inv:
        for j, term in enumerate(order_term):
            r = t[(t.investor == inv) & (t.term == term)].iloc[0]
            head = KO_INV[inv] if j == 0 else ""
            lines.append(
                f"{head} & {KO_TERM[term]} & {f4(r.adam_mean)} ({pct(r.adam_sign_consistency)}) & "
                f"{f4(r.exact_mean)} ({pct(r.exact_sign_consistency)}) & {r.diff_mean:+.4f} & "
                f"{pct(r.exact_zero_rate)} \\\\"
            )
        if inv != order_inv[-1]:
            lines.append("\\addlinespace")
    (OUT / "tabA1_body.tex").write_text("\n".join(lines) + "%\n", encoding="utf-8")


# ---------------------------------------------------------------- 부록 A2
def tableA2():
    t = pd.read_csv(RUNS / "task2_induced_vs_observed.csv")
    ko = {"beta:momentum": "$\\beta^{mom}$", "alpha:kospi_return_1d": "$\\alpha^{KOSPI}$",
          "alpha:fx_level_z_252": "$\\alpha^{FX}$", "beta:herd": "$\\beta^{herd}$"}
    lines = []
    for term in ["beta:momentum", "alpha:kospi_return_1d", "alpha:fx_level_z_252", "beta:herd"]:
        g = t[t.term == term]
        f_obs = float(g["observed_foreign(adam)"].iloc[0])
        r_obs = float(g["observed_retail(adam)"].iloc[0])
        ind = g["induced_retail(adam)"]
        share = g["share_of_observed_pct(adam)"]
        lo, hi = (ind.min(), ind.max()) if f_obs > 0 else (ind.min(), ind.max())
        s_lo, s_hi = share.min(), share.max()
        note = "해당 없음$^{\\dagger}$" if term == "beta:herd" else f"{s_lo:.0f}--{s_hi:.0f}"
        lines.append(
            f"{ko[term]} & {f4(f_obs)} & ${lo:+.4f}\\sim{hi:+.4f}$ & {f4(r_obs)} & {note} \\\\"
        )
    (OUT / "tabA2_body.tex").write_text("\n".join(lines) + "%\n", encoding="utf-8")

    q = pd.read_csv(RUNS / "task2_null_sim_summary.csv")
    q = q[q.spec == "feat3"]
    sem = pd.read_csv(RUNS / "task2_seminull_summary.csv")
    lines = []
    for inv in ["foreign", "retail"]:
        for j, term in enumerate(["beta:momentum", "alpha:kospi_return_1d",
                                  "alpha:fx_level_z_252", "beta:herd"]):
            n = q[(q.investor == inv) & (q.term == term)].iloc[0]
            s = sem[(sem.investor == inv) & (sem.term == term)].iloc[0]
            head = KO_INV[inv] if j == 0 else ""
            lines.append(
                f"{head} & {KO_TERM[term]} & $[{n.q2_5:+.4f},\\,{n.q97_5:+.4f}]$ & "
                f"$[{s.q2_5:+.4f},\\,{s.q97_5:+.4f}]$ \\\\"
            )
        if inv == "foreign":
            lines.append("\\addlinespace")
    (OUT / "tabA2b_body.tex").write_text("\n".join(lines) + "%\n", encoding="utf-8")


# ---------------------------------------------------------------- 부록 A3
def tableA3():
    u = pd.read_csv(RUNS / "task4_unified_lambda_summary.csv")
    u = u[u.spec == "feat3"]
    order = [("foreign", "beta:momentum"), ("retail", "beta:momentum"),
             ("foreign", "alpha:kospi_return_1d"), ("retail", "alpha:kospi_return_1d"),
             ("foreign", "alpha:fx_level_z_252"), ("retail", "alpha:fx_level_z_252"),
             ("foreign", "beta:herd"), ("institution", "beta:herd"), ("retail", "beta:herd"),
             ("foreign", "beta:underwater"), ("retail", "beta:underwater"),
             ("institution", "beta:momentum"), ("institution", "beta:underwater")]
    lines = []
    for inv, term in order:
        cells = []
        for lam in [0.0, 0.0005, 0.01]:
            r = u[(u.investor == inv) & (u.term == term) & (u.unified_lambda == lam)].iloc[0]
            cells.append(f"{f4(r['mean'])} ({pct(r.sign_consistency)})")
        lines.append(f"{KO_INV[inv]} & {KO_TERM[term]} & " + " & ".join(cells) + " \\\\")
    (OUT / "tabA3_body.tex").write_text("\n".join(lines) + "%\n", encoding="utf-8")


# ---------------------------------------------------------------- 부록 A4
def tableA4():
    t = pd.read_csv(RUNS / "task5_sensitivity_summary.csv")
    lines = []
    for rho in [0.95, 0.98, 0.99]:
        for j, w in enumerate([10, 20, 60]):
            g = t[(t.rho == rho) & (t.momentum_window == w)]
            def cell(inv, term):
                r = g[(g.investor == inv) & (g.term == term)].iloc[0]
                return f"{f4(r['mean'])} ({100 * r.sign_consistency:.0f})"
            head = f"{rho}" if j == 0 else ""
            star = "$^{\\ddagger}$" if (rho, w) == (0.98, 20) else ""
            lines.append(
                f"{head} & {w}{star} & {cell('foreign','beta:momentum')} & "
                f"{cell('retail','beta:momentum')} & {cell('retail','beta:underwater')} & "
                f"{cell('foreign','beta:underwater')} \\\\"
            )
        if rho != 0.99:
            lines.append("\\addlinespace")
    (OUT / "tabA4_body.tex").write_text("\n".join(lines) + "%\n", encoding="utf-8")


# ---------------------------------------------------------------- 부록 F (명제 1.1)
def tableF():
    d = pd.read_csv(RUNS / "prop11_feasibility_uniqueness.csv")
    s = pd.read_csv(RUNS / "prop11_global_search.csv")
    lines = []
    for inv in ["foreign", "institution", "retail"]:
        g = d[d.investor == inv]
        gs = s[s.investor == inv]
        uniq = "rank$(Z)=11$" if g["lambda"].iloc[0] == 0 else \
               f"rank$(Z_E)=|E|$ ({int(g.equicorr_size.max())})"
        lines.append(
            f"{KO_INV[inv]} & {g.max_abs_q_train.max():.3f} & {g.cond.max():.2f} & {uniq} & "
            f"{g.F_hat.max():.4f} & {g.k_star_share_pct.max():.1f} & "
            f"{g.full_saturation_lower_bound.min():.3f} & {int(gs.beaten.sum())}/{len(gs)} \\\\"
        )
    (OUT / "tabF_body.tex").write_text("\n".join(lines) + "%\n", encoding="utf-8")


# ---------------------------------------------------------------- 4쪽본 표
def table_4p():
    """표 2: 안정 계수의 V5a 무선호·V5b 조건부 청산 대조 (정확해, 3특징).

    4쪽본은 (F2)를 '유형 간 시차 교차연관'으로 부르므로 라벨을 beta^{cross}로 쓴다."""
    KO_TERM["beta:herd"] = "$\\beta^{cross}$"
    c = pd.read_csv(RUNS / "task1_cpcv_coefficients_summary.csv")
    c = c[c.spec == "feat3"]
    n = pd.read_csv(RUNS / "task2_null_sim_summary.csv")
    n = n[n.spec == "feat3"]
    cn = pd.read_csv(RUNS / "task2_conditional_null_summary.csv")
    b = pd.read_csv("runs/protocol_validation/calendar_month_bootstrap.csv")
    rows = [
        ("beta:momentum", "foreign", "V5a"),
        ("alpha:kospi_return_1d", "foreign", "V5a"),
        ("alpha:fx_level_z_252", "foreign", "V5a"),
        ("beta:momentum", "retail", "V5b"),
        ("alpha:kospi_return_1d", "retail", "V5b"),
        ("alpha:fx_level_z_252", "retail", "V5b"),
        ("beta:herd", "foreign", "V5a"),
        ("beta:herd", "retail", "V5a"),
    ]
    lines = []
    for k, (term, inv, test) in enumerate(rows):
        g = c[(c.investor == inv) & (c.term == term)].iloc[0]
        band = (
            cn[(cn.investor == inv) & (cn.term == term)].iloc[0]
            if test == "V5b"
            else n[(n.investor == inv) & (n.term == term)].iloc[0]
        )
        bb = b[(b.investor == inv) & (b.term == term)].iloc[0]
        ci_ok = bool(bb.excludes_zero)
        outside = bool(g["mean"] < band.q2_5 or g["mean"] > band.q97_5)
        if term == "beta:herd":
            verdict = "기계 성분" if not outside else "대역 밖"
        elif test == "V5b":
            verdict = "\\textbf{잔여 반응}" if outside else "청산 대역 내"
        else:
            verdict = "\\textbf{고유 반응}" if ci_ok and outside else "---"
        lines.append(
            f"{KO_TERM[term]} & {KO_INV[inv]} & {f4(g['mean'])} ({pct(g.sign_consistency)}) & "
            f"{'예' if ci_ok else '아니오'} & {test} & "
            f"$[{band.q2_5:+.4f},\\,{band.q97_5:+.4f}]$ & "
            f"{'예' if outside else '아니오'} & {verdict} \\\\"
        )
        if k in (2, 5):
            lines.append("\\addlinespace")
    (OUT / "tab4p_body.tex").write_text("\n".join(lines) + "%\n", encoding="utf-8")
    print("  [4p] V5a/V5b 핵심 8행 작성")


if __name__ == "__main__":
    table_4p()
    table3()
    table1_rows()
    tableA1()
    tableA2()
    tableA3()
    tableA4()
    print("wrote", sorted(p.name for p in OUT.glob("*.tex")))
