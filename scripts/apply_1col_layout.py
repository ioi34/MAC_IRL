"""원고를 단단(single-column) 플로트 판형으로 변환한다 — 8쪽 맞추기용.

§4 본문·수치·판정 문장은 건드리지 않는다. 바꾸는 것은 플로트 구조뿐이다:

  1. \\figorbox 를 \\linewidth 기반으로  (단단 전환 시 넘침 방지)
  2. Fig actual-vs-pred : figure* -> figure, 단단 70mm, 새 이미지
  3. Fig beta-stability : figure* -> figure, 단단 90mm, alpha 패널 흡수한 새 이미지
  4. Fig context-effects: 삭제 (3에 흡수), \\ref 는 병합 그림으로 연결
  5. Table context-main : 삭제, 6행을 reward-weights 표 하단에 흡수

라벨은 구 §4(fig:beta-stability 등)와 신 §4(fig:reward-validation 등) 양쪽을 지원한다.
그림 삽입도 \\figorbox / \\includegraphics 양쪽을 처리한다.

동료가 §4를 수정한 뒤에도 그대로 다시 돌리면 된다 (멱등하지 않으므로 항상 원본에서 실행).

사용:
  python3 scripts/apply_1col_layout.py paper/mac_irl_icaif26.tex -o /tmp/patched.tex
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# Table 2 하단에 붙일 alpha 6행. 값 출처:
#   mean/CI  runs/continuous_reward3_persist_epochs75_validation/
#            weight_bootstrap/bootstrap_context_main_weights_summary.csv
#   V1       runs/continuous_reward3_persist_epochs75/context_main_weights_summary.csv
#            (direction_consistency)
#   V3       runs/.../walk_forward_alphafix/context_main_weights_summary.csv
#            (2026-07-31 버그 수정 후 재실행분. CPCV 평균 부호를 유지한 창의 수)
ALPHA_ROWS = r"""\midrule
\multicolumn{6}{@{}l}{\emph{Direct context effects} $\alpha$ (bootstrap mean and 95\% interval)}\\
Foreign       & KOSPI 200 return & $+0.0325$ $[+0.0178, +0.0483]$ & 100.0\% & Yes & --- \\
              & FX level         & $-0.0225$ $[-0.0479, -0.0002]$ & 97.8\%  & Yes & --- \\
Institutional & KOSPI 200 return & $-0.0061$ $[-0.0119, -0.0002]$ & 100.0\% & Yes & --- \\
              & FX level         & $-0.0038$ $[-0.0122, +0.0003]$ & 97.8\%  & No  & --- \\
Retail        & KOSPI 200 return & $-0.0217$ $[-0.0424, -0.0059]$ & 100.0\% & Yes & --- \\
              & FX level         & $+0.0322$ $[+0.0017, +0.0643]$ & 100.0\% & Yes & --- \\
"""

FIG_WEIGHTS_CAPTION = r"""\caption{Recovered weights and their sample stability. Top: reward weights
$\beta$, showing CPCV means with one standard deviation. Bottom: direct
market-context effects $\alpha$, showing calendar-month block-bootstrap means
with 95\% intervals. Filled markers indicate that the V2 interval excludes
zero; open markers indicate that it includes zero.}
\Description{Two stacked panels of horizontal point-and-interval plots, grouped by investor type: reward weights above and direct context effects below.}"""


# 신 §4(동료판)는 Table 2 가 7열이다: Investor | Feature | beta | V1 | V2 | Identified | V3
ALPHA_ROWS_7 = r"""\midrule
\multicolumn{7}{@{}l}{\emph{Direct context effects} $\alpha$ (bootstrap mean [95\% interval])}\\
Foreign       & KOSPI return & $+0.0325\ [+0.0178, +0.0483]$ & 100.0\% & Yes & Yes & 3/3 \\
              & FX level     & $-0.0225\ [-0.0479, -0.0002]$ & 97.8\%  & Yes & Yes & 2/3 \\
Institutional & KOSPI return & $-0.0061\ [-0.0119, -0.0002]$ & 100.0\% & Yes & Yes & 3/3 \\
              & FX level     & $-0.0038\ [-0.0122, +0.0003]$ & 97.8\%  & No  & No  & 3/3 \\
Retail        & KOSPI return & $-0.0217\ [-0.0424, -0.0059]$ & 100.0\% & Yes & Yes & 2/3 \\
              & FX level     & $+0.0322\ [+0.0017, +0.0643]$ & 100.0\% & Yes & Yes & 3/3 \\
"""

CAPTION_NEW = r"""\caption{Recovered weights and their validation. Top: reward weights $\beta$,
showing CPCV means with one standard deviation. Bottom: direct market-context
effects $\alpha$, showing calendar-month block-bootstrap means with 95\%
intervals. Filled markers are identified by V1+V2; open markers are not
identified.}"""

LAYOUTS = {
    "new": {
        "table": "tab:reward-weights",
        "alpha_rows": ALPHA_ROWS_7,
        "drop": (("figure*", "fig:context-effects"), ("table*", "tab:context-main")),
        "refmap": {"fig:context-effects": "fig:reward-validation",
                   "tab:context-main": "tab:reward-weights"},
        "figs": (("fig:actual-prediction-timeseries", "fig_actual_vs_pred_1col", "70mm", None),
                 ("fig:reward-validation", "fig_weights_1col", "90mm", CAPTION_NEW)),
    },
    "old": {
        "table": "tab:beta",
        "alpha_rows": ALPHA_ROWS,
        "drop": (("figure*", "fig:alpha"), ("table*", "tab:alpha")),
        "refmap": {"fig:alpha": "fig:beta-stability", "tab:alpha": "tab:beta"},
        "figs": (("fig:actual-vs-pred", "fig_actual_vs_pred_1col", "70mm", None),
                 ("fig:beta-stability", "fig_weights_1col", "90mm", FIG_WEIGHTS_CAPTION)),
    },
}


def _find_env(text: str, env: str, label: str) -> tuple[int, int]:
    pat = re.compile(r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}", re.S)
    for m in pat.finditer(text):
        if label in m.group(0):
            return m.start(), m.end()
    raise SystemExit(f"not found: {env} with {label}")


def _set_figure(block: str, newfig: str, height: str) -> str:
    """\figorbox / \includegraphics 어느 형식이든 단단용 figorbox 로 바꾼다."""
    repl = rf"\\figorbox{{{newfig}}}{{{height}}}"
    if r"\figorbox" in block:
        return re.sub(r"\\figorbox\{[^}]+\}\{[^}]+\}", repl, block)
    return re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]+\}", repl, block)


def apply(text: str) -> str:
    layout = LAYOUTS["new" if "tab:reward-weights" in text else "old"]

    # 1. figorbox 를 linewidth 기반으로
    text = text.replace(r"\includegraphics[width=\textwidth]", r"\includegraphics[width=\linewidth]")
    text = text.replace(r"\dimexpr\textwidth-2\fboxsep", r"\dimexpr\linewidth-2\fboxsep")

    # 2. alpha 그림·표 블록 제거 (먼저 제거해야 뒤 인덱스가 안 흔들린다)
    for env, label in layout["drop"]:
        s, e = _find_env(text, env, label)
        text = text[:s] + text[e:]
    for old, new in layout["refmap"].items():
        text = text.replace(rf"\ref{{{old}}}", rf"\ref{{{new}}}")

    # 3. alpha 행을 reward-weights 표 하단(\bottomrule 직전)에 삽입
    s, e = _find_env(text, "table*", layout["table"])
    block = text[s:e]
    if r"\bottomrule" not in block:
        raise SystemExit(f"{layout['table']} has no bottomrule")
    block = block.replace(r"\bottomrule", layout["alpha_rows"] + r"\bottomrule", 1)
    text = text[:s] + block + text[e:]

    # 4. 두 그림을 단단으로
    for label, newfig, height, caption in layout["figs"]:
        s, e = _find_env(text, "figure*", label)
        block = text[s:e]
        block = block.replace(r"\begin{figure*}", r"\begin{figure}").replace(
            r"\end{figure*}", r"\end{figure}")
        block = _set_figure(block, newfig, height)
        if caption:
            block = re.sub(r"\\caption\{.*?\}(\s*\\Description\{.*?\})?",
                           lambda _m: caption, block, count=1, flags=re.S)
        text = text[:s] + block + text[e:]

    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("-o", "--out", required=True)
    args = p.parse_args()
    Path(args.out).write_text(apply(Path(args.source).read_text()))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
