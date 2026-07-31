"""원고를 단단(single-column) 플로트 판형으로 변환한다 — 8쪽 맞추기용.

§4 본문·수치·판정 문장은 건드리지 않는다. 바꾸는 것은 플로트 구조뿐이다:

  1. \\figorbox 를 \\linewidth 기반으로  (단단 전환 시 넘침 방지)
  2. Fig actual-vs-pred : figure* -> figure, 단단 70mm, 새 이미지
  3. Fig beta-stability : figure* -> figure, 단단 90mm, alpha 패널 흡수한 새 이미지
  4. Fig alpha-context  : 삭제 (3에 흡수), \\ref 는 beta-stability 로 연결
  5. Table alpha        : 삭제, 6행을 Table beta 하단에 흡수, \\ref 는 tab:beta 로 연결

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
#   V3       없음. scripts/train_continuous_walk_forward.py 가 모델 생성 시
#            context_main_effect 를 전달하지 않아 walk-forward 는 alpha 를 아예
#            추정하지 않는다(2026-07-30 부트스트랩 버그 수정이 이 스크립트에는
#            적용되지 않음). 따라서 V3 열은 '---' 로 둔다. 추측값을 넣지 말 것.
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


def _find_env(text: str, env: str, label: str) -> tuple[int, int]:
    pat = re.compile(r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}", re.S)
    for m in pat.finditer(text):
        if label in m.group(0):
            return m.start(), m.end()
    raise SystemExit(f"not found: {env} with {label}")


def apply(text: str) -> str:
    # 1. figorbox -> linewidth
    text = text.replace(r"\includegraphics[width=\textwidth]", r"\includegraphics[width=\linewidth]")
    text = text.replace(r"\dimexpr\textwidth-2\fboxsep", r"\dimexpr\linewidth-2\fboxsep")

    # 4/5. alpha 그림·표 블록 제거 (먼저 제거해야 뒤 인덱스가 안 흔들린다)
    for env, label in (("figure*", "fig:alpha"), ("table*", "tab:alpha")):
        s, e = _find_env(text, env, label)
        text = text[:s] + text[e:]
    text = text.replace(r"\ref{fig:alpha}", r"\ref{fig:beta-stability}")
    text = text.replace(r"\ref{tab:alpha}", r"\ref{tab:beta}")

    # 5. alpha 6행을 Table 2 하단(\bottomrule 직전)에 삽입
    s, e = _find_env(text, "table*", "tab:beta")
    block = text[s:e]
    if r"\bottomrule" not in block:
        raise SystemExit("tab:beta has no bottomrule")
    block = block.replace(r"\bottomrule", ALPHA_ROWS + r"\bottomrule", 1)
    text = text[:s] + block + text[e:]

    # 2/3. 두 그림을 단단으로
    for label, newfig, height, caption in (
        ("fig:actual-vs-pred", "fig_actual_vs_pred_1col", "70mm", None),
        ("fig:beta-stability", "fig_weights_1col", "90mm", FIG_WEIGHTS_CAPTION),
    ):
        s, e = _find_env(text, "figure*", label)
        block = text[s:e]
        block = block.replace(r"\begin{figure*}", r"\begin{figure}").replace(
            r"\end{figure*}", r"\end{figure}"
        )
        block = re.sub(r"\\figorbox\{[^}]+\}\{[^}]+\}", rf"\\figorbox{{{newfig}}}{{{height}}}", block)
        if caption:
            block = re.sub(
                r"\\caption\{.*?\}\s*\\Description\{.*?\}",
                lambda _m: caption, block, flags=re.S,
            )
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
