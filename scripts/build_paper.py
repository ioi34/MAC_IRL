"""§1-§3 원고 + 동료 §4/§5 조각 + 레이아웃 패치를 하나의 최종 .tex 로 합친다.

동료가 §4 를 다시 주면 그대로 다시 돌리면 된다. 앵커 문자열이 안 맞으면 조용히
넘어가지 않고 즉시 실패한다 — 동료가 해당 문단을 고쳐 쓴 경우를 놓치지 않기 위해서다.

적용 내용:
  1. 동료 §4/§5 조각을 \\section{Experiments} ~ 참고문헌 직전 자리에 삽입
  2. §5 에 structural recovery 문장 추가 (B안: §3.2·Intro 의 약속 이행)
  3. §4.2 수렴 문단 삭제 (§3.5 의 정확 솔버 결과로 대체됨 — 남겨두면 모순)
  4. scripts/apply_1col_layout.py 로 단단 플로트 변환 (8쪽)

사용:
  python3 scripts/build_paper.py \\
      --base ../acmart-primary/mac_irl_icaif26.tex \\
      --chapter4 <동료 조각.tex> \\
      -o ../acmart-primary/mac_irl_icaif26.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.apply_1col_layout import apply as apply_layout  # noqa: E402

STRUCTURAL_SENTENCE = (
    " A longer step is to relax the one-step horizon: under a genuine planning "
    "horizon the reward is no longer informationally equivalent to the conditional "
    "best response, and $\\beta_i$ would carry the structural reading that the "
    "myopic estimates here deliberately withhold."
)


def _require(text: str, needle: str, what: str) -> None:
    if needle not in text:
        raise SystemExit(f"[build_paper] 앵커를 찾지 못했다 ({what}): {needle[:60]!r}")


def build(base: str, chapter4: str) -> str:
    src = Path(base).read_text()
    ch4 = Path(chapter4).read_text()

    # 2. B안 — §5 에 structural recovery 문장
    anchor = "and to finer institutional categories."
    _require(ch4, anchor, "§5 future work")
    ch4 = ch4.replace(anchor, anchor + STRUCTURAL_SENTENCE, 1)

    # 3. §4.2 수렴 문단 삭제
    start = "The 75-epoch fits also make incomplete optimization"
    end = "reflect each series' dispersion."
    _require(ch4, start, "§4.2 수렴 문단 시작")
    _require(ch4, end, "§4.2 수렴 문단 끝")
    i, j = ch4.index(start), ch4.index(end) + len(end)
    ch4 = ch4[:i] + ch4[j:]

    # 1. 삽입
    _require(src, "\\section{Experiments}", "본문 §4 시작")
    refs = "%% ---------------------------------------------------------------------\n%% References."
    _require(src, refs, "참고문헌 주석")
    merged = src[: src.index("\\section{Experiments}")] + ch4 + "\n\n" + src[src.index(refs) :]

    # 4. 레이아웃
    return apply_layout(merged)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="§1-§3 을 담은 원고 .tex")
    p.add_argument("--chapter4", required=True, help="동료 §4/§5 조각 .tex")
    p.add_argument("-o", "--out", required=True)
    args = p.parse_args()

    out = Path(args.out)
    if out.exists() and out.resolve() == Path(args.base).resolve():
        backup = out.with_suffix(".tex.bak")
        backup.write_text(out.read_text())
        print(f"백업: {backup}")
    out.write_text(build(args.base, args.chapter4))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
