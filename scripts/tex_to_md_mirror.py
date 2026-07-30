"""원고 `.tex`에서 장별 검토용 마크다운 미러를 다시 만든다.

미러를 손으로 고치다 보면 `.tex`와 어긋난다. 원고가 유일한 출처이므로 미러는
매번 생성해서 덮어쓴다. 파일 하단의 ``## Design notes`` 이후 블록은 사람이 관리하는
메모이므로 보존한다.

사용 예::

    python scripts/tex_to_md_mirror.py --section 2 --out docs/paper_ch2_en.md
    python scripts/tex_to_md_mirror.py --section 3 --out docs/paper_ch3_en.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_TEX = "../acmart-primary/mac_irl_icaif26.tex"
KEEP_MARKER = "## Design notes"

SECTION_TITLES = {1: "Introduction", 2: "Related Work", 3: "Method"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", default=DEFAULT_TEX)
    parser.add_argument("--section", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def extract_section(tex: str, number: int) -> str:
    """해당 \\section 부터 다음 \\section 직전까지."""
    titles = list(SECTION_TITLES.values())
    start_title = SECTION_TITLES[number]
    start = tex.index(f"\\section{{{start_title}}}")
    following = [tex.find(f"\\section{{{t}}}", start + 1) for t in titles]
    following += [tex.find("\\section{Experiments}", start + 1)]
    following = [i for i in following if i > 0]
    end = min(following) if following else len(tex)
    return tex[start:end]


def latex_to_markdown(body: str) -> str:
    t = body

    # 그림 환경은 캡션만 남긴다. TikZ 본문을 그대로 옮기면 미러가 읽을 수 없다.
    t = re.sub(
        r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}",
        lambda m: "\n\n**[Figure]** "
        + (
            " ".join(c.group(1).split())
            if (c := re.search(r"\\caption\{(.*?)\}\s*\n\s*\\label", m.group(0), re.S))
            else "(캡션 없음)"
        )
        + "\n\n",
        t,
        flags=re.S,
    )

    # 주석 제거 (\% 는 보존)
    t = re.sub(r"(?<!\\)%.*", "", t)

    # 제목
    t = re.sub(r"\\section\{([^}]*)\}", r"## \1", t)
    t = re.sub(r"\\subsection\{([^}]*)\}", r"### \1", t)
    t = re.sub(r"\\label\{[^}]*\}", "", t)

    # 수식 블록은 $$ 로
    t = re.sub(
        r"\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}",
        lambda m: "\n$$" + " ".join(m.group(1).split()) + "$$\n",
        t,
        flags=re.S,
    )
    t = re.sub(r"\\begin\{aligned\}|\\end\{aligned\}", "", t)

    # 인용·강조
    t = re.sub(r"\\cite\{([^}]*)\}", r"[\1]", t)
    t = re.sub(r"\\ref\{sec:related\}", "2", t)
    t = re.sub(r"\\ref\{sec:method[^}]*\}", "3", t)
    t = re.sub(r"\\ref\{sec:experiments\}", "4", t)
    t = re.sub(r"\\ref\{sec:discussion\}", "5", t)
    t = re.sub(r"\\ref\{[^}]*\}", "?", t)
    t = re.sub(r"\\textbf\{([^{}]*)\}", r"**\1**", t)
    t = re.sub(r"\\emph\{([^{}]*)\}", r"*\1*", t)

    # 목록
    t = re.sub(r"\\begin\{itemize\}|\\end\{itemize\}", "", t)
    t = re.sub(r"\\item\s*", "- ", t)

    # 문장부호
    t = t.replace("---", "—").replace("~", " ")
    t = re.sub(r"\\%", "%", t)
    t = re.sub(r"\\&", "&", t)
    t = re.sub(r"\\,|\\!|\\;", " ", t)

    # 문단 안의 줄바꿈을 하나로 (빈 줄은 유지)
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", t)]
    t = "\n\n".join(p for p in paragraphs if p)
    return t


def main() -> None:
    args = parse_args()
    tex_path = Path(args.tex)
    if not tex_path.is_absolute():
        tex_path = (Path(__file__).resolve().parent.parent / tex_path).resolve()
    tex = tex_path.read_text()

    body = latex_to_markdown(extract_section(tex, args.section))

    out = Path(args.out)
    kept = ""
    if out.exists():
        existing = out.read_text()
        if KEEP_MARKER in existing:
            kept = "\n\n---\n\n" + existing[existing.index(KEEP_MARKER):]

    title = SECTION_TITLES[args.section]
    header = (
        f"# Section {args.section} — {title} (English, for review and editing)\n\n"
        f"> **Generated from `{tex_path.name}` — do not hand-edit the prose above the design notes.**\n"
        f"> Regenerate with `python scripts/tex_to_md_mirror.py --section {args.section} "
        f"--out {args.out}` after changing the manuscript.\n\n---\n\n"
    )
    out.write_text(header + body.strip() + kept + "\n")
    print(f"wrote {out}  ({len(body.split())} words of prose)")


if __name__ == "__main__":
    main()
