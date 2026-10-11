"""Keep each solve or proof chapter's problem list identical to course/solve/<S-ID>/problems.md.

DESIGN 5.5: the problems live in `problems.md` and are also included in the
chapter. A chapter holds them between

    <!-- ol:problems S-M05 -->
    ...
    <!-- /ol:problems -->

and this script rewrites that block from `problems.md` (everything after its
first `### ` heading, so the chapter keeps its own introduction and answer
format notes). `--check` exits 1 when a chapter is out of date.

    uv run --project course/harness python course/oracle/solve/sync_chapters.py [--check]
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

COURSE = Path(__file__).resolve().parents[2]
ROOT = COURSE.parent


def body(sid: str) -> str:
    text = (COURSE / "solve" / sid / "problems.md").read_text()
    i = text.find("\n### ")
    if i < 0:
        raise SystemExit(f"{sid}: problems.md has no `### ` section")
    return text[i + 1 :].rstrip("\n")


def main(argv: list[str]) -> int:
    check = "--check" in argv
    stale = 0
    # Solve sets and the proof modules that keep a problems.md (review.*,
    # ethics.06, iv.01): the src.is_file() check below skips the rest.
    mods = sorted(
        p
        for p in (COURSE / "modules").glob("*.toml")
        if (COURSE / "solve" / p.stem / "problems.md").is_file()
    )
    for mod in mods:
        reg = tomllib.loads(mod.read_text())
        sid, chap = reg["id"], reg.get("chapter", "")
        src = COURSE / "solve" / sid / "problems.md"
        if not chap or not src.is_file():
            continue
        path = ROOT / chap
        text = path.read_text() if path.is_file() else ""
        start, end = f"<!-- ol:problems {sid} -->", "<!-- /ol:problems -->"
        if start not in text or end not in text:
            continue  # a chapter that does not include its problems this way
        new = re.sub(
            re.escape(start) + r".*?" + re.escape(end),
            lambda _: f"{start}\n\n{body(sid)}\n\n{end}",
            text,
            flags=re.S,
        )
        if new != text:
            stale += 1
            if check:
                print(f"{chap}: problems differ from {src.relative_to(ROOT)}")
            else:
                path.write_text(new)
                print(f"{chap}: problems synced")
    if check and not stale:
        print("solve chapters: problems in sync")
    return 1 if (check and stale) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
