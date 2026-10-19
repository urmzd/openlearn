#!/usr/bin/env python3
"""Fail when a tracked Markdown file links to a repository path that does not exist.

Only relative links are checked. External URLs, mail links, and same-page
anchors are skipped, and the `#fragment` of a file link is ignored.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Links inside packs/ are resolved by the pack's own build, not this repository.
SKIP_PREFIXES = ("packs/", "CHANGELOG.md")
LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`[^`]*`")


def tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    paths = [ROOT / p for p in out.split() if not p.startswith(SKIP_PREFIXES)]
    return [p for p in paths if p.exists()]


def broken_links(path: Path) -> list[tuple[int, str]]:
    broken = []
    in_fence = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # Links inside inline code are examples, not links.
        for target in LINK.findall(INLINE_CODE.sub("", line)):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
                continue
            file_part = target.split("#", 1)[0].split("?", 1)[0]
            if not file_part:
                continue
            resolved = (
                (ROOT / file_part.lstrip("/"))
                if file_part.startswith("/")
                else (path.parent / file_part)
            )
            if not resolved.exists():
                broken.append((number, target))
    return broken


def main() -> int:
    failures = 0
    for path in tracked_markdown():
        for number, target in broken_links(path):
            print(f"{path.relative_to(ROOT)}:{number}: broken link {target}")
            failures += 1
    if failures:
        print(f"{failures} broken link(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
