#!/usr/bin/env python3
"""Fail when a tracked Markdown file links to a repository path that does not exist.

Only relative links are checked; external URLs and mail links are skipped. A
`#fragment` on a link to a Markdown file (or on a same-page link) must match a
heading in that file, using GitHub's anchor rules.
"""

from __future__ import annotations

import re
import subprocess
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Links inside packs/ are resolved by the pack's own build, not this repository.
SKIP_PREFIXES = ("packs/", "CHANGELOG.md")
LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`[^`]*`")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
EXPLICIT_ANCHOR = re.compile(r"""<a\s+(?:name|id)=["']([^"']+)["']""")


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


def slug(heading: str) -> str:
    """GitHub's heading anchor: lowercase, drop punctuation, spaces become '-'."""
    text = re.sub(r"<[^>]+>", "", heading)  # inline HTML
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links keep their text
    text = text.replace("`", "").lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


@cache
def anchors(path: Path) -> frozenset[str]:
    found: set[str] = set()
    seen: dict[str, int] = {}
    in_fence = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        found.update(EXPLICIT_ANCHOR.findall(line))
        m = HEADING.match(line)
        if not m:
            continue
        base = slug(m.group(2))
        n = seen.get(base, 0)
        seen[base] = n + 1
        found.add(base if n == 0 else f"{base}-{n}")
    return frozenset(found)


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
            if re.match(r"^[a-z][a-z0-9+.-]*:", target):
                continue
            file_part, _, fragment = target.partition("#")
            file_part = file_part.split("?", 1)[0]
            if file_part:
                resolved = (
                    (ROOT / file_part.lstrip("/"))
                    if file_part.startswith("/")
                    else (path.parent / file_part)
                )
                if not resolved.exists():
                    broken.append((number, target))
                    continue
            else:
                resolved = path
            if resolved.is_dir():
                resolved = resolved / "README.md"
            if fragment and resolved.suffix == ".md" and resolved.is_file():
                if fragment.lower() not in anchors(resolved.resolve()):
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
