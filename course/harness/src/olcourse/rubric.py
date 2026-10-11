"""Self-graded rubrics: course/rubrics/<name>.md (DESIGN 3.2, 5.5).

A rubric file is markdown whose checklist lines (`- [ ] ...`) are the items.
`ask` prints each item and reads y or n: from the terminal when stdin is a
TTY, otherwise one answer per line from stdin (so a script or a test can
pipe `y\\ny\\n`). End of input means "not graded now", never "yes".
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from . import HarnessError

ITEM = re.compile(r"^\s*[-*]\s+\[[ xX]?\]\s+(.+?)\s*$")


def path(course: Path, name: str) -> Path:
    return course / "rubrics" / f"{name}.md"


def items(course: Path, name: str) -> list[str]:
    p = path(course, name)
    if not p.is_file():
        raise HarnessError(f"no rubric {name!r} at {p}")
    out = [m.group(1) for line in p.read_text().splitlines() if (m := ITEM.match(line))]
    if not out:
        raise HarnessError(f"{p}: no `- [ ] item` lines")
    return out


def _read_answer(stream, prompt: str, tty: bool) -> bool | None:
    while True:
        if tty:
            try:
                raw = input(prompt)
            except EOFError:
                return None
        else:
            print(prompt, end="", flush=True)
            raw = stream.readline()
            if raw == "":
                print()
                return None
            print(raw.strip())
        a = raw.strip().lower()
        if a in ("y", "yes"):
            return True
        if a in ("n", "no"):
            return False
        if not tty:
            return None  # a malformed piped answer is not a yes


def ask(lines: list[str], indent: str = "    ", stream=None) -> list[bool] | None:
    """Ask each line; None when input ends before every line is answered."""
    stream = stream or sys.stdin
    tty = stream.isatty() if hasattr(stream, "isatty") else False
    out = []
    for i, line in enumerate(lines, 1):
        a = _read_answer(stream, f"{indent}[{i}/{len(lines)}] {line}  y/n? ", tty)
        if a is None:
            return None
        out.append(a)
    return out
