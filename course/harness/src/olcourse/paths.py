"""path.tsv reading with includes and the fifth `check` column (DESIGN 5.15).
Mirrors expand_rows in practice/bin/ol; the bash side stays the authority
for `ol learn`, this side serves `ol next` and `ol verify course` check 11."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import ctx


@dataclass
class Row:
    owner: str
    stage: str
    title: str
    files: str
    when: str
    check: str  # "" or "-" when manual


def paths_dir(course: Path | None = None) -> Path:
    env = os.environ.get("OL_PATHS_DIR")
    return (
        Path(env)
        if env
        else ctx.site_root(course) / "content" / "courses" / "build-an-llm" / "paths"
    )


def expand(name: str, base: Path, stack: tuple[str, ...] = ()) -> list[Row]:
    tsv = base / name / "path.tsv"
    if not tsv.is_file():
        raise FileNotFoundError(f"unknown path {name!r}")
    if name in stack:
        raise ValueError("include cycle: " + " -> ".join(stack + (name,)))
    rows: list[Row] = []
    for line in tsv.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if cols[0].startswith("@"):
            rows += expand(cols[0][1:], base, stack + (name,))
            continue
        cols += [""] * (5 - len(cols))
        rows.append(Row(name, cols[0], cols[1], cols[2], cols[3], cols[4].strip()))
    return rows


def learn_dir(owner: str) -> Path:
    """Where `ol learn` keeps a path's done stages (practice/bin/ol learn_dir):
    course-* paths live in the active learner repo's .ol/learn/."""
    home = ctx.learner_home()
    if owner.startswith("course") and (home / "system.toml").is_file():
        return home / ".ol" / "learn"
    return ctx.scratch() / "learn"


def done_stages(owner: str) -> set[str]:
    p = learn_dir(owner) / f"{owner}.done"
    return set(p.read_text().split()) if p.is_file() else set()


def check_targets(check: str) -> list[tuple[str, str]]:
    """`module:L8.3` -> [("module", "L8.3")]; `all:A,B` -> [("module", A), ("module", B)]."""
    if not check or check == "-":
        return []
    kind, _, rest = check.partition(":")
    if kind == "all":
        return [("module", x) for x in rest.split(",") if x]
    return [(kind, rest)]
