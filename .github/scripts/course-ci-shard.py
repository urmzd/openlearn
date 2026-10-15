#!/usr/bin/env python3
"""Print one disjoint nightly module shard, one id per line."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULES = ROOT / "course" / "modules"
LAYERS = {"math", "spine", "systems", "operations"}


def layer_of(module: dict) -> str:
    chapter = module.get("chapter", "")
    chapter = chapter.removeprefix("content/tracks/").removeprefix("content/")
    if chapter.startswith("courses/build-an-llm/"):
        return "spine"
    top = chapter.split("/", 1)[0]
    if top in {"math", "algorithms"}:
        return "math"
    if top in {"systems", "ai-platform-engineering", "distributed-systems"}:
        return "systems"
    if top in {"infrastructure", "field-engineering", "interviews"}:
        return "operations"
    if top in {"ml", "data-engineering", "responsible-ai"}:
        return "spine"

    # Small shared chapters (primers, testing, privacy, and practices) follow
    # the pass where they are used so every registered module has one shard.
    p = int(module.get("pass", 0))
    if p <= 3:
        return "math"
    if p <= 7:
        return "spine"
    if p <= 9:
        return "systems"
    return "operations"


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in LAYERS:
        print(
            f"usage: {Path(sys.argv[0]).name} <{'|'.join(sorted(LAYERS))}>",
            file=sys.stderr,
        )
        return 2
    wanted = sys.argv[1]
    rows = []
    for path in sorted(MODULES.glob("*.toml")):
        module = tomllib.loads(path.read_text())
        if layer_of(module) == wanted:
            rows.append(module["id"])
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
