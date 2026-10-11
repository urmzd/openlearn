"""Maintainer generator for the mutants of L1.1, L1.2, L1.3, L1.4, and L1.6.

Each mutant is a (old, new) text substitution on the reference unit with
its markers dropped (what a learner's file looks like). This script writes
course/mutants/<ID>/<mid>.patch as a unified diff against that text and
rewrites course/mutants/<ID>/manifest.tsv. `ol verify course <ID>` (check 6)
then proves every patch applies and is killed by the course tests.

    uv run --project course/harness python course/oracle/tok/mutants.py [ID...]
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

COURSE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COURSE / "harness" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from olcourse.markers import drop_markers  # noqa: E402
from mutant_specs import SPECS  # noqa: E402

# Rung R2 modules (DESIGN 5.6): no mutant is required for the learner's grade,
# so the manifest's `required` column is `n` for semantic mutants too.
RUNG = {"L1.1": 2, "L1.2": 2, "L1.3": 2, "L1.4": 2, "L1.6": 2}

HEADER = "# mid\tunit\ttier\toperator\tline\trequired\tpublic\tprivate\n"


def build(mid_module: str) -> None:
    rows = []
    out = COURSE / "mutants" / mid_module
    out.mkdir(parents=True, exist_ok=True)
    for old_patch in out.glob("*.patch"):
        old_patch.unlink()
    for m in SPECS[mid_module]:
        unit = m["unit"]
        src = drop_markers((COURSE / "ref" / unit).read_text())
        edits = m["edits"]
        text = src
        first_line = None
        for old, new in edits:
            n = text.count(old)
            if n != 1:
                raise SystemExit(
                    f"{mid_module} {m['mid']}: `{old[:60]}` occurs {n} times in {unit}"
                )
            i = text.index(old)
            line = text[:i].count("\n") + 1
            first_line = line if first_line is None else min(first_line, line)
            text = text.replace(old, new)
        diff = "".join(
            difflib.unified_diff(
                src.splitlines(keepends=True),
                text.splitlines(keepends=True),
                f"a/{unit}",
                f"b/{unit}",
                n=3,
            )
        )
        (out / f"{m['mid']}.patch").write_text(diff)
        rows.append(
            "\t".join(
                [
                    m["mid"],
                    unit,
                    m["tier"],
                    m["operator"],
                    str(first_line),
                    "y"
                    if m["tier"] == "semantic" and RUNG.get(mid_module, 3) >= 3
                    else "n",
                    "(hidden until pass)" if m["tier"] == "semantic" else m["public"],
                    m["private"],
                ]
            )
        )
    (out / "manifest.tsv").write_text(HEADER + "\n".join(rows) + "\n")
    print(f"{mid_module}: {len(rows)} mutants")


if __name__ == "__main__":
    for mod in sys.argv[1:] or sorted(SPECS):
        build(mod)
