"""ol diff <ID> [--spoil]   your units against the reference, markers dropped

Meant for after you pass. Before a pass it refuses unless --spoil, which
records the module as `spoiled` in the ledger. A practice module or drill
has no units: its `artifacts` (primers/<ID>/, docs/, deploy/, ...) are
compared with course/ref/primers, course/ref/docs, or course/ref/entry."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path

from .. import EXIT_FAIL, ctx, ledger, markers, units
from ..session import open_session


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ol diff")
    ap.add_argument("id")
    ap.add_argument("--spoil", action="store_true")
    a = ap.parse_args(argv)
    s = open_session()
    m = s.module(a.id)
    v = ledger.matching(s.learner, s.reg, m.id) or ledger.latest(s.learner, m.id)
    if not (v and v.get("result") == "pass"):
        if not a.spoil:
            ctx.say(
                f"{m.id} has not passed yet; the diff would spoil it. `ol diff {m.id} --spoil` records it as spoiled."
            )
            return EXIT_FAIL
        ledger.event(s.learner, m.id, "spoiled", via="diff")
    ctx.say(
        f"{ctx.DIM}Reference on the right. Different is not the same as worse.{ctx.RST}"
    )
    for u in m.owned:
        ref = markers.drop_markers(
            units.ref_text(s.course, s.reg, u, m.id)
        ).splitlines()
        mine = (units.learner_text(s.learner, u) or "").splitlines()
        for line in difflib.unified_diff(
            mine, ref, f"yours: {u}", f"reference: {u}", lineterm=""
        ):
            ctx.say(line)
    for rel in m.artifacts:
        _diff_artifact(s.learner, s.course, rel)
    if not m.owned and not m.artifacts:
        ctx.say(f"{m.id} has no units or artifacts to compare")
    return 0


def _ref_root(course: Path, rel: str) -> Path:
    """Where the reference keeps a learner path: primers/ and docs/ under
    course/ref/, everything else (entry points, deploy/, CI) under course/ref/entry/."""
    top = rel.split("/", 1)[0]
    return course / "ref" / (rel if top in ("primers", "docs") else f"entry/{rel}")


def _diff_artifact(lr: Path, course: Path, rel: str) -> None:
    ref = _ref_root(course, rel)
    if not ref.exists():
        ctx.say(
            f"{ctx.DIM}{rel}: the reference has no copy ({ref} does not exist){ctx.RST}"
        )
        return
    mine_files = {
        p.relative_to(lr).as_posix() for p in ledger.artifact_files(lr, [rel])
    }
    ref_files = (
        {rel}
        if ref.is_file()
        else {
            f"{rel}/{p.relative_to(ref).as_posix()}"
            for p in ledger.artifact_files(ref.parent, [ref.name])
        }
    )
    for f in sorted(mine_files | ref_files):
        rp = ref if ref.is_file() else ref / f[len(rel) + 1 :]
        if f not in ref_files:
            ctx.say(f"{ctx.DIM}only yours:     {f}{ctx.RST}")
            continue
        if f not in mine_files:
            ctx.say(f"{ctx.DIM}only reference: {f}  ({rp}){ctx.RST}")
            continue
        mine = (lr / f).read_text(errors="replace").splitlines()
        theirs = markers.drop_markers(rp.read_text(errors="replace")).splitlines()
        for line in difflib.unified_diff(
            mine, theirs, f"yours: {f}", f"reference: {f}", lineterm=""
        ):
            ctx.say(line)
