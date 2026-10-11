"""ol tdd red <ID> [--ref-deps[=all|ID,...]]     your graded tests must FAIL against your current units
ol tdd green <ID>   ...and then PASS, with the same test files

Red then green (DESIGN 5.12, rung R3 and up). `red` runs [learner_tests].path
against your own units (dependencies that are not passing come from the
reference) and requires a failure: you wrote the test before the code. It
records the hash of every test file. `green` requires a pass with exactly
those test files. From R3 on, `ol check` requires, for every current graded
test file, a red record that comes before a green one.

Exit: 0 when the phase holds, 1 when it does not, 5 harness."""

from __future__ import annotations

import argparse
import hashlib

from .. import EXIT_FAIL, EXIT_PASS, HarnessError, ctx, ledger, mutation
from ..overlay import Overlay, graded_files
from ..session import Session, open_session


def _file_hashes(s: Session, rel: str) -> dict[str, str]:
    return {
        p.relative_to(s.learner).as_posix(): "sha256:"
        + hashlib.sha256(p.read_bytes()).hexdigest()
        for p in graded_files(s.learner, rel)
    }


def _unit_hash(s: Session, m) -> str:
    h = hashlib.sha256()
    for u in sorted(m.owned):
        p = s.learner / u
        h.update(u.encode() + b"\0" + (p.read_bytes() if p.is_file() else b"") + b"\0")
    return "sha256:" + h.hexdigest()


def journal_errors(s: Session, m) -> list[str]:
    """R3+: every current graded test file has a red record (with its current
    hash) that comes before a green record of the current test set."""
    lt = m.learner_tests or {}
    if int(lt.get("rung", 0)) < 3:
        return []
    spec = mutation.Spec.of(m)
    files = _file_hashes(s, spec.path)
    if not files:
        return [f"no graded test files at {spec.path}"]
    thash = mutation.tests_hash(s.learner, spec.path)
    events = [
        e
        for e in ledger.entries(s.learner)
        if e.get("id") == m.id and e.get("event") == "tdd"
    ]
    greens = [
        i
        for i, e in enumerate(events)
        if e.get("phase") == "green" and e.get("tests") == thash
    ]
    if not greens:
        return [
            f"R{lt.get('rung')}: no `ol tdd green {m.id}` for your current test files"
        ]
    last_green = greens[-1]
    errs = []
    for rel, h in sorted(files.items()):
        if not any(
            e.get("phase") == "red" and (e.get("files") or {}).get(rel) == h
            for e in events[:last_green]
        ):
            errs.append(
                f"R{lt.get('rung')}: {rel} has no red record before its green (`ol tdd red {m.id}` first)"
            )
    return errs


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ol tdd",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("phase", choices=["red", "green"])
    ap.add_argument("id")
    # Accepted for symmetry with `ol check`: by default every dependency that
    # is not passing already comes from the reference.
    ap.add_argument("--ref-deps", nargs="?", const="", default=None)
    a = ap.parse_args(argv)
    s = open_session()
    m = s.module(a.id)
    if not m.learner_tests:
        raise HarnessError(
            f"{m.id} has no [learner_tests]: `ol tdd` grades your own tests"
        )
    spec = mutation.Spec.of(m)
    files = _file_hashes(s, spec.path)
    if not files:
        ctx.say(
            f"{ctx.RED}FAIL{ctx.RST} no graded test files at {spec.path}: write a test first"
        )
        return EXIT_FAIL
    from .check import parse_ref_deps, resolve_sources

    mode, chosen = (
        ("failing", set()) if a.ref_deps is None else parse_ref_deps(a.ref_deps)
    )
    if mode == "none":
        mode = "failing"
    sources, _, _ = resolve_sources(s, m.id, mode, chosen)
    with ctx.lock(s.learner / ".ol"):
        ov = Overlay(
            s.reg,
            s.course,
            m.id,
            sources,
            s.learner,
            s.learner / ".ol" / "tdd" / m.id,
            s.learner / ".ol",
        )
        r = ov.run_learner_tests(spec.path, max(spec.time_budget_s * 3, 60))
        thash = mutation.tests_hash(s.learner, spec.path)
        if a.phase == "red":
            if r.ok:
                ctx.say(
                    f"{ctx.RED}not red{ctx.RST} {m.id}: your tests pass against your current code. "
                    "Write a test that fails first, then make it pass."
                )
                return EXIT_FAIL
            ledger.event(
                s.learner,
                m.id,
                "tdd",
                phase="red",
                tests=thash,
                files=files,
                units=_unit_hash(s, m),
                compiled=r.compiled,
            )
            ctx.say(
                f"{ctx.GRN}red{ctx.RST} {m.id}: {len(files)} test file(s) fail as they should; recorded"
            )
            ctx.say(ctx.indent(ctx.tail(r.output, 12), 4))
            return EXIT_PASS
        reds = [
            e
            for e in ledger.entries(s.learner)
            if e.get("id") == m.id
            and e.get("event") == "tdd"
            and e.get("phase") == "red"
            and e.get("tests") == thash
        ]
        if not reds:
            ctx.say(
                f"{ctx.RED}no red{ctx.RST} {m.id}: there is no `ol tdd red {m.id}` for these exact test files "
                "(changed a test since? run red again)"
            )
            return EXIT_FAIL
        if not r.ok:
            ctx.say(f"{ctx.RED}not green{ctx.RST} {m.id}: your tests still fail")
            ctx.say(ctx.indent(ctx.tail(r.output, 20), 4))
            return EXIT_FAIL
        ledger.event(
            s.learner,
            m.id,
            "tdd",
            phase="green",
            tests=thash,
            files=files,
            units=_unit_hash(s, m),
        )
        ctx.say(
            f"{ctx.GRN}green{ctx.RST} {m.id}: the same {len(files)} test file(s) pass; recorded"
        )
        return EXIT_PASS
