"""ol mutate <ID> [-j N] [--reveal-survivors] [--json] [--seed N]

The mutation grade of YOUR tests (DESIGN 5.6): [learner_tests].path runs
against the reference with one planted fault per mutant. Score = killed /
total; pass = score >= the rung's threshold and every required mutant killed.
The full grade is cached by the hash of your test files (and each mutant by
test, unit, and patch hash), so it is recomputed only when your tests change;
`ol check` then uses it.

Survivors show where the fault was planted. A surviving semantic mutant shows
only which Pitfall it came from until the module passes, or until
--reveal-survivors, which records the module as spoiled.

Exit: 0 pass, 1 fail, 5 harness."""

from __future__ import annotations

import argparse
import json

from .. import EXIT_FAIL, EXIT_HARNESS, EXIT_PASS, HarnessError, ctx, ledger, mutation
from ..session import Session, open_session


def grader_for(
    s: Session, m, sources: dict | None, seed: int = 0, jobs: int = 0, say=None
) -> mutation.Grader:
    return mutation.Grader(
        s.reg,
        s.course,
        m,
        s.learner,
        s.learner / ".ol",
        s.learner / ".ol",
        mutation.find_grade_cache(s.learner),
        impl_sources=sources,
        seed=seed,
        jobs=jobs,
        say=say,
    )


def report(
    g: mutation.Grade, revealed: bool, say=ctx.say, label: str = "mutation grade"
) -> None:
    ms = {}
    for r in g.results:
        ms[r.mid] = r
    if g.reason and not g.results:
        say(f"  {ctx.RED}{label}: 0{ctx.RST}  {g.reason.splitlines()[0]}")
        rest = "\n".join(g.reason.splitlines()[1:])
        if rest:
            say(ctx.indent(rest, 6))
        return
    tag = f" (estimate from {len(g.results)} of {g.of} mutants)" if g.sampled else ""
    color = ctx.GRN if g.passed else ctx.RED
    say(
        f"  {color}{label}: {g.score:.2f}{ctx.RST} ({g.killed}/{g.total} killed, threshold {g.threshold:.2f}"
        f", required {'ok' if g.required_ok else 'MISSED'}){tag}"
    )
    return


def survivors(course, m, g: mutation.Grade, revealed: bool, say=ctx.say) -> None:
    by = {mu.mid: mu for mu in mutation.manifest(course, m.id)}
    for r in g.results:
        if r.status == "survived":
            mu = by.get(r.mid)
            text = mu.survivor_text(revealed) if mu else r.mid
            req = " (required)" if r.required else ""
            say(f"    {ctx.RED}survived{ctx.RST} {r.mid}{req}: {text}")
        elif r.status in ("invalid", "error"):
            say(f"    {ctx.YEL}{r.status}{ctx.RST} {r.mid}: {r.detail}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ol mutate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("id")
    ap.add_argument("-j", "--jobs", type=int, default=0)  # 0: OL_MUTATION_JOBS or 4
    ap.add_argument("--reveal-survivors", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    s = open_session()
    m = s.module(a.id)
    if not m.learner_tests:
        raise HarnessError(
            f"{m.id} has no [learner_tests]: nothing of yours to grade (rung R0/R1)"
        )
    from .check import resolve_sources

    sources, blocked, _ = resolve_sources(s, m.id, "failing", set())
    say = (lambda *_: None) if a.json else ctx.say
    with ctx.lock(s.learner / ".ol"):
        say(
            f"{ctx.BLD}mutate {m.id}{ctx.RST}  your tests at {mutation.Spec.of(m).path}"
        )
        g = grader_for(s, m, sources, a.seed, a.jobs, say).grade()
        passed_before = bool(ledger.fresh_pass(s.learner, s.reg, m.id))
        revealed = passed_before or a.reveal_survivors
        if a.reveal_survivors and not passed_before:
            ledger.event(s.learner, m.id, "spoiled", via="reveal-survivors")
        ledger.event(s.learner, m.id, "mutation", **g.summary())
    if a.json:
        print(
            json.dumps(
                {
                    "id": m.id,
                    "exit": 0 if g.passed else 1,
                    "grade": g.summary(),
                    "results": [r.__dict__ for r in g.results],
                }
            )
        )
        return EXIT_PASS if g.passed else EXIT_FAIL
    report(g, revealed)
    survivors(s.course, m, g, revealed)
    if g.reason and g.results:
        say(ctx.indent(g.reason, 4))
    if g.reason.startswith("harness"):
        return EXIT_HARNESS
    say(
        f"{ctx.GRN + 'PASS' if g.passed else ctx.RED + 'FAIL'}{ctx.RST} {m.id} mutation grade"
    )
    return EXIT_PASS if g.passed else EXIT_FAIL
