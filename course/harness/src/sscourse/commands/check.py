"""ss check <ID> [--ref-deps[=all|ID,...]] [--no-cumulative] [--kind K] [--json] [--seed N]
ss check --all [--ci] [--fresh]

--all checks every started module in pass order. --ci is what the learner's
own CI runs (`ss course ci` prints that recipe): it forbids --ref-deps, skips
interactive proof rubrics (reported `self`), and runs practice checks with
SS_SMOKE=1 (no cluster tier; a practice verdict whose check skipped its
cluster tier is recorded with mode "smoke"). With nothing started it is trivially green.
--all reuses a pass recorded on exactly the current files (every owned unit,
artifact, and graded test in the repo, the same course commit and harness,
no reference code); --fresh checks every module again. Under --all the same
rule skips a dependency's cumulative smoke tests when that dependency passed
on these files (a single `ss check <ID>` always reruns them).

Order (DESIGN 5.3, 5.4): not started (2), contracts content hash (4), dependency
resolution (3), unit contract pre-check (4), cumulative smoke of every
learner-sourced dep, then the module's course tests through the overlay.
A practice check builds and runs the learner's own tree, so --ref-deps cannot
substitute into it: a practice module whose deps would come from the
reference is blocked (3) instead.

The exit code is the verdict: 0 pass, 1 fail, 2 not started, 3 blocked by deps,
4 contract drift, 5 harness or toolchain."""

from __future__ import annotations

import argparse
import json
import os

from .. import (
    EXIT_BLOCKED,
    EXIT_DRIFT,
    EXIT_FAIL,
    EXIT_HARNESS,
    EXIT_NOT_STARTED,
    EXIT_PASS,
    HarnessError,
    catalog,
    ctx,
    learner,
    ledger,
    precheck,
    tree,
)
from ..overlay import Overlay
from ..session import Session, open_session

CODE_KINDS = ("build", "side", "practice")


def parse_ref_deps(value: str | None) -> tuple[str, set[str]]:
    """None -> ("none", {}); bare -> ("failing", {}); "all"; "A,B" -> ("list", {A, B})."""
    if value is None:
        return "none", set()
    if value in ("", "failing"):
        return "failing", set()
    if value == "all":
        return "all", set()
    return "list", {x.strip() for x in value.split(",") if x.strip()}


def resolve_sources(
    s: Session, mid: str, mode: str, chosen: set[str]
) -> tuple[dict[str, str], list[str], list[str]]:
    """sources for the target and its closure, the blocked deps, and notes."""
    closure = s.reg.closure(mid)
    unknown = chosen - set(closure)
    if unknown:
        raise HarnessError(
            f"--ref-deps names {sorted(unknown)}, which {mid} does not depend on (closure: {closure})"
        )
    sources = {mid: "learner"}
    blocked, notes = [], []
    for d in closure:
        dm = s.reg.get(d)
        if not dm.owned:
            if not ledger.fresh_pass(s.learner, s.reg, d):
                notes.append(
                    f"{d} has no code to substitute (not passed yet; nothing blocks on it)"
                )
            continue
        if mode == "all" or d in chosen:
            sources[d] = "ref"
        elif ledger.fresh_pass(s.learner, s.reg, d) or learner.state(
            s.learner, s.course, s.reg, d
        ).status in ("pass", "smoke"):
            sources[d] = "learner"
        elif mode == "failing":
            sources[d] = "ref"
        else:
            blocked.append(d)
    return sources, blocked, notes


def _print_runs(runs, label: str, say=ctx.say) -> bool:
    ok = True
    for r in runs:
        if r.ok:
            say(f"  {ctx.GRN}PASS{ctx.RST} {label} [{r.lang}]")
        else:
            ok = False
            what = "does not compile" if not r.compiled else "FAIL"
            say(f"  {ctx.RED}{what}{ctx.RST} {label} [{r.lang}]")
            say(ctx.indent(ctx.tail(r.output, 60), 6))
    return ok


def check_one(
    s: Session,
    mid: str,
    ref_mode: str = "none",
    chosen: set[str] = frozenset(),
    cumulative: bool = True,
    kind: str | None = None,
    as_json: bool = False,
    seed: int = 0,
    quiet: bool = False,
    ci: bool = False,
    regrade: bool = False,
    reuse_deps: bool = False,
) -> int:
    """reuse_deps (set by --all): skip a dependency's cumulative smoke tests
    when it passed on exactly these files (it was just checked in the same
    run), instead of running them again for every dependent."""
    m = s.module(mid)
    # The whole repo as it is now: recorded with the verdict, so a later
    # `ss check --all` (or a dependent's regression step) can tell that a pass
    # was earned on exactly these files (ledger.reusable).
    repo = ledger.repo_hash(s.learner, s.reg)
    say = (lambda *_: None) if as_json else ctx.say
    if m.kind in ("solve", "proof"):
        return check_solve(s, m, ci, as_json, say, regrade)
    if m.kind == "drill":
        from .. import drills

        name = next((d.name for d in drills.all_drills(s.course) if d.id == m.id), m.id)
        v = ledger.latest(s.learner, mid)
        say(
            f"{m.id} is a drill, graded by `ss drill end`: run `ss drill start {name}`, "
            "fix the system, then `ss drill end`"
            + (f" (last graded: {v['result']})" if v else "")
        )
        if v:
            return EXIT_PASS if v.get("result") == "pass" else EXIT_FAIL
        return EXIT_NOT_STARTED
    later = learner.superseded(s.learner, s.course, s.reg, mid)
    prev = ledger.latest(s.learner, mid)
    if later and prev and prev.get("result") == "pass":
        say(
            f"{ctx.GRN}PASS{ctx.RST} {mid} (superseded by {later}; frozen verdict, its tests run as {later}'s regression)"
        )
        return EXIT_PASS
    if not learner.started(s.learner, s.course, s.reg, mid):
        say(f"{ctx.DIM}not started{ctx.RST} {mid}: ss start {mid}")
        return EXIT_NOT_STARTED

    def record(result: str, sources: dict | None = None, **extra) -> dict:
        srcs = {k: v for k, v in (sources or {}).items() if k != mid}
        smoke = extra.pop("smoke", False)
        return ledger.append(
            s.learner,
            {
                "id": mid,
                "kind": m.kind,
                "tree": ledger.tree_hash(s.learner, s.reg, mid),
                "sources": srcs,
                "result": result,
                "assisted": any(v == "ref" for v in srcs.values()),
                "mutation": None,
                **({"mode": "smoke"} if smoke else {}),
                "tainted": s.tree.tainted,
                "course_sha": s.tree.sha[:12],
                # After the run: a check may write its own artifacts.
                "repo": ledger.repo_hash(s.learner, s.reg),
                "closure": ledger.closure_hash(s.learner, s.reg, mid),
                "ci": ci,
                **extra,
            },
        )

    def finish(code: int, v: dict | None) -> int:
        if as_json:
            print(json.dumps({"id": mid, "exit": code, "verdict": v}))
        return code

    drift = tree.precheck_contracts(s.learner)
    if drift:
        say(f"{ctx.RED}DRIFT{ctx.RST} {mid}: " + "; ".join(drift))
        return finish(EXIT_DRIFT, record("drift", reason=drift))
    hint = tree.advise_semver(s.learner)
    if hint:
        say(f"{ctx.DIM}{hint}{ctx.RST}")

    sources, blocked, notes = resolve_sources(s, mid, ref_mode, set(chosen))
    if blocked:

        def _why(d: str) -> str:
            st = learner.state(s.learner, s.course, s.reg, d)
            return f"{d} ({st.status}{': ' + st.note if st.note else ''})"

        why = ", ".join(_why(d) for d in blocked)
        say(
            f"{ctx.YEL}BLOCKED{ctx.RST} {mid} needs {why}. Fix it"
            + ("" if m.kind == "practice" else ", or rerun with --ref-deps")
        )
        return finish(EXIT_BLOCKED, record("blocked", sources, blocked=blocked))
    refs = sorted(d for d, src in sources.items() if src == "ref")
    if m.kind == "practice" and refs:
        # A practice check builds and runs YOUR tree (images, services, entry
        # points; D16): the reference units would never reach what it runs.
        say(
            f"{ctx.YEL}BLOCKED{ctx.RST} {mid}'s check builds and runs your own tree, so "
            f"--ref-deps cannot stand in for {', '.join(refs)}: pass "
            f"{'it' if len(refs) == 1 else 'them'} first (ss check <ID>)"
        )
        return finish(EXIT_BLOCKED, record("blocked", sources, blocked=refs))
    for n in notes:
        say(f"  {ctx.DIM}{n}{ctx.RST}")

    scratch = s.learner / ".ss" / "build" / mid / "precheck"
    errs = precheck.module(s.learner, s.course, m, scratch)
    if errs:
        say(f"{ctx.RED}DRIFT{ctx.RST} {mid}: contract pre-check failed")
        for e in errs:
            say(ctx.indent(e, 4))
        return finish(EXIT_DRIFT, record("drift", sources, reason=errs))

    deps_line = ", ".join(f"{d} {src}" for d, src in sources.items() if d != mid)
    say(
        f"{ctx.BLD}check {mid}{ctx.RST}  {m.title}"
        + (f"  {ctx.DIM}deps: {deps_line}{ctx.RST}" if deps_line else "")
    )
    ov = Overlay(
        s.reg,
        s.course,
        mid,
        sources,
        s.learner,
        s.learner / ".ss" / "overlay" / mid,
        s.learner / ".ss",
        seed=seed,
    )

    # Which deps' tests ride along: smoke tests of learner-sourced deps, and
    # the full tests of every earlier owner of a unit this module took over.
    regress: list[tuple[str, list[str] | None]] = []
    if cumulative:
        for d in s.reg.closure(mid):
            if sources.get(d) == "learner" and s.reg.get(d).smoke:
                regress.append((d, s.reg.get(d).smoke))
        for u in m.upgrades:
            chain = s.reg.unit_chain(u)
            for prev_id in chain[: chain.index(mid)]:
                if prev_id not in [r[0] for r in regress]:
                    regress.append((prev_id, None))
                else:
                    regress = [(a, None if a == prev_id else b) for a, b in regress]
    rust_ids = [mid] + [d for d, _ in regress]

    ok = True
    regressions: list[str] = []
    for d, names in regress:
        if not ov.test_langs(d):
            continue
        if reuse_deps and names is not None and _passed_here(s, d, repo, ci):
            # d passed on exactly these files (same repo hash, course, and
            # harness): its smoke tests would run the same code to the same
            # result. A unit this module took over (names None) always reruns.
            if not quiet:
                say(f"  {ctx.DIM}ok   {d} smoke (passed on this tree){ctx.RST}")
            continue
        runs = ov.run_tests(d, names, rust_tests=rust_ids)
        label = f"{d} " + (
            f"smoke: {', '.join(names)}" if names else f"tests (taken over by {mid})"
        )
        if not all(r.ok for r in runs):
            ok = False
            regressions.append(d)
            say(
                f"  {ctx.RED}REGRESSION{ctx.RST} {d} ({'smoke: ' + ', '.join(names) if names else 'full suite'})"
            )
            _print_runs([r for r in runs if not r.ok], label, say)
        elif not quiet:
            say(f"  {ctx.DIM}ok   {label}{ctx.RST}")

    names = None
    if kind:
        names = [
            t.name
            for t in catalog.collect(catalog.files_for(s.course, mid, ov.test_dir(mid)))
            if kind in t.kinds
        ]
        if not names:
            say(f"  no {mid} test has KIND {kind}")
            return finish(EXIT_HARNESS, None)
    runs = ov.run_tests(mid, names, rust_tests=rust_ids)
    if not runs:
        say(
            f"  {ctx.YEL}no course tests found for {mid}{ctx.RST} (looked in {ov.test_dir(mid)})"
        )
        return finish(EXIT_HARNESS, None)
    ok = _print_runs(runs, mid, say) and ok

    grade = None
    if m.learner_tests and not ok:
        say(
            f"  {ctx.DIM}mutation grade of your tests: skipped while the course tests fail{ctx.RST}"
        )
    elif m.learner_tests:
        ok, grade = _mutation_grade(s, m, sources, seed, ci, say)
    result = "pass" if ok else "fail"
    # Smoke only when the check really skipped its cluster tier under
    # SS_SMOKE (_lib/practice.py says so); a check without one is a full pass.
    smoke = (
        m.kind == "practice"
        and os.environ.get("SS_SMOKE", "") not in ("", "0")
        and any("cluster tier skipped" in r.output for r in runs)
    )
    v = record(
        result,
        sources,
        smoke=smoke,
        regressions=regressions,
        **({"kind_filter": kind} if kind else {}),
        **({"mutation": grade} if grade else {}),
    )
    tag = " (assisted)" if v["assisted"] else ""
    if ok:
        say(f"{ctx.GRN}PASS{ctx.RST} {mid}{tag}")
    else:
        say(
            f"{ctx.RED}FAIL{ctx.RST} {mid}{tag}"
            + (f"  regressions: {', '.join(regressions)}" if regressions else "")
        )
        say(f"  {ctx.DIM}ss tests {mid}  reads what each test checks and why{ctx.RST}")
    return finish(EXIT_PASS if ok else EXIT_FAIL, v)


def _passed_here(s: Session, mid: str, repo: str, ci: bool) -> dict | None:
    """The latest verdict of `mid` when it is a pass a rerun would repeat."""
    v = ledger.latest(s.learner, mid)
    ok = ledger.reusable(
        v,
        repo,
        ledger.closure_hash(s.learner, s.reg, mid),
        s.tree.sha[:12],
        s.tree.tainted,
        ci,
        sample_ok=os.environ.get("SS_MUTATION_SAMPLE") == "1",
    )
    return v if ok else None


def _mutation_grade(
    s: Session, m, sources: dict, seed: int, ci: bool, say
) -> tuple[bool, dict]:
    """[learner_tests]: the red-then-green journal (R3 and up), then the
    mutation grade: the cached full grade when your tests are unchanged since
    `ss mutate`; otherwise --ci computes the full grade and a local check runs
    the required mutants plus a seeded sample of up to 8 (an estimate)."""
    from . import mutate as mut_cmd
    from .tdd import journal_errors

    ok = True
    for e in journal_errors(s, m):
        say(f"  {ctx.RED}TDD{ctx.RST}  {e}")
        ok = False
    grader = mut_cmd.grader_for(s, m, sources, seed)
    g = grader.cached_grade()
    label = "mutation grade (full, cached)"
    if g is None:
        # SS_MUTATION_SAMPLE=1 (set by `ss verify course --e2e`) samples even
        # under --ci: the reference learner's full grade is verify check 6.
        sample = not ci or os.environ.get("SS_MUTATION_SAMPLE") == "1"
        g = grader.grade(sample=sample, sample_seed=seed)
        if ci and g.sampled and not g.passed:
            # An estimate below the bar is not a verdict under --ci: grade
            # every mutant before failing.
            g = grader.grade(sample=False, sample_seed=seed)
        label = "mutation grade" + (" (estimate)" if g.sampled else "")
    mut_cmd.report(g, False, say, label)
    revealed = bool(ledger.fresh_pass(s.learner, s.reg, m.id)) or ledger.has_event(
        s.learner, m.id, "spoiled"
    )
    mut_cmd.survivors(s.course, m, g, revealed, say)
    if g.reason and g.results:
        say(ctx.indent(g.reason, 4))
    if g.sampled:
        say(
            f"  {ctx.DIM}`ss mutate {m.id}` computes the full grade once and caches it{ctx.RST}"
        )
    return ok and g.passed, g.summary()


def check_solve(
    s: Session, m, ci: bool, as_json: bool, say, regrade: bool = False
) -> int:
    """A solve set (DESIGN 5.5): SymPy-checked answers, self-graded proofs.
    --ci never asks a rubric: an ungraded proof is reported `self` (skipped)."""
    from .. import rubric, solve

    if not learner.started(s.learner, s.course, s.reg, m.id):
        say(f"{ctx.DIM}not started{ctx.RST} {m.id}: ss start {m.id}")
        return EXIT_NOT_STARTED

    def recorded(qid: str, h: str) -> bool | None:
        for e in reversed(ledger.entries(s.learner)):
            if (
                e.get("id") == m.id
                and e.get("event") == "rubric"
                and e.get("qid") == qid
                and e.get("hash") == h
            ):
                return bool(e.get("ok"))
        return None

    def ask(qid: str, lines: list[str], h: str):
        if ci or as_json:
            return None
        res = rubric.ask(lines)
        if res is not None:
            ledger.event(s.learner, m.id, "rubric", qid=qid, hash=h, ok=all(res))
        return res

    say(f"{ctx.BLD}check {m.id}{ctx.RST}  {m.title}")
    results = solve.grade(
        s.course, s.learner, m.id, ask, None if regrade else recorded, say
    )
    color = {"pass": ctx.GRN, "self": ctx.GRN, "self-skipped": ctx.DIM}
    for r in results:
        label = {"self-skipped": "self (skipped)", "missing": "missing"}.get(
            r.status, r.status.upper() if r.status in ("pass", "fail") else r.status
        )
        say(
            f"  {color.get(r.status, ctx.RED)}{label:<14}{ctx.RST} {r.qid}"
            + (f"  {r.msg}" if r.msg and r.status != "pass" else "")
        )
    skipped = [r.qid for r in results if r.status == "self-skipped"]
    bad = [r.qid for r in results if r.status in ("fail", "missing")]
    if skipped and not ci:
        bad += skipped  # a proof nobody graded is not a pass outside --ci
    result = "fail" if bad else "pass"
    is_self = any(r.status in ("self", "self-skipped") for r in results)
    v = ledger.append(
        s.learner,
        {
            "id": m.id,
            "kind": m.kind,
            "tree": ledger.tree_hash(s.learner, s.reg, m.id),
            "sources": {},
            "result": result,
            "assisted": False,
            "self": is_self,
            **({"self_skipped": skipped} if skipped else {}),
            "questions": {r.qid: r.status for r in results},
            "tainted": s.tree.tainted,
            "course_sha": s.tree.sha[:12],
        },
    )
    if as_json:
        print(
            json.dumps({"id": m.id, "exit": 0 if result == "pass" else 1, "verdict": v})
        )
    elif result == "pass":
        say(
            f"{ctx.GRN}PASS{ctx.RST} {m.id}"
            + (" (self-graded proofs)" if is_self else "")
        )
    else:
        say(
            f"{ctx.RED}FAIL{ctx.RST} {m.id}  {len(bad)} of {len(results)} question(s) not done: {', '.join(bad)}"
        )
    return EXIT_PASS if result == "pass" else EXIT_FAIL


def check_all(s: Session, ci: bool, as_json: bool, fresh: bool = False) -> int:
    """Every started module in pass order. A module whose latest verdict is a
    pass earned on exactly the current files (ledger.reusable) is reported
    from that verdict instead of being checked again; --fresh checks all."""
    rows = []
    worst = 0
    if ci:
        # The learner's CI has no cluster: practice checks skip their cluster
        # tier (named in their output), which the milestone kind steps grade.
        os.environ["SS_SMOKE"] = "1"
    for m in s.reg.ordered():
        if not learner.started(s.learner, s.course, s.reg, m.id):
            continue
        if m.kind == "drill":
            rows.append((m.id, "skipped", "graded by `ss drill end`"))
            continue
        if m.kind in ("solve", "proof"):
            # --ci never runs an interactive rubric: a proof is reported `self` and skipped.
            code = check_solve(s, m, ci, False, lambda *_: None)
            v = ledger.latest(s.learner, m.id) or {}
            st = {0: "pass", 1: "fail", 2: "not started"}.get(code, "error")
            if code == 0 and v.get("self"):
                st = "self"
            note = (
                "self-graded rubric, skipped in --ci"
                if v.get("self_skipped")
                else ("proofs self-graded" if v.get("self") else "")
            )
            rows.append((m.id, st, note))
            if code != EXIT_NOT_STARTED:
                worst = max(worst, code)
            continue
        if not fresh:
            v = _passed_here(s, m.id, ledger.repo_hash(s.learner, s.reg), ci)
            if v:
                rows.append((m.id, "pass", f"unchanged since {v.get('ts', '?')}"))
                continue
        code = check_one(
            s,
            m.id,
            "none",
            set(),
            True,
            None,
            False,
            quiet=True,
            ci=ci,
            reuse_deps=not fresh,
        )
        st = {0: "pass", 1: "fail", 2: "not started", 3: "blocked", 4: "drift"}.get(
            code, "error"
        )
        rows.append((m.id, st, ""))
        if code != EXIT_NOT_STARTED:
            worst = max(worst, code)
    if as_json:
        print(json.dumps([{"id": a, "status": b, "note": c} for a, b, c in rows]))
    else:
        ctx.say(
            f"\n{ctx.BLD}ss check --all{' --ci' if ci else ''}{ctx.RST}: {len(rows)} started module(s)"
        )
        for a, b, c in rows:
            ctx.say(f"  {a:<12} {b:<10} {c}")
        if not rows:
            ctx.say("  nothing started yet: trivially green")
    return worst


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ss check",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("id", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--ci", action="store_true")
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="--all: check every module, even one that passed on these exact files",
    )
    ap.add_argument("--ref-deps", nargs="?", const="", default=None)
    ap.add_argument("--no-cumulative", action="store_true")
    ap.add_argument("--kind")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--regrade",
        action="store_true",
        help="solve sets: ask every proof rubric again",
    )
    a = ap.parse_args(argv)
    s = open_session()
    with ctx.lock(s.learner / ".ss"):
        return _run(s, a, ap)


def _run(s: Session, a, ap) -> int:
    if a.all:
        if a.ref_deps is not None and a.ci:
            raise HarnessError("--ci forbids --ref-deps")
        return check_all(s, a.ci, a.json, a.fresh)
    if not a.id:
        ap.print_usage()
        return EXIT_HARNESS
    mode, chosen = parse_ref_deps(a.ref_deps)
    return check_one(
        s,
        a.id,
        mode,
        chosen,
        not a.no_cumulative,
        a.kind,
        a.json,
        a.seed,
        ci=a.ci,
        regrade=a.regrade,
    )
