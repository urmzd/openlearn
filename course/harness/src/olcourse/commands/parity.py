"""ol parity [<suite>...] [--fuzz [--cases N] [--seed N]] [--ref] [--json]

Parity suites (DESIGN 5.8): every implementation of an algorithm (Python,
C, Rust, Go) against the same golden oracle outputs, or, with --fuzz,
against each other on generated inputs. With a learner repo, each
implementation is YOUR code and runs only once its module has a fresh pass
(otherwise it is pending); --ref (and CI's parity-golden job, which has no
learner) runs the reference implementations.

Suites: course/conformance/parity/*.toml. Verdicts are ledger lines with id
`parity:<suite>`.

Exit: 0 every runnable implementation agrees (pending is fine), 1 a mismatch,
5 a driver or toolchain error."""

from __future__ import annotations

import argparse
import json

from .. import EXIT_FAIL, EXIT_HARNESS, EXIT_PASS, ctx, learner, ledger, parity
from ..overlay import Overlay
from ..runner import open_run
from ..session import Session


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ol parity",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("suites", nargs="*")
    ap.add_argument("--fuzz", action="store_true")
    ap.add_argument("--cases", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--ref", action="store_true", help="run the reference implementations"
    )
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    r = open_run(need_learner=False)
    suites = parity.which(a.suites, parity.load_all(r.course))
    if not suites:
        ctx.say(f"no parity suites in {parity.suite_dir(r.course)} yet")
        return EXIT_PASS
    ref_mode = a.ref or r.learner is None
    work = (ctx.scratch() / ".course-parity") if ref_mode else (r.learner / ".ol")
    sess = None if ref_mode else Session(r.learner, r.tree, r.reg)

    def overlay_for(im):
        closure = r.reg.closure(im.module)
        if ref_mode:
            src = {x: "ref" for x in [im.module] + closure}
            return (
                Overlay(
                    r.reg,
                    r.course,
                    im.module,
                    src,
                    None,
                    work / "parity" / im.module,
                    work,
                ),
                "",
                "ref",
            )
        st = learner.state(r.learner, r.course, r.reg, im.module)
        if st.status not in ("pass", "smoke", "assisted", "spoiled", "self"):
            return (
                None,
                f"{im.module} has no pass of yours yet ({st.status})",
                "learner",
            )
        from .check import resolve_sources

        src, _, _ = resolve_sources(sess, im.module, "failing", set())
        label = "learner" + (
            " (assisted)"
            if any(v == "ref" for k, v in src.items() if k != im.module)
            else ""
        )
        ov = Overlay(
            r.reg,
            r.course,
            im.module,
            src,
            r.learner,
            r.learner / ".ol" / "overlay" / f"parity-{im.module}",
            r.learner / ".ol",
        )
        return ov, "", label

    fails = errors = 0
    report = []
    lock = ctx.lock(work)
    with lock:
        for s in suites:
            mode = "fuzz" if a.fuzz else "golden"
            if a.fuzz and not s.fuzz.get("generator"):
                res = [
                    parity.ImplResult(
                        "-", "pending", "no [fuzz].generator for this suite"
                    )
                ]
            else:
                res = parity.run_suite(
                    r.course, r.reg, s, overlay_for, mode, a.seed, a.cases
                )
            status = (
                "error"
                if any(x.status == "error" for x in res)
                else "fail"
                if any(x.status == "fail" for x in res)
                else "pass"
                if any(x.status == "pass" for x in res)
                else "pending"
            )
            fails += status == "fail"
            errors += status == "error"
            report.append(
                {
                    "suite": s.id,
                    "mode": mode,
                    "status": status,
                    "impls": [x.__dict__ for x in res],
                }
            )
            if not a.json:
                color = {
                    "pass": ctx.GRN,
                    "pending": ctx.DIM,
                    "fail": ctx.RED,
                    "error": ctx.RED,
                }[status]
                ctx.say(
                    f"{ctx.BLD}parity {s.id}{ctx.RST} ({mode})  {s.title}  {color}{status}{ctx.RST}"
                )
                for x in res:
                    c = {"pass": ctx.GRN, "pending": ctx.DIM}.get(x.status, ctx.RED)
                    ctx.say(
                        f"  {c}{x.status:<8}{ctx.RST} {x.impl:<12} {ctx.DIM}{x.source}{ctx.RST}  {x.detail.splitlines()[0] if x.detail else ''}"
                    )
                    if x.status in ("fail", "error") and len(x.detail.splitlines()) > 1:
                        ctx.say(ctx.indent("\n".join(x.detail.splitlines()[1:20]), 6))
            if not ref_mode and status != "pending":
                ledger.append(
                    r.learner,
                    {
                        "id": f"parity:{s.id}",
                        "kind": "parity",
                        "mode": mode,
                        "result": "pass" if status == "pass" else "fail",
                        "assisted": any("assisted" in x.source for x in res),
                        "impls": {x.impl: x.status for x in res},
                        "tainted": r.tree.tainted,
                        "course_sha": r.tree.sha[:12],
                    },
                )
    if a.json:
        print(json.dumps(report))
    return EXIT_FAIL if fails else (EXIT_HARNESS if errors else EXIT_PASS)
