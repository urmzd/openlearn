"""ol bench --calibrate [--in-cluster]
ol bench <ID>... [--assert]
ol bench course [--assert]

Course performance budgets (DESIGN 5.11). --calibrate times the harness's
reference kernels (a tiled matmul and a decode step) on this machine and
stores the result; every budget is relative to it, so "at least half the
calibrated matmul GFLOP/s" means the same on a laptop and in CI.
--in-cluster runs the same kernel as a Job in [deploy].namespace (through
the drill safety gate), for SLOs checked on kind.

`ol bench <ID>` runs the module's [bench] against YOUR units (the reference
when there is no learner repo) and compares the metric with its budget;
`ol bench course` does it for every module with a budget (started ones, for
a learner). Without --assert a missed budget is reported and the exit is 0;
with --assert it is exit 1. Benchmarks run locally only, never in CI.

(`ol bench [lang] [--assert]` without a course id keeps its practice meaning.)

Exit: 0, 1 (a budget missed under --assert), 5 harness."""

from __future__ import annotations

import argparse

from .. import (
    EXIT_FAIL,
    EXIT_HARNESS,
    EXIT_PASS,
    HarnessError,
    bench,
    ctx,
    learner,
    ledger,
)
from ..overlay import Overlay
from ..runner import open_run
from ..session import Session


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ol bench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--in-cluster", action="store_true")
    ap.add_argument("--assert", dest="assert_", action="store_true")
    a = ap.parse_args(argv)
    r = open_run(need_learner=False)
    if a.calibrate:
        if a.in_cluster:
            if r.system is None:
                raise HarnessError(
                    "--in-cluster needs a learner repo with [deploy] in system.toml"
                )
            doc = bench.calibrate_in_cluster(r.system.deploy)
            where = f"in {doc['context']}/{doc['namespace']}"
        else:
            doc = bench.calibrate()
            where = f"on {doc['host']} ({doc['machine']}, {doc['cpus']} CPUs)"
        m = doc["metrics"]
        ctx.say(
            f"{ctx.GRN}calibrated{ctx.RST} {where}: matmul {m['matmul_gflops']:.2f} GFLOP/s, "
            f"decode {m['decode_tok_s']:.0f} steps/s  {ctx.DIM}{bench.calibration_path(a.in_cluster)}{ctx.RST}"
        )
        return EXIT_PASS
    if a.in_cluster:
        raise HarnessError("--in-cluster goes with --calibrate")
    if not a.targets:
        ap.print_usage()
        return EXIT_HARNESS
    calib = bench.load_calibration()
    with_bench = [m for m in r.reg.ordered() if m.bench]
    if a.targets == ["course"]:
        mods = with_bench
    else:
        mods = [r.reg.get(t) for t in a.targets]
        for m in mods:
            if not m.bench:
                raise HarnessError(f"{m.id} has no [bench] budget")
    ref_mode = r.learner is None
    sess = None if ref_mode else Session(r.learner, r.tree, r.reg)
    missed = errors = ran = 0
    work = ctx.scratch() / ".course-bench" if ref_mode else r.learner / ".ol"
    with ctx.lock(work):
        for m in mods:
            spec = m.bench
            if ref_mode:
                src = {x: "ref" for x in [m.id] + r.reg.closure(m.id)}
                ov = Overlay(
                    r.reg, r.course, m.id, src, None, work / "bench" / m.id, work
                )
                who = "ref"
            else:
                if not learner.started(r.learner, r.course, r.reg, m.id):
                    if a.targets != ["course"]:
                        ctx.say(
                            f"{ctx.DIM}not started{ctx.RST} {m.id}: ol start {m.id}"
                        )
                        errors += 1
                    continue
                from .check import resolve_sources

                src, _, _ = resolve_sources(sess, m.id, "failing", set())
                ov = Overlay(
                    r.reg,
                    r.course,
                    m.id,
                    src,
                    r.learner,
                    r.learner / ".ol" / "overlay" / f"bench-{m.id}",
                    r.learner / ".ol",
                )
                who = "yours"
            ran += 1
            try:
                metrics, out = bench.run_bench(ov, r.course, m.id, spec)
            except HarnessError as e:
                ctx.say(f"  {ctx.RED}ERROR{ctx.RST} {m.id}: {e}")
                errors += 1
                continue
            metric = str(spec.get("metric", ""))
            if metrics is None or metric not in metrics:
                ctx.say(
                    f"  {ctx.RED}ERROR{ctx.RST} {m.id}: no metric {metric!r}\n{ctx.indent(ctx.tail(out, 12))}"
                )
                errors += 1
                continue
            try:
                ok, why = bench.check_budget(
                    float(metrics[metric]), str(spec.get("budget", "")), calib
                )
            except HarnessError as e:
                ctx.say(f"  {ctx.RED}ERROR{ctx.RST} {m.id}: {e}")
                errors += 1
                continue
            missed += not ok
            color = ctx.GRN if ok else (ctx.RED if a.assert_ else ctx.YEL)
            ctx.say(
                f"  {color}{'ok  ' if ok else 'SLOW'}{ctx.RST} {m.id:<10} {metric} {why}  {ctx.DIM}[{spec.get('lang', 'c')}, {who}]{ctx.RST}"
            )
            if not ref_mode:
                ledger.event(
                    r.learner,
                    m.id,
                    "bench",
                    metric=metric,
                    value=metrics[metric],
                    budget=spec.get("budget"),
                    ok=ok,
                )
    if not ran and not errors:
        ctx.say("no module with a [bench] budget to run")
    if errors:
        return EXIT_HARNESS
    if missed and a.assert_:
        ctx.say(f"{ctx.RED}{missed} budget(s) missed{ctx.RST}")
        return EXIT_FAIL
    if missed:
        ctx.say(
            f"{ctx.YEL}{missed} budget(s) missed{ctx.RST} (report only; --assert makes it the verdict)"
        )
    return EXIT_PASS
