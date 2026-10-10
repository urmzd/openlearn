"""ss milestone <MS-ID> [--smoke] [--ref-deps] [--step NAME] [--json]
ss milestone [list]

Runs a milestone through YOUR entry points (DESIGN 5.7, P9): [build].steps
first, then the [services.*] the steps need, started in `after` order on
allocated ports with a generated runtime.toml each, then every step, then
teardown. Logs land in .ss/milestones/<MS-ID>/<timestamp>/.

--smoke runs only the steps marked `smoke = true` that do not need a cluster
(what PR CI runs). Without it, `ci = "kind"` steps run against the cluster in
[deploy]; when that cluster is not reachable they are skipped with the reason
and the verdict is `incomplete`.

Every module in `requires` must have a fresh pass of your own; otherwise the
milestone is blocked (exit 3). --ref-deps runs it anyway and records the
verdict as `assisted` (entry points are always yours, D16).

Exit: 0 pass, 1 fail or incomplete, 3 blocked, 5 harness or toolchain."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from .. import (
    EXIT_BLOCKED,
    EXIT_FAIL,
    EXIT_HARNESS,
    EXIT_PASS,
    HarnessError,
    ctx,
    kube,
    ledger,
    matchers,
)
from .. import learner as learner_mod
from .. import milestones as ms_mod
from .. import placeholders, services, system, web
from ..runner import Run, open_run


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")[:80] or "step"


def list_milestones(r: Run) -> int:
    ids = ms_mod.all_ids(r.course)
    if not ids:
        ctx.say(f"no milestones in {r.course / 'milestones'} yet")
        return 0
    for i in ids:
        m = ms_mod.load(r.course, i)
        v = ledger.latest(r.learner, i, full_only=True) if r.learner else None
        sv = ledger.latest(r.learner, i) if r.learner else None
        state = (v or {}).get("result", "todo")
        if sv and sv.get("mode") == "smoke" and state != "pass":
            state += f" (smoke {sv.get('result')})"
        ctx.say(f"  {i:<16} pass {m.pass_:<3} {state:<22} {m.title}")
    return 0


def _next_step(r: Run, mid: str, smoke: bool) -> str:
    """The command that earns mid's pass, with what the ledger says now."""
    if mid not in r.reg.modules:
        return f"ss milestone {mid}" if mid.startswith("MS-") else ""
    m = r.reg.get(mid)
    st = (
        learner_mod.state(r.learner, r.course, r.reg, mid).status
        if r.learner
        else "todo"
    )
    if m.kind == "drill":
        from .. import drills

        name = next((d.name for d in drills.all_drills(r.course) if d.id == mid), mid)
        return f"({st}) ss drill start {name}, then ss drill end"
    if st == "smoke" and not smoke:
        return f"(passed under SS_SMOKE only) ss check {mid}, with the cluster in [deploy] up"
    return f"({st}) ss check {mid}"


class Runner:
    def __init__(
        self, r: Run, plan: ms_mod.Plan, smoke: bool, logdir: Path, seed: int = 0
    ):
        self.r = r
        self.plan = plan
        self.smoke = smoke
        self.logdir = logdir
        self.env = r.env(seed)
        self.stack: services.Stack | None = None
        self.outs: dict[str, Path] = {}
        self.prev_out: Path | None = None
        self.started = time.time()
        self.target = plan.target.id
        self.seed = seed
        self.recorder: dict | None = None

    def who(self) -> str:
        return f"milestone {self.target}"

    def step_lookup(self, step: ms_mod.Step, out: Path, matrix: dict):
        def ckpt() -> str | None:
            latest = out / "ckpt" / "LATEST"
            if not latest.is_file():
                return None
            return str(out / "ckpt" / latest.read_text().strip())

        def f(name: str):
            if name in matrix:
                return str(matrix[name])
            if self.smoke and name in step.smoke_vars:
                return step.smoke_vars[name]
            if name in step.vars:
                return step.vars[name]
            if name == "seed":
                return str(self.seed)
            if name == "out":
                return str(out)
            if name.startswith("out:"):
                p = self.outs.get(name[4:])
                if p is None:
                    raise HarnessError(
                        f"{{{name}}}: no earlier step named {name[4:]!r} ran in this milestone"
                    )
                return str(p)
            if name == "ckpt":
                v = ckpt()
                if v is None:
                    raise HarnessError(f"{{ckpt}}: {out}/ckpt/LATEST does not exist")
                return v
            if name == "queue":
                return step.queue
            if name == "model_dir":
                if step.model_dir:
                    return placeholders.expand(step.model_dir, self.r.lookup)
                return str(self.prev_out) if self.prev_out else None
            return None

        parts = [f]
        if self.stack is not None:
            parts.append(self.stack.lookup)
        parts.append(self.r.lookup)
        return placeholders.chain(*parts)

    def run_role(
        self,
        role: str,
        argv: list[str],
        cwd: Path | None = None,
        env: dict | None = None,
        timeout: float = 120,
    ) -> tuple[int, str]:
        cmd = self.r.system.entry(role, self.who()) + argv
        return ctx.run(
            cmd, cwd=cwd or self.r.learner, env=env or self.env, timeout=timeout
        )

    def run_step(self, step: ms_mod.Step, matrix: dict, label: str) -> matchers.Verdict:
        out = self.logdir / "out" / _slug(label)
        out.mkdir(parents=True, exist_ok=True)
        lk = self.step_lookup(step, out, matrix)
        where = f"{self.who()} step {label!r}"
        so = matchers.StepOut(rc=None, out_dir=out)
        argv: list[str] = []
        if step.run:
            argv = placeholders.expand_argv(step.argv, lk, where)
            env = {
                **self.env,
                **{
                    k: placeholders.expand(str(v), lk, where)
                    for k, v in step.env.items()
                },
            }
            cmd = self.r.system.entry(step.run, self.who()) + argv
            rc, text = ctx.run(cmd, cwd=self.r.learner, env=env, timeout=step.timeout_s)
            so = matchers.StepOut(rc=rc, stdout=text, out_dir=out, argv=cmd)
            (out / "stdout.txt").write_text(f"$ {' '.join(cmd)}\n{text}")
        elif step.http:
            h = placeholders.expand_obj(dict(step.http), lk, where)
            headers = dict(h.get("headers", {}))
            if h.get("auth"):
                key = self.r.api_key()
                if not key:
                    return matchers.Verdict(
                        "fail", f"step needs an API key in ${self.r.system.api_key_env}"
                    )
                headers["Authorization"] = f"Bearer {key}"
            resp = web.request(
                h.get("method", "GET"),
                h["url"],
                json_body=h.get("json"),
                headers=headers,
                timeout=float(h.get("timeout_s", step.timeout_s)),
            )
            if resp.status == 0:
                return matchers.Verdict("fail", resp.error)
            so = matchers.StepOut(
                rc=None, stdout=resp.text, http_status=resp.status, out_dir=out
            )
            (out / "response.txt").write_text(f"HTTP {resp.status}\n{resp.text}")
        mc = matchers.MatchCtx(
            run=self.r,
            lookup=lk,
            step_name=label,
            started_at=self.started,
            kind_mode=step.kind,
            stack=self.stack,
            run_role=self.run_role,
            step_argv=argv,
            step_role=step.run,
            step_key=f"{step.origin}/{step.name}",
            smoke=self.smoke,
            recorder=self.recorder,
        )
        v = matchers.evaluate(step.expect, so, mc)
        self.outs[label] = out
        self.outs.setdefault(step.name, out)
        self.prev_out = out
        return v


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ss milestone",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("id", nargs="?")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ref-deps", nargs="?", const="", default=None)
    ap.add_argument("--step", action="append", default=[])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--record-thresholds",
        action="store_true",
        help="maintainers: run over --seeds reference seeds and write ref-thresholds.tsv rows",
    )
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args(argv)
    r = open_run(need_learner=a.id not in (None, "list"))
    if a.id in (None, "list"):
        return list_milestones(r)
    errs = system.schema_errors(r.learner, r.system.raw)
    if errs:
        ctx.err(
            "system.toml does not match contracts/config/system.schema.json:\n"
            + ctx.indent("\n".join(errs[:20]))
        )
        return EXIT_HARNESS
    with ctx.lock(r.learner / ".ss" / "milestones"):
        if a.record_thresholds:
            return record_thresholds(r, a.id, a.smoke, a.step, a.seeds)
        return run_milestone(
            r, a.id, a.smoke, a.ref_deps is not None, a.step, a.json, a.seed
        )


def record_thresholds(
    r: Run, msid: str, smoke: bool, only: list[str], seeds: int
) -> int:
    """Run the milestone over seeds 0..seeds-1 (`{seed}` and SS_SEED) against
    this system (the reference learner), collect every `calibrated` metric,
    and write mean +/- 3 sd rows to course/fixtures/ref-thresholds.tsv."""
    from .. import thresholds

    if not r.tree.live:
        raise HarnessError(
            "--record-thresholds writes the live course tree; this learner's contracts/VERSION "
            "names another commit (ss contracts sync)"
        )
    seen: dict[tuple[str, str, str], list[float]] = {}
    for sd in range(seeds):
        rec: dict = {}
        code = run_milestone(r, msid, smoke, True, only, False, sd, recorder=rec)
        if code != EXIT_PASS:
            ctx.err(
                f"seed {sd}: the milestone did not pass on this system; no thresholds written"
            )
            return EXIT_FAIL
        for k, v in rec.items():
            seen.setdefault(k, []).append(v)
    if not seen:
        ctx.err(f"{msid} has no `calibrated` metric in the steps that ran")
        return EXIT_HARNESS
    mode = "smoke" if smoke else "full"
    rows = [
        thresholds.compute(k, m, d, mode, v) for (k, m, d), v in sorted(seen.items())
    ]
    p = thresholds.write(ctx.live_course(), rows)
    for row in rows:
        ctx.say(
            f"  {row.key} {row.metric} ({mode}): mean {row.mean:.6g}, sd {row.sd:.3g}, "
            f"threshold {row.threshold:.6g} over {row.n} seeds"
        )
    ctx.say(
        f"{ctx.GRN}wrote{ctx.RST} {len(rows)} row(s) to {p} (and its MANIFEST.tsv row)"
    )
    return EXIT_PASS


def run_milestone(
    r: Run,
    msid: str,
    smoke: bool,
    ref_deps: bool,
    only: list[str],
    as_json: bool,
    seed: int = 0,
    recorder: dict | None = None,
) -> int:
    say = (lambda *_: None) if as_json else ctx.say
    plan = ms_mod.plan(r.course, msid, smoke)
    steps = [
        s for s in plan.steps if not only or any(o in s.qualified(msid) for o in only)
    ]
    mode = "smoke" if smoke else "full"
    say(
        f"{ctx.BLD}milestone {msid}{ctx.RST}  {plan.target.title}  ({mode}, {len(steps)} step(s))"
    )

    def record(result: str, **extra) -> int:
        code = {"pass": EXIT_PASS, "blocked": EXIT_BLOCKED, "error": EXIT_HARNESS}.get(
            result, EXIT_FAIL
        )
        if recorder is not None:
            return code  # a recording run is not a verdict
        v = ledger.append(
            r.learner,
            {
                "id": msid,
                "kind": "milestone",
                "mode": mode,
                "result": result,
                "assisted": bool(extra.pop("assisted", False)),
                "tainted": r.tree.tainted,
                "course_sha": r.tree.sha[:12],
                **extra,
            },
        )
        if as_json:
            print(json.dumps({"id": msid, "exit": code, "verdict": v}))
        return code

    requires = list(plan.requires)
    if smoke:
        # A drill or a nightly-tier module is graded on a cluster, which a
        # --smoke run (PR CI) never has: the full milestone gates on it.
        cluster_only = [
            m
            for m in requires
            if m in r.reg.modules
            and (r.reg.get(m).kind == "drill" or r.reg.get(m).ci == "nightly")
        ]
        if cluster_only:
            say(
                f"  {ctx.DIM}not required under --smoke (graded on a cluster by the full run): "
                f"{', '.join(cluster_only)}{ctx.RST}"
            )
        requires = [m for m in requires if m not in cluster_only]
    waiting = [m for m in requires if not r.module_passed(m, smoke_ok=smoke)]
    if waiting and not ref_deps:
        say(
            f"{ctx.YEL}BLOCKED{ctx.RST} {msid} needs these to pass first, or rerun with "
            "--ref-deps (recorded as assisted):"
        )
        for m in waiting:
            say(f"  {m:<10} {_next_step(r, m, smoke)}")
        return record("blocked", blocked=waiting)
    assisted = bool(waiting)
    if assisted:
        say(
            f"  {ctx.YEL}assisted{ctx.RST}: {', '.join(waiting)} not passing; your entry points run as they are"
        )
    if not steps:
        if not smoke:
            ctx.err(f"{msid} selects no steps ({plan.target.path})")
            return EXIT_HARNESS
        say(f"  {ctx.DIM}no smoke steps outside kind: trivially green{ctx.RST}")
        return record("pass", assisted=assisted, steps=[])

    logdir = (
        r.learner
        / ".ss"
        / "milestones"
        / msid
        / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    logdir.mkdir(parents=True, exist_ok=True)
    run = Runner(r, plan, smoke, logdir, seed)
    run.recorder = recorder
    results: list[dict] = []

    local = [s for s in steps if not s.kind]
    kind_steps = [s for s in steps if s.kind]
    kind_ok, kind_why = (True, "")
    if kind_steps:
        kind_ok, kind_why = kube.cluster_ready(r.system.deploy)

    if local and r.system.build_steps:
        ok, why = services.run_build(
            r.system, logdir / "build.log", run.env, lookup=r.lookup
        )
        say(
            f"  {ctx.GRN if ok else ctx.RED}{'ok  ' if ok else 'FAIL'}{ctx.RST} [build] ({len(r.system.build_steps)} step(s))"
        )
        if not ok:
            say(ctx.indent(why, 6))
            return _finish(
                say,
                record,
                results,
                logdir,
                assisted,
                "fail",
                f"build failed: {why.splitlines()[0]}",
            )

    # Each contributing milestone's steps run against services of their own,
    # started fresh (a pass gate composes its component milestones and the
    # earlier gates' smoke steps): a component's state, such as MS-gateway's
    # usage ledger or a key's rate-limit budget, is the one it fixes, never
    # what an earlier milestone's steps left behind.
    groups: list[list] = []
    for s in steps:
        if groups and groups[-1][0].origin == s.origin:
            groups[-1].append(s)
        else:
            groups.append([s])
    try:
        for group in groups:
            needed: list[str] = []
            for s in group:
                if s.kind:
                    continue
                for n in s.services:
                    if n not in needed:
                        needed.append(n)
            gdir = logdir if group[0].origin == msid else logdir / group[0].origin
            gdir.mkdir(parents=True, exist_ok=True)
            with services.Stack(r.system, gdir, r.lookup, run.env, run.who()) as stack:
                run.stack = stack
                if needed:
                    stack.start(needed)
                for s in group:
                    if s.kind and not kind_ok:
                        results.append(
                            {
                                "step": s.qualified(msid),
                                "status": "skipped",
                                "detail": kind_why,
                            }
                        )
                        say(
                            f"  {ctx.YEL}skip{ctx.RST} {s.qualified(msid)}  {ctx.DIM}kind step: {kind_why}{ctx.RST}"
                        )
                        continue
                    for variant in s.variants():
                        label = s.qualified(msid) + (
                            "[" + ",".join(f"{k}={v}" for k, v in variant.items()) + "]"
                            if variant
                            else ""
                        )
                        t0 = time.time()
                        try:
                            v = run.run_step(s, variant, label)
                        except services.ServiceError:
                            raise
                        except HarnessError as e:
                            if e.code != EXIT_HARNESS:
                                raise
                            v = matchers.Verdict("error", str(e))
                        results.append(
                            {
                                "step": label,
                                "status": v.status,
                                "detail": v.detail[:2000],
                                "seconds": round(time.time() - t0, 2),
                            }
                        )
                        color = ctx.GRN if v.ok else ctx.RED
                        say(
                            f"  {color}{v.status:<8}{ctx.RST} {label}"
                            + (
                                f"  {ctx.DIM}{v.detail.splitlines()[0][:120]}{ctx.RST}"
                                if v.ok and v.detail
                                else ""
                            )
                        )
                        if not v.ok:
                            say(ctx.indent(ctx.tail(v.detail, 30), 11))
    except services.ServiceError as e:
        say(f"  {ctx.RED}FAIL{ctx.RST} {e}")
        return _finish(
            say, record, results, logdir, assisted, "fail", str(e).splitlines()[0]
        )

    if any(x["status"] == "error" for x in results):
        return _finish(
            say, record, results, logdir, assisted, "error", "harness error in a step"
        )
    failed = [x for x in results if x["status"] == "fail"]
    skipped = [x for x in results if x["status"] == "skipped"]
    result = "fail" if failed else ("incomplete" if skipped else "pass")
    return _finish(say, record, results, logdir, assisted, result, "")


def _finish(
    say, record, results, logdir: Path, assisted: bool, result: str, reason: str
) -> int:
    summary = {"result": result, "reason": reason, "steps": results}
    (logdir / "summary.json").write_text(json.dumps(summary, indent=1))
    tag = " (assisted)" if assisted and result == "pass" else ""
    color = (
        ctx.GRN
        if result == "pass"
        else (ctx.YEL if result == "incomplete" else ctx.RED)
    )
    label = {
        "pass": "PASS",
        "fail": "FAIL",
        "incomplete": "INCOMPLETE",
        "error": "ERROR",
    }[result]
    say(
        f"{color}{label}{ctx.RST}{tag}"
        + (f"  {reason}" if reason else "")
        + (
            f"  ({len([x for x in results if x['status'] == 'skipped'])} kind step(s) skipped)"
            if result == "incomplete"
            else ""
        )
        + f"  {ctx.DIM}logs: {logdir}{ctx.RST}"
    )
    return record(
        result,
        assisted=assisted,
        steps=[{"step": x["step"], "status": x["status"]} for x in results],
        log=str(logdir),
        **({"reason": reason} if reason else {}),
    )
