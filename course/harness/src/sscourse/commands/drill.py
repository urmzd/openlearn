"""ss drill list
ss drill start <name|id> [--seed N]   inject, then print only the pager symptom
ss drill status                      elapsed time only
ss drill end                         grade: detected (TTD), resolved (TTM; [[resolve.check]] and
                                     the [[resolve.run]] commands), trace evidence, postmortem
ss drill reset                       replay the undo journal in reverse
ss drill run <name|id> [--seed N] --respond   start, wait for the alert, run the
                                     scripted responder (respond.sh), grade without the time limit (CI)

A drill's [workload] (a build the faults land in) is started by `start` in
the background through your [entry] role, before the injections or, with
after_inject, after them; `reset` kills it.

Faults go into YOUR kind deployment (DESIGN 5.10); git-branch and
contract-bump drills work on a branch made in a scratch copy of your repo
and need no cluster. The safety gate refuses
unless `kubectl config current-context` equals [deploy].kube_context, that
context starts with kind- or k3d-, and [deploy].namespace exists; every action
stays in that namespace. Every injection writes its undo to
.ss/drills/<run>/journal.jsonl. There is no override flag."""

from __future__ import annotations

import argparse
import json
import random
import time

from .. import (
    EXIT_BLOCKED,
    EXIT_FAIL,
    EXIT_PASS,
    HarnessError,
    ctx,
    drills,
    kube,
    ledger,
    matchers,
    placeholders,
)
from ..runner import Run, open_run


def _runs_dir(r: Run):
    return r.learner / ".ss" / "drills"


def _current(r: Run) -> tuple[str, dict] | None:
    cur = _runs_dir(r) / "current"
    if not cur.is_file():
        return None
    run = cur.read_text().strip()
    st = _runs_dir(r) / run / "state.json"
    return (run, json.loads(st.read_text())) if st.is_file() else None


def _save(r: Run, run: str, state: dict) -> None:
    (_runs_dir(r) / run / "state.json").write_text(json.dumps(state, indent=1))


def cmd_list(r: Run) -> int:
    ds = drills.all_drills(r.course)
    if not ds:
        ctx.say(f"no drills in {r.course / 'drills'} yet")
        return 0
    for d in ds:
        v = ledger.latest(r.learner, d.id) if r.learner else None
        ctx.say(
            f"  {d.name:<20} {d.id:<8} {(v or {}).get('result', 'todo'):<6} {d.title}"
        )
    return 0


def cmd_start(r: Run, name: str, seed: int | None) -> int:
    d = drills.find(r.course, name)
    cur = _current(r)
    if cur and cur[1].get("status") == "active":
        raise HarnessError(
            f"drill {cur[1]['drill']} is still running (run {cur[0]}): `ss drill end`, then `ss drill reset`"
        )
    waiting = [x for x in d.requires if not r.module_passed(x)]
    if waiting:
        ctx.say(
            f"{ctx.YEL}BLOCKED{ctx.RST} {d.id} needs {', '.join(waiting)} to pass first"
        )
        return EXIT_BLOCKED
    k = kube.safety_gate(r.system.deploy) if d.needs_cluster else None
    if d.slo_profile:
        errs = drills.slo_errors(r.learner, d)
        if errs:
            raise kube.GateRefused(
                f"drill {d.id} runs under the `{d.slo_profile}` SLO profile, and your alert rules are not ready:\n"
                + ctx.indent("\n".join(errs))
            )
    seed = seed if seed is not None else int(time.time()) % 100000
    rng = random.Random(seed)
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{d.name}"
    rdir = _runs_dir(r) / run
    rdir.mkdir(parents=True, exist_ok=True)
    (_runs_dir(r) / "current").write_text(run)
    state = {
        "run": run,
        "drill": d.name,
        "id": d.id,
        "seed": seed,
        "status": "active",
        "started": time.time(),
        "injected": None,
        "picks": [],
        "context": k.context if k else None,
        "namespace": k.namespace if k else None,
        "slo_profile": d.slo_profile,
    }
    _save(r, run, state)
    j = drills.Journal.open(rdir)
    try:
        _inject_all(r, d, k, j, rng, state, seed, rdir)
    except BaseException:
        # A failed start must not leave half a fault behind: undo what was done.
        done = drills.undo(k, j)
        state["status"] = "aborted"
        _save(r, run, state)
        if done:
            ctx.say(f"  {ctx.YEL}start failed; undid {', '.join(done)}{ctx.RST}")
        raise
    symptom = (
        d.raw.get("symptom")
        or f"PAGE: {r.system.name} is degraded; users report errors and slow responses"
    )
    ctx.say(f"{ctx.RED}{symptom}{ctx.RST}")
    ctx.say(
        f"{ctx.DIM}drill running (seed hidden until `ss drill end`); time limit {d.raw.get('time_limit_min', 45)} min{ctx.RST}"
    )
    return EXIT_PASS


def _inject_all(r: Run, d, k, j, rng, state: dict, seed: int, rdir) -> None:
    run = state["run"]
    lk = placeholders.chain(r.lookup)
    extra = {
        "learner": r.learner,
        "course": r.course,
        "reg": r.reg,
        "run_dir": rdir,
        "env": r.env(seed),
        "drill": d.name,
        "role": lambda role: r.system.entry(role, f"drill {d.id}"),
    }
    injects = list(d.injects)
    wl = d.raw.get("workload")
    if wl and not wl.get("after_inject"):
        _start_workload(r, d, k, j, wl, lk, seed, rdir, state)
    cm = (r.system.deploy or {}).get("slo_configmap")
    if d.slo_profile and cm and k is not None:
        # Switch the learner's SLO windows to the drill profile for the run;
        # the journal switches them back.
        injects.insert(
            0,
            {
                "kind": "config-drift",
                "configmap": cm,
                "set": {"slo_profile": d.slo_profile},
                "target": (r.system.deploy or {}).get("slo_reloader", ""),
            },
        )
    for n, inj in enumerate(injects):
        at = drills.AT.match(str(inj.get("at", "start")))
        delay = float(at.group(3) or 0)
        if at.group(2):  # loadgen+Ns: start the learner's loadgen first
            lg = d.raw.get("loadgen", {})
            argv = placeholders.expand_argv(
                r.system.entry("loadgen", f"drill {d.id}")
                + list(
                    lg.get("argv", ["--rate", "20", "--target", "{deploy.gateway_url}"])
                ),
                lk,
            )
            pid = drills.start_background(
                argv, r.learner, r.env(seed), rdir / "loadgen.log"
            )
            j.append(
                {
                    "n": f"loadgen-{n}",
                    "kind": "loadgen",
                    "target": " ".join(argv),
                    "undo": [["kill", str(pid)]],
                }
            )
        if delay:
            time.sleep(delay)
        did, undo, pick = drills.inject(k, inj, rng, lk, extra)
        j.append(
            {
                "n": n,
                "kind": inj["kind"],
                "target": inj.get("target", ""),
                "did": did,
                "undo": undo,
            }
        )
        state["picks"].append({"inject": n, "kind": inj["kind"], "did": did, **pick})
        state["injected"] = state["injected"] or time.time()
        _save(r, run, state)
    if wl and wl.get("after_inject"):
        _start_workload(r, d, k, j, wl, lk, seed, rdir, state)


def _start_workload(r: Run, d, k, j, wl: dict, lk, seed: int, rdir, state) -> None:
    """[workload]: wait for `wait_rollout` (kubectl rollout status), then start
    the role's command in the background; the journal's undo kills it."""
    where = f"{d.path} [workload]"
    target = wl.get("wait_rollout")
    if target:
        if k is None:
            raise HarnessError(f"{where}: wait_rollout needs the cluster safety gate")
        target = placeholders.expand(str(target), lk, where)
        rc, out = k.run(["rollout", "status", target, "--timeout=300s"], timeout=315)
        if rc != 0:
            raise HarnessError(
                f"{where}: rollout status {target} exited {rc}:\n{ctx.indent(ctx.tail(out, 10))}"
            )
    argv = r.system.entry(str(wl["role"]), f"drill {d.id}") + placeholders.expand_argv(
        list(wl.get("argv", [])), lk, where
    )
    env = {
        **r.env(seed),
        **{
            k_: placeholders.expand(str(v), lk, where)
            for k_, v in dict(wl.get("env", {})).items()
        },
    }
    pid = drills.start_background(argv, r.learner, env, rdir / "workload.log")
    j.append(
        {
            "n": "workload",
            "kind": "workload",
            "target": " ".join(argv),
            "undo": [["kill", str(pid)]],
        }
    )
    state["workload"] = {"pid": pid, "log": str(rdir / "workload.log")}
    _save(r, state["run"], state)


def cmd_status(r: Run) -> int:
    cur = _current(r)
    if not cur:
        ctx.say("no drill run yet")
        return EXIT_PASS
    run, st = cur
    mins = (time.time() - st["started"]) / 60
    ctx.say(f"drill {st['drill']}: {st['status']}, {mins:.1f} min elapsed")
    return EXIT_PASS


def cmd_end(r: Run, scripted: bool = False) -> int:
    cur = _current(r)
    if not cur or cur[1].get("status") != "active":
        raise HarnessError("no active drill: ss drill start <name>")
    run, st = cur
    d = drills.find(r.course, st["drill"])
    now = time.time()
    t0 = st.get("injected") or st["started"]
    step = float(d.raw.get("step_s", 15))
    prom = str(r.system.deploy.get("prometheus", "")).rstrip("/")
    report: dict = {
        "run": run,
        "seed": st["seed"],
        "picks": st["picks"],
        "elapsed_s": round(now - st["started"], 1),
    }
    ok = True
    lines = []

    det = d.raw.get("detect")
    if det:
        if not prom:
            raise HarnessError("grading detection needs [deploy].prometheus")
        q = f'ALERTS{{alertname="{det["alert"]}",alertstate="firing"}}'
        series, err = drills.query_range(prom, q, t0, now, step)
        if series is None:
            raise HarnessError(f"Prometheus: {err}")
        times = drills.true_times(series)
        ttd = (times[0] - t0) if times else None
        detected = ttd is not None and ttd <= float(det.get("within_s", 300))
        report.update(detected=detected, ttd_s=None if ttd is None else round(ttd, 1))
        ok &= detected
        lines.append(
            (
                "detected",
                detected,
                f"{det['alert']} "
                + (
                    f"fired after {ttd:.0f}s (limit {det.get('within_s', 300)}s)"
                    if ttd is not None
                    else "never fired"
                ),
            )
        )
    else:
        lines.append(("detected", True, "no [detect]: graded manually"))

    resolved, ttm = True, 0.0
    for i, chk in enumerate((d.raw.get("resolve") or {}).get("check", [])):
        if "promql" in chk:
            if not prom:
                raise HarnessError("grading resolution needs [deploy].prometheus")
            series, err = drills.query_range(
                prom, placeholders.expand(chk["promql"], r.lookup), t0, now, step
            )
            if series is None:
                raise HarnessError(f"Prometheus: {err}")
            start = drills.final_run_start(drills.true_times(series), now, step)
            hold = float(chk.get("hold_s", 0))
            good = start is not None and (now - start) >= hold
            if good:
                ttm = max(ttm, start - t0)
            lines.append(
                (
                    f"resolve {i + 1}",
                    good,
                    f"held {0 if start is None else now - start:.0f}s of {hold:.0f}s: {chk['promql'][:70]}",
                )
            )
        elif "rollout" in chk:
            k = kube.safety_gate(r.system.deploy)
            target = placeholders.expand(str(chk["rollout"]), r.lookup)
            rc, out = k.run(
                [
                    "rollout",
                    "status",
                    target,
                    f"--timeout={int(chk.get('timeout_s', 60))}s",
                ],
                timeout=float(chk.get("timeout_s", 60)) + 15,
            )
            good = rc == 0
            lines.append(
                (
                    f"resolve {i + 1}",
                    good,
                    f"rollout status {target}: "
                    + (out.strip().splitlines() or ["(no output)"])[-1][:120],
                )
            )
        elif "suite" in chk:
            from .conform import run_suite
            from ..conform import parse_suite

            suite = parse_suite(chk["suite"])
            base = str(r.system.deploy.get("gateway_url", ""))
            res = run_suite(r, suite, base, r.system.deploy.get("gateway_health_url"))
            good = bool(res) and all(x.status != "fail" for x in res)
            lines.append(
                (
                    f"resolve {i + 1}",
                    good,
                    f"{suite.id}: {sum(x.status == 'fail' for x in res)} failing case(s)",
                )
            )
        else:
            raise HarnessError(
                f"{d.path}: resolve.check {i + 1} needs promql, rollout, or suite"
            )
        resolved &= good
    for i, rr in enumerate((d.raw.get("resolve") or {}).get("run", [])):
        good, detail = _resolve_run(r, d, st, i, rr)
        lines.append((f"resolve run {i + 1}", good, detail))
        resolved &= good
    report.update(resolved=resolved, ttm_s=round(ttm, 1) if resolved else None)
    ok &= resolved

    tr = (d.raw.get("evidence") or {}).get("trace")
    if tr:
        base = str(r.system.deploy.get("traces", "")).rstrip("/")
        tr = placeholders.expand_obj(dict(tr), r.lookup, f"{d.path} [evidence.trace]")
        t, why = matchers.find_trace(base, tr, t0, now)
        lines.append(
            ("trace evidence", t is not None, f"trace {t.get('traceID')}" if t else why)
        )
        ok &= t is not None
    for i, doc in enumerate(d.raw.get("doc", []), 1):
        label = str(doc.get("label") or f"doc {i}")
        path, errs = drills.postmortem_errors(r.learner, doc, what=label)
        lines.append(
            (label, not errs, errs[0] if errs else str(path.relative_to(r.learner)))
        )
        ok &= not errs
    pm = d.raw.get("postmortem")
    if pm:
        path, errs = drills.postmortem_errors(r.learner, pm)
        lines.append(
            (
                "postmortem",
                not errs,
                errs[0] if errs else str(path.relative_to(r.learner)),
            )
        )
        ok &= not errs
    limit = float(d.raw.get("time_limit_min", 45)) * 60
    if scripted:
        # CI's scripted responder (5.10): detection and resolution count, not the clock.
        lines.append(("time limit", True, "not graded (scripted responder)"))
    else:
        within = now - st["started"] <= limit
        lines.append(
            (
                "time limit",
                within,
                f"{(now - st['started']) / 60:.1f} of {limit / 60:.0f} min",
            )
        )
        ok &= within
    if st.get("slo_profile"):
        lines.append(
            (
                "slo profile",
                True,
                f"{st['slo_profile']}: windows {drills.SLO_PROFILES[st['slo_profile']]}",
            )
        )

    for label, good, detail in lines:
        ctx.say(
            f"  {ctx.GRN + 'ok  ' if good else ctx.RED + 'FAIL'}{ctx.RST} {label:<15} {detail}"
        )
    ctx.say(
        f"  {ctx.DIM}seed {st['seed']}; injected: "
        + "; ".join(p["did"] for p in st["picks"])
        + ctx.RST
    )
    times = [
        f"{k} {report[v]}s"
        for k, v in (("TTD", "ttd_s"), ("TTM", "ttm_s"))
        if report.get(v) is not None
    ]
    if times:
        ctx.say("  " + "  ".join(times))
    st["status"] = "ended"
    st["ended"] = now
    _save(r, run, st)
    tree_ = ledger.tree_hash(r.learner, r.reg, d.id) if d.id in r.reg.modules else None
    ledger.append(
        r.learner,
        {
            "id": d.id,
            "kind": "drill",
            "tree": tree_,
            "result": "pass" if ok else "fail",
            "assisted": False,
            "drill": d.name,
            **({"responder": "scripted"} if scripted else {}),
            **report,
        },
    )
    if pm and not scripted:
        from .. import rubric

        try:
            items = rubric.items(r.course, "postmortem")
            ctx.say(
                f"  {ctx.DIM}self-review your postmortem against course/rubrics/postmortem.md ({len(items)} items){ctx.RST}"
            )
        except HarnessError:
            pass
    pend = len(drills.Journal.open(_runs_dir(r) / run).pending_undos())
    ctx.say(
        f"{ctx.GRN + 'PASS' if ok else ctx.RED + 'FAIL'}{ctx.RST} drill {d.id}"
        + (
            f"  {ctx.DIM}`ss drill reset` undoes {pend} injection(s){ctx.RST}"
            if pend
            else ""
        )
    )
    return EXIT_PASS if ok else EXIT_FAIL


def _resolve_run(r: Run, d, st: dict, i: int, rr: dict) -> tuple[bool, str]:
    """[[resolve.run]] i: run the role's command from the learner repo and
    grade it with a milestone matcher (`expect`, default exit-code)."""
    where = f"{d.path} resolve.run {i + 1}"
    lk = placeholders.chain(r.lookup)
    argv = placeholders.expand_argv(list(rr.get("argv", [])), lk, where)
    cmd = r.system.entry(str(rr["role"]), f"drill {d.id}") + argv
    env = {
        **r.env(st["seed"]),
        **{
            k: placeholders.expand(str(v), lk, where)
            for k, v in dict(rr.get("env", {})).items()
        },
    }
    out_dir = _runs_dir(r) / st["run"] / f"resolve-run-{i + 1}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rc, text = ctx.run(
        cmd, cwd=r.learner, env=env, timeout=float(rr.get("timeout_s", 600))
    )
    (out_dir / "stdout.txt").write_text(f"$ {' '.join(cmd)}\n{text}")

    def run_role(role, argv_):  # the near-tie rerun of tokens-equal
        return ctx.run(
            r.system.entry(role, f"drill {d.id}") + argv_,
            cwd=r.learner,
            env=env,
            timeout=120,
        )

    mc = matchers.MatchCtx(
        run=r,
        lookup=lk,
        step_name=f"resolve.run {i + 1}",
        started_at=st["started"],
        run_role=run_role,
        step_argv=argv,
        step_role=str(rr["role"]),
    )
    v = matchers.evaluate(
        dict(rr.get("expect", {"match": "exit-code"})),
        matchers.StepOut(rc=rc, stdout=text, out_dir=out_dir, argv=cmd),
        mc,
    )
    head = f"{rr['role']} {' '.join(argv)[:60]}"
    first = (v.detail.splitlines() or [""])[0][:120]
    return v.ok, f"{head}: {first}" if first else head


def cmd_reset(r: Run) -> int:
    cur = _current(r)
    if not cur:
        ctx.say("no drill run to reset")
        return EXIT_PASS
    run, st = cur
    j = drills.Journal.open(_runs_dir(r) / run)
    pending = j.pending_undos()
    if not pending:
        ctx.say(f"drill run {run}: nothing to undo")
    else:
        local = ("kill", "http", "git-branch-remove")
        k = None
        if any(a[0] not in local for e in pending for a in e["undo"]):
            k = kube.safety_gate(r.system.deploy)
            if (k.context, k.namespace) != (st.get("context"), st.get("namespace")):
                raise kube.GateRefused(
                    f"refusing: run {run} injected into {st.get('context')}/{st.get('namespace')}, "
                    f"but [deploy] now names {k.context}/{k.namespace}"
                )
        for what in drills.undo(k, j):
            ctx.say(f"  undid {what}")
    if st.get("status") == "active":
        st["status"] = "reset"
        _save(r, run, st)
    ctx.say(f"{ctx.GRN}reset{ctx.RST} {run}")
    return EXIT_PASS


def cmd_run(
    r: Run, name: str, seed: int | None, respond: bool, poll_s: float, max_wait_s: float
) -> int:
    """start, wait for the alert, run the scripted responder, wait for the
    resolve checks to hold, then grade without the time limit (CI, 5.10)."""
    code = cmd_start(r, name, seed)
    if code != EXIT_PASS:
        return code
    d = drills.find(r.course, name)
    det = d.raw.get("detect")
    prom = str(r.system.deploy.get("prometheus", "")).rstrip("/")
    if det and prom:
        q = f'ALERTS{{alertname="{det["alert"]}",alertstate="firing"}}'
        deadline = time.monotonic() + min(float(det.get("within_s", 300)), max_wait_s)
        while time.monotonic() < deadline:
            res, _ = matchers.promql_instant(prom, q)
            if res:
                ctx.say(f"  {ctx.DIM}{det['alert']} is firing{ctx.RST}")
                break
            time.sleep(poll_s)
    if respond:
        script = d.path.parent / "respond.sh"
        if not script.is_file():
            raise HarnessError(f"drill {d.name} has no scripted responder ({script})")
        rc, out = ctx.run(
            ["bash", str(script), str(r.learner)],
            cwd=r.learner,
            env=r.env(),
            timeout=max_wait_s,
        )
        ctx.say(f"  {ctx.DIM}respond.sh exited {rc}{ctx.RST}")
        if rc != 0:
            ctx.say(ctx.indent(ctx.tail(out, 20), 4))
    hold = max(
        [
            float(c.get("hold_s", 0))
            for c in (d.raw.get("resolve") or {}).get("check", [])
            if "promql" in c
        ]
        + [0.0]
    )
    if hold:
        time.sleep(min(hold + float(d.raw.get("step_s", 15)), max_wait_s))
    return cmd_end(r, scripted=respond)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ss drill",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("verb", choices=["list", "start", "status", "end", "reset", "run"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--seed", type=int)
    ap.add_argument(
        "--respond", action="store_true", help="run: use the drill's scripted responder"
    )
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--max-wait-s", type=float, default=900.0)
    a = ap.parse_args(argv)
    r = open_run(need_learner=a.verb != "list")
    if a.verb == "list":
        return cmd_list(r)
    with ctx.lock(r.learner / ".ss" / "drills"):
        if a.verb == "start":
            if not a.name:
                raise HarnessError("usage: ss drill start <name|id> [--seed N]")
            return cmd_start(r, a.name, a.seed)
        if a.verb == "status":
            if a.name:
                # `drill:<id>` path checks call `ss drill status <id>`: answer with the last verdict.
                d = drills.find(r.course, a.name)
                v = ledger.latest(r.learner, d.id)
                ctx.say(f"drill {d.id}: {(v or {}).get('result', 'not run')}")
                return EXIT_PASS if v and v.get("result") == "pass" else EXIT_FAIL
            return cmd_status(r)
        if a.verb == "end":
            return cmd_end(r)
        if a.verb == "run":
            if not a.name:
                raise HarnessError(
                    "usage: ss drill run <name|id> [--seed N] [--respond]"
                )
            return cmd_run(r, a.name, a.seed, a.respond, a.poll_s, a.max_wait_s)
        return cmd_reset(r)
