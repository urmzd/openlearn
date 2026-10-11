"""ol conform openapi[:v0|v1|v2][:engine|gateway][:smoke] [--target engine|gateway]
           [--base URL] [--health-base URL] [--model NAME] [--no-build] [--json]

Runs the OpenAPI conformance cases (DESIGN 5.8) against one tier. Without
--base it runs [build].steps, starts that tier's [services.*] entry (and the
services it comes `after`) on allocated ports, and tears them down after.
With --base it tests a running server (a kind NodePort, your own process);
/healthz is served on the health port, so pass --health-base for that case
(without it the case is pending).

Every response is validated against the vendored contract
contracts/openapi/openai-subset.<version>.yaml. A case whose `requires`
module has not passed yet is `pending`, not failed. The gateway tier sends
`Authorization: Bearer $<[endpoints].api_key_env>` (default TL_API_KEY).
Exit 0 when no case fails, 1 otherwise, 5 on a harness or toolchain error."""

from __future__ import annotations

import argparse
import json
import time

from .. import (
    EXIT_FAIL,
    EXIT_HARNESS,
    EXIT_PASS,
    HarnessError,
    conform,
    ctx,
    ledger,
    services,
)
from ..runner import Run, open_run


def run_suite(
    r: Run,
    suite: conform.Suite,
    base: str,
    health_base: str | None = None,
    model: str | None = None,
    overrides: dict | None = None,
) -> list[conform.Result]:
    client = r.conform_client(suite, base, health_base, model)
    cases = conform.load_cases(r.course)
    if not cases:
        raise HarnessError(
            f"no conformance cases in {r.course / 'conformance' / 'openapi' / 'cases'}"
        )
    return conform.run(client, cases, suite, r.module_passed, overrides)


def _fake_cases(r: Run, suite: conform.Suite) -> list[conform.Case]:
    return [
        c
        for c in conform.load_cases(r.course)
        if c.upstream == "fake"
        and suite.version in c.versions
        and suite.tier in c.tiers
        and (c.smoke or not suite.smoke)
    ]


def _fake_upstream_pass(
    r: Run, suite: conform.Suite, logdir, env: dict, model: str | None, say
) -> list[conform.Result]:
    """Cases that must see what the gateway sends upstream (priority.internal):
    the services the gateway comes after are replaced by one recording fake
    engine, whose port fills their {<service>.port} placeholders."""
    m = model or (r.system.endpoints.get("model") if r.system else None) or "tracer"
    fake = conform.FakeUpstream(m)
    try:
        with services.Stack(
            r.system,
            logdir / "fake-upstream",
            r.lookup,
            env,
            f"ol conform {suite.id} (recording upstream)",
        ) as stack:
            for name in r.system.start_order([suite.tier], stack.who):
                if name == suite.tier:
                    continue
                svc = r.system.service(name, stack.who)
                ports = {p: fake.port for p in services.PORT_NAMES}
                stack.instances[name] = services.Instance(
                    svc,
                    ports,
                    logdir,
                    logdir / f"{name}.fake.toml",
                    logdir / f"{name}.fake.log",
                )
            stack.start([suite.tier])
            inst = stack.instances[suite.tier]
            say(
                f"  {ctx.DIM}gateway against the recording upstream on :{fake.port}{ctx.RST}"
            )
            client = r.conform_client(
                suite, inst.base, f"http://127.0.0.1:{inst.ports['health_port']}", model
            )
            client.fake_upstream = fake
            return conform.run(client, _fake_cases(r, suite), suite, r.module_passed)
    finally:
        fake.close()


def report(results: list[conform.Result], say=ctx.say) -> tuple[int, int, int]:
    counts = {"pass": 0, "fail": 0, "pending": 0}
    for res in results:
        counts[res.status] += 1
        color = {"pass": ctx.GRN, "fail": ctx.RED, "pending": ctx.YEL}[res.status]
        say(
            f"  {color}{res.status:<7}{ctx.RST} {res.case}"
            + (f"  {ctx.DIM}{res.detail}{ctx.RST}" if res.status == "pending" else "")
        )
        if res.status == "fail":
            say(ctx.indent(ctx.tail(res.detail, 20), 10))
    return counts["pass"], counts["fail"], counts["pending"]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="ol conform",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("suite")
    ap.add_argument("--target")
    ap.add_argument("--base")
    ap.add_argument("--health-base")
    ap.add_argument("--model")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    suite = conform.parse_suite(a.suite, a.target)
    r = open_run(need_learner=a.base is None)
    say = (lambda *_: None) if a.json else ctx.say
    say(f"{ctx.BLD}conform {suite.id}{ctx.RST}")
    t0 = time.time()
    if a.base:
        # /healthz lives on the health port (spec/cli-roles.md), never on the
        # API port: without --health-base the healthz case is pending.
        results = run_suite(r, suite, a.base, a.health_base, a.model)
    else:
        if r.system is None:
            raise HarnessError(
                "no learner repo: ol course init --name <system>, or pass --base URL"
            )
        logdir = (
            r.learner
            / ".ol"
            / "conform"
            / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        )
        env = r.env()
        if not a.no_build:
            ok, why = services.run_build(
                r.system, logdir / "build.log", env, lookup=r.lookup
            )
            if not ok:
                say(f"{ctx.RED}FAIL{ctx.RST} build: {why}")
                return _record(r, suite, "fail", [], a.json, reason=why)
        try:
            with services.Stack(
                r.system, logdir, r.lookup, env, f"ol conform {suite.id}"
            ) as stack:
                stack.start([suite.tier])
                inst = stack.instances[suite.tier]
                results = run_suite(
                    r,
                    suite,
                    inst.base,
                    f"http://127.0.0.1:{inst.ports['health_port']}",
                    a.model,
                )
            if suite.tier == "gateway" and _fake_cases(r, suite):
                fake = _fake_upstream_pass(r, suite, logdir, env, a.model, say)
                done = {x.case for x in fake}
                results = [x for x in results if x.case not in done] + fake
        except services.ServiceError as e:
            say(f"{ctx.RED}FAIL{ctx.RST} {e}")
            return _record(r, suite, "fail", [], a.json, reason=str(e))
    if not results:
        ctx.err(f"no conformance case of {suite.id} applies (versions, tiers, smoke)")
        return EXIT_HARNESS
    p, f, n = report(results, say)
    result = "pass" if f == 0 else "fail"
    say(
        f"{(ctx.GRN + 'PASS') if f == 0 else (ctx.RED + 'FAIL')}{ctx.RST} {suite.id}: {p} pass, {f} fail, {n} pending"
        f"  ({time.time() - t0:.1f}s)"
    )
    return _record(r, suite, result, results, a.json)


def _record(
    r: Run, suite: conform.Suite, result: str, results, as_json: bool, **extra
) -> int:
    code = EXIT_PASS if result == "pass" else EXIT_FAIL
    cases = {x.case: x.status for x in results}
    v = None
    if r.learner is not None:
        v = ledger.append(
            r.learner,
            {
                "id": f"conform:{suite.id}",
                "kind": "conform",
                "result": result,
                "assisted": False,
                "cases": cases,
                "course_sha": r.tree.sha[:12],
                **extra,
            },
        )
    if as_json:
        print(
            json.dumps({"suite": suite.id, "exit": code, "cases": cases, "verdict": v})
        )
    return code
