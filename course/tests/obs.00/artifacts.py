"""obs.00 artifact check: one request is one trace, gateway.proxy -> POST /v1/completions.

Run by `ol check obs.00` in your repo. Two tiers:

  local    your [build] steps, then your engine and gateway started from system.toml
           as local processes (as `ol milestone MS-P1 --smoke` does) with
           OTEL_EXPORTER_OTLP_ENDPOINT pointing at a course OTLP sink. One streamed
           request must produce the gateway's CLIENT span `gateway.proxy` and, as its
           child in the same trace, the engine's SERVER span `POST /v1/completions`.
  cluster  on [deploy].kube_context: a request through [deploy].gateway_url shows
           up in Jaeger ([deploy].traces) as one trace across <system>-gateway and
           <system>-engine.

The cluster tier fails when the context is missing or down; with OL_SMOKE=1 it is
skipped with the reason instead.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib.practice import Ctx, Fail, run  # noqa: E402
from _otlpsink import Sink, Span  # noqa: E402

GATEWAY_SPAN = "gateway.proxy"
ENGINE_SPAN = "POST /v1/completions"
CLIENT, SERVER = 3, 2
KEY = "tl_check_obs00probe"
# The W3C example ids: a caller's trace the gateway must continue.
CALLER_TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
CALLER_SPAN = "00f067aa0ba902b7"
CORPUS = "Once upon a time a request left the gateway and came back as tokens.\n" * 8


# -- local stack ----------------------------------------------------------------


def _harness():
    try:
        from olcourse import placeholders, services, system
    except ImportError:
        raise Fail(
            "the course harness is not importable: run this check through `ol check obs.00`"
        ) from None
    return placeholders, services, system


def _model(c: Ctx) -> str:
    if "model" not in c.cache:
        argv = (c.system().get("entry") or {}).get("tinyllm")
        if not argv:
            raise Fail(
                "system.toml has no [entry].tinyllm to train the model your engine serves"
            )
        work = c.path(".ol/check/obs.00")
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        (work / "corpus.txt").write_text(CORPUS)
        c.sh(
            [
                *argv,
                "train",
                "bigram",
                "--data",
                str(work / "corpus.txt"),
                "--out",
                str(work / "model"),
            ],
            timeout=300,
        )
        c.cache["model"] = str(work / "model")
    return c.cache["model"]


class Local:
    """Your engine and gateway as local processes, exporting to `sink`."""

    def __init__(self, c: Ctx, sink: Sink, tag: str) -> None:
        ph, services, system = _harness()
        self.c, self.sink = c, sink
        sysm = system.load(c.root)
        log = c.path(f".ol/check/obs.00/{tag}")
        log.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)
        env.pop("OTEL_SERVICE_NAME", None)
        env.update(
            {"OTEL_EXPORTER_OTLP_ENDPOINT": sink.endpoint, sysm.api_key_env: KEY}
        )
        if not c.cache.get("built"):
            ok, why = services.run_build(sysm, log / "build.log", env)
            if not ok:
                raise Fail(f"your [build] steps failed: {why}")
            c.cache["built"] = True
        local = {"model_dir": _model(c), "out": str(log), "system": sysm.name}
        entries = sysm.raw.get("entry", {})
        lookup = ph.chain(
            local.get,
            lambda k: list(entries[k]) if k in entries else None,
            lambda k: (
                ph.from_dict(sysm.deploy, "")(k[len("deploy.") :])
                if k.startswith("deploy.")
                else None
            ),
        )
        self.stack = services.Stack(sysm, log, lookup, env, "ol check obs.00")
        try:
            self.stack.start(["engine", "gateway"])
        except (
            Exception
        ) as e:  # ServiceError or HarnessError: a failed check, with the reason
            self.stack.stop()
            raise Fail(str(e)) from None
        self.base = self.stack.instances["gateway"].base

    def request(
        self, traceparent: str | None = None, max_tokens: int = 8, timeout: float = 20
    ) -> tuple[int, float]:
        headers = {"Authorization": f"Bearer {KEY}"}
        if traceparent:
            headers["traceparent"] = traceparent
        t0 = time.monotonic()
        status, _, _ = self.c.http(
            "POST",
            self.base + "/v1/completions",
            {
                "model": "tracer",
                "prompt": "Once",
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": True,
            },
            headers,
            timeout=timeout,
        )
        return status, time.monotonic() - t0

    def close(self) -> None:
        self.stack.stop()


def _trace(c: Ctx) -> dict:
    """One request with no caller context, the spans it produced, and the sink."""
    if "trace" not in c.cache:
        c.cache["trace"] = None
        sink = Sink().__enter__()
        c.cleanups.append(lambda: sink.__exit__(None, None, None))
        loc = Local(c, sink, "trace")
        c.cleanups.append(loc.close)
        status, _ = loc.request()
        if status != 200:
            raise Fail(f"the streamed request through your gateway answered {status}")

        def pair(spans: list[Span]):
            gws = [s for s in spans if s.name == GATEWAY_SPAN]
            for e in (s for s in spans if s.name == ENGINE_SPAN):
                for g in gws:
                    if e.trace_id == g.trace_id and e.parent_span_id == g.span_id:
                        return g, e
            return None

        spans = sink.wait(lambda s: pair(s) is not None, timeout=8)
        c.cache["trace"] = {
            "spans": spans,
            "pair": pair(spans),
            "sink": sink,
            "local": loc,
        }
    if c.cache["trace"] is None:
        raise Fail(
            "the local stack did not serve a request (see test_one_request_is_one_trace)"
        )
    return c.cache["trace"]


def _describe(spans: list[Span], sink: Sink) -> str:
    if not spans:
        why = f"the sink received {sink.requests} export request(s) and no spans"
        if sink.errors:
            why += f"; rejected: {sink.errors[-1]}"
        return why
    return "spans received: " + "; ".join(
        f"{s.name!r} kind={s.kind} trace={s.trace_id[:8]}.. span={s.span_id[:8]}.. parent={s.parent_span_id[:8] or '-'}"
        for s in spans[:8]
    )


# -- local tier -------------------------------------------------------------------


def test_one_request_is_one_trace(c: Ctx) -> None:
    # WHY: the point of tracing: the engine's server span names the gateway's span
    #      as its parent, so a slow or failed request is one tree you can read, not
    #      two unrelated records. It needs both halves: the gateway sends
    #      traceparent with its own span id, and both processes export.
    # KIND: conformance
    # CHAPTER: obs.00 section 5, Pitfalls 1 and 2
    t = _trace(c)
    if t["pair"] is None:
        raise Fail(
            f"no {ENGINE_SPAN!r} span whose parent is a {GATEWAY_SPAN!r} span in the same trace; "
            + _describe(t["spans"], t["sink"])
        )


def test_span_kinds_and_nesting(c: Ctx) -> None:
    # WHY: kinds tell a trace viewer which side of a call a span is (the gateway is
    #      the CLIENT of the engine, the engine is a SERVER), and a child span lies
    #      inside its parent's time window: the engine finishes before the gateway's
    #      proxy call does.
    # KIND: unit
    # CHAPTER: obs.00 section 3, Worked example
    t = _trace(c)
    if t["pair"] is None:
        raise Fail("no parent-child pair (see test_one_request_is_one_trace)")
    g, e = t["pair"]
    errs = []
    if g.kind != CLIENT:
        errs.append(f"{GATEWAY_SPAN} has kind {g.kind}; want 3 (SPAN_KIND_CLIENT)")
    if e.kind != SERVER:
        errs.append(f"{ENGINE_SPAN} has kind {e.kind}; want 2 (SPAN_KIND_SERVER)")
    for s in (g, e):
        if not (0 < s.start_ns <= s.end_ns):
            errs.append(
                f"{s.name}: start {s.start_ns} and end {s.end_ns} are not a valid interval (unix nanoseconds)"
            )
    slack = 50_000_000  # 50 ms: two processes, one clock, separate timestamps
    if (
        g.start_ns
        and e.start_ns
        and not (g.start_ns - slack <= e.start_ns and e.end_ns <= g.end_ns + slack)
    ):
        errs.append(
            f"{ENGINE_SPAN} [{e.start_ns}, {e.end_ns}] is not inside {GATEWAY_SPAN} [{g.start_ns}, {g.end_ns}]"
        )
    if len(g.trace_id) != 32 or len(g.span_id) != 16 or g.trace_id == "0" * 32:
        errs.append(
            f"ids must be 16 and 8 random bytes in hex; got trace {g.trace_id!r}, span {g.span_id!r}"
        )
    if errs:
        raise Fail("\n".join(errs))


def test_caller_trace_is_continued(c: Ctx) -> None:
    # WHY: a client that is itself traced (the agent in Pass 10, a curl with a
    #      traceparent) must see the gateway's span inside its own trace: the gateway
    #      joins an incoming W3C context instead of always starting a new one.
    # KIND: conformance
    # CHAPTER: obs.00 section 5, Pitfall 3
    loc = _trace(c)["local"]
    sink = _trace(c)["sink"]
    # A request that differs from the earlier tests' (max_tokens 9): from
    # gw.06 on, a repeated temperature-0 request may be answered from the
    # response cache without reaching the engine, and this test needs the hop.
    status, _ = loc.request(
        traceparent=f"00-{CALLER_TRACE}-{CALLER_SPAN}-01", max_tokens=9
    )
    if status != 200:
        raise Fail(f"a request carrying traceparent answered {status}")
    # Wait for both spans: each service exports on its own schedule.
    spans = sink.wait(
        lambda s: all(
            any(x.name == n and x.trace_id == CALLER_TRACE for x in s)
            for n in (ENGINE_SPAN, GATEWAY_SPAN)
        ),
        timeout=8,
    )
    gw = [s for s in spans if s.name == GATEWAY_SPAN and s.trace_id == CALLER_TRACE]
    if not gw:
        raise Fail(
            f"no {GATEWAY_SPAN} span in the caller's trace {CALLER_TRACE}; "
            + _describe(spans, sink)
        )
    if gw[0].parent_span_id != CALLER_SPAN:
        raise Fail(
            f"{GATEWAY_SPAN} has parent {gw[0].parent_span_id!r}; want the caller's span {CALLER_SPAN}"
        )
    if not any(
        s.name == ENGINE_SPAN
        and s.trace_id == CALLER_TRACE
        and s.parent_span_id == gw[0].span_id
        for s in spans
    ):
        raise Fail(
            f"the engine span is not the child of {GATEWAY_SPAN} in the caller's trace"
        )


def test_export_never_blocks_requests(c: Ctx) -> None:
    # WHY: telemetry is best effort and requests are not. With a collector that
    #      accepts connections and never answers, a request must take no longer than
    #      it does with a healthy one; an exporter that posts inside the request path
    #      turns a collector outage into a gateway outage.
    # KIND: fault
    # CHAPTER: obs.00 section 5, Pitfall 4
    _trace(c)  # the stack builds once; a second one starts with the hanging sink
    with Sink(hang=True) as dead:
        loc = Local(c, dead, "hang")
        try:
            times = []
            for _ in range(3):
                # 3 s is twice the limit below: a request still waiting then is
                # blocked on the collector, and waiting longer proves nothing.
                try:
                    status, took = loc.request(timeout=3)
                except Fail as e:
                    raise Fail(
                        f"with a hanging collector a request did not finish in 3 s ({e}): "
                        "export must not wait on the collector (queue it, with a timeout)"
                    ) from None
                if status != 200:
                    raise Fail(f"with a hanging collector a request answered {status}")
                times.append(took)
        finally:
            loc.close()
    if max(times) > 1.5:
        raise Fail(
            f"with a hanging collector requests took {', '.join(f'{x:.2f}s' for x in times)}; "
            "export must not wait on the collector (queue it, with a timeout)"
        )
    if dead.requests == 0:
        raise Fail(
            "nothing was exported to the hanging collector, so this test proved nothing: "
            "is OTEL_EXPORTER_OTLP_ENDPOINT read at startup?"
        )


# -- cluster tier ---------------------------------------------------------------------


def test_trace_in_jaeger_on_kind(c: Ctx) -> None:
    # WHY: the deployed version of the first test: the charts set OTEL_SERVICE_NAME
    #      and point OTEL_EXPORTER_OTLP_ENDPOINT at the jaeger Service, and Jaeger's
    #      query API (the MS-P1 trace step) shows one trace across both services.
    # KIND: conformance
    n = c.system_name()
    d = c.system().get("deploy") or {}
    if not d.get("kube_context"):
        raise Fail("system.toml has no [deploy].kube_context (dep.00)")
    c.need_cluster(d["kube_context"])
    env_name = (c.system().get("endpoints") or {}).get("api_key_env", "TL_API_KEY")
    key = os.environ.get(env_name)
    if not key:
        raise Fail(f"export {env_name} with the key stored in your API key Secret")
    trace_id = secrets.token_hex(16)
    status, _, _ = c.http(
        "POST",
        d["gateway_url"].rstrip("/") + "/v1/completions",
        {
            "model": "tracer",
            "prompt": "Once",
            "max_tokens": 8,
            "temperature": 0,
            "stream": True,
        },
        {
            "Authorization": f"Bearer {key}",
            "traceparent": f"00-{trace_id}-{CALLER_SPAN}-01",
        },
        timeout=30,
    )
    if status != 200:
        raise Fail(f"POST {d['gateway_url']}/v1/completions answered {status}")
    url = d["traces"].rstrip("/") + f"/api/traces/{trace_id}"
    deadline, last = time.monotonic() + 30, ""
    while time.monotonic() < deadline:
        code, _, body = c.http("GET", url, timeout=10)
        if code == 200:
            data = (json.loads(body).get("data") or [{}])[0]
            procs = {
                k: v.get("serviceName")
                for k, v in (data.get("processes") or {}).items()
            }
            spans = data.get("spans") or []
            by_id = {s["spanID"]: s for s in spans}
            for s in spans:
                if s.get("operationName") != ENGINE_SPAN:
                    continue
                for ref in s.get("references") or []:
                    parent = by_id.get(ref.get("spanID"))
                    if (
                        parent
                        and parent.get("operationName") == GATEWAY_SPAN
                        and procs.get(parent.get("processID")) == f"{n}-gateway"
                        and procs.get(s.get("processID")) == f"{n}-engine"
                    ):
                        return
            last = f"trace has spans {[x.get('operationName') for x in spans]} from {sorted(set(procs.values()))}"
        else:
            last = f"HTTP {code}"
        time.sleep(1)
    raise Fail(
        f"Jaeger never showed {GATEWAY_SPAN} ({n}-gateway) -> {ENGINE_SPAN} ({n}-engine) "
        f"for trace {trace_id}: {last}"
    )


if __name__ == "__main__":
    raise SystemExit(run(globals()))
