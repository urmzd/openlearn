"""obs.05 artifact check: observability for the control plane.

Run by `ol check obs.05` in your repo. Three tiers, in order:

  static   deploy/observability/dashboards/control-plane.json is portable
           Grafana JSON whose queries parse (promtool) and name only contract
           metrics, with panels for queue depth by queue, the dead-letter
           queue, redeliveries as a rate, schedule-to-start p95, the WAL size,
           and the training gauges Python pushes; control-plane-monitors.yaml
           scrapes the durable server, the workers, and the collector's
           Prometheus exporter; your worker chart (dep.06) sends Python's
           OTLP/HTTP to port 4318 and the Go worker's OTLP/gRPC to 4317
  python   your telemetry.py (dur.09), run as a worker runs it, continues the
           activity span from TRACEPARENT: train.run under the activity span,
           train.step every 50th step under train.run, all in one trace,
           received by an OTLP/HTTP sink inside this check
  cluster  on [deploy]: every panel query runs in Prometheus, and Tempo holds
           a trace whose spans include workflow TrainRun, activity train,
           train.run, and train.step (MS-durable's kind run makes one)

The cluster tier fails when the context is missing or down; with OL_SMOKE=1 it
is skipped instead.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "obs.03"))

import _promtool  # noqa: E402  (shared with obs.03 and obs.04)
from _lib.practice import Ctx, Fail, Skip, containers, one, run, smoke_mode  # noqa: E402

DASH = "deploy/observability/dashboards/control-plane.json"
MONITORS = "deploy/observability/control-plane-monitors.yaml"
TP_TRACE, TP_PARENT = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"


def _obs04():
    """obs.04's PromQL helpers (metric_names, substitute, _contract_names)."""
    spec = importlib.util.spec_from_file_location(
        "obs04_artifacts", HERE.parent / "obs.04" / "artifacts.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


O4 = _obs04()


def dashboard(c: Ctx) -> dict:
    if "dash" not in c.cache:
        try:
            c.cache["dash"] = json.loads(c.require_file(DASH).read_text())
        except json.JSONDecodeError as e:
            raise Fail(f"{DASH} is not JSON: {e}") from None
    return c.cache["dash"]


def queries(c: Ctx) -> list[tuple[dict, str]]:
    out = [
        (p, str(t.get("expr", "")))
        for p in O4._panels(dashboard(c))
        for t in p.get("targets") or []
        if str(t.get("expr", "")).strip()
    ]
    if not out:
        raise Fail(f"{DASH} has no panel queries")
    return out


def panels_with(c: Ctx, pred) -> list[dict]:
    return [p for p, e in queries(c) if pred(e)]


# -- static -----------------------------------------------------------------------


def test_dashboard_is_portable(c: Ctx) -> None:
    # WHY: the dashboard is code: it must load into any Grafana of the stack
    #      the same way, so it has a stable uid, no instance id, and reads
    #      Prometheus through the ${datasource} variable, never a uid from
    #      someone's Grafana.
    # KIND: unit
    # CHAPTER: obs.05 section 4, The artifact and its check
    d = dashboard(c)
    errs = []
    if not str(d.get("uid", "")).strip():
        errs.append("no uid: Grafana makes a new dashboard on every import")
    if d.get("id") not in (None, 0):
        errs.append(f"id is {d.get('id')}: an instance id; set it to null")
    names = [
        v.get("name")
        for v in (d.get("templating") or {}).get("list") or []
        if v.get("type") == "datasource"
    ]
    if not names:
        errs.append("no datasource variable in templating.list")
    for p, _ in queries(c):
        ds = p.get("datasource") or {}
        uid = ds.get("uid") if isinstance(ds, dict) else ds
        if not (isinstance(uid, str) and (uid.startswith("$") or uid == "prometheus")):
            errs.append(
                f"panel {p.get('title')!r}: datasource {ds!r} is not the variable or the stack's uid"
            )
    if errs:
        raise Fail("\n".join(dict.fromkeys(errs)))


def test_queries_name_contract_metrics(c: Ctx) -> None:
    # WHY: a panel on a metric no service emits is an empty graph during the
    #      one incident it was built for. Every query names only metrics of
    #      contracts/otel/metrics.yaml (or ALERTS).
    # KIND: conformance
    # CHAPTER: obs.05 section 2.2
    allowed = O4._contract_names(c) | {"ALERTS"}
    errs = []
    for p, e in queries(c):
        bad = sorted(O4.metric_names(O4.substitute(e)) - allowed)
        if bad:
            errs.append(
                f"panel {p.get('title')!r}: {', '.join(bad)} not in contracts/otel/metrics.yaml"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_queries_parse(c: Ctx) -> None:
    # WHY: Grafana shows a PromQL syntax error only when someone opens the
    #      panel; promtool parses every query now.
    # KIND: conformance
    # CHAPTER: obs.05 section 4, The artifact and its check
    import yaml

    rules = [
        {"record": f"cp:query{i}", "expr": O4.substitute(e)}
        for i, (_, e) in enumerate(queries(c))
    ]
    r = _promtool.run(
        c,
        c.path(".ol/check/obs.05/parse"),
        ["check", "rules", "queries.yaml"],
        {
            "queries.yaml": yaml.safe_dump(
                {"groups": [{"name": "cp", "rules": rules}]}, sort_keys=False
            )
        },
        timeout=110,
    )
    if r.returncode != 0:
        raise Fail(
            "promtool rejects a query:\n" + (r.stdout + r.stderr).strip()[-1500:]
        )


def test_queue_panels(c: Ctx) -> None:
    # WHY: the three questions of a queue-based system during an incident:
    #      is work piling up (depth, per queue, since one stuck queue hides in
    #      a total), is work being given up on (the dead-letter queue), and are
    #      workers dying (redeliveries, a counter, so only its rate means
    #      anything: the raw total only ever grows).
    # KIND: unit
    # CHAPTER: obs.05 section 2.2
    errs = []
    depth = panels_with(c, lambda e: "tl_durable_task_queue_depth" in e)
    if not depth:
        errs.append("no panel shows tl_durable_task_queue_depth")
    elif not any(
        re.search(r"\bby\s*\([^)]*\bqueue\b", e)
        or re.search(r"tl_durable_task_queue_depth\s*(\{[^}]*\})?\s*$", e)
        for p, e in queries(c)
        if "tl_durable_task_queue_depth" in e
    ):
        errs.append(
            "the queue-depth panel sums the queues together: keep `by (queue)` so a stuck queue shows"
        )
    if not panels_with(c, lambda e: "tl_durable_dlq_size" in e):
        errs.append("no panel shows tl_durable_dlq_size (the dead-letter queue)")
    redel = [e for _, e in queries(c) if "tl_durable_redeliveries_total" in e]
    if not redel:
        errs.append("no panel shows tl_durable_redeliveries_total")
    elif not all(
        re.search(r"\b(rate|irate|increase)\s*\(\s*tl_durable_redeliveries_total", e)
        for e in redel
    ):
        errs.append(
            "redeliveries are a counter: show rate(...) or increase(...), never the raw total"
        )
    if errs:
        raise Fail("\n".join(errs))


def test_schedule_to_start_quantile(c: Ctx) -> None:
    # WHY: how long a task waits for a worker is the latency KEDA's scaling
    #      exists to keep low. A histogram quantile must aggregate the bucket
    #      rates and keep `le`; dropping `le` from the `by` clause gives NaN,
    #      and a quantile of the raw buckets mixes the whole history.
    # KIND: boundary
    # CHAPTER: obs.05 section 3, Worked example by hand
    exprs = [
        e
        for _, e in queries(c)
        if "tl_durable_task_schedule_to_start_seconds_bucket" in e
    ]
    if not exprs:
        raise Fail(
            "no panel shows tl_durable_task_schedule_to_start_seconds (schedule-to-start)"
        )
    good = [
        e
        for e in exprs
        if re.search(r"histogram_quantile\s*\(\s*0?\.\d+", e)
        and re.search(r"\bby\s*\([^)]*\ble\b", e)
        and re.search(
            r"\b(rate|increase)\s*\(\s*tl_durable_task_schedule_to_start_seconds_bucket",
            e,
        )
    ]
    if not good:
        raise Fail(
            "schedule-to-start needs histogram_quantile(q, sum by (le, ...) (rate(..._bucket[...]))): "
            + "; ".join(exprs)
        )


def test_storage_and_training_panels(c: Ctx) -> None:
    # WHY: the WAL's size against its quota is what warns before appends
    #      fail (ops.11), and the training gauges Python pushes over OTLP are
    #      the only view of a run in progress.
    # KIND: unit
    # CHAPTER: obs.05 section 2.2
    errs = []
    if not panels_with(c, lambda e: "tl_durable_wal_bytes" in e):
        errs.append("no panel shows tl_durable_wal_bytes")
    if not panels_with(c, lambda e: "tl_train_loss" in e):
        errs.append(
            "no panel shows tl_train_loss (pushed by Python through the collector)"
        )
    if errs:
        raise Fail("\n".join(errs))


def test_monitors_scrape_the_control_plane(c: Ctx) -> None:
    # WHY: a dashboard over metrics nobody scrapes is empty. Prometheus
    #      selects only monitors labelled release: observability
    #      (contracts/helm/observability.md); the durable server and the
    #      workers serve /metrics on the port named health; Python's pushed
    #      metrics appear on the collector's Prometheus exporter.
    # KIND: unit
    # CHAPTER: obs.05 section 2.3
    docs = [d for d in c.yaml_file(MONITORS) if isinstance(d, dict)]
    mons = [d for d in docs if d.get("kind") in ("PodMonitor", "ServiceMonitor")]
    errs = []
    for d in mons:
        if ((d.get("metadata") or {}).get("labels") or {}).get(
            "release"
        ) != "observability":
            errs.append(
                f"{d['kind']} {d['metadata'].get('name')}: no label release: observability, so Prometheus ignores it"
            )

    def selects(component: str) -> bool:
        for d in mons:
            ml = ((d.get("spec") or {}).get("selector") or {}).get("matchLabels") or {}
            eps = (
                (d.get("spec") or {}).get("podMetricsEndpoints")
                or (d.get("spec") or {}).get("endpoints")
                or []
            )
            if ml.get("app.kubernetes.io/component") == component and any(
                e.get("port") == "health" for e in eps
            ):
                return True
        return False

    for comp in ("durable", "worker"):
        if not selects(comp):
            errs.append(
                f"no monitor scrapes the port named health of pods with app.kubernetes.io/component: {comp}"
            )
    if not any(
        any(
            e.get("port") == "prom-exporter"
            for e in (
                (d.get("spec") or {}).get("podMetricsEndpoints")
                or (d.get("spec") or {}).get("endpoints")
                or []
            )
        )
        for d in mons
    ):
        errs.append(
            "no monitor scrapes the collector's prom-exporter port: Python's tl_train_* gauges never reach Prometheus"
        )
    if errs:
        raise Fail("\n".join(errs))


def test_python_exports_over_http(c: Ctx) -> None:
    # WHY: two exporters share one pod. The Go worker speaks OTLP over gRPC
    #      to 4317; the Python children post OTLP/HTTP JSON (dur.09), which
    #      the collector receives on 4318. Pointing Python at 4317 drops every
    #      Python span silently (the gRPC port answers HTTP with an error
    #      telemetry.py swallows by design).
    # KIND: boundary
    # CHAPTER: obs.05 section 5, Pitfalls
    n = c.system_name()
    docs = c.helm_template(f"deploy/helm/{n}-worker", f"{n}-worker", namespace=n)
    dep = one(docs, "Deployment", f"deploy/helm/{n}-worker")
    env = {
        e.get("name"): str(e.get("value", ""))
        for e in containers(dep)[0].get("env") or []
    }
    errs = []
    go_ep = env.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    py_ep = env.get("TL_PYTHON_OTLP_ENDPOINT", "")
    if not go_ep.endswith(":4317"):
        errs.append(
            f"OTEL_EXPORTER_OTLP_ENDPOINT is {go_ep!r}; the Go worker exports OTLP/gRPC to the collector's 4317"
        )
    if not py_ep:
        errs.append(
            "TL_PYTHON_OTLP_ENDPOINT is not set: the worker cannot tell its Python children where to export"
        )
    elif not (py_ep.startswith("http") and py_ep.rstrip("/").endswith(":4318")):
        errs.append(
            f"TL_PYTHON_OTLP_ENDPOINT is {py_ep!r}; Python posts OTLP/HTTP JSON to the collector's 4318"
        )
    if errs:
        raise Fail("\n".join(errs))


# -- python -----------------------------------------------------------------------


class _Sink:
    def __init__(self) -> None:
        self.spans: list[dict] = []
        sink = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", "0")))
                )
                if self.path == "/v1/traces":
                    for rs in body.get("resourceSpans", []):
                        for ss in rs.get("scopeSpans", []):
                            sink.spans += ss.get("spans", [])
                self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(
            target=self.srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        ).start()

    def close(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()


TRAIN = """
from tinyllm.io.telemetry import Tracer, should_sample_step
tr = Tracer.from_env()
with tr.span("train.run", {"tl.run.id": "train-1", "tl.train.steps": 120}):
    for step in range(120):
        if should_sample_step(step):
            with tr.span("train.step", {"tl.train.step": step, "tl.train.loss": 2.0, "tl.train.tokens": 64 * step}):
                pass
    tr.gauge("tl.train.loss", 2.0, {"tl.workflow.id": "train-1"})
tr.shutdown()
"""


def test_python_span_tree_from_traceparent(c: Ctx) -> None:
    # WHY: the bottom of the control-plane trace is Python. Run the way the
    #      worker runs it (TRACEPARENT of the activity span, the endpoint and
    #      service name from the environment), your telemetry.py must make
    #      train.run a child of the activity span in the same trace and keep
    #      train.step only every 50th step (0, 50, 100 of 120).
    # KIND: conformance
    # CHAPTER: obs.05 section 3, Worked example by hand
    if not c.path("python/tinyllm/io/telemetry.py").is_file():
        raise Fail("python/tinyllm/io/telemetry.py is missing (dur.09)")
    sink = _Sink()
    c.cleanups.append(sink.close)
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(c.path("python")),
            "TRACEPARENT": f"00-{TP_TRACE}-{TP_PARENT}-01",
            "OTEL_EXPORTER_OTLP_ENDPOINT": sink.url,
            "OTEL_SERVICE_NAME": f"{c.system_name()}-python",
        }
    )
    with tempfile.TemporaryDirectory() as d:
        p = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(TRAIN)],
            cwd=d,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            out, _ = p.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9)
            raise Fail("the telemetry run did not finish in 60 s") from None
    if p.returncode != 0:
        raise Fail("running your telemetry.py failed:\n" + out[-1500:])
    by_name: dict[str, list[dict]] = {}
    for s in sink.spans:
        by_name.setdefault(s.get("name"), []).append(s)
    errs = []
    runs = by_name.get("train.run", [])
    if len(runs) != 1:
        errs.append(f"{len(runs)} train.run spans reached the collector; want 1")
    else:
        r = runs[0]
        if r.get("traceId") != TP_TRACE or r.get("parentSpanId") != TP_PARENT:
            errs.append(
                f"train.run is in trace {r.get('traceId')} under {r.get('parentSpanId')}; want trace {TP_TRACE} under the activity span {TP_PARENT}"
            )
        steps = by_name.get("train.step", [])
        got = sorted(
            int(a["value"].get("intValue", -1))
            for s in steps
            for a in s.get("attributes", [])
            if a.get("key") == "tl.train.step"
        )
        if got != [0, 50, 100]:
            errs.append(
                f"train.step spans for steps {got}; want [0, 50, 100] (every 50th of 120)"
            )
        if any(
            s.get("parentSpanId") != r.get("spanId") or s.get("traceId") != TP_TRACE
            for s in steps
        ):
            errs.append(
                "a train.step span is not a child of train.run in the same trace"
            )
    if errs:
        raise Fail("\n".join(errs))


# -- cluster ----------------------------------------------------------------------


def _deploy(c: Ctx, key: str) -> str:
    d = c.system().get("deploy") or {}
    url = str(d.get(key, "")).rstrip("/")
    if d.get("kube_context"):
        c.need_cluster(str(d["kube_context"]))
    if not url:
        if smoke_mode():
            c.skipped_cluster = True
            raise Skip(
                f"cluster tier skipped under OL_SMOKE=1: system.toml has no [deploy].{key}"
            )
        raise Fail(f"system.toml has no [deploy].{key}")
    return url


def test_queries_run_in_prometheus(c: Ctx) -> None:
    # WHY: the deployed Prometheus evaluates every panel query without error.
    # KIND: conformance
    # CHAPTER: obs.05 section 4, The artifact and its check
    import urllib.parse

    prom = _deploy(c, "prometheus")
    errs = []
    for p, e in queries(c):
        status, _, body = c.http(
            "GET",
            f"{prom}/api/v1/query?query={urllib.parse.quote(O4.substitute(e), safe='')}",
            timeout=15,
        )
        if status != 200:
            errs.append(f"panel {p.get('title')!r}: HTTP {status}: {body[:200]!r}")
    if errs:
        raise Fail("\n".join(errs))


def test_train_trace_in_tempo(c: Ctx) -> None:
    # WHY: one TrainRun is one trace, from the CLI through the workflow and
    #      activity spans to sampled Python steps (otel/semconv.md). Tempo
    #      must hold a trace that has all four layers.
    # KIND: conformance
    # CHAPTER: obs.05 section 2.1
    import urllib.parse

    tempo = _deploy(c, "traces")
    q = urllib.parse.quote('{ name = "train.step" }', safe="")
    status, _, body = c.http("GET", f"{tempo}/api/search?q={q}&limit=5", timeout=20)
    if status != 200:
        raise Fail(f"Tempo search: HTTP {status}")
    traces = json.loads(body).get("traces") or []
    if not traces:
        raise Fail(
            "no trace with a train.step span: run `<system> train --spec specs/tiny.json` on kind first"
        )
    need = {"workflow TrainRun", "activity train", "train.run", "train.step"}
    for t in traces:
        status, _, tb = c.http("GET", f"{tempo}/api/traces/{t['traceID']}", timeout=20)
        names = set(re.findall(r'"name"\s*:\s*"([^"]+)"', tb.decode(errors="replace")))
        if need <= names:
            return
    raise Fail(
        f"no train.step trace also holds {sorted(need)}: the context is lost between the worker and Python"
    )


if __name__ == "__main__":
    raise SystemExit(run(globals()))
