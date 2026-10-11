"""obs.02 artifact check: metrics on every service, and logs you can join to traces.

Run by `ol check obs.02` in your repo. Three tiers:

  static   Prometheus scrapes every gateway and engine pod your charts render:
           a ServiceMonitor or PodMonitor under deploy/observability/ selects
           it on its health port (9464) at /metrics, every 10 s or faster,
           names the job <system>-gateway or <system>-engine, and carries
           release: observability; deploy/observability/collector.yaml
           receives OTLP on 4317 and 4318 and sends traces to Tempo and
           metrics to its prometheus exporter, through memory_limiter and batch
  local    your [build] steps, then [services.engine] and [services.gateway]
           started from system.toml (as `ol milestone MS-prod --smoke` does);
           after a few requests, /metrics on each health port serves the
           contract instruments of contracts/otel/metrics.yaml with their
           types, buckets, labels, and label values; capped labels stay under
           their cap when a client sends 80 different model names; both
           services log JSON lines carrying the trace id of a traced request,
           and never the prompt
  cluster  on [deploy].kube_context: every scrape target of both jobs is up in
           [deploy].prometheus, and `kubectl logs` of the gateway finds the
           trace id of a request sent through [deploy].gateway_url

The cluster tier fails when the context is missing or down; with OL_SMOKE=1 it
is skipped with the reason instead.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.practice import Ctx, Fail, Skip, objects, pod_spec, run, smoke_mode  # noqa: E402

OBS = "deploy/observability"
COLLECTOR = f"{OBS}/collector.yaml"
HEALTH_PORT = 9464
MAX_SCRAPE_S = 10  # the drill profile's 25 s window needs two samples
KEY = "tl_check_obs02probe"
CALLER_TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
CALLER_SPAN = "00f067aa0ba902b7"
CANARY = "canary-prompt-7Q2xv"  # must never reach a log line
CORPUS = "Once upon a time a metric met a log line and they shared a trace id.\n" * 8
CAP = 64  # otel/metrics.yaml: a capped label keeps at most 64 values, then _other
# Instruments every Pass 7 service serves (metrics.yaml emitted_by), minus the
# ones a later module or another mode adds: spec_accept_rate (L10.8),
# kv.transfer.* (disaggregated roles), policy.denials (gw.08, Pass 10).
LATER = {
    "tl_engine_spec_accept_rate",
    "tl_kv_transfer_bytes_total",
    "tl_kv_transfer_blocks_total",
    "tl_gateway_policy_denials_total",
}


# -- contract ---------------------------------------------------------------------


def _contract(c: Ctx) -> dict[str, dict]:
    if "contract" not in c.cache:
        doc = c.yaml_file("contracts/otel/metrics.yaml")[0]
        c.cache["contract"] = {i["prometheus"]: i for i in doc["instruments"]}
    return c.cache["contract"]


def required(c: Ctx, component: str) -> list[str]:
    return sorted(
        p
        for p, i in _contract(c).items()
        if component in i.get("emitted_by", []) and p not in LATER
    )


# -- Prometheus text format ---------------------------------------------------------

_SAMPLE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{(.*)\})?\s+(\S+)(\s+\S+)?$")
_LABEL = re.compile(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"\s*,?')


def parse_exposition(text: str) -> tuple[dict[str, str], list[tuple[str, dict, float]]]:
    """(# TYPE per name, samples as (name, labels, value)) of a /metrics body."""
    types, samples = {}, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = line.split(None, 3)
            if len(parts) >= 4 and parts[1] == "TYPE":
                types[parts[2]] = parts[3].strip()
            continue
        m = _SAMPLE.match(line)
        if not m:
            raise ValueError(f"not a sample line: {raw[:120]!r}")
        labels, body = {}, m.group(3) or ""
        pos = 0
        while pos < len(body):
            lm = _LABEL.match(body, pos)
            if not lm:
                raise ValueError(f"bad labels in {raw[:120]!r}")
            labels[lm.group(1)] = re.sub(
                r"\\(.)",
                lambda e: "\n" if e.group(1) == "n" else e.group(1),
                lm.group(2),
            )
            pos = lm.end()
        samples.append((m.group(1), labels, float(m.group(4))))
    return types, samples


def _le(v: str) -> float:
    return float("inf") if v in ("+Inf", "Inf") else float(v)


def contract_errors(c: Ctx, component: str, text: str) -> list[str]:
    """Every way a service's /metrics body breaks contracts/otel/metrics.yaml."""
    try:
        types, samples = parse_exposition(text)
    except ValueError as e:
        return [f"{component}: /metrics is not Prometheus text: {e}"]
    errs = []
    contract = _contract(c)
    by_base: dict[str, list] = {}
    for name, labels, value in samples:
        for base, inst in contract.items():
            names = {base}
            if inst["type"] == "histogram":
                names = {f"{base}_bucket", f"{base}_sum", f"{base}_count"}
            if name in names:
                by_base.setdefault(base, []).append((name, labels, value))
    for base in required(c, component):
        if base not in by_base:
            errs.append(f"{component}: {base} is missing from /metrics")
    for base, rows in sorted(by_base.items()):
        inst = contract[base]
        want = {"histogram": "histogram", "counter": "counter", "gauge": "gauge"}[
            inst["type"]
        ]
        got = types.get(base) or types.get(base.removesuffix("_total"))
        if got != want:
            errs.append(f"{component}: {base} has # TYPE {got}; want {want}")
        attrs = {
            k.replace(".", "_"): v for k, v in (inst.get("attributes") or {}).items()
        }
        allowed = set(attrs) | ({"le"} if inst["type"] == "histogram" else set())
        seen: dict[str, set] = {}
        for name, labels, _ in rows:
            extra = sorted(set(labels) - allowed)
            if extra:
                errs.append(
                    f"{component}: {name} has label(s) {extra}, not in metrics.yaml (allowed: {sorted(allowed - {'le'})})"
                )
            for k, v in labels.items():
                seen.setdefault(k, set()).add(v)
                spec = attrs.get(k) or {}
                if spec.get("values") and v not in spec["values"]:
                    errs.append(
                        f"{component}: {name}{{{k}={v!r}}}: {k} takes only {spec['values']}"
                    )
        for k, vals in seen.items():
            if (attrs.get(k) or {}).get("capped") and len(vals - {"_other"}) > CAP:
                errs.append(
                    f"{component}: {base} has {len(vals)} values of capped label {k}; at most {CAP} plus _other"
                )
        if inst["type"] == "histogram":
            want_le = sorted([float(b) for b in inst["buckets"]] + [float("inf")])
            series: dict[tuple, set] = {}
            for name, labels, _ in rows:
                if name.endswith("_bucket"):
                    key = tuple(sorted((k, v) for k, v in labels.items() if k != "le"))
                    series.setdefault(key, set()).add(_le(labels.get("le", "nan")))
            for key, les in series.items():
                if sorted(les) != want_le:
                    errs.append(
                        f"{component}: {base}{dict(key)} buckets {sorted(les)}; want metrics.yaml's {want_le}"
                    )
                    break
    return list(dict.fromkeys(errs))


# -- static -----------------------------------------------------------------------


def _monitors(c: Ctx) -> list[tuple[str, dict]]:
    if "monitors" not in c.cache:
        out = []
        d = c.path(OBS)
        for f in sorted(d.glob("*.y*ml")) if d.is_dir() else []:
            rel = str(f.relative_to(c.root))
            for doc in c.yaml_file(rel):
                if isinstance(doc, dict) and doc.get("kind") in (
                    "ServiceMonitor",
                    "PodMonitor",
                ):
                    out.append((rel, doc))
        c.cache["monitors"] = out
    return c.cache["monitors"]


def _render(c: Ctx, part: str, role: str = "") -> list:
    n = c.system_name()
    if not role:
        return c.helm_template(f"deploy/helm/{n}-{part}", f"{n}-{part}", namespace=n)
    # Pass 7 installs one engine release per role (MS-prod: <system>-engine-decode).
    return c.helm_template(
        f"deploy/helm/{n}-{part}",
        f"{n}-{part}-{role}",
        sets={"role": role},
        namespace=n,
    )


def _job_override(ep: dict) -> str:
    """A relabeling that sets job to a fixed value, as the job's name."""
    for rl in ep.get("relabelings") or []:
        if (
            rl.get("targetLabel") == "job"
            and rl.get("action", "replace") == "replace"
            and not rl.get("sourceLabels")
            and rl.get("replacement")
        ):
            return str(rl["replacement"])
    return ""


def _selects(sel: dict, labels: dict) -> bool:
    ml = (sel or {}).get("matchLabels") or {}
    if not ml and not (sel or {}).get("matchExpressions"):
        return False  # an empty selector is a bug, not "everything"
    if any(labels.get(k) != v for k, v in ml.items()):
        return False
    for e in (sel or {}).get("matchExpressions") or []:
        k, op, vals = e.get("key"), e.get("operator"), e.get("values") or []
        if op == "In" and labels.get(k) not in vals:
            return False
        if op == "NotIn" and labels.get(k) in vals:
            return False
        if op == "Exists" and k not in labels:
            return False
        if op == "DoesNotExist" and k in labels:
            return False
    return True


def _in_namespace(mon: dict, ns: str) -> bool:
    nss = mon.get("spec", {}).get("namespaceSelector") or {}
    if nss.get("any"):
        return True
    if nss.get("matchNames"):
        return ns in nss["matchNames"]
    return (mon.get("metadata") or {}).get("namespace", ns) == ns


def _scrape(
    c: Ctx, part: str, workload: dict, role: str = ""
) -> tuple[list[str], dict | None, str]:
    """(problems, the monitor endpoint that scrapes the workload, its job name)."""
    ns = c.system_name()
    pod_labels = (workload["spec"]["template"].get("metadata") or {}).get(
        "labels"
    ) or {}
    ctr_ports = {
        p.get("name"): int(p["containerPort"])
        for ctr in pod_spec(workload).get("containers") or []
        for p in ctr.get("ports") or []
    }
    svcs = objects(_render(c, part, role), "Service")
    for rel, mon in _monitors(c):
        spec = mon.get("spec") or {}
        if not _in_namespace(mon, ns):
            continue
        if mon["kind"] == "PodMonitor":
            if not _selects(spec.get("selector"), pod_labels):
                continue
            for ep in spec.get("podMetricsEndpoints") or []:
                port = ep.get("port")
                if ctr_ports.get(port) == HEALTH_PORT or str(
                    ep.get("targetPort")
                ) == str(HEALTH_PORT):
                    job = _job_override(ep) or pod_labels.get(
                        spec.get("jobLabel", ""),
                        f"{(mon.get('metadata') or {}).get('namespace', ns)}/{mon['metadata'].get('name')}",
                    )
                    return [], dict(ep, _mon=mon, _rel=rel), job
        else:
            for svc in svcs:
                sel = (svc.get("spec") or {}).get("selector") or {}
                if not sel or any(pod_labels.get(k) != v for k, v in sel.items()):
                    continue
                if not _selects(
                    spec.get("selector"),
                    (svc.get("metadata") or {}).get("labels") or {},
                ):
                    continue
                for ep in spec.get("endpoints") or []:
                    for sp in (svc.get("spec") or {}).get("ports") or []:
                        tgt = sp.get("targetPort", sp.get("port"))
                        cport = ctr_ports.get(tgt, tgt if str(tgt).isdigit() else None)
                        if (
                            ep.get("port") == sp.get("name")
                            and int(cport or 0) == HEALTH_PORT
                        ):
                            labels = (svc.get("metadata") or {}).get("labels") or {}
                            job = _job_override(ep) or labels.get(
                                spec.get("jobLabel", ""), svc["metadata"]["name"]
                            )
                            return [], dict(ep, _mon=mon, _rel=rel), job
    return (
        [
            f"no ServiceMonitor or PodMonitor in {OBS}/ scrapes {workload['kind']} {workload['metadata']['name']} on its health port {HEALTH_PORT}"
        ],
        None,
        "",
    )


def _workloads(c: Ctx) -> list[tuple[str, str, dict]]:
    """(part, role, workload) for the gateway and for the engine released as
    unified and as each Pass 7 role (prefill, decode)."""
    out = []
    for part, roles in (("gateway", [""]), ("engine", ["", "prefill", "decode"])):
        for role in roles:
            docs = _render(c, part, role)
            for w in objects(docs, "Deployment") + objects(docs, "StatefulSet"):
                out.append((part, role, w))
    if not any(p == "gateway" for p, _, _ in out) or not any(
        p == "engine" for p, _, _ in out
    ):
        raise Fail(
            "the gateway and engine charts must each render a Deployment or StatefulSet"
        )
    return out


def test_every_pod_is_scraped_on_its_health_port(c: Ctx) -> None:
    # WHY: Prometheus finds targets only through ServiceMonitor or PodMonitor
    #      objects. Every gateway and engine pod your charts render (every
    #      engine role) must be selected, on the port where /metrics lives
    #      (9464), with path /metrics. A monitor that selects nothing is valid
    #      YAML and an empty graph.
    # KIND: conformance
    # CHAPTER: obs.02 section 5, Pitfall 1
    errs = []
    for part, role, w in _workloads(c):
        why, ep, _ = _scrape(c, part, w, role)
        errs += why
        if ep and ep.get("path", "/metrics") != "/metrics":
            errs.append(
                f"{ep['_rel']}: path {ep.get('path')!r}; your services serve /metrics"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_monitors_are_selected_by_the_stack(c: Ctx) -> None:
    # WHY: kube-prometheus-stack picks up only monitors labelled release:
    #      observability (contracts/helm/observability.md); one without the
    #      label is ignored without a word.
    # KIND: conformance
    errs = []
    if not _monitors(c):
        raise Fail(f"no ServiceMonitor or PodMonitor under {OBS}/")
    for rel, mon in _monitors(c):
        lab = (mon.get("metadata") or {}).get("labels") or {}
        if lab.get("release") != "observability":
            errs.append(
                f"{rel}: {mon['kind']} {mon['metadata'].get('name')} lacks label release: observability"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_scrape_job_names_and_interval(c: Ctx) -> None:
    # WHY: SLO rules (obs.03) and dashboards (obs.04) select series by
    #      job="<system>-gateway" and job="<system>-engine". A PodMonitor names
    #      the job "<namespace>/<monitor>" unless jobLabel says otherwise. The
    #      drill profile's 25 s window needs two samples, so scrape every 10 s
    #      or faster (the stack's default is 30 s).
    # KIND: unit
    # CHAPTER: obs.02 section 5, Pitfall 2
    n = c.system_name()
    errs = []
    for part, role, w in _workloads(c):
        _, ep, job = _scrape(c, part, w, role)
        if ep is None:
            continue
        if job != f"{n}-{part}":
            errs.append(
                f"{ep['_rel']}: {w['metadata']['name']} is scraped as job {job!r}; want {n}-{part} (set jobLabel)"
            )
        iv = str(ep.get("interval", ""))
        secs = sum(
            float(x) * {"s": 1, "m": 60}[u]
            for x, u in re.findall(r"(\d+(?:\.\d+)?)([sm])", iv)
        )
        if not iv or secs > MAX_SCRAPE_S:
            errs.append(
                f"{ep['_rel']}: scrape interval {iv or 'unset (30s)'} for {part}; want {MAX_SCRAPE_S}s or less"
            )
    if errs:
        raise Fail("\n".join(dict.fromkeys(errs)))


def _collector(c: Ctx) -> dict:
    docs = c.yaml_file(COLLECTOR)
    doc = docs[0] if docs and isinstance(docs[0], dict) else {}
    # A raw collector config, values for the opentelemetry-collector chart
    # (`config:`), or values for an umbrella chart that aliases it
    # (`otel-collector: {config: ...}`, the dep.03 reference).
    for cand in (
        doc,
        doc.get("config"),
        (doc.get("otel-collector") or {}).get("config"),
        (doc.get("opentelemetry-collector") or {}).get("config"),
    ):
        if isinstance(cand, dict) and "service" in cand:
            return cand
    raise Fail(
        f"{COLLECTOR}: no collector config (receivers, processors, exporters, service) at the top, under `config:`, or under `otel-collector.config:`"
    )


def test_collector_pipelines(c: Ctx) -> None:
    # WHY: one door for telemetry (DESIGN 2.11): OTLP gRPC 4317 and HTTP 4318
    #      in; traces out to Tempo; the Python subprocesses' pushed metrics out
    #      through the prometheus exporter that Prometheus scrapes. Every
    #      pipeline starts with memory_limiter (a collector that OOMs loses
    #      everything) and batches before exporting.
    # KIND: unit
    # CHAPTER: obs.02 section 5, Pitfall 5
    cfg = _collector(c)
    errs = []
    otlp = (cfg.get("receivers") or {}).get("otlp") or {}
    protos = otlp.get("protocols") or {}
    for proto, port in (("grpc", "4317"), ("http", "4318")):
        ep = str((protos.get(proto) or {}).get("endpoint", ""))
        if proto not in protos or (ep and not ep.endswith(f":{port}")):
            errs.append(
                f"receivers.otlp.protocols.{proto} must listen on :{port} (got {ep or 'nothing'})"
            )
        elif ep.startswith(("localhost", "127.0.0.1")):
            errs.append(
                f"receivers.otlp.protocols.{proto}.endpoint {ep} accepts only in-pod clients; use 0.0.0.0:{port} or ${{env:MY_POD_IP}}:{port}"
            )
    pipes = (cfg.get("service") or {}).get("pipelines") or {}
    exporters = cfg.get("exporters") or {}

    def pipe(kind: str) -> list[tuple[str, dict]]:
        return [
            (k, v or {})
            for k, v in pipes.items()
            if k == kind or k.startswith(kind + "/")
        ]

    traces = pipe("traces")
    if not any(
        any(
            str((exporters.get(e) or {}).get("endpoint", "")).find("tempo") >= 0
            for e in p.get("exporters") or []
        )
        for _, p in traces
    ):
        errs.append(
            "no traces pipeline exports to Tempo (an otlp exporter whose endpoint names the tempo Service)"
        )
    if not any(
        any(
            e == "prometheus" or e.startswith("prometheus/")
            for e in p.get("exporters") or []
        )
        for _, p in pipe("metrics")
    ):
        errs.append(
            "no metrics pipeline exports through the prometheus exporter (Python's pushed metrics)"
        )
    for name, p in traces + pipe("metrics"):
        procs = p.get("processors") or []
        if not procs or not str(procs[0]).startswith("memory_limiter"):
            errs.append(
                f"pipeline {name}: memory_limiter must be the first processor (got {procs})"
            )
        if not any(str(x).startswith("batch") for x in procs):
            errs.append(f"pipeline {name}: no batch processor")
        if not any(str(r).startswith("otlp") for r in p.get("receivers") or []):
            errs.append(f"pipeline {name}: does not receive from otlp")
    if errs:
        raise Fail(f"{COLLECTOR}:\n" + "\n".join(errs))


# -- local ------------------------------------------------------------------------


def _harness():
    try:
        from olcourse import placeholders, services, system
    except ImportError:
        raise Fail(
            "the course harness is not importable: run this check through `ol check obs.02`"
        ) from None
    return placeholders, services, system


def _model(c: Ctx) -> str:
    if "model" not in c.cache:
        argv = (c.system().get("entry") or {}).get("tinyllm")
        if not argv:
            raise Fail(
                "system.toml has no [entry].tinyllm to train the model your engine serves"
            )
        work = c.path(".ol/check/obs.02")
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


def _stack(c: Ctx) -> dict:
    """Engine and gateway as local processes, warmed with traffic; their scrapes and logs."""
    if "stack" in c.cache:
        if c.cache["stack"] is None:
            raise Fail(
                "the local stack did not start (see test_services_serve_contract_metrics)"
            )
        return c.cache["stack"]
    c.cache["stack"] = None
    ph, services, system = _harness()
    model = _model(c)
    sysm = system.load(c.root)
    log = c.path(".ol/check/obs.02/run")
    log.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)  # no collector: logs and metrics only
    env[sysm.api_key_env] = KEY
    ok, why = services.run_build(sysm, log / "build.log", env)
    if not ok:
        raise Fail(f"your [build] steps failed: {why}")
    entries = sysm.raw.get("entry", {})
    lookup = ph.chain(
        {"model_dir": model, "out": str(log), "system": sysm.name}.get,
        lambda k: list(entries[k]) if k in entries else None,
        lambda k: (
            ph.from_dict(sysm.deploy, "")(k[len("deploy.") :])
            if k.startswith("deploy.")
            else None
        ),
    )
    stack = services.Stack(sysm, log, lookup, env, "ol check obs.02")
    c.cleanups.append(stack.stop)
    try:
        stack.start(["engine", "gateway"])
    except (
        Exception
    ) as e:  # ServiceError or HarnessError: a failed check, with the reason
        raise Fail(str(e)) from None
    gw, eng = stack.instances["gateway"], stack.instances["engine"]
    model_id = (sysm.raw.get("endpoints") or {}).get("model", "tracer")
    hdr = {"Authorization": f"Bearer {KEY}"}

    def chat(
        base: str,
        model: str,
        prompt: str,
        extra: dict | None = None,
        stream: bool = False,
    ) -> int:
        st, _, _ = c.http(
            "POST",
            base + "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8,
                "temperature": 0,
                "stream": stream,
            },
            dict(hdr, **(extra or {})),
            timeout=30,
        )
        return st

    codes = [chat(gw.base, model_id, f"Once {i}", stream=bool(i % 2)) for i in range(6)]
    if not any(s == 200 for s in codes):
        raise Fail(
            f"no chat completion through your gateway succeeded (statuses {codes})"
        )
    traced = chat(
        gw.base,
        model_id,
        f"{CANARY} Once upon a time",
        {"traceparent": f"00-{CALLER_TRACE}-{CALLER_SPAN}-01"},
    )
    for i in range(80):  # a client sending random model names: the cardinality attack
        chat(gw.base, f"model-{i:03d}", "Once")
        chat(eng.base, f"model-{i:03d}", "Once")
    scrapes = {}
    for name, inst in (("gateway", gw), ("engine", eng)):
        st, _, body = c.http(
            "GET", f"http://127.0.0.1:{inst.ports['health_port']}/metrics", timeout=10
        )
        scrapes[name] = (st, body.decode("utf-8", "replace"))
    time.sleep(0.5)  # let log lines of the last requests reach the files
    c.cache["stack"] = {
        "stack": stack,
        "scrapes": scrapes,
        "traced": traced,
        "logs": {n: stack.instances[n].log for n in ("gateway", "engine")},
    }
    return c.cache["stack"]


def test_services_serve_contract_metrics(c: Ctx) -> None:
    # WHY: Prometheus and the course tests read the same thing: GET /metrics on
    #      the health port, in Prometheus text format, with the names, types,
    #      buckets, labels, and label values of contracts/otel/metrics.yaml.
    #      SLO rules compare bucket le="0.5"; a histogram with other bounds, a
    #      gauge served as a counter, or a renamed metric breaks them silently.
    # KIND: conformance
    # CHAPTER: obs.02 section 4, The interface
    errs = []
    for name, (status, body) in _stack(c)["scrapes"].items():
        if status != 200:
            errs.append(f"{name}: GET /metrics on the health port answered {status}")
            continue
        errs += [e for e in contract_errors(c, name, body) if "capped label" not in e]
    if errs:
        raise Fail("\n".join(errs))


def test_capped_labels_stay_under_the_cap(c: Ctx) -> None:
    # WHY: a label that copies client input (the model name) makes one series
    #      per distinct value: 80 random model names are 80 x 16 buckets of
    #      TTFT alone, and an attacker can send millions. Capped labels keep 64
    #      values per process and record the rest as _other.
    # KIND: fault
    # CHAPTER: obs.02 section 5, Pitfall 3
    errs = []
    for name, (status, body) in _stack(c)["scrapes"].items():
        if status != 200:
            continue
        errs += [e for e in contract_errors(c, name, body) if "capped label" in e]
    if errs:
        raise Fail("\n".join(errs))


def _json_lines(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def test_logs_carry_the_trace_id(c: Ctx) -> None:
    # WHY: there is no log backend (DESIGN 2.11): joining logs to a trace is
    #      `kubectl logs | grep <trace id>`. That works only if every line a
    #      service writes while handling a request is JSON with trace_id and
    #      span_id. The traced request here continues the caller's trace, so
    #      both services must log its trace id.
    # KIND: conformance
    # CHAPTER: obs.02 section 3, Worked example
    st = _stack(c)
    if st["traced"] != 200:
        raise Fail(f"the traced request answered {st['traced']}")
    errs = []
    for name, path in st["logs"].items():
        lines = _json_lines(path)
        if not lines:
            errs.append(f"{name}: no JSON log lines in {path.relative_to(c.root)}")
            continue
        hits = [x for x in lines if x.get("trace_id") == CALLER_TRACE]
        if not hits:
            errs.append(
                f"{name}: no log line carries trace_id {CALLER_TRACE} (the traced request's trace)"
            )
            continue
        for x in hits:
            missing = [
                k for k in ("ts", "level", "msg", "service", "span_id") if k not in x
            ]
            if missing or not re.fullmatch(r"[0-9a-f]{16}", str(x.get("span_id", ""))):
                errs.append(
                    f"{name}: line {json.dumps(x)[:160]} lacks {missing or 'a 16-hex span_id'} (otel/semconv.md, Logs)"
                )
                break
    if errs:
        raise Fail("\n".join(errs))


def test_logs_never_contain_the_prompt(c: Ctx) -> None:
    # WHY: logs are kept longer and read by more people than any request;
    #      a prompt in a log line is user data in the wrong place (semconv.md:
    #      no prompt, completion, or key in logs).
    # KIND: boundary
    # CHAPTER: obs.02 section 5, Pitfall 4
    st = _stack(c)
    errs = [
        f"{n}: a log line contains the prompt"
        for n, p in st["logs"].items()
        if CANARY in p.read_text(errors="replace")
    ]
    errs += [
        f"{n}: a log line contains the API key"
        for n, p in st["logs"].items()
        if KEY in p.read_text(errors="replace")
    ]
    if errs:
        raise Fail("\n".join(errs))


# -- cluster ----------------------------------------------------------------------


def _deploy(c: Ctx) -> dict:
    d = c.system().get("deploy") or {}
    if not d.get("kube_context"):
        if smoke_mode():
            c.skipped_cluster = True
            raise Skip(
                "cluster tier skipped under OL_SMOKE=1: system.toml has no [deploy].kube_context"
            )
        raise Fail("system.toml has no [deploy].kube_context")
    c.need_cluster(str(d["kube_context"]))
    if not d.get("prometheus"):
        raise Fail("system.toml has no [deploy].prometheus (http://127.0.0.1:30090)")
    return d


def test_targets_up_in_prometheus(c: Ctx) -> None:
    # WHY: the deployed proof of the static tier: Prometheus has scrape
    #      targets for both jobs and every one is up.
    # KIND: conformance
    import urllib.parse

    d = _deploy(c)
    n = c.system_name()
    q = urllib.parse.quote(f'up{{job=~"{n}-(gateway|engine)"}}', safe="")
    status, _, body = c.http(
        "GET", f"{str(d['prometheus']).rstrip('/')}/api/v1/query?query={q}", timeout=10
    )
    if status != 200:
        raise Fail(f"Prometheus answered {status}")
    res = json.loads(body)["data"]["result"]
    jobs = {r["metric"].get("job") for r in res}
    down = [r["metric"].get("instance") for r in res if r["value"][1] != "1"]
    if jobs != {f"{n}-gateway", f"{n}-engine"} or down:
        raise Fail(
            f"scrape targets: jobs {sorted(jobs)} (want {n}-gateway and {n}-engine), down: {down}"
        )


def test_kubectl_logs_find_the_trace(c: Ctx) -> None:
    # WHY: the correlation recipe of the chapter, on the cluster: a request
    #      with a known traceparent through the gateway NodePort, then
    #      `kubectl logs deploy/<system>-gateway | grep <trace id>`.
    # KIND: conformance
    d = _deploy(c)
    env_name = (c.system().get("endpoints") or {}).get("api_key_env", "TL_API_KEY")
    key = os.environ.get(env_name)
    if not key:
        raise Fail(f"export {env_name} with the key stored in your API key Secret")
    trace = "5bf92f3577b34da6a3ce929d0e0e4737"
    model = (c.system().get("endpoints") or {}).get("model", "tracer")
    c.http(
        "POST",
        str(d["gateway_url"]).rstrip("/") + "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Once"}],
            "max_tokens": 4,
        },
        {
            "Authorization": f"Bearer {key}",
            "traceparent": f"00-{trace}-{CALLER_SPAN}-01",
        },
        timeout=30,
    )
    target = (
        (d.get("services") or {}).get("gateway")
    ) or f"deploy/{c.system_name()}-gateway"
    out = c.sh(
        [
            "kubectl",
            "--context",
            d["kube_context"],
            "-n",
            d.get("namespace", c.system_name()),
            "logs",
            target,
            "--since=5m",
        ],
        timeout=30,
    ).stdout
    if trace not in out:
        raise Fail(f"`kubectl logs {target}` has no line with trace id {trace}")


if __name__ == "__main__":
    raise SystemExit(run(globals()))
