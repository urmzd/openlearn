"""obs.04 artifact check: dashboards as code for the serving path.

Run by `ol check obs.04` in your repo. Two tiers:

  static   every deploy/observability/dashboards/*.json is a portable Grafana
           dashboard (stable uid, no instance id, a Prometheus datasource that
           is a variable or the stack's "prometheus" uid); every panel query
           parses (promtool) and names only contract metrics
           (contracts/otel/metrics.yaml), ALERTS, or recording rules your
           rules files define; and the required panels exist: SLO status for
           TTFT, TPOT, and availability, TTFT and TPOT heatmaps, KV cache
           usage, and per-tenant usage
  cluster  every panel query runs without error in [deploy].prometheus, and a
           ConfigMap labelled grafana_dashboard=1 carries each dashboard uid
           (the Grafana sidecar of kube-prometheus-stack loads those)

Grafana variables are replaced before parsing: $__rate_interval, $__interval,
and $__range by 5m, every other $var or ${var} by .* (use them inside label
matchers, `tenant=~"$tenant"`).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "obs.03"))

import _promtool  # noqa: E402  (shared with obs.03)
from _lib.practice import Ctx, Fail, Skip, run, smoke_mode  # noqa: E402

DASH_DIR = "deploy/observability/dashboards"
RULES_DIR = "deploy/observability/rules"
TTFT = "gen_ai_server_time_to_first_token_seconds"
TPOT = "gen_ai_server_time_per_output_token_seconds"
HTTP = "http_server_request_duration_seconds"
SLO_METRIC = {"ttft": TTFT, "tpot": TPOT, "availability": HTTP}
STATUS_TYPES = {
    "stat",
    "gauge",
    "bargauge",
    "table",
    "state-timeline",
    "status-history",
}
# The kube-prometheus-stack provisions its Prometheus datasource with this uid.
STACK_DS_UID = "prometheus"
KEYWORDS = {
    "and",
    "or",
    "unless",
    "bool",
    "offset",
    "by",
    "without",
    "on",
    "ignoring",
    "group_left",
    "group_right",
    "inf",
    "nan",
    "Inf",
    "NaN",
    "atan2",
    "start",
    "end",
}


# -- loading ----------------------------------------------------------------------


def _dashboards(c: Ctx) -> dict[str, dict]:
    if "dash" not in c.cache:
        d = c.path(DASH_DIR)
        files = sorted(d.glob("*.json")) if d.is_dir() else []
        if not files:
            raise Fail(f"no dashboards: {DASH_DIR}/*.json is empty")
        out = {}
        for f in files:
            try:
                out[str(f.relative_to(c.root))] = json.loads(f.read_text())
            except json.JSONDecodeError as e:
                raise Fail(f"{f.relative_to(c.root)} is not JSON: {e}") from None
        c.cache["dash"] = out
    return c.cache["dash"]


def _panels(dash: dict) -> list[dict]:
    """Every panel, rows' nested panels included, rows themselves excluded."""
    out = []
    for p in dash.get("panels") or []:
        if p.get("type") == "row":
            out += [q for q in p.get("panels") or [] if q.get("type") != "row"]
        else:
            out.append(p)
    return out


def _queries(c: Ctx) -> list[tuple[str, dict, dict]]:
    """(file, panel, target) for every Prometheus query of every panel."""
    out = []
    for rel, dash in _dashboards(c).items():
        for p in _panels(dash):
            for t in p.get("targets") or []:
                if str(t.get("expr", "")).strip():
                    out.append((rel, p, t))
    if not out:
        raise Fail("the dashboards hold no panel queries (targets[].expr)")
    return out


def _contract_names(c: Ctx) -> set[str]:
    doc = c.yaml_file("contracts/otel/metrics.yaml")[0]
    names = set()
    for i in doc["instruments"]:
        p = i["prometheus"]
        names |= (
            {f"{p}_bucket", f"{p}_sum", f"{p}_count"}
            if i["type"] == "histogram"
            else {p}
        )
    return names


def _recording_rules(c: Ctx) -> dict[str, list[tuple[dict, str]]]:
    """record name -> [(labels, expr)] from every rules file you keep."""
    if "rec" not in c.cache:
        out: dict[str, list] = {}
        d = c.path(RULES_DIR)
        for f in sorted(d.glob("*.y*ml")) if d.is_dir() else []:
            for doc in c.yaml_file(str(f.relative_to(c.root))):
                if not isinstance(doc, dict):
                    continue
                groups = (
                    (doc.get("spec") or {}).get("groups") or doc.get("groups") or []
                )
                for g in groups:
                    for r in g.get("rules") or []:
                        if "record" in r:
                            out.setdefault(str(r["record"]), []).append(
                                (r.get("labels") or {}, str(r.get("expr", "")))
                            )
        c.cache["rec"] = out
    return c.cache["rec"]


def substitute(expr: str) -> str:
    """The query Prometheus would receive, with Grafana variables filled in."""
    s = re.sub(r"\$\{?__(rate_interval|interval|range)(_ms|_s)?\}?", "5m", expr)
    # $1 in label_replace is PromQL's own, not a variable: names start with a letter.
    s = re.sub(r"\[\[\s*\w+\s*\]\]|\$\{[A-Za-z_]\w*(:\w+)?\}|\$[A-Za-z_]\w*", ".*", s)
    return s


def metric_names(expr: str) -> set[str]:
    """Metric names a PromQL expression selects (functions, keywords, label
    names, durations, and strings excluded)."""
    s = re.sub(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|`[^`]*`', '""', expr)
    s = re.sub(r"\{[^}]*\}", "{}", s)
    s = re.sub(r"\[[^\]]*\]", "[]", s)
    s = re.sub(r"\b(by|without|on|ignoring|group_left|group_right)\s*\([^)]*\)", " ", s)
    names = set()
    for m in re.finditer(r"(?<![\w.])[A-Za-z_:][A-Za-z0-9_:]*", s):
        name = m.group(0)
        rest = s[m.end() :].lstrip()
        if name in KEYWORDS or rest.startswith("("):
            continue
        names.add(name)
    return names


def _selectors(expr: str, name: str) -> list[str]:
    """The label-matcher bodies of every selector of `name` in expr ('' when bare)."""
    out = []
    for m in re.finditer(rf"(?<![\w:]){re.escape(name)}\s*(\{{([^}}]*)\}})?", expr):
        out.append(m.group(2) or "")
    return out


def slos_covered(c: Ctx, expr: str) -> set[str]:
    """Which SLOs a query reports on: through the SLO's own metric, an
    ALERTS selector naming its alerts, or a recording rule built from it
    (narrowed by an slo="..." matcher when the rule records several)."""
    got = set()
    names = metric_names(expr)
    for slo, metric in SLO_METRIC.items():
        if any(n.startswith(metric) for n in names):
            got.add(slo)
    if "ALERTS" in names:
        for body in _selectors(expr, "ALERTS") or [""]:
            m = re.search(r'alertname\s*(=~|=)\s*"([^"]*)"', body)
            pat = m.group(2) if m else ".*"
            for slo, prefix in (
                ("ttft", "TTFT"),
                ("tpot", "TPOT"),
                ("availability", "Availability"),
            ):
                if any(
                    re.fullmatch(pat, f"{prefix}BudgetBurn{w}")
                    for w in ("Fast", "Slow")
                ):
                    got.add(slo)
    for name, recs in _recording_rules(c).items():
        if name not in names:
            continue
        for labels, rexpr in recs:
            rec_slos = {
                s
                for s, mt in SLO_METRIC.items()
                if any(n.startswith(mt) for n in metric_names(rexpr))
            }
            for body in _selectors(expr, name):
                m = re.search(r'\bslo\s*(=~|=)\s*"([^"]*)"', body)
                if (
                    m
                    and "slo" in labels
                    and not re.fullmatch(m.group(2), str(labels["slo"]))
                ):
                    continue
                got |= rec_slos
    return got


# -- static -----------------------------------------------------------------------


def test_dashboards_are_portable_grafana_json(c: Ctx) -> None:
    # WHY: a dashboard is code only if a fresh Grafana loads it unchanged: a
    #      stable uid (links and the sidecar key on it), unique across files,
    #      and no numeric "id" from the instance it was exported from.
    # KIND: unit
    # CHAPTER: obs.04 section 5, Pitfall 1
    errs, uids = [], {}
    for rel, d in _dashboards(c).items():
        uid = str(d.get("uid") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", uid):
            errs.append(f"{rel}: uid {uid!r} must be 1 to 40 of [A-Za-z0-9_-]")
        elif uid in uids:
            errs.append(f"{rel}: uid {uid!r} is also used by {uids[uid]}")
        uids[uid] = rel
        if d.get("id") is not None:
            errs.append(
                f'{rel}: "id": {d.get("id")} is the exporting instance\'s database id; set it to null'
            )
        if (
            not d.get("title")
            or not isinstance(d.get("panels"), list)
            or not d.get("schemaVersion")
        ):
            errs.append(
                f"{rel}: a dashboard needs title, schemaVersion, and a panels list"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_panels_use_a_portable_prometheus_datasource(c: Ctx) -> None:
    # WHY: an exported dashboard pins the datasource uid of the Grafana it came
    #      from, and every panel says "datasource not found" anywhere else.
    #      Point panels at a datasource variable (type datasource, query
    #      prometheus) or at the stack's provisioned uid "prometheus".
    # KIND: unit
    # CHAPTER: obs.04 section 5, Pitfall 2
    errs = []
    for rel, d in _dashboards(c).items():
        dsvars = {
            v.get("name")
            for v in (d.get("templating") or {}).get("list") or []
            if v.get("type") == "datasource" and v.get("query") == "prometheus"
        }
        for p in _panels(d):
            for where, ds in [("panel", p.get("datasource"))] + [
                (f"target {t.get('refId')}", t.get("datasource"))
                for t in p.get("targets") or []
            ]:
                if ds is None:
                    continue
                uid = ds.get("uid") if isinstance(ds, dict) else str(ds)
                typ = ds.get("type") if isinstance(ds, dict) else None
                var = re.fullmatch(r"\$\{?(\w+)\}?", str(uid or ""))
                if var and var.group(1) in dsvars:
                    continue
                if uid == STACK_DS_UID and typ in (None, "prometheus"):
                    continue
                errs.append(
                    f"{rel}: panel {p.get('title')!r} {where} datasource {ds!r}: use a prometheus datasource variable or uid {STACK_DS_UID!r}"
                )
    if errs:
        raise Fail("\n".join(dict.fromkeys(errs)))


def test_every_query_names_contract_metrics(c: Ctx) -> None:
    # WHY: a panel is only as good as the series it reads. Every metric a query
    #      selects is in contracts/otel/metrics.yaml (with its _bucket, _sum,
    #      _count, or _total suffix), is ALERTS, or is a recording rule your
    #      rules files define; a typo or an invented name draws an empty panel
    #      that looks like "no traffic".
    # KIND: conformance
    # CHAPTER: obs.04 section 5, Pitfall 3
    ok = _contract_names(c) | set(_recording_rules(c)) | {"ALERTS", "ALERTS_FOR_STATE"}
    errs = []
    for rel, p, t in _queries(c):
        bad = sorted(metric_names(substitute(t["expr"])) - ok)
        if bad:
            errs.append(
                f"{rel}: panel {p.get('title')!r}: {', '.join(bad)} is not a contract metric or one of your recording rules"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_every_query_parses(c: Ctx) -> None:
    # WHY: Grafana shows a PromQL syntax error only when someone opens the
    #      panel. promtool parses every query (as a recording rule) now.
    # KIND: conformance
    import yaml

    qs = _queries(c)
    rules = [
        {"record": f"dashboard:query{i}", "expr": substitute(t["expr"])}
        for i, (_, _, t) in enumerate(qs)
    ]
    r = _promtool.run(
        c,
        c.path(".ol/check/obs.04/parse"),
        ["check", "rules", "queries.yaml"],
        {
            "queries.yaml": yaml.safe_dump(
                {"groups": [{"name": "dashboards", "rules": rules}]}, sort_keys=False
            )
        },
        timeout=300,
    )
    if r.returncode != 0:
        out = (r.stdout + r.stderr).strip()
        hint = "\n".join(
            f"  dashboard:query{i} = {rel}: {p.get('title')!r}"
            for i, (rel, p, _) in enumerate(qs)
        )
        raise Fail(f"promtool rejects a query:\n{out[-1500:]}\nwhere\n{hint}")


def test_slo_status_panels_cover_the_three_slos(c: Ctx) -> None:
    # WHY: the first row answers "are we within our SLOs right now?": one
    #      status panel (stat, gauge, bar gauge, table, or state timeline) per
    #      SLO, reading its SLI, its burn-rate recording rule, or its alerts.
    # KIND: unit
    # CHAPTER: obs.04 section 4, The interface
    covered: set[str] = set()
    for _, p, t in _queries(c):
        if p.get("type") in STATUS_TYPES:
            covered |= slos_covered(c, t["expr"])
    missing = sorted(set(SLO_METRIC) - covered)
    if missing:
        raise Fail(
            f"no status panel ({', '.join(sorted(STATUS_TYPES))}) reports on SLO(s) {missing}"
        )


def test_latency_heatmaps(c: Ctx) -> None:
    # WHY: a p95 line hides the shape: a heatmap of the TTFT and TPOT
    #      histograms shows the second mode a cold prefix cache or a preempted
    #      batch adds. It is right only when the query sums bucket rates by le
    #      and the target's format is "heatmap" (otherwise Grafana stacks the
    #      cumulative buckets and every cell is wrong).
    # KIND: unit
    # CHAPTER: obs.04 section 5, Pitfall 4
    errs = []
    for label, metric in (("TTFT", TTFT), ("TPOT", TPOT)):
        cand = [
            (rel, p, t)
            for rel, p, t in _queries(c)
            if p.get("type") == "heatmap"
            and f"{metric}_bucket" in metric_names(t["expr"])
        ]
        if not cand:
            errs.append(f"no heatmap panel queries {metric}_bucket ({label})")
            continue
        good = [
            1
            for _, _, t in cand
            if re.search(r"\bby\s*\([^)]*\ble\b[^)]*\)", t["expr"])
            and t.get("format") == "heatmap"
        ]
        if not good:
            rel, p, t = cand[0]
            errs.append(
                f'{rel}: heatmap {p.get("title")!r} must sum rates by (le) and set the target\'s "format": "heatmap" (format is {t.get("format")!r})'
            )
    if errs:
        raise Fail("\n".join(errs))


def test_kv_cache_usage_panel(c: Ctx) -> None:
    # WHY: KV blocks are the engine's capacity: free, used, and cached blocks
    #      (tl_engine_kv_blocks{state}) show a leak (used never falls) and
    #      preemption pressure before TTFT does.
    # KIND: unit
    found = [
        p for _, p, t in _queries(c) if "tl_engine_kv_blocks" in metric_names(t["expr"])
    ]
    if not found:
        raise Fail("no panel queries tl_engine_kv_blocks")


def test_per_tenant_usage_panel(c: Ctx) -> None:
    # WHY: ops.09 (noisy neighbor) starts on this panel: request rate (or
    #      errors) per tenant, from tl_gateway_requests_total broken out by
    #      the tenant label. Summed over tenants, one tenant's flood is invisible.
    # KIND: unit
    # CHAPTER: obs.04 section 5, Pitfall 5
    ok = [
        p
        for _, p, t in _queries(c)
        if "tl_gateway_requests_total" in metric_names(t["expr"])
        and re.search(r"\bby\s*\([^)]*\btenant\b[^)]*\)", t["expr"])
    ]
    if not ok:
        raise Fail("no panel shows tl_gateway_requests_total by (tenant)")


def test_rate_windows_survive_the_scrape_interval(c: Ctx) -> None:
    # WHY: rate() needs at least two samples in its window. A fixed [1m] at a
    #      30 s scrape interval returns nothing whenever one scrape is late;
    #      Grafana's $__rate_interval is always at least four scrape intervals.
    # KIND: boundary
    errs = []
    for rel, p, t in _queries(c):
        for fn, win in re.findall(
            r"\b(rate|irate|increase)\s*\([^\[]*\[([^\]]+)\]", t["expr"]
        ):
            if "$__rate_interval" in win or "${__rate_interval}" in win:
                continue
            secs = sum(
                float(n) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[u]
                for n, u in re.findall(r"(\d+)([smhd])", win)
            )
            if secs < 120:
                errs.append(
                    f"{rel}: panel {p.get('title')!r}: {fn}(...[{win}]); use $__rate_interval (or at least 2m)"
                )
    if errs:
        raise Fail("\n".join(errs))


# -- cluster ----------------------------------------------------------------------


def _prometheus(c: Ctx) -> str:
    d = c.system().get("deploy") or {}
    prom = str(d.get("prometheus", "")).rstrip("/")
    if not prom:
        if smoke_mode():
            c.skipped_cluster = True
            raise Skip(
                "cluster tier skipped under OL_SMOKE=1: system.toml has no [deploy].prometheus"
            )
        raise Fail("system.toml has no [deploy].prometheus (http://127.0.0.1:30090)")
    if d.get("kube_context"):
        c.need_cluster(str(d["kube_context"]))
    return prom


def test_queries_run_in_prometheus(c: Ctx) -> None:
    # WHY: the deployed Prometheus evaluates every panel query without error
    #      (an unknown function or a many-to-many match fails only at run time).
    # KIND: conformance
    import urllib.parse

    prom = _prometheus(c)
    errs = []
    for rel, p, t in _queries(c):
        q = urllib.parse.quote(substitute(t["expr"]), safe="")
        status, _, body = c.http("GET", f"{prom}/api/v1/query?query={q}", timeout=15)
        if status != 200:
            errs.append(
                f"{rel}: panel {p.get('title')!r}: HTTP {status}: {body[:200]!r}"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_dashboards_provisioned_for_grafana(c: Ctx) -> None:
    # WHY: Grafana in kube-prometheus-stack loads dashboards from ConfigMaps
    #      labelled grafana_dashboard=1; a dashboard that is only in git is
    #      not on the screen during a load run or a drill.
    # KIND: conformance
    _prometheus(c)
    ctx = str((c.system().get("deploy") or {}).get("kube_context", ""))
    cms = c.kubectl_json(ctx, ["get", "configmap", "-A", "-l", "grafana_dashboard=1"])
    blob = "\n".join(
        v for cm in cms.get("items", []) for v in (cm.get("data") or {}).values()
    )
    missing = [
        d.get("uid") for d in _dashboards(c).values() if f'"{d.get("uid")}"' not in blob
    ]
    if missing:
        raise Fail(
            f"no ConfigMap labelled grafana_dashboard=1 holds the dashboard(s) {missing}"
        )


if __name__ == "__main__":
    raise SystemExit(run(globals()))
