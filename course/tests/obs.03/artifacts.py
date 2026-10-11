"""obs.03 artifact check: SLOs and multi-window burn-rate alerts.

Run by `ol check obs.03` in your repo. Three tiers:

  static   deploy/observability/slo.yaml against contracts/otel/slo.schema.json;
           rules/slo-prod.yaml and rules/slo-drill.yaml are PrometheusRule
           objects the kube-prometheus-stack selects (label release:
           observability), define the six alerts with their severities and
           their profile's windows, and pass `promtool check rules`
  behavior `promtool test rules` replays course synthetic traffic through your
           rules, once per profile: a fast burn pages and then resets, a slow
           burn opens a ticket without paging, and noise plus a two-minute spike
           stays quiet. The traffic is generated from YOUR slo.yaml (thresholds
           and objectives), so the rules must agree with the file they came from
  cluster  [deploy].prometheus has loaded all six alerts (the rule selector
           matched)

promtool is taken from your PATH, else from the pinned prom/prometheus image
(Docker). The cluster tier fails when [deploy].prometheus is unset or down;
with OL_SMOKE=1 it is skipped with the reason instead.

The synthetic series carry these labels, so your selectors may use any of
them: job="<system>-gateway" (and a copy of every series as
job="<system>-engine"), namespace="<system>", tl_engine_role="gateway"
("unified" on the engine copy), gen_ai_operation_name="chat",
gen_ai_request_model="smol-135m"; HTTP series add http_request_method="POST",
http_route="/v1/chat/completions", http_response_status_code ("200" or
"503"); tl_gateway_requests_total carries route, code, and tenant="acme".
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _promtool  # noqa: E402
from _lib.practice import Ctx, Fail, Skip, run, smoke_mode  # noqa: E402

SLO = "deploy/observability/slo.yaml"
RULES = {
    "prod": "deploy/observability/rules/slo-prod.yaml",
    "drill": "deploy/observability/rules/slo-drill.yaml",
}
ALERTS = {
    "TTFTBudgetBurnFast": ("ttft", "fast"),
    "TTFTBudgetBurnSlow": ("ttft", "slow"),
    "TPOTBudgetBurnFast": ("tpot", "fast"),
    "TPOTBudgetBurnSlow": ("tpot", "slow"),
    "AvailabilityBudgetBurnFast": ("availability", "fast"),
    "AvailabilityBudgetBurnSlow": ("availability", "slow"),
}
PREFIX = {"ttft": "TTFT", "tpot": "TPOT", "availability": "Availability"}
# Windows per profile (DESIGN 2.11; otel/slo.schema.json).
WINDOWS = {
    "prod": {"fast": ("1h", "5m"), "slow": ("6h", "30m")},
    "drill": {"fast": ("5m", "25s"), "slow": ("30m", "2m30s")},
}
# A rule group evaluated less often than this cannot follow its short window.
MAX_INTERVAL_S = {"prod": 60, "drill": 10}
# The synthetic traffic: one sample (and one rule evaluation) per step; drill
# timelines are the prod ones divided by 12, like the windows.
STEP_S = {"prod": 15, "drill": 5}
SCALE = {"prod": 1, "drill": 12}
REQS = 1000  # requests per step, so a bad fraction has 0.1% resolution
# Course default targets (DESIGN 2.11): stricter is allowed, looser is not.
DEFAULT_MS = {"ttft": 500, "tpot": 60}
LATENCY_METRIC = {
    "ttft": "gen_ai_server_time_to_first_token_seconds",
    "tpot": "gen_ai_server_time_per_output_token_seconds",
}
HTTP = "http_server_request_duration_seconds"
# The kube-prometheus-stack rule selector (contracts/helm/observability.md).
SELECTOR = ("release", "observability")


# -- loading ----------------------------------------------------------------------


def _slo(c: Ctx) -> dict:
    if "slo" not in c.cache:
        docs = c.yaml_file(SLO)
        if len(docs) != 1 or not isinstance(docs[0], dict):
            raise Fail(f"{SLO} must hold one YAML mapping")
        c.cache["slo"] = docs[0]
    return c.cache["slo"]


def _metrics(c: Ctx) -> dict:
    if "metrics" not in c.cache:
        doc = c.yaml_file("contracts/otel/metrics.yaml")[0]
        c.cache["metrics"] = {i["prometheus"]: i for i in doc["instruments"]}
    return c.cache["metrics"]


def _buckets(c: Ctx, slo: str) -> list[float]:
    return [float(b) for b in _metrics(c)[LATENCY_METRIC[slo]]["buckets"]]


def _rule_objects(c: Ctx, profile: str) -> list[dict]:
    key = f"rules:{profile}"
    if key not in c.cache:
        docs = [d for d in c.yaml_file(RULES[profile]) if isinstance(d, dict)]
        if not docs:
            raise Fail(f"{RULES[profile]} holds no YAML object")
        c.cache[key] = docs
    return c.cache[key]


def _groups(c: Ctx, profile: str) -> list[dict]:
    out = []
    for d in _rule_objects(c, profile):
        out += list((d.get("spec") or {}).get("groups") or [])
    if not out:
        raise Fail(f"{RULES[profile]}: spec.groups is empty")
    return out


def _alert_rules(c: Ctx, profile: str) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {}
    for g in _groups(c, profile):
        for r in g.get("rules") or []:
            if "alert" in r:
                found.setdefault(str(r["alert"]), []).append(r)
    return found


def _budget(c: Ctx, slo: str) -> float:
    return round(1 - float(_slo(c)["slos"][slo]["objective"]), 9)


def _seconds(d: str) -> float:
    total = 0.0
    for n, u in re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h|d|w|y)", str(d)):
        total += (
            float(n)
            * {
                "ms": 0.001,
                "s": 1,
                "m": 60,
                "h": 3600,
                "d": 86400,
                "w": 604800,
                "y": 31536000,
            }[u]
        )
    return total


# -- static -----------------------------------------------------------------------


def test_slo_file_matches_the_schema(c: Ctx) -> None:
    # WHY: contracts/otel/slo.schema.json fixes the three SLOs, the six alert
    #      names, and each profile's windows. ol drill, MS-prod, and the
    #      dashboards (obs.04) read the file by those names.
    # KIND: conformance
    # CHAPTER: obs.03 section 4, The interface
    try:
        from olcourse import schema
    except ImportError:
        raise Fail(
            "the course harness is not importable: run this check through `ol check obs.03`"
        ) from None
    sch = json.loads(c.require_file("contracts/otel/slo.schema.json").read_text())
    errs = schema.validate(_slo(c), sch, root=sch)
    if errs:
        raise Fail(
            f"{SLO} does not match contracts/otel/slo.schema.json:\n"
            + "\n".join(errs[:12])
        )
    if _slo(c).get("profile") != "prod":
        raise Fail(
            f"{SLO} must declare profile: prod; the drill profile is rendered from it (rules/slo-drill.yaml)"
        )


def test_latency_thresholds_are_bucket_bounds(c: Ctx) -> None:
    # WHY: a latency SLI "fraction of requests at most T" is exact only when T
    #      is a bucket bound of the histogram (otel/metrics.yaml): the SLI reads
    #      the cumulative count of bucket le="T". 60 ms is not a TPOT bound, so
    #      the 60 ms default becomes 50 ms (stricter) or 75 ms (not allowed).
    # KIND: boundary
    # CHAPTER: obs.03 section 5, Pitfall 3
    errs = []
    for slo in ("ttft", "tpot"):
        t = float(_slo(c)["slos"][slo]["threshold_ms"]) / 1000
        b = _buckets(c, slo)
        if not any(abs(t - x) < 1e-12 for x in b):
            errs.append(
                f"slos.{slo}.threshold_ms = {t * 1000:g}: not a bucket bound of {LATENCY_METRIC[slo]} ({', '.join(f'{x * 1000:g}' for x in b)} ms)"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_targets_are_no_looser_than_the_course_defaults(c: Ctx) -> None:
    # WHY: DESIGN 2.11: you may promise more than the course defaults (TTFT
    #      p95 500 ms, TPOT 60 ms, 99.5% available), never less. The objective
    #      floors are in the schema; the latency thresholds are checked here.
    # KIND: boundary
    errs = []
    for slo, ms in DEFAULT_MS.items():
        got = float(_slo(c)["slos"][slo]["threshold_ms"])
        if got > ms:
            errs.append(
                f"slos.{slo}.threshold_ms = {got:g} is looser than the course default {ms} ms"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_rule_files_are_selected_prometheus_rules(c: Ctx) -> None:
    # WHY: kube-prometheus-stack loads only PrometheusRule objects labelled
    #      release: observability (contracts/helm/observability.md). A rule
    #      file without the label is valid YAML that Prometheus never sees:
    #      the alerts exist in git and nowhere else.
    # KIND: conformance
    # CHAPTER: obs.03 section 5, Pitfall 1
    errs = []
    for profile, rel in RULES.items():
        for d in _rule_objects(c, profile):
            where = f"{rel} ({d.get('kind')} {((d.get('metadata') or {}).get('name'))})"
            if (
                d.get("apiVersion") != "monitoring.coreos.com/v1"
                or d.get("kind") != "PrometheusRule"
            ):
                errs.append(
                    f"{where}: want apiVersion monitoring.coreos.com/v1, kind PrometheusRule"
                )
            labels = (d.get("metadata") or {}).get("labels") or {}
            if labels.get(SELECTOR[0]) != SELECTOR[1]:
                errs.append(
                    f"{where}: metadata.labels.{SELECTOR[0]} is {labels.get(SELECTOR[0])!r}; want {SELECTOR[1]!r}"
                )
    if errs:
        raise Fail("\n".join(errs))


def test_rules_define_the_six_alerts(c: Ctx) -> None:
    # WHY: ol drill [detect] and MS-prod look alerts up by the six names of
    #      the schema, and the page/ticket split is slo.yaml's severity. Each
    #      name exactly once per profile: a duplicate fires twice per incident.
    # KIND: unit
    # CHAPTER: obs.03 section 4, The interface
    want_sev = {k: v["severity"] for k, v in (_slo(c).get("alerts") or {}).items()}
    errs = []
    for profile, rel in RULES.items():
        found = _alert_rules(c, profile)
        for name in ALERTS:
            rules = found.get(name, [])
            if len(rules) != 1:
                errs.append(
                    f"{rel}: alert {name} is defined {len(rules)} times; want once"
                )
                continue
            sev = (rules[0].get("labels") or {}).get("severity")
            if sev != want_sev.get(name):
                errs.append(
                    f"{rel}: {name} has labels.severity {sev!r}; slo.yaml says {want_sev.get(name)!r}"
                )
        extra = sorted(set(found) - set(ALERTS))
        if extra:
            errs.append(
                f"{rel}: alerts {extra} are not in the schema (put other alerts in another file)"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_rules_use_the_profile_windows(c: Ctx) -> None:
    # WHY: each alert compares its long AND its short window. The drill
    #      profile compresses both 12x so `ol drill` can see a page within
    #      minutes; ol drill refuses to start when they are missing. A window
    #      may be spelled either way (2m30s or 150s).
    # KIND: unit
    # CHAPTER: obs.03 section 5, Pitfall 4
    errs = []
    for profile, rel in RULES.items():
        text = c.require_file(rel).read_text()
        used = {_seconds(w) for w in re.findall(r"\[([0-9smhdwy]+)\]", text)}
        for kind, (long_, short) in WINDOWS[profile].items():
            for w in (long_, short):
                if _seconds(w) not in used:
                    errs.append(
                        f"{rel}: no range selector [{w}] ({profile} {kind} window {long_}/{short})"
                    )
    if errs:
        raise Fail("\n".join(errs))


def test_rule_groups_evaluate_often_enough(c: Ctx) -> None:
    # WHY: a group evaluated every 5 minutes cannot react to a 25 s window,
    #      and a drill's page arrives late or never.
    # KIND: boundary
    errs = []
    for profile, rel in RULES.items():
        for g in _groups(c, profile):
            iv = _seconds(g.get("interval", "30s"))
            if iv > MAX_INTERVAL_S[profile]:
                errs.append(
                    f"{rel}: group {g.get('name')} interval {g.get('interval')}; at most {MAX_INTERVAL_S[profile]}s for the {profile} profile"
                )
    if errs:
        raise Fail("\n".join(errs))


def _rules_file(c: Ctx, profile: str) -> str:
    import yaml

    return yaml.safe_dump({"groups": _groups(c, profile)}, sort_keys=False)


def test_promtool_accepts_the_rules(c: Ctx) -> None:
    # WHY: PromQL syntax errors and duplicate rule names surface here, not as
    #      an operator log line after the rules were applied.
    # KIND: conformance
    errs = []
    for profile in RULES:
        r = _promtool.run(
            c,
            c.path(f".ol/check/obs.03/check-{profile}"),
            ["check", "rules", "rules.yaml"],
            {"rules.yaml": _rules_file(c, profile)},
            timeout=300,
        )
        if r.returncode != 0:
            errs.append(f"{RULES[profile]}:\n{(r.stdout + r.stderr).strip()[-1500:]}")
    if errs:
        raise Fail("\n".join(errs))


# -- behavior: synthetic traffic through promtool -----------------------------------------


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.4f}".rstrip("0")


def _counter(incs: list[float]) -> str:
    """promtool series notation for a counter that grows by incs[i] at step i+1."""
    out = ["0"]
    v = 0.0
    i = 0
    while i < len(incs):
        j = i
        while j + 1 < len(incs) and incs[j + 1] == incs[i]:
            j += 1
        n = j - i + 1
        out.append(f"{_num(v + incs[i])}+{_num(incs[i])}x{n - 1}")
        v += incs[i] * n
        i = j + 1
    return " ".join(out)


def _steps(phases: list[tuple[float, float]], profile: str) -> list[float]:
    """Per-step bad fraction from (prod minutes, fraction) phases."""
    per_min = 60 / SCALE[profile] / STEP_S[profile]
    out: list[float] = []
    for minutes, f in phases:
        out += [f] * round(minutes * per_min)
    return out


def _series(c: Ctx, bad: dict[str, list[float]]) -> list[dict]:
    """Input series for promtool from per-SLO per-step bad fractions."""
    name = c.system_name()
    base = {
        "namespace": name,
        "gen_ai_operation_name": "chat",
        "gen_ai_request_model": "smol-135m",
    }
    roles = [("gateway", f"{name}-gateway"), ("unified", f"{name}-engine")]
    out: list[dict] = []

    def sel(metric: str, labels: dict) -> str:
        inner = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{metric}{{{inner}}}"

    for slo in ("ttft", "tpot"):
        bs = _buckets(c, slo)
        t = float(_slo(c)["slos"][slo]["threshold_ms"]) / 1000
        i = min(range(len(bs)), key=lambda k: abs(bs[k] - t))
        nbad = [round(REQS * f) for f in bad[slo]]
        for role, job in roles:
            lab = dict(base, tl_engine_role=role, job=job)
            for k, b in enumerate(bs):
                # good requests land in bucket i (le >= bs[i]), bad ones in bucket i+1
                incs = [
                    0.0 if k < i else (REQS - nb if k == i else REQS) for nb in nbad
                ]
                out.append(
                    {
                        "series": sel(
                            LATENCY_METRIC[slo] + "_bucket", dict(lab, le=repr(b))
                        ),
                        "values": _counter(incs),
                    }
                )
            out.append(
                {
                    "series": sel(
                        LATENCY_METRIC[slo] + "_bucket", dict(lab, le="+Inf")
                    ),
                    "values": _counter([REQS] * len(nbad)),
                }
            )
            out.append(
                {
                    "series": sel(LATENCY_METRIC[slo] + "_count", lab),
                    "values": _counter([REQS] * len(nbad)),
                }
            )
            sums = [
                (REQS - nb) * bs[i] * 0.9
                + nb * (bs[i + 1] if i + 1 < len(bs) else bs[i] * 2) * 0.9
                for nb in nbad
            ]
            out.append(
                {
                    "series": sel(LATENCY_METRIC[slo] + "_sum", lab),
                    "values": _counter([round(s, 3) for s in sums]),
                }
            )
    nbad = [round(REQS * f) for f in bad["availability"]]
    for role, job in roles:
        http = dict(
            namespace=name,
            job=job,
            http_request_method="POST",
            http_route="/v1/chat/completions",
        )
        for code, incs in (
            ("200", [REQS - n for n in nbad]),
            ("503", list(map(float, nbad))),
        ):
            lab = dict(http, http_response_status_code=code)
            out.append({"series": sel(HTTP + "_count", lab), "values": _counter(incs)})
            out.append(
                {
                    "series": sel(HTTP + "_bucket", dict(lab, le="+Inf")),
                    "values": _counter(incs),
                }
            )
            if role == "gateway":
                gw = dict(
                    namespace=name,
                    job=job,
                    route="/v1/chat/completions",
                    code=code,
                    tenant="acme",
                )
                out.append(
                    {
                        "series": sel("tl_gateway_requests_total", gw),
                        "values": _counter(incs),
                    }
                )
    return out


def _at(minutes: float, profile: str) -> str:
    return f"{round(minutes * 60 / SCALE[profile])}s"


def _firing(names: str, profile: str, minutes: float, want: bool) -> dict:
    expr = f'clamp_max(count(ALERTS{{alertstate="firing",alertname=~"{names}"}}), 1)'
    return {
        "expr": expr,
        "eval_time": _at(minutes, profile),
        "exp_samples": [{"labels": "{}", "value": 1}] if want else [],
    }


def _scenario(c: Ctx, profile: str, kind: str) -> list[dict]:
    budget = {s: _budget(c, s) for s in ("ttft", "tpot", "availability")}
    fast = float(_slo(c)["windows"]["fast"]["burn_rate"])
    slow = float(_slo(c)["windows"]["slow"]["burn_rate"])
    if kind == "noise":
        bad = {
            s: _steps(
                [
                    (120, 0.5 * budget[s]),
                    (2, min(1.0, 100 * budget[s])),
                    (60, 0.5 * budget[s]),
                ],
                profile,
            )
            for s in budget
        }
        tests = [
            _firing(".*", profile, m, False) for m in (60, 121, 122, 125, 135, 150, 180)
        ]
        name = f"{profile}: every SLO at 0.5x its budget with a 2-minute spike at 100x"
        return [
            {
                "name": name,
                "interval": f"{STEP_S[profile]}s",
                "input_series": _series(c, bad),
                "promql_expr_test": tests,
            }
        ]
    groups = []
    for slo in ("ttft", "tpot", "availability"):
        p = PREFIX[slo]
        others = "|".join(PREFIX[o] + ".*" for o in budget if o != slo)
        if kind == "fast":
            # 30 min healthy, 75 min at twice the fast threshold's bad ratio, 30 min healthy.
            f = min(1.0, 2 * fast * budget[slo])
            bad = {
                s: _steps([(30, 0), (75, f if s == slo else 0), (30, 0)], profile)
                for s in budget
            }
            tests = [
                _firing(".*", profile, 30, False),
                _firing(f"{p}BudgetBurnFast", profile, 100, True),
                _firing(others, profile, 100, False),
                _firing(f"{p}BudgetBurnFast", profile, 113, False),
            ]
            name = f"{profile}: {slo} burns at {f / budget[slo]:g}x its budget for 75 min, then recovers"
        else:
            f = 1.5 * slow * budget[slo]
            bad = {s: _steps([(150, f if s == slo else 0)], profile) for s in budget}
            tests = [
                _firing(f"{p}BudgetBurnSlow", profile, 120, True),
                _firing(f"{p}BudgetBurnFast", profile, 120, False),
                _firing(others, profile, 120, False),
            ]
            name = (
                f"{profile}: {slo} burns at {f / budget[slo]:g}x its budget for 2.5 h"
            )
        groups.append(
            {
                "name": name,
                "interval": f"{STEP_S[profile]}s",
                "input_series": _series(c, bad),
                "promql_expr_test": tests,
            }
        )
    return groups


def _behavior(c: Ctx, kind: str) -> None:
    import yaml

    errs = []
    for profile in RULES:
        test = {
            "rule_files": ["rules.yaml"],
            "evaluation_interval": f"{STEP_S[profile]}s",
            "tests": _scenario(c, profile, kind),
        }
        r = _promtool.run(
            c,
            c.path(f".ol/check/obs.03/{kind}-{profile}"),
            ["test", "rules", "test.yaml"],
            {
                "rules.yaml": _rules_file(c, profile),
                "test.yaml": yaml.safe_dump(test, sort_keys=False),
            },
            timeout=300,
        )
        if r.returncode != 0:
            out = (r.stdout + r.stderr).strip()
            errs.append(
                f"{RULES[profile]}, {kind} scenario (inputs kept in .ol/check/obs.03/{kind}-{profile}/):\n{out[-2500:]}"
            )
    if errs:
        raise Fail("\n".join(errs))


def test_fast_burn_pages_then_resets(c: Ctx) -> None:
    # WHY: the page you want: when one SLO burns its budget at twice the fast
    #      threshold, its *BudgetBurnFast alert fires once the long window has
    #      filled, the other SLOs' alerts stay quiet, and the page clears a few
    #      minutes after the burn stops, because the short window recovers
    #      first. A long-window-only alert would keep paging for an hour.
    # KIND: conformance
    # CHAPTER: obs.03 section 3, Worked example; section 5, Pitfalls 2 and 5
    _behavior(c, "fast")


def test_slow_burn_tickets_without_paging(c: Ctx) -> None:
    # WHY: a burn between the slow and the fast threshold (9x) exhausts the
    #      30-day budget in about 3 days: worth a ticket, not a 3 a.m. page.
    # KIND: conformance
    # CHAPTER: obs.03 section 5, Pitfall 6
    _behavior(c, "slow")


def test_noise_and_short_spikes_stay_quiet(c: Ctx) -> None:
    # WHY: burning at half the budget is the budget working as intended, and a
    #      2-minute total outage is caught by the short windows but vetoed by
    #      the long ones. Alerts that fire on either train people to ignore
    #      pages.
    # KIND: conformance
    # CHAPTER: obs.03 section 5, Pitfall 2
    _behavior(c, "noise")


# -- cluster ----------------------------------------------------------------------


def test_rules_loaded_in_prometheus(c: Ctx) -> None:
    # WHY: the proof that the selector matched: Prometheus at
    #      [deploy].prometheus lists all six alerts among its loaded rules.
    # KIND: conformance
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
    status, _, body = c.http("GET", prom + "/api/v1/rules?type=alert", timeout=10)
    if status != 200:
        raise Fail(f"GET {prom}/api/v1/rules answered {status}")
    names = {
        r.get("name")
        for g in json.loads(body)["data"]["groups"]
        for r in g.get("rules", [])
    }
    missing = sorted(set(ALERTS) - names)
    if missing:
        raise Fail(
            f"Prometheus has not loaded {missing}: is the PrometheusRule applied with label release: observability?"
        )


if __name__ == "__main__":
    raise SystemExit(run(globals()))
