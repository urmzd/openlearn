#!/usr/bin/env python3
"""Render deploy/observability/slo.yaml into PrometheusRule files (obs.03 reference).

    python3 deploy/observability/render_slo_rules.py [--system forge]

writes rules/slo-prod.yaml and rules/slo-drill.yaml next to this file. Both
hold the same six alerts (names fixed by contracts/otel/slo.schema.json);
only the windows differ: prod 1h/5m and 6h/30m, drill the same ratios with
windows compressed 12x (5m/25s and 30m/2m30s) so a drill can page within
minutes. Both may be installed at once: every recorded series and alert
carries slo_profile, so the two never collide, and Alertmanager can route
slo_profile="drill" pages to the drill channel only.

Each SLO becomes one error-ratio recording rule per window,
slo:sli_error:ratio_rate<window>{slo="<name>",slo_profile="<profile>"}, and each alert fires when the
burn rate (error ratio / (1 - objective)) exceeds its factor over BOTH its
long and its short window:

    error_ratio[long] > factor * (1 - objective)  and  error_ratio[short] > factor * (1 - objective)

Stdlib only: slo.yaml is read with a small parser for exactly its shape.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILES = {
    "prod": {
        "fast": ("1h", "5m"),
        "slow": ("6h", "30m"),
        "interval": "30s",
        "for": {"fast": "2m", "slow": "15m"},
    },
    # 2m30s is written 150s: the same duration, and the spelling `ol drill`'s
    # rule gate looks for.
    "drill": {
        "fast": ("5m", "25s"),
        "slow": ("30m", "150s"),
        "interval": "5s",
        "for": {"fast": "10s", "slow": "75s"},
    },
}
# The SLIs, measured where the user is: the gateway. Prometheus names the
# scrape job after the Service its ServiceMonitor selects, so every series the
# gateway serves carries job="<system>-gateway".
TTFT = "gen_ai_server_time_to_first_token_seconds"
TPOT = "gen_ai_server_time_per_output_token_seconds"
HTTP = "http_server_request_duration_seconds"


def parse_slo(text: str) -> dict:
    """slo.yaml's shape: top-level scalars and two-level flow mappings."""
    doc: dict = {}
    section = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        m = re.match(r"^(\s*)([A-Za-z_]+):\s*(.*)$", line)
        if not m:
            raise SystemExit(f"slo.yaml: cannot read line {raw!r}")
        indent, key, val = m.groups()
        if not indent:
            section = key
            doc[key] = val if val else {}
            continue
        if val.startswith("{"):
            body = {}
            for part in val.strip("{}").split(","):
                k, v = (x.strip() for x in part.split(":", 1))
                body[k] = float(v) if re.fullmatch(r"[0-9.]+", v) else v
            doc[section][key] = body
    return doc


def fmt_le(seconds: float) -> str:
    """The `le` label value Prometheus stores for a bucket bound."""
    return repr(float(seconds))


def error_ratio(slo: str, spec: dict, window: str, system: str) -> str:
    sel = f'job="{system}-gateway"'
    if slo == "availability":
        return (
            f'sum(rate({HTTP}_count{{{sel},http_response_status_code=~"5.."}}[{window}]))\n'
            f"/\nsum(rate({HTTP}_count{{{sel}}}[{window}]))"
        )
    metric = TTFT if slo == "ttft" else TPOT
    le = fmt_le(spec["threshold_ms"] / 1000)
    return (
        f'1 - (\n  sum(rate({metric}_bucket{{{sel},le="{le}"}}[{window}]))\n'
        f"  /\n  sum(rate({metric}_count{{{sel}}}[{window}]))\n)"
    )


def indent(block: str, n: int) -> str:
    return "\n".join(" " * n + line for line in block.splitlines())


def render(doc: dict, profile: str, system: str) -> str:
    p = PROFILES[profile]
    windows = sorted(
        {w for kind in ("fast", "slow") for w in p[kind]}, key=lambda w: (len(w), w)
    )
    out = [
        "# Rendered by render_slo_rules.py from slo.yaml: do not edit by hand.",
        f"# Profile {profile}: fast {p['fast'][0]}/{p['fast'][1]}, slow {p['slow'][0]}/{p['slow'][1]}.",
        "apiVersion: monitoring.coreos.com/v1",
        "kind: PrometheusRule",
        "metadata:",
        f"  name: {system}-slo-{profile}",
        "  namespace: observability",
        "  labels:",
        "    release: observability          # the kube-prometheus-stack rule selector (contracts/helm/observability.md)",
        f"    app.kubernetes.io/part-of: {system}",
        "spec:",
        "  groups:",
        f"    - name: {system}-slo-{profile}",
        f"      interval: {p['interval']}",
        "      rules:",
    ]
    for slo in ("ttft", "tpot", "availability"):
        spec = doc["slos"][slo]
        for w in windows:
            out += [
                f"        - record: slo:sli_error:ratio_rate{w}",
                "          expr: |",
                indent(error_ratio(slo, spec, w, system), 12),
                "          labels:",
                f"            slo: {slo}",
                f"            slo_profile: {profile}",
            ]
    factors = {k: float(v["burn_rate"]) for k, v in doc["windows"].items()}
    for name, a in doc["alerts"].items():
        slo, kind, sev = a["slo"], a["window"], a["severity"]
        budget = round(1 - float(doc["slos"][slo]["objective"]), 6)
        long_, short = p[kind]
        thr = f"({factors[kind]:g} * {budget:g})"
        out += [
            f"        - alert: {name}",
            "          expr: |",
            f'            slo:sli_error:ratio_rate{long_}{{slo="{slo}",slo_profile="{profile}"}} > {thr}',
            "            and",
            f'            slo:sli_error:ratio_rate{short}{{slo="{slo}",slo_profile="{profile}"}} > {thr}',
            f"          for: {p['for'][kind]}",
            "          labels:",
            f"            severity: {sev}",
            f"            slo: {slo}",
            f"            slo_profile: {profile}",
            "          annotations:",
            f"            summary: {slo} error budget burning at more than {factors[kind]:g}x over {long_} and {short}",
            f"            runbook: docs/runbooks/{name}.md",
        ]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--system", default="forge")
    args = ap.parse_args()
    doc = parse_slo((HERE / "slo.yaml").read_text())
    (HERE / "rules").mkdir(exist_ok=True)
    for profile in PROFILES:
        (HERE / "rules" / f"slo-{profile}.yaml").write_text(
            render(doc, profile, args.system)
        )
        print(f"wrote rules/slo-{profile}.yaml")


if __name__ == "__main__":
    main()
