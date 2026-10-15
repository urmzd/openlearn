<!-- ol:module obs.04 -->
# Dashboards as code for the serving path

## Overview

| | |
|---|---|
| **Module** | `obs.04` · practice · ops · Pass 7 · 3 to 4 h |
| **You build** | `deploy/observability/dashboards/*.json`: the serving dashboard (SLO status, TTFT and TPOT heatmaps, KV cache usage, per-tenant usage), as Grafana JSON in git, loaded by the stack's Grafana from a ConfigMap |
| **Contract** | [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml) (the only metric names a panel may read), your `obs.03` rules (recording rules and alerts), [`helm/observability.md`](../../../../course/contracts/helm/observability.md) (Grafana on NodePort 30300) |
| **Tests** | `course/tests/obs.04/` (`check` runs `artifacts.py`; section 4) |
| **Needs** | `obs.02` (the series), `obs.03` (the burn-rate recording rules and the six alerts) |
| **Used by** | `ops.01` and every later drill start from this dashboard; `ops.09` (noisy neighbor) from its tenant row |
| **Milestone** | MS-prod (Grafana during a load run) |
| **Optional depth** | [Grafana dashboard JSON model](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/view-dashboard-json-model/) (free), [Heatmaps of Prometheus histograms](https://grafana.com/docs/grafana/latest/panels-visualizations/visualizations/heatmap/) (free), *The RED method* and *The USE method* (free articles) |

## Key Takeaways

- A dashboard is **code** when a fresh Grafana loads the file from git unchanged: a stable `uid`, no instance `id`, and a datasource that is a variable, not the uid of the Grafana you exported from.
- Every query names only **contract metrics**, your recording rules, or `ALERTS`, and parses; the check proves both with `promtool`.
- A **heatmap** of a Prometheus histogram needs bucket rates summed `by (le)` and the target format `heatmap`; the p95 line alone hides a second mode.
- One row per question: are we within SLO (status), how are latencies distributed (heatmaps), is the engine full (KV blocks), who is using it (per tenant).
- `rate()` windows follow the scrape interval: `$__rate_interval`, never a fixed `[1m]` at a 30 s scrape.

## How to work this chapter

```bash
ol start obs.04                       # records the start; there are no stubs
ol tests obs.04                       # read the test catalog first
# build the dashboard in Grafana (http://127.0.0.1:30300), export JSON, clean it (section 5), then:
OL_SMOKE=1 ol check obs.04            # the static tier
kubectl -n observability create configmap forge-dashboards --from-file=deploy/observability/dashboards \
  --dry-run=client -o yaml | kubectl label --local -f - grafana_dashboard=1 -o yaml | kubectl apply -f -
ol check obs.04                       # adds: queries run in Prometheus, the ConfigMap is there
```

---

## 1. Why now

`ops.01` will page you with `TTFTBudgetBurnFast`. The page says the budget is burning; it does not say why. The first five minutes of every incident are the same questions: which SLO, since when, is latency uniformly worse or is there a new slow mode, is the engine out of KV blocks, is one tenant flooding the gateway. Typing PromQL during an incident is slow and error-prone, and a dashboard someone clicked together in Grafana last month is gone the next time the cluster is recreated (`dep.02` recreates it whenever port mappings change). Dashboards kept as JSON in your repo are reviewed, versioned, and redeployed like every other artifact.

## 2. Principles

### 2.1 The dashboard JSON model

| Field | Meaning | Rule |
|---|---|---|
| `uid` | the dashboard's identity in URLs and provisioning | stable, set by you, at most 40 of `[A-Za-z0-9_-]` |
| `id` | the database row id in one Grafana | `null` in git: another Grafana assigns its own |
| `panels[]` | each panel: `type`, `title`, `gridPos`, `datasource`, `targets[]` | `type: row` panels group the others |
| `targets[].expr` | the PromQL a panel runs | contract metrics, recording rules, `ALERTS` only |
| `templating.list[]` | variables, e.g. `datasource` of type `datasource`, query `prometheus` | panels refer to `${datasource}` |

### 2.2 Queries per panel kind

| Panel | Question | Query shape |
|---|---|---|
| stat | is this SLO burning now? | the burn rate, `slo:sli_error:ratio_rate1h{slo="ttft",slo_profile="prod"} / 0.01`, thresholds at 6 and 14.4 |
| table | which alerts fire? | `ALERTS{alertstate="firing",alertname=~".+BudgetBurn(Fast\|Slow)"}`, instant, format table |
| heatmap | how are latencies distributed over time? | `sum by (le) (rate(<histogram>_bucket{job="<system>-gateway"}[$__rate_interval]))`, format `heatmap` |
| time series | KV blocks by state, queue depth | `sum by (state) (tl_engine_kv_blocks)` |
| time series | who sends what | `sum by (tenant) (rate(tl_gateway_requests_total[$__rate_interval]))` |

| Symbol | Meaning | Type |
|---|---|---|
| $C_i(t)$ | cumulative count of bucket $b_i$ (`obs.02`) | counter |
| $r_i = \text{rate}(C_i)$ | requests per second at most $b_i$ | per second |
| $r_i - r_{i-1}$ | requests per second in $(b_{i-1}, b_i]$: one heatmap cell | per second |

A heatmap cell is the **difference** of neighbouring cumulative rates. Grafana computes it only when the target's format is `heatmap`; with the default `time_series` format it stacks the cumulative rates, so every cell counts all faster requests again.

### 2.3 Rates and scrape intervals

`rate(x[w])` uses the samples inside $w$; with fewer than two it returns nothing. Grafana's `$__rate_interval` is at least four scrape intervals and grows with the zoom, so it is never too short. A fixed `[1m]` at a 30 s scrape is two samples on a good day and none after one late scrape: the panel flickers between a value and "No data".

### 2.4 Provisioning

The stack's Grafana runs a sidecar that loads every ConfigMap labelled `grafana_dashboard: "1"` as dashboards. The JSON in git is the source; the ConfigMap is generated from it (by `kubectl create configmap --from-file`, a Helm chart, or your Tiltfile); nothing is edited in the UI without being exported back to the file.

## 3. Worked example by hand

Two engine replicas, scraped every 10 s. Over the last minute the gateway's TTFT buckets grew by:

| `le` | $\Delta C_i$ over 60 s | $r_i$ (per s) | cell $(b_{i-1}, b_i]$ (per s) |
|---|---|---|---|
| 0.1 | 300 | 5.0 | 5.0 in (0.08, 0.1] and below |
| 0.25 | 540 | 9.0 | 4.0 in (0.1, 0.25] |
| 0.5 | 570 | 9.5 | 0.5 in (0.25, 0.5] |
| 0.75 | 570 | 9.5 | 0 |
| 1.0 | 594 | 9.9 | 0.4 in (0.75, 1.0] |
| `+Inf` | 600 | 10.0 | 0.1 above 1.0 |

The heatmap column for this minute has its mass in the two lowest rows and a separate small band near one second: about 5% of requests ($0.5 / 10$) are in a second mode, here the requests that queued behind a long prefill. The p95 line misses it entirely: the 95th percentile ($0.95 \times 10 = 9.5$ per s) lands at the 0.5 s bound. With the default format Grafana would draw 10.0, 9.9, 9.5, 9.5, 9.0, 5.0 stacked in the column, and the slow band would look like the busiest cell.

The SLO status stat next to it shows the 1 h TTFT burn: if the hour looked like this minute, $e = 0.5 / 10 = 0.05$ of requests over 500 ms, and with $\beta = 0.01$ the burn is $5$: still green, one step below the orange line at 6 (the slow alert's factor); red starts at 14.4 (the fast one's).

## 4. The interface

`deploy/observability/dashboards/serving.json` (the reference's rows):

| Row | Panels |
|---|---|
| SLOs | stat: TTFT, TPOT, and availability burn rate (1 h); table: firing `*BudgetBurn*` alerts |
| Latency | heatmap: TTFT; heatmap: TPOT; time series: TTFT and TPOT p95 |
| Engine capacity | KV cache blocks by state (stacked); queue depth and running sequences per pod |
| Tenants | requests per second by tenant; 5xx per second by tenant |

Panel and target `datasource`: `{"type": "prometheus", "uid": "${datasource}"}` with a `datasource` variable of type `datasource`, query `prometheus`, or the stack's provisioned uid `prometheus`. Before checking, the course replaces `$__rate_interval`, `$__interval`, and `$__range` with `5m` and every other variable with `.*`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_dashboards_are_portable_grafana_json` | unit | valid JSON, unique stable `uid`, `id` null, title, schemaVersion, panels | the file loads in any Grafana |
| `test_panels_use_a_portable_prometheus_datasource` | unit | every panel and target uses a datasource variable or uid `prometheus` | no "datasource not found" after a rebuild |
| `test_every_query_names_contract_metrics` | conformance | metric names are in `metrics.yaml`, your recording rules, or `ALERTS` | no empty panel that looks like "no traffic" |
| `test_every_query_parses` | conformance | `promtool` parses every query | syntax errors show up now, not mid-incident |
| `test_slo_status_panels_cover_the_three_slos` | unit | a status panel (stat, gauge, bar gauge, table, state timeline) per SLO | the first question of every page |
| `test_latency_heatmaps` | unit | TTFT and TPOT heatmaps: bucket rates `by (le)`, format `heatmap` | section 3's second mode |
| `test_kv_cache_usage_panel` | unit | a panel reads `tl_engine_kv_blocks` | capacity, leaks, preemption pressure |
| `test_per_tenant_usage_panel` | unit | `tl_gateway_requests_total` by `tenant` | `ops.09`, noisy neighbors |
| `test_rate_windows_survive_the_scrape_interval` | boundary | every `rate`/`irate`/`increase` window is `$__rate_interval` or at least 2 m | no flickering panels |
| `test_queries_run_in_prometheus` | conformance | on kind: each query runs in `[deploy].prometheus` without error | many-to-many and runtime errors |
| `test_dashboards_provisioned_for_grafana` | conformance | on kind: a ConfigMap labelled `grafana_dashboard=1` holds each `uid` | the dashboard is on screen during a drill |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. Exported with `"id": 17` and a generated `uid` | a second import overwrites or duplicates; links break on every re-export | `test_dashboards_are_portable_grafana_json` |
| 2. The exporting Grafana's datasource uid in every panel | "datasource not found" on a fresh cluster | `test_panels_use_a_portable_prometheus_datasource` |
| 3. A metric name from a blog post or a typo (`ttft_seconds_bucket`) | an empty panel that reads as zero traffic | `test_every_query_names_contract_metrics` |
| 4. A heatmap without `by (le)` or with the time series format | the busiest-looking cell is the slow tail | `test_latency_heatmaps` |
| 5. Usage summed over all tenants | one tenant's flood looks like organic growth | `test_per_tenant_usage_panel` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `obs.02` | every series a panel reads, under its job names |
| Back | `obs.03` | `slo:sli_error:*` recording rules and the six alerts in the status row |
| Forward | `ops.01` | the drill's diagnosis starts at the TTFT heatmap and the queue depth per pod |
| Forward | `ops.09` | the tenant row shows the noisy neighbor |
| Forward | `obs.05` | the control-plane dashboard (queues, DLQ) follows the same rules |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| hand-edited JSON | Grafonnet (Jsonnet), the Grafana Foundation SDK, Grafana's `grafanactl` | dashboards generated from code, shared panel libraries | [Grafonnet](https://grafana.github.io/grafonnet/) |
| a ConfigMap per folder | Grafana provisioning from Git, or the Grafana Operator | sync, folders, permissions as code | [Grafana Operator](https://grafana.github.io/grafana-operator/) |
| one serving dashboard | RED (rate, errors, duration) per service, USE (utilization, saturation, errors) per resource | a standard layout every team reads the same way | *The RED Method* (Wilkie), *The USE Method* (Gregg) |
| burn-rate stats | SLO dashboards from Sloth or Pyrra | budget remaining over the period, burn history | [Pyrra](https://github.com/pyrra-dev/pyrra) |
