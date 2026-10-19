# Dashboards as code for the serving path

## Overview

| | |
|---|---|
| **Module** | `obs.04` · practice · ops · Pass 7 · 3 to 4 h |
| **Solution** | [`deploy/observability/dashboards/serving.json`](../../../../course/solutions/entry/deploy/observability/dashboards/serving.json): the serving dashboard (`uid: forge-serving`; rows SLOs, Latency, Engine capacity, Tenants), as Grafana JSON in git, loaded by the stack's Grafana from a ConfigMap |
| **Contract** | [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml) (the only metric names a panel may read), the `obs.03` rules (recording rules and alerts), [`helm/observability.md`](../../../../course/contracts/helm/observability.md) (Grafana on NodePort 30300) |
| **Needs** | `obs.02` (the series), `obs.03` (the burn-rate recording rules and the six alerts) |
| **Used by** | `ops.01` and every later drill start from this dashboard; `ops.09` (noisy neighbor) from its tenant row; the production system shows it in Grafana during a load run |
| **Optional depth** | [Grafana dashboard JSON model](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/view-dashboard-json-model/) (free), [Heatmaps of Prometheus histograms](https://grafana.com/docs/grafana/latest/panels-visualizations/visualizations/heatmap/) (free), *The RED method* and *The USE method* (free articles) |

## Key Takeaways

- A dashboard is **code** when a fresh Grafana loads the file from git unchanged: a stable `uid`, no instance `id`, and a datasource that is a variable, not the uid of the Grafana you exported from.
- Every query names only **contract metrics**, the `obs.03` recording rules, or `ALERTS`, and parses (`promtool` can prove both).
- A **heatmap** of a Prometheus histogram needs bucket rates summed `by (le)` and the target format `heatmap`; the p95 line alone hides a second mode.
- One row per question: are we within SLO (status), how are latencies distributed (heatmaps), is the engine full (KV blocks), who is using it (per tenant).
- `rate()` windows follow the scrape interval: `$__rate_interval`, never a fixed `[1m]` at a 30 s scrape.

## How to read this chapter

Sections 1 to 3 build the dashboard's questions from first principles and work one heatmap column by hand. Section 4 fixes the rows and what a portable dashboard must satisfy. Section 5 walks through the solution's JSON panel by panel and explains why each choice is the right one. Try to predict each step before reading it.

A dashboard like this is usually built in Grafana (http://127.0.0.1:30300), exported as JSON, and cleaned (section 6). The solution is deployed like this:

```bash
kubectl -n observability create configmap forge-dashboards --from-file=deploy/observability/dashboards \
  --dry-run=client -o yaml | kubectl label --local -f - grafana_dashboard=1 -o yaml | kubectl apply -f -
```

---

## 1. Why now

`ops.01` pages the on-call engineer with `TTFTBudgetBurnFast`. The page says the budget is burning; it does not say why. The first five minutes of every incident are the same questions: which SLO, since when, is latency uniformly worse or is there a new slow mode, is the engine out of KV blocks, is one tenant flooding the gateway. Typing PromQL during an incident is slow and error-prone, and a dashboard someone clicked together in Grafana last month is gone the next time the cluster is recreated (`dep.02` recreates it whenever port mappings change). Dashboards kept as JSON in the repo are reviewed, versioned, and redeployed like every other artifact.

## 2. Principles

### 2.1 The dashboard JSON model

| Field | Meaning | Rule |
|---|---|---|
| `uid` | the dashboard's identity in URLs and provisioning | stable, chosen by the author, at most 40 of `[A-Za-z0-9_-]` |
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

The stack's Grafana runs a sidecar that loads every ConfigMap labelled `grafana_dashboard: "1"` as dashboards. The JSON in git is the source; the ConfigMap is generated from it (by `kubectl create configmap --from-file`, a Helm chart, or a Tiltfile); nothing is edited in the UI without being exported back to the file.

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

`deploy/observability/dashboards/serving.json` (the solution's rows):

| Row | Panels |
|---|---|
| SLOs | stat: TTFT, TPOT, and availability burn rate (1 h); table: firing `*BudgetBurn*` alerts |
| Latency | heatmap: TTFT; heatmap: TPOT; time series: TTFT and TPOT p95 |
| Engine capacity | KV cache blocks by state (stacked); queue depth and running sequences per pod |
| Tenants | requests per second by tenant; 5xx per second by tenant |

Panel and target `datasource`: `{"type": "prometheus", "uid": "${datasource}"}` with a `datasource` variable of type `datasource`, query `prometheus`, or the stack's provisioned uid `prometheus`. To check queries outside Grafana (with `promtool` or the Prometheus API), replace `$__rate_interval`, `$__interval`, and `$__range` with `5m` and every other variable with `.*`.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Portable Grafana JSON | unit | valid JSON, unique stable `uid`, `id` null, title, schemaVersion, panels | the file loads in any Grafana |
| A portable Prometheus datasource | unit | every panel and target uses a datasource variable or uid `prometheus` | no "datasource not found" after a rebuild |
| Every query names contract metrics | conformance | metric names are in `metrics.yaml`, the recording rules, or `ALERTS` | no empty panel that looks like "no traffic" |
| Every query parses | conformance | `promtool` parses every query | syntax errors show up now, not mid-incident |
| SLO status for all three SLOs | unit | a status panel (stat, gauge, bar gauge, table, state timeline) per SLO | the first question of every page |
| Latency heatmaps | unit | TTFT and TPOT heatmaps: bucket rates `by (le)`, format `heatmap` | section 3's second mode |
| KV cache usage | unit | a panel reads `tl_engine_kv_blocks` | capacity, leaks, preemption pressure |
| Per-tenant usage | unit | `tl_gateway_requests_total` by `tenant` | `ops.09`, noisy neighbors |
| Rate windows survive the scrape interval | boundary | every `rate`/`irate`/`increase` window is `$__rate_interval` or at least 2 m | no flickering panels |
| Queries run in Prometheus (cluster) | conformance | on kind: each query runs in the cluster's Prometheus without error | many-to-many and runtime errors |
| Dashboards provisioned for Grafana (cluster) | conformance | on kind: a ConfigMap labelled `grafana_dashboard=1` holds each `uid` | the dashboard is on screen during a drill |

## 5. The solution, step by step

### 5.1 From first principles

1. **Start from the page.** An alert from `obs.03` names an SLO. The first row must therefore show, per SLO, how fast its budget is burning, with the same thresholds as the alerts (6 and 14.4), plus which alerts are firing.
2. **Then the shape of latency.** A burn says "too many slow requests" but not whether everything shifted or a new slow mode appeared. Only the full distribution answers that, so the second row is heatmaps of the TTFT and TPOT histograms, with a p95 line beside them for the familiar summary.
3. **Then the likeliest cause.** For an LLM engine that is capacity: KV blocks and the queue per pod. The third row reads exactly those gauges.
4. **Then who.** A flood from one tenant looks like organic growth unless usage is split by tenant, so the last row groups by `tenant`.
5. **Portability is forced by "a fresh cluster loads the file".** That fixes a stable `uid`, `"id": null`, and a datasource variable instead of an instance's uid.
6. **Correctness of each query is forced by the data** (sections 2.2 and 2.3): only contract names, bucket rates `by (le)` with format `heatmap`, `$__rate_interval` windows.

### 5.2 The code, annotated

**The dashboard header** (top-level fields; `panels` follows):

```json
{
  "id": null,
  "uid": "forge-serving",
  "title": "forge: serving path",
  "tags": ["forge", "serving", "slo"],
  "timezone": "utc",
  "schemaVersion": 39,
  "refresh": "30s",
  "time": {"from": "now-1h", "to": "now"},
  "templating": {"list": [
    {"name": "datasource", "type": "datasource", "query": "prometheus", "label": "Prometheus", "current": {}, "hide": 0, "refresh": 1, "regex": "", "options": []}
  ]}
}
```

`"id": null` lets each Grafana assign its own row id; `uid` is what URLs, links from runbooks, and the provisioning sidecar key on, so it is chosen once and never changes. The `datasource` variable resolves to whichever Prometheus datasource the Grafana has, so the file contains no instance-specific uid. UTC keeps the times on the dashboard identical to the timestamps in logs and postmortems.

**An SLO status stat.** The query turns the recorded error ratio into a burn rate by dividing by the budget ($\beta = 0.01$ for TTFT and TPOT, $0.005$ for availability), so the thresholds can be the alerts' factors directly:

```json
{
  "title": "TTFT error budget burn (1h)",
  "type": "stat",
  "datasource": {"type": "prometheus", "uid": "${datasource}"},
  "gridPos": {"x": 0, "y": 1, "w": 6, "h": 5},
  "targets": [{
    "datasource": {"type": "prometheus", "uid": "${datasource}"},
    "refId": "A",
    "expr": "slo:sli_error:ratio_rate1h{slo=\"ttft\",slo_profile=\"prod\"} / 0.01",
    "legendFormat": "1h burn rate"
  }],
  "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "background"},
  "fieldConfig": {"defaults": {"unit": "x", "decimals": 1,
    "thresholds": {"mode": "absolute", "steps": [
      {"color": "green", "value": null},
      {"color": "orange", "value": 6},
      {"color": "red", "value": 14.4}
    ]}}, "overrides": []}
}
```

Orange at 6 and red at 14.4 mean the colour changes exactly when the slow and fast alerts' long-window condition holds; section 3's burn of 5 is green. The fourth panel of the row is a table of `ALERTS{alertstate="firing",alertname=~".+BudgetBurn(Fast|Slow)"}`, an instant query with format `table`, so it lists what is firing now rather than drawing a history.

**A latency heatmap**, the panel section 3 computes by hand:

```json
{
  "title": "TTFT distribution",
  "type": "heatmap",
  "targets": [{
    "refId": "A",
    "expr": "sum by (le) (rate(gen_ai_server_time_to_first_token_seconds_bucket{job=\"forge-gateway\"}[$__rate_interval]))",
    "format": "heatmap",
    "legendFormat": "{{le}}"
  }],
  "options": {"calculate": false, "yAxis": {"unit": "s"}, "cellGap": 1,
              "color": {"scheme": "Oranges", "mode": "scheme"}}
}
```

(datasource and gridPos elided). `sum by (le)` merges pods and models but keeps one series per bound, which is what the cumulative-to-cell subtraction needs. `"format": "heatmap"` tells Grafana the series are cumulative buckets, so it computes $r_i - r_{i-1}$ per cell. `"calculate": false` stops Grafana from re-bucketing data that is already bucketed. `legendFormat` `{{le}}` labels each row by its bound.

**The rest of the panels** are one-line queries, each answering one question:

| Panel | Query |
|---|---|
| TTFT and TPOT p95 | `histogram_quantile(0.95, sum by (le) (rate(<histogram>_bucket{job="forge-gateway"}[$__rate_interval])))` |
| KV cache blocks by state | `sum by (state) (tl_engine_kv_blocks)` |
| Queue depth and running sequences | `sum by (pod) (tl_engine_queue_depth)`, `sum by (pod) (tl_engine_active_sequences)` |
| Requests per second by tenant | `sum by (tenant) (rate(tl_gateway_requests_total[$__rate_interval]))` |
| Errors (5xx) per second by tenant | `sum by (tenant) (rate(tl_gateway_requests_total{code=~"5.."}[$__rate_interval]))` |

The gauges need no `rate`; every counter does, and every `rate` uses `$__rate_interval`. Rows are `type: row` panels at full width (`w: 24`), so the layout reads top to bottom in the order of section 5.1.

Full solution: [`serving.json`](../../../../course/solutions/entry/deploy/observability/dashboards/serving.json).

### 5.3 Why this solution

- **Cost.** Thirteen queries, each over at most a few dozen series (the label caps of `obs.02`), refreshed every 30 s. The SLO stats read precomputed recording rules instead of recomputing 1 h rates.
- **Why JSON by hand instead of Grafonnet or the Foundation SDK?** One dashboard does not repay a code generator; the JSON is the format the provisioning sidecar reads, and every field is reviewable in a diff. Generators are the right step once there are many dashboards sharing panels (Going further).
- **Why a heatmap and not more percentile lines?** Percentiles summarise one point of the distribution each; a bimodal distribution (section 3) can keep p95 flat while 5% of requests take ten times longer. A heatmap shows every bucket.
- **Why a ConfigMap and not the Grafana API?** The sidecar reconciles from the cluster's state, so a recreated cluster gets the dashboard back with no extra step, and the file in git stays the only source.

### 5.4 Explain it back

1. Why must the heatmap target use `"format": "heatmap"`?
   <details><summary>Answer</summary>Prometheus buckets are cumulative: the `le="1.0"` series counts every request at most 1 s, including the fast ones. With the default time series format, Grafana draws the cumulative rates as they are, so the top cell holds the total and looks busiest. The heatmap format subtracts neighbouring bounds, giving the requests per second in each interval.</details>
2. Why divide the recorded error ratio by 0.01 in the TTFT stat, and why are the thresholds 6 and 14.4?
   <details><summary>Answer</summary>The recording rule stores the error ratio $e$; the burn rate is $e / \beta$ with $\beta = 1 - 0.99$. Showing the burn rate lets the thresholds be the slow and fast alerts' factors, so the colour matches the alert logic over the long window.</details>
3. What goes wrong after recreating the cluster if the panels name the datasource uid of the Grafana the dashboard was exported from?
   <details><summary>Answer</summary>The new Grafana's Prometheus datasource has a different uid, so every panel shows "datasource not found". The `${datasource}` variable resolves to whatever Prometheus datasource exists.</details>
4. Why `$__rate_interval` instead of `[1m]`?
   <details><summary>Answer</summary>`rate` needs at least two samples in the window. `$__rate_interval` is at least four scrape intervals and grows with the zoom; a fixed `[1m]` at a 30 s scrape holds two samples at best, and none after a late scrape, so the panel flickers to "No data".</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. Exported with `"id": 17` and a generated `uid` | a second import overwrites or duplicates; links break on every re-export | `"id": null` and the hand-chosen `uid` `forge-serving` |
| 2. The exporting Grafana's datasource uid in every panel | "datasource not found" on a fresh cluster | every panel and target uses `${datasource}` |
| 3. A metric name from a blog post or a typo (`ttft_seconds_bucket`) | an empty panel that reads as zero traffic | queries use only `metrics.yaml` names, `slo:sli_error:*`, and `ALERTS` |
| 4. A heatmap without `by (le)` or with the time series format | the busiest-looking cell is the slow tail | `sum by (le)` with `"format": "heatmap"` and `"calculate": false` |
| 5. Usage summed over all tenants | one tenant's flood looks like organic growth | the tenant row groups `by (tenant)` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `obs.02` | every series a panel reads, under its job names |
| Back | `obs.03` | `slo:sli_error:*` recording rules and the six alerts in the status row |
| Forward | `ops.01` | the drill's diagnosis starts at the TTFT heatmap and the queue depth per pod |
| Forward | `ops.09` | the tenant row shows the noisy neighbor |
| Forward | `obs.05` | the control-plane dashboard (queues, DLQ) follows the same rules |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| hand-edited JSON | Grafonnet (Jsonnet), the Grafana Foundation SDK, Grafana's `grafanactl` | dashboards generated from code, shared panel libraries | [Grafonnet](https://grafana.github.io/grafonnet/) |
| a ConfigMap per folder | Grafana provisioning from Git, or the Grafana Operator | sync, folders, permissions as code | [Grafana Operator](https://grafana.github.io/grafana-operator/) |
| one serving dashboard | RED (rate, errors, duration) per service, USE (utilization, saturation, errors) per resource | a standard layout every team reads the same way | *The RED Method* (Wilkie), *The USE Method* (Gregg) |
| burn-rate stats | SLO dashboards from Sloth or Pyrra | budget remaining over the period, burn history | [Pyrra](https://github.com/pyrra-dev/pyrra) |
