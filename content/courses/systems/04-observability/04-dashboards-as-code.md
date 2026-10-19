# Dashboards as code for the serving path

In the drill [A decode worker dies mid-stream](../05-incident-response-and-chaos/01-kill-decode.md), the on-call engineer is paged with `TTFTBudgetBurnFast`. The page says the budget is burning; it does not say why. The first five minutes of every incident are the same questions: which SLO, since when, is latency uniformly worse or is there a new slow mode, is the engine out of KV blocks, is one tenant flooding the gateway. Typing PromQL during an incident is slow and error-prone, and a dashboard someone clicked together in Grafana last month is gone the next time the cluster is recreated, which [kind cluster, local registry, and port mappings](../../infrastructure/01-containers-kubernetes/02-kind-cluster-and-local-registry.md) does whenever port mappings change.

Dashboards kept as JSON in the repo are reviewed, versioned, and redeployed like every other artifact. This section builds one: the serving dashboard, with one row per incident question, reading only the series from [Metrics and log correlation](02-metrics-and-log-correlation.md) and the recording rules and alerts from [SLOs and multi-window burn-rate alerts](03-slos-and-burn-rate-alerts.md). Along the way you will see why a heatmap of a Prometheus histogram is easy to get subtly wrong, and how to make a dashboard file that any fresh Grafana loads unchanged.

The solution is [`deploy/observability/dashboards/serving.json`](../../../../course/solutions/entry/deploy/observability/dashboards/serving.json): the serving dashboard (`uid: forge-serving`; rows SLOs, Latency, Engine capacity, Tenants), as Grafana JSON in git, loaded by the stack's Grafana from a ConfigMap. The contracts are [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml) (the only metric names a panel may read), the SLO rules of the previous section, and [`helm/observability.md`](../../../../course/contracts/helm/observability.md) (Grafana on NodePort 30300).

> **Learning objectives**
> - Write Grafana dashboard JSON that is portable: a stable `uid`, no instance `id`, and a datasource variable.
> - Build a correct heatmap of a Prometheus histogram, and compute one heatmap column by hand from cumulative bucket counts.
> - Choose `rate()` windows that survive the scrape interval.
> - Lay out a dashboard as one row per incident question: SLO status, latency distribution, engine capacity, tenant usage.
> - Provision dashboards to Grafana from git through a labelled ConfigMap.

## The dashboard JSON model

A Grafana dashboard is a JSON document, and a handful of its fields decide whether it is portable:

| Field | Meaning | Rule |
|---|---|---|
| `uid` | the dashboard's identity in URLs and provisioning | stable, chosen by the author, at most 40 of `[A-Za-z0-9_-]` |
| `id` | the database row id in one Grafana | `null` in git: another Grafana assigns its own |
| `panels[]` | each panel: `type`, `title`, `gridPos`, `datasource`, `targets[]` | `type: row` panels group the others |
| `targets[].expr` | the PromQL a panel runs | contract metrics, recording rules, `ALERTS` only |
| `templating.list[]` | variables, e.g. `datasource` of type `datasource`, query `prometheus` | panels refer to `${datasource}` |

A dashboard is **code** when a fresh Grafana loads the file from git unchanged: a stable `uid`, no instance `id`, and a datasource that is a variable, not the uid of the Grafana you exported from. Every query names only contract metrics, the SLO recording rules, or `ALERTS`, and parses; `promtool` can prove both.

## Queries per panel kind

Each panel kind answers a different kind of question, and each has a characteristic query shape:

| Panel | Question | Query shape |
|---|---|---|
| stat | is this SLO burning now? | the burn rate, `slo:sli_error:ratio_rate1h{slo="ttft",slo_profile="prod"} / 0.01`, thresholds at 6 and 14.4 |
| table | which alerts fire? | `ALERTS{alertstate="firing",alertname=~".+BudgetBurn(Fast\|Slow)"}`, instant, format table |
| heatmap | how are latencies distributed over time? | `sum by (le) (rate(<histogram>_bucket{job="<system>-gateway"}[$__rate_interval]))`, format `heatmap` |
| time series | KV blocks by state, queue depth | `sum by (state) (tl_engine_kv_blocks)` |
| time series | who sends what | `sum by (tenant) (rate(tl_gateway_requests_total[$__rate_interval]))` |

The heatmap deserves a closer look, because it is the panel that most often lies. Write the histogram's quantities as follows:

| Symbol | Meaning | Type |
|---|---|---|
| $C_i(t)$ | cumulative count of bucket $b_i$ | counter |
| $r_i = \text{rate}(C_i)$ | requests per second at most $b_i$ | per second |
| $r_i - r_{i-1}$ | requests per second in $(b_{i-1}, b_i]$: one heatmap cell | per second |

A heatmap cell is the **difference** of neighbouring cumulative rates. Grafana computes it only when the target's format is `heatmap`. With the default `time_series` format it stacks the cumulative rates, so every cell counts all faster requests again. A heatmap of a Prometheus histogram therefore needs bucket rates summed `by (le)` and the target format `heatmap`.

## Rates and scrape intervals

Every counter panel uses `rate`, and its window has to match how often the data arrives. `rate(x[w])` uses the samples inside $w$; with fewer than two it returns nothing. Grafana's `$__rate_interval` is at least four scrape intervals and grows with the zoom, so it is never too short. A fixed `[1m]` at a 30 s scrape is two samples on a good day and none after one late scrape: the panel flickers between a value and "No data".

## Provisioning

The last principle is how the file reaches Grafana. The stack's Grafana runs a sidecar that loads every ConfigMap labelled `grafana_dashboard: "1"` as dashboards. The JSON in git is the source; the ConfigMap is generated from it (by `kubectl create configmap --from-file`, a Helm chart, or a Tiltfile); and nothing is edited in the UI without being exported back to the file.

## Working one heatmap column by hand

Two engine replicas are scraped every 10 s. Over the last minute the gateway's TTFT buckets grew as follows, and the last column is what a correct heatmap draws:

| `le` | $\Delta C_i$ over 60 s | $r_i$ (per s) | cell $(b_{i-1}, b_i]$ (per s) |
|---|---|---|---|
| 0.1 | 300 | 5.0 | 5.0 in (0.08, 0.1] and below |
| 0.25 | 540 | 9.0 | 4.0 in (0.1, 0.25] |
| 0.5 | 570 | 9.5 | 0.5 in (0.25, 0.5] |
| 0.75 | 570 | 9.5 | 0 |
| 1.0 | 594 | 9.9 | 0.4 in (0.75, 1.0] |
| `+Inf` | 600 | 10.0 | 0.1 above 1.0 |

The heatmap column for this minute has its mass in the two lowest rows and a separate small band near one second: about 5% of requests ($0.5 / 10$) are in a second mode, here the requests that queued behind a long prefill. The p95 line misses it entirely, because the 95th percentile ($0.95 \times 10 = 9.5$ per s) lands at the 0.5 s bound. With the default format, Grafana would draw 10.0, 9.9, 9.5, 9.5, 9.0, 5.0 stacked in the column, and the slow band would look like the busiest cell.

The SLO status stat next to it shows the 1 h TTFT burn. If the hour looked like this minute, $e = 0.5 / 10 = 0.05$ of requests would be over 500 ms, and with $\beta = 0.01$ the burn is $5$: still green, one step below the orange line at 6 (the slow alert's factor); red starts at 14.4 (the fast alert's).

## Laying out the dashboard

The solution's `deploy/observability/dashboards/serving.json` has four rows, one per question: are we within SLO (status), how are latencies distributed (heatmaps), is the engine full (KV blocks), and who is using it (per tenant).

| Row | Panels |
|---|---|
| SLOs | stat: TTFT, TPOT, and availability burn rate (1 h); table: firing `*BudgetBurn*` alerts |
| Latency | heatmap: TTFT; heatmap: TPOT; time series: TTFT and TPOT p95 |
| Engine capacity | KV cache blocks by state (stacked); queue depth and running sequences per pod |
| Tenants | requests per second by tenant; 5xx per second by tenant |

Every panel and target sets `datasource` to `{"type": "prometheus", "uid": "${datasource}"}` with a `datasource` variable of type `datasource`, query `prometheus`; the stack's provisioned uid `prometheus` is also acceptable. To check queries outside Grafana (with `promtool` or the Prometheus API), replace `$__rate_interval`, `$__interval`, and `$__range` with `5m` and every other variable with `.*`.

A dashboard like this is usually built in Grafana (http://127.0.0.1:30300), exported as JSON, and cleaned of the export mistakes listed at the end of this section. The solution is deployed like this:

```bash
kubectl -n observability create configmap forge-dashboards --from-file=deploy/observability/dashboards \
  --dry-run=client -o yaml | kubectl label --local -f - grafana_dashboard=1 -o yaml | kubectl apply -f -
```

## Building it panel by panel

The layout follows the incident. Start from the page: an SLO alert names an SLO, so the first row must show, per SLO, how fast its budget is burning, with the same thresholds as the alerts (6 and 14.4), plus which alerts are firing. Then the shape of latency: a burn says "too many slow requests" but not whether everything shifted or a new slow mode appeared. Only the full distribution answers that, so the second row is heatmaps of the TTFT and TPOT histograms, with a p95 line beside them for the familiar summary. Then the likeliest cause, which for an LLM engine is capacity: KV blocks and the queue per pod, so the third row reads exactly those gauges. Then who: a flood from one tenant looks like organic growth unless usage is split by tenant, so the last row groups by `tenant`.

Two more constraints come from outside the incident. Portability is forced by the requirement that a fresh cluster loads the file, which fixes a stable `uid`, `"id": null`, and a datasource variable instead of an instance's uid. And the correctness of each query is forced by the data: only contract names, bucket rates `by (le)` with format `heatmap`, and `$__rate_interval` windows.

### The dashboard header

The top-level fields come first; `panels` follows them:

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

`"id": null` lets each Grafana assign its own row id. `uid` is what URLs, links from runbooks, and the provisioning sidecar key on, so it is chosen once and never changes. The `datasource` variable resolves to whichever Prometheus datasource the Grafana has, so the file contains no instance-specific uid. UTC keeps the times on the dashboard identical to the timestamps in logs and postmortems.

### An SLO status stat

The query turns the recorded error ratio into a burn rate by dividing by the budget ($\beta = 0.01$ for TTFT and TPOT, $0.005$ for availability), so the thresholds can be the alerts' factors directly:

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

Orange at 6 and red at 14.4 mean the colour changes exactly when the slow and fast alerts' long-window condition holds; the worked example's burn of 5 is green. The fourth panel of the row is a table of `ALERTS{alertstate="firing",alertname=~".+BudgetBurn(Fast|Slow)"}`, an instant query with format `table`, so it lists what is firing now rather than drawing a history.

### A latency heatmap

This is the panel computed by hand above (datasource and gridPos elided):

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

`sum by (le)` merges pods and models but keeps one series per bound, which is what the cumulative-to-cell subtraction needs. `"format": "heatmap"` tells Grafana the series are cumulative buckets, so it computes $r_i - r_{i-1}$ per cell. `"calculate": false` stops Grafana from re-bucketing data that is already bucketed. `legendFormat` `{{le}}` labels each row by its bound.

### The remaining panels

The rest of the panels are one-line queries, each answering one question:

| Panel | Query |
|---|---|
| TTFT and TPOT p95 | `histogram_quantile(0.95, sum by (le) (rate(<histogram>_bucket{job="forge-gateway"}[$__rate_interval])))` |
| KV cache blocks by state | `sum by (state) (tl_engine_kv_blocks)` |
| Queue depth and running sequences | `sum by (pod) (tl_engine_queue_depth)`, `sum by (pod) (tl_engine_active_sequences)` |
| Requests per second by tenant | `sum by (tenant) (rate(tl_gateway_requests_total[$__rate_interval]))` |
| Errors (5xx) per second by tenant | `sum by (tenant) (rate(tl_gateway_requests_total{code=~"5.."}[$__rate_interval]))` |

The gauges need no `rate`; every counter does, and every `rate` uses `$__rate_interval`. Rows are `type: row` panels at full width (`w: 24`), so the layout reads top to bottom in the order of the incident questions. The full solution is [`serving.json`](../../../../course/solutions/entry/deploy/observability/dashboards/serving.json).

### Why this design

The dashboard runs thirteen queries, each over at most a few dozen series (thanks to the label caps on the metrics), refreshed every 30 s. The SLO stats read precomputed recording rules instead of recomputing 1 h rates.

It is hand-written JSON rather than Grafonnet or the Foundation SDK because one dashboard does not repay a code generator. The JSON is the format the provisioning sidecar reads, and every field is reviewable in a diff. Generators are the right step once there are many dashboards sharing panels.

It uses a heatmap rather than more percentile lines because percentiles summarise one point of the distribution each. A bimodal distribution, like the one in the worked example, can keep p95 flat while 5% of requests take ten times longer. A heatmap shows every bucket.

It is provisioned by ConfigMap rather than the Grafana API because the sidecar reconciles from the cluster's state, so a recreated cluster gets the dashboard back with no extra step, and the file in git stays the only source.

## What can go wrong

A correct dashboard must pass a set of checks. The file must be valid JSON with a unique, stable `uid`, a null `id`, a title, a `schemaVersion`, and panels, so it loads in any Grafana. Every panel and target must use a datasource variable or the uid `prometheus`, so there is no "datasource not found" after a rebuild. Every metric name must be in `metrics.yaml`, the recording rules, or `ALERTS`, so no panel is empty in a way that looks like "no traffic", and `promtool` must parse every query, so syntax errors show up now rather than mid-incident. There must be a status panel (stat, gauge, bar gauge, table, or state timeline) per SLO, since that is the first question of every page; TTFT and TPOT heatmaps with bucket rates `by (le)` and format `heatmap`, to catch the second mode; a panel reading `tl_engine_kv_blocks`, for capacity, leaks, and preemption pressure; and `tl_gateway_requests_total` by `tenant`, for [A noisy neighbor](../05-incident-response-and-chaos/09-noisy-neighbor.md). Every `rate`, `irate`, or `increase` window must be `$__rate_interval` or at least 2 m, so no panel flickers. On kind, each query must run in the cluster's Prometheus without error (which catches many-to-many and other runtime errors), and a ConfigMap labelled `grafana_dashboard=1` must hold each `uid`, so the dashboard is on screen during a drill.

Most failures come from exporting a dashboard out of the UI without cleaning it:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Exported with `"id": 17` and a generated `uid` | a second import overwrites or duplicates; links break on every re-export | `"id": null` and the hand-chosen `uid` `forge-serving` |
| The exporting Grafana's datasource uid in every panel | "datasource not found" on a fresh cluster | every panel and target uses `${datasource}` |
| A metric name from a blog post or a typo (`ttft_seconds_bucket`) | an empty panel that reads as zero traffic | queries use only `metrics.yaml` names, `slo:sli_error:*`, and `ALERTS` |
| A heatmap without `by (le)` or with the time series format | the busiest-looking cell is the slow tail | `sum by (le)` with `"format": "heatmap"` and `"calculate": false` |
| Usage summed over all tenants | one tenant's flood looks like organic growth | the tenant row groups `by (tenant)` |

## Where this leads

Every series a panel reads comes from [Metrics and log correlation](02-metrics-and-log-correlation.md), under its job names, and the status row reads the `slo:sli_error:*` recording rules and six alerts of [SLOs and multi-window burn-rate alerts](03-slos-and-burn-rate-alerts.md). Going forward, the diagnosis in [A decode worker dies mid-stream](../05-incident-response-and-chaos/01-kill-decode.md) starts at the TTFT heatmap and the queue depth per pod, the tenant row shows the culprit in [A noisy neighbor](../05-incident-response-and-chaos/09-noisy-neighbor.md), and the control-plane dashboard (queues, DLQ) in [Observability for the control plane](05-control-plane-observability.md) follows the same rules.

Each piece has a production equivalent. Hand-edited JSON gives way to Grafonnet (Jsonnet), the Grafana Foundation SDK, or Grafana's `grafanactl`: dashboards generated from code with shared panel libraries. A ConfigMap per folder gives way to Grafana provisioning from Git or the Grafana Operator, with sync, folders, and permissions as code. One serving dashboard grows into a standard layout every team reads the same way: RED (rate, errors, duration) per service and USE (utilization, saturation, errors) per resource. And burn-rate stats grow into SLO dashboards from Sloth or Pyrra, showing budget remaining over the period and burn history.

### Further reading

- [Grafana dashboard JSON model](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/view-dashboard-json-model/) (free)
- [Heatmaps of Prometheus histograms](https://grafana.com/docs/grafana/latest/panels-visualizations/visualizations/heatmap/) (free)
- Tom Wilkie, *The RED Method*, and Brendan Gregg, *The USE Method* (free articles)
- [Grafonnet](https://grafana.github.io/grafonnet/)
- [Grafana Operator](https://grafana.github.io/grafana-operator/)
- [Pyrra](https://github.com/pyrra-dev/pyrra)

## Key terms

- **dashboard as code**: a dashboard whose JSON lives in git and loads unchanged into any fresh Grafana.
- **`uid`**: the stable, author-chosen identity of a dashboard, used in URLs and provisioning.
- **datasource variable**: a template variable that resolves to whichever Prometheus datasource the Grafana has.
- **heatmap format**: the target format that tells Grafana to subtract neighbouring cumulative bucket rates into cells.
- **`$__rate_interval`**: a Grafana variable of at least four scrape intervals, used as the `rate()` window.
- **provisioning sidecar**: the Grafana companion that loads every ConfigMap labelled `grafana_dashboard: "1"`.

## Summary

- A dashboard is code when a fresh Grafana loads the file from git unchanged: a stable `uid`, no instance `id`, and a datasource variable.
- Every query names only contract metrics, the SLO recording rules, or `ALERTS`, and parses.
- A heatmap of a Prometheus histogram needs bucket rates summed `by (le)` and the target format `heatmap`; the p95 line alone hides a second mode.
- One row per question: are we within SLO, how are latencies distributed, is the engine full, who is using it.
- `rate()` windows follow the scrape interval: `$__rate_interval`, never a fixed `[1m]` at a 30 s scrape.

## Review questions

1. Why must the heatmap target use `"format": "heatmap"`?
   <details><summary>Answer</summary>Prometheus buckets are cumulative: the `le="1.0"` series counts every request at most 1 s, including the fast ones. With the default time series format, Grafana draws the cumulative rates as they are, so the top cell holds the total and looks busiest. The heatmap format subtracts neighbouring bounds, giving the requests per second in each interval.</details>
2. Why divide the recorded error ratio by 0.01 in the TTFT stat, and why are the thresholds 6 and 14.4?
   <details><summary>Answer</summary>The recording rule stores the error ratio $e$; the burn rate is $e / \beta$ with $\beta = 1 - 0.99$. Showing the burn rate lets the thresholds be the slow and fast alerts' factors, so the colour matches the alert logic over the long window.</details>
3. What goes wrong after recreating the cluster if the panels name the datasource uid of the Grafana the dashboard was exported from?
   <details><summary>Answer</summary>The new Grafana's Prometheus datasource has a different uid, so every panel shows "datasource not found". The `${datasource}` variable resolves to whatever Prometheus datasource exists.</details>
4. Why `$__rate_interval` instead of `[1m]`?
   <details><summary>Answer</summary>`rate` needs at least two samples in the window. `$__rate_interval` is at least four scrape intervals and grows with the zoom; a fixed `[1m]` at a 30 s scrape holds two samples at best, and none after a late scrape, so the panel flickers to "No data".</details>
