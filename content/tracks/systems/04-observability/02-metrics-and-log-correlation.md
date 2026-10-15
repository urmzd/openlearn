<!-- ol:module obs.02 -->
# Metrics and log correlation

## Overview

| | |
|---|---|
| **Module** | `obs.02` · practice · ops, go · Pass 7 · 4 to 6 h |
| **You build** | `/metrics` on the health port (9464) of your gateway (in `go/cmd/gateway`, recording the gateway's instruments) and engine (`L10.5` gauges, `L10.7` histograms); JSON logs with `trace_id` and `span_id` (`otelx.LogHandler`); `deploy/observability/monitors.yaml` (what Prometheus scrapes) and `deploy/observability/collector.yaml` (the OpenTelemetry Collector) |
| **Contract** | [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml) (names, types, buckets, labels), [`otel/semconv.md`](../../../../course/contracts/otel/semconv.md) (log fields), [`helm/observability.md`](../../../../course/contracts/helm/observability.md) (the selector label, the collector Service) |
| **Tests** | `course/tests/obs.02/` (`check` runs `artifacts.py`; what each test checks: section 4) |
| **Needs** | `obs.01` (`go/otelx`: `LogHandler`, `Setup`, the server span), `dep.03` (the charts whose pods are scraped), `L10.7` (the engine's serve loop that records the GenAI and HTTP histograms and logs each request in its span); reading: `obs.00`, `L10.5`, and `gw.01` (the other code it scrapes) |
| **Used by** | `obs.03` turns these series into SLOs; `obs.04` draws them; `ops.01` reads them during the drill |
| **Milestone** | MS-prod (contract metrics scraped on kind) |
| **Optional depth** | [Prometheus exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/) (free), [Histograms and summaries](https://prometheus.io/docs/practices/histograms/) (free), *Observability Engineering*, ch. 8 and 9 |

## Key Takeaways

- A histogram is a set of **cumulative counters**, one per bucket bound `le`. With the contract's bounds, "fraction of requests at most 500 ms" is one exact division; with other bounds it cannot be computed at all.
- Prometheus finds targets only through **ServiceMonitor** or **PodMonitor** objects the stack selects (`release: observability`), and names the series' `job` from them. SLO rules and dashboards select by that `job`.
- A label that copies client input (a model name) multiplies series; **capped** labels keep 64 values and fold the rest into `_other`.
- There is no log backend: log-to-trace correlation is `kubectl logs | grep <trace_id>`, which works only if every line is JSON with `trace_id`, and never holds a prompt or a key.
- The collector is the one door for telemetry: OTLP in on 4317 and 4318, traces to Tempo, pushed Python metrics out through its Prometheus exporter; `memory_limiter` first and `batch` in every pipeline.

## How to work this chapter

```bash
ol start obs.02                # records the start; there are no stubs
ol tests obs.02                # read the test catalog first
# add /metrics and JSON logs to the gateway, monitors.yaml and collector.yaml, then:
OL_SMOKE=1 ol check obs.02     # static and local tiers: your services as local processes
curl -s 127.0.0.1:<health_port>/metrics | grep '^gen_ai_server_time_to_first_token_seconds_bucket'
# on kind: apply the monitors, install the collector, then the full check
kubectl apply -f deploy/observability/monitors.yaml
helm upgrade --install observability deploy/observability -n observability \
  -f deploy/observability/values.yaml -f deploy/observability/collector.yaml   # dep.03's umbrella chart
ol check obs.02
```

---

## 1. Why now

Traces (`obs.01`) explain one request. They do not say whether TTFT is getting worse for everyone, how many KV blocks are left, or which tenant is flooding the gateway: that takes **metrics**, counts and distributions aggregated over all requests, cheap enough to keep for every request. Your engine already serves a few gauges on its health port (`L10.5`); the gateway serves none, the histograms the SLOs need (`obs.03`) do not exist yet, and nothing in the cluster scrapes either. When `ops.01` kills a decode pod, the page fires from these series, and the person paged starts from a dashboard (`obs.04`), then needs the log lines of one slow request. This module makes all of that possible.

## 2. Principles

### 2.1 Counters, gauges, histograms

| Symbol | Meaning | Type |
|---|---|---|
| $c(t)$ | a **counter**: a total that only grows (requests served) | non-negative, resets to 0 on restart |
| $g(t)$ | a **gauge**: a value that goes up and down (free KV blocks) | real |
| $b_1 < \dots < b_k$ | the bucket bounds of a histogram (`le`, "less or equal") | seconds, from `metrics.yaml` |
| $C_i(t)$ | requests observed so far with value $\le b_i$ | counter; $C_{+\infty}$ is the total |
| $\Delta C_i$ | the increase of $C_i$ over a window ($\text{rate} \times$ window) | non-negative |

A **histogram** is the counters $C_1 \le C_2 \le \dots \le C_{+\infty}$ plus a sum. They are **cumulative**: an observation of 0.07 s increments every $C_i$ with $b_i \ge 0.07$. Over a window, the fraction of requests at most $b_i$ is

$$\text{frac}(b_i) = \frac{\Delta C_i}{\Delta C_{+\infty}}$$

which is exact **only at a bound**. That is why `metrics.yaml` fixes the bounds and why the check compares them exactly: an engine with its own bounds silently breaks every SLO rule that reads `le="0.5"`.

### 2.2 The exposition format and names

Every service answers `GET /metrics` on its health port with Prometheus text:

```
# TYPE gen_ai_server_time_to_first_token_seconds histogram
gen_ai_server_time_to_first_token_seconds_bucket{gen_ai_operation_name="chat",gen_ai_request_model="smol-135m",tl_engine_role="gateway",le="0.5"} 3
...
gen_ai_server_time_to_first_token_seconds_count{gen_ai_operation_name="chat",gen_ai_request_model="smol-135m",tl_engine_role="gateway"} 4
```

OTel instrument names become Prometheus names by replacing `.` with `_` and adding the unit (`_seconds`); counters end in `_total`. The `prometheus` column of `metrics.yaml` is the exact result; serve that name.

### 2.3 Labels and cardinality

Every distinct combination of label values is its own **series**, stored and scanned separately. A histogram with 16 bounds is 19 series per combination: 16 buckets, the `+Inf` bucket, `_sum`, and `_count`. A label taking client input (the `model` field of a request) lets a client mint series: 80 random model names times 19 is 1 520 series for TTFT alone. `metrics.yaml` marks such labels `capped`: a process keeps the first 64 distinct values and records every later one as `_other`. Labels with `values` take only those.

### 2.4 How Prometheus finds your pods

kube-prometheus-stack runs the Prometheus Operator, which turns **ServiceMonitor** objects (scrape the endpoints behind a Service port) and **PodMonitor** objects (scrape a named container port of matching pods) into scrape config, but only the ones carrying the stack's selector label `release: observability`. Each target's series get a `job` label: a ServiceMonitor names it after the Service, a PodMonitor after `<namespace>/<monitor name>`, unless `jobLabel` names a label to take it from or a relabeling (`targetLabel: job`) sets it. The scrape interval decides the shortest window you can compute a rate over: `rate()` needs two samples, so the drill profile's 25 s window needs a scrape every 10 s or faster.

### 2.5 The collector

Services export traces as OTLP to one address (`[otel].endpoint`), the **OpenTelemetry Collector**, which forwards them to Tempo. Python subprocesses (training, corpus stages) cannot be scraped because they live seconds to hours and run outside any Service (D9), so they **push** OTLP metrics to the collector, whose `prometheus` exporter exposes them for Prometheus to scrape. A collector pipeline is `receivers -> processors -> exporters`; `memory_limiter` goes first so an overload drops data instead of the pod, and `batch` before export.

### 2.6 Logs you can join to a trace

Each line a service writes is one JSON object (`semconv.md`, Logs): `ts`, `level`, `msg`, `service`, and, inside a request, `trace_id` and `span_id` of the current span. `otelx.LogHandler` adds the two ids to every record logged with the request's context. The prompt, the completion, and the API key never appear: logs are retained longer and read by more people than any request.

## 3. Worked example by hand

Four requests reach the gateway, with first-token times 0.03 s, 0.07 s, 0.30 s, and 0.60 s. The TTFT bounds of `metrics.yaml` around them are 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75.

| `le` | requests $\le$ `le` | $C_i$ |
|---|---|---|
| 0.02 | none | 0 |
| 0.04 | 0.03 | 1 |
| 0.06 | 0.03 | 1 |
| 0.08 | 0.03, 0.07 | 2 |
| 0.1, 0.25 | 0.03, 0.07 | 2 |
| 0.5 | 0.03, 0.07, 0.30 | 3 |
| 0.75 and up, `+Inf` | all four | 4 |

The fraction within the 500 ms SLO threshold is $C_{0.5} / C_{+\infty} = 3/4 = 0.75$, exact. Had the engine used bounds 0.4 and 0.8, the 0.30 and the 0.60 s requests would fall in buckets that straddle 0.5, and no division of counters would give the answer.

**Cardinality.** After 80 requests with model names `model-000` to `model-079`, an uncapped gateway serves 81 values of `gen_ai_request_model` (the 80 plus `tracer`): $81 \times 19 = 1\,539$ TTFT series. Capped, it serves 64 named values plus `_other`: at most $65 \times 19 = 1\,235$, and that ceiling holds for a million names.

**Correlation.** One request carries `traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01`. The gateway logs:

```json
{"ts":"2026-10-09T10:00:01Z","level":"info","msg":"proxied","service":"forge-gateway","status":200,"trace_id":"4bf92f3577b34da6a3ce929d0e0e4736","span_id":"a1b2c3d4e5f60718"}
```

`kubectl -n forge logs deploy/forge-gateway | grep 4bf92f3577b34da6a3ce929d0e0e4736` finds it; the same grep on the engine finds the engine's lines, whose `span_id` is a different span of the same trace. That is the check `test_logs_carry_the_trace_id` runs on your local processes.

## 4. The interface

| Artifact | Requirement |
|---|---|
| `GET /metrics` on the health port of the gateway and of every engine | Prometheus text; every instrument of `metrics.yaml` its `emitted_by` names, except the later ones (`tl.engine.spec_accept_rate` with `L10.8`, `tl.kv.transfer.*` in disaggregated roles, `tl.gateway.policy.denials` with `gw.08`); exact names, `# TYPE`, bounds, label keys, enumerated values; capped labels at most 64 values plus `_other` |
| logs | one JSON object per line on stdout: `ts`, `level`, `msg`, `service`, and `trace_id`, `span_id` inside a request; no prompt, no key |
| `deploy/observability/monitors.yaml` | a PodMonitor or ServiceMonitor per component, label `release: observability`, selecting every pod your charts render (every engine role release too), on the 9464 port, path `/metrics`, `interval` of 10 s or less, `job` = `<system>-gateway` / `<system>-engine` |
| `deploy/observability/collector.yaml` | the collector config, under `otel-collector.config:` (values for `dep.03`'s umbrella chart), `config:` (the chart alone), or at the top (raw): OTLP on `0.0.0.0:4317` and `:4318`; a `traces` pipeline to Tempo; a `metrics` pipeline to the `prometheus` exporter; `memory_limiter` first and `batch` in each |

The reference uses PodMonitors that select pods by `app.kubernetes.io/component` (shared by every release of a chart, including the engine's per-role releases `<system>-engine-prefill` and `-decode`) and set `job` with a relabeling, so no Service needs a metrics port.

### What the tests check

The **local tier** runs your `[build]` steps and starts `[services.engine]` and `[services.gateway]` from `system.toml` as `ol milestone` does, sends six chat completions, one traced request whose prompt holds a canary string, and 80 requests with random model names to each service, then scrapes both health ports and reads both logs.

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_every_pod_is_scraped_on_its_health_port` | conformance | each rendered gateway workload, and the engine rendered as unified, prefill, and decode releases, is selected by a monitor on port 9464, path `/metrics` | no target, no series |
| `test_monitors_are_selected_by_the_stack` | conformance | every monitor carries `release: observability` | the operator ignores the rest |
| `test_scrape_job_names_and_interval` | unit | job names `<system>-gateway` and `<system>-engine`; interval 10 s or less | `obs.03` rules and `obs.04` panels select by job; drill windows |
| `test_collector_pipelines` | unit | the four collector requirements of the table | traces reach Tempo; Python metrics reach Prometheus |
| `test_services_serve_contract_metrics` | conformance | names, types, bounds, labels, values of both scrapes | every rule and panel downstream |
| `test_capped_labels_stay_under_the_cap` | fault | after 80 model names, at most 64 values plus `_other` per capped label | an attacker cannot exhaust Prometheus |
| `test_logs_carry_the_trace_id` | conformance | both services log JSON with the traced request's `trace_id` and a `span_id` | the section 3 grep |
| `test_logs_never_contain_the_prompt` | boundary | the canary prompt and the API key appear in no log line | user data stays out of logs |
| `test_targets_up_in_prometheus` | conformance | on kind: `up` is 1 for every target of both jobs | the deployed proof of the static tier |
| `test_kubectl_logs_find_the_trace` | conformance | on kind: a request through the gateway NodePort, then `kubectl logs` finds its trace id | correlation in the cluster, during `ops.01` |

The last two are the **cluster tier**; `OL_SMOKE=1` skips them with the reason.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. A monitor whose selector matches no pod (selecting `app.kubernetes.io/name: <system>-engine` misses the `-decode` release), or the wrong port | Prometheus shows no target; graphs are empty, not zero | `test_every_pod_is_scraped_on_its_health_port` |
| 2. A PodMonitor without a job relabeling (or `jobLabel`), or the default 30 s interval | series under `job="observability/forge-gateway"`, so every SLO rule is silent; drill windows hold one sample | `test_scrape_job_names_and_interval` |
| 3. The model name (or a user id) as an uncapped label | Prometheus memory grows with traffic until it is OOM-killed | `test_capped_labels_stay_under_the_cap` |
| 4. Logging the request body "for debugging" | prompts and keys in `kubectl logs`, retained for weeks | `test_logs_never_contain_the_prompt` |
| 5. Collector OTLP bound to `localhost:4317`, no `memory_limiter`, or traces exported to the debug exporter only | services cannot reach it; the collector OOMs under a burst; Tempo stays empty | `test_collector_pipelines` |
| 6. Your own bucket bounds ("these fit our latencies better") | SLO rules reading `le="0.5"` find nothing | `test_services_serve_contract_metrics` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `obs.01` | `LogHandler` stamps the ids; `Setup` and the SERVER span give each log line its span |
| Back | `dep.03` | the charts whose pods and container ports the monitors select |
| Back | `L10.7` | the local tier runs your engine: its serve loop records the TTFT, TPOT, duration, and HTTP histograms this module scrapes, and writes the JSON log line with the trace id |
| Forward | `obs.03` | SLIs over the TTFT and TPOT histograms and the 5xx ratio, selected by `job` |
| Forward | `obs.04` | heatmaps of the same buckets, KV usage, per-tenant usage |
| Forward | `ops.01` | the drill's detection and resolution queries read these series |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| hand-written exposition | `prometheus/client_golang` or the OTel Prometheus exporter | registries, collectors, exemplars | [client_golang](https://github.com/prometheus/client_golang) |
| classic histograms with fixed bounds | native histograms | exponential buckets, no bounds to choose, far fewer series | [Native histograms](https://prometheus.io/docs/specs/native_histograms/) |
| `kubectl logs \| grep` | Loki or the OTel log pipeline | indexed search, retention, log-to-trace links in Grafana | [Loki](https://grafana.com/docs/loki/latest/) |
| PodMonitors per component | the collector's Prometheus receiver or target allocator | one scraper, relabelling in the pipeline | Collector `prometheusreceiver` |
| exemplars: not used | trace exemplars on histogram buckets | jump from a slow bucket straight to a trace | [Exemplars](https://prometheus.io/docs/prometheus/latest/feature_flags/#exemplars-storage) |
