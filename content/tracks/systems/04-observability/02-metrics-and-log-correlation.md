# Metrics and log correlation

## Overview

| | |
|---|---|
| **Module** | `obs.02` · practice · ops, go · Pass 7 · 4 to 6 h |
| **Solution** | the gateway's telemetry in [`go/cmd/gateway/telemetry.go`](../../../../course/solutions/entry/go/cmd/gateway/telemetry.go) (`newObsTelemetry`, `wrap`, `withMetrics`, `obsHist`, `obsCapped`, `obsMetrics.observe`, `exposition`), wired in [`main.go`](../../../../course/solutions/entry/go/cmd/gateway/main.go); the engine's `/metrics` (`L10.5` gauges, `L10.7` histograms); [`deploy/observability/monitors.yaml`](../../../../course/solutions/entry/deploy/observability/monitors.yaml) (two PodMonitors) and [`deploy/observability/collector.yaml`](../../../../course/solutions/entry/deploy/observability/collector.yaml) (the OpenTelemetry Collector) |
| **Contract** | [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml) (names, types, buckets, labels), [`otel/semconv.md`](../../../../course/contracts/otel/semconv.md) (log fields), [`helm/observability.md`](../../../../course/contracts/helm/observability.md) (the selector label, the collector Service) |
| **Needs** | `obs.01` (`go/otelx`: `LogHandler`, `Setup`, the server span), `dep.03` (the charts whose pods are scraped), `L10.7` (the engine's serve loop that records the GenAI and HTTP histograms and logs each request in its span); reading: `obs.00`, `L10.5`, and `gw.01` (the other code it scrapes) |
| **Used by** | `obs.03` turns these series into SLOs; `obs.04` draws them; `ops.01` reads them during the drill. After this module the production system's contract metrics are scraped on kind |
| **Optional depth** | [Prometheus exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/) (free), [Histograms and summaries](https://prometheus.io/docs/practices/histograms/) (free), *Observability Engineering*, ch. 8 and 9 |

## Key Takeaways

- A histogram is a set of **cumulative counters**, one per bucket bound `le`. With the contract's bounds, "fraction of requests at most 500 ms" is one exact division; with other bounds it cannot be computed at all.
- Prometheus finds targets only through **ServiceMonitor** or **PodMonitor** objects the stack selects (`release: observability`), and names the series' `job` from them. SLO rules and dashboards select by that `job`.
- A label that copies client input (a model name) multiplies series; **capped** labels keep 64 values and fold the rest into `_other`.
- There is no log backend: log-to-trace correlation is `kubectl logs | grep <trace_id>`, which works only if every line is JSON with `trace_id`, and never holds a prompt or a key.
- The collector is the one door for telemetry: OTLP in on 4317 and 4318, traces to Tempo, pushed Python metrics out through its Prometheus exporter; `memory_limiter` first and `batch` in every pipeline.

## How to read this chapter

Sections 1 to 3 build metrics and log correlation from first principles and work a histogram, a cardinality bound, and a log grep by hand. Section 4 fixes the artifacts and what a correct setup must handle. Section 5 walks through the solution (the gateway's telemetry code, the PodMonitors, and the collector config) and explains why each choice is the right one. Try to predict each step before reading it.

The solution is used like this:

```bash
curl -s 127.0.0.1:<health_port>/metrics | grep '^gen_ai_server_time_to_first_token_seconds_bucket'
# on kind: apply the monitors, install the collector
kubectl apply -f deploy/observability/monitors.yaml
helm upgrade --install observability deploy/observability -n observability \
  -f deploy/observability/values.yaml -f deploy/observability/collector.yaml   # dep.03's umbrella chart
```

---

## 1. Why now

Traces (`obs.01`) explain one request. They do not say whether TTFT is getting worse for everyone, how many KV blocks are left, or which tenant is flooding the gateway: that takes **metrics**, counts and distributions aggregated over all requests, cheap enough to keep for every request. The engine already serves a few gauges on its health port (`L10.5`); the gateway serves none, the histograms the SLOs need (`obs.03`) do not exist yet, and nothing in the cluster scrapes either. When `ops.01` kills a decode pod, the page fires from these series, and the person paged starts from a dashboard (`obs.04`), then needs the log lines of one slow request. This module makes all of that possible.

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

which is exact **only at a bound**. That is why `metrics.yaml` fixes the bounds and why a correct service serves them exactly: an engine with its own bounds silently breaks every SLO rule that reads `le="0.5"`.

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

### 2.4 How Prometheus finds the pods

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

`kubectl -n forge logs deploy/forge-gateway | grep 4bf92f3577b34da6a3ce929d0e0e4736` finds it; the same grep on the engine finds the engine's lines, whose `span_id` is a different span of the same trace.

## 4. The interface

| Artifact | Requirement |
|---|---|
| `GET /metrics` on the health port of the gateway and of every engine | Prometheus text; every instrument of `metrics.yaml` its `emitted_by` names, except the later ones (`tl.engine.spec_accept_rate` with `L10.8`, `tl.kv.transfer.*` in disaggregated roles, `tl.gateway.policy.denials` with `gw.08`); exact names, `# TYPE`, bounds, label keys, enumerated values; capped labels at most 64 values plus `_other` |
| logs | one JSON object per line on stdout: `ts`, `level`, `msg`, `service`, and `trace_id`, `span_id` inside a request; no prompt, no key |
| `deploy/observability/monitors.yaml` | a PodMonitor or ServiceMonitor per component, label `release: observability`, selecting every pod the charts render (every engine role release too), on the 9464 port, path `/metrics`, `interval` of 10 s or less, `job` = `<system>-gateway` / `<system>-engine` |
| `deploy/observability/collector.yaml` | the collector config, under `otel-collector.config:` (values for `dep.03`'s umbrella chart), `config:` (the chart alone), or at the top (raw): OTLP on `0.0.0.0:4317` and `:4318`; a `traces` pipeline to Tempo; a `metrics` pipeline to the `prometheus` exporter; `memory_limiter` first and `batch` in each |

The solution uses PodMonitors that select pods by `app.kubernetes.io/component` (shared by every release of a chart, including the engine's per-role releases `<system>-engine-prefill` and `-decode`) and set `job` with a relabeling, so no Service needs a metrics port.

### What a correct solution must handle

The local cases start the engine and the gateway as local processes, send six chat completions, one traced request whose prompt holds a canary string, and 80 requests with random model names to each service, then scrape both health ports and read both logs. The cluster cases run the same on kind.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Every pod is scraped on its health port | conformance | each rendered gateway workload, and the engine rendered as unified, prefill, and decode releases, is selected by a monitor on port 9464, path `/metrics` | no target, no series |
| Monitors are selected by the stack | conformance | every monitor carries `release: observability` | the operator ignores the rest |
| Job names and interval | unit | job names `<system>-gateway` and `<system>-engine`; interval 10 s or less | `obs.03` rules and `obs.04` panels select by job; drill windows |
| Collector pipelines | unit | the four collector requirements of the table | traces reach Tempo; Python metrics reach Prometheus |
| Services serve the contract metrics | conformance | names, types, bounds, labels, values of both scrapes | every rule and panel downstream |
| Capped labels stay under the cap | fault | after 80 model names, at most 64 values plus `_other` per capped label | an attacker cannot exhaust Prometheus |
| Logs carry the trace id | conformance | both services log JSON with the traced request's `trace_id` and a `span_id` | the section 3 grep |
| Logs never contain the prompt | boundary | the canary prompt and the API key appear in no log line | user data stays out of logs |
| Targets are up in Prometheus (cluster) | conformance | on kind: `up` is 1 for every target of both jobs | the deployed proof of the static setup |
| `kubectl logs` finds the trace (cluster) | conformance | on kind: a request through the gateway NodePort, then `kubectl logs` finds its trace id | correlation in the cluster, during `ops.01` |

## 5. The solution, step by step

### 5.1 From first principles

1. **The SLO must be an exact division of counters** (section 2.1). So the histograms must use the contract's bounds verbatim, and the solution copies them into constants rather than choosing its own.
2. **A counter-only representation is the cheapest correct one.** Each observation increments one bucket counter, a sum, and a total under one mutex; the cumulative form Prometheus wants is computed only at scrape time. No background goroutine and no I/O on the request path.
3. **Series must be bounded by the system, not by clients** (section 2.3). Every label whose value comes from the request (model, route, status code, tenant) passes through a cap of 64 values per process.
4. **Prometheus must find every pod, under the job name the rules use** (section 2.4). Scraping pods directly (PodMonitor) makes every replica and every role release its own target; selecting by `app.kubernetes.io/component` covers all releases of a chart; a relabeling fixes `job`; 10 s keeps two samples in a 25 s window.
5. **One door for pushed telemetry** (section 2.5): the collector receives OTLP on all interfaces, protects itself first (`memory_limiter`), batches, and splits traces to Tempo and metrics to a Prometheus endpoint.
6. **Logs join to traces by id, and hold nothing sensitive** (section 2.6): JSON lines through `otelx.LogHandler`, with method, route, status, and timing only.

### 5.2 The code, annotated

**Bounds and the cap** are constants copied from `metrics.yaml`:

```go
// Bucket bounds of otel/metrics.yaml, in seconds (tokens for usage).
var (
	obsTTFTBounds = []float64{0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0}
	// ... TPOT, end-to-end, token usage, HTTP duration
)

const obsCap = 64 // values per capped label (metrics.yaml)
```

**A histogram** stores per-bucket (non-cumulative) counts per label set, and accumulates them only when written. An observation lands in the first bucket whose bound is at least the value; a value above every bound only increments the total, which becomes the `+Inf` bucket:

```go
func (h *obsHist) observe(labels string, v float64) {
	c, ok := h.counts[labels]
	if !ok {
		c = make([]uint64, len(h.bounds))
		h.counts[labels] = c
	}
	for i, b := range h.bounds {
		if v <= b { // "le": less or equal
			c[i]++
			break // one bucket per observation; cumulation happens at scrape
		}
	}
	h.sums[labels] += v
	h.totals[labels]++ // the +Inf bucket and _count
}

func (h *obsHist) write(sb *strings.Builder, name string) {
	fmt.Fprintf(sb, "# TYPE %s histogram\n", name)
	for _, labels := range obsSorted(h.totals) { // sorted: a stable, diffable scrape
		// ... sep is "," when labels is non-empty
		var cum uint64
		for i, b := range h.bounds {
			cum += h.counts[labels][i] // C_i = sum of buckets up to b_i (section 2.1)
			fmt.Fprintf(sb, "%s_bucket{%s%sle=\"%s\"} %d\n", name, labels, sep, strconv.FormatFloat(b, 'f', -1, 64), cum)
		}
		fmt.Fprintf(sb, "%s_bucket{%s%sle=\"+Inf\"} %d\n", name, labels, sep, h.totals[labels])
		fmt.Fprintf(sb, "%s_sum{%s} %g\n", name, labels, h.sums[labels])
		fmt.Fprintf(sb, "%s_count{%s} %d\n", name, labels, h.totals[labels])
	}
}
```

`strconv.FormatFloat(b, 'f', -1, 64)` prints `0.5` and `1`, the shortest exact form, so a rule reading `le="0.5"` matches the bucket string byte for byte. With section 3's four requests this writes $C_0.5 = 3$ and `+Inf` = 4.

**The cap** keeps the first 64 distinct values it sees, then folds everything new into `_other`:

```go
func obsCapped(seen map[string]bool, v string) string {
	if seen[v] {
		return v // already admitted: keep recording under its own name
	}
	if len(seen) >= obsCap {
		return "_other" // the 65th and later values share one series
	}
	seen[v] = true
	return v
}
```

That bounds TTFT to $65 \times 19$ series per other-label combination, however many names a client invents (section 3).

**Recording a request.** `obsMetrics.observe` takes the lock once, caps every client-controlled label, then fills the instruments the request qualifies for:

```go
func (m *obsMetrics) observe(method, route, model string, w *obsWriter, start, end time.Time) {
	m.mu.Lock()
	defer m.mu.Unlock()
	code := w.status()
	codeS := obsCapped(m.codes, strconv.Itoa(code))
	routeL := obsCapped(m.routes, route)
	// ... http duration and tl_gateway_requests_total for every request; 429s count as rejections
	op := obsOperation(route)
	if op == "" || method != http.MethodPost {
		return // only GenAI operations get the gen_ai_* histograms
	}
	modelL := obsCapped(m.models, model) // the client-chosen label, capped
	// ... end-to-end duration, with error_type for 4xx and 5xx
	if code != http.StatusOK || op == "embeddings" || w.first.IsZero() {
		return // no first token: TTFT and TPOT are undefined, not zero
	}
	lat := obsLabels("gen_ai_operation_name", op, "gen_ai_request_model", modelL, "tl_engine_role", "gateway")
	m.ttft.observe(lat, w.first.Sub(start).Seconds())
	if w.events >= 2 { // TPOT needs at least two tokens: (end - first) / (events - 1)
		m.tpot.observe(lat, end.Sub(w.first).Seconds()/float64(w.events-1))
	}
	// ... output token usage from the event count
}
```

`obsWriter` is the gateway's own `ResponseWriter` wrapper: it notes the time of the first non-empty write (the first token) and counts SSE `data:` events minus `[DONE]`, and like `otelx`'s recorder it forwards `Flush` and `Unwrap`.

**Wiring and logs.** `newObsTelemetry` builds the provider with `otelx.Setup`, then a JSON handler whose keys match `semconv.md` (`ts` in RFC 3339 UTC, lower-case `level`), wrapped in `otelx.LogHandler` and pinned with `service`:

```go
	return &obsTelemetry{
		service: service,
		tp:      tp,
		log:     slog.New(otelx.LogHandler(h)).With("service", service), // trace_id, span_id on every line in a span
		m:       newObsMetrics(),
	}, nil
```

`wrap` puts `otelx.Middleware` (named by `obsRoute`, which maps `/v1/models/smol-135m` to `/v1/models/{model}`) around the API handler, reads only the `model` field from the body (and restores the body for the proxy), and logs one line per request:

```go
			// No prompt, completion, or key in logs: method, route, status, timing only.
			t.log.InfoContext(r.Context(), "request", "method", r.Method, "route", route,
				"status", rec.status(), "duration_ms", time.Since(start).Milliseconds())
```

`InfoContext` with the request's context is what lets `LogHandler` find the SERVER span. `withMetrics` answers `/metrics` on the health handler and probes the upstream's `/healthz` to set `tl_gateway_workers`.

**The PodMonitor** (the engine's is the same with `component: engine`):

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: forge-gateway
  namespace: observability
  labels:
    release: observability          # the stack's monitor selector (contracts/helm/observability.md)
spec:
  namespaceSelector:
    matchNames: [forge]
  selector:
    matchLabels:
      app.kubernetes.io/component: gateway   # shared by every release of the chart
  podMetricsEndpoints:
    - port: health                  # the named container port, 9464
      path: /metrics
      interval: 10s                 # two samples inside the drill's 25 s window
      relabelings:
        - targetLabel: job          # without this, job = observability/forge-gateway
          replacement: forge-gateway
```

**The collector**, as values for the umbrella chart's `otel-collector` subchart:

```yaml
  config:
    receivers:
      otlp:
        protocols:
          grpc: {endpoint: "0.0.0.0:4317"}   # all interfaces: other pods must reach it
          http: {endpoint: "0.0.0.0:4318"}
    processors:
      # First in every pipeline: refuse data before the pod is OOM-killed.
      memory_limiter:
        check_interval: 1s
        limit_percentage: 80
        spike_limit_percentage: 25
      batch:
        timeout: 2s
        send_batch_size: 1024
    exporters:
      otlp/tempo:
        endpoint: tempo.observability:4317
        tls: {insecure: true}
      # The Python subprocesses' metrics (they cannot be scraped, D9), exposed
      # for Prometheus with the names of otel/metrics.yaml.
      prometheus:
        endpoint: "0.0.0.0:8889"
        resource_to_telemetry_conversion: {enabled: false}
    # ...
    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [memory_limiter, batch]
          exporters: [otlp/tempo]
        metrics:
          receivers: [otlp]
          processors: [memory_limiter, batch]
          exporters: [prometheus]
```

`resource_to_telemetry_conversion` stays off so resource attributes do not become labels on every pushed series (more cardinality). The chart's `ports.prom-exporter` opens 8889 so Prometheus can scrape the exporter.

Full solution: [`telemetry.go`](../../../../course/solutions/entry/go/cmd/gateway/telemetry.go), [`monitors.yaml`](../../../../course/solutions/entry/deploy/observability/monitors.yaml), [`collector.yaml`](../../../../course/solutions/entry/deploy/observability/collector.yaml).

### 5.3 Why this solution

- **Cost.** An observation is $O(k)$ for $k$ bounds (at most 16) under one mutex; a scrape is $O(S \cdot k)$ for $S$ live label sets, and $S$ is capped. Memory is bounded by the caps, not by traffic.
- **Why hand-written exposition instead of `client_golang`?** Every byte of the format is visible and checked against `metrics.yaml`; the library is the production choice (Going further) and adds registries and exemplars this module does not need.
- **Why PodMonitors and not ServiceMonitors?** A ServiceMonitor needs a metrics port on each Service and names the job after the Service; per-role engine releases would each need one. PodMonitors scrape the existing `health` port of every pod, and one relabeling sets the job.
- **Why a cap of first-come values instead of an allow-list?** The model list changes with deployments; an allow-list would need a config change per model. The cap's cost is that which names keep their own series depends on arrival order since the process started, which is acceptable for a guard whose job is to bound memory.
- **Why logs to stdout and no log backend?** `kubectl logs` already collects stdout; JSON with ids makes grep an exact join. Loki is an upgrade, not a prerequisite.

### 5.4 Explain it back

1. Why can no rule compute "fraction of TTFT within 500 ms" if the engine's bounds are 0.4 and 0.8?
   <details><summary>Answer</summary>Buckets only say how many observations are at most each bound. Between 0.4 and 0.8 the counters cannot tell 0.45 from 0.75, so the count at most 0.5 is not determined by any combination of them. Only a bound at exactly 0.5 makes the fraction an exact division.</details>
2. Why does the histogram store non-cumulative counts and cumulate in `write`?
   <details><summary>Answer</summary>An observation then touches one bucket, not every bucket above it, and the cumulative series Prometheus needs (non-decreasing in `le`) is produced in one pass at scrape time. Both give the same exposition; this one makes recording cheaper.</details>
3. What happens to Prometheus if the gateway records the model label uncapped and a client sends a million distinct model names?
   <details><summary>Answer</summary>Each name mints about 19 TTFT series plus the end-to-end and usage series. Prometheus holds every active series in memory, so it grows with the attack until it is OOM-killed and all monitoring stops. With the cap, at most 65 values exist per label.</details>
4. Why does the PodMonitor need a `job` relabeling?
   <details><summary>Answer</summary>A PodMonitor's default job is `<namespace>/<monitor name>` (`observability/forge-gateway`). The SLO rules and dashboards select `job="forge-gateway"`, so without the relabeling every rule evaluates to no data and alerts never fire.</details>
5. Why is `memory_limiter` first in each collector pipeline?
   <details><summary>Answer</summary>It must refuse data before anything else buffers it. Placed after `batch`, a burst would already be queued in memory when the limiter sees it, and the pod would be OOM-killed, losing all telemetry instead of some.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. A monitor whose selector matches no pod (selecting `app.kubernetes.io/name: <system>-engine` misses the `-decode` release), or the wrong port | Prometheus shows no target; graphs are empty, not zero | selects `app.kubernetes.io/component`, shared by every release, on the named `health` port |
| 2. A PodMonitor without a job relabeling (or `jobLabel`), or the default 30 s interval | series under `job="observability/forge-gateway"`, so every SLO rule is silent; drill windows hold one sample | a `targetLabel: job` relabeling and `interval: 10s` |
| 3. The model name (or a user id) as an uncapped label | Prometheus memory grows with traffic until it is OOM-killed | every client-controlled label goes through `obsCapped` |
| 4. Logging the request body "for debugging" | prompts and keys in `kubectl logs`, retained for weeks | `wrap` logs method, route, status, and duration only |
| 5. Collector OTLP bound to `localhost:4317`, no `memory_limiter`, or traces exported to the debug exporter only | services cannot reach it; the collector OOMs under a burst; Tempo stays empty | receivers on `0.0.0.0`, `memory_limiter` first, `otlp/tempo` exporter |
| 6. Custom bucket bounds ("these fit our latencies better") | SLO rules reading `le="0.5"` find nothing | the bounds are copied verbatim from `metrics.yaml` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `obs.01` | `LogHandler` stamps the ids; `Setup` and the SERVER span give each log line its span |
| Back | `dep.03` | the charts whose pods and container ports the monitors select |
| Back | `L10.7` | the engine's serve loop records the TTFT, TPOT, duration, and HTTP histograms this module scrapes, and writes the JSON log line with the trace id |
| Forward | `obs.03` | SLIs over the TTFT and TPOT histograms and the 5xx ratio, selected by `job` |
| Forward | `obs.04` | heatmaps of the same buckets, KV usage, per-tenant usage |
| Forward | `ops.01` | the drill's detection and resolution queries read these series |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| hand-written exposition | `prometheus/client_golang` or the OTel Prometheus exporter | registries, collectors, exemplars | [client_golang](https://github.com/prometheus/client_golang) |
| classic histograms with fixed bounds | native histograms | exponential buckets, no bounds to choose, far fewer series | [Native histograms](https://prometheus.io/docs/specs/native_histograms/) |
| `kubectl logs \| grep` | Loki or the OTel log pipeline | indexed search, retention, log-to-trace links in Grafana | [Loki](https://grafana.com/docs/loki/latest/) |
| PodMonitors per component | the collector's Prometheus receiver or target allocator | one scraper, relabelling in the pipeline | Collector `prometheusreceiver` |
| exemplars: not used | trace exemplars on histogram buckets | jump from a slow bucket straight to a trace | [Exemplars](https://prometheus.io/docs/prometheus/latest/feature_flags/#exemplars-storage) |
