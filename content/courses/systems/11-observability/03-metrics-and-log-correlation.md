# Metrics and log correlation

Traces, built in [Tracing for the serving path](02-tracing-for-the-serving-path.md), explain one request. They do not say whether TTFT is getting worse for everyone, how many KV blocks are left, or which tenant is flooding the gateway. That takes **metrics**: counts and distributions aggregated over all requests, cheap enough to keep for every request. The engine already serves a few gauges on its health port, from [tl-serve: OpenAI-compatible HTTP + SSE on tokio/hyper](../../ai-engineering/23-model-serving/06-openai-server-on-tokio.md) in *AI Engineering*. The gateway serves none, the histograms the SLOs of the next section need do not exist yet, and nothing in the cluster scrapes either.

That gap matters at three in the morning. When the drill in [A decode worker dies mid-stream](../12-incident-response-and-chaos/02-kill-decode.md) kills a decode pod, the page fires from these series, and the person paged starts from a dashboard, then needs the log lines of one slow request. This section makes all of that possible: contract metrics served by the gateway and the engine, Prometheus scraping them on kind, an OpenTelemetry Collector as the one door for pushed telemetry, and JSON logs that join to traces by id.

It builds on the `go/otelx` kit (`LogHandler`, `Setup`, the SERVER span), on the charts from [Helm charts and the observability stack](../10-containers-kubernetes/04-helm-charts-and-observability-stack.md) in this book whose pods are scraped, and on the engine's serve loop from [Serving metrics and tracing](../../ai-engineering/23-model-serving/08-serving-metrics-and-tracing.md), which records the GenAI and HTTP histograms and logs each request in its span. The other code it scrapes is the gateway's [server skeleton](../../ai-engineering/25-ai-gateway/02-server-skeleton.md). The solution is the gateway's telemetry in [`go/cmd/gateway/telemetry.go`](../../../../course/solutions/entry/go/cmd/gateway/telemetry.go) (`newObsTelemetry`, `wrap`, `withMetrics`, `obsHist`, `obsCapped`, `obsMetrics.observe`, `exposition`), wired in [`main.go`](../../../../course/solutions/entry/go/cmd/gateway/main.go), plus [`deploy/observability/monitors.yaml`](../../../../course/solutions/entry/deploy/observability/monitors.yaml) (two PodMonitors) and [`deploy/observability/collector.yaml`](../../../../course/solutions/entry/deploy/observability/collector.yaml) (the collector). The contracts are [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml) (names, types, buckets, labels), [`otel/semconv.md`](../../../../course/contracts/otel/semconv.md) (log fields), and [`helm/observability.md`](../../../../course/contracts/helm/observability.md) (the selector label, the collector Service).

> **Learning objectives**
> - Explain why a histogram's cumulative bucket counters give an exact SLO fraction only at a bucket bound, and compute one by hand.
> - Bound the series a client can create with capped labels, and compute the series count with and without the cap.
> - Configure PodMonitors that Prometheus actually selects, with the `job` names and scrape interval the rules need.
> - Build an OpenTelemetry Collector pipeline that receives OTLP, protects itself, and routes traces and metrics.
> - Correlate logs with a trace using JSON lines that carry `trace_id` and never carry a prompt or key.

## Counters, gauges, and histograms

Three kinds of metric cover everything this system measures:

| Symbol | Meaning | Type |
|---|---|---|
| $c(t)$ | a **counter**: a total that only grows (requests served) | non-negative, resets to 0 on restart |
| $g(t)$ | a **gauge**: a value that goes up and down (free KV blocks) | real |
| $b_1 < \dots < b_k$ | the bucket bounds of a histogram (`le`, "less or equal") | seconds, from `metrics.yaml` |
| $C_i(t)$ | requests observed so far with value $\le b_i$ | counter; $C_{+\infty}$ is the total |
| $\Delta C_i$ | the increase of $C_i$ over a window ($\text{rate} \times$ window) | non-negative |

A **histogram** is the counters $C_1 \le C_2 \le \dots \le C_{+\infty}$ plus a sum. They are **cumulative**: an observation of 0.07 s increments every $C_i$ with $b_i \ge 0.07$. Over a window, the fraction of requests at most $b_i$ is

$$\text{frac}(b_i) = \frac{\Delta C_i}{\Delta C_{+\infty}}$$

which is exact **only at a bound**. With the contract's bounds, "fraction of requests at most 500 ms" is one exact division; with other bounds it cannot be computed at all. That is why `metrics.yaml` fixes the bounds and why a correct service serves them exactly: an engine with its own bounds silently breaks every SLO rule that reads `le="0.5"`.

## The exposition format and names

Prometheus reads those counters over HTTP. Every service answers `GET /metrics` on its health port with Prometheus text:

```
# TYPE gen_ai_server_time_to_first_token_seconds histogram
gen_ai_server_time_to_first_token_seconds_bucket{gen_ai_operation_name="chat",gen_ai_request_model="smol-135m",tl_engine_role="gateway",le="0.5"} 3
...
gen_ai_server_time_to_first_token_seconds_count{gen_ai_operation_name="chat",gen_ai_request_model="smol-135m",tl_engine_role="gateway"} 4
```

OTel instrument names become Prometheus names by replacing `.` with `_` and adding the unit (`_seconds`); counters end in `_total`. The `prometheus` column of `metrics.yaml` is the exact result, and that is the name to serve.

## Labels and cardinality

The lines above carry labels, and labels are where metrics get expensive. Every distinct combination of label values is its own **series**, stored and scanned separately. A histogram with 16 bounds is 19 series per combination: 16 buckets, the `+Inf` bucket, `_sum`, and `_count`.

A label taking client input, such as the `model` field of a request, lets a client mint series: 80 random model names times 19 is 1 520 series for TTFT alone. `metrics.yaml` therefore marks such labels `capped`: a process keeps the first 64 distinct values and records every later one as `_other`. Labels with `values` take only those.

## How Prometheus finds the pods

A service that exposes metrics is useless until something scrapes it. kube-prometheus-stack runs the Prometheus Operator, which turns **ServiceMonitor** objects (scrape the endpoints behind a Service port) and **PodMonitor** objects (scrape a named container port of matching pods) into scrape config, but only the ones carrying the stack's selector label `release: observability`.

Each target's series get a `job` label, and SLO rules and dashboards select by it. A ServiceMonitor names the job after the Service, a PodMonitor after `<namespace>/<monitor name>`, unless `jobLabel` names a label to take it from or a relabeling (`targetLabel: job`) sets it. The scrape interval decides the shortest window you can compute a rate over: `rate()` needs two samples, so the drill profile's 25 s window needs a scrape every 10 s or faster.

## The collector

Not all telemetry can be scraped. Services export traces as OTLP to one address (`[otel].endpoint`), the **OpenTelemetry Collector**, which forwards them to Tempo. Python subprocesses (training, corpus stages) cannot be scraped either, because they live seconds to hours and run outside any Service (design decision D9). So they **push** OTLP metrics to the collector, whose `prometheus` exporter exposes them for Prometheus to scrape.

The collector is thus the one door for telemetry: OTLP in on 4317 and 4318, traces out to Tempo, pushed Python metrics out through its Prometheus exporter. A collector pipeline is `receivers -> processors -> exporters`. `memory_limiter` goes first, so that an overload drops data instead of the pod, and `batch` comes before export.

## Logs you can join to a trace

The last signal is logs, and here there is deliberately no log backend: log-to-trace correlation is `kubectl logs | grep <trace_id>`. That works only if every line is JSON with `trace_id`. Each line a service writes is one JSON object (`semconv.md`, Logs): `ts`, `level`, `msg`, `service`, and, inside a request, the `trace_id` and `span_id` of the current span. `otelx.LogHandler` adds the two ids to every record logged with the request's context.

The prompt, the completion, and the API key never appear. Logs are retained longer and read by more people than any request.

## Working it by hand

Four requests reach the gateway, with first-token times 0.03 s, 0.07 s, 0.30 s, and 0.60 s. The TTFT bounds of `metrics.yaml` around them are 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75. The cumulative counters come out as follows:

| `le` | requests $\le$ `le` | $C_i$ |
|---|---|---|
| 0.02 | none | 0 |
| 0.04 | 0.03 | 1 |
| 0.06 | 0.03 | 1 |
| 0.08 | 0.03, 0.07 | 2 |
| 0.1, 0.25 | 0.03, 0.07 | 2 |
| 0.5 | 0.03, 0.07, 0.30 | 3 |
| 0.75 and up, `+Inf` | all four | 4 |

The fraction within the 500 ms SLO threshold is $C_{0.5} / C_{+\infty} = 3/4 = 0.75$, exact. Had the engine used bounds 0.4 and 0.8, the 0.30 s and 0.60 s requests would fall in buckets that straddle 0.5, and no division of counters would give the answer.

Now the cardinality. After 80 requests with model names `model-000` to `model-079`, an uncapped gateway serves 81 values of `gen_ai_request_model` (the 80 plus `tracer`): $81 \times 19 = 1\,539$ TTFT series. Capped, it serves 64 named values plus `_other`: at most $65 \times 19 = 1\,235$, and that ceiling holds for a million names.

Finally, correlation. One request carries `traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01`. The gateway logs:

```json
{"ts":"2026-10-09T10:00:01Z","level":"info","msg":"proxied","service":"forge-gateway","status":200,"trace_id":"4bf92f3577b34da6a3ce929d0e0e4736","span_id":"a1b2c3d4e5f60718"}
```

`kubectl -n forge logs deploy/forge-gateway | grep 4bf92f3577b34da6a3ce929d0e0e4736` finds it. The same grep on the engine finds the engine's lines, whose `span_id` is a different span of the same trace.

## What the setup must provide

Four artifacts make up the work. The gateway and every engine serve `GET /metrics` on the health port, in Prometheus text, with every instrument of `metrics.yaml` that its `emitted_by` names. The exceptions are instruments added later: `tl.engine.spec_accept_rate` with [Speculative decoding in the engine](../../ai-engineering/23-model-serving/09-speculative-decoding-in-the-engine.md), `tl.kv.transfer.*` in disaggregated roles, and `tl.gateway.policy.denials` with [Usage policy enforcement](../../ai-engineering/25-ai-gateway/06-usage-policy-enforcement.md). Names, `# TYPE`, bounds, label keys, and enumerated values must be exact, and capped labels must hold at most 64 values plus `_other`.

Logs are one JSON object per line on stdout: `ts`, `level`, `msg`, `service`, and `trace_id`, `span_id` inside a request; no prompt, no key.

`deploy/observability/monitors.yaml` holds a PodMonitor or ServiceMonitor per component, labelled `release: observability`, selecting every pod the charts render (every engine role release too), on the 9464 port, path `/metrics`, with an `interval` of 10 s or less and `job` set to `<system>-gateway` or `<system>-engine`.

`deploy/observability/collector.yaml` holds the collector config, under `otel-collector.config:` (values for the umbrella chart), `config:` (the chart alone), or at the top (raw). It receives OTLP on `0.0.0.0:4317` and `:4318`, has a `traces` pipeline to Tempo and a `metrics` pipeline to the `prometheus` exporter, and puts `memory_limiter` first and `batch` in each.

The solution uses PodMonitors that select pods by `app.kubernetes.io/component` (shared by every release of a chart, including the engine's per-role releases `<system>-engine-prefill` and `-decode`) and set `job` with a relabeling, so no Service needs a metrics port. Once built, it is used like this:

```bash
curl -s 127.0.0.1:<health_port>/metrics | grep '^gen_ai_server_time_to_first_token_seconds_bucket'
# on kind: apply the monitors, install the collector
kubectl apply -f deploy/observability/monitors.yaml
helm upgrade --install observability deploy/observability -n observability \
  -f deploy/observability/values.yaml -f deploy/observability/collector.yaml   # the umbrella chart
```

## Building it step by step

Six observations shape the solution. The SLO must be an exact division of counters, so the histograms must use the contract's bounds verbatim, and the solution copies them into constants rather than choosing its own.

A counter-only representation is the cheapest correct one. Each observation increments one bucket counter, a sum, and a total under one mutex; the cumulative form Prometheus wants is computed only at scrape time. There is no background goroutine and no I/O on the request path.

Series must be bounded by the system, not by clients. Every label whose value comes from the request (model, route, status code, tenant) passes through a cap of 64 values per process.

Prometheus must find every pod, under the job name the rules use. Scraping pods directly (PodMonitor) makes every replica and every role release its own target; selecting by `app.kubernetes.io/component` covers all releases of a chart; a relabeling fixes `job`; and 10 s keeps two samples in a 25 s window.

There is one door for pushed telemetry. The collector receives OTLP on all interfaces, protects itself first (`memory_limiter`), batches, and splits traces to Tempo and metrics to a Prometheus endpoint.

Logs join to traces by id and hold nothing sensitive: JSON lines through `otelx.LogHandler`, with method, route, status, and timing only.

### Bounds and the cap

Both are constants copied from `metrics.yaml`:

```go
// Bucket bounds of otel/metrics.yaml, in seconds (tokens for usage).
var (
	obsTTFTBounds = []float64{0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0}
	// ... TPOT, end-to-end, token usage, HTTP duration
)

const obsCap = 64 // values per capped label (metrics.yaml)
```

### The histogram

A histogram stores per-bucket (non-cumulative) counts per label set, and accumulates them only when written. An observation lands in the first bucket whose bound is at least the value; a value above every bound only increments the total, which becomes the `+Inf` bucket:

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
			cum += h.counts[labels][i] // C_i = sum of buckets up to b_i
			fmt.Fprintf(sb, "%s_bucket{%s%sle=\"%s\"} %d\n", name, labels, sep, strconv.FormatFloat(b, 'f', -1, 64), cum)
		}
		fmt.Fprintf(sb, "%s_bucket{%s%sle=\"+Inf\"} %d\n", name, labels, sep, h.totals[labels])
		fmt.Fprintf(sb, "%s_sum{%s} %g\n", name, labels, h.sums[labels])
		fmt.Fprintf(sb, "%s_count{%s} %d\n", name, labels, h.totals[labels])
	}
}
```

`strconv.FormatFloat(b, 'f', -1, 64)` prints `0.5` and `1`, the shortest exact form, so a rule reading `le="0.5"` matches the bucket string byte for byte. With the four requests of the worked example this writes $C_{0.5} = 3$ and `+Inf` = 4.

### The cap

The cap keeps the first 64 distinct values it sees, then folds everything new into `_other`:

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

That bounds TTFT to $65 \times 19$ series per other-label combination, however many names a client invents.

### Recording a request

`obsMetrics.observe` takes the lock once, caps every client-controlled label, then fills the instruments the request qualifies for:

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

`obsWriter` is the gateway's own `ResponseWriter` wrapper. It notes the time of the first non-empty write (the first token) and counts SSE `data:` events minus `[DONE]`, and like `otelx`'s recorder it forwards `Flush` and `Unwrap`.

### Wiring and logs

`newObsTelemetry` builds the provider with `otelx.Setup`, then a JSON handler whose keys match `semconv.md` (`ts` in RFC 3339 UTC, lower-case `level`), wrapped in `otelx.LogHandler` and pinned with `service`:

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

### The PodMonitor

The engine's PodMonitor is the same as the gateway's, with `component: engine`:

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

### The collector

The collector is configured as values for the umbrella chart's `otel-collector` subchart:

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

`resource_to_telemetry_conversion` stays off so that resource attributes do not become labels on every pushed series, which would add cardinality. The chart's `ports.prom-exporter` opens 8889 so Prometheus can scrape the exporter.

The full solution is in [`telemetry.go`](../../../../course/solutions/entry/go/cmd/gateway/telemetry.go), [`monitors.yaml`](../../../../course/solutions/entry/deploy/observability/monitors.yaml), and [`collector.yaml`](../../../../course/solutions/entry/deploy/observability/collector.yaml).

### Why this design

An observation is $O(k)$ for $k$ bounds (at most 16) under one mutex; a scrape is $O(S \cdot k)$ for $S$ live label sets, and $S$ is capped. Memory is bounded by the caps, not by traffic.

The exposition is hand-written rather than built on `client_golang` so that every byte of the format is visible and checked against `metrics.yaml`. The library is the production choice and adds registries and exemplars that are not needed here.

PodMonitors win over ServiceMonitors because a ServiceMonitor needs a metrics port on each Service and names the job after the Service; per-role engine releases would each need one. PodMonitors scrape the existing `health` port of every pod, and one relabeling sets the job.

The cap admits first-come values instead of using an allow-list because the model list changes with deployments, and an allow-list would need a config change per model. The cost is that which names keep their own series depends on arrival order since the process started, which is acceptable for a guard whose job is to bound memory.

Logs go to stdout with no log backend because `kubectl logs` already collects stdout, and JSON with ids makes grep an exact join. Loki is an upgrade, not a prerequisite.

## What can go wrong

The checks for a correct setup start the engine and the gateway as local processes, send each service six chat completions, one traced request whose prompt holds a canary string, and 80 requests with random model names, then scrape both health ports and read both logs. Every rendered gateway workload, and the engine rendered as unified, prefill, and decode releases, must be selected by a monitor on port 9464, path `/metrics`: no target, no series. Every monitor must carry `release: observability`, because the operator ignores the rest. Job names must be `<system>-gateway` and `<system>-engine`, with an interval of 10 s or less, because the SLO rules and dashboard panels select by job and the drill windows need two samples. The collector must meet its four requirements, so that traces reach Tempo and Python metrics reach Prometheus. Both scrapes must serve the contract's names, types, bounds, labels, and values, since every rule and panel downstream reads them. After 80 model names, each capped label must hold at most 64 values plus `_other`, so that an attacker cannot exhaust Prometheus. Both services must log JSON with the traced request's `trace_id` and a `span_id`, so the grep from the worked example works. And the canary prompt and the API key must appear in no log line. On kind, the same runs end to end: `up` must be 1 for every target of both jobs, and after a request through the gateway NodePort, `kubectl logs` must find its trace id, which is what the decode-kill drill relies on.

The usual mistakes, and how the solution avoids each:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| A monitor whose selector matches no pod (selecting `app.kubernetes.io/name: <system>-engine` misses the `-decode` release), or the wrong port | Prometheus shows no target; graphs are empty, not zero | selects `app.kubernetes.io/component`, shared by every release, on the named `health` port |
| A PodMonitor without a job relabeling (or `jobLabel`), or the default 30 s interval | series under `job="observability/forge-gateway"`, so every SLO rule is silent; drill windows hold one sample | a `targetLabel: job` relabeling and `interval: 10s` |
| The model name (or a user id) as an uncapped label | Prometheus memory grows with traffic until it is OOM-killed | every client-controlled label goes through `obsCapped` |
| Logging the request body "for debugging" | prompts and keys in `kubectl logs`, retained for weeks | `wrap` logs method, route, status, and duration only |
| Collector OTLP bound to `localhost:4317`, no `memory_limiter`, or traces exported to the debug exporter only | services cannot reach it; the collector OOMs under a burst; Tempo stays empty | receivers on `0.0.0.0`, `memory_limiter` first, `otlp/tempo` exporter |
| Custom bucket bounds ("these fit our latencies better") | SLO rules reading `le="0.5"` find nothing | the bounds are copied verbatim from `metrics.yaml` |

## Where this leads

Looking back, `LogHandler` from [Tracing for the serving path](02-tracing-for-the-serving-path.md) stamps the ids, and `Setup` and the SERVER span give each log line its span. The charts of [Helm charts and the observability stack](../10-containers-kubernetes/04-helm-charts-and-observability-stack.md) supply the pods and container ports the monitors select. The engine's serve loop from [Serving metrics and tracing](../../ai-engineering/23-model-serving/08-serving-metrics-and-tracing.md) records the TTFT, TPOT, duration, and HTTP histograms scraped here, and writes the JSON log line with the trace id.

Looking forward, [SLOs and multi-window burn-rate alerts](04-slos-and-burn-rate-alerts.md) defines SLIs over the TTFT and TPOT histograms and the 5xx ratio, selected by `job`. [Dashboards as code for the serving path](05-dashboards-as-code.md) draws heatmaps of the same buckets, KV usage, and per-tenant usage. And the detection and resolution queries of [A decode worker dies mid-stream](../12-incident-response-and-chaos/02-kill-decode.md) read these series.

Each simplification has a production equivalent. Hand-written exposition gives way to `prometheus/client_golang` or the OTel Prometheus exporter, with registries, collectors, and exemplars. Classic histograms with fixed bounds give way to native histograms: exponential buckets, no bounds to choose, far fewer series. `kubectl logs | grep` gives way to Loki or the OTel log pipeline, with indexed search, retention, and log-to-trace links in Grafana. PodMonitors per component can be replaced by the collector's Prometheus receiver (`prometheusreceiver`) or target allocator: one scraper, with relabelling in the pipeline. And exemplars, unused here, attach trace ids to histogram buckets so that you can jump from a slow bucket straight to a trace.

### Further reading

- [Prometheus exposition format](https://prometheus.io/docs/instrumenting/exposition_formats/) (free)
- [Histograms and summaries](https://prometheus.io/docs/practices/histograms/) (free)
- [client_golang](https://github.com/prometheus/client_golang)
- [Native histograms](https://prometheus.io/docs/specs/native_histograms/)
- [Loki](https://grafana.com/docs/loki/latest/)
- [Exemplars](https://prometheus.io/docs/prometheus/latest/feature_flags/#exemplars-storage)
- Charity Majors, Liz Fong-Jones, and George Miranda, *Observability Engineering* (O'Reilly), chapters 8 and 9

## Key terms

- **counter**: a metric whose value only grows, resetting to 0 on restart.
- **gauge**: a metric whose value goes up and down.
- **histogram**: a set of cumulative bucket counters, one per bound `le`, plus a sum and a count.
- **series**: one distinct combination of metric name and label values, stored separately.
- **capped label**: a label that keeps its first 64 distinct values per process and folds later ones into `_other`.
- **PodMonitor**: a Prometheus Operator object that scrapes a named container port of matching pods.
- **OpenTelemetry Collector**: the service that receives OTLP, processes it, and exports traces and metrics to their backends.

## Summary

- A histogram is a set of cumulative counters, one per bucket bound `le`; with the contract's bounds, "fraction of requests at most 500 ms" is one exact division, and with other bounds it cannot be computed at all.
- Prometheus finds targets only through ServiceMonitor or PodMonitor objects the stack selects (`release: observability`), and names the series' `job` from them; SLO rules and dashboards select by that `job`.
- A label that copies client input multiplies series; capped labels keep 64 values and fold the rest into `_other`.
- With no log backend, log-to-trace correlation is `kubectl logs | grep <trace_id>`, which works only if every line is JSON with `trace_id`, and never holds a prompt or a key.
- The collector is the one door for telemetry: OTLP in on 4317 and 4318, traces to Tempo, pushed Python metrics out through its Prometheus exporter, with `memory_limiter` first and `batch` in every pipeline.

## Review questions

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
