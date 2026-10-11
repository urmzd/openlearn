<!-- ol:module L10.7 -->
# Serving metrics, SLO histograms, OTel spans and propagation

## Overview

| | |
|---|---|
| **Module** | `L10.7` · build · Rust · Pass 7 · 6 to 9 h |
| **You build** | `rust/crates/tl-serve/src/metrics.rs`: counters, gauges, and cumulative histograms, the Prometheus text exposition, and `EngineMetrics` with every engine instrument of the contract · `rust/crates/tl-serve/src/telemetry.rs`: W3C trace context, the engine's span tree, OTLP/HTTP JSON export, JSON log lines with the trace · `rust/crates/tl-serve/src/server.rs`, taken over from `L10.5`: the serve loop records every request in `EngineMetrics` and ends it with one JSON log line and its span tree |
| **Contract** | [`otel/metrics.yaml`](../../../course/contracts/otel/metrics.yaml) (names, types, buckets, labels) · [`otel/semconv.md`](../../../course/contracts/otel/semconv.md) (spans, attributes, propagation, logs) · [`otel/slo.schema.json`](../../../course/contracts/otel/slo.schema.json) (what the histograms feed) |
| **Tests** | `course/tests/rust/l10_7.rs`, 17 tests (what they check: section 4); `L10.5`'s tests run as the regression of the server you take over |
| **Needs** | `L10.5` the engine loop's `EngineStats` and the `/metrics` route ([chapter](05-openai-server-on-tokio.md)) · reading: `M07.4` quantiles and intervals ([chapter](../../../math/07-probability-statistics/04-lln-clt-confidence-intervals-bootstrap.md)), `obs.00` the Pass 1 trace ([chapter](../../../systems/04-observability/00-one-trace.md)) · or `--ref-deps` |
| **Used by** | `L10.9` takes the serve loop over next and keeps recording through it; `obs.02` runs your engine and scrapes these histograms from its `/metrics`; `obs.01`, `obs.03`, `obs.04` trace and alert on them; `load.01` cross-checks its numbers; the drills grade burn-rate alerts computed from it |
| **Milestone** | `MS-L10` (and `MS-prod`, where Prometheus scrapes it) |
| **Optional depth** | [Prometheus exposition formats](https://prometheus.io/docs/instrumenting/exposition_formats/) (free); [W3C Trace Context](https://www.w3.org/TR/trace-context/) (free); [OTLP JSON encoding](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding) (free); [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) (free); *Site Reliability Engineering*, ch. 6 "Monitoring Distributed Systems" (free online) |

## Key Takeaways

- A histogram bucket `le="b"` counts observations at most `b`, so a value equal to a bound belongs to that bucket, the wire buckets are cumulative, and `+Inf` equals `_count` (`hand_example_histogram`, `histogram_count_equals_requests`).
- The names, types, buckets, and label keys are the contract's, character for character: dashboards and alerts query them (`names_and_buckets_match_the_contract`).
- TPOT is $(E2E - TTFT)/(n - 1)$, defined only for $n \ge 2$ tokens (`tpot_hand_example`).
- Counters never decrease, unbounded labels are capped, and label values are escaped; any of the three broken and the scrape, the cardinality, or `rate()` breaks (`counters_never_go_down`, `label_cap_and_allowed_values`, `label_values_are_escaped`).
- One request is one trace: the engine's SERVER span is the child of the caller's `traceparent`, its queue, prefill, and decode spans are its children, and the export is OTLP/HTTP JSON (`request_span_tree`, `otlp_json_shape`).

## How to work this chapter

```bash
ol start L10.7               # stubs metrics.rs and telemetry.rs
ol tests L10.7
ol check L10.7               # exit code is the verdict
ol check L10.7 --ref-deps    # only if L10.5 is not passing yet
ol diff  L10.7
```

Declare `pub mod metrics; pub mod telemetry;` in `tl-serve/src/lib.rs`. This module takes over `server.rs` from `L10.5` (`upgrades`), so `ol check L10.7` also runs every `L10.5` test as a regression. Your server keeps one `EngineMetrics` behind a mutex: `record_http` for every request, `record_request` when a generation ends (early errors too, with their `code` as `error_type`), `set_engine(&EngineGauges::from_stats(&engine.stats(), spec_rate))` before rendering `/metrics`. The engine thread stamps when it takes a request and when it samples each token; at the end of each request the server builds `request_spans` from those times, writes one `log_line` inside the SERVER span to stderr, and posts `otlp_json` to `$OTEL_EXPORTER_OTLP_ENDPOINT/v1/traces` on a thread of its own.

---

## 1. Why now

Your engine (`L10.5`) answers requests, and nobody can tell how well. The SLOs the course holds you to (DESIGN 2.11: TTFT p95 under 500 ms, TPOT p95 under 60 ms, 99.5% availability) are statements about distributions; the burn-rate alerts that page you in the drills are ratios of counts over windows; the gateway (`gw.05`) needs queue depth and KV usage to route. All of that comes from the numbers this module exposes. The Pass 1 tracer wrote one span by hand; from now on a request crosses the gateway, a prefill engine, a KV transfer, and a decode engine, and only a shared trace id lets you see one request's path through all four.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $b_1 < \dots < b_m$ | a histogram's bucket upper bounds (seconds) | `f64` |
| $c_j$ | observations $x$ with $b_{j-1} < x \le b_j$ ($b_0 = -\infty$, $b_{m+1} = +\infty$) | count |
| $C_j = \sum_{i \le j} c_i$ | the cumulative count served as `_bucket{le="b_j"}` | count |
| $n$ | output tokens of a request | count |
| $T_{ttft}, T_{e2e}$ | receipt to first token; receipt to the end | seconds |

### 2.1 Three kinds of metric

A **counter** only goes up (requests served, bytes moved, evictions); Prometheus differentiates it with `rate()` and reads any drop as a process restart. A **gauge** goes both ways (queue depth, free blocks). A **histogram** counts observations into fixed buckets, plus their sum and count, so any quantile can be estimated later over any window and aggregated across replicas, which a precomputed p95 cannot.

### 2.2 Histograms on the wire

A value $x$ goes into the first bucket with $x \le b_j$. The exposition serves cumulative counts $C_j$, then `le="+Inf"` with the total, then `_sum` and `_count`; so $C_{m+1}$ = `_count` always, and $C_j$ never decreases in $j$. `histogram_quantile(0.95, ...)` interpolates inside the bucket where the cumulative count crosses 95%, which is why the contract's buckets are dense around the SLO bounds (0.5 s for TTFT, 0.06 s for TPOT).

### 2.3 Labels and cardinality

Each distinct label combination is a separate series Prometheus stores forever. A label with a fixed value set (`state`, `gen_ai.operation.name`) is cheap; a label fed by users (`gen_ai.request.model`, `http.route`) is capped at 64 distinct values per process, the rest recorded as `_other`. Label values are user data: `\`, `"`, and newline are escaped as `\\`, `\"`, `\n`, or a single odd model name breaks the whole scrape.

### 2.4 The request metrics

For a request with $n$ output tokens: TTFT $= T_{ttft}$ when a token came; TPOT $= (T_{e2e} - T_{ttft})/(n-1)$ when $n \ge 2$ (the first token's time is TTFT, not TPOT); the request duration $= T_{e2e}$ always, with `error_type` set on failure. The engine gauges come from the step loop's `EngineStats`: waiting requests are the queue depth, running ones the active sequences, and free + used + cached KV blocks equal the pool.

### 2.5 Trace context

A `traceparent` header is `00-<32 hex trace id>-<16 hex parent span id>-<2 hex flags>`, lowercase, ids not all zero, version `ff` invalid, version `00` with exactly four fields. Invalid means "start a new trace", never "half trust it". The engine's SERVER span takes the trace id and records the caller's span as its parent; its children carry the trace id and the SERVER span as parent. Outgoing calls (`kv.transfer`) carry the current span as `traceparent`, in HTTP headers and gRPC metadata alike.

### 2.6 OTLP/HTTP JSON

An export is one `ExportTraceServiceRequest`: `resourceSpans[].resource.attributes` (service.name `<system>-engine` first), `scopeSpans[].spans[]` with `traceId` and `spanId` as lowercase hex, `parentSpanId` only on children, `kind` as a number (1 internal, 2 server, 3 client), times and 64-bit integers as strings, doubles as numbers, string arrays as `arrayValue`, and `status.code = 2` on error.

## 3. Worked example by hand

**A histogram** (test `hand_example_histogram`). Bounds $[0.1, 0.5, 1.0]$; observations $0.05, 0.1, 0.3, 2.0$.

1. $0.05 \le 0.1$: bucket 1. $0.1 \le 0.1$: bucket 1 too (le means at most). $0.3$: bucket 2. $2.0$: above every bound, the `+Inf` bucket.
2. $c = [2, 1, 0, 1]$, so the wire buckets are $C = [2, 3, 3, 4]$; the sum is $2.45$; the count 4.

```text
x_seconds_bucket{op="chat",le="0.1"} 2
x_seconds_bucket{op="chat",le="0.5"} 3
x_seconds_bucket{op="chat",le="1"} 3
x_seconds_bucket{op="chat",le="+Inf"} 4
x_seconds_sum{op="chat"} 2.45
x_seconds_count{op="chat"} 4
```

**TPOT** (test `tpot_hand_example`). TTFT 0.2 s, E2E 1.0 s, 5 tokens: TPOT $= 0.8/4 = 0.2$ s, counted in the `le="0.2"` bucket and not in `le="0.15"`.

**A trace** (test `traceparent_hand_example`). The W3C example `00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01`: trace id `4bf9...4736`, parent span `00f0...02b7`, flags `01` (sampled). The engine's SERVER span gets trace id `4bf9...4736`, a fresh span id, and parent `00f0...02b7`; its `engine.queue` child gets the same trace id and the SERVER span id as parent.

## 4. The interface

```rust
// rust/crates/tl-serve/src/metrics.rs
pub const TTFT_BUCKETS: &[f64]; pub const TPOT_BUCKETS: &[f64]; pub const REQUEST_BUCKETS: &[f64]; pub const HTTP_BUCKETS: &[f64];
pub struct Histogram { pub bounds: Vec<f64>, pub counts: Vec<u64>, pub sum: f64, pub count: u64 }
impl Histogram { pub fn new(bounds: &[f64]) -> Histogram; pub fn observe(&mut self, v: f64); pub fn cumulative(&self) -> Vec<u64>; }
pub struct LabelSpec { pub key: &'static str, pub values: Option<&'static [&'static str]>, pub capped: bool }
pub enum MetricError { UnknownMetric(String), Labels(String), Value { label: String, value: String }, Decrease(String), Kind(String) }
impl Registry {
    pub fn register(&mut self, name, help, kind: Kind, labels: Vec<LabelSpec>, buckets: &'static [f64]);
    pub fn inc(&mut self, name: &str, labels: &[(&str, &str)], by: f64) -> Result<(), MetricError>;
    pub fn set_total(&mut self, ...) -> Result<(), MetricError>;   // a running total; never lower
    pub fn set(&mut self, ...) -> Result<(), MetricError>;
    pub fn observe(&mut self, ...) -> Result<(), MetricError>;
    pub fn value(&self, name: &str, labels: &[(&str, &str)]) -> Option<f64>;
    pub fn render(&self) -> String;                                // Prometheus text format 0.0.4
}
pub struct RequestRecord { pub operation: String, pub model: String, pub role: String, pub ttft_s: Option<f64>, pub e2e_s: f64, pub output_tokens: usize, pub error_type: Option<String> }
pub struct EngineGauges { /* kv_free, kv_used, kv_cached, kv_evictions, queue_depth, active_sequences, batch_tokens, prefix_hit_ratio, spec_accept_rate, preemptions */ }
impl EngineGauges { pub fn from_stats(s: &tl_engine::engine::EngineStats, spec_accept_rate: f64) -> EngineGauges; }
impl EngineMetrics {
    pub fn new() -> EngineMetrics;                                 // every engine instrument of metrics.yaml
    pub fn record_request(&mut self, r: &RequestRecord) -> Result<(), MetricError>;
    pub fn record_http(&mut self, method: &str, route: &str, status: u16, seconds: f64) -> Result<(), MetricError>;
    pub fn set_engine(&mut self, g: &EngineGauges) -> Result<(), MetricError>;
    pub fn record_kv_transfer(&mut self, sent: bool, bytes: u64, blocks_sent: u64, blocks_deduped: u64) -> Result<(), MetricError>;
    pub fn render(&self) -> String;
}

// rust/crates/tl-serve/src/telemetry.rs
pub struct TraceContext { pub trace_id: [u8; 16], pub span_id: [u8; 8], pub flags: u8 }
impl TraceContext { pub fn parse(header: &str) -> Option<TraceContext>; pub fn header(&self) -> String; pub fn sampled(&self) -> bool; }
pub fn inject(ctx: &TraceContext, set: &mut dyn FnMut(&str, &str));
pub fn extract(get: &dyn Fn(&str) -> Option<String>) -> Option<TraceContext>;
pub struct IdGen;  pub enum SpanKind { Internal = 1, Server = 2, Client = 3 }
pub enum AttrValue { Str(String), Int(i64), Float(f64), Bool(bool), StrArray(Vec<String>) }
pub struct Span { /* name, kind, ids, parent, start/end ns, attributes, events, error */ }
pub fn request_spans(ids: &mut IdGen, incoming: Option<&TraceContext>, t: &RequestTiming) -> Vec<Span>;
pub fn otlp_json(resource: &[(String, AttrValue)], scope: &str, spans: &[Span]) -> String;
pub fn engine_resource(system: &str, version: &str, role: &str) -> Vec<(String, AttrValue)>;
pub fn export(endpoint: &str, body: &str, timeout: Duration) -> Result<u16, String>;
pub fn log_line(ts: &str, level: &str, msg: &str, service: &str, ctx: Option<&TraceContext>) -> String;

// rust/crates/tl-serve/src/server.rs, taken over from L10.5: its public API is unchanged
// (ServeConfig, spawn, run, ServerHandle); /metrics renders EngineMetrics, and every request
// is recorded, logged, and traced as above.
```

### What the tests check

The exposition is parsed by the test's own Prometheus text reader and compared with `course/fixtures/L10.7/engine_metrics.json`, the engine instruments extracted from `metrics.yaml`.

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_histogram` | unit | section 3's buckets and the exact six lines | you and the scraper agree on `le` |
| `names_and_buckets_match_the_contract` | conformance | every engine instrument: name, TYPE, HELP, label keys, bucket bounds; nothing extra | dashboards and SLO rules find their series |
| `tpot_hand_example` | unit | TPOT 0.2 s in `le="0.2"`; 1 token no TPOT; failures carry `error_type` | the TPOT SLO measures what it says |
| `histogram_count_equals_requests` | property | 500 seeded requests: count, sum, +Inf == count, buckets ascending | quantiles are computed from these |
| `counters_never_go_down` | fault | lower totals and negative or NaN increments refused | `rate()` and the burn-rate alerts |
| `counters_and_gauges_start_at_zero` | boundary | unlabeled series served before any event; KV gauges sum to the pool | alerts on a fresh engine |
| `gauges_from_engine_stats` | unit | waiting is the queue depth, running the active sequences | queueing alerts point at the right gauge |
| `label_cap_and_allowed_values` | boundary | 64 models then `_other`; values outside a list and missing labels refused | Prometheus memory under many tenants |
| `label_values_are_escaped` | boundary | `"`, `\`, newline escaped and parsed back | one odd model name cannot break the scrape |
| `traceparent_hand_example` | unit | the W3C example parsed and written back | the gateway's context reaches the engine |
| `traceparent_rejects_invalid_headers` | boundary | uppercase, zero ids, version ff, extra fields on 00, bad lengths | a broken header starts a new trace |
| `request_span_tree` | conformance | SERVER under the caller, three children with their intervals and attributes, token events every 32 | `obs.01`'s span tree |
| `failed_request_sets_error_status` | unit | 5xx is ERROR with `error.type`; 429 is not | trace search finds real failures |
| `otlp_json_shape` | conformance | hex ids, parent only on children, numeric kind, string times and ints, arrays, status | the collector accepts it |
| `export_posts_to_a_collector` | fault | one POST `/v1/traces`, JSON content type, the body unchanged; a dead collector errors in time | export never hangs a request |
| `propagation_and_log_correlation` | unit | `traceparent` injected and extracted; log lines carry trace and span ids; seeded ids | `grep <trace_id>` over `kubectl logs` |
| `serve_loop_records_metrics_and_spans` | conformance | a live server: TTFT, TPOT, and request duration per generation (stream or not), a failure with its `error_type`, the HTTP histogram per route and status, the KV gauges, and the exported span tree continuing the caller's trace | `obs.02` scrapes these from your engine; `obs.01` joins its trace |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| `le` treated as "less than" | values exactly on a bound land one bucket late: p95 drifts up | `hand_example_histogram` (mutant `s01`) |
| A metric name not the contract's | the SLO rules and dashboards show no data | `names_and_buckets_match_the_contract` (mutant `s02`) |
| TPOT on the TTFT buckets | no resolution around the 60 ms SLO bound | `names_and_buckets_match_the_contract` (mutant `s03`) |
| TPOT divided by $n$, not $n - 1$ | TPOT reads low; the SLO passes while users wait | `tpot_hand_example` (mutant `s04`) |
| Observations above the top bound not counted | `+Inf` < `_count`: quantiles and rates go wrong for slow requests | `histogram_count_equals_requests` (mutant `s05`) |
| A counter accepting a lower total | `rate()` reads a restart and spikes the burn rate | `counters_never_go_down` (mutant `s06`) |
| An off-by-one label cap | 65 series per label: the cap is not the contract's | `label_cap_and_allowed_values` (mutant `s07`) |
| A quote not escaped in a label value | the whole scrape fails to parse | `label_values_are_escaped` (mutant `s08`) |
| Flags written as one hex digit | downstream services reject the header; the trace splits | `traceparent_hand_example` (mutant `s09`) |
| Uppercase hex accepted | a header other services reject is propagated | `traceparent_rejects_invalid_headers` (mutant `s10`) |
| The SERVER span dropping its parent | the engine's spans float as a separate trace | `request_span_tree` (mutant `s11`) |
| Children parented to the caller | the engine's internal spans appear as siblings of its own SERVER span | `request_span_tree` (mutant `s12`) |
| 64-bit ints written as JSON numbers | OTLP JSON readers lose precision or refuse the body | `otlp_json_shape` (mutant `s13`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L10.5` | `EngineStats` from the step loop feeds the gauges; you take over its `server.rs`, whose `/metrics` route now renders the registry |
| Forward | `obs.01` | the collector receives these spans; the span tree of a request crosses gateway, prefill, transfer, decode |
| Forward | `L10.9` | takes `server.rs` over next: tool requests are recorded and traced through the same calls |
| Forward | `obs.02` | runs your engine and checks its `/metrics` (names, types, buckets, labels against `metrics.yaml`) and its JSON log lines |
| Forward | `obs.03` | the SLO burn-rate rules are written over `gen_ai_server_time_to_first_token_seconds_bucket` and friends |
| Forward | `load.01` | the load generator's client-side TTFT and TPOT are compared with these server-side numbers |

`L10.6` records `tl_kv_transfer_bytes_total` through `record_kv_transfer`; `L10.8` feeds `tl_engine_spec_accept_rate`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a hand-written registry | the `prometheus` and `opentelemetry` crates | exemplars, native histograms, push and pull from one SDK | [opentelemetry-rust](https://github.com/open-telemetry/opentelemetry-rust) |
| fixed buckets | Prometheus native histograms | exponential buckets with no bucket choice and lower error | [native histograms](https://prometheus.io/docs/specs/native_histograms/) |
| spans built at request end | `tracing` + `tracing-opentelemetry` | spans opened and closed as the code runs, with context in thread-locals | [tracing-opentelemetry](https://github.com/tokio-rs/tracing-opentelemetry) |
| OTLP/HTTP JSON per request | the OTel SDK batch processor over gRPC | batching, retries, sampling decisions in one place | [OTLP exporter spec](https://opentelemetry.io/docs/specs/otel/protocol/exporter/) |
| vLLM's metrics | vLLM `/metrics` | the same GenAI instruments plus scheduler and cache internals | [vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics.html) |
