# Serving metrics, SLO histograms, OTel spans and propagation

## Overview

| | |
|---|---|
| **Module** | `L10.7` · build · Rust · Pass 7 · 3 to 5 h reading |
| **Solution** | [`tl-serve/src/metrics.rs`](../../../../course/solutions/rust/crates/tl-serve/src/metrics.rs): `Histogram`, `Registry` (counters, gauges, cumulative histograms, the Prometheus text exposition), and `EngineMetrics` with every engine instrument of the contract · [`tl-serve/src/telemetry.rs`](../../../../course/solutions/rust/crates/tl-serve/src/telemetry.rs): `TraceContext` (W3C trace context), `request_spans`, `otlp_json`, `export`, `log_line` · [`tl-serve/src/server.rs`](../../../../course/solutions/history/L10.7/rust/crates/tl-serve/src/server.rs) (as of this module, taken over from `L10.5`; [later version](../../../../course/solutions/rust/crates/tl-serve/src/server.rs)): the serve loop records every request in `EngineMetrics` and ends it with one JSON log line and its span tree (`finish_request`, `export_spans`) |
| **Contract** | [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml) (names, types, buckets, labels) · [`otel/semconv.md`](../../../../course/contracts/otel/semconv.md) (spans, attributes, propagation, logs) · [`otel/slo.schema.json`](../../../../course/contracts/otel/slo.schema.json) (what the histograms feed) |
| **Needs** | `L10.5` the engine loop's `EngineStats` and the `/metrics` route ([chapter](05-openai-server-on-tokio.md)) · reading: `M07.4` quantiles and intervals ([chapter](../../../tracks/math/07-probability-statistics/04-lln-clt-confidence-intervals-bootstrap.md)), `obs.00` the Pass 1 trace ([chapter](../../../tracks/systems/04-observability/00-one-trace.md)) |
| **Used by** | `L10.9` takes the serve loop over next and keeps recording through it; `obs.02` runs the engine and scrapes these histograms from its `/metrics`; `obs.01`, `obs.03`, `obs.04` trace and alert on them; `load.01` cross-checks its numbers; the incident walkthroughs compute burn-rate alerts from it. In the production deployment Prometheus scrapes this endpoint |
| **Optional depth** | [Prometheus exposition formats](https://prometheus.io/docs/instrumenting/exposition_formats/) (free); [W3C Trace Context](https://www.w3.org/TR/trace-context/) (free); [OTLP JSON encoding](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding) (free); [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) (free); *Site Reliability Engineering*, ch. 6 "Monitoring Distributed Systems" (free online) |

## Key Takeaways

- A histogram bucket `le="b"` counts observations at most `b`, so a value equal to a bound belongs to that bucket, the wire buckets are cumulative, and `+Inf` equals `_count`.
- The names, types, buckets, and label keys are the contract's, character for character: dashboards and alerts query them.
- TPOT is $(E2E - TTFT)/(n - 1)$, defined only for $n \ge 2$ tokens.
- Counters never decrease, unbounded labels are capped, and label values are escaped; any of the three broken and the scrape, the cardinality, or `rate()` breaks.
- One request is one trace: the engine's SERVER span is the child of the caller's `traceparent`, its queue, prefill, and decode spans are its children, and the export is OTLP/HTTP JSON.

## How to read this chapter

Sections 1 to 3 explain why the engine needs histograms and traces, the rules of the Prometheus and W3C formats, and work a histogram, a TPOT, and a trace by hand. Section 4 fixes the interface. Section 5 walks through the solution: the histogram, the single validating write path, the exposition, the request metrics, the trace parser, and the span tree. Try to predict each step before reading it.

The solution declares `pub mod metrics; pub mod telemetry;` in `tl-serve/src/lib.rs` and takes over `server.rs` from `L10.5`, keeping all of its behaviour. The server keeps one `EngineMetrics` behind a mutex: `record_http` for every request, `record_request` when a generation ends (early errors too, with their `code` as `error_type`), `set_engine(&EngineGauges::from_stats(&engine.stats(), spec_rate))` before rendering `/metrics`. The engine thread stamps when it takes a request and when it samples each token; at the end of each request the server builds `request_spans` from those times, writes one `log_line` inside the SERVER span to stderr, and posts `otlp_json` to `$OTEL_EXPORTER_OTLP_ENDPOINT/v1/traces` on a thread of its own.

---

## 1. Why now

The engine of `L10.5` answers requests, and nobody can tell how well. The system's SLOs (TTFT p95 under 500 ms, TPOT p95 under 60 ms, 99.5% availability) are statements about distributions; the burn-rate alerts that page an on-call engineer are ratios of counts over windows; the gateway (`gw.05`) needs queue depth and KV usage to route. All of that comes from the numbers this module exposes. The Pass 1 tracer wrote one span by hand; from now on a request crosses the gateway, a prefill engine, a KV transfer, and a decode engine, and only a shared trace id lets you see one request's path through all four.

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

**A histogram.** Bounds $[0.1, 0.5, 1.0]$; observations $0.05, 0.1, 0.3, 2.0$.

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

**TPOT.** TTFT 0.2 s, E2E 1.0 s, 5 tokens: TPOT $= 0.8/4 = 0.2$ s, counted in the `le="0.2"` bucket and not in `le="0.15"`.

**A trace.** The W3C example `00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01`: trace id `4bf9...4736`, parent span `00f0...02b7`, flags `01` (sampled). The engine's SERVER span gets trace id `4bf9...4736`, a fresh span id, and parent `00f0...02b7`; its `engine.queue` child gets the same trace id and the SERVER span id as parent.

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

### What a correct solution must handle

The exposition is read back with a Prometheus text parser and compared with the engine instruments of `metrics.yaml`.

| Case | Checks | Why it matters downstream |
|---|---|---|
| The hand histogram | section 3's buckets and the exact six lines | the engine and the scraper agree on `le` |
| Names and buckets match the contract | every engine instrument: name, TYPE, HELP, label keys, bucket bounds; nothing extra | dashboards and SLO rules find their series |
| The hand TPOT | TPOT 0.2 s in `le="0.2"`; 1 token no TPOT; failures carry `error_type` | the TPOT SLO measures what it says |
| Count equals requests | 500 seeded requests: count, sum, +Inf == count, buckets ascending | quantiles are computed from these |
| Counters never go down | lower totals and negative or NaN increments refused | `rate()` and the burn-rate alerts |
| Start at zero | unlabeled series served before any event; KV gauges sum to the pool | alerts on a fresh engine |
| Gauges from engine stats | waiting is the queue depth, running the active sequences | queueing alerts point at the right gauge |
| Label cap and allowed values | 64 models then `_other`; values outside a list and missing labels refused | Prometheus memory under many tenants |
| Escaped label values | `"`, `\`, newline escaped and parsed back | one odd model name cannot break the scrape |
| The hand `traceparent` | the W3C example parsed and written back | the gateway's context reaches the engine |
| Invalid `traceparent` | uppercase, zero ids, version ff, extra fields on 00, bad lengths | a broken header starts a new trace |
| The span tree | SERVER under the caller, three children with their intervals and attributes, token events every 32 | `obs.01`'s span tree |
| A failed request | 5xx is ERROR with `error.type`; 429 is not | trace search finds real failures |
| OTLP JSON shape | hex ids, parent only on children, numeric kind, string times and ints, arrays, status | the collector accepts it |
| Export to a collector | one POST `/v1/traces`, JSON content type, the body unchanged; a dead collector errors in time | export never hangs a request |
| Propagation and log correlation | `traceparent` injected and extracted; log lines carry trace and span ids; seeded ids | `grep <trace_id>` over `kubectl logs` |
| The live serve loop | TTFT, TPOT, and request duration per generation (stream or not), a failure with its `error_type`, the HTTP histogram per route and status, the KV gauges, and the exported span tree continuing the caller's trace | `obs.02` scrapes these from the engine; `obs.01` joins its trace |

## 5. The solution, step by step

### 5.1 From first principles

1. **Quantiles must be computable later and across replicas.** A p95 computed in each pod cannot be averaged into a fleet p95. Counts in fixed buckets can be summed, so the engine exports histograms, not percentiles (section 2.1).
2. **The bucket rule must match the scraper's.** Prometheus reads `le="b"` as "at most $b$". So `observe` puts $x$ in the first bucket with $x \le b_j$, and the exposition writes running sums, ending in `+Inf` = `_count` (section 2.2, the $[2, 1, 0, 1] \to [2, 3, 3, 4]$ example).
3. **Names and buckets are an interface, not a choice.** Dashboards and alert rules query exact strings. So every name, bucket list, and label key is a constant transcribed from `metrics.yaml`, registered once in `EngineMetrics::new`.
4. **Bad input must be refused, not stored.** A counter that goes down reads as a restart to `rate()`; an unknown label value or a 65th model name grows Prometheus memory without bound. So every write goes through one function (`key`) that validates labels, and the counter writers refuse decreases and NaN.
5. **TPOT excludes the first token.** The first token's latency is TTFT. The remaining $n - 1$ tokens arrived in $T_{e2e} - T_{ttft}$, so TPOT $= (T_{e2e} - T_{ttft})/(n-1)$, undefined for $n < 2$ (section 2.4: $0.8/4 = 0.2$ s).
6. **A trace context is untrusted input.** The parser accepts exactly the W3C shape and nothing else; anything doubtful starts a new trace.
7. **Spans are cheap to build at the end.** The engine thread already stamps when a request was admitted and when each token was sampled. Building the four spans from those times after the request ends keeps tracing off the hot path; export happens on a separate thread, bounded by a timeout.

### 5.2 The histogram

[`metrics.rs`](../../../../course/solutions/rust/crates/tl-serve/src/metrics.rs) stores per-bucket counts and accumulates only when rendering:

```rust
pub struct Histogram {
    pub bounds: Vec<f64>,
    /// counts[i]: observations in (bounds[i-1], bounds[i]]; the last entry
    /// counts those above every bound.
    pub counts: Vec<u64>,
    pub sum: f64,
    pub count: u64,
}

impl Histogram {
    pub fn new(bounds: &[f64]) -> Histogram {
        // one more slot than bounds: the +Inf bucket, so nothing is ever dropped
        Histogram { bounds: bounds.to_vec(), counts: vec![0; bounds.len() + 1], sum: 0.0, count: 0 }
    }

    /// The `le` rule: a value equal to a bound belongs to that bucket.
    pub fn observe(&mut self, v: f64) {
        let i = self.bounds.iter().position(|&b| v <= b).unwrap_or(self.bounds.len());
        self.counts[i] += 1;
        self.sum += v;
        self.count += 1;
    }

    /// The `_bucket` values: running sums, ending at the +Inf bucket (== count).
    pub fn cumulative(&self) -> Vec<u64> {
        let mut acc = 0;
        self.counts.iter().map(|c| { acc += c; acc }).collect()
    }
}
```

In the worked example, `0.1` makes `position(|&b| 0.1 <= b)` return 0, so it lands in `le="0.1"`; `2.0` matches no bound and `unwrap_or` sends it to the last slot. `position` is a linear scan; with 13 to 16 bounds a binary search would not be measurably faster.

### 5.3 One gate for every write

`Registry::key` turns a label list into a series key and is the only path into a series, so every rule is enforced in one place:

```rust
fn key(&mut self, name: &str, labels: &[(&str, &str)]) -> Result<(usize, Vec<String>), MetricError> {
    let fi = self.families.iter().position(|f| f.name == name).ok_or_else(|| MetricError::UnknownMetric(name.to_string()))?;
    let f = &mut self.families[fi];
    if labels.len() != f.labels.len() { /* MetricError::Labels */ }
    let mut key = Vec::with_capacity(labels.len());
    for (i, spec) in f.labels.iter().enumerate() {           // declared order: one key per series
        let Some(&(_, v)) = labels.iter().find(|(k, _)| *k == spec.key) else { /* missing label */ };
        if let Some(allowed) = spec.values {
            if !v.is_empty() && !allowed.contains(&v) {      // fixed sets: state, operation, role
                return Err(MetricError::Value { label: spec.key.to_string(), value: v.to_string() });
            }
        }
        let mut v = v.to_string();
        if spec.capped && !v.is_empty() {
            let seen = f.seen.entry(i).or_default();
            if !seen.contains(&v) {
                if seen.len() < LABEL_CAP {                  // 64 distinct values, then _other
                    seen.push(v.clone());
                } else {
                    v = OTHER.to_string();
                }
            }
        }
        key.push(v);
    }
    Ok((fi, key))
}
```

`seen.len() < LABEL_CAP` admits exactly 64 values: the 65th distinct value finds `len() == 64` and becomes `_other`. An empty value means "this optional label is absent" (`error_type` on a success), and `render` leaves it out of the series.

The two counter writers refuse the two ways a counter could go down:

```rust
pub fn inc(&mut self, name: &str, labels: &[(&str, &str)], by: f64) -> Result<(), MetricError> {
    // ... kind must be Counter
    if by.is_nan() || by < 0.0 {                             // NaN < 0.0 is false: test NaN explicitly
        return Err(MetricError::Decrease(format!("{name}: inc by {by}")));
    }
    // ... add to the series
}

pub fn set_total(&mut self, name: &str, labels: &[(&str, &str)], total: f64) -> Result<(), MetricError> {
    // a running total read from elsewhere (evictions, preemptions): never lower
    // ...
    match cur {
        Value::Scalar(x) if total >= *x => { *x = total; Ok(()) }
        Value::Scalar(x) => Err(MetricError::Decrease(format!("{name}: {total} < {x}"))),
        // ...
    }
}
```

`register` inserts a zero series for every unlabeled counter and gauge, so a fresh engine exposes `tl_engine_queue_depth 0` before any request: an alert written as "absent or high" works from the first scrape.

### 5.4 The exposition

```rust
// per family: # HELP, # TYPE, then the series sorted by label values (BTreeMap)
for (key, val) in &f.series {
    let pairs: Vec<String> = f.labels.iter().zip(key)
        .filter(|(_, v)| !v.is_empty())                                  // absent optional labels
        .map(|(s, v)| format!("{}=\"{}\"", s.key, escape_label(v)))      // \  "  newline escaped
        .collect();
    // ...
    match val {
        Value::Scalar(x) => { let _ = writeln!(out, "{}{} {}", f.name, braces(None), fmt_value(*x)); }
        Value::Hist(h) => {
            let cum = h.cumulative();
            for (i, b) in h.bounds.iter().enumerate() {
                let _ = writeln!(out, "{}_bucket{} {}", f.name, braces(Some(format!("le=\"{}\"", fmt_value(*b)))), cum[i]);
            }
            let _ = writeln!(out, "{}_bucket{} {}", f.name, braces(Some("le=\"+Inf\"".to_string())), h.count);
            let _ = writeln!(out, "{}_sum{} {}", f.name, braces(None), fmt_value(h.sum));
            let _ = writeln!(out, "{}_count{} {}", f.name, braces(None), h.count);
        }
    }
}
```

`fmt_value` uses Rust's shortest round-trip decimal, so the bound `1.0` prints as `le="1"`, matching section 3's exact lines. `le` is appended last, after the series' own labels.

### 5.5 The engine instruments

`EngineMetrics::new` registers every engine instrument of `metrics.yaml`; the TTFT and TPOT histograms get their own bucket lists, dense near their SLO bounds:

```rust
pub const TTFT_BUCKETS: &[f64] = &[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0];
pub const TPOT_BUCKETS: &[f64] = &[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5];
```

`record_request` is section 2.4 in code:

```rust
pub fn record_request(&mut self, r: &RequestRecord) -> Result<(), MetricError> {
    let gen = [("gen_ai_operation_name", r.operation.as_str()), ("gen_ai_request_model", r.model.as_str()), ("tl_engine_role", r.role.as_str())];
    if let Some(ttft) = r.ttft_s {                       // no token, no TTFT (an early error)
        self.reg.observe(TTFT, &gen, ttft)?;
        if r.output_tokens >= 2 {                        // n - 1 intervals; none for one token
            self.reg.observe(TPOT, &gen, (r.e2e_s - ttft) / (r.output_tokens - 1) as f64)?;
        }
    }
    let err = r.error_type.as_deref().unwrap_or("");    // "": the label is left out on success
    self.reg.observe(REQUEST_DURATION, &[("gen_ai_operation_name", r.operation.as_str()), ("gen_ai_request_model", r.model.as_str()), ("error_type", err)], r.e2e_s)?;
    Ok(())
}
```

`EngineGauges::from_stats` maps the step loop's `EngineStats` (`L10.5`): `waiting` is the queue depth and `running` the active sequences, and `set_engine` writes free, used, and cached KV blocks as three series of one gauge, plus evictions and preemptions through `set_total`.

### 5.6 Trace context

[`telemetry.rs`](../../../../course/solutions/rust/crates/tl-serve/src/telemetry.rs) parses `traceparent` strictly:

```rust
/// Lowercase hex digits only (W3C forbids uppercase).
fn unhex<const N: usize>(s: &str) -> Option<[u8; N]> {
    if s.len() != 2 * N || !s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
        return None;
    }
    // ... two digits per byte
}

pub fn parse(header: &str) -> Option<TraceContext> {
    let h = header.trim();
    let parts: Vec<&str> = h.split('-').collect();
    if parts.len() < 4 { return None; }
    let version = unhex::<1>(parts[0])?[0];
    // ff is invalid; 00 has exactly four fields; a later version may append more
    if version == 0xff || (version == 0 && parts.len() != 4) { return None; }
    let trace_id = unhex::<16>(parts[1])?;
    let span_id = unhex::<8>(parts[2])?;
    let flags = unhex::<1>(parts[3])?[0];
    if trace_id == [0; 16] || span_id == [0; 8] { return None; }   // all-zero ids are invalid
    Some(TraceContext { trace_id, span_id, flags })
}

pub fn header(&self) -> String {
    format!("00-{}-{}-{:02x}", hex(&self.trace_id), hex(&self.span_id), self.flags)   // {:02x}: flags are two digits
}
```

`unhex` checks the length before converting, so a 31-digit trace id or a one-digit flag field fails cleanly.

### 5.7 The span tree

```rust
pub fn request_spans(ids: &mut IdGen, incoming: Option<&TraceContext>, t: &RequestTiming) -> Vec<Span> {
    // SERVER continues the caller's trace: same trace id, parent = the caller's span id
    let mut server = Span::start(ids, &format!("POST {}", t.route), SpanKind::Server, incoming, t.received_ns);
    server.end_ns = t.end_ns;
    // ... http.* and gen_ai.* attributes
    if t.status >= 500 || t.error_type.is_some() { /* error.type, status ERROR */ }
    let ctx = server.context();                     // children hang under SERVER, not under the caller
    let mut queue = Span::start(ids, "engine.queue", SpanKind::Internal, Some(&ctx), t.received_ns);
    queue.end_ns = t.admitted_ns;                   // [received, admitted]
    // ...
    let mut prefill = Span::start(ids, "engine.prefill", SpanKind::Internal, Some(&ctx), t.admitted_ns);
    prefill.end_ns = t.first_token_ns;              // [admitted, first token]
    // ...
    let mut decode = Span::start(ids, "engine.decode", SpanKind::Internal, Some(&ctx), t.first_token_ns);
    decode.end_ns = t.end_ns;                       // [first token, end]
    // ... a `token` event at every 32nd token
    vec![server, queue, prefill, decode]
}
```

The three children tile the SERVER interval exactly, so in a trace viewer the queue wait, the prefill, and the decode add up to the request. In the section 3 trace, `Span::start` with `Some(incoming)` copies trace id `4bf9...4736` and sets parent `00f0...02b7`; `server.context()` then gives the children the SERVER's fresh span id as their parent.

The serve loop ([`server.rs`](../../../../course/solutions/history/L10.7/rust/crates/tl-serve/src/server.rs) as of this module) passes `error_type` into the spans only for a 5xx (`out.error_type.clone().filter(|_| out.status >= 500)`), so a 429 is recorded in the duration histogram with its `error_type` but is not an ERROR span: overload is a client-visible condition, not a server failure.

### 5.8 OTLP JSON, logs, and export

`otlp_json` writes the ExportTraceServiceRequest by hand. The details that collectors reject when wrong:

```rust
fn any_value(v: &AttrValue) -> String {
    match v {
        AttrValue::Str(s) => format!("{{\"stringValue\":{}}}", json_str(s)),
        AttrValue::Int(i) => format!("{{\"intValue\":\"{i}\"}}"),        // 64-bit ints as strings (proto3 JSON)
        AttrValue::Float(f) if f.is_finite() => format!("{{\"doubleValue\":{f}}}"),
        AttrValue::Float(_) => "{\"doubleValue\":0}".to_string(),       // JSON has no NaN or Inf
        // ... Bool, StrArray as arrayValue
    }
}
// per span: "traceId" and "spanId" as lowercase hex, "parentSpanId" only when Some,
// "kind" as the enum's number, start and end times as strings,
// "status": {"code": 2, ...} only on error
```

`log_line` writes one JSON object with `ts`, `level`, `msg`, `service`, and, inside a span, `trace_id` and `span_id`. The serve loop's `export_spans` logs that line on stderr for every request and, only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, posts the body on a thread of its own with a 2 s timeout:

```rust
std::thread::spawn(move || {
    if let Err(e) = telemetry::export(&endpoint, &body, Duration::from_secs(2)) {
        eprintln!("tl-serve: span export failed: {e}");   // counted and dropped, never retried on the request path
    }
});
```

Full solution: [`metrics.rs`](../../../../course/solutions/rust/crates/tl-serve/src/metrics.rs), [`telemetry.rs`](../../../../course/solutions/rust/crates/tl-serve/src/telemetry.rs), and the serve loop's recording in [`server.rs`](../../../../course/solutions/history/L10.7/rust/crates/tl-serve/src/server.rs) (`finish_request`, `export_spans`, `server_span`).

### 5.9 Why this solution

- **Cost.** `observe` is $O(m)$ for $m \le 16$ bounds; `key` is $O(\ell)$ label checks plus a scan of at most 64 seen values. Rendering is $O(\text{series} \cdot m)$, paid only per scrape. All of it is a mutex-guarded update per request, negligible next to a decode step.
- **Why a hand-written registry, not an SDK?** The contract is the point: writing the exposition by hand makes every rule (`le`, cumulative buckets, escaping, `+Inf` = `_count`) visible and checkable against `metrics.yaml`, and it costs a few hundred lines. The `prometheus` crate is not in [`allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml); the OpenTelemetry crates are, and are the production route (see Going further), but they hide exactly the rules this module is about.
- **Why build spans at the end instead of opening them live?** The engine thread would have to touch tracing state every step. Timestamps it already records are enough to reconstruct the intervals exactly, and the request path pays nothing for an unreachable collector.
- **Why fold to `_other` instead of refusing?** Refusing would lose the observation; folding keeps the totals correct (every request still counts) while bounding the series count.
- **Tradeoffs accepted.** One export POST per request does not batch; at high request rates a batching exporter (the OTel SDK's) is cheaper. `prefix_hit_tokens` is 0 per request because the engine reports hits only as a ratio gauge.

### 5.10 Explain it back

**An observation of exactly 0.5 s arrives. Which `_bucket` lines change, and why does it matter for a 500 ms SLO?**

<details><summary>Answer</summary>

`position(|&b| 0.5 <= b)` finds the `0.5` bound, so `le="0.5"` and every larger bound, plus `+Inf`, go up by one (they are cumulative). If `le` were treated as "less than", it would land in `0.75`, and a request that met the SLO exactly would count as a violation; across many such requests p95 drifts up.

</details>

**Why is TPOT divided by $n - 1$ and not $n$, and why is it skipped for $n = 1$?**

<details><summary>Answer</summary>

$T_{e2e} - T_{ttft}$ is the time after the first token, during which $n - 1$ tokens arrived; the first token's latency is already TTFT. Dividing by $n$ makes TPOT read low, so the SLO passes while users wait. With $n = 1$ there are no intervals, and $0/0$ would be NaN.

</details>

**Why does `inc` test `by.is_nan()` separately from `by < 0.0`?**

<details><summary>Answer</summary>

Every comparison with NaN is false, so `NaN < 0.0` is false and a NaN increment would pass a lone `< 0.0` check. Added to a counter, it makes the counter NaN forever, and `rate()` over it is NaN.

</details>

**The children of SERVER are parented to `server.context()`, not to `incoming`. What would a viewer show if they used `incoming`?**

<details><summary>Answer</summary>

`engine.queue`, `engine.prefill`, and `engine.decode` would appear as siblings of the engine's SERVER span under the gateway's proxy span, so the tree would no longer show that they are parts of the engine's handling of the request, and their durations would not nest inside it.

</details>

**Why is `traceparent` with uppercase hex refused instead of lowercased?**

<details><summary>Answer</summary>

W3C requires lowercase, and the other services in the path refuse uppercase. If the engine alone accepted it, the engine would join a trace that every other hop had already split off, so the system would disagree about where the trace starts. Every hop applying the same rule, "invalid means start a new trace", keeps the trace boundaries consistent.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| `le` treated as "less than" | values exactly on a bound land one bucket late: p95 drifts up | `observe` takes the first bound with `v <= b`; see 5.2 |
| A metric name not the contract's | the SLO rules and dashboards show no data | every name is a constant transcribed from `metrics.yaml` and registered once in `EngineMetrics::new`; see 5.5 |
| TPOT on the TTFT buckets | no resolution around the 60 ms SLO bound | TPOT is registered with `TPOT_BUCKETS`, its own list; see 5.5 |
| TPOT divided by $n$, not $n - 1$ | TPOT reads low; the SLO passes while users wait | `record_request` divides by `output_tokens - 1` and skips $n < 2$; see 5.5 |
| Observations above the top bound not counted | `+Inf` < `_count`: quantiles and rates go wrong for slow requests | `counts` has one more slot than `bounds`, and `+Inf` is written from `count`; see 5.2 and 5.4 |
| A counter accepting a lower total | `rate()` reads a restart and spikes the burn rate | `set_total` refuses `total < current`, `inc` refuses negative and NaN; see 5.3 |
| An off-by-one label cap | 65 series per label: the cap is not the contract's | `seen.len() < LABEL_CAP` admits exactly 64 values; see 5.3 |
| A quote not escaped in a label value | the whole scrape fails to parse | `escape_label` escapes `\`, `"`, and newline in every rendered value; see 5.4 |
| Flags written as one hex digit | downstream services reject the header; the trace splits | `header` formats flags with `{:02x}`; see 5.6 |
| Uppercase hex accepted | a header other services reject is propagated | `unhex` accepts only `0-9a-f`; see 5.6 |
| The SERVER span dropping its parent | the engine's spans float as a separate trace | `Span::start` with the incoming context copies its trace id and sets its span id as parent; see 5.7 |
| Children parented to the caller | the engine's internal spans appear as siblings of its own SERVER span | children start from `server.context()`; see 5.7 |
| 64-bit ints written as JSON numbers | OTLP JSON readers lose precision or refuse the body | `any_value` writes `intValue` and times as strings; see 5.8 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L10.5` | `EngineStats` from the step loop feeds the gauges; this module takes over its `server.rs`, whose `/metrics` route now renders the registry |
| Forward | `obs.01` | the collector receives these spans; the span tree of a request crosses gateway, prefill, transfer, decode |
| Forward | `L10.9` | takes `server.rs` over next: tool requests are recorded and traced through the same calls |
| Forward | `obs.02` | runs the engine and checks its `/metrics` (names, types, buckets, labels against `metrics.yaml`) and its JSON log lines |
| Forward | `obs.03` | the SLO burn-rate rules are written over `gen_ai_server_time_to_first_token_seconds_bucket` and friends |
| Forward | `load.01` | the load generator's client-side TTFT and TPOT are compared with these server-side numbers |

`L10.6` records `tl_kv_transfer_bytes_total` through `record_kv_transfer`; `L10.8` feeds `tl_engine_spec_accept_rate`.

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a hand-written registry | the `prometheus` and `opentelemetry` crates | exemplars, native histograms, push and pull from one SDK | [opentelemetry-rust](https://github.com/open-telemetry/opentelemetry-rust) |
| fixed buckets | Prometheus native histograms | exponential buckets with no bucket choice and lower error | [native histograms](https://prometheus.io/docs/specs/native_histograms/) |
| spans built at request end | `tracing` + `tracing-opentelemetry` | spans opened and closed as the code runs, with context in thread-locals | [tracing-opentelemetry](https://github.com/tokio-rs/tracing-opentelemetry) |
| OTLP/HTTP JSON per request | the OTel SDK batch processor over gRPC | batching, retries, sampling decisions in one place | [OTLP exporter spec](https://opentelemetry.io/docs/specs/otel/protocol/exporter/) |
| vLLM's metrics | vLLM `/metrics` | the same GenAI instruments plus scheduler and cache internals | [vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics.html) |
