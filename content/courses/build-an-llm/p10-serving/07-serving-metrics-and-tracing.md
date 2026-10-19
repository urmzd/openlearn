# Serving metrics and tracing

The engine from [the OpenAI-compatible server on tokio](05-openai-server-on-tokio.md) answers requests, and nobody can tell how well. The system's SLOs (TTFT p95 under 500 ms, TPOT p95 under 60 ms, 99.5% availability) are statements about distributions. The burn-rate alerts that page an on-call engineer are ratios of counts over windows. The gateway's [worker registry and router](../../ai-platform-engineering/12-gateway/05-routing.md) needs queue depth and KV usage to route. All of those numbers come from what this section exposes.

Tracing has the same story. [Your first endpoint](00-your-first-endpoint.md) wrote one span by hand. From here on, a request crosses the gateway, a prefill engine, a KV transfer, and a decode engine (see [disaggregated prefill and decode](06-disaggregated-prefill-decode.md)), and only a shared trace id lets you see one request's path through all four.

This section builds both halves: a Prometheus registry with SLO histograms, and W3C trace propagation with OTLP export. It builds on the engine loop's `EngineStats` and the `/metrics` route from the tokio server. For the statistics of quantiles and intervals, see [LLN, CLT, confidence intervals, and the bootstrap](../../mathematics-for-computing/07-probability-statistics/04-lln-clt-confidence-intervals-bootstrap.md) in *Mathematics for Computing*; for the tracer that came first, see [one trace from gateway to engine](../../systems/04-observability/00-one-trace.md) in *Systems*.

The contract has three parts: [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml) fixes names, types, buckets, and labels; [`otel/semconv.md`](../../../../course/contracts/otel/semconv.md) fixes spans, attributes, propagation, and logs; and [`otel/slo.schema.json`](../../../../course/contracts/otel/slo.schema.json) describes what the histograms feed. The solution is [`tl-serve/src/metrics.rs`](../../../../course/solutions/rust/crates/tl-serve/src/metrics.rs) (`Histogram`, `Registry` with counters, gauges, cumulative histograms, and the Prometheus text exposition, and `EngineMetrics` with every engine instrument of the contract), [`tl-serve/src/telemetry.rs`](../../../../course/solutions/rust/crates/tl-serve/src/telemetry.rs) (`TraceContext`, `request_spans`, `otlp_json`, `export`, `log_line`), and the serve loop in [`server.rs` as of this section](../../../../course/solutions/history/L10.7/rust/crates/tl-serve/src/server.rs) (taken over from the tokio server; there is also a [later version](../../../../course/solutions/rust/crates/tl-serve/src/server.rs)), which records every request in `EngineMetrics` and ends it with one JSON log line and its span tree (`finish_request`, `export_spans`).

> **Learning objectives**
> - Place an observation in the right Prometheus histogram bucket and write the cumulative exposition by hand.
> - Compute TPOT correctly and say when it is undefined.
> - Enforce the contract's metric rules (no decreasing counters, capped and escaped labels) through a single write path.
> - Parse and emit a W3C `traceparent` strictly, and build an engine span tree that continues the caller's trace.
> - Export spans as OTLP/HTTP JSON without ever blocking a request.

## Three kinds of metric

Prometheus has three instrument kinds worth knowing here. A **counter** only goes up: requests served, bytes moved, evictions. Prometheus differentiates it with `rate()` and reads any drop as a process restart. A **gauge** goes both ways: queue depth, free blocks. A **histogram** counts observations into fixed buckets, plus their sum and count, so any quantile can be estimated later, over any window, and aggregated across replicas. A precomputed p95 cannot do that: a p95 computed in each pod cannot be averaged into a fleet p95, but counts in fixed buckets can be summed. That is why the engine exports histograms, not percentiles.

A little notation helps for what follows.

| Symbol | Meaning | Type |
|---|---|---|
| $b_1 < \dots < b_m$ | a histogram's bucket upper bounds (seconds) | `f64` |
| $c_j$ | observations $x$ with $b_{j-1} < x \le b_j$ ($b_0 = -\infty$, $b_{m+1} = +\infty$) | count |
| $C_j = \sum_{i \le j} c_i$ | the cumulative count served as `_bucket{le="b_j"}` | count |
| $n$ | output tokens of a request | count |
| $T_{ttft}, T_{e2e}$ | receipt to first token; receipt to the end | seconds |

## Histograms on the wire

A value $x$ goes into the first bucket with $x \le b_j$: Prometheus reads `le="b"` as "at most $b$", so a value equal to a bound belongs to that bucket. The exposition serves cumulative counts $C_j$, then `le="+Inf"` with the total, then `_sum` and `_count`. So $C_{m+1}$ equals `_count` always, and $C_j$ never decreases in $j$.

`histogram_quantile(0.95, ...)` interpolates inside the bucket where the cumulative count crosses 95%. That is why the contract's buckets are dense around the SLO bounds: 0.5 s for TTFT, 0.06 s for TPOT. Bucket placement is resolution, and resolution is only useful where you make decisions.

## Labels and cardinality

Each distinct label combination is a separate series that Prometheus stores forever. A label with a fixed value set (`state`, `gen_ai.operation.name`) is cheap. A label fed by users (`gen_ai.request.model`, `http.route`) is capped at 64 distinct values per process, with the rest recorded as `_other`. Label values are also user data: `\`, `"`, and newline must be escaped as `\\`, `\"`, `\n`, or a single odd model name breaks the whole scrape.

## The request metrics

For a request with $n$ output tokens, three numbers matter. TTFT is $T_{ttft}$, recorded when a token came. TPOT is

$$\text{TPOT} = \frac{T_{e2e} - T_{ttft}}{n - 1}, \qquad n \ge 2,$$

because the first token's time is TTFT, not TPOT: the remaining $n - 1$ tokens arrived in $T_{e2e} - T_{ttft}$. For $n < 2$ TPOT is undefined. The request duration is $T_{e2e}$ always, with `error_type` set on failure.

The engine gauges come from the step loop's `EngineStats`: waiting requests are the queue depth, running ones are the active sequences, and free + used + cached KV blocks equal the pool.

## Trace context

Traces need their own rules. A `traceparent` header is `00-<32 hex trace id>-<16 hex parent span id>-<2 hex flags>`, lowercase, with ids not all zero; version `ff` is invalid, and version `00` has exactly four fields. Invalid means "start a new trace", never "half trust it".

The engine's SERVER span takes the trace id and records the caller's span as its parent. Its children carry the trace id and the SERVER span as parent. Outgoing calls (such as `kv.transfer`) carry the current span as `traceparent`, in HTTP headers and gRPC metadata alike.

## OTLP/HTTP JSON

The export format is one `ExportTraceServiceRequest`. It holds `resourceSpans[].resource.attributes` (with service.name `<system>-engine` first) and `scopeSpans[].spans[]`. Each span has `traceId` and `spanId` as lowercase hex, `parentSpanId` only on children, `kind` as a number (1 internal, 2 server, 3 client), times and 64-bit integers as strings, doubles as numbers, string arrays as `arrayValue`, and `status.code = 2` on error.

## Working it by hand

Three small examples pin down all of the rules above.

**A histogram.** Take bounds $[0.1, 0.5, 1.0]$ and observations $0.05, 0.1, 0.3, 2.0$.

1. $0.05 \le 0.1$: bucket 1. $0.1 \le 0.1$: bucket 1 too, because `le` means at most. $0.3$: bucket 2. $2.0$ is above every bound, so it goes to the `+Inf` bucket.
2. $c = [2, 1, 0, 1]$, so the wire buckets are $C = [2, 3, 3, 4]$; the sum is $2.45$ and the count 4.

```text
x_seconds_bucket{op="chat",le="0.1"} 2
x_seconds_bucket{op="chat",le="0.5"} 3
x_seconds_bucket{op="chat",le="1"} 3
x_seconds_bucket{op="chat",le="+Inf"} 4
x_seconds_sum{op="chat"} 2.45
x_seconds_count{op="chat"} 4
```

**TPOT.** TTFT 0.2 s, E2E 1.0 s, 5 tokens: TPOT $= 0.8/4 = 0.2$ s, counted in the `le="0.2"` bucket and not in `le="0.15"`.

**A trace.** The W3C example `00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01` has trace id `4bf9...4736`, parent span `00f0...02b7`, and flags `01` (sampled). The engine's SERVER span gets trace id `4bf9...4736`, a fresh span id, and parent `00f0...02b7`; its `engine.queue` child gets the same trace id and the SERVER span id as parent.

## Designing the interface

The interface splits into a metrics module and a telemetry module; the server's public API does not change.

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

// rust/crates/tl-serve/src/server.rs, taken over from the tokio server: its public API is unchanged
// (ServeConfig, spawn, run, ServerHandle); /metrics renders EngineMetrics, and every request
// is recorded, logged, and traced as above.
```

The solution declares `pub mod metrics; pub mod telemetry;` in `tl-serve/src/lib.rs` and takes over `server.rs` from the tokio server, keeping all of its behaviour. The server keeps one `EngineMetrics` behind a mutex. It calls `record_http` for every request, `record_request` when a generation ends (early errors too, with their `code` as `error_type`), and `set_engine(&EngineGauges::from_stats(&engine.stats(), spec_rate))` before rendering `/metrics`. The engine thread stamps when it takes a request and when it samples each token. At the end of each request the server builds `request_spans` from those times, writes one `log_line` inside the SERVER span to stderr, and posts `otlp_json` to `$OTEL_EXPORTER_OTLP_ENDPOINT/v1/traces` on a thread of its own.

## Building it step by step

### Reasoning from first principles

The design follows from seven observations.

1. **Quantiles must be computable later and across replicas.** Counts in fixed buckets can be summed across pods; percentiles cannot. So the engine exports histograms.
2. **The bucket rule must match the scraper's.** Prometheus reads `le="b"` as "at most $b$". So `observe` puts $x$ in the first bucket with $x \le b_j$, and the exposition writes running sums ending in `+Inf` = `_count`, exactly the $[2, 1, 0, 1] \to [2, 3, 3, 4]$ example.
3. **Names and buckets are an interface, not a choice.** Dashboards and alert rules query exact strings. So every name, bucket list, and label key is a constant transcribed from `metrics.yaml`, registered once in `EngineMetrics::new`.
4. **Bad input must be refused, not stored.** A counter that goes down reads as a restart to `rate()`; an unknown label value or a 65th model name grows Prometheus memory without bound. So every write goes through one function (`key`) that validates labels, and the counter writers refuse decreases and NaN.
5. **TPOT excludes the first token.** The first token's latency is TTFT, so TPOT $= (T_{e2e} - T_{ttft})/(n-1)$, undefined for $n < 2$ ($0.8/4 = 0.2$ s in the example).
6. **A trace context is untrusted input.** The parser accepts exactly the W3C shape and nothing else; anything doubtful starts a new trace.
7. **Spans are cheap to build at the end.** The engine thread already stamps when a request was admitted and when each token was sampled. Building the four spans from those times after the request ends keeps tracing off the hot path, and export happens on a separate thread, bounded by a timeout.

### The histogram

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

In the worked example, `0.1` makes `position(|&b| 0.1 <= b)` return 0, so it lands in `le="0.1"`; `2.0` matches no bound, and `unwrap_or` sends it to the last slot. `position` is a linear scan; with 13 to 16 bounds a binary search would not be measurably faster.

### One gate for every write

The histogram is only storage. The rules about labels live in `Registry::key`, which turns a label list into a series key and is the only path into a series, so every rule is enforced in one place:

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

`register` inserts a zero series for every unlabeled counter and gauge, so a fresh engine exposes `tl_engine_queue_depth 0` before any request. An alert written as "absent or high" works from the first scrape.

### The exposition

With valid series stored, rendering is a careful loop over families:

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

`fmt_value` uses Rust's shortest round-trip decimal, so the bound `1.0` prints as `le="1"`, matching the exact lines of the hand example. `le` is appended last, after the series' own labels.

### The engine instruments

`EngineMetrics::new` registers every engine instrument of `metrics.yaml`. The TTFT and TPOT histograms get their own bucket lists, dense near their SLO bounds:

```rust
pub const TTFT_BUCKETS: &[f64] = &[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0];
pub const TPOT_BUCKETS: &[f64] = &[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5];
```

`record_request` is the request-metrics definition in code:

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

`EngineGauges::from_stats` maps the step loop's `EngineStats` from the tokio server: `waiting` is the queue depth and `running` the active sequences. `set_engine` writes free, used, and cached KV blocks as three series of one gauge, plus evictions and preemptions through `set_total`.

### Trace context

Metrics done, the tracing half starts with parsing. [`telemetry.rs`](../../../../course/solutions/rust/crates/tl-serve/src/telemetry.rs) parses `traceparent` strictly:

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

### The span tree

Given a parsed context and the timestamps the engine thread recorded, `request_spans` builds the four spans after the fact:

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

The three children tile the SERVER interval exactly, so in a trace viewer the queue wait, the prefill, and the decode add up to the request. In the hand-worked trace, `Span::start` with `Some(incoming)` copies trace id `4bf9...4736` and sets parent `00f0...02b7`; `server.context()` then gives the children the SERVER's fresh span id as their parent.

The serve loop passes `error_type` into the spans only for a 5xx (`out.error_type.clone().filter(|_| out.status >= 500)`). A 429 is therefore recorded in the duration histogram with its `error_type` but is not an ERROR span: overload is a client-visible condition, not a server failure.

### OTLP JSON, logs, and export

`otlp_json` writes the ExportTraceServiceRequest by hand. These are the details collectors reject when they are wrong:

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

The full solution is [`metrics.rs`](../../../../course/solutions/rust/crates/tl-serve/src/metrics.rs), [`telemetry.rs`](../../../../course/solutions/rust/crates/tl-serve/src/telemetry.rs), and the serve loop's recording in [`server.rs`](../../../../course/solutions/history/L10.7/rust/crates/tl-serve/src/server.rs) (`finish_request`, `export_spans`, `server_span`).

### Why this design

The cost is negligible. `observe` is $O(m)$ for $m \le 16$ bounds; `key` is $O(\ell)$ label checks plus a scan of at most 64 seen values. Rendering is $O(\text{series} \cdot m)$, paid only per scrape. All of it is a mutex-guarded update per request, small next to a decode step.

Why a hand-written registry rather than an SDK? Because the contract is the point. Writing the exposition by hand makes every rule (`le`, cumulative buckets, escaping, `+Inf` = `_count`) visible and checkable against `metrics.yaml`, and it costs a few hundred lines. The `prometheus` crate is not in [`allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml); the OpenTelemetry crates are, and they are the production route, but they hide exactly the rules this section is about.

Why build spans at the end instead of opening them live? Opening them live would make the engine thread touch tracing state every step. The timestamps it already records are enough to reconstruct the intervals exactly, and the request path pays nothing for an unreachable collector.

Why fold to `_other` instead of refusing? Refusing would lose the observation. Folding keeps the totals correct (every request still counts) while bounding the series count.

Two tradeoffs are accepted. One export POST per request does not batch, and at high request rates a batching exporter (the OTel SDK's) is cheaper. And `prefix_hit_tokens` is 0 per request, because the engine reports hits only as a ratio gauge.

## What can go wrong

The tests read the exposition back with a Prometheus text parser and compare it with the engine instruments of `metrics.yaml`. A correct solution handles each of these cases:

| Case | Checks | Why it matters downstream |
|---|---|---|
| The hand histogram | the worked example's buckets and the exact six lines | the engine and the scraper agree on `le` |
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
| The span tree | SERVER under the caller, three children with their intervals and attributes, token events every 32 | the serving-path span tree in *Systems* |
| A failed request | 5xx is ERROR with `error.type`; 429 is not | trace search finds real failures |
| OTLP JSON shape | hex ids, parent only on children, numeric kind, string times and ints, arrays, status | the collector accepts it |
| Export to a collector | one POST `/v1/traces`, JSON content type, the body unchanged; a dead collector errors in time | export never hangs a request |
| Propagation and log correlation | `traceparent` injected and extracted; log lines carry trace and span ids; seeded ids | `grep <trace_id>` over `kubectl logs` |
| The live serve loop | TTFT, TPOT, and request duration per generation (stream or not), a failure with its `error_type`, the HTTP histogram per route and status, the KV gauges, and the exported span tree continuing the caller's trace | the observability sections scrape these and join the trace |

The pitfalls cluster around the histogram rules, the label rules, and the trace format.

On histograms: treating `le` as "less than" makes values exactly on a bound land one bucket late, and p95 drifts up; `observe` takes the first bound with `v <= b`. Putting TPOT on the TTFT buckets leaves no resolution around the 60 ms SLO bound, so TPOT is registered with `TPOT_BUCKETS`, its own list. Dividing TPOT by $n$ instead of $n - 1$ makes it read low, so the SLO passes while users wait; `record_request` divides by `output_tokens - 1` and skips $n < 2$. Not counting observations above the top bound makes `+Inf` less than `_count`, and quantiles and rates go wrong for slow requests; `counts` has one more slot than `bounds`, and `+Inf` is written from `count`.

> **Warning:** A metric name that is not the contract's character for character produces no error at all. The SLO rules and dashboards simply show no data. Every name is a constant transcribed from `metrics.yaml` and registered once in `EngineMetrics::new` for exactly this reason.

On counters and labels: a counter that accepts a lower total makes `rate()` read a restart and spikes the burn rate; `set_total` refuses `total < current`, and `inc` refuses negative and NaN. An off-by-one label cap gives 65 series per label, which is not the contract's cap; `seen.len() < LABEL_CAP` admits exactly 64 values. A quote not escaped in a label value makes the whole scrape fail to parse; `escape_label` escapes `\`, `"`, and newline in every rendered value.

On traces: flags written as one hex digit make downstream services reject the header, and the trace splits; `header` formats flags with `{:02x}`. Accepting uppercase hex propagates a header other services reject; `unhex` accepts only `0-9a-f`. A SERVER span that drops its parent makes the engine's spans float as a separate trace; `Span::start` with the incoming context copies its trace id and sets its span id as parent. Children parented to the caller make the engine's internal spans appear as siblings of its own SERVER span; children start from `server.context()`. And 64-bit ints written as JSON numbers make OTLP JSON readers lose precision or refuse the body; `any_value` writes `intValue` and times as strings.

## Where this leads

This section takes over the tokio server's `server.rs`, whose `/metrics` route now renders the registry, and feeds its gauges from the step loop's `EngineStats`. [Disaggregated prefill and decode](06-disaggregated-prefill-decode.md) records `tl_kv_transfer_bytes_total` through `record_kv_transfer`, and [speculative decoding in the engine](08-speculative-decoding-in-the-engine.md) feeds `tl_engine_spec_accept_rate`. [Tool calls and constrained decoding](09-tool-calls-and-constrained-decoding.md) takes `server.rs` over next, and tool requests are recorded and traced through the same calls. The [load generator](10-load-generator.md) compares its client-side TTFT and TPOT with these server-side numbers.

In *Systems*, the observability chapter builds on all of it. [Tracing for the serving path](../../systems/04-observability/01-tracing-for-the-serving-path.md) has the collector receive these spans, with a request's span tree crossing gateway, prefill, transfer, and decode. [Metrics and log correlation](../../systems/04-observability/02-metrics-and-log-correlation.md) runs the engine and checks its `/metrics` (names, types, buckets, labels against `metrics.yaml`) and its JSON log lines. [SLOs and burn-rate alerts](../../systems/04-observability/03-slos-and-burn-rate-alerts.md) writes its rules over `gen_ai_server_time_to_first_token_seconds_bucket` and friends, and [dashboards as code](../../systems/04-observability/04-dashboards-as-code.md) charts them. The incident walkthroughs compute burn-rate alerts from these series, and in the production deployment Prometheus scrapes this endpoint.

Production systems replace each hand-built piece with something richer. The `prometheus` and `opentelemetry` crates add exemplars, native histograms, and push and pull from one SDK. Prometheus native histograms use exponential buckets, removing the bucket choice and lowering error. `tracing` with `tracing-opentelemetry` opens and closes spans as the code runs, with context in thread-locals, instead of building them at request end. The OTel SDK's batch processor over gRPC replaces one OTLP/HTTP JSON POST per request with batching, retries, and sampling decisions in one place. And vLLM's `/metrics` exposes the same GenAI instruments plus scheduler and cache internals.

### Further reading

- [Prometheus exposition formats](https://prometheus.io/docs/instrumenting/exposition_formats/) (free) and [native histograms](https://prometheus.io/docs/specs/native_histograms/).
- [W3C Trace Context](https://www.w3.org/TR/trace-context/) (free).
- [OTLP JSON encoding](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding) (free) and the [OTLP exporter spec](https://opentelemetry.io/docs/specs/otel/protocol/exporter/).
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) (free).
- [opentelemetry-rust](https://github.com/open-telemetry/opentelemetry-rust) and [tracing-opentelemetry](https://github.com/tokio-rs/tracing-opentelemetry).
- [vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics.html).
- *Site Reliability Engineering*, ch. 6, "Monitoring Distributed Systems" (free online).

## Key terms

- **Counter**: a metric that only increases; `rate()` reads any drop as a restart.
- **Gauge**: a metric that can go up or down, such as queue depth or free KV blocks.
- **Histogram**: counts of observations in fixed buckets plus sum and count, from which quantiles can be estimated and aggregated across replicas.
- **Cumulative bucket (`le`)**: the exposed count of observations at most the bound; `le="+Inf"` equals `_count`.
- **Cardinality**: the number of distinct label combinations, each a stored series; user-fed labels are capped at 64 values with the rest folded into `_other`.
- **TPOT**: time per output token, $(T_{e2e} - T_{ttft})/(n - 1)$, defined only for $n \ge 2$.
- **`traceparent`**: the W3C header `00-<trace id>-<parent span id>-<flags>` that carries trace context between services.
- **OTLP/HTTP JSON**: the OpenTelemetry export format used here, one `ExportTraceServiceRequest` POSTed to `/v1/traces`.

## Summary

- A histogram bucket `le="b"` counts observations at most `b`, so a value equal to a bound belongs to that bucket, the wire buckets are cumulative, and `+Inf` equals `_count`.
- The names, types, buckets, and label keys are the contract's, character for character: dashboards and alerts query them.
- TPOT is $(E2E - TTFT)/(n - 1)$, defined only for $n \ge 2$ tokens.
- Counters never decrease, unbounded labels are capped, and label values are escaped; break any of the three and the scrape, the cardinality, or `rate()` breaks.
- One request is one trace: the engine's SERVER span is the child of the caller's `traceparent`, its queue, prefill, and decode spans are its children, and the export is OTLP/HTTP JSON on a separate thread that never blocks the request.

## Review questions

1. An observation of exactly 0.5 s arrives. Which `_bucket` lines change, and why does it matter for a 500 ms SLO?
   <details><summary>Answer</summary>

   `position(|&b| 0.5 <= b)` finds the `0.5` bound, so `le="0.5"` and every larger bound, plus `+Inf`, go up by one (they are cumulative). If `le` were treated as "less than", it would land in `0.75`, and a request that met the SLO exactly would count as a violation; across many such requests p95 drifts up.

   </details>

2. Why is TPOT divided by $n - 1$ and not $n$, and why is it skipped for $n = 1$?
   <details><summary>Answer</summary>

   $T_{e2e} - T_{ttft}$ is the time after the first token, during which $n - 1$ tokens arrived; the first token's latency is already TTFT. Dividing by $n$ makes TPOT read low, so the SLO passes while users wait. With $n = 1$ there are no intervals, and $0/0$ would be NaN.

   </details>

3. Why does `inc` test `by.is_nan()` separately from `by < 0.0`?
   <details><summary>Answer</summary>

   Every comparison with NaN is false, so `NaN < 0.0` is false and a NaN increment would pass a lone `< 0.0` check. Added to a counter, it makes the counter NaN forever, and `rate()` over it is NaN.

   </details>

4. The children of SERVER are parented to `server.context()`, not to `incoming`. What would a viewer show if they used `incoming`?
   <details><summary>Answer</summary>

   `engine.queue`, `engine.prefill`, and `engine.decode` would appear as siblings of the engine's SERVER span under the gateway's proxy span, so the tree would no longer show that they are parts of the engine's handling of the request, and their durations would not nest inside it.

   </details>

5. Why is `traceparent` with uppercase hex refused instead of lowercased?
   <details><summary>Answer</summary>

   W3C requires lowercase, and the other services in the path refuse uppercase. If the engine alone accepted it, the engine would join a trace that every other hop had already split off, so the system would disagree about where the trace starts. Every hop applying the same rule, "invalid means start a new trace", keeps the trace boundaries consistent.

   </details>

6. A request returns 429 because the engine is overloaded. What does it record in metrics, and is its span marked ERROR?
   <details><summary>Answer</summary>

   It is recorded in the request-duration histogram with its `error_type`, but the serve loop passes `error_type` into the spans only for a 5xx, so the SERVER span is not ERROR. Overload is a client-visible condition, not a server failure, and keeping it out of ERROR spans means trace search finds real failures.

   </details>
