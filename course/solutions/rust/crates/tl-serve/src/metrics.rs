//! Serving metrics (L10.7): counters, gauges, and SLO histograms, served in
//! the Prometheus text format on the health port's `/metrics`.
//!
//! The instruments, their Prometheus names, buckets, and label keys are
//! fixed by course/contracts/otel/metrics.yaml (the entries an engine
//! emits): the dashboards (obs.04), the SLO rules (obs.03), the load
//! generator's cross-checks (load.01), and the drills all read these names.
//! A histogram is cumulative on the wire: `<name>_bucket{le="b"}` counts the
//! observations <= b, the last bucket is `le="+Inf"` and equals `_count`.
//!
//! Chapter: content/courses/build-an-llm/11-serving/08-serving-metrics-and-tracing.md.

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// The most distinct values a `capped` label keeps per process; later ones
/// are recorded as `_other` (metrics.yaml).
pub const LABEL_CAP: usize = 64;

/// The value a capped label takes past [`LABEL_CAP`].
pub const OTHER: &str = "_other";

/// Bucket upper bounds (seconds) of the GenAI and HTTP histograms, as
/// metrics.yaml lists them.
pub const TTFT_BUCKETS: &[f64] = &[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0];
pub const TPOT_BUCKETS: &[f64] = &[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5];
pub const REQUEST_BUCKETS: &[f64] = &[0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92];
pub const HTTP_BUCKETS: &[f64] = &[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0];

/// What a metric is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Counter,
    Gauge,
    Histogram,
}

/// A recording the contract does not allow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetricError {
    UnknownMetric(String),
    /// A label key the family does not declare, or one it declares missing.
    Labels(String),
    /// A value outside a label's `values` list.
    Value { label: String, value: String },
    /// A counter may only go up.
    Decrease(String),
    /// Wrong kind of recording for the metric (observe on a counter, ...).
    Kind(String),
}

/// One label: its key and either a fixed value set or a cap.
#[derive(Clone, Debug, PartialEq)]
pub struct LabelSpec {
    pub key: &'static str,
    /// Some: only these values (metrics.yaml `values`).
    pub values: Option<&'static [&'static str]>,
    /// The label keeps at most LABEL_CAP distinct values.
    pub capped: bool,
}

/// Cumulative-on-the-wire histogram over fixed upper bounds.
#[derive(Clone, Debug, PartialEq)]
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
        Histogram { bounds: bounds.to_vec(), counts: vec![0; bounds.len() + 1], sum: 0.0, count: 0 }
    }

    /// Adds one observation to the first bucket whose bound is >= v (the
    /// `le` rule: a value equal to a bound belongs to that bucket).
    pub fn observe(&mut self, v: f64) {
        let i = self.bounds.iter().position(|&b| v <= b).unwrap_or(self.bounds.len());
        self.counts[i] += 1;
        self.sum += v;
        self.count += 1;
    }

    /// The `_bucket` values: for each bound, the observations <= it, then
    /// the +Inf bucket (== count).
    pub fn cumulative(&self) -> Vec<u64> {
        let mut acc = 0;
        self.counts.iter().map(|c| {
            acc += c;
            acc
        }).collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Value {
    Scalar(f64),
    Hist(Histogram),
}

/// One metric family and its series.
#[derive(Clone, Debug)]
pub struct Family {
    pub name: &'static str,
    pub help: &'static str,
    pub kind: Kind,
    pub labels: Vec<LabelSpec>,
    pub buckets: &'static [f64],
    series: BTreeMap<Vec<String>, Value>,
    /// Distinct values seen per capped label (index into `labels`).
    seen: BTreeMap<usize, Vec<String>>,
}

/// Every family of one process.
#[derive(Clone, Debug, Default)]
pub struct Registry {
    families: Vec<Family>,
}

/// The value with `\`, `"`, and newline escaped (exposition format).
pub fn escape_label(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    for c in v.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out
}

/// A sample value or bound in the exposition: shortest decimal, `+Inf`,
/// `-Inf`, `NaN`.
pub fn fmt_value(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v == f64::INFINITY {
        "+Inf".to_string()
    } else if v == f64::NEG_INFINITY {
        "-Inf".to_string()
    } else {
        format!("{v}")
    }
}

impl Registry {
    pub fn new() -> Registry {
        Registry { families: Vec::new() }
    }

    /// Declares a family. `buckets` is used by histograms only. A counter
    /// or gauge without labels starts with one series at 0, so it is
    /// scrapeable (and rate() works) before its first change.
    pub fn register(&mut self, name: &'static str, help: &'static str, kind: Kind, labels: Vec<LabelSpec>, buckets: &'static [f64]) {
        let mut series = BTreeMap::new();
        if labels.is_empty() && kind != Kind::Histogram {
            series.insert(Vec::new(), Value::Scalar(0.0));
        }
        self.families.push(Family { name, help, kind, labels, buckets, series, seen: BTreeMap::new() });
    }

    /// The family names, in registration order.
    pub fn names(&self) -> Vec<&'static str> {
        self.families.iter().map(|f| f.name).collect()
    }

    /// The series key: label values in declared order, checked against the
    /// declaration (every declared key exactly once, `values` respected,
    /// capped labels folded into `_other` past the cap). A key may be
    /// omitted only when it is optional, which this registry marks by
    /// passing the value "" (the series then lacks that label).
    fn key(&mut self, name: &str, labels: &[(&str, &str)]) -> Result<(usize, Vec<String>), MetricError> {
        let fi = self.families.iter().position(|f| f.name == name).ok_or_else(|| MetricError::UnknownMetric(name.to_string()))?;
        let f = &mut self.families[fi];
        if labels.len() != f.labels.len() {
            return Err(MetricError::Labels(format!("{name}: {} labels given, {} declared", labels.len(), f.labels.len())));
        }
        let mut key = Vec::with_capacity(labels.len());
        for (i, spec) in f.labels.iter().enumerate() {
            let Some(&(_, v)) = labels.iter().find(|(k, _)| *k == spec.key) else {
                return Err(MetricError::Labels(format!("{name}: label {} missing", spec.key)));
            };
            if let Some(allowed) = spec.values {
                if !v.is_empty() && !allowed.contains(&v) {
                    return Err(MetricError::Value { label: spec.key.to_string(), value: v.to_string() });
                }
            }
            let mut v = v.to_string();
            if spec.capped && !v.is_empty() {
                let seen = f.seen.entry(i).or_default();
                if !seen.contains(&v) {
                    if seen.len() < LABEL_CAP {
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

    /// Adds `by` (>= 0) to a counter.
    pub fn inc(&mut self, name: &str, labels: &[(&str, &str)], by: f64) -> Result<(), MetricError> {
        let (fi, key) = self.key(name, labels)?;
        let f = &mut self.families[fi];
        if f.kind != Kind::Counter {
            return Err(MetricError::Kind(format!("{name} is not a counter")));
        }
        if by.is_nan() || by < 0.0 {
            return Err(MetricError::Decrease(format!("{name}: inc by {by}")));
        }
        match f.series.entry(key).or_insert(Value::Scalar(0.0)) {
            Value::Scalar(x) => *x += by,
            Value::Hist(_) => unreachable!(),
        }
        Ok(())
    }

    /// Sets a counter to a running total read from elsewhere (the KV pool's
    /// eviction count, the scheduler's preemptions). A total below the
    /// current value is an error: counters never go down.
    pub fn set_total(&mut self, name: &str, labels: &[(&str, &str)], total: f64) -> Result<(), MetricError> {
        let (fi, key) = self.key(name, labels)?;
        let f = &mut self.families[fi];
        if f.kind != Kind::Counter {
            return Err(MetricError::Kind(format!("{name} is not a counter")));
        }
        let cur = f.series.entry(key).or_insert(Value::Scalar(0.0));
        match cur {
            Value::Scalar(x) if total >= *x => {
                *x = total;
                Ok(())
            }
            Value::Scalar(x) => Err(MetricError::Decrease(format!("{name}: {total} < {x}"))),
            Value::Hist(_) => unreachable!(),
        }
    }

    /// Sets a gauge.
    pub fn set(&mut self, name: &str, labels: &[(&str, &str)], v: f64) -> Result<(), MetricError> {
        let (fi, key) = self.key(name, labels)?;
        let f = &mut self.families[fi];
        if f.kind != Kind::Gauge {
            return Err(MetricError::Kind(format!("{name} is not a gauge")));
        }
        f.series.insert(key, Value::Scalar(v));
        Ok(())
    }

    /// Adds an observation to a histogram.
    pub fn observe(&mut self, name: &str, labels: &[(&str, &str)], v: f64) -> Result<(), MetricError> {
        let (fi, key) = self.key(name, labels)?;
        let f = &mut self.families[fi];
        if f.kind != Kind::Histogram {
            return Err(MetricError::Kind(format!("{name} is not a histogram")));
        }
        let buckets = f.buckets;
        match f.series.entry(key).or_insert_with(|| Value::Hist(Histogram::new(buckets))) {
            Value::Hist(h) => h.observe(v),
            Value::Scalar(_) => unreachable!(),
        }
        Ok(())
    }

    /// The current value of a counter or gauge series (None if unseen).
    pub fn value(&self, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
        let f = self.families.iter().find(|f| f.name == name)?;
        let key: Option<Vec<String>> = f.labels.iter().map(|s| labels.iter().find(|(k, _)| *k == s.key).map(|(_, v)| v.to_string())).collect();
        match f.series.get(&key?)? {
            Value::Scalar(x) => Some(*x),
            Value::Hist(h) => Some(h.count as f64),
        }
    }

    /// The Prometheus text exposition (format 0.0.4): per family `# HELP`,
    /// `# TYPE`, then its series sorted by label values; histograms as
    /// `_bucket{...,le="b"}` for every bound and `+Inf`, then `_sum` and
    /// `_count`. A label whose value is "" is left out of the series.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for f in &self.families {
            let kind = match f.kind {
                Kind::Counter => "counter",
                Kind::Gauge => "gauge",
                Kind::Histogram => "histogram",
            };
            let help = f.help.replace('\\', "\\\\").replace('\n', "\\n");
            let _ = writeln!(out, "# HELP {} {}", f.name, help);
            let _ = writeln!(out, "# TYPE {} {}", f.name, kind);
            for (key, val) in &f.series {
                let pairs: Vec<String> = f
                    .labels
                    .iter()
                    .zip(key)
                    .filter(|(_, v)| !v.is_empty())
                    .map(|(s, v)| format!("{}=\"{}\"", s.key, escape_label(v)))
                    .collect();
                let braces = |extra: Option<String>| {
                    let mut all = pairs.clone();
                    all.extend(extra);
                    if all.is_empty() {
                        String::new()
                    } else {
                        format!("{{{}}}", all.join(","))
                    }
                };
                match val {
                    Value::Scalar(x) => {
                        let _ = writeln!(out, "{}{} {}", f.name, braces(None), fmt_value(*x));
                    }
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
        }
        out
    }
}

/// Names of the engine's instruments (metrics.yaml `prometheus`).
pub const TTFT: &str = "gen_ai_server_time_to_first_token_seconds";
pub const TPOT: &str = "gen_ai_server_time_per_output_token_seconds";
pub const REQUEST_DURATION: &str = "gen_ai_server_request_duration_seconds";
pub const HTTP_DURATION: &str = "http_server_request_duration_seconds";
pub const KV_BLOCKS: &str = "tl_engine_kv_blocks";
pub const KV_EVICTIONS: &str = "tl_engine_kv_evictions_total";
pub const QUEUE_DEPTH: &str = "tl_engine_queue_depth";
pub const ACTIVE_SEQUENCES: &str = "tl_engine_active_sequences";
pub const BATCH_TOKENS: &str = "tl_engine_batch_tokens";
pub const PREFIX_HIT_RATIO: &str = "tl_engine_prefix_cache_hit_ratio";
pub const SPEC_ACCEPT_RATE: &str = "tl_engine_spec_accept_rate";
pub const PREEMPTIONS: &str = "tl_engine_preemptions_total";
pub const KV_TRANSFER_BYTES: &str = "tl_kv_transfer_bytes_total";
pub const KV_TRANSFER_BLOCKS: &str = "tl_kv_transfer_blocks_total";

const OPS: &[&str] = &["chat", "text_completion"];
const OPS_E: &[&str] = &["chat", "text_completion", "embeddings"];
const ROLES: &[&str] = &["unified", "prefill", "decode", "gateway"];

fn fixed(key: &'static str, values: &'static [&'static str]) -> LabelSpec {
    LabelSpec { key, values: Some(values), capped: false }
}

fn capped(key: &'static str) -> LabelSpec {
    LabelSpec { key, values: None, capped: true }
}

/// One finished request as the engine saw it.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestRecord {
    /// "chat", "text_completion", or "embeddings".
    pub operation: String,
    pub model: String,
    /// The engine's role: unified, prefill, or decode.
    pub role: String,
    /// Receipt to the first output token; None when none was produced.
    pub ttft_s: Option<f64>,
    /// Receipt to the end of the response.
    pub e2e_s: f64,
    pub output_tokens: usize,
    /// The error `code` (or gRPC status name) of a failed request.
    pub error_type: Option<String>,
}

/// The engine's registry with every engine instrument of metrics.yaml.
pub struct EngineMetrics {
    pub reg: Registry,
}

/// Point-in-time engine state for the gauges and running totals.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EngineGauges {
    pub kv_free: u32,
    pub kv_used: u32,
    pub kv_cached: u32,
    pub kv_evictions: u64,
    pub queue_depth: usize,
    pub active_sequences: usize,
    pub batch_tokens: usize,
    pub prefix_hit_ratio: f64,
    pub spec_accept_rate: f64,
    pub preemptions: u64,
}

impl EngineGauges {
    /// The gauges of the engine loop's stats (L10.5) and the speculative
    /// accept rate (L10.8, 0 without speculation): waiting requests are the
    /// queue depth, running ones the active sequences.
    pub fn from_stats(s: &tl_engine::engine::EngineStats, spec_accept_rate: f64) -> EngineGauges {
        EngineGauges {
            kv_free: s.kv_free,
            kv_used: s.kv_used,
            kv_cached: s.kv_cached,
            kv_evictions: s.kv_evictions,
            queue_depth: s.waiting,
            active_sequences: s.running,
            batch_tokens: s.batch_tokens,
            prefix_hit_ratio: s.prefix_hit_rate,
            spec_accept_rate,
            preemptions: s.preemptions,
        }
    }
}

impl Default for EngineMetrics {
    fn default() -> Self {
        EngineMetrics::new()
    }
}

impl EngineMetrics {
    pub fn new() -> EngineMetrics {
        let mut r = Registry::new();
        r.register(TTFT, "Time from request receipt to the first output token. The TTFT SLO.", Kind::Histogram,
            vec![fixed("gen_ai_operation_name", OPS), capped("gen_ai_request_model"), fixed("tl_engine_role", ROLES)], TTFT_BUCKETS);
        r.register(TPOT, "(E2E - TTFT) / (output tokens - 1) per request. The TPOT SLO.", Kind::Histogram,
            vec![fixed("gen_ai_operation_name", OPS), capped("gen_ai_request_model"), fixed("tl_engine_role", ROLES)], TPOT_BUCKETS);
        r.register(REQUEST_DURATION, "End-to-end request time.", Kind::Histogram,
            vec![fixed("gen_ai_operation_name", OPS_E), capped("gen_ai_request_model"), capped("error_type")], REQUEST_BUCKETS);
        r.register(HTTP_DURATION, "Every HTTP request. The availability SLO is the ratio of http_response_status_code >= 500.", Kind::Histogram,
            vec![fixed("http_request_method", &["GET", "POST", "PUT", "DELETE"]), capped("http_route"), capped("http_response_status_code")], HTTP_BUCKETS);
        r.register(KV_BLOCKS, "KV pool blocks by state; free + used + cached equals the pool size.", Kind::Gauge, vec![fixed("state", &["free", "used", "cached"])], &[]);
        r.register(KV_EVICTIONS, "Cached blocks reclaimed.", Kind::Counter, vec![], &[]);
        r.register(QUEUE_DEPTH, "Requests admitted and waiting to run.", Kind::Gauge, vec![], &[]);
        r.register(ACTIVE_SEQUENCES, "Sequences in the running batch.", Kind::Gauge, vec![], &[]);
        r.register(BATCH_TOKENS, "Tokens processed by the last step.", Kind::Gauge, vec![], &[]);
        r.register(PREFIX_HIT_RATIO, "Prompt tokens served from the prefix cache over prompt tokens, since start.", Kind::Gauge, vec![], &[]);
        r.register(SPEC_ACCEPT_RATE, "Accepted draft tokens over proposed draft tokens, since start.", Kind::Gauge, vec![], &[]);
        r.register(PREEMPTIONS, "Requests preempted by recompute.", Kind::Counter, vec![], &[]);
        r.register(KV_TRANSFER_BYTES, "Envelope bytes pushed or received over tl.kv.v1.", Kind::Counter, vec![fixed("direction", &["sent", "received"])], &[]);
        r.register(KV_TRANSFER_BLOCKS, "Blocks moved over tl.kv.v1, by outcome.", Kind::Counter, vec![fixed("outcome", &["sent", "deduped"])], &[]);
        EngineMetrics { reg: r }
    }

    /// One finished request: TTFT (when a token came), TPOT = (E2E - TTFT)
    /// / (output tokens - 1) (when there were at least 2), and the request
    /// duration (with error_type on failure).
    pub fn record_request(&mut self, r: &RequestRecord) -> Result<(), MetricError> {
        let gen = [("gen_ai_operation_name", r.operation.as_str()), ("gen_ai_request_model", r.model.as_str()), ("tl_engine_role", r.role.as_str())];
        if let Some(ttft) = r.ttft_s {
            self.reg.observe(TTFT, &gen, ttft)?;
            if r.output_tokens >= 2 {
                self.reg.observe(TPOT, &gen, (r.e2e_s - ttft) / (r.output_tokens - 1) as f64)?;
            }
        }
        let err = r.error_type.as_deref().unwrap_or("");
        self.reg.observe(REQUEST_DURATION, &[("gen_ai_operation_name", r.operation.as_str()), ("gen_ai_request_model", r.model.as_str()), ("error_type", err)], r.e2e_s)?;
        Ok(())
    }

    /// One HTTP request (every route, health checks included).
    pub fn record_http(&mut self, method: &str, route: &str, status: u16, seconds: f64) -> Result<(), MetricError> {
        let code = status.to_string();
        self.reg.observe(HTTP_DURATION, &[("http_request_method", method), ("http_route", route), ("http_response_status_code", &code)], seconds)
    }

    /// The gauges and running totals from the engine's stats.
    pub fn set_engine(&mut self, g: &EngineGauges) -> Result<(), MetricError> {
        self.reg.set(KV_BLOCKS, &[("state", "free")], g.kv_free as f64)?;
        self.reg.set(KV_BLOCKS, &[("state", "used")], g.kv_used as f64)?;
        self.reg.set(KV_BLOCKS, &[("state", "cached")], g.kv_cached as f64)?;
        self.reg.set_total(KV_EVICTIONS, &[], g.kv_evictions as f64)?;
        self.reg.set(QUEUE_DEPTH, &[], g.queue_depth as f64)?;
        self.reg.set(ACTIVE_SEQUENCES, &[], g.active_sequences as f64)?;
        self.reg.set(BATCH_TOKENS, &[], g.batch_tokens as f64)?;
        self.reg.set(PREFIX_HIT_RATIO, &[], g.prefix_hit_ratio)?;
        self.reg.set(SPEC_ACCEPT_RATE, &[], g.spec_accept_rate)?;
        self.reg.set_total(PREEMPTIONS, &[], g.preemptions as f64)?;
        Ok(())
    }

    /// One KV transfer's blocks and bytes, as the sender (sent) or the
    /// receiver (received) counts them.
    pub fn record_kv_transfer(&mut self, sent: bool, bytes: u64, blocks_sent: u64, blocks_deduped: u64) -> Result<(), MetricError> {
        self.reg.inc(KV_TRANSFER_BYTES, &[("direction", if sent { "sent" } else { "received" })], bytes as f64)?;
        self.reg.inc(KV_TRANSFER_BLOCKS, &[("outcome", "sent")], blocks_sent as f64)?;
        self.reg.inc(KV_TRANSFER_BLOCKS, &[("outcome", "deduped")], blocks_deduped as f64)?;
        Ok(())
    }

    /// The `/metrics` body.
    pub fn render(&self) -> String {
        self.reg.render()
    }
}
