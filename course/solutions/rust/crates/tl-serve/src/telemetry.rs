//! Traces and logs (L10.7): W3C trace context, the engine's span tree, OTLP
//! export, and JSON log lines that carry the trace.
//!
//! One request is one trace (course/contracts/otel/semconv.md): the engine's
//! SERVER span `POST /v1/chat/completions` is the child of the caller's
//! `traceparent` (the gateway's proxy span), and `engine.queue`,
//! `engine.prefill`, and `engine.decode` are its children, with the
//! attributes semconv.md requires. A disaggregated request adds
//! `kv.transfer` (CLIENT on the prefill engine, SERVER on the decode
//! engine); the context crosses gRPC in metadata under the same
//! `traceparent` key ([`inject`], [`extract`]).
//!
//! Export is OTLP/HTTP with the JSON encoding (`POST <endpoint>/v1/traces`,
//! ids as lowercase hex, 64-bit integers as strings), which the OpenTelemetry
//! collector, Jaeger, and Tempo all accept.
//!
//! Chapter: content/courses/build-an-llm/p10-serving/07-serving-metrics-and-tracing.md.

use std::fmt::Write as _;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// A W3C trace context (https://www.w3.org/TR/trace-context/).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TraceContext {
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    /// trace-flags; bit 0 is "sampled".
    pub flags: u8,
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        let _ = write!(s, "{x:02x}");
    }
    s
}

/// Lowercase hex digits only (W3C forbids uppercase).
fn unhex<const N: usize>(s: &str) -> Option<[u8; N]> {
    if s.len() != 2 * N || !s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
        return None;
    }
    let mut out = [0u8; N];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(out)
}

impl TraceContext {
    /// Parses a `traceparent` header: `version-traceid-parentid-flags` with
    /// version 00 (exactly four fields) or a later version (four fields and
    /// optionally more after a '-'), 32 and 16 lowercase hex digits that are
    /// not all zero, 2 hex digits of flags. Version ff, uppercase hex, and
    /// any other shape are invalid (None): the engine then starts a new
    /// trace.
    pub fn parse(header: &str) -> Option<TraceContext> {
        let h = header.trim();
        let parts: Vec<&str> = h.split('-').collect();
        if parts.len() < 4 {
            return None;
        }
        let version = unhex::<1>(parts[0])?[0];
        if version == 0xff || (version == 0 && parts.len() != 4) {
            return None;
        }
        let trace_id = unhex::<16>(parts[1])?;
        let span_id = unhex::<8>(parts[2])?;
        let flags = unhex::<1>(parts[3])?[0];
        if trace_id == [0; 16] || span_id == [0; 8] {
            return None;
        }
        Some(TraceContext { trace_id, span_id, flags })
    }

    /// `00-<trace id>-<span id>-<flags>`.
    pub fn header(&self) -> String {
        format!("00-{}-{}-{:02x}", hex(&self.trace_id), hex(&self.span_id), self.flags)
    }

    pub fn sampled(&self) -> bool {
        self.flags & 1 == 1
    }

    pub fn trace_hex(&self) -> String {
        hex(&self.trace_id)
    }

    pub fn span_hex(&self) -> String {
        hex(&self.span_id)
    }
}

/// Writes the context into outgoing headers or gRPC metadata.
pub fn inject(ctx: &TraceContext, set: &mut dyn FnMut(&str, &str)) {
    set("traceparent", &ctx.header());
}

/// Reads the context from incoming headers or gRPC metadata (keys are
/// lowercase in both); None when absent or invalid.
pub fn extract(get: &dyn Fn(&str) -> Option<String>) -> Option<TraceContext> {
    get("traceparent").and_then(|h| TraceContext::parse(&h))
}

/// Span ids and trace ids from SplitMix64: seeded, so tests see fixed ids;
/// never all zero.
#[derive(Clone, Debug)]
pub struct IdGen {
    state: u64,
}

impl IdGen {
    pub fn new(seed: u64) -> IdGen {
        IdGen { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn span_id(&mut self) -> [u8; 8] {
        loop {
            let v = self.next();
            if v != 0 {
                return v.to_be_bytes();
            }
        }
    }

    pub fn trace_id(&mut self) -> [u8; 16] {
        let mut t = [0u8; 16];
        t[..8].copy_from_slice(&self.span_id());
        t[8..].copy_from_slice(&self.next().to_be_bytes());
        t
    }
}

/// OTLP span kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanKind {
    Internal = 1,
    Server = 2,
    Client = 3,
}

/// An attribute value (OTLP AnyValue subset).
#[derive(Clone, Debug, PartialEq)]
pub enum AttrValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    StrArray(Vec<String>),
}

/// A span event (semconv: `token` every 32 tokens on engine.decode).
#[derive(Clone, Debug, PartialEq)]
pub struct SpanEvent {
    pub name: String,
    pub time_ns: u64,
    pub attributes: Vec<(String, AttrValue)>,
}

/// One finished span.
#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub name: String,
    pub kind: SpanKind,
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    pub parent_span_id: Option<[u8; 8]>,
    pub start_ns: u64,
    pub end_ns: u64,
    pub attributes: Vec<(String, AttrValue)>,
    pub events: Vec<SpanEvent>,
    /// Some(message): status ERROR.
    pub error: Option<String>,
}

impl Span {
    /// A span that continues `parent` (same trace, parent span id = the
    /// context's span id) or, without one, starts a new trace.
    pub fn start(ids: &mut IdGen, name: &str, kind: SpanKind, parent: Option<&TraceContext>, start_ns: u64) -> Span {
        let (trace_id, parent_span_id) = match parent {
            Some(p) => (p.trace_id, Some(p.span_id)),
            None => (ids.trace_id(), None),
        };
        Span { name: name.to_string(), kind, trace_id, span_id: ids.span_id(), parent_span_id, start_ns, end_ns: start_ns, attributes: Vec::new(), events: Vec::new(), error: None }
    }

    /// The context to propagate from this span (sampled).
    pub fn context(&self) -> TraceContext {
        TraceContext { trace_id: self.trace_id, span_id: self.span_id, flags: 1 }
    }

    pub fn attr(&mut self, key: &str, v: AttrValue) {
        self.attributes.push((key.to_string(), v));
    }
}

/// Timestamps (Unix ns) and facts of one engine request.
#[derive(Clone, Debug, PartialEq)]
pub struct RequestTiming {
    /// The route, for the span name `POST <route>` and `http.route`.
    pub route: String,
    /// "chat", "text_completion", or "embeddings".
    pub operation: String,
    pub model: String,
    pub status: u16,
    /// The error code of a failed request (status ERROR, error.type).
    pub error_type: Option<String>,
    pub received_ns: u64,
    /// Admitted to the running batch (end of engine.queue).
    pub admitted_ns: u64,
    /// The first output token was sampled (end of engine.prefill).
    pub first_token_ns: u64,
    pub end_ns: u64,
    pub priority: i32,
    pub prompt_tokens: usize,
    pub prefix_hit_tokens: usize,
    pub chunks: usize,
    /// When each output token was sampled, in order.
    pub token_times_ns: Vec<u64>,
    pub finish_reasons: Vec<String>,
    pub spec_accept_rate: f64,
}

/// Every how many tokens engine.decode records a `token` event.
pub const TOKEN_EVENT_EVERY: usize = 32;

/// The engine's spans for one request: the SERVER span (child of the
/// caller's context, or a new trace) and its children engine.queue
/// [received, admitted], engine.prefill [admitted, first token], and
/// engine.decode [first token, end], with semconv.md's attributes and a
/// `token` event (attribute tl.engine.tokens = tokens so far) at every
/// 32nd token. SERVER first.
pub fn request_spans(ids: &mut IdGen, incoming: Option<&TraceContext>, t: &RequestTiming) -> Vec<Span> {
    let mut server = Span::start(ids, &format!("POST {}", t.route), SpanKind::Server, incoming, t.received_ns);
    server.end_ns = t.end_ns;
    server.attr("http.request.method", AttrValue::Str("POST".into()));
    server.attr("http.route", AttrValue::Str(t.route.clone()));
    server.attr("http.response.status_code", AttrValue::Int(t.status as i64));
    server.attr("gen_ai.operation.name", AttrValue::Str(t.operation.clone()));
    server.attr("gen_ai.request.model", AttrValue::Str(t.model.clone()));
    if t.status >= 500 || t.error_type.is_some() {
        let et = t.error_type.clone().unwrap_or_else(|| t.status.to_string());
        server.attr("error.type", AttrValue::Str(et.clone()));
        server.error = Some(et);
    }
    let ctx = server.context();
    let mut queue = Span::start(ids, "engine.queue", SpanKind::Internal, Some(&ctx), t.received_ns);
    queue.end_ns = t.admitted_ns;
    queue.attr("tl.engine.queue_ms", AttrValue::Float(t.admitted_ns.saturating_sub(t.received_ns) as f64 / 1e6));
    queue.attr("tl.engine.priority", AttrValue::Int(t.priority as i64));
    let mut prefill = Span::start(ids, "engine.prefill", SpanKind::Internal, Some(&ctx), t.admitted_ns);
    prefill.end_ns = t.first_token_ns;
    prefill.attr("gen_ai.usage.input_tokens", AttrValue::Int(t.prompt_tokens as i64));
    prefill.attr("tl.engine.prefix_hit_tokens", AttrValue::Int(t.prefix_hit_tokens as i64));
    prefill.attr("tl.engine.chunks", AttrValue::Int(t.chunks as i64));
    let mut decode = Span::start(ids, "engine.decode", SpanKind::Internal, Some(&ctx), t.first_token_ns);
    decode.end_ns = t.end_ns;
    decode.attr("gen_ai.usage.output_tokens", AttrValue::Int(t.token_times_ns.len() as i64));
    decode.attr("gen_ai.response.finish_reasons", AttrValue::StrArray(t.finish_reasons.clone()));
    decode.attr("tl.engine.spec_accept_rate", AttrValue::Float(t.spec_accept_rate));
    for (i, &ts) in t.token_times_ns.iter().enumerate() {
        if (i + 1) % TOKEN_EVENT_EVERY == 0 {
            decode.events.push(SpanEvent { name: "token".into(), time_ns: ts, attributes: vec![("tl.engine.tokens".into(), AttrValue::Int(i as i64 + 1))] });
        }
    }
    vec![server, queue, prefill, decode]
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn any_value(v: &AttrValue) -> String {
    match v {
        AttrValue::Str(s) => format!("{{\"stringValue\":{}}}", json_str(s)),
        AttrValue::Int(i) => format!("{{\"intValue\":\"{i}\"}}"),
        AttrValue::Float(f) if f.is_finite() => format!("{{\"doubleValue\":{f}}}"),
        AttrValue::Float(_) => "{\"doubleValue\":0}".to_string(),
        AttrValue::Bool(b) => format!("{{\"boolValue\":{b}}}"),
        AttrValue::StrArray(a) => format!(
            "{{\"arrayValue\":{{\"values\":[{}]}}}}",
            a.iter().map(|s| format!("{{\"stringValue\":{}}}", json_str(s))).collect::<Vec<_>>().join(",")
        ),
    }
}

fn attrs_json(a: &[(String, AttrValue)]) -> String {
    format!("[{}]", a.iter().map(|(k, v)| format!("{{\"key\":{},\"value\":{}}}", json_str(k), any_value(v))).collect::<Vec<_>>().join(","))
}

/// An OTLP/HTTP JSON ExportTraceServiceRequest: one resource with
/// `resource` attributes (service.name first), one scope, the spans.
pub fn otlp_json(resource: &[(String, AttrValue)], scope: &str, spans: &[Span]) -> String {
    let mut items = Vec::with_capacity(spans.len());
    for s in spans {
        let mut o = format!(
            "{{\"traceId\":\"{}\",\"spanId\":\"{}\"",
            hex(&s.trace_id),
            hex(&s.span_id)
        );
        if let Some(p) = s.parent_span_id {
            let _ = write!(o, ",\"parentSpanId\":\"{}\"", hex(&p));
        }
        let _ = write!(
            o,
            ",\"name\":{},\"kind\":{},\"startTimeUnixNano\":\"{}\",\"endTimeUnixNano\":\"{}\",\"attributes\":{}",
            json_str(&s.name),
            s.kind as i32,
            s.start_ns,
            s.end_ns,
            attrs_json(&s.attributes)
        );
        if !s.events.is_empty() {
            let ev: Vec<String> = s
                .events
                .iter()
                .map(|e| format!("{{\"timeUnixNano\":\"{}\",\"name\":{},\"attributes\":{}}}", e.time_ns, json_str(&e.name), attrs_json(&e.attributes)))
                .collect();
            let _ = write!(o, ",\"events\":[{}]", ev.join(","));
        }
        if let Some(m) = &s.error {
            let _ = write!(o, ",\"status\":{{\"code\":2,\"message\":{}}}", json_str(m));
        }
        o.push('}');
        items.push(o);
    }
    format!(
        "{{\"resourceSpans\":[{{\"resource\":{{\"attributes\":{}}},\"scopeSpans\":[{{\"scope\":{{\"name\":{}}},\"spans\":[{}]}}]}}]}}",
        attrs_json(resource),
        json_str(scope),
        items.join(",")
    )
}

/// The resource of an engine (semconv.md): service.name
/// `<system>-engine`, service.namespace `<system>`, service.version, and
/// tl.engine.role.
pub fn engine_resource(system: &str, version: &str, role: &str) -> Vec<(String, AttrValue)> {
    vec![
        ("service.name".into(), AttrValue::Str(format!("{system}-engine"))),
        ("service.namespace".into(), AttrValue::Str(system.to_string())),
        ("service.version".into(), AttrValue::Str(version.to_string())),
        ("tl.engine.role".into(), AttrValue::Str(role.to_string())),
    ]
}

/// POSTs one OTLP JSON body to `<endpoint>/v1/traces` (endpoint
/// `http://host:port`) over a plain HTTP/1.1 connection and returns the
/// response status. The call is bounded by `timeout`; export failures are
/// the caller's to count and drop, never to retry inside a request.
pub fn export(endpoint: &str, body: &str, timeout: Duration) -> Result<u16, String> {
    let rest = endpoint.strip_prefix("http://").ok_or_else(|| format!("endpoint {endpoint:?} is not http://"))?;
    let host = rest.trim_end_matches('/');
    let addr = host.to_socket_addrs().map_err(|e| format!("{host}: {e}"))?.next().ok_or_else(|| format!("{host}: no address"))?;
    let mut s = TcpStream::connect_timeout(&addr, timeout).map_err(|e| format!("connect {host}: {e}"))?;
    s.set_read_timeout(Some(timeout)).map_err(|e| e.to_string())?;
    s.set_write_timeout(Some(timeout)).map_err(|e| e.to_string())?;
    let req = format!(
        "POST /v1/traces HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    s.write_all(req.as_bytes()).and_then(|_| s.write_all(body.as_bytes())).map_err(|e| format!("write: {e}"))?;
    let mut buf = Vec::new();
    let _ = s.read_to_end(&mut buf);
    let head = String::from_utf8_lossy(&buf);
    let status = head.split_whitespace().nth(1).and_then(|c| c.parse::<u16>().ok()).ok_or_else(|| format!("bad response {:?}", head.lines().next()))?;
    Ok(status)
}

/// One JSON log line (semconv.md "Logs"): ts, level, msg, service, and,
/// inside a span, trace_id and span_id, so `grep <trace_id>` finds every
/// line of a request. No prompt text belongs in `msg`.
pub fn log_line(ts_rfc3339: &str, level: &str, msg: &str, service: &str, ctx: Option<&TraceContext>) -> String {
    let mut o = format!("{{\"ts\":{},\"level\":{},\"msg\":{},\"service\":{}", json_str(ts_rfc3339), json_str(level), json_str(msg), json_str(service));
    if let Some(c) = ctx {
        let _ = write!(o, ",\"trace_id\":\"{}\",\"span_id\":\"{}\"", c.trace_hex(), c.span_hex());
    }
    o.push('}');
    o
}
