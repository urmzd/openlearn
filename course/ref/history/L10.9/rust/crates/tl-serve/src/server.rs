//! The engine server (L10.5): API v1 (engine tier) over HTTP/1.1 and SSE,
//! on tokio and hyper, in front of one `Engine`. L10.7 takes it over to
//! measure it: every request is recorded in the `EngineMetrics` of
//! otel/metrics.yaml (served on the health port's `/metrics`) and becomes
//! one trace of otel/semconv.md (the SERVER span and its engine.queue,
//! engine.prefill, and engine.decode children, exported as OTLP/HTTP JSON
//! when `OTEL_EXPORTER_OTLP_ENDPOINT` is set). L10.9 takes it over to
//! route tool calls: `tools`, `tool_choice`, and `response_format` are read
//! from the raw body (tools.rs), the prompt describes the tools, a request
//! with a grammar is decoded under its `Constraint` (constrain.rs), and the
//! answer is parsed into `tool_calls` (a message, or stream deltas).
//!
//! Two worlds meet here. HTTP is I/O bound: thousands of connections that
//! mostly wait, which async tasks on a few threads handle well (lang.09).
//! The model is CPU bound: one step at a time, never to be interrupted by
//! an `.await`. So the engine runs on its own OS thread, and the two talk
//! through channels:
//!
//! ```text
//! handler task --try_send(Cmd)--> [bounded admission queue] --> engine thread
//! handler task <--Ev (tokens)---- [one unbounded channel per request] ---
//! ```
//!
//! - **Admission is bounded.** A full queue answers 429 with `Retry-After`
//!   at once instead of letting latency grow without limit.
//! - **Disconnect aborts.** A request's event channel closes when its
//!   handler or SSE body is dropped (the client went away, or a stop string
//!   ended it); the engine checks before every step and aborts, so the
//!   request's KV blocks return within one step.
//! - **SIGTERM drains.** Listeners close, `/readyz` turns 503, new requests
//!   on open connections get 503, in-flight requests finish, then the
//!   process exits 0 (spec/cli-roles.md).
//!
//! `--config <runtime.toml>` reads `[engine]` (config/runtime.schema.json)
//! with `TL_ENGINE__<KEY>` environment overrides for scalar keys.

use std::collections::HashMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full};
use hyper::body::{Body, Frame, Incoming};
use hyper::header::HeaderValue;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use serde_json::Value;
use tokio::sync::{mpsc, oneshot, watch};

use tl_engine::constrain::{constrained_sample, regex_to_dfa, Constraint, Json, TokenIndex, DEAD};
use tl_engine::engine::{Engine, EngineStats, GenRequest};
use tl_engine::model::TokenizerKind;
use tl_engine::runner::{EngineConfig, KvConfig, ModelRunner, PrefixCache, Quant, SchedPolicy};
use tl_engine::sample::{mix64, sample, stream, PURPOSE_SAMPLE};
use tl_engine::spec::{self, RunnerTarget, SpecConfig, Target};
use tl_engine::sched::{AdmitError, FinishReason, RequestId};
use tl_tok::{ByteBpe, ByteTokenizer, Tokenizer};

use crate::metrics::{EngineGauges, EngineMetrics, RequestRecord};
use crate::openai::{self, ApiError, GenSpec, Prompt};
use crate::sse::{self, TextStream};
use crate::telemetry::{self, AttrValue, IdGen, RequestTiming, Span, SpanKind, TraceContext};
use crate::template::{Template, DEFAULT_TEMPLATE};
use crate::tools::{self, ToolChoice, ToolEvent, ToolParser, ToolRequest};

/// The response body: a whole JSON document or an SSE stream.
pub type RespBody = BoxBody<Bytes, Infallible>;

/// What the server needs from runtime.toml.
#[derive(Clone, Debug, PartialEq)]
pub struct ServeConfig {
    pub model_dir: PathBuf,
    pub model_id: String,
    pub http_listen: String,
    pub health_listen: String,
    pub engine: EngineConfig,
    pub queue_capacity: usize,
    pub queue_deadline_ms: u64,
    /// Optional request-local speculative decoder (L10.8).
    pub speculative: Option<SpecConfig>,
    /// Exit 0 on SIGTERM after draining (the binary sets it; tests do not).
    pub handle_signals: bool,
    /// Sleep after every engine step (tests use it to hold requests in
    /// flight; never set from runtime.toml).
    pub step_delay: Duration,
}

/// `":8000"` (every interface) or `"host:port"` as a socket address.
pub fn listen_addr(s: &str) -> Result<SocketAddr, String> {
    // SOLUTION-BEGIN L10.9
    let full = if s.starts_with(':') { format!("0.0.0.0{s}") } else { s.to_string() };
    full.parse().map_err(|_| format!("listen address {s:?} is not host:port"))
    // SOLUTION-END
}

/// The model id: `model_id`, else the model directory's parent name and
/// then its own (`models/<id>/<version>/`).
fn default_model_id(dir: &Path) -> String {
    // SOLUTION-BEGIN L10.9
    let name = |p: &Path| p.file_name().map(|n| n.to_string_lossy().to_string());
    let own = name(dir).unwrap_or_else(|| "model".to_string());
    // models/<id>/v3/ is served as <id>; any other directory by its own name
    let is_version = own.len() > 1 && own.starts_with('v') && own[1..].chars().all(|c| c.is_ascii_digit());
    match dir.parent().and_then(name) {
        Some(parent) if is_version => parent,
        _ => own,
    }
    // SOLUTION-END
}

impl ServeConfig {
    /// Defaults of config/runtime.schema.json for a model directory.
    pub fn for_model(dir: &Path) -> ServeConfig {
        // SOLUTION-BEGIN L10.9
        ServeConfig {
            model_dir: dir.to_path_buf(),
            model_id: default_model_id(dir),
            http_listen: ":8000".to_string(),
            health_listen: ":9464".to_string(),
            engine: EngineConfig {
                max_batch_tokens: 2048,
                max_seqs: 64,
                prefill_chunk: 512,
                prefix_cache: PrefixCache::Radix,
                kv: KvConfig { blocks: 2048, block_size: 16 },
                policy: SchedPolicy::Priority,
                threads: 0,
                quant: None,
            },
            queue_capacity: 256,
            queue_deadline_ms: 30000,
            speculative: None,
            handle_signals: false,
            step_delay: Duration::ZERO,
        }
        // SOLUTION-END
    }

    /// Parses runtime.toml's `[engine]`, then applies `TL_ENGINE__<KEY>`
    /// overrides from `env`. Unknown keys and values this engine cannot
    /// serve (roles other than unified, KV format 2, int8) are errors.
    pub fn from_toml(text: &str, env: &[(String, String)]) -> Result<ServeConfig, String> {
        // SOLUTION-BEGIN L10.9
        let doc: toml::Table = text.parse().map_err(|e| format!("runtime.toml: {e}"))?;
        let mut eng = doc.get("engine").and_then(|v| v.as_table()).cloned().ok_or("runtime.toml: no [engine] table")?;
        for (k, v) in env {
            let Some(key) = k.strip_prefix("TL_ENGINE__") else { continue };
            let key = key.to_ascii_lowercase();
            let parsed = match v.parse::<i64>() {
                Ok(n) => toml::Value::Integer(n),
                Err(_) => toml::Value::String(v.clone()),
            };
            eng.insert(key, parsed);
        }
        const KNOWN: [&str; 21] = [
            "role", "model_dir", "model_id", "http_listen", "health_listen", "grpc_listen", "kv_listen", "gateway_registry", "advertise_host",
            "kv_blocks", "block_size", "kv_format", "prefix_cache", "max_batch_tokens", "max_seqs", "prefill_chunk", "queue_capacity",
            "queue_deadline_ms", "threads", "quant", "speculative",
        ];
        if let Some(k) = eng.keys().find(|k| !KNOWN.contains(&k.as_str())) {
            return Err(format!("runtime.toml: unknown key engine.{k}"));
        }
        let s = |k: &str| eng.get(k).and_then(|v| v.as_str()).map(str::to_string);
        let n = |k: &str, d: i64, lo: i64| -> Result<usize, String> {
            match eng.get(k) {
                None => Ok(d as usize),
                Some(v) => match v.as_integer() {
                    Some(x) if x >= lo => Ok(x as usize),
                    _ => Err(format!("runtime.toml: engine.{k} must be an integer >= {lo}")),
                },
            }
        };
        let model_dir = PathBuf::from(s("model_dir").ok_or("runtime.toml: engine.model_dir is required")?);
        let mut c = ServeConfig::for_model(&model_dir);
        if let Some(id) = s("model_id") {
            c.model_id = id;
        }
        if let Some(r) = s("role") {
            if r != "unified" {
                return Err(format!("runtime.toml: engine.role {r:?} needs disaggregation (L10.6)"));
            }
        }
        if n("kv_format", 1, 1)? != 1 {
            return Err("runtime.toml: engine.kv_format 2 arrives with the craft.13 migration".to_string());
        }
        c.http_listen = s("http_listen").unwrap_or(c.http_listen);
        c.health_listen = s("health_listen").unwrap_or(c.health_listen);
        listen_addr(&c.http_listen)?;
        listen_addr(&c.health_listen)?;
        c.engine.kv = KvConfig { blocks: n("kv_blocks", 2048, 1)?, block_size: n("block_size", 16, 1)? };
        c.engine.max_batch_tokens = n("max_batch_tokens", 2048, 1)?;
        c.engine.max_seqs = n("max_seqs", 64, 1)?;
        c.engine.prefill_chunk = n("prefill_chunk", 512, 1)?;
        c.engine.threads = n("threads", 0, 0)?;
        c.queue_capacity = n("queue_capacity", 256, 1)?;
        c.queue_deadline_ms = n("queue_deadline_ms", 30000, 1)? as u64;
        c.engine.prefix_cache = match s("prefix_cache").as_deref() {
            None | Some("radix") => PrefixCache::Radix,
            Some("hash") => PrefixCache::Hash,
            Some("none") => PrefixCache::None,
            Some(o) => return Err(format!("runtime.toml: engine.prefix_cache {o:?} is not none, hash, or radix")),
        };
        c.engine.quant = match s("quant").as_deref() {
            None | Some("none") => None,
            Some("int4") => Some(Quant::Int4 { group: 32 }),
            Some(o) => return Err(format!("runtime.toml: engine.quant {o:?} is not supported by this engine (none, int4)")),
        };
        c.speculative = match eng.get("speculative").and_then(|v| v.as_table()) {
            None => None,
            Some(sp) => {
                let draft = sp.get("draft").and_then(|v| v.as_str()).unwrap_or("none");
                let k = sp.get("k").and_then(|v| v.as_integer()).unwrap_or(4);
                Some(SpecConfig::parse(draft, k).map_err(|e| format!("runtime.toml: {e}"))?)
            }
        };
        Ok(c)
        // SOLUTION-END
    }

    /// Reads a runtime.toml file with the process environment's overrides.
    pub fn load(path: &Path) -> Result<ServeConfig, String> {
        // SOLUTION-BEGIN L10.9
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let env: Vec<(String, String)> = std::env::vars().collect();
        ServeConfig::from_toml(&text, &env)
        // SOLUTION-END
    }
}

/// The chat template, special-token strings, and EOS ids of a model
/// directory: generation_config.json first, then tokenizer_config.json,
/// then config.json's EOS ids and the default template.
pub fn chat_settings(dir: &Path, config_eos: &[u32]) -> Result<(Template, String, String, Vec<u32>), String> {
    // SOLUTION-BEGIN L10.9
    let read = |name: &str| -> Option<Value> { std::fs::read_to_string(dir.join(name)).ok().and_then(|t| serde_json::from_str(&t).ok()) };
    let gen = read("generation_config.json").unwrap_or(Value::Null);
    let tokc = read("tokenizer_config.json").unwrap_or(Value::Null);
    let text = |v: Option<&Value>| -> Option<String> {
        match v? {
            Value::String(s) => Some(s.clone()),
            Value::Object(o) => o.get("content").and_then(Value::as_str).map(str::to_string),
            _ => None,
        }
    };
    let src = text(gen.get("chat_template")).or_else(|| text(tokc.get("chat_template"))).unwrap_or_else(|| DEFAULT_TEMPLATE.to_string());
    let template = Template::parse(&src)?;
    let bos = text(gen.get("bos_token")).or_else(|| text(tokc.get("bos_token"))).unwrap_or_default();
    let eos = text(gen.get("eos_token")).or_else(|| text(tokc.get("eos_token"))).unwrap_or_default();
    let eos_ids = match gen.get("eos_token_id") {
        Some(Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_u64).map(|x| x as u32).collect(),
        _ => config_eos.to_vec(),
    };
    Ok((template, bos, eos, eos_ids))
    // SOLUTION-END
}

/// Mean of `hidden` [T, d] over T, divided by its L2 norm (/v1/embeddings).
pub fn pool_embedding(hidden: &[f32], d: usize) -> Vec<f32> {
    // SOLUTION-BEGIN L10.9
    let t = hidden.len() / d.max(1);
    let mut m = vec![0.0f64; d];
    for row in hidden.chunks_exact(d) {
        for (a, x) in m.iter_mut().zip(row) {
            *a += *x as f64;
        }
    }
    for a in &mut m {
        *a /= t.max(1) as f64;
    }
    let norm = m.iter().map(|x| x * x).sum::<f64>().sqrt();
    m.iter().map(|x| if norm > 0.0 { (x / norm) as f32 } else { 0.0 }).collect()
    // SOLUTION-END
}

/// A command to the engine thread.
enum Cmd {
    Generate { req: GenRequest, events: mpsc::UnboundedSender<Ev>, constraint: Option<Constrained> },
    Embed { inputs: Vec<Vec<u32>>, reply: oneshot::Sender<Result<Vec<Vec<f32>>, String>> },
}

/// A request decoded under a grammar (L10.9).
struct Constrained {
    /// Shared by every request with the same grammar.
    index: Arc<TokenIndex>,
    /// Some: free text until the output contains it (tool_choice auto).
    trigger: Option<String>,
}

/// The validated tool fields of a chat request and its raw messages (keys
/// in the order written).
struct ToolReq {
    req: ToolRequest,
    messages: Json,
}

/// An event from the engine thread to one request's handler. Times are
/// Unix nanoseconds stamped on the engine thread.
#[derive(Debug)]
enum Ev {
    /// The engine thread took the request off the admission queue.
    Admitted { at_ns: u64 },
    Rejected(ApiError),
    Token { token: u32, logprob: f64, finish: Option<FinishReason>, at_ns: u64 },
    Failed(String),
}

/// State every handler shares.
struct Shared {
    cmds: SyncSender<Cmd>,
    tok: Arc<dyn Tokenizer>,
    template: Template,
    bos: String,
    eos: String,
    eos_ids: Vec<u32>,
    model_id: String,
    hidden: usize,
    created: u64,
    max_model_len: usize,
    draining: AtomicBool,
    ready: AtomicBool,
    inflight: AtomicUsize,
    seq: AtomicU64,
    stats: Mutex<EngineStats>,
    /// Every engine instrument of otel/metrics.yaml (L10.7).
    metrics: Mutex<EngineMetrics>,
    /// Span and trace ids.
    ids: Mutex<IdGen>,
    /// Draft tokens proposed and accepted by the speculative path, since start.
    spec: Mutex<(u64, u64)>,
    /// `[engine].prefill_chunk`, for tl.engine.chunks.
    prefill_chunk: usize,
    /// Logit rows per step (the model's vocabulary, padding included).
    vocab: usize,
    /// One TokenIndex per grammar pattern (masks are cached inside it).
    grammars: Mutex<HashMap<String, Arc<TokenIndex>>>,
}

/// Unix time in nanoseconds.
fn now_ns() -> u64 {
    // SOLUTION-BEGIN L10.9
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0)
    // SOLUTION-END
}

/// One live request on the engine thread.
struct Live {
    tx: mpsc::UnboundedSender<Ev>,
    since: Instant,
    started: bool,
}

/// The AdmitError of the scheduler as an HTTP error.
fn admit_error(e: AdmitError) -> ApiError {
    // SOLUTION-BEGIN L10.9
    match e {
        AdmitError::QueueFull => ApiError::queue_full(),
        AdmitError::Empty => ApiError::invalid(Some("messages"), "the prompt is empty"),
        AdmitError::TooLong { tokens, limit } => ApiError::context_length(format!("the prompt has {tokens} tokens; the context is {limit}")),
        AdmitError::NeverFits { blocks, total } => ApiError::context_length(format!("the request needs {blocks} KV blocks; the pool has {total}")),
    }
    // SOLUTION-END
}

/// Decodes one request under its grammar on the engine thread, outside the
/// batch as the speculative path is: the request's own KV blocks
/// (spec::RunnerTarget), L10.1's sampler on masked logits
/// (constrain::constrained_sample), and the request's generator
/// `stream(seed, sample)`. With a trigger (tool_choice auto) decoding is
/// free until the output contains it; the constraint then starts in the
/// state the bytes after the trigger lead to (none when they already left
/// the grammar). A grammar complete with no token left to allow (no EOS in
/// the vocabulary) ends the answer with Stop. `emit` gets each token, its
/// logprob, and the finish (on the last only); false means the client left.
fn constrained_generate(runner: &mut ModelRunner, c: &Constrained, req: &GenRequest, tok: &dyn Tokenizer, emit: &mut dyn FnMut(u32, f64, Option<FinishReason>) -> bool) -> Result<(), String> {
    // SOLUTION-BEGIN L10.9
    let mut target = RunnerTarget::new(runner);
    let mut rng = stream(req.seed, PURPOSE_SAMPLE);
    let mut cons = c.trigger.is_none().then(|| Constraint::new(Arc::clone(&c.index)));
    let trigger = c.trigger.as_deref().unwrap_or("").as_bytes();
    let mut text: Vec<u8> = Vec::new();
    let mut triggered = false;
    let mut out: Vec<u32> = Vec::new();
    let mut pending: Option<(u32, f64)> = None;
    let mut row = target.extend(&req.prompt).map_err(|e| e.to_string())?.pop().ok_or("the prompt gave no logits")?;
    let finish = loop {
        if out.len() >= req.max_tokens {
            break FinishReason::Length;
        }
        let (t, lp) = match cons.as_mut() {
            Some(k) if !k.mask().iter().any(|&m| m) => break FinishReason::Stop,
            Some(k) => constrained_sample(&row, k, &req.params, &req.prompt, &out, &mut rng).map_err(|e| e.to_string())?,
            None => sample(&row, &req.params, &req.prompt, &out, &mut rng),
        };
        if let Some((p, plp)) = pending.take() {
            if !emit(p, plp, None) {
                return Ok(());
            }
        }
        if req.stop_ids.contains(&t) {
            emit(t, lp, Some(FinishReason::Stop));
            return Ok(());
        }
        out.push(t);
        pending = Some((t, lp));
        if cons.is_none() && !triggered && !trigger.is_empty() {
            text.extend_from_slice(tok.token_bytes(t).unwrap_or(&[]));
            if let Some(at) = text.windows(trigger.len()).position(|w| w == trigger) {
                let state = c.index.dfa.step(0, &text[at + trigger.len()..]).unwrap_or(DEAD);
                if state != DEAD {
                    cons = Some(Constraint { index: Arc::clone(&c.index), state, done: false });
                }
                triggered = true;
            }
        }
        if out.len() < req.max_tokens {
            row = target.extend(&[t]).map_err(|e| e.to_string())?.pop().ok_or("a step gave no logits")?;
        }
    };
    if let Some((p, plp)) = pending {
        emit(p, plp, Some(finish));
    }
    Ok(())
    // SOLUTION-END
}

/// The engine thread: take commands, drop requests whose client is gone
/// or whose queue deadline passed, run one step, deliver its tokens.
fn engine_loop(mut engine: Engine, rx: Receiver<Cmd>, shared: Arc<Shared>, stop: Arc<AtomicBool>, deadline: Duration, step_delay: Duration, speculative: Option<SpecConfig>) {
    // SOLUTION-BEGIN L10.9
    let mut live: HashMap<RequestId, Live> = HashMap::new();
    loop {
        if stop.load(Ordering::SeqCst) {
            // the drain is over: whatever still runs is cut
            engine.abort_all();
            for (_, l) in live.drain() {
                let _ = l.tx.send(Ev::Failed("the engine stopped".to_string()));
            }
            return;
        }
        loop {
            let cmd = if engine.has_work() {
                match rx.try_recv() {
                    Ok(c) => c,
                    Err(_) => break,
                }
            } else {
                match rx.recv_timeout(Duration::from_millis(20)) {
                    Ok(c) => c,
                    Err(RecvTimeoutError::Timeout) if stop.load(Ordering::SeqCst) => return,
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            };
            match cmd {
                Cmd::Generate { req, events, constraint: Some(c) } => {
                    let _ = events.send(Ev::Admitted { at_ns: now_ns() });
                    let tok = Arc::clone(&shared.tok);
                    let mut emit = |token, logprob, finish| events.send(Ev::Token { token, logprob, finish, at_ns: now_ns() }).is_ok();
                    if let Err(e) = constrained_generate(engine.runner_mut(), &c, &req, &*tok, &mut emit) {
                        let _ = events.send(Ev::Failed(e));
                    }
                }
                Cmd::Generate { req, events, .. } if speculative.is_some() => {
                    let cfg = speculative.as_ref().expect("checked above");
                    let _ = events.send(Ev::Admitted { at_ns: now_ns() });
                    let result = spec::generate_with_runner(engine.runner_mut(), cfg, &req.prompt, &req.params, req.seed, req.max_tokens, &req.stop_ids);
                    match result {
                        Ok(g) => {
                            if let Ok(mut s) = shared.spec.lock() {
                                s.0 += g.stats.drafted as u64;
                                s.1 += g.stats.accepted as u64;
                            }
                            let finish = Some(match g.finish { spec::Finish::Stop => FinishReason::Stop, spec::Finish::Length => FinishReason::Length });
                            let at_ns = now_ns();
                            for (i, (&token, &logprob)) in g.tokens.iter().zip(&g.logprobs).enumerate() {
                                let end = (i + 1 == g.tokens.len()).then_some(finish).flatten();
                                if events.send(Ev::Token { token, logprob, finish: end, at_ns }).is_err() { break; }
                            }
                        }
                        Err(e) => { let _ = events.send(Ev::Failed(e.to_string())); }
                    }
                }
                Cmd::Generate { req, events, .. } => match engine.add_request(req) {
                    Ok(id) => {
                        let _ = events.send(Ev::Admitted { at_ns: now_ns() });
                        live.insert(id, Live { tx: events, since: Instant::now(), started: false });
                    }
                    Err(e) => {
                        let _ = events.send(Ev::Rejected(admit_error(e)));
                    }
                },
                Cmd::Embed { inputs, reply } => {
                    let d = shared.hidden;
                    let out: Result<Vec<Vec<f32>>, String> = inputs
                        .iter()
                        .map(|t| engine.runner_mut().run_once(t).map(|(h, _)| pool_embedding(&h, d)).map_err(|e| e.to_string()))
                        .collect();
                    let _ = reply.send(out);
                }
            }
        }
        // clients that left, and requests that waited too long for KV
        let mut drop_ids = Vec::new();
        for (&id, l) in &live {
            if l.tx.is_closed() {
                drop_ids.push((id, None));
            } else if !l.started && l.since.elapsed() > deadline {
                drop_ids.push((id, Some(ApiError::no_capacity("no KV blocks became free before the queue deadline"))));
            }
        }
        for (id, err) in drop_ids {
            engine.abort(id);
            if let (Some(l), Some(e)) = (live.remove(&id), err) {
                let _ = l.tx.send(Ev::Rejected(e));
            }
        }
        if engine.has_work() {
            match engine.step() {
                Ok(outs) => {
                    let at_ns = now_ns();
                    for o in outs {
                        let Some(l) = live.get_mut(&o.id) else {
                            engine.abort(o.id);
                            continue;
                        };
                        l.started = true;
                        let sent = l.tx.send(Ev::Token { token: o.token, logprob: o.logprob, finish: o.finish, at_ns }).is_ok();
                        if o.finish.is_some() {
                            live.remove(&o.id);
                        } else if !sent {
                            engine.abort(o.id);
                            live.remove(&o.id);
                        }
                    }
                }
                Err(e) => {
                    // a failed step fails every request in flight; the
                    // engine itself keeps serving
                    engine.abort_all();
                    for (_, l) in live.drain() {
                        let _ = l.tx.send(Ev::Failed(e.to_string()));
                    }
                }
            }
            if !step_delay.is_zero() {
                std::thread::sleep(step_delay);
            }
        }
        if let Ok(mut s) = shared.stats.lock() {
            *s = engine.stats();
        }
    }
    // SOLUTION-END
}

/// The SSE body: frames arrive on a channel from the producer task. When
/// the client goes away hyper drops this body, the producer's next send
/// fails, and the producer drops the request's event channel.
struct SseBody {
    rx: mpsc::UnboundedReceiver<Bytes>,
}

impl Body for SseBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        // SOLUTION-BEGIN L10.9
        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(b)) => Poll::Ready(Some(Ok(Frame::data(b)))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
        // SOLUTION-END
    }
}

/// Decrements the in-flight count when a request (or its stream) ends.
struct Inflight(Arc<Shared>);

impl Drop for Inflight {
    fn drop(&mut self) {
        // SOLUTION-BEGIN L10.9
        self.0.inflight.fetch_sub(1, Ordering::SeqCst);
        // SOLUTION-END
    }
}

fn full(status: StatusCode, content_type: &str, body: String) -> Response<RespBody> {
    // SOLUTION-BEGIN L10.9
    let mut r = Response::new(Full::new(Bytes::from(body)).boxed());
    *r.status_mut() = status;
    r.headers_mut().insert("content-type", HeaderValue::from_str(content_type).expect("ascii"));
    r
    // SOLUTION-END
}

/// An error in the OpenAI shape; 429 also carries `Retry-After`.
fn error_response(e: &ApiError) -> Response<RespBody> {
    // SOLUTION-BEGIN L10.9
    let mut r = full(StatusCode::from_u16(e.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), "application/json", e.to_json());
    if e.status == 429 {
        r.headers_mut().insert("retry-after", HeaderValue::from_static("1"));
    }
    r
    // SOLUTION-END
}

/// A fresh hex id.
fn new_id(shared: &Shared) -> String {
    // SOLUTION-BEGIN L10.9
    let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
    format!("{:016x}", mix64(t ^ shared.seq.fetch_add(1, Ordering::SeqCst).wrapping_mul(0x9E3779B97F4A7C15)))
    // SOLUTION-END
}

fn unix_now() -> u64 {
    // SOLUTION-BEGIN L10.9
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
    // SOLUTION-END
}

/// What the measurements need about one API request.
struct Obs {
    method: String,
    /// The route label (`route_of`).
    route: &'static str,
    received: Instant,
    received_ns: u64,
    /// The caller's `traceparent`, when valid.
    parent: Option<TraceContext>,
}

/// The route label of a path: the routes this server answers, the model
/// route with its parameter, and `_other` for everything else (a capped
/// label must not grow with arbitrary paths).
fn route_of(path: &str) -> &'static str {
    // SOLUTION-BEGIN L10.9
    match path {
        "/v1/chat/completions" => "/v1/chat/completions",
        "/v1/completions" => "/v1/completions",
        "/v1/embeddings" => "/v1/embeddings",
        "/v1/tokenize" => "/v1/tokenize",
        "/v1/models" => "/v1/models",
        "/healthz" => "/healthz",
        "/readyz" => "/readyz",
        "/metrics" => "/metrics",
        p if p.starts_with("/v1/models/") => "/v1/models/{model}",
        _ => "_other",
    }
    // SOLUTION-END
}

/// How a generation ended.
#[derive(Default)]
struct Outcome {
    status: u16,
    /// The error `code` (or type) of a failed request.
    error_type: Option<String>,
    prompt_tokens: usize,
    priority: i32,
    /// When the engine thread took the request; None when it never got there.
    admitted_ns: Option<u64>,
    /// When each output token was sampled.
    token_times_ns: Vec<u64>,
    finish: Option<String>,
}

impl Outcome {
    fn failed(e: &ApiError) -> Outcome {
        // SOLUTION-BEGIN L10.9
        Outcome { status: e.status, error_type: Some(e.code.map(str::to_string).unwrap_or_else(|| e.kind.to_string())), ..Outcome::default() }
        // SOLUTION-END
    }
}

/// The accept rate of the speculative path since start (0 without it).
fn spec_rate(shared: &Shared) -> f64 {
    // SOLUTION-BEGIN L10.9
    match shared.spec.lock() {
        Ok(s) if s.0 > 0 => s.1 as f64 / s.0 as f64,
        _ => 0.0,
    }
    // SOLUTION-END
}

/// A Unix time in nanoseconds as RFC 3339 UTC with milliseconds.
fn rfc3339(ns: u64) -> String {
    // SOLUTION-BEGIN L10.9
    let secs = ns / 1_000_000_000;
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // civil date from days since 1970-01-01 (Hinnant's algorithm)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{:03}Z", rem / 3600, rem / 60 % 60, rem % 60, ns / 1_000_000 % 1000)
    // SOLUTION-END
}

/// Ends a request's telemetry: one JSON log line on stderr inside its
/// SERVER span (otel/semconv.md "Logs": trace_id and span_id, never the
/// prompt), then the spans posted to `$OTEL_EXPORTER_OTLP_ENDPOINT/v1/traces`
/// on a thread of their own (never on the request path) when it is set.
fn export_spans(spans: Vec<Span>) {
    // SOLUTION-BEGIN L10.9
    let service = std::env::var("OTEL_SERVICE_NAME").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "tl-engine".to_string());
    if let Some(root) = spans.first() {
        let status = root.attributes.iter().find(|(k, _)| k == "http.response.status_code").map(|(_, v)| match v {
            AttrValue::Int(i) => i.to_string(),
            _ => String::new(),
        });
        let msg = format!("{} {}", root.name, status.unwrap_or_default());
        let level = if root.error.is_some() { "error" } else { "info" };
        eprintln!("{}", telemetry::log_line(&rfc3339(now_ns()), level, &msg, &service, Some(&root.context())));
    }
    let Some(endpoint) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok().filter(|s| !s.is_empty()) else {
        return;
    };
    let system = service.strip_suffix("-engine").unwrap_or(&service).to_string();
    let body = telemetry::otlp_json(&telemetry::engine_resource(&system, env!("CARGO_PKG_VERSION"), "unified"), "tl-serve", &spans);
    std::thread::spawn(move || {
        if let Err(e) = telemetry::export(&endpoint, &body, Duration::from_secs(2)) {
            eprintln!("tl-serve: span export failed: {e}");
        }
    });
    // SOLUTION-END
}

/// The SERVER span alone, for a request that never reached the engine
/// (validation errors, embeddings, tokenize, models). Only a 5xx is ERROR.
fn server_span(shared: &Shared, obs: &Obs, operation: Option<&str>, status: u16, error_type: Option<&str>) -> Span {
    // SOLUTION-BEGIN L10.9
    let mut ids = shared.ids.lock().unwrap_or_else(|p| p.into_inner());
    let mut s = Span::start(&mut ids, &format!("{} {}", obs.method, obs.route), SpanKind::Server, obs.parent.as_ref(), obs.received_ns);
    s.end_ns = now_ns();
    s.attr("http.request.method", AttrValue::Str(obs.method.clone()));
    s.attr("http.route", AttrValue::Str(obs.route.to_string()));
    s.attr("http.response.status_code", AttrValue::Int(i64::from(status)));
    if let Some(op) = operation {
        s.attr("gen_ai.operation.name", AttrValue::Str(op.to_string()));
    }
    s.attr("gen_ai.request.model", AttrValue::Str(shared.model_id.clone()));
    if status >= 500 {
        let et = error_type.map(str::to_string).unwrap_or_else(|| status.to_string());
        s.attr("error.type", AttrValue::Str(et.clone()));
        s.error = Some(et);
    }
    s
    // SOLUTION-END
}

/// One finished generation: the request metrics (TTFT, TPOT, duration)
/// and its trace (the full tree once the engine took it, else the SERVER
/// span alone).
fn finish_request(shared: &Shared, obs: &Obs, operation: &str, model: &str, out: Outcome) {
    // SOLUTION-BEGIN L10.9
    let end_ns = now_ns();
    let first = out.token_times_ns.first().copied();
    let secs = |ns: u64| ns.saturating_sub(obs.received_ns) as f64 / 1e9;
    let rec = RequestRecord {
        operation: operation.to_string(),
        model: model.to_string(),
        role: "unified".to_string(),
        ttft_s: first.map(secs),
        e2e_s: obs.received.elapsed().as_secs_f64(),
        output_tokens: out.token_times_ns.len(),
        error_type: out.error_type.clone(),
    };
    if let Ok(mut m) = shared.metrics.lock() {
        let _ = m.record_request(&rec);
    }
    let span_error = out.error_type.clone().filter(|_| out.status >= 500);
    let spans = match out.admitted_ns {
        None => vec![server_span(shared, obs, Some(operation), out.status, span_error.as_deref())],
        Some(admitted_ns) => {
            let chunk = shared.prefill_chunk;
            let t = RequestTiming {
                route: obs.route.to_string(),
                operation: operation.to_string(),
                model: model.to_string(),
                status: out.status,
                error_type: span_error,
                received_ns: obs.received_ns,
                admitted_ns,
                first_token_ns: first.unwrap_or(end_ns),
                end_ns,
                priority: out.priority,
                prompt_tokens: out.prompt_tokens,
                // L10.5's engine does not report prefix-cache hits per
                // request; the hit ratio is the tl_engine_prefix_cache_hit_ratio gauge.
                prefix_hit_tokens: 0,
                chunks: if chunk > 0 { out.prompt_tokens.div_ceil(chunk).max(1) } else { 1 },
                token_times_ns: out.token_times_ns,
                finish_reasons: out.finish.into_iter().collect(),
                spec_accept_rate: spec_rate(shared),
            };
            let mut ids = shared.ids.lock().unwrap_or_else(|p| p.into_inner());
            telemetry::request_spans(&mut ids, obs.parent.as_ref(), &t)
        }
    };
    export_spans(spans);
    // SOLUTION-END
}

/// The TokenIndex of a grammar over this model's vocabulary, built once per
/// pattern. EOS (the first EOS id) is allowed where the grammar accepts;
/// the other EOS ids and the BOS and EOS marker tokens never; ids past the
/// tokenizer's vocabulary (padding rows) never.
fn grammar_index(shared: &Shared, pattern: &str) -> Result<Arc<TokenIndex>, ApiError> {
    // SOLUTION-BEGIN L10.9
    let mut cache = shared.grammars.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(i) = cache.get(pattern) {
        return Ok(Arc::clone(i));
    }
    let dfa = regex_to_dfa(pattern).map_err(|e| ApiError::unsupported("tools", e.0))?;
    let mut specials = shared.eos_ids.clone();
    for marker in [&shared.bos, &shared.eos] {
        let ids = shared.tok.encode(marker);
        if ids.len() == 1 {
            specials.push(ids[0]);
        }
    }
    let mut vocab = tools::vocab_bytes(&*shared.tok, &specials);
    vocab.resize(shared.vocab, None);
    let eos = shared.eos_ids.first().copied().filter(|&e| (e as usize) < shared.vocab);
    let index = Arc::new(TokenIndex::new(dfa, &vocab, eos).map_err(|e| ApiError::internal(e.0))?);
    cache.insert(pattern.to_string(), Arc::clone(&index));
    Ok(index)
    // SOLUTION-END
}

/// A chat body: the tool fields first, from the raw text so a schema's
/// properties keep their order (L10.9), then L10.5's validation of the rest.
fn chat_request(body: &[u8]) -> Result<(GenSpec, Option<ToolReq>), ApiError> {
    // SOLUTION-BEGIN L10.9
    const KEYS: [&str; 3] = ["tools", "tool_choice", "response_format"];
    let obj = match serde_json::from_slice::<Value>(body) {
        Ok(Value::Object(o)) if KEYS.iter().any(|k| o.get(*k).is_some_and(|v| !v.is_null())) => o,
        _ => return openai::parse_chat(body).map(|s| (s, None)),
    };
    let text = std::str::from_utf8(body).map_err(|_| ApiError::invalid(None, "the body is not UTF-8"))?;
    let req = tools::parse_tool_request(text).map_err(|e| e.to_api_error())?;
    let mut rest = obj;
    for k in KEYS {
        rest.remove(k);
    }
    let spec = openai::parse_chat(serde_json::to_string(&rest).expect("a JSON object").as_bytes())?;
    let messages = Json::parse(text).ok().and_then(|d| d.get("messages").cloned()).unwrap_or(Json::Arr(Vec::new()));
    Ok((spec, Some(ToolReq { req, messages })))
    // SOLUTION-END
}

/// A chat.completion.chunk around a ready `delta` object.
fn delta_chunk(job: &Job, created: u64, delta: Value, finish: Option<&str>) -> String {
    // SOLUTION-BEGIN L10.9
    serde_json::json!({"id": job.id, "object": "chat.completion.chunk", "created": created, "model": job.model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}).to_string()
    // SOLUTION-END
}

/// One ToolEvent as a stream chunk.
fn event_chunk(job: &Job, created: u64, e: &ToolEvent) -> String {
    // SOLUTION-BEGIN L10.9
    delta_chunk(job, created, serde_json::from_str(&tools::delta_json(e)).expect("delta_json writes JSON"), None)
    // SOLUTION-END
}

/// What a generation needs after validation.
struct Job {
    id: String,
    chat: bool,
    model: String,
    prompt_tokens: usize,
    stream: bool,
    include_usage: bool,
    logprobs: bool,
    stops: Vec<String>,
    /// "chat" or "text_completion".
    operation: &'static str,
    priority: i32,
    admitted_ns: u64,
    /// The answer may call tools: parse it into tool_calls.
    tool_calls: bool,
}

/// Turns tokens into text pieces with stop strings and EOS handled.
struct Detok {
    text: TextStream,
    completion_tokens: usize,
    finish: Option<&'static str>,
}

impl Detok {
    /// One token: the text it releases, and its logprob entry.
    fn token(&mut self, shared: &Shared, token: u32, logprob: f64, finish: Option<FinishReason>) -> (String, Option<Value>) {
        // SOLUTION-BEGIN L10.9
        self.completion_tokens += 1;
        let is_eos = finish == Some(FinishReason::Stop) && shared.eos_ids.contains(&token);
        let bytes = if is_eos { &[][..] } else { shared.tok.token_bytes(token).unwrap_or(&[]) };
        let mut piece = self.text.push(bytes);
        let lp = (!is_eos).then(|| openai::token_logprob(&String::from_utf8_lossy(bytes), bytes, logprob));
        if self.text.stopped() {
            self.finish = Some("stop");
        } else if let Some(f) = finish {
            piece.push_str(&self.text.finish());
            self.finish = Some(match f {
                FinishReason::Stop => "stop",
                FinishReason::Length => "length",
            });
        }
        (piece, lp)
        // SOLUTION-END
    }
}

/// Validates, renders the prompt, and queues a generation; the response is
/// decided by the first event (an error, or the first token).
async fn generate(shared: Arc<Shared>, headers: &hyper::HeaderMap, spec: GenSpec, chat: bool, obs: Obs, tool: Option<ToolReq>) -> Response<RespBody> {
    // SOLUTION-BEGIN L10.9
    let operation = if chat { "chat" } else { "text_completion" };
    // every early answer is a finished request too: recorded, then sent
    let fail = |e: ApiError, out: Outcome| {
        finish_request(&shared, &obs, operation, &spec.model, Outcome { admitted_ns: out.admitted_ns, prompt_tokens: out.prompt_tokens, priority: out.priority, ..Outcome::failed(&e) });
        error_response(&e)
    };
    if spec.model != shared.model_id {
        return fail(ApiError::model_not_found(&spec.model), Outcome::default());
    }
    if shared.draining.load(Ordering::SeqCst) {
        return fail(ApiError::no_capacity("the engine is draining"), Outcome::default());
    }
    if headers.contains_key("x-tl-kv-handle") {
        let mut e = ApiError::invalid(None, "unknown KV handle: this engine serves no disaggregated hand-offs (L10.6)");
        e.status = 404;
        e.code = Some("kv_handle_not_found");
        return fail(e, Outcome::default());
    }
    let priority = match headers.get("x-tl-priority").map(|v| v.to_str().ok().and_then(|s| s.trim().parse::<i32>().ok())) {
        None => 0,
        Some(Some(p)) if (-100..=100).contains(&p) => p,
        Some(_) => return fail(ApiError::invalid(None, "X-TL-Priority must be an integer in [-100, 100]"), Outcome::default()),
    };
    let constraint = match tool.as_ref().and_then(|t| tools::grammar(&t.req)) {
        None => None,
        Some(g) => match grammar_index(&shared, &g.pattern) {
            Ok(index) => Some(Constrained { index, trigger: g.trigger }),
            Err(e) => return fail(e, Outcome::default()),
        },
    };
    let tool_calls = tool.as_ref().is_some_and(|t| !t.req.tools.is_empty() && t.req.choice != ToolChoice::None);
    let prompt = match &spec.prompt {
        Prompt::Text(t) => t.clone(),
        Prompt::Chat(_) if tool.is_some() => {
            let t = tool.as_ref().expect("checked above");
            match tools::render_prompt(&shared.template, &t.messages, &t.req.tools, &shared.bos, &shared.eos) {
                Ok(p) => p,
                Err(e) => return fail(e.to_api_error(), Outcome::default()),
            }
        }
        Prompt::Chat(msgs) => {
            let m: Vec<Value> = msgs.iter().map(|x| serde_json::json!({"role": x.role, "content": x.content.clone().unwrap_or_default()})).collect();
            match shared.template.render_chat(&Value::Array(m), true, &shared.bos, &shared.eos, None) {
                Ok(p) => p,
                Err(e) => return fail(ApiError::internal(format!("chat template: {e}")), Outcome::default()),
            }
        }
    };
    let tokens = shared.tok.encode(&prompt);
    if tokens.is_empty() {
        return fail(ApiError::invalid(Some(if chat { "messages" } else { "prompt" }), "the prompt has no tokens"), Outcome::default());
    }
    let base = Outcome { prompt_tokens: tokens.len(), priority, ..Outcome::default() };
    let room = shared.max_model_len.saturating_sub(tokens.len());
    let max_tokens = match spec.max_tokens {
        Some(n) if n > room => {
            return fail(
                ApiError::context_length(format!("{} prompt tokens plus max_tokens {n} exceed the context of {}", tokens.len(), shared.max_model_len)),
                base,
            )
        }
        Some(n) => n,
        None if room == 0 => return fail(ApiError::context_length(format!("{} prompt tokens fill the context", tokens.len())), base),
        None => room,
    };
    let seed = spec.seed.unwrap_or_else(|| mix64(u64::from_str_radix(&new_id(&shared), 16).unwrap_or(0)) >> 1);
    let req = GenRequest { prompt: tokens.clone(), params: spec.params.clone(), seed, max_tokens, stop_ids: shared.eos_ids.clone(), priority };
    // in flight from here until the response (or its stream) ends: a drain
    // waits for it
    shared.inflight.fetch_add(1, Ordering::SeqCst);
    let guard = Inflight(Arc::clone(&shared));
    let (tx, mut rx) = mpsc::unbounded_channel();
    match shared.cmds.try_send(Cmd::Generate { req, events: tx, constraint }) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => return fail(ApiError::queue_full(), base),
        Err(TrySendError::Disconnected(_)) => return fail(ApiError::no_capacity("the engine stopped"), base),
    }
    let mut admitted_ns = None;
    let first = loop {
        let at = Outcome { admitted_ns, ..Outcome { prompt_tokens: tokens.len(), priority, ..Outcome::default() } };
        match rx.recv().await {
            Some(Ev::Admitted { at_ns }) => admitted_ns = Some(at_ns),
            Some(Ev::Rejected(e)) => return fail(e, at),
            Some(Ev::Failed(m)) => return fail(ApiError::internal(m), at),
            Some(ev) => break ev,
            None => return fail(ApiError::internal("the engine dropped the request"), at),
        }
    };
    let job = Job {
        id: format!("{}-{}", if chat { "chatcmpl" } else { "cmpl" }, new_id(&shared)),
        chat,
        model: spec.model.clone(),
        prompt_tokens: tokens.len(),
        stream: spec.stream,
        include_usage: spec.include_usage,
        logprobs: spec.logprobs,
        stops: spec.stop.clone(),
        operation,
        priority,
        admitted_ns: admitted_ns.unwrap_or_else(now_ns),
        tool_calls,
    };
    if job.stream {
        let (btx, brx) = mpsc::unbounded_channel::<Bytes>();
        tokio::spawn(stream_task(Arc::clone(&shared), job, first, rx, btx, guard, obs));
        let mut r = Response::new(SseBody { rx: brx }.boxed());
        r.headers_mut().insert("content-type", HeaderValue::from_static("text/event-stream"));
        r.headers_mut().insert("cache-control", HeaderValue::from_static("no-cache"));
        return r;
    }
    let mut d = Detok { text: TextStream::new(job.stops.clone()), completion_tokens: 0, finish: None };
    let mut content = String::new();
    let mut lps = Vec::new();
    let mut times = Vec::new();
    let mut ev = Some(first);
    while let Some(e) = ev.take() {
        let at = |times: &Vec<u64>| Outcome { admitted_ns: Some(job.admitted_ns), prompt_tokens: job.prompt_tokens, priority, token_times_ns: times.clone(), ..Outcome::default() };
        match e {
            Ev::Token { token, logprob, finish, at_ns } => {
                times.push(at_ns);
                let (piece, lp) = d.token(&shared, token, logprob, finish);
                content.push_str(&piece);
                lps.extend(lp);
                if d.finish.is_some() {
                    break;
                }
            }
            Ev::Admitted { .. } => {}
            Ev::Failed(m) => return fail(ApiError::internal(m), at(&times)),
            Ev::Rejected(e) => return fail(e, at(&times)),
        }
        ev = rx.recv().await;
    }
    drop(rx); // a stop string ended it: the engine aborts the rest
    drop(guard);
    let mut finish = d.finish.unwrap_or("length").to_string();
    let mut message = None;
    if job.tool_calls {
        let mut p = ToolParser::new(&job.id);
        let mut events = p.push(&content);
        events.extend(p.finish());
        let (text, calls) = tools::collect(&events);
        finish = tools::finish_reason(calls.len(), &finish);
        message = Some(tools::message_json(text.as_deref(), &calls));
    }
    let finish = finish.as_str();
    finish_request(
        &shared,
        &obs,
        operation,
        &spec.model,
        Outcome { status: 200, prompt_tokens: job.prompt_tokens, priority, admitted_ns: Some(job.admitted_ns), token_times_ns: times, finish: Some(finish.to_string()), ..Outcome::default() },
    );
    let usage = openai::usage(job.prompt_tokens, d.completion_tokens);
    let body = if let Some(m) = message {
        let msg: Value = serde_json::from_str(&m).expect("message_json writes JSON");
        let mut choice = serde_json::json!({"index": 0, "message": msg, "finish_reason": finish});
        if job.logprobs {
            choice["logprobs"] = serde_json::json!({"content": lps});
        }
        serde_json::json!({"id": job.id, "object": "chat.completion", "created": shared.created, "model": job.model, "choices": [choice], "usage": usage}).to_string()
    } else if chat {
        openai::chat_completion(&job.id, shared.created, &job.model, &content, finish, usage, job.logprobs.then_some(lps))
    } else {
        openai::completion(&job.id, shared.created, &job.model, &content, finish, usage)
    };
    full(StatusCode::OK, "application/json", body)
    // SOLUTION-END
}

/// Writes one request's SSE stream: a role delta (chat), one chunk per
/// piece of text, the finishing chunk, the usage chunk when asked, then
/// `[DONE]`; a ping comment after 15 s without a token.
async fn stream_task(shared: Arc<Shared>, job: Job, first: Ev, mut rx: mpsc::UnboundedReceiver<Ev>, out: mpsc::UnboundedSender<Bytes>, _guard: Inflight, obs: Obs) {
    // SOLUTION-BEGIN L10.9
    let mut times: Vec<u64> = Vec::new();
    // the request ends here whichever way the stream ends
    let done = |times: Vec<u64>, finish: Option<&str>, err: Option<&ApiError>| {
        let mut o = Outcome { status: 200, prompt_tokens: job.prompt_tokens, priority: job.priority, admitted_ns: Some(job.admitted_ns), token_times_ns: times, finish: finish.map(str::to_string), ..Outcome::default() };
        if let Some(e) = err {
            o.status = e.status;
            o.error_type = Outcome::failed(e).error_type;
        }
        finish_request(&shared, &obs, job.operation, &job.model, o);
    };
    let send = |s: String| out.send(Bytes::from(s)).is_ok();
    let chunk = |text: &str, role: bool, finish: Option<&str>, lp: Option<Vec<Value>>| {
        if job.chat {
            openai::chat_chunk(&job.id, shared.created, &job.model, role, text, finish, lp)
        } else {
            openai::completion_chunk(&job.id, shared.created, &job.model, text, finish)
        }
    };
    let gone = ApiError { status: 499, kind: "client_closed", message: "the client went away".to_string(), param: None, code: Some("client_closed") };
    if job.chat && !send(sse::event(&chunk("", true, None, None))) {
        done(times, None, Some(&gone));
        return;
    }
    let mut d = Detok { text: TextStream::new(job.stops.clone()), completion_tokens: 0, finish: None };
    let mut parser = job.tool_calls.then(|| ToolParser::new(&job.id));
    let mut final_finish = None;
    let mut ev = Some(first);
    loop {
        let e = match ev.take() {
            Some(e) => e,
            None => match tokio::time::timeout(Duration::from_secs(15), rx.recv()).await {
                Ok(Some(e)) => e,
                Ok(None) => {
                    let err = ApiError::internal("the engine dropped the request");
                    send(sse::event(&err.to_json()));
                    done(times, None, Some(&err));
                    return;
                }
                Err(_) => {
                    if !send(sse::PING.to_string()) {
                        done(times, None, Some(&gone));
                        return;
                    }
                    continue;
                }
            },
        };
        match e {
            Ev::Token { token, logprob, finish, at_ns } => {
                times.push(at_ns);
                let (piece, lp) = d.token(&shared, token, logprob, finish);
                if let Some(p) = parser.as_mut() {
                    // tool mode: the text becomes content and tool_calls deltas
                    let mut events = p.push(&piece);
                    if d.finish.is_some() {
                        events.extend(p.finish());
                    }
                    for e in &events {
                        if !send(sse::event(&event_chunk(&job, shared.created, e))) {
                            done(times, None, Some(&gone));
                            return;
                        }
                    }
                    if let Some(f) = d.finish {
                        let f = tools::finish_reason(p.calls(), f);
                        if !send(sse::event(&delta_chunk(&job, shared.created, serde_json::json!({}), Some(&f)))) {
                            done(times, None, Some(&gone));
                            return;
                        }
                        final_finish = Some(f);
                        break;
                    }
                    continue;
                }
                let lp = if job.logprobs { Some(lp.into_iter().collect()) } else { None };
                if let Some(f) = d.finish {
                    if !send(sse::event(&chunk(&piece, false, Some(f), lp))) {
                        done(times, None, Some(&gone));
                        return;
                    }
                    break;
                }
                if (!piece.is_empty() || job.logprobs) && !send(sse::event(&chunk(&piece, false, None, lp))) {
                    done(times, None, Some(&gone));
                    return;
                }
            }
            Ev::Admitted { .. } => {}
            Ev::Rejected(e) => {
                send(sse::event(&e.to_json()));
                done(times, None, Some(&e));
                return;
            }
            Ev::Failed(m) => {
                let e = ApiError::internal(m);
                send(sse::event(&e.to_json()));
                done(times, None, Some(&e));
                return;
            }
        }
    }
    drop(rx);
    done(times, final_finish.as_deref().or(d.finish), None);
    if job.include_usage {
        let object = if job.chat { "chat.completion.chunk" } else { "text_completion" };
        send(sse::event(&openai::usage_chunk(&job.id, shared.created, &job.model, object, openai::usage(job.prompt_tokens, d.completion_tokens))));
    }
    send(sse::DONE.to_string());
    // SOLUTION-END
}

/// `/v1/embeddings`.
async fn embeddings(shared: Arc<Shared>, body: &[u8], obs: &Obs) -> Response<RespBody> {
    // SOLUTION-BEGIN L10.9
    let (model, inputs) = match openai::parse_embeddings(body) {
        Ok(x) => x,
        Err(e) => return error_response(&e),
    };
    let result = embed(&shared, &model, inputs).await;
    if let Ok(mut m) = shared.metrics.lock() {
        let error_type = result.as_ref().err().map(|e| Outcome::failed(e).error_type.unwrap_or_default());
        let rec = RequestRecord { operation: "embeddings".into(), model: model.clone(), role: "unified".into(), ttft_s: None, e2e_s: obs.received.elapsed().as_secs_f64(), output_tokens: 0, error_type };
        let _ = m.record_request(&rec);
    }
    match result {
        Ok(body) => full(StatusCode::OK, "application/json", body),
        Err(e) => error_response(&e),
    }
    // SOLUTION-END
}

/// Runs validated embedding inputs on the engine thread: the response body.
async fn embed(shared: &Arc<Shared>, model: &str, inputs: Vec<String>) -> Result<String, ApiError> {
    // SOLUTION-BEGIN L10.9
    if model != shared.model_id {
        return Err(ApiError::model_not_found(model));
    }
    let toks: Vec<Vec<u32>> = inputs.iter().map(|s| shared.tok.encode(s)).collect();
    if let Some(t) = toks.iter().find(|t| t.is_empty() || t.len() > shared.max_model_len) {
        return Err(ApiError::context_length(format!("an input has {} tokens; the context is {}", t.len(), shared.max_model_len)));
    }
    let n: usize = toks.iter().map(Vec::len).sum();
    let (tx, rx) = oneshot::channel();
    if shared.cmds.try_send(Cmd::Embed { inputs: toks, reply: tx }).is_err() {
        return Err(ApiError::queue_full());
    }
    match rx.await {
        Ok(Ok(v)) => Ok(openai::embeddings(model, &v, n)),
        Ok(Err(m)) => Err(ApiError::no_capacity(m)),
        Err(_) => Err(ApiError::internal("the engine dropped the request")),
    }
    // SOLUTION-END
}

/// The `/metrics` body: the engine gauges refreshed from the step loop's
/// stats, then every instrument of otel/metrics.yaml.
fn metrics_body(shared: &Shared) -> String {
    // SOLUTION-BEGIN L10.9
    let stats = *shared.stats.lock().unwrap_or_else(|p| p.into_inner());
    let rate = spec_rate(shared);
    let mut m = shared.metrics.lock().unwrap_or_else(|p| p.into_inner());
    let _ = m.set_engine(&EngineGauges::from_stats(&stats, rate));
    m.render()
    // SOLUTION-END
}

/// Routes of the API port.
async fn api(shared: Arc<Shared>, req: Request<Incoming>) -> Result<Response<RespBody>, Infallible> {
    // SOLUTION-BEGIN L10.9
    let rid = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty() && v.len() <= 128)
        .map(str::to_string)
        .unwrap_or_else(|| new_id(&shared));
    let (parts, body) = req.into_parts();
    let path = parts.uri.path().to_string();
    let obs = || Obs {
        method: parts.method.to_string(),
        route: route_of(&path),
        received: Instant::now(),
        received_ns: now_ns(),
        parent: telemetry::extract(&|k| parts.headers.get(k).and_then(|v| v.to_str().ok()).map(str::to_string)),
    };
    let o = obs();
    let body = match body.collect().await {
        Ok(b) => b.to_bytes(),
        Err(_) => Bytes::new(),
    };
    // generations record their own request metrics and spans (they may
    // outlive this handler as a stream); the other routes get a SERVER span here
    let mut traced = false;
    let mut resp = match (&parts.method, path.as_str()) {
        (&Method::POST, "/v1/chat/completions") => {
            traced = true;
            match chat_request(&body) {
                Ok((spec, tool)) => generate(Arc::clone(&shared), &parts.headers, spec, true, obs(), tool).await,
                Err(e) => {
                    finish_request(&shared, &o, "chat", &shared.model_id, Outcome::failed(&e));
                    error_response(&e)
                }
            }
        }
        (&Method::POST, "/v1/completions") => {
            traced = true;
            match openai::parse_completion(&body) {
                Ok(spec) => generate(Arc::clone(&shared), &parts.headers, spec, false, obs(), None).await,
                Err(e) => {
                    finish_request(&shared, &o, "text_completion", &shared.model_id, Outcome::failed(&e));
                    error_response(&e)
                }
            }
        }
        (&Method::POST, "/v1/embeddings") => embeddings(Arc::clone(&shared), &body, &o).await,
        (&Method::POST, "/v1/tokenize") => match openai::parse_tokenize(&body) {
            Ok((m, _, _)) if m != shared.model_id => error_response(&ApiError::model_not_found(&m)),
            Ok((_, text, _)) => full(StatusCode::OK, "application/json", serde_json::json!({"ids": shared.tok.encode(&text)}).to_string()),
            Err(e) => error_response(&e),
        },
        (&Method::GET, "/v1/models") => {
            full(StatusCode::OK, "application/json", serde_json::json!({"object": "list", "data": [openai::model_object(&shared.model_id, shared.created)]}).to_string())
        }
        (&Method::GET, p) if p.starts_with("/v1/models/") => {
            let m = &p["/v1/models/".len()..];
            if m == shared.model_id {
                full(StatusCode::OK, "application/json", openai::model_object(m, shared.created).to_string())
            } else {
                error_response(&ApiError::model_not_found(m))
            }
        }
        (&Method::GET, "/healthz") => full(StatusCode::OK, "text/plain", "ok\n".to_string()),
        (_, "/v1/chat/completions" | "/v1/completions" | "/v1/embeddings" | "/v1/tokenize" | "/v1/models" | "/healthz") => {
            let mut e = ApiError::invalid(None, format!("method {} is not allowed on {path}", parts.method));
            e.status = 405;
            error_response(&e)
        }
        _ => {
            let mut e = ApiError::invalid(None, format!("no route {path}"));
            e.status = 404;
            error_response(&e)
        }
    };
    if let Ok(v) = HeaderValue::from_str(&rid) {
        resp.headers_mut().insert("x-request-id", v);
    }
    let status = resp.status().as_u16();
    if !traced && path.starts_with("/v1/") {
        let op = (o.route == "/v1/embeddings").then_some("embeddings");
        export_spans(vec![server_span(&shared, &o, op, status, None)]);
    }
    if let Ok(mut m) = shared.metrics.lock() {
        let _ = m.record_http(&o.method, o.route, status, o.received.elapsed().as_secs_f64());
    }
    Ok(resp)
    // SOLUTION-END
}

/// Routes of the health port.
async fn health(shared: Arc<Shared>, req: Request<Incoming>) -> Result<Response<RespBody>, Infallible> {
    // SOLUTION-BEGIN L10.9
    let t0 = Instant::now();
    let path = req.uri().path().to_string();
    let resp = match path.as_str() {
        "/healthz" => full(StatusCode::OK, "text/plain", "ok\n".to_string()),
        "/readyz" => {
            if shared.ready.load(Ordering::SeqCst) && !shared.draining.load(Ordering::SeqCst) {
                full(StatusCode::OK, "text/plain", "ready\n".to_string())
            } else {
                full(StatusCode::SERVICE_UNAVAILABLE, "text/plain", "not ready\n".to_string())
            }
        }
        "/metrics" => full(StatusCode::OK, "text/plain; version=0.0.4", metrics_body(&shared)),
        _ => full(StatusCode::NOT_FOUND, "text/plain", "not found\n".to_string()),
    };
    if let Ok(mut m) = shared.metrics.lock() {
        let _ = m.record_http(req.method().as_str(), route_of(&path), resp.status().as_u16(), t0.elapsed().as_secs_f64());
    }
    Ok(resp)
    // SOLUTION-END
}

/// Accepts connections until `stop` turns true; each connection is served
/// by its own task.
async fn accept_loop<F, Fut>(listener: tokio::net::TcpListener, shared: Arc<Shared>, mut stop: watch::Receiver<bool>, route: F)
where
    F: Fn(Arc<Shared>, Request<Incoming>) -> Fut + Copy + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<Response<RespBody>, Infallible>> + Send + 'static,
{
    // SOLUTION-BEGIN L10.9
    loop {
        tokio::select! {
            _ = stop.changed() => return,
            acc = listener.accept() => {
                let Ok((stream, _)) = acc else { continue };
                let _ = stream.set_nodelay(true);
                let sh = Arc::clone(&shared);
                tokio::spawn(async move {
                    let svc = service_fn(move |req| route(Arc::clone(&sh), req));
                    let _ = hyper::server::conn::http1::Builder::new().serve_connection(TokioIo::new(stream), svc).await;
                });
            }
        }
    }
    // SOLUTION-END
}

/// A running server (tests and the binary).
pub struct ServerHandle {
    pub http_addr: SocketAddr,
    pub health_addr: SocketAddr,
    drain: watch::Sender<bool>,
    thread: Option<std::thread::JoinHandle<Result<(), String>>>,
}

impl ServerHandle {
    /// Begins draining, as SIGTERM does.
    pub fn drain(&self) {
        // SOLUTION-BEGIN L10.9
        let _ = self.drain.send(true);
        // SOLUTION-END
    }

    /// Waits until the server has drained and stopped.
    pub fn join(mut self) -> Result<(), String> {
        // SOLUTION-BEGIN L10.9
        match self.thread.take() {
            Some(t) => t.join().map_err(|_| "the server thread panicked".to_string())?,
            None => Ok(()),
        }
        // SOLUTION-END
    }
}

/// Loads the model, binds both listeners, and serves on a thread of its
/// own (a tokio runtime plus the engine thread). Returns once bound.
pub fn spawn(cfg: ServeConfig) -> Result<ServerHandle, String> {
    // SOLUTION-BEGIN L10.9
    let runner = ModelRunner::load(&cfg.model_dir, &cfg.engine).map_err(|e| format!("{e:#}"))?;
    let mcfg = runner.config().clone();
    let tok: Arc<dyn Tokenizer> = match mcfg.tokenizer {
        TokenizerKind::Bytes => Arc::new(ByteTokenizer),
        TokenizerKind::File => Arc::new(ByteBpe::from_hf_json(&cfg.model_dir.join("tokenizer.json")).map_err(|e| e.to_string())?),
    };
    let (template, bos, eos, eos_ids) = chat_settings(&cfg.model_dir, &mcfg.eos_token_ids)?;
    let engine = Engine::new(runner, &cfg.engine, cfg.queue_capacity);
    let (cmd_tx, cmd_rx) = sync_channel::<Cmd>(cfg.queue_capacity);
    let shared = Arc::new(Shared {
        cmds: cmd_tx,
        tok,
        template,
        bos,
        eos,
        eos_ids,
        model_id: cfg.model_id.clone(),
        hidden: mcfg.hidden_size,
        created: unix_now(),
        max_model_len: engine.max_model_len(),
        draining: AtomicBool::new(false),
        ready: AtomicBool::new(false),
        inflight: AtomicUsize::new(0),
        seq: AtomicU64::new(0),
        stats: Mutex::new(engine.stats()),
        metrics: Mutex::new(EngineMetrics::new()),
        ids: Mutex::new(IdGen::new(now_ns() ^ u64::from(std::process::id()).rotate_left(32))),
        spec: Mutex::new((0, 0)),
        prefill_chunk: cfg.engine.prefill_chunk,
        vocab: mcfg.vocab_size,
        grammars: Mutex::new(HashMap::new()),
    });
    let std_http = std::net::TcpListener::bind(listen_addr(&cfg.http_listen)?).map_err(|e| format!("{}: {e}", cfg.http_listen))?;
    let std_health = std::net::TcpListener::bind(listen_addr(&cfg.health_listen)?).map_err(|e| format!("{}: {e}", cfg.health_listen))?;
    let (http_addr, health_addr) = (std_http.local_addr().map_err(|e| e.to_string())?, std_health.local_addr().map_err(|e| e.to_string())?);
    std_http.set_nonblocking(true).map_err(|e| e.to_string())?;
    std_health.set_nonblocking(true).map_err(|e| e.to_string())?;
    let (drain_tx, drain_rx) = watch::channel(false);
    let stop = Arc::new(AtomicBool::new(false));
    let eng_shared = Arc::clone(&shared);
    let eng_stop = Arc::clone(&stop);
    let deadline = Duration::from_millis(cfg.queue_deadline_ms);
    let delay = cfg.step_delay;
    let speculative = cfg.speculative.clone();
    let engine_thread = std::thread::Builder::new()
        .name("tl-engine".to_string())
        .spawn(move || engine_loop(engine, cmd_rx, eng_shared, eng_stop, deadline, delay, speculative))
        .map_err(|e| e.to_string())?;
    let signals = cfg.handle_signals;
    let drain_self = drain_tx.clone();
    let thread = std::thread::Builder::new()
        .name("tl-serve".to_string())
        .spawn(move || -> Result<(), String> {
            let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().map_err(|e| e.to_string())?;
            rt.block_on(async move {
                let http = tokio::net::TcpListener::from_std(std_http).map_err(|e| e.to_string())?;
                let hl = tokio::net::TcpListener::from_std(std_health).map_err(|e| e.to_string())?;
                // the API listener closes when draining begins; the health
                // listener stays up to answer /readyz 503 until the end
                let (health_stop_tx, health_stop_rx) = watch::channel(false);
                let a = tokio::spawn(accept_loop(http, Arc::clone(&shared), drain_rx.clone(), api));
                let h = tokio::spawn(accept_loop(hl, Arc::clone(&shared), health_stop_rx, health));
                shared.ready.store(true, Ordering::SeqCst);
                let mut dr = drain_rx.clone();
                if signals {
                    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).map_err(|e| e.to_string())?;
                    tokio::select! {
                        _ = term.recv() => { let _ = drain_self.send(true); }
                        _ = dr.changed() => {}
                    }
                } else {
                    let _ = dr.changed().await;
                }
                shared.draining.store(true, Ordering::SeqCst);
                let _ = a.await;
                let t0 = Instant::now();
                while shared.inflight.load(Ordering::SeqCst) > 0 && t0.elapsed() < Duration::from_secs(30) {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                let _ = health_stop_tx.send(true);
                let _ = h.await;
                Ok::<(), String>(())
            })?;
            stop.store(true, Ordering::SeqCst);
            engine_thread.join().map_err(|_| "the engine thread panicked".to_string())?;
            Ok(())
        })
        .map_err(|e| e.to_string())?;
    Ok(ServerHandle { http_addr, health_addr, drain: drain_tx, thread: Some(thread) })
    // SOLUTION-END
}

/// The binary's entry: serve until SIGTERM, drain, return (exit 0).
pub fn run(cfg: ServeConfig) -> Result<(), String> {
    // SOLUTION-BEGIN L10.9
    let h = spawn(ServeConfig { handle_signals: true, ..cfg })?;
    h.join()
    // SOLUTION-END
}
