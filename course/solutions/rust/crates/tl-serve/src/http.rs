//! tl-serve http v0 (L10.0): the tracer engine, std only.
//!
//! One request per TCP connection, one thread per connection:
//!
//!   bytes --read_request--> Request --parse_completion--> CompletionRequest
//!         --Bigram::logits (row lookup) + sample--> byte ids
//!         --Utf8Stream--> text --JSON / SSE--> bytes back to the client
//!
//! then, when OTEL_EXPORTER_OTLP_ENDPOINT is set, one SERVER span per
//! completion request is posted as OTLP/HTTP JSON.
//!
//! Contracts: openapi/openai-subset.v0.yaml (the HTTP surface),
//! spec/cli-roles.md (the engine role), formats/safetensors.md and
//! formats/config.schema.json (the model directory), formats/tokenizer.md
//! (the byte tokenizer and incremental decoding). L10.5 takes this file over.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream, ToSocketAddrs};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The byte tokenizer's vocabulary: one id per byte value.
pub const VOCAB: usize = 256;
/// The request line plus headers may take at most this many bytes.
pub const MAX_HEAD: usize = 16 * 1024;
/// The largest request body accepted.
pub const MAX_BODY: usize = 1024 * 1024;
/// The contract's upper bound on `max_tokens`.
pub const MAX_TOKENS: u32 = 4096;

// ===========================================================================
// HTTP/1.1 requests

/// One parsed request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub method: String,
    /// Path plus an optional `?query`, as sent.
    pub target: String,
    /// In the order and case they were sent; use `header()` to look up.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    /// The first header named `name`, ignoring ASCII case.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }

    /// The target without its query string.
    pub fn path(&self) -> &str {
        self.target.split_once('?').map_or(self.target.as_str(), |(p, _)| p)
    }
}

/// A request that could not be read. `status` is the HTTP answer (400, 411,
/// 413, 431, 501); 0 means the client went away and nobody is listening.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpError {
    pub status: u16,
    pub message: String,
}

fn http_error(status: u16, message: impl Into<String>) -> HttpError {
    HttpError { status, message: message.into() }
}

/// One line ending in `\n`, at most `limit` bytes, returned without its
/// `\r\n` or `\n`. `Ok(None)` at a clean end of stream.
fn read_line<R: BufRead>(r: &mut R, limit: usize) -> Result<Option<Vec<u8>>, HttpError> {
    let mut line = Vec::new();
    let n = r
        .by_ref()
        .take(limit as u64 + 1)
        .read_until(b'\n', &mut line)
        .map_err(|e| http_error(0, format!("read: {e}")))?;
    if n == 0 {
        return Ok(None);
    }
    if line.last() != Some(&b'\n') {
        return Err(if n > limit { http_error(431, "request head too large") } else { http_error(0, "closed mid-line") });
    }
    line.pop();
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    Ok(Some(line))
}

/// Reads one HTTP/1.x request: the request line, headers up to the empty
/// line, then exactly Content-Length body bytes.
pub fn read_request<R: BufRead>(r: &mut R) -> Result<Request, HttpError> {
    let first = read_line(r, MAX_HEAD)?.ok_or_else(|| http_error(0, "closed before a request"))?;
    let first = String::from_utf8(first).map_err(|_| http_error(400, "request line is not UTF-8"))?;
    let parts: Vec<&str> = first.split(' ').collect();
    let [method, target, version] = parts[..] else {
        return Err(http_error(400, format!("request line {first:?} is not `METHOD target HTTP/1.1`")));
    };
    if method.is_empty() || !method.bytes().all(|b| b.is_ascii_uppercase()) {
        return Err(http_error(400, format!("bad method {method:?}")));
    }
    if !target.starts_with('/') {
        return Err(http_error(400, format!("target {target:?} must start with /")));
    }
    if version != "HTTP/1.1" && version != "HTTP/1.0" {
        return Err(http_error(400, format!("unsupported version {version:?}")));
    }

    let mut used = first.len() + 2;
    let mut headers = Vec::new();
    loop {
        let line = read_line(r, MAX_HEAD)?.ok_or_else(|| http_error(0, "closed in the headers"))?;
        used += line.len() + 2;
        if used > MAX_HEAD {
            return Err(http_error(431, "request head too large"));
        }
        if line.is_empty() {
            break;
        }
        let line = String::from_utf8(line).map_err(|_| http_error(400, "header is not UTF-8"))?;
        let (name, value) = line.split_once(':').ok_or_else(|| http_error(400, format!("header {line:?} has no colon")))?;
        if name.is_empty() || name.bytes().any(|b| b <= b' ' || b == 0x7f) {
            return Err(http_error(400, format!("bad header name {name:?}")));
        }
        headers.push((name.to_string(), value.trim_matches(|c| c == ' ' || c == '\t').to_string()));
    }

    let mut req = Request { method: method.to_string(), target: target.to_string(), headers, body: Vec::new() };
    if req.header("transfer-encoding").is_some() {
        return Err(http_error(501, "chunked request bodies are not supported; send Content-Length"));
    }
    let len = match req.header("content-length") {
        Some(v) if !v.is_empty() && v.bytes().all(|b| b.is_ascii_digit()) => {
            v.parse::<usize>().map_err(|_| http_error(413, "body too large"))?
        }
        Some(v) => return Err(http_error(400, format!("bad Content-Length {v:?}"))),
        None if req.method == "POST" || req.method == "PUT" => return Err(http_error(411, "Content-Length required")),
        None => 0,
    };
    if len > MAX_BODY {
        return Err(http_error(413, "body too large"));
    }
    req.body = vec![0; len];
    r.read_exact(&mut req.body).map_err(|e| http_error(0, format!("body: {e}")))?;
    Ok(req)
}

// ===========================================================================
// JSON

/// A JSON value. Integer literals that fit an i64 are `Int`; every other
/// number is `Float`. Objects keep their keys in the order written.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// The value of `key` when this is an object that has it.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

/// `s` quoted and escaped as a JSON string. `"`, `\`, and every character
/// below U+0020 are escaped; everything else is written as raw UTF-8.
pub fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Parses one complete JSON text.
pub fn parse_json(text: &str) -> Result<Json, String> {
    let mut p = Parser { b: text.as_bytes(), i: 0 };
    let v = p.value(0)?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("trailing characters at byte {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> Result<(), String> {
        if self.b[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            Ok(())
        } else {
            Err(format!("expected {lit:?} at byte {}", self.i))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > 64 {
            return Err("nested deeper than 64".into());
        }
        self.ws();
        match self.b.get(self.i) {
            None => Err("unexpected end of input".into()),
            Some(b'n') => self.eat("null").map(|_| Json::Null),
            Some(b't') => self.eat("true").map(|_| Json::Bool(true)),
            Some(b'f') => self.eat("false").map(|_| Json::Bool(false)),
            Some(b'"') => self.string().map(Json::Str),
            Some(b'[') => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Arr(items));
                        }
                        _ => return Err(format!("expected , or ] at byte {}", self.i)),
                    }
                }
            }
            Some(b'{') => {
                self.i += 1;
                let mut kv = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Obj(kv));
                }
                loop {
                    self.ws();
                    if self.b.get(self.i) != Some(&b'"') {
                        return Err(format!("expected a string key at byte {}", self.i));
                    }
                    let k = self.string()?;
                    self.ws();
                    self.eat(":")?;
                    let v = self.value(depth + 1)?;
                    kv.push((k, v));
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Obj(kv));
                        }
                        _ => return Err(format!("expected , or }} at byte {}", self.i)),
                    }
                }
            }
            Some(_) => self.number(),
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.i;
        while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
            self.i += 1;
        }
        self.i - start
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.b.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        let int_start = self.i;
        if self.digits() == 0 || (self.b[int_start] == b'0' && self.i - int_start > 1) {
            return Err(format!("bad number at byte {start}"));
        }
        let mut integer = true;
        if self.b.get(self.i) == Some(&b'.') {
            integer = false;
            self.i += 1;
            if self.digits() == 0 {
                return Err(format!("bad fraction at byte {start}"));
            }
        }
        if matches!(self.b.get(self.i), Some(b'e' | b'E')) {
            integer = false;
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if self.digits() == 0 {
                return Err(format!("bad exponent at byte {start}"));
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.i]).map_err(|e| e.to_string())?;
        if integer {
            if let Ok(v) = text.parse::<i64>() {
                return Ok(Json::Int(v));
            }
        }
        text.parse::<f64>().map(Json::Float).map_err(|e| format!("{text}: {e}"))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self.b.get(self.i..self.i + 4).ok_or("short \\u escape")?;
        let s = std::str::from_utf8(h).map_err(|e| e.to_string())?;
        let v = u32::from_str_radix(s, 16).map_err(|_| format!("bad \\u escape {s:?}"))?;
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // the opening quote
        let mut out: Vec<u8> = Vec::new();
        loop {
            let Some(&c) = self.b.get(self.i) else {
                return Err("unterminated string".into());
            };
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).map_err(|e| e.to_string()),
                b'\\' => {
                    let Some(&e) = self.b.get(self.i) else {
                        return Err("unterminated escape".into());
                    };
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hi = self.hex4()?;
                            let code = if (0xD800..0xDC00).contains(&hi) {
                                self.eat("\\u")?;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return Err("unpaired surrogate".into());
                                }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            char::from_u32(code).ok_or("unpaired surrogate")?
                        }
                        _ => return Err(format!("bad escape \\{}", e as char)),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                c if c < 0x20 => return Err("raw control character in a string".into()),
                c => out.push(c),
            }
        }
    }
}

// ===========================================================================
// API v0: errors and the completion request

/// An error in the contract's shape:
/// `{"error": {"message", "type", "param", "code"}}`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiError {
    pub status: u16,
    /// `invalid_request_error` or `server_error`.
    pub error_type: String,
    pub message: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

impl ApiError {
    /// A 400 `invalid_request_error`, naming the offending field.
    pub fn invalid(param: Option<&str>, message: impl Into<String>) -> ApiError {
        ApiError {
            status: 400,
            error_type: "invalid_request_error".to_string(),
            message: message.into(),
            param: param.map(str::to_string),
            code: None,
        }
    }

    /// The JSON body.
    pub fn to_json(&self) -> String {
        let opt = |v: &Option<String>| v.as_deref().map_or("null".to_string(), json_string);
        format!(
            "{{\"error\":{{\"message\":{},\"type\":{},\"param\":{},\"code\":{}}}}}",
            json_string(&self.message),
            json_string(&self.error_type),
            opt(&self.param),
            opt(&self.code)
        )
    }
}

/// `POST /v1/completions`, validated against `CompletionRequest` in the v0
/// contract, with its defaults filled in.
#[derive(Debug, Clone, PartialEq)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f64,
    pub seed: Option<u64>,
    pub stream: bool,
}

/// Parses and validates a request body. Unknown fields are ignored; a
/// missing, mistyped, or out-of-range field is a 400 naming it in `param`.
pub fn parse_completion(body: &[u8]) -> Result<CompletionRequest, ApiError> {
    let text = std::str::from_utf8(body).map_err(|_| ApiError::invalid(None, "body is not UTF-8"))?;
    let v = parse_json(text).map_err(|e| ApiError::invalid(None, format!("body is not JSON: {e}")))?;
    if !matches!(v, Json::Obj(_)) {
        return Err(ApiError::invalid(None, "body must be a JSON object"));
    }
    let string = |key: &str| -> Result<String, ApiError> {
        match v.get(key) {
            Some(Json::Str(s)) if !s.is_empty() => Ok(s.clone()),
            Some(Json::Str(_)) => Err(ApiError::invalid(Some(key), format!("`{key}` must not be empty"))),
            Some(_) => Err(ApiError::invalid(Some(key), format!("`{key}` must be a string"))),
            None => Err(ApiError::invalid(Some(key), format!("`{key}` is required"))),
        }
    };
    let model = string("model")?;
    let prompt = string("prompt")?;
    let max_tokens = match v.get("max_tokens") {
        None => 16,
        Some(Json::Int(n)) if (1..=i64::from(MAX_TOKENS)).contains(n) => *n as u32,
        Some(_) => {
            return Err(ApiError::invalid(Some("max_tokens"), format!("`max_tokens` must be an integer from 1 to {MAX_TOKENS}")))
        }
    };
    let temperature = match v.get("temperature") {
        None => 1.0,
        Some(Json::Int(n)) if (0..=2).contains(n) => *n as f64,
        Some(Json::Float(t)) if (0.0..=2.0).contains(t) => *t,
        Some(_) => return Err(ApiError::invalid(Some("temperature"), "`temperature` must be a number from 0 to 2")),
    };
    let seed = match v.get("seed") {
        None | Some(Json::Null) => None,
        Some(Json::Int(n)) if *n >= 0 => Some(*n as u64),
        Some(_) => return Err(ApiError::invalid(Some("seed"), "`seed` must be a non-negative integer")),
    };
    let stream = match v.get("stream") {
        None | Some(Json::Null) => false,
        Some(Json::Bool(b)) => *b,
        Some(_) => return Err(ApiError::invalid(Some("stream"), "`stream` must be a boolean")),
    };
    Ok(CompletionRequest { model, prompt, max_tokens, temperature, seed, stream })
}

// ===========================================================================
// The model: a byte bigram behind the checkpoint contract

/// The tracer model: `logits(prev) = onehot(prev) @ W`, W in R^{256 x 256}.
#[derive(Debug, Clone)]
pub struct Bigram {
    /// Row-major [VOCAB, VOCAB]: row i holds the next-token logits after i.
    weight: Vec<f32>,
}

impl Bigram {
    /// Wraps a row-major 256 x 256 weight.
    pub fn from_weight(weight: Vec<f32>) -> Result<Bigram, String> {
        if weight.len() != VOCAB * VOCAB {
            return Err(format!("bigram.weight has {} values, want {}", weight.len(), VOCAB * VOCAB));
        }
        Ok(Bigram { weight })
    }

    /// Loads a model directory: `config.json` must say `tl_arch: bigram`,
    /// `tl_tokenizer: bytes`, `vocab_size: 256`; `model.safetensors` must hold
    /// `bigram.weight`, F32, shape [256, 256].
    pub fn load(dir: &Path) -> Result<Bigram, String> {
        let cfg_path = dir.join("config.json");
        let cfg_text = std::fs::read_to_string(&cfg_path).map_err(|e| format!("{}: {e}", cfg_path.display()))?;
        let cfg = parse_json(&cfg_text).map_err(|e| format!("{}: {e}", cfg_path.display()))?;
        let field = |k: &str| cfg.get(k).cloned().unwrap_or(Json::Null);
        if field("tl_arch") != Json::Str("bigram".into()) {
            return Err(format!("config.json: tl_arch is {:?}; the tracer engine serves only \"bigram\"", field("tl_arch")));
        }
        if field("tl_tokenizer") != Json::Str("bytes".into()) {
            return Err("config.json: tl_tokenizer must be \"bytes\" (formats/tokenizer.md)".to_string());
        }
        if field("vocab_size") != Json::Int(VOCAB as i64) {
            return Err("config.json: vocab_size must be 256 for the byte tokenizer".to_string());
        }
        let st_path = dir.join("model.safetensors");
        let bytes = std::fs::read(&st_path).map_err(|e| format!("{}: {e}", st_path.display()))?;
        let weight = read_bigram_weight(&bytes).map_err(|e| format!("{}: {e}", st_path.display()))?;
        Bigram::from_weight(weight)
    }

    /// The next-token logits after byte `prev`: row `prev` of the stored
    /// transition matrix. A one-hot row times W selects that same row.
    pub fn logits(&self, prev: u8) -> Result<Vec<f32>, String> {
        let start = prev as usize * VOCAB;
        Ok(self.weight[start..start + VOCAB].to_vec())
    }
}

/// An integer field of a safetensors header entry, as a usize.
fn header_usize(v: &Json) -> Result<usize, String> {
    match v {
        Json::Int(n) if *n >= 0 => Ok(*n as usize),
        _ => Err(format!("expected a non-negative integer, got {v:?}")),
    }
}

/// Reads `bigram.weight` out of the bytes of a safetensors file, enforcing
/// the reader rules of formats/safetensors.md.
pub fn read_bigram_weight(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() < 8 {
        return Err("shorter than the 8-byte header length".to_string());
    }
    let n = u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes"));
    if n > 100_000_000 || 8 + n > bytes.len() as u64 {
        return Err(format!("header length {n} runs past the end of the file ({} bytes)", bytes.len()));
    }
    let data_start = 8 + n as usize;
    let header = std::str::from_utf8(&bytes[8..data_start]).map_err(|_| "header is not UTF-8".to_string())?;
    let Json::Obj(entries) = parse_json(header).map_err(|e| format!("header: {e}"))? else {
        return Err("header is not a JSON object".to_string());
    };

    // Rules 2 and 5: no duplicate names; the tensors tile the data buffer.
    let mut spans = Vec::new();
    for (i, (name, entry)) in entries.iter().enumerate() {
        if entries[..i].iter().any(|(k, _)| k == name) {
            return Err(format!("duplicate key {name:?}"));
        }
        if name == "__metadata__" {
            continue;
        }
        let off = match entry.get("data_offsets") {
            Some(Json::Arr(o)) if o.len() == 2 => (header_usize(&o[0])?, header_usize(&o[1])?),
            _ => return Err(format!("{name}: data_offsets must be [begin, end]")),
        };
        spans.push(off);
    }
    spans.sort();
    let mut at = 0;
    for (b, e) in &spans {
        if *b != at || e < b {
            return Err(format!("tensors do not tile the data buffer at offset {at}"));
        }
        at = *e;
    }
    if data_start + at != bytes.len() {
        return Err(format!("{} bytes after the last tensor", bytes.len() as i64 - (data_start + at) as i64));
    }

    let entry = entries.iter().find(|(k, _)| k == "bigram.weight").map(|(_, v)| v);
    let entry = entry.ok_or("no tensor named bigram.weight")?;
    match entry.get("dtype") {
        Some(Json::Str(d)) if d == "F32" => {}
        other => return Err(format!("bigram.weight dtype {other:?}: contract v0 reads F32 only")),
    }
    let shape_ok = matches!(entry.get("shape"), Some(Json::Arr(s)) if s == &[Json::Int(256), Json::Int(256)]);
    if !shape_ok {
        return Err(format!("bigram.weight shape must be [256, 256], got {:?}", entry.get("shape")));
    }
    let Some(Json::Arr(o)) = entry.get("data_offsets") else {
        return Err("bigram.weight: no data_offsets".to_string());
    };
    let (begin, end) = (header_usize(&o[0])?, header_usize(&o[1])?);
    if end - begin != VOCAB * VOCAB * 4 {
        return Err(format!("bigram.weight spans {} bytes, want {}", end - begin, VOCAB * VOCAB * 4));
    }
    // data_offsets count from the start of the data buffer, not of the file.
    let raw = &bytes[data_start + begin..data_start + end];
    Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

// ===========================================================================
// Sampling

/// PCG-XSH-RR 64/32 (O'Neill), seeded like the reference `pcg32_srandom_r`.
#[derive(Debug, Clone)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    pub fn new(seed: u64, seq: u64) -> Pcg32 {
        let mut r = Pcg32 { state: 0, inc: (seq << 1) | 1 };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// A uniform double in [0, 1) with 53 random bits, from two draws.
    pub fn uniform(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }
}

/// Picks the next token from `logits`. `temperature == 0` is greedy: the
/// largest logit, ties to the lowest id. Otherwise the softmax of
/// `logits / temperature` in f64, one uniform draw, inverse CDF. NaN logits
/// are never picked.
pub fn sample(logits: &[f32], temperature: f64, rng: &mut Pcg32) -> u8 {
    if temperature == 0.0 {
        let mut best: Option<usize> = None;
        for (i, &l) in logits.iter().enumerate() {
            if !l.is_nan() && best.is_none_or(|b| l > logits[b]) {
                best = Some(i);
            }
        }
        return best.unwrap_or(0) as u8;
    }
    let z: Vec<f64> = logits.iter().map(|&l| l as f64 / temperature).collect();
    let m = z.iter().copied().filter(|x| !x.is_nan()).fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = z.iter().map(|&x| if x.is_nan() || m == f64::NEG_INFINITY { 0.0 } else { (x - m).exp() }).collect();
    let total: f64 = w.iter().sum();
    let u = rng.uniform() * total;
    let mut cum = 0.0;
    let mut last = 0;
    for (i, &wi) in w.iter().enumerate() {
        if wi > 0.0 {
            cum += wi;
            last = i;
            if cum > u {
                return i as u8;
            }
        }
    }
    last as u8
}

/// One generation step: the logits after `prev`, then `sample`.
pub fn next_token(model: &Bigram, prev: u8, temperature: f64, rng: &mut Pcg32) -> Result<u8, String> {
    let logits = model.logits(prev)?;
    Ok(sample(&logits, temperature, rng))
}

// ===========================================================================
// The byte tokenizer, decoded incrementally (formats/tokenizer.md)

/// Turns a stream of byte ids into text one token at a time, holding back
/// the bytes of an incomplete UTF-8 sequence until it completes or fails.
#[derive(Debug, Default, Clone)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn new() -> Utf8Stream {
        Utf8Stream { pending: Vec::new() }
    }

    /// The text that `byte` completes (possibly "").
    pub fn push(&mut self, byte: u8) -> String {
        self.pending.push(byte);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..good]).expect("valid prefix"));
                    match e.error_len() {
                        // The tail could still become a character: keep it.
                        None => {
                            self.pending.drain(..good);
                            return out;
                        }
                        // An invalid sequence: one U+FFFD for its maximal subpart.
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..good + bad);
                        }
                    }
                }
            }
        }
    }

    /// Whatever is still held back, decoded with replacement.
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

// ===========================================================================
// Tracing: W3C traceparent in, one OTLP/HTTP JSON span out

/// `(trace_id, parent_span_id)` from a W3C `traceparent` header
/// (`00-<32 hex>-<16 hex>-<2 hex>`), or None when it is malformed, in which
/// case the server starts a new trace.
pub fn parse_traceparent(value: &str) -> Option<(String, String)> {
    let hex = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    let parts: Vec<&str> = value.trim().split('-').collect();
    let [version, trace, span, flags] = parts[..] else {
        return None;
    };
    let zeros = |s: &str| s.bytes().all(|b| b == b'0');
    if !hex(version, 2) || version == "ff" || !hex(trace, 32) || !hex(span, 16) || !hex(flags, 2) {
        return None;
    }
    if zeros(trace) || zeros(span) {
        return None;
    }
    Some((trace.to_string(), span.to_string()))
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// 64 random bits from the standard library's randomly keyed hasher.
pub fn random_u64() -> u64 {
    let mut h = RandomState::new().build_hasher();
    h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    h.write_u128(unix_nanos());
    h.finish()
}

/// `bytes` random bytes as lowercase hex, never all zeros.
pub fn random_hex(bytes: usize) -> String {
    loop {
        let mut s = String::with_capacity(bytes * 2);
        while s.len() < bytes * 2 {
            s.push_str(&format!("{:016x}", random_u64()));
        }
        s.truncate(bytes * 2);
        if s.bytes().any(|b| b != b'0') {
            return s;
        }
    }
}

/// Nanoseconds since the Unix epoch.
pub fn unix_nanos() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos())
}

/// An attribute value: OTLP's `stringValue` or `intValue`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttrValue {
    Str(String),
    Int(i64),
}

/// One finished span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub name: String,
    pub start_unix_nano: u128,
    pub end_unix_nano: u128,
    pub attributes: Vec<(String, AttrValue)>,
    /// True for a 5xx answer (OTel HTTP server convention).
    pub error: bool,
}

/// The OTLP/HTTP JSON body (`ExportTraceServiceRequest`) for one SERVER
/// span. Ids are lowercase hex; 64-bit integers are JSON strings.
pub fn otlp_json(service_name: &str, span: &Span) -> String {
    let attrs: Vec<String> = span
        .attributes
        .iter()
        .map(|(k, v)| {
            let value = match v {
                AttrValue::Str(s) => format!("{{\"stringValue\":{}}}", json_string(s)),
                AttrValue::Int(n) => format!("{{\"intValue\":\"{n}\"}}"),
            };
            format!("{{\"key\":{},\"value\":{value}}}", json_string(k))
        })
        .collect();
    let parent = span.parent_span_id.as_deref().map_or(String::new(), |p| format!("\"parentSpanId\":{},", json_string(p)));
    format!(
        concat!(
            "{{\"resourceSpans\":[{{\"resource\":{{\"attributes\":[{{\"key\":\"service.name\",\"value\":{{\"stringValue\":{}}}}}]}},",
            "\"scopeSpans\":[{{\"scope\":{{\"name\":\"tl-serve\",\"version\":\"0.1.0\"}},\"spans\":[{{",
            "\"traceId\":{},\"spanId\":{},{}\"name\":{},\"kind\":2,",
            "\"startTimeUnixNano\":\"{}\",\"endTimeUnixNano\":\"{}\",\"attributes\":[{}],\"status\":{{\"code\":{}}}",
            "}}]}}]}}]}}"
        ),
        json_string(service_name),
        json_string(&span.trace_id),
        json_string(&span.span_id),
        parent,
        json_string(&span.name),
        span.start_unix_nano,
        span.end_unix_nano,
        attrs.join(","),
        if span.error { 2 } else { 0 }
    )
}

/// POSTs `body` to `<endpoint>/v1/traces` (plain http only) and waits for a
/// 2xx status line. Gives up after two seconds.
pub fn export_span(endpoint: &str, body: &str) -> Result<(), String> {
    let rest = endpoint.strip_prefix("http://").ok_or("OTEL_EXPORTER_OTLP_ENDPOINT must start with http://")?;
    let (hostport, base) = rest.split_once('/').map_or((rest, ""), |(h, b)| (h, b));
    let hostport = if hostport.contains(':') { hostport.to_string() } else { format!("{hostport}:80") };
    let path = format!("/{}{}v1/traces", base.trim_end_matches('/'), if base.is_empty() { "" } else { "/" });
    let addr = hostport.to_socket_addrs().map_err(|e| format!("{hostport}: {e}"))?.next().ok_or("no address")?;
    let limit = Duration::from_secs(2);
    let mut s = TcpStream::connect_timeout(&addr, limit).map_err(|e| format!("connect {addr}: {e}"))?;
    s.set_read_timeout(Some(limit)).map_err(|e| e.to_string())?;
    s.set_write_timeout(Some(limit)).map_err(|e| e.to_string())?;
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {hostport}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    let mut status = String::new();
    BufReader::new(s).read_line(&mut status).map_err(|e| e.to_string())?;
    match status.split(' ').nth(1) {
        Some(code) if code.starts_with('2') => Ok(()),
        _ => Err(format!("collector answered {:?}", status.trim_end())),
    }
}

// ===========================================================================
// The server

/// Everything a connection needs, shared read-only across threads.
pub struct Server {
    pub model: Bigram,
    /// OTLP/HTTP base URL; None exports nothing.
    pub otlp_endpoint: Option<String>,
    /// `service.name` on exported spans.
    pub service_name: String,
}

impl Server {
    /// Loads the model and reads `OTEL_EXPORTER_OTLP_ENDPOINT` and
    /// `OTEL_SERVICE_NAME` (default `tl-engine`).
    pub fn from_env(model_dir: &Path) -> Result<Server, String> {
        let model = Bigram::load(model_dir)?;
        let otlp_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok().filter(|s| !s.is_empty());
        let service_name = std::env::var("OTEL_SERVICE_NAME").ok().filter(|s| !s.is_empty());
        Ok(Server { model, otlp_endpoint, service_name: service_name.unwrap_or_else(|| "tl-engine".to_string()) })
    }
}

/// Accepts connections forever, one thread each. Returns only when the
/// listener itself fails.
pub fn serve(listener: TcpListener, server: Arc<Server>) -> io::Result<()> {
    loop {
        let (stream, _) = listener.accept()?;
        let server = Arc::clone(&server);
        thread::spawn(move || handle_connection(stream, &server));
    }
}

/// What a completion produced, for the span.
struct Outcome {
    status: u16,
    model: String,
    prompt_tokens: usize,
    completion_tokens: usize,
}

/// Reads one request, answers it, closes the connection, then exports the
/// span of a completion request.
pub fn handle_connection(stream: TcpStream, server: &Server) {
    let start = unix_nanos();
    // Small writes (one SSE event) must leave now, not wait for more data.
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(10)));
    let Ok(read_half) = stream.try_clone() else {
        return;
    };
    let mut out = stream;
    let req = match read_request(&mut BufReader::new(read_half)) {
        Ok(req) => req,
        Err(e) if e.status == 0 => return,
        Err(e) => {
            let err = ApiError { status: e.status, ..ApiError::invalid(None, e.message) };
            let _ = write_json(&mut out, err.status, &[], &err.to_json());
            return;
        }
    };
    let mut extra = Vec::new();
    if let Some(id) = req.header("x-request-id").filter(|v| valid_request_id(v)) {
        extra.push(("X-Request-Id".to_string(), id.to_string()));
    }
    let outcome = match (req.method.as_str(), req.path()) {
        ("GET", "/healthz") => {
            let _ = write_json(&mut out, 200, &extra, "{\"status\":\"ok\"}");
            None
        }
        ("POST", "/v1/completions") => Some(complete(&req, server, &mut out, &extra)),
        (_, "/v1/completions") => {
            let mut err = ApiError::invalid(None, "use POST");
            err.status = 405;
            err.code = Some("method_not_allowed".to_string());
            extra.push(("Allow".to_string(), "POST".to_string()));
            let _ = write_json(&mut out, 405, &extra, &err.to_json());
            None
        }
        (_, path) => {
            let mut err = ApiError::invalid(None, format!("no route for {path}"));
            err.status = 404;
            err.code = Some("not_found".to_string());
            let _ = write_json(&mut out, 404, &extra, &err.to_json());
            None
        }
    };
    let _ = out.flush();
    let _ = out.shutdown(Shutdown::Both);
    drop(out);

    let (Some(o), Some(endpoint)) = (outcome, server.otlp_endpoint.as_deref()) else {
        return;
    };
    let (trace_id, parent) = match req.header("traceparent").and_then(parse_traceparent) {
        Some((t, p)) => (t, Some(p)),
        None => (random_hex(16), None),
    };
    let span = Span {
        trace_id,
        span_id: random_hex(8),
        parent_span_id: parent,
        name: "POST /v1/completions".to_string(),
        start_unix_nano: start,
        end_unix_nano: unix_nanos(),
        attributes: vec![
            ("http.request.method".to_string(), AttrValue::Str("POST".to_string())),
            ("http.route".to_string(), AttrValue::Str("/v1/completions".to_string())),
            ("http.response.status_code".to_string(), AttrValue::Int(i64::from(o.status))),
            ("gen_ai.operation.name".to_string(), AttrValue::Str("text_completion".to_string())),
            ("gen_ai.request.model".to_string(), AttrValue::Str(o.model)),
            ("gen_ai.usage.input_tokens".to_string(), AttrValue::Int(o.prompt_tokens as i64)),
            ("gen_ai.usage.output_tokens".to_string(), AttrValue::Int(o.completion_tokens as i64)),
        ],
        error: o.status >= 500,
    };
    if let Err(e) = export_span(endpoint, &otlp_json(&server.service_name, &span)) {
        eprintln!("tl-serve: span export failed: {e}");
    }
}

/// An X-Request-Id we are willing to echo: 1 to 128 visible ASCII bytes.
fn valid_request_id(v: &str) -> bool {
    !v.is_empty() && v.len() <= 128 && v.bytes().all(|b| (0x21..=0x7e).contains(&b))
}

/// The reason phrase of the statuses this server sends.
fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        413 => "Content Too Large",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        _ => "Unknown",
    }
}

/// Writes a status line and headers. `content_length: None` is a stream
/// that ends when the connection closes.
fn write_head<W: Write>(
    w: &mut W,
    status: u16,
    content_type: &str,
    content_length: Option<usize>,
    extra: &[(String, String)],
) -> io::Result<()> {
    let mut head = format!("HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\n", reason(status));
    match content_length {
        Some(n) => head.push_str(&format!("Content-Length: {n}\r\n")),
        None => head.push_str("Cache-Control: no-cache\r\n"),
    }
    head.push_str("Connection: close\r\n");
    for (k, v) in extra {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    w.write_all(head.as_bytes())
}

/// A whole JSON response; Content-Length counts the body's bytes.
fn write_json<W: Write>(w: &mut W, status: u16, extra: &[(String, String)], body: &str) -> io::Result<()> {
    write_head(w, status, "application/json", Some(body.len()), extra)?;
    w.write_all(body.as_bytes())?;
    w.flush()
}

/// One SSE event: `data: <payload>\n\n`.
pub fn sse_event(payload: &str) -> String {
    format!("data: {payload}\n\n")
}

/// One completion chunk (or, with `usage`, the whole completion) as JSON.
fn completion_json(id: &str, created: u64, model: &str, text: &str, finish: Option<&str>, usage: Option<(usize, usize)>) -> String {
    let finish = finish.map_or("null".to_string(), json_string);
    let mut s = format!(
        "{{\"id\":{},\"object\":\"text_completion\",\"created\":{created},\"model\":{},\"choices\":[{{\"index\":0,\"text\":{},\"finish_reason\":{finish}}}]",
        json_string(id),
        json_string(model),
        json_string(text)
    );
    if let Some((p, c)) = usage {
        s.push_str(&format!(",\"usage\":{{\"prompt_tokens\":{p},\"completion_tokens\":{c},\"total_tokens\":{}}}", p + c));
    }
    s.push('}');
    s
}

/// POST /v1/completions: validate, generate, answer as JSON or SSE.
fn complete(req: &Request, server: &Server, out: &mut TcpStream, extra: &[(String, String)]) -> Outcome {
    let creq = match parse_completion(&req.body) {
        Ok(c) => c,
        Err(e) => {
            let _ = write_json(out, e.status, extra, &e.to_json());
            return Outcome { status: e.status, model: String::new(), prompt_tokens: 0, completion_tokens: 0 };
        }
    };
    let prompt = creq.prompt.as_bytes();
    let mut outcome = Outcome { status: 200, model: creq.model.clone(), prompt_tokens: prompt.len(), completion_tokens: 0 };
    let seed = creq.seed.unwrap_or_else(|| random_u64() >> 1);
    let mut rng = Pcg32::new(seed, 54); // stream 54, the reference default
    let id = format!("cmpl-{}", random_hex(12));
    let created = (unix_nanos() / 1_000_000_000) as u64;
    let mut prev = *prompt.last().expect("prompt is non-empty");

    if !creq.stream {
        let mut ids = Vec::with_capacity(creq.max_tokens as usize);
        for _ in 0..creq.max_tokens {
            match next_token(&server.model, prev, creq.temperature, &mut rng) {
                Ok(t) => {
                    ids.push(t);
                    prev = t;
                }
                Err(e) => {
                    let err = ApiError { status: 500, error_type: "server_error".to_string(), message: e, param: None, code: None };
                    let _ = write_json(out, 500, extra, &err.to_json());
                    outcome.status = 500;
                    return outcome;
                }
            }
        }
        let text = String::from_utf8_lossy(&ids);
        let usage = Some((prompt.len(), ids.len()));
        let body = completion_json(&id, created, &creq.model, &text, Some("length"), usage);
        let _ = write_json(out, 200, extra, &body);
        outcome.completion_tokens = ids.len();
        return outcome;
    }

    if write_head(out, 200, "text/event-stream", None, extra).is_err() {
        return outcome;
    }
    let mut decoder = Utf8Stream::new();
    for i in 0..creq.max_tokens {
        let tok = match next_token(&server.model, prev, creq.temperature, &mut rng) {
            Ok(t) => t,
            Err(e) => {
                // After the first byte the status is already 200: report the
                // failure in the stream and end it, never truncate silently.
                let err = ApiError { status: 500, error_type: "server_error".to_string(), message: e, param: None, code: None };
                let _ = out.write_all(sse_event(&err.to_json()).as_bytes());
                outcome.status = 500;
                return outcome;
            }
        };
        prev = tok;
        outcome.completion_tokens += 1;
        let last = i + 1 == creq.max_tokens;
        let mut text = decoder.push(tok);
        if last {
            text.push_str(&decoder.finish());
        }
        let chunk = completion_json(&id, created, &creq.model, &text, if last { Some("length") } else { None }, None);
        if out.write_all(sse_event(&chunk).as_bytes()).is_err() {
            return outcome; // the client went away; stop generating
        }
    }
    let _ = out.write_all(sse_event("[DONE]").as_bytes());
    outcome
}
