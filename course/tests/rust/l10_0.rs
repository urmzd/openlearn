//! L10.0 course tests: the std-only first endpoint (tl-serve http v0).
//!
//! Annotated exemplars (DESIGN 5.12). Every model these tests serve is
//! written by hand, byte by byte, in `write_model` below, following
//! formats/safetensors.md, so no fixture file is involved. Server tests start
//! your `http::serve` on 127.0.0.1:0 in a thread and speak raw HTTP/1.1 to
//! it; the bytes are spelled out in each test. JSON responses are read with
//! the small parser in `mod j`, never with yours.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use tl_serve::http::{self, Bigram, CompletionRequest, Json, Pcg32, Server, Utf8Stream};

// ---------------------------------------------------------------------------
// helpers: model directories written by hand

static SEQ: AtomicUsize = AtomicUsize::new(0);

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ol-l10_0-{}-{}-{tag}", std::process::id(), SEQ.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A 256 x 256 weight: every entry `fill`, then each (from, to, value) edge.
fn weight(edges: &[(u8, u8, f32)], fill: f32) -> Vec<f32> {
    let mut w = vec![fill; 256 * 256];
    for &(from, to, v) in edges {
        w[from as usize * 256 + to as usize] = v;
    }
    w
}

/// A safetensors file: u64 LE header length, the header padded with spaces
/// to a multiple of 8, then the data buffer.
fn safetensors(header: &str, data: &[u8]) -> Vec<u8> {
    let mut h = header.as_bytes().to_vec();
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(data);
    out
}

fn le_bytes(w: &[f32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

const CONFIG: &str = r#"{"tl_arch":"bigram","tl_tokenizer":"bytes","vocab_size":256,"tl_format":1}"#;

/// A model directory holding `w`. The metadata is long on purpose: it makes
/// the header (and so the data buffer's start) far from offset 0.
fn write_model_with(w: &[f32], config: &str) -> PathBuf {
    let d = tmpdir("model");
    let data = le_bytes(w);
    let header = format!(
        r#"{{"__metadata__":{{"format":"tinyllm","note":"{}"}},"bigram.weight":{{"dtype":"F32","shape":[256,256],"data_offsets":[0,{}]}}}}"#,
        "x".repeat(300),
        data.len()
    );
    std::fs::write(d.join("model.safetensors"), safetensors(&header, &data)).unwrap();
    std::fs::write(d.join("config.json"), config).unwrap();
    d
}

fn write_model(edges: &[(u8, u8, f32)], fill: f32) -> PathBuf {
    write_model_with(&weight(edges, fill), CONFIG)
}

/// The chapter's worked model: after byte i, byte i + 1 (logit 2, rest 0).
fn succ_model() -> PathBuf {
    let edges: Vec<(u8, u8, f32)> = (0..=255u8).map(|i| (i, i.wrapping_add(1), 2.0)).collect();
    write_model(&edges, 0.0)
}

/// After 'x' comes 0xC3, then 0xA9 (together "é"), then 0xFF (never valid
/// UTF-8), then row 0xFF is all zeros, so greedy picks id 0 from then on.
fn utf8_model() -> PathBuf {
    write_model(&[(b'x', 0xC3, 3.0), (0xC3, 0xA9, 3.0), (0xA9, 0xFF, 3.0)], 0.0)
}

// ---------------------------------------------------------------------------
// helpers: a server in a thread, raw HTTP

fn start(model_dir: &Path, otlp: Option<String>) -> SocketAddr {
    let model = Bigram::load(model_dir).expect("Bigram::load on a valid model directory");
    let server = Server { model, otlp_endpoint: otlp, service_name: "tl-engine-test".to_string() };
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    thread::spawn(move || http::serve(listener, Arc::new(server)));
    addr
}

struct Resp {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Resp {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
    fn text(&self) -> String {
        String::from_utf8(self.body.clone()).expect("response body is UTF-8")
    }
    fn json(&self) -> j::V {
        j::parse(&self.text())
    }
}

fn parse_resp(raw: &[u8]) -> Resp {
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap_or_else(|| panic!("no blank line after the headers: {:?}", String::from_utf8_lossy(raw)));
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap();
    let mut parts = status_line.splitn(3, ' ');
    assert_eq!(parts.next(), Some("HTTP/1.1"), "status line {status_line:?}");
    let status = parts.next().and_then(|s| s.parse().ok()).unwrap_or_else(|| panic!("status line {status_line:?}"));
    let headers = lines
        .map(|l| {
            let (k, v) = l.split_once(':').unwrap_or_else(|| panic!("header line {l:?}"));
            (k.trim().to_string(), v.trim().to_string())
        })
        .collect();
    Resp { status, headers, body: raw[split + 4..].to_vec() }
}

fn send(addr: SocketAddr, raw: &[u8]) -> Resp {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    s.write_all(raw).unwrap();
    let mut out = Vec::new();
    s.read_to_end(&mut out).expect("read the response until the server closes");
    parse_resp(&out)
}

fn post(addr: SocketAddr, body: &str, extra: &str) -> Resp {
    let raw = format!("POST /v1/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{extra}\r\n{body}", body.len());
    send(addr, raw.as_bytes())
}

fn get(addr: SocketAddr, path: &str) -> Resp {
    send(addr, format!("GET {path} HTTP/1.1\r\nHost: t\r\n\r\n").as_bytes())
}

/// The `data:` payloads of an SSE body, checking the framing as it goes:
/// every event is exactly one `data: <payload>` line and a blank line.
fn sse_payloads(body: &str) -> Vec<String> {
    assert!(body.ends_with("\n\n"), "an SSE stream ends with a blank line: {body:?}");
    let events: Vec<&str> = body[..body.len() - 2].split("\n\n").collect();
    events
        .iter()
        .map(|e| {
            assert!(!e.contains('\n'), "an event is one `data:` line, got {e:?}");
            e.strip_prefix("data: ").unwrap_or_else(|| panic!("event {e:?} must start with `data: `")).to_string()
        })
        .collect()
}

/// A fake OTLP collector: answers 200 to every request and hands each
/// (request line, headers, body) to the returned channel.
fn collector() -> (String, mpsc::Receiver<(String, Vec<(String, String)>, String)>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut s) = conn else { continue };
            let mut r = BufReader::new(s.try_clone().unwrap());
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            let mut headers = Vec::new();
            let mut len = 0;
            loop {
                let mut h = String::new();
                r.read_line(&mut h).unwrap();
                let h = h.trim_end().to_string();
                if h.is_empty() {
                    break;
                }
                let (k, v) = h.split_once(':').unwrap();
                if k.eq_ignore_ascii_case("content-length") {
                    len = v.trim().parse().unwrap();
                }
                headers.push((k.trim().to_string(), v.trim().to_string()));
            }
            let mut body = vec![0; len];
            r.read_exact(&mut body).unwrap();
            let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}");
            let _ = tx.send((line.trim_end().to_string(), headers, String::from_utf8(body).unwrap()));
        }
    });
    (base, rx)
}

fn is_hex(s: &str, n: usize) -> bool {
    s.len() == n && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// A tiny JSON reader for checking responses, independent of the code under test.
mod j {
    #[derive(Debug, Clone, PartialEq)]
    pub enum V {
        Null,
        Bool(bool),
        Num(f64),
        Str(String),
        Arr(Vec<V>),
        Obj(Vec<(String, V)>),
    }

    impl V {
        pub fn get(&self, k: &str) -> &V {
            match self {
                V::Obj(kv) => kv.iter().find(|(a, _)| a == k).map(|(_, v)| v).unwrap_or_else(|| panic!("no key {k:?} in {self:?}")),
                _ => panic!("not an object: {self:?}"),
            }
        }
        pub fn has(&self, k: &str) -> bool {
            matches!(self, V::Obj(kv) if kv.iter().any(|(a, _)| a == k))
        }
        pub fn str(&self) -> &str {
            match self {
                V::Str(s) => s,
                _ => panic!("not a string: {self:?}"),
            }
        }
        pub fn num(&self) -> f64 {
            match self {
                V::Num(n) => *n,
                _ => panic!("not a number: {self:?}"),
            }
        }
        pub fn arr(&self) -> &Vec<V> {
            match self {
                V::Arr(a) => a,
                _ => panic!("not an array: {self:?}"),
            }
        }
    }

    pub fn parse(s: &str) -> V {
        let b: Vec<char> = s.chars().collect();
        let mut i = 0;
        let v = val(&b, &mut i);
        ws(&b, &mut i);
        assert_eq!(i, b.len(), "trailing characters in JSON {s:?}");
        v
    }

    fn ws(b: &[char], i: &mut usize) {
        while *i < b.len() && b[*i].is_whitespace() {
            *i += 1;
        }
    }

    fn val(b: &[char], i: &mut usize) -> V {
        ws(b, i);
        let rest: String = b[*i..].iter().take(5).collect();
        match b[*i] {
            '{' => {
                *i += 1;
                let mut kv = Vec::new();
                loop {
                    ws(b, i);
                    if b[*i] == '}' {
                        *i += 1;
                        return V::Obj(kv);
                    }
                    if b[*i] == ',' {
                        *i += 1;
                        continue;
                    }
                    let V::Str(k) = val(b, i) else { panic!("object key is not a string") };
                    ws(b, i);
                    assert_eq!(b[*i], ':');
                    *i += 1;
                    kv.push((k, val(b, i)));
                }
            }
            '[' => {
                *i += 1;
                let mut a = Vec::new();
                loop {
                    ws(b, i);
                    if b[*i] == ']' {
                        *i += 1;
                        return V::Arr(a);
                    }
                    if b[*i] == ',' {
                        *i += 1;
                        continue;
                    }
                    a.push(val(b, i));
                }
            }
            '"' => {
                *i += 1;
                let mut s = String::new();
                let mut units: Vec<u16> = Vec::new();
                loop {
                    let c = b[*i];
                    *i += 1;
                    if c != '\\' || b[*i] != 'u' {
                        if !units.is_empty() {
                            s.push_str(&String::from_utf16(&units).expect("valid \\u escapes"));
                            units.clear();
                        }
                    }
                    match c {
                        '"' => return V::Str(s),
                        '\\' => {
                            let e = b[*i];
                            *i += 1;
                            match e {
                                'n' => s.push('\n'),
                                't' => s.push('\t'),
                                'r' => s.push('\r'),
                                'b' => s.push('\u{8}'),
                                'f' => s.push('\u{c}'),
                                'u' => {
                                    let h: String = b[*i..*i + 4].iter().collect();
                                    units.push(u16::from_str_radix(&h, 16).unwrap());
                                    *i += 4;
                                }
                                other => s.push(other),
                            }
                        }
                        c => {
                            assert!(c as u32 >= 0x20, "raw control character in a JSON string");
                            s.push(c)
                        }
                    }
                }
            }
            't' if rest.starts_with("true") => {
                *i += 4;
                V::Bool(true)
            }
            'f' if rest.starts_with("false") => {
                *i += 5;
                V::Bool(false)
            }
            'n' if rest.starts_with("null") => {
                *i += 4;
                V::Null
            }
            _ => {
                let start = *i;
                while *i < b.len() && "+-.eE0123456789".contains(b[*i]) {
                    *i += 1;
                }
                let t: String = b[start..*i].iter().collect();
                V::Num(t.parse().unwrap_or_else(|_| panic!("bad JSON at {rest:?}")))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Candle-free endpoint math

#[test]
fn bigram_logits_select_the_weight_row() {
    // WHY: the endpoint's one-hot previous token selects a row directly.
    // KIND: unit
    // CATCHES: s02
    // CHAPTER: L10.0 section 3
    let mut w = vec![0.0f32; 256 * 256];
    w[3 * 256 + 7] = 2.5;
    let m = Bigram::from_weight(w).unwrap();
    let logits = m.logits(3).unwrap();
    assert_eq!(logits[7], 2.5);
    assert!(logits.iter().enumerate().all(|(i, x)| i == 7 || *x == 0.0));
}

// ---------------------------------------------------------------------------
// HTTP and JSON

#[test]
fn parses_request_line_headers_and_body() {
    // WHY: the worked request: request line, headers until the blank line,
    //      then exactly Content-Length bytes of body. The query string is not
    //      part of the path.
    // KIND: unit
    // CHAPTER: L10.0 section 3
    let raw = b"POST /v1/completions?x=1 HTTP/1.1\r\nHost: a\r\nContent-Length: 2\r\nX-Request-Id:  r1 \r\n\r\n{}";
    let req = http::read_request(&mut &raw[..]).unwrap();
    assert_eq!((req.method.as_str(), req.path(), req.target.as_str()), ("POST", "/v1/completions", "/v1/completions?x=1"));
    assert_eq!(req.header("x-request-id"), Some("r1"));
    assert_eq!(req.body, b"{}");
}

#[test]
fn header_names_are_case_insensitive() {
    // WHY: curl sends Content-Length, a Go proxy may send content-length;
    //      HTTP header names ignore case. A case-sensitive lookup misses the
    //      body and answers 411.
    // KIND: boundary
    // CATCHES: s07
    // CHAPTER: L10.0 section 5, Pitfalls
    let raw = b"POST /v1/completions HTTP/1.1\r\ncontent-length: 2\r\n\r\n{}";
    let req = http::read_request(&mut &raw[..]).unwrap();
    assert_eq!(req.body, b"{}");
    assert_eq!(req.header("CONTENT-LENGTH"), Some("2"));
}

#[test]
fn malformed_requests_are_rejected() {
    // WHY: every way a request can be unreadable has its own status, and
    //      none of them is a panic: 400 bad syntax, 411 no length, 413 too
    //      big, 501 chunked; 0 means the client left before sending anything.
    // KIND: boundary
    // CHAPTER: L10.0 section 2
    let status = |raw: &[u8]| http::read_request(&mut &raw[..]).map(|_| 200).unwrap_or_else(|e| e.status);
    assert_eq!(status(b"GARBAGE\r\n\r\n"), 400);
    assert_eq!(status(b"GET /healthz\r\n\r\n"), 400);
    assert_eq!(status(b"GET healthz HTTP/1.1\r\n\r\n"), 400);
    assert_eq!(status(b"POST /v1/completions HTTP/1.1\r\n\r\n"), 411);
    assert_eq!(status(b"POST /v1/completions HTTP/1.1\r\nContent-Length: 2000000\r\n\r\n"), 413);
    assert_eq!(status(b"POST /v1/completions HTTP/1.1\r\nContent-Length: -1\r\n\r\n"), 400);
    assert_eq!(status(b"POST /v1/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"), 501);
    assert_eq!(status(b""), 0);
    assert_eq!(status(b"GET /healthz HTTP/1.1\r\nHost: t\r\n\r\n"), 200);
}

#[test]
fn json_parses_nested_values_and_escapes() {
    // WHY: requests, config.json, and the safetensors header all go through
    //      this parser: nesting, integers vs floats, escapes including a
    //      surrogate pair (😀 is ONE character), and clean errors.
    // KIND: unit
    // CHAPTER: L10.0 section 2
    let v = http::parse_json(r#" {"a": [1, -2.5e1, null, true], "s": "hé😀\n", "n": 9223372036854775807} "#).unwrap();
    assert_eq!(
        v,
        Json::Obj(vec![
            ("a".into(), Json::Arr(vec![Json::Int(1), Json::Float(-25.0), Json::Null, Json::Bool(true)])),
            ("s".into(), Json::Str("h\u{e9}\u{1F600}\n".into())),
            ("n".into(), Json::Int(i64::MAX)),
        ])
    );
    for bad in [r#"{"a":}"#, "[1,", "01", r#"{"a":1} x"#, r#""\ud800""#, "\"raw\u{1}\""] {
        assert!(http::parse_json(bad).is_err(), "{bad:?} must not parse");
    }
}

#[test]
fn json_string_escapes_quotes_and_control_bytes() {
    // WHY: a byte model can emit any byte, including `"`, `\`, a newline, or
    //      NUL. Unescaped, one such token breaks the JSON of its chunk and
    //      the client's parser stops the stream.
    // KIND: boundary
    // CATCHES: s05
    // CHAPTER: L10.0 section 5, Pitfalls
    assert_eq!(http::json_string("a\"b\\c\n\u{1}\u{e9}"), "\"a\\\"b\\\\c\\n\\u0001\u{e9}\"");
    let s = "tab\t nul\u{0} quote\" \u{FFFD}";
    assert_eq!(http::parse_json(&http::json_string(s)), Ok(Json::Str(s.into())));
}

#[test]
fn completion_request_validation() {
    // WHY: the contract's CompletionRequest: defaults when a field is absent,
    //      unknown fields ignored, and a 400 whose `param` names the field
    //      that is missing, mistyped, or out of range.
    // KIND: boundary
    // CHAPTER: L10.0 section 4
    let ok = http::parse_completion(br#"{"model":"m","prompt":"hi","whatever":[1]}"#).unwrap();
    assert_eq!(ok, CompletionRequest { model: "m".into(), prompt: "hi".into(), max_tokens: 16, temperature: 1.0, seed: None, stream: false });
    let full = http::parse_completion(br#"{"model":"m","prompt":"hi","max_tokens":4096,"temperature":0,"seed":7,"stream":true}"#).unwrap();
    assert_eq!((full.max_tokens, full.temperature, full.seed, full.stream), (4096, 0.0, Some(7), true));
    let param = |body: &str| http::parse_completion(body.as_bytes()).unwrap_err();
    for (body, p) in [
        (r#"{"prompt":"hi"}"#, "model"),
        (r#"{"model":"m"}"#, "prompt"),
        (r#"{"model":"m","prompt":""}"#, "prompt"),
        (r#"{"model":"m","prompt":"hi","max_tokens":0}"#, "max_tokens"),
        (r#"{"model":"m","prompt":"hi","max_tokens":4097}"#, "max_tokens"),
        (r#"{"model":"m","prompt":"hi","max_tokens":2.5}"#, "max_tokens"),
        (r#"{"model":"m","prompt":"hi","temperature":-1}"#, "temperature"),
        (r#"{"model":"m","prompt":"hi","temperature":2.5}"#, "temperature"),
        (r#"{"model":"m","prompt":"hi","seed":-1}"#, "seed"),
        (r#"{"model":"m","prompt":"hi","stream":"yes"}"#, "stream"),
    ] {
        let e = param(body);
        assert_eq!((e.status, e.param.as_deref()), (400, Some(p)), "{body}");
        assert_eq!(e.error_type, "invalid_request_error");
    }
    assert_eq!(param("{not json").param, None);
    assert_eq!(param("[1]").status, 400);
}

// ---------------------------------------------------------------------------
// the model

#[test]
fn loads_hand_written_checkpoint() {
    // WHY: the engine reads exactly what L0.0 writes: config.json plus
    //      model.safetensors with bigram.weight, F32 [256, 256]. data_offsets
    //      count from the start of the DATA BUFFER (8 + header length), not
    //      of the file; reading from the wrong base gives garbage weights.
    // KIND: unit
    // CATCHES: s01
    // CHAPTER: L10.0 section 3
    let m = Bigram::load(&succ_model()).unwrap();
    let after_a = m.logits(b'a').unwrap();
    assert_eq!(after_a.len(), 256);
    assert_eq!(after_a[b'b' as usize], 2.0);
    assert_eq!(after_a.iter().filter(|&&x| x != 0.0).count(), 1);
    assert_eq!(m.logits(255).unwrap()[0], 2.0, "row 255 wraps to id 0");
}

#[test]
fn rejects_bad_checkpoints() {
    // WHY: the reader rules of formats/safetensors.md: a header length past
    //      the end, another dtype, another shape, tensors that do not tile
    //      the data buffer, and a config.json for another architecture or
    //      tokenizer are errors at load time, not wrong answers later.
    // KIND: boundary
    // CATCHES: s12
    // CHAPTER: L10.0 section 5, Pitfalls
    let w = le_bytes(&weight(&[], 0.0));
    let n = w.len();
    let bad = |header: String, data: &[u8]| http::read_bigram_weight(&safetensors(&header, data));
    let entry = |dtype: &str, shape: &str, end: usize| format!(r#"{{"bigram.weight":{{"dtype":"{dtype}","shape":{shape},"data_offsets":[0,{end}]}}}}"#);
    assert!(bad(entry("F32", "[256,256]", n), &w).is_ok());
    assert!(bad(entry("F16", "[256,256]", n), &w).is_err(), "contract v0 reads F32 only");
    assert!(bad(entry("F32", "[256,255]", n), &w).is_err(), "wrong shape");
    assert!(bad(entry("F32", "[256,256]", n), &w[..n - 4]).is_err(), "data shorter than data_offsets");
    let mut longer = w.clone();
    longer.extend_from_slice(&[0; 4]);
    assert!(bad(entry("F32", "[256,256]", n), &longer).is_err(), "trailing bytes after the last tensor");
    let mut huge = 1_000_000u64.to_le_bytes().to_vec();
    huge.extend_from_slice(b"{}");
    assert!(http::read_bigram_weight(&huge).is_err(), "header length past the end of the file");
    assert!(http::read_bigram_weight(&[1, 2, 3]).is_err());
    let llama = write_model_with(&weight(&[], 0.0), r#"{"tl_arch":"llama","tl_tokenizer":"bytes","vocab_size":256}"#);
    assert!(Bigram::load(&llama).is_err(), "the tracer engine serves tl_arch bigram only");
    let file_tok = write_model_with(&weight(&[], 0.0), r#"{"tl_arch":"bigram","vocab_size":256}"#);
    assert!(Bigram::load(&file_tok).is_err(), "tl_tokenizer defaults to file; the tracer needs bytes");
}

// ---------------------------------------------------------------------------
// sampling and decoding

#[test]
fn pcg32_matches_reference_stream() {
    // WHY: O'Neill's reference pcg32_srandom_r(42, 54) stream; a shift or
    //      rotate off by one gives numbers that look random and are wrong.
    // KIND: golden
    // CHAPTER: L10.0 section 2
    let mut r = Pcg32::new(42, 54);
    let got: Vec<u32> = (0..6).map(|_| r.next_u32()).collect();
    assert_eq!(got, [0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e]);
    let u = Pcg32::new(1, 54).uniform();
    assert!((0.0..1.0).contains(&u));
}

#[test]
fn greedy_ties_go_to_lowest_id() {
    // WHY: temperature 0 is greedy with ties to the LOWEST id (D11), so
    //      every implementation that follows the rule emits the same bytes;
    //      NaN is never the maximum.
    // KIND: boundary
    // CATCHES: s04
    // CHAPTER: L10.0 section 5, Pitfalls
    let mut rng = Pcg32::new(0, 54);
    let mut l = vec![0.0f32; 256];
    l[20] = 1.0;
    l[10] = 1.0;
    assert_eq!(http::sample(&l, 0.0, &mut rng), 10);
    assert_eq!(http::sample(&[0.0f32; 256], 0.0, &mut rng), 0);
    let mut n = vec![1.0f32; 256];
    n[0] = f32::NAN;
    assert_eq!(http::sample(&n, 0.0, &mut rng), 1);
}

#[test]
fn sampling_skips_nan_logits() {
    // WHY: a NaN logit in the selected weight row has no
    //      probability; sampling must never return its id, at any temperature.
    // KIND: boundary
    // CHAPTER: L10.0 section 5, Pitfalls
    let mut l = vec![f32::NAN; 256];
    l[65] = 0.0;
    l[66] = 0.0;
    let mut rng = Pcg32::new(3, 54);
    for _ in 0..200 {
        let t = http::sample(&l, 1.0, &mut rng);
        assert!(t == 65 || t == 66, "sampled {t}, whose logit is NaN");
    }
}

#[test]
fn temperature_sampling_matches_softmax() {
    // WHY: the inverse-CDF sampler draws from softmax(logits / T). With
    //      p(A) = 0.75, p(B) = 0.25 and T = 0.5 the tempered odds are
    //      0.75^2 : 0.25^2 = 0.9 : 0.1. Chi-square over 2000 draws at a fixed
    //      seed, p > 1e-3 (chi2 < 10.83, 1 degree of freedom). Multiplying by
    //      T instead of dividing gives 0.63 : 0.37 and fails.
    // KIND: statistical
    // CATCHES: s03
    // CHAPTER: L10.0 section 2
    let mut l = vec![-30.0f32; 256];
    l[b'A' as usize] = 0.75f32.ln();
    l[b'B' as usize] = 0.25f32.ln();
    let mut rng = Pcg32::new(2024, 54);
    let (mut a, mut b) = (0.0f64, 0.0f64);
    for _ in 0..2000 {
        match http::sample(&l, 0.5, &mut rng) {
            b'A' => a += 1.0,
            b'B' => b += 1.0,
            t => panic!("sampled {t}, whose probability is about e^-60"),
        }
    }
    let chi2 = (a - 1800.0).powi(2) / 1800.0 + (b - 200.0).powi(2) / 200.0;
    assert!(chi2 < 10.83, "A={a} B={b}: chi2 {chi2:.2} rejects the 0.9 : 0.1 split");
}

#[test]
fn utf8_stream_holds_back_incomplete_sequences() {
    // WHY: one token is one byte, and "é" is two. Decoding each byte alone
    //      would send two U+FFFD; the stream instead emits "" for 0xC3, "é"
    //      for 0xA9, and U+FFFD for 0xFF, which is never valid UTF-8
    //      (formats/tokenizer.md, worked example).
    // KIND: boundary
    // CATCHES: s06
    // CHAPTER: L10.0 section 5, Pitfalls
    let mut d = Utf8Stream::new();
    assert_eq!([d.push(0xC3), d.push(0xA9), d.push(0xFF)], ["", "\u{e9}", "\u{FFFD}"]);
    assert_eq!(d.finish(), "");
    let mut d = Utf8Stream::new();
    assert_eq!([d.push(0xE2), d.push(0x82)], ["", ""]);
    assert_eq!(d.finish(), "\u{FFFD}", "an unfinished sequence is one U+FFFD at the end");
    let mut d = Utf8Stream::new();
    assert_eq!(d.push(b'h'), "h");
}

// ---------------------------------------------------------------------------
// the server, end to end over TCP

#[test]
fn hand_example_completion() {
    // WHY: the chapter's worked example over the wire: the succ model,
    //      prompt "a", greedy, 3 tokens is "bcd", and the response has every
    //      field the v0 Completion schema requires.
    // KIND: unit
    // CATCHES: s02
    // CHAPTER: L10.0 section 3
    let addr = start(&succ_model(), None);
    let r = post(addr, r#"{"model":"tracer-test","prompt":"a","max_tokens":3,"temperature":0}"#, "");
    assert_eq!(r.status, 200, "{}", r.text());
    assert_eq!(r.header("content-type"), Some("application/json"));
    let v = r.json();
    assert!(v.get("id").str().starts_with("cmpl-"));
    assert_eq!(v.get("object").str(), "text_completion");
    assert_eq!(v.get("model").str(), "tracer-test", "v0 echoes the request's model");
    assert!(v.get("created").num() > 1.6e9, "created is Unix time in seconds");
    let c = &v.get("choices").arr()[0];
    assert_eq!((c.get("index").num(), c.get("text").str(), c.get("finish_reason").str()), (0.0, "bcd", "length"));
    let u = v.get("usage");
    assert_eq!((u.get("prompt_tokens").num(), u.get("completion_tokens").num(), u.get("total_tokens").num()), (1.0, 3.0, 4.0));
}

#[test]
fn stream_framing_is_exact() {
    // WHY: the SSE worked example: text/event-stream with no Content-Length,
    //      one `data: <chunk>` event per token (same id on every chunk,
    //      finish_reason null until the last), then `data: [DONE]`. Clients
    //      split on the blank line; a missing \n merges two events.
    // KIND: conformance
    // CATCHES: s08
    // CHAPTER: L10.0 section 3
    let addr = start(&succ_model(), None);
    let r = post(addr, r#"{"model":"m","prompt":"a","max_tokens":2,"temperature":0,"stream":true}"#, "");
    assert_eq!(r.status, 200);
    assert_eq!(r.header("content-type"), Some("text/event-stream"));
    assert_eq!(r.header("content-length"), None, "a stream ends when the connection closes");
    let p = sse_payloads(&r.text());
    assert_eq!(p.len(), 3, "two chunks and [DONE]: {p:?}");
    assert_eq!(p[2], "[DONE]");
    let (c0, c1) = (j::parse(&p[0]), j::parse(&p[1]));
    assert_eq!(c0.get("id"), c1.get("id"));
    assert_eq!(c0.get("object").str(), "text_completion");
    let ch = |c: &j::V| c.get("choices").arr()[0].clone();
    assert_eq!((ch(&c0).get("text").str(), ch(&c0).get("finish_reason")), ("b", &j::V::Null));
    assert_eq!((ch(&c1).get("text").str(), ch(&c1).get("finish_reason").str()), ("c", "length"));
}

#[test]
fn max_tokens_counts_generated_tokens_only() {
    // WHY: max_tokens bounds the NEW tokens; the prompt is counted apart in
    //      usage.prompt_tokens, in bytes ("héllo" is 6). One chunk per
    //      generated token in the stream.
    // KIND: conformance
    // CATCHES: s13
    // CHAPTER: L10.0 section 4
    let addr = start(&succ_model(), None);
    let r = post(addr, r#"{"model":"m","prompt":"héllo","max_tokens":4,"temperature":0}"#, "");
    let v = r.json();
    assert_eq!(v.get("choices").arr()[0].get("text").str(), "pqrs");
    let u = v.get("usage");
    assert_eq!((u.get("prompt_tokens").num(), u.get("completion_tokens").num(), u.get("total_tokens").num()), (6.0, 4.0, 10.0));
    let s = post(addr, r#"{"model":"m","prompt":"hello","max_tokens":4,"temperature":0,"stream":true}"#, "");
    assert_eq!(sse_payloads(&s.text()).len(), 5, "4 chunks and [DONE]");
}

#[test]
fn stream_equals_nonstream_at_temperature_0() {
    // WHY: the concatenated chunk texts equal the non-stream text. With the
    //      utf8 model the bytes are C3 A9 FF 00 00: chunk texts "", "é",
    //      U+FFFD, NUL, NUL, and the NULs travel as \u0000 inside the JSON.
    // KIND: differential
    // CHAPTER: L10.0 section 4
    let addr = start(&utf8_model(), None);
    let body = r#"{"model":"m","prompt":"x","max_tokens":5,"temperature":0}"#;
    let whole = post(addr, body, "").json().get("choices").arr()[0].get("text").str().to_string();
    assert_eq!(whole, "\u{e9}\u{FFFD}\u{0}\u{0}");
    let s = post(addr, &body.replace("}", r#","stream":true}"#), "");
    let p = sse_payloads(&s.text());
    let texts: Vec<String> = p[..p.len() - 1].iter().map(|c| j::parse(c).get("choices").arr()[0].get("text").str().to_string()).collect();
    assert_eq!(texts, ["", "\u{e9}", "\u{FFFD}", "\u{0}", "\u{0}"]);
    assert_eq!(texts.concat(), whole);
}

#[test]
fn seed_reproduces_sampling() {
    // WHY: with temperature > 0, the same seed on the same engine gives the
    //      same text (the v0 Determinism rule), because every token draws
    //      from a PCG32 seeded by the request, never from global state.
    // KIND: property
    // CHAPTER: L10.0 section 4
    let addr = start(&write_model(&[], 0.0), None);
    let body = r#"{"model":"m","prompt":"a","max_tokens":16,"temperature":1.0,"seed":1234}"#;
    let a = post(addr, body, "").json().get("choices").arr()[0].get("text").str().to_string();
    let b = post(addr, body, "").json().get("choices").arr()[0].get("text").str().to_string();
    assert_eq!(a, b);
    let other = post(addr, &body.replace("1234", "1235"), "").json().get("choices").arr()[0].get("text").str().to_string();
    assert_ne!(a, other, "16 uniform bytes from two seeds are equal with probability 2^-128");
}

#[test]
fn errors_use_the_openai_shape() {
    // WHY: every error body is {"error": {message, type, param, code}} as
    //      application/json, so an OpenAI client can show it: 400 names the
    //      field, unknown paths are 404, the wrong method is 405.
    // KIND: conformance
    // CHAPTER: L10.0 section 4
    let addr = start(&succ_model(), None);
    let r = post(addr, r#"{"model":"m","prompt":"a","temperature":-1}"#, "");
    assert_eq!((r.status, r.header("content-type")), (400, Some("application/json")));
    let e = r.json();
    let e = e.get("error");
    assert_eq!((e.get("type").str(), e.get("param").str()), ("invalid_request_error", "temperature"));
    assert!(!e.get("message").str().is_empty());
    assert_eq!(e.get("code"), &j::V::Null);
    let bad = post(addr, "{oops", "");
    assert_eq!((bad.status, bad.json().get("error").get("param")), (400, &j::V::Null));
    let nf = get(addr, "/nope");
    assert_eq!(nf.status, 404);
    assert!(nf.json().get("error").has("message"));
    assert_eq!(get(addr, "/v1/completions").status, 405);
}

#[test]
fn healthz_answers_200() {
    // WHY: the runner and Kubernetes poll GET /healthz before sending
    //      traffic; it must answer while the process can serve.
    // KIND: conformance
    // CHAPTER: L10.0 section 4
    let addr = start(&succ_model(), None);
    assert_eq!(get(addr, "/healthz").status, 200);
}

#[test]
fn content_length_counts_bytes() {
    // WHY: Content-Length is a count of BYTES. "é" and U+FFFD are two and
    //      three bytes; counting characters truncates the body, and the
    //      client's JSON parser fails on the missing tail.
    // KIND: boundary
    // CATCHES: s14
    // CHAPTER: L10.0 section 5, Pitfalls
    let addr = start(&utf8_model(), None);
    let r = post(addr, r#"{"model":"m","prompt":"x","max_tokens":3,"temperature":0}"#, "");
    assert_eq!(r.header("content-length").map(|v| v.parse::<usize>().unwrap()), Some(r.body.len()));
    assert_eq!(r.json().get("choices").arr()[0].get("text").str(), "\u{e9}\u{FFFD}");
}

#[test]
fn echoes_x_request_id() {
    // WHY: the gateway tags every request with X-Request-Id and the engine
    //      echoes it, so one id finds the request in both logs.
    // KIND: unit
    // CHAPTER: L10.0 section 4
    let addr = start(&succ_model(), None);
    let r = post(addr, r#"{"model":"m","prompt":"a","max_tokens":1,"temperature":0}"#, "X-Request-Id: req-42\r\n");
    assert_eq!(r.header("x-request-id"), Some("req-42"));
    let r = post(addr, r#"{"model":"m","prompt":"a","max_tokens":1,"temperature":0}"#, "");
    assert_eq!(r.header("x-request-id"), None);
}

#[test]
fn exports_child_span_of_traceparent() {
    // WHY: with a traceparent header, the engine's SERVER span joins the
    //      caller's trace: same trace id, parentSpanId = the caller's span
    //      id, a fresh span id. It is posted as OTLP/HTTP JSON to
    //      <endpoint>/v1/traces, which is how Jaeger draws gateway -> engine.
    // KIND: conformance
    // CATCHES: s15
    // CHAPTER: L10.0 section 2
    let (endpoint, rx) = collector();
    let addr = start(&succ_model(), Some(endpoint));
    let tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";
    let r = post(addr, r#"{"model":"m","prompt":"a","max_tokens":2,"temperature":0}"#, &format!("traceparent: {tp}\r\n"));
    assert_eq!(r.status, 200);
    let (line, headers, body) = rx.recv_timeout(Duration::from_secs(5)).expect("a span is exported within 5 s");
    assert!(line.starts_with("POST /v1/traces "), "{line}");
    assert!(headers.iter().any(|(k, v)| k.eq_ignore_ascii_case("content-type") && v.starts_with("application/json")));
    let doc = j::parse(&body);
    let rs = &doc.get("resourceSpans").arr()[0];
    let res_attrs = rs.get("resource").get("attributes").arr();
    assert!(res_attrs.iter().any(|a| a.get("key").str() == "service.name" && a.get("value").get("stringValue").str() == "tl-engine-test"));
    let span = &rs.get("scopeSpans").arr()[0].get("spans").arr()[0];
    assert_eq!(span.get("traceId").str(), "4bf92f3577b34da6a3ce929d0e0e4736");
    assert_eq!(span.get("parentSpanId").str(), "00f067aa0ba902b7");
    assert!(is_hex(span.get("spanId").str(), 16) && span.get("spanId").str() != "00f067aa0ba902b7");
    assert_eq!(span.get("name").str(), "POST /v1/completions");
    assert_eq!(span.get("kind").num(), 2.0, "SPAN_KIND_SERVER");
    let t = |k: &str| span.get(k).str().parse::<u128>().unwrap();
    assert!(t("endTimeUnixNano") >= t("startTimeUnixNano") && t("startTimeUnixNano") > 1_600_000_000_000_000_000);
    let attr = |k: &str| span.get("attributes").arr().iter().find(|a| a.get("key").str() == k).map(|a| a.get("value").clone());
    assert_eq!(attr("http.response.status_code").map(|v| v.get("intValue").str().to_string()), Some("200".to_string()));
    assert_eq!(attr("gen_ai.request.model").map(|v| v.get("stringValue").str().to_string()), Some("m".to_string()));
}

#[test]
fn starts_a_new_trace_without_traceparent() {
    // WHY: no traceparent, or a malformed one (all-zero ids, uppercase hex,
    //      wrong lengths), means this request starts a NEW trace: a random
    //      non-zero trace id and no parent, never the zero ids copied through.
    // KIND: boundary
    // CHAPTER: L10.0 section 5, Pitfalls
    assert_eq!(http::parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"),
               Some(("4bf92f3577b34da6a3ce929d0e0e4736".to_string(), "00f067aa0ba902b7".to_string())));
    for bad in [
        "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
        "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",
        "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        "garbage",
    ] {
        assert_eq!(http::parse_traceparent(bad), None, "{bad}");
    }
    let (endpoint, rx) = collector();
    let addr = start(&succ_model(), Some(endpoint));
    post(addr, r#"{"model":"m","prompt":"a","max_tokens":1,"temperature":0}"#, "traceparent: 00-00000000000000000000000000000000-0000000000000000-01\r\n");
    let (_, _, body) = rx.recv_timeout(Duration::from_secs(5)).expect("a span is exported within 5 s");
    let span = j::parse(&body).get("resourceSpans").arr()[0].get("scopeSpans").arr()[0].get("spans").arr()[0].clone();
    let trace = span.get("traceId").str().to_string();
    assert!(is_hex(&trace, 32) && trace.bytes().any(|b| b != b'0'), "{trace}");
    assert!(!span.has("parentSpanId") || span.get("parentSpanId").str().is_empty());
}

#[test]
fn serves_a_second_client_while_one_stalls() {
    // WHY: one thread per connection: a client that sends half a request
    //      line and stops must not block the health check behind it.
    // KIND: fault
    // CHAPTER: L10.0 section 2
    let addr = start(&succ_model(), None);
    let mut slow = TcpStream::connect(addr).unwrap();
    slow.write_all(b"POST /v1/comp").unwrap();
    thread::sleep(Duration::from_millis(100));
    let t0 = Instant::now();
    assert_eq!(get(addr, "/healthz").status, 200);
    assert!(t0.elapsed() < Duration::from_secs(2), "the second client waited {:?}", t0.elapsed());
    drop(slow);
}
