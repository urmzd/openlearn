//! L10.5 course tests: tl-serve's v1 server (server.rs, openai.rs, sse.rs,
//! template.rs) over the engine loop (tl-engine engine.rs).
//!
//! Annotated exemplars (DESIGN 5.12). Each test starts your server with
//! `server::spawn` on 127.0.0.1 port 0 and speaks raw HTTP/1.1 to it with
//! `Connection: close`; streamed responses arrive with chunked transfer
//! encoding, which `dechunk` below undoes. JSON is read with `mod j`, never
//! with yours. Models: the tracer bigram written by hand (after byte i
//! comes byte i + 1), and the tiny Llama of course/fixtures/L7.9.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tl_engine::runner::{EngineConfig, KvConfig, ModelRunner, PrefixCache};
use tl_serve::server::{self, pool_embedding, ServeConfig, ServerHandle};
use tl_serve::sse::TextStream;
use tl_serve::template::{Template, DEFAULT_TEMPLATE};

// ---------------------------------------------------------------------------
// helpers: models and servers

static SEQ: AtomicUsize = AtomicUsize::new(0);

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ol-l10_5-{}-{}-{tag}", std::process::id(), SEQ.fetch_add(1, Ordering::SeqCst)));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

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

/// The succ bigram in a directory named `succ`, with a chat template that
/// concatenates the contents: the prompt of [{user: "a"}] is "a".
fn succ_model() -> PathBuf {
    let d = tmpdir("m").join("succ");
    std::fs::create_dir_all(&d).unwrap();
    let mut w = vec![0.0f32; 256 * 256];
    for i in 0..256 {
        w[i * 256 + (i + 1) % 256] = 2.0;
    }
    let data: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    let header = format!(r#"{{"__metadata__":{{"format":"tinyllm"}},"bigram.weight":{{"dtype":"F32","shape":[256,256],"data_offsets":[0,{}]}}}}"#, data.len());
    std::fs::write(d.join("model.safetensors"), safetensors(&header, &data)).unwrap();
    std::fs::write(d.join("config.json"), r#"{"tl_arch":"bigram","tl_tokenizer":"bytes","vocab_size":256,"tl_format":1}"#).unwrap();
    std::fs::write(d.join("generation_config.json"), r#"{"chat_template":"{% for m in messages %}{{ m.content }}{% endfor %}"}"#).unwrap();
    d
}

fn tiny_llama() -> PathBuf {
    PathBuf::from(std::env::var("TINYLLM_FIXTURES").expect("TINYLLM_FIXTURES is set by ol")).join("L7.9/tiny-llama-2l")
}

/// Stops (drains) the server when the test ends, pass or fail.
struct Srv(Option<ServerHandle>);

impl Srv {
    fn http(&self) -> SocketAddr {
        self.0.as_ref().unwrap().http_addr
    }
    fn health(&self) -> SocketAddr {
        self.0.as_ref().unwrap().health_addr
    }
}

impl Drop for Srv {
    fn drop(&mut self) {
        if let Some(h) = self.0.take() {
            h.drain();
            let _ = h.join();
        }
    }
}

fn start(dir: &Path, tweak: impl FnOnce(&mut ServeConfig)) -> Srv {
    let mut c = ServeConfig::for_model(dir);
    c.http_listen = "127.0.0.1:0".to_string();
    c.health_listen = "127.0.0.1:0".to_string();
    c.engine.kv = KvConfig { blocks: 64, block_size: 16 };
    c.engine.prefix_cache = PrefixCache::None;
    tweak(&mut c);
    let h = server::spawn(c).expect("server::spawn");
    let s = Srv(Some(h));
    let t0 = Instant::now();
    while get(s.health(), "/readyz").status != 200 {
        assert!(t0.elapsed() < Duration::from_secs(5), "/readyz never answered 200");
        std::thread::sleep(Duration::from_millis(5));
    }
    s
}

// ---------------------------------------------------------------------------
// helpers: raw HTTP

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
        String::from_utf8(self.body.clone()).expect("UTF-8 body")
    }
    fn json(&self) -> j::V {
        j::parse(&self.text())
    }
    /// The SSE events of the body: each `data: ` payload, in order.
    fn events(&self) -> Vec<String> {
        let t = self.text();
        assert!(t.ends_with("\n\n"), "an SSE body ends with a blank line: {t:?}");
        t.split("\n\n").filter(|e| !e.is_empty()).map(|e| e.strip_prefix("data: ").unwrap_or_else(|| panic!("event without `data: `: {e:?}")).to_string()).collect()
    }
}

/// Undoes chunked transfer encoding.
fn dechunk(mut raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let nl = raw.windows(2).position(|w| w == b"\r\n").expect("chunk size line");
        let size = usize::from_str_radix(std::str::from_utf8(&raw[..nl]).unwrap().trim(), 16).expect("hex chunk size");
        raw = &raw[nl + 2..];
        if size == 0 {
            return out;
        }
        out.extend_from_slice(&raw[..size]);
        raw = &raw[size + 2..];
    }
}

fn parse_resp(raw: &[u8]) -> Resp {
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap_or_else(|| panic!("no header end in {:?}", String::from_utf8_lossy(raw)));
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let mut lines = head.split("\r\n");
    let status: u16 = lines.next().unwrap().split(' ').nth(1).unwrap().parse().unwrap();
    let headers: Vec<(String, String)> = lines.map(|l| l.split_once(':').map(|(k, v)| (k.trim().to_string(), v.trim().to_string())).unwrap()).collect();
    let mut body = raw[split + 4..].to_vec();
    if headers.iter().any(|(k, v)| k.eq_ignore_ascii_case("transfer-encoding") && v.eq_ignore_ascii_case("chunked")) {
        body = dechunk(&body);
    }
    Resp { status, headers, body }
}

fn send_raw(addr: SocketAddr, raw: &str) -> Resp {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    s.write_all(raw.as_bytes()).unwrap();
    let mut out = Vec::new();
    s.read_to_end(&mut out).expect("read until the server closes");
    parse_resp(&out)
}

fn request(addr: SocketAddr, method: &str, path: &str, body: &str, extra: &str) -> Resp {
    send_raw(
        addr,
        &format!("{method} {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{extra}\r\n{body}", body.len()),
    )
}

fn post(addr: SocketAddr, path: &str, body: &str) -> Resp {
    request(addr, "POST", path, body, "")
}

fn get(addr: SocketAddr, path: &str) -> Resp {
    send_raw(addr, &format!("GET {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n"))
}

/// Opens a streamed request and reads until `n` events have arrived; the
/// stream is returned open (drop it to disconnect).
fn open_stream(addr: SocketAddr, path: &str, body: &str, n: usize) -> TcpStream {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    s.write_all(format!("POST {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).unwrap();
    let mut seen = Vec::new();
    let mut buf = [0u8; 4096];
    while seen.windows(6).filter(|w| w == b"data: ").count() < n {
        let k = s.read(&mut buf).expect("stream read");
        assert!(k > 0, "stream closed early: {:?}", String::from_utf8_lossy(&seen));
        seen.extend_from_slice(&buf[..k]);
    }
    s
}

fn metric(addr: SocketAddr, line: &str) -> f64 {
    let t = get(addr, "/metrics").text();
    t.lines().find_map(|l| l.strip_prefix(line).map(|v| v.trim().parse::<f64>().unwrap())).unwrap_or_else(|| panic!("no {line} in /metrics:\n{t}"))
}

fn chat_body(model: &str, content: &str, extra: &str) -> String {
    format!(r#"{{"model":"{model}","messages":[{{"role":"user","content":"{content}"}}]{extra}}}"#)
}

/// A tiny JSON reader, independent of the code under test.
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
                    if !(c == '\\' && b[*i] == 'u') && !units.is_empty() {
                        s.push_str(&String::from_utf16_lossy(&units));
                        units.clear();
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
// tests

#[test]
fn hand_example_chat_completion() {
    // WHY: the chapter's worked example over the wire: the template turns
    //      [{user: "a"}] into the prompt "a" (1 byte token), greedy decoding
    //      of the succ bigram continues "bcd", and the response is a
    //      chat.completion with finish_reason length and usage 1 + 3 = 4.
    // KIND: unit
    // CATCHES: s01
    // CHAPTER: L10.5 section 3
    let s = start(&succ_model(), |_| {});
    let r = post(s.http(), "/v1/chat/completions", &chat_body("succ", "a", r#","max_tokens":3,"temperature":0"#));
    assert_eq!(r.status, 200, "{}", r.text());
    assert!(r.header("content-type").unwrap().starts_with("application/json"));
    let v = r.json();
    assert_eq!(v.get("object").str(), "chat.completion");
    assert!(v.get("id").str().starts_with("chatcmpl-"));
    assert_eq!(v.get("model").str(), "succ");
    let c = &v.get("choices").arr()[0];
    assert_eq!(c.get("index").num(), 0.0);
    assert_eq!(c.get("message").get("role").str(), "assistant");
    assert_eq!(c.get("message").get("content").str(), "bcd");
    assert_eq!(c.get("finish_reason").str(), "length");
    let u = v.get("usage");
    assert_eq!((u.get("prompt_tokens").num(), u.get("completion_tokens").num(), u.get("total_tokens").num()), (1.0, 3.0, 4.0));
}

#[test]
fn stream_framing_role_first_then_done() {
    // WHY: the v1 streaming contract: text/event-stream, every event
    //      `data: <json>\n\n`, one id throughout, the role in the first
    //      delta, finish_reason on the last content chunk, the usage chunk
    //      (choices []) when asked, then `data: [DONE]`.
    // KIND: conformance
    // CATCHES: s01, s03, s04
    // CHAPTER: L10.5 section 2
    let s = start(&succ_model(), |_| {});
    let r = post(s.http(), "/v1/chat/completions", &chat_body("succ", "a", r#","max_tokens":3,"temperature":0,"stream":true,"stream_options":{"include_usage":true}"#));
    assert_eq!(r.status, 200);
    assert_eq!(r.header("content-type"), Some("text/event-stream"));
    let ev = r.events();
    assert_eq!(ev.last().map(String::as_str), Some("[DONE]"));
    let chunks: Vec<j::V> = ev[..ev.len() - 1].iter().map(|e| j::parse(e)).collect();
    let id = chunks[0].get("id").str().to_string();
    assert!(chunks.iter().all(|c| c.get("id").str() == id && c.get("object").str() == "chat.completion.chunk"));
    let first = &chunks[0].get("choices").arr()[0];
    assert_eq!(first.get("delta").get("role").str(), "assistant");
    assert_eq!(first.get("finish_reason"), &j::V::Null);
    let usage = chunks.last().unwrap();
    assert!(usage.get("choices").arr().is_empty());
    assert_eq!(usage.get("usage").get("completion_tokens").num(), 3.0);
    let content: Vec<&j::V> = chunks[1..chunks.len() - 1].iter().map(|c| &c.get("choices").arr()[0]).collect();
    let text: String = content.iter().map(|c| c.get("delta").get("content").str()).collect();
    assert_eq!(text, "bcd");
    let (last, rest) = content.split_last().unwrap();
    assert_eq!(last.get("finish_reason").str(), "length");
    assert!(rest.iter().all(|c| c.get("finish_reason") == &j::V::Null));
}

#[test]
fn stream_equals_nonstream_at_temperature_0() {
    // WHY: conformance case chat.stream.equals_nonstream on a real model:
    //      the concatenated deltas equal the non-streamed content, byte for
    //      byte, including the incremental UTF-8 of byte tokens.
    // KIND: differential
    // CATCHES: s03
    // CHAPTER: L10.5 section 2
    let s = start(&tiny_llama(), |_| {});
    let body = chat_body("tiny-llama-2l", "Once upon a time", r#","max_tokens":24,"temperature":0"#);
    let plain = post(s.http(), "/v1/chat/completions", &body).json();
    let want = plain.get("choices").arr()[0].get("message").get("content").str().to_string();
    let r = post(s.http(), "/v1/chat/completions", &body.replace("\"temperature\":0", "\"temperature\":0,\"stream\":true"));
    let ev = r.events();
    let got: String = ev[..ev.len() - 1].iter().map(|e| j::parse(e).get("choices").arr()[0].get("delta").get("content").str().to_string()).collect();
    assert_eq!(got, want);
}

#[test]
fn seed_reproduces_sampling() {
    // WHY: with temperature > 0 the same seed on the same engine gives the
    //      same output (conformance chat.seed): each request draws from its
    //      own stream(seed, sample), whatever else runs.
    // KIND: property
    // CHAPTER: L10.5 section 2
    let s = start(&tiny_llama(), |_| {});
    let run = |seed: u64| {
        let r = post(s.http(), "/v1/chat/completions", &chat_body("tiny-llama-2l", "Hello", &format!(r#","max_tokens":20,"temperature":1.0,"seed":{seed}"#)));
        r.json().get("choices").arr()[0].get("message").get("content").str().to_string()
    };
    let a = run(7);
    assert_eq!(a, run(7));
    assert_ne!(a, run(8), "another seed should give another sample");
}

#[test]
fn stop_strings_end_generation_across_tokens() {
    // WHY: a stop string ends generation, is never returned, and can span
    //      tokens: with one byte per token, "de" arrives as two tokens and
    //      the held-back "d" is dropped; finish_reason is stop in both modes.
    // KIND: unit
    // CATCHES: s03, s04, s05, s06
    // CHAPTER: L10.5 section 5, Pitfalls
    let s = start(&succ_model(), |_| {});
    let body = chat_body("succ", "a", r#","max_tokens":10,"temperature":0,"stop":["xyz","de"]"#);
    let v = post(s.http(), "/v1/chat/completions", &body).json();
    let c = &v.get("choices").arr()[0];
    assert_eq!(c.get("message").get("content").str(), "bc");
    assert_eq!(c.get("finish_reason").str(), "stop");
    let ev = post(s.http(), "/v1/chat/completions", &body.replace("\"temperature\":0", "\"temperature\":0,\"stream\":true")).events();
    let chunks: Vec<j::V> = ev[1..ev.len() - 1].iter().map(|e| j::parse(e)).collect();
    let text: String = chunks.iter().map(|c| c.get("choices").arr()[0].get("delta").get("content").str().to_string()).collect();
    assert_eq!(text, "bc");
    assert_eq!(chunks.last().unwrap().get("choices").arr()[0].get("finish_reason").str(), "stop");
}

#[test]
fn errors_use_the_openai_shape() {
    // WHY: every error has the v1 shape {error: {message, type, param,
    //      code}} with application/json: out-of-range values are 400 naming
    //      the field, unsupported values 422 unsupported_parameter, an
    //      unknown model 404 model_not_found, a prompt past the context 400
    //      context_length_exceeded.
    // KIND: conformance
    // CATCHES: s07, s08
    // CHAPTER: L10.5 section 2
    let s = start(&succ_model(), |_| {});
    let chat = |extra: &str| post(s.http(), "/v1/chat/completions", &chat_body("succ", "a", extra));
    let cases: Vec<(Resp, u16, Option<&str>, Option<&str>)> = vec![
        (chat(r#","temperature":-1"#), 400, Some("temperature"), None),
        (chat(r#","top_p":0"#), 400, Some("top_p"), None),
        (chat(r#","n":2"#), 422, Some("n"), Some("unsupported_parameter")),
        // a response_format this engine never serves: 422 before L10.9 and after it
        (chat(r#","response_format":{"type":"json_schema"}"#), 422, Some("response_format"), Some("unsupported_parameter")),
        (chat(r#","max_tokens":0"#), 400, Some("max_tokens"), None),
        (chat(r#","max_tokens":100000"#), 400, None, Some("context_length_exceeded")),
        (post(s.http(), "/v1/chat/completions", &chat_body("nope", "a", "")), 404, Some("model"), Some("model_not_found")),
        (post(s.http(), "/v1/chat/completions", r#"{"model":"succ","messages":[{"role":"user","content":[{"type":"text","text":"a"}]}]}"#), 422, None, Some("unsupported_parameter")),
        (post(s.http(), "/v1/chat/completions", "{not json"), 400, None, None),
        (post(s.http(), "/v1/completions", r#"{"model":"succ","prompt":"a","echo":true}"#), 422, Some("echo"), Some("unsupported_parameter")),
        (request(s.http(), "POST", "/v1/chat/completions", &chat_body("succ", "a", ""), "X-TL-Priority: 500\r\n"), 400, None, None),
        (get(s.http(), "/v1/chat/completions"), 405, None, None),
        (get(s.http(), "/v2/anything"), 404, None, None),
    ];
    for (i, (r, status, param, code)) in cases.into_iter().enumerate() {
        assert_eq!(r.status, status, "case {i}: {}", r.text());
        assert!(r.header("content-type").unwrap().starts_with("application/json"), "case {i}");
        let e = r.json();
        let e = e.get("error");
        assert!(!e.get("message").str().is_empty());
        assert!(e.has("type") && e.has("param") && e.has("code"), "case {i}: {e:?}");
        if let Some(p) = param {
            assert_eq!(e.get("param").str(), p, "case {i}");
        }
        if let Some(c) = code {
            assert_eq!(e.get("code").str(), c, "case {i}");
        }
    }
}

#[test]
fn completions_endpoint_and_usage() {
    // WHY: /v1/completions serves the same engine without a template: the
    //      prompt "ab" is 2 byte tokens, the default max_tokens is 16, and
    //      the text_completion object carries usage.
    // KIND: conformance
    // CATCHES: s01
    // CHAPTER: L10.5 section 2
    let s = start(&succ_model(), |_| {});
    let v = post(s.http(), "/v1/completions", r#"{"model":"succ","prompt":"ab","temperature":0}"#).json();
    assert_eq!(v.get("object").str(), "text_completion");
    let c = &v.get("choices").arr()[0];
    assert_eq!(c.get("text").str(), "cdefghijklmnopqr");
    assert_eq!(c.get("finish_reason").str(), "length");
    assert_eq!(v.get("usage").get("prompt_tokens").num(), 2.0);
    assert_eq!(v.get("usage").get("completion_tokens").num(), 16.0);
}

#[test]
fn usage_counts_the_templated_prompt() {
    // WHY: prompt_tokens counts the prompt AFTER the chat template (roles,
    //      separators, generation prompt), which is what the model reads and
    //      what the gateway bills (conformance chat.usage); the tiny model
    //      has no template of its own, so the default one applies.
    // KIND: conformance
    // CATCHES: s02
    // CHAPTER: L10.5 section 2
    let s = start(&tiny_llama(), |_| {});
    let v = post(s.http(), "/v1/chat/completions", &chat_body("tiny-llama-2l", "hi", r#","max_tokens":2,"temperature":0"#)).json();
    let want = "<|user|>\nhi\n<|assistant|>\n".len() as f64;
    assert_eq!(v.get("usage").get("prompt_tokens").num(), want);
}

#[test]
fn full_queue_answers_429_with_retry_after() {
    // WHY: admission is bounded: with one running sequence and a waiting
    //      queue of one, a third request is refused at once with 429
    //      rate_limit_exceeded and Retry-After, instead of waiting without
    //      limit; the queued one is served once the first leaves.
    // KIND: fault
    // CATCHES: s09, s10
    // CHAPTER: L10.5 section 2
    let s = start(&tiny_llama(), |c| {
        c.engine.max_seqs = 1;
        c.queue_capacity = 1;
        c.step_delay = Duration::from_millis(10);
    });
    let a = open_stream(s.http(), "/v1/chat/completions", &chat_body("tiny-llama-2l", "first", r#","max_tokens":200,"temperature":0,"stream":true"#), 2);
    let addr = s.http();
    let b = std::thread::spawn(move || post(addr, "/v1/chat/completions", &chat_body("tiny-llama-2l", "second", r#","max_tokens":2,"temperature":0"#)));
    std::thread::sleep(Duration::from_millis(150));
    let c = post(s.http(), "/v1/chat/completions", &chat_body("tiny-llama-2l", "third", r#","max_tokens":2"#));
    assert_eq!(c.status, 429, "{}", c.text());
    assert_eq!(c.header("retry-after"), Some("1"));
    assert_eq!(c.json().get("error").get("code").str(), "rate_limit_exceeded");
    drop(a);
    let b = b.join().unwrap();
    assert_eq!(b.status, 200, "the queued request runs after the first leaves: {}", b.text());
}

#[test]
fn disconnect_frees_blocks_within_a_step() {
    // WHY: a client that leaves mid-stream must not keep its KV blocks: the
    //      engine sees the closed channel before its next step and aborts, so
    //      /metrics shows 0 active sequences and 0 used blocks within 2 s
    //      (conformance cancel.disconnect).
    // KIND: fault
    // CHAPTER: L10.5 section 2
    let s = start(&tiny_llama(), |c| c.step_delay = Duration::from_millis(5));
    let stream = open_stream(s.http(), "/v1/chat/completions", &chat_body("tiny-llama-2l", "Hello", r#","max_tokens":200,"temperature":0,"stream":true"#), 3);
    assert_eq!(metric(s.health(), "tl_engine_active_sequences "), 1.0);
    assert!(metric(s.health(), "tl_engine_kv_blocks{state=\"used\"} ") > 0.0);
    drop(stream);
    let t0 = Instant::now();
    loop {
        let active = metric(s.health(), "tl_engine_active_sequences ");
        let used = metric(s.health(), "tl_engine_kv_blocks{state=\"used\"} ");
        if active == 0.0 && used == 0.0 {
            break;
        }
        assert!(t0.elapsed() < Duration::from_secs(2), "after a disconnect: {active} active, {used} blocks used");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn drain_finishes_in_flight_and_refuses_new() {
    // WHY: SIGTERM (here `drain`, what the handler calls) turns /readyz 503
    //      so no new traffic is routed, refuses new connections, lets the
    //      in-flight stream finish to [DONE], then stops; the binary exits 0.
    // KIND: fault
    // CATCHES: s11
    // CHAPTER: L10.5 section 2
    let mut s = start(&tiny_llama(), |c| c.step_delay = Duration::from_millis(5));
    let mut stream = open_stream(s.http(), "/v1/chat/completions", &chat_body("tiny-llama-2l", "Hello", r#","max_tokens":40,"temperature":0,"stream":true"#), 2);
    let h = s.0.take().unwrap();
    let (http, health) = (h.http_addr, h.health_addr);
    h.drain();
    let t0 = Instant::now();
    while get(health, "/readyz").status != 503 {
        assert!(t0.elapsed() < Duration::from_secs(2), "/readyz must turn 503 while draining");
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(TcpStream::connect(http).is_err() || post(http, "/v1/chat/completions", &chat_body("tiny-llama-2l", "x", "")).status == 503);
    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).unwrap();
    let rest = String::from_utf8_lossy(&rest);
    assert!(rest.contains("data: [DONE]"), "the in-flight stream must finish: {rest}");
    h.join().expect("the server stops cleanly after draining");
}

#[test]
fn embeddings_are_mean_pooled_and_normalized() {
    // WHY: /v1/embeddings returns, per input in order, the mean of the
    //      final-normed hidden states over the input's tokens divided by its
    //      L2 norm (the contract's definition; the gateway's policy
    //      classifier, D33, is a linear head over it).
    // KIND: unit
    // CATCHES: s12
    // CHAPTER: L10.5 section 2
    let s = start(&tiny_llama(), |_| {});
    let v = post(s.http(), "/v1/embeddings", r#"{"model":"tiny-llama-2l","input":["abc","hello"]}"#).json();
    assert_eq!(v.get("object").str(), "list");
    assert_eq!(v.get("usage").get("prompt_tokens").num(), 8.0);
    let mut r = ModelRunner::load(&tiny_llama(), &EngineConfig { kv: KvConfig { blocks: 8, block_size: 16 }, ..EngineConfig::default() }).unwrap();
    for (k, input) in ["abc", "hello"].iter().enumerate() {
        let e = &v.get("data").arr()[k];
        assert_eq!(e.get("index").num(), k as f64);
        let got: Vec<f64> = e.get("embedding").arr().iter().map(|x| x.num()).collect();
        assert_eq!(got.len(), 32);
        let norm: f64 = got.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm {norm}");
        let toks: Vec<u32> = input.bytes().map(u32::from).collect();
        let (hidden, _) = r.run_once(&toks).unwrap();
        // the definition, computed here independently
        let mut mean = vec![0.0f64; 32];
        for row in hidden.chunks(32) {
            for (m, x) in mean.iter_mut().zip(row) {
                *m += *x as f64 / toks.len() as f64;
            }
        }
        let n: f64 = mean.iter().map(|x| x * x).sum::<f64>().sqrt();
        for (g, m) in got.iter().zip(&mean) {
            assert!((g - m / n).abs() < 1e-5, "{input}: {g} vs {}", m / n);
        }
        assert_eq!(pool_embedding(&hidden, 32).len(), 32);
    }
}

#[test]
fn tokenize_models_health_and_request_id() {
    // WHY: the engine-tier extras: /v1/tokenize gives the tokenizer's ids
    //      (the gateway's TPM cost, gw.03), /v1/models lists the served
    //      model, /healthz answers on both ports, /readyz on the health
    //      port, and X-Request-Id is echoed (or generated) on every response.
    // KIND: conformance
    // CATCHES: s14
    // CHAPTER: L10.5 section 2
    let s = start(&succ_model(), |_| {});
    let t = post(s.http(), "/v1/tokenize", r#"{"model":"succ","text":"hé"}"#).json();
    let ids: Vec<f64> = t.get("ids").arr().iter().map(|x| x.num()).collect();
    assert_eq!(ids, [104.0, 195.0, 169.0]);
    let m = get(s.http(), "/v1/models").json();
    assert_eq!(m.get("data").arr()[0].get("id").str(), "succ");
    assert_eq!(m.get("data").arr()[0].get("object").str(), "model");
    assert_eq!(get(s.http(), "/v1/models/succ").status, 200);
    assert_eq!(get(s.http(), "/v1/models/other").status, 404);
    assert_eq!(get(s.http(), "/healthz").status, 200);
    assert_eq!(get(s.health(), "/healthz").status, 200);
    let r = request(s.http(), "GET", "/v1/models", "", "X-Request-Id: req-42\r\n");
    assert_eq!(r.header("x-request-id"), Some("req-42"));
    assert!(get(s.http(), "/v1/models").header("x-request-id").is_some_and(|v| !v.is_empty()));
}

#[test]
fn concurrent_streams_equal_serial() {
    // WHY: eight streams at once share the continuous batch and each still
    //      gets exactly the tokens it gets alone at temperature 0
    //      (conformance concurrency.16, at a size a test can afford).
    // KIND: differential
    // CATCHES: s03, s04
    // CHAPTER: L10.5 section 2
    let s = start(&tiny_llama(), |c| c.engine.prefill_chunk = 8);
    let prompts: Vec<String> = (0..8).map(|i| format!("Story {i}: once")).collect();
    let serial: Vec<String> = prompts
        .iter()
        .map(|p| post(s.http(), "/v1/chat/completions", &chat_body("tiny-llama-2l", p, r#","max_tokens":16,"temperature":0"#)).json().get("choices").arr()[0].get("message").get("content").str().to_string())
        .collect();
    let addr = s.http();
    let threads: Vec<_> = prompts
        .iter()
        .map(|p| {
            let body = chat_body("tiny-llama-2l", p, r#","max_tokens":16,"temperature":0,"stream":true"#);
            std::thread::spawn(move || {
                let ev = post(addr, "/v1/chat/completions", &body).events();
                ev[1..ev.len() - 1].iter().map(|e| j::parse(e).get("choices").arr()[0].get("delta").get("content").str().to_string()).collect::<String>()
            })
        })
        .collect();
    let parallel: Vec<String> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(parallel, serial);
}

#[test]
fn chat_template_hand_example() {
    // WHY: the Jinja subset renders SmolLM2's real template exactly: a
    //      default system turn when the first message is not a system one
    //      (loop.first, messages[0]['role'], !=), string concatenation, and
    //      the generation prompt; plus trim, tojson, loop.last, elif, and
    //      whitespace control.
    // KIND: unit
    // CATCHES: s13
    // CHAPTER: L10.5 section 3
    let smol = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";
    let t = Template::parse(smol).unwrap();
    assert_eq!(
        t.render_messages(&[("user", "Hi")], true, "", "").unwrap(),
        "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    );
    assert_eq!(
        t.render_messages(&[("system", "Be brief."), ("user", "Hi")], false, "", "").unwrap(),
        "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
    );
    let t2 = Template::parse("{%- for m in messages -%}\n  {%- if m.role == 'user' %}U:{{ m.content | trim }}{% elif m.role == 'tool' %}T:{{ m | tojson }}{% else %}A{% endif %}{% if not loop.last %},{% endif %}\n{%- endfor %}").unwrap();
    assert_eq!(t2.render_messages(&[("user", "  x  "), ("tool", "1"), ("assistant", "z")], false, "", "").unwrap(), r#"U:x,T:{"content":"1","role":"tool"},A"#);
    let d = Template::parse(DEFAULT_TEMPLATE).unwrap();
    assert_eq!(d.render_messages(&[("user", "hi")], true, "", "").unwrap(), "<|user|>\nhi\n<|assistant|>\n");
}

#[test]
fn template_errors_are_caught_at_parse() {
    // WHY: a template the engine cannot render must fail when the model
    //      loads, with a message, not on the first chat request.
    // KIND: boundary
    // CHAPTER: L10.5 section 5, Pitfalls
    for bad in ["{% for m in messages %}x", "{{ m.content ", "{% if x %}a{% else %}b", "{% macro f() %}{% endmacro %}", "{{ x | upper }}", "{% endif %}"] {
        assert!(Template::parse(bad).is_err(), "should refuse {bad:?}");
    }
}

#[test]
fn text_stream_holds_back_stop_prefixes() {
    // WHY: streamed text must never show part of a stop string that later
    //      completes, and must hold an incomplete UTF-8 sequence: `<` and
    //      `</` wait until the next byte decides, `é` arrives whole.
    // KIND: unit
    // CATCHES: s05, s06
    // CHAPTER: L10.5 section 3
    let mut t = TextStream::new(vec!["</s>".to_string()]);
    assert_eq!(t.push(b"ab<"), "ab");
    assert_eq!(t.push(b"/"), "");
    assert_eq!(t.push(b"x"), "</x");
    assert_eq!(t.push(&[0xC3]), "");
    assert_eq!(t.push(&[0xA9]), "é");
    assert_eq!(t.push(b"</s"), "");
    assert_eq!(t.push(b">tail"), "");
    assert!(t.stopped());
    let mut u = TextStream::new(vec![]);
    assert_eq!(u.push(&[0xE2, 0x82]), "");
    assert_eq!(u.finish(), "\u{FFFD}");
}

#[test]
fn runtime_toml_and_env_overrides() {
    // WHY: `--config` reads [engine] of runtime.toml with the schema's
    //      defaults, TL_ENGINE__<KEY> overrides a scalar key, and keys or
    //      values this engine cannot honor are refused at start.
    // KIND: unit
    // CATCHES: s14
    // CHAPTER: L10.5 section 4
    let toml = "[engine]\nmodel_dir = \"/m/smol-135m/v3\"\nhttp_listen = \":8001\"\nhealth_listen = \"127.0.0.1:9465\"\nkv_blocks = 64\nprefix_cache = \"hash\"\nprefill_chunk = 128\n";
    let c = ServeConfig::from_toml(toml, &[]).unwrap();
    assert_eq!(c.model_id, "smol-135m");
    assert_eq!(c.engine.kv, KvConfig { blocks: 64, block_size: 16 });
    assert_eq!(c.engine.prefix_cache, PrefixCache::Hash);
    assert_eq!(c.engine.prefill_chunk, 128);
    assert_eq!(c.queue_capacity, 256);
    assert_eq!(server::listen_addr(&c.http_listen).unwrap().port(), 8001);
    let env = vec![("TL_ENGINE__KV_BLOCKS".to_string(), "128".to_string()), ("TL_ENGINE__PREFIX_CACHE".to_string(), "none".to_string()), ("OTHER".to_string(), "1".to_string())];
    let c = ServeConfig::from_toml(toml, &env).unwrap();
    assert_eq!(c.engine.kv.blocks, 128);
    assert_eq!(c.engine.prefix_cache, PrefixCache::None);
    let base = "[engine]\nmodel_dir = \"/m/x\"\n";
    assert!(ServeConfig::from_toml(base, &[]).is_ok());
    for bad in ["kv_blockz = 1", "role = \"prefill\"", "quant = \"int8\"", "kv_blocks = 0", "prefix_cache = \"lru\"", "http_listen = \"nowhere\""] {
        assert!(ServeConfig::from_toml(&format!("{base}{bad}\n"), &[]).is_err(), "should refuse {bad}");
    }
    assert!(ServeConfig::from_toml("[gateway]\nlisten = \":1\"\n", &[]).is_err(), "no [engine] table");
}

#[test]
fn speculative_runtime_config_drives_the_serving_path() {
    // WHY: the optional runtime setting is parsed once and routes generation
    // through L10.8's runner-backed verifier without changing greedy output.
    // KIND: conformance
    // CATCHES: s01
    // CHAPTER: L10.5 section 4
    let plain = start(&succ_model(), |_| {});
    let body = chat_body("succ", "aba", r#", "max_tokens":6,"temperature":0"#);
    let expected = post(plain.http(), "/v1/chat/completions", &body).json();
    drop(plain);

    let spec = start(&succ_model(), |c| {
        c.speculative = Some(tl_engine::spec::SpecConfig::parse("prompt_lookup", 4).unwrap());
    });
    let got = post(spec.http(), "/v1/chat/completions", &body).json();
    let choice = &got.get("choices").arr()[0];
    assert_eq!(choice.get("message").get("content"), expected.get("choices").arr()[0].get("message").get("content"));
    assert_eq!(choice.get("finish_reason").str(), "length");
}
