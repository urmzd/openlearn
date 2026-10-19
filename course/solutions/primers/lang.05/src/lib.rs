//! wire: HTTP/1.1, JSON, and Server-Sent Events from the bytes up (lang.05).
//!
//! Nothing in this library opens a socket. `read_request` reads from any
//! `BufRead` and the writers write to any `Write`, so a test can feed them a
//! byte slice and a `Vec<u8>`; the binary feeds them a `TcpStream`.

use std::io::{self, BufRead, Read, Write};

/// The most bytes the request line plus all headers may take.
pub const MAX_HEAD: usize = 16 * 1024;
/// The largest request body accepted.
pub const MAX_BODY: usize = 1024 * 1024;

/// One parsed HTTP/1.x request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// `GET`, `POST`, ...
    pub method: String,
    /// The request target as sent: path plus optional `?query`.
    pub target: String,
    /// Header names keep the case they were sent in; look them up with
    /// `header()`, which ignores case.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

/// Why a request could not be read. `status()` is the HTTP answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpError {
    /// The client closed the connection before sending a request line.
    Closed,
    /// Syntax error in the request line or a header (400).
    BadRequest(String),
    /// A POST without Content-Length (411).
    LengthRequired,
    /// Head over MAX_HEAD or body over MAX_BODY (413).
    TooLarge,
    /// The socket failed or the body ended early.
    Io(String),
}

impl HttpError {
    /// The status code to answer with; 0 when there is no one to answer.
    pub fn status(&self) -> u16 {
        match self {
            HttpError::Closed | HttpError::Io(_) => 0,
            HttpError::BadRequest(_) => 400,
            HttpError::LengthRequired => 411,
            HttpError::TooLarge => 413,
        }
    }
}

impl Request {
    /// The value of the first header named `name`, ignoring ASCII case.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }

    /// The target without its query string: `/count?n=3` gives `/count`.
    pub fn path(&self) -> &str {
        self.target.split_once('?').map_or(self.target.as_str(), |(p, _)| p)
    }

    /// The value of `key` in the query string (`a=1&b=2`), not decoded.
    pub fn query(&self, key: &str) -> Option<&str> {
        let (_, q) = self.target.split_once('?')?;
        q.split('&').filter_map(|kv| kv.split_once('=')).find(|(k, _)| *k == key).map(|(_, v)| v)
    }
}

/// Reads one line ending in `\n` from `r`, at most `limit` bytes, and
/// returns it without the `\r\n` or `\n`. `Ok(None)` at a clean EOF.
fn read_line<R: BufRead>(r: &mut R, limit: usize) -> Result<Option<Vec<u8>>, HttpError> {
    let mut line = Vec::new();
    let n = r
        .by_ref()
        .take(limit as u64 + 1)
        .read_until(b'\n', &mut line)
        .map_err(|e| HttpError::Io(e.to_string()))?;
    if n == 0 {
        return Ok(None);
    }
    if line.last() != Some(&b'\n') {
        return Err(if n > limit { HttpError::TooLarge } else { HttpError::Io("connection closed mid-line".into()) });
    }
    line.pop();
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    Ok(Some(line))
}

/// Reads one request: the request line, the headers up to the empty line,
/// then exactly Content-Length body bytes.
pub fn read_request<R: BufRead>(r: &mut R) -> Result<Request, HttpError> {
    let first = read_line(r, MAX_HEAD)?.ok_or(HttpError::Closed)?;
    let first = String::from_utf8(first).map_err(|_| HttpError::BadRequest("request line is not UTF-8".into()))?;
    let parts: Vec<&str> = first.split(' ').collect();
    let [method, target, version] = parts[..] else {
        return Err(HttpError::BadRequest(format!("request line {first:?} is not `METHOD target HTTP/1.1`")));
    };
    if method.is_empty() || !method.bytes().all(|b| b.is_ascii_uppercase()) {
        return Err(HttpError::BadRequest(format!("bad method {method:?}")));
    }
    if !target.starts_with('/') {
        return Err(HttpError::BadRequest(format!("target {target:?} must start with /")));
    }
    if version != "HTTP/1.1" && version != "HTTP/1.0" {
        return Err(HttpError::BadRequest(format!("unsupported version {version:?}")));
    }

    let mut used = first.len();
    let mut headers = Vec::new();
    loop {
        let line = read_line(r, MAX_HEAD)?.ok_or_else(|| HttpError::Io("connection closed in the headers".into()))?;
        used += line.len() + 2;
        if used > MAX_HEAD {
            return Err(HttpError::TooLarge);
        }
        if line.is_empty() {
            break;
        }
        let line = String::from_utf8(line).map_err(|_| HttpError::BadRequest("header is not UTF-8".into()))?;
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| HttpError::BadRequest(format!("header {line:?} has no colon")))?;
        if name.is_empty() || name.bytes().any(|b| b <= b' ' || b == 0x7f) {
            return Err(HttpError::BadRequest(format!("bad header name {name:?}")));
        }
        headers.push((name.to_string(), value.trim_matches(|c| c == ' ' || c == '\t').to_string()));
    }

    let mut req = Request { method: method.to_string(), target: target.to_string(), headers, body: Vec::new() };
    if req.header("transfer-encoding").is_some() {
        return Err(HttpError::BadRequest("chunked request bodies are not supported".into()));
    }
    let len = match req.header("content-length") {
        Some(v) => {
            if v.is_empty() || !v.bytes().all(|b| b.is_ascii_digit()) {
                return Err(HttpError::BadRequest(format!("bad Content-Length {v:?}")));
            }
            v.parse::<usize>().map_err(|_| HttpError::TooLarge)?
        }
        None if req.method == "POST" || req.method == "PUT" => return Err(HttpError::LengthRequired),
        None => 0,
    };
    if len > MAX_BODY {
        return Err(HttpError::TooLarge);
    }
    req.body = vec![0; len];
    r.read_exact(&mut req.body).map_err(|e| HttpError::Io(format!("body: {e}")))?;
    Ok(req)
}

/// The reason phrase for the status codes this server sends.
pub fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        413 => "Content Too Large",
        _ => "Unknown",
    }
}

/// Writes a complete response: status line, headers (Content-Type,
/// Content-Length, Connection: close, then `extra`), the empty line, body.
pub fn write_response<W: Write>(
    w: &mut W,
    status: u16,
    content_type: &str,
    extra: &[(&str, &str)],
    body: &[u8],
) -> io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n",
        reason(status),
        body.len()
    );
    for (k, v) in extra {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    w.write_all(head.as_bytes())?;
    w.write_all(body)?;
    w.flush()
}

/// Writes the head of an SSE response. There is no Content-Length: the
/// stream ends when the server closes the connection.
pub fn write_sse_head<W: Write>(w: &mut W) -> io::Result<()> {
    w.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
    )?;
    w.flush()
}

/// One SSE event carrying `data`. Each line of `data` becomes its own
/// `data: ` line; a blank line ends the event.
pub fn sse_event(data: &str) -> String {
    let mut out = String::new();
    for line in data.split('\n') {
        out.push_str("data: ");
        out.push_str(line);
        out.push('\n');
    }
    out.push('\n');
    out
}

/// A JSON value. Objects keep their keys in the order they were written.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
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

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// Quotes and escapes `s` as a JSON string: `"`, `\`, and every byte below
/// 0x20 are escaped; everything else, non-ASCII included, is written as is.
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

/// Parses a whole JSON text (one value, surrounding whitespace allowed).
pub fn parse_json(text: &str) -> Result<Json, String> {
    let mut p = Parser { b: text.as_bytes(), i: 0 };
    let v = p.value(0)?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("trailing characters at byte {}", p.i));
    }
    Ok(v)
}

/// A recursive-descent parser over the bytes of the text.
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

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        let digits = |p: &mut Self| {
            let s = p.i;
            while p.i < p.b.len() && p.b[p.i].is_ascii_digit() {
                p.i += 1;
            }
            p.i - s
        };
        if self.b.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        let int_start = self.i;
        if digits(self) == 0 || (self.b[int_start] == b'0' && self.i - int_start > 1) {
            return Err(format!("bad number at byte {start}"));
        }
        if self.b.get(self.i) == Some(&b'.') {
            self.i += 1;
            if digits(self) == 0 {
                return Err(format!("bad fraction at byte {start}"));
            }
        }
        if matches!(self.b.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if digits(self) == 0 {
                return Err(format!("bad exponent at byte {start}"));
            }
        }
        let text = std::str::from_utf8(&self.b[start..self.i]).map_err(|e| e.to_string())?;
        text.parse::<f64>().map(Json::Num).map_err(|e| format!("{text}: {e}"))
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
                                // A high surrogate must be followed by \uDC00..\uDFFF.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_worked_example_request() {
        let raw = b"POST /echo HTTP/1.1\r\nHost: x\r\ncontent-length: 15\r\n\r\n{\"text\":\"hi\"}\r\n";
        let req = read_request(&mut &raw[..]).unwrap();
        assert_eq!((req.method.as_str(), req.path()), ("POST", "/echo"));
        assert_eq!(req.header("Content-Length"), Some("15"));
        assert_eq!(req.body, b"{\"text\":\"hi\"}\r\n");
    }

    #[test]
    fn sse_event_frames_each_line() {
        assert_eq!(sse_event("{\"i\":0}"), "data: {\"i\":0}\n\n");
        assert_eq!(sse_event("a\nb"), "data: a\ndata: b\n\n");
    }

    #[test]
    fn json_round_trip() {
        let v = parse_json(r#" {"text": "h\u00e9 \"q\"", "n": [1, -2.5e1, null, true]} "#).unwrap();
        assert_eq!(v.get("text").and_then(Json::as_str), Some("h\u{e9} \"q\""));
        assert_eq!(json_string("h\u{e9} \"q\"\n"), "\"h\u{e9} \\\"q\\\"\\n\"");
    }
}
