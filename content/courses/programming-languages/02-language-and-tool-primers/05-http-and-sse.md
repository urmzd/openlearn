# HTTP/1.1, JSON, and Server-Sent Events from the wire up

The echo server of the [previous section](04-rust.md) speaks an invented protocol, so only a client written for it can talk to it. A real inference engine must be reachable by `curl`, by the official OpenAI client libraries, and by a Go gateway, and all of them speak the same thing: **HTTP/1.1** requests with **JSON** bodies, answered either with one JSON document or with a stream of **Server-Sent Events**, one event per generated token. The tracer engine in *Build Your Own LLM System* is std-only Rust (no TOML parser, no HTTP library), so it implements that protocol itself. When a client hangs or a stream arrives all at once, the bytes on the wire are what explain why.

This section builds a small server that does each piece once, on routes simple enough to check by eye. The solution is the `wire` package ([`Cargo.toml`](../../../../course/solutions/primers/lang.05/Cargo.toml)): the library [`src/lib.rs`](../../../../course/solutions/primers/lang.05/src/lib.rs) (`read_request`, `write_response`, `write_sse_head`, `sse_event`, `parse_json`, `json_string`, `Request`, `HttpError`, `Json`; no sockets) and the server [`src/main.rs`](../../../../course/solutions/primers/lang.05/src/main.rs) (`handle`, `route`, `echo`, `count`) with three routes: `GET /health`, `POST /echo`, and `GET /count`, an SSE stream. The routes table and the signatures below are the contract; the standards behind them are RFC 9112 (HTTP/1.1), RFC 8259 (JSON), and the WHATWG "Server-sent events" section.

You should be comfortable with the Rust from [Rust: ownership, traits, Result, cargo workspaces, std TCP](04-rust.md): ownership, `Result`, traits, `std::net`, and a thread per connection. To run the solution:

```bash
cd primers/lang.05 && cargo test && cargo run -- --port 8080
curl -N 'http://127.0.0.1:8080/count?n=5&delay_ms=500'   # watch the events arrive one by one
```

> **Learning objectives**
> - Read and write HTTP/1.1 requests and responses byte by byte, including header case rules and `Content-Length` framing.
> - Parse untrusted requests with bounded reads and refuse anything that cannot be framed.
> - Parse and emit JSON, including escapes and UTF-16 surrogate pairs, with a recursive-descent parser.
> - Frame a Server-Sent Events stream and make each event leave the server as it is produced.

## An HTTP/1.1 request, byte by byte

HTTP is a **request and response** protocol over a TCP connection: the client sends a request, the server sends one response. In HTTP/1.1 both are text framed by lines ending in **CRLF** (the two bytes `\r\n`, carriage return and line feed).

```text
POST /echo HTTP/1.1\r\n                 request line: METHOD SP target SP version
Host: 127.0.0.1:8080\r\n                headers: name ":" optional spaces value
Content-Type: application/json\r\n
Content-Length: 14\r\n
\r\n                                    an empty line ends the headers
{"text":"hé"}                           the body: exactly 14 bytes
```

Each part has a precise meaning. The **method** says what the client wants: `GET` reads a resource, `POST` sends data to it; methods are case-sensitive upper-case tokens. The **target** is the path, optionally followed by `?` and a **query string** of `key=value` pairs joined by `&`: `/count?n=3&delay_ms=0`. The **version** is `HTTP/1.1` (this server also accepts `HTTP/1.0`). A **header** is a name, a colon, and a value; spaces and tabs around the value are not part of it. Header names are **case-insensitive**: `Content-Length`, `content-length`, and `CONTENT-LENGTH` are the same header, and real clients send each spelling.

A request with a body says how long it is with `Content-Length`, a decimal count of **bytes**, not characters. The server reads exactly that many bytes after the empty line. Without it, a `POST` cannot be read. (The other way, `Transfer-Encoding: chunked`, is covered in Further reading; this server answers it with 400.)

A careful reader treats the request as untrusted input: it caps how many bytes the request line and headers may take (16 KiB here) and how large a body may be (1 MiB), so a client cannot make the server allocate without bound.

## The response and its status

The response mirrors the request, with a status line in place of the request line:

```text
HTTP/1.1 200 OK\r\n                     status line: version SP code SP reason
Content-Type: application/json\r\n
Content-Length: 11\r\n
Connection: close\r\n
\r\n
{"ok":true}
```

The **status code** is a three-digit number; its first digit is its class. This server uses six:

| Code | Meaning | This server sends it when |
|---|---|---|
| 200 OK | success | the route handled the request |
| 400 Bad Request | the request is malformed | bad request line, bad header, bad JSON body |
| 404 Not Found | no such resource | an unknown path |
| 405 Method Not Allowed | the resource exists, not for this method | `GET /echo`; the response names the allowed method in an `Allow` header |
| 411 Length Required | a body without `Content-Length` | `POST` with no length |
| 413 Content Too Large | over the size caps | a body over 1 MiB |

`Content-Type` names the body's **media type**: `application/json` for JSON, `text/event-stream` for SSE. `Connection: close` says the server closes the TCP connection after this response, so every connection carries exactly one request. That keeps the server simple and is what the tracer engine does too; reusing connections (**keep-alive**) is covered in Further reading.

## Where a body ends

A reader of a byte stream must know where each message ends. HTTP/1.1 gives three ways for a response body:

| Framing | How the client knows the body ended | Used for |
|---|---|---|
| `Content-Length: N` | after N bytes | every complete JSON response here |
| `Transfer-Encoding: chunked` | a zero-length chunk | streams on connections that stay open |
| neither, with `Connection: close` | the server closes the connection | the SSE stream here and in the tracer engine |

A wrong `Content-Length` is a silent bug: too small and the client stops early and its JSON parser fails on a cut-off document; too large and the client waits for bytes that never come. It must count **bytes** of the encoded body, so `"é"` (two bytes in UTF-8) counts as 2.

## JSON

The bodies themselves are JSON. **JSON** is a text format for values. Its grammar (RFC 8259) has six kinds of value:

| Kind | Written | Example |
|---|---|---|
| null, booleans | `null`, `true`, `false` | `true` |
| number | optional `-`, integer part with no leading zero (`0` alone is fine), optional `.` fraction, optional `e` or `E` exponent | `-2.5e1` is -25 |
| string | `"` ... `"` with escapes | `"hé \"q\""` |
| array | `[` values separated by `,` `]` | `[1, null]` |
| object | `{` `"key": value` pairs separated by `,` `}` | `{"text": "hi"}` |

Whitespace (space, tab, `\n`, `\r`) may surround any token. A JSON text is encoded as UTF-8.

Inside a string, `"` and `\` must be escaped (`\"`, `\\`), and so must every character below U+0020 (the control characters): `\n`, `\r`, `\t`, `\b`, `\f`, or the general form `\uXXXX` with four hex digits. A writer may emit every other character as raw UTF-8. A reader must also accept `\/` and `\uXXXX` for any character.

`\uXXXX` names a **UTF-16 code unit**, a 16-bit number, so it can only spell characters up to U+FFFF directly. A character above that, such as the emoji U+1F600, is written as two escapes, a **surrogate pair**: subtract 0x10000 to get a 20-bit number, put its high 10 bits after 0xD800 (the **high surrogate**, 0xD800 to 0xDBFF) and its low 10 bits after 0xDC00 (the **low surrogate**, 0xDC00 to 0xDFFF). A reader that sees a high surrogate must read the low one and combine them; a lone surrogate is an error.

A **recursive-descent parser** reads JSON with one function per kind of value: look at the next byte, and `{` means "read an object", which calls "read a value" for each member, and so on. The solution's parser keeps an index into the bytes and a depth counter (at most 64 levels), so a body of ten thousand `[` cannot exhaust the stack.

## Server-Sent Events

JSON answers one request with one document. For token streaming, the response must instead be a sequence of small messages, and that is what SSE provides. **SSE** is a one-way stream of events from server to client over a plain HTTP response with `Content-Type: text/event-stream`. The body is UTF-8 text made of **events**; each event is one or more `field: value` lines, and an **empty line** ends it:

```text
data: {"i":0}\n
\n
data: {"i":1}\n
\n
data: [DONE]\n
\n
```

The field these books use is `data`. A payload that contains a newline is sent as several `data:` lines, which the client joins with `\n`; JSON written without raw newlines needs only one. Lines starting with `:` are comments (servers send them as keep-alives). OpenAI-compatible APIs end a stream with the payload `[DONE]`, which is not JSON: a client checks for it before parsing.

`Cache-Control: no-cache` tells caches and proxies not to store or merge the stream. With no `Content-Length` and `Connection: close`, the stream ends when the server closes the connection after `[DONE]`.

## Buffering, flushing, and Nagle

Framing events correctly is not enough; they must also leave on time. Bytes a program writes can wait in three places before they reach the client: a user-space buffer (a `BufWriter` holds writes until it is full or flushed), the kernel's socket buffer, and **Nagle's algorithm**, a TCP rule that delays a small segment while an earlier one is unacknowledged, hoping to merge them. For a JSON response that is all harmless. For a stream it defeats the purpose: the client sees nothing for seconds and then every event at once.

So a streaming server writes each event and **flushes** it, and sets `TCP_NODELAY` on the socket (`stream.set_nodelay(true)` in Rust) so small writes leave immediately. Writing straight to a `TcpStream` has no user-space buffer; wrapping it in a `BufWriter` does, and then every event needs `flush()`.

## Seeing the wire

A few tools let you check every claim in this section against real bytes:

| Tool | What it shows |
|---|---|
| `curl -i URL` | the response status line and headers, then the body |
| `curl -N URL` | turns off curl's output buffer, so stream events print as they arrive |
| `curl -v URL` | the request bytes curl sent, prefixed `>`, and the response, prefixed `<` |
| `printf 'GET /health HTTP/1.1\r\nHost: x\r\n\r\n' \| nc 127.0.0.1 8080` | sends exact bytes, including malformed ones |

## Working it by hand

Here are the exact bytes the solution produces for the health, echo, and stream routes.

**Health.** The client sends `GET /health HTTP/1.1\r\nHost: test\r\n\r\n`. The body is `{"ok":true}`: count the bytes `{` `"` `o` `k` `"` `:` `t` `r` `u` `e` `}`, 11 in all, so:

```text
HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\nConnection: close\r\n\r\n{"ok":true}
```

**Echo.** The body `{"text":"hé"}` is 14 bytes: `{` (1), `"text"` (6), `:` (1), `"hé"` (5, because `é` is the two bytes `c3 a9`), `}` (1). The server parses it into the object with key `text` and the string `hé`, whose UTF-8 length is 3, and answers `{"text":"hé","bytes":3}`: 1 + 6 + 1 + 5 + 1 + 7 + 1 + 1 + 1 = 24 bytes, so `Content-Length: 24`.

**A surrogate pair.** The body `{"text":"😀"}` holds one character. High surrogate 0xD83D minus 0xD800 is 0x3D; low 0xDE00 minus 0xDC00 is 0x200. The code point is 0x10000 + (0x3D shifted left 10 bits) + 0x200 = 0x10000 + 0xF400 + 0x200 = 0x1F600, the emoji U+1F600, whose UTF-8 is the four bytes `f0 9f 98 80`. The reply's `bytes` is 4.

**The stream.** `GET /count?n=2 HTTP/1.1` answers with the head `HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n` and then the body

```text
data: {"i":0}\n\ndata: {"i":1}\n\ndata: [DONE]\n\n
```

and closes the connection. Run `curl -N 'http://127.0.0.1:8080/count?n=5&delay_ms=500'` and the five events appear half a second apart; if they appear together at the end, something is buffering.

## The routes and the interface

The package has this layout:

```text
primers/lang.05/
  Cargo.toml      package `wire`: a library and a binary named wire
  src/lib.rs      HTTP, JSON, SSE as plain functions over BufRead and Write
  src/main.rs     the server: std::net, one thread per connection, one request per connection
```

`wire --port <n>` binds `127.0.0.1:<n>` (port 0 picks a free port) and prints `listening on 127.0.0.1:<port>` as its first stdout line. Its routes are:

| Request | Response |
|---|---|
| `GET /health` | 200 `application/json` `{"ok":true}` |
| `POST /echo` with a JSON object holding a string `"text"` | 200 `application/json` `{"text": <the same string>, "bytes": <its UTF-8 length>}` |
| `GET /count?n=<0..1000>&delay_ms=<0..5000>` | 200 `text/event-stream`: `n` events `data: {"i":<k>}`, `delay_ms` apart, then `data: [DONE]`; `n` defaults to 3, `delay_ms` to 0 |
| a body that is not that JSON, or bad `n` / `delay_ms` | 400 `{"error": "<message>"}` |
| `POST /echo` without `Content-Length` | 411 |
| a known path with the wrong method | 405 with `Allow: GET` or `Allow: POST` |
| any other path | 404 |
| a malformed request line or header | 400 |

The library's interface, in `src/lib.rs`, is:

```rust
pub struct Request { pub method: String, pub target: String, pub headers: Vec<(String, String)>, pub body: Vec<u8> }
impl Request {
    pub fn header(&self, name: &str) -> Option<&str>;   // ignores ASCII case
    pub fn path(&self) -> &str;                          // target without "?query"
    pub fn query(&self, key: &str) -> Option<&str>;      // value of key=value, not decoded
}
pub enum HttpError { Closed, BadRequest(String), LengthRequired, TooLarge, Io(String) }
impl HttpError { pub fn status(&self) -> u16; }          // 400, 411, 413; 0 for Closed and Io

pub fn read_request<R: BufRead>(r: &mut R) -> Result<Request, HttpError>;
pub fn write_response<W: Write>(w: &mut W, status: u16, content_type: &str, extra: &[(&str, &str)], body: &[u8]) -> io::Result<()>;
pub fn write_sse_head<W: Write>(w: &mut W) -> io::Result<()>;
pub fn sse_event(data: &str) -> String;                 // "data: <line>\n" per line, then "\n"

pub enum Json { Null, Bool(bool), Num(f64), Str(String), Arr(Vec<Json>), Obj(Vec<(String, Json)>) }
impl Json { pub fn get(&self, key: &str) -> Option<&Json>; pub fn as_str(&self) -> Option<&str>; }
pub fn parse_json(text: &str) -> Result<Json, String>;  // one whole JSON text
pub fn json_string(s: &str) -> String;                 // quoted and escaped
```

Because the functions take any `BufRead` and any `Write`, unit tests can call `read_request(&mut &bytes[..])` on a byte slice and `write_response(&mut vec, ...)` into a `Vec<u8>`, with no socket.

## Building it step by step

Eight principles shape the code. Try to predict each function before reading it.

1. The request is untrusted bytes from a stream. So reading is bounded at every step: each line is read with a byte cap, the head as a whole is capped at 16 KiB, and the body at 1 MiB. Exceeding a cap is a 413, not an allocation.
2. The end of each piece is defined by the format, not by `read` boundaries: a line ends at `\n`, the head at an empty line, the body after exactly `Content-Length` bytes. So the reader sits on a `BufReader` and uses `read_until` and `read_exact`, which keep reading until the format says stop.
3. Header names are case-insensitive, but the values a client sent are kept as sent. So headers are stored as a list of pairs and looked up with a case-insensitive comparison, rather than normalized on the way in.
4. A request the server cannot frame (no length on a POST, chunked encoding it does not implement) must be refused, not guessed: guessing where a body ends desynchronizes the stream.
5. Every response with a known body states its byte length, and `Connection: close` makes "one request per connection" explicit. The SSE response has no length; closing the connection ends it.
6. JSON is a small recursive grammar, so a recursive-descent parser with one function per kind of value follows directly. Recursion depth is attacker-controlled, so it is capped.
7. Strings are where JSON is subtle: escapes on the way in (including surrogate pairs), escapes on the way out (quotes, backslash, controls). Both directions are one function each, used everywhere, so no code builds JSON strings with `format!`.
8. An SSE event is `data:` lines plus a blank line. Each event must reach the client when it is produced, so the socket has `TCP_NODELAY`, nothing buffers in user space, and each event is flushed.

### Bounded line reading

```rust
/// Reads one line ending in `\n` from `r`, at most `limit` bytes, and
/// returns it without the `\r\n` or `\n`. `Ok(None)` at a clean EOF.
fn read_line<R: BufRead>(r: &mut R, limit: usize) -> Result<Option<Vec<u8>>, HttpError> {
    let mut line = Vec::new();
    let n = r
        .by_ref()
        .take(limit as u64 + 1)            // never read more than limit + 1 bytes for one line
        .read_until(b'\n', &mut line)
        .map_err(|e| HttpError::Io(e.to_string()))?;
    if n == 0 {
        return Ok(None);                   // the client closed before sending anything
    }
    if line.last() != Some(&b'\n') {
        // either the cap stopped us (too large) or the stream ended mid-line
        return Err(if n > limit { HttpError::TooLarge } else { HttpError::Io("connection closed mid-line".into()) });
    }
    line.pop();                            // the \n
    if line.last() == Some(&b'\r') {
        line.pop();                        // and a \r before it
    }
    Ok(Some(line))
}
```

`take(limit + 1)` is the cap of step 1 applied inside `read_until`: one byte past the limit is enough to tell "too long" from "exactly at the limit". `by_ref()` lets `take` borrow the reader so it remains usable for the next line.

### The request line and headers

```rust
pub fn read_request<R: BufRead>(r: &mut R) -> Result<Request, HttpError> {
    let first = read_line(r, MAX_HEAD)?.ok_or(HttpError::Closed)?;
    let first = String::from_utf8(first).map_err(|_| HttpError::BadRequest("request line is not UTF-8".into()))?;
    let parts: Vec<&str> = first.split(' ').collect();
    let [method, target, version] = parts[..] else {          // exactly three parts, single spaces
        return Err(HttpError::BadRequest(format!("request line {first:?} is not `METHOD target HTTP/1.1`")));
    };
    if method.is_empty() || !method.bytes().all(|b| b.is_ascii_uppercase()) { /* 400 bad method */ }
    if !target.starts_with('/') { /* 400 */ }
    if version != "HTTP/1.1" && version != "HTTP/1.0" { /* 400 unsupported version */ }

    let mut used = first.len();
    let mut headers = Vec::new();
    loop {
        let line = read_line(r, MAX_HEAD)?.ok_or_else(|| HttpError::Io("connection closed in the headers".into()))?;
        used += line.len() + 2;            // count the CRLF too: the cap is on the whole head
        if used > MAX_HEAD {
            return Err(HttpError::TooLarge);
        }
        if line.is_empty() {
            break;                         // the empty line ends the head
        }
        let line = String::from_utf8(line).map_err(|_| HttpError::BadRequest("header is not UTF-8".into()))?;
        let (name, value) = line.split_once(':')
            .ok_or_else(|| HttpError::BadRequest(format!("header {line:?} has no colon")))?;
        if name.is_empty() || name.bytes().any(|b| b <= b' ' || b == 0x7f) {   // no spaces or controls in a name
            return Err(HttpError::BadRequest(format!("bad header name {name:?}")));
        }
        headers.push((name.to_string(), value.trim_matches(|c| c == ' ' || c == '\t').to_string()));
    }
    // ... body below
```

Splitting on single spaces and requiring exactly three parts rejects `GET  /health HTTP/1.1` (two spaces) and `GET /health` (no version), as RFC 9112 requires; a lenient parser that guessed would accept requests other servers on the path read differently. `split_once(':')` splits at the first colon, so a value like `127.0.0.1:8080` keeps its own colon.

### The body

```rust
    let mut req = Request { method: method.to_string(), target: target.to_string(), headers, body: Vec::new() };
    if req.header("transfer-encoding").is_some() {
        return Err(HttpError::BadRequest("chunked request bodies are not supported".into()));   // step 4
    }
    let len = match req.header("content-length") {            // case-insensitive lookup (step 3)
        Some(v) => {
            if v.is_empty() || !v.bytes().all(|b| b.is_ascii_digit()) {   // no sign, no spaces, no "1e3"
                return Err(HttpError::BadRequest(format!("bad Content-Length {v:?}")));
            }
            v.parse::<usize>().map_err(|_| HttpError::TooLarge)?          // all digits but overflows: too large
        }
        None if req.method == "POST" || req.method == "PUT" => return Err(HttpError::LengthRequired),
        None => 0,
    };
    if len > MAX_BODY {
        return Err(HttpError::TooLarge);   // before allocating
    }
    req.body = vec![0; len];
    r.read_exact(&mut req.body).map_err(|e| HttpError::Io(format!("body: {e}")))?;   // however many packets
    Ok(req)
}
```

`read_exact` loops until the buffer is full, so a body split across three TCP segments is read whole; a client that closes early gives `Io`, and the server simply ends that connection.

### Responses and SSE framing

```rust
pub fn write_response<W: Write>(w: &mut W, status: u16, content_type: &str,
                                extra: &[(&str, &str)], body: &[u8]) -> io::Result<()> {
    let mut head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n",
        reason(status),
        body.len()                         // bytes of the encoded body, never characters
    );
    for (k, v) in extra {
        head.push_str(&format!("{k}: {v}\r\n"));   // e.g. Allow: POST on a 405
    }
    head.push_str("\r\n");
    w.write_all(head.as_bytes())?;
    w.write_all(body)?;
    w.flush()
}

pub fn write_sse_head<W: Write>(w: &mut W) -> io::Result<()> {
    // no Content-Length: the stream ends when the server closes the connection
    w.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")?;
    w.flush()
}

pub fn sse_event(data: &str) -> String {
    let mut out = String::new();
    for line in data.split('\n') {         // a multi-line payload becomes several data: lines
        out.push_str("data: ");
        out.push_str(line);
        out.push('\n');
    }
    out.push('\n');                        // the blank line that ends the event
    out
}
```

Taking the body as `&[u8]` makes `body.len()` a byte count by construction, so the `Content-Length` of the echo example is 24, not 23.

### JSON: the parser's shape

```rust
    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > 64 {
            return Err("nested deeper than 64".into());   // step 6: bounded recursion
        }
        self.ws();
        match self.b.get(self.i) {         // one byte of lookahead decides the kind
            None => Err("unexpected end of input".into()),
            Some(b'n') => self.eat("null").map(|_| Json::Null),
            Some(b't') => self.eat("true").map(|_| Json::Bool(true)),
            Some(b'f') => self.eat("false").map(|_| Json::Bool(false)),
            Some(b'"') => self.string().map(Json::Str),
            Some(b'[') => { /* items: value(depth + 1), separated by ',' until ']' */ }
            Some(b'{') => { /* "key" ':' value(depth + 1) pairs, separated by ',' until '}' */ }
            Some(_) => self.number(),
        }
    }

pub fn parse_json(text: &str) -> Result<Json, String> {
    let mut p = Parser { b: text.as_bytes(), i: 0 };
    let v = p.value(0)?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("trailing characters at byte {}", p.i));   // `{} x` is not one JSON text
    }
    Ok(v)
}
```

`number` follows the JSON number grammar exactly (optional `-`, no leading zero, a fraction and an exponent that must have digits) and only then hands the matched text to `f64::parse`, which on its own would accept things JSON forbids, such as `inf` or `+1`. Objects keep their keys in order in a `Vec`, which is enough for lookup in tiny bodies and avoids a map dependency.

### JSON: strings in

```rust
                b'\\' => {
                    // ... read the escape letter e
                    let ch = match e {
                        b'"' => '"', b'\\' => '\\', b'/' => '/',
                        b'b' => '\u{8}', b'f' => '\u{c}', b'n' => '\n', b'r' => '\r', b't' => '\t',
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
                            // a lone LOW surrogate is not a char either: from_u32 rejects it
                            char::from_u32(code).ok_or("unpaired surrogate")?
                        }
                        _ => return Err(format!("bad escape \\{}", e as char)),
                    };
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());   // back to UTF-8 bytes
                }
                c if c < 0x20 => return Err("raw control character in a string".into()),
                c => out.push(c),          // raw bytes, already UTF-8 because the input is &str
```

This is the hand-worked surrogate arithmetic in code: `0xD83D, 0xDE00` gives `0x10000 + (0x3D << 10) + 0x200 = 0x1F600`, which `encode_utf8` writes as `f0 9f 98 80`.

### JSON: strings out

```rust
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
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),   // other controls
            c => out.push(c),              // everything else, non-ASCII included, as raw UTF-8
        }
    }
    out.push('"');
    out
}
```

### The server

The server handles one request per connection, routed by method and path:

```rust
fn handle(stream: TcpStream) -> io::Result<()> {
    // Small writes (one SSE event) must leave at once, not wait for more data.
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;  // a silent client cannot hold a thread forever
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut out = stream;                  // writes go straight to the socket: no user-space buffer
    match read_request(&mut reader) {
        Ok(req) => route(&req, &mut out),
        Err(HttpError::Closed) | Err(HttpError::Io(_)) => Ok(()),   // nobody to answer
        Err(e) => {
            let msg = format!("{e:?}");
            error(&mut out, e.status(), &msg, &[])                  // 400, 411, 413 as JSON
        }
    }
}

fn route(req: &Request, out: &mut TcpStream) -> io::Result<()> {
    match (req.method.as_str(), req.path()) {
        ("GET", "/health") => write_response(out, 200, "application/json", &[], b"{\"ok\":true}"),
        ("POST", "/echo") => echo(req, out),
        ("GET", "/count") => count(req, out),
        (_, "/health") | (_, "/count") => error(out, 405, "method not allowed", &[("Allow", "GET")]),
        (_, "/echo") => error(out, 405, "method not allowed", &[("Allow", "POST")]),
        _ => error(out, 404, &format!("no route for {}", req.path()), &[]),
    }
}
```

Matching on the `(method, path)` pair makes the 405 cases fall out naturally: a known path with any other method reaches its 405 arm, and only unknown paths reach the 404.

```rust
fn echo(req: &Request, out: &mut TcpStream) -> io::Result<()> {
    let text = std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(parse_json);
    let text = match text {
        Ok(v) => match v.get("text").and_then(Json::as_str) {
            Some(t) => t.to_string(),
            None => return error(out, 400, "body must be a JSON object with a string \"text\"", &[]),
        },
        Err(e) => return error(out, 400, &format!("bad JSON: {e}"), &[]),
    };
    let body = format!("{{\"text\":{},\"bytes\":{}}}", json_string(&text), text.len());
    write_response(out, 200, "application/json", &[], body.as_bytes())
}

fn count(req: &Request, out: &mut TcpStream) -> io::Result<()> {
    let n: u32 = match req.query("n").map(str::parse::<u32>) {
        None => 3,                         // absent: the default
        Some(Ok(n)) if n <= 1000 => n,
        _ => return error(out, 400, "n must be an integer from 0 to 1000", &[]),   // bad or out of range
    };
    // ... delay_ms the same way, 0..=5000, default 0
    write_sse_head(out)?;                  // validation is done before the 200 is sent
    for i in 0..n {
        if i > 0 && delay > 0 {
            thread::sleep(Duration::from_millis(delay));   // between events, not before the first
        }
        out.write_all(sse_event(&format!("{{\"i\":{i}}}")).as_bytes())?;
        out.flush()?; // a TcpStream has no buffer, but a BufWriter would: flush per event
    }
    out.write_all(sse_event("[DONE]").as_bytes())?;
    out.flush()                            // returning drops the socket: the stream ends
}
```

Parameters are validated before `write_sse_head`, because once the 200 head is sent the status can no longer change. The first event is written immediately, so a client sees it before the first pause, which is what "events leave as they happen" requires.

The full solution is in [`lib.rs`](../../../../course/solutions/primers/lang.05/src/lib.rs) and [`main.rs`](../../../../course/solutions/primers/lang.05/src/main.rs).

### Why this design

In cost, reading a request is linear in its size and bounded by the caps; the JSON parser is linear in the body; a stream of `n` events costs `n` small writes. Memory per connection is one buffer and the body, at most about 1 MiB.

The parser is hand-written rather than `httparse` or `hyper` because the tracer engine is std-only, and writing it once is how the bytes become readable; the [OpenAI-compatible server on tokio](../../build-an-llm/p10-serving/05-openai-server-on-tokio.md) moves to hyper once that understanding exists. One request per connection, rather than keep-alive, means there is no need to find the end of one request inside a stream that carries the next, and SSE can end by closing; the cost is a TCP handshake per request. Request-line parsing is strict rather than lenient because two parsers that disagree about where a request ends is how request smuggling happens, so rejecting anything ambiguous is the safe default. One tradeoff is accepted: query values are not percent-decoded, which is enough for `n=5` but not for arbitrary text in a query.

## What can go wrong

The checks cover both conformance and hostile input. The package must build a library and a `wire` binary, std only, because the tracer engine is std only, and its unit tests must pass without a socket. The health response must match the hand-worked bytes exactly, since every client relies on the status line and `Content-Length`. Echo must round-trip quotes, backslash, newline, tab, and non-ASCII text, with `bytes` and `Content-Length` counting bytes, because generated text can contain any byte. `é` and the surrogate pair for 😀 must decode to 6 bytes of UTF-8 in total, because prompts arrive from JSON clients with any escapes. `content-length` and `CONTENT-LENGTH` must both work, since a Go gateway's headers reach the engine. A head and body arriving in three pieces must be read whole, because TCP is a byte stream.

On the error side, five malformed JSON bodies must each produce a 400 JSON error, the same shape as the OpenAI-style errors the engine returns. A POST with no length must give 411, because a body's end must be known. Four bad request lines must each give 400, because untrusted input is rejected, not guessed. An unknown path gives 404 and a wrong method gives 405 with `Allow`. Finally, the SSE stream must be `text/event-stream` with no length, exact event bytes, and `[DONE]`, the way the engine's API streams; its first event must arrive before a 1 s pause ends; and `curl -N` must read it unchanged.

The mistakes that most often break these checks are:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Comparing header names exactly | a lower-case `content-length` is missed, so the body is never read and the client gets 411 | `Request::header` compares with `eq_ignore_ascii_case` |
| Reading the body with one `read()` call | a body that arrives in two TCP segments is cut short and the JSON parse fails | `read_exact` on the `BufReader` for exactly `Content-Length` bytes |
| `Content-Length` from `chars().count()` | non-ASCII responses are truncated by the client | `write_response` uses `body.len()` of the encoded bytes |
| Writing JSON with `format!("\"{}\"", s)` | a quote or newline in the text breaks the document | every string goes through `json_string` |
| Decoding `\uD83D` on its own | a lone surrogate is not a character: emoji become errors or garbage | `string()` reads the low surrogate after a high one and combines them; a lone one is an error |
| Ending an SSE event with one `\n` | two events merge into one, and the client's JSON parse fails | `sse_event` ends every event with an extra `\n` |
| Buffering the stream (a `BufWriter` flushed only at the end, Nagle left on) | `curl -N` shows nothing, then everything at once | `set_nodelay(true)`, no `BufWriter`, and `flush()` after every event |
| Panicking on a malformed request | the client sees the connection drop with no status | every failure is an `HttpError` or `Err(String)` turned into a status by `handle` and `echo` |

> **Warning:** If `curl -N` shows a stream all at once at the end, the framing may be perfect and the stream still broken. Look for a user-space buffer or Nagle's algorithm before anything else.

## Where this leads

The server skeleton came from the [previous section](04-rust.md): `TcpListener`, a thread per connection, `BufRead`, and `Result`. In *Build Your Own LLM System*, [Your first endpoint](../../build-an-llm/p10-serving/00-your-first-endpoint.md) grows this package into `tl-serve/src/http.rs`, which reads `POST /v1/completions`, parses its JSON, and streams one SSE chunk per token, ending with `data: [DONE]`, exactly as `/count` does. In *AI Platform Engineering*, the [streaming proxy](../../ai-platform-engineering/12-gateway/00-streaming-proxy.md) relays the engine's SSE through a Go gateway without buffering, flushing after every event for the reason you saw above, and the agent SDK's [types and provider](../../ai-platform-engineering/13-agent-sdk/01-types-and-provider.md) parse the same SSE stream as a client. The next section, [Go](06-go.md), introduces the language that gateway is written in.

### Further reading

- [RFC 9112, HTTP/1.1](https://www.rfc-editor.org/rfc/rfc9112) (free), sections 2 to 6; section 7.1 for chunked transfer coding (bodies of unknown length on a reused connection) and section 9 for keep-alive.
- [RFC 9110, HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110) (free), section 15 (status codes); RFC 9113 for HTTP/2, which multiplexes streams over one connection.
- [RFC 8259, JSON](https://www.rfc-editor.org/rfc/rfc8259) (free).
- [HTML Standard, Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html) (free): the full format, with `event:`, `id:`, `retry:`, and reconnection with `Last-Event-ID`.
- [MDN, Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) (free).
- [hyper](https://hyper.rs/) and [httparse](https://docs.rs/httparse/): a fuzzed, zero-copy parser and a full client and server.
- [serde_json](https://docs.rs/serde_json/): typed JSON parsers and writers derived from structs.

## Key terms

- **CRLF**: the two bytes `\r\n` that end every line of an HTTP/1.1 head.
- **header**: a case-insensitive name, a colon, and a value in an HTTP message head.
- **`Content-Length`**: the decimal count of bytes in a message body.
- **status code**: the three-digit number in a response's status line, whose first digit is its class.
- **surrogate pair**: two `\uXXXX` escapes, a high and a low surrogate, that together spell a character above U+FFFF.
- **recursive-descent parser**: a parser with one function per grammar rule, chosen by lookahead.
- **Server-Sent Events**: a `text/event-stream` response whose body is `data:` lines grouped into events by blank lines.
- **Nagle's algorithm**: a TCP rule that delays small segments to merge them, disabled with `TCP_NODELAY`.

## Summary

- An HTTP/1.1 message is CRLF-terminated lines: a start line, headers, an empty line, then a body whose end is given by `Content-Length` or, for a response with neither length nor chunking, by the server closing the connection.
- Header names ignore case, and `Content-Length` counts bytes, not characters.
- JSON strings are UTF-8 with escapes; `\uXXXX` names a UTF-16 code unit, so a character beyond U+FFFF is a surrogate pair.
- An SSE stream is a `text/event-stream` body of events, each `data:` lines followed by a blank line; the OpenAI convention ends it with `data: [DONE]`.
- Streaming means each event leaves the server when it is produced: write it, flush it, and do not let a buffer or Nagle's algorithm hold it.

## Review questions

1. Why does `read_line` use `take(limit + 1)` and not just check the length after `read_until`?

   <details><summary>Answer</summary>

   Checking afterwards means `read_until` has already read, and allocated for, everything up to the next `\n`, which a hostile client can make gigabytes long. `take` stops the read itself at `limit + 1` bytes; the extra byte distinguishes "too long" from "exactly at the limit".

   </details>

2. `Content-Length: 14` arrives as `content-length: 14`. Which line makes that work, and why are names not lower-cased on the way in?

   <details><summary>Answer</summary>

   `Request::header` compares with `eq_ignore_ascii_case`. Storing names as sent keeps the request faithful (useful for logging and for relaying headers in a proxy), and the comparison is the only place case needs to be ignored.

   </details>

3. Why does `count` validate `n` and `delay_ms` before calling `write_sse_head`?

   <details><summary>Answer</summary>

   The SSE head carries the status line `200 OK`. Once it is on the wire the server cannot answer 400 any more; an error would have to be an event inside a successful stream. Validating first keeps errors as proper HTTP errors.

   </details>

4. What two things together make `curl -N` show events half a second apart?

   <details><summary>Answer</summary>

   `set_nodelay(true)` disables Nagle's algorithm, so a small write is not held waiting for an acknowledgement, and the server writes each event straight to the `TcpStream` (no `BufWriter`) and flushes it. Either a user-space buffer or Nagle could hold events until the end.

   </details>

5. The parser sees `"\uDE00"`. What does it return, and why?

   <details><summary>Answer</summary>

   An error, "unpaired surrogate". `0xDE00` is not a high surrogate, so the code takes the `else` branch with `code = 0xDE00`, and `char::from_u32` returns `None` because surrogate code points are not Unicode scalar values. A lone low surrogate is not a character.

   </details>
