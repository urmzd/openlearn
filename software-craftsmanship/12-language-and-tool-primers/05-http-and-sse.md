<!-- ol:module lang.05 -->
# HTTP/1.1, JSON, and Server-Sent Events from the wire up

## Overview

| | |
|---|---|
| **Module** | `lang.05` · practice · Rust · Pass 1 · 5 to 7 h |
| **You build** | `primers/lang.05/`: the `wire` package, a library (HTTP request parser, response writer, JSON reader and writer, SSE framing, no sockets) and a `wire` server binary with three routes: `GET /health`, `POST /echo`, `GET /count` (an SSE stream) |
| **Contract** | none: the routes table and the signatures in section 4 are the contract; the standards are RFC 9112 (HTTP/1.1), RFC 8259 (JSON), and the WHATWG "Server-sent events" section |
| **Tests** | `course/tests/lang.05/check` builds your package, runs your own `cargo test`, then runs `test_lang05_http_sse.py`, which sends raw bytes over TCP and runs `curl -N` against your server (what each test checks: section 4) |
| **Needs** | reading: `lang.04` Rust (ownership, `Result`, traits, `std::net`, a thread per connection) |
| **Used by** | no call site (a primer): `L10.0` applies it next (your engine's `http.rs` is this package grown up), then `gw.00` (the gateway relays the same SSE) |
| **Milestone** | `MS-P1` |
| **Optional depth** | [RFC 9112, HTTP/1.1](https://www.rfc-editor.org/rfc/rfc9112) (free), sections 2 to 6; [RFC 9110, HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110) (free), section 15 (status codes); [RFC 8259, JSON](https://www.rfc-editor.org/rfc/rfc8259) (free); [HTML Standard, Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html) (free); [MDN, Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) (free) |

## Key Takeaways

- An HTTP/1.1 message is text lines ending in `\r\n`: a start line, headers, an empty line, then a body whose end is given by `Content-Length` (or, for a response with neither length nor chunking, by the server closing the connection) (`test_health_response_bytes`).
- Header names ignore case, and `Content-Length` counts **bytes**, not characters (`test_header_names_are_case_insensitive`, `test_echo_round_trips_json_text`).
- JSON strings are UTF-8 text with escapes; `\uXXXX` names a UTF-16 code unit, so a character beyond U+FFFF is written as two escapes, a **surrogate pair** (`test_json_unicode_escapes_and_surrogate_pairs`).
- An SSE stream is a `text/event-stream` response whose body is a sequence of events, each `data: <payload>` lines followed by a blank line; the OpenAI convention ends it with `data: [DONE]` (`test_sse_stream_framing`).
- Streaming means each event leaves the server when it is produced: write it, flush it, and do not let a buffer or Nagle's algorithm hold it (`test_sse_events_leave_as_they_happen`).

## How to work this chapter

```bash
ol start lang.05          # records that you started; the exercise lives in primers/lang.05/
ol tests lang.05          # read the test catalog first
ol check lang.05          # first run writes the starter package (stubs), then checks it
cd primers/lang.05 && cargo test && cargo run -- --port 8080
curl -N 'http://127.0.0.1:8080/count?n=5&delay_ms=500'   # watch the events arrive one by one
```

The first `ol check lang.05` writes the starter files that are missing into `primers/lang.05/`: `Cargo.toml` as given, and `src/lib.rs` and `src/main.rs` with every function body replaced by `todo!("lang.05")`. The starter compiles and fails. `ol check` never overwrites a file.

---

## 1. Why now

Your echo server (`lang.04`) speaks a protocol you invented, so only your own client can talk to it. The engine you write next (`L10.0`) must be reachable by `curl`, by the official OpenAI client libraries, and by the Go gateway (`gw.00`), and all of them speak the same thing: **HTTP/1.1** requests with **JSON** bodies, answered either with one JSON document or with a stream of **Server-Sent Events**, one event per generated token. The tracer engine is std-only Rust (`spec/cli-roles.md`: no TOML parser, no HTTP library), so you will write that protocol yourself, and when a client hangs or a stream arrives all at once you will need to read the bytes on the wire to see why. This primer builds a small server that does each piece once, on routes simple enough to check by eye.

## 2. Principles

### 2.1 An HTTP/1.1 request, byte by byte

HTTP is a **request and response** protocol over a TCP connection: the client sends a request, the server sends one response. In HTTP/1.1 both are text framed by lines ending in **CRLF** (the two bytes `\r\n`, carriage return and line feed).

```text
POST /echo HTTP/1.1\r\n                 request line: METHOD SP target SP version
Host: 127.0.0.1:8080\r\n                headers: name ":" optional spaces value
Content-Type: application/json\r\n
Content-Length: 14\r\n
\r\n                                    an empty line ends the headers
{"text":"hé"}                           the body: exactly 14 bytes
```

- The **method** says what the client wants: `GET` reads a resource, `POST` sends data to it. Methods are case-sensitive upper-case tokens.
- The **target** is the path, optionally followed by `?` and a **query string** of `key=value` pairs joined by `&`: `/count?n=3&delay_ms=0`.
- The **version** is `HTTP/1.1` (this server also accepts `HTTP/1.0`).
- A **header** is a name, a colon, and a value; spaces and tabs around the value are not part of it. Header names are **case-insensitive**: `Content-Length`, `content-length`, and `CONTENT-LENGTH` are the same header, and real clients send each spelling.
- A request with a body says how long it is with `Content-Length`, a decimal count of **bytes**. The server reads exactly that many bytes after the empty line. Without it, a `POST` cannot be read (the other way, `Transfer-Encoding: chunked`, is in "Going further"; this server answers it with 400).

A careful reader treats the request as untrusted input: it caps how many bytes the request line and headers may take (16 KiB here) and how large a body may be (1 MiB), so a client cannot make the server allocate without bound.

### 2.2 The response and its status

```text
HTTP/1.1 200 OK\r\n                     status line: version SP code SP reason
Content-Type: application/json\r\n
Content-Length: 11\r\n
Connection: close\r\n
\r\n
{"ok":true}
```

The **status code** is a three-digit number; its first digit is its class.

| Code | Meaning | This server sends it when |
|---|---|---|
| 200 OK | success | the route handled the request |
| 400 Bad Request | the request is malformed | bad request line, bad header, bad JSON body |
| 404 Not Found | no such resource | an unknown path |
| 405 Method Not Allowed | the resource exists, not for this method | `GET /echo`; the response names the allowed method in an `Allow` header |
| 411 Length Required | a body without `Content-Length` | `POST` with no length |
| 413 Content Too Large | over the size caps | a body over 1 MiB |

`Content-Type` names the body's **media type**: `application/json` for JSON, `text/event-stream` for SSE. `Connection: close` says the server closes the TCP connection after this response, so every connection carries exactly one request. That keeps the server simple and is what the tracer engine does too; reusing connections (**keep-alive**) is in "Going further".

### 2.3 Where a body ends

A reader of a byte stream must know where each message ends. HTTP/1.1 gives three ways for a response body:

| Framing | How the client knows the body ended | Used for |
|---|---|---|
| `Content-Length: N` | after N bytes | every complete JSON response here |
| `Transfer-Encoding: chunked` | a zero-length chunk | streams on connections that stay open |
| neither, with `Connection: close` | the server closes the connection | the SSE stream here and in `L10.0` |

A wrong `Content-Length` is a silent bug: too small and the client stops early and its JSON parser fails on a cut-off document; too large and the client waits for bytes that never come. It must count **bytes** of the encoded body, so `"é"` (two bytes in UTF-8) counts as 2.

### 2.4 JSON

**JSON** is a text format for values. Its grammar (RFC 8259) has six kinds of value:

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

A **recursive-descent parser** reads JSON with one function per kind of value: look at the next byte, and `{` means "read an object", which calls "read a value" for each member, and so on. Your parser keeps an index into the bytes and a depth counter, so a body of ten thousand `[` cannot exhaust the stack.

### 2.5 Server-Sent Events

**SSE** is a one-way stream of events from server to client over a plain HTTP response with `Content-Type: text/event-stream`. The body is UTF-8 text made of **events**; each event is one or more `field: value` lines, and an **empty line** ends it:

```text
data: {"i":0}\n
\n
data: {"i":1}\n
\n
data: [DONE]\n
\n
```

The field this course uses is `data`. A payload that contains a newline is sent as several `data:` lines, which the client joins with `\n`; JSON written without raw newlines needs only one. Lines starting with `:` are comments (servers send them as keep-alives). OpenAI-compatible APIs end a stream with the payload `[DONE]`, which is not JSON: a client checks for it before parsing.

`Cache-Control: no-cache` tells caches and proxies not to store or merge the stream. With no `Content-Length` and `Connection: close`, the stream ends when the server closes the connection after `[DONE]`.

### 2.6 Buffering, flushing, and Nagle

Bytes you write can wait in three places before they reach the client: a user-space buffer (a `BufWriter` holds writes until it is full or flushed), the kernel's socket buffer, and **Nagle's algorithm**, a TCP rule that delays a small segment while an earlier one is unacknowledged, hoping to merge them. For a JSON response that is all harmless. For a stream it defeats the purpose: the client sees nothing for seconds and then every event at once.

So a streaming server writes each event and **flushes** it, and sets `TCP_NODELAY` on the socket (`stream.set_nodelay(true)` in Rust) so small writes leave immediately. Writing straight to a `TcpStream` has no user-space buffer; wrapping it in a `BufWriter` does, and then every event needs `flush()`.

### 2.7 Seeing the wire

| Tool | What it shows |
|---|---|
| `curl -i URL` | the response status line and headers, then the body |
| `curl -N URL` | turns off curl's output buffer, so stream events print as they arrive |
| `curl -v URL` | the request bytes curl sent, prefixed `>`, and the response, prefixed `<` |
| `printf 'GET /health HTTP/1.1\r\nHost: x\r\n\r\n' \| nc 127.0.0.1 8080` | sends exact bytes you wrote, including malformed ones |

## 3. Worked example by hand

These are the exact bytes `test_health_response_bytes`, `test_echo_round_trips_json_text`, and `test_sse_stream_framing` check.

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

and closes the connection. Run `curl -N 'http://127.0.0.1:8080/count?n=5&delay_ms=500'` and the five events appear half a second apart; if they appear together at the end, something is buffering (section 2.6).

## 4. The artifact and its check

```text
primers/lang.05/
  Cargo.toml      package `wire`: a library and a binary named wire
  src/lib.rs      HTTP, JSON, SSE as plain functions over BufRead and Write
  src/main.rs     the server: std::net, one thread per connection, one request per connection
```

Routes of `wire --port <n>` (binds `127.0.0.1:<n>`; port 0 picks a free port; the first stdout line is `listening on 127.0.0.1:<port>`):

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

The library's interface, in `src/lib.rs`:

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

Because the functions take any `BufRead` and any `Write`, your own unit tests can call `read_request(&mut &bytes[..])` on a byte slice and `write_response(&mut vec, ...)` into a `Vec<u8>`, with no socket.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_crate_builds` | conformance | the package builds a library and a `wire` binary, std only | the tracer engine is std only (`L10.0`) |
| `test_your_cargo_tests_pass` | unit | your own unit tests pass | your parser is tested without a socket |
| `test_health_response_bytes` | unit | section 3's exact response bytes | every client relies on the status line and `Content-Length` |
| `test_echo_round_trips_json_text` | unit | quotes, backslash, newline, tab, non-ASCII survive; `bytes` and `Content-Length` count bytes | generated text can contain any byte (`L10.0`) |
| `test_json_unicode_escapes_and_surrogate_pairs` | boundary | `é` and the pair `😀` decode to 6 bytes of UTF-8 | prompts arrive from JSON clients with any escapes |
| `test_header_names_are_case_insensitive` | boundary | `content-length` and `CONTENT-LENGTH` both work | the Go gateway's headers reach your engine (`gw.00`) |
| `test_body_split_across_packets` | boundary | head and body arriving in three pieces | TCP is a byte stream (`lang.04`) |
| `test_bad_json_is_a_400_with_a_json_error` | boundary | five malformed bodies, each a 400 JSON error | the OpenAI error shape in `L10.0` |
| `test_post_without_content_length_is_411` | boundary | no length, 411 | a body's end must be known |
| `test_malformed_request_line_is_400` | boundary | four bad request lines | untrusted input is rejected, not guessed |
| `test_unknown_path_and_wrong_method` | unit | 404, and 405 with `Allow` | the engine answers unknown routes the same way |
| `test_sse_stream_framing` | conformance | `text/event-stream`, no length, exact event bytes, `[DONE]` | the v0 API streams exactly this way |
| `test_sse_events_leave_as_they_happen` | fault | the first event arrives before the 1 s pause ends | tokens must reach the user as they are generated |
| `test_curl_n_reads_the_stream` | conformance | `curl -N` reads your stream unchanged | off-the-shelf clients agree with your framing |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Comparing header names exactly | a lower-case `content-length` is missed, so the body is never read and the client gets 411 | `test_header_names_are_case_insensitive` |
| Reading the body with one `read()` call | a body that arrives in two TCP segments is cut short and the JSON parse fails | `test_body_split_across_packets` |
| `Content-Length` from `chars().count()` | non-ASCII responses are truncated by the client | `test_echo_round_trips_json_text` |
| Writing JSON with `format!("\"{}\"", s)` | a quote or newline in the text breaks the document | `test_echo_round_trips_json_text` |
| Decoding `\uD83D` on its own | a lone surrogate is not a character: emoji become errors or garbage | `test_json_unicode_escapes_and_surrogate_pairs` |
| Ending an SSE event with one `\n` | two events merge into one, and the client's JSON parse fails | `test_sse_stream_framing` |
| Buffering the stream (a `BufWriter` flushed only at the end, Nagle left on) | `curl -N` shows nothing, then everything at once | `test_sse_events_leave_as_they_happen` |
| Panicking on a malformed request | the client sees the connection drop with no status | `test_malformed_request_line_is_400`, `test_bad_json_is_a_400_with_a_json_error` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `lang.04` | the server skeleton: `TcpListener`, a thread per connection, `BufRead`, `Result` |
| Forward | `L10.0` | `tl-serve/src/http.rs` reads `POST /v1/completions`, parses its JSON, and streams one SSE chunk per token, ending with `data: [DONE]`, exactly as `/count` does |
| Forward | `gw.00` | the Go gateway proxies the engine's SSE without buffering: it flushes after every event, for the reason in section 2.6 |
| Forward | `ag.01` | the agent SDK parses the same SSE stream as a client |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one request per connection | HTTP/1.1 keep-alive, HTTP/2 | many requests per connection; HTTP/2 multiplexes streams over one connection | RFC 9112 section 9; RFC 9113 |
| `Content-Length` request bodies only | chunked transfer coding | bodies of unknown length on a reused connection | RFC 9112 section 7.1 |
| your hand-written parser | [hyper](https://hyper.rs/), [httparse](https://docs.rs/httparse/) | a fuzzed, zero-copy parser and a full client and server | `L10.5` moves the engine to hyper |
| `data:` events only | the full SSE format | `event:`, `id:`, `retry:`, and reconnection with `Last-Event-ID` | HTML Standard, server-sent events |
| your JSON reader | [serde_json](https://docs.rs/serde_json/) | derives typed parsers and writers from your structs | `L10.5` |
