# Your first endpoint: a std-only Rust HTTP/1.1 + SSE server

At this point in the book you have a trained byte bigram on disk and a Python CLI that can sample from it, and nothing else can reach it. No process listens on a port, so the gateway you build in [the tracer gateway](../../ai-platform-engineering/12-gateway/01-streaming-proxy.md) (in *AI Platform Engineering*) has no upstream, [the tracer deploy](../../infrastructure/01-containers-kubernetes/01-tracer-deploy.md) has no image to run, and there is no request for [one trace](../../systems/04-observability/01-one-trace.md) to follow. Every later layer of the system is a client of an inference server, from the load generator to the agent SDK. This section writes that server.

The server is a Rust process that speaks the smallest OpenAI-compatible API (v0: `POST /v1/completions`, plain or streamed), serves the checkpoint written in [Byte bigram](../01-foundations/01-byte-bigram.md), computes each token's logits by selecting the matching row of the checkpoint, and reports each request as a span. It uses only Rust's standard library, so the HTTP framing, the JSON, the decoding, and the trace export are all code you can read. It is the engine of the first end-to-end tracer: the first point at which every layer of the system runs together.

This section builds on the [Rust primer](../../programming-languages/02-language-and-tool-primers/04-rust.md) and on [HTTP/1.1, JSON, and Server-Sent Events from the wire up](../../programming-languages/02-language-and-tool-primers/05-http-and-sse.md), both in *Programming Languages*. The contracts it implements are the HTTP API [`openapi/openai-subset.v0.yaml`](../../../../course/contracts/openapi/openai-subset.v0.yaml), the `engine` role in [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md), and the model directory formats [`formats/safetensors.md`](../../../../course/contracts/formats/safetensors.md), [`formats/config.schema.json`](../../../../course/contracts/formats/config.schema.json), and [`formats/tokenizer.md`](../../../../course/contracts/formats/tokenizer.md). The finished code is [`rust/crates/tl-serve/src/http.rs`](../../../../course/solutions/rust/crates/tl-serve/src/http.rs) (v0), with the crate root [`lib.rs`](../../../../course/solutions/rust/crates/tl-serve/src/lib.rs) and the entry point [`main.rs`](../../../../course/solutions/entry/rust/crates/tl-serve/src/main.rs), a thin wrapper over `http::serve`. This is how the finished solution is built and run:

```bash
cargo build --release --manifest-path rust/Cargo.toml
uv run --project python python python/tinyllm/__main__.py train bigram --data some.txt --out artifacts/bigram
rust/target/release/tl-serve --model-dir artifacts/bigram --port 8000 --health-port 9464
curl -N localhost:8000/v1/completions -d '{"model":"tracer","prompt":"Once","max_tokens":32,"temperature":0,"stream":true}'
```

> **Learning objectives**
> - Read a safetensors checkpoint correctly, counting tensor offsets from the start of the data buffer rather than the file.
> - Explain why the bigram's logits are exactly row `prev` of the weight matrix, and implement greedy decoding (ties to the lowest id) and seeded temperature sampling with PCG32 and the inverse CDF.
> - Stream one-byte tokens as text by holding back incomplete UTF-8 sequences, so the concatenated stream equals the non-streamed answer.
> - Frame HTTP/1.1 requests and Server-Sent Events by hand, and answer every malformed request in the OpenAI error shape.
> - Emit one OTLP span per request that joins the caller's trace through `traceparent`, without ever delaying the client.

## The request path

Before looking at any one piece, it helps to see the whole journey a request takes through the engine:

```text
TCP bytes --read_request--> Request --parse_completion--> CompletionRequest
   --for each new token: Bigram::logits (row lookup) then sample--> byte id
   --Utf8Stream--> text --JSON body, or one SSE chunk per token--> TCP bytes
   (connection closed) --otlp_json + export_span--> POST <collector>/v1/traces
```

Each connection carries one request and gets one thread, using the patterns from the Rust and HTTP primers. The model, loaded once at start, is shared read-only by every thread through an `Arc` (a reference-counted pointer that many threads may hold). The rest of this section follows the path from left to right: first the model and how it turns a byte into logits, then how a token is chosen, how bytes become text, how the checkpoint is read, what the API promises, and finally how the request is reported as a span.

## The model and the one-hot product

The notation for this section is collected in one table:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V = 256$ | vocabulary size: one id per byte value | integer |
| $x_t \in \{0, \dots, 255\}$ | the byte id at position $t$ | `u8` |
| $W \in \mathbb{R}^{V \times V}$ | `bigram.weight`: row $i$ holds the next-token logits after byte $i$ | `f32[256 * 256]`, row-major |
| $e_i \in \mathbb{R}^{V}$ | one-hot vector: 1 at position $i$, 0 elsewhere | `f32[256]` |
| $z \in \mathbb{R}^{V}$ | logits for the next token | `f32[256]` |
| $T \ge 0$ | temperature | `f64` |
| $p_j$ | probability of id $j$ after tempering | `f64` |
| $u \in [0, 1)$ | one uniform draw | `f64` |

A bigram model predicts the next byte from the previous byte only. Its logits after byte $x_t$ are row $x_t$ of $W$, which you can write as a product:

$$z = e_{x_t} W, \qquad z_j = \sum_{k} [k = x_t]\, W_{kj} = W_{x_t j}.$$

Every term but one is an exact zero, so the product equals the row bit for bit. The endpoint therefore copies row $x_t$ directly from the row-major weight array. That avoids a matrix operation for a one-hot vector and preserves the checkpoint values exactly.

The **prompt** is encoded by the byte tokenizer (its ids are its UTF-8 bytes), and generation starts from its last byte, so the prompt must be non-empty; the v0 contract requires it.

The endpoint reads the checkpoint into a Rust `Bigram` value. `Bigram::from_weight` checks that the row-major matrix has exactly 256 × 256 values, and `Bigram::logits(prev)` returns the row for the previous byte. This first endpoint uses Rust's standard library only; later sections use Candle for tensor operations in the Llama runner.

## Choosing the next token

With logits in hand, the engine has to pick one id. There are two rules, and both are fixed precisely so that every implementation produces the same stream.

**Greedy** ($T = 0$) takes $\arg\max_j z_j$, and when several ids share the maximum, the **lowest** id (decision D11). Fixing the tie rule makes the stream a function of the weights alone, so the Python CLI, this engine, and anyone else's implementation reach the same bytes.

**Sampling** ($T > 0$) divides by the temperature, subtracts the maximum (so no exponent overflows), exponentiates, and normalizes, all in `f64`:

$$p_j = \frac{\exp\big((z_j - m)/T\big)}{\sum_k \exp\big((z_k - m)/T\big)}, \qquad m = \max_k z_k.$$

Low $T$ sharpens the distribution toward greedy; $T = 1$ is the model's own distribution. The engine then draws one $u$ and walks the **cumulative distribution**, returning the first $j$ with $p_0 + \dots + p_j > u$. This is the **inverse CDF** method: each $j$ owns an interval of $[0, 1)$ as long as $p_j$.

The uniform comes from **PCG32** (PCG-XSH-RR 64/32, decision D10), a generator whose whole state is two 64-bit integers, $s$ (state) and $c$ (an odd increment that selects the stream):

$$s' = s \cdot 6364136223846793005 + c \pmod{2^{64}}, \qquad \text{out} = \mathrm{rotr}_{32}\big((((s \gg 18) \oplus s) \gg 27) \bmod 2^{32},\; s \gg 59\big),$$

where the output is computed from the **old** state. Seeding with $(\text{seed}, \text{seq})$ follows O'Neill's reference: $c = 2\,\text{seq} + 1$, $s = 0$, one step, $s \mathrel{+}= \text{seed}$, one step. A uniform with 53 random bits takes two outputs $a, b$: $u = \big((a \gg 5)\, 2^{26} + (b \gg 6)\big)\, 2^{-53}$. The engine seeds a fresh generator per request with $(\text{seed}, 54)$, so the same `seed` on the same engine gives the same text (the v0 Determinism rule); without a `seed` it picks one at random. The generator itself is derived in [Modular arithmetic, hashing, and PCG32 in Python](../../mathematics-for-computing/07-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) in *Mathematics for Computing*.

## Bytes to text, one token at a time

A token is now a byte id, but the client wants text. UTF-8 writes a character as 1 to 4 bytes, and the first byte says how many: `0xxxxxxx` is a whole character, `110xxxxx` starts a 2-byte one, `1110xxxx` a 3-byte one, `11110xxx` a 4-byte one, and each following byte is `10xxxxxx`. Bytes `C0`, `C1`, and `F5` to `FF` never appear.

A byte model emits one byte per token, so the text a token completes may be empty (the first byte of `é`), one character, or a replacement character U+FFFD for an invalid sequence. `Utf8Stream` keeps the pending bytes and, after each new byte, emits the longest valid prefix. An invalid sequence becomes one U+FFFD for its **maximal subpart** (the longest prefix of a valid sequence), and a tail that could still complete stays pending. Rust's `std::str::from_utf8` reports exactly the two numbers this needs: `valid_up_to()` and `error_len()`, which is `None` for "incomplete, wait". At the end, whatever is still pending is decoded with replacement (`String::from_utf8_lossy`). The concatenated chunk texts then equal the non-streamed text, as the v0 contract requires.

## Reading the checkpoint

So far the weights have been an abstract $W$. On disk, a model directory holds `config.json` and `model.safetensors` (see `formats/safetensors.md`). The tracer engine accepts only `"tl_arch": "bigram"`, `"tl_tokenizer": "bytes"`, and `"vocab_size": 256`. The safetensors file is laid out as

```text
offset 0      u64 little-endian N, the header length
offset 8      N bytes of JSON header, padded with spaces to a multiple of 8
offset 8 + N  the data buffer
```

and each header entry's `data_offsets` is `[begin, end)` **relative to the data buffer**, not to the file. The reader rejects a header length past the end of the file, a dtype other than `F32` (contract v0), a shape other than `[256, 256]`, duplicate names, and tensors that do not tile the data buffer exactly (gaps, overlaps, or trailing bytes). Weights are little-endian `f32`, read with `f32::from_le_bytes`.

## Serving API v0

The model side is now complete; the other side of the engine is the contract with clients. `POST /v1/completions` takes `{model, prompt, max_tokens, temperature, seed, stream}` (`openai-subset.v0.yaml`): `model` and `prompt` are non-empty strings, `max_tokens` is an integer in 1 to 4096 (default 16), `temperature` is a number in 0 to 2 (default 1), `seed` is a non-negative integer, and `stream` is a boolean. Unknown fields are ignored. Any violation is `400` with the error shape `{"error": {"message", "type": "invalid_request_error", "param": <the field>, "code": null}}`.

Without `stream`, the answer is one JSON `Completion` with `Content-Length` (in bytes) and `usage`, where `prompt_tokens` is the prompt's byte length and `completion_tokens` the number generated. With `stream: true` the answer is `text/event-stream`: one `data: <CompletionChunk>` event per generated token, all with the same `id`, `finish_reason: null` until the last chunk, which carries `"length"` and any held-back bytes, then `data: [DONE]`. Each event is written straight to the socket with `TCP_NODELAY` set, so it leaves at once (the HTTP primer explains why). An error after the first byte cannot change the status any more, so it is sent as a `data: {"error": ...}` event before the stream ends.

`GET /healthz` answers 200 on both the API port and the health port. Every response has `Connection: close`. An `X-Request-Id` header of 1 to 128 visible ASCII characters is echoed back.

## One span per request

The last thing the engine does for each request is report it. A **trace** is the tree of operations one request caused across services; each operation is a **span** with a 16-byte trace id (shared by the whole tree), an 8-byte span id, its parent's span id, a name, a kind, start and end times, and attributes. The W3C header `traceparent: 00-<trace id, 32 hex>-<parent span id, 16 hex>-<flags, 2 hex>` carries the caller's position in the tree; it uses lower-case hex only, and all-zero ids are invalid. When the header is valid, the engine's span reuses the trace id and records the caller's span as its parent; otherwise the engine starts a new trace with a random non-zero id.

When `OTEL_EXPORTER_OTLP_ENDPOINT` is set (`spec/cli-roles.md`), the engine posts each completion's span to `<endpoint>/v1/traces` as **OTLP/HTTP JSON**: `{"resourceSpans": [{"resource": {"attributes": [service.name]}, "scopeSpans": [{"scope": ..., "spans": [...]}]}]}`, with ids as lower-case hex strings, times as decimal strings of Unix nanoseconds, `kind: 2` (SERVER), and 64-bit integer attributes as strings (`{"intValue": "200"}`). It exports **after** closing the client's connection, so a slow collector never delays an answer. Jaeger accepts this format on port 4318, which is how [One trace](../../systems/04-observability/01-one-trace.md) draws gateway to engine as one trace.

Finally, `spec/cli-roles.md` asks servers to exit 0 on SIGTERM, which is what Kubernetes sends before killing a pod. The tracer's [`main.rs`](../../../../course/solutions/entry/rust/crates/tl-serve/src/main.rs) starts a thread that waits for SIGTERM and exits 0; there is nothing to drain yet. The async server later in this chapter drains in-flight requests first.

## Working it by hand

Every rule above can be checked with a pencil before you write any code.

**The model.** A small `succ` model has $W_{i,(i+1) \bmod 256} = 2$ and every other entry 0: after any byte, the next byte value is the likeliest. On disk, with the metadata `{"format":"tinyllm"}`, the header text is 113 bytes, padded with 7 spaces to $N = 120$, so the data buffer starts at byte $8 + 120 = 128$. $W_{97,98}$ (after `a`, the logit of `b`) is element $256 \cdot 97 + 98 = 24930$, at data offset $4 \cdot 24930 = 99720$ and file offset $128 + 99720 = 99848$, and holds the bytes `00 00 00 40` (2.0 in little-endian `f32`). Read from file offset 99720 instead and you get a different weight: that is the data-offset pitfall.

**Greedy, three tokens from `a`.** The prompt `a` is the id 97. $z = e_{97} W$ is row 97: 2 at index 98, 0 elsewhere, so the argmax is 98 (`b`). Then row 98 gives 99 (`c`), and row 99 gives 100 (`d`). The answer to `{"model":"m","prompt":"a","max_tokens":3,"temperature":0}` is the text `bcd` with `finish_reason: "length"` and `usage` 1, 3, 4.

**A small matrix product.** The same row-selection rule follows from the one-hot product, though this endpoint uses a direct row lookup. As a reminder of how a matrix product combines rows and columns:

$$\begin{bmatrix}1 & 2\\ 3 & 4\end{bmatrix}\begin{bmatrix}5 & 6\\ 7 & 8\end{bmatrix} = \begin{bmatrix}1\cdot5+2\cdot7 & 1\cdot6+2\cdot8\\ 3\cdot5+4\cdot7 & 3\cdot6+4\cdot8\end{bmatrix} = \begin{bmatrix}19 & 22\\ 43 & 50\end{bmatrix}.$$

**The stream.** With `"max_tokens":2,"stream":true`, the response body is three events (the `id` and `created` values vary):

```text
data: {"id":"cmpl-9d9d...","object":"text_completion","created":1791518626,"model":"m","choices":[{"index":0,"text":"b","finish_reason":null}]}\n\n
data: {"id":"cmpl-9d9d...","object":"text_completion","created":1791518626,"model":"m","choices":[{"index":0,"text":"c","finish_reason":"length"}]}\n\n
data: [DONE]\n\n
```

**Temperature.** Suppose a row has $z_A = \ln 0.75$ and $z_B = \ln 0.25$ and every other logit is $-30$. At $T = 0.5$: $z_A / T = -0.5754$ and $z_B / T = -2.7726$; subtract the maximum to get $0$ and $-2.1972$; exponentiate to get $1$ and $0.1111$; normalize to get $p_A = 0.9$ and $p_B = 0.1$ (the other ids get about $e^{-60}$). Tempering squared the odds, $0.75^2 : 0.25^2 = 9 : 1$. With $u = 0.3$ the walk stops at $A$ (cumulative 0.9 > 0.3); with $u = 0.95$ it passes $A$ and stops at $B$ (cumulative 1.0 > 0.95). Drawing 2000 times should give about 1800 and 200.

**UTF-8.** The `utf8` model emits `C3 A9 FF` after `x`. `C3` is `110 00011`, the start of a 2-byte character, so the first chunk is `""`. `A9` is `10 101001`, a continuation; together they carry the bits `00011 101001` = U+00E9, so the second chunk is `"é"`. `FF` can never appear in UTF-8, so the third chunk is U+FFFD. The non-streamed text of the same request is `"é\u{FFFD}"`, the same string.

**A trace.** A request with `traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01` produces a span with `traceId` `4bf92f3577b34da6a3ce929d0e0e4736`, `parentSpanId` `00f067aa0ba902b7`, a new random `spanId`, `name` `POST /v1/completions`, and `kind` 2.

## Designing the interface

The endpoint's model interface is deliberately small: `Bigram::load` validates `config.json` and `model.safetensors`, then `logits(prev)` returns a copied slice of the selected transition row. There is no foreign-function boundary in this endpoint. The whole public surface of `http.rs` fits in one listing:

```rust
// rust/crates/tl-serve/src/http.rs (v0); lib.rs is `pub mod http;`
pub struct Request { pub method: String, pub target: String, pub headers: Vec<(String, String)>, pub body: Vec<u8> }
impl Request { pub fn header(&self, name: &str) -> Option<&str>; pub fn path(&self) -> &str; }
pub struct HttpError { pub status: u16, pub message: String }           // 400 411 413 431 501; 0: client gone
pub fn read_request<R: BufRead>(r: &mut R) -> Result<Request, HttpError>;

pub enum Json { Null, Bool(bool), Int(i64), Float(f64), Str(String), Arr(Vec<Json>), Obj(Vec<(String, Json)>) }
impl Json { pub fn get(&self, key: &str) -> Option<&Json>; }
pub fn parse_json(text: &str) -> Result<Json, String>;
pub fn json_string(s: &str) -> String;

pub struct ApiError { pub status: u16, pub error_type: String, pub message: String, pub param: Option<String>, pub code: Option<String> }
impl ApiError { pub fn invalid(param: Option<&str>, message: impl Into<String>) -> ApiError; pub fn to_json(&self) -> String; }
pub struct CompletionRequest { pub model: String, pub prompt: String, pub max_tokens: u32, pub temperature: f64, pub seed: Option<u64>, pub stream: bool }
pub fn parse_completion(body: &[u8]) -> Result<CompletionRequest, ApiError>;

pub struct Bigram { /* weight: Vec<f32>, row-major [256, 256] */ }
impl Bigram {
    pub fn from_weight(weight: Vec<f32>) -> Result<Bigram, String>;
    pub fn load(dir: &Path) -> Result<Bigram, String>;               // config.json + model.safetensors
    pub fn logits(&self, prev: u8) -> Result<Vec<f32>, String>;       // copy row prev from W
}
pub fn read_bigram_weight(bytes: &[u8]) -> Result<Vec<f32>, String>;  // the safetensors reader rules

pub struct Pcg32 { /* state, inc */ }
impl Pcg32 { pub fn new(seed: u64, seq: u64) -> Pcg32; pub fn next_u32(&mut self) -> u32; pub fn uniform(&mut self) -> f64; }
pub fn sample(logits: &[f32], temperature: f64, rng: &mut Pcg32) -> u8;
pub fn next_token(model: &Bigram, prev: u8, temperature: f64, rng: &mut Pcg32) -> Result<u8, String>;

pub struct Utf8Stream { /* pending bytes */ }
impl Utf8Stream { pub fn new() -> Utf8Stream; pub fn push(&mut self, byte: u8) -> String; pub fn finish(&mut self) -> String; }

pub fn parse_traceparent(value: &str) -> Option<(String, String)>;    // (trace id, parent span id)
pub fn random_u64() -> u64;  pub fn random_hex(bytes: usize) -> String;  pub fn unix_nanos() -> u128;
pub enum AttrValue { Str(String), Int(i64) }
pub struct Span { pub trace_id: String, pub span_id: String, pub parent_span_id: Option<String>, pub name: String,
                  pub start_unix_nano: u128, pub end_unix_nano: u128, pub attributes: Vec<(String, AttrValue)>, pub error: bool }
pub fn otlp_json(service_name: &str, span: &Span) -> String;
pub fn export_span(endpoint: &str, body: &str) -> Result<(), String>;

pub struct Server { pub model: Bigram, pub otlp_endpoint: Option<String>, pub service_name: String }
impl Server { pub fn from_env(model_dir: &Path) -> Result<Server, String>; }   // OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_SERVICE_NAME
pub fn serve(listener: TcpListener, server: Arc<Server>) -> io::Result<()>;  // a thread per connection, forever
pub fn handle_connection(stream: TcpStream, server: &Server);
pub fn sse_event(payload: &str) -> String;
```

`HttpError` uses status 0 to mean "the client went away", in which case the caller answers nothing. The entry point [`main.rs`](../../../../course/solutions/entry/rust/crates/tl-serve/src/main.rs) parses `--model-dir`, `--port`, and `--health-port` (exit 2 on a usage error), builds the `Server` with `Server::from_env`, binds `0.0.0.0:<port>` and `0.0.0.0:<health port>`, serves the health listener on a second thread, and calls `serve` on the API listener.

## Building it step by step

### The reasoning chain

Before reading code, follow the chain of decisions that shapes it. Try to predict each step of the solution before you read it.

1. **One request per connection, `Connection: close`, one thread per connection.** That removes keep-alive, pipelining, and chunked request bodies from the parser, and a stalled client blocks only its own thread. The model is immutable after load, so threads share it through an `Arc` with no lock.
2. **The request is framed by `Content-Length` only.** The parser reads lines up to the empty line, then exactly `Content-Length` bytes. Every limit (head size, body size) is checked before allocating, so a hostile client gets 413 or 431, not an out-of-memory.
3. **The model is a table lookup.** Because $e_{x_t} W$ is exactly row $x_t$, `logits(prev)` copies 256 floats starting at `prev * 256`. In the worked example, row 97 has its 2.0 at index 98, so greedy from `a` gives `b`.
4. **Reproducibility fixes two rules:** greedy ties go to the lowest id, and sampling uses one seeded PCG32 per request, one uniform per token, and the inverse CDF in `f64`. Any other rule (a different RNG, a different draw count) produces a different stream from the same seed.
5. **Text must be emitted per token, but tokens are bytes.** So a decoder holds back an incomplete UTF-8 tail and releases it when it completes or fails; the final chunk flushes what remains. That makes the concatenated chunks equal the non-streamed text.
6. **Once the first byte of a stream is written, the status is fixed at 200.** Errors after that point must go inside the stream as an event.
7. **Tracing must never slow an answer.** So the span is built and posted after the client's socket is shut down, with a 2-second timeout on the collector.

### The request parser

The request line must split into exactly three parts; then headers are read until the empty line with a running byte count. The body length comes from `Content-Length` alone: a POST without it is 411, and `Transfer-Encoding` is refused with 501 rather than misread.

```rust
pub fn read_request<R: BufRead>(r: &mut R) -> Result<Request, HttpError> {
    // status 0 means "the client went away": the caller answers nothing
    let first = read_line(r, MAX_HEAD)?.ok_or_else(|| http_error(0, "closed before a request"))?;
    // ... request line split into exactly METHOD target HTTP/1.x ...
    let mut used = first.len() + 2;
    let mut headers = Vec::new();
    loop {
        let line = read_line(r, MAX_HEAD)?.ok_or_else(|| http_error(0, "closed in the headers"))?;
        used += line.len() + 2;
        if used > MAX_HEAD {                      // 16 KiB for the whole head, not per line
            return Err(http_error(431, "request head too large"));
        }
        if line.is_empty() { break; }             // the blank line ends the head
        // ... split at the first ':', reject empty or control-character names ...
        headers.push((name.to_string(), value.trim_matches(|c| c == ' ' || c == '\t').to_string()));
    }
    let mut req = Request { method: method.to_string(), target: target.to_string(), headers, body: Vec::new() };
    if req.header("transfer-encoding").is_some() {      // header() ignores ASCII case
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
    if len > MAX_BODY { return Err(http_error(413, "body too large")); } // check before allocating
    req.body = vec![0; len];
    r.read_exact(&mut req.body).map_err(|e| http_error(0, format!("body: {e}")))?;
    Ok(req)
}
```

`Request::header` uses `eq_ignore_ascii_case`, which is what makes `content-length` from one client and `Content-Length` from another equivalent.

### JSON output escaping

A byte model can generate any byte, including NUL and newline, so every character below U+0020 must be escaped or the chunk breaks the client's parser.

```rust
pub fn json_string(s: &str) -> String {
    // ...
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)), // NUL, U+0001, ...
            c => out.push(c),                    // everything else as raw UTF-8
        }
    }
    // ...
}
```

### Reading the checkpoint

The reader enforces the safetensors rules before trusting any offset: the header length fits in the file, there are no duplicate names, the tensors tile the data buffer with no gap, overlap, or trailing byte, and `bigram.weight` is F32 `[256, 256]`. The final slice adds `data_start`, which is the whole data-offset lesson in one expression.

```rust
pub fn read_bigram_weight(bytes: &[u8]) -> Result<Vec<f32>, String> {
    // ...
    let n = u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes"));
    if n > 100_000_000 || 8 + n > bytes.len() as u64 {        // a huge N cannot make us read past the file
        return Err(format!("header length {n} runs past the end of the file ({} bytes)", bytes.len()));
    }
    let data_start = 8 + n as usize;
    // ... parse the header; collect every tensor's [begin, end) ...
    spans.sort();
    let mut at = 0;
    for (b, e) in &spans {
        if *b != at || e < b {                                 // a gap or an overlap
            return Err(format!("tensors do not tile the data buffer at offset {at}"));
        }
        at = *e;
    }
    if data_start + at != bytes.len() {                        // trailing bytes: truncated or concatenated
        return Err(format!("{} bytes after the last tensor", bytes.len() as i64 - (data_start + at) as i64));
    }
    // ... dtype must be "F32", shape [256, 256], span 256 * 256 * 4 bytes ...
    // data_offsets count from the start of the data buffer, not of the file.
    let raw = &bytes[data_start + begin..data_start + end];
    Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}
```

`Bigram::load` checks `config.json` first (`tl_arch` must be `"bigram"`, `tl_tokenizer` `"bytes"`, `vocab_size` 256), so a Llama directory fails at startup with a clear message. Then `logits` is the row lookup:

```rust
    pub fn logits(&self, prev: u8) -> Result<Vec<f32>, String> {
        let start = prev as usize * VOCAB;            // row prev of a row-major [256, 256]
        Ok(self.weight[start..start + VOCAB].to_vec()) // == onehot(prev) @ W, bit for bit
    }
```

### PCG32

The output permutes the *old* state while the state advances, and seeding follows `pcg32_srandom_r` exactly so the stream matches every other implementation.

```rust
    pub fn new(seed: u64, seq: u64) -> Pcg32 {
        let mut r = Pcg32 { state: 0, inc: (seq << 1) | 1 }; // the increment must be odd
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc); // mod 2^64
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;                      // top 5 bits pick the rotation
        xorshifted.rotate_right(rot)
    }
    pub fn uniform(&mut self) -> f64 {                     // 27 + 26 = 53 bits: every f64 in [0, 1) step
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0) // (a * 2^26 + b) * 2^-53
    }
```

### Sampling

Greedy is a forward scan with strict `>` and NaN skipped. Sampling divides by $T$ (never multiplies), subtracts the maximum, and walks the unnormalized cumulative weights against `u * total`, which is the same as normalizing first but saves a pass.

```rust
pub fn sample(logits: &[f32], temperature: f64, rng: &mut Pcg32) -> u8 {
    if temperature == 0.0 {
        let mut best: Option<usize> = None;
        for (i, &l) in logits.iter().enumerate() {
            if !l.is_nan() && best.is_none_or(|b| l > logits[b]) { // strict >: ties to the lowest id
                best = Some(i);
            }
        }
        return best.unwrap_or(0) as u8;
    }
    let z: Vec<f64> = logits.iter().map(|&l| l as f64 / temperature).collect();   // divide by T, in f64
    let m = z.iter().copied().filter(|x| !x.is_nan()).fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = z.iter().map(|&x| if x.is_nan() || m == f64::NEG_INFINITY { 0.0 } else { (x - m).exp() }).collect();
    let total: f64 = w.iter().sum();
    let u = rng.uniform() * total;              // one draw per token
    let mut cum = 0.0;
    let mut last = 0;
    for (i, &wi) in w.iter().enumerate() {
        if wi > 0.0 {                           // a NaN (weight 0) never owns an interval
            cum += wi;
            last = i;
            if cum > u { return i as u8; }      // inverse CDF: first j with cumulative > u
        }
    }
    last as u8                                  // rounding left u at the very top: the last live id
}
```

In the temperature example, the weights are $1$ and $0.1111$; with $u = 0.3$ scaled by the total $1.1111$ to $0.333$, the walk stops at $A$.

### Incremental UTF-8

`from_utf8` either succeeds or reports how many bytes are valid and whether the error is an invalid sequence (`Some(len)`) or a tail that might still complete (`None`). The loop handles a pending buffer that contains a valid prefix, then an invalid run, then more bytes.

```rust
    pub fn push(&mut self, byte: u8) -> String {
        self.pending.push(byte);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => { out.push_str(s); self.pending.clear(); return out; }
                Err(e) => {
                    let good = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..good]).expect("valid prefix"));
                    match e.error_len() {
                        None => { self.pending.drain(..good); return out; } // C3: wait for more
                        Some(bad) => {                                      // FF: one U+FFFD per maximal subpart
                            out.push('\u{FFFD}');
                            self.pending.drain(..good + bad);
                        }
                    }
                }
            }
        }
    }
```

### The streaming loop

The head is written with no `Content-Length` (the stream ends when the connection closes). Each token becomes exactly one event; the last token's chunk also flushes the decoder and carries `"length"`. A failed write means the client left, so generation stops.

```rust
    let seed = creq.seed.unwrap_or_else(|| random_u64() >> 1);
    let mut rng = Pcg32::new(seed, 54);          // a fresh generator per request
    let mut prev = *prompt.last().expect("prompt is non-empty"); // generation starts from the last byte
    // ... non-stream branch: generate all ids, String::from_utf8_lossy, one JSON body ...
    if write_head(out, 200, "text/event-stream", None, extra).is_err() { return outcome; }
    let mut decoder = Utf8Stream::new();
    for i in 0..creq.max_tokens {
        let tok = match next_token(&server.model, prev, creq.temperature, &mut rng) {
            Ok(t) => t,
            Err(e) => {
                // status is already 200: report in-band, never truncate silently
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
        if last { text.push_str(&decoder.finish()); }  // held-back bytes go out with the last chunk
        let chunk = completion_json(&id, created, &creq.model, &text, if last { Some("length") } else { None }, None);
        if out.write_all(sse_event(&chunk).as_bytes()).is_err() { return outcome; } // client gone
    }
    let _ = out.write_all(sse_event("[DONE]").as_bytes());
```

`sse_event` is `format!("data: {payload}\n\n")`: the blank line is what ends an event. `write_json` sets `Content-Length` from `body.len()`, which in Rust is the byte length of the UTF-8 string.

### The span after the answer

`handle_connection` sets `TCP_NODELAY` (so each small event leaves immediately), answers, then flushes and shuts the socket down *before* building the span:

```rust
    let _ = out.flush();
    let _ = out.shutdown(Shutdown::Both);
    drop(out);                                   // the client has its answer now
    let (Some(o), Some(endpoint)) = (outcome, server.otlp_endpoint.as_deref()) else { return; };
    let (trace_id, parent) = match req.header("traceparent").and_then(parse_traceparent) {
        Some((t, p)) => (t, Some(p)),            // join the caller's trace
        None => (random_hex(16), None),          // malformed or absent: a new trace
    };
    // ... Span { span_id: random_hex(8), name: "POST /v1/completions", attributes, ... } ...
    if let Err(e) = export_span(endpoint, &otlp_json(&server.service_name, &span)) {
        eprintln!("tl-serve: span export failed: {e}");   // export failure never fails a request
    }
```

`parse_traceparent` accepts only lower-case hex of the right lengths, rejects version `ff` and all-zero ids, and returns `None` otherwise, so a malformed header starts a fresh trace instead of propagating invalid ids.

### The entry point

[`main.rs`](../../../../course/solutions/entry/rust/crates/tl-serve/src/main.rs) parses the three flags (exit 2 on a usage error), loads the `Server` (exit 1 if the model is bad), binds both ports, serves the health listener on its own thread, starts a SIGTERM watcher that exits 0, and calls `http::serve` on the API listener. `serve` is `accept` in a loop with `thread::spawn` per connection.

The full solution is [`http.rs`](../../../../course/solutions/rust/crates/tl-serve/src/http.rs) and [`lib.rs`](../../../../course/solutions/rust/crates/tl-serve/src/lib.rs). The latter is the final crate root, which lists the modules later sections add; this section needs only `pub mod http;`.

### Why this design

Each token costs a 1 KiB row copy, a 256-entry softmax, and one SSE write: microseconds. Each request costs one thread, and memory is the 256 KiB model plus a small per-connection buffer.

Why std only, rather than hyper and serde? Because every byte of the protocol is then code you can read, which is the point of a tracer. The price is a minimal parser with no keep-alive and no chunked bodies; [the tokio server](06-openai-server-on-tokio.md) replaces it with tokio and hyper when concurrency and the v1 API need them. For the same reason the engine uses a thread per connection rather than an event loop: at tracer scale it is the simplest design in which one slow client cannot stall another, and its weakness (a thread per open socket) matters only at high concurrency, which the tokio server handles.

Copying the row rather than multiplying by a one-hot vector gives the same bits: the product is exact but costs $256^2$ multiply-adds to compute 256 numbers. And exporting the span synchronously after closing the socket, instead of through a background batcher, keeps the code small while still never delaying the client. The cost is a thread busy for up to 2 seconds on a slow collector; [Serving metrics and tracing](08-serving-metrics-and-tracing.md) moves to the OpenTelemetry SDK's batching exporter.

## What can go wrong

A correct server handles every case in the following table. Each row names how the case is checked and why it matters to the layers that depend on this engine.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example over the wire | unit | `a` greedy 3 is `bcd` with every Completion field | the worked example over the wire |
| logits select the weight row | unit | `logits(prev)` returns the corresponding row | each generated byte uses the same checkpoint semantics |
| request line, headers, body | unit | method, path without query, trimmed header value, body | the gateway forwards these requests |
| case-insensitive header names | boundary | `content-length` works | Go's HTTP client and curl spell headers differently |
| malformed requests | boundary | 400, 411, 413, 501, and 0 for an empty connection | bad input is answered, never a panic |
| JSON parsing | unit | nesting, `Int` vs `Float`, a surrogate pair, six invalid texts | requests, `config.json`, and the safetensors header |
| JSON string escaping | boundary | `"`, `\`, newline, U+0001 escaped; round trip | a generated byte can be any byte |
| completion request validation | boundary | defaults, unknown fields ignored, ten bad bodies each naming `param` | the v0 400 contract |
| a hand-written checkpoint | unit | the succ model's row 97 has 2.0 at 98, row 255 wraps to 0 | the engine serves exactly what the bigram trainer wrote |
| bad checkpoints | boundary | F16, wrong shape, short or long data, huge header, llama config, missing `tl_tokenizer` | a bad model fails at start, not mid-request |
| the PCG32 reference stream | golden | the first six outputs of `pcg32_srandom_r(42, 54)` | one RNG across languages (D10) |
| greedy ties | boundary | ties to 10 not 20; all-equal gives 0; NaN skipped | greedy streams are identical everywhere (D11) |
| sampling skips NaN logits | boundary | 200 draws never return a NaN id | a broken weight never becomes a token |
| temperature sampling matches the softmax | statistical | 2000 draws fit 0.9 : 0.1 at T = 0.5 (chi-square < 10.83) | the sampler implements the formula, not an approximation |
| incremental UTF-8 | boundary | `C3 A9 FF` gives `""`, `"é"`, U+FFFD; `E2 82` then finish gives one U+FFFD | the incremental rule of `formats/tokenizer.md` |
| exact stream framing | conformance | event framing, one `id`, `null` then `"length"`, `[DONE]`, no `Content-Length` | the gateway and every SSE client parse these bytes |
| `max_tokens` counts generated tokens only | conformance | usage 6, 4, 10 for `héllo`; four chunks | usage is what the gateway's ledger bills later |
| stream equals non-stream at $T = 0$ | differential | chunk texts concatenate to the plain text, NULs escaped | the v0 rule that a stream and a plain answer agree |
| a seed reproduces sampling | property | the same seed twice gives the same text; another seed differs | the v0 Determinism rule |
| errors in the OpenAI shape | conformance | 400 with `param`, 404, 405, all in the error shape | OpenAI clients show the message |
| `/healthz` | conformance | `GET /healthz` is 200 | the runner and Kubernetes wait on it |
| `Content-Length` counts bytes | boundary | `Content-Length` equals the body's byte length with multi-byte text | a short count truncates the JSON |
| `X-Request-Id` echo | unit | echoed when sent, absent otherwise | one id finds the request in both logs |
| a child span of `traceparent` | conformance | trace id and parent from the header, a new span id, kind 2, status attribute, `service.name` | the trace view sees gateway and engine as one trace |
| a new trace without `traceparent` | boundary | invalid headers rejected; a new non-zero trace id and no parent | a broken header never poisons a trace |
| a stalled client | fault | a half-sent request does not block `/healthz` | a slow client never takes the engine down |

Most of the mistakes people make here produce output that looks plausible, which is what makes them dangerous. The most common is reading `data_offsets` from the start of the file: the logits are garbage and greedy text is noise, which the solution avoids by slicing at `data_start + begin`. Close behind is selecting column `prev` instead of row `prev` (the transposed product), so that after `a` you get `` ` ``; `logits` copies `weight[prev * 256 ..][..256]`.

> **Warning:** Multiplying the logits by $T$ instead of dividing inverts temperature: high temperatures become sharper and low ones flatter. The solution computes `l as f64 / temperature`.

The remaining pitfalls are compact enough for a table:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| `>=` in the argmax | ties go to the highest id; the engine and the Python CLI disagree on greedy text | strict `l > logits[b]` |
| Writing control bytes raw in JSON | a generated newline or NUL breaks the chunk; the client stops the stream | `json_string` escapes everything below U+0020 |
| Decoding each byte on its own | every multi-byte character streams as two or three U+FFFD | `Utf8Stream` holds back incomplete tails |
| Comparing header names exactly | requests from some clients have no body (411) | `Request::header` uses `eq_ignore_ascii_case` |
| Ending an SSE event with one `\n` | clients merge every chunk into one event | `sse_event` ends with `\n\n` |
| Accepting bytes after the last tensor | a truncated or concatenated file loads without complaint | `data_start + at != bytes.len()` is an error |
| Counting the prompt in characters | `usage.prompt_tokens` disagrees with the byte tokenizer | `prompt.as_bytes().len()` |
| `Content-Length` in characters | multi-byte answers are cut off by the client | `body.len()`, the UTF-8 byte length |
| Dropping the `traceparent` parent | the engine shows up as its own trace; Jaeger draws two disconnected trees | `parent_span_id` from `parse_traceparent` |
| Echoing a malformed `traceparent` instead of starting a new trace | spans with all-zero ids that backends discard | `parse_traceparent` rejects bad hex and zero ids |
| One thread for all connections | one slow client stalls everyone, health checks included | `thread::spawn` per accepted connection |

## Where this leads

This engine sits on the checkpoint `config.json` and `bigram.weight` in `model.safetensors` from [Byte bigram](../01-foundations/01-byte-bigram.md), and on `TcpListener`, threads, `Result`, HTTP/1.1 framing, JSON, SSE, and flushing from the Rust and HTTP primers. Many layers build on it directly. [The tracer gateway](../../ai-platform-engineering/12-gateway/01-streaming-proxy.md) uses it as its upstream: it checks the key and relays the engine's SSE byte for byte. [The tracer deploy](../../infrastructure/01-containers-kubernetes/01-tracer-deploy.md) packages the engine binary in a container image behind a Helm chart on kind. In [One trace](../../systems/04-observability/01-one-trace.md), the exported span appears as the child of the gateway's `gateway.proxy` span in Jaeger.

Within this chapter, [the Candle model runner](02-model-runner-and-sampler.md) adds a Llama runner and a Rust-owned KV cache while the bigram still serves; [the tokio server](06-openai-server-on-tokio.md) takes `http.rs` over with tokio and hyper, `runtime.toml`, chat completions, admission control, and the v1 API; and [Serving metrics and tracing](08-serving-metrics-and-tracing.md) replaces the hand-written OTLP JSON with the OpenTelemetry SDK and adds metrics.

Each simplification here has a production counterpart worth studying:

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a thread per connection, one request each | [vLLM's OpenAI server](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai), [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) | async I/O, keep-alive, continuous batching of many requests into one forward pass | `vllm/entrypoints/openai/api_server.py`; `tools/server/server.cpp` |
| hand-written HTTP and JSON | [hyper](https://hyper.rs/), [axum](https://github.com/tokio-rs/axum), [serde_json](https://docs.rs/serde_json/) | HTTP/2, fuzzed parsers, typed request structs | [the tokio server](06-openai-server-on-tokio.md) |
| one hand-written span | [OpenTelemetry Rust](https://github.com/open-telemetry/opentelemetry-rust) with `tracing-opentelemetry` | batching exporters, context propagation, metrics and logs | [Serving metrics and tracing](08-serving-metrics-and-tracing.md) |
| reading the whole file | memory-mapping safetensors ([safetensors](https://github.com/huggingface/safetensors), [candle](https://github.com/huggingface/candle)) | zero-copy loading of multi-gigabyte weights | [the Candle model runner](02-model-runner-and-sampler.md) |
| a byte bigram | [Hugging Face TGI](https://github.com/huggingface/text-generation-inference), [SGLang](https://github.com/sgl-project/sglang) | real tokenizers, chat templates, tool calls, quantized weights | the rest of this chapter, through [tool calls and constrained decoding](10-tool-calls-and-constrained-decoding.md) |

### Further reading

- [RFC 9112](https://www.rfc-editor.org/rfc/rfc9112), HTTP/1.1 message syntax (free).
- [W3C Trace Context](https://www.w3.org/TR/trace-context/) (free).
- [OTLP specification, JSON encoding](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding) (free).
- [PCG, A Family of Better Random Number Generators](https://www.pcg-random.org/) (free).

## Key terms

- **one-hot product**: multiplying a one-hot row vector $e_i$ by $W$, which selects row $i$ exactly.
- **inverse CDF sampling**: drawing $u \in [0,1)$ and returning the first id whose cumulative probability exceeds $u$.
- **PCG32**: a permuted congruential generator with 64-bit state and an odd stream increment, producing 32-bit outputs from the old state.
- **maximal subpart**: the longest prefix of a valid UTF-8 sequence; an invalid run is replaced by one U+FFFD per maximal subpart.
- **data buffer**: the part of a safetensors file after the 8-byte length and the JSON header, from which `data_offsets` are counted.
- **Server-Sent Events (SSE)**: a `text/event-stream` body of `data: ...` events, each terminated by a blank line.
- **span**: one operation in a trace, with trace id, span id, parent span id, name, kind, times, and attributes.
- **`traceparent`**: the W3C header carrying the caller's trace id and span id.

## Summary

- The engine reads exactly the files the bigram trainer writes: `bigram.weight` sits at an offset counted from the start of the safetensors **data buffer**, not the file.
- The next token's logits are row `prev` of the stored weight matrix, selected by `Bigram::logits`.
- Greedy decoding takes the largest logit with ties to the lowest id; sampling at temperature $T$ draws from $\mathrm{softmax}(z/T)$ with one uniform from a seeded PCG32, so a seed reproduces a stream.
- One token is one byte, so a stream must hold back an incomplete UTF-8 sequence: the chunk for `0xC3` is `""` and the chunk for `0xA9` is `"é"`.
- Once a stream's first byte is sent the status is fixed, so later errors travel as SSE events.
- One request, one server span: with a `traceparent` header the engine's span joins the caller's trace, and it is posted as OTLP/HTTP JSON after the client is answered.

## Review questions

1. Why must `Utf8Stream::finish` be called on the last token, and what would happen without it?
   <details><summary>Answer</summary>A stream that ends mid-character leaves bytes pending. Without flushing them (with replacement) into the last chunk, the concatenated stream would be missing characters that the non-streamed answer, decoded with `from_utf8_lossy`, contains.</details>
2. Why does `sample` compare `cum > u` against `u * total` instead of normalizing the weights?
   <details><summary>Answer</summary>Scaling $u$ by the total is the same inverse-CDF test as dividing every weight by the total, with one multiplication instead of 256 divisions. Using strict `>` means each id owns a half-open interval of length $p_j$.</details>
3. Why is the error after the first streamed byte sent as an SSE event and not as a 500 status?
   <details><summary>Answer</summary>The status line `HTTP/1.1 200` has already been sent; HTTP has no way to change it. Writing an error event lets the client see the failure instead of a stream that silently stops.</details>
4. What goes wrong if `read_bigram_weight` slices `bytes[begin..end]` instead of `bytes[data_start + begin..]`?
   <details><summary>Answer</summary>Offsets count from the data buffer, which starts at $8 + N$. Without the shift, every weight is read $8 + N$ bytes early, giving a different and meaningless matrix; greedy text becomes noise.</details>
5. Why does the solution create a new `Pcg32` per request instead of sharing one across the server?
   <details><summary>Answer</summary>A shared generator would make each request's stream depend on how many draws other requests made before it, so the same `seed` would not reproduce the same text. A fresh generator seeded with $(\text{seed}, 54)$ makes the output a function of the request alone.</details>
6. In the `succ` model with $N = 120$, at which file offset does $W_{97,98}$ live, and why?
   <details><summary>Answer</summary>At 99848: the element index is $256 \cdot 97 + 98 = 24930$, its data offset is $4 \cdot 24930 = 99720$, and the data buffer starts at $8 + 120 = 128$, so the file offset is $128 + 99720$.</details>
