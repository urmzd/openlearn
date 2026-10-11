<!-- ol:module L10.0 -->
# Your first endpoint: a std-only Rust HTTP/1.1 + SSE server

## Overview

| | |
|---|---|
| **Module** | `L10.0` · build · Rust · Pass 1 · 8 to 12 h |
| **You build** | `rust/crates/tl-serve/src/http.rs` (v0): the tracer engine (HTTP/1.1 parser, JSON, the byte bigram row lookup, sampling, incremental UTF-8, SSE, OTLP span export) and its crate root `lib.rs` · your entry point `rust/crates/tl-serve/src/main.rs` |
| **Contract** | HTTP: [`openapi/openai-subset.v0.yaml`](../../../course/contracts/openapi/openai-subset.v0.yaml) · the `engine` role: [`spec/cli-roles.md`](../../../course/contracts/spec/cli-roles.md) · the model directory: [`formats/safetensors.md`](../../../course/contracts/formats/safetensors.md), [`formats/config.schema.json`](../../../course/contracts/formats/config.schema.json), [`formats/tokenizer.md`](../../../course/contracts/formats/tokenizer.md) |
| **Tests** | `course/tests/rust/l10_0.rs`, 29 tests (what they check: section 4); conformance `ol conform openapi:v0:engine` from `MS-P1` |
| **Needs** | `L0.0` the checkpoint you serve ([chapter](../p00-foundations/00-byte-bigram.md)) · reading: `lang.04` Rust ([primer](../../../software-craftsmanship/12-language-and-tool-primers/04-rust.md)), `lang.05` HTTP, JSON, SSE ([primer](../../../software-craftsmanship/12-language-and-tool-primers/05-http-and-sse.md)) · or `--ref-deps` |
| **Used by** | `dep.00` builds it into the engine image · `obs.00` reads its spans; `gw.00` proxies to it over HTTP; `L10.1` adds Candle model math and `L10.5` takes `http.rs` over |
| **Milestone** | `MS-P1` (the tracer: every layer is yours and runs end to end) |
| **Optional depth** | [RFC 9112](https://www.rfc-editor.org/rfc/rfc9112) (free); [W3C Trace Context](https://www.w3.org/TR/trace-context/) (free); [OTLP specification, JSON encoding](https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding) (free); [PCG, A Family of Better Random Number Generators](https://www.pcg-random.org/) (free) |

## Key Takeaways

- Your engine reads exactly the files `L0.0` writes: `bigram.weight` sits at an offset counted from the start of the safetensors **data buffer**, not the file (`loads_hand_written_checkpoint`).
- The next token's logits are row `prev` of the stored weight matrix, selected by `Bigram::logits` (`bigram_logits_select_the_weight_row`).
- Greedy decoding takes the largest logit with ties to the lowest id; sampling at temperature $T$ draws from $\mathrm{softmax}(z/T)$ with one uniform from a seeded PCG32, so a seed reproduces a stream (`greedy_ties_go_to_lowest_id`, `temperature_sampling_matches_softmax`, `seed_reproduces_sampling`).
- One token is one byte, so a stream must hold back an incomplete UTF-8 sequence: the chunk for `0xC3` is `""` and the chunk for `0xA9` is `"é"` (`utf8_stream_holds_back_incomplete_sequences`).
- One request, one server span: with a `traceparent` header the engine's span joins the caller's trace, and it is posted as OTLP/HTTP JSON after the client is answered (`exports_child_span_of_traceparent`).

## How to work this chapter

```bash
ol start L10.0               # stubs http.rs and lib.rs; writes rust/Cargo.toml and the crate manifest
ol tests L10.0               # read the test catalog first: rung R0, you write no tests here
ol check L10.0               # exit code is the verdict
ol check L10.0 --ref-deps    # only if M03.1, rt.01, or L0.0 is not passing yet
ol diff  L10.0               # after passing: your code against the reference
```

`ol start` writes the `tl-serve` workspace member and manifest only if they are absent, and never rewrites them. The binary `src/main.rs` is yours (D16): the course ships no server. To run your engine yourself:

```bash
cargo build --release --manifest-path rust/Cargo.toml
uv run --project python python python/tinyllm/__main__.py train bigram --data some.txt --out artifacts/bigram
rust/target/release/tl-serve --model-dir artifacts/bigram --port 8000 --health-port 9464
curl -N localhost:8000/v1/completions -d '{"model":"tracer","prompt":"Once","max_tokens":32,"temperature":0,"stream":true}'
```

and declare it in `system.toml` for `MS-P1`: `[entry].engine = ["rust/target/release/tl-serve", "--model-dir", "<your trained dir>", "--port", "{port}", "--health-port", "{health_port}"]`, with `[services.engine].health = "http://127.0.0.1:{health_port}/healthz"`.

---

## 1. Why now

After `L0.0` your system has a trained byte bigram on disk and a Python CLI that can sample from it, and nothing else can reach it: no process listens on a port, so the gateway you write next (`gw.00`) has no upstream, the cluster (`dep.00`) has no image to run, and there is no request to trace (`obs.00`). Every later layer of the course is a client of an inference server, from the load generator to the agent SDK. This module writes that server: a Rust process that speaks the smallest OpenAI-compatible API (v0: `POST /v1/completions`, plain or streamed), serves the checkpoint `L0.0` wrote, computes each token's logits by selecting the matching row in the checkpoint, and reports each request as a span. It uses only Rust's standard library, so the HTTP framing, the JSON, the decoding, and the trace export are all code you can read.

## 2. Principles

### 2.1 The request path

```text
TCP bytes --read_request--> Request --parse_completion--> CompletionRequest
   --for each new token: Bigram::logits (row lookup) then sample--> byte id
   --Utf8Stream--> text --JSON body, or one SSE chunk per token--> TCP bytes
   (connection closed) --otlp_json + export_span--> POST <collector>/v1/traces
```

Each connection carries one request and gets one thread (`lang.04`, `lang.05`). The model, loaded once at start, is shared read-only by every thread through an `Arc` (a reference-counted pointer that many threads may hold).

### 2.2 The model and the one-hot product

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

A bigram model predicts the next byte from the previous byte only. Its logits after byte $x_t$ are row $x_t$ of $W$, which the engine computes as a product:

$$z = e_{x_t} W, \qquad z_j = \sum_{k} [k = x_t]\, W_{kj} = W_{x_t j}.$$

Every term but one is an exact zero, so the product equals the row bit for bit. The endpoint directly copies row $x_t$ from the row-major weight array. This avoids a matrix operation for a one-hot vector and preserves the checkpoint values exactly.

The **prompt** is encoded by the byte tokenizer (its ids are its UTF-8 bytes), and generation starts from its last byte, so the prompt must be non-empty (the v0 contract requires it).

### 2.3 Choosing the next token

**Greedy** ($T = 0$): take $\arg\max_j z_j$, and when several ids share the maximum, the **lowest** id (decision D11). Fixing the tie rule makes the stream a function of the weights alone, so your Python CLI, your engine, and anyone else's reach the same bytes.

**Sampling** ($T > 0$): divide by the temperature, subtract the maximum (so no exponent overflows), exponentiate, and normalize, in `f64`:

$$p_j = \frac{\exp\big((z_j - m)/T\big)}{\sum_k \exp\big((z_k - m)/T\big)}, \qquad m = \max_k z_k.$$

Low $T$ sharpens the distribution toward greedy; $T = 1$ is the model's own distribution. Then draw one $u$ and walk the **cumulative distribution**: return the first $j$ with $p_0 + \dots + p_j > u$ (the **inverse CDF** method: each $j$ owns an interval of $[0, 1)$ as long as $p_j$).

The uniform comes from **PCG32** (PCG-XSH-RR 64/32, decision D10), a generator whose whole state is two 64-bit integers, $s$ (state) and $c$ (an odd increment that selects the stream):

$$s' = s \cdot 6364136223846793005 + c \pmod{2^{64}}, \qquad \text{out} = \mathrm{rotr}_{32}\big((((s \gg 18) \oplus s) \gg 27) \bmod 2^{32},\; s \gg 59\big),$$

where the output is computed from the **old** state. Seeding with $(\text{seed}, \text{seq})$ follows O'Neill's reference: $c = 2\,\text{seq} + 1$, $s = 0$, one step, $s \mathrel{+}= \text{seed}$, one step. A uniform with 53 random bits takes two outputs $a, b$: $u = \big((a \gg 5)\, 2^{26} + (b \gg 6)\big)\, 2^{-53}$. The engine seeds a fresh generator per request with $(\text{seed}, 54)$, so the same `seed` on the same engine gives the same text (the v0 Determinism rule); without a `seed` it picks one at random.

### 2.4 Bytes to text, one token at a time

UTF-8 writes a character as 1 to 4 bytes. The first byte says how many: `0xxxxxxx` is a whole character, `110xxxxx` starts a 2-byte one, `1110xxxx` a 3-byte one, `11110xxx` a 4-byte one, and each following byte is `10xxxxxx`. Bytes `C0`, `C1`, and `F5` to `FF` never appear.

A byte model emits one byte per token, so the text a token completes may be empty (the first byte of `é`), one character, or a replacement character U+FFFD for an invalid sequence. `Utf8Stream` keeps the pending bytes and, after each new byte, emits the longest valid prefix; an invalid sequence becomes one U+FFFD for its **maximal subpart** (the longest prefix of a valid sequence); a tail that could still complete stays pending. Rust's `std::str::from_utf8` reports exactly the two numbers this needs: `valid_up_to()` and `error_len()`, which is `None` for "incomplete, wait". At the end, whatever is still pending is decoded with replacement (`String::from_utf8_lossy`). The concatenated chunk texts then equal the non-streamed text, which conformance case `v0.stream.equals_nonstream` checks.

### 2.5 Reading the checkpoint

A model directory holds `config.json` and `model.safetensors` (`formats/safetensors.md`). The tracer engine accepts only `"tl_arch": "bigram"`, `"tl_tokenizer": "bytes"`, `"vocab_size": 256`. The safetensors file is

```text
offset 0      u64 little-endian N, the header length
offset 8      N bytes of JSON header, padded with spaces to a multiple of 8
offset 8 + N  the data buffer
```

and each header entry's `data_offsets` is `[begin, end)` **relative to the data buffer**. The reader rejects a header length past the end of the file, a dtype other than `F32` (contract v0), a shape other than `[256, 256]`, duplicate names, and tensors that do not tile the data buffer exactly (gaps, overlaps, or trailing bytes). Weights are little-endian `f32`: `f32::from_le_bytes`.

### 2.6 The model boundary

The endpoint reads the checkpoint into a Rust `Bigram` value. `Bigram::from_weight` checks that the row-major matrix has exactly 256 × 256 values, and `Bigram::logits(prev)` returns the row for the previous byte. The first endpoint uses Rust's standard library only; later modules use Candle for tensor operations in the Llama runner.

### 2.7 Serving API v0

`POST /v1/completions` takes `{model, prompt, max_tokens, temperature, seed, stream}` (`openai-subset.v0.yaml`): `model` and `prompt` non-empty strings, `max_tokens` an integer in 1 to 4096 (default 16), `temperature` a number in 0 to 2 (default 1), `seed` a non-negative integer, `stream` a boolean; unknown fields are ignored. Any violation is `400` with the error shape `{"error": {"message", "type": "invalid_request_error", "param": <the field>, "code": null}}`.

Without `stream`, the answer is one JSON `Completion` with `Content-Length` (in bytes) and `usage` (`prompt_tokens` is the prompt's byte length, `completion_tokens` the number generated). With `stream: true` it is `text/event-stream`: one `data: <CompletionChunk>` event per generated token, all with the same `id`, `finish_reason: null` until the last chunk, which carries `"length"` and any held-back bytes, then `data: [DONE]`. Each event is written straight to the socket with `TCP_NODELAY` set, so it leaves at once (`lang.05`, section 2.6). An error after the first byte cannot change the status any more, so it is sent as a `data: {"error": ...}` event before the stream ends.

`GET /healthz` answers 200 on both the API port and the health port. Every response has `Connection: close`. An `X-Request-Id` header of 1 to 128 visible ASCII characters is echoed back.

### 2.8 One span per request

A **trace** is the tree of operations one request caused across services; each operation is a **span** with a 16-byte trace id (shared by the whole tree), an 8-byte span id, its parent's span id, a name, a kind, start and end times, and attributes. The W3C header `traceparent: 00-<trace id, 32 hex>-<parent span id, 16 hex>-<flags, 2 hex>` carries the caller's position in the tree; lower-case hex only, and all-zero ids are invalid. When it is valid, the engine's span reuses the trace id and records the caller's span as its parent; otherwise the engine starts a new trace with a random non-zero id.

When `OTEL_EXPORTER_OTLP_ENDPOINT` is set (`spec/cli-roles.md`), the engine posts each completion's span to `<endpoint>/v1/traces` as **OTLP/HTTP JSON**: `{"resourceSpans": [{"resource": {"attributes": [service.name]}, "scopeSpans": [{"scope": ..., "spans": [...]}]}]}`, with ids as lower-case hex strings, times as decimal strings of Unix nanoseconds, `kind: 2` (SERVER), and 64-bit integer attributes as strings (`{"intValue": "200"}`). It exports **after** closing the client's connection, so a slow collector never delays an answer. Jaeger accepts this format on port 4318, which is how `obs.00` draws gateway to engine as one trace.

### 2.9 Stopping

`spec/cli-roles.md` asks servers to exit 0 on SIGTERM (what Kubernetes sends before killing a pod). The tracer's `main.rs` (yours, not a graded unit) starts a thread that waits for SIGTERM and exits 0; there is nothing to drain yet. The async server later in this part drains in-flight requests first.

## 3. Worked example by hand

**The model.** The tests' `succ` model has $W_{i,(i+1) \bmod 256} = 2$ and every other entry 0: after any byte, the next byte value is the likeliest. On disk, with the metadata `{"format":"tinyllm"}`, the header text is 113 bytes, padded with 7 spaces to $N = 120$, so the data buffer starts at byte $8 + 120 = 128$. $W_{97,98}$ (after `a`, the logit of `b`) is element $256 \cdot 97 + 98 = 24930$, at data offset $4 \cdot 24930 = 99720$, file offset $128 + 99720 = 99848$, and holds the bytes `00 00 00 40` (2.0 in little-endian `f32`). Read from file offset 99720 instead and you get a different weight: that is the data-offset pitfall.

**Greedy, three tokens from `a`.** The prompt `a` is the id 97. $z = e_{97} W$ is row 97: 2 at index 98, 0 elsewhere, so the argmax is 98 (`b`). Then row 98 gives 99 (`c`), row 99 gives 100 (`d`). The answer to `{"model":"m","prompt":"a","max_tokens":3,"temperature":0}` is the text `bcd` with `finish_reason: "length"` and `usage` 1, 3, 4. This is `hand_example_completion`.

**A small matrix product.** The same row-selection rule follows from the one-hot product, though this endpoint uses a direct row lookup:

$$\begin{bmatrix}1 & 2\\ 3 & 4\end{bmatrix}\begin{bmatrix}5 & 6\\ 7 & 8\end{bmatrix} = \begin{bmatrix}1\cdot5+2\cdot7 & 1\cdot6+2\cdot8\\ 3\cdot5+4\cdot7 & 3\cdot6+4\cdot8\end{bmatrix} = \begin{bmatrix}19 & 22\\ 43 & 50\end{bmatrix}.$$

**The stream.** With `"max_tokens":2,"stream":true`, the response body is three events (the `id` and `created` values vary):

```text
data: {"id":"cmpl-9d9d...","object":"text_completion","created":1791518626,"model":"m","choices":[{"index":0,"text":"b","finish_reason":null}]}\n\n
data: {"id":"cmpl-9d9d...","object":"text_completion","created":1791518626,"model":"m","choices":[{"index":0,"text":"c","finish_reason":"length"}]}\n\n
data: [DONE]\n\n
```

**Temperature.** Suppose a row has $z_A = \ln 0.75$ and $z_B = \ln 0.25$ and every other logit is $-30$. At $T = 0.5$: $z_A / T = -0.5754$, $z_B / T = -2.7726$; subtract the maximum: $0$ and $-2.1972$; exponentiate: $1$ and $0.1111$; normalize: $p_A = 0.9$, $p_B = 0.1$ (the other ids get about $e^{-60}$). Tempering squared the odds, $0.75^2 : 0.25^2 = 9 : 1$. With $u = 0.3$ the walk stops at $A$ (cumulative 0.9 > 0.3); with $u = 0.95$ it passes $A$ and stops at $B$ (cumulative 1.0 > 0.95). `temperature_sampling_matches_softmax` draws 2000 times and expects about 1800 and 200.

**UTF-8.** The `utf8` model emits `C3 A9 FF` after `x`. `C3` is `110 00011`, the start of a 2-byte character, so the first chunk is `""`. `A9` is `10 101001`, a continuation; together they carry the bits `00011 101001` = U+00E9, so the second chunk is `"é"`. `FF` can never appear in UTF-8, so the third chunk is U+FFFD. The non-streamed text of the same request is `"é\u{FFFD}"`, the same string.

**A trace.** A request with `traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01` produces a span with `traceId` `4bf92f3577b34da6a3ce929d0e0e4736`, `parentSpanId` `00f067aa0ba902b7`, a new random `spanId`, `name` `POST /v1/completions`, `kind` 2.

## 4. The interface

The endpoint's model interface is deliberately small: `Bigram::load` validates `config.json` and `model.safetensors`, then `logits(prev)` returns a copied slice of the selected transition row. There is no foreign-function boundary in this endpoint.

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

Your `main.rs` parses `--model-dir`, `--port`, `--health-port` (exit 2 on a usage error), builds the `Server` with `Server::from_env`, binds `0.0.0.0:<port>` and `0.0.0.0:<health port>`, serves the health listener on a second thread, and calls `serve` on the API listener. It is about 70 lines.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_completion` | unit | `a` greedy 3 is `bcd` with every Completion field | the worked example over the wire |
| `bigram_logits_select_the_weight_row` | unit | `logits(prev)` returns the corresponding row | each generated byte uses the same checkpoint semantics |
| `parses_request_line_headers_and_body` | unit | method, path without query, trimmed header value, body | the gateway forwards these requests (`gw.00`) |
| `header_names_are_case_insensitive` | boundary | `content-length` works | Go's HTTP client and curl spell headers differently |
| `malformed_requests_are_rejected` | boundary | 400, 411, 413, 501, and 0 for an empty connection | bad input is answered, never a panic |
| `json_parses_nested_values_and_escapes` | unit | nesting, `Int` vs `Float`, a surrogate pair, six invalid texts | requests, `config.json`, and the safetensors header |
| `json_string_escapes_quotes_and_control_bytes` | boundary | `"`, `\`, newline, U+0001 escaped; round trip | a generated byte can be any byte |
| `completion_request_validation` | boundary | defaults, unknown fields ignored, ten bad bodies each naming `param` | the v0 400 contract |
| `loads_hand_written_checkpoint` | unit | the succ model's row 97 has 2.0 at 98, row 255 wraps to 0 | you serve exactly what `L0.0` wrote |
| `rejects_bad_checkpoints` | boundary | F16, wrong shape, short or long data, huge header, llama config, missing `tl_tokenizer` | a bad model fails at start, not mid-request |
| `pcg32_matches_reference_stream` | golden | the first six outputs of `pcg32_srandom_r(42, 54)` | one RNG across languages (D10) |
| `greedy_ties_go_to_lowest_id` | boundary | ties to 10 not 20; all-equal gives 0; NaN skipped | greedy streams are identical everywhere (D11) |
| `sampling_skips_nan_logits` | boundary | 200 draws never return a NaN id | a broken weight never becomes a token |
| `temperature_sampling_matches_softmax` | statistical | 2000 draws fit 0.9 : 0.1 at T = 0.5 (chi-square < 10.83) | the sampler implements the formula, not an approximation |
| `utf8_stream_holds_back_incomplete_sequences` | boundary | `C3 A9 FF` gives `""`, `"é"`, U+FFFD; `E2 82` then finish gives one U+FFFD | the incremental rule of `formats/tokenizer.md` |
| `stream_framing_is_exact` | conformance | event framing, one `id`, `null` then `"length"`, `[DONE]`, no `Content-Length` | `gw.00` and every SSE client parse these bytes |
| `max_tokens_counts_generated_tokens_only` | conformance | usage 6, 4, 10 for `héllo`; four chunks | usage is what the gateway's ledger bills later |
| `stream_equals_nonstream_at_temperature_0` | differential | chunk texts concatenate to the plain text, NULs escaped | conformance case `v0.stream.equals_nonstream` |
| `seed_reproduces_sampling` | property | the same seed twice gives the same text; another seed differs | conformance case `v0.seed` |
| `errors_use_the_openai_shape` | conformance | 400 with `param`, 404, 405, all in the error shape | OpenAI clients show the message |
| `healthz_answers_200` | conformance | `GET /healthz` is 200 | the runner and Kubernetes wait on it |
| `content_length_counts_bytes` | boundary | `Content-Length` equals the body's byte length with multi-byte text | a short count truncates the JSON |
| `echoes_x_request_id` | unit | `X-Request-Id` echoed when sent, absent otherwise | one id finds the request in both logs |
| `exports_child_span_of_traceparent` | conformance | trace id and parent from the header, a new span id, kind 2, status attribute, `service.name` | `obs.00` sees gateway and engine as one trace |
| `starts_a_new_trace_without_traceparent` | boundary | invalid headers rejected; a new non-zero trace id and no parent | a broken header never poisons a trace |
| `serves_a_second_client_while_one_stalls` | fault | a half-sent request does not block `/healthz` | a slow client never takes the engine down |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Reading `data_offsets` from the start of the file | garbage logits; greedy text is noise | `loads_hand_written_checkpoint` (mutant `s01`) |
| Passing `trans_b = 1` for the one-hot product | column `prev` instead of row `prev`: after `a` you get `` ` `` | `hand_example_completion` (mutant `s02`) |
| Multiplying the logits by $T$ instead of dividing | high temperatures become sharper, low ones flatter | `temperature_sampling_matches_softmax` (mutant `s03`) |
| `>=` in the argmax | ties go to the highest id; your engine and your Python disagree on greedy text | `greedy_ties_go_to_lowest_id` (mutant `s04`) |
| Writing control bytes raw in JSON | a generated newline or NUL breaks the chunk; the client stops the stream | `json_string_escapes_quotes_and_control_bytes` (mutant `s05`) |
| Decoding each byte on its own | every multi-byte character streams as two or three U+FFFD | `utf8_stream_holds_back_incomplete_sequences` (mutant `s06`) |
| Comparing header names exactly | requests from some clients have no body (411) | `header_names_are_case_insensitive` (mutant `s07`) |
| Ending an SSE event with one `\n` | clients merge every chunk into one event | `stream_framing_is_exact` (mutant `s08`) |

| Accepting bytes after the last tensor | a truncated or concatenated file loads without complaint | `rejects_bad_checkpoints` (mutant `s12`) |
| Counting the prompt in characters | `usage.prompt_tokens` disagrees with the byte tokenizer | `max_tokens_counts_generated_tokens_only` (mutant `s13`) |
| `Content-Length` in characters | multi-byte answers are cut off by the client | `content_length_counts_bytes` (mutant `s14`) |
| Dropping the `traceparent` parent | the engine shows up as its own trace; Jaeger draws two disconnected trees | `exports_child_span_of_traceparent` (mutant `s15`) |
| Echoing a malformed `traceparent` instead of starting a new trace | spans with all-zero ids that backends discard | `starts_a_new_trace_without_traceparent` |
| One thread for all connections | one slow client stalls everyone, health checks included | `serves_a_second_client_while_one_stalls` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|

| Back | `L0.0` | the checkpoint: `config.json` and `bigram.weight` in `model.safetensors` |
| Back | `lang.04` | `TcpListener`, a thread per connection, `Result` |
| Back | `lang.05` | HTTP/1.1 framing, JSON, SSE, flushing |
| Forward | `gw.00` | the gateway's upstream: it checks the key and relays your SSE byte for byte |
| Forward | `dep.00` | your engine binary in a container image, behind a Helm chart on kind |
| Forward | `obs.00` | your exported span is the child of the gateway's `gateway.proxy` span in Jaeger |
| Forward | `L10.1` | adds the Candle-backed Llama runner and Rust-owned KV cache; the bigram still serves |
| Forward | `L10.5` | takes `http.rs` over: tokio and hyper, `runtime.toml`, chat completions, admission control, the v1 API |
| Forward | `L10.7` | replaces the hand-written OTLP JSON with the OpenTelemetry SDK and adds metrics |

If you skip this module, `MS-P1` has no engine to start and `ol conform openapi:v0 --target engine` has nothing to test.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| a thread per connection, one request each | [vLLM's OpenAI server](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai), [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) | async I/O, keep-alive, continuous batching of many requests into one forward pass | `vllm/entrypoints/openai/api_server.py`; `tools/server/server.cpp` |
| hand-written HTTP and JSON | [hyper](https://hyper.rs/), [axum](https://github.com/tokio-rs/axum), [serde_json](https://docs.rs/serde_json/) | HTTP/2, fuzzed parsers, typed request structs | `L10.5` |
| one hand-written span | [OpenTelemetry Rust](https://github.com/open-telemetry/opentelemetry-rust) with `tracing-opentelemetry` | batching exporters, context propagation, metrics and logs | `L10.7` |
| reading the whole file | memory-mapping safetensors ([safetensors](https://github.com/huggingface/safetensors), [candle](https://github.com/huggingface/candle)) | zero-copy loading of multi-gigabyte weights | `L10.1` |
| a byte bigram | [Hugging Face TGI](https://github.com/huggingface/text-generation-inference), [SGLang](https://github.com/sgl-project/sglang) | real tokenizers, chat templates, tool calls, quantized weights | `L10.5` to `L10.9` |
