# tl-serve: OpenAI-compatible HTTP + SSE on tokio/hyper

## Overview

| | |
|---|---|
| **Module** | `L10.5` · build · Rust · Pass 7 · 8 to 12 h reading |
| **Solution** | [`tl-engine/src/engine.rs`](../../../../course/solutions/rust/crates/tl-engine/src/engine.rs): `Engine`, the step loop over runner, scheduler, block manager, and per-request samplers · [`tl-serve/src/server.rs`](../../../../course/solutions/history/L10.5/rust/crates/tl-serve/src/server.rs) (as of this module; [later version](../../../../course/solutions/rust/crates/tl-serve/src/server.rs)): `ServeConfig`, `spawn`, `run`, the engine thread `engine_loop`, `generate`, `stream_task`, bounded admission, abort on disconnect, drain on SIGTERM, health, readiness, `/metrics` · [`openai.rs`](../../../../course/solutions/rust/crates/tl-serve/src/openai.rs): `ApiError`, `parse_chat`, `parse_completion`, the response and chunk documents · [`sse.rs`](../../../../course/solutions/rust/crates/tl-serve/src/sse.rs): `event`, `TextStream` (incremental UTF-8, stop strings) · [`template.rs`](../../../../course/solutions/rust/crates/tl-serve/src/template.rs): `Template`, the chat-template Jinja subset · the entry point's `--config` form in [`main.rs`](../../../../course/solutions/entry/rust/crates/tl-serve/src/main.rs) |
| **Contract** | HTTP: [`openapi/openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml) (engine tier) · the `engine` role, config form: [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md) · `[engine]`: [`config/runtime.schema.json`](../../../../course/contracts/config/runtime.schema.json) · chat templates: [`formats/generation-config.schema.json`](../../../../course/contracts/formats/generation-config.schema.json) · metric names: [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml) |
| **Needs** | `L10.1` runner and sampler ([chapter](01-model-runner-and-sampler.md)) · `L10.2` scheduler ([chapter](02-continuous-batching.md)) · `L10.3` chunked prefill ([chapter](03-chunked-prefill.md)) · `L10.4` block manager ([chapter](04-block-manager-and-prefix-cache.md)) · `L10.8` speculative decoding ([chapter](08-speculative-decoding-in-the-engine.md)) · `L1.5` tl-tok: the byte tokenizer and tokenizer.json BPE ([chapter](../p01-tokenizers/05-rust-fast-bpe.md)) · reading: `lang.09` async Rust and tokio ([primer](../../../tracks/software-craftsmanship/12-language-and-tool-primers/09-async-rust-and-tokio.md)), `lang.05` HTTP and SSE, `L10.0` the tracer server, `L8.2` generate and the detokenizer |
| **Used by** | `L10.6` (disaggregation), `L10.7` (metrics and tracing), and `L10.9` (tool calls) build on this server, and `gw.04`, `ag.01`, `load.01`, and `L12.3` reach it over HTTP. This is the v1 engine that the OpenAI conformance cases, and the official `openai` Python client, run against |
| **Optional depth** | [OpenAI API reference: chat completions and streaming](https://platform.openai.com/docs/api-reference/chat) (free); [hyper 1.x guide](https://hyper.rs/guides/1/) (free); [Tokio tutorial: channels, select](https://tokio.rs/tokio/tutorial) (free); [Jinja template designer docs](https://jinja.palletsprojects.com/en/stable/templates/) (free); [vLLM's OpenAI server](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai) (free) |

## Key Takeaways

- The model runs on one OS thread and HTTP on tokio tasks; they meet in channels: a bounded admission queue in, one event channel per request out, so eight concurrent streams give the same answers as eight serial ones.
- A chat request is rendered through the model's own chat template, with the generation prompt, before tokenization, and `usage.prompt_tokens` counts that rendered prompt.
- Streaming is `data: <json>\n\n` events: the role first, text as it becomes valid UTF-8 and safe from a stop string, `finish_reason` on the last content chunk, the usage chunk when asked, then `[DONE]`.
- Every error has the v1 shape, and the status says what kind: 400 bad value, 422 unsupported value, 404 unknown model, 429 full queue with `Retry-After`.
- A client that leaves frees its KV blocks within one step, and SIGTERM drains: `/readyz` turns 503, in-flight streams finish, then exit 0.

## How to read this chapter

Sections 1 to 3 explain why the engine needs an HTTP front end, the principles that shape it (one compute thread, bounded admission, templates, streaming, errors, draining), and work one request by hand. Section 4 fixes the interface. Section 5 walks through the solution, from the engine step to the SSE task, and explains why each choice is the right one. Try to predict each step before reading it.

The solution adds `pub mod engine;` to `tl-engine/src/lib.rs` and `pub mod openai; pub mod server; pub mod sse; pub mod template;` to `tl-serve/src/lib.rs`. `tl-serve` depends on `tl-engine` and `tl-tok` by path, `tokio` (full), `hyper` (server, http1), `hyper-util` (tokio), `http-body-util`, `bytes`, `serde_json`, and `toml` (the allowed list is [`allowed-deps.toml`](../../../../course/contracts/allowed-deps.toml)). The tracer `http.rs` stays as it is: `--model-dir --port --health-port` keeps serving API v0, and `--config <runtime.toml>` starts `server::run`:

```toml
# runtime.toml
[engine]
model_dir = "artifacts/models/smol-135m/v3"
http_listen = ":8000"
health_listen = ":9464"
prefix_cache = "radix"
prefill_chunk = 512
```

```bash
rust/target/release/tl-serve --config runtime.toml
curl -N localhost:8000/v1/chat/completions -d '{"model":"smol-135m","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

---

## 1. Why now

`L10.1` to `L10.4` built an engine that batches requests, chunks prompts, and shares prefixes, and nothing outside a Rust test can use it: the tracer server (`L10.0`) still runs one bigram request per thread. Every client in the rest of the course (the gateway, the load generator, the agent SDK, the official OpenAI client) speaks the v1 API: chat messages, streaming with roles, stop sequences, error codes. This module puts the engine behind that API, on tokio and hyper, with the operational behaviors a server needs to live in a cluster: bounded admission, disconnect handling, readiness, and draining.

## 2. Principles

### 2.1 One compute thread, many I/O tasks

HTTP connections spend almost all their time waiting; tokio runs thousands of them as tasks on a few threads (`lang.09`). The model is the opposite: each step is milliseconds of pure CPU, and an `.await` inside it would block a runtime thread. So the engine runs on its own OS thread and the two sides talk through channels:

```text
handler task  --try_send(Generate)-->  bounded admission queue  -->  engine thread
                                                                      loop:
                                                                        take commands (add_request)
                                                                        drop requests whose channel closed
                                                                        Engine::step (schedule, plan, forward, sample, on_step)
handler / SSE task  <--Ev::Token-------  one unbounded channel per request  --
```

`Engine::step` is the whole of continuous batching: `schedule` (`L10.2`), `plan` (`L10.3`) over the block manager's tables (`L10.4`), `ModelRunner::forward` (`L10.1`), one `sample` per sampled row with the request's own parameters and generator `stream(seed, sample)`, then `on_step`. A token in `stop_ids` (EOS) finishes the request with `stop`.

### 2.2 The admission queue is bounded

An unbounded queue turns overload into unbounded latency: every new request waits behind all the others and times out at the client anyway. The admission channel and the scheduler's waiting queue have capacity `queue_capacity`; a request that does not fit is answered **at once** with 429 `rate_limit_exceeded` and `Retry-After: 1`, which load balancers and the gateway (`gw.04`) understand. A request admitted but kept waiting for KV blocks past `queue_deadline_ms` gets 503 `no_capacity`.

### 2.3 A chat becomes a prompt

The model was trained on conversations written in one exact text format. The model directory carries it as a template (`generation_config.json`'s `chat_template`, else `tokenizer_config.json`'s); `template.rs` renders `messages` through it with `add_generation_prompt`, the opening of an assistant turn, so the model continues as the assistant. SmolLM2's template is

```jinja
{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
```

The interpreter is a small compiler: split the source into text, `{{ expression }}`, and `{% statement %}` segments (with `-` trimming whitespace on that side), parse statements into a tree (`for`, `if` / `elif` / `else`), parse expressions by recursive descent (`or` < `and` < `not` < `==`, `!=`, `is defined` < `+`, `~` < `.name`, `[key]`, `| filter` < literals and names), and evaluate over JSON values. Syntax errors surface when the model loads, never on a request. A model with no template (the tracer bigram, the tiny test models) gets a plain default, `<|role|>\ncontent\n` per message and `<|assistant|>\n`.

The rendered prompt is tokenized with the model's tokenizer (`tl-tok`: bytes, or tokenizer.json BPE with its special tokens matched first), and `usage.prompt_tokens` counts those tokens: what the model reads and what the gateway's ledger bills.

### 2.4 Streaming text

| Symbol | Meaning | Type |
|---|---|---|
| $b_1, b_2, \dots$ | bytes of the generated tokens, in order | `u8` |
| $P$ | bytes received but not yet a complete UTF-8 sequence | bytes |
| $T$ | decoded text not yet sent | string |
| $\Sigma$ | the stop strings of the request (at most 4) | strings |

`TextStream::push` appends a token's bytes to $P$, moves the longest valid UTF-8 prefix of $P$ into $T$ (an invalid sequence becomes U+FFFD; an incomplete tail waits), then: if some $\sigma \in \Sigma$ occurs in $T$, it sends $T$ up to the first occurrence and stops; otherwise it holds back the longest suffix of $T$ that is a proper prefix of some $\sigma$ and sends the rest. So a stop string never appears in the output, even split across tokens, and the concatenation of everything sent equals the non-streamed text.

An SSE stream is the bytes `data: <ChatCompletionChunk json>\n\n` per chunk: the first with `delta.role: "assistant"`, then `delta.content` pieces (possibly empty), the last content chunk with `finish_reason` (`length`, or `stop` for EOS and stop strings), the usage chunk (`choices: []`) when `stream_options.include_usage` is set, and `data: [DONE]\n\n`. After 15 s without a token the server sends the comment `: ping\n\n`. hyper frames a body of unknown length with chunked transfer encoding; every event is written as soon as it exists.

### 2.5 Validation and errors

Every request body is checked against the v1 schema before it reaches the engine: a value out of range (`temperature` outside 0 to 2, `top_p` not in (0, 1], `max_tokens` 0, a stop list of 5) is 400 `invalid_request_error` with `param` naming the field; a supported field with a value this engine does not serve (`n` > 1, `echo`, content as an array of parts, and until `L10.9` `tools`, `tool_choice`, `response_format`) is 422 `unsupported_parameter`; an unknown model is 404 `model_not_found`; a prompt plus `max_tokens` past the context is 400 `context_length_exceeded`. Unknown fields are ignored. Every error body is `{"error": {"message", "type", "param", "code"}}` with `application/json`, and every response carries `X-Request-Id` (echoed, or generated).

### 2.6 Disconnects and draining

When a client goes away, hyper drops the response body. For a stream, the body is the receiving end of a channel fed by the request's SSE task; that task's next send fails, it returns, and it drops the request's event channel. The engine thread checks `is_closed()` on every live request before each step and aborts the closed ones: the scheduler releases their blocks within one step. A stop string ends a request the same way: the handler stops reading and drops its channel.

On SIGTERM (Kubernetes' stop signal) the server closes its API listener, answers `/readyz` with 503 so no new traffic is routed, answers new requests on open connections with 503, waits (up to 30 s) until every in-flight request has finished, stops the engine thread, and exits 0. `/healthz` stays 200 throughout: the process is alive.

### 2.7 Embeddings

`/v1/embeddings` runs each input once (`ModelRunner::run_once`, between steps on the engine thread), takes the final-normed hidden states $h_1, \dots, h_T \in \mathbb{R}^d$, and returns $e = \bar h / \lVert \bar h \rVert_2$ with $\bar h = \frac{1}{T} \sum_t h_t$. The gateway's usage-policy classifier (D33) is a linear head over exactly this vector.

## 3. Worked example by hand

The succ bigram (after byte $i$, byte $i + 1$) in a directory `succ`, with the template `{% for m in messages %}{{ m.content }}{% endfor %}`.

**Request.** `POST /v1/chat/completions` with `{"model": "succ", "messages": [{"role": "user", "content": "a"}], "max_tokens": 3, "temperature": 0}`.

1. Validation: `model` is the served id; `temperature` 0 is in range; `max_tokens` 3 plus the prompt fits the context.
2. Template: one message, so the loop emits its content: the prompt is `a`.
3. Tokenize: the byte tokenizer gives `[97]`; `prompt_tokens` = 1.
4. Engine: step 1 prefills position 0 and samples greedily from row 97: `b` (98); steps 2 and 3 decode `c`, `d`. The third token reaches `max_tokens`: finish `length`.
5. Response: `{"object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "bcd"}, "finish_reason": "length"}], "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4}, ...}`. This is `hand_example_chat_completion`.

With `"stream": true, "stream_options": {"include_usage": true}` the body is six events: role (`content: ""`), `b`, `c`, `d` with `finish_reason: "length"`, the usage chunk, `[DONE]`.

**A stop string across tokens.** With `"stop": ["de"]` and `max_tokens` 10, the tokens arrive as `b`, `c`, `d`, `e`, ... After `d` the text not yet sent is `d`, a prefix of `de`: held back. After `e` it is `de`: the stop string, so nothing more is sent and the answer is `bc` with `finish_reason: "stop"` (`stop_strings_end_generation_across_tokens`).

**The template, by hand** (`chat_template_hand_example`). SmolLM2's template with `[{"role": "user", "content": "Hi"}]`: `loop.first` is true and `messages[0]['role']` is `user`, not `system`, so the default system turn is emitted first; then `<|im_start|>user\nHi<|im_end|>\n`; then, because `add_generation_prompt` is true, `<|im_start|>assistant\n`.

## 4. The interface

```rust
// rust/crates/tl-engine/src/engine.rs
pub struct GenRequest { pub prompt: Vec<u32>, pub params: SamplingParams, pub seed: u64, pub max_tokens: usize, pub stop_ids: Vec<u32>, pub priority: i32 }
pub struct TokenOut { pub id: RequestId, pub token: u32, pub logprob: f64, pub finish: Option<FinishReason> }
impl Engine {
    pub fn new(runner: ModelRunner, cfg: &EngineConfig, max_waiting: usize) -> Engine;
    pub fn add_request(&mut self, r: GenRequest) -> Result<RequestId, AdmitError>;
    pub fn abort(&mut self, id: RequestId);  pub fn abort_all(&mut self);
    pub fn step(&mut self) -> anyhow::Result<Vec<TokenOut>>;
    pub fn has_work(&self) -> bool;  pub fn stats(&self) -> EngineStats;  pub fn max_model_len(&self) -> usize;
    pub fn runner_mut(&mut self) -> &mut ModelRunner;
}
```

```rust
// rust/crates/tl-serve/src/{server,openai,sse,template}.rs
pub struct ServeConfig { pub model_dir: PathBuf, pub model_id: String, pub http_listen: String, pub health_listen: String,
                         pub engine: EngineConfig, pub queue_capacity: usize, pub queue_deadline_ms: u64,
                         pub handle_signals: bool, pub step_delay: Duration }
impl ServeConfig { pub fn for_model(dir: &Path) -> ServeConfig;
                   pub fn from_toml(text: &str, env: &[(String, String)]) -> Result<ServeConfig, String>;
                   pub fn load(path: &Path) -> Result<ServeConfig, String>; }
pub fn spawn(cfg: ServeConfig) -> Result<ServerHandle, String>;     // binds, then serves on its own threads
impl ServerHandle { pub fn drain(&self); pub fn join(self) -> Result<(), String>; }   // pub http_addr, health_addr
pub fn run(cfg: ServeConfig) -> Result<(), String>;                 // the binary: serve until SIGTERM, drain, return
pub fn pool_embedding(hidden: &[f32], d: usize) -> Vec<f32>;  pub fn metrics_text(s: &EngineStats) -> String;

pub struct ApiError { pub status: u16, pub kind: &'static str, pub message: String, pub param: Option<String>, pub code: Option<&'static str> }
pub fn parse_chat(body: &[u8]) -> Result<GenSpec, ApiError>;   pub fn parse_completion(body: &[u8]) -> Result<GenSpec, ApiError>;
pub fn parse_embeddings(body: &[u8]) -> Result<(String, Vec<String>), ApiError>;   pub fn parse_tokenize(body: &[u8]) -> Result<(String, String, bool), ApiError>;

pub fn event(payload: &str) -> String;   pub const DONE: &str;   pub const PING: &str;
pub struct TextStream { /* pending bytes, held text, stops */ }
impl TextStream { pub fn new(stops: Vec<String>) -> Self; pub fn push(&mut self, bytes: &[u8]) -> String; pub fn finish(&mut self) -> String; pub fn stopped(&self) -> bool; }

pub struct Template { /* parsed nodes */ }
impl Template { pub fn parse(src: &str) -> Result<Template, String>;  pub fn render(&self, ctx: &serde_json::Value) -> Result<String, String>;
                pub fn render_chat(&self, messages: &Value, add_generation_prompt: bool, bos: &str, eos: &str, tools: Option<&Value>) -> Result<String, String>;
                pub fn render_messages(&self, messages: &[(&str, &str)], add_generation_prompt: bool, bos: &str, eos: &str) -> Result<String, String>; }
pub const DEFAULT_TEMPLATE: &str;
```

Routes: API port `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings`, `POST /v1/tokenize`, `GET /v1/models`, `GET /v1/models/{id}`, `GET /healthz`; health port `GET /healthz`, `GET /readyz`, `GET /metrics` (`tl_engine_kv_blocks{state}`, `tl_engine_queue_depth`, `tl_engine_active_sequences`, `tl_engine_batch_tokens`, `tl_engine_prefix_cache_hit_ratio`, `tl_engine_preemptions_total`, `tl_engine_kv_evictions_total`; `L10.7` adds the histograms and traces). `X-TL-Priority` (an integer in -100 to 100) sets the request's priority; `X-TL-KV-Handle` answers 404 `kv_handle_not_found` until `L10.6`.

Your `main.rs`: `--config <path>` loads `ServeConfig::load` (with `TL_ENGINE__<KEY>` overrides) and calls `server::run`; exit 0 after the drain.

### What a correct solution must handle

| Case | Checks | Why it matters downstream |
|---|---|---|
| The hand example | section 3: `bcd`, `length`, usage 1 + 3, every field of `chat.completion` | the worked example over the wire |
| Stream framing | content type, one id, role first, finish on the last content chunk, usage chunk, `[DONE]` | every SSE client parses this |
| Stream equals non-stream at temperature 0 | deltas concatenate to the plain content on the tiny Llama | conformance case `chat.stream.equals_nonstream` |
| Seeds reproduce sampling | the same seed repeats, another seed differs | case `chat.seed` |
| Stop strings across tokens | `de` split across tokens stops at `bc`, plain and streamed | case `chat.stop` |
| Errors in the OpenAI shape | thirteen bad requests: status, shape, `param`, `code` | cases `err.400`, `err.422`, `route` errors |
| `/v1/completions` and usage | default 16 tokens, usage | the text endpoint of v1 |
| Usage counts the templated prompt | `prompt_tokens` = tokens of the rendered default template | case `chat.usage`, the gateway's billing |
| A full queue | queue of one: the third request gets 429 with `Retry-After` at once; the queued one runs later | bounded admission |
| A disconnect | `/metrics` shows 0 active sequences and 0 used blocks within 2 s | case `cancel.disconnect` |
| A drain | `/readyz` 503, new connections refused, the open stream reaches `[DONE]` | rolling deploys on Kubernetes |
| Embeddings | each vector is the normalized mean of the hidden states, in input order | the policy classifier's input (D33) |
| Tokenize, models, health, request id | `/v1/tokenize`, `/v1/models`, health on both ports, `X-Request-Id` | the gateway's TPM cost and routing |
| Concurrent streams | eight parallel streams equal the serial answers | case `concurrency.16` |
| The chat template by hand | SmolLM2's real template, filters, `elif`, `loop.last`, whitespace control, the default template | the agent model (D20) gets the prompt it was trained on |
| Template errors at parse | unclosed blocks, unknown statements and filters refused at parse | a broken template fails at load |
| Stop-prefix hold-back | stop-prefix hold-back and incremental UTF-8, byte by byte | streams never leak a stop string |
| runtime.toml and env overrides | `[engine]` keys, defaults, `TL_ENGINE__` overrides, refused keys and values | the config form of spec/cli-roles.md |
| Speculative config | `[engine].speculative` selects the runner-backed L10.8 path; greedy output matches ordinary serving | prompt lookup can be enabled without changing greedy answers |

## 5. The solution, step by step

### 5.1 From first principles

1. **The model step is CPU-bound and long; HTTP is I/O-bound and wide.** A step costs milliseconds of matrix work, a connection mostly waits. Putting a step inside an async task blocks a tokio worker for its whole duration and stalls every connection scheduled there (section 2.1). So the step must run on a thread tokio does not own.
2. **One engine thread forces message passing.** The scheduler, block manager, and runner are mutable state with one owner. Handlers cannot call into them; they send a command (`Cmd::Generate`) and receive events (`Ev::Token`). The engine owns all state, so it needs no locks around the model.
3. **Admission must be bounded and must answer at once.** If the command channel were unbounded, overload would become queueing delay. A `sync_channel(queue_capacity)` with `try_send` turns "full" into an immediate 429 (section 2.2), and the scheduler's own `max_waiting` refuses with `QueueFull` behind it.
4. **Per-request events need per-request channels.** Each request gets its own unbounded `mpsc` sender. Unbounded is safe here because the engine produces at most one token per request per step; the producer, not the channel, sets the rate. The channel also carries cancellation for free: when the receiver is dropped, `is_closed()` turns true on the engine side.
5. **Text cannot be produced per token.** A token is bytes, possibly half a character, and a stop string may span tokens. So a stateful `TextStream` sits between tokens and output: it releases only complete UTF-8 and holds back any tail that could still become a stop string (section 2.4, the `de` example of section 3).
6. **The response kind is decided by the first event.** A request that the scheduler refuses (too long, never fits) must get an HTTP error, not a 200 stream that then fails. So `generate` waits for the first event before choosing between an error, a JSON body, and an SSE body.
7. **Draining is a count.** A rolling deploy needs "stop taking new work, finish old work". An atomic in-flight counter, incremented before admission and decremented by a guard's `Drop` when the response or its stream ends, makes "finished" observable without tracking each request.

### 5.2 The engine step

[`engine.rs`](../../../../course/solutions/rust/crates/tl-engine/src/engine.rs) composes the earlier modules. Each request keeps its own sampling state, so a request's tokens do not depend on what else is batched with it:

```rust
struct ReqState {
    params: SamplingParams,
    rng: Pcg32,          // stream(seed, PURPOSE_SAMPLE): this request's own generator
    stop_ids: Vec<u32>,  // EOS ids: finish with Stop
}

pub fn add_request(&mut self, r: GenRequest) -> Result<RequestId, AdmitError> {
    // the scheduler can refuse (QueueFull, TooLong, NeverFits); only then is
    // there no sampler to create
    let id = self.sched.add(Request { prompt: r.prompt, max_tokens: r.max_tokens, priority: r.priority })?;
    self.reqs.insert(id, ReqState { params: r.params, rng: stream(r.seed, PURPOSE_SAMPLE), stop_ids: r.stop_ids });
    Ok(id)
}
```

The step is the whole of continuous batching in about twenty lines:

```rust
pub fn step(&mut self) -> Result<Vec<TokenOut>> {
    let out = self.sched.schedule();              // L10.2: who runs, how many tokens each
    self.batch_tokens = out.num_tokens();
    if out.is_empty() {
        return Ok(Vec::new());
    }
    let mut sampled = Vec::new();
    {
        let p = plan(&self.sched, &out);          // L10.3: the mixed batch over L10.4's block tables
        let logits = self.runner.forward(&p.batch)?;   // L10.1: one forward for every row
        // only rows that end a sequence's scheduled span are sampled; a
        // prefill chunk that stops mid-prompt samples nothing
        for &(row, id) in &p.sample {
            let st = self.reqs.get_mut(&id).ok_or_else(|| anyhow!("no sampler for request {id}"))?;
            let toks = self.sched.tokens(id).ok_or_else(|| anyhow!("unknown request {id}"))?;
            let plen = self.sched.prompt_len(id).unwrap_or(toks.len());
            // prompt and generated tokens are passed separately: the
            // repetition penalties need to know which is which
            let (token, logprob) = sample(logits.row(row), &st.params, &toks[..plen], &toks[plen..], &mut st.rng);
            sampled.push((id, token, logprob, st.stop_ids.contains(&token)));
        }
    }
    let outputs: Vec<StepOutput> = sampled.iter().map(|&(id, token, _, stop)| StepOutput { id, token, stop }).collect();
    let mut finished: HashMap<RequestId, FinishReason> = HashMap::new();
    // the scheduler decides finish (EOS -> Stop, max_tokens -> Length) and frees blocks
    for ev in self.sched.on_step(&outputs) {
        if let RequestEvent::Finished { id, reason } = ev {
            finished.insert(id, reason);
            self.reqs.remove(&id);
        }
    }
    Ok(sampled.into_iter().map(|(id, token, logprob, _)| TokenOut { id, token, logprob, finish: finished.get(&id).copied() }).collect())
}
```

The finish reason comes from the scheduler, not from the server: the server never has to guess whether the last token was a length cut, which is how a length cut is never reported as `stop`.

### 5.3 The engine thread

The engine loop in [`server.rs`](../../../../course/solutions/history/L10.5/rust/crates/tl-serve/src/server.rs) has three phases per iteration: take commands, drop the dead, step.

```rust
fn engine_loop(mut engine: Engine, rx: Receiver<Cmd>, shared: Arc<Shared>, stop: Arc<AtomicBool>, deadline: Duration, step_delay: Duration, speculative: Option<SpecConfig>) {
    let mut live: HashMap<RequestId, Live> = HashMap::new();
    loop {
        // ... on `stop`: abort_all, send Ev::Failed to each live request, return
        loop {
            // with work: drain what is queued without waiting; idle: block
            // briefly so the thread does not spin
            let cmd = if engine.has_work() {
                match rx.try_recv() { Ok(c) => c, Err(_) => break }
            } else {
                match rx.recv_timeout(Duration::from_millis(20)) {
                    Ok(c) => c,
                    Err(RecvTimeoutError::Timeout) if stop.load(Ordering::SeqCst) => return,
                    Err(RecvTimeoutError::Timeout) => break,
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            };
            match cmd {
                // ... Cmd::Generate with [engine].speculative: the L10.8 path
                Cmd::Generate { req, events } => match engine.add_request(req) {
                    Ok(id) => { live.insert(id, Live { tx: events, since: Instant::now(), started: false }); }
                    Err(e) => { let _ = events.send(Ev::Rejected(admit_error(e))); }
                },
                // ... Cmd::Embed: run_once + pool_embedding, between steps
            }
        }
        // clients that left, and requests that waited too long for KV
        let mut drop_ids = Vec::new();
        for (&id, l) in &live {
            if l.tx.is_closed() {
                drop_ids.push((id, None));                 // receiver dropped: abort silently
            } else if !l.started && l.since.elapsed() > deadline {
                drop_ids.push((id, Some(ApiError::no_capacity("no KV blocks became free before the queue deadline"))));
            }
        }
        for (id, err) in drop_ids {
            engine.abort(id);                              // blocks back to the pool before the next step
            if let (Some(l), Some(e)) = (live.remove(&id), err) {
                let _ = l.tx.send(Ev::Rejected(e));
            }
        }
        if engine.has_work() {
            match engine.step() {
                Ok(outs) => {
                    for o in outs {
                        // ...
                        l.started = true;
                        let sent = l.tx.send(Ev::Token { token: o.token, logprob: o.logprob, finish: o.finish }).is_ok();
                        if o.finish.is_some() {
                            live.remove(&o.id);
                        } else if !sent {
                            engine.abort(o.id);            // the receiver went away during the step
                            live.remove(&o.id);
                        }
                    }
                }
                Err(e) => {
                    // a failed step fails every request in flight; the engine keeps serving
                    engine.abort_all();
                    for (_, l) in live.drain() { let _ = l.tx.send(Ev::Failed(e.to_string())); }
                }
            }
            // ... optional step_delay (tests slow the engine down)
        }
        if let Ok(mut s) = shared.stats.lock() { *s = engine.stats(); }   // /metrics reads this copy
    }
}
```

The `is_closed()` sweep runs before every step, so an abandoned request is aborted at most one step after its client leaves. That is the "frees its blocks within a step" guarantee, and it needs no message from the handler: dropping the receiver is the message. `/metrics` reads a copy of `EngineStats` under a mutex, so a scrape never touches the engine.

### 5.4 Admission and the first event

`generate` validates, renders, tokenizes, and admits:

```rust
async fn generate(shared: Arc<Shared>, headers: &hyper::HeaderMap, spec: GenSpec, chat: bool) -> Response<RespBody> {
    // ... unknown model -> 404, draining -> 503, X-TL-KV-Handle -> 404 (until L10.6), X-TL-Priority in [-100, 100]
    let prompt = match &spec.prompt {
        Prompt::Text(t) => t.clone(),
        Prompt::Chat(msgs) => {
            let m: Vec<Value> = msgs.iter().map(|x| serde_json::json!({"role": x.role, "content": x.content.clone().unwrap_or_default()})).collect();
            // add_generation_prompt = true: the prompt ends by opening the assistant turn
            match shared.template.render_chat(&Value::Array(m), true, &shared.bos, &shared.eos, None) {
                Ok(p) => p,
                Err(e) => return error_response(&ApiError::internal(format!("chat template: {e}"))),
            }
        }
    };
    let tokens = shared.tok.encode(&prompt);   // usage.prompt_tokens is tokens.len(): the rendered prompt
    // ... empty prompt -> 400
    let room = shared.max_model_len.saturating_sub(tokens.len());
    let max_tokens = match spec.max_tokens {
        Some(n) if n > room => return error_response(&ApiError::context_length(/* ... */)),
        Some(n) => n,
        None if room == 0 => return error_response(&ApiError::context_length(/* ... */)),
        None => room,                          // chat default: the rest of the context
    };
    // ... seed: the request's, or derived from a fresh id
    let req = GenRequest { prompt: tokens.clone(), params: spec.params.clone(), seed, max_tokens, stop_ids: shared.eos_ids.clone(), priority };
    // in flight from here until the response (or its stream) ends: a drain waits for it
    shared.inflight.fetch_add(1, Ordering::SeqCst);
    let guard = Inflight(Arc::clone(&shared));
    let (tx, mut rx) = mpsc::unbounded_channel();
    match shared.cmds.try_send(Cmd::Generate { req, events: tx }) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => return error_response(&ApiError::queue_full()),   // 429 + Retry-After: 1
        Err(TrySendError::Disconnected(_)) => return error_response(&ApiError::no_capacity("the engine stopped")),
    }
    let first = match rx.recv().await {
        Some(Ev::Rejected(e)) => return error_response(&e),   // scheduler refused, or queue deadline
        Some(Ev::Failed(m)) => return error_response(&ApiError::internal(m)),
        Some(ev) => ev,
        None => return error_response(&ApiError::internal("the engine dropped the request")),
    };
    // ... build Job; if streaming, spawn stream_task(first, rx, guard) and return an SseBody
    // ... otherwise loop over events into one chat.completion document
}
```

`try_send` is the line that makes admission bounded: it never awaits. The `Inflight` guard is created before the send, so any early return drops it and the count stays right.

`error_response` adds the one header a full queue needs:

```rust
fn error_response(e: &ApiError) -> Response<RespBody> {
    let mut r = full(StatusCode::from_u16(e.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), "application/json", e.to_json());
    if e.status == 429 {
        r.headers_mut().insert("retry-after", HeaderValue::from_static("1"));
    }
    r
}
```

Validation in [`openai.rs`](../../../../course/solutions/rust/crates/tl-serve/src/openai.rs) separates the two error classes the contract names. A value out of range is 400 with `param`; a value the engine does not serve is 422:

```rust
fn sampling(m: &Map<String, Value>) -> Result<(SamplingParams, Option<u64>, Vec<String>, bool, bool), ApiError> {
    let params = SamplingParams {
        temperature: number(m, "temperature", 1.0, 0.0, false, 2.0)?,   // [0, 2]
        top_p: number(m, "top_p", 1.0, 0.0, true, 1.0)?,                // (0, 1]: lo excluded
        // ... top_k, min_p, penalties
    };
    // ... seed, stop (a string, or 1 to 4 non-empty strings), stream, stream_options.include_usage
}

pub fn parse_chat(body: &[u8]) -> Result<GenSpec, ApiError> {
    // ... tools, tool_choice, response_format -> 422 until L10.9
    if integer(&m, "n", 1)?.unwrap_or(1) != 1 {
        return Err(ApiError::unsupported("n", "only n = 1 is supported"));   // never silently 1 choice
    }
    // ... messages: roles checked, content as an array of parts -> 422
}
```

### 5.5 Detokenizing and the SSE task

`Detok` wraps `TextStream` and turns the engine's finish into the wire's `finish_reason`:

```rust
fn token(&mut self, shared: &Shared, token: u32, logprob: f64, finish: Option<FinishReason>) -> (String, Option<Value>) {
    self.completion_tokens += 1;
    // EOS counts as a completion token but contributes no text
    let is_eos = finish == Some(FinishReason::Stop) && shared.eos_ids.contains(&token);
    let bytes = if is_eos { &[][..] } else { shared.tok.token_bytes(token).unwrap_or(&[]) };
    let mut piece = self.text.push(bytes);
    // ... logprob entry
    if self.text.stopped() {
        self.finish = Some("stop");               // a stop string: the handler stops reading
    } else if let Some(f) = finish {
        piece.push_str(&self.text.finish());      // flush what was held back
        self.finish = Some(match f { FinishReason::Stop => "stop", FinishReason::Length => "length" });
    }
    (piece, lp)
}
```

`TextStream::push` in [`sse.rs`](../../../../course/solutions/rust/crates/tl-serve/src/sse.rs) is section 2.4's algorithm:

```rust
pub fn push(&mut self, bytes: &[u8]) -> String {
    if self.stopped {
        return String::new();
    }
    self.pending.extend_from_slice(bytes);
    self.decode(false);   // complete UTF-8 into `held`; an incomplete tail stays in `pending`
    // the earliest occurrence of any stop string ends the text there
    if let Some(at) = self.stops.iter().filter_map(|s| self.held.find(s.as_str())).min() {
        self.stopped = true;
        let out = self.held[..at].to_string();
        self.held.clear();
        self.pending.clear();
        return out;
    }
    // hold back the longest tail that is a proper prefix of a stop string
    let mut keep = 0;
    for s in &self.stops {
        for (i, _) in s.char_indices().skip(1) {   // char boundaries only: never split a character
            if self.held.ends_with(&s[..i]) {
                keep = keep.max(i);
            }
        }
    }
    let cut = self.held.len() - keep;
    let out = self.held[..cut].to_string();
    self.held.drain(..cut);
    out
}
```

In the `de` example: after `d`, `held` is `d`, which equals the prefix `de[..1]`, so `keep = 1` and nothing is sent; after `e`, `held` is `de`, `find` returns 0, and the stream stops with `bc` already sent. `decode` uses `Utf8Error::error_len()`: `Some(n)` means an invalid sequence (emit U+FFFD, skip `n` bytes), `None` means an incomplete one (wait, unless this is `finish`).

The SSE task writes one event per piece and owns the guard, so the drain waits for the stream, not just the handler:

```rust
async fn stream_task(shared: Arc<Shared>, job: Job, first: Ev, mut rx: mpsc::UnboundedReceiver<Ev>, out: mpsc::UnboundedSender<Bytes>, _guard: Inflight) {
    let send = |s: String| out.send(Bytes::from(s)).is_ok();   // false: hyper dropped the body
    // ... `chunk` builds chat.completion.chunk or text_completion
    if job.chat && !send(sse::event(&chunk("", true, None, None))) {   // role first
        return;
    }
    let mut d = Detok { text: TextStream::new(job.stops.clone()), completion_tokens: 0, finish: None };
    let mut ev = Some(first);
    loop {
        let e = match ev.take() {
            Some(e) => e,
            None => match tokio::time::timeout(Duration::from_secs(15), rx.recv()).await {
                Ok(Some(e)) => e,
                // ... Ok(None): the engine dropped the request -> an error event
                Err(_) => { if !send(sse::PING.to_string()) { return; } continue; }   // keep proxies open
            },
        };
        match e {
            Ev::Token { token, logprob, finish } => {
                let (piece, lp) = d.token(&shared, token, logprob, finish);
                // ...
                if let Some(f) = d.finish {
                    if !send(sse::event(&chunk(&piece, false, Some(f), lp))) { return; }   // finish on the last content chunk
                    break;
                }
                if (!piece.is_empty() || job.logprobs) && !send(sse::event(&chunk(&piece, false, None, lp))) {
                    return;
                }
            }
            // ... Rejected / Failed: an error event, then return
        }
    }
    drop(rx);   // a stop string may end us before the engine finishes: closing rx aborts it
    if job.include_usage {
        // ... the usage chunk with choices: []
    }
    send(sse::DONE.to_string());
}
```

Every `return` on a failed `send` drops `rx`, which is how a disconnect reaches the engine's `is_closed()` sweep. The response body is `SseBody`, a `Body` whose `poll_frame` just polls the `Bytes` channel; hyper frames it with chunked encoding and writes each event as it arrives.

### 5.6 Draining

The server thread in `spawn` runs the tokio runtime and owns the shutdown sequence:

```rust
// the API listener closes when draining begins; the health listener stays
// up to answer /readyz 503 until the end
let (health_stop_tx, health_stop_rx) = watch::channel(false);
let a = tokio::spawn(accept_loop(http, Arc::clone(&shared), drain_rx.clone(), api));
let h = tokio::spawn(accept_loop(hl, Arc::clone(&shared), health_stop_rx, health));
shared.ready.store(true, Ordering::SeqCst);
// ... wait for SIGTERM (or ServerHandle::drain in tests)
shared.draining.store(true, Ordering::SeqCst);   // /readyz -> 503, new requests on open connections -> 503
let _ = a.await;                                  // API listener closed: new connections refused
let t0 = Instant::now();
while shared.inflight.load(Ordering::SeqCst) > 0 && t0.elapsed() < Duration::from_secs(30) {
    tokio::time::sleep(Duration::from_millis(10)).await;
}
let _ = health_stop_tx.send(true);
let _ = h.await;
// ... then stop.store(true) and join the engine thread; run() returns Ok -> exit 0
```

The order matters: readiness drops first so the load balancer stops routing, then the listener closes, then in-flight work finishes, and only then does the engine stop. `/healthz` stays 200 the whole time because the process is alive and should not be restarted.

### 5.7 The chat template and embeddings

[`template.rs`](../../../../course/solutions/rust/crates/tl-serve/src/template.rs) parses once (every syntax error surfaces at model load, in `Template::parse`) and renders a node tree. The `for` node is where `loop.first` comes from:

```rust
Node::For { var, iter, body } => {
    let items = match eval(iter, env)? {
        Some(Value::Array(a)) => a,
        Some(Value::Null) | None => Vec::new(),   // an absent list loops zero times, as in Jinja
        Some(other) => return Err(format!("template: cannot loop over {other}")),
    };
    let n_items = items.len();
    for (i, item) in items.into_iter().enumerate() {
        // index is 1-based, index0 0-based; first is i == 0, not index == 0
        let lp = serde_json::json!({
            "first": i == 0, "last": i + 1 == n_items, "index": i + 1, "index0": i, "length": n_items
        });
        env.scopes.push(vec![(var.clone(), item), ("loop".to_string(), lp)]);
        let r = render_nodes(body, env, out);
        env.scopes.pop();   // pop even on error, then propagate
        r?;
    }
}
```

`chat_settings` finds the template in `generation_config.json`, then `tokenizer_config.json`, then falls back to `DEFAULT_TEMPLATE`, so every model directory renders. Embeddings pool in `f64` and normalize:

```rust
pub fn pool_embedding(hidden: &[f32], d: usize) -> Vec<f32> {
    let t = hidden.len() / d.max(1);
    let mut m = vec![0.0f64; d];
    for row in hidden.chunks_exact(d) {
        for (a, x) in m.iter_mut().zip(row) { *a += *x as f64; }
    }
    for a in &mut m { *a /= t.max(1) as f64; }
    let norm = m.iter().map(|x| x * x).sum::<f64>().sqrt();
    // unit length: cosine similarity becomes a dot product for the policy head
    m.iter().map(|x| if norm > 0.0 { (x / norm) as f32 } else { 0.0 }).collect()
}
```

`ServeConfig::from_toml` parses `[engine]`, applies `TL_ENGINE__<KEY>` environment overrides on top (lower-cased, integers parsed as integers), rejects unknown keys, and refuses values this engine cannot serve (a role other than `unified`, `kv_format` 2, `quant` other than `none` or `int4`) at load rather than at first request.

Full solution: [`engine.rs`](../../../../course/solutions/rust/crates/tl-engine/src/engine.rs), [`server.rs`](../../../../course/solutions/history/L10.5/rust/crates/tl-serve/src/server.rs), [`openai.rs`](../../../../course/solutions/rust/crates/tl-serve/src/openai.rs), [`sse.rs`](../../../../course/solutions/rust/crates/tl-serve/src/sse.rs), [`template.rs`](../../../../course/solutions/rust/crates/tl-serve/src/template.rs).

### 5.8 Why this solution

- **Cost.** Per step the engine thread does $O(L)$ work for $L$ live requests on top of the forward pass (the closed-channel sweep and token delivery), negligible next to the model. `TextStream::push` is $O(|T| \cdot \sum |\sigma|)$ for the held text $T$, and $T$ is at most the longest stop string, so it is constant per token in practice.
- **Why not `spawn_blocking` per step, or a mutex around the engine?** Both put several tasks in contention for one model; a mutex serializes them anyway and makes batching impossible, because batching needs one owner to see every request at once. The single owner thread is the batching.
- **Why not stream tokens straight to the socket?** The engine would block on a slow client. With a per-request channel the engine never waits on I/O; a slow reader only grows its own channel by one token per step.
- **Why decide the status on the first event?** Once headers are sent as 200 `text/event-stream`, a later rejection can only be an error event inside the stream, which many clients treat as a success with garbage. Waiting for the first token costs one step of latency on the header and buys correct status codes.
- **Tradeoffs accepted.** The engine thread polls with a 20 ms timeout when idle, so the first request after idle can wait up to 20 ms. Drain has a 30 s ceiling, after which remaining streams are cut, matching Kubernetes' default grace period.

### 5.9 Explain it back

**Why does the engine use `try_send` on a bounded channel instead of `send().await`?**

<details><summary>Answer</summary>

`send().await` on a full channel waits, so overload becomes unbounded latency, and the client times out anyway. `try_send` fails immediately on `Full`, which becomes 429 with `Retry-After: 1`: a signal the gateway and load balancers can act on (retry elsewhere, back off) instead of a hung connection.

</details>

**How does a client disconnect reach the scheduler, given that no code sends an "abort" message?**

<details><summary>Answer</summary>

hyper drops the `SseBody`, so the `Bytes` channel's receiver is gone; the SSE task's next `send` returns false and the task returns, dropping its `Ev` receiver. On the engine thread `l.tx.is_closed()` is now true, and the sweep before the next step calls `engine.abort(id)`, which releases the blocks. The step after the disconnect is the latest point the blocks are freed.

</details>

**Why does `TextStream` hold back `d` when the stop string is `de`, and why only at character boundaries?**

<details><summary>Answer</summary>

`d` might be the start of `de`; if it were sent and `e` followed, the client would already have shown part of the stop string, and the streamed text would differ from the non-streamed `bc`. Holding back the longest suffix that is a proper prefix of some stop string is the least you can hold and still be safe. The prefixes are taken at `char_indices` so a slice never cuts a multi-byte character, which would panic on a `str`.

</details>

**Why is `usage.prompt_tokens` the length of the rendered template and not of the user's text?**

<details><summary>Answer</summary>

The model reads the rendered prompt: system turn, role markers, and the generation prompt included. That is what occupies KV blocks and what the gateway bills. Counting only the user's text would undercount every chat by the template's overhead.

</details>

**Why does the drain keep the health listener open until the in-flight count reaches zero?**

<details><summary>Answer</summary>

`/readyz` must keep answering 503 during the drain so Kubernetes and the gateway route nothing new to this pod; closing the health port would look like a crash. `/healthz` keeps answering 200 so the pod is not killed early. Only when every stream has finished (the `Inflight` guards have all dropped) is it safe to stop both.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Reporting a length cut as `stop` | clients think the model finished its answer | the finish reason comes from the scheduler's `Finished` event, mapped one to one in `Detok::token`; see 5.2 and 5.5 |
| Rendering without the generation prompt | the model continues the user's turn instead of answering | `render_chat(..., true, ...)` always passes `add_generation_prompt = true`; see 5.4 |
| Ending an SSE event with one newline | every client merges all chunks into one event | `sse::event` writes `data: {payload}\n\n`, and `DONE` and `PING` end with a blank line too |
| No role in the first delta | the OpenAI client cannot assemble the message | `stream_task` sends `chunk("", true, ...)` before any token; see 5.5 |
| Sending text that may begin a stop string | the stream shows `d` of `de` before it stops | `TextStream::push` holds back the longest stop-string prefix at the tail; see 5.5 |
| Returning the stop string itself | answers end with the stop sequence | `push` returns `held[..at]`, the text before the first occurrence |
| Accepting out-of-range values | `temperature: -1` reaches the sampler | `number` checks every field's range and returns 400 with `param`; see 5.4 |
| Ignoring `n` | a client asking for 2 choices silently gets 1 | `n != 1` is 422 `unsupported_parameter` |
| A full queue answered with 503 or a wait | clients and the gateway cannot tell overload from failure; latency grows unbounded | `try_send` on the bounded channel; `Full` becomes 429 with `Retry-After: 1`; see 5.4 |
| `abort` that does not reach the scheduler | a gone client keeps generating and holding blocks | `Engine::abort` calls `sched.abort`, and the engine loop sweeps `is_closed()` before every step; see 5.3 |
| Exiting on SIGTERM without waiting | every rolling deploy cuts streams mid-answer | the drain waits for the `Inflight` count to reach zero before stopping the engine; see 5.6 |
| Returning the mean without normalizing | cosine similarities and the policy head are off | `pool_embedding` divides by the L2 norm; see 5.7 |
| `loop.first` off by one | the default system turn appears in the wrong place | `first` is `i == 0` over the 0-based enumerate; see 5.7 |
| Ignoring `TL_ENGINE__` overrides | a ConfigMap cannot be tuned per pod | `from_toml` merges the overrides into `[engine]` before reading any key |
| Dropping `X-Request-Id` | one request cannot be followed through gateway and engine logs | every response echoes the header, or generates one |
| Running a step inside an async task | one long prefill stalls every connection on that runtime thread | the engine runs on its own OS thread, `tl-engine`; see 5.1 and 5.3 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L10.1` | the runner (also for embeddings) and the sampler with per-request `stream(seed, sample)` |
| Back | `L10.2` | the scheduler: `add`, `abort`, `schedule`, `on_step`; `QueueFull` becomes 429 |
| Back | `L10.3` | `plan` builds every step; `prefill_chunk` turns chunking on |
| Back | `L10.4` | the block manager in the mode `prefix_cache` names; hit ratio and block counts in `/metrics` |
| Back | `L10.8` | runner-backed speculative generation when `[engine].speculative` is configured |
| Back | `L1.5` | tl-tok: byte tokenizer and tokenizer.json BPE for prompts and token bytes |
| Forward | `L10.6` | disaggregated prefill and decode add `EngineControl` and the `X-TL-KV-Handle` resume to this server |
| Forward | `L10.7` | histograms, OpenTelemetry spans, and trace propagation join `/metrics` |
| Forward | `L10.9` | `tools`, `tool_choice`, and `response_format` replace today's 422 |
| Forward | `gw.04`, `load.01`, `ag.01` | the gateway proxies this API; the load generator and the agent SDK are its clients |
| Forward | `L12.3` | GRPO uses this serving API and its chat template |

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one engine thread and channels | vLLM's `AsyncLLM` with a separate engine core process | the engine in another process (ZeroMQ), so Python's GIL never blocks it | `vllm/v1/engine/async_llm.py` |
| a Jinja subset | [minijinja](https://github.com/mitsuhiko/minijinja), Hugging Face `apply_chat_template` | the full template language: macros, `raise_exception`, namespaces | `transformers` `apply_chat_template` |
| bounded queue + 429 | admission control with token buckets and fairness | per-tenant limits, priority classes | `gw.03`, `gw.04` |
| hand-written validation | OpenAPI-generated servers ([utoipa](https://github.com/juhaku/utoipa), [progenitor](https://github.com/oxidecomputer/progenitor)) | schema and code from one source | `craft.14`'s v2 migration |
| `/metrics` gauges | OpenTelemetry metrics and traces | histograms of TTFT and TPOT, spans per request | `L10.7` |
