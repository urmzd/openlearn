<!-- ol:module L10.5 -->
# tl-serve: OpenAI-compatible HTTP + SSE on tokio/hyper

## Overview

| | |
|---|---|
| **Module** | `L10.5` · build · Rust · Pass 7 · 16 to 24 h |
| **You build** | `rust/crates/tl-engine/src/engine.rs`: `Engine`, the step loop over runner, scheduler, block manager, and per-request samplers · `rust/crates/tl-serve/src/server.rs`: the v1 server on tokio and hyper (runtime.toml, the engine thread, bounded admission, abort on disconnect, drain on SIGTERM, health, readiness, `/metrics`) · `openai.rs` (validation and the response and chunk documents) · `sse.rs` (SSE framing, incremental UTF-8, stop strings) · `template.rs` (the chat-template Jinja subset) · the module lines of your `L10.0` crate root `tl-serve/src/lib.rs` · your entry point's `--config` form |
| **Contract** | HTTP: [`openapi/openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml) (engine tier) · the `engine` role, config form: [`spec/cli-roles.md`](../../../../course/contracts/spec/cli-roles.md) · `[engine]`: [`config/runtime.schema.json`](../../../../course/contracts/config/runtime.schema.json) · chat templates: [`formats/generation-config.schema.json`](../../../../course/contracts/formats/generation-config.schema.json) · metric names: [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml) |
| **Tests** | `course/tests/rust/l10_5.rs`, 19 tests (what they check: section 4); conformance `ol conform openapi:v1 --target engine` from `MS-L10`, including the official `openai` Python client |
| **Needs** | `L10.1` runner and sampler ([chapter](01-model-runner-and-sampler.md)) · `L10.2` scheduler ([chapter](02-continuous-batching.md)) · `L10.3` chunked prefill ([chapter](03-chunked-prefill.md)) · `L10.4` block manager ([chapter](04-block-manager-and-prefix-cache.md)) · `L10.8` speculative decoding ([chapter](08-speculative-decoding-in-the-engine.md)) · `L1.5` tl-tok: the byte tokenizer and tokenizer.json BPE ([chapter](../p01-tokenizers/05-rust-fast-bpe.md)) · reading: `lang.09` async Rust and tokio ([primer](../../../tracks/software-craftsmanship/12-language-and-tool-primers/09-async-rust-and-tokio.md)), `lang.05` HTTP and SSE, `L10.0` the tracer server, `L8.2` generate and the detokenizer · or `--ref-deps` |
| **Used by** | `L10.6` (disaggregation), `L10.7` (metrics and tracing), and `L10.9` (tool calls) build on this server, and `gw.04`, `ag.01`, `load.01`, and `L12.3` reach it over HTTP |
| **Milestone** | `MS-L10` (`ol conform openapi:v1 --target engine` 100%) |
| **Optional depth** | [OpenAI API reference: chat completions and streaming](https://platform.openai.com/docs/api-reference/chat) (free); [hyper 1.x guide](https://hyper.rs/guides/1/) (free); [Tokio tutorial: channels, select](https://tokio.rs/tokio/tutorial) (free); [Jinja template designer docs](https://jinja.palletsprojects.com/en/stable/templates/) (free); [vLLM's OpenAI server](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai) (free) |

## Key Takeaways

- The model runs on one OS thread and HTTP on tokio tasks; they meet in channels: a bounded admission queue in, one event channel per request out (`concurrent_streams_equal_serial`).
- A chat request is rendered through the model's own chat template, with the generation prompt, before tokenization, and `usage.prompt_tokens` counts that rendered prompt (`hand_example_chat_completion`, `usage_counts_the_templated_prompt`).
- Streaming is `data: <json>\n\n` events: the role first, text as it becomes valid UTF-8 and safe from a stop string, `finish_reason` on the last content chunk, the usage chunk when asked, then `[DONE]` (`stream_framing_role_first_then_done`, `text_stream_holds_back_stop_prefixes`).
- Every error has the v1 shape, and the status says what kind: 400 bad value, 422 unsupported value, 404 unknown model, 429 full queue with `Retry-After` (`errors_use_the_openai_shape`, `full_queue_answers_429_with_retry_after`).
- A client that leaves frees its KV blocks within one step, and SIGTERM drains: `/readyz` turns 503, in-flight streams finish, then exit 0 (`disconnect_frees_blocks_within_a_step`, `drain_finishes_in_flight_and_refuses_new`).

## How to work this chapter

```bash
ol start L10.5               # stubs engine.rs, server.rs, openai.rs, sse.rs, template.rs
ol tests L10.5
ol check L10.5
ol conform openapi:v1 --target engine     # once your entry point serves --config (MS-L10 runs it)
```

Add to your `tl-serve/Cargo.toml` what the course manifest lists: `tl-engine` and `tl-tok` by path, `tokio` (full), `hyper` (server, http1), `hyper-util` (tokio), `http-body-util`, `bytes`, `serde_json`, `toml` (`contracts/allowed-deps.toml`). Add `pub mod engine;` to `tl-engine/src/lib.rs` and `pub mod openai; pub mod server; pub mod sse; pub mod template;` to `tl-serve/src/lib.rs`. Your `http.rs` (the tracer) stays as it is: `--model-dir --port --health-port` keeps serving API v0, and `--config <runtime.toml>` now starts `server::run`:

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_chat_completion` | unit | section 3: `bcd`, `length`, usage 1 + 3, every field of `chat.completion` | the worked example over the wire |
| `stream_framing_role_first_then_done` | conformance | content type, one id, role first, finish on the last content chunk, usage chunk, `[DONE]` | every SSE client parses this |
| `stream_equals_nonstream_at_temperature_0` | differential | deltas concatenate to the plain content on the tiny Llama | case `chat.stream.equals_nonstream` |
| `seed_reproduces_sampling` | property | the same seed repeats, another seed differs | case `chat.seed` |
| `stop_strings_end_generation_across_tokens` | unit | `de` split across tokens stops at `bc`, plain and streamed | case `chat.stop` |
| `errors_use_the_openai_shape` | conformance | thirteen bad requests: status, shape, `param`, `code` | cases `err.400`, `err.422`, `route` errors |
| `completions_endpoint_and_usage` | conformance | `/v1/completions`: default 16 tokens, usage | the text endpoint of v1 |
| `usage_counts_the_templated_prompt` | conformance | `prompt_tokens` = tokens of the rendered default template | case `chat.usage`, the gateway's billing |
| `full_queue_answers_429_with_retry_after` | fault | queue of one: the third request gets 429 with `Retry-After` at once; the queued one runs later | bounded admission |
| `disconnect_frees_blocks_within_a_step` | fault | `/metrics` shows 0 active sequences and 0 used blocks within 2 s of a disconnect | case `cancel.disconnect` |
| `drain_finishes_in_flight_and_refuses_new` | fault | `drain`: `/readyz` 503, new connections refused, the open stream reaches `[DONE]` | rolling deploys on Kubernetes |
| `embeddings_are_mean_pooled_and_normalized` | unit | each vector is the normalized mean of the hidden states, in input order | the policy classifier's input (D33) |
| `tokenize_models_health_and_request_id` | conformance | `/v1/tokenize`, `/v1/models`, health on both ports, `X-Request-Id` | the gateway's TPM cost and routing |
| `concurrent_streams_equal_serial` | differential | eight parallel streams equal the serial answers | case `concurrency.16` |
| `chat_template_hand_example` | unit | SmolLM2's real template, filters, `elif`, `loop.last`, whitespace control, the default template | the agent model (D20) gets the prompt it was trained on |
| `template_errors_are_caught_at_parse` | boundary | unclosed blocks, unknown statements and filters refused at parse | a broken template fails at load |
| `text_stream_holds_back_stop_prefixes` | unit | stop-prefix hold-back and incremental UTF-8, byte by byte | streams never leak a stop string |
| `runtime_toml_and_env_overrides` | unit | `[engine]` keys, defaults, `TL_ENGINE__` overrides, refused keys and values | the config form of spec/cli-roles.md |
| `speculative_runtime_config_drives_the_serving_path` | integration | `[engine].speculative` selects the runner-backed L10.8 path; greedy output matches ordinary serving | prompt lookup can be enabled without changing greedy answers |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Reporting a length cut as `stop` | clients think the model finished its answer | `hand_example_chat_completion` (mutant `s01`) |
| Rendering without the generation prompt | the model continues the user's turn instead of answering | `usage_counts_the_templated_prompt` (mutant `s02`) |
| Ending an SSE event with one newline | every client merges all chunks into one event | `stream_framing_role_first_then_done` (mutant `s03`) |
| No role in the first delta | the OpenAI client cannot assemble the message | `stream_framing_role_first_then_done` (mutant `s04`) |
| Sending text that may begin a stop string | the stream shows `d` of `de` before it stops | `text_stream_holds_back_stop_prefixes` (mutant `s05`) |
| Returning the stop string itself | answers end with the stop sequence | `stop_strings_end_generation_across_tokens` (mutant `s06`) |
| Accepting out-of-range values | `temperature: -1` reaches the sampler | `errors_use_the_openai_shape` (mutant `s07`) |
| Ignoring `n` | a client asking for 2 choices silently gets 1 | `errors_use_the_openai_shape` (mutant `s08`) |
| A full queue answered with 503 or a wait | clients and the gateway cannot tell overload from failure; latency grows unbounded | `full_queue_answers_429_with_retry_after` (mutant `s09`) |
| `abort` that does not reach the scheduler | a gone client keeps generating and holding blocks | `full_queue_answers_429_with_retry_after` (mutant `s10`) |
| Exiting on SIGTERM without waiting | every rolling deploy cuts streams mid-answer | `drain_finishes_in_flight_and_refuses_new` (mutant `s11`) |
| Returning the mean without normalizing | cosine similarities and the policy head are off | `embeddings_are_mean_pooled_and_normalized` (mutant `s12`) |
| `loop.first` off by one | the default system turn appears in the wrong place | `chat_template_hand_example` (mutant `s13`) |
| Ignoring `TL_ENGINE__` overrides | a ConfigMap cannot be tuned per pod | `runtime_toml_and_env_overrides` (mutant `s14`) |
| Dropping `X-Request-Id` | one request cannot be followed through gateway and engine logs | `tokenize_models_health_and_request_id` (mutant `s15`) |
| Running a step inside an async task | one long prefill stalls every connection on that runtime thread | `concurrent_streams_equal_serial` |

## 6. Where it's used next
| Forward | `L12.3` | Registered module relationship. |

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

If you skip this module, MS-L10's conformance run has no v1 engine to test.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one engine thread and channels | vLLM's `AsyncLLM` with a separate engine core process | the engine in another process (ZeroMQ), so Python's GIL never blocks it | `vllm/v1/engine/async_llm.py` |
| a Jinja subset | [minijinja](https://github.com/mitsuhiko/minijinja), Hugging Face `apply_chat_template` | the full template language: macros, `raise_exception`, namespaces | `transformers` `apply_chat_template` |
| bounded queue + 429 | admission control with token buckets and fairness | per-tenant limits, priority classes | `gw.03`, `gw.04` |
| hand-written validation | OpenAPI-generated servers ([utoipa](https://github.com/juhaku/utoipa), [progenitor](https://github.com/oxidecomputer/progenitor)) | schema and code from one source | `craft.14`'s v2 migration |
| `/metrics` gauges | OpenTelemetry metrics and traces | histograms of TTFT and TPOT, spans per request | `L10.7` |
