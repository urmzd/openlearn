<!-- ol:module L10.9 -->
# Tool calls and constrained JSON decoding in the engine

## Overview

| | |
|---|---|
| **Module** | `L10.9` · build · Rust · Pass 7 · 12 to 16 h |
| **You build** | `rust/crates/tl-engine/src/constrain.rs`: the regex subset to a canonical byte DFA (Thompson NFA, subset construction over byte classes, dead states removed, Moore minimization, breadth-first numbering), the JSON-schema subset to a regex, token masks over a byte trie, `constrained_sample`, an order-keeping JSON reader that writes Python's compact JSON · `rust/crates/tl-serve/src/tools.rs`: `tools`, `tool_choice`, `response_format` validated, tools rendered into the prompt, the grammar per `tool_choice`, a streaming tool-call parser, and the OpenAI `tool_calls` message and deltas · `rust/crates/tl-serve/src/server.rs`, taken over from `L10.7`: the chat route reads the tool fields, decodes a request with a grammar under its `Constraint`, and answers with `tool_calls` |
| **Contract** | the Python specification you port: [`py/tinyllm/infer/constrain.pyi`](../../../course/contracts/py/tinyllm/infer/constrain.pyi) (L8.7) · HTTP: [`openapi/openai-subset.v1.yaml`](../../../course/contracts/openapi/openai-subset.v1.yaml) (`tools`, `tool_choice`, `response_format`, `tool_calls` deltas, `finish_reason: tool_calls`) · conformance `tools.call`, `tools.stream`, `tools.choice` |
| **Tests** | `course/tests/rust/l10_9.rs`, 20 tests (what they check: section 4); the `L10.5` and `L10.7` tests run as the regression of the server you take over |
| **Needs** | `L8.7` your Python constrained decoding, the specification ([chapter](../p08-inference/07-constrained-decoding.md)) · `L10.1` the sampler ([chapter](01-model-runner-and-sampler.md)) · `L10.5` the chat template and the API error shape ([chapter](05-openai-server-on-tokio.md)) · `L1.5` the `Tokenizer` trait ([chapter](../p01-tokenizers/05-rust-fast-bpe.md)) · `L10.7` the serve loop you take over, which records metrics and spans ([chapter](07-serving-metrics-and-tracing.md)) · reading: `M06.1` graphs and reachability ([chapter](../../../math/06-discrete-math-2/01-graphs-dags-and-topological-sort.md)) · or `--ref-deps` |
| **Used by** | the serve loop you take over, and so its call site `obs.02`, which runs your engine; `ag.01` and `MS-agent` read `tool_calls` from it over HTTP; `craft.14` takes the serve loop over next to serve API v1 and v2 side by side |
| **Milestone** | `MS-L10` (and `MS-agent`, where SmolLM2-135M-Instruct calls your agent's tools through it) |
| **Optional depth** | Hopcroft, Motwani, Ullman, *Introduction to Automata Theory*, ch. 2 to 4; [Willard and Louf, Efficient Guided Generation](https://arxiv.org/abs/2307.09702) (free); [XGrammar](https://arxiv.org/abs/2411.15100) (free); [OpenAI function calling guide](https://platform.openai.com/docs/guides/function-calling) (free) |

## Key Takeaways

- A minimized DFA numbered breadth-first over bytes in order is canonical: your Rust tables equal your Python's for every pattern and schema, so the masks are the same (`hand_example_dfa`, `dfa_matches_python_l8_7`, `masks_match_python_l8_7`).
- A token is allowed when walking ALL its bytes stays live; EOS only where the output is a whole string; specials never (`token_index_rules`).
- Whatever the logits, constrained output parses and satisfies its schema: that is why a random tiny model can still make a valid forced tool call (`constrained_output_always_parses`, `forced_call_from_a_random_model_is_valid`).
- `tool_choice` decides the grammar: a named function is one call of it, `required` one to four calls, `auto` free text until the model writes `<tool_call>`, `none` free text (`grammar_by_tool_choice`).
- The parser streams argument fragments as they arrive and never leaks markup into the content, wherever the tokens split the text (`parser_hand_example`, `parser_is_split_invariant`).

## How to work this chapter

```bash
ol start L10.9               # stubs constrain.rs and tools.rs
ol tests L10.9
ol check L10.9               # exit code is the verdict
ol check L10.9 --ref-deps    # only if L8.7, L10.1, L10.5, or L1.5 is not passing yet
ol diff  L10.9
ol conform openapi:v1 --target engine   # tools.* turn from pending to checked once L10.9 passes
```

Declare `pub mod constrain;` in `tl-engine/src/lib.rs` and `pub mod tools;` in `tl-serve/src/lib.rs`. This module takes over `server.rs` from `L10.7` (`upgrades`), so `ol check L10.9` also runs the `L10.5` and `L10.7` tests as regressions; one of `L10.5`'s error cases is `response_format` `json_schema`, which stays 422 here. Your chat handler calls `parse_tool_request` on the raw body (before the generic parser, so property order survives), renders with `render_prompt`, builds one `TokenIndex` per (grammar, model) and caches it, runs each sequence through a `Constraint` (from the first token, or from the trigger in `auto` mode), and turns the text through `ToolParser` into `delta_json` chunks or `message_json`. A request with a grammar runs on the engine thread outside the batch, as the speculative path does: its own KV blocks through `spec::RunnerTarget`, `constrained_sample` per token, and the request's own generator; a grammar that is complete with no token left to allow (a vocabulary without EOS) ends the answer with `stop`.

---

## 1. Why now

Agents (`ag.01` onward) need structured output from your engine: a tool name and JSON arguments that a program can execute. Asking nicely in the prompt works for large models some of the time and for a 135M model almost never. In Python (`L8.7`) you built the machinery that makes it work every time: compile the allowed language to a DFA over bytes and mask, at every step, every token that would leave it. Your Rust engine answers `tools` and `response_format` with 422 `unsupported_parameter` today. This module ports the automaton to the engine (the masks must equal the Python ones exactly, so the port is held to them) and adds the OpenAI tool-calling surface around it.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $\Sigma$ | the 256 byte values | alphabet |
| $\delta(s, b)$ | the state after byte $b$ from state $s$; $\bot$ (DEAD) when no accepting state is reachable | `i32` |
| $A$ | the accepting states | `bool[n]` |
| $w_t$ | the bytes of token $t$ | `Vec<u8>` |
| $\delta^*(s, w)$ | $\delta$ applied byte by byte; $\bot$ stays $\bot$ | `i32` |

### 2.1 From regex to a canonical DFA

The pipeline is L8.7's, step for step. Parse the subset (literals, escapes, classes `\d \w \s` and negations, `.`, brackets, groups, `|`, `* + ? {m} {m,} {m,n}`) into a tree of byte sets. Thompson's construction turns each node into a small NFA fragment glued with epsilon edges. The subset construction runs over **byte classes** (bytes every edge treats alike) instead of 256 bytes, so a JSON string pattern with its UTF-8 ranges stays small. Remove the states from which no accepting state is reachable (their transitions become $\bot$), Moore-minimize (split blocks until every state of a block moves to the same block on every class), and number the blocks breadth-first from the start state taking bytes 0 to 255 in order. Two patterns with the same language get identical tables.

### 2.2 From a schema to a regex

The JSON-schema subset of `constrain.pyi` maps to a regex over **compact** JSON: strings (well-formed UTF-8, the eight escapes, `\uXXXX`), integers, numbers, booleans, null, `enum`/`const` (each value as Python's compact `json.dumps`, escaped), arrays with `minItems`/`maxItems`, objects whose properties appear **in the order given**, required ones always and optional ones possibly, and `anyOf`. Every other keyword is refused (422), never ignored. Property order is part of the language, so the schema is read with a JSON reader that keeps key order.

### 2.3 Masks over a byte trie

Token $t$ is allowed in state $s$ when $\delta^*(s, w_t) \ne \bot$; EOS is allowed exactly when $s \in A$; a special token or an empty token never. Walking every token's bytes per state costs $O(\sum_t |w_t|)$; a trie over the vocabulary shares the walk between tokens with a common prefix and prunes a whole subtree at the first $\bot$. Masks are cached per state and shared by every request using the same grammar. Sampling then runs L10.1's `sample` on the logits with every disallowed token at $-\infty$; the logprob is renormalized over the allowed tokens.

### 2.4 Tool calls

The model writes calls as `<tool_call>{"name":"<name>","arguments":<object>}</tool_call>`, the format small instruct models are trained on, separated by `\n`. `tool_choice` picks the grammar:

| `tool_choice` | grammar |
|---|---|
| `{"type":"function","function":{"name":"f"}}` | exactly one call of `f`, from the first token |
| `"required"` | one call of any tool, or up to 4 with `parallel_tool_calls` |
| `"auto"` (default with tools) | free text; once the output ends with `<tool_call>`, the rest of one call |
| `"none"` | free text (or a JSON object with `response_format: json_object`, nested at most 3 levels: the language of all JSON objects is not regular) |

The tools are described to the model in a system message (appended to the first system message if there is one); an assistant turn's `tool_calls` are written back as `<tool_call>` blocks and a `tool` message as a `<tool_response>` user turn, so a conversation renders the way it was generated.

### 2.5 Streaming the answer

The parser keeps a small state machine: outside a call it emits text but holds back a suffix that could be the start of `<tool_call>`; after the opener it waits for the header `{"name":"...","arguments":` and emits the call's id and name; inside the arguments it tracks brace depth and string escapes and passes every new piece on as an `arguments` fragment until the value closes; then it consumes `}</tool_call>`. Ids are `call_<request id>_<index>`. The stream's chunks are `{"tool_calls":[{"index":i,"id":...,"type":"function","function":{"name":...,"arguments":""}}]}` for a call's start and `{"tool_calls":[{"index":i,"function":{"arguments":"<fragment>"}}]}` after; `finish_reason` is `tool_calls` when any call was made.

## 3. Worked example by hand

**The automaton** (test `hand_example_dfa`). `(cat|car|dog)s?`: breadth-first from state 0 with bytes in order: `c` (0x63) gets id 1, `d` (0x64) id 2; from 1, `a` gives 3; from 2, `o` gives 4; from 3, `r` and `t` both reach the state after a whole word, and so does `g` from 4: minimization merges "car", "cat", and "dog" into state 5 (accepting); `s` from 5 gives 6 (accepting, no transitions). Seven states, accepting {5, 6}, eight live edges.

**A mask** (test `constrained_sample_hand_example`). Pattern `ab?`, vocabulary `[a, b, c, EOS]`, logits `[0, 1, 5, 2]`. In state 0 only `a` walks live: greedy takes `a` although `c` has the largest logit, with logprob $\ln 1 = 0$. In the state after `a` (accepting) `b` and EOS are allowed: greedy takes EOS ($2 > 1$) with logprob $2 - \ln(e^1 + e^2)$.

**A schema** (`grammar_by_tool_choice`). The conformance tool's parameters `{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}` become `\{\"city\":"(?:...)*"\}`: the escaped key, a colon, the JSON string pattern. Optional members nest: two optional integers `x`, `y` give `\{(?:\"x\":INT(?:,\"y\":INT)?|(?:\"y\":INT|))\}`, so `{}`, `{"x":1}`, `{"y":2}`, and `{"x":1,"y":2}` match and `{"y":2,"x":1}` does not.

**A streamed call** (test `parser_hand_example`). Pieces `Sure.<tool_`, `call>{"name":"get_weather","argu`, `ments":{"city":"Pa`, `ris"}}</tool_call>`:

| Piece | Events |
|---|---|
| 1 | content `Sure.` (`<tool_` held back) |
| 2 | nothing (the header is incomplete) |
| 3 | call 0 starts (`call_req7_0`, `get_weather`); arguments `{"city":"Pa` |
| 4 | arguments `ris"}`; `}</tool_call>` consumed |

Folded: content `Sure.`, one call with arguments `{"city":"Paris"}`, `finish_reason` `tool_calls`.

## 4. The interface

```rust
// rust/crates/tl-engine/src/constrain.rs
pub const DEAD: i32 = -1;
pub struct Dfa { pub n_states: usize, pub accept: Vec<bool>, pub trans: Vec<i32> /* n * 256 */ }
impl Dfa { pub fn step(&self, state: i32, data: &[u8]) -> Result<i32, ConstrainError>; pub fn matches(&self, data: &[u8]) -> bool; }
pub fn parse(pattern: &str) -> Result<Node, ConstrainError>;
pub fn regex_to_dfa(pattern: &str) -> Result<Dfa, ConstrainError>;
pub enum Json { Null, Bool(bool), Int(String), Float(f64), Str(String), Arr(Vec<Json>), Obj(Vec<(String, Json)>) }
impl Json { pub fn parse(text: &str) -> Result<Json, ConstrainError>; pub fn get(&self, key: &str) -> Option<&Json>; pub fn dumps(&self) -> String; }
pub fn json_quote(s: &str) -> String;  pub fn python_float(x: f64) -> String;  pub fn escape(text: &str) -> String;
pub fn json_schema_to_regex(schema: &Json) -> Result<String, ConstrainError>;
pub fn json_object_regex(depth: usize) -> String;
pub struct TokenIndex { pub dfa: Dfa, pub vocab_size: usize, pub eos_id: Option<u32>, /* trie, cache */ }
impl TokenIndex { pub fn new(dfa: Dfa, vocab: &[Option<Vec<u8>>], eos_id: Option<u32>) -> Result<TokenIndex, ConstrainError>;
                  pub fn mask(&self, state: i32) -> Result<Vec<bool>, ConstrainError>; pub fn next_state(&self, state: i32, token_id: u32) -> Result<i32, ConstrainError>; }
pub struct Constraint { pub index: Arc<TokenIndex>, pub state: i32, pub done: bool }   // new, mask, advance, is_complete
pub fn apply_mask(logits: &[f32], mask: &[bool]) -> Result<Vec<f32>, ConstrainError>;
pub fn constrained_sample(logits: &[f32], c: &mut Constraint, p: &SamplingParams, prompt: &[u32], output: &[u32], rng: &mut Pcg32) -> Result<(u32, f64), ConstrainError>;

// rust/crates/tl-serve/src/tools.rs
pub struct ToolError { pub status: u16, pub param: String, pub code: Option<String>, pub message: String }   // to_api_error, to_json
pub struct Tool { pub name: String, pub description: Option<String>, pub parameters: Json }
pub enum ToolChoice { Auto, None, Required, Named(String) }   pub enum ResponseFormat { Text, JsonObject }
pub struct ToolRequest { pub tools: Vec<Tool>, pub choice: ToolChoice, pub response_format: ResponseFormat, pub parallel: bool }
pub fn parse_tool_request(body: &str) -> Result<ToolRequest, ToolError>;
pub struct Grammar { pub pattern: String, pub trigger: Option<String> }
pub fn grammar(r: &ToolRequest) -> Option<Grammar>;
pub fn render_messages(messages: &Json, tools: &[Tool]) -> Result<Vec<(String, String)>, ToolError>;
pub fn render_prompt(t: &Template, messages: &Json, tools: &[Tool], bos: &str, eos: &str) -> Result<String, ToolError>;
pub fn vocab_bytes(tok: &dyn Tokenizer, specials: &[u32]) -> Vec<Option<Vec<u8>>>;
pub enum ToolEvent { Content(String), CallStart { index: usize, id: String, name: String }, Arguments { index: usize, fragment: String } }
impl ToolParser { pub fn new(request_id: &str) -> ToolParser; pub fn push(&mut self, piece: &str) -> Vec<ToolEvent>; pub fn finish(&mut self) -> Vec<ToolEvent>; }
pub fn collect(events: &[ToolEvent]) -> (Option<String>, Vec<ToolCall>);
pub fn finish_reason(calls: usize, engine: &str) -> String;
pub fn message_json(content: Option<&str>, calls: &[ToolCall]) -> String;
pub fn delta_json(e: &ToolEvent) -> String;

// rust/crates/tl-serve/src/server.rs, taken over from L10.7: its public API is unchanged
// (ServeConfig, spawn, run, ServerHandle); POST /v1/chat/completions accepts tools,
// tool_choice, response_format, and parallel_tool_calls.
```

### What the tests check

`course/fixtures/L10.9/constrain_golden.json` (`course/oracle/L10.9/constrain_golden.py`) records, from the Python reference of L8.7, the regex and canonical DFA of 13 schemas and 8 patterns, the masks along three seeded paths of each through a 123-entry vocabulary (printable bytes, multi-byte and UTF-8 tokens, a special, an empty token, EOS), Python's compact `json.dumps` of 19 JSON texts, and the schemas and patterns the subset refuses; L8.7's own golden file (an independent derivative-based oracle) is replayed too.

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_dfa` | unit | section 3's seven states and eight edges; `step` and `matches` | you and Python agree on the canonical form |
| `dfa_matches_python_l8_7` | differential | regex text per schema; states, accepting flags, every transition per pattern | identical tables, so identical masks |
| `masks_match_python_l8_7` | differential | allowed ids and completeness at every step of every recorded path | what the sampler sees is Python's |
| `subset_errors_are_refused` | boundary | 15 bad patterns and 11 bad schemas refused | a 422, never a silently different language |
| `dumps_matches_python` | unit | key order, `-0`, float repr, escapes, raw non-ASCII | enum values the model can actually produce |
| `token_index_rules` | boundary | specials, empty tokens, EOS rules, forbidden advances | the edge cases of the mask |
| `constrained_sample_hand_example` | unit | section 3's mask and logprobs; an all-false mask refused | the sampler sees the mask first |
| `constrained_output_always_parses` | property | 25 random walks per schema parse and validate | the guarantee itself |
| `json_object_grammar` | boundary | objects up to depth 3; arrays, scalars, broken objects refused | `response_format: json_object` |
| `parse_tool_request_hand_example` | unit | defaults, a named choice, 400s and the 422 naming the keyword | the request surface of `openai-subset.v1` |
| `grammar_by_tool_choice` | unit | named, required, parallel off, auto with its trigger, none | `tools.choice` |
| `render_messages_with_tools` | unit | tools in the system message; calls and results rendered back; through the chat template | multi-turn agent loops (`ag.03`) |
| `vocab_bytes_marks_specials` | unit | specials map to None | chat markers are never generated |
| `parser_hand_example` | unit | section 3's stream, fragment by fragment | `tools.stream` |
| `parser_holds_back_only_real_markup` | boundary | `a<b`, a trailing `<tool`, an opener without a header | no markup in content, no content lost |
| `parser_parallel_calls` | unit | indices, ids, the newline between calls | parallel calls |
| `parser_is_split_invariant` | property | every 2-way split and 200 random splits fold to the same answer | tokens cut text anywhere |
| `message_and_delta_json_shapes` | conformance | the message and both delta shapes, arguments as a string | what `ag.01` and the `openai` client parse |
| `forced_call_from_a_random_model_is_valid` | property | 40 random walks under a named choice give one valid `get_weather` call | MS-L10's PR check on a tiny random model |
| `serve_loop_routes_tool_calls` | conformance | a live server: a forced call comes back as schema-valid `tool_calls` with `finish_reason` `tool_calls`, from the prompt `render_prompt` builds; streamed deltas reassemble to the same call; `tool_choice` none calls nothing; tool-field errors keep the OpenAI shape | `ag.01` and the `tools.*` conformance cases read exactly this |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Stopping minimization after one refinement | equal languages, different tables: masks differ from Python's | `dfa_matches_python_l8_7` (mutant `s01`) |
| Sorting properties by name | the model must write keys in an order the schema did not ask for | `dfa_matches_python_l8_7` (mutant `s02`) |
| No comma before a later optional member | `{"x":1"y":2}` is "valid" and does not parse | `constrained_output_always_parses` (mutant `s03`) |
| Enum values inserted unescaped | `2.5` matches `2x5`; quotes break the pattern | `dfa_matches_python_l8_7` (mutant `s04`) |
| `maxItems` off by one | one item too many passes the grammar and fails the schema | `constrained_output_always_parses` (mutant `s05`) |
| Walking a token's later bytes from the wrong state | multi-byte tokens allowed that leave the language | `masks_match_python_l8_7` (mutant `s06`) |
| EOS allowed before a whole string | outputs end mid-object | `token_index_rules` (mutant `s07`) |
| An empty language not refused | a request that can never finish | `subset_errors_are_refused` (mutant `s08`) |
| Floats printed unlike Python | `1.5e-05` written `0.000015`: the enum value is unreachable | `dumps_matches_python` (mutant `s09`) |
| Tokens still allowed after EOS | generation continues past the end of the language | `masks_match_python_l8_7` (mutant `s10`) |
| The sampler seeing the unmasked logits | the constraint is checked after the fact and fails | `constrained_sample_hand_example` (mutant `s11`) |
| Parameters outside the subset accepted | the schema is silently weakened | `parse_tool_request_hand_example` (mutant `s12`) |
| A named `tool_choice` allowing every tool | the model calls the wrong function | `grammar_by_tool_choice` (mutant `s13`) |
| Dropping argument fragments | streamed arguments arrive incomplete | `parser_hand_example` (mutant `s14`) |
| Not holding back a partial `<tool_call>` | `<tool_` shows up in the user-visible text | `parser_holds_back_only_real_markup` (mutant `s15`) |
| `content: ""` next to `tool_calls` | clients treat the turn as text | `message_and_delta_json_shapes` (mutant `s16`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L8.7` | the Python automaton and schema translation this port is held to |
| Back | `L10.1` | `sample` on the masked logits |
| Back | `L10.5` | `Template::render_messages` and `ApiError` |
| Back | `L1.5` | `Tokenizer::token_bytes` for the vocabulary's bytes |
| Back | `L10.7` | the serve loop you take over: tool requests are recorded and traced through its calls |
| Forward | `obs.02` | runs your engine, whose serve loop is now this module's `server.rs` (the call site an upgrade inherits, DESIGN 3.4) |
| Forward | `craft.14` | takes your `server.rs` over to select API v1 or v2 per request (Pass 11) |

`ag.01` parses your `tool_calls` deltas; `MS-agent` runs SmolLM2-135M-Instruct through your engine with tool calls; the conformance cases `tools.call`, `tools.stream`, `tools.choice` turn from pending to checked once this module passes.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| DFA masks cached per state | Outlines | the same automaton approach, precomputed over the whole vocabulary | [Outlines](https://github.com/dottxt-ai/outlines) |
| a regular subset of JSON schema | XGrammar, llguidance | context-free grammars (recursive JSON) with a persistent stack and fast masks | [XGrammar](https://github.com/mlc-ai/xgrammar), [llguidance](https://github.com/guidance-ai/llguidance) |
| `<tool_call>` text parsing | vLLM tool parsers | one parser per model family's call format | [vLLM tool calling](https://docs.vllm.ai/en/latest/features/tool_calling.html) |
| a lazy trigger in `auto` mode | llama.cpp grammar triggers | grammars activated by a token or pattern mid-generation | [llama.cpp grammars](https://github.com/ggml-org/llama.cpp/tree/master/grammars) |
