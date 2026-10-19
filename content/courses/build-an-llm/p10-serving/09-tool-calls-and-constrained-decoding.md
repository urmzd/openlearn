# Tool calls and constrained decoding

Agents need structured output from the engine: a tool name and JSON arguments that a program can execute. Starting with [types and the OpenAI-compatible provider](../../ai-platform-engineering/13-agent-sdk/01-types-and-provider.md) in *AI Platform Engineering*, the agent system depends on it. Asking nicely in the prompt works for large models some of the time and for a 135M-parameter model almost never.

[Constrained decoding](../p08-inference/07-constrained-decoding.md) in the inference chapter built, in Python, the machinery that makes it work every time: compile the allowed language to a DFA over bytes, and at every step mask every token that would leave it. The Rust engine today answers `tools` and `response_format` with 422 `unsupported_parameter`. This section ports the automaton to the engine and adds the OpenAI tool-calling surface around it. The masks must equal the Python ones exactly, so the port is held to them.

The Python implementation, [`py/tinyllm/infer/constrain.pyi`](../../../../course/contracts/py/tinyllm/infer/constrain.pyi), is the specification. You will also need the sampler from [the model runner and sampler](01-model-runner-and-sampler.md), the chat template and API error shape from [the tokio server](05-openai-server-on-tokio.md), the `Tokenizer` trait from [the Rust fast BPE](../p01-tokenizers/05-rust-fast-bpe.md), and the serve loop from [serving metrics and tracing](07-serving-metrics-and-tracing.md), which this section takes over and which records metrics and spans. For background on reachability, which is what dead-state removal computes, see [graphs, DAGs, and topological sort](../../mathematics-for-computing/06-discrete-math-2/01-graphs-dags-and-topological-sort.md) in *Mathematics for Computing*.

The HTTP contract is [`openapi/openai-subset.v1.yaml`](../../../../course/contracts/openapi/openai-subset.v1.yaml) (`tools`, `tool_choice`, `response_format`, `tool_calls` deltas, `finish_reason: tool_calls`), and the conformance cases are `tools.call`, `tools.stream`, and `tools.choice`. The solution has three parts. [`tl-engine/src/constrain.rs`](../../../../course/solutions/rust/crates/tl-engine/src/constrain.rs) holds `regex_to_dfa` (the regex subset to a canonical byte DFA: Thompson NFA, subset construction over byte classes, dead states removed, Moore minimization, breadth-first numbering), `json_schema_to_regex`, `TokenIndex` (token masks over a byte trie), `Constraint`, `constrained_sample`, and `Json`, an order-keeping JSON reader that writes Python's compact JSON. [`tl-serve/src/tools.rs`](../../../../course/solutions/rust/crates/tl-serve/src/tools.rs) holds `parse_tool_request` (validating `tools`, `tool_choice`, `response_format`), `render_prompt`, `grammar` per `tool_choice`, the streaming `ToolParser`, and the OpenAI `tool_calls` message and deltas. And [`server.rs` as of this section](../../../../course/solutions/history/L10.9/rust/crates/tl-serve/src/server.rs) (there is also a [later version](../../../../course/solutions/rust/crates/tl-serve/src/server.rs)) has the chat route read the tool fields, decode a request with a grammar under its `Constraint` (`constrained_generate`), and answer with `tool_calls`.

> **Learning objectives**
> - Compile a regex to a canonical minimized DFA by hand and explain why canonical numbering makes two implementations' tables identical.
> - Translate a JSON-schema subset, including optional object members, into a regex over compact JSON.
> - Compute a token mask over a byte trie and sample from masked logits with a correct logprob.
> - Choose the grammar for each `tool_choice`, including the lazy trigger in `auto` mode.
> - Stream tool-call deltas from a parser whose output does not depend on where tokens split the text.

## From regex to a canonical DFA

The core question is whether "valid output" can be checked one token at a time. If the allowed outputs form a regular language, a DFA tells you after every byte whether the text so far can still be completed. So the grammar is a regex, compiled to a DFA over bytes. Some notation:

| Symbol | Meaning | Type |
|---|---|---|
| $\Sigma$ | the 256 byte values | alphabet |
| $\delta(s, b)$ | the state after byte $b$ from state $s$; $\bot$ (DEAD) when no accepting state is reachable | `i32` |
| $A$ | the accepting states | `bool[n]` |
| $w_t$ | the bytes of token $t$ | `Vec<u8>` |
| $\delta^*(s, w)$ | $\delta$ applied byte by byte; $\bot$ stays $\bot$ | `i32` |

The pipeline is the Python reference's, step for step. First, parse the subset into a tree of byte sets: literals, escapes, the classes `\d \w \s` and their negations, `.`, brackets, groups, `|`, and `* + ? {m} {m,} {m,n}`. Thompson's construction turns each node into a small NFA fragment glued with epsilon edges. The subset construction then runs over **byte classes** (sets of bytes that every edge treats alike) instead of 256 bytes, so a JSON string pattern with its UTF-8 ranges stays small.

Next, remove the states from which no accepting state is reachable; their transitions become $\bot$. Moore-minimize: split blocks until every state of a block moves to the same block on every class. Finally, number the blocks breadth-first from the start state, taking bytes 0 to 255 in order.

That last step matters more than it looks. Two different DFAs can recognize the same language, and only a canonical form (dead states removed, minimized, numbered breadth-first with bytes in order) makes two patterns with the same language get identical tables. That is what lets the Rust tables equal the Python reference's, and so the masks equal too.

## From a schema to a regex

A JSON schema is a regex if it is bounded. The JSON-schema subset of `constrain.pyi` maps to a regex over **compact** JSON: strings (well-formed UTF-8, the eight escapes, `\uXXXX`), integers, numbers, booleans, null, `enum`/`const` (each value as Python's compact `json.dumps`, escaped), arrays with `minItems`/`maxItems`, objects whose properties appear **in the order given** (required ones always and optional ones possibly), and `anyOf`. Every other keyword is refused with 422, never ignored.

Property order is part of the language, so the schema has to be read with a JSON reader that keeps key order. Properties in a fixed order, bounded arrays, and finitely nested objects are all regular. Arbitrary JSON objects are not, which is why `response_format: json_object` has a nesting limit of 3.

## Masks over a byte trie

A token is many bytes. Token $t$ is allowed in state $s$ when $\delta^*(s, w_t) \ne \bot$; EOS is allowed exactly when $s \in A$; a special token or an empty token never.

Walking every token's bytes per state costs $O(\sum_t |w_t|)$. A trie over the vocabulary shares the walk between tokens with a common prefix and prunes a whole subtree at the first $\bot$. Masks are cached per state and shared by every request using the same grammar.

The mask must act before sampling. Sampling runs the model runner's `sample` on the logits with every disallowed token at $-\infty$, so every sampling knob still applies, but only to allowed tokens; checking after sampling would simply fail. The logprob is renormalized over the allowed tokens.

## Tool calls as a language

The model writes calls as `<tool_call>{"name":"<name>","arguments":<object>}</tool_call>`, the format small instruct models are trained on, with calls separated by `\n`. Seen this way, `tool_choice` is a choice of language. The table shows the grammar each value selects.

| `tool_choice` | grammar |
|---|---|
| `{"type":"function","function":{"name":"f"}}` | exactly one call of `f`, from the first token |
| `"required"` | one call of any tool, or up to 4 with `parallel_tool_calls` |
| `"auto"` (default with tools) | free text; once the output ends with `<tool_call>`, the rest of one call |
| `"none"` | free text (or a JSON object with `response_format: json_object`, nested at most 3 levels: the language of all JSON objects is not regular) |

The tools are described to the model in a system message, appended to the first system message if there is one. An assistant turn's `tool_calls` are written back as `<tool_call>` blocks, and a `tool` message as a `<tool_response>` user turn, so a conversation renders the way it was generated.

## Streaming the answer

The client sees a stream, not a grammar. The parser keeps a small state machine. Outside a call it emits text but holds back a suffix that could be the start of `<tool_call>`. After the opener it waits for the header `{"name":"...","arguments":` and emits the call's id and name. Inside the arguments it tracks brace depth and string escapes and passes every new piece on as an `arguments` fragment until the value closes; then it consumes `}</tool_call>`.

Ids are `call_<request id>_<index>`. A call's start is the chunk `{"tool_calls":[{"index":i,"id":...,"type":"function","function":{"name":...,"arguments":""}}]}`, and later chunks are `{"tool_calls":[{"index":i,"function":{"arguments":"<fragment>"}}]}`. `finish_reason` is `tool_calls` when any call was made.

## Working it by hand

Four examples cover the automaton, the mask, the schema translation, and the parser.

**The automaton.** Take `(cat|car|dog)s?` and number breadth-first from state 0 with bytes in order. `c` (0x63) gets id 1 and `d` (0x64) id 2. From 1, `a` gives 3; from 2, `o` gives 4. From 3, `r` and `t` both reach the state after a whole word, and so does `g` from 4: minimization merges "car", "cat", and "dog" into state 5 (accepting). `s` from 5 gives 6 (accepting, no transitions). That is seven states, accepting {5, 6}, with eight live edges.

**A mask.** Pattern `ab?`, vocabulary `[a, b, c, EOS]`, logits `[0, 1, 5, 2]`. In state 0 only `a` walks live, so greedy takes `a` although `c` has the largest logit, with logprob $\ln 1 = 0$. In the state after `a` (accepting), `b` and EOS are allowed, and greedy takes EOS ($2 > 1$) with logprob $2 - \ln(e^1 + e^2)$.

**A schema.** The conformance tool's parameters `{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}` become `\{\"city\":"(?:...)*"\}`: the escaped key, a colon, and the JSON string pattern. Optional members nest. Two optional integers `x` and `y` give `\{(?:\"x\":INT(?:,\"y\":INT)?|(?:\"y\":INT|))\}`, so `{}`, `{"x":1}`, `{"y":2}`, and `{"x":1,"y":2}` match and `{"y":2,"x":1}` does not.

**A streamed call.** The pieces are `Sure.<tool_`, `call>{"name":"get_weather","argu`, `ments":{"city":"Pa`, and `ris"}}</tool_call>`. Each piece produces these events:

| Piece | Events |
|---|---|
| 1 | content `Sure.` (`<tool_` held back) |
| 2 | nothing (the header is incomplete) |
| 3 | call 0 starts (`call_req7_0`, `get_weather`); arguments `{"city":"Pa` |
| 4 | arguments `ris"}`; `}</tool_call>` consumed |

Folded together: content `Sure.`, one call with arguments `{"city":"Paris"}`, and `finish_reason` `tool_calls`.

## Designing the interface

The automaton lives in the engine crate; the OpenAI surface lives in the server crate.

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

// rust/crates/tl-serve/src/server.rs, taken over from the metrics section: its public API is unchanged
// (ServeConfig, spawn, run, ServerHandle); POST /v1/chat/completions accepts tools,
// tool_choice, response_format, and parallel_tool_calls.
```

The solution declares `pub mod constrain;` in `tl-engine/src/lib.rs` and `pub mod tools;` in `tl-serve/src/lib.rs`, and takes over `server.rs` from the metrics section, keeping all of its behaviour; `response_format` `json_schema` stays 422.

The chat handler calls `parse_tool_request` on the raw body, before the generic parser, so property order survives. It renders with `render_prompt`, builds one `TokenIndex` per (grammar, model) and caches it, runs each sequence through a `Constraint` (from the first token, or from the trigger in `auto` mode), and turns the text through `ToolParser` into `delta_json` chunks or `message_json`. A request with a grammar runs on the engine thread outside the batch, as the speculative path does: its own KV blocks through `spec::RunnerTarget`, `constrained_sample` per token, and the request's own generator. A grammar that is complete with no token left to allow (a vocabulary without EOS) ends the answer with `stop`.

## Building it step by step

### Reasoning from first principles

The design rests on seven observations, most of which you have already met.

1. **"Valid output" must be checkable one token at a time.** If the allowed outputs form a regular language, a DFA tells after every byte whether the text so far can still be completed. So the grammar is a regex, compiled to a DFA over bytes.
2. **Masks must equal the Python reference's.** Two different DFAs can recognize the same language. Only a canonical form makes the Rust and Python tables identical, and so the masks identical.
3. **A JSON schema is a regex if it is bounded.** Properties in a fixed order, bounded arrays, and finitely nested objects are regular; arbitrary JSON objects are not, which is why `json_object` has a depth limit of 3.
4. **A token is many bytes.** Token $t$ is allowed in state $s$ exactly when the walk over all of $w_t$ stays live. Doing that per token per state is expensive, so a trie shares common prefixes and prunes a subtree at its first dead byte, and the result is cached per state.
5. **The mask must act before sampling.** Setting disallowed logits to $-\infty$ and then calling the ordinary `sample` means every sampling knob still applies, but only to allowed tokens.
6. **`tool_choice` is a choice of language.** Named: one call of that function. Required: one or more calls of any tool. Auto: free text, then a call once the model writes the opener. None: free text.
7. **The client sees a stream, not a grammar.** A parser that holds back a possible `<tool_call>` prefix and tracks JSON depth can turn the text into content, call starts, and argument fragments, however tokens split it.

### The canonical DFA

`regex_to_dfa` in [`constrain.rs`](../../../../course/solutions/rust/crates/tl-engine/src/constrain.rs) parses, builds a Thompson NFA, and runs the subset construction over byte classes, each a set of bytes every NFA edge treats alike. Then it removes states that cannot reach acceptance, refusing a pattern whose start state is dead:

```rust
// live: reachable backwards from an accepting state
let mut live = vec![false; n];
let mut stack: Vec<usize> = (0..n).filter(|&s| acc[s]).collect();
// ... reverse-edge DFS marks every state that can reach acceptance
if !live[0] {
    return err(format!("regex: {pattern:?} matches nothing"));   // an empty language: refused, never served
}
// ... every edge into a dead state becomes DEAD
```

Moore minimization refines blocks until a round splits nothing:

```rust
// start with two blocks: accepting and not
let mut block: HashMap<usize, usize> = alive.iter().map(|&s| (s, acc[s] as usize)).collect();
loop {
    // a state's signature: its block, then the block it reaches on every class
    let sig: HashMap<usize, Vec<i64>> = alive.iter().map(|&s| {
        let mut v = vec![block[&s] as i64];
        v.extend(moves[s].iter().map(|&t| if t == DEAD { -1 } else { block[&(t as usize)] as i64 }));
        (s, v)
    }).collect();
    let mut keys: Vec<&Vec<i64>> = sig.values().collect();
    keys.sort();
    keys.dedup();
    // ... new block = index of the state's signature
    let stable = keys.len() == before.len();   // no block split: the partition is final
    block = nb;
    if stable {
        break;                                 // one refinement is not enough in general: loop to the fixpoint
    }
}
```

Finally the blocks are numbered breadth-first from the start block, trying bytes 0 to 255 in order, and the table is written out over all 256 bytes:

```rust
let mut order: HashMap<usize, usize> = HashMap::from([(block[&0], 0)]);
let mut q = VecDeque::from([block[&0]]);
while let Some(b0) = q.pop_front() {
    let s = rep_of[&b0];                       // any member: all members of a block move alike
    accept_of_block.push(acc[s]);
    for byte in 0..256 {
        let t = moves[s][byte_class[byte]];
        if t == DEAD { trans.push(DEAD); continue; }
        let bt = block[&(t as usize)];
        let next = order.len();
        // first time a block is reached, it gets the next id: BFS order, bytes ascending
        let id = *order.entry(bt).or_insert_with(|| { q.push_back(bt); next });
        trans.push(id as i32);
    }
}
```

In the hand-worked `(cat|car|dog)s?`, `c` (0x63) is met before `d` (0x64) from state 0, so they get ids 1 and 2; the three word-final states share one signature (accepting, `s` to the same block) and merge into state 5.

### Schemas to regexes

Most schema types translate directly. The object case is the interesting one. It builds the member sequence from the end, carrying two strings: `first`, what may follow when nothing has been written yet, and `after`, what may follow once something has, so that each member needs a leading comma:

```rust
let mut members = Vec::new();
for (k, v) in props {                          // properties in the order given: order is part of the language
    members.push((format!("{}:{}", escape(&json_quote(k)), json_schema_to_regex(v)?), required.contains(&k.as_str())));
}
let (mut after, mut first) = (String::new(), String::new());
for (m, req) in members.iter().rev() {
    first = if *req { format!("{m}{after}") } else { format!("(?:{m}{after}|{first})") };
    after = if *req { format!(",{m}{after}") } else { format!("(?:,{m})?{after}") };
}
Ok(format!(r"\{{{first}\}}"))
```

With two optional members `x` and `y`, this yields the hand-worked `\{(?:"x":INT(?:,"y":INT)?|(?:"y":INT|))\}`: a comma appears only between two written members.

Arrays become `item(?:,item){lo-1,hi-1}`, wrapped in an optional group when `minItems` is 0. `enum` and `const` values are written with `dumps` (Python's compact `json.dumps`) and then `escape`d, so `2.5` matches only `2.5`. Any keyword outside the subset is an error naming it, which the server turns into 422.

### Masks over the trie

With a DFA in hand, the token index computes one mask per state by walking the vocabulary trie:

```rust
fn build(&self, state: usize) -> MaskEntry {
    let mut mask = vec![false; self.vocab_size];
    let mut next = vec![DEAD; self.vocab_size];
    let mut stack = vec![(0usize, state as i32)];          // (trie node, DFA state after its bytes)
    while let Some((node, s)) = stack.pop() {
        for &(b, child) in &self.trie[node].children {
            let t = self.dfa.trans[s as usize * 256 + b as usize];   // from the state reached so far, not from `state`
            if t == DEAD {
                continue;                                  // every token below this byte is dead too
            }
            for &id in &self.trie[child].ids {             // tokens that end exactly here are allowed
                mask[id as usize] = true;
                next[id as usize] = t;
            }
            stack.push((child, t));
        }
    }
    if let Some(e) = self.eos_id {
        if self.dfa.accept[state] {                        // EOS only where the output is a whole string
            mask[e as usize] = true;
            next[e as usize] = state as i32;
        }
    }
    (mask, next)
}
```

Specials (`None` in the vocabulary), empty tokens, and EOS are never inserted into the trie in `TokenIndex::new`, so they can only be allowed by the EOS rule. Each mask and its next-state table are cached per state behind a mutex, shared by every request with the same grammar and model.

`Constraint` holds one request's state. `advance` refuses a token whose next state is DEAD and marks `done` on EOS, after which `mask` is all false. Sampling is the ordinary sampler on masked logits:

```rust
pub fn constrained_sample(logits: &[f32], c: &mut Constraint, p: &SamplingParams, prompt: &[u32], output: &[u32], rng: &mut Pcg32) -> Result<(u32, f64), ConstrainError> {
    let masked = apply_mask(logits, &c.mask())?;   // -inf where forbidden; an all-false mask is an error
    let (tok, lp) = sample(&masked, p, prompt, output, rng);   // the logprob is over the allowed tokens only
    c.advance(tok)?;
    Ok((tok, lp))
}
```

On the hand-worked mask, only `a` is live in state 0, so the masked logits are `[0, -inf, -inf, -inf]` and greedy takes `a` with logprob $\ln 1 = 0$, although `c` had logit 5.

### Choosing the grammar

The server side starts by picking a language. In [`tools.rs`](../../../../course/solutions/rust/crates/tl-serve/src/tools.rs):

```rust
pub fn grammar(r: &ToolRequest) -> Option<Grammar> {
    let open = escape(CALL_OPEN);
    let close = escape(CALL_CLOSE);
    let any_body = || format!("(?:{})", r.tools.iter().map(call_body_regex).collect::<Vec<_>>().join("|"));
    let calls = |body: &str| {
        let one = format!("{open}{body}{close}");
        if r.parallel && MAX_PARALLEL > 1 {
            format!("{one}(?:\\n{one}){{0,{}}}", MAX_PARALLEL - 1)   // up to 4 calls, newline-separated
        } else {
            one
        }
    };
    match (&r.choice, r.tools.is_empty()) {
        // named: only that tool's body, from the first token
        (ToolChoice::Named(n), false) => {
            let t = r.tools.iter().find(|t| &t.name == n)?;
            Some(Grammar { pattern: format!("{open}{}{close}", call_body_regex(t)), trigger: None })
        }
        (ToolChoice::Required, false) => Some(Grammar { pattern: calls(&any_body()), trigger: None }),
        // auto: free until the text ends with <tool_call>, then the rest of one call
        (ToolChoice::Auto, false) => Some(Grammar { pattern: format!("{}{close}", any_body()), trigger: Some(CALL_OPEN.to_string()) }),
        _ => match r.response_format {
            ResponseFormat::JsonObject => Some(Grammar { pattern: json_object_regex(JSON_OBJECT_DEPTH), trigger: None }),
            ResponseFormat::Text => None,
        },
    }
}
```

`call_body_regex` is `{"name":<the escaped quoted name>,"arguments":<the parameters' regex>}`, so the name is a literal in the language: under a named choice the model cannot even spell another function.

### Decoding with a grammar

The serve loop runs a grammar request on the engine thread outside the batch, over its own blocks through `RunnerTarget`:

```rust
fn constrained_generate(runner: &mut ModelRunner, c: &Constrained, req: &GenRequest, tok: &dyn Tokenizer, emit: &mut dyn FnMut(u32, f64, Option<FinishReason>) -> bool) -> Result<(), String> {
    let mut target = RunnerTarget::new(runner);
    let mut rng = stream(req.seed, PURPOSE_SAMPLE);                 // the request's own generator
    // constrained from the first token, unless there is a trigger (auto)
    let mut cons = c.trigger.is_none().then(|| Constraint::new(Arc::clone(&c.index)));
    // ...
    let finish = loop {
        if out.len() >= req.max_tokens { break FinishReason::Length; }
        let (t, lp) = match cons.as_mut() {
            // complete with nothing left to allow (no EOS in the vocabulary): the answer is over
            Some(k) if !k.mask().iter().any(|&m| m) => break FinishReason::Stop,
            Some(k) => constrained_sample(&row, k, &req.params, &req.prompt, &out, &mut rng).map_err(|e| e.to_string())?,
            None => sample(&row, &req.params, &req.prompt, &out, &mut rng),
        };
        // ... emit the previous token (one is held so the finish can ride on the last one); EOS ends with Stop
        if cons.is_none() && !triggered && !trigger.is_empty() {
            text.extend_from_slice(tok.token_bytes(t).unwrap_or(&[]));
            if let Some(at) = text.windows(trigger.len()).position(|w| w == trigger) {
                // the trigger may end mid-token: start the DFA on the bytes after it
                let state = c.index.dfa.step(0, &text[at + trigger.len()..]).unwrap_or(DEAD);
                if state != DEAD {
                    cons = Some(Constraint { index: Arc::clone(&c.index), state, done: false });
                }
                triggered = true;
            }
        }
        // ... extend the target with t for the next row
    };
    // ... emit the held token with `finish`
}
```

### The streaming parser

The text that comes out has to become OpenAI deltas. `ToolParser::push` is a four-mode state machine. Outside a call it holds back only a real prefix of the opener:

```rust
Mode::Text => {
    if let Some(at) = self.pending.find(CALL_OPEN) {
        let before = self.pending[..at].to_string();
        // "\n" between two calls is markup, not content.
        if !(self.calls > 0 && before.trim().is_empty()) && !before.is_empty() {
            out.push(ToolEvent::Content(before));
        }
        self.pending.drain(..at + CALL_OPEN.len());
        self.mode = Mode::Header;
        continue;
    }
    // hold back the longest suffix that is a proper prefix of "<tool_call>"
    let keep = partial_suffix(&self.pending, CALL_OPEN);
    // ... emit the rest as content
}
```

`Header` waits until `parse_header` sees all of `{"name":"<n>","arguments":`, then emits `CallStart` with id `call_<request id>_<index>`; or it proves the text cannot be a header, in which case the held markup is content after all. `Args` tracks depth, strings, and escapes so that a `}` inside a string does not end the value, passing everything up to the value's end on as an `Arguments` fragment:

```rust
Mode::Args { mut depth, mut in_str, mut esc, mut started } => {
    // ...
    for (k, c) in self.pending.char_indices() {
        if in_str {
            match c { _ if esc => esc = false, '\\' => esc = true, '"' => in_str = false, _ => {} }
        } else {
            match c {
                '"' => in_str = true,
                '{' | '[' => { depth += 1; started = true; }
                '}' | ']' if depth == 0 => { end = Some(k); break; }        // a scalar ended at the call's own `}`
                '}' | ']' => { depth -= 1; if depth == 0 { end = Some(k + 1); break; } }
                // ...
            }
        }
    }
    // end found: emit the fragment up to it and go to Tail; otherwise emit everything and keep the counters
}
```

Trace the hand-worked stream. Piece 1 ends with `<tool_`, a proper prefix of the opener, so only `Sure.` is emitted. Piece 2 completes the opener but not the header. Piece 3 completes the header (`CallStart`) and passes `{"city":"Pa` on with depth 1, inside a string. Piece 4's `ris"}` closes the string and the object (depth 0: end), and `Tail` consumes `}</tool_call>`. Because all state is carried in the mode, the result does not depend on where the pieces were cut.

`message_json` writes `content` as `null` when it is empty, never `""`, so clients treat a calling turn as a call; `finish_reason` is `tool_calls` whenever a call was made.

The full solution is [`constrain.rs`](../../../../course/solutions/rust/crates/tl-engine/src/constrain.rs), [`tools.rs`](../../../../course/solutions/rust/crates/tl-serve/src/tools.rs), and the serve loop's `constrained_generate` in [`server.rs`](../../../../course/solutions/history/L10.9/rust/crates/tl-serve/src/server.rs).

### Why this design

Compiling is paid once per (grammar, model) and cached. A mask is built at most once per DFA state, in $O(\text{trie nodes reached})$, which is far less than $\sum_t |w_t|$ because dead subtrees are pruned at their first byte. Per token, the hot path is a cached mask lookup, an $O(V)$ masking pass, and one next-state lookup.

Why a DFA rather than a backtracking regex or a JSON parser? A backtracking engine cannot answer "can this prefix still match?" cheaply for every token. An incremental JSON parser can, but not for an arbitrary schema's constraints (property order, enums, bounds). The DFA answers prefix viability in one table lookup per byte.

Why canonical numbering? Only then can the Rust tables be compared to the Python reference's state for state, which is what proves the port. It also makes the per-state mask cache keys stable.

Two tradeoffs are accepted. The language must be regular, so recursive schemas are refused and `json_object` nests at most 3 levels; context-free grammars (XGrammar, llguidance) lift that limit at the cost of a stack. And a grammar request runs outside the batch, holding the engine thread until it finishes.

## What can go wrong

The automaton is checked against what the Python reference produces: the regex and canonical DFA of 13 schemas and 8 patterns; the masks along three seeded paths of each through a 123-entry vocabulary (printable bytes, multi-byte and UTF-8 tokens, a special, an empty token, EOS); Python's compact `json.dumps` of 19 JSON texts; and the schemas and patterns the subset refuses. The reference's own independent derivative-based oracle agrees too. A correct solution handles all of these cases:

| Case | Checks | Why it matters downstream |
|---|---|---|
| The hand DFA | the worked seven states and eight edges; `step` and `matches` | Rust and Python agree on the canonical form |
| DFA equals Python | regex text per schema; states, accepting flags, every transition per pattern | identical tables, so identical masks |
| Masks equal Python | allowed ids and completeness at every step of every recorded path | what the sampler sees is Python's |
| Subset errors | 15 bad patterns and 11 bad schemas refused | a 422, never a silently different language |
| `dumps` equals Python | key order, `-0`, float repr, escapes, raw non-ASCII | enum values the model can actually produce |
| Token index rules | specials, empty tokens, EOS rules, forbidden advances | the edge cases of the mask |
| The hand mask | the worked mask and logprobs; an all-false mask refused | the sampler sees the mask first |
| Constrained output always parses | 25 random walks per schema parse and validate | the guarantee itself |
| The `json_object` grammar | objects up to depth 3; arrays, scalars, broken objects refused | `response_format: json_object` |
| Parsing the tool request | defaults, a named choice, 400s and the 422 naming the keyword | the request surface of `openai-subset.v1` |
| Grammar by `tool_choice` | named, required, parallel off, auto with its trigger, none | `tools.choice` |
| Rendering messages with tools | tools in the system message; calls and results rendered back; through the chat template | multi-turn [agent loops](../../ai-platform-engineering/13-agent-sdk/03-agent-loop.md) |
| `vocab_bytes` marks specials | specials map to None | chat markers are never generated |
| The hand stream | the worked stream, fragment by fragment | `tools.stream` |
| Holding back only real markup | `a<b`, a trailing `<tool`, an opener without a header | no markup in content, no content lost |
| Parallel calls | indices, ids, the newline between calls | parallel calls |
| Split invariance | every 2-way split and 200 random splits fold to the same answer | tokens cut text anywhere |
| Message and delta shapes | the message and both delta shapes, arguments as a string | what the agent SDK and the `openai` client parse |
| A forced call from a random model | 40 random walks under a named choice give one valid `get_weather` call | a valid call even from a tiny random model |
| The live serve loop | a forced call comes back as schema-valid `tool_calls` with `finish_reason` `tool_calls`, from the prompt `render_prompt` builds; streamed deltas reassemble to the same call; `tool_choice` none calls nothing; tool-field errors keep the OpenAI shape | the agent SDK and the `tools.*` conformance cases read exactly this |

The pitfalls sort into automaton bugs, schema bugs, mask bugs, and surface bugs.

In the automaton, stopping minimization after one refinement gives equal languages with different tables, so the masks differ from Python's; Moore refinement loops until a round splits no block. An empty language that is not refused produces a request that can never finish; `regex_to_dfa` refuses a pattern whose start state is not live.

In the schema translation, sorting properties by name forces the model to write keys in an order the schema did not ask for; `Json` keeps key order and members are built in the order given. Leaving out the comma before a later optional member makes `{"x":1"y":2}` "valid" though it does not parse; the `first` / `after` construction puts a comma between every two written members. Enum values inserted unescaped let `2.5` match `2x5`, and quotes break the pattern; each value is `dumps`ed, then `escape`d. A `maxItems` off by one lets one item too many pass the grammar and fail the schema; the tail repeats `{lo-1, hi-1}` times after the first item. Floats printed unlike Python (`1.5e-05` written `0.000015`) make the enum value unreachable; `python_float` reproduces Python's `repr` for floats. Accepting parameters outside the subset silently weakens the schema; every unknown keyword is an error naming it, returned as 422.

> **Warning:** The mask bugs break the guarantee itself. Walking a token's later bytes from the wrong state allows multi-byte tokens that leave the language; the trie walk carries the state after each byte. Allowing EOS before a whole string lets outputs end mid-object; EOS is allowed only when `dfa.accept[state]`. Tokens still allowed after EOS let generation continue past the end of the language; `Constraint::mask` is all false once `done`. And a sampler that sees the unmasked logits only checks the constraint after the fact, and fails; `constrained_sample` masks before calling `sample`.

On the API surface, a named `tool_choice` that allows every tool lets the model call the wrong function; the named grammar contains only that tool's body, name included. Dropping argument fragments makes streamed arguments arrive incomplete; `Args` emits everything it has read, up to the value's end. Not holding back a partial `<tool_call>` shows `<tool_` in the user-visible text; `Text` holds back `partial_suffix(pending, CALL_OPEN)`. And `content: ""` next to `tool_calls` makes clients treat the turn as text; `collect` returns `None` for empty content and `message_json` writes `null`.

## Where this leads

This section is held to the Python automaton and schema translation of [constrained decoding](../p08-inference/07-constrained-decoding.md). It uses `sample` on the masked logits from [the model runner and sampler](01-model-runner-and-sampler.md), `Template::render_messages` and `ApiError` from [the tokio server](05-openai-server-on-tokio.md), and `Tokenizer::token_bytes` from [the Rust fast BPE](../p01-tokenizers/05-rust-fast-bpe.md) for the vocabulary's bytes. It takes over the serve loop of [serving metrics and tracing](07-serving-metrics-and-tracing.md), so tool requests are recorded and traced through its calls.

The serve loop is now this section's `server.rs`, so [metrics and log correlation](../../systems/04-observability/02-metrics-and-log-correlation.md) in *Systems* runs exactly this engine. In *AI Platform Engineering*, [types and the provider](../../ai-platform-engineering/13-agent-sdk/01-types-and-provider.md) parses these `tool_calls` deltas over HTTP, and the agent system runs SmolLM2-135M-Instruct through this engine to call its tools. In *Software Craftsmanship*, [the API v1 to v2 migration](../../software-craftsmanship/07-maintenance/02-api-v1-to-v2.md) takes this `server.rs` over to serve API v1 and v2 side by side. The conformance cases `tools.call`, `tools.stream`, and `tools.choice` exercise exactly this surface.

Production systems differ mainly in scale and expressiveness. Outlines uses the same automaton approach, precomputed over the whole vocabulary, where this solution caches DFA masks per state. XGrammar and llguidance go beyond a regular subset of JSON schema to context-free grammars (recursive JSON) with a persistent stack and fast masks. vLLM's tool parsers replace one `<tool_call>` text format with one parser per model family's call format. And llama.cpp's grammar triggers generalize the lazy trigger in `auto` mode to grammars activated by a token or pattern mid-generation.

### Further reading

- Hopcroft, Motwani, Ullman, *Introduction to Automata Theory*, ch. 2 to 4.
- Willard and Louf, [Efficient Guided Generation](https://arxiv.org/abs/2307.09702) (free), and [Outlines](https://github.com/dottxt-ai/outlines).
- [XGrammar](https://arxiv.org/abs/2411.15100) (free; [code](https://github.com/mlc-ai/xgrammar)) and [llguidance](https://github.com/guidance-ai/llguidance).
- [OpenAI function calling guide](https://platform.openai.com/docs/guides/function-calling) (free).
- [vLLM tool calling](https://docs.vllm.ai/en/latest/features/tool_calling.html) and [llama.cpp grammars](https://github.com/ggml-org/llama.cpp/tree/master/grammars).

## Key terms

- **Constrained decoding**: masking, at every step, every token that would take the output outside an allowed language.
- **Canonical DFA**: a DFA with dead states removed, Moore-minimized, and numbered breadth-first with bytes in order, so equal languages give identical tables.
- **Byte class**: a set of bytes that every NFA edge treats alike, used to keep the subset construction small.
- **Token mask**: the set of token ids whose full byte sequence keeps the DFA live from the current state, plus EOS in accepting states.
- **Byte trie**: a prefix tree over the vocabulary's bytes that shares mask walks between tokens and prunes dead subtrees.
- **`tool_choice`**: the request field that selects the call grammar: named, `required`, `auto`, or `none`.
- **Trigger**: in `auto` mode, the text `<tool_call>` whose appearance switches decoding from free text to the call grammar.

## Summary

- A minimized DFA numbered breadth-first over bytes in order is canonical: the Rust tables equal the Python reference's for every pattern and schema, so the masks are the same.
- A token is allowed when walking *all* its bytes stays live; EOS only where the output is a whole string; specials never.
- Whatever the logits, constrained output parses and satisfies its schema: that is why a random tiny model can still make a valid forced tool call.
- `tool_choice` decides the grammar: a named function is one call of it, `required` one to four calls, `auto` free text until the model writes `<tool_call>`, `none` free text.
- The parser streams argument fragments as they arrive and never leaks markup into the content, wherever the tokens split the text.

## Review questions

1. Why does the trie walk use the DFA state reached so far, not the state the mask is for, at each byte?
   <details><summary>Answer</summary>

   Token $t$ is allowed when $\delta^*(s, w_t) \ne \bot$: the bytes are applied one after another, each from the state the previous byte reached. Walking every byte from $s$ would test each byte in isolation and allow multi-byte tokens that leave the language. The stack carries `(node, state after this node's bytes)` exactly for this.

   </details>

2. Why does Moore minimization loop until no block splits, rather than refining once?
   <details><summary>Answer</summary>

   One refinement separates states whose successors are in different blocks of the old partition, but splitting a block can make two states that looked equal now differ, because their successors fall in the newly split blocks. Only at the fixpoint are all remaining states equivalent. Stopping early gives a correct but non-minimal DFA, whose tables differ from the Python reference's.

   </details>

3. A schema has two optional properties `x` and `y`. Why does the regex need both `first` and `after`, and what goes wrong with a single string?
   <details><summary>Answer</summary>

   Whether a member needs a leading comma depends on whether anything was written before it. `first` describes the rest when nothing has been written (no comma before the next member), `after` when something has (comma required). With a single string you either force a comma before the first member (`{,"y":2}`) or allow none between members (`{"x":1"y":2}`), which does not parse.

   </details>

4. In auto mode, why does the solution start the DFA on the bytes after the trigger instead of at state 0 after the trigger token?
   <details><summary>Answer</summary>

   Tokens do not respect markup boundaries: the token that completes `<tool_call>` may carry the first bytes of the call body too. `dfa.step(0, &text[at + trigger.len()..])` advances over those bytes, so the constraint starts where the text actually is. If they are already outside the call language, the state is DEAD and the answer stays free text.

   </details>

5. Why is the logprob of a constrained token computed on the masked logits?
   <details><summary>Answer</summary>

   The token was drawn from the masked distribution, so its probability is relative to the allowed tokens only; in the hand-worked mask, `a` was the only allowed token and was certain, logprob 0. Reporting the unmasked logprob would describe a distribution the sampler never used.

   </details>

6. Why is `response_format: json_object` limited to 3 levels of nesting?
   <details><summary>Answer</summary>

   The language of all JSON objects requires matching arbitrarily deep braces, which no finite automaton can do: it is not regular. Bounding the depth makes it regular, so it compiles to a DFA like any other grammar here. Lifting the limit needs a context-free grammar engine with a stack, such as XGrammar or llguidance.

   </details>
