<!-- ss:module L10.8 -->
# Speculative decoding in the engine

## Overview

| | |
|---|---|
| **Module** | `L10.8` · build · Rust · Pass 7 · 6 to 9 h |
| **You build** | `rust/crates/tl-engine/src/spec.rs`: prompt-lookup and n-gram drafts over the request's own context, the verifier (greedy acceptance, and M07.6's accept-or-resample at temperature > 0), KV rollback on rejection, and a generation loop over a `Target` |
| **Contract** | the Python specification you port: [`py/tinyllm/infer/spec.pyi`](../../../course/contracts/py/tinyllm/infer/spec.pyi) (L8.6) · sampling order and draws: [`spec/sampling.md`](../../../course/contracts/spec/sampling.md), [`spec/pcg32.md`](../../../course/contracts/spec/pcg32.md) · configuration `[engine].speculative`: [`config/runtime.schema.json`](../../../course/contracts/config/runtime.schema.json) · the gauge `tl.engine.spec_accept_rate`: [`otel/metrics.yaml`](../../../course/contracts/otel/metrics.yaml) |
| **Tests** | `course/tests/rust/l10_8.rs`, 14 tests (what they check: section 4) |
| **Needs** | `L8.6` your Python speculative decoding, the specification this port is held to ([chapter](../p08-inference/06-speculative-decoding.md)) · `L10.1` the sampler (`apply_penalties`, `distribution`, `sample`, `stream`) and `KvPool` ([chapter](01-model-runner-and-sampler.md)) · `L10.1` the Rust pool behind the rollback ([chapter](../p08-inference/08-paged-kv-block-pool.md)) · reading: `L10.2`, `L10.4` the scheduler and block manager it plugs into, `M07.6` rejection and residuals ([chapter](../../../math/07-probability-statistics/06-rejection-sampling-and-residual-distributions.md)) · or `--ref-deps` |
| **Used by** | `L10.5`'s runner-backed serving path (`[engine].speculative`); `MS-L10` checks greedy output is unchanged with prompt lookup on |
| **Milestone** | `MS-L10` |
| **Optional depth** | [Leviathan et al., Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (free); [Chen et al., Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (free); [Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding) (free) |

## Key Takeaways

- Prompt lookup copies what followed the most recent earlier occurrence of the context's suffix, trying long suffixes first; it costs no model and shines on repetitive text (`hand_example_prompt_lookup`, `prompt_lookup_matches_python_l8_6`).
- Greedy verification keeps drafts while they equal the target's argmax, so greedy output with speculation is token for token greedy output without it (`hand_example_greedy_verify`, `greedy_spec_equals_greedy_without_spec`).
- With sampling, accept-or-resample keeps the target's distribution exactly, whatever the draft proposes (`hand_example_sampled_step`, `sampled_spec_keeps_the_target_distribution`).
- The port draws exactly what L8.6 draws, in the same order (u_accept, then u_resample, both always; one draw for the bonus token), so its tokens equal your Python's on the same logits and seed (`verify_matches_python_l8_6`).
- A rejected draft's KV is given back: after every round the sequence holds exactly the blocks its cached positions need (`rollback_hand_example`, `blocks_conserved_across_rollbacks`).

## How to work this chapter

```bash
ss start L10.8               # stubs spec.rs
ss tests L10.8
ss check L10.8               # exit code is the verdict
ss check L10.8 --ref-deps    # only if L8.6, L10.1, or L10.1 is not passing yet
ss diff  L10.8
```

Declare `pub mod spec;` in `tl-engine/src/lib.rs`. Your engine turns `[engine].speculative = { draft = "prompt_lookup", k = 4 }` into a `SpecConfig` with `SpecConfig::parse`, and its step loop runs one round per sequence: `propose`, one forward over the pending token plus the draft, `verify_draft`, then `rollback` of the sequence's block table to the kept length.

---

## 1. Why now

Decoding one token costs one forward pass over the whole model, and at batch sizes your laptop can afford that pass is limited by reading the weights, not by arithmetic: checking five tokens costs about as much as producing one. If something cheap can guess the next few tokens, one target pass can verify all of them. In Python (`L8.6`) you proved the rule that makes this exact. Your Rust engine (`L10.5`) still decodes one token per pass, and the workloads it serves are full of repetition: code edits, retrieval answers quoting the context, JSON tool arguments that echo the prompt. This module ports the verifier to the engine, adds two drafts that need no second model (prompt lookup and an n-gram table over the request's own tokens), and gives back the KV of rejected drafts so speculation never leaks blocks.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $x_1 \ldots x_t$ | the context: prompt plus emitted tokens | `u32` ids |
| $d_1 \ldots d_m$ | the draft, $m \le k$ | `u32` ids |
| $p_i$ | the target's sampling distribution after context $+ d_1 \ldots d_{i-1}$ | `f64[V]` |
| $q_i$ | the draft's distribution for $d_i$ (one-hot for a deterministic draft) | `f64[V]` |
| $u_a, u_r$ | the two uniforms drawn per verified position | `f64` in $[0, 1)$ |
| $B$ | tokens per KV block | `usize` |

### 2.1 Drafts from the context

**Prompt lookup**: for $n$ from `max_ngram` down to `min_ngram`, take the last $n$ tokens and find their most recent earlier occurrence (a start $i < t - n$); the draft is the next $k$ tokens after it, which may run into the suffix itself. The first $n$ with a continuation wins. **N-gram**: for each draft token, count what followed the last $m$ tokens anywhere earlier in the context (drafted tokens included), for $m$ = `max_ngram - 1` down to 1; take the most frequent, ties to the lowest id; stop when nothing matches. Both are deterministic, so $q_i$ is one-hot.

### 2.2 One round

The target extends its cache with the pending token (the previous round's last emitted token, or the prompt in the first round) followed by $d_1 \ldots d_m$, in one forward pass, giving $m + 1$ rows of logits: row $i$ scores position $i$ of the draft, row $m$ the token after the whole draft. The verifier walks the draft:

- **Greedy** ($T = 0$): accept $d_i$ while it equals the argmax of row $i$ (penalties applied, seeing the tokens accepted so far); the first mismatch emits the argmax instead and stops; after $m$ acceptances emit the argmax of row $m$. No draws.
- **Sampled** ($T > 0$): draw $u_a$ then $u_r$ (both always, so the number of draws does not depend on the outcome); accept $d_i$ when $u_a < p_i(d_i)/q_i(d_i)$; otherwise emit a draw from the residual $r = \mathrm{normalize}(\max(0, p_i - q_i))$ with $u_r$ and stop. After $m$ acceptances emit L10.1's `sample` of row $m$ (one draw).

Every round emits between 1 and $m + 1$ tokens, never more than the request has left: the draft is shortened to `max_new - emitted - 1`.

### 2.3 Why the distribution is exact

For a proposed $x \sim q$: $P(\text{emit } x) = q(x)\min(1, p(x)/q(x)) + (1 - \sum_y \min(p(y), q(y)))\,r(x)$. Since $\min(q, p) + \max(0, p - q) = p$ pointwise and the rejected mass is exactly the residual's normalizer, the sum is $p(x)$. For a one-hot draft this is: keep $d$ with probability $p(d)$, else sample from $p$ with $d$ removed and renormalized.

### 2.4 Rollback

The forward wrote KV for all $m$ draft positions. After keeping $n$ of them, the cache must hold the base length plus the pending tokens plus $n$: the blocks past $\lceil \text{keep}/B \rceil$ are released (one reference each: a block shared through the prefix cache survives for its other owner) and the last kept block's fill is set to what it still holds. A block a rejected draft filled must never be registered in the prefix index.

## 3. Worked example by hand

**Prompt lookup** (test `hand_example_prompt_lookup`). Context `[5 6 7 8 5 6]`, $k = 3$, $n$ from 3 down to 1. The 3-suffix `[8 5 6]` never occurred earlier. The 2-suffix `[5 6]` occurred at position 0, so the draft is what followed it: `[7 8 5]`.

**A greedy round** (test `hand_example_greedy_verify`). Four ids, draft `[2 1]`, rows:

| Row | Logits | Argmax |
|---|---|---|
| 0 | `[0 1 3 0.5]` | 2 = $d_1$: accept |
| 1 | `[0 1 0.5 2]` | 3 $\ne d_2 = 1$: emit 3, stop |
| 2 | `[5 0 0 0]` | (not reached) |

Emitted `[2 3]`, one draft accepted, no draws. With draft `[2 3]` both are accepted and row 2's argmax 0 is the bonus: `[2 3 0]`. The logprob of the first emitted 2 is $3 - \ln(e^0 + e^1 + e^3 + e^{0.5})$.

**A sampled step** (test `hand_example_sampled_step`). $p = [0.2, 0.5, 0.3]$, the draft proposed $x = 0$ from $q = [0.6, 0.2, 0.2]$; accept with probability $0.2/0.6 = 1/3$. With $u_a = 0.3$: accept. With $u_a = 0.5$: reject; the residual is $\max(0, p - q) = [0, 0.3, 0.1]$, normalized $[0, 0.75, 0.25]$; $u_r = 0.8$ walks the running sum $0, 0.75, 1.0$ and lands on id 2.

**A rollback** (test `rollback_hand_example`). $B = 4$; 10 cached positions fill three blocks (4, 4, 2). Keeping 5 positions needs $\lceil 5/4 \rceil = 2$ blocks: block 3 is released, block 2's fill becomes 1.

## 4. The interface

```rust
// rust/crates/tl-engine/src/spec.rs
pub enum Draft { None, Ngram, PromptLookup }
pub struct SpecConfig { pub draft: Draft, pub k: usize, pub max_ngram: usize, pub min_ngram: usize }
impl SpecConfig { pub fn parse(draft: &str, k: i64) -> Result<SpecConfig, SpecError>; }   // "none" | "ngram" | "prompt_lookup", 1 <= k <= 16
pub fn prompt_lookup(ctx: &[u32], k: usize, max_ngram: usize, min_ngram: usize) -> Vec<u32>;
pub fn ngram_draft(ctx: &[u32], k: usize, n: usize) -> Vec<u32>;
pub fn propose(cfg: &SpecConfig, ctx: &[u32], k: usize) -> Vec<u32>;
pub fn residual(p: &[f64], q: &[f64]) -> Vec<f64>;
pub fn sample_dense(p: &[f64], u: f64) -> usize;
pub fn speculative_step(p: &[f64], q: &[f64], x: usize, u_accept: f64, u_resample: f64) -> Result<(usize, bool), SpecError>;
pub fn dense_distribution(logits: &[f32], p: &SamplingParams, prompt: &[u32], output: &[u32]) -> Vec<f64>;
pub struct Verified { pub tokens: Vec<u32>, pub logprobs: Vec<f64>, pub n_accepted: usize }
pub fn verify_draft(rows: &[Vec<f32>], draft: &[u32], draft_probs: Option<&[Vec<f64>]>, p: &SamplingParams,
                    prompt: &[u32], output: &[u32], rng: &mut Pcg32) -> Result<Verified, SpecError>;
pub fn blocks_for(tokens: usize, block_tokens: usize) -> usize;
pub fn rollback(pool: &mut KvPool, table: &mut Vec<u32>, keep: usize) -> Result<usize, SpecError>;
pub trait Target { fn cached(&self) -> usize; fn extend(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, SpecError>; fn truncate(&mut self, len: usize) -> Result<(), SpecError>; }
pub struct SpecStats { pub drafted: usize, pub accepted: usize, pub target_calls: usize }   // accept_rate()
pub fn generate(target: &mut dyn Target, cfg: &SpecConfig, prompt: &[u32], p: &SamplingParams,
                seed: u64, max_new: usize, eos: &[u32]) -> Result<Generated, SpecError>;   // given: generate_stream, never stopped
pub fn generate_stream(target: &mut dyn Target, cfg: &SpecConfig, prompt: &[u32], p: &SamplingParams,
                       seed: u64, max_new: usize, eos: &[u32],
                       on_token: &mut dyn FnMut(u32, f64) -> bool) -> Result<Generated, SpecError>;
```

You write `generate_stream`: `generate`'s loop, handing each emitted token and its logprob to `on_token` as soon as its round is verified, and returning at once with what it has when `on_token` answers false. The serve loop streams through it and stops a request whose client left. `RunnerTarget` (given) is the production `Target`: the runner returns one logits row per sequence, so it feeds every id as its own one-token sequence over the request's blocks, at its own position, in one forward.

### What the tests check

`course/fixtures/L10.8/spec_golden.json` records what the Python reference of L8.6 answers on 120 verification cases (exact f32 logits, drafts that agree for a while, optional draft distributions, every sampling knob, 64-bit seeds) and 150 prompt-lookup contexts (`course/oracle/L10.8/spec_golden.py`). The generation tests run a fake target whose logits are a fixed function of the context.

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_prompt_lookup` | unit | section 3's draft, a continuation into the suffix, no draft | the worked example |
| `prompt_lookup_matches_python_l8_6` | differential | 150 contexts equal to `PromptLookupDraft` | most recent occurrence, longest suffix first |
| `ngram_draft_hand_example` | unit | counts, ties to the lowest id, longest context first, back-off, stop | the second draft |
| `config_parses_runtime_values` | boundary | the three drafts; `k` 0 and 17 and a typo refused | a bad runtime.toml fails at startup |
| `hand_example_greedy_verify` | unit | section 3's round, no draws, logprobs, the bonus token | the greedy rule |
| `hand_example_sampled_step` | unit | accept, reject, residual, a q that never proposed x | M07.6 by hand |
| `verify_matches_python_l8_6` | differential | 120 cases: emitted tokens, n_accepted, and the generator's next u32 | the port draws exactly what Python draws |
| `verify_rejects_bad_shapes` | boundary | rows, draft distributions, out-of-vocabulary drafts | an error, not a panic, in the step loop |
| `greedy_spec_equals_greedy_without_spec` | differential | every draft kind, three prompts: the greedy tokens, cached positions, fewer passes with prompt lookup | speculation changes speed only |
| `eos_and_max_tokens_end_generation` | boundary | exactly `max_new` tokens; EOS inside an accepted run stops there | requests end where they should |
| `sampled_spec_keeps_the_target_distribution` | statistical | 4000 seeds: the first token's chi-square against the target softmax at p > 1e-3 | sampled output is unchanged in law |
| `rollback_hand_example` | unit | section 3's rollback, a full block, keep 0, `blocks_for` | the worked example |
| `rollback_drops_one_reference_only` | fault | a block shared with another sequence survives | the prefix cache is never corrupted |
| `blocks_conserved_across_rollbacks` | property | 200 seeded requests with many rejections on a real pool: blocks match positions, nothing leaks | weeks of speculation do not shrink the pool |
| `generate_stream_hands_over_each_token_and_stops_when_told` | conformance | every emitted token reaches the listener in order; a listener that says stop ends generation there | the engine streams a speculative answer and frees it when the client leaves |
| `runner_target_scores_every_fed_position` | differential | on L7.9's tiny llama: one row per fed id, each equal to a fresh run of its prefix; prompt lookup emits plain greedy's tokens | speculation works on a real model, not only the fake |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Copying after the FIRST earlier occurrence | stale drafts on long repetitive contexts; acceptance drops | `prompt_lookup_matches_python_l8_6` (mutant `s01`) |
| Stopping the continuation at the suffix | no draft on periodic text, the best case for lookup | `hand_example_prompt_lookup` (mutant `s02`) |
| N-gram ties to the highest id | drafts differ from the specified ones | `ngram_draft_hand_example` (mutant `s03`) |
| Greedy penalties not seeing tokens accepted this round | greedy speculative output differs from greedy | `verify_matches_python_l8_6` (mutant `s04`) |
| Accepting with probability $p(x)$, ignoring $q(x)$ | sampled output follows the wrong law whenever the draft is stochastic | `hand_example_sampled_step` (mutant `s05`) |
| Residual $\lvert p - q \rvert$ | rejected positions favor the drafted token | `sampled_spec_keeps_the_target_distribution` (mutant `s06`) |
| Drawing $u_r$ only on rejection | the generator drifts from Python's after the first acceptance | `verify_matches_python_l8_6` (mutant `s07`) |
| The bonus token from the last draft's row | a full acceptance emits the wrong next token | `hand_example_greedy_verify` (mutant `s08`) |
| $u_r$ drawn before $u_a$ | seeded streams differ from Python's | `verify_matches_python_l8_6` (mutant `s09`) |
| Rejected drafts' KV kept | later tokens attend to tokens that were never emitted | `greedy_spec_equals_greedy_without_spec` (mutant `s10`) |
| Drafts not shortened near `max_new` | KV written past the request's budget | `greedy_spec_equals_greedy_without_spec` (mutant `s11`) |
| Resampling from $p$, not the residual | the drafted token is over-represented | `sampled_spec_keeps_the_target_distribution` (mutant `s12`) |
| Releasing the partial last block | the kept positions lose their KV | `rollback_hand_example` (mutant `s13`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L8.6` | the Python verifier and prompt lookup this port is held to, case by case |
| Back | `L10.1` | `apply_penalties`, `distribution`, `argmax`, `sample`, `stream`, and `KvPool` |
| Back | `L10.1` | release and fill on the Rust pool during rollback |
| Forward | `L10.5` | runtime config enables speculative decoding through the request-serving path |

The serve loop (`L10.5`) routes requests through `RunnerTarget` when `[engine].speculative` names a draft. This path holds one request on the engine thread until generation completes; the ordinary scheduler path remains active when speculation is absent. `L10.7` exports `SpecStats::accept_rate` as `tl_engine_spec_accept_rate`; `MS-L10` checks greedy output is unchanged with `prompt_lookup` on.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| prompt lookup and n-gram drafts | vLLM `ngram` speculative method | the same idea inside a batched GPU engine | [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) |
| a chain of drafts | Medusa, EAGLE | extra heads that draft a tree of continuations; tree attention verifies them at once | [EAGLE](https://arxiv.org/abs/2401.15077), [Medusa](https://arxiv.org/abs/2401.10774) |
| one target pass per round | SpecInfer, Sequoia | token trees and draft budgets tuned to hardware | [SpecInfer](https://arxiv.org/abs/2305.09781) |
| a deterministic draft model | llama.cpp `--draft` | a small model of the same family as the draft | [llama.cpp speculative example](https://github.com/ggml-org/llama.cpp/tree/master/examples/speculative) |
