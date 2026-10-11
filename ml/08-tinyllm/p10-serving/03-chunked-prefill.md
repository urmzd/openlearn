<!-- ol:module L10.3 -->
# Chunked prefill (mixed prefill + decode batches)

## Overview

| | |
|---|---|
| **Module** | `L10.3` · build · Rust · Pass 7 · 5 to 8 h |
| **You build** | `rust/crates/tl-engine/src/chunk.rs`: the `Chunked` prefill policy, `plan` (one step's mixed batch for the runner and the rows to sample), and `split` |
| **Contract** | no Rust trait contract file yet: section 4 pins the API; `prefill_chunk` of `[engine]` in [`config/runtime.schema.json`](../../../course/contracts/config/runtime.schema.json) |
| **Tests** | `course/tests/rust/l10_3.rs`, 7 tests (what they check: section 4) |
| **Needs** | `L10.2` the scheduler and its `PrefillPolicy` ([chapter](02-continuous-batching.md)) · `L10.1` the runner and `ForwardBatch` ([chapter](01-model-runner-and-sampler.md)) · or `--ref-deps` |
| **Used by** | `L10.5` (the engine loop builds every step with `plan`; `[engine].prefill_chunk` turns `Chunked` on) |
| **Milestone** | `MS-L10` |
| **Optional depth** | [Agrawal et al. 2023, SARATHI: chunked prefills](https://arxiv.org/abs/2308.16369) (free); [Agrawal et al. 2024, Sarathi-Serve](https://arxiv.org/abs/2403.02310) (free); [vLLM chunked prefill docs](https://docs.vllm.ai/en/latest/configuration/optimization.html) (free) |

## Key Takeaways

- A long prompt prefilled in one step makes that step slow and every running request waits for it; in chunks of at most `prefill_chunk` tokens it shares each step with the decodes (`long_prompt_progresses_beside_decodes`).
- Every step stays within `max_batch_tokens`: decodes first, then prefill chunks in what is left (`token_budget_never_exceeded`).
- A mixed batch is one sequence per decode and one per chunk; only decodes and chunks that end their sequence are sampled (`plan_samples_only_sequence_ends`).
- Chunking changes when K and V are computed, never their values: the logits after the last chunk agree with one whole prefill within floating-point tolerance (`chunked_prefill_matches_whole_within_tolerance`).

## How to work this chapter

```bash
ol start L10.3               # stubs tl-engine/src/chunk.rs
ol tests L10.3
ol check L10.3
```

---

## 1. Why now

`L10.2` prefills a prompt whole, in one step, and a prompt longer than the budget may only run in a step of its own. Two costs follow. While a 2000-token prompt runs, every decoding request waits for that step: its time per output token (TPOT) spikes. And a long prompt can wait for a step of its own for a long time while others decode (`whole_prompt_waits_for_a_step_of_its_own`). Chunked prefill splits the prompt across steps and mixes the pieces with decodes, so both waits are bounded.

## 2. Principles

### 2.1 Two latencies, one budget

| Symbol | Meaning | Type |
|---|---|---|
| $\beta$ | token budget per step (`max_batch_tokens`) | integer |
| $\kappa$ | chunk size (`prefill_chunk`) | integer |
| $m$ | tokens of a prompt still to prefill | integer |
| $d$ | decodes in the step | integer |
| $\tau(\text{tokens})$ | time of one step, roughly linear in its tokens | seconds |

**Time to first token** (TTFT) is the time from arrival to the first generated token; for a prompt it needs every chunk done. **Time per output token** (TPOT) is the time between a running request's tokens, one step. A step's time grows with the tokens it processes, so TPOT is about $\tau(d + \text{prefill tokens in the step})$.

Chunking fixes the step's size: each step takes the $d$ decodes first, then gives each prefilling request $\min(m, \kappa, \text{budget left})$ tokens. TPOT is bounded by $\tau(\beta)$ whatever prompts arrive. A prompt of $m$ tokens needs about $\lceil m / \min(\kappa, \beta - d) \rceil$ steps to its first token instead of one big step, so TTFT for long prompts rises a little while TPOT for everyone else stops spiking. The right $\kappa$ is measured, not derived: `L10.7` exports TTFT and TPOT histograms, and `load.01` drives the engine to read them.

### 2.2 Why the answer does not change

Splitting a prompt into chunks is safe only if the K and V of each position, and the final logits, agree with one whole prefill. Absolute positions (`pos` $= p_s + i$, RoPE at the same angle) and writing each chunk's K and V to the same blocks preserve cached values. Candle can use different floating-point reduction paths for different tensor shapes, so the test compares logits within a small tolerance.

### 2.3 The mixed batch

`plan(scheduler, schedule_output)` builds what the runner takes:

- one `ForwardSeq` per decode: the newest token, at position $n_r - 1$;
- one `ForwardSeq` per chunk: tokens `start..start + len` at `start`;
- the rows to sample: every decode, and each chunk whose `last` flag says it reaches the end of the sequence. A middle chunk's logits predict a token the prompt already has, so they are computed and thrown away.

The policy itself is one line: `Chunked { chunk }.chunk_len(m, budget, _) = min(m, chunk, budget)`.

## 3. Worked example by hand

Budget $\beta = 8$, $\kappa = 4$, blocks of 4. A has a 2-token prompt and `max_tokens` 3; B has a 10-token prompt and `max_tokens` 1; both arrive at step 1.

| Step | Decodes | Chunks | Tokens | Sampled |
|---|---|---|---|---|
| 1 | none | A 0..2 (last), B 0..4 | 6 | A |
| 2 | A | B 4..8 | 5 | A |
| 3 | A | B 8..10 (last) | 3 | A, B |

B's first token comes at step 3, and no step exceeds 8 tokens. Without chunks (`WholePrompt`), B (10 > 8) cannot share a step with A's decodes, so it waits until A finishes and then runs alone in a 10-token step 4 (`whole_prompt_waits_for_a_step_of_its_own`).

`split(10, 4)` gives the boundaries a prompt alone in the engine would use: $(0, 4), (4, 4), (8, 2)$, every token once, in order (`split_hand_example`).

## 4. The interface

```rust
// rust/crates/tl-engine/src/chunk.rs
pub struct Chunked { pub chunk: usize }
impl PrefillPolicy for Chunked { fn chunk_len(&self, remaining: usize, budget: usize, alone: bool) -> usize; }
pub struct StepPlan<'a> { pub batch: ForwardBatch<'a>, pub sample: Vec<(usize, RequestId)> }   // (logits row, request)
pub fn plan<'a, B: BlockSpace>(s: &'a Scheduler<B>, out: &ScheduleOutput) -> StepPlan<'a>;
pub fn split(n: usize, chunk: usize) -> Vec<(usize, usize)>;                                   // a chunk of 0 means 1
```

Turn chunking on with `Scheduler::new(cfg, blocks).with_prefill_policy(Box::new(Chunked { chunk }))`; the engine of `L10.5` does when `prefill_chunk > 0`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_chunk_plan` | unit | section 3, step by step, through the scheduler | the worked example |
| `whole_prompt_waits_for_a_step_of_its_own` | unit | the same workload without chunks: B waits for step 4 and a 10-token step | the problem chunking solves |
| `split_hand_example` | unit | $(0,4), (4,4), (8,2)$ and the edge cases | chunk boundaries cover each token once |
| `plan_samples_only_sequence_ends` | unit | decode at $n - 1$, chunk tokens and start, only the right rows sampled | the engine samples only real next tokens |
| `token_budget_never_exceeded` | property | random prompts up to 4x the budget: every step within $\beta$, every output exact | the TPOT bound holds |
| `long_prompt_progresses_beside_decodes` | property | a 40-token prompt starts within 4 steps beside 3 decodes; without chunks it waits about 38 | why chunking exists |
| `chunked_prefill_matches_whole_within_tolerance` | differential | chunk sizes 1, 3, 7, 16 give logits within tolerance and preserve greedy continuations | chunking never changes answers |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| A chunk that ignores the budget left | steps exceed `max_batch_tokens`; TPOT spikes return | `token_budget_never_exceeded` (mutant `s01`) |
| A chunk that ignores `prefill_chunk` | long prompts still run whole | `hand_example_chunk_plan` (mutant `s02`) |
| Sampling a middle chunk's logits | extra tokens appear in the answer, or the scheduler must guard against them | `plan_samples_only_sequence_ends` (mutant `s03`) |
| Placing a decode token at position $n$ | each decode reads and writes one slot off | `plan_samples_only_sequence_ends` (mutant `s04`) |
| A last chunk of full size past the prompt | the plan reads past the prompt | `split_hand_example` (mutant `s05`) |
| Chunk-relative positions or `q_offset` 0 | chunked output differs from whole output | `chunked_prefill_matches_whole_within_tolerance` (the faults of L10.1's s13 and s14) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L10.2` | `PrefillPolicy` is consulted in phases 2 and 3; `Chunk.last` marks the sampled rows |
| Back | `L10.1` | chunk invariance of the runner makes chunking output-preserving |
| Back | `L10.1` | Candle-backed attention, absolute positions, and the Rust KV cache |
| Forward | `L10.5` | the engine loop runs `plan` every step; `prefill_chunk` in runtime.toml |

`L10.7` exports TTFT and TPOT, which is how you choose `prefill_chunk` for your machine.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one fixed chunk size | Sarathi-Serve's stall-free batching | chunk size from a TPOT target | [Sarathi-Serve](https://arxiv.org/abs/2403.02310) |
| decodes first, then chunks | vLLM v1's unified scheduler | one token budget across prefill and decode, no phases | `vllm/v1/core/sched/scheduler.py` |
| mixed batches on one engine | disaggregated prefill (`L10.6`) | prefill and decode on different machines | [DistServe](https://arxiv.org/abs/2401.09670), `L10.6` |
