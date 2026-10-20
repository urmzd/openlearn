# Chunked prefill (mixed prefill + decode batches)

[The continuous batching scheduler](03-continuous-batching.md) prefills a prompt whole, in one step, and a prompt longer than the budget may only run in a step of its own. Two costs follow. While a 2000-token prompt runs, every decoding request waits for that step, so its time per output token (TPOT) spikes. And a long prompt can wait a long time for a step of its own while others decode. **Chunked prefill** splits the prompt across steps and mixes the pieces with decodes, so both waits are bounded.

The solution is small, because the scheduler already has the seam it plugs into. It builds on the scheduler and its `PrefillPolicy` from the previous section and on the runner's `ForwardBatch` from [the Candle model runner](02-model-runner-and-sampler.md). There is no Rust trait contract file yet, so the interface below pins the API; the configuration knob is `prefill_chunk` in the `[engine]` table of [`config/runtime.schema.json`](../../../../course/contracts/config/runtime.schema.json). The code is [`rust/crates/tl-engine/src/chunk.rs`](../../../../course/solutions/rust/crates/tl-engine/src/chunk.rs): the `Chunked` prefill policy, `plan` (one step's mixed batch for the runner and the rows to sample), and `split`.

> **Learning objectives**
> - Define TTFT and TPOT and explain how chunking trades a little of the first for a bound on the second.
> - Implement the `Chunked` policy and explain where the "decodes first" rule actually lives.
> - Build a mixed batch with one sequence per decode and per chunk, sampling only rows that predict an unknown token.
> - Argue why chunking changes when K and V are computed but not their values.

## Two latencies, one budget

The quantities involved:

| Symbol | Meaning | Type |
|---|---|---|
| $\beta$ | token budget per step (`max_batch_tokens`) | integer |
| $\kappa$ | chunk size (`prefill_chunk`) | integer |
| $m$ | tokens of a prompt still to prefill | integer |
| $d$ | decodes in the step | integer |
| $\tau(\text{tokens})$ | time of one step, roughly linear in its tokens | seconds |

**Time to first token** (TTFT) is the time from arrival to the first generated token; for a prompt it needs every chunk done. **Time per output token** (TPOT) is the time between a running request's tokens, which is one step. A step's time grows with the tokens it processes, so TPOT is about $\tau(d + \text{prefill tokens in the step})$.

Chunking fixes the step's size. Each step takes the $d$ decodes first, then gives each prefilling request $\min(m, \kappa, \text{budget left})$ tokens, so TPOT is bounded by $\tau(\beta)$ whatever prompts arrive. A prompt of $m$ tokens needs about $\lceil m / \min(\kappa, \beta - d) \rceil$ steps to its first token instead of one big step, so TTFT for long prompts rises a little while TPOT for everyone else stops spiking. The right $\kappa$ is measured, not derived: [Serving metrics and tracing](08-serving-metrics-and-tracing.md) exports TTFT and TPOT histograms, and [the load generator](11-load-generator.md) drives the engine so you can read them.

## Why the answer does not change

Splitting a prompt into chunks is safe only if the K and V of each position, and the final logits, agree with one whole prefill. Absolute positions (`pos` $= p_s + i$, so RoPE turns by the same angle) and writing each chunk's K and V to the same blocks preserve the cached values. Candle can use different floating-point reduction paths for different tensor shapes, so chunked and whole logits are compared within a small tolerance rather than bit for bit.

## The mixed batch

Once the scheduler has granted chunks, `plan(scheduler, schedule_output)` builds what the runner takes:

- one `ForwardSeq` per decode: the newest token, at position $n_r - 1$;
- one `ForwardSeq` per chunk: tokens `start..start + len` at `start`;
- the rows to sample: every decode, and each chunk whose `last` flag says it reaches the end of the sequence.

A middle chunk's logits predict a token the prompt already has, so they are computed and thrown away. The policy itself is one line: `Chunked { chunk }.chunk_len(m, budget, _) = min(m, chunk, budget)`.

## Working it by hand

Take budget $\beta = 8$, $\kappa = 4$, and blocks of 4. A has a 2-token prompt and `max_tokens` 3; B has a 10-token prompt and `max_tokens` 1; both arrive at step 1. The schedule, step by step:

| Step | Decodes | Chunks | Tokens | Sampled |
|---|---|---|---|---|
| 1 | none | A 0..2 (last), B 0..4 | 6 | A |
| 2 | A | B 4..8 | 5 | A |
| 3 | A | B 8..10 (last) | 3 | A, B |

B's first token comes at step 3, and no step exceeds 8 tokens. Without chunks (`WholePrompt`), B (10 > 8) cannot share a step with A's decodes, so it waits until A finishes and then runs alone in a 10-token step 4.

`split(10, 4)` gives the boundaries a prompt alone in the engine would use: $(0, 4), (4, 4), (8, 2)$, every token once, in order.

## Designing the interface

The whole interface is four items:

```rust
// rust/crates/tl-engine/src/chunk.rs
pub struct Chunked { pub chunk: usize }
impl PrefillPolicy for Chunked { fn chunk_len(&self, remaining: usize, budget: usize, alone: bool) -> usize; }
pub struct StepPlan<'a> { pub batch: ForwardBatch<'a>, pub sample: Vec<(usize, RequestId)> }   // (logits row, request)
pub fn plan<'a, B: BlockSpace>(s: &'a Scheduler<B>, out: &ScheduleOutput) -> StepPlan<'a>;
pub fn split(n: usize, chunk: usize) -> Vec<(usize, usize)>;                                   // a chunk of 0 means 1
```

You turn chunking on with `Scheduler::new(cfg, blocks).with_prefill_policy(Box::new(Chunked { chunk }))`; [the tokio server](06-openai-server-on-tokio.md) does so when `prefill_chunk > 0`.

## Building it step by step

### The reasoning chain

Try to predict each step before reading it.

1. **A step's time grows with its tokens**, so bounding TPOT means bounding the tokens in every step: the budget $\beta$ must hold for prefill tokens too, not just decodes.
2. **The scheduler already has a seam for "how much of this prompt now"**: `PrefillPolicy::chunk_len(remaining, budget, alone)`. Chunking is therefore a new policy, not a new scheduler: take $\min(m, \kappa, \text{budget})$.
3. **Decodes are planned first by the scheduler**, so whatever the policy grants comes out of what decodes left. That is the "decodes first, then chunks" rule with no extra code.
4. **The runner already takes a batch of (tokens, start, blocks) sequences**, so a mixed batch is just more of them: one per decode (one token at $n - 1$), one per chunk (its slice at `start`). No new kernel is needed.
5. **Only rows whose logits predict an unknown token may be sampled:** every decode, and a chunk that reaches the end of the sequence. A middle chunk's last logits predict a prompt token that already exists.
6. **Chunking is correct only because positions are absolute and KV goes to the same blocks**, so the last chunk sees exactly the keys a whole prefill would have written.

### The policy

The policy is one expression. `chunk.max(1)` guards a misconfigured chunk of 0 (which would otherwise grant nothing forever), and `alone` is ignored: unlike `WholePrompt`, a chunk never needs a step of its own because it always fits the budget.

```rust
impl PrefillPolicy for Chunked {
    fn chunk_len(&self, remaining: usize, budget: usize, _alone: bool) -> usize {
        remaining.min(self.chunk.max(1)).min(budget)   // min(m, kappa, budget left after decodes)
    }
}
```

In the worked example at step 1, A's 2-token prompt gets $\min(2, 4, 8) = 2$ (and is `last`), then B gets $\min(10, 4, 6) = 4$.

### The plan

Rows are numbered in the order sequences are pushed: decodes first, then chunks. `sample` records `(row, request)` for exactly the rows whose logits are used. Each `ForwardSeq` borrows its tokens and block table from the scheduler, so building the batch copies nothing.

```rust
pub fn plan<'a, B: BlockSpace>(s: &'a Scheduler<B>, out: &ScheduleOutput) -> StepPlan<'a> {
    let mut p = StepPlan::default();
    for &id in &out.decode {
        let Some(toks) = s.tokens(id) else { continue };
        let n = toks.len();
        p.sample.push((p.batch.seqs.len(), id));              // every decode is sampled
        p.batch.seqs.push(ForwardSeq {
            tokens: &toks[n - 1..],                           // the newest token only
            start: n - 1,                                     // at position n - 1, not n
            blocks: s.blocks().block_table(id),
        });
    }
    for c in &out.prefill {
        let Some(toks) = s.tokens(c.id) else { continue };
        if c.last {
            p.sample.push((p.batch.seqs.len(), c.id));        // only a chunk that ends the sequence
        }
        p.batch.seqs.push(ForwardSeq {
            tokens: &toks[c.start..c.start + c.len],          // this chunk's slice
            start: c.start,                                   // absolute start: RoPE and q_offset follow
            blocks: s.blocks().block_table(c.id),
        });
    }
    p
}
```

The engine then runs `ModelRunner::forward(&plan.batch)`, samples `logits.row(row)` for each `(row, id)` in `plan.sample`, and passes the tokens to `Scheduler::on_step`, which ignores any request whose sequence is not fully computed.

### `split`

`split` returns the boundaries a lone prompt would use. `step_by(c)` gives the starts, and `c.min(n - s)` shortens the last chunk so it never runs past the prompt.

```rust
pub fn split(n: usize, chunk: usize) -> Vec<(usize, usize)> {
    let c = chunk.max(1);                                     // a chunk of 0 means 1
    (0..n).step_by(c).map(|s| (s, c.min(n - s))).collect()   // (start, len); last one short
}
```

`split(10, 4)` is `[(0, 4), (4, 4), (8, 2)]`.

The full solution is [`chunk.rs`](../../../../course/solutions/rust/crates/tl-engine/src/chunk.rs); the phases that call the policy are in [`sched.rs`](../../../../course/solutions/rust/crates/tl-engine/src/sched.rs).

### Why this design

`plan` is $O(\text{sequences in the step})$ and allocates only the two small vectors. Chunking adds no forward work; it only spreads it over more steps. The real cost is that a prompt of $m$ tokens takes about $\lceil m / \min(\kappa, \beta - d) \rceil$ steps to its first token.

Why a policy object instead of chunk logic inside the scheduler? The scheduler's phases, budget, and preemption stay unchanged, chunking becomes a three-line plug-in, and `WholePrompt` remains available for comparison. Why decodes first, rather than a fair share of the budget between prefill and decode? Decodes are the latency users already see (TPOT), and putting them first guarantees they never wait on a prompt. vLLM v1 unifies the two under one budget without phases, which is more flexible but harder to reason about. Why not sample middle chunks and discard the token in the scheduler? It would work, but it spends sampler work and RNG draws on tokens that are thrown away, and a stray draw would desynchronize seeded streams. Not sampling them is both cheaper and simpler.

One tradeoff is accepted: chunked and whole prefills agree within floating-point tolerance, not bit for bit, because Candle may pick different reduction paths for different shapes. Greedy continuations are preserved on the tested model, but a near-tie could in principle flip.

## What can go wrong

A correct implementation handles these cases:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked plan | unit | the worked example, step by step, through the scheduler | the worked example |
| the same workload without chunks | unit | B waits for step 4 and a 10-token step | the problem chunking solves |
| `split` boundaries | unit | $(0,4), (4,4), (8,2)$ and the edge cases | chunk boundaries cover each token once |
| `plan` samples only sequence ends | unit | decode at $n - 1$, chunk tokens and start, only the right rows sampled | the engine samples only real next tokens |
| the token budget is never exceeded | property | random prompts up to 4x the budget: every step within $\beta$, every output exact | the TPOT bound holds |
| a long prompt progresses beside decodes | property | a 40-token prompt starts within 4 steps beside 3 decodes; without chunks it waits about 38 | why chunking exists |
| chunked equals whole prefill | differential | chunk sizes 1, 3, 7, 16 give logits within tolerance and preserve greedy continuations | chunking never changes answers |

The mistakes here are mostly off-by-ones in positions or a missing `min`:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| A chunk that ignores the budget left | steps exceed `max_batch_tokens`; TPOT spikes return | `.min(budget)` in `chunk_len` |
| A chunk that ignores `prefill_chunk` | long prompts still run whole | `.min(self.chunk.max(1))` |
| Sampling a middle chunk's logits | extra tokens appear in the answer, or the scheduler must guard against them | `plan` pushes to `sample` only when `c.last` |
| Placing a decode token at position $n$ | each decode reads and writes one slot off | `start: n - 1` |
| A last chunk of full size past the prompt | the plan reads past the prompt | `split` uses `c.min(n - s)` |
| Chunk-relative positions or `q_offset` 0 | chunked output differs from whole output | `start: c.start`, absolute, passed through to RoPE and attention in the runner |

## Where this leads

Chunked prefill depends on the scheduler, which consults `PrefillPolicy` in phases 2 and 3 and marks sampled rows with `Chunk.last`, and on the runner's Candle-backed attention, absolute positions, and Rust KV cache, whose chunk invariance makes chunking output-preserving. Going forward, [the tokio server](06-openai-server-on-tokio.md) runs `plan` every step and reads `prefill_chunk` from `runtime.toml`, and [Serving metrics and tracing](08-serving-metrics-and-tracing.md) exports TTFT and TPOT, which is how `prefill_chunk` is chosen for a given machine.

Production systems go further:

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one fixed chunk size | Sarathi-Serve's stall-free batching | chunk size from a TPOT target | [Sarathi-Serve](https://arxiv.org/abs/2403.02310) |
| decodes first, then chunks | vLLM v1's unified scheduler | one token budget across prefill and decode, no phases | `vllm/v1/core/sched/scheduler.py` |
| mixed batches on one engine | [disaggregated prefill](07-disaggregated-prefill-decode.md) | prefill and decode on different machines | [DistServe](https://arxiv.org/abs/2401.09670), [Disaggregated prefill and decode](07-disaggregated-prefill-decode.md) |

### Further reading

- [Agrawal et al. 2023, SARATHI: chunked prefills](https://arxiv.org/abs/2308.16369) (free).
- [Agrawal et al. 2024, Sarathi-Serve](https://arxiv.org/abs/2403.02310) (free).
- [vLLM chunked prefill docs](https://docs.vllm.ai/en/latest/configuration/optimization.html) (free).

## Key terms

- **chunked prefill**: prefilling a prompt over several steps in pieces of at most `prefill_chunk` tokens, mixed with decodes.
- **TTFT (time to first token)**: time from a request's arrival to its first generated token.
- **TPOT (time per output token)**: time between successive tokens of a running request, about one step.
- **mixed batch**: one forward step containing both decode sequences and prefill chunks.
- **last chunk**: the chunk that reaches the end of a sequence, the only chunk whose logits are sampled.

## Summary

- A long prompt prefilled in one step makes that step slow and every running request waits for it; in chunks of at most `prefill_chunk` tokens it shares each step with the decodes.
- Every step stays within `max_batch_tokens`: decodes first, then prefill chunks in what is left.
- A mixed batch is one sequence per decode and one per chunk; only decodes and chunks that end their sequence are sampled.
- Chunking changes when K and V are computed, never their values: the logits after the last chunk agree with one whole prefill within floating-point tolerance.

## Review questions

1. Why does `plan` place a decode at position $n - 1$ and not $n$?
   <details><summary>Answer</summary>The newest token, at index $n - 1$, has not been fed yet; feeding it writes its K and V at slot $n - 1$ and produces the logits for token $n$. Placing it at $n$ would write one slot off and rotate it by the wrong RoPE angle.</details>
2. Why is a middle chunk's row not sampled?
   <details><summary>Answer</summary>Its last logits predict the next prompt token, which is already known. Sampling it would add a wrong token to the answer and consume a random draw.</details>
3. Where does the "decodes first" rule live, given that `Chunked` only computes a minimum?
   <details><summary>Answer</summary>In the scheduler's phase order: decodes take their tokens from the budget in phase 1, so the `budget` that phases 2 and 3 pass to `chunk_len` is what decodes left.</details>
4. Why does chunking not change the answer?
   <details><summary>Answer</summary>Each chunk is fed at its absolute start, so RoPE angles and causal visibility are those of a whole prefill, and its K and V go into the same blocks and slots. The last chunk then attends over the same keys and values a whole prefill would have computed.</details>
5. In the worked example, why does step 2 process 5 tokens?
   <details><summary>Answer</summary>A decodes (1 token) and B receives $\min(6, 4, 7) = 4$ prompt tokens (positions 4..8), for $1 + 4 = 5$.</details>
