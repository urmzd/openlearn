# Speculative decoding in the engine

## Overview

| | |
|---|---|
| **Module** | `L10.8` · build · Rust · Pass 7 · 3 to 5 h reading |
| **Solution** | [`tl-engine/src/spec.rs`](../../../../course/solutions/rust/crates/tl-engine/src/spec.rs): `prompt_lookup` and `ngram_draft` over the request's own context, the verifier `verify_draft` (greedy acceptance, and M07.6's accept-or-resample `speculative_step` at temperature > 0), KV `rollback` on rejection, and the generation loop `generate_stream` over a `Target` (`RunnerTarget` in production) |
| **Contract** | the Python specification this port follows: [`py/tinyllm/infer/spec.pyi`](../../../../course/contracts/py/tinyllm/infer/spec.pyi) (L8.6) · sampling order and draws: [`spec/sampling.md`](../../../../course/contracts/spec/sampling.md), [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md) · configuration `[engine].speculative`: [`config/runtime.schema.json`](../../../../course/contracts/config/runtime.schema.json) · the gauge `tl.engine.spec_accept_rate`: [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml) |
| **Needs** | `L8.6` the Python speculative decoding, the specification this port is held to ([chapter](../p08-inference/06-speculative-decoding.md)) · `L10.1` the sampler (`apply_penalties`, `distribution`, `sample`, `stream`) and `KvPool` ([chapter](01-model-runner-and-sampler.md)) · `L10.1` the Rust pool behind the rollback ([chapter](../p08-inference/08-paged-kv-block-pool.md)) · reading: `L10.2`, `L10.4` the scheduler and block manager it plugs into, `M07.6` rejection and residuals ([chapter](../../../tracks/math/07-probability-statistics/06-rejection-sampling-and-residual-distributions.md)) |
| **Used by** | `L10.5`'s runner-backed serving path (`[engine].speculative`); with prompt lookup on, the end-to-end system's greedy output is unchanged |
| **Optional depth** | [Leviathan et al., Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (free); [Chen et al., Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (free); [Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding) (free) |

## Key Takeaways

- Prompt lookup copies what followed the most recent earlier occurrence of the context's suffix, trying long suffixes first; it costs no model and shines on repetitive text.
- Greedy verification keeps drafts while they equal the target's argmax, so greedy output with speculation is token for token greedy output without it.
- With sampling, accept-or-resample keeps the target's distribution exactly, whatever the draft proposes.
- The port draws exactly what L8.6 draws, in the same order (u_accept, then u_resample, both always; one draw for the bonus token), so its tokens equal the Python reference's on the same logits and seed.
- A rejected draft's KV is given back: after every round the sequence holds exactly the blocks its cached positions need.

## How to read this chapter

Sections 1 to 3 explain why speculation pays, the drafts, the verification rule and why it is exact, and work a draft, a greedy round, a sampled step, and a rollback by hand. Section 4 fixes the interface. Section 5 walks through the solution: the drafts, the accept-or-resample step, `verify_draft`, `rollback`, and the generation loop. Try to predict each step before reading it.

The solution declares `pub mod spec;` in `tl-engine/src/lib.rs`. The engine turns `[engine].speculative = { draft = "prompt_lookup", k = 4 }` into a `SpecConfig` with `SpecConfig::parse`, and each round runs `propose`, one forward over the pending token plus the draft, `verify_draft`, then a cut of the sequence's block table to the kept length.

---

## 1. Why now

Decoding one token costs one forward pass over the whole model, and at the batch sizes a laptop can afford that pass is limited by reading the weights, not by arithmetic: checking five tokens costs about as much as producing one. If something cheap can guess the next few tokens, one target pass can verify all of them. In Python (`L8.6`) the course proved the rule that makes this exact. The Rust engine (`L10.5`) still decodes one token per pass, and the workloads it serves are full of repetition: code edits, retrieval answers quoting the context, JSON tool arguments that echo the prompt. This module ports the verifier to the engine, adds two drafts that need no second model (prompt lookup and an n-gram table over the request's own tokens), and gives back the KV of rejected drafts so speculation never leaks blocks.

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

**Prompt lookup.** Context `[5 6 7 8 5 6]`, $k = 3$, $n$ from 3 down to 1. The 3-suffix `[8 5 6]` never occurred earlier. The 2-suffix `[5 6]` occurred at position 0, so the draft is what followed it: `[7 8 5]`.

**A greedy round.** Four ids, draft `[2 1]`, rows:

| Row | Logits | Argmax |
|---|---|---|
| 0 | `[0 1 3 0.5]` | 2 = $d_1$: accept |
| 1 | `[0 1 0.5 2]` | 3 $\ne d_2 = 1$: emit 3, stop |
| 2 | `[5 0 0 0]` | (not reached) |

Emitted `[2 3]`, one draft accepted, no draws. With draft `[2 3]` both are accepted and row 2's argmax 0 is the bonus: `[2 3 0]`. The logprob of the first emitted 2 is $3 - \ln(e^0 + e^1 + e^3 + e^{0.5})$.

**A sampled step.** $p = [0.2, 0.5, 0.3]$, the draft proposed $x = 0$ from $q = [0.6, 0.2, 0.2]$; accept with probability $0.2/0.6 = 1/3$. With $u_a = 0.3$: accept. With $u_a = 0.5$: reject; the residual is $\max(0, p - q) = [0, 0.3, 0.1]$, normalized $[0, 0.75, 0.25]$; $u_r = 0.8$ walks the running sum $0, 0.75, 1.0$ and lands on id 2.

**A rollback.** $B = 4$; 10 cached positions fill three blocks (4, 4, 2). Keeping 5 positions needs $\lceil 5/4 \rceil = 2$ blocks: block 3 is released, block 2's fill becomes 1.

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
                seed: u64, max_new: usize, eos: &[u32]) -> Result<Generated, SpecError>;   // generate_stream, never stopped
pub fn generate_stream(target: &mut dyn Target, cfg: &SpecConfig, prompt: &[u32], p: &SamplingParams,
                       seed: u64, max_new: usize, eos: &[u32],
                       on_token: &mut dyn FnMut(u32, f64) -> bool) -> Result<Generated, SpecError>;
```

`generate_stream` is `generate`'s loop, handing each emitted token and its logprob to `on_token` as soon as its round is verified, and returning at once with what it has when `on_token` answers false. The serve loop streams through it and stops a request whose client left. `RunnerTarget` is the production `Target`: the runner returns one logits row per sequence, so it feeds every id as its own one-token sequence over the request's blocks, at its own position, in one forward.

### What a correct solution must handle

The verification and prompt-lookup cases are checked against what the Python reference of L8.6 answers on 120 verification cases (exact f32 logits, drafts that agree for a while, optional draft distributions, every sampling knob, 64-bit seeds) and 150 prompt-lookup contexts. The generation cases run a fake target whose logits are a fixed function of the context.

| Case | Checks | Why it matters downstream |
|---|---|---|
| The hand prompt lookup | section 3's draft, a continuation into the suffix, no draft | the worked example |
| Prompt lookup equals Python | 150 contexts equal to `PromptLookupDraft` | most recent occurrence, longest suffix first |
| The hand n-gram draft | counts, ties to the lowest id, longest context first, back-off, stop | the second draft |
| Runtime config | the three drafts; `k` 0 and 17 and a typo refused | a bad runtime.toml fails at startup |
| The hand greedy round | section 3's round, no draws, logprobs, the bonus token | the greedy rule |
| The hand sampled step | accept, reject, residual, a q that never proposed x | M07.6 by hand |
| Verification equals Python | 120 cases: emitted tokens, n_accepted, and the generator's next u32 | the port draws exactly what Python draws |
| Bad shapes | rows, draft distributions, out-of-vocabulary drafts | an error, not a panic, in the step loop |
| Greedy with speculation equals greedy without | every draft kind, three prompts: the greedy tokens, cached positions, fewer passes with prompt lookup | speculation changes speed only |
| EOS and `max_tokens` | exactly `max_new` tokens; EOS inside an accepted run stops there | requests end where they should |
| Sampled speculation keeps the target distribution | 4000 seeds: the first token's chi-square against the target softmax at p > 1e-3 | sampled output is unchanged in law |
| The hand rollback | section 3's rollback, a full block, keep 0, `blocks_for` | the worked example |
| Rollback drops one reference only | a block shared with another sequence survives | the prefix cache is never corrupted |
| Blocks conserved across rollbacks | 200 seeded requests with many rejections on a real pool: blocks match positions, nothing leaks | weeks of speculation do not shrink the pool |
| Streaming generation | every emitted token reaches the listener in order; a listener that says stop ends generation there | the engine streams a speculative answer and frees it when the client leaves |
| `RunnerTarget` on a real model | on L7.9's tiny llama: one row per fed id, each equal to a fresh run of its prefix; prompt lookup emits plain greedy's tokens | speculation works on a real model, not only the fake |

## 5. The solution, step by step

### 5.1 From first principles

1. **One target pass can score many positions.** Feeding the pending token plus $m$ draft tokens gives $m + 1$ rows of logits for about the cost of one decode step (section 1). So the round is: draft cheaply, score once, keep a prefix.
2. **The draft must be cheap and deterministic.** Prompt lookup and the n-gram table read only the request's own tokens; neither needs a model, and both give a one-hot $q$, so $p(d)/q(d) = p(d)$.
3. **Correctness must not depend on the draft.** The accept test $u_a < p(x)/q(x)$ with a residual resample on rejection makes the emitted token's law exactly $p$ (section 2.3), whatever $q$ was. At $T = 0$ this collapses to "accept while the draft equals the argmax".
4. **The port must reproduce the Python reference draw for draw.** So the draw schedule is fixed regardless of outcome: $u_a$ then $u_r$ at every verified position, one draw for the bonus. A seeded request gives the same tokens in Python and Rust.
5. **Penalties must see the round's own accepted tokens.** Row $i$ was computed after $d_1 \ldots d_{i-1}$; a repetition penalty applied to it must count them, or greedy speculative output differs from plain greedy.
6. **Rejected positions' KV must go.** The forward wrote KV for every fed position; after keeping $n$, the cache must hold exactly `base + pending + n` positions, which means releasing whole blocks past $\lceil \text{keep}/B \rceil$ and shrinking the last block's fill (section 2.4).
7. **A round never overshoots the budget.** With `room = max_new - emitted - 1`, a draft of at most `room` plus the one token the verifier always adds emits at most `max_new - emitted` tokens.

### 5.2 Drafts

[`spec.rs`](../../../../course/solutions/rust/crates/tl-engine/src/spec.rs):

```rust
pub fn prompt_lookup(ctx: &[u32], k: usize, max_ngram: usize, min_ngram: usize) -> Vec<u32> {
    let l = ctx.len();
    if k == 0 || l < 2 || min_ngram == 0 {
        return Vec::new();
    }
    let top = max_ngram.min(l - 1);              // a suffix needs at least one earlier position
    if top < min_ngram {
        return Vec::new();
    }
    for n in (min_ngram..=top).rev() {           // longest suffix first: the most specific match
        let suffix = &ctx[l - n..];
        // most recent earlier occurrence first: i from l - n - 1 down to 0
        for i in (0..l - n).rev() {
            if &ctx[i..i + n] == suffix {
                let end = (i + n + k).min(l);    // capped at l, not at l - n: may run into the suffix
                if i + n < end {
                    return ctx[i + n..end].to_vec();
                }
            }
        }
    }
    Vec::new()
}
```

On section 3's context `[5 6 7 8 5 6]` with $k = 3$: `top = 3`; $n = 3$ finds no earlier `[8 5 6]`; $n = 2$ scans $i = 3, 2, 1, 0$ and matches at $i = 0$; `end = min(5, 6) = 5`, so the draft is `ctx[2..5]` = `[7 8 5]`. The last `5` is part of the suffix itself, which is why `end` is capped at `l`: on periodic text the continuation is the period, and stopping at the suffix would give no draft in the best case.

`ngram_draft` counts what followed the last $m$ tokens, for $m$ from `n - 1` down to 1, and picks with

```rust
counts.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
```

The reversed id comparison `b.0.cmp(a.0)` makes the lower id the "greater" one on a count tie, so ties go to the lowest id deterministically, independent of `HashMap` iteration order.

### 5.3 The accept-or-resample step

```rust
pub fn residual(p: &[f64], q: &[f64]) -> Vec<f64> {
    let r: Vec<f64> = p.iter().zip(q).map(|(&a, &b)| (a - b).max(0.0)).collect();   // max(0, p - q), not |p - q|
    let mut z = 0.0f64;
    for &v in &r { z += v; }
    if z <= 0.0 {
        return p.to_vec();      // p == q: nothing is ever rejected, but stay well defined
    }
    r.iter().map(|&v| v / z).collect()
}

pub fn speculative_step(p: &[f64], q: &[f64], x: usize, u_accept: f64, u_resample: f64) -> Result<(usize, bool), SpecError> {
    // ... shape checks; q[x] must be > 0 (q proposed x)
    if u_accept < p[x] / q[x] {                 // min(1, p/q) without computing the min
        return Ok((x, true));
    }
    Ok((sample_dense(&residual(p, q), u_resample), false))
}
```

Section 3's sampled step: $p(0)/q(0) = 0.2/0.6 = 1/3$; $u_a = 0.3 < 1/3$ accepts, $u_a = 0.5$ rejects; the residual is $[0, 0.75, 0.25]$, and `sample_dense` with $u_r = 0.8$ walks $0, 0.75, 1.0$ and returns the first id where $0.8 < c$: id 2.

### 5.4 Verifying a draft

```rust
pub fn verify_draft(rows: &[Vec<f32>], draft: &[u32], draft_probs: Option<&[Vec<f64>]>, p: &SamplingParams,
                    prompt: &[u32], output: &[u32], rng: &mut Pcg32) -> Result<Verified, SpecError> {
    let m = draft.len();
    // ... rows.len() == m + 1, equal row lengths, draft_probs [m][V]: errors, not panics
    let mut hist = output.to_vec();             // grows with every emitted token: penalties see this round
    let mut out = Verified { tokens: Vec::new(), logprobs: Vec::new(), n_accepted: 0 };
    let emit = |out: &mut Verified, hist: &mut Vec<u32>, row: &[f32], tok: u32| {
        let lp = logprob_of(&apply_penalties(row, p, prompt, hist), tok as usize);
        out.tokens.push(tok);
        out.logprobs.push(lp);
        hist.push(tok);
    };
    for (i, &x) in draft.iter().enumerate() {
        // ... x must be in the vocabulary
        if p.temperature == 0.0 {
            let g = argmax(&apply_penalties(&rows[i], p, prompt, &hist)) as u32;
            emit(&mut out, &mut hist, &rows[i], g);   // the argmax is emitted whether or not it matches
            if g != x {
                out.n_accepted = i;                   // first mismatch: the argmax replaces x, stop
                return Ok(out);
            }
            continue;                                 // no draws at T = 0
        }
        let pd = dense_distribution(&rows[i], p, prompt, &hist);
        let qd = /* draft_probs[i], or one-hot on x */;
        let u_accept = rng.uniform_f64();             // u_a first,
        let u_resample = rng.uniform_f64();           // then u_r, always: the draw count is fixed
        let (y, ok) = speculative_step(&pd, &qd, x as usize, u_accept, u_resample)?;
        emit(&mut out, &mut hist, &rows[i], y as u32);
        if !ok {
            out.n_accepted = i;
            return Ok(out);
        }
    }
    // every draft accepted: the bonus comes from row m, the row after the whole draft
    let (bonus, _) = sample(&rows[m], p, prompt, &hist, rng);
    emit(&mut out, &mut hist, &rows[m], bonus);
    out.n_accepted = m;
    Ok(out)
}
```

On section 3's greedy round, row 0's argmax 2 equals $d_1$ and is emitted; row 1's argmax 3 differs from $d_2 = 1$, so 3 is emitted and the round ends with `[2 3]`, `n_accepted = 1`. With draft `[2 3]` both match and the bonus is row 2's argmax 0. At $T = 0$ the bonus goes through `sample`, which is greedy at temperature 0, so the greedy path draws nothing.

### 5.5 Rollback

```rust
pub fn rollback(pool: &mut KvPool, table: &mut Vec<u32>, keep: usize) -> Result<usize, SpecError> {
    let b = pool.cfg().block_tokens as usize;
    let need = blocks_for(keep, b);              // ceil(keep / B)
    if need > table.len() { /* Shape error */ }
    let mut freed = 0;
    while table.len() > need {
        let id = table.pop().expect("len > need >= 0");
        pool.release(id).map_err(|e| SpecError::Kv(e.to_string()))?;   // one reference: a shared block survives
        freed += 1;
    }
    if let Some(&last) = table.last() {
        let fill = keep - (need - 1) * b;       // positions the last kept block still holds
        pool.set_fill(last, fill as u32).map_err(|e| SpecError::Kv(e.to_string()))?;
    }
    Ok(freed)
}
```

Section 3's rollback: $B = 4$, `keep = 5`, `need = 2`; one block popped and released; the last kept block's fill is $5 - 1 \cdot 4 = 1$. With `keep = 0`, `need = 0`, every block is released and there is no last block to refill. Releasing rather than freeing is what protects the prefix cache: a block another sequence also references only loses this sequence's reference.

### 5.6 The generation loop

```rust
pub fn generate_stream(target: &mut dyn Target, cfg: &SpecConfig, prompt: &[u32], p: &SamplingParams,
                       seed: u64, max_new: usize, eos: &[u32], on_token: &mut dyn FnMut(u32, f64) -> bool) -> Result<Generated, SpecError> {
    // ... empty prompt -> Shape error
    let mut rng = stream(seed, PURPOSE_SAMPLE);           // the same generator as unspeculated sampling
    let mut g = Generated { /* ... */ finish: Finish::Length };
    let mut pending: Vec<u32> = prompt.to_vec();          // first round: the whole prompt
    while g.tokens.len() < max_new {
        let room = max_new - g.tokens.len() - 1;          // leave room for the verifier's own token
        let mut ctx = prompt.to_vec();
        ctx.extend_from_slice(&g.tokens);
        let draft = propose(cfg, &ctx, cfg.k.min(room));
        let base = target.cached();
        let mut feed = pending.clone();
        feed.extend_from_slice(&draft);
        let rows = target.extend(&feed)?;                 // one forward: a row per fed id
        // ... rows.len() must equal feed.len()
        let tail = &rows[pending.len() - 1..];            // the last pending row scores d_1; m + 1 rows
        let ver = verify_draft(tail, &draft, None, p, prompt, &g.tokens, &mut rng)?;
        // ... stats: target_calls, drafted, accepted
        target.truncate(base + pending.len() + ver.n_accepted)?;   // rejected drafts' KV is dropped
        for (&t, &lp) in ver.tokens.iter().zip(&ver.logprobs) {
            if eos.contains(&t) {                         // EOS inside an accepted run stops there, not emitted
                g.finish = Finish::Stop;
                return Ok(g);
            }
            g.tokens.push(t);
            g.logprobs.push(lp);
            if !on_token(t, lp) {                         // the client left: stop now
                return Ok(g);
            }
            if g.tokens.len() == max_new {
                return Ok(g);
            }
        }
        pending = vec![*ver.tokens.last().expect("verify emits at least one token")];
    }
    Ok(g)
}
```

The cache after `truncate` holds the pending tokens and the accepted drafts, but not the verifier's own last token: that token has not been fed yet, so it becomes the next round's `pending`. This is why `tail` starts at `pending.len() - 1`: the row after the last pending token is the one that scores $d_1$. `generate` is `generate_stream` with a listener that always answers true. `RunnerTarget` is the production `Target`; it feeds every id as its own one-token sequence at its own position over the request's blocks, so one forward returns one row per fed id; its `truncate` applies the same release-and-refill rule as `rollback` to the request's blocks, and its `Drop` releases every block it still holds, so an early return cannot leak.

Full solution: [`spec.rs`](../../../../course/solutions/rust/crates/tl-engine/src/spec.rs).

### 5.7 Why this solution

- **Cost.** Prompt lookup is $O(n_{\max} \cdot t \cdot n)$ in the worst case per round (a scan for each suffix length), and $t$ is at most the context; on a 2,000-token context that is microseconds against a forward pass of milliseconds. Verification is $O(mV)$ for the dense distributions. A round with $a$ accepted drafts emits $a + 1$ tokens for one target pass.
- **Why prompt lookup and n-grams rather than a draft model?** No second model to load, no draft KV to manage, and the workloads this engine serves (code edits, quoting retrieved context, JSON echoing the prompt) repeat their own context. A draft model helps on novel text but costs memory and a second forward per token.
- **Why fix the draw schedule instead of drawing only when needed?** Drawing $u_r$ only on rejection would be correct in law but would make the generator's position depend on outcomes, so the Rust port could not be checked token for token against the Python reference on the same seed.
- **Tradeoffs accepted.** The serve loop runs a speculative request outside the batch, holding the engine thread until it completes; batched speculation (a draft per sequence inside one scheduled step) is the production design. Speculation changes speed only: greedy outputs are identical, sampled outputs are identical in law.

### 5.8 Explain it back

**Why is the bonus token taken from row $m$ and not from row $m - 1$?**

<details><summary>Answer</summary>

Row $i$ scores the token at position $i$ given the context plus $d_1 \ldots d_{i-1}$. Row $m - 1$ scored $d_m$, which was accepted; the next token needs the distribution after the whole draft, which is row $m$, the row produced by feeding $d_m$. Using row $m - 1$ would emit a second candidate for $d_m$'s position.

</details>

**With a one-hot draft at $T > 0$, what is the probability of keeping $d$, and what is sampled on rejection?**

<details><summary>Answer</summary>

$q(d) = 1$, so $d$ is kept when $u_a < p(d)$: probability $p(d)$. On rejection the residual is $\max(0, p - q)$, which is $p$ with $d$'s entry zeroed (since $p(d) - 1 \le 0$), renormalized by $1 - p(d)$. The total probability of emitting $d$ is $p(d)$ and of any $y \ne d$ is $(1 - p(d)) \cdot p(y)/(1 - p(d)) = p(y)$: exactly $p$.

</details>

**Why does the solution draw $u_r$ even when the draft is accepted?**

<details><summary>Answer</summary>

So the number of draws per verified position is always two, independent of the outcome. The generator's state after a round then depends only on how many positions were verified, which makes the Rust port's draws, and therefore its tokens, equal to the Python reference's on the same seed and logits. It costs one wasted uniform per accepted token.

</details>

**After a round that keeps 5 of 10 cached positions with $B = 4$, which KV calls happen and why is the partial block not released?**

<details><summary>Answer</summary>

`need = ceil(5/4) = 2`, so the third block is released (one reference) and the second block's fill is set to 1. The second block still holds position 4, which is kept; releasing it would lose kept KV. Only the fill changes, so the next round's writes land at the right offset.

</details>

**What breaks if `room` were `max_new - emitted` instead of `max_new - emitted - 1`?**

<details><summary>Answer</summary>

A fully accepted draft of `room` tokens plus the bonus would emit one token past `max_new`. The loop would stop at `max_new` when emitting, but the forward would already have written KV for a position past the request's budget, and that position could exceed the context the scheduler reserved for it.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Copying after the FIRST earlier occurrence | stale drafts on long repetitive contexts; acceptance drops | `prompt_lookup` scans `i` from `l - n - 1` down to 0 and takes the first match; see 5.2 |
| Stopping the continuation at the suffix | no draft on periodic text, the best case for lookup | the continuation's end is capped at `l`, so it may run into the suffix; see 5.2 |
| N-gram ties to the highest id | drafts differ from the specified ones | `max_by` compares counts, then ids reversed, so the lowest id wins; see 5.2 |
| Greedy penalties not seeing tokens accepted this round | greedy speculative output differs from greedy | `hist` grows with every emitted token and every row is penalized with it; see 5.4 |
| Accepting with probability $p(x)$, ignoring $q(x)$ | sampled output follows the wrong law whenever the draft is stochastic | `speculative_step` tests `u_accept < p[x] / q[x]`; see 5.3 |
| Residual $\lvert p - q \rvert$ | rejected positions favor the drafted token | `residual` takes `(a - b).max(0.0)`; see 5.3 |
| Drawing $u_r$ only on rejection | the generator drifts from Python's after the first acceptance | `verify_draft` draws both uniforms at every sampled position; see 5.4 |
| The bonus token from the last draft's row | a full acceptance emits the wrong next token | the bonus is sampled from `rows[m]`; see 5.4 |
| $u_r$ drawn before $u_a$ | seeded streams differ from Python's | `u_accept` is drawn first, then `u_resample` |
| Rejected drafts' KV kept | later tokens attend to tokens that were never emitted | `generate_stream` truncates to `base + pending + n_accepted` after every round; see 5.6 |
| Drafts not shortened near `max_new` | KV written past the request's budget | the draft is capped at `room = max_new - emitted - 1`; see 5.6 |
| Resampling from $p$, not the residual | the drafted token is over-represented | a rejection samples `residual(p, q)`; see 5.3 |
| Releasing the partial last block | the kept positions lose their KV | `rollback` keeps `ceil(keep / B)` blocks and only refills the last; see 5.5 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L8.6` | the Python verifier and prompt lookup this port is held to, case by case |
| Back | `L10.1` | `apply_penalties`, `distribution`, `argmax`, `sample`, `stream`, and `KvPool` |
| Back | `L10.1` | release and fill on the Rust pool during rollback |
| Forward | `L10.5` | runtime config enables speculative decoding through the request-serving path |

The serve loop (`L10.5`) routes requests through `RunnerTarget` when `[engine].speculative` names a draft. This path holds one request on the engine thread until generation completes; the ordinary scheduler path remains active when speculation is absent. `L10.7` exports `SpecStats::accept_rate` as `tl_engine_spec_accept_rate`; with `prompt_lookup` on, the system's greedy output is unchanged.

## Going further

| This solution | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| prompt lookup and n-gram drafts | vLLM `ngram` speculative method | the same idea inside a batched GPU engine | [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) |
| a chain of drafts | Medusa, EAGLE | extra heads that draft a tree of continuations; tree attention verifies them at once | [EAGLE](https://arxiv.org/abs/2401.15077), [Medusa](https://arxiv.org/abs/2401.10774) |
| one target pass per round | SpecInfer, Sequoia | token trees and draft budgets tuned to hardware | [SpecInfer](https://arxiv.org/abs/2305.09781) |
| a deterministic draft model | llama.cpp `--draft` | a small model of the same family as the draft | [llama.cpp speculative example](https://github.com/ggml-org/llama.cpp/tree/master/examples/speculative) |
