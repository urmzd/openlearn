# Speculative decoding in the engine

Decoding one token costs one forward pass over the whole model, and at the batch sizes a laptop can afford, that pass is limited by reading the weights, not by arithmetic. Checking five tokens costs about as much as producing one. So if something cheap can guess the next few tokens, one target pass can verify all of them at once.

[Speculative decoding](../p08-inference/06-speculative-decoding.md) in the inference chapter proved, in Python, the rule that makes this exact. The Rust engine from [the tokio server](05-openai-server-on-tokio.md) still decodes one token per pass, and the workloads it serves are full of repetition: code edits, retrieval answers quoting the context, JSON tool arguments that echo the prompt. This section ports the verifier to the engine, adds two drafts that need no second model (prompt lookup and an n-gram table over the request's own tokens), and gives back the KV of rejected drafts so that speculation never leaks blocks.

The Python implementation is the specification this port is held to: [`py/tinyllm/infer/spec.pyi`](../../../../course/contracts/py/tinyllm/infer/spec.pyi). You will also use the sampler (`apply_penalties`, `distribution`, `sample`, `stream`) and `KvPool` from [the model runner and sampler](01-model-runner-and-sampler.md), and the Rust pool from [the paged KV block pool](../p08-inference/08-paged-kv-block-pool.md) behind the rollback. The scheduler and block manager it plugs into are in [continuous batching](02-continuous-batching.md) and [the block manager and prefix cache](04-block-manager-and-prefix-cache.md). The probability argument comes from [rejection sampling and residual distributions](../../mathematics-for-computing/07-probability-statistics/06-rejection-sampling-and-residual-distributions.md) in *Mathematics for Computing*.

The other contracts are the sampling order and draws in [`spec/sampling.md`](../../../../course/contracts/spec/sampling.md) and [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md), the configuration key `[engine].speculative` in [`config/runtime.schema.json`](../../../../course/contracts/config/runtime.schema.json), and the gauge `tl.engine.spec_accept_rate` in [`otel/metrics.yaml`](../../../../course/contracts/otel/metrics.yaml). The solution is [`tl-engine/src/spec.rs`](../../../../course/solutions/rust/crates/tl-engine/src/spec.rs): `prompt_lookup` and `ngram_draft` over the request's own context, the verifier `verify_draft` (greedy acceptance, and accept-or-resample `speculative_step` at temperature above 0), KV `rollback` on rejection, and the generation loop `generate_stream` over a `Target` (`RunnerTarget` in production).

> **Learning objectives**
> - Compute a prompt-lookup or n-gram draft from a context by hand.
> - Verify a draft greedily and with accept-or-resample, and prove that the emitted token's distribution is exactly the target's.
> - Fix a draw schedule that makes a Rust port reproduce a Python reference token for token on the same seed.
> - Roll back the KV of rejected draft positions without leaking blocks or corrupting the prefix cache.

## Drafts from the context

Some notation first.

| Symbol | Meaning | Type |
|---|---|---|
| $x_1 \ldots x_t$ | the context: prompt plus emitted tokens | `u32` ids |
| $d_1 \ldots d_m$ | the draft, $m \le k$ | `u32` ids |
| $p_i$ | the target's sampling distribution after context $+ d_1 \ldots d_{i-1}$ | `f64[V]` |
| $q_i$ | the draft's distribution for $d_i$ (one-hot for a deterministic draft) | `f64[V]` |
| $u_a, u_r$ | the two uniforms drawn per verified position | `f64` in $[0, 1)$ |
| $B$ | tokens per KV block | `usize` |

The draft must be cheap and deterministic. Both drafts here read only the request's own tokens, so neither needs a model.

**Prompt lookup** works like this: for $n$ from `max_ngram` down to `min_ngram`, take the last $n$ tokens and find their most recent earlier occurrence (a start $i < t - n$). The draft is the next $k$ tokens after it, which may run into the suffix itself. The first $n$ with a continuation wins. Long suffixes come first because they are the most specific match.

**The n-gram draft** builds one token at a time. For each draft token, count what followed the last $m$ tokens anywhere earlier in the context (drafted tokens included), for $m$ = `max_ngram - 1` down to 1, and take the most frequent, with ties going to the lowest id. Stop when nothing matches.

Both drafts are deterministic, so $q_i$ is one-hot and $p(d)/q(d) = p(d)$.

## One round of speculation

With a draft in hand, one target pass scores every position. The target extends its cache with the pending token (the previous round's last emitted token, or the prompt in the first round) followed by $d_1 \ldots d_m$, in one forward pass. That gives $m + 1$ rows of logits: row $i$ scores position $i$ of the draft, and row $m$ scores the token after the whole draft. The verifier then walks the draft.

At temperature zero (**greedy**), it accepts $d_i$ while it equals the argmax of row $i$, with penalties applied and seeing the tokens accepted so far. The first mismatch emits the argmax instead and stops. After $m$ acceptances it emits the argmax of row $m$, the bonus token. No random draws are made.

At $T > 0$ (**sampled**), it draws $u_a$ then $u_r$, both always, so the number of draws does not depend on the outcome. It accepts $d_i$ when $u_a < p_i(d_i)/q_i(d_i)$. Otherwise it emits a draw from the residual $r = \mathrm{normalize}(\max(0, p_i - q_i))$ using $u_r$, and stops. After $m$ acceptances it emits the sampler's `sample` of row $m$, which is one draw.

Every round emits between 1 and $m + 1$ tokens, and never more than the request has left: the draft is shortened to `max_new - emitted - 1`.

## Why the distribution is exact

Correctness must not depend on the draft, and the accept-or-resample rule guarantees it. For a proposed $x \sim q$,

$$P(\text{emit } x) = q(x)\min\!\left(1, \frac{p(x)}{q(x)}\right) + \left(1 - \sum_y \min(p(y), q(y))\right) r(x).$$

Since $\min(q, p) + \max(0, p - q) = p$ pointwise, and the rejected mass is exactly the residual's normalizer, the sum is $p(x)$. For a one-hot draft this reads: keep $d$ with probability $p(d)$, else sample from $p$ with $d$ removed and renormalized. At $T = 0$ the whole rule collapses to "accept while the draft equals the argmax", which is why greedy output with speculation is token for token greedy output without it.

## Rolling back rejected KV

The forward wrote KV for all $m$ draft positions, whether or not they were kept. After keeping $n$ of them, the cache must hold the base length plus the pending tokens plus $n$. The blocks past $\lceil \text{keep}/B \rceil$ are released, one reference each, so a block shared through the prefix cache survives for its other owner, and the last kept block's fill is set to what it still holds. A block that a rejected draft filled must never be registered in the prefix index.

## Working it by hand

Four small examples cover every rule.

**Prompt lookup.** Context `[5 6 7 8 5 6]`, $k = 3$, $n$ from 3 down to 1. The 3-suffix `[8 5 6]` never occurred earlier. The 2-suffix `[5 6]` occurred at position 0, so the draft is what followed it: `[7 8 5]`.

**A greedy round.** Four token ids, draft `[2 1]`, and these rows:

| Row | Logits | Argmax |
|---|---|---|
| 0 | `[0 1 3 0.5]` | 2 = $d_1$: accept |
| 1 | `[0 1 0.5 2]` | 3 $\ne d_2 = 1$: emit 3, stop |
| 2 | `[5 0 0 0]` | (not reached) |

The round emits `[2 3]` with one draft accepted and no draws. With draft `[2 3]` instead, both are accepted and row 2's argmax 0 is the bonus: `[2 3 0]`. The logprob of the first emitted 2 is $3 - \ln(e^0 + e^1 + e^3 + e^{0.5})$.

**A sampled step.** $p = [0.2, 0.5, 0.3]$, and the draft proposed $x = 0$ from $q = [0.6, 0.2, 0.2]$, so it is accepted with probability $0.2/0.6 = 1/3$. With $u_a = 0.3$: accept. With $u_a = 0.5$: reject. The residual is $\max(0, p - q) = [0, 0.3, 0.1]$, normalized $[0, 0.75, 0.25]$; $u_r = 0.8$ walks the running sum $0, 0.75, 1.0$ and lands on id 2.

**A rollback.** $B = 4$; 10 cached positions fill three blocks (4, 4, 2). Keeping 5 positions needs $\lceil 5/4 \rceil = 2$ blocks, so block 3 is released and block 2's fill becomes 1.

## Designing the interface

Everything lives in one module.

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

The solution declares `pub mod spec;` in `tl-engine/src/lib.rs`. The engine turns `[engine].speculative = { draft = "prompt_lookup", k = 4 }` into a `SpecConfig` with `SpecConfig::parse`. Each round then runs `propose`, one forward over the pending token plus the draft, `verify_draft`, and a cut of the sequence's block table to the kept length.

## Building it step by step

### Reasoning from first principles

Seven observations shape the code.

1. **One target pass can score many positions.** Feeding the pending token plus $m$ draft tokens gives $m + 1$ rows of logits for about the cost of one decode step. So the round is: draft cheaply, score once, keep a prefix.
2. **The draft must be cheap and deterministic.** Prompt lookup and the n-gram table read only the request's own tokens; neither needs a model, and both give a one-hot $q$, so $p(d)/q(d) = p(d)$.
3. **Correctness must not depend on the draft.** The accept test $u_a < p(x)/q(x)$ with a residual resample on rejection makes the emitted token's law exactly $p$, whatever $q$ was. At $T = 0$ this collapses to "accept while the draft equals the argmax".
4. **The port must reproduce the Python reference draw for draw.** So the draw schedule is fixed regardless of outcome: $u_a$ then $u_r$ at every verified position, and one draw for the bonus. A seeded request then gives the same tokens in Python and Rust.
5. **Penalties must see the round's own accepted tokens.** Row $i$ was computed after $d_1 \ldots d_{i-1}$; a repetition penalty applied to it must count them, or greedy speculative output differs from plain greedy.
6. **Rejected positions' KV must go.** The forward wrote KV for every fed position; after keeping $n$, the cache must hold exactly `base + pending + n` positions, which means releasing whole blocks past $\lceil \text{keep}/B \rceil$ and shrinking the last block's fill.
7. **A round never overshoots the budget.** With `room = max_new - emitted - 1`, a draft of at most `room` plus the one token the verifier always adds emits at most `max_new - emitted` tokens.

### The drafts

Prompt lookup is a double loop over suffix length and start position, in [`spec.rs`](../../../../course/solutions/rust/crates/tl-engine/src/spec.rs):

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

Trace the hand example `[5 6 7 8 5 6]` with $k = 3$: `top = 3`; $n = 3$ finds no earlier `[8 5 6]`; $n = 2$ scans $i = 3, 2, 1, 0$ and matches at $i = 0$; `end = min(5, 6) = 5`, so the draft is `ctx[2..5]` = `[7 8 5]`. The last `5` is part of the suffix itself, which is why `end` is capped at `l`: on periodic text the continuation is the period, and stopping at the suffix would give no draft in exactly the case where lookup works best.

`ngram_draft` counts what followed the last $m$ tokens, for $m$ from `n - 1` down to 1, and picks with

```rust
counts.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
```

The reversed id comparison `b.0.cmp(a.0)` makes the lower id the "greater" one on a count tie, so ties go to the lowest id deterministically, independent of `HashMap` iteration order.

### The accept-or-resample step

The sampled verification rule is two small functions:

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

On the hand-worked sampled step, $p(0)/q(0) = 0.2/0.6 = 1/3$; $u_a = 0.3 < 1/3$ accepts and $u_a = 0.5$ rejects. The residual is $[0, 0.75, 0.25]$, and `sample_dense` with $u_r = 0.8$ walks $0, 0.75, 1.0$ and returns the first id where $0.8 < c$: id 2.

### Verifying a draft

`verify_draft` walks the draft and applies either rule, keeping a running history so that penalties see what this round has already emitted:

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

On the hand-worked greedy round, row 0's argmax 2 equals $d_1$ and is emitted; row 1's argmax 3 differs from $d_2 = 1$, so 3 is emitted and the round ends with `[2 3]`, `n_accepted = 1`. With draft `[2 3]` both match and the bonus is row 2's argmax 0. At $T = 0$ the bonus goes through `sample`, which is greedy at temperature 0, so the greedy path draws nothing.

### Rollback

Once the verifier says how many positions to keep, `rollback` gives the rest back:

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

On the hand-worked rollback, $B = 4$, `keep = 5`, and `need = 2`: one block is popped and released, and the last kept block's fill is $5 - 1 \cdot 4 = 1$. With `keep = 0`, `need = 0`, every block is released and there is no last block to refill. Releasing rather than freeing is what protects the prefix cache: a block another sequence also references only loses this sequence's reference.

### The generation loop

The loop ties drafting, scoring, verifying, and truncating together:

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

After `truncate`, the cache holds the pending tokens and the accepted drafts, but not the verifier's own last token: that token has not been fed yet, so it becomes the next round's `pending`. That is why `tail` starts at `pending.len() - 1`: the row after the last pending token is the one that scores $d_1$. `generate` is `generate_stream` with a listener that always answers true.

`RunnerTarget` feeds every id as its own one-token sequence at its own position over the request's blocks, so one forward returns one row per fed id. Its `truncate` applies the same release-and-refill rule as `rollback` to the request's blocks, and its `Drop` releases every block it still holds, so an early return cannot leak.

The full solution is [`spec.rs`](../../../../course/solutions/rust/crates/tl-engine/src/spec.rs).

### Why this design

Start with cost. Prompt lookup is $O(n_{\max} \cdot t \cdot n)$ in the worst case per round (a scan for each suffix length), and $t$ is at most the context; on a 2,000-token context that is microseconds against a forward pass of milliseconds. Verification is $O(mV)$ for the dense distributions. A round with $a$ accepted drafts emits $a + 1$ tokens for one target pass.

Why prompt lookup and n-grams rather than a draft model? There is no second model to load and no draft KV to manage, and the workloads this engine serves (code edits, quoting retrieved context, JSON echoing the prompt) repeat their own context. A draft model helps on novel text but costs memory and a second forward per token.

Why fix the draw schedule instead of drawing only when needed? Drawing $u_r$ only on rejection would be correct in law, but it would make the generator's position depend on outcomes, so the Rust port could not be checked token for token against the Python reference on the same seed.

One tradeoff is accepted: the serve loop runs a speculative request outside the batch, holding the engine thread until it completes. Batched speculation, with a draft per sequence inside one scheduled step, is the production design. Either way, speculation changes speed only: greedy outputs are identical, and sampled outputs are identical in law.

## What can go wrong

The verification and prompt-lookup tests compare against what the Python reference answers on 120 verification cases (exact f32 logits, drafts that agree for a while, optional draft distributions, every sampling knob, 64-bit seeds) and 150 prompt-lookup contexts. The generation tests run a fake target whose logits are a fixed function of the context. A correct solution handles all of these:

| Case | Checks | Why it matters downstream |
|---|---|---|
| The hand prompt lookup | the worked draft, a continuation into the suffix, no draft | the worked example |
| Prompt lookup equals Python | 150 contexts equal to `PromptLookupDraft` | most recent occurrence, longest suffix first |
| The hand n-gram draft | counts, ties to the lowest id, longest context first, back-off, stop | the second draft |
| Runtime config | the three drafts; `k` 0 and 17 and a typo refused | a bad runtime.toml fails at startup |
| The hand greedy round | the worked round, no draws, logprobs, the bonus token | the greedy rule |
| The hand sampled step | accept, reject, residual, a q that never proposed x | rejection sampling by hand |
| Verification equals Python | 120 cases: emitted tokens, n_accepted, and the generator's next u32 | the port draws exactly what Python draws |
| Bad shapes | rows, draft distributions, out-of-vocabulary drafts | an error, not a panic, in the step loop |
| Greedy with speculation equals greedy without | every draft kind, three prompts: the greedy tokens, cached positions, fewer passes with prompt lookup | speculation changes speed only |
| EOS and `max_tokens` | exactly `max_new` tokens; EOS inside an accepted run stops there | requests end where they should |
| Sampled speculation keeps the target distribution | 4000 seeds: the first token's chi-square against the target softmax at p > 1e-3 | sampled output is unchanged in law |
| The hand rollback | the worked rollback, a full block, keep 0, `blocks_for` | the worked example |
| Rollback drops one reference only | a block shared with another sequence survives | the prefix cache is never corrupted |
| Blocks conserved across rollbacks | 200 seeded requests with many rejections on a real pool: blocks match positions, nothing leaks | weeks of speculation do not shrink the pool |
| Streaming generation | every emitted token reaches the listener in order; a listener that says stop ends generation there | the engine streams a speculative answer and frees it when the client leaves |
| `RunnerTarget` on a real model | on the tiny [Llama-family model](../p07-modern-block/09-llama-family-model.md): one row per fed id, each equal to a fresh run of its prefix; prompt lookup emits plain greedy's tokens | speculation works on a real model, not only the fake |

The pitfalls fall into three groups: drafts that are worse than specified, verification that is subtly wrong, and KV that is not given back correctly.

Draft mistakes cost speed or determinism, not correctness. Copying after the *first* earlier occurrence instead of the most recent gives stale drafts on long repetitive contexts, and acceptance drops; `prompt_lookup` scans `i` from `l - n - 1` down to 0 and takes the first match. Stopping the continuation at the suffix gives no draft on periodic text, the best case for lookup; the continuation's end is capped at `l` so it may run into the suffix. N-gram ties broken to the highest id make drafts differ from the specified ones; `max_by` compares counts, then ids reversed, so the lowest id wins.

> **Warning:** Verification mistakes are the dangerous ones, because sampled output still looks plausible while following the wrong distribution. Accepting with probability $p(x)$ and ignoring $q(x)$ gives the wrong law whenever the draft is stochastic; `speculative_step` tests `u_accept < p[x] / q[x]`. Using the residual $\lvert p - q \rvert$, or resampling from $p$ instead of the residual, makes rejected positions favor the drafted token; `residual` takes `(a - b).max(0.0)` and a rejection samples `residual(p, q)`.

Other verification mistakes break the equivalence with the reference or with plain greedy. Greedy penalties that do not see tokens accepted this round make greedy speculative output differ from greedy; `hist` grows with every emitted token and every row is penalized with it. Drawing $u_r$ only on rejection makes the generator drift from Python's after the first acceptance, and drawing $u_r$ before $u_a$ makes seeded streams differ; `verify_draft` draws `u_accept` first, then `u_resample`, at every sampled position. Taking the bonus token from the last draft's row makes a full acceptance emit the wrong next token; the bonus is sampled from `rows[m]`.

KV mistakes corrupt later tokens or overrun budgets. Keeping rejected drafts' KV makes later tokens attend to tokens that were never emitted; `generate_stream` truncates to `base + pending + n_accepted` after every round. Not shortening drafts near `max_new` writes KV past the request's budget; the draft is capped at `room = max_new - emitted - 1`. Releasing the partial last block loses the kept positions' KV; `rollback` keeps `ceil(keep / B)` blocks and only refills the last.

## Where this leads

This section is held, case by case, to the Python verifier and prompt lookup of [speculative decoding](../p08-inference/06-speculative-decoding.md). It uses `apply_penalties`, `distribution`, `argmax`, `sample`, `stream`, and `KvPool` from [the model runner and sampler](01-model-runner-and-sampler.md), and release and fill on the Rust pool during rollback.

Going forward, the serve loop of [the tokio server](05-openai-server-on-tokio.md) routes requests through `RunnerTarget` when `[engine].speculative` names a draft. This path holds one request on the engine thread until generation completes; the ordinary scheduler path remains active when speculation is absent. [Serving metrics and tracing](07-serving-metrics-and-tracing.md) exports `SpecStats::accept_rate` as `tl_engine_spec_accept_rate`. With `prompt_lookup` on, the end-to-end system's greedy output is unchanged.

Production engines push the same idea further. vLLM's `ngram` speculative method runs prompt-lookup-style drafts inside a batched GPU engine. Medusa and EAGLE replace a chain of drafts with extra heads that draft a tree of continuations, verified at once by tree attention. SpecInfer and Sequoia go beyond one target pass per round with token trees and draft budgets tuned to hardware. And llama.cpp's `--draft` uses a small model of the same family as the draft.

### Further reading

- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (free).
- Chen et al., [Accelerating LLM Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (free).
- [Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding) (free).
- [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html).
- [EAGLE](https://arxiv.org/abs/2401.15077), [Medusa](https://arxiv.org/abs/2401.10774), and [SpecInfer](https://arxiv.org/abs/2305.09781).
- The [llama.cpp speculative example](https://github.com/ggml-org/llama.cpp/tree/master/examples/speculative).

## Key terms

- **Speculative decoding**: drafting several tokens cheaply and verifying them with one target forward pass, keeping the longest acceptable prefix.
- **Prompt lookup**: a draft that copies what followed the most recent earlier occurrence of the context's suffix, longest suffix first.
- **N-gram draft**: a draft that picks, token by token, the most frequent continuation of the last few tokens in the context, ties to the lowest id.
- **Accept-or-resample**: accept a drafted $x$ when $u_a < p(x)/q(x)$, else sample from the residual; the emitted token's law is exactly $p$.
- **Residual distribution**: $\mathrm{normalize}(\max(0, p - q))$, the distribution sampled on rejection.
- **Bonus token**: the extra token sampled from row $m$ when every draft token is accepted.
- **Rollback**: releasing the KV blocks of rejected draft positions and resetting the last kept block's fill.

## Summary

- Prompt lookup copies what followed the most recent earlier occurrence of the context's suffix, trying long suffixes first; it costs no model and shines on repetitive text.
- Greedy verification keeps drafts while they equal the target's argmax, so greedy output with speculation is token for token greedy output without it.
- With sampling, accept-or-resample keeps the target's distribution exactly, whatever the draft proposes.
- The port draws exactly what the Python reference draws, in the same order ($u_a$, then $u_r$, both always; one draw for the bonus token), so its tokens equal the reference's on the same logits and seed.
- A rejected draft's KV is given back: after every round the sequence holds exactly the blocks its cached positions need, and shared blocks lose only one reference.

## Review questions

1. Why is the bonus token taken from row $m$ and not from row $m - 1$?
   <details><summary>Answer</summary>

   Row $i$ scores the token at position $i$ given the context plus $d_1 \ldots d_{i-1}$. Row $m - 1$ scored $d_m$, which was accepted; the next token needs the distribution after the whole draft, which is row $m$, the row produced by feeding $d_m$. Using row $m - 1$ would emit a second candidate for $d_m$'s position.

   </details>

2. With a one-hot draft at $T > 0$, what is the probability of keeping $d$, and what is sampled on rejection?
   <details><summary>Answer</summary>

   $q(d) = 1$, so $d$ is kept when $u_a < p(d)$: probability $p(d)$. On rejection the residual is $\max(0, p - q)$, which is $p$ with $d$'s entry zeroed (since $p(d) - 1 \le 0$), renormalized by $1 - p(d)$. The total probability of emitting $d$ is $p(d)$ and of any $y \ne d$ is $(1 - p(d)) \cdot p(y)/(1 - p(d)) = p(y)$: exactly $p$.

   </details>

3. Why does the solution draw $u_r$ even when the draft is accepted?
   <details><summary>Answer</summary>

   So the number of draws per verified position is always two, independent of the outcome. The generator's state after a round then depends only on how many positions were verified, which makes the Rust port's draws, and therefore its tokens, equal to the Python reference's on the same seed and logits. It costs one wasted uniform per accepted token.

   </details>

4. After a round that keeps 5 of 10 cached positions with $B = 4$, which KV calls happen and why is the partial block not released?
   <details><summary>Answer</summary>

   `need = ceil(5/4) = 2`, so the third block is released (one reference) and the second block's fill is set to 1. The second block still holds position 4, which is kept; releasing it would lose kept KV. Only the fill changes, so the next round's writes land at the right offset.

   </details>

5. What breaks if `room` were `max_new - emitted` instead of `max_new - emitted - 1`?
   <details><summary>Answer</summary>

   A fully accepted draft of `room` tokens plus the bonus would emit one token past `max_new`. The loop would stop at `max_new` when emitting, but the forward would already have written KV for a position past the request's budget, and that position could exceed the context the scheduler reserved for it.

   </details>

6. Context `[5 6 7 8 5 6]`, $k = 3$. Why does the prompt-lookup draft `[7 8 5]` include a token from the suffix itself, and why is that desirable?
   <details><summary>Answer</summary>

   The 2-suffix `[5 6]` matches at position 0, and the continuation runs from index 2 to `min(0 + 2 + 3, 6) = 5`, so it includes index 4, which is the first token of the suffix. Capping the end at `l` rather than at `l - n` lets the draft follow the period of periodic text; stopping at the suffix would produce no draft precisely where prompt lookup should do best.

   </details>
