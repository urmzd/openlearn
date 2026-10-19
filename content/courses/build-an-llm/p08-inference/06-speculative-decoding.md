# Speculative decoding: n-gram, prompt-lookup, and model drafts

## Overview

| | |
|---|---|
| **Module** | `L8.6` · build · Python · Pass 6 |
| **Solution** | [`spec.py`](../../../../course/solutions/python/tinyllm/infer/spec.py): `verify_draft`, `speculative_generate`, and three drafts: `NGramDraft`, `PromptLookupDraft`, `ModelDraft` (the `DraftModel` protocol is in the contract) |
| **Contract** | [`course/contracts/py/tinyllm/infer/spec.pyi`](../../../../course/contracts/py/tinyllm/infer/spec.pyi) |
| **Needs** | [`M07.6` rejection sampling](../../../tracks/math/07-probability-statistics/06-rejection-sampling-and-residual-distributions.md) (`speculative_step`), [`L8.2` KV cache and generate](02-kv-cache-and-generate.md) (`KVCache.truncate`, `cache_dims`, `Generation`), [`L2.1` n-gram LM](../p02-statistical-lm/01-ngram-kneser-ney.md) (`NGramLM`), [`L8.1` sampling](01-sampling-and-logit-processors.md) (`sample`, `sampling_distribution`, `token_logprobs`, `request_rng`). Reading: [`M11.1` entropy and KL](../../../tracks/math/11-information-theory/01-entropy-cross-entropy-and-kl.md) |
| **Used by** | `L10.8` speculative decoding in the Rust engine, held to this module on shared draft and target logits; on a real model, greedy speculation with an n-gram draft gives token-identical output to plain decoding and reports `acceptance_rate` |
| **Optional depth** | Leviathan, Kalman, and Matias, *Fast Inference from Transformers via Speculative Decoding* (ICML 2023); Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling* (2023); Saxena, *Prompt Lookup Decoding* (2023) |

## Key Takeaways

- **Decoding one token costs one pass over the weights; scoring $k + 1$ tokens costs about the same**, so a draft that guesses $k$ tokens and a target that checks them in one pass can emit several tokens per pass.
- **Greedy verification never changes the output**: keep drafts while they equal the target's argmax, then add the target's own token.
- **Sampled verification keeps the target's distribution exactly**: accept $x \sim q$ with probability $\min(1, p(x)/q(x))$, otherwise draw from $\max(0, p - q)$ normalized; enumerated exactly, the output is $p$ as fractions.
- **Rejected drafts must leave no trace in the KV cache**: truncate back to the kept length, then feed the replacement token first next round.
- **Cheap drafts are enough** for repetitive text: copying from the prompt or an n-gram model cost nothing to run.

## How to read this chapter

Sections 1 to 3 build speculative decoding from first principles, prove that the verification rule is exact, and work it by hand. Section 4 fixes the interface. Section 5 walks through the solution, `verify_draft` first and then the generation loop and the three drafts, and explains why each choice is the right one. Try to predict each step before reading it.

---

## 1. Why now

L8.2's `generate` produces one token per forward pass. Profile it on SmolLM2 and the pass is dominated by reading every weight from memory once; the arithmetic for one token is small. A pass over 5 tokens (with the KV cache, 5 new query rows) reads the same weights once and costs barely more. So decoding wastes most of what each pass could do. Speculative decoding spends that slack: something cheap guesses the next $k$ tokens, the target scores all of them in one pass, and every correct guess is a token you did not pay a pass for. The trap is correctness: a guess that is accepted when the target would not have produced it changes the output, and for sampling it changes the distribution. This module builds the verification rules that make the speedup free of either, plus three drafts, and Part 10's Rust engine (L10.8) is checked against it.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | vocabulary size | int |
| $k$ | the number of draft tokens per round | int $\ge 0$ |
| $x_1, \dots, x_m$ | the draft tokens of one round, $m \le k$ | ids |
| $p_i$ | the target's sampling distribution (L8.1) at draft position $i$, given everything before $x_i$ | float64 $[V]$ |
| $q_i$ | the draft's distribution that $x_i$ was drawn from (one-hot for a deterministic draft) | float64 $[V]$ |
| $u$ | a uniform draw in $[0, 1)$ from the request's PCG32 stream | float |
| $\alpha = \sum_v \min(p(v), q(v))$ | the acceptance probability of one draft token | $[0, 1]$ |
| $\mathrm{TV}(p, q) = \tfrac12 \sum_v \lvert p(v) - q(v) \rvert$ | total variation distance; $\alpha = 1 - \mathrm{TV}$ | $[0, 1]$ |

### 2.1 The round, and the cache

The target cache holds the positions that are **kept**. One round:

1. the draft proposes $x_1..x_m$ after the context (prompt plus generated);
2. the target runs **one** forward pass over the tokens not yet in its cache: the **pending** token (the previous round's last token, or the whole prompt in the first round) followed by $x_1..x_m$. Its last $m + 1$ rows of logits are the target's predictions after the context, after $x_1$, ..., after $x_m$;
3. `verify_draft` keeps $x_1..x_n$ for some $n \le m$ and adds one more token $y$ of the target's own (a correction, or a bonus when all $m$ were kept);
4. the cache now holds $m$ draft positions, but only $n$ are kept: `KVCache.truncate(committed + len(pending) + n)` drops the rest. $y$ was never fed, so it becomes the next round's pending token.

Every round emits $n + 1 \ge 1$ tokens for one target pass, so speculation can never be slower in passes than plain decoding (with $k = 0$ it **is** plain decoding).

### 2.2 Exactness: why rejection sampling gives exactly $p$

Draw $x \sim q$. Keep it with probability $\min(1, p(x)/q(x))$ (M07.6's `rejection_accept`: $u < p(x)/q(x)$). If it is rejected, draw $y$ from the **residual** $r = \max(0, p - q) / Z$ with $Z = \sum_v \max(0, p(v) - q(v))$. For any token $v$:

$$P(\text{out} = v) = \underbrace{q(v) \min\!\left(1, \tfrac{p(v)}{q(v)}\right)}_{\text{kept}} + \underbrace{\left(1 - \alpha\right) r(v)}_{\text{replaced}} = \min(q(v), p(v)) + \max(0, p(v) - q(v)) = p(v),$$

because $1 - \alpha = 1 - \sum_v \min(p, q) = \sum_v \max(0, p - q) = Z$. Nothing in this argument needs $q$ to be good: a bad draft is rejected more often, never wrong. A **deterministic** draft (prompt lookup, greedy n-gram) proposes a fixed $x$, which is a draw from the one-hot $q = e_x$: then $x$ is kept with probability $p(x)$ and the residual is $p$ with $x$ removed, renormalized.

At position $i > 1$ the same step runs with $p_i$, the target's distribution **after** $x_1..x_{i-1}$ (which the target already computed in the same pass), and only if $x_{i-1}$ was kept. Conditioned on the kept prefix, each position is a fresh exact step, so every emitted token follows the target's distribution given the tokens before it. Penalties (repetition, presence, frequency) are part of $p_i$: their history includes the drafts kept so far in this round.

### 2.3 Greedy is the temperature-0 limit

At temperature 0 the target's distribution is one-hot on its argmax $g_i$ (L8.1, ties to the lowest id). The rule becomes: keep $x_i$ while $x_i = g_i$; at the first mismatch emit $g_i$ and stop; after $m$ matches emit $g_{m+1}$. No uniform is drawn. The emitted tokens are exactly the target's greedy tokens, which is the claim the end-to-end run checks on a real model.

### 2.4 How fast: expected tokens per pass

If each draft token is accepted independently with probability $\alpha$, a round of $k$ drafts emits $1 + \alpha + \alpha^2 + \dots + \alpha^k = \frac{1 - \alpha^{k+1}}{1 - \alpha}$ tokens on average. With $\alpha = 0.8$ and $k = 4$ that is 3.36 tokens per target pass. The cost side is the draft: an n-gram lookup or a prompt lookup is free, a smaller model costs $k$ of its own passes per round. `acceptance_rate` (accepted drafts over proposed drafts) is the number that tells you whether a draft is worth it. **Prompt lookup** shines when the output repeats the input (code edits, summaries quoting their source); an **n-gram** model trained on similar text catches common continuations; a **model draft** generalizes best and costs the most.

### 2.5 Randomness, in a fixed order

One request generator (`request_rng(seed)`, L8.1) serves the draft and the verifier in program order. Per verified draft position the verifier always draws **two** uniforms, `u_accept` then `u_resample` (M07.6's `speculative_step` takes both, and drawing both keeps the count independent of the outcome); the bonus token is L8.1's `sample` (one uniform). The Rust port (L10.8) draws in the same order, so on the same logits it emits the same ids.

## 3. Worked example by hand

**Sampled.** $V = 3$, target $p = [\tfrac12, \tfrac14, \tfrac14]$, draft $q = [\tfrac14, \tfrac12, \tfrac14]$, draft token $x = 1$.

| Step | Computation | Result |
|---|---|---|
| acceptance probability | $\min(1, p_1/q_1) = \min(1, \tfrac{1/4}{1/2})$ | $\tfrac12$ |
| $u_{accept} = 0.7$ | $0.7 < 0.5$? no | rejected |
| residual | $\max(0, p - q) = [\tfrac14, 0, 0]$, $Z = \tfrac14$ | $r = [1, 0, 0]$ |
| $u_{resample} = 0.4$ | inverse CDF of $r$ | token 0; emitted `[0]`, 0 accepted |

With $u_{accept} = 0.3 < 0.5$ the draft is kept, $u_{resample}$ is drawn and ignored, and the bonus token comes from the target's next row with a third uniform. Averaged over the draft as well ($x \sim q$): token 1 comes out only when $x = 1$ is kept, with probability $q_1 \cdot \tfrac12 = \tfrac14 = p_1$; token 0 when $x = 0$ (always kept, $\tfrac14$) or when $x = 1$ is rejected ($\tfrac12 \cdot \tfrac12$), total $\tfrac12 = p_0$. That is section 2.2's identity; enumerating every outcome on a larger example gives the same equality as exact fractions.

**Greedy.** Target rows with argmax 2, 0, 1 (`[[0, 1, 3], [2, 1, 0], [0, 5, 1]]`) and draft `[2, 1]`: position 0, draft 2 = argmax 2, keep; position 1, draft 1 $\ne$ argmax 0, emit 0 and stop. Result `[2, 0]`, one accepted, no uniform drawn. Draft `[2, 0]` matches twice and the bonus is row 2's argmax, giving `[2, 0, 1]`.

**Prompt lookup.** Context `1 2 3 9 1 2`, $k = 3$, `max_ngram = 3`: the last 3-gram `9 1 2` never occurred earlier; the last 2-gram `1 2` occurred at the start, followed by `3 9 1`: that is the draft.

## 4. The interface

```python
class DraftModel(Protocol):
    def propose(self, ctx: Sequence[int], k: int, rng: Optional[UniformSource]) -> tuple[list[int], Optional[NDArray]]
class NGramDraft:        def __init__(self, lm: NGramLM)
class PromptLookupDraft: def __init__(self, max_ngram: int = 3, min_ngram: int = 1)
class ModelDraft:        def __init__(self, model, temperature: float = 1.0)
def verify_draft(target_logits, draft_ids, draft_probs, p, history, rng, prompt=()) -> tuple[list[int], int]
def speculative_generate(target, draft, tok, prompt, p, k=4, kv_dtype=np.float32, eos_ids=()) -> Generation
```

`rng=None` asks a draft for its greedy guess (and no distribution). `speculative_generate` reports `drafted`, `accepted`, `acceptance_rate`, and `target_calls` in `Generation.stats`, besides L8.2's fields. To check it, use a toy model whose logits at a position depend only on the tokens up to it and are computed row by row in float64 (so whether the target scores one token or five cannot change a bit), and which asserts on every call that the cache holds exactly the positions before the chunk.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The sampled hand example | unit, smoke | section 3, sampled: `[0]` with 0 accepted for $u = 0.7, 0.4$; `[1, 2]` for $0.3, 0.9, 0.1$; three uniforms | the draw order L10.8 must follow |
| The greedy hand example | unit, smoke | section 3, greedy: `[2, 0]`, `[2, 0, 1]`, `[2]`; no draw | greedy verification |
| One token is exactly the target | statistical | $V = 5$, all outcomes enumerated: the first token's distribution equals $p$ as fractions | the exactness theorem |
| Two drafts are exact at each position | statistical | $k = 2$: first token $\sim p_1$; after a kept $x_1 = y_1$, second $\sim p(\cdot \mid y_1)$ | the right row and history per position |
| A deterministic draft is exact | statistical | one-hot drafts (every $x$, including one with $p(x) = 0$) give exactly $p$ | prompt lookup and n-gram drafts at temperature > 0 |
| Penalties see the kept drafts | unit | a repetition penalty rejects a second copy of a just-kept token | penalties match plain `generate` |
| Shapes are checked | boundary | rows must be $m + 1$, draft distributions $m \times V$ | no off-by-one row |
| The prompt-lookup hand example | unit, smoke | section 3; most recent occurrence; longest n-gram first; $k = 0$ | the lookup rule L10.8 ports |
| The n-gram draft follows its model | unit | greedy ids are the argmax chain; sampled rows are the model's distribution | the n-gram draft |
| The model draft syncs its cache | unit | the draft's own cache follows a changing context; proposals equal fresh greedy decoding | a model draft after rejections |
| Greedy speculation equals greedy decoding | differential | four drafts: ids equal L8.2's `generate` and a plain argmax loop | the end-to-end greedy claim |
| Rejected drafts are rolled back | property | every target pass starts at the kept length with the right token | the cache rollback |
| Stats, and one pass per round | unit | self-draft: 40 tokens in 8 passes, acceptance 1; $k = 0$: 40 passes | the speedup is real |
| Sampled speculation matches the target distribution | statistical | end to end, 400 seeds, chi-square $p > 10^{-3}$ | sampling through the whole loop |
| EOS, stop strings, and the budget | boundary | EOS ends without being emitted; a stop string cuts the text; never more than `max_tokens` | `generate`'s contract kept |
| Bad arguments | boundary | $k < 0$, empty prompt, prompt longer than the model | caller errors early |
| Logprobs are the target's | unit | logprobs equal plain `generate`'s | the API reports the target's numbers |

## 5. The solution, step by step

### From first principles

1. **A target pass costs about the same for 1 or $k + 1$ tokens** (section 1), so the design goal is: as many emitted tokens per target pass as possible, with the output unchanged (greedy) or identically distributed (sampled).
2. **The target must score the pending token and every draft in one pass.** Its row after the pending token is the prediction for $x_1$; the row after $x_i$ predicts $x_{i+1}$; the last row is the bonus. So the pass feeds `pending + drafts` and keeps the last $m + 1$ rows.
3. **Verification walks positions left to right and stops at the first rejection**, because the rows after a rejected $x_i$ were computed with $x_i$ in the context and are no longer valid predictions.
4. **Every round emits one token of the target's own**: the correction at the rejected position, or the bonus. That makes each round at least as productive as one plain decoding step and makes $k = 0$ identical to plain decoding.
5. **The emitted extra token was never fed to the target**, and the rejected drafts were. So the cache is truncated to the kept length and the extra token becomes the next round's pending input.
6. **Exactness comes from M07.6's step**, which needs $p_i$ and $q_i$ as full distributions. $p_i$ is L8.1's `sampling_distribution` with the history including the drafts kept so far (penalties depend on it); $q_i$ is the draft's row, or a one-hot for a deterministic draft.
7. **Randomness must be consumed in a fixed order** so the Rust port reproduces the stream: two uniforms per sampled position, regardless of the outcome; one for the bonus; none when greedy.

### The code, annotated

`verify_draft` is the whole correctness argument in code:

```python
def verify_draft(target_logits, draft_ids, draft_probs, p, history, rng, prompt=()):
    rows = np.asarray(target_logits, dtype=np.float64)
    m = len(draft_ids)
    # m drafts need m + 1 rows: one per draft position plus the bonus row.
    if rows.ndim != 2 or rows.shape[0] != m + 1:
        raise ValueError(...)
    V = rows.shape[1]
    q = None if draft_probs is None else np.asarray(draft_probs, dtype=np.float64)
    if q is not None and q.shape != (m, V):
        raise ValueError(...)
    hist = [int(t) for t in history]
    emitted: list[int] = []
    for i, x in enumerate(int(t) for t in draft_ids):
        if p.temperature == 0:
            # Greedy (section 2.3): the target's argmax at this position, with
            # penalties over everything emitted so far; no uniform drawn.
            g = sample(rows[i], p, hist + emitted, rng, prompt)[0]
            emitted.append(g)
            if g != x:
                return emitted, i        # g is the correction; i drafts kept
            continue                     # g == x: the draft is kept
        # Sampled (section 2.2): p_i sees the drafts kept this round.
        P = sampling_distribution(rows[i], p, hist + emitted, prompt)
        if q is None:
            Q = np.zeros(V)              # a deterministic draft is a draw from
            Q[x] = 1.0                   # the one-hot at x
        else:
            Q = q[i]
        u_accept = rng.uniform()         # always two draws, in this order,
        u_resample = rng.uniform()       # whatever the outcome (section 2.5)
        y, ok = speculative_step(P, Q, x, u_accept, u_resample)
        emitted.append(int(y))           # x if kept, else a residual draw
        if not ok:
            return emitted, i
    # All m kept: the bonus comes from the last row.
    emitted.append(sample(rows[m], p, hist + emitted, rng, prompt)[0])
    return emitted, m
```

The worked example's sampled case: `P = [1/2, 1/4, 1/4]`, `Q = [1/4, 1/2, 1/4]`, $x = 1$, $u_{accept} = 0.7 \ge 1/2$, so `speculative_step` rejects and draws token 0 from the residual with $u_{resample} = 0.4$; the function returns `([0], 0)`.

The loop in `speculative_generate` keeps the cache honest:

```python
committed = 0                 # positions in the target cache that are kept
pending = list(prompt_ids)    # ids whose keys and values are not cached yet
while len(out) < p.max_tokens:
    room = limit - committed - len(pending)          # cache positions left for drafts
    # Never draft more than fits the cache, or than the token budget allows
    # once the round's own extra token is counted (hence the - 1).
    k_eff = max(0, min(k, p.max_tokens - len(out) - 1, room))
    drafts, q = (draft.propose(prompt_ids + out, k_eff, draft_rng)
                 if k_eff > 0 else ([], None))
    drafts = [int(t) for t in drafts[:k_eff]]
    if q is not None:
        q = np.asarray(q, dtype=np.float64)[: len(drafts)]
    # One target pass over pending + drafts; keep the last m + 1 rows
    # (the row after the last pending token, then one after each draft).
    rows = _forward_rows(target, pending + drafts, committed, cache)[len(pending) - 1 :]
    calls += 1
    emitted, n_acc = verify_draft(rows, drafts, q if drafts else None, p, out, rng, prompt_ids)
    drafted += len(drafts)
    accepted += n_acc
    committed += len(pending) + n_acc
    cache.truncate(committed)     # drop the rejected drafts' keys and values
    pending = [emitted[-1]]       # the extra token is not in the cache yet
    for j, t in enumerate(emitted):
        if t in eos:              # EOS ends generation and is not emitted
            finish = "stop"
            break
        lps.append(float(token_logprobs(rows[j], p, out, prompt_ids)[t]))  # the target's
        out.append(t)
        # ... stop strings: decode, cut the text at the first stop, finish = "stop" ...
    if finish == "stop" or committed + len(pending) >= limit:
        break
```

`draft_rng` is `None` at temperature 0 (drafts propose their greedy guess) and the request's own `rng` otherwise, so the draft's draws and the verifier's draws interleave in one stream in program order. The cache is sized `min(model_max, len(prompt_ids) + p.max_tokens + k)`: room for the prompt, every emitted token, and one round of drafts beyond them.

The three drafts. Prompt lookup searches the longest suffix n-gram first, and the most recent earlier occurrence of it:

```python
def propose(self, ctx, k, rng):
    ctx = [int(t) for t in ctx]
    L = len(ctx)
    if k <= 0:
        return [], None
    for n in range(min(self.max_ngram, L - 1), self.min_ngram - 1, -1):  # longest first
        suffix = ctx[L - n :]
        for i in range(L - n - 1, -1, -1):  # most recent earlier occurrence first
            if ctx[i : i + n] == suffix:
                cont = ctx[i + n : i + n + k]       # at most k tokens
                if cont:
                    return cont, None               # deterministic: no distribution
    return [], None
```

On the worked example `1 2 3 9 1 2`: $n = 3$ finds no earlier `9 1 2`; $n = 2$ finds `1 2` at $i = 0$ and returns `ctx[2:5] = [3, 9, 1]`.

`NGramDraft` chains $k$ draws from `lm.logprobs(cur)`, appending each token to its own context; with `rng=None` it is the argmax chain, otherwise L8.1's `sample` with the exact distribution kept as the row. `ModelDraft` owns a KV cache and keeps it in sync with a context that the verifier may have cut:

```python
# Sync the cache with ctx: keep the common prefix, but always re-feed
# at least the last id, whose logits are the ones we need.
common = 0
while common < min(len(self._ids), len(ctx) - 1) and self._ids[common] == ctx[common]:
    common += 1
self._cache.truncate(common)        # forget rejected drafts and anything after them
self._ids = ctx[:common]
logits = _forward_rows(self.model, ctx[common:], common, self._cache)[-1]
self._ids = list(ctx)
# ... then k draws, feeding each drafted token back for the next one
```

Full solution: [`spec.py`](../../../../course/solutions/python/tinyllm/infer/spec.py).

### Why this solution

- **Cost.** One target pass per round over $\lvert\text{pending}\rvert + m$ tokens, plus the draft's cost: none for prompt lookup ($O(L \cdot n_{\max})$ list comparisons), $k$ table lookups for the n-gram, $k$ small-model steps for `ModelDraft`. Expected tokens per pass $\frac{1 - \alpha^{k+1}}{1 - \alpha}$ (section 2.4).
- **Against accepting a draft when it is "likely enough".** Any threshold rule changes the distribution. The rejection-and-residual rule keeps $p$ exactly for any $q$ (section 2.2), which is why enumerating outcomes gives exactly $p$ as fractions.
- **Against re-running the target on the kept prefix.** Truncating the cache and feeding only the extra token reuses every kept position's keys and values; recomputing would spend the pass the method saved.
- **Against a separate generator for the draft.** One stream in program order is reproducible by the Rust port draw for draw; two streams would need a second seed and an agreed interleaving anyway.
- **Tradeoff accepted.** Drafts are a single path (no tree), so one early rejection discards the rest of the round. Tree speculation (Going further) verifies several paths in one pass at the cost of tree-masked attention.

### Explain it back

1. Why does a round always emit at least one token, even if the first draft is rejected?
   <details><summary>Answer</summary>The target's row at the rejected position is valid (it was computed from kept context only), so the rejection step draws a replacement from the residual (or, greedy, emits the argmax). That token is the target's own, so the round is never worse than one plain decoding step.</details>
2. Why are both uniforms drawn even when the draft is accepted?
   <details><summary>Answer</summary>So the number of draws per position does not depend on the outcome. The Rust port can then consume the same stream position by position and emit the same ids on the same logits; skipping a draw on acceptance would shift every later draw.</details>
3. After a round keeps $n$ of $m$ drafts, why is the cache truncated to `committed + len(pending) + n` and not `+ n + 1`?
   <details><summary>Answer</summary>The cache holds pending plus all $m$ drafts. Only the pending tokens and the $n$ kept drafts are valid; the extra token $y$ was emitted from a row but never fed, so it has no keys or values yet. It becomes next round's pending input.</details>
4. How does a deterministic draft fit the sampled rule, and what happens when $p(x) = 0$?
   <details><summary>Answer</summary>It is a draw from the one-hot $q = e_x$, so $x$ is kept with probability $\min(1, p(x)/1) = p(x)$, and the residual is $p$ with $x$ removed, renormalized. When $p(x) = 0$ the draft is always rejected and the output is drawn from $p$ itself, which is still exact.</details>
5. Why must the history passed to `sampling_distribution` include `emitted`?
   <details><summary>Answer</summary>Penalties depend on the tokens generated so far. Plain decoding would have seen the kept drafts before choosing the next token, so $p_i$ must see them too, or a repetition penalty would fail to discourage a second copy of a token kept earlier in the same round.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. treating the draft as if it sampled from the target ($q = p$), or ignoring `draft_probs` | every draft kept: the output follows the draft, not the target | passes the draft's own row `q[i]` as `Q` to `speculative_step` |
| 2. a deterministic draft treated as uniform, or its one-hot at the wrong index | biased output, or a crash on $q(x) = 0$ | builds `Q` as the one-hot at the draft id `x` |
| 3. resampling from $p$ instead of the residual after a rejection | tokens the draft favors are over-represented | delegates the rejected case to `speculative_step`, which draws from $\max(0, p - q)$ |
| 4. drawing the uniforms in another order | Python and Rust streams diverge | `u_accept` then `u_resample` on every sampled position, then one draw for the bonus |
| 5. greedy: keeping the draft on a mismatch | the output changes with the draft | returns at the first `g != x`, with `g` emitted |
| 6. the bonus from the wrong row | the token after a fully kept draft is wrong | the bonus samples `rows[m]`, the row after the last draft |
| 7. penalties that do not see the drafts kept this round | sampled and greedy output differ from plain `generate` with penalties | passes `hist + emitted` as the history |
| 8. prompt lookup: oldest occurrence, shortest n-gram first, or ignoring $k$ | different drafts than L10.8; longer drafts than asked | loops $n$ downward, $i$ downward, and slices at most $k$ tokens |
| 9. a model draft that does not truncate or re-feed its own cache | stale keys: proposals drift from the draft model's real greedy output | truncates to the common prefix and always re-feeds at least the last id |
| 10. no rollback, or the extra token counted as cached, or never fed | the next pass attends to rejected tokens; output diverges | `cache.truncate(committed)` then `pending = [emitted[-1]]` |
| 11. EOS kept in `ids`, or a stop string not cut | the API returns text past the stop | breaks before appending EOS; cuts the decoded text at the first stop |
| 12. a round allowed $k + 1$ tokens when fewer remain | more than `max_tokens` ids | `k_eff` is capped at `max_tokens - len(out) - 1` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.6` | `speculative_step` is the sampled verification of one position |
| Back | `L8.2` | `KVCache.truncate` is the rollback; `cache_dims` sizes the cache; `Generation` is the result; `generate` is the reference speculation must reproduce |
| Back | `L2.1` | `NGramDraft` drafts from an `NGramLM`'s `logprobs` |
| Back | `L8.1` | `sample` (greedy and bonus draws), `sampling_distribution` (the $p_i$), `token_logprobs`, `request_rng` |
| Back | `M11.1` | total variation and why $\alpha = 1 - \mathrm{TV}(p, q)$ (reading) |
| Forward | `L10.8` | the Rust engine's prompt-lookup and n-gram speculation, held to `verify_draft` on shared logits and to greedy equality |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `verify_draft` | vLLM's rejection sampler | batched verification on the GPU, per-request draft lengths, the residual in log space | vLLM `v1/sample/rejection_sampler.py` |
| `PromptLookupDraft` | vLLM's n-gram proposer, HF `prompt_lookup_num_tokens` | a KMP-style search over the whole context, minimum and maximum n-gram sizes | vLLM `v1/spec_decode/ngram_proposer.py` |
| `ModelDraft` | EAGLE, Medusa | a draft head on the target's own hidden states; a tree of candidates verified with one tree-masked attention pass | Li et al., *EAGLE* (2024); Cai et al., *Medusa* (2024) |
| one draft path | SGLang and vLLM tree speculation | several candidate continuations per round, verified together | SGLang `srt/speculative/` |
