# T5: span corruption and relative position buckets

Two pretraining objectives are now in place: [GPT's next token](01-gpt-decoder-only.md) and [BERT's masked tokens](02-bert-masked-lm.md). BERT's targets are single tokens at known positions, which an [encoder-decoder Transformer](../06-transformer-2017/05-encoder-decoder-transformer.md) cannot use well, because its decoder writes sequences. T5 recast pretraining as text to text: corrupt the input by dropping whole spans and ask the decoder to write them back. It also replaced absolute position embeddings with a small learned table indexed by relative distance, which generalizes past the training length.

Neither piece has a call site in the system this book builds, which is why this section is optional. It is still worth an evening. Span corruption is the objective of T5, UL2, and many code models, and relative buckets are the simplest relative position scheme, a good contrast to [RoPE](../08-modern-block/03-rope.md) and [ALiBi](../08-modern-block/04-context-extension.md) in Part 7. If you want to put the pieces to work, a natural side project is to train a small T5 on the book's corpus with span corruption for pretraining and the bias inside every attention layer.

The code uses only the [`Tensor`](../01-foundations/02-tensor-and-broadcasting-backward.md), the [`permute` op](../01-foundations/03-op-library-and-gradcheck.md), and [`Embedding`](../01-foundations/05-module-system-and-layers.md); it helps to have read [categorical sampling](../../mathematics-for-computing/08-probability-statistics/02-categorical-sampling.md) and [PCG32](../../mathematics-for-computing/07-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) in *Mathematics for Computing*. The reference implementation is [`python/tinyllm/obj/t5.py`](../../../../course/solutions/python/tinyllm/obj/t5.py) (`noise_span_counts`, `random_spans_noise_mask`, `span_corrupt`, `t5_relative_bucket`, `T5RelativeBias`), with the contract in [`course/contracts/py/tinyllm/obj/t5.pyi`](../../../../course/contracts/py/tinyllm/obj/t5.pyi).

> **Learning objectives**
> - Compute the number of noise tokens and spans for a text, with half-to-even rounding and both clamps.
> - Lay out a span-corruption mask from two random compositions and replay the exact `below` draws.
> - Build inputs and targets with sentinels counting down from the top of the vocabulary, and show that they restore the text.
> - Compute T5 relative position buckets by hand for bidirectional and causal attention.
> - Explain how the bias table is indexed and trained, and what `q_offset` does in cached decoding.

## Span corruption

Here is the notation for this section.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $L$ | the number of tokens of one text | `int` $\ge 2$ |
| $\rho$ | `noise_density`, the fraction to corrupt | `float` in $(0, 1)$ |
| $\mu$ | `mean_noise_span`, the average span length | `float` $\ge 1$ |
| $N$ | noise tokens, $\mathrm{clamp}(\mathrm{round}(L \rho), 1, L - 1)$ | `int` |
| $K$ | noise spans, $\mathrm{clamp}(\mathrm{round}(N / \mu), 1, \min(N, L - N))$ | `int` |
| $S_k$ | sentinel $k$, the id `sentinel_start_id` $- k$ | `int` |
| $\delta = j - i$ | relative position of key $j$ from query $i$ | `int` |
| $n$ | buckets per direction (`num_buckets`, halved when bidirectional) | `int` |
| $D$ | `max_distance` | `int` |

T5's `random_spans_noise_mask` decides **how many** before **where**. With $N$ noise tokens in $K$ spans, it splits $N$ into $K$ positive span lengths and the $L - N$ kept tokens into $K$ positive gap lengths, then lays them out as keep, noise, keep, noise, and so on. The text therefore always starts with kept tokens and ends with a noise span.

Splitting $m$ items into $k$ positive parts uniformly is a shuffle. Take $m - 1$ flags, the first $k - 1$ of them true, shuffle them, and start a new part after every true flag. The shuffle is Fisher-Yates from the end, $j = $ `rng.below(i + 1)` for $i = m - 2$ down to 1, and the noise lengths are drawn first.

Rounding is numpy's (half to even): $L = 10$, $\rho = 0.25$ gives $N = \mathrm{round}(2.5) = 2$, not 3. T5 clamps $K$ only from below. On a short text $\mathrm{round}(N / \mu)$ can exceed $L - N$, and then some gap would be empty, so this book's implementation also clamps $K \le \min(N, L - N)$.

Once the mask is fixed, each noise span $k$ is replaced in the **inputs** by sentinel $S_k$, and the **targets** are each span preceded by its sentinel. T5's sentinels are the last ids of the vocabulary counting down (`<extra_id_0>` is $V - 1$), so `sentinel_start_id` is $V - 1$ and $S_k = V - 1 - k$. Real ids must stay out of the sentinel range, or the targets become ambiguous. Replacing every sentinel of the inputs by what follows it in the targets gives the text back. The objective loses nothing, and the targets are short, about $N + K$ tokens.

## Relative position buckets

Span corruption is T5's objective; its other idea concerns positions. Self-attention without positions is permutation-equivariant. T5 adds to head $h$'s logit for query $i$ and key $j$ a learned scalar $b_h[\mathrm{bucket}(\delta)]$, with $\delta = j - i$. Small offsets matter most and get one bucket each; large offsets share logarithmically wider buckets:

$$\mathrm{bucket}(r) = \begin{cases} r & r < n/2 \\ \min\!\Big(n - 1,\ n/2 + \Big\lfloor \frac{\ln(r / (n/2))}{\ln(D / (n/2))}\,(n - n/2) \Big\rfloor\Big) & \text{otherwise} \end{cases}$$

for a distance $r \ge 0$. The two attention directions use this differently.

- **Bidirectional** (the encoder): $n$ is half of `num_buckets`, $r = \lvert\delta\rvert$, and keys after the query ($\delta > 0$) add $n$, so the two directions use disjoint halves.
- **Causal** (the decoder): $n$ is all of `num_buckets` and $r = \max(0, -\delta)$. Every future key falls into bucket 0, which the causal mask hides anyway.

The logarithm is computed in float32 and truncated toward zero, exactly as in the Hugging Face code that T5 checkpoints were trained with; a correct solution agrees with it at every offset in $-300..300$. The bias table is an `Embedding(num_buckets, n_heads)` named `relative_attention_bias`, and `T5RelativeBias.forward(q_len, k_len, q_offset)` returns `[n_heads, q_len, k_len]`, with `q_offset` the number of tokens already in a decoder's cache.

## Working it by hand

Start with the simplest case, **one span**. Take tokens 10 to 17 ($L = 8$), $\rho = 0.25$, $\mu = 2$. Then $N = \mathrm{round}(2) = 2$ and $K = \mathrm{round}(1) = 1$. One span leaves no choice: 6 kept tokens, then 2 noise tokens. With $S_0 = 99$, the inputs are $(10, 11, 12, 13, 14, 15, 99)$ and the targets $(99, 16, 17)$.

Now **two spans, with scripted draws**. Take tokens 0 to 9, $\rho = 0.4$, $\mu = 2$, so $N = 4$ and $K = 2$. The table traces both shuffles.

| step | flags before | draw | flags after | lengths |
|---|---|---|---|---|
| noise: 3 flags, first 1 true | T F F | `below(3)` = 0, swap 2 and 0 | F F T | |
| | F F T | `below(2)` = 1, swap 1 and 1 | F F T | parts after each true flag: 3, 1 |
| keep: 5 flags, first 1 true | T F F F F | `below(5)` = 2, `below(4)` = 0, `below(3)` = 2, `below(2)` = 0 | F F F T F | 4, 2 |

The layout is keep 4, noise 3, keep 2, noise 1, so tokens 4, 5, 6, and 9 are noise. The inputs are $(0, 1, 2, 3, S_0, 7, 8, S_1)$ and the targets $(S_0, 4, 5, 6, S_1, 9)$.

Finally, **buckets**. With bidirectional attention, 32 buckets ($n = 16$, exact below 8), and $D = 128$:

| $\delta$ | $r$ | bucket |
|---|---|---|
| $-3$ | 3 | 3 |
| $+3$ | 3 | $16 + 3 = 19$ |
| $-20$ | 20 | $8 + \lfloor \ln 2.5 / \ln 16 \times 8 \rfloor = 8 + \lfloor 2.64 \rfloor = 10$ |
| $+200$ | 200 | $16 + \min(15, 8 + \lfloor 9.29 \rfloor) = 31$ |
| 0 | 0 | 0 |

With causal attention and 32 buckets, $\delta = +5$ gives bucket 0 and $\delta = -5$ gives 5.

## Designing the interface

```python
def noise_span_counts(length: int, noise_density: float, mean_noise_span: float) -> tuple[int, int]: ...
def random_spans_noise_mask(length, noise_density, mean_noise_span, rng) -> NDArray: ...   # bool [L]
def span_corrupt(ids, noise_density, mean_noise_span, sentinel_start_id, rng, eos_id=None) -> tuple[NDArray, NDArray]: ...
def t5_relative_bucket(rel_pos, bidirectional: bool, num_buckets=32, max_distance=128) -> NDArray: ...
class T5RelativeBias(Module):
    def __init__(self, n_heads, num_buckets=32, max_distance=128, bidirectional=True, rng=None): ...
    def forward(self, q_len: int, k_len: int, q_offset: int = 0) -> Tensor: ...   # [H, q, k]
```

The corruption side splits into counts, mask, and corruption, so each layer of the "how many, then where, then what" decision can be tested on its own. The position side is a pure bucket function plus a module that owns the learned table.

## Building it step by step

The reasoning runs as follows.

1. An encoder-decoder's decoder writes sequences, so the corruption should produce a sequence to write. Dropping whole spans and asking for them back does that, and the targets are short: about $N + K$ tokens instead of $L$.
2. The decoder must know which hole each span came from, so each span gets its own marker: sentinel $k$ in the inputs where span $k$ was, and before span $k$ in the targets. Putting the targets back into the inputs at the sentinels recovers the text exactly, so nothing is lost.
3. The sentinels must not collide with real ids. T5 takes them from the top of the vocabulary counting down ($S_k = V - 1 - k$), so the solution refuses any id inside the sentinel range it is about to use.
4. "How many" is decided before "where": $N = \mathrm{round}(L\rho)$ and $K = \mathrm{round}(N/\mu)$, clamped so every span and every gap has at least one token. In the first worked example $N = 2$, $K = 1$.
5. "Where" is two random compositions: $N$ into $K$ positive parts and $L - N$ into $K$ positive parts, interleaved keep, noise, keep, noise. A uniform composition of $m$ into $k$ positive parts is a choice of $k - 1$ of the $m - 1$ cut points, which a shuffle of $m - 1$ flags makes. Fixing the shuffle (Fisher-Yates from the end, one `below` per swap, noise first) makes a seed replay the batch.
6. For positions, attention logits get a learned scalar per head that depends only on $\delta = j - i$. Exact buckets for small $|\delta|$ and logarithmic buckets for large ones spend the table where offsets are distinguishable and stop at `max_distance`, beyond which all offsets share the last bucket, so the scheme extends to any length.
7. To load T5 checkpoints, the bucket function must reproduce Hugging Face's float32 arithmetic and truncation, bucket for bucket.

### Counts

```python
def noise_span_counts(length, noise_density, mean_noise_span):
    # ... length >= 2, 0 < density < 1, mean span >= 1 ...
    n_noise = int(np.round(length * noise_density))   # numpy rounds half to even: round(2.5) = 2
    n_noise = min(max(n_noise, 1), length - 1)        # at least one noise and one kept token
    n_spans = max(int(np.round(n_noise / mean_noise_span)), 1)
    # Every span needs at least one token, and so does every gap before one.
    n_spans = min(n_spans, n_noise, length - n_noise)
    return n_noise, n_spans
```

### Random composition

```python
def _segment_lengths(n_items, n_segments, rng):
    """Split n_items into n_segments positive lengths, uniformly at random."""
    # n_items - 1 gaps, n_segments - 1 of them start a new segment; the gaps
    # are shuffled with Fisher-Yates from the end, one rng.below per swap.
    starts = [i < n_segments - 1 for i in range(n_items - 1)]
    for i in range(len(starts) - 1, 0, -1):
        j = rng.below(i + 1)
        starts[i], starts[j] = starts[j], starts[i]
    first = np.array([False] + starts, dtype=bool)    # item 0 always opens segment 0
    seg = np.cumsum(first)                            # segment index of every item
    return np.bincount(seg, minlength=n_segments).astype(np.int64)
```

Trace the two-span example. The noise flags start as `T F F`; `below(3) = 0` swaps positions 2 and 0, and `below(2) = 1` swaps 1 with itself, giving `F F T`. Prepending `False` and taking the cumulative sum gives segment ids $(0, 0, 0, 1)$, so the lengths are $(3, 1)$.

### The mask and the corruption

```python
def random_spans_noise_mask(length, noise_density, mean_noise_span, rng):
    n_noise, n_spans = noise_span_counts(length, noise_density, mean_noise_span)
    noise = _segment_lengths(n_noise, n_spans, rng)          # noise lengths are drawn first
    keep = _segment_lengths(length - n_noise, n_spans, rng)
    # keep_1, noise_1, keep_2, noise_2, ...: the text starts with kept tokens
    # and ends with a noise span.
    mask = np.zeros(length, dtype=bool)
    pos = 0
    for k, x in zip(keep, noise):
        pos += int(k)
        mask[pos : pos + int(x)] = True
        pos += int(x)
    return mask
```

```python
def span_corrupt(ids, noise_density, mean_noise_span, sentinel_start_id, rng, eos_id=None):
    # ... 1-D integer check ...
    mask = random_spans_noise_mask(a.size, noise_density, mean_noise_span, rng)
    n_spans = int(np.sum(mask[1:] & ~mask[:-1]) + mask[0])   # count rising edges
    lowest = sentinel_start_id - n_spans + 1
    if lowest < 0:
        raise ValueError(...)        # not enough ids below sentinel_start_id
    if np.any((a >= lowest) & (a <= sentinel_start_id)):
        raise ValueError(...)        # a real id inside the sentinel range: ambiguous targets
    inputs, targets = [], []
    k = -1
    for t in range(a.size):
        if mask[t]:
            if t == 0 or not mask[t - 1]:  # a new noise span: sentinel k
                k += 1
                inputs.append(sentinel_start_id - k)   # counting DOWN from the top
                targets.append(sentinel_start_id - k)
            targets.append(int(a[t]))
        else:
            inputs.append(int(a[t]))
    if eos_id is not None:
        inputs.append(int(eos_id))
        targets.append(int(eos_id))
    return np.array(inputs, dtype=np.int64), np.array(targets, dtype=np.int64)
```

### Buckets

```python
def t5_relative_bucket(rel_pos, bidirectional, num_buckets=32, max_distance=128):
    rel = np.asarray(rel_pos, dtype=np.int64)          # rel = key - query
    n = int(num_buckets)
    out = np.zeros(rel.shape, dtype=np.int64)
    if bidirectional:
        n //= 2
        out += (rel > 0).astype(np.int64) * n  # keys after the query: upper half
        rel = np.abs(rel)
    else:
        rel = -np.minimum(rel, 0)  # a causal decoder only sees keys at or before the query
    max_exact = n // 2
    # ... need max_exact >= 1 and max_distance > max_exact ...
    is_small = rel < max_exact
    # T5's float32 arithmetic (as torch computes it), truncated toward zero.
    # Small offsets take the exact branch, so they are lifted to max_exact
    # here only to keep log away from 0.
    logr = np.log(np.maximum(rel, max_exact).astype(np.float32) / np.float32(max_exact))
    large = max_exact + (
        logr / np.float32(math.log(max_distance / max_exact)) * np.float32(n - max_exact)
    ).astype(np.int64)                                 # astype truncates toward zero
    large = np.minimum(large, n - 1)                   # everything past max_distance: last bucket
    return out + np.where(is_small, rel, large)
```

For $\delta = -20$ (bidirectional, 32 buckets), $n = 16$ and `max_exact` $= 8$; $\ln(20/8)/\ln(128/8) \times 8 = 2.64$, truncated to 2, so the bucket is 10, as in the table.

### The bias module

```python
    def forward(self, q_len, k_len, q_offset=0):
        # ... range checks ...
        # A query at row r of this call is at absolute position q_offset + r (cached decoding).
        q = np.arange(q_len, dtype=np.int64)[:, None] + q_offset
        k = np.arange(k_len, dtype=np.int64)[None, :]
        buckets = t5_relative_bucket(k - q, self.bidirectional, self.num_buckets, self.max_distance)
        bias = self.relative_attention_bias(buckets)  # [q, k, heads], an embedding gather
        return F.permute(bias, (2, 0, 1))             # [heads, q, k], added to the attention logits
```

Because the lookup is an `Embedding`, backward scatters the upstream gradient into the bucket rows, summing over every $(i, j)$ pair that shares a bucket. The full solution is in [`t5.py`](../../../../course/solutions/python/tinyllm/obj/t5.py).

### Why this design

Corruption costs $O(L)$ with fewer than $L$ `below` draws (one per swap of each shuffle). The bias is $O(q \cdot k)$ bucket lookups and an $H \cdot q \cdot k$ tensor; Hugging Face computes it once in layer 0 and reuses it in every layer.

**Count-then-place** beats flipping a coin per token: the number of noise tokens and spans is fixed by the length, so target lengths are predictable for padding and the mean span is exactly controlled. **Composition by shuffled flags** beats drawing $K - 1$ cut points with rejection: it is uniform, uses a fixed number of draws, and is easy to replay in another language. **Float32 bucket arithmetic** is chosen over exact rationals because the buckets must agree with the code pretrained T5 tables were trained with, including where truncation lands. The tradeoff accepted is that all offsets past `max_distance` share one bucket, so very long-range order is coarse; RoPE and ALiBi make different tradeoffs.

## Testing it

A correct solution has to pass these checks.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the one-span example | unit | the worked example, with and without `eos_id` | everyone agrees on the layout |
| the two-span example with scripted draws | unit | the six scripted draws, in order, give the worked inputs and targets | the shuffle and the sentinel order |
| the bucket table | unit | the worked bucket table, both directions | the bucket formula |
| inputs and targets restore the text | property | 40 random texts: putting spans back gives the ids | the objective loses nothing |
| counts and lengths | property | $N$, $K$ (half to even), mask starts kept and ends noisy, sequence lengths | the batch shapes a trainer pads to |
| short texts never make empty spans | boundary | the upper clamp of $K$ on 10 and 6 tokens | no empty span or gap |
| draws follow the spec | unit | the exact `below` calls; same seed, same output | reproducible data |
| sentinels and validation | boundary | sentinels count down; colliding ids, no room, bad densities, $L < 2$ raise | unambiguous targets |
| buckets match Hugging Face | golden | five settings over $-300..300$ against HF's `_relative_position_bucket` | pretrained T5 tables index correctly |
| bucket properties | property | monotone, in range, disjoint halves, exact while small | the coarsening is a coarsening |
| the bias matches Hugging Face | golden | HF `compute_bias`, encoder and decoder with past tokens | the bias each head adds |
| the bias gradient lands on buckets | property | the gradient is a scatter-add of the upstream gradient by bucket | training the table |

## What can go wrong

The common mistakes are sign conventions, rounding, and anything that changes the order of random draws.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| relative position as query minus key, or a causal decoder that gives future keys their distance | past and future swapped: pretrained biases read the wrong side | `rel = k - q`; causal uses `-min(rel, 0)` |
| bidirectional buckets not halved | the two directions overlap; HF disagrees past offset 8 | `n //= 2` and keys after the query add `n` |
| rounding half up; no upper clamp on the span count | 3 noise tokens where T5 has 2; empty spans on short texts | `np.round` (half to even); `n_spans = min(n_spans, n_noise, length - n_noise)` |
| sentinels counting up, or targets without them | ids collide with real tokens; the decoder cannot tell spans apart | `sentinel_start_id - k` written to both inputs and targets |
| a text that starts with a noise span | the first tokens always corrupted, unlike T5 | the layout loop adds a keep length before every noise length |
| the shuffle's range or draw order changed | a seed no longer replays the batch | Fisher-Yates from the end with `below(i + 1)`, noise lengths drawn before keep lengths |

> **Warning:** Python's built-in `round` and numpy's `np.round` both round half to even, but many other languages round half away from zero. If you port `noise_span_counts`, port the rounding rule too, or $L = 10$, $\rho = 0.25$ will give 3 noise tokens instead of T5's 2.

## Where this leads

Nothing later in the book calls this code. The ideas, though, recur: [RoPE](../08-modern-block/03-rope.md) encodes position by rotation, and [ALiBi](../08-modern-block/04-context-extension.md) uses a fixed linear penalty with no table, two different answers to the question T5's buckets answer with a learned table.

In production, T5's `span_corruption` preprocessor (`t5/data/preprocessors.py`) and Hugging Face's `DataCollatorForT5MLM` (`transformers/examples/flax/language-modeling/run_t5_mlm_flax.py`) add packing to fixed input and target lengths, EOS handling, and `expand_inputs_and_targets` to choose the raw length. UL2's mixture of denoisers trains short spans, long spans, and prefix LM in one model, chosen by a mode token. Hugging Face's `T5Attention.compute_bias` (`transformers/models/t5/modeling_t5.py`) computes the bias once in layer 0 and reuses it in every layer.

Further reading:

- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (2020), sections 2.1, 3.1.3, and 3.3.
- Shaw, Uszkoreit, and Vaswani, "Self-Attention with Relative Position Representations" (2018).
- Tay et al., "UL2: Unifying Language Learning Paradigms" (2022).

## Key terms

- **Span corruption**: replacing whole spans of the input with sentinels and training the decoder to write the spans back.
- **Sentinel**: a reserved id marking one dropped span, counting down from the top of the vocabulary.
- **Random composition**: a uniform split of $m$ items into $k$ positive parts, made by shuffling cut-point flags.
- **Relative position bucket**: an index into a learned bias table, exact for small offsets and logarithmic for large ones.
- **`q_offset`**: the absolute position of the first query in a call, the number of tokens already cached.

## Summary

- Span corruption hides about 15% of the tokens in a few whole spans, one sentinel id per span; inputs plus targets hold every original token exactly once.
- The counts are fixed by the length, the density, and the mean span (round half to even), and the mask always starts with kept tokens and ends with a span.
- The random choices are two Fisher-Yates shuffles with one `below` draw per swap, noise lengths first, so a seed replays the batch.
- T5 has no position embeddings: each head adds a learned bias indexed by a **bucket** of the key's offset, exact when small and logarithmic up to `max_distance`.
- A causal decoder folds every future offset into bucket 0 and uses all buckets for the past.

## Review questions

1. Why does the mask always start with kept tokens and end with a noise span?
   <details><summary>Answer</summary>The layout interleaves $K$ keep lengths and $K$ noise lengths starting with a keep, and every length is positive. That is T5's convention; a mask starting with noise would always corrupt the first tokens.</details>
2. Why must $K$ be clamped to $\min(N, L - N)$ and not just to at least 1?
   <details><summary>Answer</summary>Each of the $K$ spans needs at least one noise token and each of the $K$ gaps at least one kept token. On a short text, $\mathrm{round}(N/\mu)$ can exceed $L - N$, and `_segment_lengths` would then need more positive parts than items, producing an empty gap.</details>
3. Why do sentinels count down from `sentinel_start_id` and why does `span_corrupt` refuse ids in that range?
   <details><summary>Answer</summary>T5's sentinels are the last vocabulary ids, `<extra_id_0>` $= V - 1$ and so on downward. If a real token had a sentinel's id, the targets could not be split back into spans unambiguously.</details>
4. In a causal decoder, why can every future offset map to bucket 0?
   <details><summary>Answer</summary>The causal mask removes those keys from the softmax, so their bias is never used. Folding them into one bucket frees the whole table for past offsets.</details>
5. What does `q_offset` fix during cached decoding?
   <details><summary>Answer</summary>With a cache, the new query is at absolute position `q_offset` while keys run from 0. Without the offset, the query would be treated as position 0 and every relative distance would be wrong.</details>
