<!-- ol:module L6.4 -->
# T5 span corruption and relative position buckets

## Overview

| | |
|---|---|
| **Module** | `L6.4` · side (optional, D31) · Python · Pass 5 · 2 to 3 h |
| **You build** | `python/tinyllm/obj/t5.py`: `noise_span_counts`, `random_spans_noise_mask`, `span_corrupt`, `t5_relative_bucket`, `T5RelativeBias` |
| **Contract** | [`course/contracts/py/tinyllm/obj/t5.pyi`](../../../../course/contracts/py/tinyllm/obj/t5.pyi) |
| **Tests** | `course/tests/L6.4/test_t5.py` (what they check: section 4); the bucket oracle is Hugging Face's own T5 code |
| **Needs** | `L0.1` Tensor · `L0.2` `permute` · `L0.4` `Embedding` · reading: `L5.5` encoder-decoder Transformer, `M07.1` sampling, `M06.3` PCG32 (or `--ref-deps`) |
| **Used by** | no module: the side quest `sq.t5` trains an encoder-decoder with this objective |
| **Milestone** | none (optional module) |
| **Optional depth** | Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (2020), sections 2.1, 3.1.3, and 3.3; Shaw, Uszkoreit, and Vaswani, "Self-Attention with Relative Position Representations" (2018) |

## Key Takeaways

- Span corruption hides about 15% of the tokens in a few whole spans, one sentinel id per span; inputs plus targets hold every original token exactly once (`test_inputs_and_targets_restore_the_text`).
- The counts are fixed by the length, the density, and the mean span (round half to even), and the mask always starts with kept tokens and ends with a span (`test_counts_and_lengths`).
- The random choices are two Fisher-Yates shuffles with one `below` draw per swap, noise lengths first, so a seed replays the batch (`test_draws_follow_the_spec`).
- T5 has no position embeddings: each head adds a learned bias indexed by a **bucket** of the key's offset, exact when small and logarithmic up to `max_distance` (`test_buckets_match_hf`).
- A causal decoder folds every future offset into bucket 0 and uses all buckets for the past (`test_hand_example_buckets`, `test_relative_bias_matches_hf`).

## How to work this chapter

```bash
ol start L6.4              # stubs t5.py
ol tests L6.4              # the course tests
ol check L6.4              # exit code is the verdict
ol diff  L6.4              # after passing: your code against the reference
```

---

## 1. Why now

You have now met two pretraining objectives: GPT's next token (`L6.1`) and BERT's masked tokens (`L6.2`). BERT's targets are single tokens at known positions, which an encoder-decoder (`L5.5`) cannot use well: its decoder writes sequences. T5 recast pretraining as text to text: corrupt the input by dropping whole spans and ask the decoder to write them back. It also replaced absolute position embeddings with a small learned table indexed by relative distance, which generalizes past the training length. Neither piece has a call site in your system (D31), which is why this module is optional; the side quest `sq.t5` uses both to train a small T5 on your corpus.

## 2. Principles

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

### 2.1 Span corruption

T5's `random_spans_noise_mask` decides **how many** before **where**. With $N$ noise tokens in $K$ spans, it splits $N$ into $K$ positive span lengths and the $L - N$ kept tokens into $K$ positive gap lengths, then lays them out as keep, noise, keep, noise, and so on: the text always starts with kept tokens and ends with a noise span. Splitting $m$ items into $k$ positive parts uniformly is a shuffle: take $m - 1$ flags, the first $k - 1$ of them true, shuffle them, and start a new part after every true flag. The shuffle is Fisher-Yates from the end, $j = $ `rng.below(i + 1)` for $i = m - 2$ down to 1; the noise lengths are drawn first.

Rounding is numpy's (half to even): $L = 10$, $\rho = 0.25$ gives $N = \mathrm{round}(2.5) = 2$, not 3. T5 clamps $K$ only from below; on a short text $\mathrm{round}(N / \mu)$ can exceed $L - N$, and then some gap would be empty, so the course clamps $K \le \min(N, L - N)$.

Each noise span $k$ is replaced in the **inputs** by sentinel $S_k$, and the **targets** are each span preceded by its sentinel. T5's sentinels are the last ids of the vocabulary counting down (`<extra_id_0>` is $V - 1$), so `sentinel_start_id` is $V - 1$ and $S_k = V - 1 - k$. Real ids must stay out of the sentinel range, or the targets become ambiguous. Replacing every sentinel of the inputs by what follows it in the targets gives the text back: the objective loses nothing, and the targets are short (about $N + K$ tokens).

### 2.2 Relative position buckets

Self-attention without positions is permutation-equivariant. T5 adds to head $h$'s logit for query $i$ and key $j$ a learned scalar $b_h[\mathrm{bucket}(\delta)]$, with $\delta = j - i$. Small offsets matter most and get one bucket each; large offsets share logarithmically wider buckets:

$$\mathrm{bucket}(r) = \begin{cases} r & r < n/2 \\ \min\!\Big(n - 1,\ n/2 + \Big\lfloor \frac{\ln(r / (n/2))}{\ln(D / (n/2))}\,(n - n/2) \Big\rfloor\Big) & \text{otherwise} \end{cases}$$

for a distance $r \ge 0$. **Bidirectional** (the encoder): $n$ is half of `num_buckets`, $r = \lvert\delta\rvert$, and keys after the query ($\delta > 0$) add $n$, so the two directions use disjoint halves. **Causal** (the decoder): $n$ is all of `num_buckets`, $r = \max(0, -\delta)$: every future key falls into bucket 0, which the causal mask hides anyway. The logarithm is computed in float32 and truncated toward zero, exactly as the Hugging Face code that T5 checkpoints were trained with; the test compares every offset in $-300..300$. The bias table is an `Embedding(num_buckets, n_heads)` named `relative_attention_bias`, and `T5RelativeBias.forward(q_len, k_len, q_offset)` returns `[n_heads, q_len, k_len]`, with `q_offset` the tokens already in a decoder's cache.

## 3. Worked example by hand

**One span.** Tokens 10 to 17 ($L = 8$), $\rho = 0.25$, $\mu = 2$: $N = \mathrm{round}(2) = 2$, $K = \mathrm{round}(1) = 1$. One span leaves no choice: 6 kept tokens, then 2 noise tokens. With $S_0 = 99$: inputs $(10, 11, 12, 13, 14, 15, 99)$, targets $(99, 16, 17)$.

**Two spans, scripted draws.** Tokens 0 to 9, $\rho = 0.4$, $\mu = 2$: $N = 4$, $K = 2$.

| step | flags before | draw | flags after | lengths |
|---|---|---|---|---|
| noise: 3 flags, first 1 true | T F F | `below(3)` = 0, swap 2 and 0 | F F T | |
| | F F T | `below(2)` = 1, swap 1 and 1 | F F T | parts after each true flag: 3, 1 |
| keep: 5 flags, first 1 true | T F F F F | `below(5)` = 2, `below(4)` = 0, `below(3)` = 2, `below(2)` = 0 | F F F T F | 4, 2 |

Layout: keep 4, noise 3, keep 2, noise 1, so tokens 4, 5, 6 and 9 are noise. Inputs $(0, 1, 2, 3, S_0, 7, 8, S_1)$, targets $(S_0, 4, 5, 6, S_1, 9)$.

**Buckets.** Bidirectional, 32 buckets ($n = 16$, exact below 8), $D = 128$:

| $\delta$ | $r$ | bucket |
|---|---|---|
| $-3$ | 3 | 3 |
| $+3$ | 3 | $16 + 3 = 19$ |
| $-20$ | 20 | $8 + \lfloor \ln 2.5 / \ln 16 \times 8 \rfloor = 8 + \lfloor 2.64 \rfloor = 10$ |
| $+200$ | 200 | $16 + \min(15, 8 + \lfloor 9.29 \rfloor) = 31$ |
| 0 | 0 | 0 |

Causal with 32 buckets: $\delta = +5$ gives bucket 0, $\delta = -5$ gives 5.

These are `test_hand_example_one_span`, `test_hand_example_two_spans_scripted_draws`, and `test_hand_example_buckets`.

## 4. The interface

```python
def noise_span_counts(length: int, noise_density: float, mean_noise_span: float) -> tuple[int, int]: ...
def random_spans_noise_mask(length, noise_density, mean_noise_span, rng) -> NDArray: ...   # bool [L]
def span_corrupt(ids, noise_density, mean_noise_span, sentinel_start_id, rng, eos_id=None) -> tuple[NDArray, NDArray]: ...
def t5_relative_bucket(rel_pos, bidirectional: bool, num_buckets=32, max_distance=128) -> NDArray: ...
class T5RelativeBias(Module):
    def __init__(self, n_heads, num_buckets=32, max_distance=128, bidirectional=True, rng=None): ...
    def forward(self, q_len: int, k_len: int, q_offset: int = 0) -> Tensor: ...   # [H, q, k]
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_one_span` | unit | section 3, with and without `eos_id` | you and the test agree on the layout |
| `test_hand_example_two_spans_scripted_draws` | unit | the six scripted draws, in order, give section 3's inputs and targets | the shuffle and the sentinel order |
| `test_hand_example_buckets` | unit | section 3's bucket table, both directions | the bucket formula |
| `test_inputs_and_targets_restore_the_text` | property | 40 random texts: putting spans back gives the ids | the objective loses nothing |
| `test_counts_and_lengths` | property | $N$, $K$ (half to even), mask starts kept and ends noisy, sequence lengths | the batch shapes a trainer pads to |
| `test_short_texts_never_make_empty_spans` | boundary | the upper clamp of $K$ on 10 and 6 tokens | no empty span or gap |
| `test_draws_follow_the_spec` | unit | the exact `below` calls; same seed, same output | reproducible data |
| `test_sentinels_and_validation` | boundary | sentinels count down; colliding ids, no room, bad densities, $L < 2$ raise | unambiguous targets |
| `test_buckets_match_hf` | golden | five settings over $-300..300$ against HF's `_relative_position_bucket` | pretrained T5 tables index correctly |
| `test_bucket_properties` | property | monotone, in range, disjoint halves, exact while small | the coarsening is a coarsening |
| `test_relative_bias_matches_hf` | golden | HF `compute_bias`, encoder and decoder with past tokens | the bias each head adds |
| `test_relative_bias_gradient_lands_on_buckets` | property | the gradient is a scatter-add of the upstream gradient by bucket | training the table |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. relative position as query minus key, or a causal decoder that gives future keys their distance | past and future swapped: pretrained biases read the wrong side | `test_relative_bias_matches_hf` (mutant `s01`), `test_buckets_match_hf` (mutant `s09`) |
| 2. bidirectional buckets not halved | the two directions overlap; HF disagrees past offset 8 | `test_buckets_match_hf`, `test_bucket_properties` (mutant `s02`) |
| 3. rounding half up; no upper clamp on the span count | 3 noise tokens where T5 has 2; empty spans on short texts | `test_counts_and_lengths` (mutant `s03`), `test_short_texts_never_make_empty_spans` (mutant `s10`) |
| 4. sentinels counting up, or targets without them | ids collide with real tokens; the decoder cannot tell spans apart | `test_hand_example_two_spans_scripted_draws`, `test_sentinels_and_validation` (mutants `s04`, `s05`) |
| 5. a text that starts with a noise span | the first tokens always corrupted, unlike T5 | `test_hand_example_one_span`, `test_counts_and_lengths` (mutant `s06`) |
| 6. the shuffle's range or draw order changed | a seed no longer replays the batch | `test_draws_follow_the_spec` (mutants `s07`, `s08`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.1` | the bias is a Tensor whose gradient trains the table |
| Back | `L0.2` | `permute` turns the `[q, k, H]` lookup into `[H, q, k]` |
| Back | `L0.4` | the table is an `Embedding(num_buckets, n_heads)` |
| Back | `L5.5` | the encoder-decoder that span corruption trains (reading) |
| Forward | `sq.t5` | a small T5 on your corpus: span corruption for pretraining, the bias inside every attention layer |

No module calls this code (D31): `L6.4` is optional. It is still worth an evening: span corruption is the objective of T5, UL2, and many code models, and relative buckets are the simplest relative position scheme, a good contrast to RoPE (`L7.3`) and ALiBi (`L7.4`).

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `span_corrupt` | T5's `span_corruption` preprocessor; HF `DataCollatorForT5MLM` | packing to fixed input and target lengths, EOS handling, `expand_inputs_and_targets` to choose the raw length | `t5/data/preprocessors.py`; `transformers/examples/flax/language-modeling/run_t5_mlm_flax.py` |
| the denoising mixture | UL2's mixture of denoisers | short spans, long spans, and prefix LM in one model, chosen by a mode token | Tay et al., "UL2: Unifying Language Learning Paradigms" (2022) |
| `T5RelativeBias` | HF `T5Attention.compute_bias` | the bias computed once in layer 0 and reused by every layer | `transformers/models/t5/modeling_t5.py` |
| relative positions | RoPE, ALiBi | position by rotation, or a fixed linear penalty with no table | `L7.3`, `L7.4` |
