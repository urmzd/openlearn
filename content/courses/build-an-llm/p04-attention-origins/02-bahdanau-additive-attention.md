<!-- ol:module L4.2 -->
# Bahdanau additive attention

## Overview

| | |
|---|---|
| **Module** | `L4.2` · build · Python · Pass 4 · 3 to 4 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/seq2seq/additive.py`: `length_mask` and `AdditiveAttention` (`project_keys`, `scores`, `forward`); and your own oracle tests in `python/tests/l4-2-additive/` |
| **Contract** | [`course/contracts/py/tinyllm/seq2seq/additive.pyi`](../../../../course/contracts/py/tinyllm/seq2seq/additive.pyi) |
| **Tests** | `course/tests/L4.2/test_additive.py` (what they check: section 4), golden values from torch 2.14.1 in `course/fixtures/L4.2/additive_torch.npz` (`course/oracle/L4.2/additive_torch.py`); your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | `L0.4` `Linear`, `Module` · `L0.2` `F.tanh`, `F.masked_fill`, `F.softmax`, `F.matmul` · `L0.1` `Tensor` · `M06.3` `PCG32` (default initialization) · reading: `M09.2` (the masked softmax), `S-M08` (its VJP by hand) (or `--ref-deps`) |
| **Used by** | `L4.3` builds masks with `length_mask` · `L4.1` plugs `AdditiveAttention` into the decoder · later `L6.7` the zoo's attention rows |
| **Milestone** | `MS-L4` (`em_bahdanau >= 0.97` on the dates task) |
| **Optional depth** | Bahdanau, Cho, and Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate" (ICLR 2015), section 3.1 and appendix A.1.2; Graves, "Generating Sequences With Recurrent Neural Networks" (2013), section 5 |

## Key Takeaways

- Attention is a weighted average of the encoder outputs whose weights the decoder computes: a score per source position, a softmax, a sum (`test_hand_example_weights_and_context`).
- Padding is masked on the **scores**, to $-\infty$, before the softmax: padded weights are exactly 0, the rest sum to 1, and padding content never matters (`test_masked_weights_are_exactly_zero_and_rows_sum_to_one`, `test_padding_content_never_matters`).
- An empty source reads nothing: weights and context 0, not NaN (`test_fully_masked_row_reads_nothing`).
- The key projection $U k_s + b$ does not depend on the decoder step, so it is computed once per sentence (`test_precomputed_keys_equal_recomputed`).
- Attention by itself ignores order: permute the source and the weights permute with it (`test_permuting_the_source_permutes_the_weights`).

## How to work this chapter

```bash
ol start L4.2              # stubs additive.py; prints your test path and rung (R5)
ol tests L4.2              # the course tests
# write your oracle tests in python/tests/l4-2-additive/, then:
ol check L4.2              # course tests and the mutation grade of your tests
ol diff  L4.2              # after passing: your code against the reference
```

---

## 1. Why now

Your encoder-decoder (`L4.1`, next in this pass) squeezes the whole source sentence into one vector, the decoder's first state. On short inputs that works; on a 30-character date string the decoder has forgotten the start of the input by the time it writes the end, and exact match drops sharply with length. Bahdanau's fix, the first attention mechanism in neural translation, lets every decoder step look back at **all** encoder outputs and choose which to read. It is the idea the 2017 transformer (`L5.1`) keeps while dropping the recurrence, so this is where the core operation of every model in the rest of the course is born.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $B$, $S$ | batch size, source length (padded) | `int` |
| $q$ | the query: the decoder state that asks | `float32[B, Dq]` |
| $k_s$ | the key at source position $s$: an encoder output, also the value read | `float32[B, S, Dk]` |
| $\ell_b$ | the real length of source $b$ | `int` |
| $m_{bs}$ | mask: true when $s < \ell_b$ | `bool[B, S]` |
| $W$, $U$, $b$, $v$ | parameters `query.weight` $[A, Dq]$, `key.weight` $[A, Dk]$, `key.bias` $[A]$, `v.weight` $[1, A]$ | `float32` |
| $A$ | `d_attn`, the width of the scoring network | `int` |
| $e_s$ | the score of position $s$ | `float` |
| $a_s$ | the attention weight of position $s$ | `float` in $[0, 1]$ |
| $c$ | the context: $\sum_s a_s k_s$ | `float32[B, Dk]` |

### 2.1 Reading by weighted average

A decoder that must write the next target word needs some of the source, not all of it equally. A **weighted average** of the encoder outputs, with weights that sum to 1, reads exactly that: a weight near 1 on one position copies that position; spread weights blend several. The weights must depend on what the decoder is doing now, so they are computed from the decoder state $q$.

### 2.2 Scores, weights, context

Bahdanau scores each position with a one-hidden-layer network on the pair $(q, k_s)$:

$$e_s = v^\top \tanh(W q + U k_s + b), \qquad a = \operatorname{softmax}(e), \qquad c = \sum_s a_s k_s .$$

The softmax (`M09.2`) turns arbitrary real scores into positive weights that sum to 1. Every operation is an op-library call (`L0.2`), so backward reaches $q$ (the decoder learns where to look), the keys (the encoder learns what to offer), and $W, U, b, v$. The context has the keys' width $D_k$, not $A$: the scoring network decides **how much** of each key to take, the keys themselves are what is read. Nothing in the formula looks at the position $s$ itself: permute the keys and the weights permute with them. Order has to come from the encoder.

### 2.3 Masking padding

A batch pads every source to the longest length $S$. Padded positions must never be read. The rule is to set their **scores** to $-\infty$ before the softmax: $e^{-\infty} = 0$, so their weights are exactly 0 and the real weights still sum to 1. `length_mask(lengths, S)` builds $m_{bs} = (s < \ell_b)$. Masking after the softmax (multiplying the weights by the mask) leaves the real weights summing to less than 1 and lets padding influence the normalization. A row with no real position at all (an empty source) is all $-\infty$; `M09.2` defines its softmax as zeros, so the context is 0 and no gradient flows back, where a naive softmax would give $0/0 = $ NaN.

### 2.4 Precomputing the keys

$U k_s + b$ involves only the keys, which are fixed for the whole sentence, while $W q$ changes at every decoder step. Computing $U k_s + b$ once (`project_keys`) and passing it to every step as `proj` saves $S \cdot A \cdot D_k$ multiply-adds per step; Bahdanau's appendix points this out. The result must be identical to recomputing.

## 3. Worked example by hand

Every size 1 ($D_q = D_k = A = 1$), $W = U = v = 1$, $b = 0$. Query $q = 0$, keys $k = (0, 1, 2)$, and the third position is padding (mask `[True, True, False]`).

1. Scores before the mask: $\tanh(0 + 0) = 0$, $\tanh(1) = 0.761594$, $\tanh(2) = 0.964028$. The padded one would be the largest.
2. After the mask: $(0, 0.761594, -\infty)$.
3. Softmax: $e^0 = 1$, $e^{0.761594} = 2.141688$, $e^{-\infty} = 0$; the sum is $3.141688$, so $a = (0.318300, 0.681700, 0)$.
4. Context: $c = 0.318300 \cdot 0 + 0.681700 \cdot 1 + 0 \cdot 2 = 0.681700$.

The padded key, the biggest value in the row, contributes nothing. This is `test_hand_example_weights_and_context`.

## 4. The interface

```python
def length_mask(lengths: ArrayLike, S: int) -> NDArray: ...          # bool [B, S]
class AdditiveAttention(Module):
    query_from = "previous"                                           # read by L4.1's decoder
    def __init__(self, d_query: int, d_key: int, d_attn: int, rng=None) -> None: ...
    def project_keys(self, keys: Tensor) -> Tensor: ...               # [B, S, A]
    def scores(self, query: Tensor, keys: Tensor, proj=None) -> Tensor: ...   # [B, S], unmasked
    def forward(self, query, keys, mask, proj=None) -> tuple[Tensor, Tensor]: ...  # context, weights
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_weights_and_context` | unit | section 3 to float32 tolerance | you and the test agree on the formula |
| `test_length_mask` | unit | `s < lengths[b]`, bad lengths raise | the only place lengths enter attention |
| `test_golden_torch` | golden | context, weights, and gradients of the query, keys, and all four parameters against torch, lengths 5, 3, 1 | your encoder and decoder train through this |
| `test_gradcheck_every_input_and_parameter` | gradcheck | float64 central differences with a padded row | both sides of the model learn |
| `test_masked_weights_are_exactly_zero_and_rows_sum_to_one` | property | exact zeros, positive real weights, rows sum to 1 | padding is never read |
| `test_padding_content_never_matters` | property | garbage in padded keys changes nothing bitwise; their gradient is 0 | a batch padded differently translates the same |
| `test_fully_masked_row_reads_nothing` | boundary | zeros, no NaN, no gradient | empty inputs in a batch |
| `test_precomputed_keys_equal_recomputed` | property | `proj` passed or recomputed, bitwise equal | `L4.1` passes it at every step |
| `test_permuting_the_source_permutes_the_weights` | property | weights permute, context unchanged | order comes from the encoder, not from attention |
| `test_parameter_names_and_shapes` | unit | keys, shapes, order, `query_from`, seeded init | the safetensors keys of seq2seq checkpoints |
| `test_validation` | boundary | wrong widths, missing source axis, wrong or float mask, wrong `proj` | wiring bugs fail loudly |

### Your graded tests (rung R5)

The oracle for your tests is the formula itself, written out in numpy with the module's own weights read from `state_dict()`: compute $e$, mask, softmax, and the context in float64, and compare with `forward`. Add the section 3 numbers, a finite-difference check of the gradients of the query and the keys (your own, in float64), the masking properties, and the parameter names. Import only `tinyllm.seq2seq.additive`, `tinyllm.autograd.tensor`, and `tinyllm.autograd.functional` (as `import tinyllm.autograd.functional as F`). `ol check L4.2` requires 0.80 with every pitfall fault killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. masking after the softmax | real weights no longer sum to 1; padding still shapes the normalization | `test_masked_weights_are_exactly_zero_and_rows_sum_to_one`, `test_hand_example_weights_and_context` (mutant `s01`) |
| 2. masking with $-10^9$ instead of $-\infty$ | an empty source gets uniform weights and reads padding | `test_fully_masked_row_reads_nothing` (mutant `s02`) |
| 3. softmax over the wrong axis | weights sum to 1 over the batch, not the source | `test_masked_weights_are_exactly_zero_and_rows_sum_to_one`, `test_golden_torch` (mutant `s03`) |
| 4. reading detached keys | the encoder gets no gradient through attention | `test_gradcheck_every_input_and_parameter` (mutant `s04`) |
| mask polarity inverted | only padding is read | `test_padding_content_never_matters` (mutant `s05`) |
| no `tanh` | a linear score: the golden values disagree | `test_golden_torch` (mutant `s06`) |
| `<=` in `length_mask` | one padding position is read per row | `test_length_mask` (mutant `s07`) |
| caching $\tanh(U k + b)$ instead of $U k + b$ | the query is added after the nonlinearity | `test_hand_example_weights_and_context` (mutant `s08`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.4` | `Linear` holds $W$, $U$, $b$, $v$ |
| Back | `L0.2` | `masked_fill`, `softmax`, `tanh`, `matmul` give the backward for free |
| Back | `L0.1` | `Tensor`, the type of queries, keys, and weights |
| Back | `M06.3` | `PCG32` initializes the layers when no rng is given |
| Forward | `L4.3` | Luong attention keeps the same masked read and builds its masks with `length_mask` |
| Forward | `L4.1` | the decoder calls `attention(s_{t-1}, keys, mask, proj)` before each step |
| Forward | `L5.1` | scaled dot-product attention is this read with a cheaper score |
| Forward | `L6.7` | the zoo's attention variants of the seq2seq rows |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `AdditiveAttention` | `torch.nn.MultiheadAttention`, `F.scaled_dot_product_attention` | dot-product scores (no hidden layer), many heads, fused kernels | `torch/nn/functional.py` (`multi_head_attention_forward`) |
| masking to $-\infty$ | `attn_mask` and `key_padding_mask` in torch | boolean and additive masks, broadcast over heads | PyTorch SDPA documentation |
| `project_keys` | the KV cache | keys and values projected once and reused at every step of generation | `L8.2` in this course; vLLM's paged KV cache |
| additive scores | location-sensitive attention (Tacotron 2) | adds the previous weights as a feature so speech alignment moves forward | Chorowski et al. 2015 |
