<!-- ol:module L4.3 -->
# Luong attention and input feeding

## Overview

| | |
|---|---|
| **Module** | `L4.3` · build · Python · Pass 4 · 2 to 3 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/seq2seq/luong.py`: `LuongAttention` with the `dot`, `general`, and `concat` scores (`project_keys`, `scores`, `forward`) and `attentional`, the state input feeding passes on; and your own oracle tests in `python/tests/l4-3-luong/` |
| **Contract** | [`course/contracts/py/tinyllm/seq2seq/luong.pyi`](../../../../course/contracts/py/tinyllm/seq2seq/luong.pyi) |
| **Tests** | `course/tests/L4.3/test_luong.py` (what they check: section 4), golden values from torch 2.14.1 in `course/fixtures/L4.3/luong_torch.npz` (`course/oracle/L4.3/luong_torch.py`); your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | `L4.2` `length_mask` (and the masked read you wrote there) · `L0.4` `Linear`, `Module` · `L0.2` ops · `L0.1` `Tensor` · `M06.3` `PCG32` · reading: `M09.2` (or `--ref-deps`) |
| **Used by** | `L4.1` plugs it into the decoder with input feeding · `L4.4` decodes a Luong seq2seq in its tests · later `L6.7` the zoo's attention rows |
| **Milestone** | `MS-L4` (`em_luong >= 0.97` on the dates task) |
| **Optional depth** | Luong, Pham, and Manning, "Effective Approaches to Attention-based Neural Machine Translation" (EMNLP 2015), sections 3.1 and 3.3; Britz et al., "Massive Exploration of Neural Machine Translation Architectures" (2017), on additive vs multiplicative scores |

## Key Takeaways

- Luong keeps Bahdanau's masked read (`L4.2`) and changes two things: the score function, and when the decoder asks, **after** its recurrent step (`test_hand_example_dot`).
- `dot` is $h \cdot k_s$, `general` the bilinear $h \cdot (W_a k_s)$ (with $W_a = I$ it is `dot`), `concat` a hidden layer on $[h ; k_s]$ (`test_general_with_identity_is_dot`, `test_general_is_not_symmetric`, `test_concat_projection_splits_the_weight`).
- The attentional state $\tilde h = \tanh(W_c [c ; h])$, context first, is what the output layer reads and what input feeding passes to the next step (`test_hand_example_attentional_order`).
- The plain dot score is not scaled here; scaling by $1/\sqrt{d}$ is the transformer's change in `L5.1` (`test_dot_is_not_scaled`).

## How to work this chapter

```bash
ol start L4.3              # stubs luong.py; prints your test path and rung (R5)
ol tests L4.3              # the course tests
# write your oracle tests in python/tests/l4-3-luong/, then:
ol check L4.3              # course tests and the mutation grade of your tests
ol diff  L4.3              # after passing: your code against the reference
```

---

## 1. Why now

Bahdanau's attention (`L4.2`) works, but every decoder step runs a hidden layer over every source position, and its query is the state from **before** the step, so the decoder chooses where to look before it has seen the token it was just given. Luong, Pham, and Manning simplified both: score with a dot product, and attend with the state just computed. They also noticed that the decoder forgets where it looked, and fixed that by feeding the attentional state back into the next step's input. The dot score is the one the transformer keeps, so this module is the last step before `L5.1`.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $d$ | width of both the decoder state and the keys | `int` |
| $h$ | the query: the decoder state after this step | `float32[B, d]` |
| $k_s$ | the encoder output at source position $s$ | `float32[B, S, d]` |
| $W_a$ | `score_proj.weight`: $[d, d]$ for `general`, $[d, 2d]$ for `concat` | `float32` |
| $v$ | `v.weight`, `concat` only | `float32[1, d]` |
| $W_c$ | `combine.weight` | `float32[d, 2d]` |
| $e_s$, $a_s$, $c$ | score, weight, context as in `L4.2` | |
| $\tilde h$ | the attentional state $\tanh(W_c [c ; h])$ | `float32[B, d]` |

### 2.1 Attend after the step

Bahdanau: read with $s_{t-1}$, feed the context into the cell. Luong: run the cell first to get $h_t$, then read with $h_t$, then combine:

$$h_t = \mathrm{cell}([y_{t-1} ; \tilde h_{t-1}], h_{t-1}), \quad c_t = \sum_s a_s k_s, \quad \tilde h_t = \tanh(W_c [c_t ; h_t]), \quad \text{logits}_t = W_o \tilde h_t + b_o .$$

The decoder (`L4.1`) does the cell and the output layer; this module provides the scores, the read, and $\tilde h$. The class attribute `query_from = "current"` tells the decoder which order to use.

### 2.2 Three scores

| Score | $e_s$ | Parameters |
|---|---|---|
| `dot` | $h \cdot k_s$ | none |
| `general` | $h \cdot (W_a k_s)$ | $W_a$ $[d, d]$ |
| `concat` | $v \cdot \tanh(W_a [h ; k_s])$ | $W_a$ $[d, 2d]$, $v$ |

`dot` needs $h$ and $k_s$ in the same space; `general` learns a bilinear form between them (with $W_a = I$ it is `dot`, and $W_a$ and $W_a^\top$ give different scores, so weights load only one way round); `concat` is Bahdanau's network with one weight matrix on the concatenation. Masking, the softmax, and the context are exactly `L4.2`'s: padded scores $-\infty$, exact zero weights, an empty row reads nothing.

### 2.3 Precomputing the keys

For `concat`, $W_a [h ; k_s] = W_h h + W_k k_s$ where $W_h$ is the first $d$ columns of $W_a$ and $W_k$ the last $d$. So $W_k k_s$ is computed once per sentence (`project_keys`), and each step adds $W_h h$. For `general`, `project_keys` is $W_a k_s$; for `dot`, the keys themselves.

### 2.4 Input feeding

Without it, the decoder at step $t$ does not know what it attended to at step $t - 1$; Luong's input feeding concatenates $\tilde h_{t-1}$ to the next input embedding ($\tilde h_0 = 0$). The model can then avoid translating the same source word twice, a coverage signal for free. It also makes the network deeper in time: the gradient of step $t$ reaches the attention of step $t - 1$.

## 3. Worked example by hand

$d = 2$, the `dot` score, $h = (1, 0)$, keys $k_1 = (1, 0)$, $k_2 = (0, 1)$, $k_3 = (2, 0)$, the third padding.

1. Scores: $h \cdot k = (1, 0, 2)$; after the mask $(1, 0, -\infty)$.
2. Weights: $e^1 / (e^1 + 1) = 0.731059$ and $1 / (e + 1) = 0.268941$, then 0.
3. Context: $c = 0.731059 (1, 0) + 0.268941 (0, 1) = (0.731059, 0.268941)$.
4. With $W_c = [I \; I]$ (so $W_c [c ; h] = c + h$): $\tilde h = \tanh(1.731059, 0.268941) = (0.939181, 0.262640)$.
5. With $W_c = [I \; 0]$ instead, $\tilde h = \tanh(c) = (\tanh 0.731059, \tanh 0.268941)$: the context half is the first $d$ columns.

These are `test_hand_example_dot` and `test_hand_example_attentional_order`.

## 4. The interface

```python
class LuongAttention(Module):
    query_from = "current"
    def __init__(self, d: int, score: Literal["dot", "general", "concat"], rng=None) -> None: ...
    def project_keys(self, keys: Tensor) -> Tensor: ...               # [B, S, d]
    def scores(self, query: Tensor, keys: Tensor, proj=None) -> Tensor: ...   # [B, S], unmasked
    def forward(self, query, keys, mask, proj=None) -> tuple[Tensor, Tensor]: ...  # context, weights
    def attentional(self, query: Tensor, context: Tensor) -> Tensor: ...  # tanh(W_c [c ; h])
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_dot` | unit | section 3 steps 1 to 4 | you and the test agree on the read |
| `test_hand_example_attentional_order` | unit | $W_c = [I \; 0]$ gives $\tanh(c)$ | context first, as Luong trained it |
| `test_golden_torch` | golden | each score: weights, context, $\tilde h$, and every gradient against torch | the decoder trains through this |
| `test_gradcheck_every_input_and_parameter` | gradcheck | float64 central differences, each score, a padded row | every parameter learns |
| `test_general_with_identity_is_dot` | property | `general` with $W_a = I$ equals `dot` | the relation between the scores |
| `test_general_is_not_symmetric` | unit | $W_a$, not $W_a^\top$ | checkpoints load the right way round |
| `test_concat_projection_splits_the_weight` | property | `project_keys` and the score equal $v \cdot \tanh(W_a [h ; k])$ | the per-sentence cache is exact |
| `test_masked_weights_are_exactly_zero` | property | exact zeros, no gradient, an empty row reads nothing | padding is never read |
| `test_dot_is_not_scaled` | unit | scores exactly $(1, 0, 2)$ | the scaled score belongs to `L5.1` |
| `test_parameter_names_and_shapes` | unit | each score's keys and shapes, `combine` last, `query_from` | the safetensors keys of Luong checkpoints |
| `test_validation` | boundary | unknown score, wrong widths, wrong mask | wiring bugs fail loudly |

### Your graded tests (rung R5)

As in `L4.2`, the oracle is the formula in numpy with the module's own weights from `state_dict()`: all three scores, the masked softmax, the context, and $\tilde h$ in float64. Add the section 3 numbers, your own finite-difference gradient check of the keys, an empty row, and the parameter names. Import only contract modules (`tinyllm.seq2seq.luong`, `tinyllm.seq2seq.additive` for `length_mask`, `tinyllm.autograd.tensor`, `tinyllm.autograd.functional`).

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. $W_c [h ; c]$ instead of $W_c [c ; h]$ | Luong-trained weights decode garbage | `test_hand_example_attentional_order` (mutant `s01`) |
| 2. `general` with $W_a^\top$ | the bilinear form is transposed | `test_general_is_not_symmetric` (mutant `s02`) |
| 3. masking after the softmax | real weights do not sum to 1 | `test_masked_weights_are_exactly_zero` (mutant `s04`) |
| 4. scaling the dot score by $1/\sqrt{d}$ | every weight differs from Luong's | `test_dot_is_not_scaled` (mutant `s07`) |
| 5. `concat` precomputing with the query half of $W_a$ | the cached keys are multiplied by the wrong columns | `test_concat_projection_splits_the_weight` (mutant `s03`) |
| masking with $-10^9$ | an empty row reads padding uniformly | `test_masked_weights_are_exactly_zero` (mutant `s05`) |
| $\tilde h$ without the `tanh` | the attentional state is unbounded | `test_hand_example_dot` (mutant `s06`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L4.2` | the masked read and `length_mask` |
| Back | `L0.4` | `Linear` holds $W_a$, $v$, $W_c$ |
| Back | `L0.2` | the op library gives the backward |
| Back | `L0.1` | `Tensor` |
| Back | `M06.3` | `PCG32` for the default initialization |
| Forward | `L4.1` | the decoder steps first, attends with the new state, and feeds $\tilde h$ into the next input |
| Forward | `L4.4` | its tests beam-search a Luong seq2seq, whose state carries `feed` |
| Forward | `L5.1` | scaled dot-product attention: Luong's `dot` divided by $\sqrt{d}$, over many heads |
| Forward | `L6.7` | the zoo's Luong rows |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `dot` score | `F.scaled_dot_product_attention` | the $1/\sqrt{d}$ scale, heads, causal masks, fused kernels | `L5.1`; FlashAttention |
| input feeding | the attentional decoder of OpenNMT | the same `input_feed` option, with coverage and copy attention | OpenNMT-py `onmt/decoders/decoder.py` (`InputFeedRNNDecoder`) |
| `general` score | bilinear attention in readers and parsers | the biaffine scorer of dependency parsers | Dozat and Manning, "Deep Biaffine Attention" (2017) |
| local attention | Luong's local-p attention | reads a window around a predicted position: cost per step independent of $S$ | Luong et al. 2015, section 3.2; sliding-window attention in `L7.7` |
