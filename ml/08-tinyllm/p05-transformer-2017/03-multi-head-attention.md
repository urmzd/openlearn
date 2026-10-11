<!-- ol:module L5.3 -->
# Multi-head attention

## Overview

| | |
|---|---|
| **Module** | `L5.3` · build · Python · Pass 5 · 2 to 3 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/xfmr/mha.py`: `MultiHeadAttention` with `split_heads`, `merge_heads`, `head_mask`, `attend`, `forward`, `load_packed_in_proj`; and your own oracle tests in `python/tests/l5-3-mha/` |
| **Contract** | [`course/contracts/py/tinyllm/xfmr/mha.pyi`](../../../course/contracts/py/tinyllm/xfmr/mha.pyi) |
| **Tests** | `course/tests/L5.3/test_mha.py` (what they check: section 4); the oracle is torch's `nn.MultiheadAttention` with copied weights; your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | `L5.1` scaled dot-product attention · `L0.4` `Linear`, `Module` · `L0.2` `reshape`, `transpose` · `L0.1` `Tensor` · `M06.3` `PCG32` (or `--ref-deps`) |
| **Used by** | `L5.5` encoder and decoder layers · `L6.1` GPT blocks · `L6.2` BERT layers |
| **Milestone** | `MS-L5` (`{tinyllm} train transformer`, then `translate --beam 4`) |
| **Optional depth** | Vaswani et al., "Attention Is All You Need" (2017), section 3.2.2; Michel, Levy, and Neubig, "Are Sixteen Heads Really Better than One?" (NeurIPS 2019) |

## Key Takeaways

- A head attends inside its own $d_{\text{head}} = d_{\text{model}} / H$ slice of the features, so one position can read two places at once; one wide head cannot (`test_hand_example_two_heads`, `test_hand_example_one_head_mixes_everything`).
- Splitting is a reshape to `[B, T, H, d_head]` **and** a transpose to `[B, H, T, d_head]`; merging undoes both, in that order (`test_split_and_merge_layout`).
- Each head scales by $1/\sqrt{d_{\text{head}}}$, the width it compares, not $1/\sqrt{d_{\text{model}}}$ (`test_golden_torch`).
- Self-attention and cross-attention are the same module: queries from `x_q`, keys and values from `x_kv` (`test_cross_attention_reads_keys_from_x_kv`).
- A per-sequence mask needs a head axis, or sequence $b$'s mask silently lands on head $b$ (`test_mask_shapes_agree`).

## How to work this chapter

```bash
ol start L5.3              # stubs mha.py; prints your test path and rung (R5)
ol tests L5.3              # the course tests
# write your oracle tests in python/tests/l5-3-mha/ (section 4 says what to cover), then:
ol check L5.3              # course tests and the mutation grade of your tests
ol mutate L5.3             # the full grade, cached by your test files' hash
ol diff  L5.3              # after passing: your code against the reference
```

---

## 1. Why now

`L5.1` gave you one attention: every query compares itself with every key through one dot product over all $d$ features, and gets one weight vector. Take the addition task of `MS-L5`: to write the tens digit of the sum, the decoder has to look at the tens digit of $a$ **and** the tens digit of $b$, four positions apart. A single softmax can split its mass between the two, but then it averages them into one blurred vector. Vaswani et al. split the model width into $H$ heads that each run their own attention with their own projections, and concatenate the results: head 0 can find $a$'s digit while head 1 finds $b$'s. Every model of Parts 5 and 6 (`L5.5`'s encoder-decoder, `L6.1`'s GPT, `L6.2`'s BERT) calls this one module, so its layout and names are fixed here once.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $B, T_q, T_k$ | batch, query length, key length | `int` |
| $d$ | model width `d_model` | `int` |
| $H$ | number of heads `n_heads` | `int`, divides $d$ |
| $d_h = d / H$ | width of one head `d_head` | `int` |
| $X_q, X_{kv}$ | the inputs queries and keys/values come from | `[B, Tq, d]`, `[B, Tk, d]` |
| $W_q, W_k, W_v, W_o$ | the four projections (`q_proj`, `k_proj`, `v_proj`, `out_proj`) | `float32[d, d]` each, plus biases `[d]` |
| $Q_h, K_h, V_h$ | head $h$'s slice of the projected queries, keys, values | `[B, Tq, dh]`, `[B, Tk, dh]` |
| $M$ | mask, True = may attend | bool, broadcasts to `[B, H, Tq, Tk]` |

### 2.1 One head

One head is `L5.1` after learned projections:

$$\text{head}(X_q, X_{kv}) = \operatorname{softmax}\!\Big(\frac{(X_q W_q^\top)(X_{kv} W_k^\top)^\top}{\sqrt{d}} + M\Big)\, X_{kv} W_v^\top .$$

With $H = 1$ the module is exactly this followed by $W_o$; `test_one_head_is_projected_attention` computes it in numpy from the module's own weights.

### 2.2 Heads

With $H$ heads the projections stay $d \times d$, but their output is read in $H$ slices of width $d_h$: head $h$ owns features $h d_h$ to $(h + 1) d_h - 1$. Each head attends in its own slice, scaled by $1/\sqrt{d_h}$ because a dot product of $d_h$ terms with unit-variance entries has variance $d_h$:

$$\text{head}_h = \operatorname{softmax}\!\Big(\frac{Q_h K_h^\top}{\sqrt{d_h}} + M\Big) V_h, \qquad \text{out} = [\text{head}_0 ; \dots ; \text{head}_{H-1}]\, W_o^\top .$$

In code all heads run in one batched call. A projected `[B, T, d]` tensor is reshaped to `[B, T, H, d_h]` (the slices become an axis) and transposed to `[B, H, T, d_h]`, so `L5.1` sees `H` more batch rows. Merging reverses it: transpose back to `[B, T, H, d_h]`, then reshape to `[B, T, d]`. Reshaping `[B, H, T, d_h]` straight to `[B, T, d]` is legal numpy and wrong: it glues together the rows of different positions. The cost is the same as one wide head: four $d \times d$ projections and $H$ attentions of width $d_h$.

### 2.3 Masks for every head

Masks come from `L5.2` and mean "True = may attend". A causal mask is `[Tq, Tk]` and applies to every sequence and head; a padding mask is per sequence. The module accepts `[Tq, Tk]`, `[B, Tq, Tk]` (it inserts the head axis at 1), and any shape that already broadcasts to `[B, H, Tq, Tk]`, such as `[B, 1, 1, Tk]` key padding. Numpy aligns shapes from the right, so a `[B, Tq, Tk]` mask without the inserted axis broadcasts against `[B, H, Tq, Tk]` with its batch axis on the head axis: when $B = H$ nothing fails and the masks are simply wrong.

### 2.4 Self- and cross-attention

`attend(x_q, x_kv, mask)` takes queries from `x_q` and **both** keys and values from `x_kv`. Self-attention passes the same tensor twice; the decoder's cross-attention (`L5.5`) passes its own states as `x_q` and the encoder's output as `x_kv`. The output has `x_q`'s length. Keys are a set: permuting the positions of `x_kv` (with its mask) leaves the output unchanged, and permuting `x_q` permutes the output rows. Position information therefore has to come from outside, which is `L5.4`'s job.

### 2.5 Dropout and torch's packed layout

Attention dropout zeroes attention weights at random during training (rate `dropout`, drawn from `PCG32(0).substream("dropout")` by `L5.1`) and is off in eval mode. torch's `nn.MultiheadAttention` keeps $W_q, W_k, W_v$ stacked in one `in_proj_weight` of shape `[3d, d]`, rows in the order q, k, v; `load_packed_in_proj` copies them into the three `Linear`s.

## 3. Worked example by hand

$d = 4$, $H = 2$, $d_h = 2$, and every projection the identity with zero bias, so $Q = X_q$, $K = V = X_{kv}$. One query and two keys:

| | features 0, 1 (head 0) | features 2, 3 (head 1) |
|---|---|---|
| $q = (1, 0, 0, 2)$ | $(1, 0)$ | $(0, 2)$ |
| $k_1 = (1, 0, 0, 0)$ | $(1, 0)$ | $(0, 0)$ |
| $k_2 = (0, 1, 0, 2)$ | $(0, 1)$ | $(0, 2)$ |

**Head 0.** Scores $(1, 0) / \sqrt 2 = (0.707107, 0)$; weights $\sigma(0.707107) = 0.669762$ and $0.330238$; context $0.669762 (1, 0) + 0.330238 (0, 1) = (0.669762, 0.330238)$.

**Head 1.** Scores $(0, 4) / \sqrt 2 = (0, 2.828427)$; weights $\sigma(-2.828427) = 0.055817$ and $0.944183$; context $0.944183 \cdot (0, 2) = (0, 1.888366)$.

**Merge.** Head 0's context fills features 0 and 1, head 1's fills 2 and 3: $(0.669762, 0.330238, 0, 1.888366)$, and $W_o = I$ leaves it. This is `test_hand_example_two_heads`.

With **one** head of width 4 the same numbers give scores $(q \cdot k_1, q \cdot k_2) / 2 = (0.5, 2)$, weights $(0.182426, 0.817574)$ for every feature, and output $(0.182426, 0.817574, 0, 1.635149)$: head 0's preference for $k_1$ is outvoted (`test_hand_example_one_head_mixes_everything`).

## 4. The interface

```python
class MultiHeadAttention(Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True, rng=None)
    def split_heads(self, x: Tensor) -> Tensor              # [B, T, d] -> [B, H, T, dh]
    def merge_heads(self, x: Tensor) -> Tensor              # [B, H, T, dh] -> [B, T, d]
    def head_mask(self, mask, B, Tq, Tk) -> Optional[NDArray]
    def attend(self, x_q: Tensor, x_kv: Tensor, mask=None) -> tuple[Tensor, Tensor]   # out, weights [B, H, Tq, Tk]
    def forward(self, x_q: Tensor, x_kv: Tensor, mask=None) -> Tensor
    def load_packed_in_proj(self, weight, bias=None) -> None  # torch's [3d, d], rows q | k | v
```

Parameters, in `state_dict` order: `q_proj.weight`, `q_proj.bias`, `k_proj.*`, `v_proj.*`, `out_proj.*` (L0.4 `Linear`s, drawn from one `rng`).

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_two_heads` | unit | section 3: weights per head and the merged output | you and the test agree on the layout and the scale |
| `test_hand_example_one_head_mixes_everything` | unit | the same numbers with one head | why heads exist |
| `test_golden_torch` | golden | torch `nn.MultiheadAttention`: output, per-head weights, every gradient, self (causal + padding) and cross (padding) | `L5.5`, `L6.1`, `L6.2` load torch-trained weights |
| `test_gradcheck_inputs_and_parameters` | gradcheck | both inputs and all 8 parameter tensors against frozen central differences | every model trains through this backward |
| `test_split_and_merge_layout` | property | head $h$ owns features $h d_h ..$; merge inverts split | positions never mix |
| `test_heads_are_independent` | property | changing head 1's query rows leaves head 0's weights bitwise equal | heads are separate subspaces |
| `test_one_head_is_projected_attention` | differential | $H = 1$ equals numpy softmax$(QK^\top/\sqrt d)V$ then $W_o$ | the definition, independently |
| `test_mask_shapes_agree` | property | `[Tq, Tk]`, `[B, Tq, Tk]`, `[B, H, Tq, Tk]` masks agree, with $B = H$ | the decoder's causal-and-padding masks |
| `test_masked_keys_are_never_read` | property | masked keys weigh exactly 0; their values cannot change the output | padding in every batch |
| `test_cross_attention_reads_keys_from_x_kv` | property | output length from `x_q`; `x_kv` positions are a set | `L5.5`'s cross-attention |
| `test_dropout_only_in_training` | unit | eval mode equals no dropout | evaluation and decoding are deterministic |
| `test_parameter_names_shapes_and_init` | unit | the eight names in order, seeded init, `bias=False` | checkpoint keys of Parts 5 to 7 |
| `test_load_packed_in_proj` | unit | torch's packed rows are q, k, v | loading torch weights (`L5.5`'s golden test) |
| `test_validation` | boundary | heads must tile $d$; shapes; bool masks | wiring bugs fail loudly |

### Your graded tests (rung R5)

Rung R5 asks for **oracles**. Write multi-head attention again in float64 numpy from the module's own `state_dict` (project, reshape, transpose, softmax with the mask, merge, project) and compare `attend`'s output and weights with it, for self- and cross-attention, with a per-sequence `[B, Tq, Tk]` mask where $B = H$. Add the section 3 numbers, a split/merge check, eval-mode dropout, the packed load order, and a gradient check with your own `tinyllm.num.gradcheck` (`M04.1`) in float64. Import only names in `contracts/py`. `ol check L5.3` requires a mutation score of at least 0.80 with every pitfall fault (`s01` to `s07`) killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. merging (or splitting) heads with a plain reshape | positions mix; torch disagrees; nothing crashes | `test_split_and_merge_layout`, `test_golden_torch` (mutants `s01`, `s02`) |
| 2. scaling by $1/\sqrt{d_{\text{model}}}$ | softmax too flat by a factor $\sqrt H$ | `test_hand_example_two_heads`, `test_golden_torch` (mutant `s03`) |
| 3. projecting keys from `x_q` | cross-attention reads the decoder, or crashes when lengths differ | `test_cross_attention_reads_keys_from_x_kv` (mutant `s04`) |
| 4. a `[B, Tq, Tk]` mask without a head axis | with $B = H$ head $b$ gets sequence $b$'s mask | `test_mask_shapes_agree` (mutant `s05`) |
| 5. dropout in eval mode | decoding is random, evaluation noisy | `test_dropout_only_in_training` (mutant `s06`) |
| 6. reading torch's packed `in_proj` as k, q, v | a loaded checkpoint attends with swapped roles | `test_load_packed_in_proj` (mutant `s07`) |
| the output projection skipped | heads are never mixed; shapes still fit | `test_one_head_is_projected_attention` (mutant `s08`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L5.1` | `scaled_dot_product_attention` runs all heads in one batched call |
| Back | `L0.4` | `Linear` for the four projections, `Module` for registration |
| Back | `L0.2` | `reshape` and `transpose` for split and merge |
| Back | `L0.1` | the `Tensor` every input and parameter is |
| Back | `M06.3` | `PCG32` for the default init and dropout streams |
| Forward | `L5.5` | encoder self-attention, decoder masked self-attention and cross-attention |
| Forward | `L6.1` | GPT-2's causal self-attention (c_attn split into q, k, v) |
| Forward | `L6.2` | BERT's bidirectional self-attention |
| Forward | `L7.5` | grouped-query attention shares $K, V$ heads between query heads |

If you skip this module, `ol check L5.5` stops with `needs L5.3`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `split_heads` / `merge_heads` | PyTorch `F.scaled_dot_product_attention` on `[B, H, T, d]` | fused kernels (FlashAttention, memory-efficient) behind one call | `torch/nn/functional.py`, `aten/src/ATen/native/transformers/` |
| `load_packed_in_proj` | HF `GPT2Attention` c_attn, Llama's separate q/k/v | fused QKV matmul for speed, split views | `transformers/models/gpt2/modeling_gpt2.py` |
| one K, V per head | multi-query and grouped-query attention | fewer K, V heads: a smaller KV cache | Shazeer (2019); Ainslie et al. (2023); `L7.5` |
| attention dropout | most modern LLMs train with 0 | dropout matters for small data, not trillions of tokens | Llama and Mistral configs |
