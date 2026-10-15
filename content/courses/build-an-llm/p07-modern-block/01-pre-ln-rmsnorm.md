<!-- ol:module L7.1 -->
# Pre-LN, RMSNorm

## Overview

| | |
|---|---|
| **Module** | `L7.1` · build · Python · Pass 5 · 2 to 3 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/modern/norm.py`: `RMSNorm` and `pre_norm_residual`; and your own oracle tests in `python/tests/l7-1-norm/` |
| **Contract** | [`course/contracts/py/tinyllm/modern/norm.pyi`](../../../../course/contracts/py/tinyllm/modern/norm.pyi) |
| **Tests** | `course/tests/L7.1/test_norm.py` (what they check: section 4), golden values from transformers 5.19.0 `LlamaRMSNorm` and `GemmaRMSNorm` in `course/fixtures/L7.1/rmsnorm_hf.npz` (`course/oracle/L7.1/rmsnorm_hf.py`); your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | `L0.4` `Module` (and `Linear` in the tests) · `L0.2` `F.mean`, `F.sum` · `L0.1` `Tensor` and its `**` · reading: `M08.3` (the VJP of a norm by hand), `M09.2` (why a tiny denominator needs an eps) (or `--ref-deps`) |
| **Used by** | `L7.6` the norms of MLA's query and key-value latents · `L7.9` every norm of a Llama model · later: `L9.6` the C `tl_rmsnorm_f32` parity, `L10.1` the Rust forward |
| **Milestone** | `MS-L7` (SmolLM2-135M logits match Hugging Face) |
| **Optional depth** | Zhang and Sennrich, "Root Mean Square Layer Normalization" (2019); Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020), sections 3 and 4 |

## Key Takeaways

- RMSNorm divides each vector by its root mean square and multiplies by a learned gain; it subtracts no mean and has no bias (`test_hand_example`, `test_no_mean_is_subtracted`).
- The eps goes **inside** the square root, and the difference shows on small activations (`test_eps_inside_the_root`).
- Gemma stores the gain minus one: `offset = 1` reproduces it (`test_golden_hf_gemma_offset`).
- Pre-LN normalizes only the branch input, so the residual stream is an identity path in value and in gradient (`test_pre_norm_residual_identity_path`).

## How to work this chapter

```bash
ol start L7.1              # stubs norm.py; prints your test path and rung (R5)
ol tests L7.1              # the course tests
# write your oracle tests in python/tests/l7-1-norm/, then:
ol check L7.1              # course tests and the mutation grade of your tests
ol diff  L7.1              # after passing: your code against the reference
```

---

## 1. Why now

Your 2017 transformer (`L5.5`) puts a LayerNorm after each residual add, and the MS-L5 run showed what that costs: without a learning-rate warmup the post-LN model diverged. Every model you load from here on (SmolLM2, Llama, Mistral, Qwen) makes two changes to that block. The norm moves in front of each sublayer (Pre-LN), and LayerNorm becomes RMSNorm. Part 7 rebuilds the block piece by piece toward SmolLM2-135M, which has 61 RMSNorms; if one of them differs from Hugging Face's by where an eps sits, the logits at MS-L7 drift past the 1e-3 tolerance. This module builds the norm and the residual step exactly.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x \in \mathbb{R}^d$ | one token's hidden vector | `float32[..., d]` |
| $d$ | width of the vector | `int` |
| $\operatorname{rms}(x)$ | $\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}$ | scalar per vector |
| $\epsilon$ | `eps`, a small constant that keeps the root away from 0 | `float` |
| $w$ | the learned gain, `weight` | `float32[d]` |
| $o$ | `offset`: 0 for Llama, 1 for Gemma | `float` |
| $y$ | the output, $x / \operatorname{rms}(x) \odot (o + w)$ | `float32[..., d]` |
| $F$ | a sublayer (attention or MLP) | function |

### 2.1 Normalizing by the root mean square

A deep network adds a branch output to its hidden vector at every layer, so the size of that vector drifts with depth. A norm puts every vector back on a common scale before a sublayer reads it. LayerNorm (`L0.4`) subtracts the mean and divides by the standard deviation. Zhang and Sennrich found that the re-centering contributes little: dividing by the **root mean square** alone, $\operatorname{rms}(x) = \sqrt{\operatorname{mean}(x^2) + \epsilon}$, works as well and is cheaper. After the division (with $\epsilon = 0$) the mean of the squares is exactly 1, and scaling $x$ by any $c > 0$ changes nothing: the sublayer sees the direction of $x$, not its length. Every token is normalized by its own statistic, over the last axis only.

### 2.2 The gain and Gemma's offset

A unit-RMS vector would force every feature to the same scale, so a learned per-feature gain $w$ restores what the model needs: $y = \frac{x}{\operatorname{rms}(x)} \odot w$. Llama initializes $w$ to ones. Gemma stores $w - 1$ instead (initialized to zeros) and computes $\frac{x}{\operatorname{rms}(x)} \odot (1 + w)$; weight decay then pulls the gain toward 1, not toward 0. The contract covers both with `offset`: $y = \frac{x}{\operatorname{rms}(x)} \odot (o + w)$.

### 2.3 Backward

Every step is an op of `L0.2` ($x \cdot x$, `F.mean`, a power of $-\frac12$, two products), so the backward comes from the autograd engine. `M08.3` derives it by hand: the gradient with respect to $x$ removes the component of the upstream gradient along $x$ (moving along $x$ only rescales it, which the norm undoes) and divides by the RMS. The gain gets $\frac{x}{\operatorname{rms}(x)} \odot g$ summed over tokens.

### 2.4 Pre-LN

Post-LN (2017) computes $\operatorname{norm}(x + F(x))$: the residual stream itself is normalized after every block, so the gradient that reaches early layers passes through every norm on the way. Pre-LN computes

$$x + F(\operatorname{norm}(x)).$$

The stream $x$ is never normalized; only the branch input is. The derivative of the step with respect to $x$ is $I + \frac{\partial F(\operatorname{norm}(x))}{\partial x}$, so the gradient always has a clean identity path from the loss to the embedding. Xiong et al. show this is why Pre-LN trains at full learning rate without warmup. The price is that the stream grows with depth, so a Pre-LN model applies one more RMSNorm after the last block, before the lm_head (`L7.9`).

## 3. Worked example by hand

$d = 4$, $\epsilon = 0$, $x = (3, 4, 0, 0)$.

1. Squares: $9, 16, 0, 0$; mean $25 / 4 = 6.25$.
2. $\operatorname{rms}(x) = \sqrt{6.25} = 2.5$.
3. $x / \operatorname{rms}(x) = (1.2, 1.6, 0, 0)$. Check: the mean of the squares is $(1.44 + 2.56) / 4 = 1$.
4. Gain $w = (1, 0.5, 2, 2)$: $y = (1.2, 0.8, 0, 0)$.

LayerNorm on the same $x$ would first subtract the mean 1.75 and give a different vector. This is `test_hand_example`.

## 4. The interface

```python
class RMSNorm(Module):
    def __init__(self, d: int, eps: float = 1e-6, offset: float = 0.0) -> None: ...   # weight [d]
    def forward(self, x: Tensor) -> Tensor: ...                                      # [..., d] -> [..., d]
def pre_norm_residual(x: Tensor, norm: Module, sublayer) -> Tensor: ...              # x + sublayer(norm(x))
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | section 3, with and without the gain | you and the test agree on the formula |
| `test_golden_hf_llama` | golden | output and gradients against `LlamaRMSNorm` | SmolLM2's norms in `L7.9` |
| `test_golden_hf_gemma_offset` | golden | `offset = 1` against `GemmaRMSNorm` | Gemma-style checkpoints |
| `test_eps_inside_the_root` | golden | inputs of size 1e-3 where eps dominates | tiny activations in deep layers |
| `test_gradcheck_x_and_gain` | gradcheck | float64 central differences, both offsets | everything before the norm learns through it |
| `test_unit_rms_and_scale_invariance` | property | unit RMS output, scale invariance | the sublayer sees directions |
| `test_no_mean_is_subtracted` | boundary | a constant vector stays constant | RMSNorm is not LayerNorm |
| `test_parameter_names_and_init` | unit | one parameter `weight`, float32, ones or zeros | the safetensors key of every norm |
| `test_pre_norm_residual_identity_path` | property | a zero branch gives exactly $x$ and gradient 1 | deep stacks train |
| `test_pre_norm_residual_adds_the_branch` | unit | $x + F(\operatorname{norm}(x))$, branch bounded | the Llama block in `L7.9` |
| `test_normalizes_each_vector_on_its_own` | property | per-token statistics over the last axis | no leakage across positions |
| `test_validation` | boundary | wrong width, zero width, negative eps | wiring bugs fail loudly |

### Your graded tests (rung R5)

Your oracle is the formula in numpy float64: `x / sqrt(mean(x**2, -1) + eps) * (offset + w)`. Compare `forward` with it for both offsets and for inputs scaled by 1e-3, check the gradients of $x$ and the gain with your own central differences, the initial gain, and the Pre-LN step against `x + W @ oracle(x)` for a `Linear` sublayer. Import only `tinyllm.modern.norm`, `tinyllm.nn.layers`, `tinyllm.autograd.tensor`, and `tinyllm.autograd.functional`. `ol check L7.1` requires 0.80 with every pitfall fault killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. eps outside the root: $x / (\sqrt{\operatorname{mean}(x^2)} + \epsilon)$ | small activations come out several percent off | `test_eps_inside_the_root` (mutant `s01`) |
| 2. subtracting the mean (LayerNorm statistics) | a constant vector becomes zeros | `test_no_mean_is_subtracted`, `test_hand_example` (mutant `s02`) |
| 3. ignoring Gemma's offset | a fresh Gemma norm outputs zeros | `test_golden_hf_gemma_offset` (mutant `s03`) |
| 4. post-LN inside `pre_norm_residual` | the stream is normalized; the identity path is gone | `test_pre_norm_residual_identity_path` (mutant `s04`) |
| statistic over the wrong axis | tokens are normalized by each other | `test_normalizes_each_vector_on_its_own` (mutant `s05`) |
| gain initialized the wrong way | a fresh Llama norm outputs zeros | `test_parameter_names_and_init` (mutant `s06`) |
| sum of squares instead of the mean | outputs $\sqrt{d}$ times too small | `test_hand_example` (mutant `s07`) |
| normalizing the stream, then adding the branch | the identity path carries $\operatorname{norm}(x)$ | `test_pre_norm_residual_adds_the_branch` (mutant `s08`) |
| the gain as a constant (`.data`) | the gain never trains | `test_golden_hf_llama`, `test_gradcheck_x_and_gain` (mutant `s09`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.4` | `Module` registers the gain; `Linear` is the test sublayer |
| Back | `L0.2` | `F.mean` and `F.sum` give the backward for free |
| Back | `L0.1` | `Tensor` and its power op |
| Back | `M08.3` | the RMSNorm VJP derived by hand |
| Forward | `L7.6` | `q_a_layernorm` and `kv_a_layernorm` normalize MLA's latents |
| Forward | `L7.9` | `input_layernorm`, `post_attention_layernorm`, and the final `norm` of every Llama model |
| Forward | `L9.6` | `tl_rmsnorm_f32` in C must match this to float32 tolerance |
| Forward | `L10.1` | the Rust forward calls the C norm |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `RMSNorm` | HF `LlamaRMSNorm` | upcasts bf16 input to float32 for the statistic, casts back before the gain | `transformers/models/llama/modeling_llama.py` |
| `RMSNorm` | fused RMSNorm kernels (Apex, Liger, vLLM) | one pass over memory, fused with the residual add | vLLM `csrc/layernorm_kernels.cu` (`fused_add_rms_norm`) |
| `pre_norm_residual` | DeepNorm, sandwich norm, QK-norm | scaled residuals for 1000-layer models; extra norms on queries and keys (OLMo 2, Qwen3) | Wang et al. 2022 "DeepNet"; OLMo 2 report |
