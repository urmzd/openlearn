<!-- ol:module L7.2 -->
# Gated MLPs: SwiGLU, GeGLU

## Overview

| | |
|---|---|
| **Module** | `L7.2` · build · Python · Pass 5 · 2 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/modern/mlp.py`: `GatedMLP` and `llama_ffn_dim`; and your own oracle tests in `python/tests/l7-2-mlp/` |
| **Contract** | [`course/contracts/py/tinyllm/modern/mlp.pyi`](../../../course/contracts/py/tinyllm/modern/mlp.pyi) |
| **Tests** | `course/tests/L7.2/test_mlp.py` (what they check: section 4), golden values from transformers 5.19.0 `LlamaMLP` and `GemmaMLP` in `course/fixtures/L7.2/gated_mlp_hf.npz` (`course/oracle/L7.2/gated_mlp_hf.py`); your tests are graded by mutation, threshold 0.80 with every pitfall fault required |
| **Needs** | `L0.4` `Linear`, `Module` · `L0.2` `F.silu`, `F.gelu` · `L0.1` `Tensor` · `M06.3` `PCG32` (default initialization) · reading: `M01.3` (silu and the tanh GELU) (or `--ref-deps`) |
| **Used by** | `L7.8` every expert is a gated MLP · `L7.9` every Llama MLP · later: `L9.6` the C `tl_silu_mul_f32` |
| **Milestone** | `MS-L7` (SmolLM2-135M logits match Hugging Face) |
| **Optional depth** | Shazeer, "GLU Variants Improve Transformer" (2020); Dauphin et al., "Language Modeling with Gated Convolutional Networks" (2017), section 3; Touvron et al., "LLaMA" (2023), section 2.2 |

## Key Takeaways

- A gated MLP computes two projections of the input and multiplies one by the activation of the other: $\text{down}(\text{act}(\text{gate}(x)) \odot \text{up}(x))$ (`test_hand_example`).
- The activation belongs on the **gate**: a closed gate shuts its unit off whatever the other projection says (`test_closed_gate_blocks_the_unit`).
- Three matrices at width $\frac{8}{3} d$ cost what two at $4d$ cost, which is where Llama's odd widths come from (`test_parameter_count_matches_a_4d_mlp`, `test_llama_ffn_dim`).
- Gemma's GeGLU uses the tanh approximation of GELU, not the erf form (`test_golden_hf_geglu`).

## How to work this chapter

```bash
ol start L7.2              # stubs mlp.py; prints your test path and rung (R5)
ol tests L7.2              # the course tests
# write your oracle tests in python/tests/l7-2-mlp/, then:
ol check L7.2              # course tests and the mutation grade of your tests
ol diff  L7.2              # after passing: your code against the reference
```

---

## 1. Why now

Your 2017 transformer's MLP is `Linear(d, 4d)`, ReLU or GELU, `Linear(4d, d)`. Open SmolLM2-135M's checkpoint and the MLP has three matrices per layer, `gate_proj`, `up_proj`, and `down_proj`, of width 1536 for a model width of 576: not $4 \cdot 576 = 2304$. Your MLP cannot load these weights, and if you guess how the three combine you get logits that look plausible and are wrong. Every model in the Llama family, Mistral, Qwen, Gemma, and the experts of every MoE (`L7.8`) use this gated form. This module builds it and explains the width.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | one token's hidden vector | `float32[..., d]` |
| $d$, $f$ | model width, hidden width `d_ff` | `int` |
| $W_g$, $W_u$ | gate and up weights | `float32[f, d]` |
| $W_d$ | down weight | `float32[d, f]` |
| $\sigma(z)$ | logistic sigmoid $1 / (1 + e^{-z})$ | elementwise |
| $\operatorname{silu}(z)$ | $z \sigma(z)$ | elementwise |
| $\operatorname{gelu}_{\tanh}(z)$ | $\frac{z}{2}\left(1 + \tanh\left(\sqrt{2/\pi}\,(z + 0.044715 z^3)\right)\right)$ | elementwise |
| $\odot$ | elementwise product | |

### 2.1 From one projection to two

The plain MLP is $W_2\, \phi(W_1 x)$: one hidden vector, one nonlinearity per unit. A **gated linear unit** (Dauphin et al.) computes two hidden vectors from the same input and lets one control the other elementwise: $(W_g x) \odot (W_u x)$ passed through a nonlinearity on the gate side.

### 2.2 Gating

$$\operatorname{GatedMLP}(x) = W_d \left( \operatorname{act}(W_g x) \odot W_u x \right).$$

Unit $i$ outputs $\operatorname{act}(g_i) \cdot u_i$. When $g_i$ is very negative, $\operatorname{silu}(g_i) \approx 0$ and the unit is closed regardless of $u_i$; when $g_i$ is large, $\operatorname{silu}(g_i) \approx g_i$ and the unit passes $g_i u_i$, a product of two linear functions of $x$. The network can therefore form products of input features, which a single ReLU layer cannot. Shazeer compared the activations: SwiGLU ($\operatorname{act} = \operatorname{silu}$, Llama and SmolLM2) and GeGLU ($\operatorname{act} = \operatorname{gelu}_{\tanh}$, Gemma) both reach lower perplexity than ReLU or GELU MLPs at equal parameters. Neither has a closed-form reason; it is an empirical result that every model since has kept.

### 2.3 Backward

`F.silu` and `F.gelu` (`L0.2`, from `M01.3`'s derivatives) and the products give the backward through autograd. The gate receives the upstream gradient times $u_i \cdot \operatorname{act}'(g_i)$, the up projection receives it times $\operatorname{act}(g_i)$: both projections learn, and a closed gate also stops the gradient to its up row.

### 2.4 Sizing d_ff

Three $f \times d$ matrices hold $3 d f$ numbers. To compare fairly with the plain MLP's $2 \cdot d \cdot 4d = 8 d^2$, set $f = \frac{8}{3} d$. Meta's code computes $h = \lfloor \frac{2}{3} \cdot 4d \rfloor$, optionally scales it by `ffn_dim_multiplier`, and rounds **up** to a multiple of `multiple_of` (256 or 1024) so the matrix products tile well on hardware.

## 3. Worked example by hand

$d = f = 1$, $W_g = 1$, $W_u = 2$, $W_d = 3$, $x = 1$.

1. Gate: $g = 1$. Up: $u = 2$.
2. SwiGLU: $\sigma(1) = 1 / (1 + e^{-1}) = 0.731059$, so $\operatorname{silu}(1) = 0.731059$; times $u$: $1.462117$; times $W_d$: $4.386352$.
3. GeGLU: $\sqrt{2/\pi} \cdot 1.044715 = 0.833562$, $\tanh = 0.682384$, $\operatorname{gelu}_{\tanh}(1) = 0.841192$; output $3 \cdot 2 \cdot 0.841192 = 5.047152$.

Sizes: Llama-2 7B has $d = 4096$: $\lfloor 2 \cdot 16384 / 3 \rfloor = 10922$, rounded up to a multiple of 256 is $43 \cdot 256 = 11008$. SmolLM2: $\lfloor 2 \cdot 2304 / 3 \rfloor = 1536$, already a multiple of 256.

These are `test_hand_example` and `test_llama_ffn_dim`.

## 4. The interface

```python
class GatedMLP(Module):
    def __init__(self, d, d_ff, act: Literal["silu", "gelu_tanh"] = "silu", bias=False, rng=None): ...
    def forward(self, x: Tensor) -> Tensor: ...          # [..., d] -> [..., d]
def llama_ffn_dim(d, multiple_of=256, ffn_dim_multiplier=None) -> int: ...
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | section 3 for both activations | you and the test agree on the formula |
| `test_golden_hf_swiglu` | golden | output and gradients against `LlamaMLP` | SmolLM2's MLPs in `L7.9` |
| `test_golden_hf_swiglu_bias` | golden | `mlp_bias` checkpoints | biased variants load |
| `test_golden_hf_geglu` | golden | `GemmaMLP` with `gelu_pytorch_tanh` | Gemma-style checkpoints |
| `test_gradcheck_every_parameter` | gradcheck | float64, both activations, with biases | every matrix trains |
| `test_parameter_names_order_and_shapes` | unit | HF names, order, shapes, `act` | the safetensors keys |
| `test_rng_draw_order` | unit | gate, up, down drawn from one rng in order | one seed fixes the weights everywhere |
| `test_closed_gate_blocks_the_unit` | property | a negative gate shuts the unit | the activation is on the gate |
| `test_parameter_count_matches_a_4d_mlp` | property | $3 d \cdot \frac{8}{3} d = 8 d^2$ | equal-parameter comparisons |
| `test_llama_ffn_dim` | unit | 11008, 14336, 8192, 1536 | configs without `intermediate_size` |
| `test_validation` | boundary | unknown activation, zero width | config typos fail loudly |

### Your graded tests (rung R5)

Your oracle is the formula in numpy float64 with the module's own weights from `state_dict()`: compute gate, up, the activation, the product, and down, and compare with `forward` for both activations, with and without biases. Add the hand example, the parameter names, your own central differences for $x$ and every parameter, and the published `llama_ffn_dim` values. Import only `tinyllm.modern.mlp`, `tinyllm.autograd.tensor`, and `tinyllm.autograd.functional`. `ol check L7.2` requires 0.80 with every pitfall fault killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. activation on the up projection | loads fine, every logit wrong | `test_golden_hf_swiglu`, `test_closed_gate_blocks_the_unit` (mutant `s01`) |
| 2. exact GELU for GeGLU | Gemma outputs off by about 1e-4 per unit, growing with depth | `test_golden_hf_geglu` (mutant `s02`) |
| 3. rounding `d_ff` down | 10752 instead of 11008; the checkpoint shapes do not match | `test_llama_ffn_dim` (mutant `s03`) |
| 4. activating the product, $\operatorname{act}(g \odot u)$ | a different function with the same parameters | `test_hand_example` (mutant `s04`) |
| registering up before gate | `state_dict` order and draws differ from Hugging Face | `test_parameter_names_order_and_shapes` (mutant `s05`) |
| dropping the down bias | biased checkpoints load and are off by the bias | `test_golden_hf_swiglu_bias` (mutant `s06`) |
| ignoring `ffn_dim_multiplier` | Llama-3 widths wrong | `test_llama_ffn_dim` (mutant `s07`) |
| the up projection as a constant | up never trains | `test_gradcheck_every_parameter` (mutant `s08`) |
| a fresh rng for the last layer | two seeds give the same down weights | `test_rng_draw_order` (mutant `s09`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L0.4` | three `Linear` layers |
| Back | `L0.2` | `F.silu`, `F.gelu(approximate="tanh")` with their VJPs |
| Back | `L0.1` | `Tensor` products |
| Back | `M06.3` | `PCG32` initializes the layers when no rng is given |
| Forward | `L7.8` | each expert of a mixture of experts is a `GatedMLP` |
| Forward | `L7.9` | `mlp` of every Llama decoder layer |
| Forward | `L9.6` | `tl_silu_mul_f32` fuses $\operatorname{silu}(g) \odot u$ in C |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `GatedMLP` | HF `LlamaMLP`, `GemmaMLP` | the same three matrices; tensor-parallel splits shard `gate` and `up` by rows and `down` by columns | `transformers/models/llama/modeling_llama.py` |
| `GatedMLP` | vLLM `MergedColumnParallelLinear` + `SiluAndMul` | gate and up fused into one $[2f, d]$ matrix, the activation fused with the product | vLLM `model_executor/layers/activation.py` |
| `llama_ffn_dim` | Meta's `FeedForward` | the reference rule | `llama/model.py` in Meta's llama repository |
