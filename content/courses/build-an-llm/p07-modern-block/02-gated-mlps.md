# Gated MLPs: SwiGLU, GeGLU

## Overview

| | |
|---|---|
| **Module** | `L7.2` · build · Python · Pass 5 · 45 min reading |
| **Solution** | [`python/tinyllm/modern/mlp.py`](../../../../course/solutions/python/tinyllm/modern/mlp.py): `GatedMLP` and `llama_ffn_dim`; model oracle tests in [`learner-tests/L7.2/.../test_mlp.py`](../../../../course/solutions/learner-tests/L7.2/python/tests/l7-2-mlp/test_mlp.py) |
| **Contract** | [`course/contracts/py/tinyllm/modern/mlp.pyi`](../../../../course/contracts/py/tinyllm/modern/mlp.pyi) |
| **Needs** | `L0.4` `Linear`, `Module` · `L0.2` `F.silu`, `F.gelu` · `L0.1` `Tensor` · `M06.3` `PCG32` (default initialization) · reading: `M01.3` (silu and the tanh GELU) |
| **Used by** | `L7.8` every expert is a gated MLP · `L7.9` every Llama MLP, where SmolLM2-135M's logits must match Hugging Face · later: `L9.6` the C `tl_silu_mul_f32` |
| **Optional depth** | Shazeer, "GLU Variants Improve Transformer" (2020); Dauphin et al., "Language Modeling with Gated Convolutional Networks" (2017), section 3; Touvron et al., "LLaMA" (2023), section 2.2 |

## Key Takeaways

- A gated MLP computes two projections of the input and multiplies one by the activation of the other: $\text{down}(\text{act}(\text{gate}(x)) \odot \text{up}(x))$.
- The activation belongs on the **gate**: a closed gate shuts its unit off whatever the other projection says.
- Three matrices at width $\frac{8}{3} d$ cost what two at $4d$ cost, which is where Llama's odd widths come from.
- Gemma's GeGLU uses the tanh approximation of GELU, not the erf form.

## How to read this chapter

Sections 1 to 3 derive the gated MLP and its width and work both activations and the Llama sizes by hand. Section 4 fixes the interface. Section 5 walks through the solution line by line and explains why each choice is right. Try to predict each step before reading it.

---

## 1. Why now

The 2017 transformer's MLP is `Linear(d, 4d)`, ReLU or GELU, `Linear(4d, d)`. Open SmolLM2-135M's checkpoint and the MLP has three matrices per layer, `gate_proj`, `up_proj`, and `down_proj`, of width 1536 for a model width of 576: not $4 \cdot 576 = 2304$. The 2017 MLP cannot load these weights, and a wrong guess at how the three combine gives logits that look plausible and are wrong. Every model in the Llama family, Mistral, Qwen, Gemma, and the experts of every MoE (`L7.8`) use this gated form. This module gives the gated form and explains the width.

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

## 4. The interface

```python
class GatedMLP(Module):
    def __init__(self, d, d_ff, act: Literal["silu", "gelu_tanh"] = "silu", bias=False, rng=None): ...
    def forward(self, x: Tensor) -> Tensor: ...          # [..., d] -> [..., d]
def llama_ffn_dim(d, multiple_of=256, ffn_dim_multiplier=None) -> int: ...
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked example | unit | section 3 for both activations | everyone agrees on the formula |
| agreement with Hugging Face SwiGLU | golden | output and gradients against transformers' `LlamaMLP` | SmolLM2's MLPs in `L7.9` |
| SwiGLU with biases | golden | `mlp_bias` checkpoints | biased variants load |
| agreement with Hugging Face GeGLU | golden | `GemmaMLP` with `gelu_pytorch_tanh` | Gemma-style checkpoints |
| gradients of every parameter | gradcheck | float64, both activations, with biases | every matrix trains |
| parameter names, order and shapes | unit | HF names, order, shapes, `act` | the safetensors keys |
| rng draw order | unit | gate, up, down drawn from one rng in order | one seed fixes the weights everywhere |
| a closed gate blocks the unit | property | a negative gate shuts the unit | the activation is on the gate |
| parameter count matches a $4d$ MLP | property | $3 d \cdot \frac{8}{3} d = 8 d^2$ | equal-parameter comparisons |
| `llama_ffn_dim` | unit | 11008, 14336, 8192, 1536 | configs without `intermediate_size` |
| bad arguments | boundary | unknown activation, zero width | config typos fail loudly |

A model test suite whose oracle is the formula in numpy float64 with the module's own weights from `state_dict()` is in [`test_mlp.py`](../../../../course/solutions/learner-tests/L7.2/python/tests/l7-2-mlp/test_mlp.py).

## 5. The solution, step by step

### 5.1 From first principles

1. The checkpoint fixes the shape: three matrices named `gate_proj`, `up_proj` (both $f \times d$) and `down_proj` ($d \times f$). So the module is three `Linear` layers under exactly those names.
2. The formula must be the one the weights were trained with: $W_d(\operatorname{act}(W_g x) \odot W_u x)$. The activation goes on the gate, because that is what makes "closed" mean closed: $\operatorname{silu}(g) \approx 0$ for very negative $g$ shuts unit $i$ whatever $u_i$ is.
3. The activation is a configuration choice: `silu` for SwiGLU (Llama, SmolLM2) and the **tanh** GELU for GeGLU (Gemma's `gelu_pytorch_tanh`), since the erf form differs by about $10^{-4}$ per unit.
4. Registration order sets both the `state_dict` order and the order weights are drawn from one rng. Following HF (gate, up, down) makes both reproducible.
5. A missing `intermediate_size` must be derived the way Meta does: $h = \lfloor \frac{2}{3} \cdot 4d \rfloor$, optionally times `ffn_dim_multiplier` (truncated), rounded **up** to `multiple_of`. For $d = 4096$ that gives $10922 \to 11008$.
6. The backward follows from `F.silu`, `F.gelu` and the product; all three projections are parameters used as Tensors, so all three train.

### 5.2 The code, annotated

```python
class GatedMLP(Module):
    def __init__(self, d, d_ff, act="silu", bias=False, rng=None):
        super().__init__()
        # ... positive sizes; act in ("silu", "gelu_tanh") ...
        r = rng if rng is not None else PCG32(0).substream("init")   # ONE stream for all three
        self.act = act
        # Registration order is HF's LlamaMLP order: gate, up, down.
        self.gate_proj = Linear(d, d_ff, bias=bias, rng=r)
        self.up_proj = Linear(d, d_ff, bias=bias, rng=r)
        self.down_proj = Linear(d_ff, d, bias=bias, rng=r)

    def forward(self, x: Tensor) -> Tensor:
        g = self.gate_proj(x)
        a = F.silu(g) if self.act == "silu" else F.gelu(g, approximate="tanh")
        # The activation gates the OTHER projection: act(gate) * up.
        return self.down_proj(a * self.up_proj(x))
```

On the worked example ($W_g = 1$, $W_u = 2$, $W_d = 3$, $x = 1$): $g = 1$, $a = \operatorname{silu}(1) = 0.731059$, $a \cdot u = 1.462117$, and down gives $4.386352$. Writing `act(a * up)` instead would give $3\operatorname{silu}(2) = 5.2848$, a different function with the same parameters. `bias` applies to all three layers, as HF's `mlp_bias` does.

```python
def llama_ffn_dim(d, multiple_of=256, ffn_dim_multiplier=None):
    # ... d and multiple_of positive ...
    h = int(2 * (4 * d) / 3)                       # floor(8d/3): equal parameters to a 4d MLP
    if ffn_dim_multiplier is not None:
        # ... must be positive ...
        h = int(ffn_dim_multiplier * h)            # Llama 3's widening, truncated like Meta's code
    # Round UP to a multiple: hardware-friendly matmul widths.
    return multiple_of * ((h + multiple_of - 1) // multiple_of)
```

`(h + m - 1) // m` is the integer ceiling of $h / m$. Llama 3 8B ($d = 4096$, multiplier 1.3, `multiple_of` 1024): $h = 10922$, $\lfloor 1.3 \cdot 10922 \rfloor = 14198$, rounded up to $14 \cdot 1024 = 14336$.

Full solution: [`mlp.py`](../../../../course/solutions/python/tinyllm/modern/mlp.py).

### 5.3 Why this solution

- **Cost.** Three matmuls of $d \times f$ per token: $3df$ multiply-adds and parameters, equal to a plain $4d$ MLP's $8d^2$ at $f = \frac83 d$. The elementwise activation and product are $O(f)$.
- **Three separate `Linear` layers** rather than one fused $[2f, d]$ gate-and-up matrix: the keys load from HF checkpoints directly. Production engines (vLLM) fuse them for speed; the C kernel of `L9.6` fuses $\operatorname{silu}(g) \odot u$.
- **The activation as a string option** rather than a callable: the two allowed values map to checkpoint conventions, and a typo fails at construction.
- **Rounding up**, never down: a width smaller than the checkpoint's cannot hold its weights.
- **Tradeoff accepted.** The gated form's advantage over ReLU/GELU MLPs is empirical (Shazeer 2020), not derived; the solution adopts it because every model it must load does.

### 5.4 Explain it back

1. Why is the activation applied to the gate projection and not to the up projection?
   <details><summary>Answer</summary>The trained weights assume $\operatorname{act}(W_g x) \odot W_u x$. Swapping the roles loads without error (the shapes are equal) but computes a different function, so every logit is wrong. Functionally, the activation on the gate is what lets a unit be switched off independently of its value.</details>
2. Why is $d_{ff}$ about $\frac{8}{3}d$ rather than $4d$?
   <details><summary>Answer</summary>The gated MLP has three $f \times d$ matrices instead of two. Matching the plain MLP's $8d^2$ parameters requires $3df = 8d^2$, so $f = \frac83 d$.</details>
3. Why must all three layers draw from the same rng in the order gate, up, down?
   <details><summary>Answer</summary>A seed should fix the weights in every language and implementation. One stream consumed in HF's registration order gives the same weights everywhere; a fresh stream per layer would make, for example, two different seeds produce identical down weights.</details>
4. What goes wrong if `llama_ffn_dim` rounds down?
   <details><summary>Answer</summary>For $d = 4096$ it would give 10752 instead of 11008, and the module's shapes would not match the checkpoint's `gate_proj`, `up_proj` and `down_proj`.</details>
5. Why does GeGLU use `approximate="tanh"`?
   <details><summary>Answer</summary>Gemma's config uses `gelu_pytorch_tanh`. The erf GELU differs by about $10^{-4}$ per unit, and those errors accumulate across layers until parity with Hugging Face fails.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. activation on the up projection | loads fine, every logit wrong | `a` is computed from `gate_proj(x)` and multiplies `up_proj(x)` |
| 2. exact GELU for GeGLU | Gemma outputs off by about 1e-4 per unit, growing with depth | `F.gelu(g, approximate="tanh")` |
| 3. rounding `d_ff` down | 10752 instead of 11008; the checkpoint shapes do not match | integer ceiling `(h + m - 1) // m` |
| 4. activating the product, $\operatorname{act}(g \odot u)$ | a different function with the same parameters | the activation is applied to `g` before the product |
| registering up before gate | `state_dict` order and draws differ from Hugging Face | layers assigned in the order gate, up, down |
| dropping the down bias | biased checkpoints load and are off by the bias | `bias=bias` on all three layers |
| ignoring `ffn_dim_multiplier` | Llama-3 widths wrong | `h = int(ffn_dim_multiplier * h)` before rounding |
| the up projection as a constant | up never trains | `self.up_proj(x)` is a Tensor op on a parameter |
| a fresh rng for the last layer | two seeds give the same down weights | one `r` is passed to all three layers |

## 7. Where it's used next

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

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `GatedMLP` | HF `LlamaMLP`, `GemmaMLP` | the same three matrices; tensor-parallel splits shard `gate` and `up` by rows and `down` by columns | `transformers/models/llama/modeling_llama.py` |
| `GatedMLP` | vLLM `MergedColumnParallelLinear` + `SiluAndMul` | gate and up fused into one $[2f, d]$ matrix, the activation fused with the product | vLLM `model_executor/layers/activation.py` |
| `llama_ffn_dim` | Meta's `FeedForward` | the reference rule | `llama/model.py` in Meta's llama repository |
