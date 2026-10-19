# Gated MLPs: SwiGLU and GeGLU

The 2017 transformer's MLP is `Linear(d, 4d)`, ReLU or GELU, `Linear(4d, d)`. Open SmolLM2-135M's checkpoint and the MLP looks different: three matrices per layer, `gate_proj`, `up_proj`, and `down_proj`, of width 1536 for a model width of 576, not $4 \cdot 576 = 2304$. The 2017 MLP cannot load these weights, and a wrong guess at how the three combine gives logits that look plausible and are wrong.

Every model in the Llama family, Mistral, Qwen, Gemma, and the experts of every mixture of experts use this gated form. This section derives it, explains where the odd width comes from, and builds a module whose outputs and gradients match Hugging Face's for both the SwiGLU and GeGLU variants.

You will build on `Linear` and `Module` from [Module system and basic layers](../01-foundations/05-module-system-and-layers.md), `F.silu` and `F.gelu` from [The op library and gradcheck_all](../01-foundations/03-op-library-and-gradcheck.md), the `Tensor` from [Tensor, broadcasting backward, and no_grad](../01-foundations/02-tensor-and-broadcasting-backward.md), and the `PCG32` generator from [Modular arithmetic, hashing, and PCG32 in Python](../../mathematics-for-computing/07-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) in *Mathematics for Computing*, which supplies default initialization. [Activation Functions and Their Derivatives](../../mathematics-for-computing/02-calculus-1/03-activation-functions.md) covers silu and the tanh GELU. The reference solution is [`python/tinyllm/modern/mlp.py`](../../../../course/solutions/python/tinyllm/modern/mlp.py), which provides `GatedMLP` and `llama_ffn_dim`, against the contract [`course/contracts/py/tinyllm/modern/mlp.pyi`](../../../../course/contracts/py/tinyllm/modern/mlp.pyi).

> **Learning objectives**
> - Write the gated MLP formula and explain why the activation belongs on the gate.
> - Compute SwiGLU and GeGLU outputs by hand for a scalar example.
> - Derive the $\frac{8}{3}d$ hidden width and reproduce Meta's rounding rule, including Llama 3's multiplier.
> - Build a module whose parameter names, order, and rng draws match Hugging Face.

## Notation

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

## From one projection to two

The plain MLP is $W_2\, \phi(W_1 x)$: one hidden vector, one nonlinearity per unit. A **gated linear unit** (Dauphin et al.) computes two hidden vectors from the same input and lets one control the other elementwise: $(W_g x) \odot (W_u x)$, passed through a nonlinearity on the gate side. Written out in full, the modern MLP is

$$\operatorname{GatedMLP}(x) = W_d \left( \operatorname{act}(W_g x) \odot W_u x \right).$$

Unit $i$ outputs $\operatorname{act}(g_i) \cdot u_i$. When $g_i$ is very negative, $\operatorname{silu}(g_i) \approx 0$ and the unit is closed regardless of $u_i$; when $g_i$ is large, $\operatorname{silu}(g_i) \approx g_i$ and the unit passes $g_i u_i$, a product of two linear functions of $x$. The network can therefore form products of input features, which a single ReLU layer cannot.

Shazeer compared the activations. SwiGLU ($\operatorname{act} = \operatorname{silu}$, used by Llama and SmolLM2) and GeGLU ($\operatorname{act} = \operatorname{gelu}_{\tanh}$, used by Gemma) both reach lower perplexity than ReLU or GELU MLPs at equal parameters. Neither has a closed-form reason; it is an empirical result that every model since has kept.

The backward again comes from autograd: `F.silu` and `F.gelu` carry the derivatives from *Mathematics for Computing*, and the products do the rest. The gate receives the upstream gradient times $u_i \cdot \operatorname{act}'(g_i)$, and the up projection receives it times $\operatorname{act}(g_i)$. Both projections learn, and a closed gate also stops the gradient to its up row.

## Sizing d_ff

A third matrix costs parameters, so the hidden width has to shrink to compensate. Three $f \times d$ matrices hold $3 d f$ numbers. To compare fairly with the plain MLP's $2 \cdot d \cdot 4d = 8 d^2$, set $f = \frac{8}{3} d$. Meta's code computes $h = \lfloor \frac{2}{3} \cdot 4d \rfloor$, optionally scales it by `ffn_dim_multiplier`, and rounds **up** to a multiple of `multiple_of` (256 or 1024) so the matrix products tile well on hardware.

## Working it by hand

Take scalars throughout: $d = f = 1$, $W_g = 1$, $W_u = 2$, $W_d = 3$, $x = 1$.

1. The gate is $g = 1$ and the up projection is $u = 2$.
2. For SwiGLU, $\sigma(1) = 1 / (1 + e^{-1}) = 0.731059$, so $\operatorname{silu}(1) = 0.731059$. Times $u$ that is $1.462117$, and times $W_d$ the output is $4.386352$.
3. For GeGLU, $\sqrt{2/\pi} \cdot 1.044715 = 0.833562$, its $\tanh$ is $0.682384$, and $\operatorname{gelu}_{\tanh}(1) = 0.841192$. The output is $3 \cdot 2 \cdot 0.841192 = 5.047152$.

The sizing rule works the same way on real models. Llama-2 7B has $d = 4096$, so $\lfloor 2 \cdot 16384 / 3 \rfloor = 10922$, which rounded up to a multiple of 256 is $43 \cdot 256 = 11008$. SmolLM2 has $d = 576$: $\lfloor 2 \cdot 2304 / 3 \rfloor = 1536$, already a multiple of 256.

## The interface

```python
class GatedMLP(Module):
    def __init__(self, d, d_ff, act: Literal["silu", "gelu_tanh"] = "silu", bias=False, rng=None): ...
    def forward(self, x: Tensor) -> Tensor: ...          # [..., d] -> [..., d]
def llama_ffn_dim(d, multiple_of=256, ffn_dim_multiplier=None) -> int: ...
```

`GatedMLP` maps `[..., d]` to `[..., d]` and owns three linear layers named as in the checkpoint. `act` selects between the two allowed activations, `bias` applies to all three layers, and `rng` is the single stream the weights are drawn from. `llama_ffn_dim` derives the hidden width for configs that omit `intermediate_size`.

## Building it step by step

The checkpoint fixes the shape: three matrices named `gate_proj`, `up_proj` (both $f \times d$) and `down_proj` ($d \times f$). So the module is three `Linear` layers under exactly those names.

The formula must be the one the weights were trained with, $W_d(\operatorname{act}(W_g x) \odot W_u x)$. The activation goes on the gate, because that is what makes "closed" mean closed: $\operatorname{silu}(g) \approx 0$ for very negative $g$ shuts unit $i$ whatever $u_i$ is. The activation itself is a configuration choice, `silu` for SwiGLU (Llama, SmolLM2) and the **tanh** GELU for GeGLU (Gemma's `gelu_pytorch_tanh`), since the erf form differs by about $10^{-4}$ per unit.

Registration order sets both the `state_dict` order and the order weights are drawn from one rng. Following Hugging Face (gate, up, down) makes both reproducible. A missing `intermediate_size` must be derived the way Meta does: $h = \lfloor \frac{2}{3} \cdot 4d \rfloor$, optionally times `ffn_dim_multiplier` (truncated), rounded up to `multiple_of`. For $d = 4096$ that gives $10922 \to 11008$. Finally, the backward follows from `F.silu`, `F.gelu`, and the product; all three projections are parameters used as Tensors, so all three train.

Here is the module:

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

On the worked example ($W_g = 1$, $W_u = 2$, $W_d = 3$, $x = 1$): $g = 1$, $a = \operatorname{silu}(1) = 0.731059$, $a \cdot u = 1.462117$, and down gives $4.386352$. Writing `act(a * up)` instead would give $3\operatorname{silu}(2) = 5.2848$, a different function with the same parameters. `bias` applies to all three layers, as Hugging Face's `mlp_bias` does.

The width helper is a few lines of integer arithmetic:

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

`(h + m - 1) // m` is the integer ceiling of $h / m$. For Llama 3 8B ($d = 4096$, multiplier 1.3, `multiple_of` 1024): $h = 10922$, $\lfloor 1.3 \cdot 10922 \rfloor = 14198$, rounded up to $14 \cdot 1024 = 14336$. The full solution is in [`mlp.py`](../../../../course/solutions/python/tinyllm/modern/mlp.py).

### Why this design

The cost is three matmuls of $d \times f$ per token: $3df$ multiply-adds and parameters, equal to a plain $4d$ MLP's $8d^2$ at $f = \frac83 d$. The elementwise activation and product are $O(f)$.

The solution keeps three separate `Linear` layers rather than one fused $[2f, d]$ gate-and-up matrix so the keys load from Hugging Face checkpoints directly. Production engines such as vLLM fuse them for speed, and the C kernel in [Elementwise kernels in C](../10-kernels/08-elementwise-kernels.md) fuses $\operatorname{silu}(g) \odot u$. The activation is a string option rather than a callable because the two allowed values map to checkpoint conventions, and a typo fails at construction. The width rounds up, never down, because a width smaller than the checkpoint's cannot hold its weights.

The tradeoff you accept is that the gated form's advantage over ReLU and GELU MLPs is empirical (Shazeer 2020), not derived. The solution adopts it because every model it must load does.

## What can go wrong

The tests pin down each of those decisions. The hand-worked example must come out right for both activations. Output and gradients must match transformers' `LlamaMLP`, including checkpoints with `mlp_bias`, and `GemmaMLP` with `gelu_pytorch_tanh`. A float64 gradcheck covers every parameter for both activations and with biases. Parameter names, order, shapes, and `act` must match Hugging Face so the safetensors keys line up, and gate, up, and down must be drawn from one rng in that order so one seed fixes the weights everywhere. A property test checks that a negative gate shuts its unit, which is only true if the activation is on the gate, and another checks that the parameter count at $f = \frac83 d$ is $3 d \cdot \frac{8}{3} d = 8 d^2$, matching a $4d$ MLP. `llama_ffn_dim` must return 11008, 14336, 8192, and 1536 for the corresponding configs, and an unknown activation or a zero width must fail loudly. A model test suite whose oracle is the formula in numpy float64 with the module's own weights from `state_dict()` is in [`test_mlp.py`](../../../../course/solutions/learner-tests/L7.2/python/tests/l7-2-mlp/test_mlp.py).

> **Warning:** Putting the activation on the up projection loads without error, because the gate and up matrices have the same shape, and then every logit is wrong. The same is true of activating the product, $\operatorname{act}(g \odot u)$. Neither shows up as a shape error.

The full list of mistakes and their symptoms:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| activation on the up projection | loads fine, every logit wrong | `a` is computed from `gate_proj(x)` and multiplies `up_proj(x)` |
| exact GELU for GeGLU | Gemma outputs off by about 1e-4 per unit, growing with depth | `F.gelu(g, approximate="tanh")` |
| rounding `d_ff` down | 10752 instead of 11008; the checkpoint shapes do not match | integer ceiling `(h + m - 1) // m` |
| activating the product, $\operatorname{act}(g \odot u)$ | a different function with the same parameters | the activation is applied to `g` before the product |
| registering up before gate | `state_dict` order and draws differ from Hugging Face | layers assigned in the order gate, up, down |
| dropping the down bias | biased checkpoints load and are off by the bias | `bias=bias` on all three layers |
| ignoring `ffn_dim_multiplier` | Llama-3 widths wrong | `h = int(ffn_dim_multiplier * h)` before rounding |
| the up projection as a constant | up never trains | `self.up_proj(x)` is a Tensor op on a parameter |
| a fresh rng for the last layer | two seeds give the same down weights | one `r` is passed to all three layers |

## Where this leads

The gated MLP is reused directly in two places in this chapter. Each expert in [Mixture of Experts](08-mixture-of-experts.md) is a `GatedMLP`, and the `mlp` of every decoder layer in [The Llama-Family Model: Config, Weight Loading, and Download](09-llama-family-model.md) is one too, where SmolLM2-135M's logits must match Hugging Face. Later, `tl_silu_mul_f32` in [Elementwise kernels in C](../10-kernels/08-elementwise-kernels.md) fuses $\operatorname{silu}(g) \odot u$ in C.

### Further reading

- Shazeer, ["GLU Variants Improve Transformer"](https://arxiv.org/abs/2002.05202) (2020).
- Dauphin et al., "Language Modeling with Gated Convolutional Networks" (2017), section 3.
- Touvron et al., "LLaMA" (2023), section 2.2.
- Hugging Face `LlamaMLP` and `GemmaMLP` in `transformers/models/llama/modeling_llama.py`: the same three matrices; tensor-parallel splits shard `gate` and `up` by rows and `down` by columns.
- vLLM `MergedColumnParallelLinear` and `SiluAndMul` in `model_executor/layers/activation.py`: gate and up fused into one $[2f, d]$ matrix, with the activation fused into the product.
- Meta's `FeedForward` in `llama/model.py` of Meta's llama repository: the reference sizing rule.

## Key terms

- **Gated linear unit**: two projections of the same input, one passed through an activation and multiplied elementwise into the other.
- **SwiGLU**: the gated MLP with silu on the gate (Llama, SmolLM2).
- **GeGLU**: the gated MLP with the tanh-approximated GELU on the gate (Gemma).
- **`d_ff` (intermediate size)**: the hidden width $f$, about $\frac83 d$ for a gated MLP, rounded up to `multiple_of`.
- **`ffn_dim_multiplier`**: Llama 3's widening factor applied to the hidden width before rounding.

## Summary

- A gated MLP computes $\text{down}(\text{act}(\text{gate}(x)) \odot \text{up}(x))$: two projections, one gating the other.
- The activation belongs on the gate: a closed gate shuts its unit off whatever the other projection says.
- Three matrices at width $\frac{8}{3} d$ cost what two at $4d$ cost, which is where Llama's odd widths come from; Meta rounds up to a hardware-friendly multiple.
- Gemma's GeGLU uses the tanh approximation of GELU, not the erf form.
- Layer names, registration order, and a single rng stream make the module load Hugging Face weights and reproduce seeded initialization.

## Review questions

1. Why is the activation applied to the gate projection and not to the up projection?
   <details><summary>Answer</summary>The trained weights assume $\operatorname{act}(W_g x) \odot W_u x$. Swapping the roles loads without error (the shapes are equal) but computes a different function, so every logit is wrong. Functionally, the activation on the gate is what lets a unit be switched off independently of its value.</details>
2. Why is $d_{ff}$ about $\frac{8}{3}d$ rather than $4d$?
   <details><summary>Answer</summary>The gated MLP has three $f \times d$ matrices instead of two. Matching the plain MLP's $8d^2$ parameters requires $3df = 8d^2$, so $f = \frac83 d$.</details>
3. Why must all three layers draw from the same rng in the order gate, up, down?
   <details><summary>Answer</summary>A seed should fix the weights in every language and implementation. One stream consumed in Hugging Face's registration order gives the same weights everywhere; a fresh stream per layer would make, for example, two different seeds produce identical down weights.</details>
4. What goes wrong if `llama_ffn_dim` rounds down?
   <details><summary>Answer</summary>For $d = 4096$ it would give 10752 instead of 11008, and the module's shapes would not match the checkpoint's `gate_proj`, `up_proj` and `down_proj`.</details>
5. Why does GeGLU use `approximate="tanh"`?
   <details><summary>Answer</summary>Gemma's config uses `gelu_pytorch_tanh`. The erf GELU differs by about $10^{-4}$ per unit, and those errors accumulate across layers until parity with Hugging Face fails.</details>
6. On the scalar example, what does the buggy form $W_d\operatorname{silu}(W_g x \cdot W_u x)$ give, and what does the correct one give?
   <details><summary>Answer</summary>The buggy form gives $3\operatorname{silu}(2) = 5.2848$; the correct form gives $3 \cdot \operatorname{silu}(1) \cdot 2 = 4.386352$.</details>
