# Pre-LN, RMSNorm

## Overview

| | |
|---|---|
| **Module** | `L7.1` · build · Python · Pass 5 · 45 min reading |
| **Solution** | [`python/tinyllm/modern/norm.py`](../../../../course/solutions/python/tinyllm/modern/norm.py): `RMSNorm` and `pre_norm_residual`; model oracle tests in [`learner-tests/L7.1/.../test_norm.py`](../../../../course/solutions/learner-tests/L7.1/python/tests/l7-1-norm/test_norm.py) |
| **Contract** | [`course/contracts/py/tinyllm/modern/norm.pyi`](../../../../course/contracts/py/tinyllm/modern/norm.pyi) |
| **Needs** | `L0.4` `Module` · `L0.2` `F.mean`, `F.sum` · `L0.1` `Tensor` and its `**` · reading: `M08.3` (the VJP of a norm by hand), `M09.2` (why a tiny denominator needs an eps) |
| **Used by** | `L7.6` the norms of MLA's query and key-value latents · `L7.9` every norm of a Llama model, where SmolLM2-135M's logits must match Hugging Face · later: `L9.6` the C `tl_rmsnorm_f32` parity, `L10.1` the Rust forward |
| **Optional depth** | Zhang and Sennrich, "Root Mean Square Layer Normalization" (2019); Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020), sections 3 and 4 |

## Key Takeaways

- RMSNorm divides each vector by its root mean square and multiplies by a learned gain; it subtracts no mean and has no bias.
- The eps goes **inside** the square root, and the difference shows on small activations.
- Gemma stores the gain minus one: `offset = 1` reproduces it.
- Pre-LN normalizes only the branch input, so the residual stream is an identity path in value and in gradient.

## How to read this chapter

Sections 1 to 3 derive RMSNorm and the Pre-LN step and normalize one vector by hand. Section 4 fixes the interface. Section 5 walks through the solution line by line and explains why each choice is right. Try to predict each step before reading it.

---

## 1. Why now

The 2017 transformer (`L5.5`) puts a LayerNorm after each residual add, and the Part 5 LayerNorm demo showed what that costs: without a learning-rate warmup the post-LN model diverged. Every model loaded from here on (SmolLM2, Llama, Mistral, Qwen) makes two changes to that block. The norm moves in front of each sublayer (Pre-LN), and LayerNorm becomes RMSNorm. Part 7 rebuilds the block piece by piece toward SmolLM2-135M, which has 61 RMSNorms; if one of them differs from Hugging Face's by where an eps sits, SmolLM2's logits drift past the 1e-3 tolerance against Hugging Face. This module gets the norm and the residual step exactly right.

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

LayerNorm on the same $x$ would first subtract the mean 1.75 and give a different vector.

## 4. The interface

```python
class RMSNorm(Module):
    def __init__(self, d: int, eps: float = 1e-6, offset: float = 0.0) -> None: ...   # weight [d]
    def forward(self, x: Tensor) -> Tensor: ...                                      # [..., d] -> [..., d]
def pre_norm_residual(x: Tensor, norm: Module, sublayer) -> Tensor: ...              # x + sublayer(norm(x))
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-worked vector | unit | section 3, with and without the gain | everyone agrees on the formula |
| agreement with Hugging Face Llama | golden | output and gradients against transformers' `LlamaRMSNorm` | SmolLM2's norms in `L7.9` |
| Gemma's offset | golden | `offset = 1` against `GemmaRMSNorm` | Gemma-style checkpoints |
| eps inside the root | golden | inputs of size 1e-3 where eps dominates | tiny activations in deep layers |
| gradients of $x$ and the gain | gradcheck | float64 central differences, both offsets | everything before the norm learns through it |
| unit RMS and scale invariance | property | unit RMS output, scale invariance | the sublayer sees directions |
| no mean is subtracted | boundary | a constant vector stays constant | RMSNorm is not LayerNorm |
| parameter name and init | unit | one parameter `weight`, float32, ones or zeros | the safetensors key of every norm |
| the Pre-LN identity path | property | a zero branch gives exactly $x$ and gradient 1 | deep stacks train |
| the Pre-LN step adds the branch | unit | $x + F(\operatorname{norm}(x))$, branch bounded | the Llama block in `L7.9` |
| each vector normalized on its own | property | per-token statistics over the last axis | no leakage across positions |
| bad arguments | boundary | wrong width, zero width, negative eps | wiring bugs fail loudly |

A model test suite whose oracle is the formula in numpy float64, `x / sqrt(mean(x**2, -1) + eps) * (offset + w)`, is in [`test_norm.py`](../../../../course/solutions/learner-tests/L7.1/python/tests/l7-1-norm/test_norm.py).

## 5. The solution, step by step

### 5.1 From first principles

1. The sublayer should see a vector on a fixed scale whatever the depth. Dividing by a per-vector size statistic does that; the root mean square is the cheapest one that makes the output scale-invariant (section 2.1).
2. The statistic must be over the last axis only, one per token, so tokens never influence each other's normalization.
3. A zero vector has RMS 0, so the denominator needs an $\epsilon$. Placing it inside the root, $\sqrt{\operatorname{mean}(x^2) + \epsilon}$, is the convention every Llama-family checkpoint was trained with; placing it outside changes the output on small activations.
4. A learned per-feature gain restores scale where the model needs it. Storing $w$ or $w - 1$ is a checkpoint convention, so `offset` reproduces both, and the init follows: ones for Llama, zeros for Gemma, so that a fresh norm is the plain normalization in both cases.
5. The backward needs no hand derivation if the forward is composed from differentiable ops (`x * x`, `F.mean`, `** -0.5`, products); the gain must be used as a Tensor so it receives its gradient.
6. Pre-LN is then one line: normalize only the branch input and add the branch to the untouched stream.

### 5.2 The code, annotated

```python
class RMSNorm(Module):
    def __init__(self, d: int, eps: float = 1e-6, offset: float = 0.0) -> None:
        super().__init__()
        # ... d a positive integer (bools refused); eps >= 0 ...
        self.d = int(d)
        self.eps = float(eps)
        self.offset = float(offset)
        # Llama starts the gain at 1; Gemma stores gain - 1 and starts it at 0.
        init = np.ones(self.d) if self.offset == 0.0 else np.zeros(self.d)
        # Named `weight`: the safetensors key HF uses for every norm.
        self.weight = Tensor(init.astype(np.float32), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.d:
            raise ValueError(f"RMSNorm({self.d}) got last axis {x.shape[-1]}")
        # eps goes inside the root: sqrt(mean(x^2) + eps), as Llama and T5.
        ms = F.mean(x * x, axis=-1, keepdims=True)   # one statistic per token; keepdims to broadcast
        y = x * (ms + self.eps) ** -0.5               # multiply by rsqrt rather than divide by sqrt
        # offset + w as a Tensor op, so the gradient still reaches self.weight.
        gain = self.weight + self.offset if self.offset != 0.0 else self.weight
        return y * gain
```

On the worked example: `x * x` is $(9, 16, 0, 0)$, `ms` is 6.25, `(ms + 0) ** -0.5` is 0.4, `y` is $(1.2, 1.6, 0, 0)$, and the gain $(1, 0.5, 2, 2)$ gives $(1.2, 0.8, 0, 0)$. The mean is used, not the sum: the sum would give $\sqrt{25} = 5$ and outputs $\sqrt d = 2$ times too small.

```python
def pre_norm_residual(x, norm, sublayer):
    # The residual stream x is never normalized: only the branch sees norm(x).
    return x + sublayer(norm(x))
```

With a branch that outputs zeros, this returns $x$ exactly and the gradient with respect to $x$ is exactly 1, the identity path of section 2.4.

Full solution: [`norm.py`](../../../../course/solutions/python/tinyllm/modern/norm.py).

### 5.3 Why this solution

- **Cost.** $O(d)$ per token: one square, one mean, one rsqrt, two multiplies. LayerNorm needs a second pass for the mean and a bias; RMSNorm drops both, which matters in a fused kernel where memory traffic dominates.
- **Composed from autograd ops** rather than a hand-written VJP: correctness of the backward is inherited from `L0.2`, at the cost of a few temporary arrays. `M08.3` derives the same gradient by hand for the C and Rust ports.
- **One class with `offset`** rather than separate Llama and Gemma norms: the formulas differ by a constant added to the gain.
- **Pre-LN as a function**, not a module: it has no parameters; the norm and the sublayer own theirs.
- **Tradeoff accepted.** Pre-LN lets the stream grow with depth, so models need a final norm before the head (`L7.9`).

### 5.4 Explain it back

1. Why must $\epsilon$ be inside the square root to match Hugging Face?
   <details><summary>Answer</summary>The checkpoints were trained with $x / \sqrt{\operatorname{mean}(x^2) + \epsilon}$. For activations whose mean square is comparable to $\epsilon$ (around $10^{-6}$), $\sqrt{m + \epsilon}$ and $\sqrt m + \epsilon$ differ by a large relative amount, so outputs drift by percents and the logits fail a 1e-3 parity check.</details>
2. Why does a Gemma norm start its weight at zeros while a Llama norm starts at ones?
   <details><summary>Answer</summary>Gemma computes the gain as $1 + w$. A zero init makes the gain 1, the same starting function as Llama's $w = 1$. Initializing Gemma's weight to ones would double every output.</details>
3. Why is the gain written as `self.weight + self.offset` and not built from `self.weight.data`?
   <details><summary>Answer</summary>Using `.data` would produce a constant array with no link to the parameter, so the gain would never receive a gradient and never train. Adding a Python float to the Tensor is an op that passes the gradient through unchanged.</details>
4. What does a constant vector $(c, c, c, c)$ become, and why does that distinguish RMSNorm from LayerNorm?
   <details><summary>Answer</summary>RMSNorm gives $(1, 1, 1, 1) \odot w$ (for $c > 0$, with $\epsilon$ small): its RMS is $|c|$. LayerNorm subtracts the mean first and gives zeros. RMSNorm keeps the direction, including any constant component.</details>
5. Why does Pre-LN train without warmup while Post-LN does not?
   <details><summary>Answer</summary>In Pre-LN the step's Jacobian is $I + \partial F(\operatorname{norm}(x))/\partial x$, so the gradient reaches every layer through an identity path. In Post-LN it passes through a norm at every layer, which at initialization makes the last layers' gradients large and early updates unstable.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. eps outside the root: $x / (\sqrt{\operatorname{mean}(x^2)} + \epsilon)$ | small activations come out several percent off | `(ms + self.eps) ** -0.5` |
| 2. subtracting the mean (LayerNorm statistics) | a constant vector becomes zeros | no mean is computed; only `mean(x * x)` |
| 3. ignoring Gemma's offset | a fresh Gemma norm outputs zeros | `gain = self.weight + self.offset` when `offset != 0` |
| 4. post-LN inside `pre_norm_residual` | the stream is normalized; the identity path is gone | returns `x + sublayer(norm(x))`, never `norm(x + ...)` |
| statistic over the wrong axis | tokens are normalized by each other | `axis=-1, keepdims=True` |
| gain initialized the wrong way | a fresh Llama norm outputs zeros | ones when `offset == 0`, zeros otherwise |
| sum of squares instead of the mean | outputs $\sqrt{d}$ times too small | `F.mean`, not `F.sum` |
| normalizing the stream, then adding the branch | the identity path carries $\operatorname{norm}(x)$ | the stream term is the raw `x` |
| the gain as a constant (`.data`) | the gain never trains | the gain is the parameter Tensor itself |

## 7. Where it's used next

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

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `RMSNorm` | HF `LlamaRMSNorm` | upcasts bf16 input to float32 for the statistic, casts back before the gain | `transformers/models/llama/modeling_llama.py` |
| `RMSNorm` | fused RMSNorm kernels (Apex, Liger, vLLM) | one pass over memory, fused with the residual add | vLLM `csrc/layernorm_kernels.cu` (`fused_add_rms_norm`) |
| `pre_norm_residual` | DeepNorm, sandwich norm, QK-norm | scaled residuals for 1000-layer models; extra norms on queries and keys (OLMo 2, Qwen3) | Wang et al. 2022 "DeepNet"; OLMo 2 report |
