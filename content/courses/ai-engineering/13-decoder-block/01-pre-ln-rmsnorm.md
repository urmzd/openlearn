# Pre-LN and RMSNorm

The 2017 transformer you built in [Encoder-decoder Transformer: post-LN and pre-LN, Noam, label smoothing](../11-transformer/05-encoder-decoder-transformer.md) puts a LayerNorm after each residual add, and the LayerNorm demo in that section showed what that costs: without a learning-rate warmup, the post-LN model diverged. Every model you will load from here on (SmolLM2, Llama, Mistral, Qwen) makes two changes to that block. The norm moves in front of each sublayer, which is called Pre-LN, and LayerNorm becomes RMSNorm.

This chapter rebuilds the block piece by piece toward SmolLM2-135M, and that model has 61 RMSNorms. If one of them differs from Hugging Face's by where an eps sits, SmolLM2's logits drift past the 1e-3 tolerance against Hugging Face. So this section gets the norm and the residual step exactly right, down to the placement of one constant.

You will need the `Module` system from [Module system and basic layers](../06-numerics-and-autograd/05-module-system-and-layers.md), the ops `F.mean` and `F.sum` from [The op library and gradcheck_all](../06-numerics-and-autograd/03-op-library-and-gradcheck.md), and the `Tensor` with its `**` operator from [Tensor, broadcasting backward, and no_grad](../06-numerics-and-autograd/02-tensor-and-broadcasting-backward.md). Two sections of *Mathematics* are useful companion reading: [Matrix differentials, the trace trick, and closed-form VJPs](../../mathematics/14-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md) derives the VJP of a norm by hand, and [Stable numerics](../../mathematics/15-numerical-methods-and-floating-point/02-stable-numerics.md) explains why a tiny denominator needs an eps. The reference solution is [`python/tinyllm/modern/norm.py`](../../../../course/solutions/python/tinyllm/modern/norm.py), which provides `RMSNorm` and `pre_norm_residual`, and the interface it implements is the contract [`course/contracts/py/tinyllm/modern/norm.pyi`](../../../../course/contracts/py/tinyllm/modern/norm.pyi).

> **Learning objectives**
> - Compute RMSNorm by hand and explain why it drops LayerNorm's mean subtraction and bias.
> - Place eps inside the square root and explain why the placement shows on small activations.
> - Reproduce Gemma's stored-gain convention with an `offset` and pick the matching initialization.
> - Write the Pre-LN residual step and explain why it gives the gradient an identity path.

## Notation

The section uses a handful of symbols, collected here for reference.

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

## Normalizing by the root mean square

A deep network adds a branch output to its hidden vector at every layer, so the size of that vector drifts with depth. A norm puts every vector back on a common scale before a sublayer reads it. LayerNorm, which you built in [Module system and basic layers](../06-numerics-and-autograd/05-module-system-and-layers.md), subtracts the mean and divides by the standard deviation. Zhang and Sennrich found that the re-centering contributes little: dividing by the **root mean square** alone, $\operatorname{rms}(x) = \sqrt{\operatorname{mean}(x^2) + \epsilon}$, works as well and is cheaper.

After the division (with $\epsilon = 0$) the mean of the squares is exactly 1, and scaling $x$ by any $c > 0$ changes nothing. The sublayer sees the direction of $x$, not its length. Every token is normalized by its own statistic, over the last axis only.

## The gain and Gemma's offset

A unit-RMS vector would force every feature to the same scale, so a learned per-feature gain $w$ restores what the model needs:

$$y = \frac{x}{\operatorname{rms}(x)} \odot w.$$

Llama initializes $w$ to ones. Gemma stores $w - 1$ instead (initialized to zeros) and computes $\frac{x}{\operatorname{rms}(x)} \odot (1 + w)$; weight decay then pulls the gain toward 1, not toward 0. The contract covers both conventions with a single `offset` parameter:

$$y = \frac{x}{\operatorname{rms}(x)} \odot (o + w).$$

## The backward pass

You do not need to write a backward for RMSNorm. Every step is an op from [The op library and gradcheck_all](../06-numerics-and-autograd/03-op-library-and-gradcheck.md) ($x \cdot x$, `F.mean`, a power of $-\frac12$, two products), so the autograd engine produces the gradient. [Matrix differentials, the trace trick, and closed-form VJPs](../../mathematics/14-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md) derives it by hand, and the result has a clean reading: the gradient with respect to $x$ removes the component of the upstream gradient along $x$ (moving along $x$ only rescales it, which the norm undoes) and divides by the RMS. The gain receives $\frac{x}{\operatorname{rms}(x)} \odot g$ summed over tokens.

## Pre-LN

With the norm defined, the remaining question is where it goes. Post-LN (2017) computes $\operatorname{norm}(x + F(x))$: the residual stream itself is normalized after every block, so the gradient that reaches early layers passes through every norm on the way. Pre-LN computes

$$x + F(\operatorname{norm}(x)).$$

The stream $x$ is never normalized; only the branch input is. The derivative of the step with respect to $x$ is $I + \frac{\partial F(\operatorname{norm}(x))}{\partial x}$, so the gradient always has a clean identity path from the loss to the embedding. Xiong et al. show this is why Pre-LN trains at full learning rate without warmup. The price is that the stream grows with depth, so a Pre-LN model applies one more RMSNorm after the last block, before the lm_head, as you will see in [The Llama-Family Model: Config, Weight Loading, and Download](09-llama-family-model.md).

## Working it by hand

Take $d = 4$, $\epsilon = 0$, and $x = (3, 4, 0, 0)$.

1. The squares are $9, 16, 0, 0$, with mean $25 / 4 = 6.25$.
2. So $\operatorname{rms}(x) = \sqrt{6.25} = 2.5$.
3. Dividing gives $x / \operatorname{rms}(x) = (1.2, 1.6, 0, 0)$. As a check, the mean of the squares is $(1.44 + 2.56) / 4 = 1$.
4. With gain $w = (1, 0.5, 2, 2)$, the output is $y = (1.2, 0.8, 0, 0)$.

LayerNorm on the same $x$ would first subtract the mean 1.75 and give a different vector.

## The interface

The contract has one class and one function:

```python
class RMSNorm(Module):
    def __init__(self, d: int, eps: float = 1e-6, offset: float = 0.0) -> None: ...   # weight [d]
    def forward(self, x: Tensor) -> Tensor: ...                                      # [..., d] -> [..., d]
def pre_norm_residual(x: Tensor, norm: Module, sublayer) -> Tensor: ...              # x + sublayer(norm(x))
```

`RMSNorm` owns one parameter named `weight`, float32, initialized to ones or zeros depending on the offset; that name is the safetensors key of every norm in a Hugging Face checkpoint. `forward` normalizes over the last axis and leaves every other axis alone. `pre_norm_residual` has no parameters of its own; it wires a norm and a sublayer into the residual step.

## Building it step by step

Start from what the sublayer needs. It should see a vector on a fixed scale whatever the depth, and dividing by a per-vector size statistic does that; the root mean square is the cheapest statistic that makes the output scale-invariant. The statistic must be over the last axis only, one per token, so tokens never influence each other's normalization.

A zero vector has RMS 0, so the denominator needs an $\epsilon$. Placing it inside the root, $\sqrt{\operatorname{mean}(x^2) + \epsilon}$, is the convention every Llama-family checkpoint was trained with; placing it outside changes the output on small activations.

A learned per-feature gain restores scale where the model needs it. Storing $w$ or $w - 1$ is a checkpoint convention, so `offset` reproduces both, and the initialization follows from it: ones for Llama, zeros for Gemma, so that a fresh norm is the plain normalization in both cases.

The backward needs no hand derivation as long as the forward is composed from differentiable ops (`x * x`, `F.mean`, `** -0.5`, products). The one trap is that the gain must be used as a Tensor so it receives its gradient. Pre-LN is then one line: normalize only the branch input and add the branch to the untouched stream.

Here is the class, annotated:

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

Trace the worked example through it: `x * x` is $(9, 16, 0, 0)$, `ms` is 6.25, `(ms + 0) ** -0.5` is 0.4, `y` is $(1.2, 1.6, 0, 0)$, and the gain $(1, 0.5, 2, 2)$ gives $(1.2, 0.8, 0, 0)$. Notice that the code uses the mean, not the sum: the sum would give $\sqrt{25} = 5$ and outputs $\sqrt d = 2$ times too small.

The residual step is a single expression:

```python
def pre_norm_residual(x, norm, sublayer):
    # The residual stream x is never normalized: only the branch sees norm(x).
    return x + sublayer(norm(x))
```

With a branch that outputs zeros, this returns $x$ exactly and the gradient with respect to $x$ is exactly 1: the identity path from the Pre-LN discussion above. The full solution is in [`norm.py`](../../../../course/solutions/python/tinyllm/modern/norm.py).

### Why this design

The cost is $O(d)$ per token: one square, one mean, one rsqrt, two multiplies. LayerNorm needs a second pass for the mean and a bias; RMSNorm drops both, which matters in a fused kernel where memory traffic dominates.

Composing the forward from autograd ops rather than writing a VJP by hand means the correctness of the backward is inherited from the op library, at the cost of a few temporary arrays. The hand derivation in *Mathematics* produces the same gradient for the C and Rust ports. One class with `offset`, rather than separate Llama and Gemma norms, works because the two formulas differ only by a constant added to the gain. And Pre-LN is a function, not a module, because it has no parameters; the norm and the sublayer own theirs.

The tradeoff you accept is that Pre-LN lets the stream grow with depth, so models need a final norm before the head.

## What can go wrong

A correct solution has to survive a specific set of checks. It must reproduce the hand-worked vector with and without the gain, so everyone agrees on the formula. Its output and gradients must match transformers' `LlamaRMSNorm`, which is what SmolLM2's norms depend on, and with `offset = 1` it must match `GemmaRMSNorm` for Gemma-style checkpoints. Inputs of size 1e-3, where eps dominates, test that eps is inside the root, which matters for the tiny activations of deep layers. Float64 central differences check the gradients of $x$ and the gain at both offsets, since everything before the norm learns through it. Property tests check that the output has unit RMS and is scale-invariant, so the sublayer sees directions, and that each vector is normalized on its own, with per-token statistics over the last axis and no leakage across positions. A constant vector must stay constant, because RMSNorm is not LayerNorm. There must be exactly one parameter named `weight`, float32, initialized to ones or zeros. For the residual step, a zero branch must give exactly $x$ with gradient 1 (deep stacks depend on it), and a bounded branch must be added as $x + F(\operatorname{norm}(x))$, as the Llama block requires. Finally, bad arguments (a wrong width, zero width, negative eps) must fail loudly so wiring bugs surface early. A model test suite whose oracle is the formula in numpy float64, `x / sqrt(mean(x**2, -1) + eps) * (offset + w)`, is in [`test_norm.py`](../../../../course/solutions/learner-tests/L7.1/python/tests/l7-1-norm/test_norm.py).

> **Warning:** The most common bug is eps outside the root, $x / (\sqrt{\operatorname{mean}(x^2)} + \epsilon)$. It passes every test on ordinary inputs and then makes small activations come out several percent off. Write `(ms + self.eps) ** -0.5`.

The other mistakes each have a recognizable symptom, summarized here:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| eps outside the root | small activations come out several percent off | `(ms + self.eps) ** -0.5` |
| subtracting the mean (LayerNorm statistics) | a constant vector becomes zeros | no mean is computed; only `mean(x * x)` |
| ignoring Gemma's offset | a fresh Gemma norm outputs zeros | `gain = self.weight + self.offset` when `offset != 0` |
| post-LN inside `pre_norm_residual` | the stream is normalized; the identity path is gone | returns `x + sublayer(norm(x))`, never `norm(x + ...)` |
| statistic over the wrong axis | tokens are normalized by each other | `axis=-1, keepdims=True` |
| gain initialized the wrong way | a fresh Llama norm outputs zeros | ones when `offset == 0`, zeros otherwise |
| sum of squares instead of the mean | outputs $\sqrt{d}$ times too small | `F.mean`, not `F.sum` |
| normalizing the stream, then adding the branch | the identity path carries $\operatorname{norm}(x)$ | the stream term is the raw `x` |
| the gain as a constant (`.data`) | the gain never trains | the gain is the parameter Tensor itself |

## Where this leads

The norm you just wrote appears all over the rest of the book. In [Multi-head latent attention](06-multi-head-latent-attention.md), `q_a_layernorm` and `kv_a_layernorm` normalize MLA's query and key-value latents. In [The Llama-Family Model: Config, Weight Loading, and Download](09-llama-family-model.md), it becomes the `input_layernorm`, `post_attention_layernorm`, and final `norm` of every Llama model, where SmolLM2-135M's logits must match Hugging Face. Later, the C kernel `tl_rmsnorm_f32` in [Elementwise kernels in C](../22-kernels/08-elementwise-kernels.md) must match this implementation to float32 tolerance, and the Rust forward in [Candle model runner](../23-model-serving/02-model-runner-and-sampler.md) calls that C norm.

### Further reading

- Zhang and Sennrich, "Root Mean Square Layer Normalization" (2019).
- Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020), sections 3 and 4.
- Hugging Face `LlamaRMSNorm` in `transformers/models/llama/modeling_llama.py`: upcasts bf16 input to float32 for the statistic and casts back before the gain.
- Fused RMSNorm kernels (Apex, Liger, vLLM): one pass over memory, fused with the residual add. See vLLM `csrc/layernorm_kernels.cu` (`fused_add_rms_norm`).
- DeepNorm, sandwich norm, and QK-norm: scaled residuals for 1000-layer models and extra norms on queries and keys (OLMo 2, Qwen3). See Wang et al. 2022, "DeepNet", and the OLMo 2 report.

## Key terms

- **RMSNorm**: divides a vector by its root mean square plus eps (inside the root) and multiplies by a learned gain; no mean subtraction, no bias.
- **Gain offset**: a constant added to the stored gain; 1 reproduces Gemma's convention of storing $w - 1$.
- **Pre-LN**: the residual step $x + F(\operatorname{norm}(x))$, which normalizes only the branch input.
- **Post-LN**: the 2017 step $\operatorname{norm}(x + F(x))$, which normalizes the residual stream itself.
- **Identity path**: the $I$ term in the Pre-LN Jacobian that carries the gradient straight from the loss to the embedding.

## Summary

- RMSNorm divides each vector by its root mean square and multiplies by a learned gain; it subtracts no mean and has no bias.
- The eps goes inside the square root, and the difference shows on small activations.
- Gemma stores the gain minus one: `offset = 1` with a zeros init reproduces it.
- Pre-LN normalizes only the branch input, so the residual stream is an identity path in value and in gradient; the cost is a final norm before the head.
- Composing the forward from autograd ops gives the backward for free, as long as the gain is used as a Tensor.

## Review questions

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
6. What goes wrong if you use `F.sum` instead of `F.mean` for the statistic?
   <details><summary>Answer</summary>The statistic becomes $\sum_i x_i^2$ instead of its mean, so the output is $\sqrt d$ times too small. On the worked example the denominator would be $\sqrt{25} = 5$ instead of 2.5, halving every output.</details>
