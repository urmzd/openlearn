<!-- ol:module M07.3 -->
# Variance propagation and initialization

## Overview

| | |
|---|---|
| **Module** | `M07.3` · build · Python · Pass 2 · 2 to 3 h |
| **You build** | `python/tinyllm/nn/init.py`: `fans`, `calculate_gain`, `xavier_uniform`, `xavier_normal`, `kaiming_normal`, `normal_init`, `scaled_residual_std` |
| **Contract** | [`course/contracts/py/tinyllm/nn/init.pyi`](../../course/contracts/py/tinyllm/nn/init.pyi) · draw order: [`spec/pcg32.md`](../../course/contracts/spec/pcg32.md) |
| **Tests** | `course/tests/M07.3/test_init.py` (what they check: section 4); torch's gain table and fan rule in `course/fixtures/M07.3/init_torch.json` |
| **Needs** | `M07.0` its `normal(rng, n)` draws every normal weight (or `--ref-deps`). Reading: `M01.3` (activations), `M06.3` (PCG32), `S-M02` and `S-M04` (the Gaussian integrals) |
| **Used by** | `L0.4` initializes every `Linear` and `Embedding` with it · later `L3.2` (LSTM), `L7.9` (the modern decoder), `C1` · later: `L3.3`, `L3.6`, `L5.4`, `L5.5`, `L6.1`, `L6.2`, `L6.3` |
| **Milestone** | `MS-P2` (Pass 2 closes with every math module it teaches passing) |
| **Optional depth** | Glorot and Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010), section 4.2; He et al., "Delving Deep into Rectifiers" (2015), section 2.2; Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019), section 2.3 |

## Key Takeaways

- For $z = \sum_{j=1}^{n} w_j x_j$ with independent zero-mean weights, $\mathrm{Var}(z) = n\, \mathrm{Var}(w)\, \mathbb{E}[x^2]$: the variance of a layer's output is set by its **fan-in** and the weights' variance (`test_hand_example_linear_4x3`).
- **Xavier** picks $\mathrm{Var}(w) = 2/(n_{\text{in}} + n_{\text{out}})$ for layers that are linear near 0; **Kaiming** picks $2/n_{\text{in}}$ for ReLU, which zeroes half its inputs (`test_empirical_variance_matches_the_formula`).
- With Kaiming the signal survives 20 ReLU layers; with Xavier it shrinks by about $2^{19}$ (`test_relu_signal_survives_20_layers`).
- A **gain** folds the activation's effect into one factor; fans of a convolution or a Linear weight follow torch's `(out, in, *kernel)` rule (`test_fans_and_gains_match_torch`).
- Weights are drawn in row-major order, one draw per element, from your PCG32, so a seed fixes the model in every language (`test_normal_inits_draw_spec_normals_in_c_order`).

## How to work this chapter

```bash
ol start M07.3              # stubs python/tinyllm/nn/init.py, contract alongside
ol tests M07.3              # read the test catalog first: rung R0, you write no tests here
ol check M07.3              # exit code is the verdict
ol check M07.3 --ref-deps   # only if your M07.0 is not passing yet
ol diff  M07.3              # after passing: your code against the reference
```

---

## 1. Why now

`L0.4` builds layers next, and a layer needs starting weights. The obvious choices fail. All zeros makes every unit in a layer compute the same thing and receive the same gradient forever. Standard normal weights make a layer with 512 inputs multiply the size of its input by about $\sqrt{512} \approx 23$, so after a few layers the activations overflow float32 or saturate every tanh, and the gradients are 0 or NaN. Weights a little too small do the opposite: the signal shrinks geometrically and the deep layers learn nothing. Your `MS-L0` milestone trains an MLP on digits from your own initialization, and `C1` trains a 10M-parameter Transformer the same way. This module derives the one number that decides which of these happens, the variance of the weights, from expectation and variance (`M07.0`), and builds the initializers every later model calls.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\mathbb{E}[X]$, $\mathrm{Var}(X)$ | expectation and variance of a random variable, $\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}X)^2]$ (`M07.0`) | scalars |
| $W \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}$ | a Linear weight, one row per output (torch's layout) | `float32[out, in]` |
| $x \in \mathbb{R}^{n_{\text{in}}}$ | the layer's input; $z = W x$ its pre-activation | vectors |
| $n_{\text{in}}, n_{\text{out}}$ | fan-in and fan-out: inputs feeding one output, outputs fed by one input | `int` |
| $\sigma^2 = \mathrm{Var}(w)$ | the variance every weight is drawn with | `float` |
| $f$ | the activation, $h = f(z)$ | function |
| $g$ | the gain of $f$: the factor that corrects the variance for it | `float` |
| $\mathcal{U}(-a, a)$, $\mathcal{N}(0, s^2)$ | uniform on $[-a, a)$; normal with mean 0 and standard deviation $s$ | distributions |
| $L$ | the number of Transformer blocks | `int` |

### 2.1 Expectation and variance of sums and products

Two rules from `M07.0` do all the work. Expectation is **linear**: $\mathbb{E}[aX + bY] = a\,\mathbb{E}X + b\,\mathbb{E}Y$ for any random variables. Variance adds for **independent** ones: $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$, because the cross term $2\,\mathbb{E}[(X - \mathbb{E}X)(Y - \mathbb{E}Y)]$ is a product of two zero means. For a product of independent $w$ and $x$ with $\mathbb{E}[w] = 0$:

$$\mathrm{Var}(w x) = \mathbb{E}[w^2 x^2] - (\mathbb{E}[w]\,\mathbb{E}[x])^2 = \mathbb{E}[w^2]\,\mathbb{E}[x^2] = \sigma^2\, \mathbb{E}[x^2] .$$

Note $\mathbb{E}[x^2]$, not $\mathrm{Var}(x)$: the input's mean counts too (after a ReLU, $x \ge 0$ has a positive mean).

### 2.2 Choosing the weight variance

One output of a layer is $z_i = \sum_{j=1}^{n_{\text{in}}} W_{ij} x_j$, a sum of $n_{\text{in}}$ independent products, each with mean 0. By 2.1:

$$\mathrm{Var}(z_i) = n_{\text{in}}\, \sigma^2\, \mathbb{E}[x^2] .$$

To keep the signal the same size from layer to layer, set this equal to the previous layer's value. **Linear activations** ($h = z$, zero mean): $\mathbb{E}[x^2] = \mathrm{Var}(x)$ and the condition is $n_{\text{in}} \sigma^2 = 1$. The backward pass multiplies gradients by $W^\top$, whose rows have $n_{\text{out}}$ entries, so it wants $n_{\text{out}} \sigma^2 = 1$. **Xavier** (Glorot and Bengio) splits the difference:

$$\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}} .$$

**ReLU**, $f(z) = \max(z, 0)$, keeps the positive half of a symmetric $z$ and zeroes the rest, so $\mathbb{E}[f(z)^2] = \frac{1}{2}\mathrm{Var}(z)$. The next layer then sees $\mathbb{E}[x^2] = \frac12 \mathrm{Var}(z)$, and preserving the variance needs twice as much weight variance. **Kaiming** (He et al.):

$$\sigma^2 = \frac{2}{n_{\text{in}}} \quad (\text{forward, "fan\_in"}), \qquad \sigma^2 = \frac{2}{n_{\text{out}}} \quad (\text{backward, "fan\_out"}).$$

Getting the factor wrong compounds: a variance too small by 2 per layer is $2^{-19}$ after 20 layers, and too large by 2 is $2^{19}$.

### 2.3 Fans and gains

A **gain** $g$ writes the activation's correction as a factor on the standard deviation: Kaiming is $\sigma = g / \sqrt{n}$ with $g = \sqrt{2}$ for ReLU, and Xavier takes an optional gain, $\sigma = g\sqrt{2/(n_{\text{in}} + n_{\text{out}})}$. torch's table, which you match: $g = 1$ for linear, convolutions, and sigmoid; $\sqrt{2}$ for ReLU; $\sqrt{2/(1 + s^2)}$ for leaky ReLU with negative slope $s$ (default $0.01$), because it keeps $s^2$ of the negative half's second moment; $3/4$ for SELU; and $5/3$ for tanh. The tanh value is empirical: tanh has slope 1 at 0 but shrinks larger inputs, and with $g = 5/3$ the mean square of a deep tanh stack settles near 0.42 instead of draining toward 0 (about 0.02 after 20 layers with $g = 1$).

**Fans** come from the weight's shape `(out, in, *kernel)`: $n_{\text{in}} = \mathit{in} \cdot r$, $n_{\text{out}} = \mathit{out} \cdot r$, with $r$ the product of the kernel dimensions (1 for a Linear weight). A convolution with 4 input channels and a $3 \times 3$ kernel has $n_{\text{in}} = 36$. Fewer than 2 dimensions have no fan-in and are an error.

### 2.4 Drawing the weights

Two distributions with the same variance work equally well at the start. A **uniform** $\mathcal{U}(-a, a)$ has variance $a^2/3$, so Xavier-uniform uses $a = g\sqrt{6/(n_{\text{in}} + n_{\text{out}})}$; element $k$ is $-a + 2a\,u_k$ with $u_k$ the $k$-th `rng.uniform()`. A **normal** is $s$ times a standard normal from your `M07.0` `normal(rng, n)`. Either way the array is filled in row-major (C) order, one draw per element, and nothing else is drawn, so a model initialized layer after layer from one generator is reproducible, and the Rust port (`L10.1`) can rebuild the same weights from the same seed. Results are float32, the dtype of every parameter.

### 2.5 Residual streams

A Transformer block adds its outputs to a running **residual stream**: $x_{\ell+1} = x_\ell + \mathrm{attn}(x_\ell) + \mathrm{mlp}(\cdot)$, two additions per block, $2L$ in all. If each addition is independent with variance $s^2$, the stream's variance grows to $\mathrm{Var}(x_0) + 2L s^2$. GPT-2 scales the standard deviation of the two output projections in every block to

$$s = \frac{s_{\text{base}}}{\sqrt{2L}}, \qquad \text{so} \qquad 2L\, s^2 = s_{\text{base}}^2$$

whatever the depth (`scaled_residual_std`, with $s_{\text{base}} = 0.02$ in GPT-2 and your `L7.9`).

## 3. Worked example by hand

A Linear layer with 3 inputs and 4 outputs stores $W$ as shape $(4, 3)$: $n_{\text{in}} = 3$, $n_{\text{out}} = 4$.

| Initializer | Formula | Value |
|---|---|---|
| Xavier-uniform bound | $a = \sqrt{6 / (3 + 4)}$ | $\approx 0.92582$ |
| Xavier-normal std | $\sqrt{2 / 7}$ | $\approx 0.53452$ |
| Kaiming-normal std (ReLU, fan_in) | $\sqrt{2} / \sqrt{3} = \sqrt{2/3}$ | $\approx 0.81650$ |
| GPT-2 residual std, $L = 12$ | $0.02 / \sqrt{24}$ | $\approx 0.0040825$ |

**The variance carries through.** Feed the Kaiming layer an input with $\mathbb{E}[x^2] = 1$: $\mathrm{Var}(z) = 3 \cdot \frac{2}{3} \cdot 1 = 2$. ReLU keeps half the second moment: $\mathbb{E}[\mathrm{relu}(z)^2] = 1$, exactly what the next layer received. With Xavier ($\sigma^2 = 2/7$) the same input gives $\mathrm{Var}(z) = 6/7$ and $\mathbb{E}[\mathrm{relu}(z)^2] = 3/7$: the signal loses more than half per layer.

**The draws.** With `PCG32(0)` the first uniform is $u_0 \approx 0.280312$, so Xavier-uniform's first weight is $-a + 2a u_0 = 0.92582 \cdot (2 \cdot 0.280312 - 1) \approx -0.406783$. Kaiming's first weight uses the first pair $u_0, u_1 \approx 0.489224$: $r = \sqrt{-2 \ln(1 - u_0)} \approx 0.81104$, $\cos(2\pi u_1) \approx -0.99771$, so the standard normal is $\approx -0.80918$ and the weight $0.81650 \cdot (-0.80918) \approx -0.66071$. The first test, `test_hand_example_linear_4x3`, checks every one of these against the frozen generator.

## 4. The interface

```python
def fans(shape: tuple[int, ...]) -> tuple[int, int]                    # (fan_in, fan_out)
def calculate_gain(nonlinearity: str, param: float | None = None) -> float
def xavier_uniform(shape, gain: float, rng) -> NDArray                  # float32, U(-a, a)
def xavier_normal(shape, gain: float, rng) -> NDArray                   # float32, N(0, s^2)
def kaiming_normal(shape, fan_mode: Literal["fan_in", "fan_out"], nonlinearity: str, rng) -> NDArray
def normal_init(shape, std: float, rng) -> NDArray                      # any shape, including 1-D
def scaled_residual_std(base_std: float, n_layers: int) -> float        # base / sqrt(2 L)
```

`rng` is a PCG32: your `M06.3` generator in your own code, the frozen `course/tests/_lib/pcg32.py` in the course tests. Uniforms come from `rng.uniform()`, normals from `tinyllm.prob.rv.normal(rng, n)`. Every function raises `ValueError` for a shape without a fan (fewer than 2 dimensions, or a 0 fan), a negative gain or standard deviation, an unknown nonlinearity or fan mode, and `n_layers < 1`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_linear_4x3` | unit | section 3: fans, gains, the residual std, and the first weights of `PCG32(0)` | you and the tests agree on every formula |
| `test_fans_and_gains_match_torch` | golden | torch's gain table and fan rule for Linear and conv shapes | `L0.4` compares layers with torch |
| `test_normal_inits_draw_spec_normals_in_c_order` | unit | each normal initializer is std times the spec normals, row-major, odd sizes too | reproducible weights across languages |
| `test_uniform_uses_one_draw_per_element` | unit | exact values, and the generator advanced by exactly one uniform per element | layers initialized in sequence stay aligned |
| `test_same_seed_same_weights` | property | equal seeds give equal bytes, different seeds differ | replayable runs |
| `test_empirical_variance_matches_the_formula` | statistical | mean 0 and variance within 4 standard errors for all four variants | the formulas of section 2.2, measured |
| `test_relu_signal_survives_20_layers` | property | Kaiming keeps the mean square within a factor 16; Xavier collapses it below $10^{-4}$ | deep MLPs and Transformers train at all |
| `test_tanh_signal_survives_with_xavier` | property | gain $5/3$ keeps a 20-layer tanh stack's mean square in $(0.1, 1)$; gain 1 ends at less than half that | tanh and gated recurrent layers (`L3.2`) |
| `test_scaled_residual_keeps_the_stream_bounded` | property | $2L s^2 = s_{\text{base}}^2$ for several depths | GPT-2 style output projections (`L7.9`) |
| `test_shapes_and_dtype` | unit | float32, C-contiguous, exact shapes, 1-D for `normal_init` | parameters are float32 (`L0.4`) |
| `test_rejects_bad_arguments` | boundary | 1-D fans, unknown names, negative std or depth raise `ValueError` | no silent wrong-scale default |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| reading a `(out, in)` weight as `(in, out)` | fan-in and fan-out swap; non-square layers get the wrong scale | `test_hand_example_linear_4x3`, `test_fans_and_gains_match_torch` (mutant `s01`) |
| ignoring the kernel dimensions | a $3 \times 3$ convolution is initialized 9 times too large in variance | `test_fans_and_gains_match_torch` (mutant `s02`) |
| using $\sqrt{2/(n_{\text{in}} + n_{\text{out}})}$ as the uniform bound | the uniform's variance is $a^2/3$, so this is 3 times too small | `test_empirical_variance_matches_the_formula` (mutant `s03`) |
| ReLU gain 2 (a variance) instead of $\sqrt{2}$ (a standard deviation) | activations double in mean square per layer: $2^{19}$ after 20 | `test_relu_signal_survives_20_layers` (mutant `s05`) |
| GPT-2 residual scale $1/\sqrt{L}$ instead of $1/\sqrt{2L}$ | each block adds two outputs; the stream ends twice as large | `test_scaled_residual_keeps_the_stream_bounded` (mutant `s08`) |
| filling in column-major order | same distribution, different weights: Python and the Rust port disagree for the same seed | `test_normal_inits_draw_spec_normals_in_c_order` (mutant `s09`) |
| making a new generator inside the initializer | every layer gets the same weights, and the seed does nothing | `test_same_seed_same_weights` (mutant `s13`) |

## 6. Where it's used next
| Forward | `L3.3` | Registered call site uses this module. |
| Forward | `L3.6` | Registered call site uses this module. |
| Forward | `L5.4` | Registered call site uses this module. |
| Forward | `L5.5` | Registered call site uses this module. |
| Forward | `L6.1` | Registered call site uses this module. |
| Forward | `L6.2` | Registered call site uses this module. |
| Forward | `L6.3` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.0` | `normal(rng, n)` draws every normal weight; expectation and variance are its definitions |
| Back | `M01.3` | the activations whose second moments set the gains (reading) |
| Back | `M06.3` | the PCG32 generator passed in as `rng` (reading) |
| Back | `S-M02`, `S-M04` | the Gaussian integrals behind $\mathbb{E}[\mathrm{relu}(z)^2] = \mathrm{Var}(z)/2$ (reading) |
| Forward | `L0.4` | `Linear` and `Embedding` initialize their parameters with these functions |
| Forward | `L3.2` | LSTM gate weights |
| Forward | `L7.9` and `C1` | `normal_init(0.02)` plus `scaled_residual_std` for the output projections of a Llama-style decoder |

If you skip this module, `ol check L0.4` stops with `BLOCKED ... needs M07.3`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `xavier_*`, `kaiming_normal` | `torch.nn.init` | in-place initializers, `trunc_normal_`, `orthogonal_` (your `M03.3`), `kaiming_uniform_` with `a = sqrt(5)` (torch's `Linear` default) | `torch/nn/init.py` |
| `scaled_residual_std` | GPT-NeoX, Megatron `scaled_init_method_normal` | the same $1/\sqrt{2L}$ rule, applied by parameter name | `megatron/core/utils.py` |
| a fixed $0.02$ base std | muP (maximal update parametrization) | width-dependent init and learning rates so hyperparameters transfer from small to large models | Yang et al., "Tensor Programs V" (2022) |
| variance analysis at initialization | signal propagation theory | mean-field analysis of depth and the edge of chaos; Fixup and T-Fixup train without normalization | Schoenholz et al., "Deep Information Propagation" (2017) |
