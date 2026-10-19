# Variance propagation and initialization

## Overview

| | |
|---|---|
| **Module** | `M07.3` · build · Python · Pass 2 · 1 to 1.5 h reading |
| **Solution** | [`course/solutions/python/tinyllm/nn/init.py`](../../../../course/solutions/python/tinyllm/nn/init.py): `fans`, `calculate_gain`, `xavier_uniform`, `xavier_normal`, `kaiming_normal`, `normal_init`, `scaled_residual_std` |
| **Contract** | [`course/contracts/py/tinyllm/nn/init.pyi`](../../../../course/contracts/py/tinyllm/nn/init.pyi) · draw order: [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md) |
| **Needs** | `M07.0` its `normal(rng, n)` draws every normal weight ([`rv.py`](../../../../course/solutions/python/tinyllm/prob/rv.py)). Reading: `M01.3` (activations), `M06.3` (PCG32), `S-M02` and `S-M04` (the Gaussian integrals) |
| **Used by** | `L0.4` initializes every `Linear` and `Embedding` with it · later `L3.2` (LSTM), `L7.9` (the modern decoder), `C1` · later: `L3.3`, `L3.6`, `L5.4`, `L5.5`, `L6.1`, `L6.2`, `L6.3`, `L13.1`, `L13.3`; one of the Pass 2 math foundations |
| **Optional depth** | Glorot and Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010), section 4.2; He et al., "Delving Deep into Rectifiers" (2015), section 2.2; Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019), section 2.3 |

## Key Takeaways

- For $z = \sum_{j=1}^{n} w_j x_j$ with independent zero-mean weights, $\mathrm{Var}(z) = n\, \mathrm{Var}(w)\, \mathbb{E}[x^2]$: the variance of a layer's output is set by its **fan-in** and the weights' variance.
- **Xavier** picks $\mathrm{Var}(w) = 2/(n_{\text{in}} + n_{\text{out}})$ for layers that are linear near 0; **Kaiming** picks $2/n_{\text{in}}$ for ReLU, which zeroes half its inputs.
- With Kaiming the signal survives 20 ReLU layers; with Xavier it shrinks by about $2^{19}$.
- A **gain** folds the activation's effect into one factor; fans of a convolution or a Linear weight follow torch's `(out, in, *kernel)` rule.
- Weights are drawn in row-major order, one draw per element, from the PCG32, so a seed fixes the model in every language.

## How to read this chapter

Sections 1 and 2 derive the variance-propagation rule and the Xavier, Kaiming, and residual scalings from first principles, and section 3 works them by hand. Section 4 fixes the interface, and section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

`L0.4` builds layers next, and a layer needs starting weights. The obvious choices fail. All zeros makes every unit in a layer compute the same thing and receive the same gradient forever. Standard normal weights make a layer with 512 inputs multiply the size of its input by about $\sqrt{512} \approx 23$, so after a few layers the activations overflow float32 or saturate every tanh, and the gradients are 0 or NaN. Weights a little too small do the opposite: the signal shrinks geometrically and the deep layers learn nothing. The first MLP (`L0.4`) trains on digits from this initialization, and `C1` trains a 10M-parameter Transformer the same way. This module derives the one number that decides which of these happens, the variance of the weights, from expectation and variance (`M07.0`), and builds the initializers every later model calls.

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

A **gain** $g$ writes the activation's correction as a factor on the standard deviation: Kaiming is $\sigma = g / \sqrt{n}$ with $g = \sqrt{2}$ for ReLU, and Xavier takes an optional gain, $\sigma = g\sqrt{2/(n_{\text{in}} + n_{\text{out}})}$. torch's table, which the solution matches: $g = 1$ for linear, convolutions, and sigmoid; $\sqrt{2}$ for ReLU; $\sqrt{2/(1 + s^2)}$ for leaky ReLU with negative slope $s$ (default $0.01$), because it keeps $s^2$ of the negative half's second moment; $3/4$ for SELU; and $5/3$ for tanh. The tanh value is empirical: tanh has slope 1 at 0 but shrinks larger inputs, and with $g = 5/3$ the mean square of a deep tanh stack settles near 0.42 instead of draining toward 0 (about 0.02 after 20 layers with $g = 1$).

**Fans** come from the weight's shape `(out, in, *kernel)`: $n_{\text{in}} = \mathit{in} \cdot r$, $n_{\text{out}} = \mathit{out} \cdot r$, with $r$ the product of the kernel dimensions (1 for a Linear weight). A convolution with 4 input channels and a $3 \times 3$ kernel has $n_{\text{in}} = 36$. Fewer than 2 dimensions have no fan-in and are an error.

### 2.4 Drawing the weights

Two distributions with the same variance work equally well at the start. A **uniform** $\mathcal{U}(-a, a)$ has variance $a^2/3$, so Xavier-uniform uses $a = g\sqrt{6/(n_{\text{in}} + n_{\text{out}})}$; element $k$ is $-a + 2a\,u_k$ with $u_k$ the $k$-th `rng.uniform()`. A **normal** is $s$ times a standard normal from the `M07.0` `normal(rng, n)`. Either way the array is filled in row-major (C) order, one draw per element, and nothing else is drawn, so a model initialized layer after layer from one generator is reproducible, and the Rust port (`L10.1`) can rebuild the same weights from the same seed. Results are float32, the dtype of every parameter.

### 2.5 Residual streams

A Transformer block adds its outputs to a running **residual stream**: $x_{\ell+1} = x_\ell + \mathrm{attn}(x_\ell) + \mathrm{mlp}(\cdot)$, two additions per block, $2L$ in all. If each addition is independent with variance $s^2$, the stream's variance grows to $\mathrm{Var}(x_0) + 2L s^2$. GPT-2 scales the standard deviation of the two output projections in every block to

$$s = \frac{s_{\text{base}}}{\sqrt{2L}}, \qquad \text{so} \qquad 2L\, s^2 = s_{\text{base}}^2$$

whatever the depth (`scaled_residual_std`, with $s_{\text{base}} = 0.02$ in GPT-2 and `L7.9`).

## 3. Worked example by hand

A Linear layer with 3 inputs and 4 outputs stores $W$ as shape $(4, 3)$: $n_{\text{in}} = 3$, $n_{\text{out}} = 4$.

| Initializer | Formula | Value |
|---|---|---|
| Xavier-uniform bound | $a = \sqrt{6 / (3 + 4)}$ | $\approx 0.92582$ |
| Xavier-normal std | $\sqrt{2 / 7}$ | $\approx 0.53452$ |
| Kaiming-normal std (ReLU, fan_in) | $\sqrt{2} / \sqrt{3} = \sqrt{2/3}$ | $\approx 0.81650$ |
| GPT-2 residual std, $L = 12$ | $0.02 / \sqrt{24}$ | $\approx 0.0040825$ |

**The variance carries through.** Feed the Kaiming layer an input with $\mathbb{E}[x^2] = 1$: $\mathrm{Var}(z) = 3 \cdot \frac{2}{3} \cdot 1 = 2$. ReLU keeps half the second moment: $\mathbb{E}[\mathrm{relu}(z)^2] = 1$, exactly what the next layer received. With Xavier ($\sigma^2 = 2/7$) the same input gives $\mathrm{Var}(z) = 6/7$ and $\mathbb{E}[\mathrm{relu}(z)^2] = 3/7$: the signal loses more than half per layer.

**The draws.** With `PCG32(0)` the first uniform is $u_0 \approx 0.280312$, so Xavier-uniform's first weight is $-a + 2a u_0 = 0.92582 \cdot (2 \cdot 0.280312 - 1) \approx -0.406783$. Kaiming's first weight uses the first pair $u_0, u_1 \approx 0.489224$: $r = \sqrt{-2 \ln(1 - u_0)} \approx 0.81104$, $\cos(2\pi u_1) \approx -0.99771$, so the standard normal is $\approx -0.80918$ and the weight $0.81650 \cdot (-0.80918) \approx -0.66071$. These are the first case in section 4.

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

`rng` is a PCG32 (`M06.3`, [`rng.py`](../../../../course/solutions/python/tinyllm/num/rng.py)). Uniforms come from `rng.uniform()`, normals from `tinyllm.prob.rv.normal(rng, n)`. Every function raises `ValueError` for a shape without a fan (fewer than 2 dimensions, or a 0 fan), a negative gain or standard deviation, an unknown nonlinearity or fan mode, and `n_layers < 1`.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the $4 \times 3$ layer by hand | unit | section 3: fans, gains, the residual std, and the first weights of `PCG32(0)` | everyone agrees on every formula |
| torch's fans and gains | golden | torch's gain table and fan rule for Linear and conv shapes | `L0.4` compares layers with torch |
| spec normals in C order | unit | each normal initializer is std times the spec normals, row-major, odd sizes too | reproducible weights across languages |
| one uniform per element | unit | exact values, and the generator advanced by exactly one uniform per element | layers initialized in sequence stay aligned |
| same seed, same weights | property | equal seeds give equal bytes, different seeds differ | replayable runs |
| measured variance | statistical | mean 0 and variance within 4 standard errors for all four variants | the formulas of section 2.2, measured |
| 20 ReLU layers | property | Kaiming keeps the mean square within a factor 16; Xavier collapses it below $10^{-4}$ | deep MLPs and Transformers train at all |
| 20 tanh layers | property | gain $5/3$ keeps the mean square in $(0.1, 1)$; gain 1 ends at less than half that | tanh and gated recurrent layers (`L3.2`) |
| the residual stream | property | $2L s^2 = s_{\text{base}}^2$ for several depths | GPT-2 style output projections (`L7.9`) |
| shapes and dtype | unit | float32, C-contiguous, exact shapes, 1-D for `normal_init` | parameters are float32 (`L0.4`) |
| bad arguments | boundary | 1-D fans, unknown names, negative std or depth raise `ValueError` | no silent wrong-scale default |

## 5. The solution, step by step

### From first principles

1. **Everything is set by the fans.** Section 2.2: the variance a layer needs depends only on $n_{\text{in}}$ and $n_{\text{out}}$. For a weight of shape `(out, in, *kernel)` those are `in * r` and `out * r`, with $r$ the product of the kernel dimensions (`math.prod(())` is 1, so a Linear weight needs no special case). Fewer than 2 dimensions, or a fan of 0, leaves the variance undefined: raise.
2. **The activation enters as one gain on the standard deviation.** Section 2.3: a lookup table returning $g$ (not $g^2$), so every initializer is "some $g$ over the square root of some fan".
3. **Xavier-uniform needs the uniform's own variance.** $\mathrm{Var}\,\mathcal{U}(-a, a) = a^2/3$, so matching $\sigma^2 = g^2 \cdot 2/(n_{\text{in}} + n_{\text{out}})$ forces $a = g\sqrt{6/(n_{\text{in}} + n_{\text{out}})}$. Each element is the affine map $-a + 2au$ of one uniform.
4. **Every normal initializer is "std times spec normals".** Xavier-normal, Kaiming-normal, and `normal_init` differ only in the standard deviation; one helper draws `prod(shape)` normals with `M07.0`'s `normal`, scales, reshapes in C order, and casts to float32.
5. **The generator is passed in, never created.** One `rng` threads through every layer, so the seed fixes the whole model and layers initialized in sequence get different, reproducible weights.
6. **The residual scale counts two additions per block.** $s = s_{\text{base}}/\sqrt{2L}$, so $2Ls^2 = s_{\text{base}}^2$ for any depth.

### The code, annotated

```python
def fans(shape: tuple[int, ...]) -> tuple[int, int]:
    s = _shape(shape)                         # ints, none negative
    if len(s) < 2:
        raise ValueError(f"fan_in and fan_out need at least 2 dimensions, got shape {s}")
    receptive = math.prod(s[2:])              # kernel size; 1 for a Linear weight
    # torch's layout is (out, in, *kernel): dim 1 feeds each output, dim 0 is fed by each input.
    fan_in, fan_out = s[1] * receptive, s[0] * receptive
    if fan_in == 0 or fan_out == 0:
        raise ValueError(f"shape {s} has a fan of 0")   # sqrt(2 / 0) is no scale at all
    return fan_in, fan_out


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    if nonlinearity in _GAIN_ONE:             # linear, the conv variants, sigmoid
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3.0                      # empirical: keeps a deep tanh stack's mean square stable
    if nonlinearity == "relu":
        return math.sqrt(2.0)                 # a standard deviation: ReLU halves the second moment
    if nonlinearity == "leaky_relu":
        if param is None:
            slope = 0.01
        elif isinstance(param, Real) and not isinstance(param, bool):
            slope = float(param)
        else:
            raise ValueError(f"leaky_relu slope must be a number, got {param!r}")
        return math.sqrt(2.0 / (1.0 + slope**2))   # the negative half keeps slope^2 of its moment
    if nonlinearity == "selu":
        return 3.0 / 4.0
    raise ValueError(f"unsupported nonlinearity {nonlinearity!r}")   # no silent default of 1
```

For the $(4, 3)$ weight of section 3, `receptive = 1`, `fan_in = 3`, `fan_out = 4`. A conv weight $(8, 4, 3, 3)$ gets `receptive = 9` and fans $(36, 72)$.

```python
def _normal(shape: tuple[int, ...], std: float, rng: Any) -> NDArray:
    """std * standard normals in C order, as float32."""
    n = math.prod(shape)
    # Draw in float64 (the spec's normals), scale, then fill row-major and cast once.
    return (std * normal(rng, n)).reshape(shape).astype(np.float32)


def xavier_uniform(shape: tuple[int, ...], gain: float, rng: Any) -> NDArray:
    if not gain >= 0.0:                       # also rejects NaN
        raise ValueError(f"gain must be >= 0, got {gain}")
    s = _shape(shape)
    fan_in, fan_out = fans(s)
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))   # Var U(-a, a) = a^2 / 3 = 2 g^2 / (in + out)
    u = np.array([rng.uniform() for _ in range(math.prod(s))], dtype=np.float64)   # one draw per element
    return (-a + 2.0 * a * u).reshape(s).astype(np.float32)


def xavier_normal(shape: tuple[int, ...], gain: float, rng: Any) -> NDArray:
    # (gain check as above)
    s = _shape(shape)
    fan_in, fan_out = fans(s)
    return _normal(s, gain * math.sqrt(2.0 / (fan_in + fan_out)), rng)


def kaiming_normal(shape, fan_mode, nonlinearity, rng) -> NDArray:
    if fan_mode not in ("fan_in", "fan_out"):
        raise ValueError(f"fan_mode must be 'fan_in' or 'fan_out', got {fan_mode!r}")
    s = _shape(shape)
    fan_in, fan_out = fans(s)
    fan = fan_in if fan_mode == "fan_in" else fan_out   # forward or backward variance preserved
    return _normal(s, calculate_gain(nonlinearity) / math.sqrt(fan), rng)


def normal_init(shape: tuple[int, ...], std: float, rng: Any) -> NDArray:
    if not std >= 0.0:
        raise ValueError(f"std must be >= 0, got {std}")
    return _normal(_shape(shape), std, rng)   # no fans: any shape, including 1-D biases


def scaled_residual_std(base_std: float, n_layers: int) -> float:
    # (argument checks elided)
    return base_std / math.sqrt(2.0 * n_layers)   # two residual additions per block
```

On section 3: Xavier-uniform's first element is $-0.92582 + 2 \cdot 0.92582 \cdot 0.280312 \approx -0.406783$, and Kaiming's is $\sqrt{2/3} \cdot (-0.80918) \approx -0.66071$, where $-0.80918$ is the first spec normal of `PCG32(0)`. Because `reshape` is C order, element $k$ of the flat draw becomes row $k \mathbin{//} n_{\text{in}}$, column $k \bmod n_{\text{in}}$, the same layout the Rust port reads.

Full solution: [`init.py`](../../../../course/solutions/python/tinyllm/nn/init.py).

### Why this solution

- **Cost.** Each initializer is $O(\prod \text{shape})$ draws and arithmetic; the uniform version makes exactly one `rng.uniform()` per element and the normal versions $2\lceil n/2 \rceil$ (`M07.0`). Memory is one float64 temporary plus the float32 result.
- **Draw in float64, cast once.** Scaling float64 normals and casting at the end gives the correctly rounded float32 of the exact value, which every port can reproduce; drawing or scaling in float32 would add a port-dependent rounding.
- **One generator, passed in.** A global or per-call generator makes the seed meaningless or gives every layer the same weights. Threading `rng` through keeps the whole model a function of one seed.
- **Alternatives.** Orthogonal initialization (`M03.3`) preserves norms exactly rather than in expectation and helps very deep linear stacks, at the cost of a QR per layer. Truncated normals avoid rare large weights. LSUV and data-dependent schemes measure variance on a batch. The analytic Xavier and Kaiming scales need no data and are what torch uses, which `L0.4` compares against.
- **Tradeoffs accepted.** The derivation assumes independent zero-mean weights and inputs, so it is exact only at initialization. Training breaks independence, and normalization layers (`L7.1`) then keep the scale.

### Explain it back

**1. Why is ReLU's gain $\sqrt2$ and not 2?**

<details><summary>Answer</summary>

The gain multiplies the standard deviation: Kaiming is $\sigma = g/\sqrt{n_{\text{in}}}$. ReLU halves the second moment, so the variance must double: $\sigma^2 = 2/n_{\text{in}}$, which means $g = \sqrt2$. A gain of 2 would quadruple the variance and grow the mean square by $2\times$ per layer, $2^{19}$ after 20.

</details>

**2. Why is the Xavier-uniform bound $\sqrt{6/(n_{\text{in}} + n_{\text{out}})}$ and not $\sqrt{2/(n_{\text{in}} + n_{\text{out}})}$?**

<details><summary>Answer</summary>

The target is a variance, $2/(n_{\text{in}} + n_{\text{out}})$, and a uniform on $[-a, a)$ has variance $a^2/3$, not $a^2$. Solving $a^2/3 = 2/(n_{\text{in}} + n_{\text{out}})$ gives the factor 6. Using 2 would make the variance three times too small.

</details>

**3. Why does `fans` read `shape[1]` as the fan-in?**

<details><summary>Answer</summary>

torch stores a Linear weight as `(out, in)`: row $i$ holds the $n_{\text{in}}$ weights that feed output $i$, so the number of inputs summed into one output is the length of a row, `shape[1]`. Swapping the two only matters for non-square layers, which is why the $(4, 3)$ example has different fans.

</details>

**4. What breaks if an initializer creates its own `PCG32(seed)` instead of taking `rng`?**

<details><summary>Answer</summary>

Every layer of the same shape gets identical weights (two attention heads would compute the same thing and receive the same gradient), and changing the model's seed does nothing. Passing one generator advances it by a known number of draws per layer, so the model is reproducible and different layers differ.

</details>

**5. Why $1/\sqrt{2L}$ and not $1/\sqrt{L}$ for the residual projections?**

<details><summary>Answer</summary>

Each block adds two outputs to the stream, attention and MLP, so there are $2L$ independent additions. Their variances add to $2Ls^2$; setting $s = s_{\text{base}}/\sqrt{2L}$ makes the total $s_{\text{base}}^2$ at any depth. With $1/\sqrt{L}$ the stream's added variance would be twice the target.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| reading a `(out, in)` weight as `(in, out)` | fan-in and fan-out swap; non-square layers get the wrong scale | `fan_in = s[1] * receptive`; step 1 |
| ignoring the kernel dimensions | a $3 \times 3$ convolution is initialized 9 times too large in variance | `receptive = math.prod(s[2:])`; step 1 |
| using $\sqrt{2/(n_{\text{in}} + n_{\text{out}})}$ as the uniform bound | the uniform's variance is $a^2/3$, so this is 3 times too small | `math.sqrt(6.0 / (fan_in + fan_out))`; step 3 |
| ReLU gain 2 (a variance) instead of $\sqrt{2}$ (a standard deviation) | activations double in mean square per layer: $2^{19}$ after 20 | `calculate_gain("relu")` returns `math.sqrt(2.0)`; step 2 |
| GPT-2 residual scale $1/\sqrt{L}$ instead of $1/\sqrt{2L}$ | each block adds two outputs; the stream ends twice as large | `math.sqrt(2.0 * n_layers)`; step 6 |
| filling in column-major order | same distribution, different weights: Python and the Rust port disagree for the same seed | numpy's default C-order `reshape`; step 4 |
| making a new generator inside the initializer | every layer gets the same weights, and the seed does nothing | `rng` is a parameter; step 5 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.0` | `normal(rng, n)` draws every normal weight; expectation and variance are its definitions |
| Back | `M01.3` | the activations whose second moments set the gains (reading) |
| Back | `M06.3` | the PCG32 generator passed in as `rng` (reading) |
| Back | `S-M02`, `S-M04` | the Gaussian integrals behind $\mathbb{E}[\mathrm{relu}(z)^2] = \mathrm{Var}(z)/2$ (reading) |
| Forward | `L0.4` | `Linear` and `Embedding` initialize their parameters with these functions |
| Forward | `L3.2` | LSTM gate weights |
| Forward | `L3.3`, `L3.6`, `L5.4`, `L5.5`, `L6.1`, `L6.2`, `L6.3` | further layers initialized with the same functions |
| Forward | `L7.9` and `C1` | `normal_init(0.02)` plus `scaled_residual_std` for the output projections of a Llama-style decoder |
| Forward | `L13.1` | `normal_init` scales conv weights by fan-in |
| Forward | `L13.3` | `normal_init` for patch and position embeddings |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `xavier_*`, `kaiming_normal` | `torch.nn.init` | in-place initializers, `trunc_normal_`, `orthogonal_` (`M03.3`), `kaiming_uniform_` with `a = sqrt(5)` (torch's `Linear` default) | `torch/nn/init.py` |
| `scaled_residual_std` | GPT-NeoX, Megatron `scaled_init_method_normal` | the same $1/\sqrt{2L}$ rule, applied by parameter name | `megatron/core/utils.py` |
| a fixed $0.02$ base std | muP (maximal update parametrization) | width-dependent init and learning rates so hyperparameters transfer from small to large models | Yang et al., "Tensor Programs V" (2022) |
| variance analysis at initialization | signal propagation theory | mean-field analysis of depth and the edge of chaos; Fixup and T-Fixup train without normalization | Schoenholz et al., "Deep Information Propagation" (2017) |
