# Random variables, expectation, variance, and normal draws by Box-Muller

By the end of the previous chapter you had a reproducible generator, [PCG32](../12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md), that produces uniform numbers in $[0, 1)$. Neural networks are not initialized with uniform numbers. Almost every weight in *AI Engineering*, from the first `Linear` layer in [Module system and basic layers](../../ai-engineering/06-numerics-and-autograd/05-module-system-and-layers.md) to the [Llama-family model](../../ai-engineering/13-decoder-block/09-llama-family-model.md), starts as a draw from a **normal distribution** with a carefully chosen spread. The first place this book needs one is [orthogonal initialization](../09-linear-algebra/03-orthogonality-and-householder-qr.md), which is the QR factorization of a Gaussian matrix.

Choosing that spread is a variance calculation, the subject of [Variance propagation and initialization](04-variance-propagation-and-initialization.md): too large and activations explode through 20 layers, too small and they vanish. This section lays the groundwork. It defines random variables, expectation, and variance from the beginning, shows how to compute a variance without destroying its digits, and turns uniforms into normals with the Box-Muller transform, in exactly the order the Rust and Go ports use, so that one seed means the same weights everywhere.

You will need the generator from [Modular arithmetic, hashing, and PCG32](../12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) (the solution's [`rng.py`](../../../../course/solutions/python/tinyllm/num/rng.py)), cosine, sine, and the unit circle from [Trig, rotations, and Euler's formula](../06-precalculus/02-trig-rotations-and-eulers-formula.md), and, for background, the Gaussian integral from [Calculus 2](../08-calculus-2/), which its [problem set](../08-calculus-2/03-problem-set.md) works through. The reference implementation is [`course/solutions/python/tinyllm/prob/rv.py`](../../../../course/solutions/python/tinyllm/prob/rv.py), its contract is [`rv.pyi`](../../../../course/contracts/py/tinyllm/prob/rv.pyi), and the cross-language draw order is fixed by [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md).

> **Learning objectives**
> - Compute the expectation and variance of a discrete distribution, and validate that a table really is a distribution.
> - Derive $\operatorname{Var}[aX + c] = a^2 \operatorname{Var}[X]$ and explain why independent variances add.
> - Explain why $E[X^2] - E[X]^2$ fails numerically and compute the variance in two passes instead.
> - Derive the Box-Muller transform from the polar form of two independent normals and implement it in the spec's draw order.

## Notation

The section uses the following symbols throughout.

| Symbol | Meaning | Type |
|---|---|---|
| $X$ | a random variable: a quantity whose value is drawn at random | |
| $x_i, p_i$ | the values of a discrete $X$ and their probabilities, $p_i \ge 0$, $\sum_i p_i = 1$ | `float64[k]` each |
| $E[X] = \mu$ | expectation (mean) | scalar |
| $\operatorname{Var}[X] = \sigma^2$ | variance; $\sigma$ is the standard deviation | scalar |
| $U$ | a uniform random variable on $[0, 1)$ | |
| $f(z)$ | a probability density: $\Pr[a \le Z \le b] = \int_a^b f(z)\,dz$ | |
| $\mathcal{N}(\mu, \sigma^2)$ | the normal distribution; $\mathcal{N}(0, 1)$ is the standard normal | |
| $\Phi(z)$ | the standard normal CDF, $\Pr[Z \le z] = \tfrac12\big(1 + \operatorname{erf}(z/\sqrt2)\big)$ | |
| $u_1, u_2$ | two uniform draws | `float` |
| $r, \theta$ | polar radius and angle | `float` |

## Random variables and distributions

A **random variable** assigns a number to each outcome of a random experiment. A **discrete** one takes values $x_1, \dots, x_k$ with probabilities $p_1, \dots, p_k$, where every $p_i \ge 0$ and $\sum_i p_i = 1$; that table *is* its distribution. A table whose probabilities do not sum to 1 (say, raw counts) or contain a negative entry describes no random variable, and the functions in this section reject it rather than average with it.

A **continuous** random variable has a **density** $f \ge 0$ with $\int f = 1$; probabilities are areas under $f$, and any single value has probability 0. $U$, uniform on $[0, 1)$, has $f = 1$ on that interval.

## Expectation and variance

Once you have a distribution, the first two things to ask of it are where it sits and how widely it spreads. The **expectation** answers the first. It is the probability-weighted average of the values,

$$E[X] = \sum_i p_i\, x_i \qquad \Big(\text{continuous: } \int z f(z)\,dz\Big),$$

the long-run average of many independent draws. It is **linear**: $E[aX + c] = aE[X] + c$, and $E[X + Y] = E[X] + E[Y]$ for any $X, Y$.

The **variance** answers the second, as the expected squared distance from the mean:

$$\operatorname{Var}[X] = E\big[(X - \mu)^2\big] = \sum_i p_i\,(x_i - \mu)^2, \qquad \mu = E[X].$$

From linearity, $\operatorname{Var}[aX + c] = E[(aX + c - a\mu - c)^2] = a^2 \operatorname{Var}[X]$. A shift moves the distribution without spreading it; a scale by $a$ spreads it by $\lvert a \rvert$, so the variance grows by $a^2$. For **independent** $X$ and $Y$, variances add: $\operatorname{Var}[X + Y] = \operatorname{Var}[X] + \operatorname{Var}[Y]$. A neuron $\sum_{j=1}^{n} w_j x_j$ with independent zero-mean terms therefore has variance $n \operatorname{Var}[w]\operatorname{Var}[x]$, which is why [Variance propagation and initialization](04-variance-propagation-and-initialization.md) scales initial weights by $1/\sqrt{n}$. Every initialization in the book is sized with this rule.

Two examples recur everywhere. A **Bernoulli**($p$) variable (1 with probability $p$, else 0) has $E = p$ and $\operatorname{Var} = p(1 - p)$: every accuracy you will ever measure is an average of these. A **point mass** (one value with probability 1) has variance exactly 0.

## Computing the variance without cancellation

The definition is clear; the arithmetic is where implementations go wrong. Expanding the square gives the textbook shortcut $\operatorname{Var}[X] = E[X^2] - E[X]^2$, which is algebraically equal and numerically dangerous. Take $X = 10^9 + B$ with $B$ a fair coin: the true variance is $\tfrac14$. But $E[X^2] \approx 10^{18}$ and $E[X]^2 \approx 10^{18}$, and float64 numbers near $10^{18}$ are spaced 128 apart, so their difference is a multiple of 128 near zero: the answer comes out 0 or $-64$, every digit wrong, sometimes negative.

The **two-pass** method computes $\mu$ first, then $\sum_i p_i (x_i - \mu)^2$. The deviations $x_i - \mu$ are small numbers computed with small errors, and the squares are all non-negative. [Stable numerics](../15-numerical-methods-and-floating-point/02-stable-numerics.md) returns to this problem with Welford's and Kahan's algorithms.

## The normal distribution

With expectation and variance in hand, you can describe the distribution the rest of the section is about. The **standard normal** $\mathcal{N}(0, 1)$ has density $f(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$: mean 0, variance 1, symmetric, with about 68% of its mass within one standard deviation and 99.7% within three. $\mu + \sigma Z$ is $\mathcal{N}(\mu, \sigma^2)$. The constant $\frac{1}{\sqrt{2\pi}}$ comes from the Gaussian integral $\int_{-\infty}^{\infty} e^{-z^2/2}\,dz = \sqrt{2\pi}$, computed by squaring it and switching to polar coordinates, and the same trick in reverse is how normals are generated.

## Box-Muller

Two independent standard normals $(Z_0, Z_1)$ form a point in the plane whose density $\frac{1}{2\pi} e^{-(z_0^2 + z_1^2)/2}$ depends only on the distance from the origin. In polar coordinates $(r, \theta)$, two facts follow:

- the **angle** $\theta$ is uniform on $[0, 2\pi)$, by the rotational symmetry;
- the **radius** satisfies $\Pr[R > r] = e^{-r^2/2}$ (integrate the density outside the circle of radius $r$), so $R^2/2$ is exponential with mean 1.

To sample $R$, invert its survival function: if $V$ is uniform on $(0, 1]$, $R = \sqrt{-2 \ln V}$ has exactly that distribution. With two uniforms $u_1, u_2 \in [0, 1)$:

$$r = \sqrt{-2 \ln(1 - u_1)}, \qquad \theta = 2\pi u_2, \qquad (z_0, z_1) = (r\cos\theta,\ r\sin\theta).$$

Using $1 - u_1$ instead of $u_1$ matters at the edge. `uniform()` can return exactly 0.0 (probability $2^{-53}$, but over $10^9$ draws not never), and $\ln 0 = -\infty$ would produce an infinite "normal"; $1 - u_1$ lies in $(0, 1]$, so the log is finite and $r$ is finite. Both outputs are independent standard normals; throwing the sine away wastes half the work.

**The order is a contract.** [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md) fixes it: for each pair, draw $u_1$ then $u_2$, emit $z_0$ (cosine) then $z_1$ (sine). `normal(rng, n)` fills $n$ values pair by pair; for odd $n$ it still draws a whole last pair and drops the final sine. (The spec's generator-level `normal()` keeps that sine as a spare for the next call; `normal(rng, n)` is a free function over any generator, so it keeps no state between calls.) Every other port reproduces this order, which is how normals can be compared across languages.

## Working it by hand

Before writing code, it helps to have three answers you can check it against.

**A fair die.** Values $1, \dots, 6$, each with probability $\tfrac16$:

$$E[X] = \frac{1 + 2 + 3 + 4 + 5 + 6}{6} = \frac{21}{6} = 3.5 .$$

The deviations from 3.5 are $\pm 2.5, \pm 1.5, \pm 0.5$, so

$$\operatorname{Var}[X] = \frac{2\,(6.25 + 2.25 + 0.25)}{6} = \frac{17.5}{6} = \frac{35}{12} \approx 2.9167 .$$

**Box-Muller by hand.** Choose $u_1 = 1 - e^{-2} \approx 0.8647$ and $u_2 = \tfrac18$. Then $1 - u_1 = e^{-2}$, so $r = \sqrt{-2 \ln e^{-2}} = \sqrt{4} = 2$; and $\theta = 2\pi/8 = \pi/4$. The pair is $(2\cos\frac{\pi}{4},\ 2\sin\frac{\pi}{4}) = (\sqrt2, \sqrt2) \approx (1.41421, 1.41421)$.

**The first normals of seed 0.** `PCG32(0)` gives $u_1 = 0.2803122753265841$ and $u_2 = 0.4892241428740438$ (worked by hand in the [PCG32 section](../12-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md)). Then $\ln(1 - u_1) = \ln 0.71969 = -0.328938$, $r = \sqrt{0.657876} = 0.811095$, $\theta = 3.073886$ (just under $\pi$, so the point lies left of the origin, slightly up), $\cos\theta = -0.997709$, $\sin\theta = 0.067655$, and the first two normals are $z_0 = -0.809237$ and $z_1 = 0.054875$: the first two entries of `normal["0"]` in `spec/pcg32.vectors.json`.

## Designing the interface

Four functions cover the section:

```python
def expectation(values, probs) -> float: ...     # sum p_i x_i; ValueError unless a distribution
def variance(values, probs) -> float: ...        # sum p_i (x_i - mu)^2, two passes
def box_muller(u1: float, u2: float) -> tuple[float, float]: ...   # (r cos t, r sin t)
def normal(rng, n: int) -> NDArray: ...          # float64 [n]; u1, u2 per pair; cosine first
```

`rng` is anything with a `uniform()` method returning the spec's 53-bit doubles; in this system that is [`tinyllm.num.rng.PCG32`](../../../../course/solutions/python/tinyllm/num/rng.py). Any implementation of the spec gives the same uniforms bit for bit.

## Building it step by step

The implementation follows from five observations, each of which you should try to predict before reading on.

1. **A distribution is a validated table.** Values and probabilities must be 1-D, the same non-zero length, finite, with $p_i \ge 0$ and $\sum p_i = 1$. Anything else (raw counts, a negative entry) is not a random variable, so both `expectation` and `variance` share one validator that raises instead of averaging. The sum test needs a tolerance ($10^{-9}$) because probabilities computed in floating point need not sum to exactly 1.
2. **The expectation is a dot product.** $E[X] = \sum_i p_i x_i = x \cdot p$. Weighting by $p$ (not dividing by $k$) is what makes it right for non-uniform tables.
3. **The variance centres first, then squares.** $E[X^2] - E[X]^2$ cancels catastrophically near $10^{18}$. Computing $\mu$ once, then $d = x - \mu$ and $\sum_i p_i d_i^2$, keeps every term small and non-negative, so $10^9 + \{0, 1\}$ gives exactly $\tfrac14$ and no variance can come out negative.
4. **Box-Muller samples the polar form.** $r = \sqrt{-2\ln V}$ with $V$ in $(0, 1]$ and $\theta = 2\pi u_2$. Taking $V = 1 - u_1$ maps the generator's $[0, 1)$ onto $(0, 1]$, so $u_1 = 0$ gives $r = 0$ instead of $\infty$.
5. **The draw order is part of the answer.** Per pair, $u_1$ then $u_2$; emit cosine then sine. A loop over pairs that writes `z0` always and `z1` only when there is room implements both the even and odd cases, and an odd $n$ still consumes two uniforms for the last pair, so the generator ends at the same position as in every other port.

The validator comes first:

```python
def _table(values: ArrayLike, probs: ArrayLike) -> tuple[NDArray, NDArray]:
    """values and probs as float64 vectors, checked to be a distribution."""
    x = np.asarray(values, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    # One shape check covers 2-D input, mismatched lengths, and the empty table.
    if x.ndim != 1 or p.ndim != 1 or x.shape != p.shape or x.size == 0:
        raise ValueError(f"values and probs must be 1-D of one non-zero length, got {x.shape} and {p.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite")
    if np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError("probs must be finite and >= 0")
    # A tolerance, not ==: probabilities computed in float64 (a softmax, a
    # normalized count) are only accurate to rounding.
    if abs(float(p.sum()) - 1.0) > 1e-9:
        raise ValueError(f"probs must sum to 1, got {float(p.sum())!r}")
    return x, p
```

Raw counts such as `[3, 1]` sum to 4 and are rejected here, which is the point: a caller who forgot to normalize hears about it at once rather than getting an expectation four times too large.

With a validated table, the two moments are short:

```python
def expectation(values: ArrayLike, probs: ArrayLike) -> float:
    x, p = _table(values, probs)
    return float(x @ p)                  # sum_i p_i x_i


def variance(values: ArrayLike, probs: ArrayLike) -> float:
    x, p = _table(values, probs)
    mu = float(x @ p)                    # pass 1: the mean
    d = x - mu                           # pass 2: small deviations, computed accurately
    return float((d * d) @ p)            # sum_i p_i (x_i - mu)^2, every term >= 0
```

On the die, `mu = 3.5`, `d` is $\pm 2.5, \pm 1.5, \pm 0.5$, and the weighted sum of squares is $17.5/6 = 35/12$, exactly the hand calculation. On $10^9 + \{0, 1\}$ the deviations are exactly $\pm\tfrac12$, so the result is exactly $\tfrac14$.

The transform and the sampler that drives it complete the module:

```python
def box_muller(u1: float, u2: float) -> tuple[float, float]:
    # 1 - u1 lies in (0, 1], so the log is finite and r is in [0, inf).
    r = math.sqrt(-2.0 * math.log(1.0 - u1))   # R^2 / 2 is exponential(1)
    theta = 2.0 * math.pi * u2                 # a full turn, not half
    return r * math.cos(theta), r * math.sin(theta)   # cosine first: the spec's order


def normal(rng: Any, n: int) -> NDArray:
    if n < 0:
        raise ValueError(f"normal: n must be >= 0, got {n}")
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, 2):          # one Box-Muller pair per iteration
        u1 = rng.uniform()            # u1 then u2: the cross-language draw order
        u2 = rng.uniform()
        z0, z1 = box_muller(u1, u2)
        out[i] = z0
        if i + 1 < n:                 # odd n: the last sine is dropped, but its
            out[i + 1] = z1           # uniform was still consumed
    return out
```

`normal` keeps no spare between calls: it is a free function over any object with `uniform()`, so the only state is the generator's. `n = 0` returns an empty array and draws nothing. The full solution is [`rv.py`](../../../../course/solutions/python/tinyllm/prob/rv.py).

### Why this design

The costs are modest. `expectation` is one pass over the table and `variance` two, both $O(k)$ time and $O(k)$ extra space for `d`. `normal(rng, n)` makes $2\lceil n/2 \rceil$ uniform draws and one `log`, `sqrt`, `cos`, `sin` per pair.

The two-pass variance beats the shortcut outright. $E[X^2] - E[X]^2$ also needs two sums and saves nothing in a setting where the whole table is in memory; it only loses digits. Welford's one-pass update (see Further reading) is the right choice for streams, which is not this function's job.

Box-Muller wins over its alternatives for a reason specific to this system. The Ziggurat algorithm is faster but needs tables and a rejection loop whose number of draws per normal varies, which makes cross-language reproduction harder. The polar (Marsaglia) method also rejects. Box-Muller uses exactly two uniforms per pair, so the stream position after `normal(rng, n)` is a simple function of $n$, which is what the Rust and Go ports rely on.

The tradeoff accepted is speed: the Python loop is slow for millions of normals (one Python iteration per pair). Initializers in the book draw at most a few million values, and bit-for-bit agreement with the spec matters more than speed here.

## What can go wrong

A correct implementation has to pass checks of several kinds, from hand-worked unit cases to statistical tests of the output's shape. The table lists them along with why each matters later.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the fair die | unit | $E = 7/2$, $\operatorname{Var} = 35/12$ | everyone agrees on the definitions |
| Bernoulli and point mass | unit | $E = p$, $\operatorname{Var} = p(1 - p)$; a point mass has variance 0 | every accuracy metric is a Bernoulli mean ([confidence intervals](05-lln-clt-confidence-intervals-bootstrap.md)) |
| no catastrophic cancellation | boundary | $10^9 + \{0, 1\}$ gives exactly $\tfrac14$ | statistics of large-mean quantities (token counts, timestamps) |
| shift and scale laws | property | $\operatorname{Var}[aX + c] = a^2\operatorname{Var}[X]$ and $E[aX + c] = aE[X] + c$ on 50 random tables | the scaling rule of [initialization](04-variance-propagation-and-initialization.md) |
| tables that are not distributions | boundary | sums other than 1, negative entries, shape mismatches, empty and 2-D tables raise | unnormalized counts are caught upstream |
| Box-Muller by hand | unit | $(\sqrt2, \sqrt2)$, and $(0, 2)$ at $\theta = \pi/2$ | the transform itself |
| $u_1 = 0$ stays finite | boundary | $u_1 = 0$ gives radius 0; $u_1$ just below 1 gives a large finite value | no infinite weights |
| the spec vectors | golden | the first 8 normals of seeds 0, 1, $2^{63}$ within 4 ulp | the Rust and Go ports draw the same normals |
| odd $n$ | unit | `normal(rng, 3)` uses 4 uniforms and equals the first 3 of `normal(rng, 4)`; `n = 0` and `n < 0` | the stream position after a call |
| the moments | statistical | 20 000 draws: mean and variance within 3 standard errors of 0 and 1 | a missing factor 2 halves every initial variance |
| the shape | statistical | 20 bins equally likely under $\Phi$: Pearson statistic below 43.82 (19 degrees of freedom, $p > 10^{-3}$) | the right moments are not the right shape |
| the two halves of a pair | statistical | the cosine and sine halves have correlation within 3 standard errors of 0 | independent draws |

The mistakes that break these checks are few and recognizable. Averaging the values without the probabilities is right only for uniform tables; the solution weights each value with `x @ p`. Returning the standard deviation as the variance gives $\sqrt{35/12}$ instead of $35/12$; the solution returns the weighted sum of squares with no square root. Using $E[X^2] - E[X]^2$ gives a variance of 0 or a negative number for large means; the solution centres first with `d = x - mu`. Skipping validation turns unnormalized counts into silent nonsense; `_table` checks shape, finiteness, sign, and sum.

The Box-Muller mistakes are subtler, because several of them still produce plausible-looking numbers:

> **Warning:** $\ln u_1$ instead of $\ln(1 - u_1)$ produces an infinite value once in $2^{53}$ draws and different values from every other port. Dropping the 2 in $\sqrt{-2\ln(\cdot)}$ gives variance $\tfrac12$, so every initial weight is too small by $\sqrt2$. An angle of $\pi u_2$ instead of $2\pi u_2$ makes the sine half always positive, the wrong shape. Taking both halves from the cosine gives perfectly correlated pairs.

Emitting the sine first is statistically fine but matches no other language; the solution returns `(r * cos, r * sin)` and writes `z0` first.

## Where this leads

The normals built here feed two sections directly. [Orthogonality, Householder QR, and orthogonal initialization](../09-linear-algebra/03-orthogonality-and-householder-qr.md) factors `normal(rng, rows * cols)` in `orthogonal_init`, and [Variance propagation and initialization](04-variance-propagation-and-initialization.md) builds `normal_init`, Xavier and Kaiming normals, and the propagation rule $n\operatorname{Var}[w]\operatorname{Var}[x]$ on top of them. The next section, [Categorical sampling](02-categorical-sampling.md), builds on the same uniforms to draw tokens.

### Further reading

- Blitzstein and Hwang, *Introduction to Probability*, chapters 3 to 5 and 7.5 (Box-Muller).
- Grinstead and Snell, *Introduction to Probability*, chapter 6.
- Box and Muller, "A Note on the Generation of Random Normal Deviates" (1958).
- Welford, "Note on a Method for Calculating Corrected Sums of Squares and Products" (1962); `torch.var_mean` and Chan et al.'s parallel merging extend it to one pass over a stream, mergeable across workers.
- The Ziggurat algorithm (numpy's `standard_normal`, Rust `rand_distr`): almost always one uniform and one table lookup per normal, no `log`, `sin`, or `cos`. See `random_standard_normal` in numpy's `numpy/random/src/distributions/distributions.c`.
- Marsaglia and Bray (1964), the polar method: rejection inside the unit disc replaces $\sin$ and $\cos$.
- `torch.nn.init.normal_` (`torch/nn/init.py`) and `jax.random.normal` (`jax/_src/random.py`): per-device counter-based generators, many values per call.

## Key terms

- **Random variable**: a number assigned to each outcome of a random experiment; a discrete one is a table of values and probabilities.
- **Expectation**: the probability-weighted mean of a random variable's values.
- **Variance**: the expected squared distance from the mean; its square root is the standard deviation.
- **Catastrophic cancellation**: the loss of all significant digits when two large, nearly equal floating-point numbers are subtracted.
- **Two-pass variance**: computing the mean first, then the weighted sum of squared deviations from it.
- **Standard normal**: the distribution $\mathcal{N}(0, 1)$ with density $\frac{1}{\sqrt{2\pi}} e^{-z^2/2}$.
- **Box-Muller transform**: a map from two independent uniforms to two independent standard normals via a radius $\sqrt{-2\ln(1-u_1)}$ and angle $2\pi u_2$.

## Summary

- A discrete random variable is a table of values and probabilities; its expectation is the probability-weighted mean and its variance the expected squared distance from that mean.
- $\operatorname{Var}[aX + c] = a^2 \operatorname{Var}[X]$: shifting does nothing, scaling scales by the square. Every initialization is sized with this rule.
- Compute the variance in two passes (mean first, then squared deviations). $E[X^2] - E[X]^2$ subtracts two huge, nearly equal numbers and can return 0 or a negative variance.
- Box-Muller turns two independent uniforms into two independent standard normals: a radius $\sqrt{-2\ln(1-u_1)}$ and an angle $2\pi u_2$. Drawn in the spec's order, the same seed gives the same normals in Python, Rust, and Go.

## Review questions

1. Why does `variance` compute `d = x - mu` before squaring instead of returning `(x * x) @ p - mu * mu`?

   <details><summary>Answer</summary>

   For large means both $E[X^2]$ and $\mu^2$ are near $10^{18}$, where float64 spacing is 128, so their difference is a multiple of 128 near zero: 0 or $-64$ for a true variance of $\tfrac14$. The deviations $x_i - \mu$ are small and exact (or nearly), and their squares are non-negative, so the two-pass sum is accurate and cannot go negative.

   </details>

2. Why `math.log(1.0 - u1)` and not `math.log(u1)`? Both are uniform on the unit interval.

   <details><summary>Answer</summary>

   Distributionally they are equivalent, but the generator returns values in $[0, 1)$: $u_1 = 0$ is possible and $\ln 0 = -\infty$ gives an infinite radius. $1 - u_1$ lies in $(0, 1]$, so the log is finite. It also fixes which value every port computes from a given $u_1$, so the normals agree across languages.

   </details>

3. For odd $n$, why draw a whole pair and throw the sine away instead of drawing one uniform?

   <details><summary>Answer</summary>

   Box-Muller needs both uniforms to make even one normal (the radius uses $u_1$, the angle $u_2$). And the spec defines the stream position: `normal(rng, 3)` must consume 4 uniforms and equal the first 3 of `normal(rng, 4)`, so that the next draw from the generator is the same in Python, Rust, and Go.

   </details>

4. What would happen to the initial weights of [Variance propagation and initialization](04-variance-propagation-and-initialization.md) if the radius were $\sqrt{-\ln(1 - u_1)}$, without the 2?

   <details><summary>Answer</summary>

   $R^2$ would be exponential with mean 1 instead of 2, so each coordinate would have variance $\tfrac12$. Every weight scaled by a carefully derived $\sigma$ would actually have standard deviation $\sigma/\sqrt2$, and activations would shrink layer after layer. The mean is still 0, so only a variance check notices.

   </details>

5. Why does `_table` reject a table whose probabilities sum to 4 instead of normalizing it?

   <details><summary>Answer</summary>

   A table that does not sum to 1 is almost always a bug upstream (raw counts passed where probabilities were meant). Silently normalizing would hide it in this function and leave it in every other caller that trusts the same array. Raising makes the error visible where it is cheapest to fix.

   </details>
