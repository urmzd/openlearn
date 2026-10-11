<!-- ol:module M07.0 -->
# Random variables, expectation, variance, and normal draws by Box-Muller

## Overview

| | |
|---|---|
| **Module** | `M07.0` · build · Python · Pass 2 · 2 to 3 h |
| **You build** | `python/tinyllm/prob/rv.py`: `expectation(values, probs)`, `variance(values, probs)` (two-pass), `box_muller(u1, u2)`, `normal(rng, n)` (standard normals in the cross-language order of `spec/pcg32.md`) |
| **Contract** | [`course/contracts/py/tinyllm/prob/rv.pyi`](../../course/contracts/py/tinyllm/prob/rv.pyi) · draw order: [`spec/pcg32.md`](../../course/contracts/spec/pcg32.md) |
| **Tests** | `course/tests/M07.0/test_rv.py` (what they check: section 4) |
| **Needs** | `M06.3` the PCG32 generator every caller passes in (or `--ref-deps`). Reading: `M00.2` (cos, sin, the unit circle), [Calculus 2](../02-calculus-2/) (the Gaussian integral) |
| **Used by** | `M03.3` orthogonal initialization draws its Gaussian matrix here · `M07.3` `normal_init` and variance propagation through layers |
| **Milestone** | `MS-P2` (the foundations gate) |
| **Optional depth** | Blitzstein and Hwang, *Introduction to Probability*, ch. 3 to 5 and 7.5 (Box-Muller); Grinstead and Snell, *Introduction to Probability*, ch. 6; Box and Muller, "A Note on the Generation of Random Normal Deviates" (1958); Welford, "Note on a Method for Calculating Corrected Sums of Squares and Products" (1962) |

## Key Takeaways

- A discrete **random variable** is a table of values and probabilities; its **expectation** is the probability-weighted mean and its **variance** the expected squared distance from that mean (`test_die_hand_example`).
- $\operatorname{Var}[aX + c] = a^2 \operatorname{Var}[X]$: shifting does nothing, scaling scales by the square. `M07.3` sizes every initialization with this rule (`test_variance_shift_and_scale_laws`).
- Compute the variance in **two passes** (mean first, then squared deviations). $E[X^2] - E[X]^2$ subtracts two huge, nearly equal numbers and can return 0 or a negative variance (`test_variance_has_no_catastrophic_cancellation`).
- **Box-Muller** turns two independent uniforms into two independent standard normals: a radius $\sqrt{-2\ln(1-u_1)}$ and an angle $2\pi u_2$ (`test_box_muller_hand_example`). Drawn in the spec's order, the same seed gives the same normals in Python, Rust, and Go (`test_normal_matches_spec_vectors`).

## How to work this chapter

```bash
ol start M07.0              # stubs python/tinyllm/prob/rv.py into your repo
ol tests M07.0              # read the test catalog first: rung R0, you write no tests here
ol check M07.0              # exit code is the verdict
ol check M07.0 --ref-deps   # only if your M06.3 is not passing yet
ol diff  M07.0              # after passing: your code against the reference
```

---

## 1. Why now

Your system now has a reproducible generator (`M06.3`) that produces uniform numbers in $[0, 1)$. Neural networks are not initialized with uniform numbers: almost every weight in this course, from the first `Linear` layer (`L0.4`) to the Llama blocks (`L7.9`), starts as a draw from a **normal distribution** with a carefully chosen spread, and the next module that needs one is `M03.3`, whose orthogonal initializer is the QR factorization of a Gaussian matrix. Choosing the spread is a variance calculation (`M07.3`): too large and activations explode through 20 layers, too small and they vanish. This module defines random variables, expectation, and variance from the beginning, and turns your uniforms into normals with Box-Muller, in exactly the order the Rust and Go ports will use, so that one seed means the same weights everywhere.

## 2. Principles

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

### 2.1 Random variables and distributions

A **random variable** assigns a number to each outcome of a random experiment. A **discrete** one takes values $x_1, \dots, x_k$ with probabilities $p_1, \dots, p_k$, where every $p_i \ge 0$ and $\sum_i p_i = 1$; that table *is* its distribution. A table whose probabilities do not sum to 1 (say, raw counts) or contain a negative entry describes no random variable, and the functions in this module reject it rather than average with it.

A **continuous** random variable has a **density** $f \ge 0$ with $\int f = 1$; probabilities are areas under $f$, and any single value has probability 0. $U$, uniform on $[0, 1)$, has $f = 1$ on that interval.

### 2.2 Expectation and variance

The **expectation** is the probability-weighted average of the values,

$$E[X] = \sum_i p_i\, x_i \qquad \Big(\text{continuous: } \int z f(z)\,dz\Big),$$

the long-run average of many independent draws. It is **linear**: $E[aX + c] = aE[X] + c$, and $E[X + Y] = E[X] + E[Y]$ for any $X, Y$.

The **variance** measures spread, as the expected squared distance from the mean:

$$\operatorname{Var}[X] = E\big[(X - \mu)^2\big] = \sum_i p_i\,(x_i - \mu)^2, \qquad \mu = E[X].$$

From linearity: $\operatorname{Var}[aX + c] = E[(aX + c - a\mu - c)^2] = a^2 \operatorname{Var}[X]$. A shift moves the distribution without spreading it; a scale by $a$ spreads it by $\lvert a \rvert$, so the variance grows by $a^2$. For **independent** $X$ and $Y$, variances add: $\operatorname{Var}[X + Y] = \operatorname{Var}[X] + \operatorname{Var}[Y]$. A neuron $\sum_{j=1}^{n} w_j x_j$ with independent zero-mean terms therefore has variance $n \operatorname{Var}[w]\operatorname{Var}[x]$, which is why `M07.3` scales initial weights by $1/\sqrt{n}$.

Two examples recur everywhere. A **Bernoulli**($p$) variable (1 with probability $p$, else 0) has $E = p$ and $\operatorname{Var} = p(1 - p)$: every accuracy you will ever measure is an average of these. A **point mass** (one value with probability 1) has variance exactly 0.

### 2.3 Computing the variance without cancellation

Expanding the square gives the textbook shortcut $\operatorname{Var}[X] = E[X^2] - E[X]^2$, which is algebraically equal and numerically dangerous. Take $X = 10^9 + B$ with $B$ a fair coin: the true variance is $\tfrac14$. But $E[X^2] \approx 10^{18}$ and $E[X]^2 \approx 10^{18}$, and float64 numbers near $10^{18}$ are spaced 128 apart, so their difference is a multiple of 128 near zero: the answer comes out 0 or $-64$, every digit wrong, sometimes negative. The **two-pass** method computes $\mu$ first, then $\sum_i p_i (x_i - \mu)^2$: the deviations $x_i - \mu$ are small numbers computed with small errors, and the squares are all non-negative. (`M09.2` returns to this as Welford's and Kahan's algorithms.)

### 2.4 The normal distribution

The **standard normal** $\mathcal{N}(0, 1)$ has density $f(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$: mean 0, variance 1, symmetric, with about 68% of its mass within one standard deviation and 99.7% within three. $\mu + \sigma Z$ is $\mathcal{N}(\mu, \sigma^2)$. The constant $\frac{1}{\sqrt{2\pi}}$ comes from the Gaussian integral $\int_{-\infty}^{\infty} e^{-z^2/2}\,dz = \sqrt{2\pi}$, computed by squaring it and switching to polar coordinates, and the same trick in reverse is how normals are generated.

### 2.5 Box-Muller

Two independent standard normals $(Z_0, Z_1)$ form a point in the plane whose density $\frac{1}{2\pi} e^{-(z_0^2 + z_1^2)/2}$ depends only on the distance from the origin. In polar coordinates $(r, \theta)$:

- the **angle** $\theta$ is uniform on $[0, 2\pi)$, by the rotational symmetry;
- the **radius** satisfies $\Pr[R > r] = e^{-r^2/2}$ (integrate the density outside the circle of radius $r$), so $R^2/2$ is exponential with mean 1.

To sample $R$, invert its survival function: if $V$ is uniform on $(0, 1]$, $R = \sqrt{-2 \ln V}$ has exactly that distribution. With two uniforms $u_1, u_2 \in [0, 1)$:

$$r = \sqrt{-2 \ln(1 - u_1)}, \qquad \theta = 2\pi u_2, \qquad (z_0, z_1) = (r\cos\theta,\ r\sin\theta).$$

Using $1 - u_1$ instead of $u_1$ matters at the edge: `uniform()` can return exactly 0.0 (probability $2^{-53}$, but over $10^9$ draws not never), and $\ln 0 = -\infty$ would produce an infinite "normal"; $1 - u_1$ lies in $(0, 1]$, so the log is finite and $r$ is finite. Both outputs are independent standard normals; throwing the sine away wastes half the work.

**The order is a contract** (`spec/pcg32.md`): for each pair, draw $u_1$ then $u_2$, emit $z_0$ (cosine) then $z_1$ (sine). `normal(rng, n)` fills $n$ values pair by pair; for odd $n$ it still draws a whole last pair and drops the final sine. (The spec's generator-level `normal()` keeps that sine as a spare for the next call; `normal(rng, n)` is a free function over any generator, so it keeps no state between calls: course/DEVIATIONS.md, M070-01.) Every other port reproduces this order, which is how the parity suite can compare normals across languages.

## 3. Worked example by hand

**A fair die** (`test_die_hand_example`). Values $1, \dots, 6$, each with probability $\tfrac16$:

$$E[X] = \frac{1 + 2 + 3 + 4 + 5 + 6}{6} = \frac{21}{6} = 3.5 .$$

The deviations from 3.5 are $\pm 2.5, \pm 1.5, \pm 0.5$, so

$$\operatorname{Var}[X] = \frac{2\,(6.25 + 2.25 + 0.25)}{6} = \frac{17.5}{6} = \frac{35}{12} \approx 2.9167 .$$

**Box-Muller by hand** (`test_box_muller_hand_example`). Choose $u_1 = 1 - e^{-2} \approx 0.8647$ and $u_2 = \tfrac18$. Then $1 - u_1 = e^{-2}$, so $r = \sqrt{-2 \ln e^{-2}} = \sqrt{4} = 2$; and $\theta = 2\pi/8 = \pi/4$. The pair is $(2\cos\frac{\pi}{4},\ 2\sin\frac{\pi}{4}) = (\sqrt2, \sqrt2) \approx (1.41421, 1.41421)$.

**The first normals of seed 0** (`test_normal_matches_spec_vectors`). `PCG32(0)` gives $u_1 = 0.2803122753265841$ and $u_2 = 0.4892241428740438$ (`M06.3` section 3). Then $\ln(1 - u_1) = \ln 0.71969 = -0.328938$, $r = \sqrt{0.657876} = 0.811095$, $\theta = 3.073886$ (just under $\pi$, so the point lies left of the origin, slightly up), $\cos\theta = -0.997709$, $\sin\theta = 0.067655$, and the first two normals are $z_0 = -0.809237$ and $z_1 = 0.054875$: the first two entries of `normal["0"]` in `spec/pcg32.vectors.json`.

## 4. The interface

```python
def expectation(values, probs) -> float: ...     # sum p_i x_i; ValueError unless a distribution
def variance(values, probs) -> float: ...        # sum p_i (x_i - mu)^2, two passes
def box_muller(u1: float, u2: float) -> tuple[float, float]: ...   # (r cos t, r sin t)
def normal(rng, n: int) -> NDArray: ...          # float64 [n]; u1, u2 per pair; cosine first
```

`rng` is anything with a `uniform()` method returning the spec's 53-bit doubles: your `tinyllm.num.rng.PCG32` in the system, the frozen `course/tests/_lib/pcg32.py` copy in the course tests (D35). Both give the same uniforms bit for bit.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_die_hand_example` | unit, smoke | section 3: $E = 7/2$, $\operatorname{Var} = 35/12$ | you and the tests agree on the definitions |
| `test_bernoulli_and_point_mass` | unit | $E = p$, $\operatorname{Var} = p(1 - p)$; a point mass has variance 0 | every accuracy metric is a Bernoulli mean (`M07.4`) |
| `test_variance_has_no_catastrophic_cancellation` | boundary | $10^9 + \{0, 1\}$ gives exactly $\tfrac14$ | statistics of large-mean quantities (token counts, timestamps) |
| `test_variance_shift_and_scale_laws` | property | $\operatorname{Var}[aX + c] = a^2\operatorname{Var}[X]$ and $E[aX + c] = aE[X] + c$ on 50 random tables | the scaling rule of `M07.3` |
| `test_rejects_tables_that_are_not_distributions` | boundary | sums other than 1, negative entries, shape mismatches, empty and 2-D tables raise | unnormalized counts are caught upstream |
| `test_box_muller_hand_example` | unit, smoke | section 3: $(\sqrt2, \sqrt2)$, and $(0, 2)$ at $\theta = \pi/2$ | the transform itself |
| `test_box_muller_at_u1_zero_is_finite` | boundary | $u_1 = 0$ gives radius 0; $u_1$ just below 1 gives a large finite value | no infinite weights |
| `test_normal_matches_spec_vectors` | golden | the first 8 normals of seeds 0, 1, $2^{63}$ within 4 ulp | the Rust and Go ports draw the same normals |
| `test_odd_n_draws_whole_pairs_and_drops_the_last_sine` | unit | `normal(rng, 3)` uses 4 uniforms and equals the first 3 of `normal(rng, 4)`; `n = 0` and `n < 0` | the stream position after a call |
| `test_normal_moments_within_three_standard_errors` | statistical | 20 000 draws: mean and variance within 3 standard errors of 0 and 1 | a missing factor 2 halves every initial variance |
| `test_normal_shape_by_chi_square` | statistical | 20 bins equally likely under $\Phi$: Pearson statistic below 43.82 (19 degrees of freedom, $p > 10^{-3}$) | the right moments are not the right shape |
| `test_pair_halves_are_uncorrelated` | statistical | the cosine and sine halves have correlation within 3 standard errors of 0 | independent draws |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| averaging the values without the probabilities | right only for uniform tables | `test_bernoulli_and_point_mass` (mutant `s01`) |
| returning the standard deviation as the variance | $\sqrt{35/12}$ instead of $35/12$ | `test_die_hand_example` (mutant `s02`) |
| $E[X^2] - E[X]^2$ | variance 0 or negative for large means | `test_variance_has_no_catastrophic_cancellation` (mutant `s03`) |
| not validating the table | silent nonsense from unnormalized counts | `test_rejects_tables_that_are_not_distributions` (mutant `s04`) |
| $\ln u_1$ instead of $\ln(1 - u_1)$ | an infinite value once in $2^{53}$ draws; different values from every port | `test_box_muller_at_u1_zero_is_finite` (mutant `s05`) |
| $\sqrt{-\ln(\cdot)}$ without the 2 | variance $\tfrac12$: every initial weight too small by $\sqrt2$ | `test_normal_moments_within_three_standard_errors` (mutant `s06`) |
| angle $\pi u_2$ instead of $2\pi u_2$ | the sine half is always positive; wrong shape | `test_normal_shape_by_chi_square` (mutant `s07`) |
| sine first | statistically fine, but matches no other language | `test_normal_matches_spec_vectors` (mutant `s08`) |
| both halves from the cosine | perfectly correlated pairs | `test_pair_halves_are_uncorrelated` (mutant `s09`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M06.3` | the generator: two `uniform()` calls per pair, the same uniforms in every language |
| Back | `M00.2` | cosine, sine, and the unit circle (reading) |
| Back | [Calculus 2](../02-calculus-2/) | the Gaussian integral behind $\frac{1}{\sqrt{2\pi}}$ (reading; `S-M02` checks it) |
| Forward | `M03.3` | `orthogonal_init` factors `normal(rng, rows * cols)` |
| Forward | `M07.3` | `normal_init`, Xavier and Kaiming normals, and the variance propagation rule $n\operatorname{Var}[w]\operatorname{Var}[x]$ |
| Forward | `M07.1` | categorical sampling builds on the same uniforms (reading) |

If you skip this module, `ol check M03.3` stops with `BLOCKED ... needs M07.0`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| Box-Muller | the Ziggurat algorithm (numpy's `standard_normal`, Rust `rand_distr`) | almost always one uniform and one table lookup per normal, no `log`, `sin`, or `cos` | numpy `numpy/random/src/distributions/distributions.c`, `random_standard_normal` |
| Box-Muller | the Marsaglia polar method | rejection inside the unit disc replaces $\sin$ and $\cos$ | Marsaglia and Bray (1964) |
| two-pass variance | Welford's online algorithm, parallel merging (Chan et al.) | one pass over a stream, mergeable across workers | `torch.var_mean`, Welford (1962) |
| `normal(rng, n)` | `torch.nn.init.normal_`, `jax.random.normal` | per-device counter-based generators, many values per call | `torch/nn/init.py`, `jax/_src/random.py` |
