# Categorical sampling: inverse CDF, Gumbel-max, alias method

## Overview

| | |
|---|---|
| **Module** | `M07.1` · build · Python · Pass 3 · 1.5 to 2 h reading |
| **Solution** | [`course/solutions/python/tinyllm/prob/sampling.py`](../../../../course/solutions/python/tinyllm/prob/sampling.py): `sample_categorical`, `gumbel_noise`, `gumbel_max`, `AliasTable` (`prob`, `alias`, `sample`), `exponential_icdf`, `poisson_arrivals` |
| **Contract** | [`course/contracts/py/tinyllm/prob/sampling.pyi`](../../../../course/contracts/py/tinyllm/prob/sampling.pyi) · the sampler's op order: [`spec/sampling.md`](../../../../course/contracts/spec/sampling.md) · the generator: [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md) |
| **Needs** | no code from earlier modules · reading: `M06.3` PCG32 (the `rng`), `M07.0` random variables and the uniform, `M00.1` logs |
| **Used by** | later: `L8.1` the sampler's step 11 · `L2.3` word2vec negative sampling · `L6.2` BERT's 80/10/10 masking · `M07.4` and `M07.6` resampling · `load.01` re-implements the Poisson schedule in Go · later: `L6.3`; part of the Pass 3 tokens-and-data layer |
| **Optional depth** | Devroye, *Non-Uniform Random Variate Generation* (1986), ch. 2 and 3; Vose, "A Linear Algorithm for Generating Random Numbers with a Given Distribution" (1991) |

## Key Takeaways

- The inverse CDF turns one uniform $u$ into id $i$ exactly when $F_{i-1} \le u < F_i$, an interval of length $p_i$; strict $<$ keeps zero-probability ids out.
- Adding independent Gumbel noise $-\log(-\log U)$ to logits and taking the argmax samples $\mathrm{softmax}$ exactly, with no normalization and no running sum.
- The alias method spends $O(n)$ once to split the distribution into $n$ equal columns of two ids each, then draws in $O(1)$ from one uniform.
- Every sampler here is a deterministic function of its uniforms, so the same seed gives the same draws in Python, Rust, and Go.

## How to read this chapter

Sections 1 and 2 derive the inverse CDF, Gumbel-max, the alias method, and Poisson arrivals from first principles, and section 3 works each by hand. Section 4 fixes the interface, and section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The tracer samples text with three lines buried inside `BigramLM.sample`: a numpy generator, a cumulative sum, a search. Pass 3 needs sampling in places that line cannot serve. word2vec (`L2.3`) draws millions of negative words from a 50 000-word distribution, and a cumulative sum per draw is 50 000 additions each time. BERT's masking (`L6.2`) and the bootstrap (`M07.4`) need many small draws with a known generator position. The sampler (`L8.1`) must return the same token as the Rust engine (`L10.1`) for the same seed, which only works if the last step, uniform to id, is pinned down to the comparison (`spec/sampling.md`, step 11). And the load generator (`load.01`) needs request arrival times that look like real traffic. This module turns uniforms into draws three exact ways, each with a different cost, and makes each one a pure function of its uniforms.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $p = (p_0, \dots, p_{n-1})$ | a categorical distribution: $p_i \ge 0$, $\sum_i p_i = 1$ | `float64[n]` |
| $F_i = p_0 + \dots + p_i$ | the cumulative distribution (CDF); $F_{-1} = 0$ | `float64[n]` |
| $U$, $u$ | a uniform random variable on $[0, 1)$, and one draw of it | `float` |
| $z_i$ | a logit: $p = \mathrm{softmax}(z)$, $p_i = e^{z_i} / \sum_j e^{z_j}$ | `float64[n]` |
| $G = -\log(-\log U)$ | a standard Gumbel random variable | `float` |
| $\mathrm{prob}_i$, $\mathrm{alias}_i$ | the alias table: column $i$ keeps $i$ with probability $\mathrm{prob}_i$, else gives $\mathrm{alias}_i$ | `float64[n]`, `int64[n]` |
| $\lambda$ | a rate: events per unit time | `float` |
| $T \sim \mathrm{Exp}(\lambda)$ | an exponential waiting time, $P(T > t) = e^{-\lambda t}$ | `float` |

**Randomness comes from outside.** None of these functions creates a generator. They take a uniform $u$, or an `rng` with a `uniform()` method (the PCG32 from `M06.3`, which turns two 32-bit draws into one float64 in $[0, 1)$). So each is a plain function from uniforms to outcomes: the same uniforms give the same outcomes in every language, and a test can feed chosen uniforms to check edge cases that a real generator would hit once in $2^{53}$ draws.

**Inverse CDF (discrete).** Lay the probabilities end to end on $[0, 1)$: id $i$ owns the interval $[F_{i-1}, F_i)$, whose length is $p_i$. Draw $u$ and return the id whose interval contains it, the first $i$ with $u < F_i$. Then

$$P(\text{return } i) = P(F_{i-1} \le U < F_i) = F_i - F_{i-1} = p_i.$$

Two details decide correctness. The comparison is strict: an id with $p_i = 0$ owns the empty interval $[F_{i-1}, F_{i-1})$, and with $u \le F_i$ instead, $u = 0$ would return id 0 even when $p_0 = 0$. And floating point: ten additions of $0.1$ give $0.9999999999999999$, so the largest uniform, $1 - 2^{-53}$, is not below any running sum. Then the answer is the last id with $p_i > 0$, never simply the last id. The walk costs $O(n)$ per draw; binary search over a stored $F$ makes it $O(\log n)$.

**Inverse CDF (continuous).** The same idea works for any increasing continuous CDF $F$: if $U$ is uniform then $X = F^{-1}(U)$ has $P(X \le x) = P(U \le F(x)) = F(x)$. The exponential distribution has $F(t) = 1 - e^{-\lambda t}$, so

$$t = F^{-1}(u) = -\frac{\log(1 - u)}{\lambda}.$$

Write it with `log1p(-u)`, which is accurate when $u$ is tiny, and note $u = 0$ gives $t = 0$, never $\log 0$. (Using $-\log(u)/\lambda$ is the same distribution, since $1 - U$ is uniform too, but a different schedule from the same seed, and $u = 0$ breaks it.)

**Poisson arrivals.** Requests that arrive independently at an average rate $\lambda$ have independent $\mathrm{Exp}(\lambda)$ gaps. Adding gaps until the time passes the horizon $H$ gives the arrival times on $[0, H)$; their count is Poisson with mean $\lambda H$ and variance $\lambda H$. This is the open-loop schedule `load.01` replays against the gateway.

**Gumbel-max.** Draw independent $G_i = -\log(-\log U_i)$ and return $\arg\max_i (z_i + G_i)$. The Gumbel CDF is $P(G \le g) = \exp(-e^{-g})$, with density $e^{-g}\exp(-e^{-g})$. Id $k$ wins when $z_k + G_k = x$ and every other $z_i + G_i < x$:

$$P(k) = \int e^{-(x - z_k)} e^{-e^{-(x - z_k)}} \prod_{i \ne k} e^{-e^{-(x - z_i)}} \, dx = e^{z_k} \int e^{-x} \exp\Big(-e^{-x} \sum_i e^{z_i}\Big) dx.$$

Substitute $s = e^{-x}$ ($ds = -e^{-x} dx$): the integral is $\int_0^\infty e^{-sZ} ds = 1/Z$ with $Z = \sum_i e^{z_i}$, so $P(k) = e^{z_k}/Z = \mathrm{softmax}(z)_k$. Three consequences: the logits need no normalization (a constant shifts every sum equally); a masked logit $-\infty$ stays $-\infty$ and never wins; and there is no running sum, so every id is processed independently, which is why GPU samplers use this form. It costs $n$ uniforms per draw. The sign matters: $\log(-\log U)$ is the Gumbel of the minimum, and with it the argmax samples the wrong distribution.

**The alias method.** Scale the probabilities by $n$, so they average 1, and picture $n$ columns of height 1. A column whose scaled mass is below 1 is topped up from a column above 1, and records whom it borrowed from. Vose's algorithm does this in one pass: keep a list of "small" ids (scaled mass $< 1$) and "large" ids ($\ge 1$); repeatedly pop one small $s$ and one large $g$, set $\mathrm{prob}_s$ to $s$'s mass and $\mathrm{alias}_s = g$, and give $g$ the mass it has left, $m_g + m_s - 1$, which goes back to the small or large list. Each step finishes one column, so it ends after at most $n$ steps. Whatever remains is a whole column ($\mathrm{prob} = 1$) up to rounding. Column $i$ then holds $\mathrm{prob}_i / n$ of id $i$ and $(1 - \mathrm{prob}_i)/n$ of id $\mathrm{alias}_i$, so

$$P(i) = \frac{1}{n}\Big(\mathrm{prob}_i + \sum_{j : \mathrm{alias}_j = i} (1 - \mathrm{prob}_j)\Big) = p_i.$$

A draw picks a column uniformly and flips a biased coin. One uniform does both: $x = n u$, column $i = \lfloor x \rfloor$, coin $f = x - i$, which is uniform on $[0, 1)$ and independent of $i$. Keep the column when $f < \mathrm{prob}_i$ (strict again: a column with $\mathrm{prob}_i = 0$ is never kept, even at $f = 0$). Building costs $O(n)$ once; every draw costs $O(1)$ whatever $n$ is.

**Which one when.** Inverse CDF: one draw from a distribution that changes every time (the sampler's next token), and the op order the spec fixes. Alias: many draws from one fixed distribution (negatives from a unigram table). Gumbel-max: no normalization, vectorized, and the root of Gumbel-top-$k$, which draws $k$ ids without replacement by keeping the $k$ largest $z_i + G_i$.

## 3. Worked example by hand

Take $p = (0.1, 0.2, 0.3, 0.4, 0.0)$, five ids.

**Inverse CDF.** The running sums are $F = (0.1, 0.3, 0.6, 1.0, 1.0)$. Id 4 owns $[1.0, 1.0)$, which is empty.

| $u$ | first $i$ with $u < F_i$ | id |
|---|---|---|
| 0.05 | $0.05 < 0.1$ | 0 |
| 0.10 | $0.10 < 0.1$ is false; $0.10 < 0.3$ | 1 |
| 0.25 | $0.25 < 0.3$ | 1 |
| 0.35 | $0.35 < 0.6$ | 2 |
| 0.99 | $0.99 < 1.0$ | 3 |

**Alias table (Vose).** Scaled mass $m = 5p = (0.5, 1.0, 1.5, 2.0, 0.0)$. Small (below 1): $[0, 4]$. Large: $[1, 2, 3]$. Pop from the end of each list:

| step | small $s$ | large $g$ | set | $g$ keeps | $g$ goes to |
|---|---|---|---|---|---|
| 1 | 4 ($m = 0$) | 3 ($m = 2$) | $\mathrm{prob}_4 = 0$, $\mathrm{alias}_4 = 3$ | $2 + 0 - 1 = 1.0$ | large |
| 2 | 0 ($m = 0.5$) | 3 ($m = 1$) | $\mathrm{prob}_0 = 0.5$, $\mathrm{alias}_0 = 3$ | $1 + 0.5 - 1 = 0.5$ | small |
| 3 | 3 ($m = 0.5$) | 2 ($m = 1.5$) | $\mathrm{prob}_3 = 0.5$, $\mathrm{alias}_3 = 2$ | $1.5 + 0.5 - 1 = 1.0$ | large |
| end | | | ids 1, 2 left in large: $\mathrm{prob} = 1$ | | |

So $\mathrm{prob} = (0.5, 1, 1, 0.5, 0)$, $\mathrm{alias} = (3, 1, 2, 2, 3)$. Check id 3: its own column keeps 0.5, column 0 gives it $1 - 0.5$, column 4 gives it $1 - 0$; $(0.5 + 0.5 + 1)/5 = 0.4$. Id 2: $(1 + 0.5)/5 = 0.3$.

A draw with $u = 0.13$: $x = 0.65$, column 0, coin $0.65 \ge 0.5$, so the alias: **id 3**. With $u = 0.05$: column 0, coin $0.25 < 0.5$: **id 0**. With $u = 0.95$: column 4, coin $0.75 \ge 0$: **id 3**.

**Gumbel-max.** Logits $z = \ln(0.1, 0.2, 0.3, 0.4) = (-2.3026, -1.6094, -1.2040, -0.9163)$. Uniforms $(0.9, 0.2, 0.5, 0.1)$ give noise $G = -\log(-\log u) = (2.2504, -0.4759, 0.3665, -0.8340)$. Sums: $(-0.0522, -2.0853, -0.8375, -1.7503)$. The argmax is **id 0**, the least likely id, because its uniform was lucky; that happens with probability exactly 0.1. Four equal uniforms add equal noise and leave the argmax of $z$: id 3.

**Poisson arrivals.** Rate $\lambda = 2$, horizon 2, uniforms $0.5, 0.75, 0.9$. Gaps $-\log(1 - u)/2$: $\ln 2 / 2 = 0.3466$, $\ln 4 / 2 = 0.6931$, $\ln 10 / 2 = 1.1513$. Times: 0.3466, 1.0397, then 2.1910, which is past the horizon: two arrivals, three uniforms used.

These numbers are the first four cases in section 4.

## 4. The interface

```python
# python/tinyllm/prob/sampling.py
def sample_categorical(probs: ArrayLike, u: float) -> int          # first i with u < F_i
def gumbel_noise(u: ArrayLike) -> NDArray                          # -log(-log u), u in (0, 1)
def gumbel_max(logits: ArrayLike, gumbels: ArrayLike) -> int       # ties to the lowest id
class AliasTable:
    prob: NDArray                                                  # float64 [n]
    alias: NDArray                                                 # int64 [n]
    def __init__(self, probs: ArrayLike) -> None
    def __len__(self) -> int
    def sample(self, rng: UniformSource, n: int) -> NDArray        # one rng.uniform() per draw
def exponential_icdf(u: float, rate: float) -> float               # -log1p(-u) / rate
def poisson_arrivals(rate: float, horizon: float, rng: UniformSource) -> NDArray
```

`UniformSource` is anything with `uniform() -> float`: the PCG32 of `M06.3`, or a script of fixed values when checking edge cases. `probs` must be 1-D, finite, non-negative, and sum to 1 within $10^{-9} n + 10^{-12}$; anything else is a `ValueError`, as is $u \notin [0, 1)$.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| inverse CDF by hand | unit | the five draws of section 3 | everyone agrees on strict $<$ |
| alias table by hand | unit | the table encodes $p$; draws follow the rule on the built arrays and on the chapter's table | the contract's draw rule |
| Gumbel-max by hand | unit | the noise values and the winner of section 3 | the trick by hand |
| Poisson arrivals by hand | unit | two arrivals, three uniforms | the schedule `load.01` replays |
| inverse CDF on a grid | statistical | a 1000-point grid of $u$ hits each id $1000 p_i$ times | exact, no noise |
| a zero-probability id | boundary | $u = 0$ with $p_0 = 0$ | a filtered token never leaks |
| a zero tail after rounding | boundary | $u = 1 - 2^{-53}$ when the sums stop below it | spec step 11's fallback |
| the sampling spec's last step | unit | the spec's worked example ends on id 3 | `L8.1` calls this for step 11 |
| bad arguments | boundary | sums other than 1, negatives, NaN, 2-D, bad $u$ | upstream bugs fail loudly |
| Gumbel noise values | unit | $G(e^{-1}) = 0$; $u \in \{0, 1\}$ rejected | the sign of the noise |
| Gumbel-max is softmax | statistical | 20 000 draws fit softmax (chi-square, $p > 10^{-3}$) | the theorem of section 2 |
| masks and ties | boundary | $-\infty$ never wins; ties to the lowest id; invalid logits raise | masks from top-k and grammars |
| the alias table encodes $p$ | property | random $p$ with zeros, $n$ up to 64: $P(i) = p_i$ within $10^{-12}$ | any pairing order is fine |
| alias on a grid | statistical | grid uniforms give exact counts | column and coin from one $u$ |
| alias with a real generator | statistical | 20 000 PCG32 draws fit $p$; the $p = 0$ id never appears | the negative sampler of `L2.3` |
| one uniform per alias draw | unit | 7 draws use 7 uniforms; `n = 0` and `n < 0` | the generator's position is known |
| alias at column edges | boundary | $u = k/n$ exactly never yields a $p = 0$ id | strict coin |
| a single outcome | boundary | $n = 1$ for any $u$ | the smallest table |
| the exponential inverse CDF | unit | $0 \mapsto 0$, $\ln 2 / 2$, tiny $u$ accurate, bad args raise | `log1p`, not `log` |
| arrivals: horizon and draws | boundary | horizon 0, a gap of 0, invalid rate or horizon | $[0, H)$ is half open |
| arrivals: the rate | statistical | count mean and dispersion, gap mean from one long run | the process, not just the formula |

## 5. The solution, step by step

### From first principles

1. **Randomness is an argument.** Section 2: every function takes uniforms (or an `rng` with `uniform()`) and never builds its own generator. That makes each a pure function, so fixed uniforms test the edges and the same seed gives the same draws in every port.
2. **Validate once, with an exact sum.** All samplers share one check: 1-D, non-empty, finite, non-negative, summing to 1. The sum uses `math.fsum` (exactly rounded) and a tolerance that grows with $n$, because a long probability vector accumulates rounding of order $n \varepsilon$ in whatever produced it.
3. **Inverse CDF is a running sum with a strict comparison.** Return the first $i$ with $u < F_i$. Strict $<$ gives a $p_i = 0$ id the empty interval. If the loop ends (rounding left $F_{n-1} \le u$), return the last id with $p_i > 0$, not the last id.
4. **Gumbel-max is argmax of logits plus noise.** The noise is $-\log(-\log u)$ with $u$ strictly inside $(0, 1)$ (both ends make an infinite noise). `np.argmax` returns the first maximum, which is the lowest-id tie rule, and $-\infty + g = -\infty$ masks for free. A row that is all $-\infty$ has nothing to sample and must raise rather than return 0.
5. **Vose builds the alias table in one pass.** Scale by $n$, split into small ($< 1$) and large ($\ge 1$), pair one of each until a list empties, and give the donor $(m_g + m_s) - 1$. Anything left over is a full column up to rounding, so its `prob` is set to 1, which also ensures a leftover column never falls through to an alias.
6. **One uniform per alias draw.** $x = nu$, column $\lfloor x \rfloor$, coin $x - \lfloor x \rfloor$, keep when the coin is strictly below `prob`. Clamp the column to $n - 1$ so a product $nu$ that rounds up to $n$ cannot index past the end.
7. **Arrivals are cumulative exponential gaps.** $t \mathrel{+}= -\mathrm{log1p}(-u)/\lambda$; stop at the first $t \ge H$ and do not keep it, so the result lies in $[0, H)$.

### The code, annotated

```python
def _check_probs(probs: ArrayLike) -> NDArray:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError(f"probs must be a non-empty 1-D array, got shape {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("probs must be finite")
    if (p < 0).any():
        raise ValueError(f"probs must be non-negative, got min {p.min()}")
    total = math.fsum(p.tolist())            # exactly rounded: the check itself adds no error
    if abs(total - 1.0) > 1e-9 * p.size + 1e-12:   # tolerance grows with n
        raise ValueError(f"probs must sum to 1, got {total!r}")
    return p


def sample_categorical(probs: ArrayLike, u: float) -> int:
    p = _check_probs(probs)
    if not 0.0 <= u < 1.0:
        raise ValueError(f"u must lie in [0, 1), got {u!r}")
    c = 0.0
    for i, pi in enumerate(p.tolist()):
        c += pi                              # F_i, accumulated in spec order
        if u < c:                            # strict: id i owns [F_{i-1}, F_i)
            return i
    # Rounding left the running sum at or below u: the last id that can occur.
    return int(np.flatnonzero(p > 0)[-1])
```

The running sum is a plain left-to-right `+=` on purpose: `spec/sampling.md` fixes this op order so the Rust engine reaches the same $F_i$ bit for bit. On section 3's $u = 0.10$, $0.10 < 0.1$ is false and $0.10 < 0.3$ is true, so id 1.

```python
def gumbel_noise(u: ArrayLike) -> NDArray:
    a = np.asarray(u, dtype=np.float64)
    if not ((a > 0.0) & (a < 1.0)).all():    # u = 0 or 1 gives infinite noise
        raise ValueError("gumbel_noise needs every u in the open interval (0, 1)")
    return -np.log(-np.log(a))               # the max-Gumbel; log(-log u) is the min-Gumbel


def gumbel_max(logits: ArrayLike, gumbels: ArrayLike) -> int:
    z = np.asarray(logits, dtype=np.float64)
    g = np.asarray(gumbels, dtype=np.float64)
    if z.ndim != 1 or z.size == 0 or g.shape != z.shape:
        raise ValueError(...)            # message elided
    if np.isnan(z).any() or (z == np.inf).any():
        raise ValueError("logits must not be NaN or +inf")
    if (z == -np.inf).all():
        raise ValueError("every logit is -inf: there is nothing to sample")
    # -inf + g stays -inf, so a masked id never wins; argmax takes the first maximum.
    return int(np.argmax(z + g))
```

```python
class AliasTable:
    def __init__(self, probs: ArrayLike) -> None:
        p = _check_probs(probs)
        n = p.size
        scaled = (p * n).tolist()  # mean 1: a column holds exactly one unit of mass
        self.prob = np.zeros(n, dtype=np.float64)
        self.alias = np.arange(n, dtype=np.int64)
        small = [i for i in range(n) if scaled[i] < 1.0]
        large = [i for i in range(n) if scaled[i] >= 1.0]
        while small and large:
            s, g = small.pop(), large.pop()
            # Column s keeps its own mass and is topped up from g.
            self.prob[s] = scaled[s]
            self.alias[s] = g
            scaled[g] = (scaled[g] + scaled[s]) - 1.0   # g gave away 1 - m_s
            (small if scaled[g] < 1.0 else large).append(g)
        # What is left holds one unit each up to rounding: keep the column whole.
        for i in large + small:
            self.prob[i] = 1.0

    def sample(self, rng: UniformSource, n: int) -> NDArray:
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        k = len(self)
        # Exactly n uniforms, in order: the generator's position after the call is known.
        u = np.fromiter((rng.uniform() for _ in range(n)), dtype=np.float64, count=n)
        x = u * k
        i = np.minimum(np.floor(x).astype(np.int64), k - 1)   # column; clamp guards rounding
        f = x - i                                             # coin: uniform on [0, 1), independent of i
        return np.where(f < self.prob[i], i, self.alias[i]).astype(np.int64)
```

Tracing section 3: small $= [0, 4]$ and large $= [1, 2, 3]$, so the first `pop` pair is $(4, 3)$, then $(0, 3)$, then $(3, 2)$, exactly the table there; ids 1 and 2 remain in `large` and get `prob = 1`. A small id left over when `large` empties (its mass is $1 - \epsilon$ from rounding) also gets 1, so it is never redirected to a meaningless alias. The parenthesization `(scaled[g] + scaled[s]) - 1.0` is fixed so every port computes the same leftover.

```python
def exponential_icdf(u: float, rate: float) -> float:
    if not 0.0 <= u < 1.0:
        raise ValueError(f"u must lie in [0, 1), got {u!r}")
    if not rate > 0:
        raise ValueError(f"rate must be > 0, got {rate!r}")
    return -math.log1p(-u) / rate          # -log(1 - u) / rate, accurate for tiny u; u = 0 gives 0


def poisson_arrivals(rate: float, horizon: float, rng: UniformSource) -> NDArray:
    # (argument checks elided)
    times: list[float] = []
    t = 0.0
    while True:
        t += exponential_icdf(rng.uniform(), rate)   # one uniform per gap
        if t >= horizon:                             # the crossing arrival is not kept
            return np.asarray(times, dtype=np.float64)
        times.append(t)
```

With $\lambda = 2$, $H = 2$ and uniforms $0.5, 0.75, 0.9$, $t$ goes $0.3466, 1.0397, 2.1910$; the third crosses $H$ and the call returns two times after three uniforms.

Full solution: [`sampling.py`](../../../../course/solutions/python/tinyllm/prob/sampling.py).

### Why this solution

- **Cost.** `sample_categorical` is $O(n)$ per draw (the walk) with no extra memory; `gumbel_max` is $O(n)$ and needs $n$ uniforms; `AliasTable` builds in $O(n)$ time and memory and samples in $O(1)$ per draw from one uniform; `poisson_arrivals` costs one uniform per arrival plus one.
- **Linear walk over binary search.** A `searchsorted` over a stored cumulative sum is $O(\log n)$, but the sampler's distribution changes every token, so the cumulative sum is rebuilt each time anyway ($O(n)$), and the spec fixes a left-to-right accumulation so that two languages agree on $F_i$ exactly.
- **Vose over the naive alias build.** Walker's original sorts or scans for the smallest and largest each step, $O(n \log n)$ or $O(n^2)$. Vose's two worklists make each step $O(1)$ and the whole build $O(n)$.
- **Exceptions over sentinels.** A fully masked row, an unnormalized vector, or $u = 1$ is an upstream bug; returning id 0 would make a filtered token appear in generated text. Raising keeps those bugs visible.
- **Tradeoffs accepted.** Gumbel-max uses $n$ uniforms per draw, more than the inverse CDF's one, in exchange for needing no normalization; the alias table spends $O(n)$ memory and setup to make each draw $O(1)$, which only pays off for many draws from one fixed distribution.

### Explain it back

**1. Why does `sample_categorical` fall back to the last id with $p_i > 0$ rather than the last id?**

<details><summary>Answer</summary>

The fallback runs only when rounding left the running sum at or below $u$ (for example ten additions of 0.1 give $0.9999999999999999$ and $u = 1 - 2^{-53}$). The true answer is then the id owning the end of $[0, 1)$, which is the last id with positive mass. If the tail ids have $p = 0$ (filtered tokens), returning the last id would emit a token the sampler had removed.

</details>

**2. In `AliasTable.sample`, why is the coin `x - i` and not a second `rng.uniform()`?**

<details><summary>Answer</summary>

Given the column $i = \lfloor nu \rfloor$, the fractional part $nu - i$ is uniform on $[0, 1)$ and independent of $i$, so it is a valid coin. Using it keeps the cost at one uniform per draw, which fixes the generator's position after `sample(rng, n)` to exactly $n$ draws, matching every other port. A second uniform would give the right distribution but different draws from the same seed.

</details>

**3. Why must the coin test be `f < self.prob[i]` and not `<=`?**

<details><summary>Answer</summary>

A column with $\mathrm{prob}_i = 0$ belongs to an id with $p_i = 0$ and must always redirect to its alias. At $u = k/n$ exactly, the coin is $f = 0$, and `0 <= 0` would keep the column and emit the impossible id. Strict $<$ is the same rule as the inverse CDF's: an empty interval contains nothing.

</details>

**4. Why does `gumbel_max` raise when every logit is $-\infty$ but accept some $-\infty$ entries?**

<details><summary>Answer</summary>

A $-\infty$ logit is a mask: $-\infty + g = -\infty$, so it never wins as long as some id is finite. If all are $-\infty$, every sum is $-\infty$ and `np.argmax` would return 0, a token the mask forbade. There is no distribution to sample, so the only honest answer is an error.

</details>

**5. Why `-math.log1p(-u)` and not `-math.log(u)`, when $U$ and $1 - U$ have the same distribution?**

<details><summary>Answer</summary>

The distribution is the same but the schedule from a given seed is not: `load.01` reimplements this in Go and must produce the same arrival times. And $u = 0$ is a legal uniform: $-\log(0) = \infty$, while $-\mathrm{log1p}(0) = 0$. `log1p` also keeps full relative precision for tiny $u$, where $1 - u$ would round.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. `u <= c` (or `searchsorted(..., side="left")`) | $u = 0$ returns an id with $p = 0$ | `if u < c`; step 3 |
| 2. falling back to the last id | a zero-probability tail id comes out when rounding leaves the sums below $u$ | `np.flatnonzero(p > 0)[-1]`; step 3 |
| 3. adding the noise to probabilities, or with the wrong sign | a sampler that looks random and has the wrong distribution | `z + g` on logits, with $g = -\log(-\log u)$; step 4 |
| 4. `argmax` over a reversed array, or no check for all $-\infty$ | ties go to the highest id; a fully masked row returns id 0 | `np.argmax` takes the first maximum; the all-$-\infty$ row raises; step 4 |
| 5. coin `f <= prob` | at a column edge a zero-probability column is kept | `f < self.prob[i]`; step 6 |
| 6. two uniforms per alias draw | correct distribution, different draws from the same seed, and the generator ends in the wrong place | the coin is the fractional part of `u * k`; step 6 |
| 7. $-\log(u)$ for the exponential | same distribution, different schedule; $u = 0$ gives infinity | `-math.log1p(-u) / rate`; step 7 |
| 8. columns not scaled by $n$, the donor's leftover as $m_g - m_s$, or the coin read from the alias column | the table encodes some other distribution | `scaled = p * n`, `(scaled[g] + scaled[s]) - 1.0`, and `self.prob[i]` indexed by the drawn column; step 5 |
| 9. keeping the arrival that crossed the horizon | one extra request per window | returns before `times.append(t)` once `t >= horizon`; step 7 |
| 10. not checking that `probs` sums to 1 | an unnormalized row samples as if its tail were empty | `_check_probs` with `math.fsum`; step 2 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M06.3` | PCG32's `uniform()` is the `rng` every sampler here reads |
| Back | `M07.0` | random variables, the uniform distribution, and CDFs |
| Back | `M00.1` | $\log$ and $\exp$ in the Gumbel noise and the exponential |
| Forward | `L8.1` | the sampler's step 11 is `sample_categorical(q, rng.uniform())` |
| Forward | `L2.3` | word2vec draws negatives from `AliasTable(unigram ** 0.75)` |
| Forward | `L6.2` | BERT's 80/10/10 masking draws the replacement with `sample_categorical` |
| Forward | `L6.3` | a later call site of the same samplers |
| Forward | `M07.4`, `M07.6` | the bootstrap's resampling and speculative decoding's residual draw |
| Forward | `load.01` | re-implements `poisson_arrivals` in Go from the same rule (a port, not a call) |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `gumbel_max` | vLLM's sampler | the exponential race: `probs / Exp(1)` then argmax, the same theorem without logs, batched on GPU | `vllm/v1/sample/ops/topk_topp_sampler.py` (`random_sample`) |
| `sample_categorical` | numpy `Generator.choice(p=...)` | a cumulative sum and a binary search (`searchsorted`) per batch of uniforms | `numpy/random/_generator.pyx` |
| `AliasTable` | word2vec's unigram table | a quantized inverse CDF: an array of $10^8$ ids filled in proportion to $p^{0.75}$, one index per draw | `word2vec.c` (`InitUnigramTable`) |
