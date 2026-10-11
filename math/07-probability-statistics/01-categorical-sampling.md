<!-- ol:module M07.1 -->
# Categorical sampling: inverse CDF, Gumbel-max, alias method

## Overview

| | |
|---|---|
| **Module** | `M07.1` · build · Python · Pass 3 · 3 to 4 h |
| **You build** | `python/tinyllm/prob/sampling.py`: `sample_categorical`, `gumbel_noise`, `gumbel_max`, `AliasTable` (`prob`, `alias`, `sample`), `exponential_icdf`, `poisson_arrivals` |
| **Contract** | [`course/contracts/py/tinyllm/prob/sampling.pyi`](../../course/contracts/py/tinyllm/prob/sampling.pyi) · the sampler's op order: [`spec/sampling.md`](../../course/contracts/spec/sampling.md) · the generator: [`spec/pcg32.md`](../../course/contracts/spec/pcg32.md) |
| **Tests** | `course/tests/M07.1/` (what they check: section 4) |
| **Needs** | no code from earlier modules · reading: `M06.3` PCG32 (your `rng`), `M07.0` random variables and the uniform, `M00.1` logs |
| **Used by** | later: `L8.1` the sampler's step 11 · `L2.3` word2vec negative sampling · `L6.2` BERT's 80/10/10 masking · `M07.4` and `M07.6` resampling · `load.01` re-implements the Poisson schedule in Go · later: `L6.3` |
| **Milestone** | `MS-P3` (tokens and data) |
| **Optional depth** | Devroye, *Non-Uniform Random Variate Generation* (1986), ch. 2 and 3; Vose, "A Linear Algorithm for Generating Random Numbers with a Given Distribution" (1991) |

## Key Takeaways

- The inverse CDF turns one uniform $u$ into id $i$ exactly when $F_{i-1} \le u < F_i$, an interval of length $p_i$; strict $<$ keeps zero-probability ids out (`test_hand_example_inverse_cdf`, `test_inverse_cdf_exact_enumeration`).
- Adding independent Gumbel noise $-\log(-\log U)$ to logits and taking the argmax samples $\mathrm{softmax}$ exactly, with no normalization and no running sum (`test_gumbel_max_distribution_is_softmax`).
- The alias method spends $O(n)$ once to split the distribution into $n$ equal columns of two ids each, then draws in $O(1)$ from one uniform (`test_alias_table_encodes_the_distribution`, `test_alias_sample_chi_square`).
- Every sampler here is a deterministic function of its uniforms, so the same seed gives the same draws in Python, Rust, and Go (`test_alias_one_uniform_per_draw`, `test_hand_example_poisson_arrivals`).

## How to work this chapter

```bash
ol start M07.1              # stubs sampling.py into your repo
ol tests M07.1              # read the test catalog first
ol check M07.1              # exit code is the verdict
ol diff  M07.1              # after passing: your code against the reference
```

---

## 1. Why now

Your tracer samples text with three lines buried inside `BigramLM.sample`: a numpy generator, a cumulative sum, a search. Pass 3 needs sampling in places that line cannot serve. word2vec (`L2.3`) draws millions of negative words from a 50 000-word distribution, and a cumulative sum per draw is 50 000 additions each time. BERT's masking (`L6.2`) and the bootstrap (`M07.4`) need many small draws with a known generator position. The sampler (`L8.1`) must return the same token as your Rust engine (`L10.1`) for the same seed, which only works if the last step, uniform to id, is pinned down to the comparison (`spec/sampling.md`, step 11). And the load generator (`load.01`) needs request arrival times that look like real traffic. This module turns uniforms into draws three exact ways, each with a different cost, and makes each one a pure function of its uniforms.

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

**Randomness comes from outside.** None of these functions creates a generator. They take a uniform $u$, or an `rng` with a `uniform()` method (your PCG32 from `M06.3`, which turns two 32-bit draws into one float64 in $[0, 1)$). So each is a plain function from uniforms to outcomes: the same uniforms give the same outcomes in every language, and a test can feed chosen uniforms to check edge cases that a real generator would hit once in $2^{53}$ draws.

**Inverse CDF (discrete).** Lay the probabilities end to end on $[0, 1)$: id $i$ owns the interval $[F_{i-1}, F_i)$, whose length is $p_i$. Draw $u$ and return the id whose interval contains it, the first $i$ with $u < F_i$. Then

$$P(\text{return } i) = P(F_{i-1} \le U < F_i) = F_i - F_{i-1} = p_i.$$

Two details decide correctness. The comparison is strict: an id with $p_i = 0$ owns the empty interval $[F_{i-1}, F_{i-1})$, and with $u \le F_i$ instead, $u = 0$ would return id 0 even when $p_0 = 0$. And floating point: ten additions of $0.1$ give $0.9999999999999999$, so the largest uniform, $1 - 2^{-53}$, is not below any running sum. Then the answer is the last id with $p_i > 0$, never simply the last id. The walk costs $O(n)$ per draw; binary search over a stored $F$ makes it $O(\log n)$.

**Inverse CDF (continuous).** The same idea works for any increasing continuous CDF $F$: if $U$ is uniform then $X = F^{-1}(U)$ has $P(X \le x) = P(U \le F(x)) = F(x)$. The exponential distribution has $F(t) = 1 - e^{-\lambda t}$, so

$$t = F^{-1}(u) = -\frac{\log(1 - u)}{\lambda}.$$

Write it with `log1p(-u)`, which is accurate when $u$ is tiny, and note $u = 0$ gives $t = 0$, never $\log 0$. (Using $-\log(u)/\lambda$ is the same distribution, since $1 - U$ is uniform too, but a different schedule from the same seed, and $u = 0$ breaks it.)

**Poisson arrivals.** Requests that arrive independently at an average rate $\lambda$ have independent $\mathrm{Exp}(\lambda)$ gaps. Adding gaps until the time passes the horizon $H$ gives the arrival times on $[0, H)$; their count is Poisson with mean $\lambda H$ and variance $\lambda H$. This is the open-loop schedule `load.01` replays against your gateway.

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

These numbers are the first cases in section 4: `test_hand_example_inverse_cdf`, `test_hand_example_alias_table`, `test_hand_example_gumbel_max`, and `test_hand_example_poisson_arrivals`.

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

`UniformSource` is anything with `uniform() -> float`: your PCG32, or in the tests the frozen one, or a script of fixed values. `probs` must be 1-D, finite, non-negative, and sum to 1 within $10^{-9} n + 10^{-12}$; anything else is a `ValueError`, as is $u \notin [0, 1)$.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_inverse_cdf` | unit | the five draws of section 3 | you and the test agree on strict $<$ |
| `test_hand_example_alias_table` | unit | your table encodes $p$; draws follow the rule on your arrays and on the chapter's table | the contract's draw rule |
| `test_hand_example_gumbel_max` | unit | the noise values and the winner of section 3 | the trick by hand |
| `test_hand_example_poisson_arrivals` | unit | two arrivals, three uniforms | the schedule `load.01` replays |
| `test_inverse_cdf_exact_enumeration` | statistical | a 1000-point grid of $u$ hits each id $1000 p_i$ times | exact, no noise |
| `test_zero_probability_is_never_returned` | boundary | $u = 0$ with $p_0 = 0$ | a filtered token never leaks |
| `test_rounding_fallback_skips_a_zero_tail` | boundary | $u = 1 - 2^{-53}$ when the sums stop below it | spec step 11's fallback |
| `test_last_step_of_the_sampling_spec` | unit | the spec's worked example ends on id 3 | `L8.1` calls this for step 11 |
| `test_rejects_bad_arguments` | boundary | sums other than 1, negatives, NaN, 2-D, bad $u$ | upstream bugs fail loudly |
| `test_gumbel_noise_values` | unit | $G(e^{-1}) = 0$; $u \in \{0, 1\}$ rejected | the sign of the noise |
| `test_gumbel_max_distribution_is_softmax` | statistical | 20 000 draws fit softmax (chi-square, $p > 10^{-3}$) | the theorem of section 2 |
| `test_gumbel_max_masks_and_ties` | boundary | $-\infty$ never wins; ties to the lowest id; invalid logits raise | masks from top-k and grammars |
| `test_alias_table_encodes_the_distribution` | property | random $p$ with zeros, $n$ up to 64: $P(i) = p_i$ within $10^{-12}$ | any pairing order is fine |
| `test_alias_exact_enumeration` | statistical | grid uniforms give exact counts | column and coin from one $u$ |
| `test_alias_sample_chi_square` | statistical | 20 000 PCG32 draws fit $p$; the $p = 0$ id never appears | the negative sampler of `L2.3` |
| `test_alias_one_uniform_per_draw` | unit | 7 draws use 7 uniforms; `n = 0` and `n < 0` | the generator's position is known |
| `test_alias_zero_probability_never_drawn_at_column_edges` | boundary | $u = k/n$ exactly | strict coin |
| `test_alias_single_outcome` | boundary | $n = 1$ for any $u$ | the smallest table |
| `test_exponential_icdf` | unit | $0 \mapsto 0$, $\ln 2 / 2$, tiny $u$ accurate, bad args raise | `log1p`, not `log` |
| `test_poisson_arrivals_horizon_and_draws` | boundary | horizon 0, a gap of 0, invalid rate or horizon | $[0, H)$ is half open |
| `test_poisson_arrivals_rate` | statistical | count mean and dispersion, gap mean from one long run | the process, not just the formula |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. `u <= c` (or `searchsorted(..., side="left")`) | $u = 0$ returns an id with $p = 0$ | `test_zero_probability_is_never_returned` (mutant `s01`) |
| 2. falling back to the last id | a zero-probability tail id comes out when rounding leaves the sums below $u$ | `test_rounding_fallback_skips_a_zero_tail` (mutant `s02`) |
| 3. adding the noise to probabilities, or with the wrong sign | a sampler that looks random and has the wrong distribution | `test_gumbel_max_distribution_is_softmax` (mutants `s03`, `s04`) |
| 4. `argmax` over a reversed array, or no check for all $-\infty$ | ties go to the highest id; a fully masked row returns id 0 | `test_gumbel_max_masks_and_ties` (mutants `s12`, `s13`) |
| 5. coin `f <= prob` | at a column edge a zero-probability column is kept | `test_alias_zero_probability_never_drawn_at_column_edges` (mutant `s08`) |
| 6. two uniforms per alias draw | correct distribution, different draws from the same seed, and the generator ends in the wrong place | `test_alias_one_uniform_per_draw` (mutant `s09`) |
| 7. $-\log(u)$ for the exponential | same distribution, different schedule; $u = 0$ gives infinity | `test_exponential_icdf` (mutant `s10`) |
| 8. columns not scaled by $n$, the donor's leftover as $m_g - m_s$, or the coin read from the alias column | the table encodes some other distribution | `test_alias_table_encodes_the_distribution`, `test_hand_example_alias_table` (mutants `s05`, `s06`, `s07`) |
| 9. keeping the arrival that crossed the horizon | one extra request per window | `test_hand_example_poisson_arrivals` (mutant `s11`) |
| 10. not checking that `probs` sums to 1 | an unnormalized row samples as if its tail were empty | `test_rejects_bad_arguments` (mutant `s14`) |

## 6. Where it's used next
| Forward | `L6.3` | Registered call site uses this module. |
| Forward | `M07.6` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M06.3` | PCG32's `uniform()` is the `rng` every sampler here reads |
| Back | `M07.0` | random variables, the uniform distribution, and CDFs |
| Back | `M00.1` | $\log$ and $\exp$ in the Gumbel noise and the exponential |
| Forward | `L8.1` | the sampler's step 11 is `sample_categorical(q, rng.uniform())` |
| Forward | `L2.3` | word2vec draws negatives from `AliasTable(unigram ** 0.75)` |
| Forward | `L6.2` | BERT's 80/10/10 masking draws the replacement with `sample_categorical` |
| Forward | `M07.4`, `M07.6` | the bootstrap's resampling and speculative decoding's residual draw |
| Forward | `load.01` | re-implements `poisson_arrivals` in Go from the same rule (a port, not a call) |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `gumbel_max` | vLLM's sampler | the exponential race: `probs / Exp(1)` then argmax, the same theorem without logs, batched on GPU | `vllm/v1/sample/ops/topk_topp_sampler.py` (`random_sample`) |
| `sample_categorical` | numpy `Generator.choice(p=...)` | a cumulative sum and a binary search (`searchsorted`) per batch of uniforms | `numpy/random/_generator.pyx` |
| `AliasTable` | word2vec's unigram table | a quantized inverse CDF: an array of $10^8$ ids filled in proportion to $p^{0.75}$, one index per draw | `word2vec.c` (`InitUnigramTable`) |
