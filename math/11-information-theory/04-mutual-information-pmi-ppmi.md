<!-- ol:module M11.4 -->
# Mutual information, PMI, and PPMI

## Overview

| | |
|---|---|
| **Module** | `M11.4` · build · Python · Pass 3 · 2 to 3 h |
| **You build** | `python/tinyllm/info/pmi.py`: `mutual_information(joint)` (nats, as the KL divergence from the joint to the product of its marginals), `pmi_matrix(cooc, cds_alpha)` (pointwise mutual information with context distribution smoothing), and `ppmi(cooc, cds_alpha)` (its positive part) |
| **Contract** | [`course/contracts/py/tinyllm/info/pmi.pyi`](../../course/contracts/py/tinyllm/info/pmi.pyi) |
| **Tests** | `course/tests/M11.4/test_pmi.py` (what they check: section 4), golden values from `course/oracle/M11.4/pmi_golden.py` (mpmath, 50 digits) in `course/fixtures/M11.4/pmi_golden.json` |
| **Needs** | `M11.1` `kl` (or `--ref-deps`). Reading: `M07.2` (probabilities as count ratios), `S-M11a` |
| **Used by** | later `L2.3` the PPMI-SVD word vectors that word2vec is compared with (it joins the registry with its batch) |
| **Milestone** | `MS-P3` (the tokens-and-data gate) |
| **Optional depth** | Cover and Thomas, *Elements of Information Theory*, ch. 2.4 to 2.6; Church and Hanks, "Word association norms, mutual information, and lexicography" (1990); Levy, Goldberg, and Dagan, "Improving distributional similarity with lessons learned from word embeddings" (TACL 2015) |

## Key Takeaways

- Mutual information $I(X; Y)$ is how many nats knowing $X$ saves about $Y$: the KL divergence from the joint distribution to the product of its marginals, zero exactly when $X$ and $Y$ are independent (`test_hand_example_mutual_information`, `test_mutual_information_properties`).
- PMI is the log ratio inside that sum for one pair, positive when the pair co-occurs more than chance; mutual information is its average under the joint (`test_mutual_information_is_expected_pmi`).
- Context distribution smoothing raises context counts to $\alpha = 0.75$ and renormalizes **over contexts**, which takes PMI away from rare contexts (`test_cds_lowers_rare_context_pmi`, `test_golden_tables`).
- PPMI clips at zero, so unseen pairs ($-\infty$) and below-chance pairs become exact zeros: a finite, sparse, non-negative matrix an SVD can factor (`test_ppmi_is_finite_nonnegative_and_quiet`).

## How to work this chapter

```bash
ol start M11.4              # stubs python/tinyllm/info/pmi.py into your repo
ol tests M11.4              # read the test catalog first: rung R0, you write no tests here
ol check M11.4              # exit code is the verdict
ol check M11.4 --ref-deps   # only if your M11.1 is not passing yet
ol diff  M11.4              # after passing: your code against the reference
```

---

## 1. Why now

`L2.3` asks where word vectors come from. Before word2vec, the answer was to count: build a table of how often each word appears near each context word, then factor it. Raw counts are dominated by frequent words ("the" co-occurs with everything), so the table first has to say not "how often" but "how much more often than chance". That ratio, in logs, is pointwise mutual information, and its average is the mutual information between a word and its context. Levy and Goldberg showed that skip-gram with negative sampling implicitly factorizes a shifted PMI matrix, so this module is also the yardstick for the neural model you train next. Get the marginals, the smoothing, and the zeros wrong here and the baseline `L2.3` compares against is wrong too.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\operatorname{cooc}[w, c]$ | co-occurrence count of word $w$ with context $c$ | `float64[W, C]` |
| $D$ | total count, $\sum_{w, c} \operatorname{cooc}[w, c]$ | scalar |
| $\#(w)$, $\#(c)$ | row sum and column sum | scalars |
| $P(w, c) = \operatorname{cooc}[w, c] / D$ | joint probability (the MLE, `M07.2`) | scalar |
| $P(w) = \#(w)/D$, $P(c) = \#(c)/D$ | marginal probabilities | scalars |
| $X, Y$ | two discrete random variables with joint $P(x, y)$ | |
| $H(X)$, $H(X, Y)$ | entropy and joint entropy (`M11.1`), in nats | scalars |
| $I(X; Y)$ | mutual information, in nats | scalar $\ge 0$ |
| $\operatorname{PMI}(w, c)$ | pointwise mutual information | scalar, possibly $-\infty$ |
| $\alpha$ | context distribution smoothing exponent, $0 < \alpha \le 1$ | scalar |
| $P_\alpha(c)$ | smoothed context distribution | scalar |

### 2.1 Mutual information

If $X$ and $Y$ were independent, their joint distribution would be the product of the marginals, $P(x)P(y)$. **Mutual information** measures how far the real joint is from that:

$$I(X; Y) = \sum_{x, y} P(x, y) \ln \frac{P(x, y)}{P(x) P(y)} = D_{KL}\big(P_{XY} \,\Vert\, P_X P_Y\big).$$

Splitting the logarithm gives $I(X; Y) = H(X) + H(Y) - H(X, Y) = H(Y) - H(Y \mid X)$: the reduction in uncertainty about $Y$ from learning $X$ (S-M11b q7 asks you to prove the identity). Gibbs' inequality makes it non-negative, zero exactly when $X$ and $Y$ are independent. It is symmetric, $I(X; Y) = I(Y; X)$, even though KL is not; and since conditioning can remove at most all of the uncertainty, $I(X; Y) \le \min(H(X), H(Y))$.

The reference computes it the way the definition reads: normalize the table, take its marginals, and call `M11.1`'s `kl` on the flattened joint and the flattened outer product of the marginals. `kl` already handles $0 \ln 0 = 0$, and wherever $P(x, y) > 0$ both marginals are positive, so the product never rules out a pair the joint produces. A table of raw counts is normalized first, so counts and probabilities give the same answer.

### 2.2 Pointwise mutual information

The log ratio inside the sum, for one pair, is the **pointwise mutual information**:

$$\operatorname{PMI}(w, c) = \ln \frac{P(w, c)}{P(w) P(c)} = \ln \frac{\operatorname{cooc}[w, c] \cdot D}{\#(w)\, \#(c)} .$$

It is positive when $w$ and $c$ appear together more often than independence predicts, zero at chance, negative below chance, and $-\infty$ for a pair never seen ($\ln 0$). Mutual information is the average PMI under the joint: $I = \sum_{w, c} P(w, c) \operatorname{PMI}(w, c)$, where unseen pairs carry weight zero and contribute nothing.

### 2.3 Context distribution smoothing

PMI is biased toward rare contexts: a context seen twice, once with $w$, gets a huge ratio from tiny counts. Levy, Goldberg, and Dagan borrowed word2vec's fix, raising context counts to $\alpha = 0.75$ before normalizing **over contexts**:

$$P_\alpha(c) = \frac{\#(c)^\alpha}{\sum_{c'} \#(c')^\alpha}, \qquad \operatorname{PMI}_\alpha(w, c) = \ln P(w, c) - \ln P(w) - \ln P_\alpha(c) .$$

Since $t^\alpha$ grows slower than $t$, rare contexts get a larger share of the probability than their counts alone would give, so their PMI drops and frequent contexts' PMI rises. Three details matter. The smoothing applies to the **context** marginal only; the word marginal keeps its plain MLE. The denominator is $\sum_{c'} \#(c')^\alpha$, not $D$: dividing by $D$ leaves numbers that do not sum to 1 and shifts every PMI by the same constant. And $\alpha = 1$ is plain PMI; the contract allows $0 < \alpha \le 1$, because at $\alpha = 0$ every context, even an unused one ($0^0 = 1$), would get the same weight.

### 2.4 Positive PMI

Negative PMI values are unreliable (they need many observations to tell "below chance" from "not yet seen"), and $-\infty$ cannot go into an SVD. **PPMI** keeps only the positive part:

$$\operatorname{PPMI}(w, c) = \max(\operatorname{PMI}(w, c), 0),$$

so unseen pairs and below-chance pairs become exact zeros. The result is finite, non-negative, and mostly zeros, the matrix `L2.3` factors with `M03.5`'s SVD. A row or column that is entirely zero (a word never seen, a context never used) must come out as zeros too, with no divide-by-zero warnings: the reference computes the logs under `np.errstate(divide="ignore")`, takes the marginal logs only where the marginals are positive, and writes $-\infty$ wherever the count is 0.

## 3. Worked example by hand

Two words and three contexts:

| | pet | bark | meow | $\#(w)$ |
|---|---|---|---|---|
| cat | 2 | 0 | 2 | 4 |
| dog | 2 | 4 | 0 | 6 |
| $\#(c)$ | 4 | 4 | 2 | $D = 10$ |

**Plain PMI** ($\alpha = 1$):

| Pair | $P(w, c)$ | $P(w)P(c)$ | PMI | PPMI |
|---|---|---|---|---|
| cat, pet | 0.2 | $0.4 \cdot 0.4 = 0.16$ | $\ln 1.25 = 0.2231$ | 0.2231 |
| cat, bark | 0 | 0.16 | $-\infty$ | 0 |
| cat, meow | 0.2 | $0.4 \cdot 0.2 = 0.08$ | $\ln 2.5 = 0.9163$ | 0.9163 |
| dog, pet | 0.2 | $0.6 \cdot 0.4 = 0.24$ | $\ln(5/6) = -0.1823$ | 0 |
| dog, bark | 0.4 | $0.6 \cdot 0.4 = 0.24$ | $\ln(5/3) = 0.5108$ | 0.5108 |
| dog, meow | 0 | 0.12 | $-\infty$ | 0 |

Dog and pet co-occur, but less than chance (dogs are mostly about barking here), so PPMI drops the pair. This is `test_hand_example_pmi`.

**Mutual information** is the average PMI under the joint:
$0.2 \cdot 0.2231 + 0.2 \cdot 0.9163 + 0.2 \cdot (-0.1823) + 0.4 \cdot 0.5108 = 0.3958$ nats (to 16 digits, $0.3957527947852783$). This is `test_hand_example_mutual_information`.

**Smoothing** with $\alpha = 0.75$: $4^{0.75} = 2.8284$, $2^{0.75} = 1.6818$, so $\sum_c \#(c)^{0.75} = 2.8284 + 2.8284 + 1.6818 = 7.3386$ and $P_{0.75}(\text{meow}) = 1.6818 / 7.3386 = 0.2292$, up from 0.2. Then $\operatorname{PMI}_{0.75}(\text{cat}, \text{meow}) = \ln \frac{0.2}{0.4 \cdot 0.2292} = \ln 2.182 = 0.7801$, down from 0.9163: the rare context lost PMI. The frequent context pet went the other way, $P_{0.75}(\text{pet}) = 0.3854 < 0.4$ and $\operatorname{PMI}_{0.75}(\text{cat}, \text{pet}) = 0.2603 > 0.2231$. This is `test_cds_lowers_rare_context_pmi`.

## 4. The interface

```python
def mutual_information(joint: ArrayLike) -> float:
    """I(X; Y) in nats = kl(joint, outer(marginals)), the table normalized first."""

def pmi_matrix(cooc: ArrayLike, cds_alpha: float = 0.75) -> NDArray:
    """ln P(w, c) - ln P(w) - ln P_alpha(c); -inf where cooc == 0."""

def ppmi(cooc: ArrayLike, cds_alpha: float = 0.75) -> NDArray:
    """max(pmi_matrix(cooc, cds_alpha), 0): finite and non-negative."""
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_pmi` | unit, smoke | section 3's PMI and PPMI table at $\alpha = 1$ | you and the tests agree on the definition |
| `test_hand_example_mutual_information` | unit, smoke | $0.3957527947852783$ nats, from counts and from probabilities | nats, and normalization first |
| `test_golden_tables` | golden | five tables (random, quarter counts, independent, one row) at $\alpha = 1, 0.75, 0.5$ against 50-digit mpmath values | the smoothing renormalizes over contexts |
| `test_mutual_information_properties` | property | 0 for an outer product, symmetric under transpose, $0 \le I \le \min(H(X), H(Y))$ | what mutual information means |
| `test_mutual_information_is_expected_pmi` | property | $I = \sum P \cdot \operatorname{PMI}$ over seen pairs | PPMI is the pointwise version of $I$ |
| `test_cds_lowers_rare_context_pmi` | property | the rarest context loses PMI, the most frequent gains | why $\alpha = 0.75$ |
| `test_ppmi_is_finite_nonnegative_and_quiet` | boundary | an all-zero row and column: zeros, no NaN, no warnings; PMI is $-\infty$ exactly where the count is 0 | the SVD in `L2.3` takes this matrix |
| `test_rejects_bad_tables_and_alpha` | boundary | negative, NaN, 1-D, and all-zero tables; $\alpha \le 0$ or $> 1$ | caller bugs fail early |
| `test_integer_counts_and_inputs_untouched` | boundary | int64 counts give float64 results equal to the float ones; the caller's table is unchanged | `L2.3` reuses its count table |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. taking the context marginal from the rows (a transposed marginal) | wrong PMI everywhere; a shape error on non-square tables | `test_hand_example_pmi`, `test_golden_tables` (mutant `s01`) |
| 2. dividing the smoothed context counts by $D$ instead of $\sum_{c'} \#(c')^\alpha$ | every $\operatorname{PMI}_\alpha$ shifted by the same constant | `test_golden_tables`, `test_cds_lowers_rare_context_pmi` (mutant `s02`) |
| 3. PPMI as $\lvert \operatorname{PMI} \rvert$ | below-chance pairs count as associations; unseen pairs become $+\infty$ | `test_ppmi_is_finite_nonnegative_and_quiet` (mutant `s03`) |
| 4. mutual information in bits | values $1/\ln 2 = 1.44$ times too large next to every other number in the course | `test_hand_example_mutual_information` (mutant `s04`) |
| 5. KL of raw counts, not normalized | a number that scales with the corpus size | `test_hand_example_mutual_information`, `test_mutual_information_properties` (mutant `s05`) |
| 6. smoothing unseen pairs with $\ln(\varepsilon)$ instead of $-\infty$ | a finite but huge negative PMI that leaks into averages | `test_ppmi_is_finite_nonnegative_and_quiet` (mutant `s06`) |
| accepting $\alpha \le 0$ | unused contexts get probability through $0^0 = 1$ | `test_rejects_bad_tables_and_alpha` (mutant `s07`) |
| normalizing the caller's table in place | the counts `L2.3` keeps become probabilities | `test_integer_counts_and_inputs_untouched` (mutant `s08`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M11.1` | `mutual_information` is `kl(joint, outer(p_x, p_y))`, with its $0 \ln 0$ conventions |
| Back | `M07.2` | $P(w, c)$, $P(w)$, $P(c)$ are maximum-likelihood ratios of counts (reading) |
| Forward | `L2.3` | `ppmi_svd_embeddings` factors `ppmi(cooc)` with `M03.5`'s SVD, the count-based baseline for skip-gram |
| Forward | `S-M11b` | mutual information and PMI by hand, and the identity $I = H(X) + H(Y) - H(X, Y)$ |

If you skip this module, `L2.3` stops with `BLOCKED ... needs M11.4` once it lands: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `ppmi` on a dense table | Hyperwords (Levy and Goldberg) | sparse co-occurrence counting with dynamic windows and subsampling, PPMI, SVD with eigenvalue weighting | `hyperwords/representations/explicit.py` |
| `pmi_matrix` | gensim `Phrases` (NPMI scorer) | normalized PMI, $\operatorname{PMI} / (-\ln P(w, c))$ in $[-1, 1]$, to find collocations such as "new_york" | `gensim/models/phrases.py` |
| `mutual_information` | scikit-learn `mutual_info_score` | the same sum from two label arrays via a contingency table; `adjusted_mutual_info_score` corrects for chance | `sklearn/metrics/cluster/_supervised.py` |
| PMI by counting | SGNS (word2vec) | the same matrix, shifted by $\ln k$, factorized implicitly by gradient descent | Levy and Goldberg, NeurIPS 2014 |
