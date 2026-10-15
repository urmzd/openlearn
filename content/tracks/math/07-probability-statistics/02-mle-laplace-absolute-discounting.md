<!-- ol:module M07.2 -->
# MLE, Laplace, absolute discounting

## Overview

| | |
|---|---|
| **Module** | `M07.2` · build · Python · Pass 3 · 2 to 3 h |
| **You build** | `python/tinyllm/prob/mle.py`: `mle`, `log_likelihood`, `laplace`, `absolute_discount`, `ney_discount` |
| **Contract** | [`course/contracts/py/tinyllm/prob/mle.pyi`](../../../../course/contracts/py/tinyllm/prob/mle.pyi) |
| **Tests** | `course/tests/M07.2/` (what they check: section 4) |
| **Needs** | no code from earlier modules · reading: `M00.1` logs and nats, `S-M05` counting |
| **Used by** | `L1.4` Unigram EM re-estimates piece probabilities · later: `L2.1` interpolated Kneser-Ney, `M11.4` PMI from co-occurrence counts |
| **Milestone** | `MS-P3` (tokens and data) |
| **Optional depth** | Jurafsky and Martin, *Speech and Language Processing* (3rd ed. draft), ch. 3 "N-gram Language Models"; Chen and Goodman, "An Empirical Study of Smoothing Techniques for Language Modeling" (1998) |

## Key Takeaways

- The maximum-likelihood estimate of a categorical distribution is the observed frequency $c_k / N$; no other distribution gives the data a higher log-likelihood (`test_hand_example_mle`, `test_mle_maximizes_the_likelihood`).
- The MLE gives every unseen outcome probability 0, so a single unseen word in new text makes its likelihood 0 and its perplexity infinite (`test_log_likelihood_zero_terms_and_impossible_data`).
- Add-alpha smoothing divides by $N + \alpha V$, one pseudo-count per word of the whole vocabulary, seen or not (`test_hand_example_laplace`, `test_laplace_full_distribution_sums_to_one`).
- Absolute discounting takes a fixed $d$ from every seen count and gives exactly that mass, $dT/N$, to a backoff distribution: the core of Kneser-Ney (`test_hand_example_absolute_discount`, `test_absolute_discount_sums_to_one`).

## How to work this chapter

```bash
ol start M07.2              # stubs mle.py into your repo
ol tests M07.2              # read the test catalog first
ol check M07.2              # exit code is the verdict
ol diff  M07.2              # after passing: your code against the reference
```

---

## 1. Why now

Your tracer bigram (`L0.0`) already smooths: it adds one to every count, because a byte pair never seen in training would otherwise get probability 0, the sampler could never produce it, and its NLL on new text would be infinite. That was one formula in one place. Pass 3 estimates distributions from counts all over the system: the Unigram tokenizer (`L1.4`) re-estimates its piece probabilities from expected counts on every EM step, the n-gram model (`L2.1`) estimates the next word for millions of contexts most of which were seen once or never, and PMI (`M11.4`) divides co-occurrence counts. Each needs the same three decisions made correctly: what the best estimate from counts is, how much probability to keep for what was not seen, and where that probability goes. This module makes those decisions once, as functions with exact answers.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | the vocabulary: every outcome the model can assign probability to; $\lvert V \rvert$ its size | `int` |
| $c_k$ | how many times outcome $k$ was observed: an integer from counting, or a fraction when it is an expected count from EM | `int` or `float` |
| $N = \sum_k c_k$ | the number of observations | `int` |
| $q$ | any distribution over $V$: $q_k \ge 0$, $\sum_k q_k = 1$ | `dict[K, float]` |
| $\ell(q) = \sum_k c_k \ln q_k$ | the log-likelihood of the counts under $q$, in nats | `float` |
| $\hat p_k$ | an estimate of the true probability of $k$ | `float` |
| $\alpha > 0$ | the pseudo-count of add-alpha smoothing | `float` |
| $d \in (0, 1]$ | the absolute discount | `float` |
| $T = \lvert \{k : c_k > 0\} \rvert$ | the number of distinct outcomes seen | `int` |
| $b_k$ | a backoff distribution over $V$ (a simpler model) | `dict[K, float]` |
| $n_r$ | the number of outcomes seen exactly $r$ times (count of counts) | `int` |

**Likelihood.** If the observations are independent draws from $q$, the probability of the data is $\prod_k q_k^{c_k}$ (each observation of $k$ contributes a factor $q_k$; the order does not matter for estimating $q$). Its log is $\ell(q) = \sum_k c_k \ln q_k$. Terms with $c_k = 0$ contribute nothing, whatever $q_k$ is, by the convention $0 \cdot \ln 0 = 0$. A term with $c_k > 0$ and $q_k = 0$ is $-\infty$: the data is impossible under $q$.

**The MLE is the frequency.** The maximum-likelihood estimate is the $q$ that maximizes $\ell(q)$. It is $\hat p_k = c_k / N$, and here is why. For any $q$,

$$\ell(q) - \ell(\hat p) = \sum_k c_k \ln \frac{q_k}{\hat p_k} = N \sum_k \hat p_k \ln \frac{q_k}{\hat p_k} = -N \, \mathrm{KL}(\hat p \,\Vert\, q) \le 0,$$

because the Kullback-Leibler divergence is never negative (Gibbs' inequality, `M11.1`), and it is 0 only when $q = \hat p$. So the frequency wins against every other distribution, which is what `test_mle_maximizes_the_likelihood` checks numerically. It needs $N > 0$: with no data, $0/0$ is not an estimate.

**The zero problem.** The MLE puts all its mass on what was seen. A word with $c_k = 0$ gets $\hat p_k = 0$, and new text containing it gets likelihood 0 and perplexity $\infty$ (`M11.2`). In language most words are rare, so most of the vocabulary has $c_k = 0$ in any given context: this is the normal case, not an edge case.

**Add-alpha (Laplace) smoothing.** Pretend every word of the vocabulary was seen $\alpha$ extra times:

$$\hat p_k = \frac{c_k + \alpha}{N + \alpha \lvert V \rvert}.$$

The denominator adds $\alpha$ once per word of $V$, including the words that are not keys of `counts` at all; summing the numerator over all of $V$ gives $N + \alpha \lvert V \rvert$, so the estimate is a distribution. Every word gets at least $\alpha / (N + \alpha \lvert V \rvert) > 0$. $\alpha \to 0$ recovers the MLE and $\alpha \to \infty$ the uniform $1/\lvert V \rvert$; with no data it is exactly uniform. (Bayesian reading: this is the posterior mean under a symmetric Dirichlet($\alpha$) prior; $\alpha = 1$ is Laplace's rule of succession.) Its weakness is that it takes mass in proportion to $\alpha \lvert V \rvert$, which for a 50 000-word vocabulary and a context seen 10 times hands almost everything to unseen words.

**Absolute discounting.** Church and Gale (1991) counted bigrams in one half of a corpus and looked them up in the other: a bigram seen $c \ge 2$ times in the first half appeared on average about $c - 0.75$ times in the second. Seen counts overstate the future by a roughly constant amount. So subtract a constant $d$ from every seen count and give the freed mass to a backoff distribution $b$ (in `L2.1`, the next lower-order model):

$$\hat p_k = \frac{\max(c_k - d, 0)}{N} + \lambda \, b_k, \qquad \lambda = \frac{1}{N} \sum_k \min(c_k, d).$$

Each word gives up $\min(c_k, d)$: $d$ if it has that much, its whole count if not. So $\lambda$ is precisely the freed fraction and $\sum_k \hat p_k = (N - N\lambda)/N + \lambda \sum_k b_k = 1$. For integer counts with $d \le 1$ every seen word has at least $d$ and every unseen word has nothing, so $\lambda = dT/N$, the form in most textbooks. The $\max$ matters for keys whose count is 0, and the $\min$ for the fractional expected counts of EM (`L1.4`): a word expected 0.3 times cannot lose 0.5. A seen word keeps $(c_k - d)/N$ plus its share of the backoff; an unseen one gets only $\lambda b_k$. With $N = 0$ there is nothing to discount and the answer is $b$ itself.

**Estimating $d$.** Ney, Essen and Kneser (1994) derived $D = n_1 / (n_1 + 2 n_2)$ from leaving one observation out at a time, where $n_1$ and $n_2$ count the words seen once and twice. It lies in $(0, 1]$ when $n_1 > 0$. Modified Kneser-Ney (`L2.1`) uses three such discounts, for counts 1, 2, and 3 or more.

## 3. Worked example by hand

Counts: `a` = 3, `b` = 2, `c` = 1, `d` = 0, over the vocabulary `a b c d e` ($\lvert V \rvert = 5$; `e` is not even a key). $N = 6$, $T = 3$.

**MLE.** $\hat p = (3/6, 2/6, 1/6, 0) = (0.5, 0.3333, 0.1667, 0)$ for `a b c d`, and `e` would be 0 too.

**Log-likelihood.** $\ell(\hat p) = 3 \ln \tfrac12 + 2 \ln \tfrac13 + \ln \tfrac16 = -2.0794 - 2.1972 - 1.7918 = -6.0684$ nats. The `d` term is $0 \cdot \ln 0 = 0$.

**Laplace, $\alpha = 1$.** Denominator $6 + 1 \cdot 5 = 11$:

| word | `a` | `b` | `c` | `d` | `e` (not a key) |
|---|---|---|---|---|---|
| $c_k + 1$ | 4 | 3 | 2 | 1 | 1 |
| $\hat p_k$ | 4/11 | 3/11 | 2/11 | 1/11 | 1/11 |

The five sum to $11/11$. Dividing by $6 + 1 \cdot 4$ (only the keys) would give $4/10 + 3/10 + 2/10 + 1/10 = 1$ over the keys and leave $1/10$ for `e` on top: a total of $1.1$. Its log-likelihood is $3 \ln \tfrac4{11} + 2 \ln \tfrac3{11} + \ln \tfrac2{11} = -7.3382$, lower than the MLE's, as it must be.

**Absolute discounting, $d = 1/2$, uniform backoff $b_k = 1/5$.** The freed mass is $\lambda = d T / N = 0.5 \cdot 3 / 6 = 1/4$, so each word gets $\lambda b_k = 1/20$ from the backoff:

| word | $\max(c_k - d, 0)/N$ | $+ \lambda b_k$ | $\hat p_k$ |
|---|---|---|---|
| `a` | $2.5/6 = 5/12$ | $1/20$ | $7/15 = 0.4667$ |
| `b` | $1.5/6 = 1/4$ | $1/20$ | $3/10$ |
| `c` | $0.5/6 = 1/12$ | $1/20$ | $2/15 = 0.1333$ |
| `d` | $\max(-0.5, 0)/6 = 0$ | $1/20$ | $1/20$ |
| `e` | $0$ | $1/20$ | $1/20$ |

Sum: $28/60 + 18/60 + 8/60 + 3/60 + 3/60 = 1$. Without the clamp, `d` would get $-1/12 + 1/20 < 0$.

**Ney's discount.** One word seen once (`c`), one seen twice (`b`): $D = 1/(1 + 2 \cdot 1) = 1/3$.

These numbers are the first cases in section 4: `test_hand_example_mle`, `test_hand_example_laplace`, `test_hand_example_absolute_discount`, `test_hand_example_ney_discount`, and `test_hand_example_log_likelihood`.

## 4. The interface

```python
# python/tinyllm/prob/mle.py
def mle(counts: Mapping[K, int]) -> dict[K, float]                                  # c_k / N
def log_likelihood(counts: Mapping[K, int], probs: Mapping[K, float]) -> float       # nats; -inf if impossible
def laplace(counts: Mapping[K, int], vocab_size: int, alpha: float = 1.0) -> dict[K, float]
def absolute_discount(counts: Mapping[K, int], d: float, backoff: Mapping[K, float]) -> dict[K, float]
def ney_discount(counts: Mapping[K, int]) -> float                                   # n1 / (n1 + 2 n2)
```

Counts are finite non-negative numbers: integers from counting (numpy integers are fine) or floats from EM; NaN, infinity, bools, and strings are a `ValueError`. `laplace` returns the keys of `counts` only; the contract states the probability of every other word. `absolute_discount` returns every key of `backoff`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_mle` | unit | 1/2, 1/3, 1/6, 0 | you and the test agree on the definition |
| `test_hand_example_laplace` | unit | 4/11, 3/11, 2/11, 1/11, and 1/11 for `e` | the denominator counts the whole vocabulary |
| `test_hand_example_absolute_discount` | unit | 7/15, 3/10, 2/15, 1/20, 1/20 | the formula `L2.1` builds on |
| `test_hand_example_ney_discount` | unit | $D = 1/3$ | the starting discount of `L2.1` |
| `test_hand_example_log_likelihood` | unit | $-6.0684$ for the MLE, $-7.3382$ for Laplace | smoothing costs training likelihood |
| `test_mle_maximizes_the_likelihood` | property | random competitors and small perturbations score lower | the defining property |
| `test_mle_sums_to_one_and_zero_counts_get_zero` | property | a distribution; zeros stay 0 | no NaN, no missing key |
| `test_laplace_full_distribution_sums_to_one` | property | keys plus the unseen words sum to 1 for three $\alpha$ | the whole vocabulary is normalized |
| `test_laplace_limits` | unit | tiny $\alpha$ is the MLE, huge $\alpha$ and no data are uniform | $\alpha$ interpolates |
| `test_absolute_discount_sums_to_one` | property | random counts with zeros, random backoffs, four $d$ | $\lambda = dT/N$ exactly |
| `test_absolute_discount_zero_counts_are_clamped` | boundary | a zero-count key gets only $\lambda b_k$ | no negative probability |
| `test_absolute_discount_with_no_data_is_the_backoff` | boundary | $N = 0$ returns a copy of $b$ | an unseen context in `L2.1` |
| `test_absolute_discount_d_one_removes_singletons` | boundary | $d = 1$ | the edge of the allowed range |
| `test_rejects_bad_arguments` | boundary | negative, NaN, infinite, bool, or string counts; no data; bad $\alpha$, $d$, $V$, or backoff | upstream counting bugs fail here |
| `test_fractional_counts_from_em` | unit | the MLE of expected counts; $\lambda = \sum_k \min(c_k, d)/N$ when a count is below $d$ | EM in `L1.4` produces fractions |
| `test_ney_discount_cases` | unit | all singletons give 1; no singletons raise | $D \in (0, 1]$ |
| `test_log_likelihood_zero_terms_and_impossible_data` | boundary | $0 \ln 0 = 0$; an observed outcome with $q = 0$ gives $-\infty$ | infinite perplexity in `M11.2` |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. dividing by the number of distinct words instead of $N$ | "probabilities" that sum to $N / \lvert \mathrm{keys} \rvert$ | `test_hand_example_mle` (mutant `s01`) |
| 2. Laplace denominator $N + \alpha \cdot$ (number of keys) | the keys alone sum to 1, so the unseen words' mass is extra: the total is above 1 | `test_laplace_full_distribution_sums_to_one` (mutant `s02`) |
| 3. counting $T$ over every key, zero counts included | more mass handed to the backoff than was freed | `test_hand_example_absolute_discount` (mutant `s04`) |
| 4. no clamp at 0 | a key counted zero times gets a negative probability | `test_absolute_discount_zero_counts_are_clamped` (mutant `s06`) |
| 5. scoring zero-count terms, or indexing a missing probability | $0 \ln 0$ turns into $-\infty$ or a `KeyError` | `test_log_likelihood_zero_terms_and_impossible_data` (mutants `s09`, `s10`) |
| 6. $\lambda = dT/N$ with fractional counts | a count of 0.3 is charged 0.5, so the result sums to more than 1 | `test_fractional_counts_from_em` (mutant `s05`) |
| 7. returning the caller's backoff dict when $N = 0$ | `L2.1` edits one context's distribution and corrupts the shared lower order | `test_absolute_discount_with_no_data_is_the_backoff` (mutant `s07`) |
| 8. Ney's $D = n_1 / (n_1 + n_2)$ | a discount that is too large | `test_hand_example_ney_discount` (mutant `s08`) |
| 9. accepting NaN or infinite counts, or estimating from no data | NaN probabilities, or a `ZeroDivisionError` | `test_rejects_bad_arguments` (mutants `s11`, `s12`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.1` | natural logs, so likelihoods are in nats |
| Back | `S-M05` | counting outcomes and the size of a vocabulary |
| Forward | `L1.4` | seeds the piece probabilities with `mle` of substring counts; each EM step is the same estimate over expected counts |
| Forward | `L2.1` | interpolated Kneser-Ney is `absolute_discount` per context with `ney_discount`-style $D_1, D_2, D_{3+}$ and a continuation-count backoff |
| Forward | `M11.4` | PMI divides joint and marginal MLEs from co-occurrence counts |
| Forward | `M11.2` | $-\ell(q)/N$ is the NLL per token whose exponential is perplexity |

If you skip this module, `ol check L1.4` stops with `L1.4 needs M07.2`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `absolute_discount`, `ney_discount` | KenLM | modified Kneser-Ney with three discounts per order estimated from count-of-counts, on disk-backed sorted n-gram streams | `lm/builder/adjust_counts.cc` (discounts from the count of counts) |
| `laplace` | scikit-learn `MultinomialNB` | the same add-alpha estimate per class, as its `alpha` parameter | `sklearn/naive_bayes.py` |
| `mle` over expected counts | SentencePiece Unigram trainer | EM where the M-step is this MLE plus a digamma correction (a Bayesian variant) and pruning | `src/unigram_model_trainer.cc` |
