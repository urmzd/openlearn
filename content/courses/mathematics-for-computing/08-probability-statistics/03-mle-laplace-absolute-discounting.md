# MLE, Laplace, absolute discounting

The [byte bigram](../../build-an-llm/01-foundations/01-byte-bigram.md) at the start of *Build Your Own LLM System* already smooths: it adds one to every count, because a byte pair never seen in training would otherwise get probability 0, the sampler could never produce it, and its NLL on new text would be infinite. That was one formula in one place. Later systems estimate distributions from counts everywhere. The [Unigram tokenizer](../../build-an-llm/02-tokenizers/04-unigram-lm.md) re-estimates its piece probabilities from expected counts on every EM step, the [n-gram model with Kneser-Ney](../../build-an-llm/03-statistical-lm/01-ngram-kneser-ney.md) estimates the next word for millions of contexts most of which were seen once or never, and [PMI](../12-information-theory/03-mutual-information-pmi-ppmi.md) divides co-occurrence counts.

Each of those needs the same three decisions made correctly: what the best estimate from counts is, how much probability to keep for what was not seen, and where that probability goes. This section makes those decisions once, as functions with exact answers. You will prove that the observed frequency is the maximum-likelihood estimate, see why it fails on unseen outcomes, and build the two classical repairs: add-alpha smoothing and absolute discounting, the core of Kneser-Ney.

No code from earlier sections is needed. The reading background is natural logs and nats from [Exponents, logs, and units of information](../01-precalculus/01-exponents-logs-and-units-of-information.md) and counting from the [Discrete math 1 problem set](../06-discrete-math-1/03-problem-set.md). The reference implementation is [`course/solutions/python/tinyllm/prob/mle.py`](../../../../course/solutions/python/tinyllm/prob/mle.py), with its contract in [`mle.pyi`](../../../../course/contracts/py/tinyllm/prob/mle.pyi).

> **Learning objectives**
> - Prove that $c_k / N$ maximizes the log-likelihood of a categorical distribution, using the KL divergence.
> - Explain why the MLE makes unseen outcomes impossible and what that does to perplexity.
> - Implement add-alpha smoothing with the denominator over the whole vocabulary.
> - Implement absolute discounting so that exactly the freed mass goes to a backoff distribution, including for fractional counts.
> - Compute Ney's discount from counts of counts.

## Notation

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

## Likelihood and the MLE

If the observations are independent draws from $q$, the probability of the data is $\prod_k q_k^{c_k}$ (each observation of $k$ contributes a factor $q_k$; the order does not matter for estimating $q$). Its log is $\ell(q) = \sum_k c_k \ln q_k$. Terms with $c_k = 0$ contribute nothing, whatever $q_k$ is, by the convention $0 \cdot \ln 0 = 0$. A term with $c_k > 0$ and $q_k = 0$ is $-\infty$: the data is impossible under $q$.

The maximum-likelihood estimate is the $q$ that maximizes $\ell(q)$. It is the observed frequency, $\hat p_k = c_k / N$, and here is why. For any $q$,

$$\ell(q) - \ell(\hat p) = \sum_k c_k \ln \frac{q_k}{\hat p_k} = N \sum_k \hat p_k \ln \frac{q_k}{\hat p_k} = -N \, \mathrm{KL}(\hat p \,\Vert\, q) \le 0,$$

because the Kullback-Leibler divergence is never negative (Gibbs' inequality, proved in [Entropy, cross-entropy, and KL](../12-information-theory/01-entropy-cross-entropy-and-kl.md)), and it is 0 only when $q = \hat p$. So the frequency wins against every other distribution, which can also be checked numerically by scoring random competitors. It needs $N > 0$: with no data, $0/0$ is not an estimate.

## The zero problem

The MLE's optimality is also its weakness: it puts all its mass on what was seen. A word with $c_k = 0$ gets $\hat p_k = 0$, and new text containing it gets likelihood 0 and perplexity $\infty$ (see [Perplexity, bits per byte, and the NLL accumulator](../12-information-theory/02-perplexity-bits-per-byte-nll-accumulator.md)). In language most words are rare, so most of the vocabulary has $c_k = 0$ in any given context: this is the normal case, not an edge case. The rest of the section is about how to take some probability away from what was seen and give it to what was not.

## Add-alpha (Laplace) smoothing

The bluntest repair is to pretend every word of the vocabulary was seen $\alpha$ extra times:

$$\hat p_k = \frac{c_k + \alpha}{N + \alpha \lvert V \rvert}.$$

The denominator adds $\alpha$ once per word of $V$, including the words that are not keys of `counts` at all; summing the numerator over all of $V$ gives $N + \alpha \lvert V \rvert$, so the estimate is a distribution. Every word gets at least $\alpha / (N + \alpha \lvert V \rvert) > 0$. $\alpha \to 0$ recovers the MLE and $\alpha \to \infty$ the uniform $1/\lvert V \rvert$; with no data it is exactly uniform. (Bayesian reading: this is the posterior mean under a symmetric Dirichlet($\alpha$) prior; $\alpha = 1$ is Laplace's rule of succession.)

Its weakness is that it takes mass in proportion to $\alpha \lvert V \rvert$, which for a 50 000-word vocabulary and a context seen 10 times hands almost everything to unseen words.

## Absolute discounting

A better repair takes mass from seen words in proportion to how much their counts overstate the future. Church and Gale (1991) counted bigrams in one half of a corpus and looked them up in the other: a bigram seen $c \ge 2$ times in the first half appeared on average about $c - 0.75$ times in the second. Seen counts overstate the future by a roughly constant amount. So subtract a constant $d$ from every seen count and give the freed mass to a backoff distribution $b$ (in Kneser-Ney, the next lower-order model):

$$\hat p_k = \frac{\max(c_k - d, 0)}{N} + \lambda \, b_k, \qquad \lambda = \frac{1}{N} \sum_k \min(c_k, d).$$

Each word gives up $\min(c_k, d)$: $d$ if it has that much, its whole count if not. So $\lambda$ is precisely the freed fraction and $\sum_k \hat p_k = (N - N\lambda)/N + \lambda \sum_k b_k = 1$. For integer counts with $d \le 1$ every seen word has at least $d$ and every unseen word has nothing, so $\lambda = dT/N$, the form in most textbooks. The $\max$ matters for keys whose count is 0, and the $\min$ for the fractional expected counts of EM in the Unigram tokenizer: a word expected 0.3 times cannot lose 0.5. A seen word keeps $(c_k - d)/N$ plus its share of the backoff; an unseen one gets only $\lambda b_k$. With $N = 0$ there is nothing to discount and the answer is $b$ itself.

That leaves the question of how large $d$ should be. Ney, Essen and Kneser (1994) derived $D = n_1 / (n_1 + 2 n_2)$ from leaving one observation out at a time, where $n_1$ and $n_2$ count the words seen once and twice. It lies in $(0, 1]$ when $n_1 > 0$. Modified Kneser-Ney uses three such discounts, for counts 1, 2, and 3 or more.

## Working it by hand

Take counts `a` = 3, `b` = 2, `c` = 1, `d` = 0, over the vocabulary `a b c d e` ($\lvert V \rvert = 5$; `e` is not even a key). $N = 6$, $T = 3$.

**MLE.** $\hat p = (3/6, 2/6, 1/6, 0) = (0.5, 0.3333, 0.1667, 0)$ for `a b c d`, and `e` would be 0 too.

**Log-likelihood.** $\ell(\hat p) = 3 \ln \tfrac12 + 2 \ln \tfrac13 + \ln \tfrac16 = -2.0794 - 2.1972 - 1.7918 = -6.0684$ nats. The `d` term is $0 \cdot \ln 0 = 0$.

**Laplace, $\alpha = 1$.** The denominator is $6 + 1 \cdot 5 = 11$, giving:

| word | `a` | `b` | `c` | `d` | `e` (not a key) |
|---|---|---|---|---|---|
| $c_k + 1$ | 4 | 3 | 2 | 1 | 1 |
| $\hat p_k$ | 4/11 | 3/11 | 2/11 | 1/11 | 1/11 |

The five sum to $11/11$. Dividing by $6 + 1 \cdot 4$ (only the keys) would give $4/10 + 3/10 + 2/10 + 1/10 = 1$ over the keys and leave $1/10$ for `e` on top: a total of $1.1$. The Laplace estimate's log-likelihood is $3 \ln \tfrac4{11} + 2 \ln \tfrac3{11} + \ln \tfrac2{11} = -7.3382$, lower than the MLE's, as it must be.

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

These numbers are the first five test cases a correct implementation must reproduce.

## Designing the interface

Five functions cover the section:

```python
# python/tinyllm/prob/mle.py
def mle(counts: Mapping[K, int]) -> dict[K, float]                                  # c_k / N
def log_likelihood(counts: Mapping[K, int], probs: Mapping[K, float]) -> float       # nats; -inf if impossible
def laplace(counts: Mapping[K, int], vocab_size: int, alpha: float = 1.0) -> dict[K, float]
def absolute_discount(counts: Mapping[K, int], d: float, backoff: Mapping[K, float]) -> dict[K, float]
def ney_discount(counts: Mapping[K, int]) -> float                                   # n1 / (n1 + 2 n2)
```

Counts are finite non-negative numbers: integers from counting (numpy integers are fine) or floats from EM; NaN, infinity, bools, and strings are a `ValueError`. `laplace` returns the keys of `counts` only; the contract states the probability of every other word. `absolute_discount` returns every key of `backoff`.

## Building it step by step

Seven decisions determine the implementation.

1. **Counts are numbers, checked once.** Every function starts from `counts`, so one validator rejects what is not a finite non-negative real (bools are `int` subclasses in Python and are rejected explicitly) and returns $N$, summed with `math.fsum` so fractional EM counts do not drift.
2. **The MLE is $c_k / N$, and needs $N > 0$.** The KL argument proves the frequency is optimal; with $N = 0$ it is $0/0$, so the function raises instead of returning NaNs.
3. **The log-likelihood skips zero counts before looking up $q$.** $0 \ln 0 = 0$ by convention, and a zero-count key need not even be in `probs`. For $c_k > 0$, a missing or zero $q_k$ makes the data impossible: return $-\infty$ immediately.
4. **Laplace divides by $N + \alpha\lvert V \rvert$.** One pseudo-count per word of the vocabulary, not per key, so the keys plus the unseen words sum to 1. The function returns only the keys; every other word's probability is $\alpha/(N + \alpha\lvert V \rvert)$ by the same formula.
5. **Absolute discounting: clamp, charge exactly what was taken, spread it by $b$.** Each key loses $\min(c_k, d)$; $\lambda$ is the sum of those losses over $N$; every word of the backoff vocabulary gets $\max(c_k - d, 0)/N + \lambda b_k$. The output ranges over `backoff`'s keys because that is the vocabulary the smoothed distribution covers, so every counted key must be in it.
6. **No data means the backoff, as a fresh dict.** With $N = 0$ nothing was freed and nothing was seen; the answer is $b$. Returning a copy matters because Kneser-Ney may edit one context's distribution.
7. **Ney's $D = n_1/(n_1 + 2n_2)$ needs a singleton.** With $n_1 = 0$ the discount would be 0 (no smoothing) or $0/0$; raise.

The validator, the MLE, and the log-likelihood come first:

```python
def _check_counts(counts: Mapping[Hashable, float]) -> float:
    for k, c in counts.items():
        # bool is a subclass of int: True would silently count as 1.
        if isinstance(c, bool) or not isinstance(c, numbers.Real):
            raise ValueError(f"count of {k!r} must be a number, got {c!r}")
        if not math.isfinite(c) or c < 0:
            raise ValueError(f"count of {k!r} must be finite and >= 0, got {c}")
    return math.fsum(float(c) for c in counts.values())   # N, exactly rounded


def mle(counts: Mapping[Hashable, float]) -> dict:
    n = _check_counts(counts)
    if n == 0:
        raise ValueError("mle needs at least one observation (N = 0 gives 0 / 0)")
    return {k: float(c) / n for k, c in counts.items()}   # divide by N, not by the number of keys


def log_likelihood(counts, probs) -> float:
    _check_counts(counts)
    terms = []
    for k, c in counts.items():
        if c == 0:
            continue  # 0 * log 0 = 0: an unseen outcome costs nothing (and need not be in probs)
        q = float(probs.get(k, 0.0))
        if q <= 0.0:
            return -math.inf          # an observed outcome the model calls impossible
        terms.append(float(c) * math.log(q))
    return math.fsum(terms)
```

`numbers.Real` accepts Python ints, floats, and numpy integer and float scalars, which is what counting code produces.

Laplace smoothing is a few lines once the denominator is right:

```python
def laplace(counts, vocab_size: int, alpha: float = 1.0) -> dict:
    if not alpha > 0:                 # also rejects NaN
        raise ValueError(f"alpha must be > 0, got {alpha!r}")
    if len(counts) > vocab_size:      # more keys than words: the vocabulary is wrong
        raise ValueError(f"{len(counts)} keys do not fit a vocabulary of {vocab_size}")
    n = _check_counts(counts)
    # The denominator adds alpha once per outcome of the vocabulary, seen or not.
    denom = n + alpha * vocab_size
    return {k: (float(c) + alpha) / denom for k, c in counts.items()}
```

On the hand example, `denom` is $6 + 1 \cdot 5 = 11$ and the keys get $4, 3, 2, 1$ elevenths; `e`, not a key, has $1/11$ by the same formula, and the five sum to 1.

Absolute discounting carries most of the validation, because its inputs come from two places that must agree:

```python
def absolute_discount(counts, d: float, backoff) -> dict:
    if not 0.0 < d <= 1.0:
        raise ValueError(f"d must lie in (0, 1], got {d!r}")
    total_b = math.fsum(float(v) for v in backoff.values())
    if abs(total_b - 1.0) > 1e-9:
        raise ValueError(f"backoff must sum to 1, got {total_b!r}")
    missing = [k for k in counts if k not in backoff]
    if missing:                       # a counted word outside the output vocabulary would lose its mass
        raise ValueError(f"{missing[0]!r} is counted but not in the backoff vocabulary")
    n = _check_counts(counts)
    if n == 0:
        return {k: float(v) for k, v in backoff.items()}   # a copy, never the caller's dict
    # Exactly the mass the discount removed: d from each seen count, or the
    # whole count when it is smaller than d (fractional EM counts). For
    # integer counts this is d * T / N with T the number of seen keys.
    lam = math.fsum(min(float(c), d) for c in counts.values()) / n
    return {
        k: max(float(counts.get(k, 0)) - d, 0.0) / n + lam * float(b)   # clamp: a zero count stays >= 0
        for k, b in backoff.items()
    }
```

On the hand example, $\min(c_k, \tfrac12)$ is $\tfrac12, \tfrac12, \tfrac12, 0$, so $\lambda = 1.5/6 = \tfrac14$; `d` (count 0) gets $\max(-\tfrac12, 0)/6 + \tfrac14 \cdot \tfrac15 = \tfrac1{20}$. A zero-count key contributes $\min(0, d) = 0$ to $\lambda$, so counting $T$ only over seen keys falls out of the formula without a separate test.

Ney's discount is the last piece:

```python
def ney_discount(counts) -> float:
    _check_counts(counts)
    n1 = sum(1 for c in counts.values() if c == 1)   # words seen once
    n2 = sum(1 for c in counts.values() if c == 2)   # words seen twice
    if n1 == 0:
        raise ValueError("no outcome was seen exactly once: Ney's discount would be 0")
    return n1 / (n1 + 2 * n2)                        # in (0, 1] once n1 > 0
```

The full solution is [`mle.py`](../../../../course/solutions/python/tinyllm/prob/mle.py).

### Why this design

Every function is $O(\lvert \text{keys} \rvert)$ (and `absolute_discount` $O(\lvert V_b \rvert)$ over the backoff), one or two passes, with `fsum` adding a small constant factor for exact summation.

The general $\lambda = \sum \min(c_k, d)/N$ is preferred over the textbook $dT/N$. They agree for integer counts, but $dT/N$ is wrong for EM's fractional counts: a count of 0.3 would be charged 0.5, and the distribution would sum to more than 1. The general form costs nothing extra.

Laplace returns a sparse output. Returning a dict over the whole vocabulary would cost $O(\lvert V \rvert)$ per call, and the n-gram model calls this for millions of contexts. Returning the keys and stating the formula for the rest keeps it proportional to what was seen.

Among the alternatives, Good-Turing re-estimates each count from $n_{r+1}/n_r$ and is noisy for large $r$; add-alpha is simple but hands too much mass to a large vocabulary. Absolute discounting with a backoff is the base of Kneser-Ney, the best classical n-gram smoother, which is why the n-gram model builds on it.

One tradeoff is accepted: the backoff check uses a $10^{-9}$ tolerance and the count checks are strict, so a slightly unnormalized backoff from a long float computation is accepted but a negative count never is.

## What can go wrong

A correct solution reproduces the hand example, satisfies the normalization properties on random inputs, and handles the boundaries. The checks are listed here with why each matters later.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| MLE by hand | unit | 1/2, 1/3, 1/6, 0 | everyone agrees on the definition |
| Laplace by hand | unit | 4/11, 3/11, 2/11, 1/11, and 1/11 for `e` | the denominator counts the whole vocabulary |
| absolute discounting by hand | unit | 7/15, 3/10, 2/15, 1/20, 1/20 | the formula Kneser-Ney builds on |
| Ney's discount by hand | unit | $D = 1/3$ | the starting discount of Kneser-Ney |
| log-likelihood by hand | unit | $-6.0684$ for the MLE, $-7.3382$ for Laplace | smoothing costs training likelihood |
| the MLE maximizes the likelihood | property | random competitors and small perturbations score lower | the defining property |
| the MLE is a distribution | property | sums to 1; zeros stay 0 | no NaN, no missing key |
| Laplace over the whole vocabulary | property | keys plus the unseen words sum to 1 for three $\alpha$ | the whole vocabulary is normalized |
| Laplace limits | unit | tiny $\alpha$ is the MLE, huge $\alpha$ and no data are uniform | $\alpha$ interpolates |
| absolute discounting sums to 1 | property | random counts with zeros, random backoffs, four $d$ | $\lambda$ is exactly the freed mass |
| zero-count keys | boundary | a zero-count key gets only $\lambda b_k$ | no negative probability |
| no data | boundary | $N = 0$ returns a copy of $b$ | an unseen context in the n-gram model |
| $d = 1$ | boundary | singletons get only backoff mass | the edge of the allowed range |
| bad arguments | boundary | negative, NaN, infinite, bool, or string counts; no data; bad $\alpha$, $d$, $V$, or backoff | upstream counting bugs fail here |
| fractional counts from EM | unit | the MLE of expected counts; $\lambda = \sum_k \min(c_k, d)/N$ when a count is below $d$ | EM in the Unigram tokenizer produces fractions |
| Ney's discount edge cases | unit | all singletons give 1; no singletons raise | $D \in (0, 1]$ |
| zero terms and impossible data | boundary | $0 \ln 0 = 0$; an observed outcome with $q = 0$ gives $-\infty$ | infinite perplexity |

The mistakes behind failures of these checks fall into three groups.

The first group gets the normalizer wrong. Dividing by the number of distinct words instead of $N$ produces "probabilities" that sum to $N / \lvert \mathrm{keys} \rvert$; the solution divides by `n` from `_check_counts`. A Laplace denominator of $N + \alpha \cdot$ (number of keys) makes the keys alone sum to 1, so the unseen words' mass is extra and the total is above 1; the solution uses `denom = n + alpha * vocab_size`.

The second group charges the wrong amount in absolute discounting. Counting $T$ over every key, zero counts included, hands more mass to the backoff than was freed; $\lambda$ sums `min(c, d)`, which is 0 for a zero count. Using $\lambda = dT/N$ with fractional counts charges a count of 0.3 the full 0.5, so the result sums to more than 1. Omitting the clamp at 0 gives a key counted zero times a negative probability; the solution uses `max(... - d, 0.0)`. And Ney's discount written as $n_1 / (n_1 + n_2)$ is too large; the factor 2 matters.

> **Warning:** Returning the caller's backoff dict when $N = 0$ looks harmless, but Kneser-Ney edits one context's distribution and would corrupt the shared lower order. Always build a new dict.

The third group mishandles edge inputs. Scoring zero-count terms, or indexing a missing probability, turns $0 \ln 0$ into $-\infty$ or a `KeyError`; the solution does `continue` on `c == 0` and then `probs.get(k, 0.0)`. Accepting NaN or infinite counts, or estimating from no data, yields NaN probabilities or a `ZeroDivisionError`; `_check_counts` rejects the former and `mle` raises on $N = 0$.

## Where this leads

These estimators reappear wherever counts become probabilities. The [Unigram tokenizer](../../build-an-llm/02-tokenizers/04-unigram-lm.md) seeds its piece probabilities with `mle` of substring counts, and each EM step is the same estimate over expected counts. [Interpolated Kneser-Ney](../../build-an-llm/03-statistical-lm/01-ngram-kneser-ney.md) is `absolute_discount` per context with `ney_discount`-style $D_1, D_2, D_{3+}$ and a continuation-count backoff. [Mutual information, PMI, and PPMI](../12-information-theory/03-mutual-information-pmi-ppmi.md) divides joint and marginal MLEs from co-occurrence counts, and in [Perplexity, bits per byte, and the NLL accumulator](../12-information-theory/02-perplexity-bits-per-byte-nll-accumulator.md), $-\ell(q)/N$ is the NLL per token whose exponential is perplexity.

### Further reading

- Jurafsky and Martin, *Speech and Language Processing* (3rd ed. draft), chapter 3, "N-gram Language Models".
- Chen and Goodman, "An Empirical Study of Smoothing Techniques for Language Modeling" (1998).
- KenLM, `lm/builder/adjust_counts.cc`: modified Kneser-Ney with three discounts per order estimated from count-of-counts, on disk-backed sorted n-gram streams.
- scikit-learn's `MultinomialNB` (`sklearn/naive_bayes.py`): the same add-alpha estimate per class, as its `alpha` parameter.
- The SentencePiece Unigram trainer, `src/unigram_model_trainer.cc`: EM where the M-step is this MLE plus a digamma correction (a Bayesian variant) and pruning.

## Key terms

- **Log-likelihood**: $\sum_k c_k \ln q_k$, the log probability of the observed counts under a distribution $q$.
- **Maximum-likelihood estimate (MLE)**: the distribution that maximizes the log-likelihood; for counts, the frequency $c_k/N$.
- **Add-alpha (Laplace) smoothing**: adding a pseudo-count $\alpha$ to every word of the vocabulary before normalizing.
- **Absolute discounting**: subtracting a fixed $d$ from each seen count and giving the freed mass to a backoff distribution.
- **Backoff distribution**: a simpler model that receives the probability mass freed by discounting.
- **Count of counts**: $n_r$, the number of outcomes seen exactly $r$ times.
- **Ney's discount**: the estimate $D = n_1/(n_1 + 2n_2)$ of the absolute discount.

## Summary

- The maximum-likelihood estimate of a categorical distribution is the observed frequency $c_k / N$; no other distribution gives the data a higher log-likelihood, because the gap is $-N\,\mathrm{KL}(\hat p \Vert q)$.
- The MLE gives every unseen outcome probability 0, so a single unseen word in new text makes its likelihood 0 and its perplexity infinite.
- Add-alpha smoothing divides by $N + \alpha \lvert V \rvert$, one pseudo-count per word of the whole vocabulary, seen or not.
- Absolute discounting takes $\min(c_k, d)$ from every count and gives exactly that mass, $dT/N$ for integer counts, to a backoff distribution: the core of Kneser-Ney.
- Ney's $D = n_1/(n_1 + 2n_2)$ gives a principled starting discount and needs at least one singleton.

## Review questions

1. Why is the Laplace denominator $N + \alpha\lvert V \rvert$ and not $N + \alpha \cdot (\text{number of keys})$?

   <details><summary>Answer</summary>

   Every word of the vocabulary receives a pseudo-count, including words that are not keys of `counts` at all. Summing $c_k + \alpha$ over all of $V$ gives $N + \alpha\lvert V \rvert$. With only the keys in the denominator, the keys alone sum to 1 and the unseen words' $\alpha/(\cdot)$ comes on top, for a total above 1 (1.1 in the hand example).

   </details>

2. Why does `absolute_discount` clamp with `max(c - d, 0)` when $d \le 1$ and counts are integers?

   <details><summary>Answer</summary>

   Keys can have count 0 (they were listed but not seen), and `counts.get(k, 0)` returns 0 for every backoff word not counted. Without the clamp each of those would get $-d/N + \lambda b_k$, which can be negative ($-1/12 + 1/20$ in the hand example). The clamp says an unseen word loses nothing because it had nothing.

   </details>

3. Why does `log_likelihood` `continue` on `c == 0` before reading `probs`?

   <details><summary>Answer</summary>

   $0 \cdot \ln q = 0$ for any $q$, including $q = 0$ (by the $0 \ln 0 = 0$ convention). Reading first would either compute $0 \cdot (-\infty) = \text{NaN}$ or return $-\infty$ for data that is perfectly possible, and a zero-count key may be absent from `probs` entirely.

   </details>

4. With $N = 0$, why return a copy of the backoff rather than `backoff` itself?

   <details><summary>Answer</summary>

   The answer is mathematically $b$, but the n-gram model stores one distribution per context and may adjust it later. Returning the shared lower-order dict would let an edit to one unseen context change the backoff for every context.

   </details>

5. Why does `ney_discount` raise when no word was seen exactly once?

   <details><summary>Answer</summary>

   $D = n_1/(n_1 + 2n_2)$ is 0 when $n_1 = 0$ (or $0/0$ when $n_2 = 0$ too). A zero discount frees no mass, so unseen words would get probability 0, the problem smoothing exists to solve. Raising tells the caller that this count-of-counts gives no usable estimate.

   </details>
