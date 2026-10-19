# Entropy, cross-entropy, KL, JS, and the k3 estimator

## Overview

| | |
|---|---|
| **Module** | `M11.1` · build · Python · Pass 2 · 2 to 3 h |
| **Solution** | [`python/tinyllm/info/entropy.py`](../../../../course/solutions/python/tinyllm/info/entropy.py): `entropy`, `cross_entropy`, `kl`, `kl_from_logprobs`, `js`, `kl_k3`, `entropy_from_logits` |
| **Contract** | [`course/contracts/py/tinyllm/info/entropy.pyi`](../../../../course/contracts/py/tinyllm/info/entropy.pyi) |
| **Needs** | `M09.2` stable numerics (`log_softmax`) |
| **Used by** | `M08.3` differentiates this cross-entropy · `L0.3` the training loss · later: `L8.1` per-step entropy logging, `L8.6` acceptance rate as $1 - \mathrm{TV}$, `L12.3` the KL penalty with `kl_k3`, `M11.2` perplexity · later: `L12.2`, `L12.4`, `M11.4` |
| **Optional depth** | Cover and Thomas, *Elements of Information Theory* (2nd ed.), ch. 2; MacKay, *Information Theory, Inference, and Learning Algorithms*, ch. 2 and 4; Schulman, "Approximating KL Divergence" (2020 blog post) |

## Key Takeaways

- Entropy $H(p) = -\sum_i p_i \log p_i$ is the average surprise of a distribution, between 0 (certain) and $\log n$ (uniform).
- Cross-entropy splits as $H(p, q) = H(p) + \mathrm{KL}(p \,\Vert\, q)$: the training loss is the data's own entropy, which no model can beat, plus the model's excess.
- $\mathrm{KL} \ge 0$ with equality only at $p = q$ (Gibbs), and it is not symmetric; Jensen-Shannon is symmetric and bounded by $\ln 2$.
- From model outputs, compute KL with log-probabilities, never with $p / q$; the k3 estimator $e^r - r - 1$ estimates KL from samples without bias and is never negative.

## How to read this chapter

Sections 1 to 3 build entropy, cross-entropy, KL, JS and the k3 estimator from the idea of surprise and work every one by hand on a three-outcome example. Section 4 fixes the interface and lists what a correct implementation must handle. Section 5 walks through the solution, [`entropy.py`](../../../../course/solutions/python/tinyllm/info/entropy.py), function by function, and explains why each zero convention and each precision trick is there. Try to predict each line before reading it.

---

## 1. Why now

The tracer bigram (`L0.0`) already reports a negative log-likelihood, and `train bigram` prints it, but nothing in the system yet says what that number is measured against or what it can reach. In this pass you train the same bigram by gradient descent (`L0.5`) with a cross-entropy loss (`L0.3`), and the loss needs a meaning: how far above the floor are we, and what is the floor? Later the sampler (`L8.1`) logs the entropy of every step so a collapsing temperature is visible in the traces, speculative decoding (`L8.6`) reads its acceptance rate off the distance between two distributions, and post-training (`L12.3`) penalizes the policy for drifting from the reference model with a sampled KL. Each needs the same few quantities with the same conventions at zero, in nats, stable on the log-probabilities the models actually produce. And `M08.3` needs a forward cross-entropy to check its gradient against.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $n$ | number of outcomes (vocabulary size for a language model) | `int` |
| $p, q$ | probability distributions over the $n$ outcomes: $p_i \ge 0$, $\sum_i p_i = 1$ | `float64[n]` |
| $\log$ | the natural logarithm; results are in **nats** (divide by $\ln 2$ for bits) | |
| $H(p)$ | entropy of $p$ | scalar |
| $H(p, q)$ | cross-entropy of $q$ relative to $p$ | scalar |
| $\mathrm{KL}(p \,\Vert\, q)$ | Kullback-Leibler divergence from $q$ to $p$ | scalar, $\ge 0$ |
| $m = (p + q)/2$ | the mixture of $p$ and $q$ | `float64[n]` |
| $\mathrm{JS}(p, q)$ | Jensen-Shannon divergence | scalar in $[0, \ln 2]$ |
| $\pi, \pi_{\mathrm{ref}}$ | a policy and a reference model (`L12.3`) | distributions |
| $x \sim \pi$ | an outcome sampled from $\pi$ | index |
| $r = \log \pi_{\mathrm{ref}}(x) - \log \pi(x)$ | log ratio at a sample | scalar |
| $k_3 = e^r - r - 1$ | the k3 estimator of $\mathrm{KL}(\pi \,\Vert\, \pi_{\mathrm{ref}})$ | scalar, $\ge 0$ |
| $z$ | logits, with $\mathrm{softmax}(z)$ a distribution (`M09.2`) | `float64[n]` |

**Surprise.** An outcome of probability $p_i$ carries surprise $-\log p_i$: a certain outcome ($p_i = 1$) carries none, a rare one a lot, and the surprise of two independent outcomes adds, because $-\log(p_i p_j) = -\log p_i - \log p_j$. With $\log_2$ the unit is the bit: $-\log_2 p_i$ is the length of the best code word for outcome $i$.

**Entropy is the average surprise.**

$$H(p) = -\sum_i p_i \log p_i.$$

A term with $p_i = 0$ is defined as 0, because $x \log x \to 0$ as $x \to 0$: an outcome that never happens costs nothing. $H(p) \ge 0$ since every surprise is $\ge 0$, and $H(p) \le \log n$, with equality exactly for the uniform distribution: nothing is more unpredictable than equal chances.

**Cross-entropy measures a model against data.** If the data come from $p$ and you predict with $q$, your average surprise is

$$H(p, q) = -\sum_i p_i \log q_i.$$

This is the language-model loss. With $p$ the one-hot distribution of the observed next token $t$, it is $-\log q_t$, and averaging over positions gives the negative log-likelihood the bigram already prints. A term with $p_i > 0$ and $q_i = 0$ is $+\infty$: the model said impossible, and it happened.

**KL is the excess, and it is never negative.** Subtract the entropy:

$$\mathrm{KL}(p \,\Vert\, q) = H(p, q) - H(p) = \sum_i p_i \log \frac{p_i}{q_i}.$$

Proof that it is $\ge 0$ (Gibbs' inequality): $\log y \le y - 1$ for $y > 0$, with equality only at $y = 1$. Over the outcomes with $p_i > 0$,

$$-\mathrm{KL}(p \,\Vert\, q) = \sum_i p_i \log \frac{q_i}{p_i} \le \sum_i p_i \left(\frac{q_i}{p_i} - 1\right) = \sum_{i: p_i > 0} q_i - 1 \le 0.$$

Equality needs $q_i = p_i$ everywhere. So minimizing the cross-entropy over $q$ minimizes KL, and the loss floor is $H(p)$: the entropy of the data itself. Training a language model is pushing $\mathrm{KL}(\text{data} \,\Vert\, \text{model})$ toward 0.

**KL has a direction.** $\mathrm{KL}(p \,\Vert\, q)$ weighs the log ratio by $p$, so it punishes $q$ for being small where $p$ is large (it is "mass covering"), and is infinite if $q$ misses any outcome of $p$. $\mathrm{KL}(q \,\Vert\, p)$ punishes the opposite and prefers a $q$ that sits on one mode of $p$. They are different numbers (section 4 has one example), and `L12` picks one on purpose.

**KL from log-probabilities.** Models output log-probabilities ($\ell = \mathrm{logsoftmax}(z)$), which are finite even where the probability underflows. Write

$$\mathrm{KL}(p \,\Vert\, q) = \sum_i e^{\ell^p_i} \left(\ell^p_i - \ell^q_i\right),$$

a difference of moderate numbers, and never form $e^{\ell^p_i} / e^{\ell^q_i}$, which is $0/0$ or $\infty/\infty$ when the logits are large. Two edge cases: $\ell^p_i = -\infty$ means $p_i = 0$ and contributes 0 (even against $\ell^q_i = -\infty$, where the difference would be NaN); a finite $\ell^p_i$ against $\ell^q_i = -\infty$ is $+\infty$, even when $e^{\ell^p_i}$ underflows to 0 (where $0 \cdot \infty$ would be NaN).

**Jensen-Shannon is a symmetric, bounded cousin.** Compare both with their average $m = (p + q)/2$:

$$\mathrm{JS}(p, q) = \tfrac12 \mathrm{KL}(p \,\Vert\, m) + \tfrac12 \mathrm{KL}(q \,\Vert\, m).$$

$m$ is positive wherever $p$ or $q$ is, so JS is always finite, and since $m_i \ge p_i / 2$, every $\log(p_i / m_i) \le \log 2$: $\mathrm{JS} \le \ln 2$, reached by distributions with disjoint supports.

**Estimating KL from samples: k3.** When $n$ is a vocabulary and you only have the log-probabilities of the tokens you sampled, you estimate $\mathrm{KL}(\pi \,\Vert\, \pi_{\mathrm{ref}}) = \mathbb{E}_{x \sim \pi}[\log \pi(x) - \log \pi_{\mathrm{ref}}(x)] = \mathbb{E}[-r]$ by averaging over samples. The plain estimator $-r$ is unbiased but negative for many samples. Schulman's k3 adds a term with mean zero:

$$\mathbb{E}_{x \sim \pi}\left[e^{r}\right] = \sum_x \pi(x) \frac{\pi_{\mathrm{ref}}(x)}{\pi(x)} = \sum_x \pi_{\mathrm{ref}}(x) = 1,$$

so $k_3 = (e^r - 1) - r$ has the same mean, $\mathrm{KL}(\pi \,\Vert\, \pi_{\mathrm{ref}})$ (the sum runs over $\pi$'s support, which is everything for softmax outputs). And every sample is $\ge 0$, because the exponential lies above its tangent at 0: $e^r \ge 1 + r$.

**k3 for small $r$.** Near $r = 0$, $k_3 = r^2/2 + r^3/6 + \dots$ is tiny while $e^r \approx 1$. Computing $e^r - r - 1$ subtracts numbers near 1 and keeps only the rounding error of $e^r$, about $10^{-16}$, against a true value of $5 \times 10^{-13}$ at $r = 10^{-6}$. `np.expm1(r)` computes $e^r - 1$ directly to full relative precision, so $\mathrm{expm1}(r) - r$ keeps the answer. This is exactly the regime at the start of post-training, when the policy has barely moved.

**Entropy from logits.** With $\ell = \mathrm{logsoftmax}(z)$ from `M09.2`, $H = -\sum_i e^{\ell_i} \ell_i$, with masked entries ($\ell_i = -\infty$) contributing 0. Computing $p = \mathrm{softmax}(z)$ and then $-\sum p \log p$ meets $0 \cdot \log 0 = 0 \cdot (-\infty) = \mathrm{NaN}$ as soon as one probability underflows.

## 3. Worked example by hand

Take $n = 3$, $p = [1/2, 1/4, 1/4]$, $q = [1/4, 1/4, 1/2]$. Every log is a multiple of $\ln 2 = 0.693147$ except one.

**Entropy.** $H(p) = \tfrac12 \ln 2 + \tfrac14 \ln 4 + \tfrac14 \ln 4 = (\tfrac12 + \tfrac12 + \tfrac12) \ln 2 = 1.5 \ln 2 = 1.039721$ nats, which is 1.5 bits.

**Cross-entropy.** $H(p, q) = \tfrac12 \ln 4 + \tfrac14 \ln 4 + \tfrac14 \ln 2 = (1 + \tfrac12 + \tfrac14) \ln 2 = 1.75 \ln 2 = 1.213008$.

**KL.** Directly: $\tfrac12 \ln \frac{1/2}{1/4} + \tfrac14 \ln 1 + \tfrac14 \ln \frac{1/4}{1/2} = \tfrac12 \ln 2 - \tfrac14 \ln 2 = 0.25 \ln 2 = 0.173287$. Check: $H(p, q) - H(p) = 1.75 \ln 2 - 1.5 \ln 2$. Same number.

**Jensen-Shannon.** $m = [3/8, 1/4, 3/8]$.

| | $\log(p_i / m_i)$ | $p_i \log(p_i/m_i)$ |
|---|---|---|
| $i = 1$ | $\ln(4/3) = 0.287682$ | $0.143841$ |
| $i = 2$ | $\ln 1 = 0$ | 0 |
| $i = 3$ | $\ln(2/3) = -0.405465$ | $-0.101366$ |

So $\mathrm{KL}(p \,\Vert\, m) = 0.042475$; by symmetry of this example $\mathrm{KL}(q \,\Vert\, m)$ is the same, and $\mathrm{JS} = 0.042475 = 1.25 \ln 2 - 0.75 \ln 3$, well under $\ln 2$.

**k3.** Let $\pi = p$ and $\pi_{\mathrm{ref}} = q$. The log ratios $r = \log(q_x / p_x)$ are $[-\ln 2, 0, \ln 2]$:

| $x$ | $\pi(x)$ | $r$ | $k_3 = e^r - r - 1$ |
|---|---|---|---|
| 1 | 1/2 | $-0.693147$ | $0.5 + 0.693147 - 1 = 0.193147$ |
| 2 | 1/4 | 0 | 0 |
| 3 | 1/4 | $0.693147$ | $2 - 0.693147 - 1 = 0.306853$ |

The mean under $\pi$: $\tfrac12 \cdot 0.193147 + \tfrac14 \cdot 0.306853 = 0.096574 + 0.076713 = 0.173287 = 0.25 \ln 2$. Exactly $\mathrm{KL}(p \,\Vert\, q)$, as promised, and every row is nonnegative.

**Entropy from logits.** $z = [\ln 2, 0, 0]$ gives $e^z = [2, 1, 1]$, softmax $[1/2, 1/4, 1/4] = p$, so the entropy is again $1.5 \ln 2$.

These numbers are the first case in section 4.

## 4. The interface

```python
# python/tinyllm/info/entropy.py  (natural logs: nats)
def entropy(p: ArrayLike, axis: int = -1) -> NDArray
def cross_entropy(p: ArrayLike, q: ArrayLike, axis: int = -1) -> NDArray
def kl(p: ArrayLike, q: ArrayLike, axis: int = -1) -> NDArray
def kl_from_logprobs(logp: ArrayLike, logq: ArrayLike, axis: int = -1) -> NDArray
def js(p: ArrayLike, q: ArrayLike) -> NDArray          # last axis
def kl_k3(logp_ref: ArrayLike, logp: ArrayLike) -> NDArray   # elementwise, no reduction
def entropy_from_logits(z: ArrayLike, axis: int = -1) -> NDArray
```

Each reduces over `axis` and drops it, like `np.sum`. Probabilities are not renormalized, and a negative one is a `ValueError`. Note the argument order of `kl_k3`: the reference model first, as in $r = \log \pi_{\mathrm{ref}} - \log \pi$. The solution computes the zero conventions with `np.where` on the terms; adding a small epsilon inside the logs biases every answer and turns a true $+\infty$ into a large finite number.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example | unit | every section 3 number, in nats | everyone agrees on the definitions |
| agreement with SciPy | golden | scipy's entropy, KL, and JS on dense, sparse, peaked, and column-wise inputs | an independent implementation agrees |
| Gibbs' inequality | property | $\mathrm{KL} \ge 0$ on 200 random pairs, exactly 0 at $p = q$ | a KL penalty never rewards drift (`L12.3`) |
| cross-entropy decomposes | property | $H(p, q) = H(p) + \mathrm{KL}(p \,\Vert\, q)$ | reading the loss: floor plus excess |
| KL is not symmetric | unit | $[0.9, 0.1]$ vs uniform: 0.36806 one way, 0.51083 the other | forward and reverse KL are different objectives |
| zero-probability conventions | boundary | $0 \log 0 = 0$, $p \log(p/0) = +\infty$, no NaN | sparse distributions, masked tokens |
| negative probabilities rejected | boundary | a negative entry is a `ValueError` everywhere | a logit passed as a probability fails loudly |
| entropy bounds | property | 0 for one-hot, $\log n$ for uniform, in between otherwise | reading `L8.1`'s entropy log |
| axis and batch shapes | unit | `[B, V]` reduces to `[B]`; `axis=0` reduces columns | batched losses |
| JS properties | property | symmetric, in $[0, \ln 2]$, $\ln 2$ for disjoint supports | a bounded comparison of two models |
| KL from log-probabilities matches `kl` | differential | equals `kl` on the same distributions, $-\infty$ entries included | the form models use |
| KL from extreme logits | boundary | log-probabilities from logits near $10^4$ give a finite, correct KL | late training, low temperature |
| infinite log-probabilities | boundary | $-\infty$ against $-\infty$ is 0; finite (even $-800$) against $-\infty$ is $+\infty$ | no NaN from $\infty - \infty$ or $0 \cdot \infty$ |
| k3 is nonnegative | property | $k_3 \ge 0$ on 1000 samples, 0 at $r = 0$, $e - 2$ at $r = 1$ | per-sample penalties are never negative |
| k3 at small $r$ | boundary | relative error below $10^{-9}$ at $\lvert r\rvert \le 3 \times 10^{-5}$ | the start of `L12.3`, when the policy barely moved |
| k3's mean is KL | statistical | exact enumeration equals KL; 20000 samples within 4 standard errors | the estimator is unbiased |
| entropy from logits | boundary | equals $H(\mathrm{softmax}(z))$, finite at $10^4$, masks count as 0 | `L8.1` logs it from logits |

## 5. The solution, step by step

### 5.1 From first principles

1. **Every quantity is a sum of per-outcome terms.** $H$, $H(p, q)$, KL and JS all have the form $\sum_i t_i$, reduced over one axis. So each function builds an array of terms with the input's shape and calls `np.sum(..., axis=axis)` once; batching (`[B, V]` to `[B]`) then comes for free.
2. **The zero conventions are properties of single terms.** $0 \log 0 = 0$ and $p \log(p/0) = +\infty$ are statements about one $t_i$, so they are applied to the term array with `np.where` before the sum. They are not approximations, so no epsilon: $\log(q + 10^{-12})$ shifts every finite answer and turns a true $+\infty$ into a large finite number.
3. **NumPy will produce the wrong values first, and the `where` replaces them.** `0 * np.log(0)` is `0 * -inf = nan`, and `np.log(0)` warns. The solution lets NumPy compute the raw term under `np.errstate(...)` (silencing the warning it knows about) and then overwrites exactly the positions the convention covers. `np.where` evaluates both branches, which is why the raw computation must be allowed to produce `nan` quietly.
4. **KL needs both conventions, in a fixed priority.** $p_i = 0$ wins (the term is 0 even if $q_i = 0$); otherwise $q_i = 0$ gives $+\infty$. Writing it as `a * (log a - log b)` rather than `a * log(a / b)` keeps the two logs separate, which is the same form the log-probability version uses.
5. **From log-probabilities, the same priority, tested on the logs.** `logp = -inf` is $p = 0$, so the term is 0; otherwise `logq = -inf` is $+\infty$. Testing `np.isneginf(lq)` instead of the product matters: at `lp = -800`, `np.exp(lp)` underflows to 0 and `0 * inf` would be `nan`, but the true term is $+\infty$ because $p > 0$.
6. **JS reuses KL.** $m = (p + q)/2$ is positive wherever $p$ or $q$ is, so `kl(a, m)` never hits the $+\infty$ branch, and the result is finite and at most $\ln 2$ with no special code.
7. **k3 is one line once $r$ is right.** $r = \log \pi_{\mathrm{ref}} - \log \pi$, reference first, and $e^r - 1 - r$ is written `np.expm1(r) - r` so the $r^2/2$ survives when $r$ is tiny (section 2, last paragraphs).
8. **Entropy from logits goes through `log_softmax`.** `M09.2` already makes $\ell = z - \mathrm{LSE}(z)$ finite for huge logits, so the term is $e^{\ell}\ell$ with masked entries ($\ell = -\infty$) set to 0, the same convention as step 2 expressed on logs.

### 5.2 The code, annotated

Two private helpers carry the validation and the $0 \log 0$ rule.

```python
def _probs(p: ArrayLike, name: str) -> NDArray:
    """p as a float64 array, ValueError on a negative entry."""
    a = np.asarray(p, dtype=np.float64)
    # A negative "probability" is almost always a logit passed by mistake.
    # Fail loudly instead of returning log(negative) = nan.
    if (a < 0).any():
        raise ValueError(f"{name} has a negative probability (min {a.min()!r})")
    return a


def _xlogy(x: NDArray, y: NDArray) -> NDArray:
    """x * log(y) elementwise with 0 * log(anything) = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = x * np.log(y)          # may be 0 * -inf = nan where x == 0
    return np.where(x == 0, 0.0, t)  # the convention overrides those entries
```

`_xlogy` handles $x = 0$ but deliberately not $y = 0$ with $x > 0$: there `x * log(0)` is already $-\infty$, which is the right term for cross-entropy ($-(-\infty) = +\infty$ after the minus sign). Note that no function renormalizes its input: a row that sums to 0.9 gives the formula's value for that row, not for a corrected one.

```python
def entropy(p: ArrayLike, axis: int = -1) -> NDArray:
    a = _probs(p, "p")
    return -np.sum(_xlogy(a, a), axis=axis)        # H(p) = -sum p log p


def cross_entropy(p: ArrayLike, q: ArrayLike, axis: int = -1) -> NDArray:
    a, b = _probs(p, "p"), _probs(q, "q")
    return -np.sum(_xlogy(a, b), axis=axis)        # H(p, q) = -sum p log q
```

KL cannot use `_xlogy`, because it needs the second convention as well:

```python
def kl(p: ArrayLike, q: ArrayLike, axis: int = -1) -> NDArray:
    a, b = _probs(p, "p"), _probs(q, "q")
    with np.errstate(divide="ignore", invalid="ignore"):
        t = a * (np.log(a) - np.log(b))
    # p = 0 costs nothing; p > 0 where q = 0 is log(p / 0) = +inf.
    # The outer where is checked first, so p = q = 0 gives 0, not inf.
    t = np.where(a == 0, 0.0, np.where(b == 0, np.inf, t))
    return np.sum(t, axis=axis)
```

The log-probability version has the same structure with the tests moved onto the logs:

```python
def kl_from_logprobs(logp: ArrayLike, logq: ArrayLike, axis: int = -1) -> NDArray:
    lp = np.asarray(logp, dtype=np.float64)
    lq = np.asarray(logq, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        # A difference of moderate numbers: never exp(lp) / exp(lq).
        t = np.exp(lp) * (lp - lq)
    # logp = -inf is p = 0: no cost. logq = -inf under p > 0 is +inf, even when
    # exp(logp) underflows to 0 (0 * inf would be nan).
    t = np.where(np.isneginf(lp), 0.0, np.where(np.isneginf(lq), np.inf, t))
    return np.sum(t, axis=axis)
```

There is no `_probs` check here: log-probabilities are naturally negative, and validating them would reject every valid input.

```python
def js(p: ArrayLike, q: ArrayLike) -> NDArray:
    a, b = _probs(p, "p"), _probs(q, "q")
    m = 0.5 * (a + b)               # positive wherever a or b is: KL(., m) is finite
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def kl_k3(logp_ref: ArrayLike, logp: ArrayLike) -> NDArray:
    # Reference first: r = log pi_ref - log pi, so E_pi[e^r] = 1.
    r = np.asarray(logp_ref, dtype=np.float64) - np.asarray(logp, dtype=np.float64)
    # exp(r) - 1 - r loses everything to rounding when |r| is small: exp(r)
    # is 1 + r + r^2/2 and the 1 swallows the r^2/2. expm1 keeps it.
    return np.expm1(r) - r           # elementwise: one value per sample, no mean


def entropy_from_logits(z: ArrayLike, axis: int = -1) -> NDArray:
    lp = log_softmax(np.asarray(z, dtype=np.float64), axis=axis)   # M09.2, finite at 1e4
    with np.errstate(invalid="ignore"):
        t = np.exp(lp) * lp          # 0 * -inf = nan on masked entries...
    return -np.sum(np.where(np.isneginf(lp), 0.0, t), axis=axis)    # ...which are 0
```

`js` reduces over the last axis only (it calls `kl` with the default), which is what the contract says. `kl_k3` returns per-sample values because the caller (`L12.3`) adds them per token and averages over its own batch and mask; reducing here would throw away that structure.

Full solution: [`entropy.py`](../../../../course/solutions/python/tinyllm/info/entropy.py).

### 5.3 Why this solution

- **Cost.** Every function is one vectorized pass over the input: $O(n)$ time per row and $O(n)$ temporary memory for the term array. The `np.where` calls cost a few extra passes, negligible next to the `log` and `exp`.
- **`np.where` instead of an epsilon.** An epsilon is shorter, but it is wrong by construction: it changes finite answers, hides a true $+\infty$ (a model that assigned probability 0 to something that happened), and its "right size" depends on the dtype. The `where` gives exact conventions and the same results as SciPy.
- **`np.where` instead of boolean-mask indexing** (`t[a == 0] = 0`). Both work; `where` keeps every function a pure expression that also works on 0-d inputs, and composes cleanly for KL's two nested conventions.
- **Log-probability KL instead of `kl(exp(logp), exp(logq))`.** Exponentiating first loses everything below about $e^{-745}$ to zero and then needs the infinite-KL branch where the true answer is finite. Staying in logs keeps precision where models actually live.
- **`scipy.special.xlogy` / `rel_entr`** would do the same jobs, and the solution's `_xlogy` mirrors them; writing them out keeps the conventions visible and the module dependent only on NumPy and `M09.2`.
- **Tradeoff accepted.** No renormalization: a slightly unnormalized input gives a slightly off answer rather than being silently fixed. That surfaces bugs upstream (a softmax over the wrong axis) instead of hiding them.

### 5.4 Explain it back

1. Why does `kl` test `a == 0` before `b == 0`, and what would the other order return for $p_i = q_i = 0$?
   <details><summary>Answer</summary>
   An outcome that never happens under $p$ costs nothing, whatever $q$ says, so $p_i = 0$ must win. With the order reversed, $p_i = q_i = 0$ would give $+\infty$, and any sparse distribution compared with itself would have infinite KL instead of 0.
   </details>
2. In `kl_from_logprobs`, why test `np.isneginf(lq)` rather than relying on `np.exp(lp) * (lp - lq)` being $+\infty$?
   <details><summary>Answer</summary>
   When `lp` is finite but very negative (say $-800$), `np.exp(lp)` underflows to exactly 0, and `0 * inf` is `nan`. Mathematically $p > 0$, so the term is $+\infty$; only a test on the logs themselves can see that.
   </details>
3. Why is `kl_k3`'s argument order (reference first) essential, given that $e^r - r - 1 \ge 0$ for either sign of $r$?
   <details><summary>Answer</summary>
   The estimator is unbiased only because $\mathbb{E}_{x \sim \pi}[e^{r}] = 1$, which needs $r = \log(\pi_{\mathrm{ref}}/\pi)$ with samples from $\pi$. With the ratio flipped, every value is still nonnegative, so nothing looks wrong, but the mean is no longer $\mathrm{KL}(\pi \,\Vert\, \pi_{\mathrm{ref}})$.
   </details>
4. Why is `entropy_from_logits` not just `entropy(softmax(z))`?
   <details><summary>Answer</summary>
   The naive `-sum(p * log(p))` on `softmax(z)` gives `nan` as soon as one probability underflows to 0. Going through `log_softmax` gets $\ell_i = z_i - \mathrm{LSE}(z)$ directly, finite for every unmasked entry even at logits near $10^4$, instead of taking the log of an already-rounded probability; the only infinite entries are true masks, which the `where` sets to 0.
   </details>
5. Why does `js` never need the $+\infty$ branch?
   <details><summary>Answer</summary>
   KL is infinite only where the second argument is 0 and the first is not. The mixture $m_i = (p_i + q_i)/2$ is 0 only where both $p_i$ and $q_i$ are 0, and there the first argument is 0 too, so the term is 0.
   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. `p * np.log(p)` with $p = 0$, or an epsilon inside the logs | NaN entropy for sparse rows; a finite KL where it is infinite | computes the raw term, then overrides the convention positions with `np.where`; see `_xlogy` and `kl` in 5.2 |
| 2. forming $p/q$ from exponentiated log-probabilities | `0/0` and `inf/inf` once logits are large | `kl_from_logprobs` uses the difference `lp - lq`, weighted by `exp(lp)` |
| 3. computing $\mathrm{KL}(q \,\Vert\, p)$ when you meant $\mathrm{KL}(p \,\Vert\, q)$ | the right number only on symmetric examples | weights by the first argument everywhere: `a * (log a - log b)` |
| 4. `np.exp(r) - r - 1` | rounding noise instead of $r^2/2$ for small $r$ | `np.expm1(r) - r` in `kl_k3`; see step 7 |
| 5. `-sum(p * log(p))` with $p = \mathrm{softmax}(z)$ | NaN entropy as soon as one probability underflows | `entropy_from_logits` works on `log_softmax(z)` and masks $-\infty$ entries; see step 8 |
| 6. `np.log2` instead of the natural log | every number off by a factor of $\ln 2$ | uses `np.log` throughout: every result is in nats |
| 7. the k3 ratio upside down, $r = \log \pi - \log \pi_{\mathrm{ref}}$ | still nonnegative, but its mean is not KL | takes `logp_ref` first and computes `logp_ref - logp` |
| 8. letting $-\infty - (-\infty)$ or $0 \cdot \infty$ through | NaN where the answer is 0 or $+\infty$ | tests `np.isneginf` on `lp` first, then on `lq`; see step 5 |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M09.2` | `log_softmax` inside `entropy_from_logits`; the max shift behind every stable log-probability |
| Forward | `M08.3` | its cross-entropy VJP is checked as the gradient of this `cross_entropy` |
| Forward | `L0.3` | the training loss is $H(\text{onehot}, \mathrm{softmax}(z))$, fused and stable |
| Forward | `L8.1` | logs `entropy_from_logits` per sampling step |
| Forward | `L8.6` | speculative decoding accepts with probability $1 - \mathrm{TV}(p, q)$; JS and KL frame the same comparison |
| Forward | `L12.3` | the KL penalty to the reference model, per token, with `kl_k3` |
| Forward | `M11.2` | perplexity is $e^{H}$ of the per-token cross-entropy |
| Forward | `M11.4` | mutual information is the KL from the joint to the product of marginals |
| Forward | `L12.2`, `L12.4` | DPO and the other post-training objectives are written in terms of KL to a reference |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `entropy`, `kl` | `scipy.stats.entropy` | a `base` argument and normalization of its inputs | `scipy/stats/_entropy.py` |
| `kl_from_logprobs` | `torch.nn.functional.kl_div` | takes log $q$ as input and $p$ as target (note the order), `log_target`, batch reductions | `torch/nn/functional.py` |
| `kl_k3` | TRL's GRPO trainer | the per-token KL penalty $e^{r} - r - 1$ against the reference model | `trl/trainer/grpo_trainer.py` |
| `js` | `scipy.spatial.distance.jensenshannon` | returns the JS distance (the square root), with a `base` | `scipy/spatial/distance.py` |
