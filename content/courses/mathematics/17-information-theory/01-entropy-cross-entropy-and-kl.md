# Entropy, cross-entropy, KL, JS, and the k3 estimator

A language model's training loop prints a number, the negative log-likelihood, from its very first run. The byte bigram in [Byte bigram: counts, logits, safetensors v0](../../ai-engineering/06-numerics-and-autograd/01-byte-bigram.md) in *AI Engineering* already reports it, but nothing yet says what that number is measured against or how low it can go. Once you train the same bigram by gradient descent in [Training loop and the autograd bigram](../../ai-engineering/06-numerics-and-autograd/06-training-loop-and-the-autograd-bigram.md), with the cross-entropy loss of [Losses with a fused backward](../../ai-engineering/06-numerics-and-autograd/04-losses-with-fused-backward.md), the loss needs a meaning: what is the floor, and how far above it are you?

Information theory answers that, and the same few quantities keep returning later in the system. The sampler in [Sampling and logit processors](../../ai-engineering/21-inference/01-sampling-and-logit-processors.md) logs the entropy of every step so a collapsing temperature shows up in the traces. [Speculative decoding](../../ai-engineering/21-inference/06-speculative-decoding.md) reads its acceptance rate off the distance between two distributions. Post-training in [Group relative policy optimization with verifiable rewards](../../ai-engineering/18-post-training/03-grpo-with-verifiable-rewards.md) penalizes the policy for drifting from a reference model with a sampled KL. And [Matrix differentials, the trace trick, and closed-form VJPs](../14-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md) needs a forward cross-entropy to check its gradient against. Each of these needs the same definitions, with the same conventions at zero, in nats, and stable on the log-probabilities models actually produce.

This section builds entropy, cross-entropy, KL divergence, Jensen-Shannon divergence and Schulman's k3 estimator from the single idea of surprise, works every one by hand on a three-outcome example, and then walks through the reference implementation, [`entropy.py`](../../../../course/solutions/python/tinyllm/info/entropy.py) (contract: [`entropy.pyi`](../../../../course/contracts/py/tinyllm/info/entropy.pyi)), function by function. It builds on [Stable numerics: logsumexp, softmax, and compensated sums](../15-numerical-methods-and-floating-point/02-stable-numerics.md), whose `log_softmax` it calls. As you read the code, try to predict each line before you see it.

> **Learning objectives**
> - Define entropy, cross-entropy, KL and JS, and compute each by hand for a small distribution.
> - Prove Gibbs' inequality and explain why the cross-entropy loss has a floor at the data's entropy.
> - Compute KL and entropy from log-probabilities and logits without producing NaN or losing precision.
> - Explain why the k3 estimator is unbiased and nonnegative, and why it is computed with `expm1`.
> - Apply the conventions $0 \log 0 = 0$ and $p \log(p/0) = +\infty$ exactly, without an epsilon.

## Notation

The quantities in this section share a small vocabulary of symbols, collected here for reference.

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
| $\pi, \pi_{\mathrm{ref}}$ | a policy and a reference model (as in GRPO post-training) | distributions |
| $x \sim \pi$ | an outcome sampled from $\pi$ | index |
| $r = \log \pi_{\mathrm{ref}}(x) - \log \pi(x)$ | log ratio at a sample | scalar |
| $k_3 = e^r - r - 1$ | the k3 estimator of $\mathrm{KL}(\pi \,\Vert\, \pi_{\mathrm{ref}})$ | scalar, $\ge 0$ |
| $z$ | logits, with $\mathrm{softmax}(z)$ a distribution | `float64[n]` |

## Surprise and entropy

Start with a single outcome. An outcome of probability $p_i$ carries surprise $-\log p_i$. A certain outcome ($p_i = 1$) carries none, a rare one carries a lot, and the surprise of two independent outcomes adds, because $-\log(p_i p_j) = -\log p_i - \log p_j$. With $\log_2$ the unit is the bit, and $-\log_2 p_i$ is the length of the best code word for outcome $i$.

Entropy is the average surprise:

$$H(p) = -\sum_i p_i \log p_i.$$

A term with $p_i = 0$ is defined as 0, because $x \log x \to 0$ as $x \to 0$: an outcome that never happens costs nothing. $H(p) \ge 0$ since every surprise is $\ge 0$, and $H(p) \le \log n$, with equality exactly for the uniform distribution. Nothing is more unpredictable than equal chances.

## Cross-entropy: a model measured against data

Now suppose the data come from $p$ but you predict with a model $q$. Your average surprise is

$$H(p, q) = -\sum_i p_i \log q_i.$$

This is the language-model loss. With $p$ the one-hot distribution of the observed next token $t$, it is $-\log q_t$, and averaging over positions gives the negative log-likelihood the bigram already prints. A term with $p_i > 0$ and $q_i = 0$ is $+\infty$: the model said impossible, and it happened.

## KL divergence: the excess, never negative

Cross-entropy mixes two things: the inherent unpredictability of the data and the model's mistakes. Subtracting the entropy isolates the second:

$$\mathrm{KL}(p \,\Vert\, q) = H(p, q) - H(p) = \sum_i p_i \log \frac{p_i}{q_i}.$$

KL is never negative, a fact known as Gibbs' inequality. The proof uses $\log y \le y - 1$ for $y > 0$, with equality only at $y = 1$. Over the outcomes with $p_i > 0$,

$$-\mathrm{KL}(p \,\Vert\, q) = \sum_i p_i \log \frac{q_i}{p_i} \le \sum_i p_i \left(\frac{q_i}{p_i} - 1\right) = \sum_{i: p_i > 0} q_i - 1 \le 0.$$

Equality needs $q_i = p_i$ everywhere. So minimizing the cross-entropy over $q$ minimizes KL, and the loss floor is $H(p)$, the entropy of the data itself. Training a language model is pushing $\mathrm{KL}(\text{data} \,\Vert\, \text{model})$ toward 0; the training loss is the data's own entropy, which no model can beat, plus the model's excess.

KL also has a direction. $\mathrm{KL}(p \,\Vert\, q)$ weighs the log ratio by $p$, so it punishes $q$ for being small where $p$ is large (it is "mass covering"), and it is infinite if $q$ misses any outcome of $p$. $\mathrm{KL}(q \,\Vert\, p)$ punishes the opposite and prefers a $q$ that sits on one mode of $p$. The two are different numbers: for $[0.9, 0.1]$ against uniform, one direction gives 0.36806 and the other 0.51083. The post-training chapter of *AI Engineering* picks one direction on purpose, as [Distillation with forward and reverse KL](../../ai-engineering/18-post-training/04-distillation.md) shows.

## KL from log-probabilities

Models do not hand you probabilities; they output log-probabilities, $\ell = \mathrm{logsoftmax}(z)$, which stay finite even where the probability underflows. So write KL in terms of them:

$$\mathrm{KL}(p \,\Vert\, q) = \sum_i e^{\ell^p_i} \left(\ell^p_i - \ell^q_i\right),$$

a difference of moderate numbers. Never form $e^{\ell^p_i} / e^{\ell^q_i}$, which is $0/0$ or $\infty/\infty$ when the logits are large. Two edge cases need care. $\ell^p_i = -\infty$ means $p_i = 0$ and contributes 0, even against $\ell^q_i = -\infty$, where the difference would be NaN. A finite $\ell^p_i$ against $\ell^q_i = -\infty$ is $+\infty$, even when $e^{\ell^p_i}$ underflows to 0, where $0 \cdot \infty$ would be NaN.

## Jensen-Shannon: symmetric and bounded

KL's asymmetry and its infinities are sometimes exactly what you want and sometimes a nuisance. When you want a symmetric, always-finite comparison, compare both distributions with their average $m = (p + q)/2$:

$$\mathrm{JS}(p, q) = \tfrac12 \mathrm{KL}(p \,\Vert\, m) + \tfrac12 \mathrm{KL}(q \,\Vert\, m).$$

$m$ is positive wherever $p$ or $q$ is, so JS is always finite. And since $m_i \ge p_i / 2$, every $\log(p_i / m_i) \le \log 2$, so $\mathrm{JS} \le \ln 2$, a bound reached by distributions with disjoint supports.

## Estimating KL from samples: k3

When $n$ is a whole vocabulary and you only have the log-probabilities of the tokens you sampled, you cannot sum over every outcome. Instead you estimate $\mathrm{KL}(\pi \,\Vert\, \pi_{\mathrm{ref}}) = \mathbb{E}_{x \sim \pi}[\log \pi(x) - \log \pi_{\mathrm{ref}}(x)] = \mathbb{E}[-r]$ by averaging over samples. The plain estimator $-r$ is unbiased but negative for many samples, which is awkward for a per-token penalty. Schulman's k3 adds a term with mean zero:

$$\mathbb{E}_{x \sim \pi}\left[e^{r}\right] = \sum_x \pi(x) \frac{\pi_{\mathrm{ref}}(x)}{\pi(x)} = \sum_x \pi_{\mathrm{ref}}(x) = 1,$$

so $k_3 = (e^r - 1) - r$ has the same mean, $\mathrm{KL}(\pi \,\Vert\, \pi_{\mathrm{ref}})$. (The sum runs over $\pi$'s support, which is everything for softmax outputs.) And every sample is $\ge 0$, because the exponential lies above its tangent at 0: $e^r \ge 1 + r$.

There is a numerical trap in the formula. Near $r = 0$, $k_3 = r^2/2 + r^3/6 + \dots$ is tiny while $e^r \approx 1$. Computing $e^r - r - 1$ subtracts numbers near 1 and keeps only the rounding error of $e^r$, about $10^{-16}$, against a true value of $5 \times 10^{-13}$ at $r = 10^{-6}$. `np.expm1(r)` computes $e^r - 1$ directly to full relative precision, so $\mathrm{expm1}(r) - r$ keeps the answer. This is exactly the regime at the start of post-training, when the policy has barely moved.

## Entropy from logits

The same reasoning applies to entropy. With $\ell = \mathrm{logsoftmax}(z)$ from [Stable numerics](../15-numerical-methods-and-floating-point/02-stable-numerics.md), $H = -\sum_i e^{\ell_i} \ell_i$, with masked entries ($\ell_i = -\infty$) contributing 0. Computing $p = \mathrm{softmax}(z)$ and then $-\sum p \log p$ meets $0 \cdot \log 0 = 0 \cdot (-\infty) = \mathrm{NaN}$ as soon as one probability underflows.

## Working it by hand

With the definitions in place, it helps to see every one of them produce a number. Take $n = 3$, $p = [1/2, 1/4, 1/4]$, $q = [1/4, 1/4, 1/2]$. Every log is a multiple of $\ln 2 = 0.693147$ except one.

The **entropy** is $H(p) = \tfrac12 \ln 2 + \tfrac14 \ln 4 + \tfrac14 \ln 4 = (\tfrac12 + \tfrac12 + \tfrac12) \ln 2 = 1.5 \ln 2 = 1.039721$ nats, which is 1.5 bits.

The **cross-entropy** is $H(p, q) = \tfrac12 \ln 4 + \tfrac14 \ln 4 + \tfrac14 \ln 2 = (1 + \tfrac12 + \tfrac14) \ln 2 = 1.75 \ln 2 = 1.213008$.

The **KL divergence**, computed directly, is $\tfrac12 \ln \frac{1/2}{1/4} + \tfrac14 \ln 1 + \tfrac14 \ln \frac{1/4}{1/2} = \tfrac12 \ln 2 - \tfrac14 \ln 2 = 0.25 \ln 2 = 0.173287$. As a check, $H(p, q) - H(p) = 1.75 \ln 2 - 1.5 \ln 2$, the same number.

For **Jensen-Shannon**, the mixture is $m = [3/8, 1/4, 3/8]$, and the terms of $\mathrm{KL}(p \,\Vert\, m)$ are:

| | $\log(p_i / m_i)$ | $p_i \log(p_i/m_i)$ |
|---|---|---|
| $i = 1$ | $\ln(4/3) = 0.287682$ | $0.143841$ |
| $i = 2$ | $\ln 1 = 0$ | 0 |
| $i = 3$ | $\ln(2/3) = -0.405465$ | $-0.101366$ |

So $\mathrm{KL}(p \,\Vert\, m) = 0.042475$. By the symmetry of this example $\mathrm{KL}(q \,\Vert\, m)$ is the same, and $\mathrm{JS} = 0.042475 = 1.25 \ln 2 - 0.75 \ln 3$, well under $\ln 2$.

For **k3**, let $\pi = p$ and $\pi_{\mathrm{ref}} = q$. The log ratios $r = \log(q_x / p_x)$ are $[-\ln 2, 0, \ln 2]$:

| $x$ | $\pi(x)$ | $r$ | $k_3 = e^r - r - 1$ |
|---|---|---|---|
| 1 | 1/2 | $-0.693147$ | $0.5 + 0.693147 - 1 = 0.193147$ |
| 2 | 1/4 | 0 | 0 |
| 3 | 1/4 | $0.693147$ | $2 - 0.693147 - 1 = 0.306853$ |

The mean under $\pi$ is $\tfrac12 \cdot 0.193147 + \tfrac14 \cdot 0.306853 = 0.096574 + 0.076713 = 0.173287 = 0.25 \ln 2$. That is exactly $\mathrm{KL}(p \,\Vert\, q)$, as promised, and every row is nonnegative.

Finally, **entropy from logits**: $z = [\ln 2, 0, 0]$ gives $e^z = [2, 1, 1]$ and softmax $[1/2, 1/4, 1/4] = p$, so the entropy is again $1.5 \ln 2$.

These numbers are the first test case any implementation should pass.

## Designing the interface

The implementation lives in one module, with natural logs throughout so every result is in nats:

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

Each function reduces over `axis` and drops it, like `np.sum`. Probabilities are not renormalized, and a negative one is a `ValueError`. Note the argument order of `kl_k3`: the reference model comes first, as in $r = \log \pi_{\mathrm{ref}} - \log \pi$. The zero conventions are computed with `np.where` on the individual terms; adding a small epsilon inside the logs would bias every answer and turn a true $+\infty$ into a large finite number.

## Building it step by step

The reasoning behind the implementation is short, and each step follows from the one before.

1. **Every quantity is a sum of per-outcome terms.** $H$, $H(p, q)$, KL and JS all have the form $\sum_i t_i$, reduced over one axis. So each function builds an array of terms with the input's shape and calls `np.sum(..., axis=axis)` once; batching (`[B, V]` to `[B]`) then comes for free.
2. **The zero conventions are properties of single terms.** $0 \log 0 = 0$ and $p \log(p/0) = +\infty$ are statements about one $t_i$, so they are applied to the term array with `np.where` before the sum. They are not approximations, so there is no epsilon: $\log(q + 10^{-12})$ shifts every finite answer and turns a true $+\infty$ into a large finite number.
3. **NumPy will produce the wrong values first, and the `where` replaces them.** `0 * np.log(0)` is `0 * -inf = nan`, and `np.log(0)` warns. The solution lets NumPy compute the raw term under `np.errstate(...)`, silencing the warning it knows about, and then overwrites exactly the positions the convention covers. `np.where` evaluates both branches, which is why the raw computation must be allowed to produce `nan` quietly.
4. **KL needs both conventions, in a fixed priority.** $p_i = 0$ wins (the term is 0 even if $q_i = 0$); otherwise $q_i = 0$ gives $+\infty$. Writing it as `a * (log a - log b)` rather than `a * log(a / b)` keeps the two logs separate, the same form the log-probability version uses.
5. **From log-probabilities, the same priority is tested on the logs.** `logp = -inf` is $p = 0$, so the term is 0; otherwise `logq = -inf` is $+\infty$. Testing `np.isneginf(lq)` instead of the product matters: at `lp = -800`, `np.exp(lp)` underflows to 0 and `0 * inf` would be `nan`, but the true term is $+\infty$ because $p > 0$.
6. **JS reuses KL.** $m = (p + q)/2$ is positive wherever $p$ or $q$ is, so `kl(a, m)` never hits the $+\infty$ branch, and the result is finite and at most $\ln 2$ with no special code.
7. **k3 is one line once $r$ is right.** $r = \log \pi_{\mathrm{ref}} - \log \pi$, reference first, and $e^r - 1 - r$ is written `np.expm1(r) - r` so the $r^2/2$ survives when $r$ is tiny.
8. **Entropy from logits goes through `log_softmax`.** The stable-numerics section already makes $\ell = z - \mathrm{LSE}(z)$ finite for huge logits, so the term is $e^{\ell}\ell$ with masked entries ($\ell = -\infty$) set to 0, the same convention as step 2 expressed on logs.

Now the code. Two private helpers carry the validation and the $0 \log 0$ rule:

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

`_xlogy` handles $x = 0$ but deliberately not $y = 0$ with $x > 0$: there `x * log(0)` is already $-\infty$, which is the right term for cross-entropy ($-(-\infty) = +\infty$ after the minus sign). Note too that no function renormalizes its input: a row that sums to 0.9 gives the formula's value for that row, not for a corrected one.

With the helpers, entropy and cross-entropy are one line each:

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

The last three functions are short because the hard decisions have already been made:

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

`js` reduces over the last axis only (it calls `kl` with the default), which is what the contract says. `kl_k3` returns per-sample values because its caller in GRPO post-training adds them per token and averages over its own batch and mask; reducing here would throw away that structure.

The full solution is [`entropy.py`](../../../../course/solutions/python/tinyllm/info/entropy.py).

### Why this design

Every function is one vectorized pass over the input: $O(n)$ time per row and $O(n)$ temporary memory for the term array. The `np.where` calls cost a few extra passes, negligible next to the `log` and `exp`.

The choice of `np.where` over an epsilon is the central one. An epsilon is shorter, but it is wrong by construction: it changes finite answers, hides a true $+\infty$ (a model that assigned probability 0 to something that happened), and its "right size" depends on the dtype. The `where` gives exact conventions and the same results as SciPy. Boolean-mask indexing (`t[a == 0] = 0`) would also work, but `where` keeps every function a pure expression that also works on 0-d inputs, and it composes cleanly for KL's two nested conventions.

Computing KL from log-probabilities, rather than as `kl(exp(logp), exp(logq))`, matters because exponentiating first loses everything below about $e^{-745}$ to zero and then needs the infinite-KL branch where the true answer is finite. Staying in logs keeps precision where models actually live. SciPy's `scipy.special.xlogy` and `rel_entr` would do the same jobs, and `_xlogy` mirrors them; writing them out keeps the conventions visible and the module dependent only on NumPy and `log_softmax`.

One tradeoff is accepted deliberately: there is no renormalization. A slightly unnormalized input gives a slightly off answer rather than being silently fixed. That surfaces bugs upstream, such as a softmax over the wrong axis, instead of hiding them.

## What can go wrong

A correct implementation has to survive a specific set of checks. It must reproduce every hand-worked number above, in nats, and agree with SciPy's entropy, KL and JS on dense, sparse, peaked and column-wise inputs. It must satisfy Gibbs' inequality ($\mathrm{KL} \ge 0$ on 200 random pairs, exactly 0 at $p = q$), which guarantees a KL penalty never rewards drift, and the decomposition $H(p, q) = H(p) + \mathrm{KL}(p \,\Vert\, q)$, which is how you read a loss as floor plus excess. KL must not be symmetric: $[0.9, 0.1]$ against uniform gives 0.36806 one way and 0.51083 the other, because forward and reverse KL are different objectives.

At the boundaries, $0 \log 0 = 0$, $p \log(p/0) = +\infty$, and no NaN may appear, which is what sparse distributions and masked tokens need. A negative entry must be a `ValueError` everywhere, so a logit passed as a probability fails loudly. Entropy must be 0 for a one-hot distribution, $\log n$ for uniform, and in between otherwise; that is how you read the sampler's entropy log. Shapes must batch: `[B, V]` reduces to `[B]`, and `axis=0` reduces columns. JS must be symmetric, lie in $[0, \ln 2]$, and equal $\ln 2$ for disjoint supports.

The log-space functions carry their own checks. `kl_from_logprobs` must equal `kl` on the same distributions, $-\infty$ entries included; it must give a finite, correct KL from logits near $10^4$ (late training, low temperature); and $-\infty$ against $-\infty$ must be 0 while a finite value (even $-800$) against $-\infty$ is $+\infty$, with no NaN from $\infty - \infty$ or $0 \cdot \infty$. The k3 estimator must be nonnegative on 1000 samples, 0 at $r = 0$ and $e - 2$ at $r = 1$; it must have relative error below $10^{-9}$ at $\lvert r\rvert \le 3 \times 10^{-5}$, the regime at the start of post-training; and its mean must be KL, exactly by enumeration and within 4 standard errors over 20000 samples. Entropy from logits must equal $H(\mathrm{softmax}(z))$, stay finite at $10^4$, and count masks as 0.

Most failures come from a short list of mistakes, summarized here with how the solution avoids each:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| `p * np.log(p)` with $p = 0$, or an epsilon inside the logs | NaN entropy for sparse rows; a finite KL where it is infinite | computes the raw term, then overrides the convention positions with `np.where` (`_xlogy` and `kl`) |
| forming $p/q$ from exponentiated log-probabilities | `0/0` and `inf/inf` once logits are large | `kl_from_logprobs` uses the difference `lp - lq`, weighted by `exp(lp)` |
| computing $\mathrm{KL}(q \,\Vert\, p)$ when you meant $\mathrm{KL}(p \,\Vert\, q)$ | the right number only on symmetric examples | weights by the first argument everywhere: `a * (log a - log b)` |
| `np.exp(r) - r - 1` | rounding noise instead of $r^2/2$ for small $r$ | `np.expm1(r) - r` in `kl_k3` |
| `-sum(p * log(p))` with $p = \mathrm{softmax}(z)$ | NaN entropy as soon as one probability underflows | `entropy_from_logits` works on `log_softmax(z)` and masks $-\infty$ entries |
| `np.log2` instead of the natural log | every number off by a factor of $\ln 2$ | uses `np.log` throughout: every result is in nats |
| the k3 ratio upside down, $r = \log \pi - \log \pi_{\mathrm{ref}}$ | still nonnegative, but its mean is not KL | takes `logp_ref` first and computes `logp_ref - logp` |
| letting $-\infty - (-\infty)$ or $0 \cdot \infty$ through | NaN where the answer is 0 or $+\infty$ | tests `np.isneginf` on `lp` first, then on `lq` |

> **Warning:** A flipped k3 ratio is the most dangerous of these, because nothing looks wrong. Every value stays nonnegative; only the mean is off.

## Where this leads

This section leans backward on [Stable numerics](../15-numerical-methods-and-floating-point/02-stable-numerics.md) for `log_softmax` and the max shift behind every stable log-probability. Looking forward, [Matrix differentials, the trace trick, and closed-form VJPs](../14-matrix-calculus-and-autodiff/03-matrix-differentials-and-vjps.md) checks its cross-entropy VJP as the gradient of this `cross_entropy`. In *AI Engineering*, the training loss of [Losses with a fused backward](../../ai-engineering/06-numerics-and-autograd/04-losses-with-fused-backward.md) is $H(\text{onehot}, \mathrm{softmax}(z))$, fused and stable; [Sampling and logit processors](../../ai-engineering/21-inference/01-sampling-and-logit-processors.md) logs `entropy_from_logits` per sampling step; [Speculative decoding](../../ai-engineering/21-inference/06-speculative-decoding.md) accepts with probability $1 - \mathrm{TV}(p, q)$, a comparison JS and KL frame the same way; and [GRPO with verifiable rewards](../../ai-engineering/18-post-training/03-grpo-with-verifiable-rewards.md) applies the KL penalty to the reference model per token with `kl_k3`. [Direct preference optimization and IPO](../../ai-engineering/18-post-training/02-direct-preference-optimization.md) and [Distillation with forward and reverse KL](../../ai-engineering/18-post-training/04-distillation.md) write their objectives in terms of KL to a reference. Within this chapter, [Perplexity, bits per byte, NLL accumulator](02-perplexity-bits-per-byte-nll-accumulator.md) turns the per-token cross-entropy into perplexity $e^{H}$, and [Mutual information, PMI, and PPMI](03-mutual-information-pmi-ppmi.md) defines mutual information as the KL from the joint to the product of marginals.

The production equivalents of these functions are worth reading once you know what to look for. `scipy.stats.entropy` (in `scipy/stats/_entropy.py`) adds a `base` argument and normalizes its inputs. `torch.nn.functional.kl_div` (in `torch/nn/functional.py`) takes $\log q$ as input and $p$ as target (note the order), with `log_target` and batch reductions. TRL's GRPO trainer (`trl/trainer/grpo_trainer.py`) applies the per-token penalty $e^{r} - r - 1$ against the reference model. `scipy.spatial.distance.jensenshannon` (in `scipy/spatial/distance.py`) returns the JS distance, the square root, with a `base`.

### Further reading

- Cover and Thomas, *Elements of Information Theory* (2nd ed.), chapter 2.
- MacKay, *Information Theory, Inference, and Learning Algorithms*, chapters 2 and 4.
- Schulman, "Approximating KL Divergence" (2020 blog post).

## Key terms

- **Surprise**: $-\log p_i$, the information carried by an outcome of probability $p_i$.
- **Entropy**: $H(p) = -\sum_i p_i \log p_i$, the average surprise of a distribution.
- **Cross-entropy**: $H(p, q) = -\sum_i p_i \log q_i$, the average surprise when data from $p$ are predicted with $q$.
- **KL divergence**: $\mathrm{KL}(p \,\Vert\, q) = H(p, q) - H(p)$, the excess cost of predicting with $q$; nonnegative and asymmetric.
- **Jensen-Shannon divergence**: the average KL of $p$ and $q$ to their mixture; symmetric and bounded by $\ln 2$.
- **k3 estimator**: $e^r - r - 1$ with $r = \log \pi_{\mathrm{ref}} - \log \pi$, an unbiased, nonnegative per-sample estimate of $\mathrm{KL}(\pi \,\Vert\, \pi_{\mathrm{ref}})$.
- **Nat**: the unit of information with natural logs; one bit is $\ln 2$ nats.

## Summary

- Entropy $H(p) = -\sum_i p_i \log p_i$ is the average surprise of a distribution, between 0 (certain) and $\log n$ (uniform).
- Cross-entropy splits as $H(p, q) = H(p) + \mathrm{KL}(p \,\Vert\, q)$: the training loss is the data's own entropy, which no model can beat, plus the model's excess.
- $\mathrm{KL} \ge 0$ with equality only at $p = q$ (Gibbs), and it is not symmetric; Jensen-Shannon is symmetric and bounded by $\ln 2$.
- From model outputs, compute KL with log-probabilities, never with $p / q$; the k3 estimator $e^r - r - 1$ estimates KL from samples without bias and is never negative.
- Apply the zero conventions exactly with `np.where`; an epsilon biases every answer and hides true infinities.

## Review questions

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
6. What is the floor of the cross-entropy training loss, and why can no model go below it?
   <details><summary>Answer</summary>
   The floor is $H(p)$, the entropy of the data. Since $H(p, q) = H(p) + \mathrm{KL}(p \,\Vert\, q)$ and Gibbs' inequality gives $\mathrm{KL} \ge 0$ with equality only at $q = p$, the loss can reach $H(p)$ only when the model matches the data distribution exactly.
   </details>
