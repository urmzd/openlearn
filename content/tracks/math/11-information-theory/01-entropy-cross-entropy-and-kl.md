<!-- ol:module M11.1 -->
# Entropy, cross-entropy, KL, JS, and the k3 estimator

## Overview

| | |
|---|---|
| **Module** | `M11.1` · build · Python · Pass 2 · 2 to 3 h |
| **You build** | `python/tinyllm/info/entropy.py`: `entropy`, `cross_entropy`, `kl`, `kl_from_logprobs`, `js`, `kl_k3`, `entropy_from_logits` |
| **Contract** | [`course/contracts/py/tinyllm/info/entropy.pyi`](../../../../course/contracts/py/tinyllm/info/entropy.pyi) |
| **Tests** | `course/tests/M11.1/` (what they check: section 4) |
| **Needs** | `M09.2` stable numerics (`log_softmax`, or `--ref-deps`) |
| **Used by** | `M08.3` differentiates this cross-entropy · `L0.3` the training loss · later: `L8.1` per-step entropy logging, `L8.6` acceptance rate as $1 - \mathrm{TV}$, `L12.3` the KL penalty with `kl_k3`, `M11.2` perplexity · later: `L12.2`, `L12.4`, `M11.4` |
| **Milestone** | `MS-P2` (Pass 2 gate: every math module of the pass checks green, then your autograd bigram trains) |
| **Optional depth** | Cover and Thomas, *Elements of Information Theory* (2nd ed.), ch. 2; MacKay, *Information Theory, Inference, and Learning Algorithms*, ch. 2 and 4; Schulman, "Approximating KL Divergence" (2020 blog post) |

## Key Takeaways

- Entropy $H(p) = -\sum_i p_i \log p_i$ is the average surprise of a distribution, between 0 (certain) and $\log n$ (uniform) (`test_entropy_bounds`).
- Cross-entropy splits as $H(p, q) = H(p) + \mathrm{KL}(p \,\Vert\, q)$: the training loss is the data's own entropy, which no model can beat, plus the model's excess (`test_cross_entropy_decomposes`).
- $\mathrm{KL} \ge 0$ with equality only at $p = q$ (Gibbs), and it is not symmetric; Jensen-Shannon is symmetric and bounded by $\ln 2$ (`test_gibbs_kl_nonnegative`, `test_kl_is_not_symmetric`, `test_js_properties`).
- From model outputs, compute KL with log-probabilities, never with $p / q$; the k3 estimator $e^r - r - 1$ estimates KL from samples without bias and is never negative (`test_kl_from_logprobs_extreme_logits`, `test_kl_k3_mean_is_kl`).

## How to work this chapter

```bash
ol start M11.1              # stubs entropy.py into your repo, contract alongside
ol tests M11.1              # read the test catalog first: rung R0, you write no tests here
ol check M11.1              # exit code is the verdict
ol check M11.1 --ref-deps   # only if your M09.2 is not passing yet
ol diff  M11.1              # after passing: your code against the reference
```

---

## 1. Why now

The tracer bigram (`L0.0`) already reports a negative log-likelihood, and `train bigram` prints it, but nothing in your system says what that number is measured against or what it can reach. In this pass you train the same bigram by gradient descent (`L0.5`) with a cross-entropy loss (`L0.3`), and the loss needs a meaning: how far above the floor are we, and what is the floor? Later the sampler (`L8.1`) logs the entropy of every step so a collapsing temperature is visible in your traces, speculative decoding (`L8.6`) reads its acceptance rate off the distance between two distributions, and post-training (`L12.3`) penalizes the policy for drifting from the reference model with a sampled KL. Each needs the same few quantities with the same conventions at zero, in nats, stable on the log-probabilities your models actually produce. And `M08.3` needs a forward cross-entropy to check its gradient against.

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

This is the language-model loss. With $p$ the one-hot distribution of the observed next token $t$, it is $-\log q_t$, and averaging over positions gives the negative log-likelihood your bigram already prints. A term with $p_i > 0$ and $q_i = 0$ is $+\infty$: the model said impossible, and it happened.

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

These numbers are the first case in section 4, `test_hand_example`.

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

Each reduces over `axis` and drops it, like `np.sum`. Probabilities are not renormalized, and a negative one is a `ValueError`. Note the argument order of `kl_k3`: the reference model first, as in $r = \log \pi_{\mathrm{ref}} - \log \pi$. Compute the zero conventions with `np.where` on the terms; adding a small epsilon inside the logs biases every answer and turns a true $+\infty$ into a large finite number.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | every section 3 number, in nats | you and the test agree on the definitions |
| `test_matches_scipy_golden` | golden | scipy's entropy, KL, and JS on dense, sparse, peaked, and column-wise inputs | an independent implementation agrees |
| `test_gibbs_kl_nonnegative` | property | $\mathrm{KL} \ge 0$ on 200 random pairs, exactly 0 at $p = q$ | a KL penalty never rewards drift (`L12.3`) |
| `test_cross_entropy_decomposes` | property | $H(p, q) = H(p) + \mathrm{KL}(p \,\Vert\, q)$ | reading the loss: floor plus excess |
| `test_kl_is_not_symmetric` | unit | $[0.9, 0.1]$ vs uniform: 0.36806 one way, 0.51083 the other | forward and reverse KL are different objectives |
| `test_zero_probability_conventions` | boundary | $0 \log 0 = 0$, $p \log(p/0) = +\infty$, no NaN | sparse distributions, masked tokens |
| `test_rejects_negative_probabilities` | boundary | a negative entry is a `ValueError` everywhere | a logit passed as a probability fails loudly |
| `test_entropy_bounds` | property | 0 for one-hot, $\log n$ for uniform, in between otherwise | reading `L8.1`'s entropy log |
| `test_axis_and_batch_shapes` | unit | `[B, V]` reduces to `[B]`; `axis=0` reduces columns | batched losses |
| `test_js_properties` | property | symmetric, in $[0, \ln 2]$, $\ln 2$ for disjoint supports | a bounded comparison of two models |
| `test_kl_from_logprobs_matches_kl` | differential | equals `kl` on the same distributions, $-\infty$ entries included | the form models use |
| `test_kl_from_logprobs_extreme_logits` | boundary | log-probabilities from logits near $10^4$ give a finite, correct KL | late training, low temperature |
| `test_kl_from_logprobs_infinite_cases` | boundary | $-\infty$ against $-\infty$ is 0; finite (even $-800$) against $-\infty$ is $+\infty$ | no NaN from $\infty - \infty$ or $0 \cdot \infty$ |
| `test_kl_k3_is_nonnegative` | property | $k_3 \ge 0$ on 1000 samples, 0 at $r = 0$, $e - 2$ at $r = 1$ | per-sample penalties are never negative |
| `test_kl_k3_small_r_precision` | boundary | relative error below $10^{-9}$ at $\lvert r\rvert \le 3 \times 10^{-5}$ | the start of `L12.3`, when the policy barely moved |
| `test_kl_k3_mean_is_kl` | statistical | exact enumeration equals KL; 20000 samples within 4 standard errors | the estimator is unbiased |
| `test_entropy_from_logits` | boundary | equals $H(\mathrm{softmax}(z))$, finite at $10^4$, masks count as 0 | `L8.1` logs it from logits |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. `p * np.log(p)` with $p = 0$, or an epsilon inside the logs | NaN entropy for sparse rows; a finite KL where it is infinite | `test_zero_probability_conventions` (mutants `s01`, `s02`) |
| 2. forming $p/q$ from exponentiated log-probabilities | `0/0` and `inf/inf` once logits are large | `test_kl_from_logprobs_extreme_logits` (mutant `s05`) |
| 3. computing $\mathrm{KL}(q \,\Vert\, p)$ when you meant $\mathrm{KL}(p \,\Vert\, q)$ | the right number only on symmetric examples | `test_kl_is_not_symmetric` (mutant `s03`) |
| 4. `np.exp(r) - r - 1` | rounding noise instead of $r^2/2$ for small $r$ | `test_kl_k3_small_r_precision` (mutant `s07`) |
| 5. `-sum(p * log(p))` with $p = \mathrm{softmax}(z)$ | NaN entropy as soon as one probability underflows | `test_entropy_from_logits` (mutant `s10`) |
| 6. `np.log2` instead of the natural log | every number off by a factor of $\ln 2$ | `test_hand_example` (mutant `s11`) |
| 7. the k3 ratio upside down, $r = \log \pi - \log \pi_{\mathrm{ref}}$ | still nonnegative, but its mean is not KL | `test_kl_k3_mean_is_kl` (mutant `s08`) |
| 8. letting $-\infty - (-\infty)$ or $0 \cdot \infty$ through | NaN where the answer is 0 or $+\infty$ | `test_kl_from_logprobs_infinite_cases` (mutants `s06`, `s13`) |

## 6. Where it's used next
| Forward | `L12.2` | Registered module relationship. |
| Forward | `L12.4` | Registered module relationship. |
| Forward | `M11.4` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M09.2` | `log_softmax` inside `entropy_from_logits`; the max shift behind every stable log-probability |
| Forward | `M08.3` | its cross-entropy VJP is checked as the gradient of this `cross_entropy` |
| Forward | `L0.3` | the training loss is $H(\text{onehot}, \mathrm{softmax}(z))$, fused and stable |
| Forward | `L8.1` | logs `entropy_from_logits` per sampling step |
| Forward | `L8.6` | speculative decoding accepts with probability $1 - \mathrm{TV}(p, q)$; JS and KL frame the same comparison |
| Forward | `L12.3` | the KL penalty to the reference model, per token, with `kl_k3` |
| Forward | `M11.2` | perplexity is $e^{H}$ of the per-token cross-entropy |

If you skip this module, `ol check M08.3` stops with `M08.3 needs M11.1`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `entropy`, `kl` | `scipy.stats.entropy` | a `base` argument and normalization of its inputs | `scipy/stats/_entropy.py` |
| `kl_from_logprobs` | `torch.nn.functional.kl_div` | takes log $q$ as input and $p$ as target (note the order), `log_target`, batch reductions | `torch/nn/functional.py` |
| `kl_k3` | TRL's GRPO trainer | the per-token KL penalty $e^{r} - r - 1$ against the reference model | `trl/trainer/grpo_trainer.py` |
| `js` | `scipy.spatial.distance.jensenshannon` | returns the JS distance (the square root), with a `base` | `scipy/spatial/distance.py` |
