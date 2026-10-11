<!-- ol:module S-M11a -->
# Information theory problem set, part a: entropy, chain rules, KL and cross-entropy

## Overview

| | |
|---|---|
| **Module** | `S-M11a` · solve · none · Pass 2 · 3 to 4 h |
| **You build** | answers in `solve/S-M11a.toml` (16 checked by SymPy) and 2 proofs in `solve/S-M11a/qN.md` (self-graded against their rubrics) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M11a/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M11a/problems.md` and in section 4 |
| **Needs** | `S-M07a` (distributions, expectation, Bayes). Reading: the [Information Theory topic](README.md), entropy and KL sections |
| **Used by** | no call site (a solve set). It checks the definitions behind `M11.1` (`entropy`, `cross_entropy`, `kl`, `kl_from_logprobs`, `js`, `kl_k3`), which `M08.3`, `L0.3`, `L8.1`, and `L12.3` call; part b, `S-M11b` in Pass 3, covers coding, mutual information, and maximum entropy |
| **Milestone** | `MS-P2` (the Pass 2 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | Cover and Thomas, *Elements of Information Theory*, ch. 2; MacKay, *Information Theory, Inference, and Learning Algorithms* (free), ch. 2 and 4; Schulman, "Approximating KL Divergence" (2020 blog note) for k3 |

## Key Takeaways

- Entropy is the expected surprise $-\log p$; it is $\log V$ for a uniform distribution over $V$ outcomes and less for anything else, so an untrained byte model starts at $\log 256 \approx 5.55$ nats (q1, q2, q7).
- The unit is the base of the logarithm: bits for $\log_2$, nats for $\ln$; mixing them is the most common wrong answer (q1, q3, q7).
- The chain rule $H(X, Y) = H(X) + H(Y \mid X)$ is why next-token cross-entropy summed over a sequence is the sequence's log-loss (q4, q5).
- Cross-entropy is entropy plus KL divergence, KL is never negative, and it is not symmetric (q6, q9).
- The k3 estimator $e^r - r - 1$ is nonnegative for every sample, unlike the plain log-ratio (q8).

## How to work this chapter

```bash
ol start S-M11a             # writes solve/S-M11a.toml and one file per proof
ol check S-M11a             # SymPy checks the answers, then asks each proof rubric (y/n)
ol check S-M11a --regrade   # ask the rubrics again after you change a proof
```

---

## 1. Why now

Your Pass 1 bigram reported a loss of about 3.2 nats per byte, and the number meant nothing yet. In Pass 2 it starts to carry weight: `L0.3` trains with cross-entropy, `M11.1` writes `entropy`, `cross_entropy`, and `kl` that the sampler (`L8.1`) and the evaluation suite (`L6.7`) log, and the gap between your model's loss and the data's entropy is the KL divergence your optimizer is shrinking. Later, `L12.3` keeps a fine-tuned policy close to its reference with a KL penalty estimated by k3. This set fixes the definitions, the units, and the identities (chain rule, cross-entropy equals entropy plus KL, KL is nonnegative) before you code them.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $p, q$ | probability distributions over a finite set, $p_i \ge 0$, $\sum_i p_i = 1$ | `float[V]` |
| $\log_b$ | logarithm in base $b$: $b = 2$ gives bits, $b = e$ gives nats | |
| $H(p)$ | entropy, $-\sum_i p_i \log p_i$ | scalar $\ge 0$ |
| $H(p, q)$ | cross-entropy, $-\sum_i p_i \log q_i$ | scalar |
| $D_{KL}(p \,\Vert\, q)$ | KL divergence, $\sum_i p_i \log (p_i / q_i)$ | scalar $\ge 0$ |
| $H(X, Y)$ | joint entropy of a pair of random variables | scalar |
| $H(Y \mid X)$ | conditional entropy, $\sum_x P(x) H(Y \mid X = x)$ | scalar |
| $H_b(p)$ | binary entropy of a coin with $P(1) = p$ | scalar |
| $r$ | log-ratio $\log p_{\text{ref}}(x) - \log p(x)$ of one sample | scalar |

### 2.1 Entropy

The **surprise** of an outcome with probability $p_i$ is $-\log p_i$: certain outcomes surprise you not at all, rare ones a lot, and the surprises of independent outcomes add. **Entropy** is the expected surprise, $H(p) = -\sum_i p_i \log p_i$, with $0 \log 0 = 0$ (the limit of $t \log t$ as $t \to 0$). In bits it is the average number of yes/no questions an optimal strategy needs to identify the outcome: 1 for a fair coin, 8 for a uniform byte. It is 0 for a certain outcome and at most $\log V$ over $V$ outcomes, with the maximum exactly at the uniform distribution. Changing the base rescales it: $H_{\text{nats}} = H_{\text{bits}} \cdot \ln 2$.

For a coin, $H_b(p) = -p \log_2 p - (1-p)\log_2(1-p)$, symmetric about $p = 1/2$, where it peaks at 1 bit.

### 2.2 Joint and conditional entropy

For a pair $(X, Y)$ the **joint entropy** is the entropy of the joint distribution. The **conditional entropy** $H(Y \mid X)$ averages, over $x$, the entropy of $Y$'s distribution once $X = x$ is known. The **chain rule** $H(X, Y) = H(X) + H(Y \mid X)$ (q5) says that the surprise of a pair is the surprise of the first plus the surprise of the second given the first; for a sequence, $H(X_1, \dots, X_T) = \sum_t H(X_t \mid X_{<t})$. A language model is a product of next-token conditionals, and its training loss is this sum. Conditioning never increases entropy on average: $H(Y \mid X) \le H(Y)$.

### 2.3 Cross-entropy and KL divergence

If data come from $p$ but you encode or predict with $q$, the expected surprise is the **cross-entropy** $H(p, q) = -\sum_i p_i \log q_i$. A model's training loss on a corpus is an estimate of $H(p_{\text{data}}, q_{\text{model}})$. The excess over the best possible, $D_{KL}(p \,\Vert\, q) = H(p, q) - H(p) = \sum_i p_i \log \frac{p_i}{q_i}$, is the **KL divergence**. Gibbs' inequality (q9) says it is at least 0, with equality only when $q = p$, so minimizing cross-entropy in $q$ drives $q$ toward $p$. KL is not symmetric and is not a distance: $D_{KL}(p \,\Vert\, q)$ is infinite when $q$ gives zero probability to something $p$ can produce, which is why a model must never assign probability exactly 0.

### 2.4 Estimating KL from samples

In RL fine-tuning you see one sample $x \sim p$ at a time and want $D_{KL}(p \,\Vert\, p_{\text{ref}})$. With $r = \log p_{\text{ref}}(x) - \log p(x)$, the estimator $k_1 = -r$ is unbiased but often negative for single samples. **k3** $= e^r - r - 1$ is also unbiased, because $E_p[e^r] = \sum_x p(x) \frac{p_{\text{ref}}(x)}{p(x)} = 1$, so $E[k_3] = 1 + E[-r] - 1 = D_{KL}$; and since $e^r \ge 1 + r$ (the same tangent-line bound as Gibbs'), every single value is nonnegative.

## 3. Worked example by hand

This is a sibling of q4 and q6, not one of the graded problems.

**Joint and conditional entropy.** $X$ and $Y$ in $\{0, 1\}$ with $P(0,0) = 1/4$, $P(0,1) = 1/4$, $P(1,0) = 1/2$, $P(1,1) = 0$.

- Joint, in bits: $H(X, Y) = \tfrac14 \cdot 2 + \tfrac14 \cdot 2 + \tfrac12 \cdot 1 + 0 = 3/2$.
- Marginal: $P(X=0) = 1/2$, $P(X=1) = 1/2$, so $H(X) = 1$.
- Conditional, computed directly: given $X = 0$, $Y$ is $(1/2, 1/2)$ with entropy 1; given $X = 1$, $Y = 0$ surely, entropy 0. So $H(Y \mid X) = \tfrac12 \cdot 1 + \tfrac12 \cdot 0 = 1/2$.
- Chain rule check: $H(X) + H(Y \mid X) = 1 + 1/2 = 3/2 = H(X, Y)$.

**Cross-entropy and KL.** $p = (1, 0)$ and $q = (1/2, 1/2)$: $H(p) = 0$, $H(p, q) = -1 \cdot \log_2(1/2) = 1$ bit, so $D_{KL}(p \,\Vert\, q) = 1$ bit. The other direction, $D_{KL}(q \,\Vert\, p) = \tfrac12 \log_2 \frac{1/2}{1} + \tfrac12 \log_2 \frac{1/2}{0}$, is infinite: $p$ gives zero probability to an outcome $q$ produces. In `solve/` a bits answer such as $1 - \tfrac12\log_2 3$ is written `1 - log(3, 2)/2`.

## 4. The problem set

Write each answer in `solve/S-M11a.toml`; lettered parts are their own tables:

```toml
[q1.c]
answer = "3/2"
[q2]
answer = "log(V)"
[q6.b]
answer = "1 - log(3, 2)/2"
[q9]
proof = "S-M11a/q9.md"
```

Bits use `log(x, 2)`; nats use `log(x)`. Give exact values, not decimals.

<!-- ol:problems S-M11a -->

### Entropy and chain rules

**q1.** Give the entropy in bits of (a) a fair coin, (b) a uniform byte (256 equally likely values), (c) the distribution $p = (1/2, 1/4, 1/4)$. `[number]`

**q2.** Give the entropy in nats of the uniform distribution over $V$ outcomes. `[expr in V]`

**q3.** The binary entropy is $H_b(p) = -p \log_2 p - (1 - p) \log_2 (1 - p)$ bits. (a) Which $p$ maximizes it? (b) Give $H_b(1/4)$. `[number]`

**q4.** $X$ and $Y$ take values in $\{0, 1\}$ with joint distribution $P(0,0) = 1/2$, $P(0,1) = 1/4$, $P(1,0) = 0$, $P(1,1) = 1/4$. In bits, give (a) $H(X, Y)$, (b) $H(X)$, (c) $H(Y \mid X)$. `[number]`

**q5.** Prove the chain rule $H(X, Y) = H(X) + H(Y \mid X)$ for discrete random variables, where $H(Y \mid X) = \sum_x P(x) H(Y \mid X = x)$. `[proof]`

### Cross-entropy and KL divergence

**q6.** Let $p = (1/2, 1/2)$ and $q = (1/4, 3/4)$. In bits, give (a) the cross-entropy $H(p, q)$, (b) $D_{KL}(p \,\Vert\, q)$, (c) $D_{KL}(q \,\Vert\, p)$. `[number]` (d) Is $D_{KL}(p \,\Vert\, q) = D_{KL}(q \,\Vert\, p)$ for all distributions $p, q$? `[bool]`

**q7.** A byte-level model that assigns probability $1/256$ to every next byte is evaluated on any text. Give its cross-entropy loss in nats per byte. `[number]`

**q8.** The k3 estimator of KL used in RL fine-tuning is $k_3(r) = e^r - r - 1$, with $r = \log p_{\text{ref}}(x) - \log p(x)$ for a sample $x \sim p$. (a) Give $k_3(\log 2)$. `[number]` (b) Is $k_3(r) \ge 0$ for every real $r$? `[bool]`

**q9.** Prove Gibbs' inequality: $D_{KL}(p \,\Vert\, q) \ge 0$ for distributions $p, q$ on a finite set with $q_i > 0$ wherever $p_i > 0$, with equality exactly when $p = q$. `[proof]`

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Reporting nats where bits were asked, or the reverse | bits-per-byte off by a factor $\ln 2$ | q1 (canary log(2)), q2 (canary in bits), q7 (canary 8) |
| Mixing a natural log into a bits formula | a value that is neither unit | q3 (canary 2 - 3*log(3)/4) |
| Counting a zero-probability cell | entropy too high, or NaN from $0 \log 0$ in code | q4 (canary 2) |
| Taking $H(Y)$ or $H(X, Y)$ for $H(Y \mid X)$ | sequence losses that do not add up | q4 (canaries) |
| Swapping the arguments of KL | a penalty that weights the wrong distribution's errors | q6 (canaries: the other direction) |
| Reporting the cross-entropy as the KL | "the model is 1.2 bits from the data" when it is 0.2 | q6 (canary: the cross-entropy) |
| Using the raw log-ratio as a per-sample KL | negative KL values in training logs | q8 (canary log(2)) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M07a` | distributions, expectation, and conditioning |
| Forward | `M11.1` | `entropy`, `cross_entropy`, `kl`, `js`, `kl_k3` are this set as code, with the Gibbs property test |
| Forward | `M08.3` | the cross-entropy VJP, softmax minus one-hot |
| Forward | `L0.3` | fused cross-entropy, the training loss in nats |
| Forward | `L8.1` | the sampler logs the entropy of each next-token distribution |
| Forward | `M11.2` | perplexity $e^{H}$ and bits per byte from the same NLL sums |
| Forward | `L12.3` | the KL penalty to a reference policy, estimated with k3 |
