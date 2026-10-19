# Information theory problem set, part a: entropy, chain rules, KL and cross-entropy

## Overview

| | |
|---|---|
| **Module** | `S-M11a` · solve · none · Pass 2 · 3 to 4 h |
| **Solution** | worked answers to all 9 problems (16 exact parts and 2 proofs), each under its problem in section 4 |
| **Contract** | none: a pen and paper set |
| **Needs** | `S-M07a` (distributions, expectation, Bayes). Reading: the [Information Theory topic](README.md), entropy and KL sections |
| **Used by** | no call site (a problem set). It fixes the definitions behind `M11.1` (`entropy`, `cross_entropy`, `kl`, `kl_from_logprobs`, `js`, `kl_k3`), which `M08.3`, `L0.3`, `L8.1`, and `L12.3` call; part b, `S-M11b` in Pass 3, covers coding, mutual information, and maximum entropy |
| **Optional depth** | Cover and Thomas, *Elements of Information Theory*, ch. 2; MacKay, *Information Theory, Inference, and Learning Algorithms* (free), ch. 2 and 4; Schulman, "Approximating KL Divergence" (2020 blog note) for k3 |

## Key Takeaways

- Entropy is the expected surprise $-\log p$; it is $\log V$ for a uniform distribution over $V$ outcomes and less for anything else, so an untrained byte model starts at $\log 256 \approx 5.55$ nats (q1, q2, q7).
- The unit is the base of the logarithm: bits for $\log_2$, nats for $\ln$; mixing them is the most common wrong answer (q1, q3, q7).
- The chain rule $H(X, Y) = H(X) + H(Y \mid X)$ is why next-token cross-entropy summed over a sequence is the sequence's log-loss (q4, q5).
- Cross-entropy is entropy plus KL divergence, KL is never negative, and it is not symmetric (q6, q9).
- The k3 estimator $e^r - r - 1$ is nonnegative for every sample, unlike the plain log-ratio (q8).

## How to read this chapter

Sections 1 and 2 explain why these definitions matter and derive them from the idea of surprise; section 3 works two sibling problems by hand. Section 4 states every problem with its worked answer folded underneath: try each one on paper first, then open the answer and compare the derivation, not just the number. Each answer names the common wrong answer, most often a unit mix-up or a swapped KL, and why it is wrong.

---

## 1. Why now

The Pass 1 bigram reported a loss of about 3.2 nats per byte, and the number meant nothing yet. In Pass 2 it starts to carry weight: `L0.3` trains with cross-entropy, `M11.1` writes `entropy`, `cross_entropy`, and `kl` that the sampler (`L8.1`) and the evaluation suite (`L6.7`) log, and the gap between the model's loss and the data's entropy is the KL divergence the optimizer is shrinking. Later, `L12.3` keeps a fine-tuned policy close to its reference with a KL penalty estimated by k3. This set fixes the definitions, the units, and the identities (chain rule, cross-entropy equals entropy plus KL, KL is nonnegative) before the code implements them.

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

This is a sibling of q4 and q6, not one of the problems below.

**Joint and conditional entropy.** $X$ and $Y$ in $\{0, 1\}$ with $P(0,0) = 1/4$, $P(0,1) = 1/4$, $P(1,0) = 1/2$, $P(1,1) = 0$.

- Joint, in bits: $H(X, Y) = \tfrac14 \cdot 2 + \tfrac14 \cdot 2 + \tfrac12 \cdot 1 + 0 = 3/2$.
- Marginal: $P(X=0) = 1/2$, $P(X=1) = 1/2$, so $H(X) = 1$.
- Conditional, computed directly: given $X = 0$, $Y$ is $(1/2, 1/2)$ with entropy 1; given $X = 1$, $Y = 0$ surely, entropy 0. So $H(Y \mid X) = \tfrac12 \cdot 1 + \tfrac12 \cdot 0 = 1/2$.
- Chain rule check: $H(X) + H(Y \mid X) = 1 + 1/2 = 3/2 = H(X, Y)$.

**Cross-entropy and KL.** $p = (1, 0)$ and $q = (1/2, 1/2)$: $H(p) = 0$, $H(p, q) = -1 \cdot \log_2(1/2) = 1$ bit, so $D_{KL}(p \,\Vert\, q) = 1$ bit. The other direction, $D_{KL}(q \,\Vert\, p) = \tfrac12 \log_2 \frac{1/2}{1} + \tfrac12 \log_2 \frac{1/2}{0}$, is infinite: $p$ gives zero probability to an outcome $q$ produces.

## 4. The problem set

Each problem carries a tag for the kind of answer it wants (`[number]`, `[expr in ...]`, `[bool]`, `[proof]`). Answers are exact; $\log_2$ means bits and $\ln$ means nats. Open the answer only after you have one of your own.

### Entropy and chain rules

**q1.** Give the entropy in bits of (a) a fair coin, (b) a uniform byte (256 equally likely values), (c) the distribution $p = (1/2, 1/4, 1/4)$. `[number]`

<details><summary>Answer</summary>

**(a) 1, (b) 8, (c) $3/2$ bits.**

1. (a) $-\tfrac12\log_2\tfrac12 - \tfrac12\log_2\tfrac12 = \tfrac12 + \tfrac12 = 1$. A common mistake is $\ln 2$: that is the same entropy in nats, not bits.
2. (b) Uniform over 256: $H = \log_2 256 = 8$. A common mistake is 256, the number of outcomes rather than its log.
3. (c) $\tfrac12 \cdot 1 + \tfrac14 \cdot 2 + \tfrac14 \cdot 2 = 3/2$. A common mistake is $\log_2 3 \approx 1.585$, treating the three outcomes as equally likely; the uneven distribution is less uncertain than the uniform one, as the bound $H \le \log V$ says.

</details>

**q2.** Give the entropy in nats of the uniform distribution over $V$ outcomes. `[expr in V]`

<details><summary>Answer</summary>

**$\ln V$.**

Every $p_i = 1/V$, so $H = -\sum_i \frac1V \ln\frac1V = \ln V$. This is the loss of an untrained model with a uniform output, the number `L0.5` starts near. A common mistake is $\log_2 V$ ($\ln V / \ln 2$), which is bits; another is $1/V$, the probability itself.

</details>

**q3.** The binary entropy is $H_b(p) = -p \log_2 p - (1 - p) \log_2 (1 - p)$ bits. (a) Which $p$ maximizes it? (b) Give $H_b(1/4)$. `[number]`

<details><summary>Answer</summary>

**(a) $p = 1/2$, (b) $H_b(1/4) = 2 - \tfrac34\log_2 3 \approx 0.811$ bits.**

1. (a) $H_b'(p) = \log_2\frac{1-p}{p}$, which is zero only at $p = 1/2$, and $H_b$ is concave, so that is the maximum (1 bit). A common mistake is 1, the maximum value instead of where it occurs.
2. (b) $-\tfrac14\log_2\tfrac14 - \tfrac34\log_2\tfrac34 = \tfrac14 \cdot 2 + \tfrac34(2 - \log_2 3) = 2 - \tfrac34\log_2 3$. A common mistake is $1/2$, keeping only the first term; another is $2 - \tfrac34\ln 3$, a natural log inside a bits answer, which is neither unit.

</details>

**q4.** $X$ and $Y$ take values in $\{0, 1\}$ with joint distribution $P(0,0) = 1/2$, $P(0,1) = 1/4$, $P(1,0) = 0$, $P(1,1) = 1/4$. In bits, give (a) $H(X, Y)$, (b) $H(X)$, (c) $H(Y \mid X)$. `[number]`

<details><summary>Answer</summary>

**(a) $3/2$, (b) $2 - \tfrac34\log_2 3 \approx 0.811$, (c) $\tfrac34\log_2 3 - \tfrac12 \approx 0.689$ bits.**

1. (a) The nonzero cells are $1/2, 1/4, 1/4$: $H(X, Y) = \tfrac12 \cdot 1 + \tfrac14 \cdot 2 + \tfrac14 \cdot 2 = 3/2$. The cell $P(1,0) = 0$ contributes $0 \log 0 = 0$. A common mistake is 2, treating all four cells as $1/4$.
2. (b) $P(X = 0) = 3/4$, $P(X = 1) = 1/4$, so $H(X) = H_b(1/4)$, the q3(b) value. A common mistake is 1, assuming $X$ is uniform.
3. (c) By the chain rule, $H(Y \mid X) = H(X, Y) - H(X) = \tfrac32 - 2 + \tfrac34\log_2 3$. Directly: given $X = 0$, $Y$ is $(2/3, 1/3)$ with entropy $\log_2 3 - 2/3$; given $X = 1$, $Y = 1$ surely, entropy 0; so $H(Y \mid X) = \tfrac34(\log_2 3 - \tfrac23) = \tfrac34\log_2 3 - \tfrac12$. The two routes agree. A common mistake is to report $H(X, Y)$ or $H(X)$ here.

</details>

**q5.** Prove the chain rule $H(X, Y) = H(X) + H(Y \mid X)$ for discrete random variables, where $H(Y \mid X) = \sum_x P(x) H(Y \mid X = x)$. `[proof]`

<details><summary>Answer</summary>

**Proof.** For every pair with $P(x, y) > 0$, $P(x, y) = P(x)\,P(y \mid x)$, so

$$-\log P(x, y) = -\log P(x) - \log P(y \mid x).$$

Multiply by $P(x, y)$ and sum over all pairs with $P(x, y) > 0$ (pairs with $P(x, y) = 0$ contribute $0 \log 0 = 0$ to every sum below, so dropping them changes nothing):

$$H(X, Y) = -\sum_{x, y} P(x, y)\log P(x) \;-\; \sum_{x, y} P(x, y)\log P(y \mid x).$$

In the first sum, $\log P(x)$ does not depend on $y$, and $\sum_y P(x, y) = P(x)$, so it is $-\sum_x P(x)\log P(x) = H(X)$. In the second, write $P(x, y) = P(x)P(y \mid x)$:

$$-\sum_x P(x) \sum_y P(y \mid x)\log P(y \mid x) = \sum_x P(x)\, H(Y \mid X = x) = H(Y \mid X).$$

So $H(X, Y) = H(X) + H(Y \mid X)$. $\blacksquare$

The structure is "take logs of the product rule, then average": the same step, applied repeatedly to $P(x_1, \dots, x_T) = \prod_t P(x_t \mid x_{<t})$, gives the sequence chain rule that makes a language model's summed next-token loss equal the sequence's log-loss.

</details>

### Cross-entropy and KL divergence

**q6.** Let $p = (1/2, 1/2)$ and $q = (1/4, 3/4)$. In bits, give (a) the cross-entropy $H(p, q)$, (b) $D_{KL}(p \,\Vert\, q)$, (c) $D_{KL}(q \,\Vert\, p)$. `[number]` (d) Is $D_{KL}(p \,\Vert\, q) = D_{KL}(q \,\Vert\, p)$ for all distributions $p, q$? `[bool]`

<details><summary>Answer</summary>

**(a) $2 - \tfrac12\log_2 3 \approx 1.208$, (b) $1 - \tfrac12\log_2 3 \approx 0.208$, (c) $\tfrac34\log_2 3 - 1 \approx 0.189$ bits, (d) false.**

1. (a) $H(p, q) = -\tfrac12\log_2\tfrac14 - \tfrac12\log_2\tfrac34 = \tfrac12 \cdot 2 + \tfrac12(2 - \log_2 3) = 2 - \tfrac12\log_2 3$. A common mistake is 1, which is $H(p)$.
2. (b) $D_{KL}(p \,\Vert\, q) = H(p, q) - H(p) = 2 - \tfrac12\log_2 3 - 1$. A common mistake is to report the cross-entropy itself: "the model is 1.2 bits from the data" when the excess is 0.2.
3. (c) $D_{KL}(q \,\Vert\, p) = \tfrac14\log_2\frac{1/4}{1/2} + \tfrac34\log_2\frac{3/4}{1/2} = -\tfrac14 + \tfrac34(\log_2 3 - 1) = \tfrac34\log_2 3 - 1$. A common mistake is to give the (b) value, swapping the arguments.
4. (d) (b) and (c) already differ (0.208 vs 0.189), so KL is not symmetric in general. It weights the log-ratio by its first argument.

</details>

**q7.** A byte-level model that assigns probability $1/256$ to every next byte is evaluated on any text. Give its cross-entropy loss in nats per byte. `[number]`

<details><summary>Answer</summary>

**$8\ln 2 = \ln 256 \approx 5.545$ nats per byte.**

The loss on each byte is $-\ln(1/256) = \ln 256$, whatever the byte is, so the cross-entropy is $\ln 256$ for any text. A common mistake is 8, which is the same number in bits; another is $1/256$, the probability rather than its surprise.

</details>

**q8.** The k3 estimator of KL used in RL fine-tuning is $k_3(r) = e^r - r - 1$, with $r = \log p_{\text{ref}}(x) - \log p(x)$ for a sample $x \sim p$. (a) Give $k_3(\log 2)$. `[number]` (b) Is $k_3(r) \ge 0$ for every real $r$? `[bool]`

<details><summary>Answer</summary>

**(a) $1 - \ln 2 \approx 0.307$, (b) true.**

1. (a) $k_3(\ln 2) = e^{\ln 2} - \ln 2 - 1 = 2 - \ln 2 - 1 = 1 - \ln 2$. A common mistake is $1 + \ln 2$, a sign slip on $r$; another is $\ln 2$, which is the plain log-ratio estimator, not k3.
2. (b) $e^r \ge 1 + r$ for every real $r$, because the exponential is convex and $1 + r$ is its tangent line at 0. So $e^r - r - 1 \ge 0$, with equality only at $r = 0$. The plain estimator $-r$ is negative whenever $p_{\text{ref}}(x) > p(x)$, which is why k3 is what `L12.3` logs.

</details>

**q9.** Prove Gibbs' inequality: $D_{KL}(p \,\Vert\, q) \ge 0$ for distributions $p, q$ on a finite set with $q_i > 0$ wherever $p_i > 0$, with equality exactly when $p = q$. `[proof]`

<details><summary>Answer</summary>

**Proof.** Let $S = \{i : p_i > 0\}$. Terms outside $S$ are 0, and by assumption $q_i > 0$ on $S$, so

$$-D_{KL}(p \,\Vert\, q) = \sum_{i \in S} p_i \log\frac{q_i}{p_i}.$$

Use $\log t \le t - 1$ for $t > 0$ (the concave $\log$ lies below its tangent at $t = 1$), with equality only at $t = 1$, and $t = q_i/p_i$:

$$-D_{KL}(p \,\Vert\, q) \le \sum_{i \in S} p_i\left(\frac{q_i}{p_i} - 1\right) = \sum_{i \in S} q_i - \sum_{i \in S} p_i = \sum_{i \in S} q_i - 1 \le 0,$$

since $q$'s mass on $S$ is at most 1. So $D_{KL}(p \,\Vert\, q) \ge 0$.

Equality needs both inequalities to be tight. The first is tight only if $q_i/p_i = 1$ for every $i \in S$; the second only if $\sum_{i \in S} q_i = 1$, so $q$ has no mass outside $S$, where $p$ is also 0. Together, $q = p$. Conversely $p = q$ makes every log-ratio 0. $\blacksquare$

(In natural log; in base 2 the same argument goes through, since $\log_2 t = \ln t / \ln 2$ only rescales.) The second equality condition is easy to forget: matching $p$ on its support is not enough if $q$ also puts mass elsewhere, because then $q_i < p_i$ somewhere on $S$.

</details>

## 5. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Reporting nats where bits were asked, or the reverse | bits-per-byte off by a factor $\ln 2$ | fixes the base before computing: $\log_2$ for bits, $\ln$ for nats; see q1, q2, q7 |
| Mixing a natural log into a bits formula | a value that is neither unit | every log in a bits answer is $\log_2$; see q3(b) |
| Counting a zero-probability cell | entropy too high, or NaN from $0 \log 0$ in code | drops zero cells by $0 \log 0 = 0$; see q4(a) |
| Taking $H(Y)$ or $H(X, Y)$ for $H(Y \mid X)$ | sequence losses that do not add up | computes $H(Y \mid X)$ both directly and by the chain rule; see q4(c) |
| Swapping the arguments of KL | a penalty that weights the wrong distribution's errors | weights the log-ratio by the first argument; see q6(b) and (c) |
| Reporting the cross-entropy as the KL | "the model is 1.2 bits from the data" when it is 0.2 | subtracts $H(p)$ from $H(p, q)$; see q6(b) |
| Using the raw log-ratio as a per-sample KL | negative KL values in training logs | uses $k_3 = e^r - r - 1$, nonnegative by the tangent-line bound; see q8 |

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
