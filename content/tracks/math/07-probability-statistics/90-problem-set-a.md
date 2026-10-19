# Probability problem set, part a: axioms, Bayes, random variables, Box-Muller

## Overview

| | |
|---|---|
| **Module** | `S-M07a` · solve · none · Pass 2 · 3 to 4 h |
| **Solution** | a worked answer after every problem in section 4: 29 exact answers (counting lettered parts) and 1 proof (q6; the model proof is also in [`course/solutions/solve/S-M07a/`](../../../../course/solutions/solve/S-M07a/)) |
| **Contract** | none: a pen and paper set |
| **Needs** | `S-M05` (counting, sets). Reading: the [Probability and Statistics topic](README.md), probability and random variable sections, and the Gaussian integral of `S-M02` |
| **Used by** | no call site (a solve set). Do it before `M07.0` (random variables, the normal, and Box-Muller over PCG32) and `M07.3` (variance propagation for initialization); later parts `S-M07b` to `S-M07d` follow `M07.1`, `M07.4`, and `M07.5` |
| **Optional depth** | Blitzstein and Hwang, *Introduction to Probability* (free), ch. 1 to 5; Box and Muller, "A Note on the Generation of Random Normal Deviates" (1958) |

## Key Takeaways

- Probability is a function on events with three axioms; inclusion-exclusion and conditioning follow from them (q1, q6).
- Bayes' rule turns a classifier's per-class rates into the probability a flag is right, and with a 1% base rate a 90%-sensitive classifier is right only 2 times in 13 (q3).
- Expectation is linear and variance is $E[X^2] - E[X]^2$; variances of independent sums add, and a scale $a$ multiplies variance by $a^2$ (q7, q8, q14).
- An inverse CDF turns one uniform into any distribution, and its comparison is strict: $u < c$, so a $u$ landing exactly on a boundary goes to the next id (q10, q13).
- Box-Muller turns two uniforms into two independent standard normals, cosine first, sine as the spare (q15).

## How to read this chapter

Sections 1 to 3 review the ideas and work two examples by hand. Section 4 is the problem set: every problem is followed by a collapsed worked answer with the final result, the derivation, and the common wrong answers and why they are wrong. Try each problem before opening its answer.

---

## 1. Why now

Every random number in the system comes from one generator, PCG32 (`M06.3`), and from Pass 2 on it feeds weight initialization, dropout, data shuffling, and sampling. `M07.0` turns its uniforms into normals with Box-Muller, and `M07.3` chooses initialization scales so that activations neither explode nor vanish through 20 layers, which is an argument about variances. The sampler of `L8.1` walks a cumulative distribution with a strict comparison that must agree bit for bit with the Rust port. The gateway's usage-policy classifier (`gw.08`) reports a precision that can only be interpreted with Bayes' rule. This set builds the vocabulary for all of it: the axioms, conditional probability, Bayes, discrete and continuous random variables, their moments, and the normal distribution.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\Omega$ | sample space: the set of possible outcomes | set |
| $A, B \subseteq \Omega$ | events | sets |
| $P(A)$ | probability of $A$ | real in $[0, 1]$ |
| $P(A \mid B)$ | conditional probability, $P(A \cap B)/P(B)$ for $P(B) > 0$ | real |
| $X, Y$ | random variables: functions from $\Omega$ to $\mathbb{R}$ | |
| $p_X(k)$ | probability mass function, $P(X = k)$ | real |
| $f_X(x), F_X(x)$ | density and cumulative distribution function, $F_X(x) = P(X \le x)$ | real |
| $E[X]$ | expectation, $\sum_k k\, p_X(k)$ or $\int x f_X(x)\, dx$ | real |
| $\operatorname{Var}(X)$ | variance, $E[(X - E[X])^2]$ | real |
| $U$ | a uniform random variable on $[0, 1)$ | real |
| $Z$ | a standard normal, density $\varphi(z) = e^{-z^2/2}/\sqrt{2\pi}$ | real |
| $\lambda$ (`lam`) | rate of a Poisson or exponential distribution | positive real |

### 2.1 The axioms and what follows

A **probability** assigns each event $A \subseteq \Omega$ a number $P(A)$ such that $P(A) \ge 0$, $P(\Omega) = 1$, and the probability of a countable union of pairwise disjoint events is the sum of their probabilities. Everything else is derived: $P(\emptyset) = 0$, $P(A^c) = 1 - P(A)$, and inclusion-exclusion $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ (q6). On a finite $\Omega$ of equally likely outcomes, $P(A) = \lvert A \rvert / \lvert \Omega \rvert$, which makes probability a counting problem (`S-M05`).

### 2.2 Conditioning, independence, Bayes

**Conditional probability** $P(A \mid B) = P(A \cap B)/P(B)$ restricts the sample space to $B$ and renormalizes. Events are **independent** when $P(A \cap B) = P(A)P(B)$, equivalently $P(A \mid B) = P(A)$. The **law of total probability** splits on a partition $B_1, \dots, B_k$ of $\Omega$: $P(A) = \sum_i P(A \mid B_i) P(B_i)$. **Bayes' rule** inverts a conditional:
$$P(B \mid A) = \frac{P(A \mid B)\,P(B)}{P(A)}.$$
For a classifier, $P(\text{flag} \mid \text{bad})$ is its **sensitivity** (recall), $P(\text{flag} \mid \text{ok})$ its false positive rate, and $P(\text{bad} \mid \text{flag})$ its **precision**, which depends on the base rate $P(\text{bad})$ as much as on the classifier.

### 2.3 Discrete random variables

A **random variable** $X$ is a numeric function of the outcome. Its **distribution** is described by the mass function $p_X(k) = P(X = k)$. **Expectation** $E[X] = \sum_k k\, p_X(k)$ is linear, $E[aX + bY] = aE[X] + bE[Y]$, for any $X$ and $Y$, independent or not. **Variance** $\operatorname{Var}(X) = E[(X - \mu)^2] = E[X^2] - \mu^2$ with $\mu = E[X]$; it satisfies $\operatorname{Var}(aX + b) = a^2 \operatorname{Var}(X)$, and for independent $X, Y$, $\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)$.

| Distribution | Values | $P(X = k)$ | $E[X]$ |
|---|---|---|---|
| Bernoulli($p$) | $\{0, 1\}$ | $p$ for 1, $1 - p$ for 0 | $p$ |
| Binomial($n, p$) | $\{0, \dots, n\}$ | $\binom{n}{k} p^k (1-p)^{n-k}$ | $np$ (a sum of $n$ Bernoullis) |
| Geometric($p$) | $\{1, 2, \dots\}$ | $(1-p)^{k-1} p$ | derive it in q9 |
| Poisson($\lambda$) | $\{0, 1, \dots\}$ | $e^{-\lambda} \lambda^k / k!$ | $\lambda$ |
| Categorical($q$) | $\{0, \dots, V-1\}$ | $q_k$ | $\sum_k k\, q_k$ |

A language model's next-token distribution is a categorical over the vocabulary. To sample it from one uniform $u$, walk the ids in order with a running total $c$ and return the first id with $u < c$ (`spec/sampling.md` step 11). The id returned is $k$ exactly when $q_0 + \dots + q_{k-1} \le u < q_0 + \dots + q_k$, an interval of length $q_k$, so each id comes up with its probability.

### 2.4 Continuous random variables

A **continuous** $X$ has a density $f_X \ge 0$ with $\int f_X = 1$, and $P(a \le X \le b) = \int_a^b f_X(x)\,dx$; its **CDF** is $F_X(x) = \int_{-\infty}^x f_X$. Expectation becomes an integral, $E[g(X)] = \int g(x) f_X(x)\,dx$, with the same linearity and variance rules. The **uniform** on $[0, 1]$ has $f = 1$ there. The **exponential** with rate $\lambda$ has $F(x) = 1 - e^{-\lambda x}$ for $x \ge 0$. **Inverse-CDF sampling** sets $X = F^{-1}(U)$: then $P(X \le x) = P(U \le F(x)) = F(x)$, so one uniform becomes a draw from any distribution whose CDF can be inverted.

The **standard normal** $Z$ has density $\varphi(z) = e^{-z^2/2}/\sqrt{2\pi}$, mean 0, and variance 1; the constant comes from the Gaussian integral (q16). $X = \mu + \sigma Z$ is normal with mean $\mu$ and variance $\sigma^2$. Its CDF has no closed form, so inverse-CDF sampling is awkward; **Box-Muller** sidesteps it. For two independent standard normals, the squared radius $R^2 = Z_0^2 + Z_1^2$ is exponential with rate $1/2$ and the angle is uniform on $[0, 2\pi)$, independent of it. Reversing that: draw $u_1, u_2$, set $r = \sqrt{-2 \ln(1 - u_1)}$ (inverse-CDF sampling of the exponential, with $1 - u_1 > 0$ so the log is finite) and $\theta = 2\pi u_2$, and return $z_0 = r\cos\theta$ and $z_1 = r \sin\theta$. `spec/pcg32.md` fixes the order: $z_0$ is returned, $z_1$ is kept as the spare for the next call.

## 3. Worked example by hand

These are siblings of q3 and q15, not problems of the set.

**Bayes.** A spam filter catches 80% of spam and wrongly flags 2% of good mail, and 10% of mail is spam. What fraction of flagged mail is spam?

Take 1000 messages: 100 spam, 900 good. Flagged spam: $0.8 \cdot 100 = 80$. Flagged good: $0.02 \cdot 900 = 18$. Flagged in total: $98$. So $P(\text{spam} \mid \text{flag}) = 80/98 = 40/49 \approx 0.816$. The same through the formula: $P(\text{flag}) = 0.8 \cdot 0.1 + 0.02 \cdot 0.9 = 0.098$, and $P(\text{spam} \mid \text{flag}) = 0.08 / 0.098 = 40/49$.

**Box-Muller.** Take $1 - u_1 = e^{-1/2}$ and $u_2 = 1/4$. Then $r = \sqrt{-2 \cdot (-1/2)} = 1$ and $\theta = 2\pi/4 = \pi/2$, so $z_0 = \cos(\pi/2) = 0$ and the spare is $z_1 = \sin(\pi/2) = 1$. The bookkeeping `M07.0` relies on: two uniforms in, two normals out, $z_0$ first.

## 4. The problem set

Write each answer in `solve/S-M07a.toml`; lettered parts are their own tables:

```toml
[q1.a]
answer = "7/10"
[q8.a]
answer = "p*(1 - p)"
[q13.a]
answer = "-log(1 - u)/lam"
[q6]
proof = "S-M07a/q6.md"
```

Probabilities are exact fractions; `0.7` fails a question marked exact. Write $\lambda$ as `lam`, $e$ as `E` or `exp(1)`, and $\pi$ as `pi`.

<!-- ol:problems S-M07a -->

### Axioms, conditional probability, and Bayes

**q1.** Events $A$ and $B$ have $P(A) = 1/2$, $P(B) = 2/5$, and $P(A \cap B) = 1/5$.
(a) $P(A \cup B)$. (b) $P(A \mid B)$. `[number]` (c) Are $A$ and $B$ independent? `[bool]`

**q2.** Roll two fair six-sided dice. (a) $P(\text{the sum is } 7)$. (b) $P(\text{at least one die shows } 6)$. (c) $P(\text{the sum is } 7 \mid \text{at least one die shows } 6)$. `[number]`

**q3.** Your gateway's usage-policy classifier (`gw.08`) flags 90% of violating requests and 5% of allowed ones, and 1% of requests violate the policy. (a) What fraction of all requests is flagged? (b) Given that a request is flagged, what is the probability that it violates the policy? `[number]`

**q4.** Draw 3 bytes independently and uniformly from the 256 byte values. What is the probability that all three are different? `[number]`

**q5.** A bag holds a fair coin and a coin that lands heads with probability $3/4$. Pick one of the two uniformly at random and flip it. (a) $P(\text{heads})$. (b) $P(\text{the biased coin was picked} \mid \text{heads})$. `[number]`

**q6.** From the three axioms ($P(E) \ge 0$; $P(\Omega) = 1$; $P$ of a countable union of pairwise disjoint events is the sum of their probabilities) prove $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ for any events $A$ and $B$. `[proof]`

### Discrete random variables

**q7.** $X$ is a fair die roll, uniform on $\{1, \dots, 6\}$. (a) $E[X]$. (b) $E[X^2]$. (c) $\operatorname{Var}(X)$. `[number]`

**q8.** (a) Give $\operatorname{Var}(X)$ for $X \sim \mathrm{Bernoulli}(p)$. `[expr in p]` (b) Give $\operatorname{Var}(Y)$ for $Y \sim \mathrm{Binomial}(n, p)$, the number of successes in $n$ independent Bernoulli($p$) trials. `[expr in n, p]`

**q9.** $X$ counts independent Bernoulli($p$) trials up to and including the first success, so $X \in \{1, 2, 3, \dots\}$. Give $E[X]$. `[expr in p]`

**q10.** A sampler draws a token id from the categorical distribution $q = (1/10, 2/10, 3/10, 4/10)$ over ids $0, 1, 2, 3$. (a) Give $E[\text{id}]$. The sampler then uses the inverse CDF of `spec/sampling.md`: walk the ids in ascending order adding $q_i$ to a running total $c$, and return the first id with $u < c$. Which id does it return for (b) $u = 0.35$ and (c) $u = 0.3$ (exactly)? `[number]`

**q11.** $X \sim \mathrm{Poisson}(\lambda)$ has $P(X = k) = e^{-\lambda} \lambda^k / k!$. Give $P(X = 0)$. `[expr in lam]`

### Continuous random variables, the normal, and Box-Muller

**q12.** $U$ is uniform on $[0, 1]$. (a) $E[U]$. (b) $\operatorname{Var}(U)$. `[number]`

**q13.** $X \sim \mathrm{Exponential}(\lambda)$ has CDF $F(x) = 1 - e^{-\lambda x}$ for $x \ge 0$.
(a) Give the inverse-CDF sampler: the $x$ with $F(x) = u$. `[expr in u, lam]` (b) Give $E[X]$. `[expr in lam]`

**q14.** $Z$ is a standard normal and $a, b$ are constants. Give $\operatorname{Var}(aZ + b)$. `[expr in a, b]`

**q15.** `spec/pcg32.md` turns two uniforms into two normals by Box-Muller: $r = \sqrt{-2 \ln(1 - u_1)}$, then $z_0 = r \cos(2\pi u_2)$ is returned and $z_1 = r \sin(2\pi u_2)$ is kept as the spare. For $u_1$ with $1 - u_1 = e^{-2}$ and $u_2 = 1/12$, give (a) $z_0$ and (b) $z_1$. `[number]`

**q16.** Give $\int_{-\infty}^{\infty} e^{-x^2/2}\, dx$, the constant that normalizes the standard normal density. `[number]`

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Adding probabilities of overlapping events | $P(A \cup B) > 1$ for large events | q1 (canary 9/10), q2 (canary 1/3) |
| Confusing $P(A \mid B)$ with $P(B \mid A)$ | a classifier's precision quoted as its recall | q3 (canary 9/10) |
| Treating unequal outcomes as equally likely | the 11 dice sums given probability 1/11 each | q2 (canary 1/11) |
| Variance as $E[X^2]$, or with $n - 1$ in a population formula | initialization scales off by the squared mean | q7 (canaries 49/4 and 7/2), q12 (canary 1/3) |
| A shift changing variance | `normal_init` with a nonzero mean breaks M07.3's variance argument | q14 (canary a^2 + b) |
| $u \le c$ instead of $u < c$ in the inverse CDF | a Python and a Rust sampler disagree at boundaries | q10 (canary 1) |
| Swapping sine and cosine, or forgetting the radius | normals differ from `spec/pcg32.md` | q15 (canaries 1 and sqrt(3)/2) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M05` | counting finite sample spaces (q2, q4) |
| Forward | `M07.0` | `normal`, `expectation`, `variance` over PCG32, checked by sample moments and a chi-square test |
| Forward | `M07.3` | variance propagation picks `xavier_*` and `kaiming_normal` scales |
| Forward | `M07.1` | inverse CDF, Gumbel-max, and alias sampling of categoricals (`S-M07b` follows it) |
| Forward | `L8.1` | the sampler's inverse-CDF step with the strict comparison of q10 |
| Forward | `gw.08` | the policy head's precision and recall, read with q3 |
