# Probability problem set, part a: axioms, Bayes, random variables, Box-Muller

Every random number in an LLM system comes from one generator, PCG32, and it feeds weight initialization, dropout, data shuffling, and sampling. [Random variables, expectation, variance, and normal draws by Box-Muller](00-random-variables-and-the-normal.md) turns its uniforms into normals, and [Variance propagation and initialization](03-variance-propagation-and-initialization.md) chooses initialization scales so that activations neither explode nor vanish through 20 layers, which is an argument about variances. The token sampler of [Sampling and logit processors](../../build-an-llm/p08-inference/01-sampling-and-logit-processors.md) in *Build Your Own LLM System* walks a cumulative distribution with a strict comparison that must agree bit for bit with a Rust port, and a gateway's usage-policy classifier ([Usage policy enforcement](../../ai-platform-engineering/12-gateway/08-usage-policy-enforcement.md) in *AI Platform Engineering*) reports a precision that can only be interpreted with Bayes' rule.

This problem set builds the vocabulary for all of that: the axioms, conditional probability, Bayes, discrete and continuous random variables, their moments, and the normal distribution. It is a pen-and-paper set with 29 exact answers (counting lettered parts) and one proof. The section first reviews the ideas and works two examples by hand; then come the problems, each followed by a collapsed worked answer with the final result, the derivation, and the common wrong answers and why they are wrong. Try each problem before opening its answer. Work it before reading the random-variables and initialization sections of this chapter; the later parts of the set, [B](91-problem-set-b.md), [C](92-problem-set-c.md), and [D](93-problem-set-d.md), follow the categorical-sampling, limit-theorem, and hypothesis-test sections.

You need counting and sets from the [Discrete math 1 problem set](../05-discrete-math-1/90-problem-set.md), the probability and random-variable material in the [chapter opening](README.md), and the Gaussian integral from the [Calculus 2 problem set](../02-calculus-2/90-problem-set.md).

> **Learning objectives**
> - Derive inclusion-exclusion and conditional probability from the three axioms.
> - Use Bayes' rule and the law of total probability to turn a classifier's per-class rates into its precision.
> - Compute expectations and variances of discrete and continuous distributions, using linearity and the scaling rules.
> - Sample any invertible distribution from one uniform, with the strict inverse-CDF comparison.
> - Turn two uniforms into two standard normals by Box-Muller, in the order the generator spec fixes.

## Notation

The set uses these symbols.

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

## The axioms and what follows

A **probability** assigns each event $A \subseteq \Omega$ a number $P(A)$ such that $P(A) \ge 0$, $P(\Omega) = 1$, and the probability of a countable union of pairwise disjoint events is the sum of their probabilities. Everything else is derived: $P(\emptyset) = 0$, $P(A^c) = 1 - P(A)$, and inclusion-exclusion $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ (Problem 6). On a finite $\Omega$ of equally likely outcomes, $P(A) = \lvert A \rvert / \lvert \Omega \rvert$, which makes probability a counting problem.

## Conditioning, independence, Bayes

**Conditional probability** $P(A \mid B) = P(A \cap B)/P(B)$ restricts the sample space to $B$ and renormalizes. Events are **independent** when $P(A \cap B) = P(A)P(B)$, equivalently $P(A \mid B) = P(A)$. The **law of total probability** splits on a partition $B_1, \dots, B_k$ of $\Omega$: $P(A) = \sum_i P(A \mid B_i) P(B_i)$. **Bayes' rule** inverts a conditional:
$$P(B \mid A) = \frac{P(A \mid B)\,P(B)}{P(A)}.$$
For a classifier, $P(\text{flag} \mid \text{bad})$ is its **sensitivity** (recall), $P(\text{flag} \mid \text{ok})$ its false positive rate, and $P(\text{bad} \mid \text{flag})$ its **precision**, which depends on the base rate $P(\text{bad})$ as much as on the classifier.

## Discrete random variables

A **random variable** $X$ is a numeric function of the outcome, and its **distribution** is described by the mass function $p_X(k) = P(X = k)$. **Expectation** $E[X] = \sum_k k\, p_X(k)$ is linear, $E[aX + bY] = aE[X] + bE[Y]$, for any $X$ and $Y$, independent or not. **Variance** is $\operatorname{Var}(X) = E[(X - \mu)^2] = E[X^2] - \mu^2$ with $\mu = E[X]$; it satisfies $\operatorname{Var}(aX + b) = a^2 \operatorname{Var}(X)$, and for independent $X, Y$, $\operatorname{Var}(X + Y) = \operatorname{Var}(X) + \operatorname{Var}(Y)$. The distributions the set uses are these.

| Distribution | Values | $P(X = k)$ | $E[X]$ |
|---|---|---|---|
| Bernoulli($p$) | $\{0, 1\}$ | $p$ for 1, $1 - p$ for 0 | $p$ |
| Binomial($n, p$) | $\{0, \dots, n\}$ | $\binom{n}{k} p^k (1-p)^{n-k}$ | $np$ (a sum of $n$ Bernoullis) |
| Geometric($p$) | $\{1, 2, \dots\}$ | $(1-p)^{k-1} p$ | derive it in Problem 9 |
| Poisson($\lambda$) | $\{0, 1, \dots\}$ | $e^{-\lambda} \lambda^k / k!$ | $\lambda$ |
| Categorical($q$) | $\{0, \dots, V-1\}$ | $q_k$ | $\sum_k k\, q_k$ |

A language model's next-token distribution is a categorical over the vocabulary. To sample it from one uniform $u$, walk the ids in order with a running total $c$ and return the first id with $u < c$ ([`spec/sampling.md`](../../../../course/contracts/spec/sampling.md) step 11). The id returned is $k$ exactly when $q_0 + \dots + q_{k-1} \le u < q_0 + \dots + q_k$, an interval of length $q_k$, so each id comes up with its probability.

## Continuous random variables and the normal

A **continuous** $X$ has a density $f_X \ge 0$ with $\int f_X = 1$, and $P(a \le X \le b) = \int_a^b f_X(x)\,dx$; its **CDF** is $F_X(x) = \int_{-\infty}^x f_X$. Expectation becomes an integral, $E[g(X)] = \int g(x) f_X(x)\,dx$, with the same linearity and variance rules. The **uniform** on $[0, 1]$ has $f = 1$ there. The **exponential** with rate $\lambda$ has $F(x) = 1 - e^{-\lambda x}$ for $x \ge 0$. **Inverse-CDF sampling** sets $X = F^{-1}(U)$: then $P(X \le x) = P(U \le F(x)) = F(x)$, so one uniform becomes a draw from any distribution whose CDF can be inverted.

The **standard normal** $Z$ has density $\varphi(z) = e^{-z^2/2}/\sqrt{2\pi}$, mean 0, and variance 1; the constant comes from the Gaussian integral (Problem 16). $X = \mu + \sigma Z$ is normal with mean $\mu$ and variance $\sigma^2$. Its CDF has no closed form, so inverse-CDF sampling is awkward, and **Box-Muller** sidesteps it. For two independent standard normals, the squared radius $R^2 = Z_0^2 + Z_1^2$ is exponential with rate $1/2$, and the angle is uniform on $[0, 2\pi)$, independent of it. Reversing that: draw $u_1, u_2$, set $r = \sqrt{-2 \ln(1 - u_1)}$ (inverse-CDF sampling of the exponential, with $1 - u_1 > 0$ so the log is finite) and $\theta = 2\pi u_2$, and return $z_0 = r\cos\theta$ and $z_1 = r \sin\theta$. [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md) fixes the order: $z_0$ is returned, and $z_1$ is kept as the spare for the next call.

## Two examples by hand

These are siblings of Problems 3 and 15, not problems of the set.

**Bayes.** A spam filter catches 80% of spam and wrongly flags 2% of good mail, and 10% of mail is spam. What fraction of flagged mail is spam? Take 1000 messages: 100 spam, 900 good. Flagged spam: $0.8 \cdot 100 = 80$. Flagged good: $0.02 \cdot 900 = 18$. Flagged in total: $98$. So $P(\text{spam} \mid \text{flag}) = 80/98 = 40/49 \approx 0.816$. The same through the formula: $P(\text{flag}) = 0.8 \cdot 0.1 + 0.02 \cdot 0.9 = 0.098$, and $P(\text{spam} \mid \text{flag}) = 0.08 / 0.098 = 40/49$.

**Box-Muller.** Take $1 - u_1 = e^{-1/2}$ and $u_2 = 1/4$. Then $r = \sqrt{-2 \cdot (-1/2)} = 1$ and $\theta = 2\pi/4 = \pi/2$, so $z_0 = \cos(\pi/2) = 0$ and the spare is $z_1 = \sin(\pi/2) = 1$. This is the bookkeeping the random-variables section relies on: two uniforms in, two normals out, $z_0$ first.

## The problems

Probabilities are exact fractions ($7/10$, not 0.7), and lettered parts are answered separately. The tag after each problem says what kind of answer is expected: `[number]` an exact value, `[expr in ...]` a formula in the named variables, `[bool]` true or false, `[proof]` a written proof.

### Axioms, conditional probability, and Bayes

The first six problems exercise the axioms and conditioning, ending with a proof of inclusion-exclusion.

**Problem 1.** Events $A$ and $B$ have $P(A) = 1/2$, $P(B) = 2/5$, and $P(A \cap B) = 1/5$.
(a) $P(A \cup B)$. (b) $P(A \mid B)$. `[number]` (c) Are $A$ and $B$ independent? `[bool]`

<details><summary>Answer</summary>

**(a) $7/10$. (b) $1/2$. (c) Yes.** (a) Inclusion-exclusion (Problem 6): $P(A \cup B) = \frac12 + \frac25 - \frac15 = \frac{5 + 4 - 2}{10} = \frac{7}{10}$. A common mistake is $9/10$, which adds $P(A) + P(B)$ and counts the overlap twice. (b) $P(A \mid B) = P(A \cap B)/P(B) = \frac{1/5}{2/5} = \frac12$. A common mistake is $2/5$, which is $P(B \mid A) = \frac{1/5}{1/2}$: the conditioning event goes in the denominator. (c) $P(A)P(B) = \frac12 \cdot \frac25 = \frac15 = P(A \cap B)$, so they are independent; equivalently $P(A \mid B) = \frac12 = P(A)$.

</details>

**Problem 2.** Roll two fair six-sided dice. (a) $P(\text{the sum is } 7)$. (b) $P(\text{at least one die shows } 6)$. (c) $P(\text{the sum is } 7 \mid \text{at least one die shows } 6)$. `[number]`

<details><summary>Answer</summary>

**(a) $1/6$. (b) $11/36$. (c) $2/11$.** The 36 ordered pairs are equally likely. (a) Six pairs sum to 7, $(1,6), (2,5), \ldots, (6,1)$: $6/36 = 1/6$. A common mistake is $1/11$, treating the 11 possible sums $2, \ldots, 12$ as equally likely; they are not (one way to make 2, six ways to make 7). (b) The complement is "no 6 on either die", $(5/6)^2 = 25/36$, so $1 - 25/36 = 11/36$. Counting directly: 6 pairs with a 6 first, 6 with a 6 second, minus $(6, 6)$ counted twice. A common mistake is $1/6 + 1/6 = 1/3$, which double-counts $(6, 6)$. (c) Conditioning shrinks the sample space to the 11 pairs with a 6; of those, $(1, 6)$ and $(6, 1)$ sum to 7: $2/11$. Common mistakes: $1/6$ ignores the condition, and $2/36$ is the joint probability $P(\text{sum } 7 \cap \text{a } 6)$, not the conditional.

</details>

**Problem 3.** The gateway's usage-policy classifier flags 90% of violating requests and 5% of allowed ones, and 1% of requests violate the policy. (a) What fraction of all requests is flagged? (b) Given that a request is flagged, what is the probability that it violates the policy? `[number]`

<details><summary>Answer</summary>

**(a) $117/2000$. (b) $2/13$.** (a) Total probability over violate / allowed: $P(\text{flag}) = 0.9 \cdot 0.01 + 0.05 \cdot 0.99 = 0.009 + 0.0495 = 0.0585 = 117/2000$. A common mistake is $9/1000$, which counts only the true positives and forgets that 99% of traffic is allowed, so 5% of it is a much larger pool of flags. (b) Bayes: $P(\text{bad} \mid \text{flag}) = 0.009/0.0585 = 90/585 = 2/13 \approx 0.154$. In 10 000 requests: 100 violate and 90 are flagged; 9900 are allowed and 495 are flagged; of 585 flags, 90 are right. The common mistake is $9/10$, the sensitivity $P(\text{flag} \mid \text{bad})$ quoted as if it were the precision; the safety report of [Bias and safety evals](../../responsible-ai/04-bias-and-safety-evals/01-bias-and-safety-evals.md) in *Responsible AI* must quote precision, which the base rate drags down.

</details>

**Problem 4.** Draw 3 bytes independently and uniformly from the 256 byte values. What is the probability that all three are different? `[number]`

<details><summary>Answer</summary>

**$32385/32768$.** The first byte is anything; the second avoids one value, the third two: $\frac{256}{256} \cdot \frac{255}{256} \cdot \frac{254}{256} = \frac{255 \cdot 254}{65536} = \frac{64770}{65536} = \frac{32385}{32768} \approx 0.98831$. This is the birthday computation over the byte alphabet. A common mistake is $1 - 3/256$: the union bound says $P(\text{some pair equal}) \le \binom32/256 = 3/256$, which is an inequality, not the exact answer ($1 - 3/256 = 0.98828$ is slightly too small because the three pair-collision events overlap).

</details>

**Problem 5.** A bag holds a fair coin and a coin that lands heads with probability $3/4$. Pick one of the two uniformly at random and flip it. (a) $P(\text{heads})$. (b) $P(\text{the biased coin was picked} \mid \text{heads})$. `[number]`

<details><summary>Answer</summary>

**(a) $5/8$. (b) $3/5$.** (a) Total probability: $\frac12 \cdot \frac12 + \frac12 \cdot \frac34 = \frac14 + \frac38 = \frac58$. A common mistake is $3/4$, the biased coin's rate, as if it were always picked. (b) Bayes: $\frac{P(\text{biased} \cap \text{heads})}{P(\text{heads})} = \frac{3/8}{5/8} = \frac35$. A common mistake is $3/8$, the joint probability, without dividing by $P(\text{heads})$. Heads is evidence for the biased coin, so the posterior $3/5$ rises above the prior $1/2$.

</details>

**Problem 6.** From the three axioms ($P(E) \ge 0$; $P(\Omega) = 1$; $P$ of a countable union of pairwise disjoint events is the sum of their probabilities) prove $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ for any events $A$ and $B$. `[proof]`

<details><summary>Answer (model proof)</summary>

**Claim.** For any events $A, B$, $P(A \cup B) = P(A) + P(B) - P(A \cap B)$.

**Method.** Split each set into disjoint pieces and apply the third axiom. A finite union is a countable one padded with empty sets, and $P(\emptyset) = 0$ follows from additivity applied to $\emptyset = \emptyset \cup \emptyset \cup \cdots$ together with $P \ge 0$. Write $B \setminus A = B \cap A^c$.

1. $A \cup B = A \cup (B \setminus A)$, and $A$ and $B \setminus A$ are disjoint (an element of $B \setminus A$ is not in $A$). By additivity, $P(A \cup B) = P(A) + P(B \setminus A)$.
2. $B = (A \cap B) \cup (B \setminus A)$, and these two are disjoint (one lies inside $A$, the other outside it). By additivity, $P(B) = P(A \cap B) + P(B \setminus A)$, so $P(B \setminus A) = P(B) - P(A \cap B)$.
3. Substituting step 2 into step 1: $P(A \cup B) = P(A) + P(B) - P(A \cap B)$. $\blacksquare$

Every step used only set identities and the additivity axiom. A complete proof must (i) write both $A \cup B$ and $B$ as disjoint unions, (ii) apply additivity to each, (iii) eliminate $P(B \setminus A)$, and (iv) name each step. A common gap is to "subtract the overlap because it was counted twice", which is the intuition, not a proof from the axioms. The model proof is also in [`course/solutions/solve/S-M07a/q6.md`](../../../../course/solutions/solve/S-M07a/q6.md).

</details>

### Discrete random variables

The next five problems compute means and variances of the standard discrete distributions and walk the inverse CDF of a categorical.

**Problem 7.** $X$ is a fair die roll, uniform on $\{1, \dots, 6\}$. (a) $E[X]$. (b) $E[X^2]$. (c) $\operatorname{Var}(X)$. `[number]`

<details><summary>Answer</summary>

**(a) $7/2$. (b) $91/6$. (c) $35/12$.** (a) $(1 + \cdots + 6)/6 = 21/6 = 7/2$; a common slip is 3, which is not the midpoint of $1..6$. (b) $(1 + 4 + 9 + 16 + 25 + 36)/6 = 91/6$. A common mistake is $49/4 = (E[X])^2$: the mean of the squares is not the square of the mean. (c) $E[X^2] - E[X]^2 = \frac{91}{6} - \frac{49}{4} = \frac{182 - 147}{12} = \frac{35}{12}$, matching the two-pass computation in [Random variables, expectation, variance, and normal draws by Box-Muller](00-random-variables-and-the-normal.md). Common mistakes: $35/2$, the sum of squared deviations $2(6.25 + 2.25 + 0.25) = 17.5$ not divided by 6; and $7/2$, that sum divided by $n - 1 = 5$, which is the *sample* variance formula, not the variance of a known distribution.

</details>

**Problem 8.** (a) Give $\operatorname{Var}(X)$ for $X \sim \mathrm{Bernoulli}(p)$. `[expr in p]` (b) Give $\operatorname{Var}(Y)$ for $Y \sim \mathrm{Binomial}(n, p)$, the number of successes in $n$ independent Bernoulli($p$) trials. `[expr in n, p]`

<details><summary>Answer</summary>

**(a) $p(1 - p)$. (b) $np(1 - p)$.** (a) $X^2 = X$ for a 0/1 variable, so $E[X^2] = p$ and $\operatorname{Var}(X) = p - p^2 = p(1-p)$. Common mistakes: $p$ (the mean) and $p^2$ (the squared mean). (b) $Y = X_1 + \cdots + X_n$ with independent $X_i$, and variances of independent sums add: $n p(1-p)$. A common mistake is $n^2 p(1 - p)$, which is $\operatorname{Var}(nX_1)$: one trial scaled by $n$, perfectly correlated, not $n$ independent trials. The difference is why the standard error of an accuracy shrinks like $1/\sqrt n$ ([LLN, CLT, confidence intervals, bootstrap](04-lln-clt-confidence-intervals-bootstrap.md), and the evaluation harness of [The evaluation harness and the model zoo](../../build-an-llm/p06-objectives/07-lm-evaluation-harness-and-the-model-zoo.md)).

</details>

**Problem 9.** $X$ counts independent Bernoulli($p$) trials up to and including the first success, so $X \in \{1, 2, 3, \dots\}$. Give $E[X]$. `[expr in p]`

<details><summary>Answer</summary>

**$1/p$.** First-step analysis: one trial is always made; with probability $1 - p$ it fails and the process starts afresh, so $E[X] = 1 + (1 - p)E[X]$, giving $pE[X] = 1$. Directly: $E[X] = \sum_{k \ge 1} k (1-p)^{k-1} p = p \cdot \frac{1}{(1 - (1-p))^2} = \frac1p$, using $\sum_{k \ge 1} k r^{k-1} = 1/(1-r)^2$. A common mistake is $(1 - p)/p$, the expected number of *failures* before the first success; it differs by the success trial itself.

</details>

**Problem 10.** A sampler draws a token id from the categorical distribution $q = (1/10, 2/10, 3/10, 4/10)$ over ids $0, 1, 2, 3$. (a) Give $E[\text{id}]$. The sampler then uses the inverse CDF of `spec/sampling.md`: walk the ids in ascending order adding $q_i$ to a running total $c$, and return the first id with $u < c$. Which id does it return for (b) $u = 0.35$ and (c) $u = 0.3$ (exactly)? `[number]`

<details><summary>Answer</summary>

**(a) $2$. (b) $2$. (c) $2$.** (a) $0 \cdot 0.1 + 1 \cdot 0.2 + 2 \cdot 0.3 + 3 \cdot 0.4 = 0.2 + 0.6 + 1.2 = 2$. A common mistake is $3/2$, the plain mean of the ids, which ignores $q$. (b) The running totals are $0.1, 0.3, 0.6, 1.0$; $0.35 < 0.1$ and $0.35 < 0.3$ fail, $0.35 < 0.6$ holds: id 2. (c) $0.3 < 0.1$ fails and $0.3 < 0.3$ fails (strict), $0.3 < 0.6$ holds: id 2. Id 1 owns $[0.1, 0.3)$, half open, so its right end belongs to id 2. The common mistake is id 1, from $u \le c$; then a Python and a Rust sampler that disagree on the comparison return different tokens at boundaries. (In float64 the running total $0.1 + 0.2$ is $0.30000000000000004$, slightly above the double $0.3$, so a float sampler returns id 1 here; the question asks for the exact-arithmetic rule, and the spec fixes the float operations so every port rounds the same way.)

</details>

**Problem 11.** $X \sim \mathrm{Poisson}(\lambda)$ has $P(X = k) = e^{-\lambda} \lambda^k / k!$. Give $P(X = 0)$. `[expr in lam]`

<details><summary>Answer</summary>

**$e^{-\lambda}$.** Set $k = 0$: $\lambda^0/0! = 1$. This is also the probability that an exponential waiting time with rate $\lambda$ exceeds 1, which is how Poisson arrivals (in [Categorical sampling](01-categorical-sampling.md)) and the exponential connect. A common mistake is $1 - e^{-\lambda}$, which is $P(X \ge 1)$.

</details>

### Continuous random variables, the normal, and Box-Muller

The last five problems move to densities: the uniform and exponential, scaling a normal, Box-Muller, and the Gaussian integral that normalizes it all.

**Problem 12.** $U$ is uniform on $[0, 1]$. (a) $E[U]$. (b) $\operatorname{Var}(U)$. `[number]`

<details><summary>Answer</summary>

**(a) $1/2$. (b) $1/12$.** (a) $\int_0^1 u\,du = 1/2$. (b) $E[U^2] = \int_0^1 u^2\,du = 1/3$, so $\operatorname{Var}(U) = 1/3 - 1/4 = 1/12$. A common mistake is $1/3$, which is $E[U^2]$ without subtracting the squared mean. Scaling gives $\operatorname{Var}\,\mathcal{U}(-a, a) = (2a)^2/12 = a^2/3$, the fact behind Xavier-uniform's bound in [Variance propagation and initialization](03-variance-propagation-and-initialization.md).

</details>

**Problem 13.** $X \sim \mathrm{Exponential}(\lambda)$ has CDF $F(x) = 1 - e^{-\lambda x}$ for $x \ge 0$.
(a) Give the inverse-CDF sampler: the $x$ with $F(x) = u$. `[expr in u, lam]` (b) Give $E[X]$. `[expr in lam]`

<details><summary>Answer</summary>

**(a) $-\ln(1 - u)/\lambda$. (b) $1/\lambda$.** (a) $1 - e^{-\lambda x} = u \Rightarrow e^{-\lambda x} = 1 - u \Rightarrow x = -\ln(1-u)/\lambda$. This is `exponential_icdf` from [Categorical sampling](01-categorical-sampling.md). Common mistakes: $-\lambda\ln(1-u)$, multiplying by the rate (a higher rate must mean *shorter* waits), and $e^{-\lambda u}$, the survival function evaluated at $u$, not its inverse. (b) $E[X] = \int_0^\infty P(X > x)\,dx = \int_0^\infty e^{-\lambda x}\,dx = 1/\lambda$. A common mistake is $\lambda$: the rate is events per unit time, and the mean wait is its reciprocal.

</details>

**Problem 14.** $Z$ is a standard normal and $a, b$ are constants. Give $\operatorname{Var}(aZ + b)$. `[expr in a, b]`

<details><summary>Answer</summary>

**$a^2$.** $\operatorname{Var}(aZ + b) = E[(aZ + b - b)^2] = a^2 E[Z^2] = a^2$, since $E[aZ + b] = b$ and $\operatorname{Var}(Z) = 1$. `normal_init(std)` in [Variance propagation and initialization](03-variance-propagation-and-initialization.md) is $\text{std} \cdot Z$, with variance $\text{std}^2$. Common mistakes: $a^2 + b$ (a shift does not spread the distribution) and $a$, which confuses the variance with the standard deviation $\lvert a \rvert$.

</details>

**Problem 15.** `spec/pcg32.md` turns two uniforms into two normals by Box-Muller: $r = \sqrt{-2 \ln(1 - u_1)}$, then $z_0 = r \cos(2\pi u_2)$ is returned and $z_1 = r \sin(2\pi u_2)$ is kept as the spare. For $u_1$ with $1 - u_1 = e^{-2}$ and $u_2 = 1/12$, give (a) $z_0$ and (b) $z_1$. `[number]`

<details><summary>Answer</summary>

**(a) $\sqrt3$. (b) $1$.** $r = \sqrt{-2 \ln e^{-2}} = \sqrt4 = 2$ and $\theta = 2\pi/12 = \pi/6$. (a) $z_0 = 2\cos(\pi/6) = 2 \cdot \frac{\sqrt3}{2} = \sqrt3$. (b) $z_1 = 2\sin(\pi/6) = 2 \cdot \frac12 = 1$. Common mistakes for (a): $1$, from swapping sine and cosine (the spec emits the cosine first, and every port must agree); and $\sqrt3/2$, from forgetting the radius. For (b), $\sqrt3$ is the same swap.

</details>

**Problem 16.** Give $\int_{-\infty}^{\infty} e^{-x^2/2}\, dx$, the constant that normalizes the standard normal density. `[number]`

<details><summary>Answer</summary>

**$\sqrt{2\pi}$.** Square the integral and switch to polar coordinates: $I^2 = \iint e^{-(x^2 + y^2)/2}\,dx\,dy = \int_0^{2\pi}\!\!\int_0^\infty e^{-r^2/2}\, r\,dr\,d\theta = 2\pi \cdot \bigl[-e^{-r^2/2}\bigr]_0^\infty = 2\pi$, so $I = \sqrt{2\pi}$. The same polar picture, run backwards, is Box-Muller (see "Continuous random variables and the normal" above). Common mistakes: $\sqrt\pi$, which is $\int e^{-x^2}\,dx$ (no factor $\tfrac12$ in the exponent), and $1$, which is the integral of the already normalized density.

</details>

## What can go wrong

The wrong answers in this set cluster around a handful of habits, and each has a downstream cost. Adding the probabilities of overlapping events can push $P(A \cup B)$ above 1 for large events; the fix is to subtract the overlap (Problems 1a and 2b). Confusing $P(A \mid B)$ with $P(B \mid A)$ quotes a classifier's recall as its precision; Bayes with the base rate in the numerator and $P(\text{flag})$ below gets it right (Problems 3b and 5b). Treating unequal outcomes as equally likely gives each of the 11 dice sums probability 1/11; counting the 36 equally likely ordered pairs does not (Problem 2a).

> **Warning:** Computing variance as $E[X^2]$, or with $n - 1$ in a population formula, puts initialization scales off by the squared mean. Use $E[X^2] - E[X]^2$ over the known distribution (Problems 7c and 12b). Likewise, a shift never changes variance, $\operatorname{Var}(aZ + b) = a^2$ (Problem 14), so a `normal_init` with a nonzero mean would break the variance argument of the initialization section.

The two sampling pitfalls matter for cross-language parity. Using $u \le c$ instead of $u < c$ in the inverse CDF makes a Python and a Rust sampler disagree at boundaries; half-open intervals send $u = 0.3$ to id 2 (Problem 10c). Swapping sine and cosine, or forgetting the radius, makes normals differ from `spec/pcg32.md`; the spec has $z_0 = r\cos\theta$, with $r = 2$ in Problem 15.

## Where this leads

The counting behind Problems 2 and 4 comes from the [Discrete math 1 problem set](../05-discrete-math-1/90-problem-set.md). Next in this chapter, [Random variables, expectation, variance, and normal draws by Box-Muller](00-random-variables-and-the-normal.md) builds `normal`, `expectation`, and `variance` over PCG32, checked by sample moments and a chi-square test; [Variance propagation and initialization](03-variance-propagation-and-initialization.md) uses variance propagation to pick the `xavier_*` and `kaiming_normal` scales; and [Categorical sampling: inverse CDF, Gumbel-max, alias method](01-categorical-sampling.md) samples categoricals three ways, followed by [problem set B](91-problem-set-b.md). Further on, the sampler of [Sampling and logit processors](../../build-an-llm/p08-inference/01-sampling-and-logit-processors.md) in *Build Your Own LLM System* uses the inverse-CDF step with the strict comparison of Problem 10, and [Usage policy enforcement](../../ai-platform-engineering/12-gateway/08-usage-policy-enforcement.md) in *AI Platform Engineering* reports the policy head's precision and recall, which you now read with Problem 3.

### Further reading

- Blitzstein and Hwang, *Introduction to Probability* (free), ch. 1 to 5.
- Box and Muller, "A Note on the Generation of Random Normal Deviates" (1958).

## Key terms

- **Probability axioms**: non-negativity, $P(\Omega) = 1$, and countable additivity over disjoint events; every other rule is derived from them.
- **Conditional probability**: $P(A \mid B) = P(A \cap B)/P(B)$, probability restricted to $B$ and renormalized.
- **Bayes' rule**: $P(B \mid A) = P(A \mid B) P(B) / P(A)$, which inverts a conditional using the base rate.
- **Precision**: $P(\text{bad} \mid \text{flag})$, the fraction of flags that are right; unlike sensitivity it depends on the base rate.
- **Variance**: $E[X^2] - E[X]^2$; it scales by $a^2$ and adds over independent sums.
- **Inverse-CDF sampling**: $X = F^{-1}(U)$, which turns one uniform into a draw from $F$.
- **Box-Muller**: the transform $r = \sqrt{-2\ln(1-u_1)}$, $\theta = 2\pi u_2$ that turns two uniforms into two independent standard normals.

## Summary

- Probability is a function on events with three axioms; inclusion-exclusion and conditioning follow from them (Problems 1 and 6).
- Bayes' rule turns a classifier's per-class rates into the probability a flag is right, and with a 1% base rate a 90%-sensitive classifier is right only 2 times in 13 (Problem 3).
- Expectation is linear and variance is $E[X^2] - E[X]^2$; variances of independent sums add, and a scale $a$ multiplies variance by $a^2$ (Problems 7, 8, and 14).
- An inverse CDF turns one uniform into any distribution, and its comparison is strict: $u < c$, so a $u$ landing exactly on a boundary goes to the next id (Problems 10 and 13).
- Box-Muller turns two uniforms into two independent standard normals, cosine first, sine as the spare (Problem 15).
