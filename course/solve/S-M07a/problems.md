# S-M07a problems: probability axioms, Bayes, discrete and continuous random variables

Answer every question in `solve/S-M07a.toml` (written by `ol start S-M07a`).
The tag after each question is its answer type: `[number]` is an exact value
(`7/10`, `sqrt(3)`, `exp(-2)`), `[expr]` a formula in the named variables
(`p*(1 - p)`; write $\lambda$ as `lam`), `[bool]` is `true` or `false`, and
`[proof]` a file `solve/S-M07a/qN.md` graded against its rubric.

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
