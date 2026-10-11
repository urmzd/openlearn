# S-M11a problems: entropy and chain rules, cross-entropy and KL divergence

Answer every question in `solve/S-M11a.toml` (written by `ol start S-M11a`).
The tag after each question is its answer type: `[number]` is an exact value
(`3/2`, `2 - 3*log(3, 2)/4`, `8*log(2)`), `[expr]` a formula in the named
variables, `[bool]` `true` or `false`, and `[proof]` a file
`solve/S-M11a/qN.md` graded against its rubric. Write $\log_2 x$ as
`log(x, 2)` or `log(x)/log(2)`, and the natural log as `log(x)`. The
convention $0 \log 0 = 0$ holds throughout.

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
