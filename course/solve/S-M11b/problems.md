# S-M11b problems: coding theorems and Huffman, mutual information, maximum entropy, rate-distortion

Answer every question in `solve/S-M11b.toml` (written by `ol start S-M11b`).
The tag after each question is its answer type: `[number]` is an exact value
(`7/4`, `log(3, 2)/2`), `[vector]` a flat list `[1, 2, 3]`, `[bool]` `true`
or `false`, `[choice]` the letter of one option, and `[proof]` a file
`solve/S-M11b/qN.md` graded against its rubric. Write $\log_2 x$ as
`log(x, 2)` and the natural log as `log(x)`. Lettered parts are answered
separately (`[q1.a]`, `[q1.b]`). The convention $0 \log 0 = 0$ holds
throughout.

### Coding theorems and Huffman

**q1.** A source emits four symbols with probabilities $(1/2, 1/4, 1/8, 1/8)$. (a) Give the codeword lengths of a binary Huffman code, in the order of the probabilities. `[vector]` (b) Give its expected length in bits. `[number]` (c) Is that expected length equal to the entropy of the source? `[bool]`

**q2.** For probabilities $(0.4, 0.3, 0.2, 0.1)$: (a) give the expected length in bits of a binary Huffman code; (b) give the Kraft sum $\sum_i 2^{-\ell_i}$ of its codeword lengths $\ell_i$. `[number]`

**q3.** For the same probabilities $(0.4, 0.3, 0.2, 0.1)$: (a) give the expected length in bits of the Shannon code, whose lengths are $\ell_i = \lceil \log_2 (1/p_i) \rceil$. `[number]` (b) Does a binary prefix code with codeword lengths $(1, 2, 2, 3)$ exist? `[bool]`

**q4.** Prove the Kraft inequality: the codeword lengths $\ell_1, \dots, \ell_m$ of any binary prefix code satisfy $\sum_i 2^{-\ell_i} \le 1$. `[proof]`

### Mutual information

**q5.** $X$ and $Y$ take values in $\{0, 1\}$ with $P(0, 0) = P(1, 1) = 3/8$ and $P(0, 1) = P(1, 0) = 1/8$. In bits, give (a) the mutual information $I(X; Y)$, (b) the pointwise mutual information $\operatorname{PMI}(0, 0) = \log_2 \frac{P(0, 0)}{P_X(0) P_Y(0)}$, (c) $\operatorname{PMI}(0, 1)$. `[number]`

**q6.** (a) $X$ is uniform on $\{0, 1, 2, 3\}$ and $Y = X \bmod 2$. Give $I(X; Y)$ in bits. `[number]` (b) Is $I(X; Y) = I(Y; X)$ for every joint distribution? `[bool]`

**q7.** Prove that $I(X; Y) = H(X) + H(Y) - H(X, Y)$, where $I(X; Y) = \sum_{x, y} P(x, y) \log \frac{P(x, y)}{P(x) P(y)}$, and that $I(X; Y) \ge 0$ with equality exactly when $X$ and $Y$ are independent. You may use Gibbs' inequality (S-M11a q9). `[proof]`

### Maximum entropy

**q8.** A softmax at temperature $T$ turns logits $z$ into $p_i = e^{z_i / T} / \sum_j e^{z_j / T}$. Let $z = (0, \ln 2, \ln 3)$. (a) Give $p_3$ at $T = 1$. (b) Give $p_3$ at $T = 1/2$. `[number]` (c) As $T \to \infty$, the distribution tends to: (a) the one-hot vector on the largest logit, (b) the uniform distribution, (c) the distribution at $T = 1$. `[choice]`

**q9.** Derive, with a Lagrange multiplier for each constraint, that among distributions $p$ on $\{1, \dots, V\}$ with a fixed mean energy $\sum_i p_i E_i = \bar{E}$, the one with maximum entropy has the form $p_i = e^{-\beta E_i} / Z(\beta)$ with $Z(\beta) = \sum_j e^{-\beta E_j}$: a softmax of the logits $-E_i$ at temperature $1/\beta$. `[proof]`

### Rate-distortion

**q10.** A fair binary source ($P(1) = 1/2$) with Hamming distortion has the rate-distortion function $R(D) = 1 - H_b(D)$ bits per symbol for $0 \le D \le 1/2$, where $H_b$ is the binary entropy. Give (a) $R(1/8)$ and (b) $R(1/2)$. `[number]`

**q11.** A Gaussian source with variance $\sigma^2$ and squared-error distortion has $R(D) = \frac12 \log_2 (\sigma^2 / D)$ bits per sample for $0 < D \le \sigma^2$. (a) How many bits per sample are needed to reach $D = \sigma^2 / 16$? (b) Each extra bit per sample divides the smallest achievable distortion by what factor? `[number]`
