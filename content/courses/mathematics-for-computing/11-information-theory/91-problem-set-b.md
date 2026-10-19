# Information theory problem set, part b: coding and Huffman, mutual information, maximum entropy, rate-distortion

This is where information theory becomes things the system does. [Perplexity, bits per byte, NLL accumulator](02-perplexity-bits-per-byte-nll-accumulator.md) turned the models' losses into bits per byte, which is literally a compression rate: a model with 2 bits per byte could drive an arithmetic coder that shrinks text to a quarter of its size, and this set proves why no code can beat the entropy. [Mutual information, PMI, and PPMI](04-mutual-information-pmi-ppmi.md) built mutual information and PMI for the word-vector baseline of [word2vec skip-gram with negative sampling, and the PPMI-SVD baseline](../../build-an-llm/p02-statistical-lm/03-word2vec-and-ppmi-svd.md) in *Build Your Own LLM System*; here you compute them by hand.

Two topics have no section of their own in this book and live only in this set. The first is the maximum-entropy reading of softmax, which is what the temperature knob of the sampler in [Sampling and logit processors](../../build-an-llm/p08-inference/01-sampling-and-logit-processors.md) means. The second is rate-distortion, the theory behind the quantizers of [Quantization](../../build-an-llm/p08-inference/05-quantization.md) trading bits for error. The coding problems also supply the theorems behind source coding with Huffman and arithmetic codes, whose greedy construction is treated as an algorithm in [Greedy algorithms](../../algorithms/08-greedy/README.md) in *Algorithms*.

The set has eleven problems, nineteen exact parts and three proofs (Problems 4, 7, and 9), each with a worked answer folded underneath. It builds on [Problem set A](90-problem-set-a.md) for entropy, KL, and Gibbs' inequality, and on the perplexity and mutual-information sections of this chapter.

> **Learning objectives**
> - Build a Huffman code, compute its expected length and Kraft sum, and compare it with the entropy and the Shannon code.
> - Prove Kraft's inequality and the identity $I(X; Y) = H(X) + H(Y) - H(X, Y)$.
> - Compute mutual information and PMI for a small joint table, in bits.
> - Derive softmax at temperature $T$ as the maximum-entropy distribution and describe its limits as $T \to 0$ and $T \to \infty$.
> - Evaluate the rate-distortion function for a binary and a Gaussian source.

## The ideas you need

A **prefix code** assigns each symbol a bit string so that no codeword is the start of another; a stream of codewords then decodes without separators. Write $\ell_i$ for the length of symbol $i$'s codeword and $L = \sum_i p_i \ell_i$ for the expected code length. **Kraft's inequality** (Problem 4) says any prefix code's lengths satisfy $\sum_i 2^{-\ell_i} \le 1$, and conversely any lengths satisfying it have a prefix code. Minimizing $L$ under that constraint gives $\ell_i = -\log_2 p_i$ and $L = H(p)$: no code beats the entropy. Lengths must be integers, so the **Shannon code** rounds up, $\ell_i = \lceil -\log_2 p_i \rceil$, and lands within one bit: $H \le L < H + 1$. The **Huffman code** is optimal among prefix codes: repeatedly merge the two least likely symbols (or groups) into one node; each symbol's length is the number of merges it took part in.

Mutual information connects to this through the same KL machinery. $I(X; Y) = \sum_{x, y} P(x, y) \log \frac{P(x, y)}{P(x)P(y)}$ is the KL divergence from the joint to the product of its marginals, the expected PMI, where $\operatorname{PMI}(x, y) = \log \frac{P(x, y)}{P(x) P(y)}$, and also $H(X) + H(Y) - H(X, Y) = H(Y) - H(Y \mid X)$. It is symmetric, non-negative, and zero exactly when $X$ and $Y$ are independent. A single pair's PMI can be negative (seen together less than chance); only the average must be non-negative.

Maximum entropy explains temperature. Among all distributions with a given expected energy, the one with the most entropy is the **Gibbs distribution** $p_i = e^{-\beta E_i}/Z(\beta)$ (Problem 9). With logits $z_i = -E_i$ and temperature $T = 1/\beta$, that is softmax at temperature $T$: $p_i \propto e^{z_i / T}$. Dividing logits by $T < 1$ sharpens the distribution and $T > 1$ flattens it. As $T \to \infty$ every $z_i/T \to 0$ and the distribution becomes uniform; as $T \to 0$ it concentrates on the largest logit, which is greedy decoding (the sampler treats $T = 0$ as greedy).

Rate-distortion is the lossy counterpart of the coding theorem. It asks for the fewest bits per symbol, $R(D)$, that reproduce a source within an average distortion $D$. For a fair coin with Hamming distortion (the fraction of flipped bits), $R(D) = 1 - H_b(D)$ for $0 \le D \le 1/2$, where $H_b(q) = -q\log_2 q - (1-q)\log_2(1-q)$ is the binary entropy: allowing errors saves exactly the entropy of the error pattern, and at $D = 1/2$ the decoder can guess without receiving anything. For a Gaussian source with variance $\sigma^2$ and squared error, $R(D) = \frac12 \log_2(\sigma^2/D)$, so $D = \sigma^2 \cdot 2^{-2R}$: every extra bit divides the error by 4, the "6 dB per bit" rule of quantization.

## Four warm-up examples

These are siblings of Problems 1, 5, 8, and 10, worked in full before you try the real ones.

For **Huffman**, take probabilities $(1/2, 1/4, 1/4)$. Merge the two quarters into one node of $1/2$, then merge that node with the $1/2$. The lengths are $(1, 2, 2)$, $L = 1/2 + 1/2 + 1/2 = 3/2$ bits, and $H = 1/2 \cdot 1 + 2 \cdot 1/4 \cdot 2 = 3/2$: dyadic probabilities are coded at exactly the entropy. The Kraft sum is $1/2 + 1/4 + 1/4 = 1$.

For **mutual information**, take $P(0, 0) = P(1, 1) = 1/2$ and $P(0, 1) = P(1, 0) = 0$, so $Y = X$, a fair bit. The marginals are uniform, $\operatorname{PMI}(0, 0) = \log_2 \frac{1/2}{1/4} = 1$ bit, and $I = 2 \cdot \frac12 \cdot 1 = 1$ bit $= H(X)$: knowing $X$ removes all of $Y$'s uncertainty.

For **temperature**, take logits $z = (0, \ln 3)$. At $T = 1$, $p = (1/4, 3/4)$; at $T = 1/2$, $e^{2 z} = (1, 9)$ and $p = (1/10, 9/10)$, sharper.

For **rate-distortion**, a fair coin at $D = 1/4$ has $H_b(1/4) = 2 - \frac34 \log_2 3$, so $R(1/4) = \frac34\log_2 3 - 1 \approx 0.19$ bits per symbol. A Gaussian at $D = \sigma^2/4$ needs $R = \frac12\log_2 4 = 1$ bit.

## The problems

Each problem carries a tag for the kind of answer it wants (`[number]`, `[vector]`, `[bool]`, `[choice]`, `[proof]`). Answers are exact and in bits unless stated. Try each one on paper first, then open the answer and compare the derivation. Each answer names the common wrong answer and why it is wrong.

### Coding theorems and Huffman


**Problem 1.** A source emits four symbols with probabilities $(1/2, 1/4, 1/8, 1/8)$. (a) Give the codeword lengths of a binary Huffman code, in the order of the probabilities. `[vector]` (b) Give its expected length in bits. `[number]` (c) Is that expected length equal to the entropy of the source? `[bool]`

<details><summary>Answer</summary>

**(a) $[1, 2, 3, 3]$, (b) $7/4$ bits, (c) true.**

1. (a) Merge the two $1/8$s into a $1/4$ node; merge that with the $1/4$ symbol into a $1/2$ node; merge that with the $1/2$ symbol. The $1/2$ symbol took part in one merge, the $1/4$ in two, each $1/8$ in three. A common mistake is $[3, 3, 2, 1]$: the right multiset listed in the wrong order, which describes a different (bad) code. Another is $[2, 2, 2, 2]$, the fixed-length code.
2. (b) $L = \tfrac12 \cdot 1 + \tfrac14 \cdot 2 + \tfrac18 \cdot 3 + \tfrac18 \cdot 3 = 7/4$. A common mistake is 2, the fixed-length code's length.
3. (c) $H = \tfrac12 \cdot 1 + \tfrac14 \cdot 2 + 2 \cdot \tfrac18 \cdot 3 = 7/4$. Every probability is a power of $1/2$ (dyadic), so the ideal lengths $-\log_2 p_i$ are integers and Huffman reaches them exactly.

</details>

**Problem 2.** For probabilities $(0.4, 0.3, 0.2, 0.1)$: (a) give the expected length in bits of a binary Huffman code; (b) give the Kraft sum $\sum_i 2^{-\ell_i}$ of its codeword lengths $\ell_i$. `[number]`

<details><summary>Answer</summary>

**(a) $19/10$ bits, (b) 1.**

1. (a) Merge $0.1 + 0.2 = 0.3$; the two smallest are now $0.3$ and $0.3$, merge them to $0.6$; merge $0.4 + 0.6$. Lengths $(1, 2, 3, 3)$ for $(0.4, 0.3, 0.2, 0.1)$, and $L = 0.4 + 0.6 + 0.6 + 0.3 = 1.9$. A common mistake is the entropy $H \approx 1.846$: that is the lower bound, which no integer-length code reaches here since the probabilities are not dyadic. Another is 2, the fixed-length code.
2. (b) $2^{-1} + 2^{-2} + 2^{-3} + 2^{-3} = 1$. A Huffman tree is complete (every internal node has two children), so its Kraft sum is exactly 1. A sum below 1 would mean a codeword could be shortened.

</details>

**Problem 3.** For the same probabilities $(0.4, 0.3, 0.2, 0.1)$: (a) give the expected length in bits of the Shannon code, whose lengths are $\ell_i = \lceil \log_2 (1/p_i) \rceil$. `[number]` (b) Does a binary prefix code with codeword lengths $(1, 2, 2, 3)$ exist? `[bool]`

<details><summary>Answer</summary>

**(a) $12/5$ bits, (b) false.**

1. (a) $\log_2(1/p_i) \approx (1.32, 1.74, 2.32, 3.32)$, rounded up to $(2, 2, 3, 4)$. $L = 0.8 + 0.6 + 0.6 + 0.4 = 2.4$, within one bit of $H \approx 1.846$ as promised, and worse than Huffman's 1.9. A common mistake is 1.9, the Huffman length; another is rounding down instead of up, which gives lengths whose Kraft sum exceeds 1, so no prefix code has them.
2. (b) The Kraft sum is $\tfrac12 + \tfrac14 + \tfrac14 + \tfrac18 = 9/8 > 1$, so by Kraft's inequality no prefix code with these lengths exists. Concretely: after `0`, `10`, `11` use up the whole tree, there is no room left for a length-3 codeword.

</details>

**Problem 4.** Prove the Kraft inequality: the codeword lengths $\ell_1, \dots, \ell_m$ of any binary prefix code satisfy $\sum_i 2^{-\ell_i} \le 1$. `[proof]`

<details><summary>Answer</summary>

**Proof.** Let $L = \max_i \ell_i$ and picture the complete binary tree of depth $L$, with the root as the empty string and each left/right step appending a 0 or a 1. Codeword $i$ is a node at depth $\ell_i$. Below it lie exactly $2^{L - \ell_i}$ of the $2^L$ leaves at depth $L$ (all ways to extend it to length $L$).

By the prefix property no codeword is an ancestor of another, so the sets of leaves below different codewords are disjoint: a leaf below both $i$ and $j$ would make one of them a prefix of the other. Disjoint subsets of $2^L$ leaves have at most $2^L$ elements in total:

$$\sum_i 2^{L - \ell_i} \le 2^L.$$

Dividing by $2^L$ gives $\sum_i 2^{-\ell_i} \le 1$. $\blacksquare$

The intuition: a codeword of length $\ell$ "uses up" a fraction $2^{-\ell}$ of the tree, and prefix-freeness means the used-up parts cannot overlap.

</details>

### Mutual information

**Problem 5.** $X$ and $Y$ take values in $\{0, 1\}$ with $P(0, 0) = P(1, 1) = 3/8$ and $P(0, 1) = P(1, 0) = 1/8$. In bits, give (a) the mutual information $I(X; Y)$, (b) the pointwise mutual information $\operatorname{PMI}(0, 0) = \log_2 \frac{P(0, 0)}{P_X(0) P_Y(0)}$, (c) $\operatorname{PMI}(0, 1)$. `[number]`

<details><summary>Answer</summary>

**(a) $\tfrac34\log_2 3 - 1 \approx 0.189$, (b) $\log_2 3 - 1 \approx 0.585$, (c) $-1$ bits.**

The marginals are $P_X = P_Y = (1/2, 1/2)$, so each $P(x)P(y) = 1/4$.

1. (b) $\operatorname{PMI}(0, 0) = \log_2\frac{3/8}{1/4} = \log_2\frac32 = \log_2 3 - 1$. A common mistake is $\log_2(3/8)$, the log of the joint alone.
2. (c) $\operatorname{PMI}(0, 1) = \log_2\frac{1/8}{1/4} = -1$: the pair occurs half as often as chance. A common mistake is 0, which is PPMI (the clipped value), not PMI; another is $+1$, a flipped ratio.
3. (a) $I$ is the expected PMI: $2 \cdot \tfrac38(\log_2 3 - 1) + 2 \cdot \tfrac18 \cdot (-1) = \tfrac34\log_2 3 - \tfrac34 - \tfrac14 = \tfrac34\log_2 3 - 1$. A common mistake is the natural-log value $\tfrac34\ln 3 - \ln 2$ (nats); another is 1, which is $H(X)$, the most $I$ could be.

</details>

**Problem 6.** (a) $X$ is uniform on $\{0, 1, 2, 3\}$ and $Y = X \bmod 2$. Give $I(X; Y)$ in bits. `[number]` (b) Is $I(X; Y) = I(Y; X)$ for every joint distribution? `[bool]`

<details><summary>Answer</summary>

**(a) 1 bit, (b) true.**

1. (a) $I(X; Y) = H(Y) - H(Y \mid X)$. $Y$ is a fair bit, so $H(Y) = 1$, and $Y$ is a function of $X$, so $H(Y \mid X) = 0$. Hence $I = 1$. A common mistake is 2, which is $H(X)$: knowing $Y$ only tells you the parity, one of the two bits of $X$.
2. (b) The definition $\sum P(x, y)\log\frac{P(x, y)}{P(x)P(y)}$ is unchanged when $x$ and $y$ swap roles. A common mistake is false, reasoning that KL is not symmetric: true, but here both arguments of the KL ($P_{XY}$ and $P_X P_Y$) are themselves symmetric in $X$ and $Y$.

</details>

**Problem 7.** Prove that $I(X; Y) = H(X) + H(Y) - H(X, Y)$, where $I(X; Y) = \sum_{x, y} P(x, y) \log \frac{P(x, y)}{P(x) P(y)}$, and that $I(X; Y) \ge 0$ with equality exactly when $X$ and $Y$ are independent. You may use Gibbs' inequality ([Problem set A](90-problem-set-a.md), Problem 9). `[proof]`

<details><summary>Answer</summary>

**Proof.** Split the log, $\log\frac{P(x, y)}{P(x)P(y)} = \log P(x, y) - \log P(x) - \log P(y)$, and sum each term against $P(x, y)$ over the pairs with $P(x, y) > 0$:

$$I(X; Y) = \sum_{x, y} P(x, y)\log P(x, y) - \sum_{x, y} P(x, y)\log P(x) - \sum_{x, y} P(x, y)\log P(y).$$

The first sum is $-H(X, Y)$. In the second, $\log P(x)$ does not depend on $y$ and $\sum_y P(x, y) = P(x)$, so it is $-\sum_x P(x)\log P(x) = H(X)$ with the minus sign in front; the third is $H(Y)$ likewise. So $I(X; Y) = H(X) + H(Y) - H(X, Y)$.

For non-negativity, the definition is exactly $D_{KL}(P_{XY} \,\Vert\, P_X P_Y)$, and wherever $P(x, y) > 0$ both marginals are positive, so Gibbs' inequality ([Problem set A](90-problem-set-a.md), Problem 9) applies: $I(X; Y) \ge 0$, with equality exactly when $P(x, y) = P(x)P(y)$ for every $x, y$, which is the definition of independence. $\blacksquare$

</details>

### Maximum entropy

**Problem 8.** A softmax at temperature $T$ turns logits $z$ into $p_i = e^{z_i / T} / \sum_j e^{z_j / T}$. Let $z = (0, \ln 2, \ln 3)$. (a) Give $p_3$ at $T = 1$. (b) Give $p_3$ at $T = 1/2$. `[number]` (c) As $T \to \infty$, the distribution tends to: (a) the one-hot vector on the largest logit, (b) the uniform distribution, (c) the distribution at $T = 1$. `[choice]`

<details><summary>Answer</summary>

**(a) $1/2$, (b) $9/14$, (c) (b), the uniform distribution.**

1. (a) $e^{z} = (1, 2, 3)$, so $p = (1, 2, 3)/6$ and $p_3 = 1/2$. A common mistake is $1/3$, ignoring the logits.
2. (b) At $T = 1/2$, $e^{z/T} = e^{2z} = (1, 4, 9)$, so $p_3 = 9/14 \approx 0.643 > 1/2$: a temperature below 1 sharpens. A common mistake is to multiply by $T$ instead of dividing, giving $e^{z/2} = (1, \sqrt2, \sqrt3)$ and $p_3 = \sqrt3/(1 + \sqrt2 + \sqrt3)$, which flattens.
3. (c) As $T \to \infty$, every $z_i/T \to 0$, every weight tends to $e^0 = 1$, and $p \to (1/3, 1/3, 1/3)$. A common mistake is (a): the one-hot limit is $T \to 0$, greedy decoding.

</details>

**Problem 9.** Derive, with a Lagrange multiplier for each constraint, that among distributions $p$ on $\{1, \dots, V\}$ with a fixed mean energy $\sum_i p_i E_i = \bar{E}$, the one with maximum entropy has the form $p_i = e^{-\beta E_i} / Z(\beta)$ with $Z(\beta) = \sum_j e^{-\beta E_j}$: a softmax of the logits $-E_i$ at temperature $1/\beta$. `[proof]`

<details><summary>Answer</summary>

**Derivation.** Maximize $H(p) = -\sum_i p_i \ln p_i$ subject to $\sum_i p_i = 1$ and $\sum_i p_i E_i = \bar E$. With one multiplier per constraint, the Lagrangian is

$$\mathcal{L} = -\sum_i p_i \ln p_i - \lambda\Big(\sum_i p_i - 1\Big) - \beta\Big(\sum_i p_i E_i - \bar E\Big).$$

Setting $\partial\mathcal{L}/\partial p_i = 0$:

$$-\ln p_i - 1 - \lambda - \beta E_i = 0 \quad\Longrightarrow\quad p_i = e^{-1-\lambda}\, e^{-\beta E_i}.$$

The factor $e^{-1-\lambda}$ is the same for every $i$, so normalization fixes it: $e^{-1-\lambda} = 1/\sum_j e^{-\beta E_j} = 1/Z(\beta)$, giving $p_i = e^{-\beta E_i}/Z(\beta)$. ($\beta$ is then whatever value makes the mean energy equal $\bar E$.)

This stationary point is the maximum: $H$ is strictly concave on the simplex (its Hessian is $\mathrm{diag}(-1/p_i)$, negative definite) and the constraints are linear, so the feasible set is convex and a stationary point of the Lagrangian is the unique global maximum. Every $p_i > 0$, so the boundary $p_i = 0$, where $\ln p_i$ is not differentiable, plays no role.

With logits $z_i = -E_i$ and $T = 1/\beta$, $p_i = e^{z_i/T}/\sum_j e^{z_j/T}$: softmax at temperature $T$. $\blacksquare$

</details>

### Rate-distortion

**Problem 10.** A fair binary source ($P(1) = 1/2$) with Hamming distortion has the rate-distortion function $R(D) = 1 - H_b(D)$ bits per symbol for $0 \le D \le 1/2$, where $H_b$ is the binary entropy. Give (a) $R(1/8)$ and (b) $R(1/2)$. `[number]`

<details><summary>Answer</summary>

**(a) $\tfrac78\log_2 7 - 2 \approx 0.456$, (b) 0 bits.**

1. (a) $H_b(1/8) = \tfrac18\log_2 8 + \tfrac78\log_2\tfrac87 = \tfrac38 + \tfrac78(3 - \log_2 7) = 3 - \tfrac78\log_2 7 \approx 0.544$, so $R(1/8) = 1 - H_b(1/8) = \tfrac78\log_2 7 - 2$. Tolerating one flipped bit in eight saves more than half the rate. A common mistake is to report $H_b(1/8)$ itself; another is $7/8$, which is $1 - D$.
2. (b) $H_b(1/2) = 1$, so $R = 0$: at distortion $1/2$ the decoder can output anything (or a fixed guess) and be wrong half the time without receiving a single bit. A common mistake is 1.

</details>

**Problem 11.** A Gaussian source with variance $\sigma^2$ and squared-error distortion has $R(D) = \frac12 \log_2 (\sigma^2 / D)$ bits per sample for $0 < D \le \sigma^2$. (a) How many bits per sample are needed to reach $D = \sigma^2 / 16$? (b) Each extra bit per sample divides the smallest achievable distortion by what factor? `[number]`

<details><summary>Answer</summary>

**(a) 2 bits, (b) 4.**

1. (a) $R = \tfrac12\log_2 16 = \tfrac12 \cdot 4 = 2$. A common mistake is 4, dropping the $\tfrac12$; another is 16, the ratio itself.
2. (b) Inverting, $D = \sigma^2\, 2^{-2R}$, so $R \to R + 1$ multiplies $D$ by $2^{-2} = 1/4$. A common mistake is 2: one extra bit halves the error's standard deviation $\sqrt D$, not the squared error. In decibels, a factor 4 is $10\log_{10} 4 \approx 6$ dB, the rule the quantizers of [Quantization](../../build-an-llm/p08-inference/05-quantization.md) follow.

</details>

## Common mistakes

The wrong answers in this set fall into recognizable patterns, each tied to the problem where it appears.

| Mistake | Symptom | How to avoid it |
|---|---|---|
| Listing code lengths in an order other than the probabilities' | the right multiset, the wrong code | report each symbol's merge count in the probabilities' order (Problem 1(a)) |
| Reporting the entropy where the code length was asked | a non-integer-length code that no prefix code achieves | compute $\sum p_i \ell_i$ from the Huffman lengths (Problem 2(a)) |
| Rounding $-\log_2 p$ down for Shannon lengths | lengths that violate Kraft | use the ceiling (Problem 3(a)) |
| Trusting lengths without checking Kraft | a "prefix code" that cannot exist | compute the Kraft sum first (Problem 3(b)) |
| Mixing nats into a bits answer | mutual information off by $\ln 2$ | use $\log_2$ throughout (Problem 5(a)) |
| Clipping PMI at zero when PMI was asked | PPMI, not PMI | keep the negative value (Problem 5(c)) |
| Confusing $I(X; Y)$ with $H(X)$ | "knowing X tells everything about Y" when it does not | compute $I = H(Y) - H(Y \mid X)$ (Problems 5(a), 6(a)) |
| Multiplying logits by $T$ instead of dividing | a temperature that sharpens when it should flatten | use $e^{z_i / T}$ (Problem 8(b)) |
| Reading $T \to \infty$ as greedy | the opposite limit | take the limit of $z_i/T$, which is 0 (Problem 8(c)) |
| $H_b(D)$ for $R(D)$, or dropping the $1/2$ in the Gaussian rate | bit budgets off by a factor 2 or worse | write out $1 - H_b(D)$ and $\tfrac12\log_2(\sigma^2/D)$ in full (Problems 10, 11) |

## Where this leads

This set rests on [Problem set A](90-problem-set-a.md) for entropy, KL, the chain rule, and Gibbs' inequality, and on [Perplexity, bits per byte, NLL accumulator](02-perplexity-bits-per-byte-nll-accumulator.md), where bits per byte is a code length per byte. `mutual_information` and `pmi_matrix` in [Mutual information, PMI, and PPMI](04-mutual-information-pmi-ppmi.md) are Problems 5 to 7 as code. Looking forward in *Build Your Own LLM System*, [word2vec skip-gram with negative sampling, and the PPMI-SVD baseline](../../build-an-llm/p02-statistical-lm/03-word2vec-and-ppmi-svd.md) builds PPMI word vectors from positive PMI only; the sampler of [Sampling and logit processors](../../build-an-llm/p08-inference/01-sampling-and-logit-processors.md) divides logits by the temperature, with $T = 0$ as greedy; and the quantizers of [Quantization](../../build-an-llm/p08-inference/05-quantization.md) trade bits per weight for squared error, about a factor 4 per bit.

### Further reading

- Cover and Thomas, *Elements of Information Theory*, sections 2.4 to 2.6, 5.1 to 5.8, 10.1 to 10.3, and 12.1.
- MacKay, *Information Theory, Inference, and Learning Algorithms* (free online), chapters 5 and 8.
- Jaynes, "Information theory and statistical mechanics" (1957).

## Summary

- The entropy is the best expected code length: a Huffman code reaches it exactly for dyadic probabilities and stays within one bit of it otherwise (Problems 1, 2), and every prefix code obeys Kraft's inequality (Problems 3, 4).
- Mutual information is the expected PMI; it is symmetric, zero only for independent variables, and equals $H(X) + H(Y) - H(X, Y)$ (Problems 5 to 7).
- Softmax at temperature $T$ is the maximum-entropy distribution with a fixed expected logit; $T \to \infty$ flattens it to uniform and $T \to 0$ sharpens it to the argmax (Problems 8, 9).
- Rate-distortion gives the fewest bits per symbol for an allowed distortion; for a Gaussian, each extra bit divides the mean squared error by 4 (Problems 10, 11).
