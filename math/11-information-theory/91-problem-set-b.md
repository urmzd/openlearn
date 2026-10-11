<!-- ol:module S-M11b -->
# Information theory problem set, part b: coding and Huffman, mutual information, maximum entropy, rate-distortion

## Overview

| | |
|---|---|
| **Module** | `S-M11b` · solve · none · Pass 3 · 4 to 5 h |
| **You build** | answers in `solve/S-M11b.toml` (19 checked by SymPy) and 3 proofs in `solve/S-M11b/q4.md`, `q7.md`, and `q9.md` (self-graded against their rubrics) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M11b/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M11b/problems.md` and in section 4 |
| **Needs** | no module. Reading: `S-M11a` (entropy, KL, Gibbs' inequality), `M11.2` (perplexity and bits per byte), `M11.4` (mutual information and PMI), and the [Information Theory topic](README.md) |
| **Used by** | no call site (a solve set). It checks the definitions behind `M11.4` (which `L2.3` calls), the temperature semantics of `L8.1`'s sampler (the solve-only `M11.5`), the bits-per-sample view of quantization in `L8.5` (the solve-only `M11.6`), and the coding theorems behind the optional `M11.3` |
| **Milestone** | `MS-P3` (the Pass 3 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | Cover and Thomas, *Elements of Information Theory*, ch. 2.4 to 2.6, 5.1 to 5.8, 10.1 to 10.3, 12.1; MacKay, *Information Theory, Inference, and Learning Algorithms* (free), ch. 5 and 8; Jaynes, "Information theory and statistical mechanics" (1957) |

## Key Takeaways

- The entropy is the best expected code length: a Huffman code reaches it exactly for dyadic probabilities and stays within one bit of it otherwise (q1, q2), and every prefix code obeys Kraft's inequality (q3, q4).
- Mutual information is the expected PMI; it is symmetric, zero only for independent variables, and equals $H(X) + H(Y) - H(X, Y)$ (q5 to q7).
- Softmax at temperature $T$ is the maximum-entropy distribution with a fixed expected logit; $T \to \infty$ flattens it to uniform and $T \to 0$ sharpens it to the argmax (q8, q9).
- Rate-distortion gives the fewest bits per symbol for an allowed distortion; for a Gaussian, each extra bit divides the mean squared error by 4 (q10, q11).

## How to work this chapter

```bash
ol start S-M11b             # writes solve/S-M11b.toml and one file per proof
ol check S-M11b             # SymPy checks the answers, then asks each proof rubric (y/n)
ol check S-M11b --regrade   # ask the rubrics again after you change a proof
```

---

## 1. Why now

Pass 3 is where information theory becomes things your system does. `M11.2` turned your models' losses into bits per byte, which is literally a compression rate: a model with 2 bits per byte could drive an arithmetic coder that shrinks text to a quarter of its size, and this set proves why no code can beat the entropy. `M11.4` built mutual information and PMI for the word-vector baseline of `L2.3`; here you compute them by hand. Two topics have no module of their own and live only in this set: the maximum-entropy reading of softmax, which is what the temperature knob of `L8.1`'s sampler means, and rate-distortion, the theory behind `L8.5`'s quantizers trading bits for error.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $p_i$ | probability of symbol $i$ | scalar |
| $\ell_i$ | length in bits of symbol $i$'s codeword | `int` |
| $L = \sum_i p_i \ell_i$ | expected code length | bits |
| $H(p) = -\sum_i p_i \log_2 p_i$ | entropy | bits |
| $\sum_i 2^{-\ell_i}$ | Kraft sum | scalar |
| $I(X; Y)$ | mutual information | bits or nats |
| $\operatorname{PMI}(x, y) = \log \frac{P(x, y)}{P(x) P(y)}$ | pointwise mutual information | scalar |
| $z_i$, $T$ | logits and temperature | scalars |
| $E_i$, $\beta$ | energies and inverse temperature ($z_i = -E_i$, $T = 1/\beta$) | scalars |
| $D$ | allowed average distortion | scalar |
| $R(D)$ | rate-distortion function: fewest bits per symbol at distortion $D$ | bits |
| $H_b(q)$ | binary entropy $-q\log_2 q - (1-q)\log_2(1-q)$ | bits |

### 2.1 Codes, Kraft, and Huffman

A **prefix code** assigns each symbol a bit string so that no codeword is the start of another; a stream of codewords then decodes without separators. **Kraft's inequality** (q4) says any prefix code's lengths satisfy $\sum_i 2^{-\ell_i} \le 1$, and conversely any lengths satisfying it have a prefix code. Minimizing $L$ under that constraint gives $\ell_i = -\log_2 p_i$ and $L = H(p)$: no code beats the entropy. Lengths must be integers, so the **Shannon code** rounds up, $\ell_i = \lceil -\log_2 p_i \rceil$, and lands within one bit: $H \le L < H + 1$. The **Huffman code** is optimal among prefix codes: repeatedly merge the two least likely symbols (or groups) into one node; each symbol's length is the number of merges it took part in.

### 2.2 Mutual information and PMI

$I(X; Y) = \sum_{x, y} P(x, y) \log \frac{P(x, y)}{P(x)P(y)}$: the KL divergence from the joint to the product of its marginals, the expected PMI, and also $H(X) + H(Y) - H(X, Y) = H(Y) - H(Y \mid X)$. It is symmetric, non-negative, and zero exactly when $X$ and $Y$ are independent. A single pair's PMI can be negative (seen together less than chance); only the average must be non-negative.

### 2.3 Maximum entropy and temperature

Among all distributions with a given expected energy, the one with the most entropy is the **Gibbs distribution** $p_i = e^{-\beta E_i}/Z(\beta)$ (q9). With logits $z_i = -E_i$ and $T = 1/\beta$, that is softmax at temperature $T$: $p_i \propto e^{z_i / T}$. Dividing logits by $T < 1$ sharpens the distribution, $T > 1$ flattens it; as $T \to \infty$ every $z_i/T \to 0$ and the distribution becomes uniform, and as $T \to 0$ it concentrates on the largest logit, which is greedy decoding (D11 treats $T = 0$ as greedy).

### 2.4 Rate-distortion

Lossy compression asks for the fewest bits per symbol that reproduce a source within an average distortion $D$. For a fair coin with Hamming distortion (the fraction of flipped bits), $R(D) = 1 - H_b(D)$ for $0 \le D \le 1/2$: allowing errors saves exactly the entropy of the error pattern, and at $D = 1/2$ you can guess without sending anything. For a Gaussian source with variance $\sigma^2$ and squared error, $R(D) = \frac12 \log_2(\sigma^2/D)$, so $D = \sigma^2 \cdot 2^{-2R}$: every extra bit divides the error by 4, the "6 dB per bit" rule of quantization.

## 3. Worked example by hand

These are siblings of q1, q5, q8, and q10, not graded problems.

**Huffman.** Probabilities $(1/2, 1/4, 1/4)$: merge the two quarters (one node of $1/2$), then that node with the $1/2$. Lengths $(1, 2, 2)$, $L = 1/2 + 1/2 + 1/2 = 3/2$ bits, and $H = 1/2 \cdot 1 + 2 \cdot 1/4 \cdot 2 = 3/2$: dyadic probabilities are coded at exactly the entropy. Kraft sum: $1/2 + 1/4 + 1/4 = 1$.

**Mutual information.** $P(0, 0) = P(1, 1) = 1/2$, $P(0, 1) = P(1, 0) = 0$ ($Y = X$, a fair bit): the marginals are uniform, $\operatorname{PMI}(0, 0) = \log_2 \frac{1/2}{1/4} = 1$ bit, and $I = 2 \cdot \frac12 \cdot 1 = 1$ bit $= H(X)$: knowing $X$ removes all of $Y$'s uncertainty. Written in `solve/` as `1`.

**Temperature.** Logits $z = (0, \ln 3)$: at $T = 1$, $p = (1/4, 3/4)$; at $T = 1/2$, $e^{2 z} = (1, 9)$ and $p = (1/10, 9/10)$, sharper.

**Rate-distortion.** A fair coin at $D = 1/4$: $H_b(1/4) = 2 - \frac34 \log_2 3$, so $R(1/4) = \frac34\log_2 3 - 1 \approx 0.19$ bits per symbol, written `3*log(3, 2)/4 - 1`. A Gaussian at $D = \sigma^2/4$: $R = \frac12\log_2 4 = 1$ bit.

## 4. The problem set

Write each answer in `solve/S-M11b.toml`; lettered parts are their own tables:

```toml
[q1.a]
answer = "[1, 2, 3, 3]"
[q5.a]
answer = "3*log(3, 2)/4 - 1"
[q8.c]
answer = "b"
[q4]
proof = "S-M11b/q4.md"
```

Bits use `log(x, 2)`; give exact values, not decimals.

<!-- ol:problems S-M11b -->

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

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Listing code lengths in an order other than the probabilities' | the right multiset, the wrong code | q1 a (canary) |
| Reporting the entropy where the code length was asked | a non-integer-length code that no prefix code achieves | q2 a (canary) |
| Rounding $-\log_2 p$ down for Shannon lengths | lengths that violate Kraft | q3 a (canary 9/5) |
| Trusting lengths without checking Kraft | a "prefix code" that cannot exist | q3 b (canary) |
| Mixing nats into a bits answer | mutual information off by $\ln 2$ | q5 a (canary in nats) |
| Clipping PMI at zero when PMI was asked | PPMI, not PMI | q5 c (canary 0) |
| Confusing $I(X; Y)$ with $H(X)$ | "knowing X tells everything about Y" when it does not | q5 a, q6 a (canaries) |
| Multiplying logits by $T$ instead of dividing | a temperature that sharpens when it should flatten | q8 b (canary) |
| Reading $T \to \infty$ as greedy | the opposite limit | q8 c (canary a) |
| $H_b(D)$ for $R(D)$, or dropping the $1/2$ in the Gaussian rate | bit budgets off by a factor 2 or worse | q10 a, q11 (canaries) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M11a` | entropy, KL, the chain rule, Gibbs' inequality |
| Back | `M11.2` | bits per byte is a code length per byte |
| Back | `M11.4` | `mutual_information` and `pmi_matrix` are q5 to q7 as code |
| Forward | `L2.3` | PPMI word vectors: positive PMI only |
| Forward | `L8.1` | the sampler divides logits by the temperature; $T = 0$ is greedy |
| Forward | `L8.5` | quantizers trade bits per weight for squared error, about a factor 4 per bit |
