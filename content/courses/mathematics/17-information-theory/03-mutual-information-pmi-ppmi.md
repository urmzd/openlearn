# Mutual information, PMI, and PPMI

Where do word vectors come from? Before word2vec, the answer was to count: build a table of how often each word appears near each context word, then factor it. Raw counts are dominated by frequent words ("the" co-occurs with everything), so the table first has to say not "how often" but "how much more often than chance". That ratio, in logs, is pointwise mutual information (PMI), and its average is the mutual information between a word and its context.

This matters beyond history. Levy and Goldberg showed that skip-gram with negative sampling implicitly factorizes a shifted PMI matrix, so the PPMI matrix built here is the yardstick for the neural model in [word2vec skip-gram with negative sampling, and the PPMI-SVD baseline](../../ai-engineering/08-statistical-language-models/03-word2vec-and-ppmi-svd.md) in *AI Engineering*. Get the marginals, the smoothing, or the zeros wrong here and the baseline that model is compared against is wrong too.

This section derives mutual information, PMI, context distribution smoothing and PPMI, works all of them by hand on a two-word table, and walks through the reference implementation, [`pmi.py`](../../../../course/solutions/python/tinyllm/info/pmi.py) (contract: [`pmi.pyi`](../../../../course/contracts/py/tinyllm/info/pmi.pyi)). It reuses `kl` from [Entropy, cross-entropy, KL, JS, and the k3 estimator](01-entropy-cross-entropy-and-kl.md), and reads probabilities as count ratios the way [MLE, Laplace, absolute discounting](../13-probability-statistics/03-mle-laplace-absolute-discounting.md) does. [Problem set A](04-problem-set-a.md) is useful practice with the entropy identities first.

> **Learning objectives**
> - Define mutual information as a KL divergence and derive $I(X; Y) = H(X) + H(Y) - H(X, Y)$.
> - Compute PMI, smoothed PMI and PPMI for every cell of a small co-occurrence table by hand.
> - Explain why context distribution smoothing lowers PMI for rare contexts, and why it renormalizes over contexts.
> - Implement PMI so that unseen pairs are exactly $-\infty$ and zero rows and columns produce no NaN or warnings.

## Notation

The section uses these symbols.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\operatorname{cooc}[w, c]$ | co-occurrence count of word $w$ with context $c$ | `float64[W, C]` |
| $D$ | total count, $\sum_{w, c} \operatorname{cooc}[w, c]$ | scalar |
| $\#(w)$, $\#(c)$ | row sum and column sum | scalars |
| $P(w, c) = \operatorname{cooc}[w, c] / D$ | joint probability (the MLE) | scalar |
| $P(w) = \#(w)/D$, $P(c) = \#(c)/D$ | marginal probabilities | scalars |
| $X, Y$ | two discrete random variables with joint $P(x, y)$ | |
| $H(X)$, $H(X, Y)$ | entropy and joint entropy, in nats | scalars |
| $I(X; Y)$ | mutual information, in nats | scalar $\ge 0$ |
| $\operatorname{PMI}(w, c)$ | pointwise mutual information | scalar, possibly $-\infty$ |
| $\alpha$ | context distribution smoothing exponent, $0 < \alpha \le 1$ | scalar |
| $P_\alpha(c)$ | smoothed context distribution | scalar |

## Mutual information

If $X$ and $Y$ were independent, their joint distribution would be the product of the marginals, $P(x)P(y)$. **Mutual information** measures how far the real joint is from that:

$$I(X; Y) = \sum_{x, y} P(x, y) \ln \frac{P(x, y)}{P(x) P(y)} = D_{KL}\big(P_{XY} \,\Vert\, P_X P_Y\big).$$

Splitting the logarithm gives $I(X; Y) = H(X) + H(Y) - H(X, Y) = H(Y) - H(Y \mid X)$: the reduction in uncertainty about $Y$ from learning $X$. ([Problem set B](05-problem-set-b.md) proves the identity.) Gibbs' inequality makes it non-negative, and zero exactly when $X$ and $Y$ are independent. It is symmetric, $I(X; Y) = I(Y; X)$, even though KL is not; and since conditioning can remove at most all of the uncertainty, $I(X; Y) \le \min(H(X), H(Y))$.

The implementation computes it the way the definition reads: normalize the table, take its marginals, and call `kl` on the flattened joint and the flattened outer product of the marginals. `kl` already handles $0 \ln 0 = 0$, and wherever $P(x, y) > 0$ both marginals are positive, so the product never rules out a pair the joint produces. A table of raw counts is normalized first, so counts and probabilities give the same answer.

## Pointwise mutual information

Mutual information is a single number for the whole table. For word vectors you want one number per cell, and the sum already contains it. The log ratio inside the sum, for one pair, is the **pointwise mutual information**:

$$\operatorname{PMI}(w, c) = \ln \frac{P(w, c)}{P(w) P(c)} = \ln \frac{\operatorname{cooc}[w, c] \cdot D}{\#(w)\, \#(c)} .$$

It is positive when $w$ and $c$ appear together more often than independence predicts, zero at chance, negative below chance, and $-\infty$ for a pair never seen ($\ln 0$). Mutual information is the average PMI under the joint, $I = \sum_{w, c} P(w, c) \operatorname{PMI}(w, c)$, where unseen pairs carry weight zero and contribute nothing.

## Context distribution smoothing

Plain PMI has a known bias toward rare contexts: a context seen twice, once with $w$, gets a huge ratio from tiny counts. Levy, Goldberg, and Dagan borrowed word2vec's fix, raising context counts to $\alpha = 0.75$ before normalizing **over contexts**:

$$P_\alpha(c) = \frac{\#(c)^\alpha}{\sum_{c'} \#(c')^\alpha}, \qquad \operatorname{PMI}_\alpha(w, c) = \ln P(w, c) - \ln P(w) - \ln P_\alpha(c) .$$

Since $t^\alpha$ grows slower than $t$, rare contexts get a larger share of the probability than their counts alone would give, so their PMI drops and frequent contexts' PMI rises.

Three details matter. The smoothing applies to the **context** marginal only; the word marginal keeps its plain MLE. The denominator is $\sum_{c'} \#(c')^\alpha$, not $D$: dividing by $D$ leaves numbers that do not sum to 1 and shifts every PMI by the same constant. And $\alpha = 1$ is plain PMI; the contract allows $0 < \alpha \le 1$, because at $\alpha = 0$ every context, even an unused one ($0^0 = 1$), would get the same weight.

## Positive PMI

One problem remains before the matrix can be factored. Negative PMI values are unreliable, since they need many observations to tell "below chance" from "not yet seen", and $-\infty$ cannot go into an SVD. **PPMI** keeps only the positive part:

$$\operatorname{PPMI}(w, c) = \max(\operatorname{PMI}(w, c), 0),$$

so unseen pairs and below-chance pairs become exact zeros. The result is finite, non-negative, and mostly zeros: the matrix the word2vec section factors with the SVD of [SVD, Eckart-Young low rank, and least squares](../09-linear-algebra/05-svd-low-rank-and-least-squares.md). A row or column that is entirely zero (a word never seen, a context never used) must come out as zeros too, with no divide-by-zero warnings. The implementation computes the logs under `np.errstate(divide="ignore")`, takes the marginal logs only where the marginals are positive, and writes $-\infty$ wherever the count is 0.

## Working it by hand

Here is a table with two words and three contexts:

| | pet | bark | meow | $\#(w)$ |
|---|---|---|---|---|
| cat | 2 | 0 | 2 | 4 |
| dog | 2 | 4 | 0 | 6 |
| $\#(c)$ | 4 | 4 | 2 | $D = 10$ |

Plain PMI ($\alpha = 1$) for every cell:

| Pair | $P(w, c)$ | $P(w)P(c)$ | PMI | PPMI |
|---|---|---|---|---|
| cat, pet | 0.2 | $0.4 \cdot 0.4 = 0.16$ | $\ln 1.25 = 0.2231$ | 0.2231 |
| cat, bark | 0 | 0.16 | $-\infty$ | 0 |
| cat, meow | 0.2 | $0.4 \cdot 0.2 = 0.08$ | $\ln 2.5 = 0.9163$ | 0.9163 |
| dog, pet | 0.2 | $0.6 \cdot 0.4 = 0.24$ | $\ln(5/6) = -0.1823$ | 0 |
| dog, bark | 0.4 | $0.6 \cdot 0.4 = 0.24$ | $\ln(5/3) = 0.5108$ | 0.5108 |
| dog, meow | 0 | 0.12 | $-\infty$ | 0 |

Dog and pet co-occur, but less than chance (dogs are mostly about barking here), so PPMI drops the pair.

Mutual information is the average PMI under the joint:
$0.2 \cdot 0.2231 + 0.2 \cdot 0.9163 + 0.2 \cdot (-0.1823) + 0.4 \cdot 0.5108 = 0.3958$ nats (to 16 digits, $0.3957527947852783$).

Now smooth with $\alpha = 0.75$. $4^{0.75} = 2.8284$ and $2^{0.75} = 1.6818$, so $\sum_c \#(c)^{0.75} = 2.8284 + 2.8284 + 1.6818 = 7.3386$ and $P_{0.75}(\text{meow}) = 1.6818 / 7.3386 = 0.2292$, up from 0.2. Then $\operatorname{PMI}_{0.75}(\text{cat}, \text{meow}) = \ln \frac{0.2}{0.4 \cdot 0.2292} = \ln 2.182 = 0.7801$, down from 0.9163: the rare context lost PMI. The frequent context pet went the other way, with $P_{0.75}(\text{pet}) = 0.3854 < 0.4$ and $\operatorname{PMI}_{0.75}(\text{cat}, \text{pet}) = 0.2603 > 0.2231$.

## Designing the interface

Three functions cover the section:

```python
def mutual_information(joint: ArrayLike) -> float:
    """I(X; Y) in nats = kl(joint, outer(marginals)), the table normalized first."""

def pmi_matrix(cooc: ArrayLike, cds_alpha: float = 0.75) -> NDArray:
    """ln P(w, c) - ln P(w) - ln P_alpha(c); -inf where cooc == 0."""

def ppmi(cooc: ArrayLike, cds_alpha: float = 0.75) -> NDArray:
    """max(pmi_matrix(cooc, cds_alpha), 0): finite and non-negative."""
```

`mutual_information` returns nats, as the KL divergence from the joint to the product of its marginals. `pmi_matrix` returns pointwise mutual information with context distribution smoothing, and `ppmi` its positive part.

## Building it step by step

The implementation follows from seven observations.

1. **Everything starts from one validated table.** All three functions read a 2-D, finite, non-negative count table with a positive total; anything else (a 1-D vector, a NaN, a negative count, an empty corpus) has no probability interpretation. So one helper validates and returns a float64 **copy**, which also guarantees the caller's table is never modified when the functions normalize it.
2. **Mutual information is already a KL divergence, so reuse `kl`.** $I(X; Y) = D_{KL}(P_{XY} \Vert P_X P_Y)$ is a KL between two distributions over the $W \cdot C$ pairs. Flattening the joint and the outer product of marginals turns it into exactly the 1-D case `kl` handles, zero conventions included. The only preparation is normalizing the table so that counts and probabilities give the same answer.
3. **The product never has a zero where the joint is positive.** If $P(x, y) > 0$ then both $P(x) \ge P(x, y) > 0$ and $P(y) > 0$, so $P(x)P(y) > 0$ and `kl`'s $+\infty$ branch is never reached. Unseen pairs have $P(x, y) = 0$ and contribute 0 by $0 \ln 0 = 0$.
4. **PMI is three logs, kept apart.** $\ln P(w, c) - \ln P(w) - \ln P_\alpha(c)$. Computing the logs separately, rather than forming the ratio $\frac{\operatorname{cooc} \cdot D}{\#(w)\#(c)}$, makes the one legitimate infinity ($\ln 0$ for an unseen pair) appear only in the joint term, and lets the marginal terms be guarded on their own.
5. **Smoothing touches only the context marginal, renormalized over contexts.** $P_\alpha(c) = \#(c)^\alpha / \sum_{c'} \#(c')^\alpha$, so the smoothed values sum to 1. The word marginal stays $\#(w)/D$.
6. **Unseen pairs are exactly $-\infty$, and zero rows and columns are quiet.** For a word with $\#(w) = 0$ (or a context with $\#(c) = 0$), every one of its counts is 0, so its PMI entries are $-\infty$ by the rule for unseen pairs anyway. The marginal log is therefore replaced by $\ln 1 = 0$ where the marginal is 0, purely to avoid computing $\ln 0$ and $0/0$, and the final `np.where(C > 0, ..., -inf)` writes the answer.
7. **PPMI is a clip.** $\max(\operatorname{PMI}, 0)$ maps $-\infty$ and every negative value to exactly 0, so the result is finite and non-negative by construction.

The validation helper comes first:

```python
def _table(x: ArrayLike, who: str) -> NDArray:
    """A validated float64 copy of a 2-D non-negative table with a positive sum."""
    t = np.array(x, dtype=np.float64)   # np.array, not np.asarray: always a copy,
                                        # so later in-place division is safe
    if t.ndim != 2:
        raise ValueError(f"{who}: need a 2-D table, got shape {t.shape}")
    if not np.isfinite(t).all() or (t < 0).any():
        raise ValueError(f"{who}: entries must be finite and non-negative")
    if not t.sum() > 0:                 # "not > 0" also rejects a NaN sum
        raise ValueError(f"{who}: the table sums to 0")
    return t
```

Converting to float64 here is also what makes int64 count tables give the same results as float tables.

Mutual information is then four lines:

```python
def mutual_information(joint: ArrayLike) -> float:
    P = _table(joint, "mutual_information")  # our own copy
    P /= P.sum()                 # counts or probabilities: same answer
    px = P.sum(axis=1)           # P(x), row sums
    py = P.sum(axis=0)           # P(y), column sums
    # I(X; Y) = KL(P_XY || P_X P_Y): flatten both tables into one
    # distribution over pairs. Wherever P(x, y) > 0 the product is > 0 too.
    return float(kl(P.ravel(), np.outer(px, py).ravel()))
```

`kl` uses the natural log, so the result is in nats like every other quantity in this chapter. Because $P$ and the outer product are both normalized, the result does not scale with the corpus size.

PMI carries the smoothing and the zero handling:

```python
def pmi_matrix(cooc: ArrayLike, cds_alpha: float = 0.75) -> NDArray:
    C = _table(cooc, "pmi_matrix")
    # alpha = 0 would give an unused context weight 0^0 = 1; alpha > 1 sharpens
    # instead of smoothing. Neither is context distribution smoothing.
    if not 0.0 < cds_alpha <= 1.0:
        raise ValueError(f"pmi_matrix: need 0 < cds_alpha <= 1, got {cds_alpha}")
    D = C.sum()
    nw = C.sum(axis=1, keepdims=True)  # #(w), [W, 1]
    nc = C.sum(axis=0, keepdims=True)  # #(c), [1, C]
    # Smooth the CONTEXT distribution only, and renormalize over contexts:
    # P_alpha(c) = #(c)^alpha / sum_c' #(c')^alpha.
    nca = nc**cds_alpha
    pc = nca / nca.sum()
    with np.errstate(divide="ignore"):
        # Logs of the three probabilities, kept apart: log 0 = -inf only for
        # the joint, and the marginals of a seen pair are positive.
        # Where a marginal is 0, log(1.0) = 0 is a placeholder; those entries
        # are overwritten with -inf below because their counts are all 0.
        logp = (
            np.log(C / D)
            - np.log(np.where(nw > 0, nw / D, 1.0))
            - np.log(np.where(pc > 0, pc, 1.0))
        )
    return np.where(C > 0, logp, -np.inf)   # exactly -inf for unseen pairs
```

`keepdims=True` gives the marginals shapes `[W, 1]` and `[1, C]`, so the subtraction broadcasts to the full `[W, C]` table with the word marginal along rows and the context marginal along columns; transposing them is impossible without a shape error on a non-square table. At $\alpha = 1$, `pc` is `nc / D` and the function is plain PMI.

PPMI is the clip:

```python
def ppmi(cooc: ArrayLike, cds_alpha: float = 0.75) -> NDArray:
    return np.maximum(pmi_matrix(cooc, cds_alpha), 0.0)   # -inf and negatives -> 0
```

The full solution is [`pmi.py`](../../../../course/solutions/python/tinyllm/info/pmi.py).

### Why this design

The cost is $O(WC)$ time and memory: a few passes over the table and a handful of `[W, C]` temporaries. For the word2vec section's vocabularies the dense table is the bottleneck, not the arithmetic; a production system would use sparse counts, but the formulas are the same.

Reusing `kl` instead of writing the double sum means one implementation of the zero conventions, already tested, and code that reads like the definition. Writing $\sum P \ln(P / (p_x p_y))$ again would duplicate the $0 \ln 0$ handling and invite a second, subtly different version. Using three logs instead of one ratio matters for degenerate rows: the ratio form $\ln(\operatorname{cooc} \cdot D / (\#(w)\#(c)))$ is algebraically equal, but for an all-zero row it computes $0 \cdot D / 0 = \mathrm{NaN}$, while the separated form isolates each zero to a term that can be guarded. And an exact $-\infty$ beats $\ln \varepsilon$: a finite stand-in for unseen pairs is a large negative number that leaks into any average and depends on the arbitrary $\varepsilon$. $-\infty$ is the true value, and PPMI turns it into the exact 0 the SVD wants.

One tradeoff is accepted: `pmi_matrix` returns a dense matrix full of $-\infty$, which is not directly usable downstream. That is deliberate, since `ppmi` is the downstream form and `pmi_matrix` is the honest one.

## What can go wrong

A correct implementation must reproduce the hand-worked PMI and PPMI table at $\alpha = 1$, and the mutual information $0.3957527947852783$ nats from both counts and probabilities (nats, and normalization first). It must match five golden tables (random, quarter counts, independent, one row) at $\alpha = 1, 0.75, 0.5$ against 50-digit values, which checks that the smoothing renormalizes over contexts. Mutual information must be 0 for an outer product, symmetric under transpose, and satisfy $0 \le I \le \min(H(X), H(Y))$; it must also equal $\sum P \cdot \operatorname{PMI}$ over seen pairs, which is what makes PPMI the pointwise version of $I$. Smoothing must lower the rarest context's PMI and raise the most frequent one's, which is the whole point of $\alpha = 0.75$.

At the boundaries, an all-zero row and column must give zeros in PPMI with no NaN and no warnings, and PMI must be $-\infty$ exactly where the count is 0, because the SVD in the word2vec section takes this matrix. Negative, NaN, 1-D and all-zero tables must be rejected, as must $\alpha \le 0$ or $\alpha > 1$, so caller bugs fail early. Integer counts must give float64 results equal to the float ones, and the caller's table must be unchanged, since the word2vec section reuses its count table.

The common mistakes, and how the solution avoids each:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| taking the context marginal from the rows (a transposed marginal) | wrong PMI everywhere; a shape error on non-square tables | `nw` sums `axis=1` to `[W, 1]`, `nc` sums `axis=0` to `[1, C]`; broadcasting fixes their roles |
| dividing the smoothed context counts by $D$ instead of $\sum_{c'} \#(c')^\alpha$ | every $\operatorname{PMI}_\alpha$ shifted by the same constant | `pc = nca / nca.sum()` |
| PPMI as $\lvert \operatorname{PMI} \rvert$ | below-chance pairs count as associations; unseen pairs become $+\infty$ | `np.maximum(pmi, 0.0)` |
| mutual information in bits | values $1/\ln 2 = 1.44$ times too large next to every other number in the book | `kl` uses the natural log: nats |
| KL of raw counts, not normalized | a number that scales with the corpus size | `P /= P.sum()` before the marginals |
| smoothing unseen pairs with $\ln(\varepsilon)$ instead of $-\infty$ | a finite but huge negative PMI that leaks into averages | `np.where(C > 0, logp, -np.inf)` |
| accepting $\alpha \le 0$ | unused contexts get probability through $0^0 = 1$ | rejects anything outside $0 < \alpha \le 1$ |
| normalizing the caller's table in place | the counts the word2vec section keeps become probabilities | `_table` copies with `np.array`, so `P /= ...` touches only the copy |

## Where this leads

Looking back, `mutual_information` is `kl(joint, outer(p_x, p_y))` from [Entropy, cross-entropy, KL, JS, and the k3 estimator](01-entropy-cross-entropy-and-kl.md), with its $0 \ln 0$ conventions, and $P(w, c)$, $P(w)$, $P(c)$ are the maximum-likelihood count ratios of [MLE, Laplace, absolute discounting](../13-probability-statistics/03-mle-laplace-absolute-discounting.md). Looking forward, `ppmi_svd_embeddings` in [word2vec skip-gram with negative sampling, and the PPMI-SVD baseline](../../ai-engineering/08-statistical-language-models/03-word2vec-and-ppmi-svd.md) factors `ppmi(cooc)` with the SVD, the count-based baseline for skip-gram, and [Problem set B](05-problem-set-b.md) works mutual information and PMI by hand, including the identity $I = H(X) + H(Y) - H(X, Y)$.

Several production systems build on the same ideas. Hyperwords, by Levy and Goldberg (`hyperwords/representations/explicit.py`), adds sparse co-occurrence counting with dynamic windows and subsampling, PPMI, and SVD with eigenvalue weighting. gensim's `Phrases` with the NPMI scorer (`gensim/models/phrases.py`) uses normalized PMI, $\operatorname{PMI} / (-\ln P(w, c))$ in $[-1, 1]$, to find collocations such as "new_york". scikit-learn's `mutual_info_score` (`sklearn/metrics/cluster/_supervised.py`) computes the same sum from two label arrays via a contingency table, and `adjusted_mutual_info_score` corrects for chance. And SGNS, word2vec's objective, factorizes the same matrix shifted by $\ln k$, implicitly by gradient descent (Levy and Goldberg, NeurIPS 2014).

### Further reading

- Cover and Thomas, *Elements of Information Theory*, sections 2.4 to 2.6.
- Church and Hanks, "Word association norms, mutual information, and lexicography" (1990).
- Levy, Goldberg, and Dagan, "Improving distributional similarity with lessons learned from word embeddings" (TACL 2015).
- Levy and Goldberg, "Neural word embedding as implicit matrix factorization" (NeurIPS 2014).

## Key terms

- **Mutual information**: $I(X; Y) = D_{KL}(P_{XY} \Vert P_X P_Y)$, the nats that knowing one variable saves about the other.
- **Pointwise mutual information (PMI)**: $\ln \frac{P(w, c)}{P(w)P(c)}$ for a single pair; positive above chance, $-\infty$ for unseen pairs.
- **Context distribution smoothing**: raising context counts to $\alpha < 1$ and renormalizing over contexts, which lowers PMI for rare contexts.
- **PPMI**: $\max(\operatorname{PMI}, 0)$, a finite, sparse, non-negative matrix suitable for SVD.
- **Marginal**: a row or column sum of the joint table, normalized to a distribution.

## Summary

- Mutual information $I(X; Y)$ is how many nats knowing $X$ saves about $Y$: the KL divergence from the joint distribution to the product of its marginals, zero exactly when $X$ and $Y$ are independent.
- PMI is the log ratio inside that sum for one pair, positive when the pair co-occurs more than chance; mutual information is its average under the joint.
- Context distribution smoothing raises context counts to $\alpha = 0.75$ and renormalizes **over contexts**, which takes PMI away from rare contexts.
- PPMI clips at zero, so unseen pairs ($-\infty$) and below-chance pairs become exact zeros: a finite, sparse, non-negative matrix an SVD can factor.
- Keep the three logs apart and guard zero marginals, so degenerate rows and columns produce no NaN.

## Review questions

1. Why is $P_\alpha(c)$ divided by $\sum_{c'} \#(c')^\alpha$ and not by $D$?
   <details><summary>Answer</summary>
   $\#(c)^\alpha$ no longer sums to $D$ when $\alpha < 1$, so dividing by $D$ gives numbers that do not sum to 1. The error is a common factor for every context, so it shifts every $\operatorname{PMI}_\alpha$ by the same constant, $\ln(\sum_{c'} \#(c')^\alpha / D)$, which then changes which pairs PPMI keeps.
   </details>
2. Why can `mutual_information` call `kl` without ever getting $+\infty$?
   <details><summary>Answer</summary>
   KL is infinite only where the second distribution is 0 and the first is positive. If $P(x, y) > 0$ then $P(x) \ge P(x, y) > 0$ and $P(y) \ge P(x, y) > 0$, so the product is positive wherever the joint is.
   </details>
3. The marginal logs use `np.where(nw > 0, nw / D, 1.0)`. Why is replacing a zero marginal by 1 not a bug?
   <details><summary>Answer</summary>
   A zero marginal means every count in that row (or column) is 0, so every entry it affects is an unseen pair, and the final `np.where(C > 0, logp, -np.inf)` overwrites all of them with $-\infty$. The placeholder only prevents $\ln 0$ and $0/0$ from being computed, and therefore the NaN and warnings they would bring.
   </details>
4. Why does smoothing lower PMI for rare contexts and raise it for frequent ones?
   <details><summary>Answer</summary>
   $t \mapsto t^\alpha$ with $\alpha < 1$ compresses large counts more than small ones, so after renormalizing a rare context's share $P_\alpha(c)$ is larger than $\#(c)/D$ and a frequent context's share is smaller. PMI subtracts $\ln P_\alpha(c)$, so a larger share means lower PMI. In the worked example, meow goes from 0.2 to 0.2292 and its PMI with cat falls from 0.9163 to 0.7801.
   </details>
5. Why is PPMI the max with 0 rather than the absolute value?
   <details><summary>Answer</summary>
   The point is to keep associations stronger than chance. $\lvert \operatorname{PMI} \rvert$ would turn below-chance pairs into apparent associations and unseen pairs into $+\infty$, which an SVD cannot take.
   </details>
