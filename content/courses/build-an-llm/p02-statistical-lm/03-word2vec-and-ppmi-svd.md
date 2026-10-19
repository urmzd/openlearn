# word2vec skip-gram with negative sampling, and the PPMI-SVD baseline

## Overview

| | |
|---|---|
| **Module** | `L2.3` · build · Python · Pass 3 |
| **Solution** | [`word2vec.py`](../../../../course/solutions/python/tinyllm/lm/word2vec.py): `sgns_loss` (loss and hand-derived gradients), `SkipGramNS` with `set_counts`, `noise_probs`, `keep_probs`, `pairs`, `negatives`, `step` (manual sparse SGD, no autograd), `fit`, `embeddings`; `cooccurrence` and `ppmi_svd_embeddings`; `word_vocab`, `analogy`, `spearman`, `word_similarity` · model tests [`test_word2vec.py`](../../../../course/solutions/learner-tests/L2.3/python/tests/l2-3-word2vec/test_word2vec.py) |
| **Contract** | [`course/contracts/py/tinyllm/lm/word2vec.pyi`](../../../../course/contracts/py/tinyllm/lm/word2vec.pyi) |
| **Needs** | `M07.1` `AliasTable` (the negative sampler) · `M03.5` `svd` · `M03.6` `normalize`, `cosine_sim`, `topk_cosine` · `M11.4` `ppmi` · `M06.3` `PCG32`. Reading: `L2.2` (embeddings learned inside a language model), `S-M11b`, `S-M03b` |
| **Used by** | later `L6.7`: the zoo's word-similarity task and embedding utilities; `L3.6`: an optional pretrained embedding init. word2vec has no CLI step of its own in the statistical-LM layer; its call site is the zoo |
| **Optional depth** | Mikolov et al., "Distributed Representations of Words and Phrases and their Compositionality" (NeurIPS 2013); Goldberg and Levy, "word2vec Explained" (2014, arXiv:1402.3722); Levy and Goldberg, "Neural Word Embedding as Implicit Matrix Factorization" (NeurIPS 2014); Levy, Goldberg, and Dagan, "Improving Distributional Similarity with Lessons Learned from Word Embeddings" (TACL 2015); Jurafsky and Martin, *Speech and Language Processing*, 3rd ed., ch. 6 |

## Key Takeaways

- Skip-gram with negative sampling turns "which words appear near which" into a logistic regression per (center, context) pair against a few random negatives; its gradient is three lines that can be derived by hand and checked by finite differences.
- The two distributions that make it work are exact formulas: negatives from counts to the power 0.75, and subsampling that keeps a token with probability $\sqrt{t / f}$, applied before the window is cut.
- A sparse update must add every gradient a row receives: a word that appears twice in a batch is updated twice.
- PPMI plus SVD factorizes the same co-occurrences explicitly and is a strong baseline; the course's Jacobi SVD agrees with LAPACK on the cosines.
- Embeddings are judged by how well their cosine similarities rank gold word pairs (Spearman's rho), the zoo's word2vec metric.

## How to read this chapter

Sections 1 and 2 derive SGNS from the cost of the softmax, then the PPMI-SVD baseline from what SGNS optimises. Section 3 works one SGNS pair, the dynamic windows, the co-occurrence counts, and an analogy by hand. Section 4 fixes the interface. Section 5 walks through the solution and explains why each line is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

The NPLM (`L2.2`) learned an embedding table as a side effect of predicting the next byte, and the table is where its generalization lives: windows that share similar vectors share predictions. word2vec isolates that table. It throws away the language model and keeps only the question "which words appear near which?", answered at a cost so low that it trained on billions of words in 2013. The result is the classic demonstration that meaning can be read off co-occurrence: on the Pass 3 stories, animals end up near animals and foods near foods, and the vectors rank 611 gold word pairs with a Spearman correlation around 0.73. The PPMI-SVD baseline gets about 0.78 on the same pairs with no training at all, which is the second lesson: the network was implicitly factorizing a matrix that can be built and factored directly with the `M11.4` and `M03.5` code. Later, the zoo (`L6.7`) scores every embedding table this way, and the recurrent LM (`L3.6`) can start from these vectors.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | vocabulary size; words are ids $0 \dots V-1$ | `int` |
| $d$ | embedding size (`dim`) | `int` |
| $v_w$ | input (center) vector of word $w$, row $w$ of `W_in` | `float64[d]` |
| $u_w$ | output (context) vector of word $w$, row $w$ of `W_out` | `float64[d]` |
| $(c, o)$ | a center word and one context word near it | ids |
| $k_1 \dots k_K$ | $K$ negatives (`n_neg`) drawn from $P_n$ | ids |
| $\sigma(x) = 1 / (1 + e^{-x})$ | the logistic function; $1 - \sigma(x) = \sigma(-x)$ | |
| $s = u_o \cdot v_c$, $t_j = u_{k_j} \cdot v_c$ | the positive and negative scores | scalars |
| $\#(w)$, $f(w)$ | count of $w$ in the corpus, and its frequency $\#(w) / \sum \#$ | |
| $P_n(w) \propto \#(w)^{3/4}$ | the noise distribution | `float64[V]` |
| $t$ | the subsampling threshold (`subsample_t`) | `float` |
| $L$ | the window size: each center gets $r \in \{1 \dots L\}$ | `int` |
| $\eta$ | the learning rate | `float` |
| $M$ | the PPMI matrix (`M11.4`) | `float64[V, V]` |
| $\rho$ | Spearman's rank correlation | `float` |

### 2.1 Skip-gram and why the softmax is too expensive

Skip-gram predicts the words around each word: for every position, every word within the window is a (center, context) pair, and the model wants $P(o \mid c) = \mathrm{softmax}_o(u \cdot v_c)$ to be high. That softmax runs over all $V$ output vectors for every pair, $O(V d)$ work per pair; with $V$ in the hundreds of thousands it is the whole cost.

### 2.2 Negative sampling, the noise distribution, and subsampling

Negative sampling (Mikolov et al. 2013) replaces the softmax by a binary question: is $(c, o)$ a pair from the corpus, or a word drawn at random? The real context should score high, $K$ random **negatives** should score low:

$$\ell(c, o, k_{1..K}) = -\log \sigma(u_o \cdot v_c) - \sum_{j=1}^{K} \log \sigma(-u_{k_j} \cdot v_c).$$

Each pair now costs $O((K + 1) d)$. The negatives come from $P_n(w) \propto \#(w)^{3/4}$. With power 1 the frequent words ("the", ".") would be nearly all the negatives; the 0.75 power flattens the distribution and lifts rare words, which Mikolov et al. found clearly better. Drawing from it is `M07.1`'s alias table, built once in `set_counts`: one uniform per draw, $O(1)$ each.

**Subsampling.** Frequent words carry little information per occurrence ("the" near "cat" says almost nothing) and dominate the pairs. Each token of $w$ is kept with probability

$$p_{\text{keep}}(w) = \min\left(1, \sqrt{t / f(w)}\right),$$

where $f(w)$ is the **frequency**, not the count: with a count the threshold $t$ would mean something different on every corpus size. Subsampling happens before the windows are cut, so removing a "the" makes its neighbours adjacent: it widens the effective window around frequent words, which is part of why it helps.

**Dynamic window.** Each center draws $r = 1 + \text{below}(L)$ and pairs with the kept tokens within distance $r$. Near words are in more windows than far ones: a distance-1 neighbour always pairs, a distance-$L$ neighbour with probability $1/L$. This is word2vec.c's way of weighting closeness.

### 2.3 The gradient, by hand, and sparse SGD

With $\frac{d}{dx}[-\log \sigma(x)] = \sigma(x) - 1$ and $\frac{d}{dx}[-\log \sigma(-x)] = \sigma(x)$:

$$\frac{\partial \ell}{\partial v_c} = (\sigma(s) - 1)\, u_o + \sum_j \sigma(t_j)\, u_{k_j}, \qquad \frac{\partial \ell}{\partial u_o} = (\sigma(s) - 1)\, v_c, \qquad \frac{\partial \ell}{\partial u_{k_j}} = \sigma(t_j)\, v_c.$$

That is all of backpropagation for this model, so word2vec needs no autograd. A step on a batch of $B$ pairs computes these from the current rows and subtracts $\eta$ times the gradient from exactly the rows used: $B$ input rows and $B (K + 1)$ output rows, never the whole $V \times d$ tables. A row used several times (a frequent word appears in many pairs of one batch) receives the **sum** of its gradients, as if the pairs had been processed one after another from the same starting point. In numpy that is `np.add.at(W, idx, -lr * g)`; `W[idx] -= lr * g` keeps only one of the repeated writes.

`-log σ(x)` must be computed stably. Literally, $\sigma(-1000) = 1/(1 + e^{1000})$ overflows to $1 / \infty = 0$ and $-\log 0 = \infty$. As $\mathrm{softplus}(-x) = \log(1 + e^{-|x|}) + \max(-x, 0)$ it is exact for every finite $x$, and $\sigma$ itself is computed with $e^{-|x|}$ only.

**Initialization and the schedule.** word2vec.c starts the input vectors uniform in $[-0.5/d, 0.5/d)$ and the output vectors at zero (so the first step moves only the output vectors). `fit` counts the corpus, then for each epoch draws fresh pairs and steps through them in order, with the learning rate decayed linearly over the run to $10^{-4}$ of its start, as word2vec does. Every random choice (the initial vectors, subsampling, windows, negatives) comes from the one PCG32 given to the model, in call order, so a seed fixes the whole run. `embeddings()` returns the input table.

### 2.4 PPMI-SVD

Levy and Goldberg (2014) showed that SGNS at its optimum sets $u_o \cdot v_c = \mathrm{PMI}(c, o) - \log K$: it implicitly factorizes a shifted PMI matrix. So build the matrix and factor it. `cooccurrence(ids, V, L)` counts every ordered pair of positions at distance at most $L$, in both directions (unweighted, no subsampling). `M11.4`'s `ppmi` turns the counts into $\max(\mathrm{PMI}, 0)$ with context smoothing 0.75 (the same 0.75 as the noise distribution, for the same reason). `M03.5`'s `svd` gives $M = U S V^T$, and the embeddings are $U_d \sqrt{S_d}$: the symmetric split of the singular values between words and contexts, which Levy, Goldberg, and Dagan (2015) found better than $U_d S_d$ for similarity. Each column is defined only up to sign, but cosines between rows do not depend on column signs, so two correct SVDs give the same similarities.

### 2.5 Evaluating embeddings

**Word similarity.** A gold list of word pairs with scores (here 3: same class, 1: same part of speech, 0: otherwise). For each covered pair take the cosine similarity of the two vectors, then Spearman's $\rho$ between the cosines and the gold scores: the Pearson correlation of their **ranks**. Tied values share the average of the ranks they occupy, which matters because the gold scores have only three levels. Pairs with a word outside the vocabulary are skipped.

**Analogies.** "a is to b as c is to ?" by 3CosAdd: the word whose vector is most cosine-similar to $\hat b - \hat a + \hat c$ (hats: unit vectors). The query words themselves are usually the nearest vectors to that point and are excluded.

**Vocabulary.** `word_vocab` assigns ids by count, most frequent first, ties broken by the word, so a text gives the same ids on every machine.

## 3. Worked example by hand

**One SGNS pair** in $d = 2$: $v_c = (0.5, -1)$, $u_o = (2, 1)$, one negative $u_k = (0, 1)$.

- Scores: $s = 2 \cdot 0.5 + 1 \cdot (-1) = 0$ and $t = 0 \cdot 0.5 + 1 \cdot (-1) = -1$.
- Loss: $-\log \sigma(0) - \log \sigma(1) = \ln 2 + \ln(1 + e^{-1}) = 0.693147 + 0.313262 = 1.006409$.
- Weights: $\sigma(s) - 1 = -0.5$ and $\sigma(t) = \sigma(-1) = 0.268941$.
- $\partial \ell / \partial v_c = -0.5 \cdot (2, 1) + 0.268941 \cdot (0, 1) = (-1, -0.231059)$.
- $\partial \ell / \partial u_o = -0.5 \cdot (0.5, -1) = (-0.25, 0.5)$; $\partial \ell / \partial u_k = 0.268941 \cdot (0.5, -1) = (0.134471, -0.268941)$.

These are the first hand case of section 4. One step with $\eta = 0.1$ gives $v_c = (0.6, -0.976894)$, $u_o = (2.025, 0.95)$, $u_k = (-0.013447, 1.026894)$: the positive score rises to $0.286951$, the negative falls to $-1.011235$, and the loss drops to $0.870182$.

**Pairs.** Ids 0 1 2 3, no subsampling, window $L = 2$, and the generator's `below(2)` returning 1, 0, 0, 1, so $r = 2, 1, 1, 2$. Center 0 pairs with 1 and 2; center 1 with 0 and 2; center 2 with 1 and 3; center 3 with 1 and 2.

**Co-occurrence.** Ids 0 1 2 0 with $L = 1$: the adjacent pairs (0, 1), (1, 2), (2, 0) and their mirrors give

$$\begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix};$$

with $L = 2$ the pairs (0, 2) at positions 0 and 2 and (1, 0) at positions 1 and 3 are added in both directions.

**Analogy.** With man $(1, 0, 0)$, woman $(0, 1, 0)$, king $(1, 0, 1)$, queen $(0.5, 1, 0.5)$, apple $(0, 0, -1)$: $\hat{\text{king}} - \hat{\text{man}} + \hat{\text{woman}} = (-0.293, 1, 0.707)$, whose cosines are woman 0.794, queen 0.783, king 0.233. Woman, a query word, is nearest; excluding the three query words, queen wins, then apple.

## 4. The interface

```python
def word_vocab(words, min_count=1) -> tuple[list[str], NDArray]: ...
def sgns_loss(v, u_pos, u_neg) -> tuple[NDArray, NDArray, NDArray, NDArray]: ...  # loss[B], g_v, g_pos, g_neg

class SkipGramNS:
    W_in: NDArray; W_out: NDArray                                # float64 [V, d]
    def __init__(self, vocab, dim, n_neg=5, window=5, subsample_t=1e-5, rng=None): ...
    def set_counts(self, counts) -> None: ...
    def noise_probs(self) -> NDArray: ...                        # counts^0.75, normalized
    def keep_probs(self) -> NDArray: ...                         # min(1, sqrt(t / f))
    def pairs(self, ids) -> tuple[NDArray, NDArray]: ...         # subsample, then dynamic windows
    def negatives(self, n: int) -> NDArray: ...
    def step(self, centers, contexts, lr: float) -> float: ...   # mean loss before the update
    def fit(self, ids, epochs, batch_size, lr) -> list[float]: ...
    def embeddings(self) -> NDArray: ...

def cooccurrence(ids, vocab, window) -> NDArray: ...
def ppmi_svd_embeddings(cooc, dim) -> NDArray: ...               # U_d sqrt(S_d) of ppmi(cooc)
def analogy(emb, vocab, a, b, c, k=1) -> list[str]: ...
def spearman(x, y) -> float: ...
def word_similarity(emb, vocab, pairs) -> float: ...
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Hand example of the SGNS loss | unit, smoke | section 3's loss and three gradients | the loss is right before any training |
| SGNS gradcheck | gradcheck | every gradient entry against frozen central differences, 5 pairs, 3 negatives | the update is only as good as the hand-derived gradient |
| Stable at large scores | boundary | scores of $\pm 1000$ give finite, exact losses and gradients | trained vectors have large dot products |
| Noise and keep probabilities | unit | $P_n \propto \#^{0.75}$, $p_{\text{keep}} = \sqrt{t / f}$ with $f$ a frequency, unseen words | the two distributions word2vec's quality rests on |
| Negative sampler, chi-square | statistical | 20 000 negatives fit $P_n$ (p > 1e-3), the zero-count word never drawn, one uniform per draw | negatives are drawn from the right distribution |
| Pairs of the hand example | unit | section 3's dynamic windows, pair order, draw counts | reproducible pairs from a seed |
| Subsampling drops before windowing | unit | a dropped frequent word makes its neighbours adjacent | the effective window of the real algorithm |
| Init and embeddings | unit | `W_in` from one uniform per entry, `W_out` zero, `embeddings()` a copy of `W_in` | the zoo scores the input table |
| A step is a sparse sum of gradients | unit | a batch with repeated rows against a one-pair-at-a-time update; untouched rows unchanged; mean loss before the update | the update every step relies on |
| `fit` is pairs then decayed steps | differential | `fit` against the recipe rebuilt step by step with the linear decay | one seed, one run |
| SGNS learns word classes | learning | five epochs on the Pass 3 story words reach the reference's Spearman $\rho$ (mean - 3 sd over 5 seeds) | the embeddings mean something |
| Co-occurrence hand example | unit | section 3's symmetric counts for windows 1 and 2 | the PPMI input |
| PPMI-SVD matches numpy | differential | cosines of every gold pair against numpy's PMI and LAPACK SVD; $\rho > 0.7$ | the `M11.4` and `M03.5` code on a real matrix |
| Analogy hand example | unit | 3CosAdd excludes the query words; top 2 | the classic embedding check |
| Spearman with ties | unit | average ranks for ties, $\rho = \pm 1$ for monotone data, constant input rejected | gold scores have three levels |
| Word similarity covers known words only | unit | unknown words skipped, fewer than two covered pairs rejected | the zoo scores any vocabulary |
| Vocabulary order | unit | ids by count, ties by word; `min_count` drops rare words | the same ids on every machine |
| Input validation | boundary | bad sizes, counts, ids, calls before `set_counts`, mismatched shapes | caller bugs fail at the call |

The model tests in [`test_word2vec.py`](../../../../course/solutions/learner-tests/L2.3/python/tests/l2-3-word2vec/test_word2vec.py) start from the hand pair: `sgns_loss([[0.5, -1.0]], [[2.0, 1.0]], [[[0.0, 1.0]]])` must give loss $\ln 2 + \ln(1 + e^{-1})$ and $g_v = (-1, -0.5 + 1/(1 + e))$. They drive `pairs` with a scripted generator (a small class whose `uniform()` and `below()` return listed values), which is how a test can pin the dynamic windows of section 3 exactly.

## 5. The solution, step by step

### 5.1 From first principles

1. **Replace the softmax with a classifier.** The full softmax costs $O(Vd)$ per pair. Asking "real pair or noise?" for the true context and $K$ noise words costs $O((K + 1) d)$ and still pushes $u_o \cdot v_c$ up and the noise scores down.
2. **The gradient is closed-form.** The loss is a sum of $-\log \sigma(\pm \text{score})$ terms, each with derivative $\sigma(x) - 1$ or $\sigma(x)$, and each score is a dot product. So the three gradients are outer products of one scalar weight with one vector, and no autograd is needed.
3. **Only the rows used change.** Each pair reads one input row and $K + 1$ output rows, so an SGD step can touch exactly those rows. Rows repeat inside a batch, so the update must be a scatter-add, which equals applying each pair's gradient from the same starting point.
4. **Two distributions shape what is learned.** Noise from $\#^{0.75}$ (flatter than unigram) and a keep probability from the frequency (comparable across corpus sizes). Subsampling must come before the windows, because dropping a frequent word is meant to bring its neighbours closer.
5. **One stream of randomness.** Init, subsampling, windows, and negatives all draw from the model's one PCG32 in call order, so a seed fixes the run and a scripted generator can pin it.
6. **The explicit baseline.** If SGNS at its optimum factorises shifted PMI, then counting co-occurrences, taking positive PMI, and keeping the top singular directions builds comparable vectors with no training. Cosines are invariant to column signs, so any correct SVD gives the same similarities.
7. **Evaluate by rank, not value.** Gold similarity scores are coarse levels; Spearman's $\rho$ compares rankings, with ties sharing average ranks, so the metric does not depend on the order of the pair list.

### 5.2 The code, annotated

**Stable primitives.** Every `exp` takes a non-positive argument, so neither function can overflow.

```python
def _neg_log_sigmoid(x: NDArray) -> NDArray:
    """-log sigma(x) = softplus(-x), finite for every finite x."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(-x, 0.0)

def _sigmoid(x: NDArray) -> NDArray:
    """sigma(x) without overflow: exp of a non-positive number only."""
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))   # e/(1+e) = sigma(x) for x < 0
```

At $x = -1000$, `_neg_log_sigmoid` gives $\log(1 + e^{-1000}) + 1000 = 1000$ exactly instead of $\infty$.

**The loss and its gradients.** Batched with `einsum`: `v` and `u_pos` are `[B, d]`, `u_neg` is `[B, K, d]`.

```python
s = np.einsum("bd,bd->b", up, v)              # positive scores u_o . v_c
t = np.einsum("bkd,bd->bk", un, v)            # negative scores u_k . v_c
loss = _neg_log_sigmoid(s) + _neg_log_sigmoid(-t).sum(axis=1)
a = _sigmoid(s) - 1.0                         # d loss / d s   (negative: pull together)
b = _sigmoid(t)                               # d loss / d t_j (positive: push apart)
g_v = a[:, None] * up + np.einsum("bk,bkd->bd", b, un)   # both terms: pulled AND pushed
g_pos = a[:, None] * v
g_neg = b[:, :, None] * v[:, None, :]
return loss, g_v, g_pos, g_neg
```

On section 3's pair: $s = 0$, $t = -1$, $a = -0.5$, $b = 0.268941$, so $g_v = -0.5 (2, 1) + 0.268941 (0, 1) = (-1, -0.231059)$.

**Initialisation.** word2vec.c's scheme, from the one generator: one `uniform()` per input entry, in row-major order, and zero output vectors.

```python
self._rng = rng if rng is not None else PCG32(0).substream("init")
u = np.array([self._rng.uniform() for _ in range(self.vocab * self.dim)], dtype=np.float64)
self.W_in = ((u - 0.5) / self.dim).reshape(self.vocab, self.dim)   # uniform in [-0.5/d, 0.5/d)
self.W_out = np.zeros((self.vocab, self.dim), dtype=np.float64)    # first step moves only W_out
```

Zero output vectors make every first score 0, so $\sigma = 1/2$ everywhere and the first gradients are well scaled; with `W_out` zero, `g_v` is zero on the first step, which is why only `W_out` moves at first.

**The two distributions.** `set_counts` stores the counts and builds the alias table for $P_n$ once.

```python
def noise_probs(self) -> NDArray:
    w = self._need_counts() ** NOISE_POWER        # 0.75: flatter than unigram
    return w / w.sum()

def keep_probs(self) -> NDArray:
    c = self._need_counts()
    f = c / c.sum()  # a frequency, not a raw count
    keep = np.ones(self.vocab, dtype=np.float64)
    seen = f > 0                                  # unseen words: keep 1, never divide by 0
    keep[seen] = np.minimum(1.0, np.sqrt(self.subsample_t / f[seen]))
    return keep
```

**Pairs.** Subsample the whole sequence first, then cut windows over the survivors.

```python
def pairs(self, ids: ArrayLike) -> tuple[NDArray, NDArray]:
    a = self._ids(ids, "pairs")
    keep = self.keep_probs()
    kept = [int(w) for w in a.tolist() if self._rng.uniform() < keep[w]]  # one uniform per token
    centers, contexts = [], []
    n = len(kept)
    for j in range(n):
        r = 1 + self._rng.below(self.window)     # dynamic window: near words count more
        for k in range(max(0, j - r), min(n, j + r + 1)):
            if k != j:
                centers.append(kept[j])
                contexts.append(kept[k])
    return np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)
```

The draw order is fixed: all subsampling uniforms first (one per token, even when the keep probability is 1), then one `below(L)` per kept center. On section 3's `0 1 2 3` with `below` returning 1, 0, 0, 1: center 0 has $r = 2$ (pairs with 1, 2), center 1 $r = 1$ (0, 2), center 2 $r = 1$ (1, 3), center 3 $r = 2$ (1, 2).

**One step.** Draw $B K$ negatives, gather the rows, compute the gradients from the current rows, and scatter-add.

```python
negs = self.negatives(c.size * self.n_neg).reshape(c.size, self.n_neg)
loss, g_v, g_pos, g_neg = sgns_loss(self.W_in[c], self.W_out[o], self.W_out[negs])
# Scatter-add: a word that appears twice in the batch gets both updates.
np.add.at(self.W_in, c, -lr * g_v)
np.add.at(self.W_out, o, -lr * g_pos)
np.add.at(self.W_out, negs, -lr * g_neg)        # negs is [B, K]; g_neg is [B, K, d]
return float(loss.mean())                       # the loss before the update
```

`W_in[c]` is a copy (fancy indexing), so every gradient is computed from the rows as they were before the step; `np.add.at` then accumulates every contribution, unlike `W[idx] -= ...`, whose repeated indices overwrite each other.

**`fit`.** Count, then per epoch draw fresh pairs and step through them in order with a linearly decayed rate.

```python
self.set_counts(np.bincount(a, minlength=self.vocab))
for e in range(epochs):
    cen, ctx = self.pairs(a)                      # new subsample and windows every epoch
    # ... no pairs: ValueError ...
    starts = range(0, cen.size, batch_size)
    n = len(starts)
    for b, i in enumerate(starts):
        rate = lr * max(LR_FLOOR, 1.0 - (e + b / n) / epochs)   # linear decay to 1e-4 * lr
        losses.append(self.step(cen[i : i + batch_size], ctx[i : i + batch_size], rate))
    out.append(float(np.mean(losses)))
```

`(e + b / n) / epochs` is the fraction of the run completed, so the rate falls linearly across epochs and within them, never below `LR_FLOOR` times the start. `embeddings()` returns `self.W_in.copy()`, the input table, so callers cannot mutate the model.

**PPMI-SVD.** Co-occurrence is a scatter-add per distance in both directions; the embedding is $U_d \sqrt{S_d}$.

```python
def cooccurrence(ids, vocab, window):
    # ... validation ...
    m = np.zeros((vocab, vocab), dtype=np.float64)
    for d in range(1, window + 1):  # every pair at distance d, in both directions
        np.add.at(m, (a[:-d], a[d:]), 1.0)
        np.add.at(m, (a[d:], a[:-d]), 1.0)        # mirror: the matrix is symmetric
    return m

def ppmi_svd_embeddings(cooc, dim):
    m = ppmi(cooc)                                # M11.4: max(PMI, 0), context smoothing 0.75
    # ... 1 <= dim <= V ...
    u, s, _ = svd(m)                              # M03.5, singular values descending
    return u[:, :dim] * np.sqrt(s[:dim])          # symmetric split: U_d sqrt(S_d)
```

On `0 1 2 0` with $L = 1$, the $d = 1$ pairs are (0, 1), (1, 2), (2, 0) and their mirrors: the matrix of section 3.

**Analogies.** 3CosAdd over unit vectors, asking `topk_cosine` for `k + 3` candidates so that, after removing the three query words, `k` remain.

```python
ia, ib, ic = index[a], index[b], index[c]
q = normalize(e[ib]) - normalize(e[ia]) + normalize(e[ic])   # b - a + c, each a unit vector
idx, _ = topk_cosine(q, e, min(len(vocab), k + 3))
out = [vocab[int(i)] for i in idx if int(i) not in (ia, ib, ic)]
return out[:k]
```

**Spearman with ties.** Ranks come from a stable sort; each run of equal values shares the average of the 1-based ranks it spans, and $\rho$ is the Pearson correlation of the centred ranks.

```python
def _ranks(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and x[order[j + 1]] == x[order[i]]:
            j += 1                                 # extend the run of ties
        r[order[i : j + 1]] = (i + j) / 2.0 + 1.0  # average of ranks i+1 .. j+1
        i = j + 1
    return r
```

`spearman` raises when either ranking is constant (a zero denominator), and `word_similarity` skips pairs with an unknown word and raises with fewer than two covered pairs. `word_vocab` sorts by `(-count, word)`, so ties never depend on text order.

Full solution: [`word2vec.py`](../../../../course/solutions/python/tinyllm/lm/word2vec.py).

### 5.3 Why this solution

- **Complexity.** A step on $B$ pairs is $O(B K d)$ time and touches $O(B (K + 1))$ rows. An epoch over $N$ kept tokens produces about $N (L + 1)$ pairs (each center pairs with up to $2r$ neighbours, and $E[r] = (L + 1)/2$). Negatives are $O(1)$ each through the alias table. PPMI-SVD is $O(V^2)$ memory and the SVD's cost on a dense $V \times V$ matrix, fine for small vocabularies and the reason SGNS wins at scale.
- **Why manual gradients instead of autograd?** The model is two gathers, dot products, and scalar weights; the gradient is three lines, and a graph would allocate per pair what a scatter-add does in place. It also makes the sparse update explicit.
- **Why batched `np.add.at` instead of word2vec.c's per-pair updates?** Per-pair updates in Python are slow; batching vectorises them. The price is that pairs in one batch see the same starting rows (a mini-batch of sparse gradients) instead of each other's updates; the scatter-add keeps the sum exact.
- **Why $U_d \sqrt{S_d}$?** Splitting $S$ symmetrically between word and context vectors keeps the top directions from dominating cosines; Levy, Goldberg, and Dagan found it better for similarity than $U_d S_d$.
- **Why exclude the query words in analogies?** $\hat b - \hat a + \hat c$ is usually closest to $\hat b$ or $\hat c$ themselves; without exclusion "king - man + woman" answers "woman", as section 3 shows.
- **Tradeoffs accepted.** One process and no Hogwild threads; a dense co-occurrence matrix; `pairs` materialises every pair of an epoch in memory.

### 5.4 Explain it back

1. Why does the center gradient have two terms, and what happens without the second?
   <details><summary>Answer</summary>$v_c$ appears in the positive score and in every negative score, so it receives $(\sigma(s) - 1) u_o$ (pulled toward the context) and $\sum_j \sigma(t_j) u_{k_j}$ (pushed from the noise). Without the second term nothing pushes centers apart, and all vectors drift toward each other.</details>
2. Why does `keep_probs` use $f = \# / \sum \#$ rather than the raw count?
   <details><summary>Answer</summary>The threshold $t$ (for example $10^{-5}$) is a frequency. With raw counts, $\sqrt{t / \#}$ is tiny for any word seen more than once, so almost every token is dropped, and the meaning of $t$ would change with corpus size.</details>
3. A batch contains the center "cat" twice. What does `self.W_in[c] -= lr * g_v` do that `np.add.at` does not?
   <details><summary>Answer</summary>With repeated indices, buffered assignment writes each index once, so only one of the two gradients for "cat" is applied. `np.add.at` accumulates both, matching processing the pairs one at a time from the same starting rows.</details>
4. Why are the output vectors initialised to zero and the input vectors to small uniform values, not both zero?
   <details><summary>Answer</summary>If both were zero, every score and every gradient would be zero forever (each gradient is a scalar times the other table's row). Small random input vectors break the symmetry; zero output vectors make every first prediction $\sigma(0) = 1/2$, a neutral start.</details>
5. Why do two correct SVDs give the same word similarities even if their columns differ in sign?
   <details><summary>Answer</summary>Flipping the sign of column $j$ flips coordinate $j$ of every row, which leaves all dot products and norms between rows unchanged, so cosines are identical.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. the sparse update written `W[idx] -= lr * g` | a word repeated in a batch keeps one of its gradients: frequent words learn slower than they should | `np.add.at` for all three updates |
| 2. negatives from the plain unigram distribution | "the" and "." are most of the negatives; rare words are never pushed apart | `noise_probs` raises counts to `NOISE_POWER = 0.75` |
| 3. subsampling from the raw count instead of the frequency | almost every token is dropped (or kept), whatever the corpus size | `keep_probs` divides by `c.sum()` first |
| 4. $-\log \sigma(x)$ computed literally | an infinite loss and NaN vectors once scores grow | `_neg_log_sigmoid` as a softplus whose `exp` argument is never positive |
| 5. the center gradient without the negatives' term | centers are only pulled toward contexts, never pushed from noise: everything collapses together | `g_v` adds the `einsum` over the negatives |
| the positive weight $\sigma(s)$ instead of $\sigma(s) - 1$ | the context is pushed away instead of pulled closer | `a = _sigmoid(s) - 1.0` |
| a fixed window ($r = L$ always) | far words count as much as near ones | `r = 1 + below(window)` per center |
| subsampling that keeps every token | frequent words dominate the pairs | `pairs` filters with `uniform() < keep[w]` |
| PPMI-SVD embeddings $U_d S_d$ instead of $U_d \sqrt{S_d}$ | the top directions dominate the cosines | `u[:, :dim] * np.sqrt(s[:dim])` |
| analogies that keep the query words | "king - man + woman" answers "woman" | asks for `k + 3` neighbours and filters `ia, ib, ic` |
| ties ranked by position instead of averaged | Spearman depends on the order of the pair list | `_ranks` gives each run of ties its average rank |
| `fit` without the learning-rate decay | the last epochs keep jumping around instead of settling | `rate = lr * max(LR_FLOOR, 1 - progress)` |
| `embeddings()` returning the output table | the zoo scores vectors that start at zero | returns `W_in.copy()` |
| co-occurrence counted in one direction | an asymmetric matrix and wrong marginals for PMI | two `np.add.at` calls per distance, one per direction |
| vocabulary ties broken by first appearance | ids change with the order of the text | `word_vocab` sorts by `(-count, word)` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.1` | `AliasTable` draws the negatives in $O(1)$ each |
| Back | `M03.5` | `svd` factors the PPMI matrix |
| Back | `M03.6` | `normalize`, `cosine_sim`, `topk_cosine` for similarity and analogies |
| Back | `M11.4` | `ppmi` with context smoothing builds the matrix SGNS factorizes implicitly |
| Back | `M06.3` | `PCG32`: one generator for initialization, subsampling, windows, and negatives |
| Back | `L2.2` | the embedding table a language model learns as a side effect (reading) |
| Forward | `L6.7` | the zoo trains SGNS and PPMI-SVD and reports `word_similarity` next to every other family |
| Forward | `L3.6` | the recurrent LM can initialize its embedding table from `embeddings()` (optional) |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| batched sparse SGD in numpy | word2vec.c and gensim | lock-free multi-threaded SGD (Hogwild) over the shared tables, a precomputed sigmoid table, a 100M-entry unigram table for negatives | `word2vec.c` (`TrainModelThread`), `gensim/models/word2vec_inner.pyx` |
| word ids | fastText (Bojanowski et al. 2017) | vectors for character n-grams, so unseen and misspelled words get embeddings | `facebookresearch/fastText`, `src/fasttext.cc` |
| PPMI-SVD | GloVe (Pennington et al. 2014) | a weighted least-squares fit to log co-occurrence counts | `stanfordnlp/GloVe`, `src/glove.c` |
| static vectors | contextual embeddings (ELMo, BERT; `L3.5`, `L6.2`) | one vector per occurrence, not per word type | `L6.2` in this course |
| the gold pairs of a synthetic corpus | WordSim-353, SimLex-999, MEN | human similarity judgments; SimLex separates similarity from relatedness | Hill, Reichart, Korhonen (2015) |
