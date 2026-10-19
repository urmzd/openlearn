# word2vec skip-gram with negative sampling, and the PPMI-SVD baseline

[Bengio's NPLM](02-bengio-nplm.md) learned an embedding table as a side effect of predicting the next byte, and the table is where its generalization lives: windows that share similar vectors share predictions. word2vec isolates that table. It throws away the language model and keeps only the question "which words appear near which?", answered at a cost so low that it trained on billions of words in 2013.

The result is the classic demonstration that meaning can be read off co-occurrence. On the story corpus, animals end up near animals and foods near foods, and the vectors rank 611 gold word pairs with a Spearman correlation around 0.73. The PPMI-SVD baseline gets about 0.78 on the same pairs with no training at all, which is the second lesson of this section: the network was implicitly factorizing a matrix that you can build and factor directly.

You will derive skip-gram with negative sampling (SGNS) from the cost of the softmax, then the PPMI-SVD baseline from what SGNS optimizes. You will work one SGNS pair, the dynamic windows, the co-occurrence counts, and an analogy by hand, and then walk through a solution that needs no autograd at all. From *Mathematics for Computing* the section uses [the alias method](../../mathematics-for-computing/08-probability-statistics/02-categorical-sampling.md) (`AliasTable`, the negative sampler), [SVD and low rank](../../mathematics-for-computing/04-linear-algebra/05-svd-low-rank-and-least-squares.md) (`svd`), [inner products, cosine similarity, and top-k](../../mathematics-for-computing/04-linear-algebra/06-inner-products-projections-cosine-and-top-k.md) (`normalize`, `cosine_sim`, `topk_cosine`), [mutual information, PMI, and PPMI](../../mathematics-for-computing/12-information-theory/03-mutual-information-pmi-ppmi.md) (`ppmi`), and [PCG32](../../mathematics-for-computing/07-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md). The problem sets on [information theory, part b](../../mathematics-for-computing/12-information-theory/05-problem-set-b.md) and [linear algebra, part b](../../mathematics-for-computing/04-linear-algebra/08-problem-set-b.md) are good preparation.

The reference solution is [`word2vec.py`](../../../../course/solutions/python/tinyllm/lm/word2vec.py): `sgns_loss` (loss and hand-derived gradients); `SkipGramNS` with `set_counts`, `noise_probs`, `keep_probs`, `pairs`, `negatives`, `step` (manual sparse SGD), `fit`, and `embeddings`; `cooccurrence` and `ppmi_svd_embeddings`; and the evaluation helpers `word_vocab`, `analogy`, `spearman`, and `word_similarity`. Its contract is [`word2vec.pyi`](../../../../course/contracts/py/tinyllm/lm/word2vec.pyi) and its tests are [`test_word2vec.py`](../../../../course/solutions/learner-tests/L2.3/python/tests/l2-3-word2vec/test_word2vec.py). word2vec has no CLI step of its own; its call site is the model zoo.

> **Learning objectives**
> - Derive the SGNS loss and its three gradients by hand, and explain why negative sampling replaces the softmax.
> - State the noise distribution and the subsampling rule exactly, and explain why subsampling happens before windowing.
> - Implement a sparse SGD step that correctly accumulates repeated rows.
> - Build PPMI-SVD embeddings and explain why they approximate what SGNS learns.
> - Evaluate embeddings by Spearman's $\rho$ with tied ranks and by 3CosAdd analogies.

## Notation

The symbols below are used throughout the section.

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
| $M$ | the PPMI matrix | `float64[V, V]` |
| $\rho$ | Spearman's rank correlation | `float` |

## Skip-gram and why the softmax is too expensive

Skip-gram predicts the words around each word. For every position, every word within the window forms a (center, context) pair, and the model wants $P(o \mid c) = \mathrm{softmax}_o(u \cdot v_c)$ to be high. That softmax runs over all $V$ output vectors for every pair, $O(V d)$ work per pair; with $V$ in the hundreds of thousands it is the whole cost. The trick that made word2vec fast is to stop asking for a full distribution at all.

## Negative sampling, the noise distribution, and subsampling

Negative sampling (Mikolov et al. 2013) replaces the softmax with a binary question: is $(c, o)$ a pair from the corpus, or a word drawn at random? The real context should score high, and $K$ random **negatives** should score low:

$$\ell(c, o, k_{1..K}) = -\log \sigma(u_o \cdot v_c) - \sum_{j=1}^{K} \log \sigma(-u_{k_j} \cdot v_c).$$

Each pair now costs $O((K + 1) d)$. The negatives come from $P_n(w) \propto \#(w)^{3/4}$. With power 1 the frequent words ("the", ".") would be nearly all the negatives; the 0.75 power flattens the distribution and lifts rare words, which Mikolov et al. found clearly better. Drawing from it uses the alias table, built once in `set_counts`: one uniform per draw, $O(1)$ each.

Frequent words have a second problem. They carry little information per occurrence ("the" near "cat" says almost nothing) and dominate the pairs. **Subsampling** keeps each token of $w$ with probability

$$p_{\text{keep}}(w) = \min\left(1, \sqrt{t / f(w)}\right),$$

where $f(w)$ is the **frequency**, not the count: with a count, the threshold $t$ would mean something different on every corpus size. Subsampling happens before the windows are cut, so removing a "the" makes its neighbours adjacent. It widens the effective window around frequent words, which is part of why it helps.

The window itself is **dynamic**. Each center draws $r = 1 + \text{below}(L)$ and pairs with the kept tokens within distance $r$. Near words are in more windows than far ones: a distance-1 neighbour always pairs, a distance-$L$ neighbour with probability $1/L$. This is word2vec.c's way of weighting closeness.

## The gradient by hand, and sparse SGD

With $\frac{d}{dx}[-\log \sigma(x)] = \sigma(x) - 1$ and $\frac{d}{dx}[-\log \sigma(-x)] = \sigma(x)$, the chain rule through each dot product gives

$$\frac{\partial \ell}{\partial v_c} = (\sigma(s) - 1)\, u_o + \sum_j \sigma(t_j)\, u_{k_j}, \qquad \frac{\partial \ell}{\partial u_o} = (\sigma(s) - 1)\, v_c, \qquad \frac{\partial \ell}{\partial u_{k_j}} = \sigma(t_j)\, v_c.$$

That is all of backpropagation for this model, so word2vec needs no autograd. A step on a batch of $B$ pairs computes these from the current rows and subtracts $\eta$ times the gradient from exactly the rows used: $B$ input rows and $B (K + 1)$ output rows, never the whole $V \times d$ tables. A row used several times (a frequent word appears in many pairs of one batch) receives the **sum** of its gradients, as if the pairs had been processed one after another from the same starting point. In numpy that is `np.add.at(W, idx, -lr * g)`; `W[idx] -= lr * g` keeps only one of the repeated writes.

The loss term $-\log \sigma(x)$ must be computed stably. Literally, $\sigma(-1000) = 1/(1 + e^{1000})$ overflows to $1 / \infty = 0$ and $-\log 0 = \infty$. Written as $\mathrm{softplus}(-x) = \log(1 + e^{-|x|}) + \max(-x, 0)$ it is exact for every finite $x$, and $\sigma$ itself is computed with $e^{-|x|}$ only.

Initialization and the schedule follow word2vec.c. The input vectors start uniform in $[-0.5/d, 0.5/d)$ and the output vectors at zero, so the first step moves only the output vectors. `fit` counts the corpus, then for each epoch draws fresh pairs and steps through them in order, with the learning rate decayed linearly over the run to $10^{-4}$ of its start. Every random choice (the initial vectors, subsampling, windows, negatives) comes from the one PCG32 given to the model, in call order, so a seed fixes the whole run. `embeddings()` returns the input table.

## PPMI-SVD: factoring the matrix directly

Levy and Goldberg (2014) showed that SGNS at its optimum sets $u_o \cdot v_c = \mathrm{PMI}(c, o) - \log K$: it implicitly factorizes a shifted PMI matrix. So build the matrix and factor it.

`cooccurrence(ids, V, L)` counts every ordered pair of positions at distance at most $L$, in both directions (unweighted, no subsampling). `ppmi` from [Mutual information, PMI, and PPMI](../../mathematics-for-computing/12-information-theory/03-mutual-information-pmi-ppmi.md) turns the counts into $\max(\mathrm{PMI}, 0)$ with context smoothing 0.75, the same 0.75 as the noise distribution, for the same reason. `svd` from [SVD, Eckart-Young low rank, and least squares](../../mathematics-for-computing/04-linear-algebra/05-svd-low-rank-and-least-squares.md) gives $M = U S V^T$, and the embeddings are $U_d \sqrt{S_d}$: the symmetric split of the singular values between words and contexts, which Levy, Goldberg, and Dagan (2015) found better than $U_d S_d$ for similarity. Each column is defined only up to sign, but cosines between rows do not depend on column signs, so two correct SVDs give the same similarities. In particular, the book's Jacobi SVD agrees with LAPACK on the cosines.

## Evaluating embeddings

With two ways to build vectors, we need a way to compare them. **Word similarity** uses a gold list of word pairs with scores (here 3: same class, 1: same part of speech, 0: otherwise). For each covered pair take the cosine similarity of the two vectors, then Spearman's $\rho$ between the cosines and the gold scores: the Pearson correlation of their **ranks**. Tied values share the average of the ranks they occupy, which matters because the gold scores have only three levels. Pairs with a word outside the vocabulary are skipped.

**Analogies** ask "a is to b as c is to ?" and answer by 3CosAdd: the word whose vector is most cosine-similar to $\hat b - \hat a + \hat c$ (hats denote unit vectors). The query words themselves are usually the nearest vectors to that point and are excluded.

Finally, the **vocabulary** must be reproducible. `word_vocab` assigns ids by count, most frequent first, ties broken by the word, so a text gives the same ids on every machine.

## Working it by hand

Start with one SGNS pair in $d = 2$: $v_c = (0.5, -1)$, $u_o = (2, 1)$, and one negative $u_k = (0, 1)$. The scores are $s = 2 \cdot 0.5 + 1 \cdot (-1) = 0$ and $t = 0 \cdot 0.5 + 1 \cdot (-1) = -1$. The loss is $-\log \sigma(0) - \log \sigma(1) = \ln 2 + \ln(1 + e^{-1}) = 0.693147 + 0.313262 = 1.006409$. The gradient weights are $\sigma(s) - 1 = -0.5$ and $\sigma(t) = \sigma(-1) = 0.268941$, so

$$\frac{\partial \ell}{\partial v_c} = -0.5 \cdot (2, 1) + 0.268941 \cdot (0, 1) = (-1, -0.231059),$$

$$\frac{\partial \ell}{\partial u_o} = -0.5 \cdot (0.5, -1) = (-0.25, 0.5), \qquad \frac{\partial \ell}{\partial u_k} = 0.268941 \cdot (0.5, -1) = (0.134471, -0.268941).$$

One step with $\eta = 0.1$ gives $v_c = (0.6, -0.976894)$, $u_o = (2.025, 0.95)$, $u_k = (-0.013447, 1.026894)$. The positive score rises to $0.286951$, the negative falls to $-1.011235$, and the loss drops to $0.870182$. The tests pin exactly this case: `sgns_loss([[0.5, -1.0]], [[2.0, 1.0]], [[[0.0, 1.0]]])` must give loss $\ln 2 + \ln(1 + e^{-1})$ and $g_v = (-1, -0.5 + 1/(1 + e))$.

Next, the pairs. Take ids 0 1 2 3, no subsampling, window $L = 2$, and a generator whose `below(2)` returns 1, 0, 0, 1, so $r = 2, 1, 1, 2$. Center 0 pairs with 1 and 2; center 1 with 0 and 2; center 2 with 1 and 3; center 3 with 1 and 2. The tests drive `pairs` with exactly such a scripted generator (a small class whose `uniform()` and `below()` return listed values), which is how they can pin the dynamic windows.

For co-occurrence, take ids 0 1 2 0 with $L = 1$. The adjacent pairs (0, 1), (1, 2), (2, 0) and their mirrors give

$$\begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 1 \\ 1 & 1 & 0 \end{pmatrix};$$

with $L = 2$ the pairs (0, 2) at positions 0 and 2 and (1, 0) at positions 1 and 3 are added in both directions.

Finally, an analogy. With man $(1, 0, 0)$, woman $(0, 1, 0)$, king $(1, 0, 1)$, queen $(0.5, 1, 0.5)$, and apple $(0, 0, -1)$, the query is $\hat{\text{king}} - \hat{\text{man}} + \hat{\text{woman}} = (-0.293, 1, 0.707)$, whose cosines are woman 0.794, queen 0.783, king 0.233. Woman, a query word, is nearest; excluding the three query words, queen wins, then apple.

## Designing the API

The surface splits into the SGNS model, the PPMI-SVD baseline, and evaluation helpers that work on any embedding matrix.

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

`sgns_loss` is a pure function of the gathered rows, which keeps the gradient testable in isolation; `step` returns the mean loss before its update.

## Building it step by step

Seven observations determine the solution. First, replace the softmax with a classifier: the full softmax costs $O(Vd)$ per pair, while asking "real pair or noise?" for the true context and $K$ noise words costs $O((K + 1) d)$ and still pushes $u_o \cdot v_c$ up and the noise scores down. Second, the gradient is closed-form: the loss is a sum of $-\log \sigma(\pm \text{score})$ terms, each with derivative $\sigma(x) - 1$ or $\sigma(x)$, and each score is a dot product, so the three gradients are products of one scalar weight with one vector and no autograd is needed. Third, only the rows used change: each pair reads one input row and $K + 1$ output rows, so an SGD step can touch exactly those rows, and because rows repeat inside a batch, the update must be a scatter-add, which equals applying each pair's gradient from the same starting point.

Fourth, two distributions shape what is learned: noise from $\#^{0.75}$ (flatter than unigram) and a keep probability from the frequency (comparable across corpus sizes), with subsampling before the windows because dropping a frequent word is meant to bring its neighbours closer. Fifth, there is one stream of randomness: init, subsampling, windows, and negatives all draw from the model's one PCG32 in call order, so a seed fixes the run and a scripted generator can pin it. Sixth, the explicit baseline follows from the theory: if SGNS at its optimum factorizes shifted PMI, then counting co-occurrences, taking positive PMI, and keeping the top singular directions builds comparable vectors with no training. Seventh, evaluate by rank, not value: gold similarity scores are coarse levels, and Spearman's $\rho$ compares rankings, with ties sharing average ranks, so the metric does not depend on the order of the pair list.

### Stable primitives

Every `exp` takes a non-positive argument, so neither function can overflow.

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

### The loss and its gradients

The loss is batched with `einsum`: `v` and `u_pos` are `[B, d]`, `u_neg` is `[B, K, d]`.

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

On the hand pair, $s = 0$, $t = -1$, $a = -0.5$, $b = 0.268941$, so $g_v = -0.5 (2, 1) + 0.268941 (0, 1) = (-1, -0.231059)$.

### Initialization

This is word2vec.c's scheme, drawn from the one generator: one `uniform()` per input entry, in row-major order, and zero output vectors.

```python
self._rng = rng if rng is not None else PCG32(0).substream("init")
u = np.array([self._rng.uniform() for _ in range(self.vocab * self.dim)], dtype=np.float64)
self.W_in = ((u - 0.5) / self.dim).reshape(self.vocab, self.dim)   # uniform in [-0.5/d, 0.5/d)
self.W_out = np.zeros((self.vocab, self.dim), dtype=np.float64)    # first step moves only W_out
```

Zero output vectors make every first score 0, so $\sigma = 1/2$ everywhere and the first gradients are well scaled. With `W_out` zero, `g_v` is zero on the first step, which is why only `W_out` moves at first.

### The two distributions

`set_counts` stores the counts and builds the alias table for $P_n$ once.

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

### Pairs

`pairs` subsamples the whole sequence first, then cuts windows over the survivors.

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

The draw order is fixed: all subsampling uniforms first (one per token, even when the keep probability is 1), then one `below(L)` per kept center. On the hand example's `0 1 2 3` with `below` returning 1, 0, 0, 1, center 0 has $r = 2$ (pairs with 1, 2), center 1 $r = 1$ (0, 2), center 2 $r = 1$ (1, 3), and center 3 $r = 2$ (1, 2).

### One step

A step draws $B K$ negatives, gathers the rows, computes the gradients from the current rows, and scatter-adds.

```python
negs = self.negatives(c.size * self.n_neg).reshape(c.size, self.n_neg)
loss, g_v, g_pos, g_neg = sgns_loss(self.W_in[c], self.W_out[o], self.W_out[negs])
# Scatter-add: a word that appears twice in the batch gets both updates.
np.add.at(self.W_in, c, -lr * g_v)
np.add.at(self.W_out, o, -lr * g_pos)
np.add.at(self.W_out, negs, -lr * g_neg)        # negs is [B, K]; g_neg is [B, K, d]
return float(loss.mean())                       # the loss before the update
```

`W_in[c]` is a copy (fancy indexing), so every gradient is computed from the rows as they were before the step. `np.add.at` then accumulates every contribution, unlike `W[idx] -= ...`, whose repeated indices overwrite each other.

### Fitting

`fit` counts, then per epoch draws fresh pairs and steps through them in order with a linearly decayed rate.

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

### PPMI-SVD

Co-occurrence is a scatter-add per distance in both directions; the embedding is $U_d \sqrt{S_d}$.

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

On `0 1 2 0` with $L = 1$, the $d = 1$ pairs are (0, 1), (1, 2), (2, 0) and their mirrors: the matrix from the hand example.

### Analogies

`analogy` computes 3CosAdd over unit vectors, asking `topk_cosine` for `k + 3` candidates so that, after removing the three query words, `k` remain.

```python
ia, ib, ic = index[a], index[b], index[c]
q = normalize(e[ib]) - normalize(e[ia]) + normalize(e[ic])   # b - a + c, each a unit vector
idx, _ = topk_cosine(q, e, min(len(vocab), k + 3))
out = [vocab[int(i)] for i in idx if int(i) not in (ia, ib, ic)]
return out[:k]
```

### Spearman with ties

Ranks come from a stable sort. Each run of equal values shares the average of the 1-based ranks it spans, and $\rho$ is the Pearson correlation of the centred ranks.

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

The full solution is [`word2vec.py`](../../../../course/solutions/python/tinyllm/lm/word2vec.py).

### Why this design

A step on $B$ pairs is $O(B K d)$ time and touches $O(B (K + 1))$ rows. An epoch over $N$ kept tokens produces about $N (L + 1)$ pairs (each center pairs with up to $2r$ neighbours, and $E[r] = (L + 1)/2$). Negatives are $O(1)$ each through the alias table. PPMI-SVD needs $O(V^2)$ memory and the SVD's cost on a dense $V \times V$ matrix, which is fine for small vocabularies and the reason SGNS wins at scale.

Manual gradients beat autograd here because the model is two gathers, dot products, and scalar weights. The gradient is three lines, and a graph would allocate per pair what a scatter-add does in place; writing it by hand also makes the sparse update explicit. Batched `np.add.at` replaces word2vec.c's per-pair updates because per-pair updates in Python are slow. The price is that pairs in one batch see the same starting rows (a mini-batch of sparse gradients) instead of each other's updates; the scatter-add keeps the sum exact.

The $U_d \sqrt{S_d}$ split shares $S$ symmetrically between word and context vectors, which keeps the top directions from dominating cosines; Levy, Goldberg, and Dagan found it better for similarity than $U_d S_d$. Excluding the query words in analogies is necessary because $\hat b - \hat a + \hat c$ is usually closest to $\hat b$ or $\hat c$ themselves; without exclusion "king - man + woman" answers "woman", as the hand example shows. The tradeoffs accepted are one process with no Hogwild threads, a dense co-occurrence matrix, and `pairs` materializing every pair of an epoch in memory.

## What can go wrong

The tests start from the hand pair's loss and three gradients, then gradcheck every gradient entry against frozen central differences on 5 pairs with 3 negatives, since the update is only as good as the hand-derived gradient. Scores of $\pm 1000$ must give finite, exact losses and gradients, because trained vectors have large dot products. $P_n \propto \#^{0.75}$ and $p_{\text{keep}} = \sqrt{t / f}$ must hold with $f$ a frequency, including unseen words. A chi-square test checks that 20 000 negatives fit $P_n$ (p > 1e-3), that the zero-count word is never drawn, and that each draw uses one uniform.

The hand example's dynamic windows, pair order, and draw counts must match, and a dropped frequent word must make its neighbours adjacent, confirming subsampling happens before windowing. `W_in` must come from one uniform per entry, `W_out` must start at zero, and `embeddings()` must be a copy of `W_in`, because the zoo scores the input table. A batch with repeated rows must equal a one-pair-at-a-time update from the same start, untouched rows must stay unchanged, and the returned loss must be the mean before the update. `fit` must equal the recipe rebuilt step by step with the linear decay. Five epochs on the story words must reach the reference's Spearman $\rho$ (mean - 3 sd over 5 seeds), so the embeddings mean something.

On the baseline side, co-occurrence must give the hand example's symmetric counts for windows 1 and 2, and PPMI-SVD cosines on every gold pair must match numpy's PMI with LAPACK's SVD, with $\rho > 0.7$. The analogy must exclude query words and return the top 2. Spearman must average ranks for ties, give $\rho = \pm 1$ for monotone data, and reject constant input; word similarity must skip unknown words and reject fewer than two covered pairs; the vocabulary must order ids by count with ties by word, with `min_count` dropping rare words. Bad sizes, counts, ids, calls before `set_counts`, and mismatched shapes all fail at the call.

The table collects the bugs these tests are designed to catch.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| the sparse update written `W[idx] -= lr * g` | a word repeated in a batch keeps one of its gradients: frequent words learn slower than they should | `np.add.at` for all three updates |
| negatives from the plain unigram distribution | "the" and "." are most of the negatives; rare words are never pushed apart | `noise_probs` raises counts to `NOISE_POWER = 0.75` |
| subsampling from the raw count instead of the frequency | almost every token is dropped (or kept), whatever the corpus size | `keep_probs` divides by `c.sum()` first |
| $-\log \sigma(x)$ computed literally | an infinite loss and NaN vectors once scores grow | `_neg_log_sigmoid` as a softplus whose `exp` argument is never positive |
| the center gradient without the negatives' term | centers are only pulled toward contexts, never pushed from noise: everything collapses together | `g_v` adds the `einsum` over the negatives |
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

> **Warning:** `W[idx] -= lr * g` is the most natural line to write and it is silently wrong whenever `idx` repeats. In SGNS it repeats constantly, because frequent words appear in many pairs of every batch and the negatives are drawn with replacement.

## Where this leads

[The model zoo](../07-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) trains SGNS and PPMI-SVD and reports `word_similarity` next to every other model family, and [the RNN language model](../04-recurrent/06-rnn-language-model-stateful-tbptt.md) can optionally initialize its embedding table from `embeddings()`. The next chapter, [recurrent models](../04-recurrent/README.md), replaces the fixed window of the NPLM with a state.

Each piece here has a production counterpart:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| batched sparse SGD in numpy | word2vec.c and gensim | lock-free multi-threaded SGD (Hogwild) over the shared tables, a precomputed sigmoid table, a 100M-entry unigram table for negatives | `word2vec.c` (`TrainModelThread`), `gensim/models/word2vec_inner.pyx` |
| word ids | fastText (Bojanowski et al. 2017) | vectors for character n-grams, so unseen and misspelled words get embeddings | `facebookresearch/fastText`, `src/fasttext.cc` |
| PPMI-SVD | GloVe (Pennington et al. 2014) | a weighted least-squares fit to log co-occurrence counts | `stanfordnlp/GloVe`, `src/glove.c` |
| static vectors | contextual embeddings ([ELMo](../04-recurrent/05-elmo-bilm-scalarmix-probes.md), [BERT](../07-objectives/02-bert-masked-lm.md)) | one vector per occurrence, not per word type | [BERT: the bidirectional encoder and masked language modeling](../07-objectives/02-bert-masked-lm.md) in this book |
| the gold pairs of a synthetic corpus | WordSim-353, SimLex-999, MEN | human similarity judgments; SimLex separates similarity from relatedness | Hill, Reichart, Korhonen (2015) |

### Further reading

- Mikolov et al., ["Distributed Representations of Words and Phrases and their Compositionality"](https://arxiv.org/abs/1310.4546) (NeurIPS 2013).
- Goldberg and Levy, "word2vec Explained" (2014, arXiv:1402.3722).
- Levy and Goldberg, "Neural Word Embedding as Implicit Matrix Factorization" (NeurIPS 2014).
- Levy, Goldberg, and Dagan, "Improving Distributional Similarity with Lessons Learned from Word Embeddings" (TACL 2015).
- Jurafsky and Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/), 3rd ed., chapter 6.

## Key terms

- **skip-gram**: a model that predicts the context words around each center word.
- **negative sampling**: replacing the softmax with logistic classification of the real context against $K$ words drawn from a noise distribution.
- **noise distribution**: $P_n(w) \propto \#(w)^{3/4}$, the distribution negatives are drawn from.
- **subsampling**: randomly dropping frequent tokens before windowing, keeping each with probability $\min(1, \sqrt{t/f})$.
- **dynamic window**: a per-center window radius $r$ drawn uniformly from $1 \dots L$, so nearer words pair more often.
- **scatter-add**: an update that accumulates every contribution to a repeated index (`np.add.at`).
- **PPMI**: positive pointwise mutual information, $\max(\mathrm{PMI}, 0)$.
- **3CosAdd**: answering an analogy with the word nearest in cosine to $\hat b - \hat a + \hat c$, excluding the query words.

## Summary

- Skip-gram with negative sampling turns "which words appear near which" into a logistic regression per (center, context) pair against a few random negatives; its gradient is three lines that can be derived by hand and checked by finite differences.
- The two distributions that make it work are exact formulas: negatives from counts to the power 0.75, and subsampling that keeps a token with probability $\sqrt{t / f}$, applied before the window is cut.
- A sparse update must add every gradient a row receives: a word that appears twice in a batch is updated twice.
- PPMI plus SVD factorizes the same co-occurrences explicitly and is a strong baseline; the book's Jacobi SVD agrees with LAPACK on the cosines.
- Embeddings are judged by how well their cosine similarities rank gold word pairs (Spearman's $\rho$), the zoo's word2vec metric.

## Review questions

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
