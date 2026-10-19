# Inner products, projections, cosine similarity, and top-k

[word2vec skip-gram with negative sampling, and the PPMI-SVD baseline](../../build-an-llm/03-statistical-lm/03-word2vec-and-ppmi-svd.md) in *Build Your Own LLM System* turns every word into a vector, and from then on every question about words becomes a question about angles. Which words are closest to "king"? Which word completes "man is to king as woman is to ?" How well do the model's similarities agree with human ratings? All of these are cosine similarities and a sorted list of the best $k$.

Each of them is easy to get subtly wrong. A version that ranks by raw dot product quietly prefers frequent words, because their vectors are longer. One that divides by a zero norm fills the table with NaN. One that breaks ties by whatever order the sort happens to leave makes two runs disagree. The same search later runs over document embeddings in [Retrieval: BM25, Flat and IVF Vectors, RRF, and MMR](../../ai-platform-engineering/07-retrieval-and-rag/02-retrieval.md) in *AI Platform Engineering*, re-implemented in Go. This section pins the definitions down once.

Nothing needs to be built first. It helps to remember from [Vectors, matrices, row-major layout, and matmul](01-vectors-matrices-and-matmul.md) that the dot product is the inner loop of matmul, and the [linear algebra problem set, part a](07-problem-set-a.md) is useful practice. The solution is [`inner.py`](../../../../course/solutions/python/tinyllm/linalg/inner.py), with contract [`inner.pyi`](../../../../course/contracts/py/tinyllm/linalg/inner.pyi): `normalize(x)`, `cosine_sim(a, b)` with each norm clamped on its own, `project(x, v)` onto the line through $v$, and `topk_cosine(query, matrix, k)`, the exact top-$k$ rows by cosine with ties to the lowest index.

> **Learning objectives**
> - Derive $a \cdot b = \lVert a \rVert \lVert b \rVert \cos\theta$ and explain why cosine, not the raw dot product, measures similarity of direction.
> - Clamp norms so that zero vectors give 0 and tiny parallel vectors still give 1.
> - Project a vector onto a line, and show the residual is orthogonal and the result independent of the line vector's length.
> - Implement exact top-$k$ search that is reproducible across runs and languages.

## Notation

| Symbol | Meaning | Type / shape |
|---|---|---|
| $a, b, x, v$ | vectors of length $d$ | `float64[d]` |
| $a \cdot b = \sum_i a_i b_i$ | inner (dot) product | scalar |
| $\lVert a \rVert = \sqrt{a \cdot a}$ | Euclidean length | scalar |
| $\theta$ | angle between $a$ and $b$ | radians |
| $\varepsilon$ | clamp for tiny norms: $10^{-8}$ in `cosine_sim`, $10^{-12}$ in `normalize` | scalar |
| $\hat{x}$ | the unit vector $x / \lVert x \rVert$ | `float64[d]` |
| $\operatorname{proj}_v x$ | orthogonal projection of $x$ onto the line through $v$ | `float64[d]` |
| $M$ | a matrix whose $n$ rows are the candidates | `float64[n, d]` |
| $q$ | a query vector, or a batch of them | `float64[d]` or `[q, d]` |
| $k$ | how many results to return | `int` |

## Inner products and angles

The inner product of two vectors is $a \cdot b = \sum_i a_i b_i$, and $a \cdot a = \lVert a \rVert^2$. Expanding $\lVert a - b \rVert^2 = \lVert a \rVert^2 - 2\,a \cdot b + \lVert b \rVert^2$ and comparing with the law of cosines for the triangle with sides $a$, $b$, $a - b$ gives

$$a \cdot b = \lVert a \rVert \, \lVert b \rVert \cos\theta .$$

So the inner product is positive when the vectors point the same way, zero when they are perpendicular (**orthogonal**), and negative when they point apart. It mixes length and angle. The **Cauchy-Schwarz inequality** $\lvert a \cdot b \rvert \le \lVert a \rVert \lVert b \rVert$ (problem 3 of the [linear algebra problem set, part b](08-problem-set-b.md) proves it) is what makes $\cos\theta$ a number in $[-1, 1]$.

## Cosine similarity and normalization

If "similar meaning" is about direction, you want to remove length from the picture. **Cosine similarity** divides out both lengths:

$$\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert} .$$

It depends only on directions: scaling $a$ or $b$ by a positive number leaves it unchanged, a negative one flips its sign, and $\cos(a, a) = 1$. **Normalizing** a vector, $\hat{x} = x / \lVert x \rVert$, makes its length 1, and for unit vectors cosine similarity is just the dot product, $\cos(a, b) = \hat{a} \cdot \hat{b}$. That is how a vector index stores embeddings: normalize once, then every search is one matrix product.

In code, vectors lie along an `axis` (the last by default) and the other axes broadcast: `cosine_sim(A, b)` with $A$ of shape $[7, 5]$ and $b$ of shape $[5]$ returns 7 similarities. The reduced axis is dropped.

## Zero and tiny vectors

The formula breaks at one point. A zero vector has no direction, and $0 / 0$ is NaN. The contract follows PyTorch's `cosine_similarity` and clamps **each norm separately**:

$$\cos(a, b) = \frac{a \cdot b}{\max(\lVert a \rVert, \varepsilon)\,\max(\lVert b \rVert, \varepsilon)}, \qquad \varepsilon = 10^{-8} .$$

A zero vector then has similarity 0 with everything. Clamping the product instead, $\max(\lVert a \rVert \lVert b \rVert, \varepsilon)$, looks equivalent and is not. Two parallel vectors of length $10^{-5}$ have a product of norms $10^{-10} < \varepsilon$, so the product clamp returns $10^{-10}/10^{-8} = 0.01$ for what is plainly the same direction, while separate clamps leave both norms alone and return 1. `normalize` uses the same rule with $\varepsilon = 10^{-12}$, so it maps the zero vector to itself.

## Projection onto a line

Cosine tells you how aligned two vectors are; projection tells you how much of one lies along the other. The point on the line through $v \ne 0$ closest to $x$ is

$$\operatorname{proj}_v x = \frac{x \cdot v}{v \cdot v}\, v .$$

To see it, write $x = c\,v + r$ and ask that the residual $r$ be orthogonal to $v$: $0 = r \cdot v = x \cdot v - c\,(v \cdot v)$, so $c = (x \cdot v)/(v \cdot v)$. By Pythagoras, any other point $c' v$ on the line is farther: $\lVert x - c'v \rVert^2 = \lVert r \rVert^2 + (c - c')^2 \lVert v \rVert^2$.

Three consequences follow, and a correct implementation must show all of them: projecting twice changes nothing; the residual $x - \operatorname{proj}_v x$ is orthogonal to $v$; and the result does not depend on the length of $v$, because $v$ appears once on top and twice below. That last point is where the common bug lives: dividing by $\lVert v \rVert$ instead of $v \cdot v$ is correct only for unit $v$. The zero vector spans no line, so projecting onto it raises. The least squares of [SVD, Eckart-Young low rank, and least squares](05-svd-low-rank-and-least-squares.md) is the same idea with a whole column space in place of a line.

## Exact top-k

Given a query $q$ and candidate rows $M_1, \dots, M_n$, `topk_cosine` returns the indices of the $k$ rows with the largest $\cos(q, M_i)$, in descending order of score, together with the scores. Two rules make it reproducible.

First, **ties go to the lowest index**, the same rule greedy sampling uses. Duplicate rows, and rows that are positive multiples of each other, score exactly the same. A stable sort on $-\text{score}$ keeps tied rows in index order; sorting ascending and reversing puts the highest index first instead.

Second, **identical rows get bitwise identical scores**. The solution computes each query's scores through `cosine_sim` itself, an elementwise product summed along each row. A BLAS matrix product can round equal rows differently in different blocks, which would break ties at random.

This is "exact" (brute force) search: it scores every row, $O(nd)$ per query, which is what word-vector work and the evaluation zoo need for vocabularies of tens of thousands of words. Approximate indexes (IVF, HNSW) trade exactness for speed and arrive with the retrieval chapter of *AI Platform Engineering*.

## Working it by hand

Take $a = (3, 4)$ and $b = (4, 3)$. Every quantity from the section follows in a line:

| Quantity | Value |
|---|---|
| $a \cdot b$ | $12 + 12 = 24$ |
| $\lVert a \rVert$, $\lVert b \rVert$ | $5$, $5$ |
| $\cos(a, b)$ | $24/25 = 0.96$ |
| $\hat{a}$ | $(0.6, 0.8)$ |
| $\operatorname{proj}_b a = \frac{24}{25} (4, 3)$ | $(3.84, 2.88)$ |
| residual $a - \operatorname{proj}_b a$ | $(-0.84, 1.12)$; check: $-0.84 \cdot 4 + 1.12 \cdot 3 = -3.36 + 3.36 = 0$ |

Now find the top 3 for the query $q = (3, 4)$ among five rows:

| Row | Vector | $q \cdot M_i$ | $\lVert M_i \rVert$ | cosine |
|---|---|---|---|---|
| 0 | $(4, 3)$ | 24 | 5 | $24/25 = 0.96$ |
| 1 | $(6, 8)$ | 50 | 10 | $50/50 = 1$ |
| 2 | $(-3, -4)$ | $-25$ | 5 | $-1$ |
| 3 | $(10, 0)$ | 30 | 10 | $30/50 = 0.6$ |
| 4 | $(0, 2)$ | 8 | 2 | $8/10 = 0.8$ |

By cosine the top 3 are rows $1, 0, 4$ with scores $1, 0.96, 0.8$. By raw dot product they would be rows $1, 3, 0$: the long vector $(10, 0)$, which points $53°$ away from $q$, would beat $(0, 2)$, only $37°$ away. Ranking by raw dot product lets long vectors win; ranking by cosine does not.

## Designing the API

```python
def normalize(x: ArrayLike, axis: int = -1, eps: float = 1e-12) -> NDArray: ...
def cosine_sim(a: ArrayLike, b: ArrayLike, axis: int = -1, eps: float = 1e-8) -> NDArray: ...
def project(x: ArrayLike, v: ArrayLike, axis: int = -1) -> NDArray: ...
def topk_cosine(query: ArrayLike, matrix: ArrayLike, k: int) -> tuple[NDArray, NDArray]:
    """(indices int64, scores float64), [k] or [q, k], descending, ties to the lowest index."""
```

Every function reduces along one `axis` and broadcasts over the rest, and every function that takes two vectors checks that their lengths agree along that axis, because anything else would broadcast silently into a wrong answer. `topk_cosine` accepts a single query of shape `[d]`, returning shape `[k]`, or a batch `[q, d]`, returning `[q, k]`; it requires $1 \le k \le n$.

## Building it step by step

Because every function reduces along one axis of vectors and broadcasts over the rest, the solution needs one helper: a norm along `axis` with `keepdims=True`, so the result divides back into the input without reshaping. Cosine is $a \cdot b$ divided by both norms, each clamped separately to avoid $0/0$ while still scoring tiny parallel vectors as 1. `normalize` is the same division with one vector, clamped at $10^{-12}$, so the zero vector maps to itself. Projection needs $c = (x \cdot v)/(v \cdot v)$; dividing by $v \cdot v$, not $\lVert v \rVert$, is forced by the derivation, and a zero $v$ raises.

Exact top-$k$ is "score every row, sort descending, keep $k$". Reproducibility forces two more choices: the scores of equal rows must be bitwise equal, so they are computed elementwise, one query at a time, through `cosine_sim`; and the sort must be stable on $-\text{score}$ so ties keep index order. Treating a single query as a batch of one and stripping the axis at the end keeps one code path.

The two helpers:

```python
def _norm(x, axis):
    """Euclidean norm along axis, kept as a size-1 axis for broadcasting."""
    return np.sqrt(np.sum(x * x, axis=axis, keepdims=True))

def _check_lengths(a, b, axis, who):
    # vectors must have the same length along the reduced axis; anything else
    # would broadcast silently into a wrong answer
    if a.ndim == 0 or b.ndim == 0 or a.shape[axis] != b.shape[axis]:
        raise ValueError(
            f"{who}: vector lengths differ along axis {axis}: {a.shape} vs {b.shape}"
        )
```

Normalization and cosine, with the per-norm clamp:

```python
def normalize(x, axis=-1, eps=1e-12):
    v = np.asarray(x, dtype=np.float64)
    return v / np.maximum(_norm(v, axis), eps)   # zero vector: 0 / eps = 0

def cosine_sim(a, b, axis=-1, eps=1e-8):
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    _check_lengths(x, y, axis, "cosine_sim")
    # Clamp each norm on its own: two parallel vectors of norm 1e-5 keep their
    # norms and score 1; clamping the product would give 0.01.
    nx = np.maximum(_norm(x, axis), eps)
    ny = np.maximum(_norm(y, axis), eps)
    # the reduced axis is dropped from the result
    return np.squeeze(np.sum(x * y, axis=axis, keepdims=True) / (nx * ny), axis=axis)
```

Projection divides by $v \cdot v$ and refuses the zero direction. On the worked example, $x \cdot v = 24$, $v \cdot v = 25$, and the result is $0.96\,(4, 3) = (3.84, 2.88)$:

```python
def project(x, v, axis=-1):
    xv = np.asarray(x, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    _check_lengths(xv, vv, axis, "project")
    vdotv = np.sum(vv * vv, axis=axis, keepdims=True)
    if (vdotv == 0.0).any():
        raise ValueError("project: cannot project onto the zero vector")
    # Divide by v . v, not by ||v||: the coefficient multiplies v itself.
    return np.sum(xv * vv, axis=axis, keepdims=True) / vdotv * vv
```

Top-$k$ comes after validating shapes and $1 \le k \le n$:

```python
    Q = q[None, :] if q.ndim == 1 else q        # one code path for [d] and [q, d]
    # One query at a time through cosine_sim itself: identical rows get bitwise
    # identical scores (a BLAS product may round equal rows differently).
    scores = np.stack([cosine_sim(Q[i], M) for i in range(Q.shape[0])])  # [q, n]
    # Stable sort on -score: descending, and equal scores stay in row order,
    # so ties go to the lowest index.
    idx = np.argsort(-scores, axis=1, kind="stable")[:, :k].astype(np.int64)
    top = np.take_along_axis(scores, idx, axis=1)
    return (idx[0], top[0]) if q.ndim == 1 else (idx, top)
```

For the worked query the scores are $(0.96, 1, -1, 0.6, 0.8)$; sorting $-\text{score}$ gives rows $1, 0, 4, 3, 2$, and the first three are returned. The full solution is [`inner.py`](../../../../course/solutions/python/tinyllm/linalg/inner.py).

### Why this design

`cosine_sim`, `normalize`, and `project` are $O(d)$ per vector. `topk_cosine` is $O(nd)$ to score and $O(n \log n)$ to sort, per query, with an $[n, d]$ temporary per query rather than a $[q, n, d]$ one for the whole batch.

Three faster-looking alternatives were rejected. `Q @ M.T` for the scores is faster, but BLAS may round identical rows differently depending on how it blocks the product, which turns exact ties into random order; tie stability across runs and across the Go port matters more here than speed. `argpartition` is $O(n)$ instead of $O(n \log n)$, but it does not preserve the order of equal elements, so the lowest-index tie rule would need a second pass; a full stable sort is simpler and fast enough for vocabularies of tens of thousands. And sorting ascending then reversing puts tied rows in *descending* index order, which is why the solution sorts $-\text{score}$. The tradeoff accepted is that brute force scales linearly in $n$; approximate indexes, listed under further reading, are the answer when that becomes too slow.

## What can go wrong

A correct solution passes checks at each layer. The hand example must reproduce the cosine, projection, residual, and unit vector, and the top 3 must be rows $1, 0, 4$ with scores $1, 0.96, 0.8$, since cosine, not dot product, ranks word-vector neighbours. A $[7, 5]$ batch against one vector must work along the last axis and along axis 0, because the evaluation zoo scores whole vocabularies at once. Cosine must be symmetric, scale invariant, flip sign under negation, stay in $[-1, 1]$, and satisfy $\cos(a, a) = 1$: the angle and nothing else. A zero vector must give 0 (and `normalize` must keep it 0), while parallel vectors of norm $10^{-5}$ give 1. Normalized rows must have unit length, rescaling must recover $x$, and `axis` must be respected, since vector indexes store unit rows.

Projection must satisfy $P^2 = P$, give a residual orthogonal to $v$, and not depend on $v$'s length; projecting onto 0 and mismatched lengths must raise rather than produce NaN or broadcast silently. Top-$k$ must match a full sort on 6 queries against 200 rows of mixed lengths, indices and scores both. Duplicate and rescaled rows must come back in index order, so results are reproducible across runs and languages. And $k = 0$, $k > n$, and a wrong query length must raise, while $k = n$ returns every row.

The mistakes that cause these failures:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| ranking by the raw dot product | frequent words (long vectors) are everyone's nearest neighbour | scores with `cosine_sim`, which divides by both norms |
| ties broken toward the highest index (sort ascending, then reverse) | two equal candidates come back in the other order; the Go port disagrees | stable sort on `-scores` |
| clamping the product of the norms | tiny but parallel vectors score 0.01 instead of 1 | clamps `nx` and `ny` separately |
| projecting with $\lVert v \rVert$ in the denominator | correct only for unit $v$; the residual is not orthogonal | divides by `vdotv` |
| no clamp at all | NaN for any zero vector, and NaN sorts unpredictably | `np.maximum(norm, eps)` on every norm |
| projecting onto the zero vector | NaN instead of an error | raises when `vdotv == 0` |
| returning the top-$k$ in ascending order | the worst match first | sorts $-\text{score}$, so the best match is first |
| summing over the last axis whatever `axis` says | wrong results for column-stored vectors | passes `axis` to every sum, norm, and squeeze |

## Where this leads

In *Build Your Own LLM System*, `analogy` in [word2vec skip-gram with negative sampling, and the PPMI-SVD baseline](../../build-an-llm/03-statistical-lm/03-word2vec-and-ppmi-svd.md) ranks $b - a + c$ against the vocabulary with `topk_cosine` and uses it for nearest-neighbour checks of the embeddings, and the word-similarity task in [The evaluation harness and the model zoo](../../build-an-llm/07-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) correlates `cosine_sim` with human ratings (Spearman). In *AI Platform Engineering*, the Go retriever of [Retrieval: BM25, Flat and IVF Vectors, RRF, and MMR](../../ai-platform-engineering/07-retrieval-and-rag/02-retrieval.md) re-implements cosine and exact top-$k$ with the same tie rule; it is a port, not a call site. With this section the chapter's tools are complete, and the [problem sets](07-problem-set-a.md) give you practice with all of them.

### Further reading

- `torch.nn.functional.cosine_similarity` (`torch/nn/functional.py`, `aten/src/ATen/native/Distance.cpp`): the same per-norm clamp, batched on GPU.
- FAISS `IndexFlatIP` over normalized rows (`faiss/IndexFlat.cpp`): exact search with SIMD and BLAS inner products and a heap per query.
- FAISS IVF and HNSW indexes (`faiss/IndexIVF.cpp`, `faiss/IndexHNSW.cpp`): sublinear search with a recall knob (`nprobe`, `efSearch`).
- pgvector's `<=>` cosine distance (`pgvector/src/vector.c`): the same search inside Postgres, with IVFFlat and HNSW indexes.
- Strang, *Introduction to Linear Algebra*, sections 1.2 and 4.2; Mikolov, Yih, and Zweig, "Linguistic regularities in continuous space word representations" (2013).

## Key terms

- **Inner product**: $a \cdot b = \sum_i a_i b_i = \lVert a \rVert \lVert b \rVert \cos\theta$.
- **Orthogonal**: having inner product zero.
- **Cosine similarity**: the inner product divided by both lengths; depends only on direction.
- **Normalization**: dividing a vector by its length to get a unit vector.
- **Orthogonal projection**: $\operatorname{proj}_v x = \frac{x \cdot v}{v \cdot v} v$, the closest point to $x$ on the line through $v$.
- **Exact top-$k$**: scoring every candidate and returning the $k$ best, ties to the lowest index.

## Summary

- The inner product $a \cdot b = \lVert a \rVert \lVert b \rVert \cos\theta$ mixes length and angle; dividing out both lengths leaves the angle alone, which is what "similar meaning" should measure.
- Ranking by raw dot product lets long vectors win; ranking by cosine does not.
- Clamp each norm separately with $\varepsilon$: a zero vector gives 0, not NaN, and two tiny parallel vectors still give 1.
- The projection onto $v$ divides by $v \cdot v$, not by $\lVert v \rVert$, and leaves a residual orthogonal to $v$.
- Exact top-$k$ breaks ties toward the lowest index, so every run and every language returns the same list.

## Review questions

1. Why does `cosine_sim` clamp $\lVert a\rVert$ and $\lVert b\rVert$ separately instead of clamping their product?

   <details><summary>Answer</summary>

   With norms of $10^{-5}$ each, the product $10^{-10}$ is below $\varepsilon = 10^{-8}$, so a product clamp divides $10^{-10}$ by $10^{-8}$ and returns 0.01 for two parallel vectors. Separate clamps leave each norm untouched (both exceed $10^{-8}$) and return 1, while a zero vector still gives 0.

   </details>

2. Why is the projection coefficient divided by $v \cdot v$ and not $\lVert v\rVert$?

   <details><summary>Answer</summary>

   The coefficient multiplies $v$ itself, which carries one factor of $\lVert v\rVert$. The orthogonality condition gives $c = (x\cdot v)/(v\cdot v)$, so the result $(x \cdot \hat v)\hat v$ does not depend on the length of $v$. Dividing by $\lVert v\rVert$ is right only when $\lVert v\rVert = 1$.

   </details>

3. On the worked example, why would raw dot product rank row 3 above row 4?

   <details><summary>Answer</summary>

   $q \cdot (10, 0) = 30 > q \cdot (0, 2) = 8$ because $(10, 0)$ is five times longer, even though it points $53°$ away from $q$ against $37°$. Cosine divides the lengths out and gives $0.6 < 0.8$.

   </details>

4. Why does `topk_cosine` score queries one at a time through `cosine_sim` instead of a matrix product?

   <details><summary>Answer</summary>

   So that identical rows get bitwise identical scores. Elementwise products summed along each row round the same way for equal rows; a blocked BLAS product may not, and then the stable sort would order "tied" rows by rounding noise.

   </details>
