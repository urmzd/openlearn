# Inner products, projections, cosine similarity, and top-k

## Overview

| | |
|---|---|
| **Module** | `M03.6` · build · Python · Pass 3 · about 1 to 2 h reading |
| **Solution** | [`inner.py`](../../../../course/solutions/python/tinyllm/linalg/inner.py): `normalize(x)`, `cosine_sim(a, b)` (each norm clamped on its own), `project(x, v)` (onto the line through $v$), and `topk_cosine(query, matrix, k)` (exact top-$k$ rows by cosine, ties to the lowest index) |
| **Contract** | [`course/contracts/py/tinyllm/linalg/inner.pyi`](../../../../course/contracts/py/tinyllm/linalg/inner.pyi) |
| **Needs** | nothing to call. Reading: `M03.1` (the dot product is the inner loop of matmul), `S-M03a` |
| **Used by** | later `L2.3` word analogies and nearest neighbours, `L6.7` the zoo's word-similarity task; `ag.07` re-implements it in Go for vector retrieval |
| **Optional depth** | Strang, *Introduction to Linear Algebra*, sections 1.2 and 4.2; Mikolov, Yih, and Zweig, "Linguistic regularities in continuous space word representations" (2013) |

## Key Takeaways

- The inner product $a \cdot b = \lVert a \rVert \lVert b \rVert \cos\theta$ mixes length and angle; dividing out both lengths leaves the angle alone, which is what "similar meaning" should measure.
- Ranking by raw dot product lets long vectors win; ranking by cosine does not.
- Clamp each norm separately with $\varepsilon$: a zero vector gives 0, not NaN, and two tiny parallel vectors still give 1.
- The projection onto $v$ divides by $v \cdot v$, not by $\lVert v \rVert$, and leaves a residual orthogonal to $v$.
- Exact top-$k$ breaks ties toward the lowest index, so every run and every language returns the same list.

## How to read this chapter

Sections 1 to 3 build inner products, cosine similarity, projection, and exact top-$k$ from first principles and work them by hand. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

`L2.3` turns every word into a vector, and then every question about words becomes a question about angles: which words are closest to "king", which word completes "man is to king as woman is to ?", how well the model's similarities agree with human ratings. All of these are cosine similarities and a sorted list of the best $k$. A version that ranks by raw dot product quietly prefers frequent words (their vectors are longer); one that divides by a zero norm fills the table with NaN; one that breaks ties by whatever order the sort happens to leave makes two runs disagree. The same search later runs over document embeddings in the agent's retrieval (`ag.07`, in Go). This chapter pins the definitions down once.

## 2. Principles

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

### 2.1 Inner products and angles

The inner product of two vectors is $a \cdot b = \sum_i a_i b_i$, and $a \cdot a = \lVert a \rVert^2$. Expanding $\lVert a - b \rVert^2 = \lVert a \rVert^2 - 2\,a \cdot b + \lVert b \rVert^2$ and comparing with the law of cosines for the triangle with sides $a$, $b$, $a - b$ gives

$$a \cdot b = \lVert a \rVert \, \lVert b \rVert \cos\theta .$$

So the inner product is positive when the vectors point the same way, zero when they are perpendicular (**orthogonal**), and negative when they point apart. The **Cauchy-Schwarz inequality** $\lvert a \cdot b \rvert \le \lVert a \rVert \lVert b \rVert$ (problem 3 of [S-M03b](91-problem-set-b.md) proves it) is what makes $\cos\theta$ a number in $[-1, 1]$.

### 2.2 Cosine similarity and normalization

**Cosine similarity** divides out both lengths:

$$\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert} .$$

It depends only on directions: scaling $a$ or $b$ by a positive number leaves it unchanged, a negative one flips its sign, and $\cos(a, a) = 1$. **Normalizing** a vector, $\hat{x} = x / \lVert x \rVert$, makes its length 1, and for unit vectors cosine similarity is just the dot product, $\cos(a, b) = \hat{a} \cdot \hat{b}$. That is how a vector index stores embeddings: normalize once, then every search is one matrix product.

In code, vectors lie along an `axis` (the last by default) and the other axes broadcast: `cosine_sim(A, b)` with $A$ of shape $[7, 5]$ and $b$ of shape $[5]$ returns 7 similarities. The reduced axis is dropped.

### 2.3 Zero and tiny vectors

A zero vector has no direction, and $0 / 0$ is NaN. The contract follows PyTorch's `cosine_similarity` and clamps **each norm separately**:

$$\cos(a, b) = \frac{a \cdot b}{\max(\lVert a \rVert, \varepsilon)\,\max(\lVert b \rVert, \varepsilon)}, \qquad \varepsilon = 10^{-8} .$$

A zero vector then has similarity 0 with everything. Clamping the product instead, $\max(\lVert a \rVert \lVert b \rVert, \varepsilon)$, looks equivalent and is not: two parallel vectors of length $10^{-5}$ have a product of norms $10^{-10} < \varepsilon$, so the product clamp returns $10^{-10}/10^{-8} = 0.01$ for what is plainly the same direction, while separate clamps leave both norms alone and return 1. `normalize` uses the same rule with $\varepsilon = 10^{-12}$, so it maps the zero vector to itself.

### 2.4 Projection onto a line

The point on the line through $v \ne 0$ closest to $x$ is

$$\operatorname{proj}_v x = \frac{x \cdot v}{v \cdot v}\, v .$$

To see it, write $x = c\,v + r$ and ask that the residual $r$ be orthogonal to $v$: $0 = r \cdot v = x \cdot v - c\,(v \cdot v)$, so $c = (x \cdot v)/(v \cdot v)$. By Pythagoras, any other point $c' v$ on the line is farther: $\lVert x - c'v \rVert^2 = \lVert r \rVert^2 + (c - c')^2 \lVert v \rVert^2$. Three consequences a correct implementation must show: projecting twice changes nothing; the residual $x - \operatorname{proj}_v x$ is orthogonal to $v$; and the result does not depend on the length of $v$, because $v$ appears once on top and twice below. That last point is where the common bug lives: dividing by $\lVert v \rVert$ instead of $v \cdot v$ is correct only for unit $v$. The zero vector spans no line, so projecting onto it raises. `M03.5`'s least squares is the same idea with a whole column space in place of a line.

### 2.5 Exact top-$k$

Given a query $q$ and candidate rows $M_1, \dots, M_n$, `topk_cosine` returns the indices of the $k$ rows with the largest $\cos(q, M_i)$, in descending order of score, with the scores. Two rules make it reproducible:

- **Ties go to the lowest index** (the same rule as greedy sampling, D11). Duplicate rows, and rows that are positive multiples of each other, score exactly the same. A stable sort on $-\text{score}$ keeps tied rows in index order; sorting ascending and reversing puts the highest index first instead.
- **Identical rows get bitwise identical scores.** The solution computes each query's scores through `cosine_sim` itself, an elementwise product summed along each row; a BLAS matrix product can round equal rows differently in different blocks, which would break ties at random.

This is "exact" (brute force) search: it scores every row, $O(nd)$ per query, which is what `L2.3` and `L6.7` need for vocabularies of tens of thousands of words. Approximate indexes (IVF, HNSW) trade exactness for speed and arrive with `ag.07`.

## 3. Worked example by hand

$a = (3, 4)$ and $b = (4, 3)$.

| Quantity | Value |
|---|---|
| $a \cdot b$ | $12 + 12 = 24$ |
| $\lVert a \rVert$, $\lVert b \rVert$ | $5$, $5$ |
| $\cos(a, b)$ | $24/25 = 0.96$ |
| $\hat{a}$ | $(0.6, 0.8)$ |
| $\operatorname{proj}_b a = \frac{24}{25} (4, 3)$ | $(3.84, 2.88)$ |
| residual $a - \operatorname{proj}_b a$ | $(-0.84, 1.12)$; check: $-0.84 \cdot 4 + 1.12 \cdot 3 = -3.36 + 3.36 = 0$ |

**Top 3 for the query $q = (3, 4)$** among the rows

| Row | Vector | $q \cdot M_i$ | $\lVert M_i \rVert$ | cosine |
|---|---|---|---|---|
| 0 | $(4, 3)$ | 24 | 5 | $24/25 = 0.96$ |
| 1 | $(6, 8)$ | 50 | 10 | $50/50 = 1$ |
| 2 | $(-3, -4)$ | $-25$ | 5 | $-1$ |
| 3 | $(10, 0)$ | 30 | 10 | $30/50 = 0.6$ |
| 4 | $(0, 2)$ | 8 | 2 | $8/10 = 0.8$ |

By cosine the top 3 are rows $1, 0, 4$ with scores $1, 0.96, 0.8$. By raw dot product they would be rows $1, 3, 0$: the long vector $(10, 0)$, which points $53°$ away from $q$, would beat $(0, 2)$, only $37°$ away.

## 4. The interface

```python
def normalize(x: ArrayLike, axis: int = -1, eps: float = 1e-12) -> NDArray: ...
def cosine_sim(a: ArrayLike, b: ArrayLike, axis: int = -1, eps: float = 1e-8) -> NDArray: ...
def project(x: ArrayLike, v: ArrayLike, axis: int = -1) -> NDArray: ...
def topk_cosine(query: ArrayLike, matrix: ArrayLike, k: int) -> tuple[NDArray, NDArray]:
    """(indices int64, scores float64), [k] or [q, k], descending, ties to the lowest index."""
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Hand example, cosine and projection | unit | section 3's cosine, projection, residual, and unit vector | pins the definitions |
| Hand example, top 3 | unit | rows $1, 0, 4$ with scores $1, 0.96, 0.8$ | cosine, not dot product, ranks the neighbours of `L2.3` |
| Batches and `axis` | differential | a $[7, 5]$ batch against one vector, along the last axis and along axis 0 | the zoo scores whole vocabularies at once |
| Cosine properties | property | symmetric, scale invariant, sign flips, in $[-1, 1]$, $\cos(a, a) = 1$ | the angle and nothing else |
| Zero and tiny vectors | boundary | zero vector gives 0 (and `normalize` keeps it 0); parallel vectors of norm $10^{-5}$ give 1 | the per-norm clamp |
| Normalization | property | unit rows, rescaling recovers $x$, `axis` respected | vector indexes store unit rows |
| Projection properties | property | $P^2 = P$, residual orthogonal to $v$, independent of $v$'s length | divide by $v \cdot v$ |
| Bad projection inputs | boundary | projecting onto 0 and mismatched lengths raise | no NaN, no silent broadcast |
| Top-$k$ against a full sort | differential | 6 queries against 200 rows of mixed lengths: indices and scores equal | exact top-$k$ |
| Ties | boundary | duplicate and rescaled rows come back in index order | reproducible across runs and languages |
| Bad $k$ | boundary | $k = 0$, $k > n$, and a wrong query length raise; $k = n$ returns every row | caller bugs fail early |

## 5. The solution, step by step

### From first principles

1. Every function reduces along one axis of vectors and broadcasts over the rest (2.2). So the solution needs one helper, a norm along `axis` with `keepdims=True`, so the result divides back into the input without reshaping.
2. Cosine is $a \cdot b$ divided by both norms; to avoid $0/0$ and still score tiny parallel vectors as 1, each norm is clamped separately (2.3).
3. `normalize` is the same division with one vector, clamped at $10^{-12}$, so the zero vector maps to itself.
4. Projection needs $c = (x \cdot v)/(v \cdot v)$ (2.4). Dividing by $v \cdot v$, not $\lVert v \rVert$, is forced by the derivation; a zero $v$ has no line to project onto, so it raises.
5. Exact top-$k$ is "score every row, sort descending, keep $k$" (2.5). Reproducibility forces two more choices: the scores of equal rows must be bitwise equal (compute them elementwise, one query at a time, through `cosine_sim`), and the sort must be stable on $-\text{score}$ so ties keep index order.
6. A single query of shape `[d]` returns shape `[k]`; a batch `[q, d]` returns `[q, k]`. Treating a single query as a batch of one and stripping the axis at the end keeps one code path.

### The code, annotated

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

Projection, which divides by $v \cdot v$ and refuses the zero direction. On the worked example, $x \cdot v = 24$, $v \cdot v = 25$, and the result is $0.96\,(4, 3) = (3.84, 2.88)$:

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

Top-$k$, after validating shapes and $1 \le k \le n$:

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

For the worked query the scores are $(0.96, 1, -1, 0.6, 0.8)$; sorting $-\text{score}$ gives rows $1, 0, 4, 3, 2$, and the first three are returned.

Full solution: [`inner.py`](../../../../course/solutions/python/tinyllm/linalg/inner.py).

### Why this solution

- **Cost.** `cosine_sim`, `normalize`, and `project` are $O(d)$ per vector. `topk_cosine` is $O(nd)$ to score and $O(n \log n)$ to sort, per query, with an $[n, d]$ temporary per query rather than a $[q, n, d]$ one for the whole batch.
- **Why not `Q @ M.T` for the scores.** It is faster, but BLAS may round identical rows differently depending on how it blocks the product, which turns exact ties into random order. Tie stability across runs and across the Go port matters more here than speed.
- **Why not `argpartition`.** It is $O(n)$ instead of $O(n \log n)$, but it does not preserve the order of equal elements, so the lowest-index tie rule would need a second pass. A full stable sort is simpler and fast enough for vocabularies of tens of thousands.
- **Why sort $-\text{score}$ rather than sort ascending and reverse.** Reversing a stable ascending sort puts tied rows in *descending* index order.
- **Tradeoff accepted.** Brute force scales linearly in $n$; approximate indexes (section "Going further") are the answer when that becomes too slow.

### Explain it back

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

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. ranking by the raw dot product | frequent words (long vectors) are everyone's nearest neighbour | scores with `cosine_sim`, which divides by both norms |
| 2. ties broken toward the highest index (sort ascending, then reverse) | two equal candidates come back in the other order; the Go port disagrees | stable sort on `-scores`; see step 5 |
| 3. clamping the product of the norms | tiny but parallel vectors score 0.01 instead of 1 | clamps `nx` and `ny` separately; see step 2 |
| 4. projecting with $\lVert v \rVert$ in the denominator | correct only for unit $v$; the residual is not orthogonal | divides by `vdotv`; see step 4 |
| 5. no clamp at all | NaN for any zero vector, and NaN sorts unpredictably | `np.maximum(norm, eps)` on every norm |
| projecting onto the zero vector | NaN instead of an error | raises when `vdotv == 0` |
| returning the top-$k$ in ascending order | the worst match first | sorts $-\text{score}$, so the best match is first |
| summing over the last axis whatever `axis` says | wrong results for column-stored vectors | passes `axis` to every sum, norm, and squeeze |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M03.1` | the dot product is the inner loop of matmul; a row-major $[n, d]$ matrix keeps each candidate contiguous (reading) |
| Forward | `L2.3` | `analogy` ranks $b - a + c$ against the vocabulary with `topk_cosine`; nearest-neighbour checks of the embeddings |
| Forward | `L6.7` | the zoo's word-similarity task correlates `cosine_sim` with human ratings (Spearman) |
| Forward | `ag.07` | the Go retriever re-implements cosine and exact top-$k$ with the same tie rule (not a call site) |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `cosine_sim` | `torch.nn.functional.cosine_similarity` | the same per-norm clamp, batched on GPU | `torch/nn/functional.py`, `aten/src/ATen/native/Distance.cpp` |
| `topk_cosine` (exact) | FAISS `IndexFlatIP` over normalized rows | SIMD and BLAS inner products, a heap per query | `faiss/IndexFlat.cpp` |
| exact search | FAISS IVF and HNSW indexes | sublinear search with a recall knob (`nprobe`, `efSearch`) | `faiss/IndexIVF.cpp`, `faiss/IndexHNSW.cpp` |
| `normalize` + dot | pgvector's `<=>` cosine distance | the same search inside Postgres, with IVFFlat and HNSW indexes | `pgvector/src/vector.c` |
