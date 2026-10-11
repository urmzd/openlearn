<!-- ol:module M03.6 -->
# Inner products, projections, cosine similarity, and top-k

## Overview

| | |
|---|---|
| **Module** | `M03.6` · build · Python · Pass 3 · 2 to 3 h |
| **You build** | `python/tinyllm/linalg/inner.py`: `normalize(x)`, `cosine_sim(a, b)` (each norm clamped on its own), `project(x, v)` (onto the line through $v$), and `topk_cosine(query, matrix, k)` (exact top-$k$ rows by cosine, ties to the lowest index) |
| **Contract** | [`course/contracts/py/tinyllm/linalg/inner.pyi`](../../course/contracts/py/tinyllm/linalg/inner.pyi) |
| **Tests** | `course/tests/M03.6/test_inner.py` (what they check: section 4) |
| **Needs** | nothing to call. Reading: `M03.1` (the dot product is the inner loop of matmul), `S-M03a` |
| **Used by** | later `L2.3` word analogies and nearest neighbours, `L6.7` the zoo's word-similarity task (each joins the registry with its batch); `ag.07` re-implements it in Go for vector retrieval |
| **Milestone** | `MS-P3` (the tokens-and-data gate) |
| **Optional depth** | Strang, *Introduction to Linear Algebra*, sections 1.2 and 4.2; Mikolov, Yih, and Zweig, "Linguistic regularities in continuous space word representations" (2013) |

## Key Takeaways

- The inner product $a \cdot b = \lVert a \rVert \lVert b \rVert \cos\theta$ mixes length and angle; dividing out both lengths leaves the angle alone, which is what "similar meaning" should measure (`test_hand_example_cosine_and_projection`, `test_cosine_properties`).
- Ranking by raw dot product lets long vectors win; ranking by cosine does not (`test_hand_example_topk`).
- Clamp each norm separately with $\varepsilon$: a zero vector gives 0, not NaN, and two tiny parallel vectors still give 1 (`test_zero_and_tiny_vectors`).
- The projection onto $v$ divides by $v \cdot v$, not by $\lVert v \rVert$, and leaves a residual orthogonal to $v$ (`test_projection_properties`).
- Exact top-$k$ breaks ties toward the lowest index, so every run and every language returns the same list (`test_topk_ties_go_to_lowest_index`).

## How to work this chapter

```bash
ol start M03.6              # stubs python/tinyllm/linalg/inner.py into your repo
ol tests M03.6              # read the test catalog first: rung R0, you write no tests here
ol check M03.6              # exit code is the verdict
ol diff  M03.6              # after passing: your code against the reference
```

---

## 1. Why now

`L2.3` turns every word into a vector, and then every question about words becomes a question about angles: which words are closest to "king", which word completes "man is to king as woman is to ?", how well the model's similarities agree with human ratings. All of these are cosine similarities and a sorted list of the best $k$. A version that ranks by raw dot product quietly prefers frequent words (their vectors are longer); one that divides by a zero norm fills the table with NaN; one that breaks ties by whatever order the sort happens to leave makes two runs disagree. The same search later runs over document embeddings in the agent's retrieval (`ag.07`, in Go). This module pins the definitions down once.

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

So the inner product is positive when the vectors point the same way, zero when they are perpendicular (**orthogonal**), and negative when they point apart. The **Cauchy-Schwarz inequality** $\lvert a \cdot b \rvert \le \lVert a \rVert \lVert b \rVert$ (S-M03b q3 asks you to prove it) is what makes $\cos\theta$ a number in $[-1, 1]$.

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

To see it, write $x = c\,v + r$ and ask that the residual $r$ be orthogonal to $v$: $0 = r \cdot v = x \cdot v - c\,(v \cdot v)$, so $c = (x \cdot v)/(v \cdot v)$. By Pythagoras, any other point $c' v$ on the line is farther: $\lVert x - c'v \rVert^2 = \lVert r \rVert^2 + (c - c')^2 \lVert v \rVert^2$. Three consequences the tests check: projecting twice changes nothing; the residual $x - \operatorname{proj}_v x$ is orthogonal to $v$; and the result does not depend on the length of $v$, because $v$ appears once on top and twice below. That last point is where the common bug lives: dividing by $\lVert v \rVert$ instead of $v \cdot v$ is correct only for unit $v$. The zero vector spans no line, so projecting onto it raises. `M03.5`'s least squares is the same idea with a whole column space in place of a line.

### 2.5 Exact top-$k$

Given a query $q$ and candidate rows $M_1, \dots, M_n$, `topk_cosine` returns the indices of the $k$ rows with the largest $\cos(q, M_i)$, in descending order of score, with the scores. Two rules make it reproducible:

- **Ties go to the lowest index** (the same rule as greedy sampling, D11). Duplicate rows, and rows that are positive multiples of each other, score exactly the same. A stable sort on $-\text{score}$ keeps tied rows in index order; sorting ascending and reversing puts the highest index first instead.
- **Identical rows get bitwise identical scores.** The reference computes each query's scores through `cosine_sim` itself, an elementwise product summed along each row; a BLAS matrix product can round equal rows differently in different blocks, which would break ties at random.

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

This is `test_hand_example_cosine_and_projection`.

**Top 3 for the query $q = (3, 4)$** among the rows

| Row | Vector | $q \cdot M_i$ | $\lVert M_i \rVert$ | cosine |
|---|---|---|---|---|
| 0 | $(4, 3)$ | 24 | 5 | $24/25 = 0.96$ |
| 1 | $(6, 8)$ | 50 | 10 | $50/50 = 1$ |
| 2 | $(-3, -4)$ | $-25$ | 5 | $-1$ |
| 3 | $(10, 0)$ | 30 | 10 | $30/50 = 0.6$ |
| 4 | $(0, 2)$ | 8 | 2 | $8/10 = 0.8$ |

By cosine the top 3 are rows $1, 0, 4$ with scores $1, 0.96, 0.8$. By raw dot product they would be rows $1, 3, 0$: the long vector $(10, 0)$, which points $53°$ away from $q$, would beat $(0, 2)$, only $37°$ away. This is `test_hand_example_topk`.

## 4. The interface

```python
def normalize(x: ArrayLike, axis: int = -1, eps: float = 1e-12) -> NDArray: ...
def cosine_sim(a: ArrayLike, b: ArrayLike, axis: int = -1, eps: float = 1e-8) -> NDArray: ...
def project(x: ArrayLike, v: ArrayLike, axis: int = -1) -> NDArray: ...
def topk_cosine(query: ArrayLike, matrix: ArrayLike, k: int) -> tuple[NDArray, NDArray]:
    """(indices int64, scores float64), [k] or [q, k], descending, ties to the lowest index."""
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_cosine_and_projection` | unit, smoke | section 3's cosine, projection, residual, and unit vector | you and the tests agree on the definitions |
| `test_hand_example_topk` | unit, smoke | rows $1, 0, 4$ with scores $1, 0.96, 0.8$ | cosine, not dot product, ranks the neighbours of `L2.3` |
| `test_cosine_matches_formula_and_broadcasts` | differential | a $[7, 5]$ batch against one vector, along the last axis and along axis 0 | the zoo scores whole vocabularies at once |
| `test_cosine_properties` | property | symmetric, scale invariant, sign flips, in $[-1, 1]$, $\cos(a, a) = 1$ | the angle and nothing else |
| `test_zero_and_tiny_vectors` | boundary | zero vector gives 0 (and `normalize` keeps it 0); parallel vectors of norm $10^{-5}$ give 1 | the per-norm clamp |
| `test_normalize_gives_unit_vectors` | property | unit rows, rescaling recovers $x$, `axis` respected | vector indexes store unit rows |
| `test_projection_properties` | property | $P^2 = P$, residual orthogonal to $v$, independent of $v$'s length | divide by $v \cdot v$ |
| `test_projection_rejects_zero_direction` | boundary | projecting onto 0 and mismatched lengths raise | no NaN, no silent broadcast |
| `test_topk_matches_brute_force` | differential | 6 queries against 200 rows of mixed lengths: indices and scores equal a full sort | exact top-$k$ |
| `test_topk_ties_go_to_lowest_index` | boundary | duplicate and rescaled rows come back in index order | reproducible across runs and languages |
| `test_topk_rejects_bad_k` | boundary | $k = 0$, $k > n$, and a wrong query length raise; $k = n$ returns every row | caller bugs fail early |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. ranking by the raw dot product | frequent words (long vectors) are everyone's nearest neighbour | `test_hand_example_topk`, `test_topk_matches_brute_force` (mutant `s01`) |
| 2. ties broken toward the highest index (sort ascending, then reverse) | two equal candidates come back in the other order; the Go port disagrees | `test_topk_ties_go_to_lowest_index` (mutant `s02`) |
| 3. clamping the product of the norms | tiny but parallel vectors score 0.01 instead of 1 | `test_zero_and_tiny_vectors` (mutant `s03`) |
| 4. projecting with $\lVert v \rVert$ in the denominator | correct only for unit $v$; the residual is not orthogonal | `test_hand_example_cosine_and_projection`, `test_projection_properties` (mutant `s04`) |
| 5. no clamp at all | NaN for any zero vector, and NaN sorts unpredictably | `test_zero_and_tiny_vectors` (mutant `s05`) |
| projecting onto the zero vector | NaN instead of an error | `test_projection_rejects_zero_direction` (mutant `s06`) |
| returning the top-$k$ in ascending order | the worst match first | `test_hand_example_topk` (mutant `s07`) |
| summing over the last axis whatever `axis` says | wrong results for column-stored vectors | `test_cosine_matches_formula_and_broadcasts` (mutant `s08`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M03.1` | the dot product is the inner loop of matmul; a row-major $[n, d]$ matrix keeps each candidate contiguous (reading) |
| Forward | `L2.3` | `analogy` ranks $b - a + c$ against the vocabulary with `topk_cosine`; nearest-neighbour checks of the embeddings |
| Forward | `L6.7` | the zoo's word-similarity task correlates `cosine_sim` with human ratings (Spearman) |
| Forward | `ag.07` | the Go retriever re-implements cosine and exact top-$k$ with the same tie rule (not a call site) |

If you skip this module, `L2.3` and `L6.7` stop with `BLOCKED ... needs M03.6` once they land: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `cosine_sim` | `torch.nn.functional.cosine_similarity` | the same per-norm clamp, batched on GPU | `torch/nn/functional.py`, `aten/src/ATen/native/Distance.cpp` |
| `topk_cosine` (exact) | FAISS `IndexFlatIP` over normalized rows | SIMD and BLAS inner products, a heap per query | `faiss/IndexFlat.cpp` |
| exact search | FAISS IVF and HNSW indexes | sublinear search with a recall knob (`nprobe`, `efSearch`) | `faiss/IndexIVF.cpp`, `faiss/IndexHNSW.cpp` |
| `normalize` + dot | pgvector's `<=>` cosine distance | the same search inside Postgres, with IVFFlat and HNSW indexes | `pgvector/src/vector.c` |
