<!-- ol:module S-M03b -->
# Linear algebra problem set, part b: inner products, SVD and low rank, matmul accounting and the roofline

## Overview

| | |
|---|---|
| **Module** | `S-M03b` · solve · none · Pass 3 · 3 to 4 h |
| **You build** | answers in `solve/S-M03b.toml` (14 checked by SymPy) and 2 proofs in `solve/S-M03b/q3.md` and `q8.md` (self-graded against their rubrics) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M03b/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M03b/problems.md` and in section 4 |
| **Needs** | no module. Reading: `S-M03a`, `M03.5` (SVD, Eckart-Young, least squares), `M03.6` (inner products, cosine, projection), and the [Linear Algebra topic](README.md) |
| **Used by** | no call site (a solve set). It checks the definitions behind `M03.5` and `M03.6`, which `L2.3`, `L6.6`, `L7.6`, and `C1` call, and the FLOP and byte counts `L9.1` tiles its matmul around |
| **Milestone** | `MS-P3` (the Pass 3 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | Strang, *Introduction to Linear Algebra*, sections 1.2, 4.2, 7.1 to 7.2; Trefethen and Bau, *Numerical Linear Algebra*, lectures 4 and 5; Williams, Waterman, and Patterson, "Roofline: an insightful visual performance model for multicore architectures" (CACM 2009) |

## Key Takeaways

- Cosine similarity divides the inner product by **both** lengths, and Cauchy-Schwarz keeps it in $[-1, 1]$ (q1, q3).
- A projection onto a line divides by $a \cdot a$, not by $\lVert a \rVert$ (q2).
- Singular values are the square roots of the eigenvalues of $A^\top A$, never the diagonal of $A$, and a matrix has $\min(m, n)$ of them (q4, q5, q8).
- Truncating the SVD after $r$ terms leaves an error of exactly the first dropped singular value in the spectral norm and the root of the sum of squares of the dropped ones in the Frobenius norm (q6); a rank-$r$ factorization of a $d \times d$ matrix costs $2dr$ numbers (q7).
- A matmul does $2MNK$ FLOPs on $O(MK + KN + MN)$ bytes, so its intensity grows with size; a matrix-vector product does about half a FLOP per byte whatever its size, which is why decoding is memory-bound (q9, q10).

## How to work this chapter

```bash
ol start S-M03b             # writes solve/S-M03b.toml and one file per proof
ol check S-M03b             # SymPy checks the answers, then asks each proof rubric (y/n)
ol check S-M03b --regrade   # ask the rubrics again after you change a proof
```

---

## 1. Why now

Pass 3 builds the first models that live on vectors: `L2.3` factors a word co-occurrence matrix with `M03.5`'s SVD and ranks neighbours with `M03.6`'s cosine similarity, and later `L6.6` and `L7.6` reuse the same low-rank factorization on weight matrices. Before the code, you need to compute these quantities by hand on small cases: an angle, a projection, a singular value, the error of a truncation. The set closes with arithmetic you will need again in Pass 6, where `L9.1` tiles the C matmul: how many FLOPs and how many bytes a product costs, and when the machine is waiting on memory instead of computing.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $a \cdot b$, $\lVert a \rVert$ | inner product $\sum_i a_i b_i$ and length $\sqrt{a \cdot a}$ | scalars |
| $\cos\theta = \frac{a \cdot b}{\lVert a \rVert \lVert b \rVert}$ | cosine similarity | scalar in $[-1, 1]$ |
| $\operatorname{proj}_a b = \frac{a \cdot b}{a \cdot a} a$ | projection of $b$ onto the line through $a$ | vector |
| $\sigma_1 \ge \sigma_2 \ge \dots$ | singular values | scalars $\ge 0$ |
| $A_r$ | truncated SVD of rank $r$ | matrix |
| $\lVert \cdot \rVert_2$, $\lVert \cdot \rVert_F$ | spectral and Frobenius norms | scalars |
| FLOP | one floating-point add or multiply | |
| $I$ | arithmetic intensity: FLOPs per byte moved between memory and the processor | FLOP/byte |
| $P$, $\beta$ | peak compute (FLOP/s) and memory bandwidth (byte/s) | |

### 2.1 Inner products, cosine, projection

The inner product $a \cdot b = \lVert a \rVert \lVert b \rVert \cos\theta$ carries both length and angle; cosine similarity keeps the angle. Projecting $b$ onto the line through $a$ keeps the component of $b$ along $a$: the coefficient $c$ in $b = c\,a + r$ with $r \perp a$ is $c = (a \cdot b)/(a \cdot a)$. The Cauchy-Schwarz inequality $\lvert a \cdot b \rvert \le \lVert a \rVert \lVert b \rVert$ (q3) is what makes the cosine a cosine.

### 2.2 SVD and low rank

$A = U\Sigma V^\top$ with orthonormal $U$, $V$ and $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$. Multiplying out, $A^\top A = V\Sigma^2 V^\top$, so $\sigma_i^2$ are the eigenvalues of $A^\top A$ (q8). The spectral norm is $\sigma_1$ and the Frobenius norm is $\sqrt{\sum_i \sigma_i^2}$. Eckart-Young: the truncated SVD $A_r$ is the closest rank-$r$ matrix, with $\lVert A - A_r \rVert_2 = \sigma_{r+1}$ and $\lVert A - A_r \rVert_F = \sqrt{\sum_{i > r}\sigma_i^2}$.

### 2.3 FLOPs, bytes, and the roofline

$C = AB$ with $A$ of shape $M \times K$ and $B$ of shape $K \times N$ has $MN$ entries, each a sum of $K$ products: $K$ multiplies and $K$ adds, so $2MNK$ FLOPs (the usual convention counts the $K$-th add even though a sum of $K$ terms needs only $K - 1$). The least it can move is reading $A$ and $B$ once and writing $C$ once: $4(MK + KN + MN)$ bytes in fp32. **Arithmetic intensity** is the ratio, FLOPs per byte. The **roofline** model says a kernel runs at most at $\min(P, I\beta)$ FLOP/s: below the **ridge point** $I = P/\beta$ it is memory-bound (faster memory helps, more compute does not), above it compute-bound. For a square $N \times N$ matmul the intensity grows like $N$; for a matrix-vector product it stays near $1/2$, whatever $N$.

## 3. Worked example by hand

These are siblings of q1, q6, and q10, not graded problems.

**Cosine.** $u = (1, 0, 1)$, $v = (1, 1, 0)$: $u \cdot v = 1$, $\lVert u \rVert = \lVert v \rVert = \sqrt{2}$, $\cos\theta = 1/2$ ($60°$). Written in `solve/` as `1/2`.

**Singular values and truncation.** $A = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}$ is already diagonal with non-negative entries in order, so $U = V = I$ and $\sigma = (2, 1)$. $A_1 = \begin{pmatrix} 2 & 0 \\ 0 & 0 \end{pmatrix}$, and $A - A_1 = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$ has $\lVert \cdot \rVert_2 = \lVert \cdot \rVert_F = 1 = \sigma_2$. For $\begin{pmatrix} 0 & -3 \\ 1 & 0 \end{pmatrix}$ the diagonal says nothing: $A^\top A = \begin{pmatrix} 1 & 0 \\ 0 & 9 \end{pmatrix}$, so $\sigma = (3, 1)$.

**Roofline.** A machine with $P = 1$ TFLOP/s and $\beta = 50$ GB/s has its ridge at $10^{12}/(5 \times 10^{10}) = 20$ FLOP/byte. An fp32 $N \times N$ matmul has intensity $2N^3 / (12 N^2) = N/6$, so it is compute-bound from $N/6 \ge 20$, $N \ge 120$. A matrix-vector product at intensity about $1/2$ runs at most at $50 \times 10^9 / 2 = 25$ GFLOP/s, 2.5% of peak.

## 4. The problem set

Write each answer in `solve/S-M03b.toml`; lettered parts are their own tables:

```toml
[q1.b]
answer = "4/(3*sqrt(5))"
[q5]
answer = "[3, 2]"
[q9.a]
answer = "2*M*N*K"
[q3]
proof = "S-M03b/q3.md"
```

Give exact values (`sqrt(5)`, not `2.236`).

<!-- ol:problems S-M03b -->

### Inner products and projections

**q1.** Let $u = (1, 2, 2)$ and $v = (2, 0, 1)$. Give (a) the inner product $u \cdot v$ and (b) the cosine of the angle between $u$ and $v$. `[number]`

**q2.** Give the orthogonal projection of $b = (1, 2, 3)$ onto the line spanned by $a = (1, 1, 1)$. `[vector]`

**q3.** Prove the Cauchy-Schwarz inequality $\lvert a \cdot b \rvert \le \lVert a \rVert \, \lVert b \rVert$ for vectors $a, b \in \mathbb{R}^n$, with equality exactly when one is a multiple of the other. Conclude that cosine similarity lies in $[-1, 1]$. `[proof]`

### SVD and low rank

**q4.** $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$. Give (a) its largest singular value $\sigma_1$ and (b) its smallest $\sigma_2$. `[number]`

**q5.** Give the singular values of $A = \begin{pmatrix} 0 & 2 \\ 3 & 0 \\ 0 & 0 \end{pmatrix}$ in descending order. `[vector]`

**q6.** A $3 \times 3$ matrix $A$ has singular values $5, 3, 1$, and $A_r$ is its truncated SVD of rank $r$. Give (a) $\lVert A - A_1 \rVert_2$, (b) $\lVert A - A_1 \rVert_F$, (c) the smallest $r$ with $\lVert A - A_r \rVert_F \le 3/2$. `[number]`

**q7.** A weight matrix of shape $4096 \times 4096$ is replaced by a product $BC$ with $B$ of shape $4096 \times r$ and $C$ of shape $r \times 4096$ (a rank-$r$ adapter). Give the number of parameters in $B$ and $C$ together. `[expr in r]`

**q8.** Let $A = U \Sigma V^\top$ be an SVD of a real $m \times n$ matrix. Prove that (i) the columns of $V$ are eigenvectors of $A^\top A$ with eigenvalues $\sigma_i^2$, so the singular values are the square roots of the eigenvalues of $A^\top A$, and (ii) $\max_{\lVert x \rVert = 1} \lVert A x \rVert = \sigma_1$. `[proof]`

### Matmul accounting and the roofline

**q9.** (a) Give the number of floating-point operations (one multiply and one add per term) of $C = AB$ with $A$ of shape $M \times K$ and $B$ of shape $K \times N$. `[expr in M, N, K]` (b) A matrix-vector product $y = Wx$ with $W$ of shape $N \times N$ in fp32 (4 bytes per number) reads $W$ and $x$ once and writes $y$ once. Give its arithmetic intensity, floating-point operations per byte moved. `[expr in N]`

**q10.** A square $N \times N$ fp32 matmul reads $A$ and $B$ once and writes $C$ once. (a) Give its arithmetic intensity in FLOP per byte. `[expr in N]` (b) A machine has a peak of $10$ TFLOP/s and a memory bandwidth of $100$ GB/s. Give the smallest $N$ at which this matmul reaches the compute-bound side of the roofline (intensity at least the ridge point). `[number]`

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Dividing the inner product by one length, or by squared lengths | a "cosine" outside $[-1, 1]$ or one that changes when a vector is rescaled | q1 b (canaries 4/15, 4/5) |
| Dividing by $\lVert a \rVert$ instead of $a \cdot a$ in a projection | the projection scaled by $\lVert a \rVert$ | q2 (canary with sqrt(3)) |
| Reading singular values off the diagonal | wrong whenever $A$ is not diagonal with non-negative entries | q4 (canary 1) |
| Reporting eigenvalues of $A^\top A$ as singular values | the square of the right answer | q4 (canaries without the square root) |
| Listing $\max(m, n)$ singular values, or not sorting them | a padded or unordered spectrum; `low_rank` keeps the wrong directions | q5 (canaries) |
| Confusing the spectral and the Frobenius error, or summing values instead of squares | the wrong truncation rank chosen for a budget | q6 (canaries) |
| Counting multiply-adds as FLOPs, or numbers as bytes | intensities off by a factor 2 or 4; the wrong side of the ridge | q9 a, q10 a and b (canaries) |
| Forgetting the vector reads in a matrix-vector product | an intensity of exactly 1/2 instead of slightly less | q9 b (canary) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M03a` | elimination, bases, eigenvalues, QR |
| Back | `M03.6` | `cosine_sim` and `project` are q1 and q2 as code |
| Back | `M03.5` | `svd`, `low_rank` (q4 to q7), and the eigenvalue bridge of q8 |
| Forward | `L2.3` | PPMI-SVD embeddings keep the top singular directions; analogies rank by cosine |
| Forward | `L6.6` | LoRA's $2dr$ parameters (q7) and PiSSA's truncated SVD |
| Forward | `L9.1` | the tiled matmul: intensity and the ridge point decide the tile size |
| Forward | `L8.2` | decoding is a matrix-vector product per token: memory-bound (q9 b) |
