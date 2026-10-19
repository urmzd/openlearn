# Linear algebra problem set, part b: inner products, SVD and low rank, matmul accounting and the roofline

The later sections of this chapter build the first tools that live on vectors. [word2vec skip-gram with negative sampling, and the PPMI-SVD baseline](../../build-an-llm/03-statistical-lm/03-word2vec-and-ppmi-svd.md) in *Build Your Own LLM System* factors a word co-occurrence matrix with the SVD of [SVD, Eckart-Young low rank, and least squares](05-svd-low-rank-and-least-squares.md) and ranks neighbours with the cosine similarity of [Inner products, projections, cosine similarity, and top-k](06-inner-products-projections-cosine-and-top-k.md). Later, [LoRA: low-rank adaptation with PiSSA initialization and merging](../../build-an-llm/07-objectives/06-lora-pissa-and-merge.md) and [Multi-head latent attention](../../build-an-llm/08-modern-block/06-multi-head-latent-attention.md) reuse the same low-rank factorization on weight matrices, and the [TinyStories capstone](../../build-an-llm/capstones/01-tinystories.md) fits a scaling law with the same least squares.

Before the code, it pays to compute these quantities by hand on small cases: an angle, a projection, a singular value, the error of a truncation. The set closes with arithmetic that comes back when [Cache-blocked, packed, batch-invariant matmul in C](../../build-an-llm/10-kernels/03-tiled-batch-invariant-matmul.md) tiles the C matmul: how many FLOPs and how many bytes a product costs, and when the machine is waiting on memory instead of computing.

This is a pen-and-paper set. It assumes the [linear algebra problem set, part a](07-problem-set-a.md), the SVD and inner-product sections, and the [chapter opening](README.md). Every problem is followed by a collapsed worked answer with the final result, the derivation, and the common wrong answers and why they are wrong. There are 14 exact answers (counting lettered parts) and 2 proofs (q3, q8); model proofs for those two are also in [`course/solutions/solve/S-M03b/`](../../../../course/solutions/solve/S-M03b/). Try each problem before opening its answer.

> **Learning objectives**
> - Compute inner products, cosines, and projections, and prove the Cauchy-Schwarz inequality.
> - Find singular values through the eigenvalues of $A^\top A$ and compute the error of a truncated SVD in both norms.
> - Count the parameters of a low-rank factorization.
> - Count the FLOPs and bytes of a matmul and a matrix-vector product, and use the roofline to decide whether a kernel is memory-bound or compute-bound.

## The ideas in brief

These are the facts the problems draw on. The notation:

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

### Inner products, cosine, projection

The inner product $a \cdot b = \lVert a \rVert \lVert b \rVert \cos\theta$ carries both length and angle; cosine similarity keeps the angle by dividing by **both** lengths. Projecting $b$ onto the line through $a$ keeps the component of $b$ along $a$: the coefficient $c$ in $b = c\,a + r$ with $r \perp a$ is $c = (a \cdot b)/(a \cdot a)$, divided by $a \cdot a$ and not by $\lVert a \rVert$. The Cauchy-Schwarz inequality $\lvert a \cdot b \rvert \le \lVert a \rVert \lVert b \rVert$ (q3) is what makes the cosine a cosine, keeping it in $[-1, 1]$.

### SVD and low rank

$A = U\Sigma V^\top$ with orthonormal $U$, $V$ and $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$. Multiplying out, $A^\top A = V\Sigma^2 V^\top$, so $\sigma_i^2$ are the eigenvalues of $A^\top A$ (q8): singular values are never simply the diagonal of $A$, and a matrix has $\min(m, n)$ of them. The spectral norm is $\sigma_1$ and the Frobenius norm is $\sqrt{\sum_i \sigma_i^2}$. Eckart-Young: the truncated SVD $A_r$ is the closest rank-$r$ matrix, with $\lVert A - A_r \rVert_2 = \sigma_{r+1}$ and $\lVert A - A_r \rVert_F = \sqrt{\sum_{i > r}\sigma_i^2}$. A rank-$r$ factorization of a $d \times d$ matrix costs $2dr$ numbers.

### FLOPs, bytes, and the roofline

$C = AB$ with $A$ of shape $M \times K$ and $B$ of shape $K \times N$ has $MN$ entries, each a sum of $K$ products: $K$ multiplies and $K$ adds, so $2MNK$ FLOPs (the usual convention counts the $K$-th add even though a sum of $K$ terms needs only $K - 1$). The least it can move is reading $A$ and $B$ once and writing $C$ once: $4(MK + KN + MN)$ bytes in fp32. **Arithmetic intensity** is the ratio, FLOPs per byte. The **roofline** model says a kernel runs at most at $\min(P, I\beta)$ FLOP/s: below the **ridge point** $I = P/\beta$ it is memory-bound (faster memory helps, more compute does not), above it compute-bound. For a square $N \times N$ matmul the intensity grows like $N$; for a matrix-vector product it stays near $1/2$, whatever $N$, which is why decoding is memory-bound.

## Three examples worked by hand

These are siblings of q1, q6, and q10, not problems of the set.

**Cosine.** $u = (1, 0, 1)$, $v = (1, 1, 0)$: $u \cdot v = 1$, $\lVert u \rVert = \lVert v \rVert = \sqrt{2}$, $\cos\theta = 1/2$ ($60°$).

**Singular values and truncation.** $A = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}$ is already diagonal with non-negative entries in order, so $U = V = I$ and $\sigma = (2, 1)$. $A_1 = \begin{pmatrix} 2 & 0 \\ 0 & 0 \end{pmatrix}$, and $A - A_1 = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}$ has $\lVert \cdot \rVert_2 = \lVert \cdot \rVert_F = 1 = \sigma_2$. For $\begin{pmatrix} 0 & -3 \\ 1 & 0 \end{pmatrix}$ the diagonal says nothing: $A^\top A = \begin{pmatrix} 1 & 0 \\ 0 & 9 \end{pmatrix}$, so $\sigma = (3, 1)$.

**Roofline.** A machine with $P = 1$ TFLOP/s and $\beta = 50$ GB/s has its ridge at $10^{12}/(5 \times 10^{10}) = 20$ FLOP/byte. An fp32 $N \times N$ matmul has intensity $2N^3 / (12 N^2) = N/6$, so it is compute-bound from $N/6 \ge 20$, $N \ge 120$. A matrix-vector product at intensity about $1/2$ runs at most at $50 \times 10^9 / 2 = 25$ GFLOP/s, 2.5% of peak.

## How the answers are graded

The problems are grouped by topic, in the order above. Answers are exact ($\sqrt5$, not 2.236). Lettered parts are answered separately. The tag after each problem says what kind of answer is expected.

## Inner products and projections

**q1.** Let $u = (1, 2, 2)$ and $v = (2, 0, 1)$. Give (a) the inner product $u \cdot v$ and (b) the cosine of the angle between $u$ and $v$. `[number]`

<details><summary>Answer</summary>

**(a) $4$. (b) $\frac{4}{3\sqrt5} = \frac{4\sqrt5}{15} \approx 0.596$.** (a) $u \cdot v = 1 \cdot 2 + 2 \cdot 0 + 2 \cdot 1 = 4$. (b) $\lVert u\rVert = \sqrt{1 + 4 + 4} = 3$ and $\lVert v\rVert = \sqrt{4 + 0 + 1} = \sqrt5$, so $\cos\theta = \frac{4}{3\sqrt5}$. The `cosine_sim` of [Inner products, projections, cosine similarity, and top-k](06-inner-products-projections-cosine-and-top-k.md) divides by both lengths in the same way. Common mistakes for (b): $\frac{4}{15}$, which divides by $\lVert u\rVert\,\lVert v\rVert^2$, and $\frac45$, which divides by $\lVert v\rVert^2$ only; neither is invariant under rescaling the vectors.

</details>

**q2.** Give the orthogonal projection of $b = (1, 2, 3)$ onto the line spanned by $a = (1, 1, 1)$. `[vector]`

<details><summary>Answer</summary>

**$(2, 2, 2)$.** $\frac{a \cdot b}{a \cdot a}a = \frac{6}{3}(1, 1, 1)$. Its `project` divides by $v \cdot v$ for the same reason. Common mistakes: $(2\sqrt3, 2\sqrt3, 2\sqrt3)$, which divides by $\lVert a\rVert$ instead of $a \cdot a$, and $(-1, 0, 1)$, which is the residual $b - \operatorname{proj}_a b$ rather than the projection.

</details>

**q3.** Prove the Cauchy-Schwarz inequality $\lvert a \cdot b \rvert \le \lVert a \rVert \, \lVert b \rVert$ for vectors $a, b \in \mathbb{R}^n$, with equality exactly when one is a multiple of the other. Conclude that cosine similarity lies in $[-1, 1]$. `[proof]`

<details><summary>Answer</summary>

**Claim.** For $a, b \in \mathbb{R}^n$, $\lvert a \cdot b\rvert \le \lVert a\rVert\,\lVert b\rVert$, with equality exactly when one is a scalar multiple of the other.

**Proof.** If $b = 0$, both sides are 0 and $b = 0 \cdot a$: equality holds. Let $b \ne 0$. A squared length is never negative, so for every real $t$,
$$0 \le \lVert a - tb\rVert^2 = \lVert a\rVert^2 - 2t\,(a \cdot b) + t^2\lVert b\rVert^2 .$$
Choose $t = (a \cdot b)/\lVert b\rVert^2$:
$$0 \le \lVert a\rVert^2 - 2\frac{(a \cdot b)^2}{\lVert b\rVert^2} + \frac{(a \cdot b)^2}{\lVert b\rVert^2} = \lVert a\rVert^2 - \frac{(a \cdot b)^2}{\lVert b\rVert^2} .$$
Multiplying by $\lVert b\rVert^2 > 0$ gives $(a \cdot b)^2 \le \lVert a\rVert^2\lVert b\rVert^2$, and taking non-negative square roots gives the inequality.

*Equality.* If equality holds, the chain gives $\lVert a - tb\rVert^2 = 0$ for that $t$, so $a = tb$. Conversely, if $a = sb$, then $\lvert a \cdot b\rvert = \lvert s\rvert\,\lVert b\rVert^2 = \lVert a\rVert\,\lVert b\rVert$ (and symmetrically if $b = sa$).

*Cosine.* For nonzero $a, b$, dividing by $\lVert a\rVert\,\lVert b\rVert > 0$ gives $-1 \le \frac{a \cdot b}{\lVert a\rVert\,\lVert b\rVert} \le 1$, with $\pm 1$ exactly for parallel or opposite vectors.

A complete proof treats $b = 0$ separately, expands $\lVert a - tb\rVert^2 \ge 0$, picks $t$ (or uses the discriminant of the quadratic in $t$), and proves both directions of the equality case. Model proof also in [`q3.md`](../../../../course/solutions/solve/S-M03b/q3.md).

</details>

## SVD and low rank

**q4.** $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$. Give (a) its largest singular value $\sigma_1$ and (b) its smallest $\sigma_2$. `[number]`

<details><summary>Answer</summary>

**(a) $\sigma_1 = \frac{1 + \sqrt5}{2}$. (b) $\sigma_2 = \frac{\sqrt5 - 1}{2}$.** $A^\top A = \begin{pmatrix} 1 & 1 \\ 1 & 2 \end{pmatrix}$ has trace 3 and determinant 1, so its eigenvalues are $\frac{3 \pm \sqrt5}{2}$. The singular values are their square roots, and $\left(\frac{1 \pm \sqrt5}{2}\right)^2 = \frac{6 \pm 2\sqrt5}{4} = \frac{3 \pm \sqrt5}{2}$ gives them in closed form ($\approx 1.618$ and $0.618$). Check: $\sigma_1\sigma_2 = 1 = \lvert\det A\rvert$. Common mistakes: 1 for either part, read off the diagonal (singular values are not the diagonal unless $A$ is diagonal and non-negative), and $\frac{3 \pm \sqrt5}{2}$, the eigenvalues of $A^\top A$ without the square root.

</details>

**q5.** Give the singular values of $A = \begin{pmatrix} 0 & 2 \\ 3 & 0 \\ 0 & 0 \end{pmatrix}$ in descending order. `[vector]`

<details><summary>Answer</summary>

**$(3, 2)$.** $A^\top A = \begin{pmatrix} 9 & 0 \\ 0 & 4 \end{pmatrix}$, so the singular values are 3 and 2, listed in descending order as the `svd` of [SVD, Eckart-Young low rank, and least squares](05-svd-low-rank-and-least-squares.md) returns them (the signs and the order of the columns live in $U$ and $V$). Common mistakes: $(2, 3)$, the column order, and $(3, 2, 0)$: a $3 \times 2$ matrix has $\min(m, n) = 2$ singular values.

</details>

**q6.** A $3 \times 3$ matrix $A$ has singular values $5, 3, 1$, and $A_r$ is its truncated SVD of rank $r$. Give (a) $\lVert A - A_1 \rVert_2$, (b) $\lVert A - A_1 \rVert_F$, (c) the smallest $r$ with $\lVert A - A_r \rVert_F \le 3/2$. `[number]`

<details><summary>Answer</summary>

**(a) $3$. (b) $\sqrt{10}$. (c) $r = 2$.** By Eckart-Young, $A - A_1$ has singular values $3, 1$. (a) Its spectral norm is the largest, $\sigma_2 = 3$: the first dropped singular value. (b) Its Frobenius norm is $\sqrt{3^2 + 1^2} = \sqrt{10} \approx 3.16$. (c) $\lVert A - A_1\rVert_F = \sqrt{10} > \frac32$, while $\lVert A - A_2\rVert_F = 1 \le \frac32$, so $r = 2$. Common mistakes: 1 for (a) (the last singular value rather than the first dropped), $\sqrt{10}$ for (a) (the Frobenius error), and 4 for (b) (summing the dropped values instead of their squares).

</details>

**q7.** A weight matrix of shape $4096 \times 4096$ is replaced by a product $BC$ with $B$ of shape $4096 \times r$ and $C$ of shape $r \times 4096$ (a rank-$r$ adapter). Give the number of parameters in $B$ and $C$ together. `[expr in r]`

<details><summary>Answer</summary>

**$8192\,r$.** $B$ has $4096r$ entries and $C$ has $r \cdot 4096$, so together $2 \cdot 4096 \cdot r$: the $2dr$ that LoRA ([LoRA: low-rank adaptation with PiSSA initialization and merging](../../build-an-llm/07-objectives/06-lora-pissa-and-merge.md)) trains instead of $d^2 = 16\,777\,216$. At $r = 8$ that is 65 536 parameters, 0.4% of the full matrix. Common mistakes: $4096r$ (one factor only) and $r^2$.

</details>

**q8.** Let $A = U \Sigma V^\top$ be an SVD of a real $m \times n$ matrix. Prove that (i) the columns of $V$ are eigenvectors of $A^\top A$ with eigenvalues $\sigma_i^2$, so the singular values are the square roots of the eigenvalues of $A^\top A$, and (ii) $\max_{\lVert x \rVert = 1} \lVert A x \rVert = \sigma_1$. `[proof]`

<details><summary>Answer</summary>

**Claim.** Let $A = U\Sigma V^\top$ with $U$ ($m \times m$) and $V$ ($n \times n$) orthogonal and $\Sigma$ ($m \times n$) zero except $\Sigma_{ii} = \sigma_i \ge 0$, $\sigma_1 \ge \sigma_2 \ge \dots$. Then (i) $A^\top A v_i = \sigma_i^2 v_i$ for each column $v_i$ of $V$ (with $\sigma_i = 0$ beyond $\min(m, n)$), and (ii) $\max_{\lVert x\rVert = 1}\lVert Ax\rVert = \sigma_1$.

**Proof.** (i) Using $U^\top U = I$, $A^\top A = V\Sigma^\top U^\top U\Sigma V^\top = V(\Sigma^\top\Sigma)V^\top$, where $\Sigma^\top\Sigma$ is diagonal with entries $\sigma_i^2$. With $v_i = Ve_i$ and $V^\top V = I$: $A^\top A v_i = V(\Sigma^\top\Sigma)e_i = \sigma_i^2 v_i$. The $v_i$ form a basis, so these are all the eigenvalues, and $\sigma_i = \sqrt{\lambda_i(A^\top A)}$.

(ii) Let $\lVert x\rVert = 1$ and $c = V^\top x$; $\lVert c\rVert = 1$ because $V^\top$ is orthogonal. Then
$$\lVert Ax\rVert^2 = x^\top A^\top A x = c^\top(\Sigma^\top\Sigma)c = \sum_i \sigma_i^2 c_i^2 \le \sigma_1^2\sum_i c_i^2 = \sigma_1^2 .$$
At $x = v_1$, $c = e_1$ and $\lVert Av_1\rVert = \sigma_1$, so the bound is attained.

The essential steps are the two orthogonality cancellations ($U^\top U$ in (i), $\lVert V^\top x\rVert = \lVert x\rVert$ in (ii)) and showing the maximum is attained, not just bounded. This is why Eckart-Young's spectral error is $\sigma_{r+1}$. Model proof also in [`q8.md`](../../../../course/solutions/solve/S-M03b/q8.md).

</details>

## Matmul accounting and the roofline

**q9.** (a) Give the number of floating-point operations (one multiply and one add per term) of $C = AB$ with $A$ of shape $M \times K$ and $B$ of shape $K \times N$. `[expr in M, N, K]` (b) A matrix-vector product $y = Wx$ with $W$ of shape $N \times N$ in fp32 (4 bytes per number) reads $W$ and $x$ once and writes $y$ once. Give its arithmetic intensity, floating-point operations per byte moved. `[expr in N]`

<details><summary>Answer</summary>

**(a) $2MNK$. (b) $\frac{2N^2}{4N^2 + 8N} = \frac{N}{2N + 4}$.** (a) $C$ has $MN$ entries, each $K$ multiplies and $K$ adds. (b) A matrix-vector product does $2N^2$ FLOPs and moves $4(N^2 + N + N)$ bytes ($W$, $x$, $y$), so its intensity is just under $\frac12$ for every $N$. That is why decoding, one matrix-vector product per weight matrix per token, is memory-bound. Common mistakes: $MNK$ for (a), counting multiply-adds instead of FLOPs; $\frac{2N^2}{4N^2} = \frac12$ for (b), forgetting to move $x$ and $y$; and $N/6$, the square matmul's intensity.

</details>

**q10.** A square $N \times N$ fp32 matmul reads $A$ and $B$ once and writes $C$ once. (a) Give its arithmetic intensity in FLOP per byte. `[expr in N]` (b) A machine has a peak of $10$ TFLOP/s and a memory bandwidth of $100$ GB/s. Give the smallest $N$ at which this matmul reaches the compute-bound side of the roofline (intensity at least the ridge point). `[number]`

<details><summary>Answer</summary>

**(a) $N/6$. (b) $N = 600$.** (a) $2N^3$ FLOPs over $4 \cdot 3N^2 = 12N^2$ bytes gives $\frac{N}{6}$. It grows with $N$, which is why tiling a large matmul ([Cache-blocked, packed, batch-invariant matmul in C](../../build-an-llm/10-kernels/03-tiled-batch-invariant-matmul.md)) pays off. (b) The ridge point is $\frac{10 \times 10^{12}}{100 \times 10^9} = 100$ FLOP/byte, and $\frac{N}{6} \ge 100$ gives $N \ge 600$. Common mistakes: $\frac{2N}{3}$ for (a) (numbers counted instead of 4-byte values), $\frac{N}{12}$ (multiply-adds instead of FLOPs), 100 for (b) (the ridge point itself, not $N$), and 150 (from the intensity $2N/3$).

</details>

## Common mistakes

The wrong answers above fall into a few patterns, each with a recognizable symptom:

| Mistake | Symptom | The habit that avoids it |
|---|---|---|
| Dividing the inner product by one length, or by squared lengths | a "cosine" outside $[-1, 1]$ or one that changes when a vector is rescaled | divide by $\lVert u\rVert\,\lVert v\rVert$, each length once (q1 b) |
| Dividing by $\lVert a \rVert$ instead of $a \cdot a$ in a projection | the projection scaled by $\lVert a \rVert$ | divide by $a \cdot a$ (q2) |
| Reading singular values off the diagonal | wrong whenever $A$ is not diagonal with non-negative entries | go through the eigenvalues of $A^\top A$ (q4) |
| Reporting eigenvalues of $A^\top A$ as singular values | the square of the right answer | take the square root at the end (q4) |
| Listing $\max(m, n)$ singular values, or not sorting them | a padded or unordered spectrum; `low_rank` keeps the wrong directions | list $\min(m, n)$ values in descending order (q5) |
| Confusing the spectral and the Frobenius error, or summing values instead of squares | the wrong truncation rank chosen for a budget | use $\sigma_{r+1}$ for the spectral error and the root of the sum of squares for the Frobenius error (q6) |
| Counting multiply-adds as FLOPs, or numbers as bytes | intensities off by a factor 2 or 4; the wrong side of the ridge | count 2 FLOPs per term and 4 bytes per fp32 number (q9 a, q10) |
| Forgetting the vector reads in a matrix-vector product | an intensity of exactly 1/2 instead of slightly less | count the bytes of $W$, $x$, and $y$ (q9 b) |

## Where this leads

Looking back, this set builds on the elimination, bases, eigenvalues, and QR of the [linear algebra problem set, part a](07-problem-set-a.md). q1 and q2 are `cosine_sim` and `project` from [Inner products, projections, cosine similarity, and top-k](06-inner-products-projections-cosine-and-top-k.md) done by hand, and q4 to q8 are `svd`, `low_rank`, and the eigenvalue bridge of [SVD, Eckart-Young low rank, and least squares](05-svd-low-rank-and-least-squares.md).

Looking forward, in *Build Your Own LLM System*: the PPMI-SVD embeddings of [word2vec skip-gram with negative sampling, and the PPMI-SVD baseline](../../build-an-llm/03-statistical-lm/03-word2vec-and-ppmi-svd.md) keep the top singular directions, and its analogies rank by cosine. [LoRA: low-rank adaptation with PiSSA initialization and merging](../../build-an-llm/07-objectives/06-lora-pissa-and-merge.md) trains LoRA's $2dr$ parameters (q7) and initializes from PiSSA's truncated SVD. In [Cache-blocked, packed, batch-invariant matmul in C](../../build-an-llm/10-kernels/03-tiled-batch-invariant-matmul.md), intensity and the ridge point decide the tile size. And in [The KV cache, incremental decoding, and generate](../../build-an-llm/09-inference/02-kv-cache-and-generate.md), decoding is a matrix-vector product per token, and so memory-bound (q9 b).

### Further reading

- Strang, *Introduction to Linear Algebra*, sections 1.2, 4.2, 7.1 to 7.2.
- Trefethen and Bau, *Numerical Linear Algebra*, lectures 4 and 5.
- Williams, Waterman, and Patterson, "Roofline: an insightful visual performance model for multicore architectures" (CACM 2009).

## Summary

- Cosine similarity divides the inner product by both lengths, and Cauchy-Schwarz keeps it in $[-1, 1]$ (q1, q3).
- A projection onto a line divides by $a \cdot a$, not by $\lVert a \rVert$ (q2).
- Singular values are the square roots of the eigenvalues of $A^\top A$, never the diagonal of $A$, and a matrix has $\min(m, n)$ of them (q4, q5, q8).
- Truncating the SVD after $r$ terms leaves an error of exactly the first dropped singular value in the spectral norm and the root of the sum of squares of the dropped ones in the Frobenius norm (q6); a rank-$r$ factorization of a $d \times d$ matrix costs $2dr$ numbers (q7).
- A matmul does $2MNK$ FLOPs on $O(MK + KN + MN)$ bytes, so its intensity grows with size; a matrix-vector product does about half a FLOP per byte whatever its size, which is why decoding is memory-bound (q9, q10).
