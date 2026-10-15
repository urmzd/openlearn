<!-- ol:module M03.5 -->
# SVD, Eckart-Young low rank, and least squares

## Overview

| | |
|---|---|
| **Module** | `M03.5` · build · Python · Pass 3 · 4 to 5 h |
| **You build** | `python/tinyllm/linalg/svd.py`: `svd(A)` (reduced SVD by one-sided Jacobi rotations, singular values descending, a fixed sign rule), `low_rank(A, r)` (the best rank-$r$ approximation as balanced factors $B C$), and `lstsq(A, b)` (least squares by QR) |
| **Contract** | [`course/contracts/py/tinyllm/linalg/svd.pyi`](../../../../course/contracts/py/tinyllm/linalg/svd.pyi) |
| **Tests** | `course/tests/M03.5/test_svd.py` (what they check: section 4) |
| **Needs** | `M03.3` `qr_householder`, which `lstsq` factors with (or `--ref-deps`). Reading: `M03.4` (eigenvalues; singular values are their square roots for $A^\top A$), `S-M03a` |
| **Used by** | later `L2.3` PPMI-SVD word vectors, `L6.6` LoRA's PiSSA initialization, `L7.6` the MLA conversion, `C1` the scaling-law fit, `M10.6` Muon (each joins the registry with its batch) · later: `M09.3` |
| **Milestone** | `MS-P3` (the tokens-and-data gate) |
| **Optional depth** | Trefethen and Bau, *Numerical Linear Algebra*, lectures 4, 5, 11, 31; Demmel and Veselic, "Jacobi's method is more accurate than QR" (1992); Eckart and Young, "The approximation of one matrix by another of lower rank" (1936); Meng, Wang, and Zhang, "PiSSA" (2024) |

## Key Takeaways

- Every real matrix is $A = U \Sigma V^\top$: rotate, stretch along the axes by the singular values, rotate again. The singular values are the square roots of the eigenvalues of $A^\top A$, and the largest is the most $A$ can stretch a unit vector (`test_hand_example_svd`, `test_random_matrices_match_numpy`).
- One-sided Jacobi rotates pairs of columns until all are orthogonal; the test for "orthogonal enough" must be relative to the column lengths, or a matrix scaled by $10^{-12}$ is never rotated (`test_scale_invariance`).
- Keeping the $r$ largest singular triples gives the closest rank-$r$ matrix, and the error is exactly the first dropped singular value (Eckart-Young, `test_eckart_young_error`); splitting $\sqrt{\sigma}$ into each factor makes them start at the same scale (`test_low_rank_factors_are_balanced`).
- Least squares goes through QR, never $A^\top A$: the normal equations square the condition number and lose half the digits (`test_lstsq_ill_conditioned_beats_normal_equations`).

## How to work this chapter

```bash
ol start M03.5              # stubs python/tinyllm/linalg/svd.py into your repo
ol tests M03.5              # read the test catalog first: rung R0, you write no tests here
ol check M03.5              # exit code is the verdict
ol check M03.5 --ref-deps   # only if your M03.3 is not passing yet
ol diff  M03.5              # after passing: your code against the reference
```

---

## 1. Why now

Pass 3 starts treating text as numbers you can factor. `L2.3` counts which words appear near which others in a matrix with tens of thousands of rows and asks for 64 numbers per word that keep most of its structure: that is the best rank-64 approximation of the matrix, and the singular value decomposition is the only tool that gives it with a proof of optimality. The same decomposition comes back three more times in the system: `L6.6` starts a LoRA adapter on the top singular directions of a weight matrix, `L7.6` compresses attention's keys and values into a low-rank latent, and the capstone fits a scaling law $\log L = a + b \log N$ to a handful of runs by least squares. You already have the pieces: orthogonal matrices and QR (`M03.3`) and eigenvalues (`M03.4`). This module puts them together.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $A$ | the matrix, $m$ rows and $n$ columns | `float64[m, n]` |
| $k = \min(m, n)$ | number of singular values | `int` |
| $U$ | left singular vectors, orthonormal columns ($U^\top U = I$) | `float64[m, k]` |
| $\sigma_1 \ge \dots \ge \sigma_k \ge 0$ | singular values; $\Sigma = \operatorname{diag}(\sigma)$ | `float64[k]` |
| $V$ | right singular vectors, orthonormal columns; the contract returns $V^\top$ | `float64[n, k]` |
| $u_i, v_i$ | column $i$ of $U$ and of $V$ | vectors |
| $\lVert x \rVert$ | Euclidean length $\sqrt{x^\top x}$ | scalar |
| $\lVert A \rVert_2 = \max_{\lVert x \rVert = 1} \lVert Ax \rVert$ | spectral norm | scalar |
| $\lVert A \rVert_F = \sqrt{\sum_{ij} A_{ij}^2}$ | Frobenius norm | scalar |
| $A_r$ | the truncated SVD $\sum_{i \le r} \sigma_i u_i v_i^\top$ | `float64[m, n]` |
| $w_i, w_j$ | two columns of the working matrix $W$ | vectors |
| $\alpha, \beta, \gamma$ | $w_i^\top w_i$, $w_j^\top w_j$, $w_i^\top w_j$ | scalars |
| $\zeta, t, c, s$ | rotation parameters: $t = \tan\theta$, $c = \cos\theta$, $s = \sin\theta$ | scalars |
| $\varepsilon$ | float64 machine epsilon, $2^{-52} \approx 2.2 \times 10^{-16}$ | |
| $\kappa(A) = \sigma_1 / \sigma_k$ | condition number | scalar |
| $Q, R$ | the QR factors of `M03.3` | `float64[m, k]`, `float64[k, n]` |

### 2.1 What the SVD says

Every real $m \times n$ matrix factors as

$$A = U \Sigma V^\top = \sum_{i=1}^{k} \sigma_i\, u_i v_i^\top ,$$

with orthonormal columns in $U$ and $V$ and non-negative $\sigma_i$ in descending order. Read it right to left on a vector $x$: $V^\top x$ takes the coordinates of $x$ along the directions $v_i$, $\Sigma$ stretches coordinate $i$ by $\sigma_i$, and $U$ sends the result out along the directions $u_i$. So $A v_i = \sigma_i u_i$: the unit sphere goes to an ellipsoid whose semi-axes are $\sigma_i u_i$. This is the **reduced** SVD ($U$ is $m \times k$); the full one pads $U$ to a square matrix, which nothing in the course needs.

Each pair $(u_i, v_i)$ can be negated together without changing $A$ ($(-u_i)\sigma_i(-v_i)^\top = u_i \sigma_i v_i^\top$). When the singular values are distinct this is the only freedom, so the contract fixes it with a **sign rule**: the entry of largest absolute value in each row of $V^\top$ is positive (the first such entry on ties). With it, your factors and LAPACK's (after the same rule) agree entry by entry.

### 2.2 Singular values, eigenvalues, and norms

Multiply out $A^\top A = V \Sigma^\top U^\top U \Sigma V^\top = V \Sigma^2 V^\top$: the $v_i$ are eigenvectors of the symmetric matrix $A^\top A$ with eigenvalues $\sigma_i^2$ (S-M03b q8 asks you to prove it). So everything `M03.4` said about eigenvalues applies, with two differences: singular values exist for every matrix, square or not, and they are never negative or complex.

Two norms come straight from them. For a unit $x$ with coordinates $c = V^\top x$, $\lVert Ax \rVert^2 = \sum_i \sigma_i^2 c_i^2 \le \sigma_1^2$, with equality at $x = v_1$: the **spectral norm** is $\lVert A \rVert_2 = \sigma_1$. Orthogonal factors do not change the sum of squares of entries, so the **Frobenius norm** is $\lVert A \rVert_F = \sqrt{\sum_i \sigma_i^2}$.

Forming $A^\top A$ and calling an eigensolver would compute the SVD, but badly: squaring the singular values squares the condition number, and a $\sigma_i$ below $\sqrt{\varepsilon}\,\sigma_1 \approx 10^{-8}\sigma_1$ drowns in the rounding of $\sigma_1^2$. Jacobi works on $A$ itself.

### 2.3 One-sided Jacobi

Work on a tall matrix ($m \ge n$; for a wide one, decompose $A^\top$ and swap $U$ and $V$ at the end). Start with $W = A$ and $V = I$, and keep the invariant $W = AV$ with $V$ orthogonal. If all columns of $W$ were mutually orthogonal, their lengths would be the singular values and their directions the $u_i$: $W = U\Sigma$, so $A = U \Sigma V^\top$.

To make columns $i$ and $j$ orthogonal, rotate them in their plane:

$$w_i \leftarrow c\, w_i - s\, w_j, \qquad w_j \leftarrow s\, w_i + c\, w_j ,$$

and apply the same rotation to columns $i$ and $j$ of $V$ (so $W = AV$ still holds; the rotation is orthogonal, so $V$ stays orthogonal). The new inner product is $cs(\alpha - \beta) + (c^2 - s^2)\gamma$. Setting it to zero with $t = s/c$ gives $t^2 + 2\zeta t - 1 = 0$, $\zeta = (\beta - \alpha) / (2\gamma)$. Take the smaller root,

$$t = \frac{\operatorname{sign}(\zeta)}{\lvert \zeta \rvert + \sqrt{1 + \zeta^2}}, \qquad c = \frac{1}{\sqrt{1 + t^2}}, \qquad s = c\,t ,$$

with $\operatorname{sign}(0) = +1$: it is the rotation by at most $45°$, the one that converges. A **sweep** visits every pair $i < j$ once. A rotation for one pair can spoil an earlier pair a little, but the off-diagonal mass shrinks every sweep and, near the end, quadratically; random $30 \times 12$ matrices finish in 7 or 8 sweeps, the last one only confirming that nothing needs rotating.

When is a pair "orthogonal enough"? The test must be **relative**: rotate only when $\lvert \gamma \rvert > m \varepsilon \sqrt{\alpha \beta}$, that is when the cosine of the angle between the columns exceeds $m\varepsilon$. A fixed threshold such as $\lvert\gamma\rvert > 10^{-15}$ skips every rotation on a matrix scaled by $10^{-12}$ (all inner products are about $10^{-24}$) and returns garbage; scaled by $10^{12}$ it never stops. Stop after the first sweep that rotates nothing (the contract caps sweeps at 64).

Finally $\sigma_j = \lVert w_j \rVert$, sorted descending (stable), $u_j = w_j / \sigma_j$, and the sign rule.

### 2.4 Rank deficiency

If $A$ has rank $\rho < k$, then $k - \rho$ singular values are zero and their columns $w_j$ are (numerically) zero: $w_j / \sigma_j$ is $0/0$. The contract treats $\sigma_j \le m \varepsilon \sigma_1$ as zero and fills $u_j$ with a unit vector orthogonal to the $u$'s already chosen: take the first standard basis vector $e_i$ that has more than half its length left after projecting out the previous columns ($e \leftarrow e - U(U^\top e)$, done twice for accuracy), and normalize it. Any such completion is a valid SVD, because it multiplies a zero singular value; the contract fixes this one so the result is deterministic.

### 2.5 Eckart-Young and the balanced split

The truncated SVD $A_r = \sum_{i \le r} \sigma_i u_i v_i^\top$ has rank $r$, and $A - A_r = \sum_{i > r} \sigma_i u_i v_i^\top$ is itself an SVD, so

$$\lVert A - A_r \rVert_2 = \sigma_{r+1}, \qquad \lVert A - A_r \rVert_F = \sqrt{\textstyle\sum_{i > r} \sigma_i^2} .$$

The **Eckart-Young theorem** says no matrix of rank at most $r$ does better in either norm: the truncated SVD is the best low-rank approximation. That is why it is the right way to compress a co-occurrence matrix (`L2.3`) or initialize a low-rank adapter (`L6.6`).

`low_rank` returns $A_r$ as two factors $B$ ($m \times r$) and $C$ ($r \times n$) with $BC = A_r$. Many splits multiply to the same product; the contract uses the **balanced** one,

$$B = U_r \Sigma_r^{1/2}, \qquad C = \Sigma_r^{1/2} V_r^\top ,$$

so $B^\top B = C C^\top = \Sigma_r$: both factors carry the same scale. PiSSA initializes LoRA this way because gradient descent on $BC$ updates each factor in proportion to the other's size; a split such as $(U_r\Sigma_r, V_r^\top)$ trains one factor much faster than the other.

### 2.6 Least squares by QR

For a tall $A$ with independent columns, $Ax = b$ usually has no solution; least squares picks $x$ minimizing $\lVert Ax - b \rVert$. The minimizer makes the residual $r = b - Ax$ orthogonal to every column of $A$: $A^\top r = 0$, the **normal equations** $A^\top A x = A^\top b$. Solving them as written is the classic mistake. $\kappa(A^\top A) = \kappa(A)^2$, so a polynomial fit with $\kappa(A) \approx 3.5 \times 10^6$ (degree 9 at 40 points of $[0, 1]$) recovers its coefficients only to about $10^{-3}$.

QR avoids the square. With $A = QR$ (`M03.3`), $Q$ has orthonormal columns, so the residual splits into a part inside the column space and a part orthogonal to it, and only the first depends on $x$:

$$\lVert Ax - b \rVert^2 = \lVert Rx - Q^\top b \rVert^2 + \lVert (I - QQ^\top) b \rVert^2 .$$

So $x$ solves the triangular system $R x = Q^\top b$, by **back substitution** from the last row up: $x_i = (z_i - \sum_{j > i} R_{ij} x_j) / R_{ii}$ with $z = Q^\top b$. The error now grows like $\kappa(A)\varepsilon$: $5 \times 10^{-10}$ for the same fit. If some $\lvert R_{ii} \rvert$ is tiny ($\le \max(m, n)\,\varepsilon \max_j \lvert R_{jj} \rvert$), the columns are dependent, the minimizer is not unique, and the contract raises instead of dividing by almost zero.

## 3. Worked example by hand

$$A = \begin{pmatrix} 3 & 0 \\ 4 & 5 \end{pmatrix}.$$

**Jacobi.** One pair, $(0, 1)$.

| Quantity | Value |
|---|---|
| $\alpha = \lVert w_0 \rVert^2$, $\beta = \lVert w_1 \rVert^2$, $\gamma = w_0^\top w_1$ | $9 + 16 = 25$, $0 + 25 = 25$, $0 + 20 = 20$ |
| $\zeta = (\beta - \alpha)/(2\gamma)$ | $0$, so $t = 1$, $c = s = 1/\sqrt{2}$ (a $45°$ rotation) |
| new $w_0 = c\,w_0 - s\,w_1$ | $(3 - 0,\ 4 - 5)/\sqrt{2} = (3, -1)/\sqrt{2}$, length $\sqrt{5}$ |
| new $w_1 = s\,w_0 + c\,w_1$ | $(3 + 0,\ 4 + 5)/\sqrt{2} = (3, 9)/\sqrt{2}$, length $\sqrt{45} = 3\sqrt{5}$ |
| new $V$ columns | $v_0 = (1, -1)/\sqrt{2}$, $v_1 = (1, 1)/\sqrt{2}$ |
| second sweep | $w_0^\top w_1 = (9 - 9)/2 = 0$: nothing to rotate, stop |

Sorting puts $\sigma_1 = 3\sqrt{5} \approx 6.708$ first, then $\sigma_2 = \sqrt{5} \approx 2.236$; check: $\sigma_1^2 + \sigma_2^2 = 50 = 9 + 16 + 25 = \lVert A \rVert_F^2$, and $\sigma_1\sigma_2 = 15 = \lvert \det A \rvert$. Dividing each $w$ by its length:

$$U = \frac{1}{\sqrt{10}}\begin{pmatrix} 1 & 3 \\ 3 & -1 \end{pmatrix}, \quad \Sigma = \begin{pmatrix} 3\sqrt{5} & 0 \\ 0 & \sqrt{5} \end{pmatrix}, \quad V^\top = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}.$$

The sign rule holds already: each row of $V^\top$ has its largest entries tied, and the first is positive. This is `test_hand_example_svd`.

**Rank 1.** $A_1 = \sigma_1 u_1 v_1^\top = 3\sqrt{5} \cdot \frac{1}{\sqrt{20}} \begin{pmatrix} 1 & 1 \\ 3 & 3 \end{pmatrix} = \begin{pmatrix} 1.5 & 1.5 \\ 4.5 & 4.5 \end{pmatrix}$. The error $A - A_1 = \begin{pmatrix} 1.5 & -1.5 \\ -0.5 & 0.5 \end{pmatrix} = \sigma_2 u_2 v_2^\top$ has spectral and Frobenius norm $\sqrt{2.25 + 2.25 + 0.25 + 0.25} = \sqrt{5} = \sigma_2$, as Eckart-Young says. The balanced factors are $B = 45^{1/4}(1, 3)^\top/\sqrt{10}$ and $C = 45^{1/4}(1, 1)/\sqrt{2}$. This is `test_hand_example_rank_one`.

**Least squares.** Fit a line $x_0 + x_1 t$ through $(0, 1)$, $(1, 2)$, $(2, 2)$: $A = \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix}$, $b = (1, 2, 2)$. By hand, the residual condition $A^\top(b - Ax) = 0$ reads $3x_0 + 3x_1 = 5$ and $3x_0 + 5x_1 = 6$, so $x_1 = 1/2$ and $x_0 = 7/6$. Check: the residual $b - Ax = (1 - 7/6,\ 2 - 5/3,\ 2 - 13/6) = (-1/6, 1/3, -1/6)$ sums to 0 (orthogonal to the first column) and $0 \cdot (-1/6) + 1 \cdot (1/3) + 2 \cdot (-1/6) = 0$ (orthogonal to the second). Writing out the normal equations is fine for a $2 \times 2$ system on paper with exact fractions; your code must still go through QR, because in floating point the square of the condition number is what hurts. This is `test_hand_example_lstsq`.

## 4. The interface

```python
def svd(A: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
    """Reduced SVD by one-sided Jacobi: U [m, k], S [k] descending, Vt [k, n],
    A == U @ diag(S) @ Vt, the largest |entry| of each Vt row positive."""

def low_rank(A: ArrayLike, r: int) -> tuple[NDArray, NDArray]:
    """B = U_r sqrt(S_r) [m, r], C = sqrt(S_r) Vt_r [r, n]: B @ C is the best rank-r approximation."""

def lstsq(A: ArrayLike, b: ArrayLike) -> NDArray:
    """argmin ||A x - b|| for a tall, full-column-rank A, by QR and back substitution."""
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_svd` | unit, smoke | section 3's $U$, $\Sigma$, $V^\top$ to $10^{-14}$ | you and the tests agree on the rotation, the order, and the sign rule |
| `test_hand_example_rank_one` | unit, smoke | $A_1$, the balanced factors, and the error $\sqrt{5}$ | the numbers of section 3 |
| `test_hand_example_lstsq` | unit, smoke | $x = (7/6, 1/2)$ | the line fit of section 3 |
| `test_random_matrices_match_numpy` | differential | tall, wide, square, and $1 \times n$ shapes against `np.linalg.svd` with the same sign rule, entry by entry | the decomposition is the unique one |
| `test_scale_invariance` | property | `svd(c A)` is $c$ times `svd(A)` for $c = 10^{-12}, 10^{12}$ | the rotation test is relative |
| `test_rank_deficient_completes_u` | boundary | a rank-2 matrix and the zero matrix: $U$ orthonormal, no NaN | co-occurrence matrices are often rank deficient |
| `test_eckart_young_error` | property | spectral and Frobenius error of `low_rank` for $r = 1, 2, 4, 6$ | the optimality `L2.3` and `L6.6` rely on |
| `test_low_rank_factors_are_balanced` | property | $B^\top B = C C^\top = \Sigma_r$ | PiSSA's equal-scale factors |
| `test_low_rank_rejects_bad_rank` | boundary | $r = 0$, $r < 0$, $r > \min(m, n)$ raise | config errors fail early |
| `test_lstsq_matches_numpy` | differential | one and three right-hand sides against `np.linalg.lstsq` | the minimizer is unique |
| `test_lstsq_residual_is_orthogonal` | property | $A^\top (b - Ax) = 0$ | the defining property |
| `test_lstsq_ill_conditioned_beats_normal_equations` | boundary | a degree-9 polynomial fit ($\kappa \approx 3.5 \times 10^6$) recovers its coefficients to $10^{-6}$ | the capstone's scaling-law fit |
| `test_lstsq_rejects_rank_deficient_and_wide` | boundary | dependent columns, a wide system, and a wrong-length $b$ raise | no division by a zero pivot |
| `test_inputs_not_modified_and_validated` | boundary | $A$ is unchanged; 1-D and NaN inputs raise | `L6.6` keeps the weight it decomposes |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. solving the normal equations $A^\top A x = A^\top b$ | fine on easy fits, three correct digits on a degree-9 polynomial fit | `test_lstsq_ill_conditioned_beats_normal_equations` (mutant `s01`) |
| 2. leaving singular values in sweep order | `low_rank` keeps the wrong directions; the error is not $\sigma_{r+1}$ | `test_hand_example_svd`, `test_eckart_young_error` (mutant `s02`) |
| 3. the rotation with the wrong sign ($s = -ct$) | the columns never become orthogonal; 64 sweeps of noise | `test_random_matrices_match_numpy` (mutant `s03`) |
| 4. stopping after one sweep | the columns are far from orthogonal: singular values of a random $30 \times 12$ matrix off by 0.2 to 0.8 | `test_random_matrices_match_numpy` (mutant `s04`) |
| 5. the unbalanced split $(U_r\Sigma_r, V_r^\top)$ | same product, factors at different scales | `test_low_rank_factors_are_balanced`, `test_hand_example_rank_one` (mutant `s06`) |
| 6. dividing a zero column by its zero length | NaN in $U$ for rank-deficient matrices | `test_rank_deficient_completes_u` (mutant `s07`) |
| 7. no sign rule | factors disagree with LAPACK's on some columns; downstream tests that fix a seed see flipped vectors | `test_random_matrices_match_numpy` (mutant `s11`) |
| 8. an absolute threshold $\lvert\gamma\rvert > 10^{-15}$ | tiny-scaled matrices are returned unrotated | `test_scale_invariance` (mutant `s12`) |
| decomposing a wide matrix without transposing | shapes come out wrong | `test_random_matrices_match_numpy` (mutant `s05`) |
| back substitution from the top row | wrong $x$ whenever $R$ is not diagonal | `test_hand_example_lstsq`, `test_lstsq_matches_numpy` (mutant `s08`) |
| no rank check in `lstsq` | division by a zero diagonal entry of $R$: inf | `test_lstsq_rejects_rank_deficient_and_wide` (mutant `s09`) |
| rotating the caller's array in place | the weight you decomposed is now $U\Sigma$ | `test_inputs_not_modified_and_validated` (mutant `s10`) |

## 6. Where it's used next
| Forward | `M09.3` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M03.3` | `lstsq` factors $A = QR$ with `qr_householder` and solves $Rx = Q^\top b$ |
| Back | `M03.4` | singular values are the square roots of the eigenvalues of $A^\top A$ (reading) |
| Forward | `L2.3` | `ppmi_svd_embeddings` keeps the top singular directions of the PPMI matrix as word vectors |
| Forward | `L6.6` | PiSSA initializes LoRA with `low_rank(W, r)` and trains the residual |
| Forward | `L7.6` | `mha_to_mla` compresses the key and value projections to a rank-$r$ latent |
| Forward | `C1` | the scaling-law fit is `lstsq` on $\log$ loss against $\log$ parameters |
| Forward | `M10.6` | Muon's Newton-Schulz iteration approximates $UV^\top$, checked against this SVD |

If you skip this module, the modules above stop with `BLOCKED ... needs M03.5` once they land: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| one-sided Jacobi | LAPACK `dgesvj` (Drmac and Veselic) | preconditioning by QR with column pivoting, de Rijk's pivoting, blocked rotations; Jacobi's high relative accuracy at near-LAPACK speed | LAPACK `SRC/dgesvj.f`, `dgejsv.f` |
| `svd` | LAPACK `dgesdd` (what numpy calls) | Householder bidiagonalization, then divide and conquer on the bidiagonal matrix; much faster for large matrices | `numpy/linalg/_linalg.py`, LAPACK `SRC/dgesdd.f` |
| `low_rank` | randomized SVD (Halko, Martinsson, Tropp 2011) | the top $r$ triples of a huge sparse matrix from a few matrix products with random vectors | `sklearn.utils.extmath.randomized_svd` |
| `lstsq` | `numpy.linalg.lstsq` (LAPACK `dgelsd`) | the minimum-norm solution for rank-deficient $A$, via the SVD | `numpy/linalg/_linalg.py` |
| `low_rank` for LoRA | PEFT's PiSSA initializer | the same balanced split on GPU, then the residual $W - BC$ frozen as the base weight | `peft/tuners/lora/layer.py` (`pissa_init`) |
