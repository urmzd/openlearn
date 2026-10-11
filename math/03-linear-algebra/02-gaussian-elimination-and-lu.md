<!-- ol:module M03.2 -->
# Gaussian elimination and LU with partial pivoting

## Overview

| | |
|---|---|
| **Module** | `M03.2` · build · Python · Pass 2 · 3 to 4 h |
| **You build** | `python/tinyllm/linalg/lu.py`: `lu(A)` returns $P, L, U$ with $PA = LU$; `lu_solve(P, L, U, b)` solves $Ax = b$ by two triangular solves |
| **Contract** | [`course/contracts/py/tinyllm/linalg/lu.pyi`](../../course/contracts/py/tinyllm/linalg/lu.pyi) |
| **Tests** | `course/tests/M03.2/test_lu.py` (what they check: section 4) |
| **Needs** | nothing to build first. Reading: `M03.1` (matrices, the product, row-major layout) |
| **Used by** | `M03.4` inverse iteration (one `lu`, a `lu_solve` per step) · later `M07.7` (one linear solve per IRLS step of logistic regression) and `M10.5` (Newton steps) |
| **Milestone** | `MS-P2` (the foundations gate) |
| **Optional depth** | Trefethen and Bau, *Numerical Linear Algebra*, lectures 20 to 22 (elimination, pivoting, stability); Strang, *Introduction to Linear Algebra*, ch. 2; Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 9 |

## Key Takeaways

- **Gaussian elimination** turns $Ax = b$ into an upper triangular system by subtracting multiples of a pivot row from the rows below it; the multipliers form a **unit lower triangular** $L$ and the result is $U$, with $A = LU$ when no rows move (`test_hand_example_factors`).
- **Partial pivoting** first swaps the row with the largest $\lvert\text{entry}\rvert$ into the pivot position, so every multiplier has $\lvert \ell_{ik} \rvert \le 1$; the swaps are a permutation matrix $P$ and $PA = LU$ (`test_pa_equals_lu_on_random_matrices`).
- Without pivoting, a zero pivot stops elimination on an invertible matrix and a tiny one destroys the answer (`test_zero_leading_pivot_needs_a_swap`, `test_tiny_pivot_loses_everything_without_partial_pivoting`).
- Factor once in $\frac{2}{3}n^3$ operations, then each new right-hand side costs only $2n^2$: forward substitution $Ly = Pb$, back substitution $Ux = y$ (`test_hand_example_solve`, `test_solve_matches_numpy`).
- A **singular** matrix still factors, with a zero on $U$'s diagonal; solving with it must fail loudly (`test_singular_matrix_factors_but_does_not_solve`).

## How to work this chapter

```bash
ol start M03.2              # stubs python/tinyllm/linalg/lu.py into your repo
ol tests M03.2              # read the test catalog first: rung R0, you write no tests here
ol check M03.2              # exit code is the verdict
ol diff  M03.2              # after passing: your code against the reference
```

---

## 1. Why now

Your system can multiply matrices (`M03.1`) but cannot undo a multiplication: given $A$ and $b$, find the $x$ with $Ax = b$. That is the core step of every second-order method you will meet. Logistic regression by IRLS (`M07.7`, the classifier behind the usage-policy head) solves one linear system per iteration; Newton's method (`M10.5`) solves one per step; and the next module, `M03.4`, finds the eigenvalue of a matrix nearest a chosen number by solving with the same matrix again and again. Calling `np.linalg.solve` would work, but the course builds the method from first principles so you can see why it is fast (factor once, solve many times) and why the naive version of it silently returns garbage on perfectly ordinary matrices.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $A$ | a square matrix, $a_{ij}$ its entries | `float64[n, n]` |
| $x, b$ | the unknown and the right-hand side of $Ax = b$ | `float64[n]` or `[n, k]` |
| $L$ | unit lower triangular: ones on the diagonal, zeros above | `float64[n, n]` |
| $U$ | upper triangular: zeros below the diagonal | `float64[n, n]` |
| $P$ | a permutation matrix: the identity with its rows reordered | `float64[n, n]` |
| $\ell_{ik}$ | the multiplier that eliminates entry $(i, k)$, stored in $L$ | scalar |
| $u_{kk}$ | the $k$-th pivot | scalar |
| $\varepsilon$ | float64 unit roundoff, $2^{-53} \approx 1.1 \times 10^{-16}$ | |

### 2.1 Elimination is a sequence of row operations

Subtracting $\ell$ times row $k$ from row $i$ does not change the solution set of $Ax = b$ when you do the same to $b$: it combines two true equations into a third. Gaussian elimination uses that operation to clear the entries below the diagonal, one column at a time. For column $k$, the **pivot** is $u_{kk}$, and for each row $i > k$,

$$\ell_{ik} = \frac{a_{ik}}{u_{kk}}, \qquad \text{row}_i \leftarrow \text{row}_i - \ell_{ik}\, \text{row}_k .$$

After $n - 1$ columns the matrix is upper triangular, $U$.

### 2.2 The multipliers form L

The row operation "row $i$ minus $\ell$ times row $k$" is multiplication on the left by $E = I - \ell\, e_i e_k^\top$, the identity with $-\ell$ at position $(i, k)$. Its inverse is $I + \ell\, e_i e_k^\top$ (add the row back). Elimination computes $E_{\text{last}} \cdots E_1 A = U$, so

$$A = E_1^{-1} \cdots E_{\text{last}}^{-1}\, U = L U,$$

and the product of those inverses, taken in this order, is simply the identity with every multiplier $\ell_{ik}$ placed at position $(i, k)$: no arithmetic is needed to form $L$, you write each multiplier where it was used. $L$ is the record of elimination, which is what lets you replay it on any $b$ later.

### 2.3 Pivoting

Elimination divides by $u_{kk}$, and nothing guarantees it is non-zero. $\begin{pmatrix} 0 & 1 \\ 1 & 1 \end{pmatrix}$ is invertible, yet its first pivot is 0. Swapping the two rows (equations may be listed in any order) fixes it. A pivot that is merely *small* is worse, because nothing fails visibly. Take $\epsilon = 10^{-20}$:

$$\begin{pmatrix} \epsilon & 1 \\ 1 & 1 \end{pmatrix} x = \begin{pmatrix} 1 \\ 2 \end{pmatrix}, \qquad x \approx (1, 1).$$

Pivoting on $\epsilon$ gives $\ell_{21} = 10^{20}$, and $u_{22} = 1 - 10^{20}$ rounds to $-10^{20}$: the 1 in $a_{22}$ is lost, because the spacing of float64 numbers near $10^{20}$ is about $16\,000$. Back substitution then gets $x_2 = 1$ and $x_1 = (1 - x_2)/\epsilon = 0$. Every digit of $x_1$ is wrong.

**Partial pivoting** chooses, for column $k$, the row $p \ge k$ with the largest $\lvert u_{pk} \rvert$ and swaps it into row $k$ before eliminating. Then every multiplier satisfies $\lvert \ell_{ik} \rvert \le 1$, row operations never amplify entries by more than a factor 2 per step, and in practice elimination with partial pivoting is **backward stable**: the computed $x$ solves a system $(A + \delta A)x = b$ with $\lVert \delta A \rVert$ a small multiple of $\varepsilon \lVert A \rVert$. Three details make it right:

- Search **rows $k$ and below** only. Rows above $k$ are finished pivot rows.
- Compare **absolute values**. $-3$ is a better pivot than $1$.
- Swap the whole rows of the working matrix, and also the multipliers already stored in $L$ for those two rows (columns $0$ to $k - 1$): a multiplier belongs to its equation, and the equation moved. $L$'s diagonal stays where it is.

Ties go to the lowest row index (`np.argmax` returns the first maximum), which makes $P$ deterministic. If the best $\lvert u_{pk} \rvert$ is 0, the whole column below the diagonal is already zero: there is nothing to eliminate, so skip it. The matrix is then singular, and $U$ carries a 0 on its diagonal.

### 2.4 Permutation matrices and solving

Record the swaps in an array `perm` where row $i$ of the permuted matrix is row `perm[i]` of $A$; the permutation matrix has $P_{i,\mathrm{perm}[i]} = 1$, so $(PA)_{i,:} = A_{\mathrm{perm}[i],:}$. $P$ is orthogonal, $P^{-1} = P^\top$, and in general $P \ne P^\top$ (a 3-cycle is the smallest example).

With $PA = LU$, the system $Ax = b$ becomes $LUx = Pb$, solved in two triangular steps:

$$\text{forward:}\quad y_i = (Pb)_i - \sum_{j < i} \ell_{ij}\, y_j \quad (i = 0, 1, \dots), \qquad \text{back:}\quad x_i = \frac{y_i - \sum_{j > i} u_{ij}\, x_j}{u_{ii}} \quad (i = n-1, \dots, 0).$$

Forward substitution divides by nothing ($L$ has a unit diagonal); back substitution divides by each pivot, so a zero pivot means the system has no unique solution and `lu_solve` raises. Several right-hand sides at once ($b$ of shape $[n, k]$) use the same formulas column by column.

### 2.5 Cost

Eliminating column $k$ updates an $(n-k-1) \times (n-k)$ block with one multiply and one subtract per entry, so the factorization costs $\sum_k 2(n-k)^2 \approx \frac{2}{3} n^3$ floating-point operations. Each triangular solve costs about $n^2$. For $n = 1000$: about $6.7 \times 10^8$ operations to factor and $2 \times 10^6$ per solve. That 300-fold gap is why `M03.4` factors once and solves at every step.

## 3. Worked example by hand

$$A = \begin{pmatrix} 2 & 1 & 1 \\ 4 & -6 & 0 \\ -2 & 7 & 2 \end{pmatrix}, \qquad b = \begin{pmatrix} 5 \\ -2 \\ 9 \end{pmatrix}.$$

**Column 0.** The candidates are $2, 4, -2$; the largest absolute value is $4$ in row 1, so swap rows 0 and 1 (`perm = [1, 0, 2]`). The pivot is 4; the multipliers are $\ell_{10} = 2/4 = 0.5$ and $\ell_{20} = -2/4 = -0.5$:

$$\text{row}_1: (2, 1, 1) - 0.5\,(4, -6, 0) = (0, 4, 1), \qquad \text{row}_2: (-2, 7, 2) + 0.5\,(4, -6, 0) = (0, 4, 2).$$

**Column 1.** The candidates (rows 1 and 2) are $4$ and $4$: a tie, so row 1 stays. $\ell_{21} = 4/4 = 1$ and $\text{row}_2: (0, 4, 2) - (0, 4, 1) = (0, 0, 1)$.

$$P = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}, \quad L = \begin{pmatrix} 1 & 0 & 0 \\ 0.5 & 1 & 0 \\ -0.5 & 1 & 1 \end{pmatrix}, \quad U = \begin{pmatrix} 4 & -6 & 0 \\ 0 & 4 & 1 \\ 0 & 0 & 1 \end{pmatrix}.$$

Check one entry of $LU$: row 2 of $L$ times column 0 of $U$ is $-0.5 \cdot 4 = -2 = (PA)_{20}$. This is `test_hand_example_factors`.

**Solve.** $Pb = (-2, 5, 9)$. Forward: $y_0 = -2$, $y_1 = 5 - 0.5 \cdot (-2) = 6$, $y_2 = 9 - (-0.5)(-2) - 1 \cdot 6 = 2$. Back: $x_2 = 2 / 1 = 2$, $x_1 = (6 - 1 \cdot 2) / 4 = 1$, $x_0 = (-2 - (-6) \cdot 1 - 0 \cdot 2) / 4 = 1$. So $x = (1, 1, 2)$, and indeed $2 + 1 + 2 = 5$, $4 - 6 + 0 = -2$, $-2 + 7 + 4 = 9$. This is `test_hand_example_solve`. Every number here is a small dyadic fraction, so the code gets them exactly.

## 4. The interface

```python
def lu(A: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
    """P, L, U (float64 [n, n]) with P @ A == L @ U; |L| <= 1; ties to the lowest
    row; zero columns skipped (singular A still factors). A is not modified."""

def lu_solve(P: ArrayLike, L: ArrayLike, U: ArrayLike, b: ArrayLike) -> NDArray:
    """x with A @ x == b for b of shape [n] or [n, k]. ValueError on a zero pivot."""
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_factors` | unit, smoke | section 3's $P$, $L$, $U$ exactly | you and the tests agree on the pivot and tie rules |
| `test_hand_example_solve` | unit, smoke | $x = (1, 1, 2)$ exactly | forward then back substitution |
| `test_pa_equals_lu_on_random_matrices` | property | $PA = LU$, $P$ a permutation, $L$ unit lower with $\lvert L \rvert \le 1$, $U$ upper, sizes 1 to 12 | the structure every caller relies on |
| `test_solve_matches_numpy` | differential | against `np.linalg.solve`, one and three right-hand sides, $n$ up to 40 | IRLS in `M07.7` |
| `test_zero_leading_pivot_needs_a_swap` | boundary | $\begin{pmatrix} 0 & 1 \\ 1 & 1 \end{pmatrix}$ factors and solves | pivoting is required, not optional |
| `test_tiny_pivot_loses_everything_without_partial_pivoting` | boundary | the $\epsilon = 10^{-20}$ system gives $(1, 1)$ | silent garbage without pivoting |
| `test_pivot_uses_absolute_value` | boundary | $-3$ beats $1$ as the pivot | $\lvert L \rvert \le 1$ |
| `test_pivot_search_ignores_finished_rows` | boundary | a large entry in a finished row is not chosen | the search covers rows $k$ and below |
| `test_singular_matrix_factors_but_does_not_solve` | boundary | a rank-2 matrix and the zero matrix factor; solving raises | a zero pivot is reported, never divided by |
| `test_input_is_not_modified_and_shapes_are_checked` | boundary | $A$ unchanged; non-square inputs and a wrong-length $b$ raise | callers reuse $A$ |
| `test_permutation_applied_to_b_not_its_transpose` | unit | a 3-cycle permutation: $P b$, not $P^\top b$ | the permutation direction |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| no pivoting at all | division by zero on invertible matrices, or every digit lost to a tiny pivot | `test_tiny_pivot_loses_everything_without_partial_pivoting` (mutant `s01`) |
| choosing the largest value instead of the largest absolute value | multipliers above 1 and error growth | `test_pivot_uses_absolute_value` (mutant `s02`) |
| swapping rows of $U$ but not the multipliers already in $L$ | $PA \ne LU$ as soon as a later column swaps | `test_pa_equals_lu_on_random_matrices` (mutant `s03`) |
| applying $P^\top$ to $b$ instead of $P$ | right answers whenever $P$ is its own inverse, wrong otherwise | `test_permutation_applied_to_b_not_its_transpose` (mutant `s04`) |
| forgetting to divide by the pivot in back substitution | $x$ scaled wrongly except where pivots are 1 | `test_hand_example_solve` (mutant `s05`) |
| storing the multiplier with the wrong sign | $L U \ne PA$ | `test_hand_example_factors` (mutant `s06`) |
| dividing by a zero pivot when solving | `inf` and `nan` instead of an error | `test_singular_matrix_factors_but_does_not_solve` (mutant `s07`) |
| searching the whole column for the pivot | a finished row is swapped back down | `test_pivot_search_ignores_finished_rows` (mutant `s08`) |
| eliminating in the caller's array | the next solve with $A$ uses $U$ | `test_input_is_not_modified_and_shapes_are_checked` (mutant `s09`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M03.1` | row-major matrices and the matrix product (reading) |
| Forward | `M03.4` | `inverse_iteration` factors $A - \sigma I$ once and calls `lu_solve` at every step |
| Forward | `M07.7` | IRLS solves $(X^\top W X)\,\delta = X^\top (y - p)$ at every iteration of logistic regression |
| Forward | `M10.5` | Newton steps solve with the Hessian |

If you skip this module, `ol check M03.4` stops with `BLOCKED ... needs M03.2`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `lu` | LAPACK `dgetrf` | blocked (right-looking) factorization that spends almost all its time in matrix-matrix products, so it runs near the speed of `M03.1`'s GEMM | LAPACK `SRC/dgetrf.f`, `dgetrf2.f` |
| `lu_solve` | LAPACK `dgetrs`, `scipy.linalg.lu_solve` | triangular solves with many right-hand sides at once | `scipy/linalg/_decomp_lu.py` |
| partial pivoting | rook and complete pivoting | stronger growth bounds at extra search cost | Higham, ch. 9 |
| dense LU | sparse LU (SuperLU, UMFPACK) | reorders rows and columns to keep $L$ and $U$ sparse | `scipy.sparse.linalg.splu` |
| explicit solve | Cholesky for symmetric positive definite systems | half the work, no pivoting needed; the natural choice for IRLS's $X^\top W X$ | LAPACK `dpotrf` |
