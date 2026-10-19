# Gaussian elimination and LU with partial pivoting

The previous section, [Vectors, matrices, row-major layout, and matmul](01-vectors-matrices-and-matmul.md), let you multiply matrices. It did not let you undo a multiplication: given $A$ and $b$, find the $x$ with $Ax = b$. That is the core step of every second-order method you will meet. Logistic regression by IRLS, in [Logistic regression (IRLS), ROC-AUC, calibration](../08-probability-statistics/08-logistic-regression-roc-auc-calibration.md) (the classifier behind the usage-policy head), solves one linear system per iteration. Newton's method solves one per step. And the section after this one, [Eigenvalues, power iteration, and the spectral radius](04-eigenvalues-and-power-iteration.md), finds the eigenvalue of a matrix nearest a chosen number by solving with the same matrix again and again.

Calling `np.linalg.solve` would work. This section builds the method from first principles instead, so you can see why it is fast (factor once, solve many times) and why the naive version silently returns garbage on perfectly ordinary matrices. By the end you will have `lu(A)`, which returns $P, L, U$ with $PA = LU$, and `lu_solve(P, L, U, b)`, which solves $Ax = b$ by two triangular solves. The solution is [`lu.py`](../../../../course/solutions/python/tinyllm/linalg/lu.py) and its contract is [`lu.pyi`](../../../../course/contracts/py/tinyllm/linalg/lu.pyi). Nothing needs to be built first, but the matrix product and row-major layout from the previous section are assumed.

> **Learning objectives**
> - Carry out Gaussian elimination with partial pivoting by hand and read off $P$, $L$, and $U$ with $PA = LU$.
> - Explain why the multipliers are exactly the entries of $L$, and why pivoting bounds them by 1.
> - Show, with a concrete $10^{-20}$ pivot, how elimination without pivoting loses every digit of the answer.
> - Solve $Ax = b$ by forward and back substitution, and say why a zero pivot must raise.
> - Compare the cost of factoring ($\frac{2}{3}n^3$) with the cost of each solve ($2n^2$).

## Notation

The section uses a small set of symbols throughout:

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

## Elimination is a sequence of row operations

Subtracting $\ell$ times row $k$ from row $i$ does not change the solution set of $Ax = b$, provided you do the same to $b$: it combines two true equations into a third. Gaussian elimination uses that operation to clear the entries below the diagonal, one column at a time. For column $k$, the **pivot** is $u_{kk}$, and for each row $i > k$,

$$\ell_{ik} = \frac{a_{ik}}{u_{kk}}, \qquad \text{row}_i \leftarrow \text{row}_i - \ell_{ik}\, \text{row}_k .$$

After $n - 1$ columns the matrix is upper triangular. That is $U$.

## The multipliers form L

Elimination looks like a procedure, but it is also a factorization. The row operation "row $i$ minus $\ell$ times row $k$" is multiplication on the left by $E = I - \ell\, e_i e_k^\top$, the identity with $-\ell$ at position $(i, k)$. Its inverse is $I + \ell\, e_i e_k^\top$: add the row back. Elimination computes $E_{\text{last}} \cdots E_1 A = U$, so

$$A = E_1^{-1} \cdots E_{\text{last}}^{-1}\, U = L U,$$

and the product of those inverses, taken in this order, is simply the identity with every multiplier $\ell_{ik}$ placed at position $(i, k)$. No arithmetic is needed to form $L$: you write each multiplier where it was used. $L$ is the record of elimination, which is what lets you replay it on any $b$ later.

## Pivoting

So far the procedure divides by $u_{kk}$, and nothing guarantees that is non-zero. The matrix $\begin{pmatrix} 0 & 1 \\ 1 & 1 \end{pmatrix}$ is invertible, yet its first pivot is 0. Swapping the two rows fixes it, since equations may be listed in any order.

A pivot that is merely *small* is worse, because nothing fails visibly. Take $\epsilon = 10^{-20}$:

$$\begin{pmatrix} \epsilon & 1 \\ 1 & 1 \end{pmatrix} x = \begin{pmatrix} 1 \\ 2 \end{pmatrix}, \qquad x \approx (1, 1).$$

Pivoting on $\epsilon$ gives $\ell_{21} = 10^{20}$, and $u_{22} = 1 - 10^{20}$ rounds to $-10^{20}$. The 1 in $a_{22}$ is lost, because the spacing of float64 numbers near $10^{20}$ is about $16\,000$. Back substitution then gets $x_2 = 1$ and $x_1 = (1 - x_2)/\epsilon = 0$. Every digit of $x_1$ is wrong.

**Partial pivoting** fixes both problems. For column $k$ it chooses the row $p \ge k$ with the largest $\lvert u_{pk} \rvert$ and swaps it into row $k$ before eliminating. Then every multiplier satisfies $\lvert \ell_{ik} \rvert \le 1$, row operations never amplify entries by more than a factor 2 per step, and in practice elimination with partial pivoting is **backward stable**: the computed $x$ solves a system $(A + \delta A)x = b$ with $\lVert \delta A \rVert$ a small multiple of $\varepsilon \lVert A \rVert$. On the $\epsilon$ system, the `1, 1` row becomes the pivot, $\ell_{21} = 10^{-20}$, $u_{22} = 1 - 10^{-20} \approx 1$ keeps its information, and back substitution returns $x \approx (1, 1)$.

Three details make pivoting right, and each is a place where implementations go wrong:

- Search **rows $k$ and below** only. Rows above $k$ are finished pivot rows.
- Compare **absolute values**. $-3$ is a better pivot than $1$.
- Swap the whole rows of the working matrix, and also the multipliers already stored in $L$ for those two rows (columns $0$ to $k - 1$). A multiplier belongs to its equation, and the equation moved. $L$'s diagonal stays where it is.

Ties go to the lowest row index (`np.argmax` returns the first maximum), which makes $P$ deterministic. If the best $\lvert u_{pk} \rvert$ is 0, the whole column below the diagonal is already zero: there is nothing to eliminate, so skip it. The matrix is then singular, and $U$ carries a 0 on its diagonal.

## Permutation matrices and solving

With pivoting in place, you need a way to record the swaps and use them when solving. Keep an array `perm` where row $i$ of the permuted matrix is row `perm[i]` of $A$. The permutation matrix has $P_{i,\mathrm{perm}[i]} = 1$, so $(PA)_{i,:} = A_{\mathrm{perm}[i],:}$. $P$ is orthogonal, $P^{-1} = P^\top$, and in general $P \ne P^\top$; a 3-cycle is the smallest example.

With $PA = LU$, the system $Ax = b$ becomes $LUx = Pb$, solved in two triangular steps:

$$\text{forward:}\quad y_i = (Pb)_i - \sum_{j < i} \ell_{ij}\, y_j \quad (i = 0, 1, \dots), \qquad \text{back:}\quad x_i = \frac{y_i - \sum_{j > i} u_{ij}\, x_j}{u_{ii}} \quad (i = n-1, \dots, 0).$$

Forward substitution divides by nothing, because $L$ has a unit diagonal. Back substitution divides by each pivot, so a zero pivot means the system has no unique solution and `lu_solve` raises. Several right-hand sides at once ($b$ of shape $[n, k]$) use the same formulas column by column.

## What it costs

Eliminating column $k$ updates an $(n-k-1) \times (n-k)$ block with one multiply and one subtract per entry, so the factorization costs $\sum_k 2(n-k)^2 \approx \frac{2}{3} n^3$ floating-point operations. Each triangular solve costs about $n^2$. For $n = 1000$, that is about $6.7 \times 10^8$ operations to factor and $2 \times 10^6$ per solve. That 300-fold gap is why inverse iteration in [Eigenvalues, power iteration, and the spectral radius](04-eigenvalues-and-power-iteration.md) factors once and solves at every step.

## Working it by hand

Now put it all together on a 3 by 3 system:

$$A = \begin{pmatrix} 2 & 1 & 1 \\ 4 & -6 & 0 \\ -2 & 7 & 2 \end{pmatrix}, \qquad b = \begin{pmatrix} 5 \\ -2 \\ 9 \end{pmatrix}.$$

**Column 0.** The candidates are $2, 4, -2$; the largest absolute value is $4$ in row 1, so swap rows 0 and 1 (`perm = [1, 0, 2]`). The pivot is 4, and the multipliers are $\ell_{10} = 2/4 = 0.5$ and $\ell_{20} = -2/4 = -0.5$:

$$\text{row}_1: (2, 1, 1) - 0.5\,(4, -6, 0) = (0, 4, 1), \qquad \text{row}_2: (-2, 7, 2) + 0.5\,(4, -6, 0) = (0, 4, 2).$$

**Column 1.** The candidates in rows 1 and 2 are $4$ and $4$: a tie, so row 1 stays. $\ell_{21} = 4/4 = 1$ and $\text{row}_2: (0, 4, 2) - (0, 4, 1) = (0, 0, 1)$. The factors are

$$P = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}, \quad L = \begin{pmatrix} 1 & 0 & 0 \\ 0.5 & 1 & 0 \\ -0.5 & 1 & 1 \end{pmatrix}, \quad U = \begin{pmatrix} 4 & -6 & 0 \\ 0 & 4 & 1 \\ 0 & 0 & 1 \end{pmatrix}.$$

Check one entry of $LU$: row 2 of $L$ times column 0 of $U$ is $-0.5 \cdot 4 = -2 = (PA)_{20}$.

**Solve.** $Pb = (-2, 5, 9)$. Forward: $y_0 = -2$, $y_1 = 5 - 0.5 \cdot (-2) = 6$, $y_2 = 9 - (-0.5)(-2) - 1 \cdot 6 = 2$. Back: $x_2 = 2 / 1 = 2$, $x_1 = (6 - 1 \cdot 2) / 4 = 1$, $x_0 = (-2 - (-6) \cdot 1 - 0 \cdot 2) / 4 = 1$. So $x = (1, 1, 2)$, and indeed $2 + 1 + 2 = 5$, $4 - 6 + 0 = -2$, $-2 + 7 + 4 = 9$. Every number here is a small dyadic fraction, so the solution reproduces them exactly.

## Designing the API

The interface splits factoring from solving, because that split is what makes the factor-once pattern possible:

```python
def lu(A: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
    """P, L, U (float64 [n, n]) with P @ A == L @ U; |L| <= 1; ties to the lowest
    row; zero columns skipped (singular A still factors). A is not modified."""

def lu_solve(P: ArrayLike, L: ArrayLike, U: ArrayLike, b: ArrayLike) -> NDArray:
    """x with A @ x == b for b of shape [n] or [n, k]. ValueError on a zero pivot."""
```

`lu` never raises on a singular matrix: a singular matrix still has a valid factorization with a zero on $U$'s diagonal, and that zero is useful information, since it makes the rank deficiency visible. Only the solve needs to divide by the pivot, so that is where the error belongs. Neither function modifies its inputs, because callers reuse $A$.

## Building it step by step

The reasoning chain runs straight from the principles. Elimination only ever subtracts multiples of a pivot row from rows below it, so the work is a loop over columns $k = 0, \dots, n-2$, each updating the trailing block `U[k+1:, k:]`. The multipliers are exactly the entries of $L$, so the solution writes each one into `L[i, k]` at the moment it is computed; no separate pass forms $L$. Dividing by the pivot is safe and stable only if the pivot is the largest $\lvert\cdot\rvert$ among the rows still being worked on. That forces three things: search `U[k:, k]` (not the whole column), compare absolute values, and swap the stored multipliers in `L[:, :k]` with the rows. A column whose best pivot is 0 has nothing to eliminate; skipping it, rather than raising, lets a singular matrix factor and leaves the 0 on $U$'s diagonal where `lu_solve` can see it. Swaps are cheap to record as an index array `perm` and convert to $P$ once at the end. Solving replays elimination on $Pb$ and then undoes $U$ from the bottom row up, checking for a zero pivot before any division.

The factorization works on a float64 copy so the caller's `A` survives:

```python
def lu(A: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
    U = np.array(A, dtype=np.float64)  # a copy: the caller's A is never touched
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError(f"lu: A must be a square matrix, got shape {U.shape}")
    n = U.shape[0]
    L = np.eye(n)
    perm = np.arange(n)  # row i of P A is row perm[i] of A
```

Each column picks its pivot from rows $k$ and below, by absolute value, ties to the first. `np.argmax` on the slice returns an offset into that slice, so adding `k` converts it back to a row index of `U`:

```python
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(U[k:, k])))  # offset by k: rows above are finished
        if U[p, k] == 0.0:
            continue  # the column is already zero below k: nothing to eliminate
```

The swap moves the working rows, the multipliers already stored for those rows, and the permutation record. Only columns `k:` of `U` need to move, since columns left of `k` are already zero in both rows; only columns `:k` of `L` hold multipliers, and the unit diagonal stays in place:

```python
        if p != k:
            U[[k, p], k:] = U[[p, k], k:]
            L[[k, p], :k] = L[[p, k], :k]   # a multiplier belongs to its equation
            perm[[k, p]] = perm[[p, k]]
```

Elimination of the whole column is one vectorised rank-1 update. On the worked example at $k = 0$ this computes `m = [0.5, -0.5]` and turns rows 1 and 2 into $(0, 4, 1)$ and $(0, 4, 2)$:

```python
        m = U[k + 1 :, k] / U[k, k]  # |m| <= 1 thanks to the pivot choice
        L[k + 1 :, k] = m            # store with a + sign: row_i -= m * row_k
        U[k + 1 :, k:] -= np.outer(m, U[k, k:])
        U[k + 1 :, k] = 0.0  # exactly zero by construction, not just rounded
    P = np.zeros((n, n))
    P[np.arange(n), perm] = 1.0     # P[i, perm[i]] = 1, so (P A)[i] = A[perm[i]]
    return P, L, U
```

The solve validates shapes, refuses a zero pivot before dividing, and runs the two substitutions. Because `y` is indexed by its first axis only, the same loop handles `b` of shape `[n]` and `[n, k]`:

```python
    d = np.diag(U)
    if np.any(d == 0.0):
        raise ValueError("lu_solve: U has a zero pivot, A is singular")
    # Forward substitution, L y = P b: unit diagonal, no division.
    y = P @ b                       # P, not P.T: replays the row swaps on b
    for i in range(n):
        y[i] = y[i] - L[i, :i] @ y[:i]
    # Back substitution, U x = y, from the last row up.
    x = y                           # in place: y[i] is no longer needed once x[i] is set
    for i in range(n - 1, -1, -1):
        x[i] = (x[i] - U[i, i + 1 :] @ x[i + 1 :]) / d[i]
    return x
```

`P @ b` allocates a new array, so the caller's `b` is not modified either. The full solution is [`lu.py`](../../../../course/solutions/python/tinyllm/linalg/lu.py).

### Why this design

The cost is $\frac{2}{3}n^3$ flops to factor, about $2n^2$ per right-hand side, and $O(n^2)$ memory for the factors. Separating `lu` from `lu_solve` is what lets inverse iteration and IRLS pay the cubic cost once.

You might be tempted to invert $A$ instead. Forming $A^{-1}$ costs about $2n^3$ and is less accurate than solving with the factors, and it throws away the zero-pivot signal. Complete pivoting, which searches the whole trailing block for the largest entry, gives a tighter growth bound but costs $O(n^3)$ comparisons and a column permutation as well; partial pivoting is backward stable in practice and is what LAPACK uses. Returning $P$ as a dense matrix matches the contract and makes $PA = LU$ directly checkable; production code returns the `perm` vector instead, and the solution keeps it internally for the same reason. The one tradeoff knowingly accepted is speed: the loops are Python-level over $k$ and $i$, with NumPy doing each row or block. That is slow for large $n$ but makes every step of the algorithm visible.

## What can go wrong

The test cases a correct solution must pass map directly onto the ideas above. The hand example must reproduce $P$, $L$, $U$ and $x = (1, 1, 2)$ exactly, which pins the pivot and tie rules and the order of forward then back substitution. On random matrices of sizes 1 to 12, the structural properties every caller relies on must hold: $PA = LU$, $P$ a permutation, $L$ unit lower with $\lvert L \rvert \le 1$, and $U$ upper. The answers must agree with `np.linalg.solve` for one and three right-hand sides with $n$ up to 40, which is what IRLS needs.

The boundary cases are where naive code breaks. The zero leading pivot $\begin{pmatrix} 0 & 1 \\ 1 & 1 \end{pmatrix}$ must factor and solve, which shows pivoting is required, not optional. The $\epsilon = 10^{-20}$ system must give $(1, 1)$. A negative candidate $-3$ must beat $1$ as the pivot. A large entry in a finished row must not be chosen, because the search covers rows $k$ and below only. A rank-2 matrix and the zero matrix must factor while solving with them raises. $A$ must be unchanged after the call, and non-square inputs or a wrong-length $b$ must raise. Finally, a 3-cycle permutation checks that the solve uses $Pb$ and not $P^\top b$: $P^\top$ is the inverse permutation and gives the same result only when $P$ is its own inverse (a product of disjoint swaps).

The common mistakes, their symptoms, and the line that prevents each are collected here:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| no pivoting at all | division by zero on invertible matrices, or every digit lost to a tiny pivot | always swaps the largest $\lvert\cdot\rvert$ into the pivot row |
| choosing the largest value instead of the largest absolute value | multipliers above 1 and error growth | searches `np.abs(U[k:, k])` |
| swapping rows of $U$ but not the multipliers already in $L$ | $PA \ne LU$ as soon as a later column swaps | swaps `L[[k, p], :k]` with the rows |
| applying $P^\top$ to $b$ instead of $P$ | right answers whenever $P$ is its own inverse, wrong otherwise | starts the forward pass from `y = P @ b` |
| forgetting to divide by the pivot in back substitution | $x$ scaled wrongly except where pivots are 1 | divides each back-substitution row by `d[i]` |
| storing the multiplier with the wrong sign | $L U \ne PA$ | stores `m` itself in `L` and subtracts `outer(m, row_k)` |
| dividing by a zero pivot when solving | `inf` and `nan` instead of an error | checks `np.any(d == 0.0)` before any division and raises |
| searching the whole column for the pivot | a finished row is swapped back down | searches only `U[k:, k]` and offsets by `k` |
| eliminating in the caller's array | the next solve with $A$ uses $U$ | works on `np.array(A, dtype=np.float64)`, a copy |

> **Warning:** Forgetting to swap the stored multipliers passes every test in which only one swap happens. The bug shows up only when a later column swaps again, so test with matrices that need several swaps.

## Where this leads

The next consumer is [Eigenvalues, power iteration, and the spectral radius](04-eigenvalues-and-power-iteration.md), whose `inverse_iteration` factors $A - \sigma I$ once and calls `lu_solve` at every step. Later, [Logistic regression (IRLS), ROC-AUC, calibration](../08-probability-statistics/08-logistic-regression-roc-auc-calibration.md) solves $(X^\top W X)\,\delta = X^\top (y - p)$ at every iteration of logistic regression, and Newton steps in optimization solve with the Hessian.

### Further reading

- LAPACK `dgetrf` (`SRC/dgetrf.f`, `dgetrf2.f`): a blocked, right-looking factorization that spends almost all its time in matrix-matrix products, so it runs near the speed of GEMM.
- LAPACK `dgetrs` and `scipy.linalg.lu_solve` (`scipy/linalg/_decomp_lu.py`): triangular solves with many right-hand sides at once.
- Rook and complete pivoting (Higham, ch. 9): stronger growth bounds at extra search cost.
- Sparse LU (SuperLU, UMFPACK; `scipy.sparse.linalg.splu`): reorders rows and columns to keep $L$ and $U$ sparse.
- Cholesky for symmetric positive definite systems (LAPACK `dpotrf`): half the work and no pivoting needed; the natural choice for IRLS's $X^\top W X$.
- Trefethen and Bau, *Numerical Linear Algebra*, lectures 20 to 22 (elimination, pivoting, stability); Strang, *Introduction to Linear Algebra*, ch. 2; Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 9.

## Key terms

- **Pivot**: the diagonal entry $u_{kk}$ that elimination divides by in column $k$.
- **Multiplier**: $\ell_{ik} = a_{ik}/u_{kk}$, the amount of the pivot row subtracted from row $i$; stored in $L$.
- **Unit lower triangular**: lower triangular with ones on the diagonal.
- **Partial pivoting**: swapping the row with the largest absolute entry (among rows $k$ and below) into the pivot position before eliminating.
- **Permutation matrix**: the identity with its rows reordered; orthogonal, with $P^{-1} = P^\top$.
- **Backward stable**: the computed answer is the exact answer to a slightly perturbed problem, $(A + \delta A)x = b$ with $\lVert \delta A \rVert$ a small multiple of $\varepsilon \lVert A \rVert$.

## Summary

- Gaussian elimination turns $Ax = b$ into an upper triangular system by subtracting multiples of a pivot row from the rows below it; the multipliers form a unit lower triangular $L$ and the result is $U$, with $A = LU$ when no rows move.
- Partial pivoting swaps the row with the largest $\lvert\text{entry}\rvert$ into the pivot position, so every $\lvert \ell_{ik} \rvert \le 1$; the swaps form $P$ and $PA = LU$.
- Without pivoting, a zero pivot stops elimination on an invertible matrix and a tiny one destroys the answer.
- Factor once in $\frac{2}{3}n^3$ operations; each new right-hand side then costs only $2n^2$: forward substitution $Ly = Pb$, back substitution $Ux = y$.
- A singular matrix still factors, with a zero on $U$'s diagonal; solving with it must fail loudly.

## Review questions

1. Why does the pivot search use `U[k:, k]` and add `k` to the result?

   <details><summary>Answer</summary>

   Rows above $k$ are finished pivot rows; swapping one of them back down would undo earlier elimination. `np.argmax` on the slice returns an offset into that slice, so `k` converts it back to a row index of `U`.

   </details>

2. Why must `L[[k, p], :k]` be swapped along with the rows of `U`?

   <details><summary>Answer</summary>

   Each stored multiplier says how much of an earlier pivot row was subtracted from a particular equation. When that equation moves to a different row, its history must move with it, or $LU$ reconstructs the wrong rows and $PA \ne LU$ after the first later swap.

   </details>

3. On the $\epsilon = 10^{-20}$ system, what does pivoting change, number by number?

   <details><summary>Answer</summary>

   Row 1 (the `1, 1` row) becomes the pivot, so $\ell_{21} = 10^{-20}$ instead of $10^{20}$. Then $u_{22} = 1 - 10^{-20} \approx 1$ keeps its information, and back substitution gives $x \approx (1, 1)$ instead of $(0, 1)$.

   </details>

4. Why does `lu` skip a zero column instead of raising, while `lu_solve` raises?

   <details><summary>Answer</summary>

   A singular matrix still has a valid factorization $PA = LU$ with a zero on $U$'s diagonal; that is useful information (the rank deficiency is visible). Only the solve needs to divide by the pivot, so that is where the error belongs.

   </details>

5. Why `y = P @ b` and not `P.T @ b`?

   <details><summary>Answer</summary>

   $PA = LU$ means the factored equations are in the order `perm`; $Ax = b$ becomes $LUx = Pb$. $P^\top$ is the inverse permutation and gives the same result only when $P$ is its own inverse (a product of disjoint swaps), so a 3-cycle exposes the difference.

   </details>

6. For $n = 1000$, roughly how many operations does factoring cost compared with one solve, and what design does that justify?

   <details><summary>Answer</summary>

   About $6.7 \times 10^8$ to factor and $2 \times 10^6$ per solve, a 300-fold gap. It justifies separating `lu` from `lu_solve` so a caller that solves many times with the same matrix factors only once.

   </details>
