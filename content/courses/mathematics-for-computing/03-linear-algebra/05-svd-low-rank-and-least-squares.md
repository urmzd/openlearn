# SVD, Eckart-Young low rank, and least squares

Sooner or later you start treating text as numbers you can factor. [word2vec skip-gram with negative sampling, and the PPMI-SVD baseline](../../build-an-llm/p02-statistical-lm/03-word2vec-and-ppmi-svd.md) in *Build Your Own LLM System* counts which words appear near which others in a matrix with tens of thousands of rows and asks for 64 numbers per word that keep most of its structure. That is the best rank-64 approximation of the matrix, and the singular value decomposition is the only tool that gives it with a proof of optimality.

The same decomposition comes back three more times. [LoRA with PiSSA init and merge](../../build-an-llm/p06-objectives/06-lora-pissa-and-merge.md) starts a LoRA adapter on the top singular directions of a weight matrix, [Multi-head latent attention](../../build-an-llm/p07-modern-block/06-multi-head-latent-attention.md) compresses attention's keys and values into a low-rank latent, and the [TinyStories capstone](../../build-an-llm/capstones/01-tinystories.md) fits a scaling law $\log L = a + b \log N$ to a handful of runs by least squares. The pieces are already in place: orthogonal matrices and QR from [Orthogonality, Householder QR, and orthogonal initialization](03-orthogonality-and-householder-qr.md), and eigenvalues from [Eigenvalues, power iteration, and the spectral radius](04-eigenvalues-and-power-iteration.md). This section puts them together. The first half of the [linear algebra problem set, part a](90-problem-set-a.md) is useful practice beforehand.

The solution is [`svd.py`](../../../../course/solutions/python/tinyllm/linalg/svd.py), with contract [`svd.pyi`](../../../../course/contracts/py/tinyllm/linalg/svd.pyi). It provides `svd(A)`, the reduced SVD by one-sided Jacobi rotations with singular values descending and a fixed sign rule; `low_rank(A, r)`, the best rank-$r$ approximation as balanced factors $BC$; and `lstsq(A, b)`, least squares by QR using `qr_householder` from [`qr.py`](../../../../course/solutions/python/tinyllm/linalg/qr.py).

> **Learning objectives**
> - Read $A = U\Sigma V^\top$ geometrically and relate singular values to the eigenvalues of $A^\top A$ and to the spectral and Frobenius norms.
> - Compute an SVD by one-sided Jacobi rotations, by hand for a 2 by 2 matrix, with a scale-free convergence test.
> - State the Eckart-Young theorem, compute the error of a truncated SVD, and explain the balanced split used by PiSSA.
> - Solve least squares by QR and explain why the normal equations lose accuracy.

## Notation

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
| $Q, R$ | the QR factors from the previous sections | `float64[m, k]`, `float64[k, n]` |

## What the SVD says

Every real $m \times n$ matrix factors as

$$A = U \Sigma V^\top = \sum_{i=1}^{k} \sigma_i\, u_i v_i^\top ,$$

with orthonormal columns in $U$ and $V$ and non-negative $\sigma_i$ in descending order. Read it right to left on a vector $x$: $V^\top x$ takes the coordinates of $x$ along the directions $v_i$, $\Sigma$ stretches coordinate $i$ by $\sigma_i$, and $U$ sends the result out along the directions $u_i$. So $A v_i = \sigma_i u_i$: rotate, stretch along the axes, rotate again. The unit sphere goes to an ellipsoid whose semi-axes are $\sigma_i u_i$. This is the **reduced** SVD ($U$ is $m \times k$); the full one pads $U$ to a square matrix, which nothing in this book needs.

Each pair $(u_i, v_i)$ can be negated together without changing $A$, since $(-u_i)\sigma_i(-v_i)^\top = u_i \sigma_i v_i^\top$. When the singular values are distinct this is the only freedom, so the contract fixes it with a **sign rule**: the entry of largest absolute value in each row of $V^\top$ is positive (the first such entry on ties). With it, the solution's factors and LAPACK's (after the same rule) agree entry by entry.

## Singular values, eigenvalues, and norms

Multiply out $A^\top A = V \Sigma^\top U^\top U \Sigma V^\top = V \Sigma^2 V^\top$: the $v_i$ are eigenvectors of the symmetric matrix $A^\top A$ with eigenvalues $\sigma_i^2$ (problem 8 of the [linear algebra problem set, part b](91-problem-set-b.md) proves it). So everything the previous section said about eigenvalues applies, with two differences: singular values exist for every matrix, square or not, and they are never negative or complex.

Two norms come straight from them. For a unit $x$ with coordinates $c = V^\top x$, $\lVert Ax \rVert^2 = \sum_i \sigma_i^2 c_i^2 \le \sigma_1^2$, with equality at $x = v_1$: the **spectral norm** is $\lVert A \rVert_2 = \sigma_1$, the most $A$ can stretch a unit vector. Orthogonal factors do not change the sum of squares of entries, so the **Frobenius norm** is $\lVert A \rVert_F = \sqrt{\sum_i \sigma_i^2}$.

That connection suggests a shortcut: form $A^\top A$ and call an eigensolver. It would compute the SVD, but badly. Squaring the singular values squares the condition number, and a $\sigma_i$ below $\sqrt{\varepsilon}\,\sigma_1 \approx 10^{-8}\sigma_1$ drowns in the rounding of $\sigma_1^2$. Jacobi works on $A$ itself.

## One-sided Jacobi

Work on a tall matrix ($m \ge n$; for a wide one, decompose $A^\top$ and swap $U$ and $V$ at the end, since $n$ orthogonal columns can only exist in $\mathbb{R}^m$ when $m \ge n$). Start with $W = A$ and $V = I$, and keep the invariant $W = AV$ with $V$ orthogonal. If all columns of $W$ were mutually orthogonal, their lengths would be the singular values and their directions the $u_i$: $W = U\Sigma$, so $A = U \Sigma V^\top$.

To make columns $i$ and $j$ orthogonal, rotate them in their plane:

$$w_i \leftarrow c\, w_i - s\, w_j, \qquad w_j \leftarrow s\, w_i + c\, w_j ,$$

and apply the same rotation to columns $i$ and $j$ of $V$, so $W = AV$ still holds; the rotation is orthogonal, so $V$ stays orthogonal. The new inner product is $cs(\alpha - \beta) + (c^2 - s^2)\gamma$. Setting it to zero with $t = s/c$ gives $t^2 + 2\zeta t - 1 = 0$, $\zeta = (\beta - \alpha) / (2\gamma)$. Take the smaller root,

$$t = \frac{\operatorname{sign}(\zeta)}{\lvert \zeta \rvert + \sqrt{1 + \zeta^2}}, \qquad c = \frac{1}{\sqrt{1 + t^2}}, \qquad s = c\,t ,$$

with $\operatorname{sign}(0) = +1$. It is the rotation by at most $45°$, the one that converges. A **sweep** visits every pair $i < j$ once. A rotation for one pair can spoil an earlier pair a little, but the off-diagonal mass shrinks every sweep and, near the end, quadratically; random $30 \times 12$ matrices finish in 7 or 8 sweeps, the last one only confirming that nothing needs rotating.

When is a pair "orthogonal enough"? The test must be **relative**: rotate only when $\lvert \gamma \rvert > m \varepsilon \sqrt{\alpha \beta}$, that is, when the cosine of the angle between the columns exceeds $m\varepsilon$. A fixed threshold such as $\lvert\gamma\rvert > 10^{-15}$ skips every rotation on a matrix scaled by $10^{-12}$ (all inner products are about $10^{-24}$) and returns garbage; on a matrix scaled by $10^{12}$ it never stops. Stop after the first sweep that rotates nothing (the contract caps sweeps at 64). Finally, $\sigma_j = \lVert w_j \rVert$, sorted descending (stably), $u_j = w_j / \sigma_j$, and the sign rule.

## Rank deficiency

The last step divides by $\sigma_j$, which fails when $A$ is rank deficient. If $A$ has rank $\rho < k$, then $k - \rho$ singular values are zero and their columns $w_j$ are (numerically) zero: $w_j / \sigma_j$ is $0/0$. The contract treats $\sigma_j \le m \varepsilon \sigma_1$ as zero and fills $u_j$ with a unit vector orthogonal to the $u$'s already chosen: take the first standard basis vector $e_i$ that has more than half its length left after projecting out the previous columns ($e \leftarrow e - U(U^\top e)$, done twice for accuracy), and normalize it. Any such completion is a valid SVD, because it multiplies a zero singular value; the contract fixes this one so the result is deterministic.

## Eckart-Young and the balanced split

Now the payoff. The truncated SVD $A_r = \sum_{i \le r} \sigma_i u_i v_i^\top$ has rank $r$, and $A - A_r = \sum_{i > r} \sigma_i u_i v_i^\top$ is itself an SVD, so

$$\lVert A - A_r \rVert_2 = \sigma_{r+1}, \qquad \lVert A - A_r \rVert_F = \sqrt{\textstyle\sum_{i > r} \sigma_i^2} .$$

The **Eckart-Young theorem** says no matrix of rank at most $r$ does better in either norm: the truncated SVD is the best low-rank approximation. That is why it is the right way to compress a co-occurrence matrix or initialize a low-rank adapter.

`low_rank` returns $A_r$ as two factors $B$ ($m \times r$) and $C$ ($r \times n$) with $BC = A_r$. Many splits multiply to the same product; the contract uses the **balanced** one,

$$B = U_r \Sigma_r^{1/2}, \qquad C = \Sigma_r^{1/2} V_r^\top ,$$

so $B^\top B = C C^\top = \Sigma_r$ and both factors carry the same scale. PiSSA initializes LoRA this way because gradient descent on $BC$ updates each factor in proportion to the other's size; a split such as $(U_r\Sigma_r, V_r^\top)$ trains one factor much faster than the other.

## Least squares by QR

The last tool solves systems with more equations than unknowns. For a tall $A$ with independent columns, $Ax = b$ usually has no solution; least squares picks $x$ minimizing $\lVert Ax - b \rVert$. The minimizer makes the residual $r = b - Ax$ orthogonal to every column of $A$: $A^\top r = 0$, the **normal equations** $A^\top A x = A^\top b$. Solving them as written is the classic mistake. $\kappa(A^\top A) = \kappa(A)^2$, so a polynomial fit with $\kappa(A) \approx 3.5 \times 10^6$ (degree 9 at 40 points of $[0, 1]$) recovers its coefficients only to about $10^{-3}$.

QR avoids the square. With $A = QR$, $Q$ has orthonormal columns, so the residual splits into a part inside the column space and a part orthogonal to it, and only the first depends on $x$:

$$\lVert Ax - b \rVert^2 = \lVert Rx - Q^\top b \rVert^2 + \lVert (I - QQ^\top) b \rVert^2 .$$

So $x$ solves the triangular system $R x = Q^\top b$, by **back substitution** from the last row up: $x_i = (z_i - \sum_{j > i} R_{ij} x_j) / R_{ii}$ with $z = Q^\top b$. The error now grows like $\kappa(A)\varepsilon$: $5 \times 10^{-10}$ for the same fit. If some $\lvert R_{ii} \rvert$ is tiny ($\le \max(m, n)\,\varepsilon \max_j \lvert R_{jj} \rvert$), the columns are dependent, the minimizer is not unique, and the contract raises instead of dividing by almost zero. The test is relative to the largest pivot so it does not depend on the units of $A$.

## Working it by hand

Take

$$A = \begin{pmatrix} 3 & 0 \\ 4 & 5 \end{pmatrix}.$$

**Jacobi.** There is one pair, $(0, 1)$. The table follows the rotation quantity by quantity:

| Quantity | Value |
|---|---|
| $\alpha = \lVert w_0 \rVert^2$, $\beta = \lVert w_1 \rVert^2$, $\gamma = w_0^\top w_1$ | $9 + 16 = 25$, $0 + 25 = 25$, $0 + 20 = 20$ |
| $\zeta = (\beta - \alpha)/(2\gamma)$ | $0$, so $t = 1$, $c = s = 1/\sqrt{2}$ (a $45°$ rotation) |
| new $w_0 = c\,w_0 - s\,w_1$ | $(3 - 0,\ 4 - 5)/\sqrt{2} = (3, -1)/\sqrt{2}$, length $\sqrt{5}$ |
| new $w_1 = s\,w_0 + c\,w_1$ | $(3 + 0,\ 4 + 5)/\sqrt{2} = (3, 9)/\sqrt{2}$, length $\sqrt{45} = 3\sqrt{5}$ |
| new $V$ columns | $v_0 = (1, -1)/\sqrt{2}$, $v_1 = (1, 1)/\sqrt{2}$ |
| second sweep | $w_0^\top w_1 = (9 - 9)/2 = 0$: nothing to rotate, stop |

Sorting puts $\sigma_1 = 3\sqrt{5} \approx 6.708$ first, then $\sigma_2 = \sqrt{5} \approx 2.236$. Check: $\sigma_1^2 + \sigma_2^2 = 50 = 9 + 16 + 25 = \lVert A \rVert_F^2$, and $\sigma_1\sigma_2 = 15 = \lvert \det A \rvert$. Dividing each $w$ by its length:

$$U = \frac{1}{\sqrt{10}}\begin{pmatrix} 1 & 3 \\ 3 & -1 \end{pmatrix}, \quad \Sigma = \begin{pmatrix} 3\sqrt{5} & 0 \\ 0 & \sqrt{5} \end{pmatrix}, \quad V^\top = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}.$$

The sign rule holds already: each row of $V^\top$ has its largest entries tied, and the first is positive.

**Rank 1.** $A_1 = \sigma_1 u_1 v_1^\top = 3\sqrt{5} \cdot \frac{1}{\sqrt{20}} \begin{pmatrix} 1 & 1 \\ 3 & 3 \end{pmatrix} = \begin{pmatrix} 1.5 & 1.5 \\ 4.5 & 4.5 \end{pmatrix}$. The error $A - A_1 = \begin{pmatrix} 1.5 & -1.5 \\ -0.5 & 0.5 \end{pmatrix} = \sigma_2 u_2 v_2^\top$ has spectral and Frobenius norm $\sqrt{2.25 + 2.25 + 0.25 + 0.25} = \sqrt{5} = \sigma_2$, as Eckart-Young says. The balanced factors are $B = 45^{1/4}(1, 3)^\top/\sqrt{10}$ and $C = 45^{1/4}(1, 1)/\sqrt{2}$.

**Least squares.** Fit a line $x_0 + x_1 t$ through $(0, 1)$, $(1, 2)$, $(2, 2)$: $A = \begin{pmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{pmatrix}$, $b = (1, 2, 2)$. By hand, the residual condition $A^\top(b - Ax) = 0$ reads $3x_0 + 3x_1 = 5$ and $3x_0 + 5x_1 = 6$, so $x_1 = 1/2$ and $x_0 = 7/6$. Check: the residual $b - Ax = (1 - 7/6,\ 2 - 5/3,\ 2 - 13/6) = (-1/6, 1/3, -1/6)$ sums to 0 (orthogonal to the first column) and $0 \cdot (-1/6) + 1 \cdot (1/3) + 2 \cdot (-1/6) = 0$ (orthogonal to the second). Writing out the normal equations is fine for a $2 \times 2$ system on paper with exact fractions; the solution still goes through QR, because in floating point the square of the condition number is what hurts.

## Designing the API

```python
def svd(A: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
    """Reduced SVD by one-sided Jacobi: U [m, k], S [k] descending, Vt [k, n],
    A == U @ diag(S) @ Vt, the largest |entry| of each Vt row positive."""

def low_rank(A: ArrayLike, r: int) -> tuple[NDArray, NDArray]:
    """B = U_r sqrt(S_r) [m, r], C = sqrt(S_r) Vt_r [r, n]: B @ C is the best rank-r approximation."""

def lstsq(A: ArrayLike, b: ArrayLike) -> NDArray:
    """argmin ||A x - b|| for a tall, full-column-rank A, by QR and back substitution."""
```

All three leave $A$ unchanged, since a LoRA initializer keeps the weight it decomposes. `svd` rejects 1-D and non-finite inputs, `low_rank` requires $1 \le r \le \min(m, n)$, and `lstsq` raises on dependent columns, a wide system, or a wrong-length $b$.

## Building it step by step

The design follows from the principles in order. The SVD must be accurate for small singular values, which rules out an eigensolver on $A^\top A$; working on $A$ directly with orthogonal rotations keeps every operation norm-preserving. If $W = AV$ with $V$ orthogonal and $W$'s columns mutually orthogonal, then $W = U\Sigma$ and the SVD is read off, so the algorithm keeps that invariant and drives the column inner products to zero, one pair at a time. For a pair, the rotation that zeroes $\gamma$ solves $t^2 + 2\zeta t - 1 = 0$; the smaller root keeps $\lvert\theta\rvert \le 45°$, which is what makes sweeps converge (in the worked example $\zeta = 0$, $t = 1$, a $45°$ rotation). "Orthogonal enough" must be scale-free, so the test compares $\lvert\gamma\rvert$ with $m\varepsilon\sqrt{\alpha\beta}$: a cosine, not an absolute size.

The algorithm needs $m \ge n$, so a wide matrix is decomposed as $A^\top$ with $U$ and $V$ swapped. Column lengths come out in sweep order, but Eckart-Young needs descending order, so sort stably for determinism. A column whose length is negligible relative to $\sigma_1$ cannot be normalized and gets a deterministic orthonormal completion. Signs are fixed last, on $V$, so the result is unique and comparable with LAPACK. `low_rank` is then the truncated SVD split as $\sqrt{\Sigma}$ into each side, and `lstsq` is QR followed by back substitution on $Rx = Q^\top b$, with a relative rank check before dividing.

The Jacobi kernel works on a copy of a tall matrix and accumulates $V$:

```python
EPS = 2.0**-52
MAX_SWEEPS = 64

def _jacobi_tall(T):
    W = np.array(T, dtype=np.float64)  # a copy; its columns become U * S
    rows, cols = W.shape
    V = np.eye(cols)                   # invariant: W == T @ V, V orthogonal
    tol = rows * EPS
    for _ in range(MAX_SWEEPS):
        rotated = False
        for i in range(cols - 1):
            for j in range(i + 1, cols):
                wi, wj = W[:, i], W[:, j]
                alpha = float(wi @ wi)
                beta = float(wj @ wj)
                gamma = float(wi @ wj)
                if alpha == 0.0 or beta == 0.0:
                    continue  # a zero column is orthogonal to everything
                if abs(gamma) <= tol * math.sqrt(alpha * beta):
                    continue  # relative test: the cosine is below m * eps
                rotated = True
```

The rotation itself applies the same $(c, s)$ to $W$ and to $V$ so the invariant holds. Both new columns are combinations of both old ones, so $V$'s columns are copied before being overwritten; the `W` update avoids the problem by building both columns in one `column_stack`:

```python
                zeta = (beta - alpha) / (2.0 * gamma)
                sign = 1.0 if zeta >= 0.0 else -1.0       # sign(0) = +1
                t = sign / (abs(zeta) + math.sqrt(1.0 + zeta * zeta))  # smaller root, no cancellation
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = c * t
                W[:, [i, j]] = np.column_stack((c * wi - s * wj, s * wi + c * wj))
                vi, vj = V[:, i].copy(), V[:, j].copy()   # copies: both new columns use the old ones
                V[:, i] = c * vi - s * vj
                V[:, j] = s * vi + c * vj
        if not rotated:
            break                                         # a sweep with no rotation: converged
```

Reading off the factors takes lengths, a stable descending sort, and normalization or completion. The worked example ends with $\sqrt 5$ in column 0 and $3\sqrt 5$ in column 1, which the sort swaps:

```python
    S = np.sqrt(np.einsum("ij,ij->j", W, W))   # column lengths
    order = np.argsort(-S, kind="stable")
    S, W, V = S[order], W[:, order], V[:, order]
    U = np.zeros((rows, cols))
    floor = rows * EPS * (S[0] if cols else 0.0)
    for j in range(cols):
        if S[j] > floor and S[j] > 0.0:
            U[:, j] = W[:, j] / S[j]
        else:
            U[:, j] = _complete(U[:, :j], rows)  # zero singular value: any orthonormal u works
    return U, S, V
```

`_complete` takes the first $e_i$ with more than half its length left after projecting out the previous columns twice (one projection loses orthogonality to rounding; two is enough):

```python
def _complete(Uprev, rows):
    for i in range(rows):
        e = np.zeros(rows)
        e[i] = 1.0
        for _ in range(2):  # twice is enough (Kahan, Parlett)
            e -= Uprev @ (Uprev.T @ e)
        norm = math.sqrt(float(e @ e))
        if norm > 0.5:
            return e / norm
    raise AssertionError("unreachable: fewer than rows orthonormal columns")
```

The public `svd` validates, handles empty and wide shapes, and applies the sign rule to $V$ and $U$ together:

```python
def svd(A):
    M = np.asarray(A, dtype=np.float64)  # a view; _jacobi_tall copies
    if M.ndim != 2:
        raise ValueError(f"svd: A must be 2-D, got shape {M.shape}")
    if not np.isfinite(M).all():
        raise ValueError("svd: A has a NaN or infinite entry")
    m, n = M.shape
    k = min(m, n)
    if k == 0:
        return np.zeros((m, 0)), np.zeros(0), np.zeros((0, n))
    if m >= n:
        U, S, V = _jacobi_tall(M)
    else:
        # A^T = U' S V'^T, so A = V' S U'^T: the roles swap.
        Ut, S, Vt_ = _jacobi_tall(M.T)
        U, V = Vt_, Ut
    big = np.argmax(np.abs(V), axis=0)          # first largest |entry| per column of V
    d = np.where(V[big, np.arange(k)] < 0.0, -1.0, 1.0)
    return U * d, S, (V * d).T                  # flip u_i and v_i together: A unchanged
```

`low_rank` is three lines once `svd` exists:

```python
def low_rank(A, r):
    U, S, Vt = svd(A)
    k = S.shape[0]
    if not 1 <= r <= k:
        raise ValueError(f"low_rank: need 1 <= r <= min(m, n) = {k}, got r = {r}")
    root = np.sqrt(S[:r])
    return U[:, :r] * root, root[:, None] * Vt[:r]   # B^T B = C C^T = Sigma_r
```

`lstsq` factors with QR, checks the rank relative to the largest pivot, and back-substitutes. Indexing `x[i]` on the first axis lets one loop handle `b` of shape `[m]` or `[m, k]`:

```python
    Q, R = qr_householder(M)
    diag = np.abs(np.diag(R))
    if n == 0:
        return np.zeros((0,) + y.shape[1:])
    if diag.min() <= max(m, n) * EPS * diag.max():
        raise ValueError("lstsq: A is rank deficient (a diagonal entry of R is ~0)")
    z = Q.T @ y  # the coordinates of b's projection onto range(A)
    x = np.zeros((n,) + y.shape[1:])
    for i in range(n - 1, -1, -1):  # back substitution, last row first
        x[i] = (z[i] - R[i, i + 1 :] @ x[i + 1 :]) / R[i, i]
    return x
```

The full solution is [`svd.py`](../../../../course/solutions/python/tinyllm/linalg/svd.py).

### Why this design

A sweep visits $n(n-1)/2$ pairs at $O(m)$ each, so it costs $O(mn^2)$, with a handful of sweeps in practice; memory is $O(mn + n^2)$. `lstsq` is one QR, $O(mn^2)$, plus $O(n^2)$ back substitution.

Eigen-decomposing $A^\top A$ squares $\kappa$ and loses every singular value below about $10^{-8}\sigma_1$. Golub-Kahan bidiagonalization, what LAPACK does, is faster for large matrices but much longer to write, and its relative accuracy for tiny singular values is weaker than Jacobi's; for the sizes this book decomposes, Jacobi's simplicity and accuracy win. QR rather than the normal equations makes the least-squares error scale with $\kappa(A)$ instead of $\kappa(A)^2$: $5 \times 10^{-10}$ instead of $10^{-3}$ on the degree-9 fit. Two tradeoffs are accepted: `lstsq` refuses rank-deficient $A$ rather than returning the minimum-norm solution (that needs the SVD and is what `numpy.linalg.lstsq` does), and the sweep cap of 64 is a safety stop, never reached on well-formed input.

## What can go wrong

A correct solution passes a set of checks that mirror the theory. The hand example must reproduce $U$, $\Sigma$, $V^\top$ to $10^{-14}$, pinning the rotation, the order, and the sign rule; $A_1$, the balanced factors, and the error $\sqrt{5}$; and the line fit $x = (7/6, 1/2)$. Random tall, wide, square, and $1 \times n$ matrices must match `np.linalg.svd` entry by entry after the same sign rule, since the decomposition is the unique one. Scaling by $10^{\pm 12}$ must scale the result exactly, `svd(c A)` being $c$ times `svd(A)`, because the rotation test is relative. Rank-deficient and zero matrices, common among co-occurrence matrices, must give orthonormal $U$ and no NaN.

For `low_rank`, the spectral and Frobenius errors for $r = 1, 2, 4, 6$ must match Eckart-Young, which is the optimality word vectors and PiSSA rely on; the factors must satisfy $B^\top B = C C^\top = \Sigma_r$; and $r = 0$, $r < 0$, and $r > \min(m, n)$ must raise. For `lstsq`, one and three right-hand sides must match `np.linalg.lstsq`; the residual must satisfy $A^\top (b - Ax) = 0$; the degree-9 polynomial fit ($\kappa \approx 3.5 \times 10^6$) must recover its coefficients to $10^{-6}$, as the capstone's scaling-law fit needs; and dependent columns, a wide system, and a wrong-length $b$ must raise. Throughout, $A$ must be unchanged, and 1-D and NaN inputs must raise.

The mistakes that break these checks:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| solving the normal equations $A^\top A x = A^\top b$ | fine on easy fits, three correct digits on a degree-9 polynomial fit | factors with `qr_householder` and solves $Rx = Q^\top b$ |
| leaving singular values in sweep order | `low_rank` keeps the wrong directions; the error is not $\sigma_{r+1}$ | sorts with `np.argsort(-S, kind="stable")` |
| the rotation with the wrong sign ($s = -ct$) | the columns never become orthogonal; 64 sweeps of noise | `s = c * t` with $t$ the root of $t^2 + 2\zeta t - 1 = 0$ |
| stopping after one sweep | the columns are far from orthogonal: singular values of a random $30 \times 12$ matrix off by 0.2 to 0.8 | repeats sweeps until one rotates nothing |
| the unbalanced split $(U_r\Sigma_r, V_r^\top)$ | same product, factors at different scales | multiplies `sqrt(S[:r])` into both factors |
| dividing a zero column by its zero length | NaN in $U$ for rank-deficient matrices | calls `_complete` when `S[j]` is below the floor |
| no sign rule | factors disagree with LAPACK's on some columns; downstream tests that fix a seed see flipped vectors | makes the largest $\lvert\text{entry}\rvert$ of each $V$ column positive |
| an absolute threshold $\lvert\gamma\rvert > 10^{-15}$ | tiny-scaled matrices are returned unrotated | compares with `tol * sqrt(alpha * beta)` |
| decomposing a wide matrix without transposing | shapes come out wrong | decomposes `M.T` and swaps the roles of $U$ and $V$ |
| back substitution from the top row | wrong $x$ whenever $R$ is not diagonal | loops `range(n - 1, -1, -1)` |
| no rank check in `lstsq` | division by a zero diagonal entry of $R$: inf | raises when `diag.min()` is below the relative floor |
| rotating the caller's array in place | the decomposed weight is now $U\Sigma$ | rotates `np.array(T)`, a copy |

> **Warning:** The normal equations are the classic trap because they work on every easy example you try by hand. The damage only shows on ill-conditioned fits, exactly the ones (like polynomial or scaling-law fits) where you most need the digits.

## Where this leads

In *Build Your Own LLM System*, `ppmi_svd_embeddings` in [word2vec skip-gram with negative sampling, and the PPMI-SVD baseline](../../build-an-llm/p02-statistical-lm/03-word2vec-and-ppmi-svd.md) keeps the top singular directions of the PPMI matrix as word vectors; [LoRA with PiSSA init and merge](../../build-an-llm/p06-objectives/06-lora-pissa-and-merge.md) initializes LoRA with `low_rank(W, r)` and trains the residual; `mha_to_mla` in [Multi-head latent attention](../../build-an-llm/p07-modern-block/06-multi-head-latent-attention.md) compresses the key and value projections to a rank-$r$ latent; and the scaling-law fit in the [TinyStories capstone](../../build-an-llm/capstones/01-tinystories.md) is `lstsq` on $\log$ loss against $\log$ parameters. In this book, [Error analysis, condition numbers, tolerance budgets](../09-numerical-methods-and-floating-point/03-error-analysis-condition-numbers-and-tolerance-budgets.md) reads condition numbers $\sigma_1/\sigma_k$ from `svd`, and later optimization material on the Muon optimizer checks its Newton-Schulz approximation of $UV^\top$ against this SVD. The next section, [Inner products, projections, cosine similarity, and top-k](06-inner-products-projections-cosine-and-top-k.md), returns to the geometry of projections.

### Further reading

- LAPACK `dgesvj` (Drmac and Veselic; `SRC/dgesvj.f`, `dgejsv.f`): preconditioning by QR with column pivoting, de Rijk's pivoting, and blocked rotations, giving Jacobi's high relative accuracy at near-LAPACK speed.
- LAPACK `dgesdd`, what NumPy calls (`numpy/linalg/_linalg.py`, `SRC/dgesdd.f`): Householder bidiagonalization, then divide and conquer on the bidiagonal matrix; much faster for large matrices.
- Randomized SVD (Halko, Martinsson, Tropp 2011; `sklearn.utils.extmath.randomized_svd`): the top $r$ triples of a huge sparse matrix from a few matrix products with random vectors.
- `numpy.linalg.lstsq` (LAPACK `dgelsd`): the minimum-norm solution for rank-deficient $A$, via the SVD.
- PEFT's PiSSA initializer (`peft/tuners/lora/layer.py`, `pissa_init`): the same balanced split on GPU, then the residual $W - BC$ frozen as the base weight.
- Trefethen and Bau, *Numerical Linear Algebra*, lectures 4, 5, 11, 31; Demmel and Veselic, "Jacobi's method is more accurate than QR" (1992); Eckart and Young, "The approximation of one matrix by another of lower rank" (1936); Meng, Wang, and Zhang, "PiSSA" (2024).

## Key terms

- **Singular value decomposition**: $A = U\Sigma V^\top$ with orthonormal $U$, $V$ and non-negative descending $\sigma_i$.
- **Spectral norm**: $\lVert A \rVert_2 = \sigma_1$, the largest stretch of a unit vector.
- **Frobenius norm**: $\lVert A \rVert_F = \sqrt{\sum_i \sigma_i^2}$, the square root of the sum of squared entries.
- **One-sided Jacobi**: an SVD algorithm that rotates pairs of columns until all are mutually orthogonal.
- **Sweep**: one pass of Jacobi rotations over every column pair $i < j$.
- **Eckart-Young theorem**: the truncated SVD $A_r$ is the closest rank-$r$ matrix in spectral and Frobenius norm.
- **Balanced split**: $B = U_r\Sigma_r^{1/2}$, $C = \Sigma_r^{1/2}V_r^\top$, so both factors carry the same scale.
- **Normal equations**: $A^\top A x = A^\top b$, the least-squares optimality condition, which squares the condition number if solved directly.

## Summary

- Every real matrix is $A = U \Sigma V^\top$: rotate, stretch along the axes by the singular values, rotate again. The singular values are the square roots of the eigenvalues of $A^\top A$, and the largest is the most $A$ can stretch a unit vector.
- One-sided Jacobi rotates pairs of columns until all are orthogonal; the test for "orthogonal enough" must be relative to the column lengths, or a matrix scaled by $10^{-12}$ is never rotated.
- Keeping the $r$ largest singular triples gives the closest rank-$r$ matrix, and the spectral error is exactly the first dropped singular value (Eckart-Young); splitting $\sqrt{\sigma}$ into each factor makes them start at the same scale.
- Least squares by QR solves $Rx = Q^\top b$ with error growing like $\kappa(A)$; the normal equations square it.
- Sorting, a sign rule, and a deterministic completion for zero singular values make the decomposition unique and reproducible.

## Review questions

1. Why is the rotation test $\lvert\gamma\rvert \le m\varepsilon\sqrt{\alpha\beta}$ and not $\lvert\gamma\rvert \le 10^{-15}$?

   <details><summary>Answer</summary>

   $\gamma/\sqrt{\alpha\beta}$ is the cosine of the angle between the columns, which does not change when $A$ is scaled. An absolute threshold depends on the scale: at $10^{-12}$ every $\gamma$ is about $10^{-24}$ and nothing is rotated; at $10^{12}$ nothing ever passes.

   </details>

2. Why does the solution sort the singular values after the sweeps instead of trusting their order?

   <details><summary>Answer</summary>

   Jacobi leaves the lengths in whatever column they ended up in. Eckart-Young keeps the $r$ largest, so `low_rank` taking the first $r$ columns is only correct after a descending sort. The worked example ends with $\sqrt 5$ in column 0 and $3\sqrt 5$ in column 1, which sorting swaps.

   </details>

3. Why are $V$'s columns copied before being overwritten in the rotation?

   <details><summary>Answer</summary>

   Both new columns are combinations of both old ones. Updating `V[:, i]` first and then reading it for `V[:, j]` would mix a new column into the second formula. The `W` update avoids this by building both columns in one `column_stack`.

   </details>

4. Why split $\sqrt{\Sigma_r}$ into each factor of `low_rank`?

   <details><summary>Answer</summary>

   Any split $B = U_r D$, $C = D^{-1}\Sigma_r V_r^\top$ multiplies to $A_r$, but the gradient of a product with respect to one factor scales with the other. The balanced split gives $B^\top B = CC^\top = \Sigma_r$, so both factors train at the same rate (PiSSA's choice).

   </details>

5. Why does `lstsq` compare the smallest $\lvert R_{ii}\rvert$ with $\max(m, n)\varepsilon\max\lvert R_{jj}\rvert$ before solving?

   <details><summary>Answer</summary>

   A near-zero pivot means the columns are dependent and the minimizer is not unique; dividing by it gives huge or infinite coefficients. The test is relative to the largest pivot so it does not depend on the units of $A$.

   </details>
