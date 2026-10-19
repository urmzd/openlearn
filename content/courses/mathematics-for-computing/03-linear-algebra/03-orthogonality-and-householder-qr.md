# Orthogonality, Householder QR, and orthogonal initialization

## Overview

| | |
|---|---|
| **Module** | `M03.3` · build · Python · Pass 2 · about 2 h reading |
| **Solution** | [`qr.py`](../../../../course/solutions/python/tinyllm/linalg/qr.py): `qr_householder(A)` (reduced $A = QR$ by reflections, with $\operatorname{diag}(R) \ge 0$) and `orthogonal_init(shape, gain, rng)` (weights with orthonormal rows or columns, as `torch.nn.init.orthogonal_` makes them) |
| **Contract** | [`course/contracts/py/tinyllm/linalg/qr.pyi`](../../../../course/contracts/py/tinyllm/linalg/qr.pyi) |
| **Needs** | `M07.0` normal draws for the Gaussian matrix ([`rv.py`](../../../../course/solutions/python/tinyllm/prob/rv.py)). Reading: `M03.2` (elimination, the contrast) |
| **Used by** | `M03.4` the QR algorithm for the spectral radius · later `L0.4` default initializers, `L3.1` and `L3.2` recurrent weight init, `M03.5` least squares · later: `L3.3`, `L3.6` |
| **Optional depth** | Trefethen and Bau, *Numerical Linear Algebra*, lectures 7 to 10 (QR, Gram-Schmidt, Householder); Saxe, McClelland, and Ganguli, "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks" (2014); Mezzadri, "How to generate random matrices from the classical compact groups" (2007) |

## Key Takeaways

- A matrix $Q$ with **orthonormal columns** ($Q^\top Q = I$) preserves lengths and angles; multiplying by it can neither blow up nor shrink a signal.
- **Householder QR** zeroes each column below the diagonal with a reflection $H = I - 2vv^\top$; reflections are orthogonal by construction, so $Q$ stays orthogonal to machine precision even when Gram-Schmidt does not.
- The reflector must send $x$ to $-\operatorname{sign}(x_0)\lVert x \rVert e_1$; the other sign cancels catastrophically.
- Requiring $\operatorname{diag}(R) \ge 0$ makes the factorization unique, so it can be compared with LAPACK entry by entry, and makes orthogonal initialization uniformly distributed.

## How to read this chapter

Sections 1 to 3 build orthogonality and Householder reflections from first principles and factor a 2 by 2 matrix by hand. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

In Pass 2 the system starts initializing networks, and the first recurrent model (`L3.1`) multiplies its hidden state by the same matrix $W$ at every one of hundreds of steps. If $W$ stretches some direction by 1.1, that direction grows by $1.1^{100} \approx 13\,781$; if it shrinks one by 0.9, the gradient along it falls to $0.9^{100} \approx 2.7 \times 10^{-5}$. A matrix that stretches nothing and shrinks nothing is an **orthogonal** one, and the standard way to produce a random one is the QR factorization of a Gaussian matrix. QR is also the engine of the next module: `M03.4` finds every eigenvalue's size, including the complex ones that power iteration cannot see, by repeating QR. The solution builds QR with Householder reflections, the version that stays accurate on the nearly dependent columns real weight matrices have.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\langle u, v \rangle = u^\top v$ | inner (dot) product | scalar |
| $\lVert x \rVert = \sqrt{x^\top x}$ | Euclidean length | scalar |
| $A$ | the matrix to factor, $m$ rows and $n$ columns | `float64[m, n]` |
| $k = \min(m, n)$ | the number of columns of $Q$ in the reduced factorization | `int` |
| $Q$ | orthonormal columns: $Q^\top Q = I_k$ | `float64[m, k]` |
| $R$ | upper triangular, non-negative diagonal | `float64[k, n]` |
| $e_1$ | $(1, 0, \dots, 0)$ | vector |
| $v$ | a unit vector defining a reflection | vector |
| $H = I - 2vv^\top$ | the Householder reflection across the hyperplane orthogonal to $v$ | square matrix |
| $\alpha$ | the value a reflected column keeps in its first entry | scalar |
| $g$ | the gain of the initializer | scalar |
| $\varepsilon$ | float64 unit roundoff, about $1.1 \times 10^{-16}$ | |

### 2.1 Orthogonality

Two vectors are **orthogonal** when $\langle u, v \rangle = 0$; a set of vectors is **orthonormal** when every pair is orthogonal and each has length 1. A matrix $Q$ whose columns are orthonormal satisfies $Q^\top Q = I$, because entry $(i, j)$ of $Q^\top Q$ is $\langle q_i, q_j \rangle$. Then

$$\lVert Qx \rVert^2 = x^\top Q^\top Q x = x^\top x = \lVert x \rVert^2, \qquad \langle Qx, Qy \rangle = \langle x, y \rangle :$$

$Q$ preserves lengths and angles. A square $Q$ with $Q^\top Q = I$ is an **orthogonal matrix**; its inverse is its transpose, and its columns and rows are both orthonormal.

### 2.2 QR and why reflections

Every real $m \times n$ matrix with $m \ge n$ factors as $A = QR$, $Q$ with orthonormal columns and $R$ upper triangular: the first $j$ columns of $Q$ span the first $j$ columns of $A$, and $R$ holds the coordinates. The textbook construction, **Gram-Schmidt**, subtracts from each column its projections on the previous $q$'s and normalizes. In floating point it computes the projections with rounding errors, and when two columns are nearly parallel (condition number $10^{10}$), what is left after subtracting is mostly those errors: the computed $q$'s are far from orthogonal.

**Householder** takes the other route: it applies orthogonal matrices to $A$ until it becomes triangular, $H_{k-1} \cdots H_1 H_0\, A = R$, so $Q = H_0 H_1 \cdots H_{k-1}$. Each $H_j = I - 2 v v^\top$ with $\lVert v \rVert = 1$ is a **reflection**: it maps $v$ to $-v$ and leaves every vector orthogonal to $v$ alone. It is symmetric and its own inverse ($H^2 = I - 4vv^\top + 4v(v^\top v)v^\top = I$), hence orthogonal. Since $Q$ is a product of exactly orthogonal matrices, rounding errors only perturb it slightly: $\lVert Q^\top Q - I \rVert$ stays near $\varepsilon$ whatever the conditioning of $A$.

### 2.3 Choosing the reflector

Step $j$ looks at $x = $ column $j$ of the working matrix from row $j$ down, and wants a reflection that maps $x$ to $\alpha e_1$, zeroing everything below the diagonal. Reflections preserve length, so $\lvert \alpha \rvert = \lVert x \rVert$, and the reflection that swaps $x$ and $\alpha e_1$ has $v$ along their difference:

$$v = \frac{x - \alpha e_1}{\lVert x - \alpha e_1 \rVert}, \qquad H x = \alpha e_1 .$$

Either sign of $\alpha$ works in exact arithmetic. In floating point, the first entry of $x - \alpha e_1$ is $x_0 - \alpha$: if $\alpha$ has the *same* sign as $x_0$ and $x$ is already almost along $e_1$, this subtracts two nearly equal numbers and leaves only rounding error. With $x = (1, 10^{-9}, 2 \times 10^{-9})$, $\lVert x \rVert = 1 + 2.5 \times 10^{-18}$, which rounds to exactly 1, so $x_0 - \lVert x \rVert = 0$ and $v$ points the wrong way. The fix is to choose $\alpha = -\operatorname{sign}(x_0) \lVert x \rVert$ (with $\operatorname{sign}(0) = +1$): then $x_0 - \alpha = x_0 + \operatorname{sign}(x_0)\lVert x \rVert$ adds two numbers of the same sign and never cancels. If $\lVert x \rVert = 0$ the column is already zero and the step is skipped.

Applying $H$ never forms the $m \times m$ matrix: $H B = B - 2 v (v^\top B)$ costs one product and one outer product. The solution applies it to the trailing block of $R$ (columns $j$ onward) and accumulates $Q \leftarrow Q H$ as $Q - 2(Qv)v^\top$ on columns $j$ onward, then writes $\alpha$ and exact zeros into column $j$.

### 2.4 The sign rule and uniqueness

If $A = QR$, then also $A = (QD)(DR)$ for any diagonal $D$ of $\pm 1$'s, since $D^2 = I$. Fixing $\operatorname{diag}(R) \ge 0$ removes that freedom: for a matrix of full column rank the reduced factorization with a positive diagonal is **unique**. The contract applies the rule at the end: $d_i = -1$ where $R_{ii} < 0$, else $+1$; then $Q \leftarrow Q\operatorname{diag}(d)$ and $R \leftarrow \operatorname{diag}(d) R$. LAPACK (`np.linalg.qr`) uses its own signs, so a comparison applies the same rule to numpy's answer before comparing entries.

A wide matrix ($m < n$) gives $k = m$: $Q$ is square and $R$ is $m \times n$, upper trapezoidal. A tall one gives $k = n$.

### 2.5 Orthogonal initialization

`torch.nn.init.orthogonal_` builds a weight of shape $(r, c)$ like this: draw $G$ with independent standard normal entries; if $r < c$ factor $G^\top$ instead; take $Q$ from QR with the sign rule; transpose back if needed; multiply by the gain $g$. The result has orthonormal columns (tall) or rows (wide), scaled by $g$, so every singular value equals $g$ and $W^\top W = g^2 I$ (or $WW^\top = g^2 I$).

The sign rule is what makes $Q$ **uniformly distributed** over orthogonal matrices (the Haar measure). The Gaussian distribution of $G$ is unchanged by any orthogonal transformation, so the distribution of $Q$ would be too, except that the factorization's own sign convention couples $Q$'s signs to $R$'s. With this module's reflector convention, $R_{00} = \alpha$ always has the opposite sign of $G_{00}$, so without the rule $Q_{00} = G_{00}/R_{00}$ would be negative every single time. With the rule, $Q_{00} = G_{00}/\lVert G_{:,0} \rVert$ is positive exactly half the time (Mezzadri 2007).

Determinism: $G$ is `normal(rng, r * c)` from `M07.0` reshaped in C order, so the same generator state gives the same weights in every run.

## 3. Worked example by hand

$$A = \begin{pmatrix} 3 & 1 \\ 4 & 2 \end{pmatrix}.$$

One reflector suffices ($j = 0$; a $2 \times 2$ matrix has one entry below the diagonal).

| Quantity | Value |
|---|---|
| $x$ = column 0 | $(3, 4)$, $\lVert x \rVert = 5$ |
| $\alpha = -\operatorname{sign}(3) \cdot 5$ | $-5$ |
| $x - \alpha e_1$ | $(3 + 5, 4) = (8, 4)$, length $\sqrt{80}$ |
| $H = I - 2\,\frac{(8, 4)(8, 4)^\top}{80}$ | $I - \frac{1}{40}\begin{pmatrix} 64 & 32 \\ 32 & 16 \end{pmatrix} = \begin{pmatrix} -0.6 & -0.8 \\ -0.8 & 0.6 \end{pmatrix}$ |
| $H \cdot$ column 0 | $(-0.6 \cdot 3 - 0.8 \cdot 4,\ -0.8 \cdot 3 + 0.6 \cdot 4) = (-5, 0)$ |
| $H \cdot$ column 1 | $(-0.6 - 1.6,\ -0.8 + 1.2) = (-2.2, 0.4)$ |

So $R = \begin{pmatrix} -5 & -2.2 \\ 0 & 0.4 \end{pmatrix}$ and $Q = H$. The sign rule flips row 0 of $R$ and column 0 of $Q$ (only $R_{00}$ is negative):

$$Q = \begin{pmatrix} 0.6 & -0.8 \\ 0.8 & 0.6 \end{pmatrix}, \qquad R = \begin{pmatrix} 5 & 2.2 \\ 0 & 0.4 \end{pmatrix}.$$

Check: $QR = \begin{pmatrix} 0.6 \cdot 5 & 0.6 \cdot 2.2 - 0.8 \cdot 0.4 \\ 0.8 \cdot 5 & 0.8 \cdot 2.2 + 0.6 \cdot 0.4 \end{pmatrix} = \begin{pmatrix} 3 & 1 \\ 4 & 2 \end{pmatrix}$, and the columns of $Q$ are orthonormal: $0.36 + 0.64 = 1$, $-0.48 + 0.48 = 0$.

## 4. The interface

```python
def qr_householder(A: ArrayLike) -> tuple[NDArray, NDArray]:
    """Reduced QR, k = min(m, n): Q [m, k] with Q^T Q = I, R [k, n] upper
    triangular with diag(R) >= 0, A == Q @ R. A is not modified."""

def orthogonal_init(shape: tuple[int, int], gain: float, rng) -> NDArray:
    """gain * (orthonormal rows or columns), from the QR of normal(rng, rows * cols)."""
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Hand example | unit | section 3's $Q$ and $R$ to $10^{-15}$ | pins the reflector and the sign rule |
| Random tall, wide, square | differential | against `np.linalg.qr` with the same sign rule, entry by entry | the factorization is the unique one |
| First column almost along $e_1$ | boundary | still gives $A = QR$ | the reflector's sign |
| Condition number about $10^{10}$ | property | $Q^\top Q = I$ to $10^{-13}$ | why Householder and not Gram-Schmidt |
| Zero and dependent columns | boundary | finite, orthonormal, exact | singular weight matrices, padded inputs |
| Input and dimensionality | boundary | $A$ unchanged; a vector is rejected | `M03.4` keeps iterating on its matrix |
| Initializer, five shapes | property | $W^\top W = g^2 I$ or $WW^\top = g^2 I$ | every singular value equals the gain |
| Initializer draw order | unit | equals the sign-fixed QR of `normal(rng, r * c)` reshaped in C order | reproducible weights from a seed |
| Initializer over 400 seeds | statistical | $W_{00} > 0$ about half the time | the Haar distribution |
| Initializer bad shapes | boundary | zero or negative dimensions raise | config errors fail early |

## 5. The solution, step by step

### From first principles

1. $Q$ must be orthogonal to working precision whatever $A$ is (2.2). Building it as a product of reflections, each exactly orthogonal in exact arithmetic, is the only one of the two constructions whose error does not grow with the conditioning of $A$. So the algorithm is: for each column, reflect the part on and below the diagonal onto a multiple of $e_1$.
2. The reflection must not cancel (2.3), which fixes $\alpha = -\operatorname{sign}(x_0)\lVert x\rVert$ with $\operatorname{sign}(0) = +1$. In the worked example $x = (3, 4)$ gives $\alpha = -5$ and $v \propto (8, 4)$.
3. $H$ is never formed: $HB = B - 2v(v^\top B)$ costs $O(\text{rows} \cdot \text{cols})$ instead of a full matrix product. It is applied to `R[j:, j:]` only, since rows above $j$ are not touched by $H_j$ and columns left of $j$ are already zero below the diagonal.
4. $A = H_0 H_1 \cdots H_{k-1} R$, so $Q$ is accumulated by multiplying **on the right**, $Q \leftarrow QH_j$. $H_j$ is the identity outside rows/columns $j$ onward, so only `Q[:, j:]` changes.
5. A square or tall matrix needs a reflector for each column that has entries below the diagonal: `min(m - 1, n)` steps. A zero sub-column is already done and is skipped, which avoids $0/0$.
6. The reduced factorization keeps the first $k$ columns of $Q$ and rows of $R$; the sign rule (2.4) then makes it unique and, for the initializer, Haar-distributed (2.5).
7. The initializer needs orthonormal rows when the weight is wide. QR gives orthonormal columns only for a tall input, so a wide $G$ is factored as $G^\top$ and the $Q$ transposed back.

### The code, annotated

Set up a working copy that becomes $R$, and an $m \times m$ identity that becomes $Q$:

```python
def qr_householder(A: ArrayLike) -> tuple[NDArray, NDArray]:
    R = np.array(A, dtype=np.float64)  # a copy, reduced to R in place
    if R.ndim != 2:
        raise ValueError(f"qr_householder: A must be 2-D, got shape {R.shape}")
    m, n = R.shape
    k = min(m, n)
    Q = np.eye(m)
```

Each step builds the unit reflector $v$ from the current sub-column:

```python
    for j in range(min(m - 1, n)):     # the last row has nothing below it
        x = R[j:, j]
        norm_x = math.sqrt(float(x @ x))
        if norm_x == 0.0:
            continue  # nothing below the diagonal to zero; avoids 0/0
        # alpha has the opposite sign of x_0, so v_0 = x_0 - alpha never cancels.
        # sign(0) is +1, hence alpha = -||x|| when x_0 == 0.
        alpha = -math.copysign(norm_x, x[0]) if x[0] != 0.0 else -norm_x
        v = x.copy()                   # x is a view into R; do not overwrite it
        v[0] -= alpha                  # v = x - alpha e_1
        v /= math.sqrt(float(v @ v))   # unit length, so H = I - 2 v v^T
```

Then it applies $H_j$ to $R$ from the left and to $Q$ from the right, each as a rank-1 update:

```python
        R[j:, j:] -= 2.0 * np.outer(v, v @ R[j:, j:])   # R <- H R on the trailing block
        Q[:, j:] -= 2.0 * np.outer(Q[:, j:] @ v, v)     # Q <- Q H
        R[j, j] = alpha
        R[j + 1 :, j] = 0.0  # exact zeros, not rounding noise
```

Finally the reduction to $k$ columns and the sign rule. Multiplying `Q` by `d` scales its columns; `R * d[:, None]` scales its rows, so $QDDR = QR$:

```python
    Q, R = Q[:, :k], np.triu(R[:k, :])
    d = np.where(np.diag(R) < 0.0, -1.0, 1.0)   # zero diagonal counts as +1
    return Q * d, R * d[:, None]
```

The initializer is a direct transcription of 2.5:

```python
def orthogonal_init(shape: tuple[int, int], gain: float, rng: Any) -> NDArray:
    rows, cols = int(shape[0]), int(shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"orthogonal_init: shape must be positive, got {shape}")
    G = normal(rng, rows * cols).reshape(rows, cols)   # C order: reproducible from the seed
    # QR gives orthonormal columns only for a tall input: factor G^T when wide.
    Q, _ = qr_householder(G.T if rows < cols else G)
    W = Q.T if rows < cols else Q
    return gain * W                                     # gain applied once
```

Full solution: [`qr.py`](../../../../course/solutions/python/tinyllm/linalg/qr.py).

### Why this solution

- **Cost.** Each step is two rank-1 updates of an $(m - j) \times (n - j)$ and an $m \times (m - j)$ block, so factoring costs about $2mn^2 - \frac{2}{3}n^3$ flops for $R$ plus the work to accumulate the explicit $Q$. Memory is $O(m^2 + mn)$ because the full $m \times m$ $Q$ is accumulated and then truncated.
- **Gram-Schmidt** (classical or modified) is cheaper to explain and produces $Q$ directly, but loses orthogonality in proportion to the condition number of $A$; on the $10^{10}$ case it is useless. Householder's loss is about $\varepsilon$ regardless.
- **Givens rotations** also give an orthogonal $Q$ to working precision, but zero one entry at a time and cost about 1.5 times more on dense matrices. They win for sparse or banded matrices, which this module does not need.
- **`np.linalg.qr`** would be faster but its sign convention is LAPACK's, and the point of the module is to see where stability comes from. The sign rule makes the two comparable entry by entry anyway.
- **Tradeoff accepted.** Accumulating the full $m \times m$ $Q$ is wasteful for a very tall matrix (LAPACK stores the reflectors instead), but makes $Q$ trivially available to the initializer and to `M03.4`.

### Explain it back

1. Why is $\alpha$ chosen with the sign opposite to $x_0$?

   <details><summary>Answer</summary>

   The first entry of $v$ is $x_0 - \alpha$. With opposite signs it is $x_0 + \operatorname{sign}(x_0)\lVert x\rVert$, a sum of two same-sign numbers, which is never smaller than $\lVert x\rVert$. With the same sign it is a difference that can round to zero when $x$ is nearly along $e_1$, and $v$ then points in a direction made of rounding error.

   </details>

2. Why is `Q` updated as `Q[:, j:] -= 2 * outer(Q[:, j:] @ v, v)` and not by multiplying $H$ on the left?

   <details><summary>Answer</summary>

   $H_{k-1} \cdots H_0 A = R$, so $A = H_0 \cdots H_{k-1} R$ (each $H$ is its own inverse). $Q$ is that product in that order, which means each new reflector multiplies on the right. Multiplying on the left accumulates $H_{k-1} \cdots H_0 = Q^\top$ instead: still orthogonal, but $Q^\top R \ne A$ in general.

   </details>

3. Without the sign rule, why would `orthogonal_init` have $W_{00} < 0$ every time?

   <details><summary>Answer</summary>

   The first reflector sets $R_{00} = \alpha = -\operatorname{sign}(G_{00})\lVert G_{:,0}\rVert$, and $Q_{00} = G_{00}/R_{00}$, which is then always negative. The sign rule makes $R_{00} > 0$, so $Q_{00} = G_{00}/\lVert G_{:,0}\rVert$ has the sign of $G_{00}$, positive half the time.

   </details>

4. Why does the loop run `min(m - 1, n)` times and not `n` times?

   <details><summary>Answer</summary>

   A reflector is needed only where there are entries below the diagonal. For a square matrix the last column has none, and for a wide matrix columns $m$ onward have no diagonal entry at all; the sign rule still fixes the last diagonal sign.

   </details>

5. Why does a wide $(2, 5)$ initializer factor $G^\top$?

   <details><summary>Answer</summary>

   QR of a $2 \times 5$ matrix gives a $2 \times 2$ $Q$, which carries no information about the 5-dimensional rows. Factoring the $5 \times 2$ transpose gives a $5 \times 2$ $Q$ with orthonormal columns; its transpose is a $2 \times 5$ matrix with orthonormal rows, which is what $WW^\top = g^2 I$ requires.

   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| accumulating $Q$ from the wrong side ($HQ$ instead of $QH$) | $Q$ is orthogonal but $QR \ne A$ | updates `Q[:, j:]` from the right; see step 4 |
| $\alpha = +\operatorname{sign}(x_0)\lVert x \rVert$ | catastrophic cancellation when a column is nearly along $e_1$ | `alpha = -copysign(norm_x, x[0])`; see step 2 |
| classical Gram-Schmidt | $Q^\top Q$ drifts far from $I$ on ill-conditioned inputs | uses reflections, never projections; see step 1 |
| normalizing a zero column | division by zero, NaN everywhere | skips a column with `norm_x == 0.0` |
| no sign rule | factors disagree with LAPACK's; the initializer is biased ($W_{00} < 0$ every time) | flips rows of $R$ and columns of $Q$ where `diag(R) < 0` |
| reducing the caller's array in place | the next use of $A$ sees $R$ | reduces `np.array(A, dtype=np.float64)`, a copy |
| factoring a wide matrix without transposing | rows are not orthonormal | factors `G.T` when `rows < cols` and transposes back |
| applying the gain twice (or squared) | every weight scaled by $g^2$ | multiplies by `gain` once, at the end |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.0` | `normal(rng, n)` draws the Gaussian matrix $G$ |
| Back | `M03.2` | elimination, the factorization this one is contrasted with (reading) |
| Forward | `M03.4` | `spectral_radius` runs the QR algorithm, $A_{t+1} = R_t Q_t$, with `qr_householder` |
| Forward | `L0.4` | `orthogonal_init` is one of the default initializers of `Linear` |
| Forward | `L3.1`, `L3.2` | recurrent weights start orthogonal, so the hidden state neither explodes nor vanishes at step 0 |
| Forward | `M03.5` | least squares $\min \lVert Ax - b \rVert$ by solving $Rx = Q^\top b$ |
| Forward | `L3.3`, `L3.6` | later recurrent and sequence models reuse `orthogonal_init` for their weights |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `qr_householder` | LAPACK `dgeqrf` + `dorgqr` | stores reflectors compactly below $R$ and applies blocks of them as $I - VTV^\top$ (the WY form) with matrix-matrix products | LAPACK `SRC/dgeqrf.f`, `dlarft.f` |
| explicit $Q$ | `numpy.linalg.qr(mode="reduced")` | returns $Q$ only when asked; least squares never forms it | `numpy/linalg/_linalg.py` |
| `orthogonal_init` | `torch.nn.init.orthogonal_` | the same recipe on any tensor shape (flattened to 2-D) | `torch/nn/init.py` |
| one QR | tall-skinny QR (TSQR) | QR of row blocks combined in a tree, for distributed and GPU settings | Demmel et al., "Communication-optimal parallel and sequential QR" (2012) |
