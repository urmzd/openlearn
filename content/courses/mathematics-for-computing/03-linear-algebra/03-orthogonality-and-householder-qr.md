# Orthogonality, Householder QR, and orthogonal initialization

Once you start initializing networks, one problem appears almost immediately. The first recurrent model, [Vanilla RNN with manual BPTT and truncation](../../build-an-llm/p03-recurrent/01-vanilla-rnn-manual-bptt.md) in *Build Your Own LLM System*, multiplies its hidden state by the same matrix $W$ at every one of hundreds of steps. If $W$ stretches some direction by 1.1, that direction grows by $1.1^{100} \approx 13\,781$; if it shrinks one by 0.9, the gradient along it falls to $0.9^{100} \approx 2.7 \times 10^{-5}$. A matrix that stretches nothing and shrinks nothing is an **orthogonal** one, and the standard way to produce a random one is the QR factorization of a Gaussian matrix.

QR is also the engine of the next section, [Eigenvalues, power iteration, and the spectral radius](04-eigenvalues-and-power-iteration.md), which finds every eigenvalue's size, including the complex ones that power iteration cannot see, by repeating QR. This section builds QR with Householder reflections, the version that stays accurate on the nearly dependent columns real weight matrices have, and then uses it to build an orthogonal initializer.

The solution is [`qr.py`](../../../../course/solutions/python/tinyllm/linalg/qr.py), with contract [`qr.pyi`](../../../../course/contracts/py/tinyllm/linalg/qr.pyi). It provides `qr_householder(A)`, the reduced $A = QR$ by reflections with $\operatorname{diag}(R) \ge 0$, and `orthogonal_init(shape, gain, rng)`, which makes weights with orthonormal rows or columns the way `torch.nn.init.orthogonal_` does. The Gaussian matrix comes from the normal draws of [Random variables, expectation, variance, and normal draws by Box-Muller](../07-probability-statistics/00-random-variables-and-the-normal.md) ([`rv.py`](../../../../course/solutions/python/tinyllm/prob/rv.py)), and elimination from [Gaussian elimination and LU with partial pivoting](02-gaussian-elimination-and-lu.md) is the useful contrast.

> **Learning objectives**
> - Show that a matrix with orthonormal columns preserves lengths and angles.
> - Explain why Householder QR stays orthogonal to machine precision when Gram-Schmidt does not.
> - Choose the sign of a Householder reflector so that it never cancels, and factor a 2 by 2 matrix by hand.
> - State the sign rule that makes QR unique and explain why it makes orthogonal initialization uniformly distributed.
> - Implement `qr_householder` and `orthogonal_init` and test them against LAPACK.

## Notation

These symbols recur through the section:

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

## Orthogonality

Two vectors are **orthogonal** when $\langle u, v \rangle = 0$; a set of vectors is **orthonormal** when every pair is orthogonal and each has length 1. A matrix $Q$ whose columns are orthonormal satisfies $Q^\top Q = I$, because entry $(i, j)$ of $Q^\top Q$ is $\langle q_i, q_j \rangle$. Then

$$\lVert Qx \rVert^2 = x^\top Q^\top Q x = x^\top x = \lVert x \rVert^2, \qquad \langle Qx, Qy \rangle = \langle x, y \rangle .$$

$Q$ preserves lengths and angles: multiplying by it can neither blow up nor shrink a signal. A square $Q$ with $Q^\top Q = I$ is an **orthogonal matrix**; its inverse is its transpose, and its columns and rows are both orthonormal.

## QR, and why reflections

Every real $m \times n$ matrix with $m \ge n$ factors as $A = QR$, with $Q$ having orthonormal columns and $R$ upper triangular. The first $j$ columns of $Q$ span the first $j$ columns of $A$, and $R$ holds the coordinates. The question is how to compute it without losing orthogonality.

The textbook construction, **Gram-Schmidt**, subtracts from each column its projections on the previous $q$'s and normalizes. In floating point it computes the projections with rounding errors, and when two columns are nearly parallel (condition number $10^{10}$), what is left after subtracting is mostly those errors: the computed $q$'s are far from orthogonal.

**Householder** takes the other route. It applies orthogonal matrices to $A$ until it becomes triangular, $H_{k-1} \cdots H_1 H_0\, A = R$, so $Q = H_0 H_1 \cdots H_{k-1}$. Each $H_j = I - 2 v v^\top$ with $\lVert v \rVert = 1$ is a **reflection**: it maps $v$ to $-v$ and leaves every vector orthogonal to $v$ alone. It is symmetric and its own inverse ($H^2 = I - 4vv^\top + 4v(v^\top v)v^\top = I$), hence orthogonal. Since $Q$ is a product of exactly orthogonal matrices, rounding errors only perturb it slightly: $\lVert Q^\top Q - I \rVert$ stays near $\varepsilon$ whatever the conditioning of $A$.

## Choosing the reflector

That settles the strategy. The remaining question is which reflection to use at each step. Step $j$ looks at $x = $ column $j$ of the working matrix from row $j$ down, and wants a reflection that maps $x$ to $\alpha e_1$, zeroing everything below the diagonal. Reflections preserve length, so $\lvert \alpha \rvert = \lVert x \rVert$, and the reflection that swaps $x$ and $\alpha e_1$ has $v$ along their difference:

$$v = \frac{x - \alpha e_1}{\lVert x - \alpha e_1 \rVert}, \qquad H x = \alpha e_1 .$$

Either sign of $\alpha$ works in exact arithmetic. In floating point, the first entry of $x - \alpha e_1$ is $x_0 - \alpha$, and if $\alpha$ has the *same* sign as $x_0$ and $x$ is already almost along $e_1$, this subtracts two nearly equal numbers and leaves only rounding error. With $x = (1, 10^{-9}, 2 \times 10^{-9})$, $\lVert x \rVert = 1 + 2.5 \times 10^{-18}$, which rounds to exactly 1, so $x_0 - \lVert x \rVert = 0$ and $v$ points the wrong way. The fix is to choose $\alpha = -\operatorname{sign}(x_0) \lVert x \rVert$, with $\operatorname{sign}(0) = +1$. Then $x_0 - \alpha = x_0 + \operatorname{sign}(x_0)\lVert x \rVert$ adds two numbers of the same sign and never cancels. If $\lVert x \rVert = 0$ the column is already zero and the step is skipped.

Applying $H$ never forms the $m \times m$ matrix: $H B = B - 2 v (v^\top B)$ costs one product and one outer product. The solution applies it to the trailing block of $R$ (columns $j$ onward) and accumulates $Q \leftarrow Q H$ as $Q - 2(Qv)v^\top$ on columns $j$ onward, then writes $\alpha$ and exact zeros into column $j$.

## The sign rule and uniqueness

QR as described is not quite unique. If $A = QR$, then also $A = (QD)(DR)$ for any diagonal $D$ of $\pm 1$'s, since $D^2 = I$. Fixing $\operatorname{diag}(R) \ge 0$ removes that freedom: for a matrix of full column rank the reduced factorization with a positive diagonal is **unique**. The contract applies the rule at the end: $d_i = -1$ where $R_{ii} < 0$, else $+1$; then $Q \leftarrow Q\operatorname{diag}(d)$ and $R \leftarrow \operatorname{diag}(d) R$. LAPACK (`np.linalg.qr`) uses its own signs, so a comparison applies the same rule to NumPy's answer before comparing entries.

A wide matrix ($m < n$) gives $k = m$: $Q$ is square and $R$ is $m \times n$, upper trapezoidal. A tall one gives $k = n$.

## Orthogonal initialization

With a unique, stable QR in hand, the initializer is a short recipe. `torch.nn.init.orthogonal_` builds a weight of shape $(r, c)$ like this: draw $G$ with independent standard normal entries; if $r < c$, factor $G^\top$ instead; take $Q$ from QR with the sign rule; transpose back if needed; multiply by the gain $g$. The result has orthonormal columns (tall) or rows (wide), scaled by $g$, so every singular value equals $g$ and $W^\top W = g^2 I$ (or $WW^\top = g^2 I$).

The sign rule is what makes $Q$ **uniformly distributed** over orthogonal matrices (the Haar measure). The Gaussian distribution of $G$ is unchanged by any orthogonal transformation, so the distribution of $Q$ would be too, except that the factorization's own sign convention couples $Q$'s signs to $R$'s. With this section's reflector convention, $R_{00} = \alpha$ always has the opposite sign of $G_{00}$, so without the rule $Q_{00} = G_{00}/R_{00}$ would be negative every single time. With the rule, $Q_{00} = G_{00}/\lVert G_{:,0} \rVert$ is positive exactly half the time (Mezzadri 2007).

The draw is also deterministic: $G$ is `normal(rng, r * c)` reshaped in C order, so the same generator state gives the same weights in every run.

## Working it by hand

Factor

$$A = \begin{pmatrix} 3 & 1 \\ 4 & 2 \end{pmatrix}.$$

One reflector suffices ($j = 0$), since a $2 \times 2$ matrix has one entry below the diagonal. The table walks through it quantity by quantity:

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

## Designing the API

Two functions cover both uses:

```python
def qr_householder(A: ArrayLike) -> tuple[NDArray, NDArray]:
    """Reduced QR, k = min(m, n): Q [m, k] with Q^T Q = I, R [k, n] upper
    triangular with diag(R) >= 0, A == Q @ R. A is not modified."""

def orthogonal_init(shape: tuple[int, int], gain: float, rng) -> NDArray:
    """gain * (orthonormal rows or columns), from the QR of normal(rng, rows * cols)."""
```

`qr_householder` returns the reduced factorization, leaves its input alone, and rejects anything that is not two-dimensional. `orthogonal_init` rejects zero or negative dimensions so that configuration errors fail early.

## Building it step by step

The reasoning chain is short. $Q$ must be orthogonal to working precision whatever $A$ is, and building it as a product of reflections is the only one of the two constructions whose error does not grow with the conditioning of $A$. So the algorithm is: for each column, reflect the part on and below the diagonal onto a multiple of $e_1$. The reflection must not cancel, which fixes $\alpha = -\operatorname{sign}(x_0)\lVert x\rVert$ with $\operatorname{sign}(0) = +1$; in the worked example $x = (3, 4)$ gives $\alpha = -5$ and $v \propto (8, 4)$.

$H$ is never formed: $HB = B - 2v(v^\top B)$ costs $O(\text{rows} \cdot \text{cols})$ instead of a full matrix product. It is applied to `R[j:, j:]` only, since rows above $j$ are not touched by $H_j$ and columns left of $j$ are already zero below the diagonal. Because $A = H_0 H_1 \cdots H_{k-1} R$, $Q$ is accumulated by multiplying **on the right**, $Q \leftarrow QH_j$, and since $H_j$ is the identity outside rows and columns $j$ onward, only `Q[:, j:]` changes.

A reflector is needed for each column that has entries below the diagonal, which is `min(m - 1, n)` steps: for a square matrix the last column has none, and for a wide matrix columns $m$ onward have no diagonal entry at all. A zero sub-column is already done and is skipped, which avoids $0/0$. The reduced factorization keeps the first $k$ columns of $Q$ and rows of $R$, and the sign rule then makes it unique and, for the initializer, Haar-distributed. Finally, the initializer needs orthonormal rows when the weight is wide; QR gives orthonormal columns only for a tall input, so a wide $G$ is factored as $G^\top$ and the $Q$ transposed back.

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

Finally comes the reduction to $k$ columns and the sign rule. Multiplying `Q` by `d` scales its columns, and `R * d[:, None]` scales its rows, so $QDDR = QR$:

```python
    Q, R = Q[:, :k], np.triu(R[:k, :])
    d = np.where(np.diag(R) < 0.0, -1.0, 1.0)   # zero diagonal counts as +1
    return Q * d, R * d[:, None]
```

The initializer is a direct transcription of the recipe:

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

The full solution is [`qr.py`](../../../../course/solutions/python/tinyllm/linalg/qr.py).

### Why this design

Each step is two rank-1 updates, of an $(m - j) \times (n - j)$ and an $m \times (m - j)$ block, so factoring costs about $2mn^2 - \frac{2}{3}n^3$ flops for $R$ plus the work to accumulate the explicit $Q$. Memory is $O(m^2 + mn)$ because the full $m \times m$ $Q$ is accumulated and then truncated.

The alternatives each lose something. Gram-Schmidt, classical or modified, is cheaper to explain and produces $Q$ directly, but loses orthogonality in proportion to the condition number of $A$; on the $10^{10}$ case it is useless, while Householder's loss is about $\varepsilon$ regardless. Givens rotations also give an orthogonal $Q$ to working precision, but zero one entry at a time and cost about 1.5 times more on dense matrices; they win for sparse or banded matrices, which this book does not need here. `np.linalg.qr` would be faster, but its sign convention is LAPACK's, and the point is to see where stability comes from; the sign rule makes the two comparable entry by entry anyway. The one tradeoff knowingly accepted is that accumulating the full $m \times m$ $Q$ is wasteful for a very tall matrix (LAPACK stores the reflectors instead), but it makes $Q$ trivially available to the initializer and to the QR algorithm in the next section.

## What can go wrong

The tests a correct solution passes follow the ideas above. The hand example must reproduce $Q$ and $R$ to $10^{-15}$, pinning the reflector and the sign rule. Random tall, wide, and square matrices must match `np.linalg.qr` entry by entry once the same sign rule is applied, which confirms the factorization is the unique one. A first column almost along $e_1$ must still give $A = QR$, which tests the reflector's sign. A matrix with condition number about $10^{10}$ must give $Q^\top Q = I$ to $10^{-13}$, which is the reason to use Householder and not Gram-Schmidt. Zero and dependent columns, which appear in singular weight matrices and padded inputs, must give finite, orthonormal, exact results. $A$ must be unchanged after the call, since the QR algorithm keeps iterating on its matrix, and a vector input must be rejected.

The initializer has its own checks. Across five shapes, $W^\top W = g^2 I$ or $WW^\top = g^2 I$ must hold, so every singular value equals the gain. The output must equal the sign-fixed QR of `normal(rng, r * c)` reshaped in C order, so that weights are reproducible from a seed. Over 400 seeds, $W_{00} > 0$ must hold about half the time, which is the Haar distribution showing through. And zero or negative dimensions must raise.

The mistakes that break those checks:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| accumulating $Q$ from the wrong side ($HQ$ instead of $QH$) | $Q$ is orthogonal but $QR \ne A$ | updates `Q[:, j:]` from the right |
| $\alpha = +\operatorname{sign}(x_0)\lVert x \rVert$ | catastrophic cancellation when a column is nearly along $e_1$ | `alpha = -copysign(norm_x, x[0])` |
| classical Gram-Schmidt | $Q^\top Q$ drifts far from $I$ on ill-conditioned inputs | uses reflections, never projections |
| normalizing a zero column | division by zero, NaN everywhere | skips a column with `norm_x == 0.0` |
| no sign rule | factors disagree with LAPACK's; the initializer is biased ($W_{00} < 0$ every time) | flips rows of $R$ and columns of $Q$ where `diag(R) < 0` |
| reducing the caller's array in place | the next use of $A$ sees $R$ | reduces `np.array(A, dtype=np.float64)`, a copy |
| factoring a wide matrix without transposing | rows are not orthonormal | factors `G.T` when `rows < cols` and transposes back |
| applying the gain twice (or squared) | every weight scaled by $g^2$ | multiplies by `gain` once, at the end |

> **Warning:** Accumulating $Q$ from the wrong side is especially sneaky, because the result is still orthogonal and passes every $Q^\top Q = I$ check. Always test $QR = A$ as well.

## Where this leads

The next section, [Eigenvalues, power iteration, and the spectral radius](04-eigenvalues-and-power-iteration.md), runs the QR algorithm, $A_{t+1} = R_t Q_t$, with `qr_householder` to compute the spectral radius. [SVD, Eckart-Young low rank, and least squares](05-svd-low-rank-and-least-squares.md) solves least squares $\min \lVert Ax - b \rVert$ by solving $Rx = Q^\top b$. In *Build Your Own LLM System*, `orthogonal_init` is one of the default initializers of `Linear` in [Module system and basic layers](../../build-an-llm/p00-foundations/04-module-system-and-layers.md); recurrent weights in [Vanilla RNN with manual BPTT and truncation](../../build-an-llm/p03-recurrent/01-vanilla-rnn-manual-bptt.md) and [LSTM](../../build-an-llm/p03-recurrent/02-lstm.md) start orthogonal, so the hidden state neither explodes nor vanishes at step 0; and [GRU](../../build-an-llm/p03-recurrent/03-gru.md) and [RNN language model with stateful TBPTT](../../build-an-llm/p03-recurrent/06-rnn-language-model-stateful-tbptt.md) reuse it for their weights.

### Further reading

- LAPACK `dgeqrf` and `dorgqr` (`SRC/dgeqrf.f`, `dlarft.f`): store reflectors compactly below $R$ and apply blocks of them as $I - VTV^\top$ (the WY form) with matrix-matrix products.
- `numpy.linalg.qr(mode="reduced")` (`numpy/linalg/_linalg.py`): returns $Q$ only when asked; least squares never forms it.
- `torch.nn.init.orthogonal_` (`torch/nn/init.py`): the same recipe on any tensor shape, flattened to 2-D.
- Tall-skinny QR (TSQR): QR of row blocks combined in a tree, for distributed and GPU settings; Demmel et al., "Communication-optimal parallel and sequential QR" (2012).
- Trefethen and Bau, *Numerical Linear Algebra*, lectures 7 to 10 (QR, Gram-Schmidt, Householder).
- Saxe, McClelland, and Ganguli, "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks" (2014).
- Mezzadri, "How to generate random matrices from the classical compact groups" (2007).

## Key terms

- **Orthonormal**: a set of unit-length vectors that are pairwise orthogonal.
- **Orthogonal matrix**: a square $Q$ with $Q^\top Q = I$; its inverse is its transpose.
- **Householder reflection**: $H = I - 2vv^\top$ with $\lVert v \rVert = 1$, which maps $v$ to $-v$ and fixes the hyperplane orthogonal to $v$.
- **Reduced QR**: $A = QR$ with $Q$ of shape $(m, k)$ and $R$ of shape $(k, n)$, $k = \min(m, n)$.
- **Sign rule**: flipping signs so that $\operatorname{diag}(R) \ge 0$, which makes the factorization unique.
- **Haar measure**: the uniform distribution over orthogonal matrices.

## Summary

- A matrix $Q$ with orthonormal columns ($Q^\top Q = I$) preserves lengths and angles; multiplying by it can neither blow up nor shrink a signal.
- Householder QR zeroes each column below the diagonal with a reflection $H = I - 2vv^\top$; reflections are orthogonal by construction, so $Q$ stays orthogonal to machine precision even when Gram-Schmidt does not.
- The reflector must send $x$ to $-\operatorname{sign}(x_0)\lVert x \rVert e_1$; the other sign cancels catastrophically.
- Requiring $\operatorname{diag}(R) \ge 0$ makes the factorization unique, so it can be compared with LAPACK entry by entry, and makes orthogonal initialization uniformly distributed.
- An orthogonal initializer is the sign-fixed QR of a Gaussian matrix (transposed when wide) times the gain, so every singular value equals the gain.

## Review questions

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
