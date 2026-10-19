# Eigenvalues, power iteration, and the spectral radius

## Overview

| | |
|---|---|
| **Module** | `M03.4` · build · Python · Pass 2 · about 2 h reading |
| **Solution** | [`eig.py`](../../../../course/solutions/python/tinyllm/linalg/eig.py): `power_iteration(matvec, n, iters, rng)` (the dominant eigenpair), `inverse_iteration(A, shift, iters, rng)` (the eigenpair nearest a shift, one LU and many solves), `spectral_radius(W, iters)` ($\max_i \lvert\lambda_i\rvert$ by the QR algorithm, complex eigenvalues included) |
| **Contract** | [`course/contracts/py/tinyllm/linalg/eig.pyi`](../../../../course/contracts/py/tinyllm/linalg/eig.pyi) |
| **Needs** | `M03.2` `lu` and `lu_solve` ([`lu.py`](../../../../course/solutions/python/tinyllm/linalg/lu.py)), `M03.3` `qr_householder` ([`qr.py`](../../../../course/solutions/python/tinyllm/linalg/qr.py)) |
| **Used by** | later `L0.5` (the training monitor reports $\rho$ of weight matrices), `L3.1` (the exploding-gradient diagnostic), `M10.5` (the top Hessian eigenvalue); `M10.1` reads it for the condition number |
| **Optional depth** | Trefethen and Bau, *Numerical Linear Algebra*, lectures 24 to 28 (eigenvalue problems, power and inverse iteration, the QR algorithm); Strang, *Introduction to Linear Algebra*, ch. 6; Pascanu, Mikolov, and Bengio, "On the difficulty of training recurrent neural networks" (2013) |

## Key Takeaways

- An **eigenvector** is a direction a matrix only stretches, $Av = \lambda v$; repeated multiplication amplifies the eigenvalue of largest modulus fastest, so **power iteration** converges to it at the rate $\lvert\lambda_2/\lambda_1\rvert$ per step.
- Return the **Rayleigh quotient** $v^\top A v$, not $\lVert Av \rVert$, and renormalize every step: the first loses the sign, the second overflows.
- **Inverse iteration** runs power iteration on $(A - \sigma I)^{-1}$, whose dominant eigenvalue belongs to the $\lambda$ nearest $\sigma$; factor $A - \sigma I$ **once** with LU and solve at every step.
- A real matrix's dominant eigenvalues can be a **complex pair** (a rotation), where power iteration never converges. The **QR algorithm** $A_{t+1} = R_t Q_t$ exposes them as $2 \times 2$ blocks, and the **spectral radius** $\rho(W) = \max_i \lvert\lambda_i\rvert$ reads off the blocks.

## How to read this chapter

Sections 1 to 3 build power iteration, inverse iteration, and the QR algorithm from first principles and run them by hand. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

Training a recurrent network (`L3.1`) multiplies the hidden state by the same weight matrix $W$ at every step, and backpropagation multiplies the gradient by $W^\top$ the same number of times. Whether that product grows without bound or fades to nothing over 200 steps is decided by one number, the spectral radius $\rho(W)$: above 1 the gradients explode, below 1 they vanish. The training monitor (`L0.5`) prints it for every weight matrix, and `M10.5` uses the same machinery to find the largest curvature of the loss, which bounds the learning rate. `numpy.linalg.eigvals` would answer, but the method is simple enough to build from scratch: multiply, normalize, repeat. This chapter builds it, sees where it fails (rotations), and fixes that failure with the QR factorization from `M03.3`.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $A, W$ | a real square matrix | `float64[n, n]` |
| $\lambda_i, v_i$ | eigenvalues and eigenvectors: $A v_i = \lambda_i v_i$, $v_i \ne 0$ | complex scalar, vector |
| $\lvert\lambda_1\rvert > \lvert\lambda_2\rvert \ge \dots$ | eigenvalues ordered by modulus | |
| $\rho(A) = \max_i \lvert\lambda_i\rvert$ | the spectral radius | scalar |
| $v_t$ | the unit vector after $t$ steps of an iteration | `float64[n]` |
| $r(v) = v^\top A v$ | the Rayleigh quotient of a unit vector $v$ | scalar |
| $\sigma$ | a shift: a guess near the wanted eigenvalue | scalar |
| $A_t = Q_t R_t$ | the QR factorization at step $t$ of the QR algorithm | |
| $i$ | the imaginary unit, $i^2 = -1$ | |

### 2.1 Eigenvalues and eigenvectors

A non-zero vector $v$ is an **eigenvector** of $A$ with **eigenvalue** $\lambda$ when $Av = \lambda v$: $A$ maps $v$ onto its own line. The eigenvalues are the roots of the characteristic polynomial $\det(A - \lambda I) = 0$, a polynomial of degree $n$, so there are $n$ of them counted with multiplicity, possibly complex (a real polynomial's complex roots come in conjugate pairs $a \pm bi$). For a $2 \times 2$ matrix $\begin{pmatrix} a & b \\ c & d \end{pmatrix}$ the polynomial is $\lambda^2 - (a + d)\lambda + (ad - bc)$: trace and determinant, solved by the quadratic formula. A **symmetric** matrix has only real eigenvalues and an orthonormal basis of eigenvectors.

### 2.2 Power iteration

Write a start vector in the eigenvector basis, $v_0 = \sum_i c_i v_i$. Then

$$A^t v_0 = \sum_i c_i \lambda_i^t v_i = \lambda_1^t \Big( c_1 v_1 + \sum_{i \ge 2} c_i \big(\tfrac{\lambda_i}{\lambda_1}\big)^t v_i \Big).$$

If $\lvert\lambda_1\rvert > \lvert\lambda_2\rvert$ and $c_1 \ne 0$, every other term shrinks like $\lvert\lambda_2/\lambda_1\rvert^t$, so the **direction** of $A^t v_0$ converges to $v_1$. Three practical rules follow.

1. **Normalize every step**, $v_{t+1} = Av_t / \lVert Av_t \rVert$. The direction is all that matters, and $\lambda_1^t$ overflows: $10^{400}$ is `inf` in float64, and `inf/inf` is NaN.
2. **Start at random.** A start with $c_1 = 0$ never finds $v_1$ (in exact arithmetic): $e_1$ is exactly such a start for $\operatorname{diag}(1, 3)$. Random entries make $c_1 = 0$ an event of probability zero. The contract draws $v_0$ from the caller's generator, $2u - 1$ for $n$ uniforms, so runs are reproducible.
3. **Report the Rayleigh quotient** $r(v) = v^\top A v$. For an exact eigenvector it equals $\lambda$, **with its sign**; $\lVert Av \rVert$ equals $\lvert\lambda\rvert$ and turns $-5$ into $5$. For a symmetric matrix its error is the *square* of the direction's error, so it converges twice as fast.

If some $Av_t$ is exactly zero, $v_t$ lies in the null space: the eigenvalue is 0, and dividing would produce NaN, so return 0.

### 2.3 Inverse iteration

The matrix $(A - \sigma I)^{-1}$ has the same eigenvectors as $A$, with eigenvalues $1/(\lambda_i - \sigma)$. The largest of those belongs to the $\lambda_i$ **nearest** $\sigma$, so power iteration on $(A - \sigma I)^{-1}$ finds it, at the rate $\lvert\lambda_{\text{near}} - \sigma\rvert / \lvert\lambda_{\text{next}} - \sigma\rvert$ per step: very fast for a good shift. Each step needs $w = (A - \sigma I)^{-1} v$, that is, a solve with $A - \sigma I$. Never form the inverse, and never refactor: `lu` once ($\frac{2}{3}n^3$ operations, `M03.2`), then `lu_solve` per step ($2n^2$). The answer reported is $v^\top A v$, an eigenvalue of $A$ itself. If $\sigma$ is exactly an eigenvalue, $A - \sigma I$ is singular and there is nothing to solve with: raise.

### 2.4 Complex pairs and the QR algorithm

The rotation $W = \begin{pmatrix} 0 & -2 \\ 2 & 0 \end{pmatrix}$ has trace 0 and determinant 4, so $\lambda^2 + 4 = 0$ and $\lambda = \pm 2i$: equal moduli, no real eigenvector. Power iteration sends $(1, 0)$ to $(0, 2)$, then $(-4, 0)$, then $(0, -8)$: the direction turns by $90°$ every step and never settles. Recurrent weight matrices are full of such rotations.

The **QR algorithm** handles them. Start from $A_0 = W$ and repeat

$$A_t = Q_t R_t \ \text{(QR factorization, M03.3)}, \qquad A_{t+1} = R_t Q_t .$$

Since $R_t = Q_t^\top A_t$, $A_{t+1} = Q_t^\top A_t Q_t$: a **similarity transform**, which never changes the eigenvalues. It is power iteration on all $n$ directions at once, kept orthonormal by the QR step: the first column follows $\lambda_1$, the first two span the dominant 2-dimensional invariant subspace, and so on. When the moduli separate, $A_t$ converges to **block upper triangular** form (the real Schur form): entry $(j+1, j)$ decays like $\lvert\lambda_{j+1}/\lambda_j\rvert^t$, and what remains on the diagonal are $1 \times 1$ blocks (real eigenvalues) and $2 \times 2$ blocks (complex pairs, whose subdiagonal never vanishes). The spectral radius is the largest modulus over the blocks: $\lvert a \rvert$ for a $1 \times 1$ block, the quadratic formula of 2.1 for a $2 \times 2$ one. `spectral_radius` treats a subdiagonal entry below $10^{-12}$ times the largest entry as zero.

### 2.5 Why the spectral radius decides explosion

For any vector norm, $\lVert W^t \rVert^{1/t} \to \rho(W)$ as $t \to \infty$ (Gelfand's formula), so $\lVert W^t h \rVert$ grows like $\rho^t$ when $\rho > 1$ and shrinks like $\rho^t$ when $\rho < 1$, up to factors polynomial in $t$. Backpropagation through $t$ steps of $h_t = W h_{t-1}$ multiplies by $(W^\top)^t$, which has the same spectral radius. Note $\rho(W) \le \lVert W \rVert_2$, with equality for symmetric (and orthogonal) $W$: an orthogonal initialization (`M03.3`) starts every recurrent weight at exactly $\rho = 1$.

## 3. Worked example by hand

**Power iteration on** $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$, with eigenvalues 3 (eigenvector $(1, 1)$) and 1 (eigenvector $(1, -1)$). Take the start $(1, 0) = \tfrac12 (1, 1) + \tfrac12 (1, -1)$ (the solution draws a random one) and skip the normalization to keep integers:

| $t$ | $A^t v_0 = \tfrac12 3^t (1, 1) + \tfrac12 (1, -1)$ | Rayleigh quotient $\dfrac{x^\top A x}{x^\top x}$ |
|---|---|---|
| 0 | $(1, 0)$ | $2/1 = 2$ |
| 1 | $(2, 1)$ | $(2 \cdot 5 + 1 \cdot 4)/5 = 14/5 = 2.8$ |
| 2 | $(5, 4)$ | $(5 \cdot 14 + 4 \cdot 13)/41 = 122/41 \approx 2.9756$ |
| 3 | $(14, 13)$ | $(14 \cdot 41 + 13 \cdot 40)/365 = 1094/365 \approx 2.99726$ |

The wrong component stays $\tfrac12(1, -1)$ while the right one triples, so the angle error shrinks by $1/3$ per step, and the Rayleigh quotient's error by $1/9$: $3 - 2.8 = 0.2$, $3 - 2.9756 = 0.0244$, $3 - 2.99726 = 0.0027$. After 60 steps both are exact to rounding: $\lambda = 3$ to $10^{-12}$ and $\lvert v \rvert = (2^{-1/2}, 2^{-1/2})$.

**The spectral radius of a scaled rotation**, $W = 1.5 \begin{pmatrix} \cos 0.7 & -\sin 0.7 \\ \sin 0.7 & \cos 0.7 \end{pmatrix}$. Its QR factorization is $Q =$ the rotation itself, $R = 1.5 I$, so $A_1 = RQ = W$: the QR algorithm leaves it unchanged, and the $2 \times 2$ block never becomes triangular. That is the expected outcome for a complex pair. The block's trace is $3\cos 0.7$ and its determinant $2.25$, so

$$\lambda = 1.5\cos 0.7 \pm \sqrt{2.25\cos^2 0.7 - 2.25} = 1.5(\cos 0.7 \pm i \sin 0.7), \qquad \lvert\lambda\rvert = 1.5 .$$

## 4. The interface

```python
def power_iteration(matvec, n: int, iters: int, rng) -> tuple[float, NDArray]:
    """v0 = n draws of 2 * rng.uniform() - 1, normalized; iters steps of v = Av / ||Av||;
    returns (v . A v, v). (0.0, v) if some Av is exactly 0."""

def inverse_iteration(A, shift: float, iters: int, rng) -> tuple[float, NDArray]:
    """power_iteration on (A - shift I)^-1 via one lu and a lu_solve per step;
    returns (v . A v, v). ValueError if A - shift I is singular."""

def spectral_radius(W, iters: int = 100) -> float:
    """max |lambda| by the unshifted QR algorithm with qr_householder; 0.0 for 0 x 0."""
```

`power_iteration` takes a function, not a matrix, so `M10.5` can pass a Hessian-vector product without ever forming the Hessian.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Hand example, power iteration | unit | section 3: $\lambda = 3$, $v = \pm(1, 1)/\sqrt 2$ | pins the method |
| Negative dominant eigenvalue | boundary | $\operatorname{diag}(-5, 1, 2)$ gives $-5$ | `M10.5` tells a maximum from a saddle by the sign |
| A matrix where $e_1$ is a bad start | boundary | $\operatorname{diag}(1, 3)$ gives 3, not 1 | a fixed start can be orthogonal to $v_1$ |
| Start vector draw order | unit | with 0 steps, $v$ is the normalized $2u - 1$ draws | reproducible from the seed (P11) |
| Long run, $\lambda = 10$ | boundary | 400 steps stay finite | long runs |
| Symmetric matrices | differential | 20 matrices, dominant $\pm 2$ against $1.5$, versus `np.linalg.eigh` | the sign and the vector, at a slow ratio |
| Zero map, bad arguments | boundary | $Av = 0$ gives 0 without NaN; bad `n` and `iters` raise | rank-deficient weights |
| Inverse iteration, spectrum $\{1, 3, 7\}$ | unit | shift 2.9 finds 3, shift 0 finds 1; an exact eigenvalue as the shift raises | the smallest Hessian eigenvalue in `M10.5` |
| Inverse iteration, 25 steps | unit | `lu` is called exactly once | the point of a factorization |
| Scaled rotation | unit | section 3: $\rho = 1.5$ | recurrent weights rotate |
| Non-symmetric matrices | differential | five matrices with real, negative, and complex dominant eigenvalues, versus `np.linalg.eigvals` | the `L3.1` diagnostic on realistic matrices |
| Edge sizes | boundary | $1 \times 1$, $0 \times 0$, a nilpotent matrix (all eigenvalues 0), non-square input | |
| Input preserved | boundary | the caller's $W$ is unchanged | the monitor reads live weights |

## 5. The solution, step by step

### From first principles

1. Power iteration needs only products $Av$ (2.2), so the solution takes a `matvec` callable instead of a matrix. That one choice lets inverse iteration pass a solve and `M10.5` pass a Hessian-vector product, with no new code.
2. The start must have $c_1 \ne 0$ almost surely and be reproducible, so it is $n$ draws of $2u - 1$ from the caller's generator, normalized.
3. Only the direction matters and $\lambda_1^t$ overflows, so each step divides by $\lVert Av \rVert$. A zero norm means $v$ is in the null space: return eigenvalue 0 rather than $0/0$.
4. The eigenvalue estimate must keep its sign, so the return value is the Rayleigh quotient $v^\top A v$ of the final unit vector, which costs one more `matvec`. In the worked example this is the column that converges to 3 with error shrinking by $1/9$ per step.
5. Inverse iteration is power iteration with `matvec = solve with A - \sigma I` (2.3). Factor once, outside the loop; check the pivots once so a shift equal to an eigenvalue raises. The reported value must belong to $A$, not to the inverse, so it is recomputed as $v^\top A v$.
6. The spectral radius cannot come from power iteration because a complex pair never converges (2.4). The QR algorithm $T \leftarrow RQ$ preserves eigenvalues and drives $T$ to block triangular form; reading moduli from $1 \times 1$ and $2 \times 2$ diagonal blocks then covers both real and complex cases.
7. "Zero" on the subdiagonal must be relative to the matrix's scale, so the threshold is $10^{-12} \max\lvert T\rvert$.

### The code, annotated

The start vector:

```python
def _start(n: int, rng: Any) -> NDArray:
    """A random unit vector: almost surely not orthogonal to any eigenvector."""
    v = np.array([2.0 * rng.uniform() - 1.0 for _ in range(n)], dtype=np.float64)
    norm = math.sqrt(float(v @ v))
    if norm == 0.0:
        v = np.ones(n)  # probability zero with 53-bit uniforms; stay defined
        norm = math.sqrt(n)
    return v / norm
```

Power iteration, with the three rules of 2.2 visible as three lines:

```python
def power_iteration(matvec, n, iters, rng):
    if n < 1:
        raise ValueError(f"power_iteration: n must be >= 1, got {n}")
    if iters < 0:
        raise ValueError(f"power_iteration: iters must be >= 0, got {iters}")
    v = _start(n, rng)                       # rule 2: random start
    for _ in range(iters):
        w = np.asarray(matvec(v), dtype=np.float64)
        norm = math.sqrt(float(w @ w))
        if norm == 0.0:
            return 0.0, v  # v is in the null space: eigenvalue 0
        v = w / norm       # rule 1: renormalize every step, or |lambda|^t overflows
    # rule 3: the Rayleigh quotient keeps the sign that ||A v|| would lose
    return float(v @ np.asarray(matvec(v), dtype=np.float64)), v
```

Inverse iteration reuses it with a solve as the `matvec`. The lambda closes over the factors, so `lu` runs exactly once whatever `iters` is:

```python
def inverse_iteration(A, shift, iters, rng):
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"inverse_iteration: A must be square, got shape {A.shape}")
    n = A.shape[0]
    P, L, U = lu(A - shift * np.eye(n))  # factor once ...
    if np.any(np.diag(U) == 0.0):
        raise ValueError("inverse_iteration: A - shift I is singular (shift is an eigenvalue)")
    _, v = power_iteration(lambda x: lu_solve(P, L, U, x), n, iters, rng)  # ... solve every step
    # discard the eigenvalue of the inverse; report the one of A
    return float(v @ (A @ v)), v
```

The spectral radius runs the unshifted QR algorithm on a copy, then reads the diagonal blocks:

```python
def spectral_radius(W, iters=100):
    T = np.array(W, dtype=np.float64)        # a copy: the monitored weights stay put
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError(f"spectral_radius: W must be square, got shape {T.shape}")
    if T.shape[0] == 0:
        return 0.0
    for _ in range(iters):
        Q, R = qr_householder(T)
        T = R @ Q                            # = Q^T T Q: same eigenvalues
    return max(_block_moduli(T))
```

```python
def _block_moduli(T):
    n = T.shape[0]
    scale = float(np.abs(T).max()) or 1.0    # the all-zero matrix still has a scale
    out = []
    i = 0
    while i < n:
        if i == n - 1 or abs(T[i + 1, i]) <= 1e-12 * scale:
            out.append(abs(float(T[i, i])))  # 1x1 block: a real eigenvalue
            i += 1
            continue
        a, b, c, d = T[i, i], T[i, i + 1], T[i + 1, i], T[i + 1, i + 1]
        # 2x2 block: roots of t^2 - (a + d) t + (ad - bc), possibly complex
        tr, det = a + d, a * d - b * c
        disc = cmath.sqrt(tr * tr / 4.0 - det)
        out += [abs(tr / 2.0 + disc), abs(tr / 2.0 - disc)]
        i += 2
    return out
```

On the scaled rotation, $T$ stays equal to $W$, the subdiagonal $1.5 \sin 0.7$ is far above the threshold, and the $2 \times 2$ formula gives $\lvert\lambda\rvert = 1.5$ twice.

Full solution: [`eig.py`](../../../../course/solutions/python/tinyllm/linalg/eig.py).

### Why this solution

- **Cost.** Power iteration costs `iters` matvecs plus one; for a dense matrix that is $O(\text{iters} \cdot n^2)$ and $O(n)$ memory. Inverse iteration adds one $\frac{2}{3}n^3$ factorization and then $2n^2$ per step. The unshifted QR algorithm costs $O(\text{iters} \cdot n^3)$, acceptable for the small weight matrices a monitor inspects.
- **Why not $\lVert Av\rVert$.** It loses the sign and converges at the angle rate, not its square.
- **Why not power iteration for $\rho$.** It fails on exactly the matrices recurrent networks produce (rotations), and it cannot be patched by more iterations.
- **Why not shifts and Hessenberg reduction.** Production QR (LAPACK) reduces to Hessenberg form and uses Francis double shifts for quadratic convergence. That is faster but adds bookkeeping that hides the idea; the unshifted version converges linearly, at the ratio of consecutive moduli, which is enough for a diagnostic. The accepted tradeoff: if two eigenvalue moduli are very close but distinct, 100 iterations may leave a large subdiagonal and the pair is read as a $2 \times 2$ block, which still gives the right moduli.

### Explain it back

1. Why does `power_iteration` take a function and not a matrix?

   <details><summary>Answer</summary>

   The algorithm only ever needs $Av$. A callable lets inverse iteration substitute a solve with the LU factors and lets `M10.5` substitute a Hessian-vector product, without forming either matrix.

   </details>

2. What is returned for $A = \operatorname{diag}(-5, 1, 2)$, and why would $\lVert Av\rVert$ be wrong?

   <details><summary>Answer</summary>

   $v \to \pm e_1$ and $v^\top A v = -5$. $\lVert Av\rVert = 5$ for either sign of $v$, so it cannot distinguish $-5$ from $5$.

   </details>

3. In `inverse_iteration`, why is the result `v @ (A @ v)` and not the value returned by `power_iteration`?

   <details><summary>Answer</summary>

   `power_iteration` would return the Rayleigh quotient of $(A - \sigma I)^{-1}$, which is $1/(\lambda - \sigma)$. The caller wants $\lambda$, which is the Rayleigh quotient of $A$ at the same vector.

   </details>

4. Why does the QR algorithm leave the scaled rotation unchanged, and why is that the right outcome?

   <details><summary>Answer</summary>

   $W = 1.5\,\text{rot}(0.7)$ factors as $Q = \text{rot}(0.7)$, $R = 1.5 I$, so $RQ = W$. A complex pair cannot be split into real $1 \times 1$ blocks, so the $2 \times 2$ block is the final form, and `_block_moduli` reads both moduli from it.

   </details>

5. Why is the subdiagonal threshold `1e-12 * scale` and not a fixed `1e-12`?

   <details><summary>Answer</summary>

   Whether an entry is negligible depends on the size of the matrix entries. For weights of size $10^{-8}$ a fixed threshold would call every entry zero and drop complex pairs; for entries of size $10^{6}$ it would never declare convergence.

   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| returning $\lVert Av \rVert$ instead of $v^\top A v$ | $+5$ reported for eigenvalue $-5$ | returns `v @ matvec(v)`; see step 4 |
| a fixed start such as $e_1$ | the wrong eigenvalue whenever $e_1$ is orthogonal to $v_1$ | draws the start from the generator in `_start` |
| normalizing only at the end | `inf`, then NaN, after a few hundred steps | `v = w / norm` inside the loop |
| dividing by $\lVert Av \rVert = 0$ | NaN for a vector in the null space | returns `(0.0, v)` when `norm == 0.0` |
| iterating with $A - \sigma I$ instead of its inverse | finds the eigenvalue farthest from $\sigma$ | passes `lu_solve` as the `matvec` |
| refactoring at every step | correct but `iters` times slower | calls `lu` before the loop; the lambda only solves |
| spectral radius by power iteration | never converges on rotations; wrong $\rho$ for complex pairs | uses the QR algorithm, not power iteration; see step 6 |
| reading only the diagonal after the QR algorithm | complex pairs reported as their real parts | `_block_moduli` solves each $2 \times 2$ block in closed form |
| iterating in the caller's array | the weights being monitored change | iterates on `np.array(W)`, a copy |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M03.2` | `lu` once and `lu_solve` per step in `inverse_iteration` |
| Back | `M03.3` | `qr_householder` drives the QR algorithm in `spectral_radius` |
| Forward | `L0.5` | the training monitor logs $\rho$ of each weight matrix |
| Forward | `L3.1` | the exploding-gradient diagnostic: gradient norms grow when $\rho(W_{hh}) > 1$ |
| Forward | `M10.5` | the top Hessian eigenvalue by `power_iteration` on Hessian-vector products; the edge of stability $2/\lambda_{\max}$ |
| Forward | `M10.1` | the condition number $\kappa = L/\mu$ of a quadratic is $\lambda_{\max}/\lambda_{\min}$ (reading) |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| unshifted QR algorithm | LAPACK `dhseqr` (via `numpy.linalg.eigvals`) | reduction to Hessenberg form first ($O(n^2)$ per step instead of $O(n^3)$), Wilkinson and Francis double shifts for quadratic convergence, deflation, aggressive early deflation | LAPACK `SRC/dhseqr.f`, `dlahqr.f` |
| power iteration on a `matvec` | Lanczos and Arnoldi (ARPACK, `scipy.sparse.linalg.eigsh`) | keep every iterate and extract eigenvalues from the whole Krylov subspace: many eigenpairs, far fewer products | `scipy/sparse/linalg/_eigen/arpack/` |
| Hessian top eigenvalue (`M10.5`) | PyHessian | power iteration and Lanczos on Hessian-vector products of a real network | Yao et al., "PyHessian" (2020) |
| spectral radius monitoring | spectral normalization in GANs | one power-iteration step per training step to keep $\lVert W \rVert_2 \le 1$ | `torch.nn.utils.parametrizations.spectral_norm` |
