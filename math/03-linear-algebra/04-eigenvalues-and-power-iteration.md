<!-- ol:module M03.4 -->
# Eigenvalues, power iteration, and the spectral radius

## Overview

| | |
|---|---|
| **Module** | `M03.4` · build · Python · Pass 2 · 3 to 4 h |
| **You build** | `python/tinyllm/linalg/eig.py`: `power_iteration(matvec, n, iters, rng)` (the dominant eigenpair), `inverse_iteration(A, shift, iters, rng)` (the eigenpair nearest a shift, one LU and many solves), `spectral_radius(W, iters)` ($\max_i \lvert\lambda_i\rvert$ by the QR algorithm, complex eigenvalues included) |
| **Contract** | [`course/contracts/py/tinyllm/linalg/eig.pyi`](../../course/contracts/py/tinyllm/linalg/eig.pyi) |
| **Tests** | `course/tests/M03.4/test_eig.py` (what they check: section 4) |
| **Needs** | `M03.2` `lu` and `lu_solve`, `M03.3` `qr_householder` (or `--ref-deps`) |
| **Used by** | later `L0.5` (the training monitor reports $\rho$ of weight matrices), `L3.1` (the exploding-gradient diagnostic), `M10.5` (the top Hessian eigenvalue); `M10.1` reads it for the condition number. No registered module calls it yet (section 6) |
| **Milestone** | `MS-P2` (the foundations gate) |
| **Optional depth** | Trefethen and Bau, *Numerical Linear Algebra*, lectures 24 to 28 (eigenvalue problems, power and inverse iteration, the QR algorithm); Strang, *Introduction to Linear Algebra*, ch. 6; Pascanu, Mikolov, and Bengio, "On the difficulty of training recurrent neural networks" (2013) |

## Key Takeaways

- An **eigenvector** is a direction a matrix only stretches, $Av = \lambda v$; repeated multiplication amplifies the eigenvalue of largest modulus fastest, so **power iteration** converges to it at the rate $\lvert\lambda_2/\lambda_1\rvert$ per step (`test_hand_example_power_iteration`).
- Return the **Rayleigh quotient** $v^\top A v$, not $\lVert Av \rVert$, and renormalize every step: the first loses the sign, the second overflows (`test_negative_dominant_eigenvalue_keeps_its_sign`, `test_renormalizes_every_step`).
- **Inverse iteration** runs power iteration on $(A - \sigma I)^{-1}$, whose dominant eigenvalue belongs to the $\lambda$ nearest $\sigma$; factor $A - \sigma I$ **once** with LU and solve at every step (`test_inverse_iteration_factors_once`).
- A real matrix's dominant eigenvalues can be a **complex pair** (a rotation), where power iteration never converges. The **QR algorithm** $A_{t+1} = R_t Q_t$ exposes them as $2 \times 2$ blocks, and the **spectral radius** $\rho(W) = \max_i \lvert\lambda_i\rvert$ reads off the blocks (`test_spectral_radius_of_a_rotation`, `test_spectral_radius_matches_numpy`).

## How to work this chapter

```bash
ol start M03.4              # stubs python/tinyllm/linalg/eig.py into your repo
ol tests M03.4              # read the test catalog first: rung R0, you write no tests here
ol check M03.4              # exit code is the verdict
ol check M03.4 --ref-deps   # only if your M03.2 or M03.3 is not passing yet
ol diff  M03.4              # after passing: your code against the reference
```

---

## 1. Why now

Training a recurrent network (`L3.1`) multiplies the hidden state by the same weight matrix $W$ at every step, and backpropagation multiplies the gradient by $W^\top$ the same number of times. Whether that product grows without bound or fades to nothing over 200 steps is decided by one number, the spectral radius $\rho(W)$: above 1 the gradients explode, below 1 they vanish. Your training monitor (`L0.5`) will print it for every weight matrix, and `M10.5` uses the same machinery to find the largest curvature of the loss, which bounds the learning rate. `numpy.linalg.eigvals` would answer, but you are building the system yourself, and the method is simple: multiply, normalize, repeat. This module builds it, sees where it fails (rotations), and fixes that failure with the QR factorization from `M03.3`.

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

**Power iteration on** $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$, with eigenvalues 3 (eigenvector $(1, 1)$) and 1 (eigenvector $(1, -1)$). Take the start $(1, 0) = \tfrac12 (1, 1) + \tfrac12 (1, -1)$ (the tests draw a random one) and skip the normalization to keep integers:

| $t$ | $A^t v_0 = \tfrac12 3^t (1, 1) + \tfrac12 (1, -1)$ | Rayleigh quotient $\dfrac{x^\top A x}{x^\top x}$ |
|---|---|---|
| 0 | $(1, 0)$ | $2/1 = 2$ |
| 1 | $(2, 1)$ | $(2 \cdot 5 + 1 \cdot 4)/5 = 14/5 = 2.8$ |
| 2 | $(5, 4)$ | $(5 \cdot 14 + 4 \cdot 13)/41 = 122/41 \approx 2.9756$ |
| 3 | $(14, 13)$ | $(14 \cdot 41 + 13 \cdot 40)/365 = 1094/365 \approx 2.99726$ |

The wrong component stays $\tfrac12(1, -1)$ while the right one triples, so the angle error shrinks by $1/3$ per step, and the Rayleigh quotient's error by $1/9$: $3 - 2.8 = 0.2$, $3 - 2.9756 = 0.0244$, $3 - 2.99726 = 0.0027$. After 60 steps both are exact to rounding: `test_hand_example_power_iteration` checks $\lambda = 3$ to $10^{-12}$ and $\lvert v \rvert = (2^{-1/2}, 2^{-1/2})$.

**The spectral radius of a scaled rotation**, $W = 1.5 \begin{pmatrix} \cos 0.7 & -\sin 0.7 \\ \sin 0.7 & \cos 0.7 \end{pmatrix}$ (`test_spectral_radius_of_a_rotation`). Its QR factorization is $Q =$ the rotation itself, $R = 1.5 I$, so $A_1 = RQ = W$: the QR algorithm leaves it unchanged, and the $2 \times 2$ block never becomes triangular. That is the expected outcome for a complex pair. The block's trace is $3\cos 0.7$ and its determinant $2.25$, so

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_power_iteration` | unit, smoke | section 3: $\lambda = 3$, $v = \pm(1, 1)/\sqrt 2$ | you and the tests agree on the method |
| `test_negative_dominant_eigenvalue_keeps_its_sign` | boundary | $\operatorname{diag}(-5, 1, 2)$ gives $-5$ | `M10.5` tells a maximum from a saddle by the sign |
| `test_random_start_finds_the_dominant_direction` | boundary | $\operatorname{diag}(1, 3)$ gives 3, not 1 | a fixed start can be orthogonal to $v_1$ |
| `test_start_vector_draws_from_rng_in_order` | unit | with 0 steps, $v$ is the normalized $2u - 1$ draws | reproducible from the seed (P11) |
| `test_renormalizes_every_step` | boundary | $\lambda = 10$ over 400 steps stays finite | long runs |
| `test_symmetric_matrices_match_numpy` | differential | 20 symmetric matrices, dominant $\pm 2$ against $1.5$, versus `np.linalg.eigh` | the sign and the vector, at a slow ratio |
| `test_zero_map_returns_zero` | boundary | $Av = 0$ gives 0 without NaN; bad `n` and `iters` raise | rank-deficient weights |
| `test_inverse_iteration_finds_the_eigenvalue_nearest_the_shift` | unit | spectrum $\{1, 3, 7\}$: shift 2.9 finds 3, shift 0 finds 1; an exact eigenvalue as the shift raises | the smallest Hessian eigenvalue in `M10.5` |
| `test_inverse_iteration_factors_once` | unit | `lu` is called exactly once for 25 steps | the point of a factorization |
| `test_spectral_radius_of_a_rotation` | unit, smoke | section 3: $\rho = 1.5$ for a scaled rotation | recurrent weights rotate |
| `test_spectral_radius_matches_numpy` | differential | five non-symmetric matrices with real, negative, and complex dominant eigenvalues, versus `np.linalg.eigvals` | the `L3.1` diagnostic on realistic matrices |
| `test_spectral_radius_edge_cases` | boundary | $1 \times 1$, $0 \times 0$, a nilpotent matrix (all eigenvalues 0), non-square input | |
| `test_input_not_modified` | boundary | the caller's $W$ is unchanged | the monitor reads live weights |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| returning $\lVert Av \rVert$ instead of $v^\top A v$ | $+5$ reported for eigenvalue $-5$ | `test_negative_dominant_eigenvalue_keeps_its_sign` (mutant `s01`) |
| a fixed start such as $e_1$ | the wrong eigenvalue whenever $e_1$ is orthogonal to $v_1$ | `test_random_start_finds_the_dominant_direction` (mutant `s02`) |
| normalizing only at the end | `inf`, then NaN, after a few hundred steps | `test_renormalizes_every_step` (mutant `s03`) |
| dividing by $\lVert Av \rVert = 0$ | NaN for a vector in the null space | `test_zero_map_returns_zero` (mutant `s04`) |
| iterating with $A - \sigma I$ instead of its inverse | finds the eigenvalue farthest from $\sigma$ | `test_inverse_iteration_finds_the_eigenvalue_nearest_the_shift` (mutant `s05`) |
| refactoring at every step | correct but `iters` times slower | `test_inverse_iteration_factors_once` (mutant `s06`) |
| spectral radius by power iteration | never converges on rotations; wrong $\rho$ for complex pairs | `test_spectral_radius_of_a_rotation` (mutant `s07`) |
| reading only the diagonal after the QR algorithm | complex pairs reported as their real parts | `test_spectral_radius_matches_numpy` (mutant `s08`) |
| iterating in the caller's array | the weights being monitored change | `test_input_not_modified` (mutant `s09`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M03.2` | `lu` once and `lu_solve` per step in `inverse_iteration` |
| Back | `M03.3` | `qr_householder` drives the QR algorithm in `spectral_radius` |
| Forward | `L0.5` | the training monitor logs $\rho$ of each weight matrix |
| Forward | `L3.1` | the exploding-gradient diagnostic: gradient norms grow when $\rho(W_{hh}) > 1$ |
| Forward | `M10.5` | the top Hessian eigenvalue by `power_iteration` on Hessian-vector products; the edge of stability $2/\lambda_{\max}$ |
| Forward | `M10.1` | the condition number $\kappa = L/\mu$ of a quadratic is $\lambda_{\max}/\lambda_{\min}$ (reading) |

None of the forward modules is in the registry with `M03.4` in its `deps` yet, so `ol verify course M03.4` reports no call site until one lands (course/DEVIATIONS.md, M034-02).

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| unshifted QR algorithm | LAPACK `dhseqr` (via `numpy.linalg.eigvals`) | reduction to Hessenberg form first ($O(n^2)$ per step instead of $O(n^3)$), Wilkinson and Francis double shifts for quadratic convergence, deflation, aggressive early deflation | LAPACK `SRC/dhseqr.f`, `dlahqr.f` |
| power iteration on a `matvec` | Lanczos and Arnoldi (ARPACK, `scipy.sparse.linalg.eigsh`) | keep every iterate and extract eigenvalues from the whole Krylov subspace: many eigenpairs, far fewer products | `scipy/sparse/linalg/_eigen/arpack/` |
| Hessian top eigenvalue (`M10.5`) | PyHessian | power iteration and Lanczos on Hessian-vector products of a real network | Yao et al., "PyHessian" (2020) |
| spectral radius monitoring | spectral normalization in GANs | one power-iteration step per training step to keep $\lVert W \rVert_2 \le 1$ | `torch.nn.utils.parametrizations.spectral_norm` |
