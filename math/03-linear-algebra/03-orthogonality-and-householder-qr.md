<!-- ol:module M03.3 -->
# Orthogonality, Householder QR, and orthogonal initialization

## Overview

| | |
|---|---|
| **Module** | `M03.3` · build · Python · Pass 2 · 3 to 4 h |
| **You build** | `python/tinyllm/linalg/qr.py`: `qr_householder(A)` (reduced $A = QR$ by reflections, with $\operatorname{diag}(R) \ge 0$) and `orthogonal_init(shape, gain, rng)` (weights with orthonormal rows or columns, as `torch.nn.init.orthogonal_` makes them) |
| **Contract** | [`course/contracts/py/tinyllm/linalg/qr.pyi`](../../course/contracts/py/tinyllm/linalg/qr.pyi) |
| **Tests** | `course/tests/M03.3/test_qr.py` (what they check: section 4) |
| **Needs** | `M07.0` normal draws for the Gaussian matrix (or `--ref-deps`). Reading: `M03.2` (elimination, the contrast) |
| **Used by** | `M03.4` the QR algorithm for the spectral radius · later `L0.4` default initializers, `L3.1` and `L3.2` recurrent weight init, `M03.5` least squares · later: `L3.3`, `L3.6` |
| **Milestone** | `MS-P2` (the foundations gate) |
| **Optional depth** | Trefethen and Bau, *Numerical Linear Algebra*, lectures 7 to 10 (QR, Gram-Schmidt, Householder); Saxe, McClelland, and Ganguli, "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks" (2014); Mezzadri, "How to generate random matrices from the classical compact groups" (2007) |

## Key Takeaways

- A matrix $Q$ with **orthonormal columns** ($Q^\top Q = I$) preserves lengths and angles; multiplying by it can neither blow up nor shrink a signal (`test_orthogonal_init_is_orthogonal_and_scaled`).
- **Householder QR** zeroes each column below the diagonal with a reflection $H = I - 2vv^\top$; reflections are orthogonal by construction, so $Q$ stays orthogonal to machine precision even when Gram-Schmidt does not (`test_orthogonality_survives_ill_conditioning`).
- The reflector must send $x$ to $-\operatorname{sign}(x_0)\lVert x \rVert e_1$; the other sign cancels catastrophically (`test_cancellation_sign_choice`).
- Requiring $\operatorname{diag}(R) \ge 0$ makes the factorization unique, so it can be compared with LAPACK entry by entry (`test_random_matrices_match_numpy`), and makes orthogonal initialization uniformly distributed (`test_orthogonal_init_is_uniform_on_signs`).

## How to work this chapter

```bash
ol start M03.3              # stubs python/tinyllm/linalg/qr.py into your repo
ol tests M03.3              # read the test catalog first: rung R0, you write no tests here
ol check M03.3              # exit code is the verdict
ol check M03.3 --ref-deps   # only if your M07.0 is not passing yet
ol diff  M03.3              # after passing: your code against the reference
```

---

## 1. Why now

In Pass 2 you start initializing networks, and the first recurrent model (`L3.1`) multiplies its hidden state by the same matrix $W$ at every one of hundreds of steps. If $W$ stretches some direction by 1.1, that direction grows by $1.1^{100} \approx 13\,781$; if it shrinks one by 0.9, the gradient along it falls to $0.9^{100} \approx 2.7 \times 10^{-5}$. A matrix that stretches nothing and shrinks nothing is an **orthogonal** one, and the standard way to produce a random one is the QR factorization of a Gaussian matrix. QR is also the engine of the next module: `M03.4` finds every eigenvalue's size, including the complex ones that power iteration cannot see, by repeating QR. This module builds QR with Householder reflections, the version that stays accurate on the nearly dependent columns real weight matrices have.

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

Applying $H$ never forms the $m \times m$ matrix: $H B = B - 2 v (v^\top B)$ costs one product and one outer product. The reference applies it to the trailing block of $R$ (columns $j$ onward) and accumulates $Q \leftarrow Q H$ as $Q - 2(Qv)v^\top$ on columns $j$ onward, then writes $\alpha$ and exact zeros into column $j$.

### 2.4 The sign rule and uniqueness

If $A = QR$, then also $A = (QD)(DR)$ for any diagonal $D$ of $\pm 1$'s, since $D^2 = I$. Fixing $\operatorname{diag}(R) \ge 0$ removes that freedom: for a matrix of full column rank the reduced factorization with a positive diagonal is **unique**. The contract applies the rule at the end: $d_i = -1$ where $R_{ii} < 0$, else $+1$; then $Q \leftarrow Q\operatorname{diag}(d)$ and $R \leftarrow \operatorname{diag}(d) R$. LAPACK (`np.linalg.qr`) uses its own signs, so the tests apply the same rule to numpy's answer before comparing entries.

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

Check: $QR = \begin{pmatrix} 0.6 \cdot 5 & 0.6 \cdot 2.2 - 0.8 \cdot 0.4 \\ 0.8 \cdot 5 & 0.8 \cdot 2.2 + 0.6 \cdot 0.4 \end{pmatrix} = \begin{pmatrix} 3 & 1 \\ 4 & 2 \end{pmatrix}$, and the columns of $Q$ are orthonormal: $0.36 + 0.64 = 1$, $-0.48 + 0.48 = 0$. This is `test_hand_example`.

## 4. The interface

```python
def qr_householder(A: ArrayLike) -> tuple[NDArray, NDArray]:
    """Reduced QR, k = min(m, n): Q [m, k] with Q^T Q = I, R [k, n] upper
    triangular with diag(R) >= 0, A == Q @ R. A is not modified."""

def orthogonal_init(shape: tuple[int, int], gain: float, rng) -> NDArray:
    """gain * (orthonormal rows or columns), from the QR of normal(rng, rows * cols)."""
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit, smoke | section 3's $Q$ and $R$ to $10^{-15}$ | you and the tests agree on the reflector and the sign rule |
| `test_random_matrices_match_numpy` | differential | tall, wide, and square shapes against `np.linalg.qr` with the same sign rule, entry by entry | the factorization is the unique one |
| `test_cancellation_sign_choice` | boundary | a first column almost along $e_1$ still gives $A = QR$ | the reflector's sign |
| `test_orthogonality_survives_ill_conditioning` | property | columns with condition number about $10^{10}$: $Q^\top Q = I$ to $10^{-13}$ | why Householder and not Gram-Schmidt |
| `test_rank_deficient_and_zero_columns` | boundary | zero and dependent columns: finite, orthonormal, exact | singular weight matrices, padded inputs |
| `test_input_not_modified_and_2d_required` | boundary | $A$ unchanged; a vector is rejected | `M03.4` keeps iterating on its matrix |
| `test_orthogonal_init_is_orthogonal_and_scaled` | property | $W^\top W = g^2 I$ or $WW^\top = g^2 I$ for five shapes | every singular value equals the gain |
| `test_orthogonal_init_draws_normals_in_c_order` | unit | equals the sign-fixed QR of `normal(rng, r * c)` reshaped in C order | reproducible weights from a seed |
| `test_orthogonal_init_is_uniform_on_signs` | statistical | over 400 seeds, $W_{00} > 0$ about half the time | the Haar distribution |
| `test_orthogonal_init_rejects_bad_shapes` | boundary | zero or negative dimensions raise | config errors fail early |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| accumulating $Q$ from the wrong side ($HQ$ instead of $QH$) | $Q$ is orthogonal but $QR \ne A$ | `test_random_matrices_match_numpy` (mutant `s01`) |
| $\alpha = +\operatorname{sign}(x_0)\lVert x \rVert$ | catastrophic cancellation when a column is nearly along $e_1$ | `test_cancellation_sign_choice` (mutant `s02`) |
| classical Gram-Schmidt | $Q^\top Q$ drifts far from $I$ on ill-conditioned inputs | `test_orthogonality_survives_ill_conditioning` (mutant `s03`) |
| normalizing a zero column | division by zero, NaN everywhere | `test_rank_deficient_and_zero_columns` (mutant `s04`) |
| no sign rule | factors disagree with LAPACK's; the initializer is biased ($W_{00} < 0$ every time) | `test_orthogonal_init_is_uniform_on_signs`, `test_hand_example` (mutant `s05`) |
| reducing the caller's array in place | the next use of $A$ sees $R$ | `test_input_not_modified_and_2d_required` (mutant `s06`) |
| factoring a wide matrix without transposing | rows are not orthonormal | `test_orthogonal_init_draws_normals_in_c_order` (mutant `s07`) |
| applying the gain twice (or squared) | every weight scaled by $g^2$ | `test_orthogonal_init_is_orthogonal_and_scaled` (mutant `s08`) |

## 6. Where it's used next
| Forward | `L3.2` | Registered call site uses this module. |
| Forward | `L3.3` | Registered call site uses this module. |
| Forward | `L3.6` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.0` | `normal(rng, n)` draws the Gaussian matrix $G$ |
| Back | `M03.2` | elimination, the factorization this one is contrasted with (reading) |
| Forward | `M03.4` | `spectral_radius` runs the QR algorithm, $A_{t+1} = R_t Q_t$, with `qr_householder` |
| Forward | `L0.4` | `orthogonal_init` is one of the default initializers of `Linear` |
| Forward | `L3.1`, `L3.2` | recurrent weights start orthogonal, so the hidden state neither explodes nor vanishes at step 0 |
| Forward | `M03.5` | least squares $\min \lVert Ax - b \rVert$ by solving $Rx = Q^\top b$ |

If you skip this module, `ol check M03.4` stops with `BLOCKED ... needs M03.3`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `qr_householder` | LAPACK `dgeqrf` + `dorgqr` | stores reflectors compactly below $R$ and applies blocks of them as $I - VTV^\top$ (the WY form) with matrix-matrix products | LAPACK `SRC/dgeqrf.f`, `dlarft.f` |
| explicit $Q$ | `numpy.linalg.qr(mode="reduced")` | returns $Q$ only when asked; least squares never forms it | `numpy/linalg/_linalg.py` |
| `orthogonal_init` | `torch.nn.init.orthogonal_` | the same recipe on any tensor shape (flattened to 2-D) | `torch/nn/init.py` |
| one QR | tall-skinny QR (TSQR) | QR of row blocks combined in a tree, for distributed and GPU settings | Demmel et al., "Communication-optimal parallel and sequential QR" (2012) |
