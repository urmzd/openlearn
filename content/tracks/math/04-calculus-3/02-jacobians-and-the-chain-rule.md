<!-- ol:module M04.2 -->
# Jacobians, multivariable chain rule, numeric JVP/VJP

## Overview

| | |
|---|---|
| **Module** | `M04.2` · build · Python · Pass 2 · 2 to 3 h |
| **You build** | `python/tinyllm/num/jacobian.py`: `jacobian`, `jvp_numeric` ($Jv$), `vjp_numeric` ($u^\top J$) |
| **Contract** | [`course/contracts/py/tinyllm/num/jacobian.pyi`](../../../../course/contracts/py/tinyllm/num/jacobian.pyi) |
| **Tests** | `course/tests/M04.2/test_jacobian.py` (what they check: section 4) |
| **Needs** | `M04.1` `numerical_grad`, on which `vjp_numeric` is built (or `--ref-deps`). Reading: `M03.1` (matrices and their product) |
| **Used by** | `M08.1` checks dual-number JVPs against `jvp_numeric` · `M08.2` and `M08.3` check reverse-mode VJPs against `vjp_numeric` and `jacobian` · later `L3.1` checks manual backpropagation through time |
| **Milestone** | `MS-P2` (the Pass 2 gate) |
| **Optional depth** | OpenStax, *Calculus Volume 3* (free), section 4.5 (the chain rule for several variables); Baydin et al., "Automatic Differentiation in Machine Learning: a Survey" (2018), sections 3.1 and 3.2 (forward and reverse mode) |

## Key Takeaways

- For $f: \mathbb{R}^n \to \mathbb{R}^m$ the **Jacobian** $J$ is the $m \times n$ matrix of partial derivatives, $J_{ij} = \partial f_i / \partial x_j$: row $i$ is the gradient of output $i$, column $j$ is how every output moves with input $j$ (`test_hand_example`).
- The **chain rule** in several variables is a matrix product, outer function on the left: $J_{g \circ f}(x) = J_g(f(x))\,J_f(x)$ (`test_chain_rule_is_a_matrix_product`).
- Autodiff never forms $J$. **Forward mode** pushes a direction through, $Jv$; **reverse mode** pulls a cotangent back, $u^\top J$, which is the gradient of the scalar $u \cdot f(x)$ (`test_hand_example_products`).
- The two products are consistent: $u^\top(Jv) = (u^\top J)v$ for every $u$ and $v$ (`test_jvp_and_vjp_agree`).
- The JVP step must scale with $v$: $h = 10^{-6}/\max|v|$, so the input moves by $10^{-6}$ whatever the size of $v$ (`test_jvp_scales_its_step_to_v`).

## How to work this chapter

```bash
ol start M04.2              # stubs python/tinyllm/num/jacobian.py into your repo
ol tests M04.2              # read the test catalog first: rung R0, you write no tests here
ol check M04.2              # exit code is the verdict
ol check M04.2 --ref-deps   # only if your M04.1 is not passing yet
ol diff  M04.2              # after passing: your code against the reference
```

---

## 1. Why now

`M04.1` checks the gradient of a scalar loss. But your autodiff engine never handles the loss as one formula: it handles a chain of layers, each mapping arrays to arrays, and its backward pass multiplies through them one at a time. The objects being multiplied are Jacobians, or rather products with Jacobians, because the matrices themselves are far too large to form (a layer from 4096 to 4096 numbers has a Jacobian of 16 million entries per token). The next three modules build automatic differentiation three ways: dual numbers (`M08.1`, forward mode), a scalar `Value` graph (`M08.2`, reverse mode), and closed-form VJPs of matrix expressions (`M08.3`). Each needs an independent numeric oracle for $Jv$ and $u^\top J$. This module defines the Jacobian and the chain rule from partial derivatives and builds those oracles.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f: \mathbb{R}^n \to \mathbb{R}^m$ | a function from $n$ numbers to $m$ numbers | `Callable[[NDArray], ArrayLike]` |
| $x$ | the input; any shape, flattened row-major to $n$ numbers | `float64[...]` |
| $f_i$ | output $i$ of $f$ (row-major order of $f(x)$) | `float` |
| $J = J_f(x)$ | the Jacobian, $J_{ij} = \partial f_i / \partial x_j$ | `float64[m, n]` |
| $v \in \mathbb{R}^n$ | a **tangent**: a direction to move the input | shape of $x$ |
| $u \in \mathbb{R}^m$ | a **cotangent**: a weighting of the outputs | shape of $f(x)$ |
| $Jv$ | the Jacobian-vector product (JVP) | shape of $f(x)$ |
| $u^\top J$ | the vector-Jacobian product (VJP) | shape of $x$ |
| $\epsilon$, $h$ | finite-difference steps | `float` |

### 2.1 The Jacobian

Each output $f_i$ is a scalar function of $x$, so it has a gradient (`M04.1`). Stack those gradients as rows and you get the **Jacobian**:

$$J = \begin{pmatrix} \partial f_1/\partial x_1 & \cdots & \partial f_1/\partial x_n \\ \vdots & & \vdots \\ \partial f_m/\partial x_1 & \cdots & \partial f_m/\partial x_n \end{pmatrix} .$$

It is the best **linear approximation** of $f$ near $x$: $f(x + \Delta) \approx f(x) + J\Delta$ for small $\Delta$, the several-variable version of the tangent line. Column $j$ is how the whole output moves per unit move of input $j$, which is also how to compute it numerically:

$$J_{:, j} \approx \frac{f(x + \epsilon e_j) - f(x - \epsilon e_j)}{(x_j + \epsilon) - (x_j - \epsilon)} ,$$

$2n$ evaluations of $f$, dividing by the step actually taken (`M01.1`). Three examples recur. A linear map $f(x) = Ax$ has $J = A$ everywhere. An elementwise map like $\tanh$ has a diagonal Jacobian, $\operatorname{diag}(1 - \tanh^2 x)$: output $i$ depends only on input $i$. Softmax, $p_i = e^{z_i}/\sum_k e^{z_k}$, couples everything: $\partial p_i / \partial z_j = p_i(\delta_{ij} - p_j)$, so $J = \operatorname{diag}(p) - pp^\top$, symmetric with columns summing to 0 (the probabilities always add to 1). `M08.3` derives that one by hand. When $x$ or $f(x)$ is a matrix, flatten both in row-major order (`M03.1`): the derivative with respect to $X[1, 2]$ of a $2 \times 3$ input is column $1 \cdot 3 + 2 = 5$.

### 2.2 The chain rule

If $y = f(x)$ and $z = g(y)$, then a small change $\Delta$ in $x$ moves $y$ by about $J_f \Delta$, which moves $z$ by about $J_g (J_f \Delta)$. So

$$J_{g \circ f}(x) = J_g\bigl(f(x)\bigr)\, J_f(x) ,$$

a product of a $p \times m$ and an $m \times n$ matrix, the outer function's Jacobian on the left, evaluated at the inner function's output. Entry by entry this is $\frac{\partial z_k}{\partial x_j} = \sum_i \frac{\partial z_k}{\partial y_i}\frac{\partial y_i}{\partial x_j}$: add up every path from $x_j$ to $z_k$. The one-variable chain rule is the $1 \times 1$ case. A deep network is a long composition, and its Jacobian is a long product.

### 2.3 Products without the matrix

Forming $J$ costs $n$ forward passes and $mn$ memory. Two products avoid it:

- **JVP**, $Jv$: the derivative of $t \mapsto f(x + tv)$ at $t = 0$, the rate at which the output moves when the input moves in direction $v$. One central difference computes it: $\frac{f(x + hv) - f(x - hv)}{2h}$. Forward-mode autodiff (`M08.1`) computes it exactly by carrying a tangent alongside every value. Along a chain, $J_g J_f v = J_g (J_f v)$: push $v$ through one layer at a time.
- **VJP**, $u^\top J$: the gradient of the scalar $u \cdot f(x) = \sum_i u_i f_i(x)$, since $\frac{\partial}{\partial x_j}\sum_i u_i f_i = \sum_i u_i J_{ij}$. `vjp_numeric` computes it with `M04.1`'s `numerical_grad`. Reverse-mode autodiff (backpropagation, `M08.2`, `L0.1`) computes it exactly by pulling $u$ back one layer at a time: $u^\top J_g J_f = (u^\top J_g) J_f$. With $u = 1$ and a scalar loss, the VJP is the gradient; that is why training uses reverse mode.

Both products describe the same $J$, so for every $u$ and $v$, $u^\top(Jv) = (u^\top J)v$: one number computed two ways, which `test_jvp_and_vjp_agree` checks at random $u$, $v$, $x$. Mixing them up (returning $Ju$ for a VJP) gives the right shape whenever $J$ is square and the wrong numbers unless it is symmetric.

**Scale the JVP step to $v$.** $h$ multiplies $v$, so the input moves by $h\max|v|$. A fixed $h = 10^{-6}$ with $|v| = 10^6$ moves $x$ by 1 and measures a secant over a whole unit. Choosing $h = 10^{-6}/\max|v|$ keeps the largest coordinate's move at $10^{-6}$; then $J(cv) = c\,Jv$ to rounding, and $v = 0$ returns 0 without computing $0/0$.

## 3. Worked example by hand

$f(x, y) = (x^2 y,\; 5x + \sin y)$ at $(1, 2)$. Partial derivatives:

$$J = \begin{pmatrix} \partial_x(x^2 y) & \partial_y(x^2 y) \\ \partial_x(5x + \sin y) & \partial_y(5x + \sin y) \end{pmatrix} = \begin{pmatrix} 2xy & x^2 \\ 5 & \cos y \end{pmatrix} = \begin{pmatrix} 4 & 1 \\ 5 & -0.4161468 \end{pmatrix} .$$

With $v = (1, 0)$: $Jv = (4, 5)$, the first column: moving $x$ moves the outputs at rates 4 and 5. With $u = (1, 0)$: $u^\top J = (4, 1)$, the first row: the gradient of the first output. $J$ is not symmetric, so the two differ, and a transposed Jacobian swaps them. Check the identity with $u = (1, 1)$ and $v = (0, 1)$: $Jv = (1, -0.4161468)$, $u \cdot Jv = 0.5838532$; $u^\top J = (9, 0.5838532)$, $(u^\top J) \cdot v = 0.5838532$. These are `test_hand_example` and `test_hand_example_products`.

## 4. The interface

```python
def jacobian(f, x: ArrayLike, eps: float = 1e-6) -> NDArray: ...   # [f(x).size, x.size]
def jvp_numeric(f, x: ArrayLike, v: ArrayLike) -> NDArray: ...      # J v, shape f(x).shape
def vjp_numeric(f, x: ArrayLike, u: ArrayLike) -> NDArray: ...      # u^T J, shape x.shape
```

All arithmetic is float64 on a private copy of `x`. `jacobian` raises `ValueError` for a bad `eps` or when `f`'s output shape changes; the products raise it when `v` does not have `x`'s shape or `u` does not have `f(x)`'s.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit, smoke | section 3's $J$ | you and the tests agree on rows and columns |
| `test_hand_example_products` | unit | $Jv = (4, 5)$ and $u^\top J = (4, 1)$ | forward and reverse mode are different products |
| `test_shapes_flatten_row_major` | unit | $x$ of shape $(2, 3)$ gives $J$ of shape $(4, 6)$ in row-major columns; product shapes; shape errors | backward passes return gradients in the input's shape |
| `test_linear_map_jacobian_is_the_matrix` | golden | $J_{Ax} = A$ | `M03.1`'s matmul is every layer's first step |
| `test_softmax_jacobian_closed_form` | golden | $\operatorname{diag}(p) - pp^\top$, columns sum to 0 | `M08.3` derives this VJP by hand |
| `test_elementwise_jacobian_is_diagonal` | golden | $J_{\tanh} = \operatorname{diag}(1 - \tanh^2)$; its VJP is elementwise | activation backward passes (`M01.3`) |
| `test_chain_rule_is_a_matrix_product` | differential | $J_{g \circ f} = J_g(f(x)) J_f(x)$ | the rule backpropagation applies |
| `test_jvp_and_vjp_agree` | property, smoke | $Jv$, $u^\top J$, and $u^\top(Jv) = (u^\top J)v$ at random points | the design's property test; forward and reverse oracles |
| `test_jvp_scales_its_step_to_v` | boundary | $J(10^6 v) = 10^6 Jv$, $J(10^{-6}v) = 10^{-6}Jv$, $J0 = 0$ | tangents of any size |
| `test_inputs_are_left_unchanged_and_float64` | unit | the caller's $x$ is untouched; $f$ sees float64 even for int $x$ | parameters are not perturbed in place |
| `test_rejects_bad_eps_and_changing_shapes` | boundary | `eps = 0` and an $f$ whose output shape changes raise | no ragged Jacobians |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the Jacobian transposed (columns stacked as rows) | $J^\top$: right shape for square $J$, wrong numbers, and a chain rule product in the wrong order | `test_hand_example`, `test_chain_rule_is_a_matrix_product` (mutant `s01`) |
| 2. perturbing the caller's $x$ and not restoring it | the input array changes during a check | `test_inputs_are_left_unchanged_and_float64` (mutant `s08`) |
| 2b. leaving coordinate $j$ at $x_j - \epsilon$ | every later column is taken at a shifted point | `test_hand_example` (mutant `s09`) |
| 3. a fixed JVP step whatever the size of $v$ | $J(10^6 v)$ is a secant over a unit step | `test_jvp_scales_its_step_to_v` (mutant `m04`) |
| 4. a VJP that computes $Ju$ instead of $u^\top J$ | the column $(4, 5)$ where the row $(4, 1)$ was wanted | `test_hand_example_products` (mutant `s04`) |
| one-sided differences for the columns | errors near $10^{-6}$; softmax's closed form missed | `test_softmax_jacobian_closed_form` (mutant `s02`) |
| dividing the JVP by $h$ instead of $2h$ | every JVP doubled | `test_hand_example_products` (mutant `s03`) |
| flattening $x$ through a copy | the perturbations never reach $f$: $J = 0$ | `test_shapes_flatten_row_major` (mutant `s05`) |
| the column difference reversed | $-J$ | `test_softmax_jacobian_closed_form` (mutant `s06`) |
| a VJP that ignores $u$ | the gradient of $\sum_i f_i$ for every $u$ | `test_jvp_and_vjp_agree` (mutant `s07`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M04.1` | `vjp_numeric` is `numerical_grad(lambda z: sum(u * f(z)), [x])` |
| Back | `M03.1` | matrices, row-major layout, and the product (reading) |
| Forward | `M08.1` | dual numbers compute $Jv$ exactly; its tests compare them with `jvp_numeric` |
| Forward | `M08.2` | the scalar `Value` graph computes $u^\top J$ by backpropagation, checked against `vjp_numeric` |
| Forward | `M08.3` | closed-form VJPs of matmul, softmax, LayerNorm, RMSNorm, and cross-entropy, checked against `jacobian` and `vjp_numeric` |
| Forward | `L3.1` | manual backpropagation through time, a chain of Jacobians over time steps |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `jvp_numeric`, `vjp_numeric` | JAX `jax.jvp`, `jax.vjp` | exact products by forward and reverse mode, composable with `jit` and `vmap` | `jax/_src/api.py` |
| `jacobian` | PyTorch `torch.autograd.functional.jacobian` (`vectorize=True`), JAX `jacfwd` and `jacrev` | the full Jacobian from $n$ JVPs or $m$ VJPs, batched; pick forward when $n < m$, reverse when $m < n$ | `torch/autograd/functional.py` |
| the chain rule | reverse-mode autodiff (backpropagation) | one VJP per layer, in reverse order, storing the forward values it needs | your `L0.1`; Baydin et al. (2018) |
| $u^\top(Jv) = (u^\top J)v$ | PyTorch `gradcheck(fast_mode=True)` | checks a whole backward with one random $u$ and $v$ instead of $mn$ entries | `torch/autograd/gradcheck.py` |
