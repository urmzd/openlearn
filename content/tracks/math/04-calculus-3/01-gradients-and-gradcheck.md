<!-- ol:module M04.1 -->
# Partial derivatives, gradients, gradcheck

## Overview

| | |
|---|---|
| **Module** | `M04.1` · build · Python · Pass 2 · 2 to 3 h |
| **You build** | `python/tinyllm/num/gradcheck.py`: `numerical_grad`, `gradcheck`, and its `GradcheckReport` |
| **Contract** | [`course/contracts/py/tinyllm/num/gradcheck.pyi`](../../../../course/contracts/py/tinyllm/num/gradcheck.pyi) |
| **Tests** | `course/tests/M04.1/test_gradcheck.py` (what they check: section 4) |
| **Needs** | `M01.1` `central_diff`, one call per coordinate (or `--ref-deps`) |
| **Used by** | `M04.2` builds `vjp_numeric` on `numerical_grad` · later `L0.2`'s `F.gradcheck_all()` runs every op of your autograd library through `gradcheck`, behind `{tinyllm} gradcheck --suite all` in `MS-L0` |
| **Milestone** | `MS-P2` (the Pass 2 gate) |
| **Optional depth** | OpenStax, *Calculus Volume 3* (free), sections 4.3 and 4.6 (partial derivatives, the gradient); Nocedal and Wright, *Numerical Optimization*, section 8.1 (finite-difference gradients); the PyTorch autograd notes on `gradcheck` |

## Key Takeaways

- A **partial derivative** $\partial f / \partial x_i$ is an ordinary derivative along one coordinate with every other coordinate frozen; the **gradient** $\nabla f$ collects one per input element, in the input's shape (`test_hand_example`, `test_matrix_inputs_and_several_inputs`).
- A numerical gradient is one central difference per element, $2n$ evaluations of $f$ for $n$ elements, exact on quadratics (`test_exact_on_quadratics`).
- An element passes when $|a - n| \le \mathrm{atol} + \mathrm{rtol}\,|n|$: absolute near zero, relative for large gradients (`test_tolerance_is_atol_plus_rtol`).
- `gradcheck` must reject the bugs backward passes really have (a transposed gradient, a factor of 2, one forgotten element) and point at the element furthest past its tolerance (`test_rejects_transposed_gradient`, `test_rejects_one_zeroed_coordinate_and_locates_it`, `test_worst_element_is_the_most_out_of_tolerance`).
- It perturbs float64 **copies**, so the caller's parameters are untouched and integer or float32 inputs still get exact steps (`test_inputs_are_left_unchanged`, `test_integer_and_float32_inputs_are_promoted`).

## How to work this chapter

```bash
ol start M04.1              # stubs python/tinyllm/num/gradcheck.py into your repo
ol tests M04.1              # read the test catalog first: rung R0, you write no tests here
ol check M04.1              # exit code is the verdict
ol check M04.1 --ref-deps   # only if your M01.1 is not passing yet
ol diff  M04.1              # after passing: your code against the reference
```

---

## 1. Why now

`L0.1` and `L0.2` are next: an autograd engine and a library of about 40 differentiable ops, each with a hand-written backward rule. A backward rule is a claim, "this array is the gradient of the loss with respect to that input", and claims need a referee. The referee is the definition of the derivative, applied one coordinate at a time: perturb one weight by $\pm\epsilon$, rerun the forward pass, divide. Your `M01.1` gives the one-coordinate version; this module turns it into the gradient of a scalar function of several arrays and into the check that compares it with an analytic gradient. `MS-L0` runs `{tinyllm} gradcheck --suite all` through your library, so every op you write in Pass 2 passes through this file. (The course's own tests use a frozen copy, `course/tests/_lib/gradcheck.py`, so a buggy `gradcheck` here can fail only this module, never pass a broken op elsewhere: design decision D35.)

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f(X_1, \ldots, X_m)$ | a scalar function of $m$ arrays (a loss) | `Callable[..., float]` |
| $X_k$ | the $k$-th input array; $X_k[i]$ one element, $i$ a multi-index | `float64[...]` |
| $\frac{\partial f}{\partial X_k[i]}$ | the partial derivative with respect to that element | `float` |
| $\nabla_{X_k} f$ | the gradient with respect to $X_k$, same shape as $X_k$ | `float64[...]` |
| $\epsilon$ | the finite-difference step (`eps`), default $10^{-6}$ | `float` |
| $a$, $n$ | one element of the analytic and of the numerical gradient | `float` |
| rtol, atol | relative and absolute tolerances, defaults $10^{-5}$ and $10^{-7}$ | `float` |
| $e_i$ | the array that is 1 at position $i$ and 0 elsewhere | same shape as $X_k$ |

### 2.1 Partial derivatives

For a function of two numbers, $f(x, y) = x^2 y + 3y$, freeze $y$ and differentiate in $x$: $\frac{\partial f}{\partial x} = 2xy$. Freeze $x$ and differentiate in $y$: $\frac{\partial f}{\partial y} = x^2 + 3$. Formally,

$$\frac{\partial f}{\partial x}(x, y) = \lim_{h \to 0}\frac{f(x + h, y) - f(x, y)}{h},$$

the one-variable derivative of `M01.1` along the $x$ direction. A partial derivative measures how $f$ responds to one input when nothing else moves, which is exactly the question a training step asks of each weight.

### 2.2 The gradient

The **gradient** lists every partial derivative: $\nabla f(x, y) = (2xy,\; x^2 + 3)$. For a function of arrays there is one partial derivative per element, and the gradient with respect to $X_k$ is arranged in $X_k$'s shape, so that a training step is simply $X_k \leftarrow X_k - \eta\,\nabla_{X_k} f$. Two facts make the gradient the right object. First, for a small change $\Delta$ in the inputs, $f$ changes by about $\sum_{k,i} \frac{\partial f}{\partial X_k[i]}\,\Delta_k[i]$, the dot product of the gradient with $\Delta$ (a first-order Taylor expansion, `M02.1`). Second, among all directions of length 1 that dot product is largest along the gradient, so $-\nabla f$ is the direction of steepest descent (`M10.1`). A gradient is only defined for a **scalar** $f$; the derivative of a vector-valued function is a matrix, the Jacobian of `M04.2`.

### 2.3 Central differences per coordinate

For each input $k$ and each element $i$:

$$n_k[i] = \frac{f(\ldots, X_k + \epsilon e_i, \ldots) - f(\ldots, X_k - \epsilon e_i, \ldots)}{2\epsilon},$$

which is `central_diff` applied to the one-variable function $t \mapsto f(\ldots, X_k \text{ with element } i \text{ set to } t, \ldots)$ at $t = X_k[i]$, with step $\epsilon$. That is how `numerical_grad` is written: it never re-derives the formula, it calls your `M01.1`. The error analysis carries over: truncation $\frac{1}{6}|f'''|\epsilon^2$, rounding about $\varepsilon |f| / \epsilon$. With $\epsilon = 10^{-6}$ and values of size 1, both are near $10^{-10}$, comfortably under the tolerances below. On a quadratic, $f(x) = \frac12 x^\top A x + b^\top x$ with gradient $Ax + b$, the third derivative is zero and the numerical gradient is exact up to rounding.

Three details decide whether this is trustworthy. **Copies**: perturb float64 copies of the inputs, never the caller's arrays (they are the model's parameters). **Restore**: put each element back before moving to the next, or every later partial derivative is taken at the wrong point. **Promote**: an integer array cannot hold $x + 10^{-6}$ (it rounds back to $x$, and the gradient comes out 0), and float32 moves $x$ only in steps of about $6 \times 10^{-8}|x|$, so the step actually taken is not $\epsilon$. The cost is $2n$ evaluations of $f$ for $n$ input elements: fine for tests on small inputs, hopeless for training, which is why backpropagation exists.

### 2.4 When is a gradient right?

Compare elementwise, with a mixed test:

$$|a - n| \le \mathrm{atol} + \mathrm{rtol}\,|n| .$$

Near zero the absolute term dominates ($10^{-7}$: a numerical gradient of $10^{-9}$ is indistinguishable from 0); for large gradients the relative term does ($10^{-5}|n|$: at $n = 100$ the allowance is $10^{-3}$). A pure relative test would reject a correct analytic 0 against a numerical $10^{-11}$; a pure absolute test would accept a gradient of 100 that is off by 0.5 percent at a loose atol, or reject a correct one at a tight one. A **nan** in either gradient must fail: every comparison with nan is false, so `diff > tol` would let it pass; test `not (diff <= tol)` instead.

The report says `ok`, the largest absolute error, the largest relative error $|a - n| / \max(|n|, \mathrm{atol})$, and **where** to look: the input and multi-index of the element whose error is the largest multiple of its own allowance. That is not the largest absolute error: an error of $5 \times 10^{-3}$ on a gradient of 1000 is within tolerance, an error of $2 \times 10^{-4}$ on a gradient of $10^{-3}$ is a bug.

## 3. Worked example by hand

$f(x) = x_0^2 x_1 + 3 x_1$ at $x = (1, 2)$, with the large step $\epsilon = 0.1$ so you can do the arithmetic. The gradient is $(2 x_0 x_1, x_0^2 + 3) = (4, 4)$.

| Element | $f(x + \epsilon e_i)$ | $f(x - \epsilon e_i)$ | quotient | analytic |
|---|---|---|---|---|
| $x_0$ | $f(1.1, 2) = 1.21 \cdot 2 + 6 = 8.42$ | $f(0.9, 2) = 0.81 \cdot 2 + 6 = 7.62$ | $0.8 / 0.2 = 4$ | 4 |
| $x_1$ | $f(1, 2.1) = 2.1 + 6.3 = 8.4$ | $f(1, 1.9) = 1.9 + 5.7 = 7.6$ | $0.8 / 0.2 = 4$ | 4 |

Both are exact even at $\epsilon = 0.1$: $f$ is quadratic in $x_0$ and linear in $x_1$, so the central difference has no truncation error (section 2.3). Note the restore: the $x_1$ row is evaluated at $x_0 = 1$, not at the 0.9 left over from the row before. A backward pass that returned $(2, 4)$ (a factor of 2 lost on $x_0$) fails at element 0 with $|2 - 4| = 2 > 10^{-7} + 10^{-5} \cdot 4$. This is `test_hand_example`.

## 4. The interface

```python
@dataclass
class GradcheckReport: ok: bool; max_abs_err: float; max_rel_err: float; worst_input: int; worst_index: tuple
def numerical_grad(f, inputs: list[ArrayLike], eps: float = 1e-6) -> list[NDArray]: ...
def gradcheck(f, inputs: list[ArrayLike], analytic: list[ArrayLike],
              eps: float = 1e-6, rtol: float = 1e-5, atol: float = 1e-7) -> GradcheckReport: ...
```

`numerical_grad` returns one float64 array per input, in its shape. `gradcheck` raises `ValueError` for mismatched lists or shapes (a bug in the caller, not a numeric disagreement), and reports everything else. `f` must return a single number.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit, smoke | section 3: $(4, 4)$ at $\epsilon = 0.1$, `ok` | you and the tests agree on the definition |
| `test_exact_on_quadratics` | property | numerical gradient of $\frac12 x^\top A x + b^\top x$ equals $Ax + b$ to $10^{-8}$ | the design's property test |
| `test_matrix_inputs_and_several_inputs` | golden | $\nabla$ of $\mathrm{sum}(\tanh(WX))$ is $G X^\top$ and $W^\top G$ | layers have weights and inputs |
| `test_rejects_transposed_gradient` | unit, smoke | a transposed weight gradient is not ok, worst input 0 | the most common backward bug |
| `test_rejects_gradient_off_by_two` | unit | $x$ instead of $2x$ is rejected, worst at the largest element, relative error 0.5 | a lost factor keeps every sign right |
| `test_rejects_one_zeroed_coordinate_and_locates_it` | unit | one zeroed element of the second input is found at $(2, 1)$ | the report says where to look |
| `test_tolerance_is_atol_plus_rtol` | boundary | the mixed test at gradients $10^{-3}$ and 100 | both regimes of section 2.4 |
| `test_worst_element_is_the_most_out_of_tolerance` | unit | the worst element is the one furthest past its allowance | not the largest absolute error |
| `test_nan_gradient_fails` | boundary | a nan analytic gradient is not ok | nan comparisons are always false |
| `test_inputs_are_left_unchanged` | unit | the caller's arrays come back bit for bit | they are the model's parameters |
| `test_integer_and_float32_inputs_are_promoted` | boundary | int and float32 inputs give exact gradients; $f$ sees float64 | steps survive the dtype |
| `test_default_eps_and_tolerances` | unit | default step $10^{-6}$ and tolerances | what `L0.2` relies on |
| `test_rejects_mismatched_arguments` | boundary | wrong counts and shapes raise `ValueError` | caller bugs are not numeric errors |
| `test_rejects_non_scalar_f` | boundary | a vector-valued $f$ raises; a 1-element array is fine | gradients are for scalars |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. combining the per-element verdicts wrongly (`all` instead of `any`) | a transposed or partly zeroed gradient is reported ok | `test_rejects_transposed_gradient`, `test_rejects_one_zeroed_coordinate_and_locates_it` (mutant `s05`) |
| 2. not restoring an element before the next one | every later partial derivative is taken at a shifted point | `test_hand_example` (mutant `s03`) |
| 2b. perturbing the caller's arrays in place | the model's weights change during a check | `test_inputs_are_left_unchanged` (mutant `s09`) |
| 3. perturbing in the caller's dtype | gradient 0 for int inputs; two digits for float32 | `test_integer_and_float32_inputs_are_promoted` (mutant `s10`) |
| 4. `diff > tol` as the failure test | a nan gradient passes | `test_nan_gradient_fails` (mutant `s08`) |
| a one-sided difference per coordinate | errors near $10^{-6}$ instead of $10^{-10}$; quadratics no longer exact | `test_exact_on_quadratics` (mutant `s01`) |
| dividing by $4\epsilon$ (the $\epsilon$ versus $2\epsilon$ confusion) | every numerical gradient halved | `test_hand_example` (mutant `s02`) |
| differentiating or comparing only the first input | the input gradient is never checked | `test_matrix_inputs_and_several_inputs` (mutant `s04`), `test_rejects_one_zeroed_coordinate_and_locates_it` (mutant `s06`) |
| reporting the largest absolute error as the worst | the report points at a correct large gradient | `test_worst_element_is_the_most_out_of_tolerance` (mutant `s07`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M01.1` | `numerical_grad` calls `central_diff(along, x_i, eps)` for every element |
| Forward | `M04.2` | `vjp_numeric(f, x, u)` is `numerical_grad` of the scalar $u \cdot f(x)$ |
| Forward | `L0.2` | `F.gradcheck_all()` checks every op's backward with your `gradcheck`, behind `{tinyllm} gradcheck --suite all` in `MS-L0` |
| Forward | `M10.1` | gradient descent steps along $-\nabla f$ |
| Forward | `L4.1` to `L7.9` | from rung R5 your own tests gradcheck every backward you write (DESIGN 5.12) |

`L0.2` joins `used_by` when it is authored (`course/DEVIATIONS.md` row B31-02).

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `gradcheck` | PyTorch `torch.autograd.gradcheck` and `gradgradcheck` | complex inputs, sparse and batched layouts, second derivatives, a fast mode that checks random projections $u^\top J v$ instead of every element | `torch/autograd/gradcheck.py` |
| `numerical_grad` | JAX `jax.test_util.check_grads` | checks forward and reverse mode up to a chosen order | `jax/_src/public_test_util.py` |
| `atol + rtol * abs(n)` | `numpy.isclose` | the same mixed test (and the same asymmetry: relative to the second argument) | `numpy/_core/numeric.py` |
| element-by-element checks | directional checks | one $u^\top J v$ per random pair costs 2 evaluations instead of $2n$; `M04.2` builds the pieces | PyTorch `fast_mode=True` |
