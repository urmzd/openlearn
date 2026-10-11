<!-- ol:module M08.1 -->
# Dual numbers and forward-mode autodiff

## Overview

| | |
|---|---|
| **Module** | `M08.1` · build · Python · Pass 2 · 2 to 3 h |
| **You build** | `python/tinyllm/autograd/dual.py`: `Dual` (arithmetic, powers, `@`, indexing), `dual_exp`, `dual_log`, `dual_tanh`, `dual_erf`, `derivative`, `jvp` |
| **Contract** | [`course/contracts/py/tinyllm/autograd/dual.pyi`](../../course/contracts/py/tinyllm/autograd/dual.pyi) |
| **Tests** | `course/tests/M08.1/` (what they check: section 4) |
| **Needs** | `M01.3` activation derivatives (the tests compare against them) · `M04.2` numeric JVP · reading: `M02.1` Taylor series (or `--ref-deps`) |
| **Used by** | `M08.2` the forward-mode oracle for reverse mode · `L0.2` the derivative oracle for every elementwise op |
| **Milestone** | `MS-P2` (Pass 2 gate: every math module of the pass checks green, then your autograd bigram trains) |
| **Optional depth** | Baydin, Pearlmutter, Radul, and Siskind, [*Automatic Differentiation in Machine Learning: a Survey*](https://www.jmlr.org/papers/v18/17-468.html) (JMLR 2018), sections 2 and 3.1; Griewank and Walther, *Evaluating Derivatives* (SIAM, 2nd ed.), ch. 3 |

## Key Takeaways

- A dual number $a + b\varepsilon$ with $\varepsilon^2 = 0$ turns Taylor's theorem into an identity: $f(a + b\varepsilon) = f(a) + f'(a)\, b\, \varepsilon$, so evaluating a function on $x + \varepsilon$ returns its value and its exact derivative together (`test_hand_example`).
- One rule per primitive is enough; the chain rule happens by itself when operations compose (`test_arithmetic_rules`, `test_chain_rule_composition`).
- With a vector tangent, one forward pass computes a Jacobian-vector product $J v$; a full Jacobian costs one pass per input (`test_jvp_matches_numeric_jvp`, `test_jvp_is_linear_in_v`).
- Dual derivatives have no step size and match closed-form derivatives to rounding, which makes them the oracle for every elementwise op in your autograd (`test_matches_activation_derivatives`).

## How to work this chapter

```bash
ol start M08.1              # stubs dual.py into your repo, contract alongside
ol tests M08.1              # read the test catalog first: rung R0, you write no tests here
ol check M08.1              # exit code is the verdict
ol check M08.1 --ref-deps   # only if your M01.3 or M04.2 is not passing yet
ol diff  M08.1              # after passing: your code against the reference
```

---

## 1. Why now

This pass replaces the tracer's count table with a bigram trained by your own autograd (`L0.1` to `L0.5`), and an autograd engine is a pile of derivative rules: one per op, each easy to get subtly wrong. So far you have two ways to check a derivative, and neither is good enough as an oracle. The closed forms of `M01.3` cover only the functions you differentiated by hand. The central differences of `M04.1` and `M04.2` work for anything, but carry a step-size error near $10^{-10}$ and cannot tell a correct rule from one that is off by $10^{-11}$. Dual numbers give a third way: write the forward computation once, run it on a number that carries its own derivative, and read off the exact derivative of the composition. `L0.2` uses this to check each elementwise op's backward, and `M08.2` checks reverse mode against it.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $\varepsilon$ | a formal symbol with $\varepsilon^2 = 0$ (and $\varepsilon \ne 0$) | |
| $a + b\varepsilon$ | a dual number: value $a$, tangent $b$ | `Dual(val, eps)` |
| $f, f'$ | a differentiable function and its derivative | |
| $x \in \mathbb{R}^n$ | an input point | `float64[n]` |
| $v \in \mathbb{R}^n$ | a tangent direction | `float64[n]` |
| $J_f(x) \in \mathbb{R}^{m \times n}$ | the Jacobian of $f: \mathbb{R}^n \to \mathbb{R}^m$, $J_{ij} = \partial f_i / \partial x_j$ | `float64[m, n]` |
| $J_f(x)\, v$ | the Jacobian-vector product (JVP) | `float64[m]` |
| $W$ | a constant matrix | `float64[m, n]` |
| $u$ | unit roundoff of float64, about $1.1 \times 10^{-16}$ | scalar |

**The algebra.** Dual numbers add and multiply like polynomials in $\varepsilon$, then drop every $\varepsilon^2$:

$$(a + b\varepsilon) + (c + d\varepsilon) = (a + c) + (b + d)\varepsilon,$$
$$(a + b\varepsilon)(c + d\varepsilon) = ac + (ad + bc)\varepsilon + bd\,\varepsilon^2 = ac + (ad + bc)\varepsilon.$$

The tangent of a product is the product rule. Division follows by multiplying with the conjugate $c - d\varepsilon$, since $(c + d\varepsilon)(c - d\varepsilon) = c^2$:

$$\frac{a + b\varepsilon}{c + d\varepsilon} = \frac{(a + b\varepsilon)(c - d\varepsilon)}{c^2} = \frac{a}{c} + \frac{bc - ad}{c^2}\varepsilon,$$

the quotient rule. A plain number $c$ is the dual number $c + 0\varepsilon$: a constant has tangent 0.

**Why it computes derivatives.** Taylor's theorem (`M02.1`) expands a smooth $f$ around $a$:

$$f(a + h) = f(a) + f'(a)\,h + \tfrac12 f''(a)\,h^2 + \dots$$

Put $h = b\varepsilon$. Every term from $h^2$ on contains $\varepsilon^2 = 0$, so

$$f(a + b\varepsilon) = f(a) + f'(a)\, b\, \varepsilon \quad \text{exactly.}$$

No step size, no truncation: the tangent is the derivative times the input tangent, and floating point adds only the rounding of each operation.

**One rule per primitive.** Read off each function's dual rule from its derivative:

| Function | Dual rule |
|---|---|
| $e^x$ | $e^a + e^a\, b\,\varepsilon$ |
| $\log x$ | $\log a + (b / a)\,\varepsilon$ |
| $\tanh x$ | $\tanh a + (1 - \tanh^2 a)\, b\,\varepsilon$ |
| $\mathrm{erf}\, x$ | $\mathrm{erf}\, a + \tfrac{2}{\sqrt\pi} e^{-a^2}\, b\,\varepsilon$ |
| $x^k$, constant $k$ | $a^k + k a^{k-1}\, b\,\varepsilon$ |
| $c^x$, constant $c > 0$ | $c^a + c^a \ln c\; b\,\varepsilon$ |
| $x^y$, both dual | $a^c + a^c\left(d \ln a + c\,b/a\right)\varepsilon$ for $x = a + b\varepsilon$, $y = c + d\varepsilon$ |

**The chain rule comes free.** If $f(a + \varepsilon) = f(a) + f'(a)\varepsilon$, then feeding that into $g$ gives $g(f(a)) + g'(f(a))\, f'(a)\,\varepsilon$: the derivative of $g \circ f$. Every program built from the primitives is differentiated by running it.

**Vectors and Jacobian-vector products.** Give every input its own tangent: $x + v\varepsilon$ with $v \in \mathbb{R}^n$. The multivariable Taylor expansion has the same shape, $f(x + v\varepsilon) = f(x) + J_f(x)\, v\,\varepsilon$, so the output tangent is the JVP. A linear map passes tangents through itself: $W(x + v\varepsilon) = Wx + (Wv)\varepsilon$. To build the whole Jacobian you run once per basis vector $e_j$, getting column $j$ each time: forward mode costs one pass per input. It is ideal for few inputs and many outputs; a loss with millions of parameters and one output is the opposite case, and the reason `M08.2` builds reverse mode.

**Arrays of dual numbers.** Store a vector of dual numbers as two arrays of one shape, `val` and `eps`, and apply every rule elementwise. Indexing and slicing take the same entries from both arrays; `@` maps the tangent by the same matrix.

**numpy on the left.** `np.float64(2.0) * d` asks numpy first. numpy's scalar and array types try to treat `d` as an element of an object array, and the tangent is lost (or you get an object array back). Setting the class attribute `__array_ufunc__ = None` tells numpy to return `NotImplemented` for any operation with a `Dual`, and Python then calls `Dual.__rmul__`. Every reflected method (`__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`, `__rpow__`, `__rmatmul__`) must keep the operand order: $c - x$ has tangent $-b$, and $c / x$ has tangent $-cb/a^2$.

**Exact versus approximate.** A central difference $(f(x + h) - f(x - h)) / 2h$ has truncation error about $h^2 |f'''|/6$ and rounding error about $u |f| / h$, balanced near $h = 10^{-5}$ at an error near $10^{-10}$ (`M01.1`). A dual derivative is as accurate as evaluating $f$ itself, about $u$ relative. That is why the tests compare dual derivatives with `M01.3`'s closed forms at a relative tolerance of $10^{-12}$, but with `M04.2`'s numeric JVP only at $10^{-6}$.

## 3. Worked example by hand

**$f(x) = x e^x + 3$ at $x = 1$.** Seed the input with tangent 1: $x = 1 + \varepsilon$.

| step | value | tangent | rule |
|---|---|---|---|
| $x$ | 1 | 1 | seed |
| $e^x$ | $e = 2.718282$ | $e \cdot 1 = 2.718282$ | exp |
| $x \cdot e^x$ | $1 \cdot e = 2.718282$ | $1 \cdot e + 1 \cdot e = 5.436564$ | product |
| $+ 3$ | $5.718282$ | $5.436564$ | constant |

So $f(1) = e + 3 = 5.718282$ and $f'(1) = 2e = 5.436564$. By hand, $f'(x) = e^x + x e^x$, which is $2e$ at 1. Same number, from one evaluation.

**A quotient, $(x + 1)/(x - 1)$ at $x = 3$.** Numerator $4 + \varepsilon$, denominator $2 + \varepsilon$; by the quotient rule the tangent is $(1 \cdot 2 - 4 \cdot 1)/2^2 = -0.5$. The function is 2 there, and its derivative $-2/(x-1)^2 = -0.5$.

**A JVP.** $f(x_1, x_2) = (x_1 x_2,\; x_1 + x_2^2)$ at $(2, 3)$ along $v = (1, 0)$: $x_1 = 2 + \varepsilon$, $x_2 = 3 + 0\varepsilon$. Then $x_1 x_2 = 6 + 3\varepsilon$ and $x_1 + x_2^2 = 11 + \varepsilon$, so $J v = (3, 1)$: the first column of $J = \begin{bmatrix} x_2 & x_1 \\ 1 & 2x_2 \end{bmatrix} = \begin{bmatrix} 3 & 2 \\ 1 & 6 \end{bmatrix}$.

The first example is the first test case in section 4, `test_hand_example`; the quotient is a row of `test_arithmetic_rules`.

## 4. The interface

```python
# python/tinyllm/autograd/dual.py
class Dual:
    __array_ufunc__ = None
    def __init__(self, val, eps=0.0) -> None    # floats, or float64 arrays of one shape
    # + - * / ** @ with Dual, numbers, or arrays on either side; unary -; d[idx]
def dual_exp(x) -> Dual; def dual_log(x) -> Dual; def dual_tanh(x) -> Dual; def dual_erf(x) -> Dual
def derivative(f: Callable[[Dual], Any], x: float) -> float
def jvp(f: Callable[[Dual], Any], x: ArrayLike, v: ArrayLike) -> tuple[NDArray, NDArray]
```

`derivative` seeds `Dual(x, 1.0)` and returns the output's tangent (0.0 when `f` ignores its input). `jvp` seeds `Dual(x, v)` and returns `(f(x), J v)` as float64 arrays. numpy has no `erf`; use `math.erf`, elementwise for arrays (`np.vectorize`).

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | $f(1) = e + 3$, $f'(1) = 2e$ for $x e^x + 3$ | you and the test agree on what a dual number carries |
| `test_arithmetic_rules` | unit | 13 primitives and reflected forms against known derivatives | each rule is right on its own |
| `test_pow_rules` | unit | $x^3$, $x^{0.5}$, $x^{-1}$, $2^x$, $x^x$ | three different rules behind `**` |
| `test_matches_activation_derivatives` | differential | sigmoid, tanh, SiLU, both GELUs, softplus at 1001 points in $[-40, 40]$ against `M01.3` | the exact oracle `L0.2` relies on |
| `test_jvp_matches_numeric_jvp` | differential | $J v$ of a map $\mathbb{R}^4 \to \mathbb{R}^3$ against `M04.2`'s `jvp_numeric` | vector forward mode is right |
| `test_numpy_operand_on_the_left` | boundary | `np.float64(2) * d`, `1 - d`, `array + d` stay `Dual` | numpy scalars appear everywhere in real code |
| `test_constant_function_has_zero_derivative` | boundary | a constant gives 0, `jvp` gives zeros, no nested `Dual` | functions that ignore their input |
| `test_chain_rule_composition` | property | $\frac{d}{dx}\tanh(x e^x)$ at 40 points | composition is the chain rule |
| `test_jvp_is_linear_in_v` | property | linearity in $v$; basis vectors give the Jacobian's columns | the cost model of forward mode |
| `test_vector_duals_index_and_matmul` | unit | slices, `W @ x`, `x @ W.T` move val and tangent together | layers are matrices |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. keeping one term of the product rule | $f'(1) = e$ instead of $2e$ in the worked example | `test_hand_example` (mutant `s01`) |
| 2. reflected operators that swap the order | $4 - x$ gets slope $+1$; $6/x$ gets the wrong sign | `test_arithmetic_rules` (mutants `s03`, `s04`, `m01`) |
| 3. no `__array_ufunc__ = None` | `np.float64(2) * d` silently loses the tangent | `test_numpy_operand_on_the_left` (mutant `s08`) |
| 4. `W @ x` that maps the value but not the tangent | JVPs of layers are wrong while scalar tests pass | `test_vector_duals_index_and_matmul` (mutant `s12`) |
| 5. a sign in the quotient rule, $1 + \tanh^2$, $\mathrm{erf}' = e^{-x^2}/\sqrt\pi$ | wrong slopes for division, tanh, and GELU | `test_arithmetic_rules` (mutants `s02`, `s07`, `s14`), `test_matches_activation_derivatives` (mutants `s05`, `s06`) |
| 6. $k x^k$ instead of $k x^{k-1}$; $c^x$ without $\ln c$ | power and exponential slopes off | `test_pow_rules` (mutants `s10`, `s11`) |
| 7. seeding the input tangent with 0 | every derivative is 0 | `test_constant_function_has_zero_derivative` (mutant `s09`) |
| 8. indexing the value but not the tangent | slices carry another entry's derivative | `test_vector_duals_index_and_matmul` (mutant `s13`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M01.3` | its closed-form derivatives are what the dual derivatives must match |
| Back | `M04.2` | its `jvp_numeric` approximates the same JVP by central differences |
| Back | `M02.1` | Taylor's theorem is why $f(a + b\varepsilon) = f(a) + f'(a)b\varepsilon$; `erf` as a series |
| Forward | `M08.2` | reverse mode is checked against `Dual` on the same expressions |
| Forward | `L0.2` | each elementwise op's backward is checked against its dual derivative |

If you skip this module, `ol check M08.2` stops with `M08.2 needs M08.1`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Dual` | PyTorch forward-mode AD | dual tensors (`fwAD.make_dual`, `unpack_dual`) and `torch.func.jvp` over every op | `torch/autograd/forward_ad.py` |
| `jvp` | JAX `jax.jvp` | forward mode by tracing, composable with `vmap` and with reverse mode (forward-over-reverse Hessian-vector products, `M08.4`) | `jax/_src/interpreters/ad.py` |
| arrays of duals | Julia ForwardDiff.jl | chunked tangents: up to 12 directions per pass, so a gradient costs $n / 12$ passes | `src/dual.jl` |
