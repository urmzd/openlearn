# Dual numbers and forward-mode autodiff

## Overview

| | |
|---|---|
| **Module** | `M08.1` · build · Python · Pass 2 · about 1 to 1.5 h reading |
| **Solution** | [`dual.py`](../../../../course/solutions/python/tinyllm/autograd/dual.py): `Dual` (arithmetic, powers, `@`, indexing), `dual_exp`, `dual_log`, `dual_tanh`, `dual_erf`, `derivative`, `jvp` |
| **Contract** | [`course/contracts/py/tinyllm/autograd/dual.pyi`](../../../../course/contracts/py/tinyllm/autograd/dual.pyi) |
| **Needs** | `M01.3` activation derivatives (the dual derivatives must match them) · `M04.2` numeric JVP · reading: `M02.1` Taylor series |
| **Used by** | `M08.2` the forward-mode oracle for reverse mode · `L0.2` the derivative oracle for every elementwise op, on the way to the Pass 2 result: a bigram trained by the course's own autograd |
| **Optional depth** | Baydin, Pearlmutter, Radul, and Siskind, [*Automatic Differentiation in Machine Learning: a Survey*](https://www.jmlr.org/papers/v18/17-468.html) (JMLR 2018), sections 2 and 3.1; Griewank and Walther, *Evaluating Derivatives* (SIAM, 2nd ed.), ch. 3 |

## Key Takeaways

- A dual number $a + b\varepsilon$ with $\varepsilon^2 = 0$ turns Taylor's theorem into an identity: $f(a + b\varepsilon) = f(a) + f'(a)\, b\, \varepsilon$, so evaluating a function on $x + \varepsilon$ returns its value and its exact derivative together.
- One rule per primitive is enough; the chain rule happens by itself when operations compose.
- With a vector tangent, one forward pass computes a Jacobian-vector product $J v$; a full Jacobian costs one pass per input.
- Dual derivatives have no step size and match closed-form derivatives to rounding, which makes them the oracle for every elementwise op in the autograd library.

## How to read this chapter

Sections 1 to 3 build dual numbers from first principles and work three examples by hand. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

This pass replaces the tracer's count table with a bigram trained by the course's own autograd (`L0.1` to `L0.5`), and an autograd engine is a pile of derivative rules: one per op, each easy to get subtly wrong. So far there are two ways to check a derivative, and neither is good enough as an oracle. The closed forms of `M01.3` cover only the functions differentiated by hand. The central differences of `M04.1` and `M04.2` work for anything, but carry a step-size error near $10^{-10}$ and cannot tell a correct rule from one that is off by $10^{-11}$. Dual numbers give a third way: write the forward computation once, run it on a number that carries its own derivative, and read off the exact derivative of the composition. `L0.2` uses this to check each elementwise op's backward, and `M08.2` checks reverse mode against it.

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

**Vectors and Jacobian-vector products.** Give every input its own tangent: $x + v\varepsilon$ with $v \in \mathbb{R}^n$. The multivariable Taylor expansion has the same shape, $f(x + v\varepsilon) = f(x) + J_f(x)\, v\,\varepsilon$, so the output tangent is the JVP. A linear map passes tangents through itself: $W(x + v\varepsilon) = Wx + (Wv)\varepsilon$. To build the whole Jacobian, run once per basis vector $e_j$, getting column $j$ each time: forward mode costs one pass per input. It is ideal for few inputs and many outputs; a loss with millions of parameters and one output is the opposite case, and the reason `M08.2` builds reverse mode.

**Arrays of dual numbers.** Store a vector of dual numbers as two arrays of one shape, `val` and `eps`, and apply every rule elementwise. Indexing and slicing take the same entries from both arrays; `@` maps the tangent by the same matrix.

**numpy on the left.** `np.float64(2.0) * d` asks numpy first. numpy's scalar and array types try to treat `d` as an element of an object array, and the tangent is lost (or an object array comes back). Setting the class attribute `__array_ufunc__ = None` tells numpy to return `NotImplemented` for any operation with a `Dual`, and Python then calls `Dual.__rmul__`. Every reflected method (`__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`, `__rpow__`, `__rmatmul__`) must keep the operand order: $c - x$ has tangent $-b$, and $c / x$ has tangent $-cb/a^2$.

**Exact versus approximate.** A central difference $(f(x + h) - f(x - h)) / 2h$ has truncation error about $h^2 |f'''|/6$ and rounding error about $u |f| / h$, balanced near $h = 10^{-5}$ at an error near $10^{-10}$ (`M01.1`). A dual derivative is as accurate as evaluating $f$ itself, about $u$ relative. That is why dual derivatives can be compared with `M01.3`'s closed forms at a relative tolerance of $10^{-12}$, but with `M04.2`'s numeric JVP only at $10^{-6}$.

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

The first example is the first case in section 4; the quotient is one of the arithmetic rules.

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

`derivative` seeds `Dual(x, 1.0)` and returns the output's tangent (0.0 when `f` ignores its input). `jvp` seeds `Dual(x, v)` and returns `(f(x), J v)` as float64 arrays. numpy has no `erf`; the solution uses `math.erf`, elementwise for arrays (`np.vectorize`).

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Hand example | unit | $f(1) = e + 3$, $f'(1) = 2e$ for $x e^x + 3$ | pins what a dual number carries |
| Arithmetic rules | unit | 13 primitives and reflected forms against known derivatives | each rule is right on its own |
| Power rules | unit | $x^3$, $x^{0.5}$, $x^{-1}$, $2^x$, $x^x$ | three different rules behind `**` |
| Activation derivatives | differential | sigmoid, tanh, SiLU, both GELUs, softplus at 1001 points in $[-40, 40]$ against `M01.3` | the exact oracle `L0.2` relies on |
| JVP against numeric JVP | differential | $J v$ of a map $\mathbb{R}^4 \to \mathbb{R}^3$ against `M04.2`'s `jvp_numeric` | vector forward mode is right |
| numpy operand on the left | boundary | `np.float64(2) * d`, `1 - d`, `array + d` stay `Dual` | numpy scalars appear everywhere in real code |
| Constant function | boundary | a constant gives 0, `jvp` gives zeros, no nested `Dual` | functions that ignore their input |
| Chain rule | property | $\frac{d}{dx}\tanh(x e^x)$ at 40 points | composition is the chain rule |
| JVP is linear in $v$ | property | linearity in $v$; basis vectors give the Jacobian's columns | the cost model of forward mode |
| Indexing and matmul | unit | slices, `W @ x`, `x @ W.T` move val and tangent together | layers are matrices |

## 5. The solution, step by step

### From first principles

1. A dual number is a pair (value, tangent), and section 2 showed that $f(a + b\varepsilon) = f(a) + f'(a)b\varepsilon$ exactly. So the representation is two fields, `val` and `eps`, and every operation must produce a new pair whose tangent is the derivative rule for that primitive applied to the input tangents.
2. Every binary operation can meet a plain number. A constant is $c + 0\varepsilon$, so the code lifts any non-`Dual` operand to `Dual(c, 0.0)` once (`_lift`) and then writes each rule only for two duals.
3. Commutative operations (`+`, `*`) can reuse the forward method for the reflected one. Non-commutative ones (`-`, `/`, `**`, `@`) cannot: $c - x$ and $x - c$ have opposite tangents, so each reflected method computes `other op self` explicitly.
4. A vector of duals is the same pair with arrays in both fields, and every rule above is already elementwise in numpy. The only array-specific operations are those that mix entries: indexing takes the same entries from both arrays, and `@` applies the same linear map to both, $W(x + v\varepsilon) = Wx + (Wv)\varepsilon$.
5. numpy must not get the first chance at `ndarray op Dual`, or it builds an object array. Setting `__array_ufunc__ = None` makes numpy return `NotImplemented`, so Python falls through to the reflected method.
6. The entry points only seed and read: `derivative` seeds tangent 1 and reads `eps`; `jvp` seeds tangent `v` and returns both fields. A function that ignores its input returns a plain number, whose derivative is 0.

### The code, annotated

The class: two fields, scalars kept as Python floats and arrays copied to one broadcast shape.

```python
class Dual:
    # numpy then returns NotImplemented for `ndarray op Dual`, and Python
    # calls Dual's reflected method instead of building an object array.
    __array_ufunc__ = None

    def __init__(self, val, eps=0.0):
        if isinstance(val, Dual) or isinstance(eps, Dual):
            # nested duals would be second-order AD; this module is first order
            raise TypeError("Dual(val, eps) takes numbers or arrays, not Dual")
        if np.ndim(val) == 0 and np.ndim(eps) == 0:
            self.val, self.eps = float(val), float(eps)   # scalar fast path
        else:
            v = np.asarray(val, dtype=np.float64)
            e = np.asarray(eps, dtype=np.float64)
            shape = np.broadcast_shapes(v.shape, e.shape)
            # own copies of one shape: val and eps must index together
            self.val = np.broadcast_to(v, shape).copy()
            self.eps = np.broadcast_to(e, shape).copy()
```

The arithmetic rules. Each tangent line is the derivative rule from section 2, with the operand order kept in the reflected forms:

```python
    def __sub__(self, other):
        o = _lift(other)
        return Dual(self.val - o.val, self.eps - o.eps)

    def __rsub__(self, other):          # other - self: tangent o.eps - self.eps
        o = _lift(other)
        return Dual(o.val - self.val, o.eps - self.eps)

    def __mul__(self, other):           # product rule: a d + b c
        o = _lift(other)
        return Dual(self.val * o.val, self.val * o.eps + self.eps * o.val)

    def __truediv__(self, other):       # quotient rule: (b c - a d) / c^2
        o = _lift(other)
        return Dual(self.val / o.val, (self.eps * o.val - self.val * o.eps) / (o.val * o.val))

    def __rtruediv__(self, other):      # other / self, so other is the numerator
        return _lift(other).__truediv__(self)

    def __pow__(self, other):
        if isinstance(other, Dual):
            # x^y = exp(y ln x): d = x^y (y' ln x + y x' / x)
            v = self.val**other.val
            return Dual(v, v * (other.eps * np.log(self.val) + other.val * self.eps / self.val))
        k = other                       # constant exponent: k a^(k-1) b
        return Dual(self.val**k, k * self.val ** (k - 1) * self.eps)

    def __rpow__(self, other):          # c^x with constant base: c^a ln c b
        v = other**self.val
        return Dual(v, v * np.log(other) * self.eps)

    def __matmul__(self, other):        # product rule for matrices
        o = _lift(other)
        return Dual(self.val @ o.val, self.val @ o.eps + self.eps @ o.val)

    def __rmatmul__(self, other):       # W @ x: a constant matrix maps both parts
        w = np.asarray(other, dtype=np.float64)
        return Dual(w @ self.val, w @ self.eps)

    def __getitem__(self, idx):         # same entries from value and tangent
        return Dual(self.val[idx], self.eps[idx])


def _lift(x):
    """x as a Dual: a constant gets tangent 0."""
    return x if isinstance(x, Dual) else Dual(x, 0.0)
```

`__add__`, `__radd__`, `__rmul__`, and `__neg__` are the one-line versions of the same pattern (addition and multiplication commute, so the reflected forms call the forward ones). Three separate rules sit behind `**`, because $x^k$, $c^x$, and $x^y$ have different derivatives; using the general $x^y$ rule for a constant exponent would call `log` on a negative base.

The elementwise functions each compute the forward value once and reuse it in the tangent:

```python
def dual_exp(x):
    d = _lift(x)
    e = np.exp(d.val)
    return Dual(e, e * d.eps)                     # (e^a)' = e^a

def dual_log(x):
    d = _lift(x)
    return Dual(np.log(d.val), d.eps / d.val)     # (log a)' = 1/a

def dual_tanh(x):
    d = _lift(x)
    t = np.tanh(d.val)
    return Dual(t, (1.0 - t * t) * d.eps)         # 1 - tanh^2, from the output

def dual_erf(x):
    d = _lift(x)
    # erf' = 2/sqrt(pi) e^{-a^2}; _erf is math.erf, vectorized for arrays
    return Dual(_erf(d.val), TWO_OVER_SQRT_PI * np.exp(-d.val * d.val) * d.eps)
```

The entry points:

```python
def derivative(f, x):
    out = f(Dual(float(x), 1.0))        # seed tangent 1: d/dx
    # a constant function returns a plain number: derivative 0
    return float(out.eps) if isinstance(out, Dual) else 0.0


def jvp(f, x, v):
    xa = np.asarray(x, dtype=np.float64)
    va = np.asarray(v, dtype=np.float64)
    if xa.shape != va.shape:
        raise ValueError(f"x has shape {xa.shape} but v has shape {va.shape}")
    out = f(Dual(xa, va))               # one pass gives f(x) and J v
    if not isinstance(out, Dual):
        y = np.asarray(out, dtype=np.float64)
        return y, np.zeros_like(y)
    y = np.asarray(out.val, dtype=np.float64)
    # an output that does not depend on x may carry a scalar tangent: broadcast it
    return y, np.broadcast_to(np.asarray(out.eps, dtype=np.float64), y.shape).copy()
```

On the worked example, `derivative(lambda x: x * dual_exp(x) + 3, 1.0)` runs `dual_exp` to get $(e, e)$, `__mul__` to get $(e, 1 \cdot e + 1 \cdot e)$, and `__radd__` with $3 + 0\varepsilon$, returning $2e$.

Full solution: [`dual.py`](../../../../course/solutions/python/tinyllm/autograd/dual.py).

### Why this solution

- **Cost.** Each operation does a constant number of extra multiplies, so a forward pass with tangents costs a small constant factor (about 2 to 3) times the plain evaluation, and memory is doubled. A JVP is one pass; a full $m \times n$ Jacobian is $n$ passes.
- **Why not finite differences.** They need a step size and give about 5 to 10 correct digits; dual numbers give the derivative to rounding, which is what an oracle must do.
- **Why not symbolic differentiation.** Symbolic derivatives of compositions grow in size (expression swell) and need a separate evaluator. Dual numbers differentiate the program as it runs, through loops and branches, with no expression tree.
- **Why operator overloading.** The functions being checked (`M01.3`'s activations, `L0.2`'s forward code) are written in numpy-style arithmetic. Overloading lets the same code run on duals unchanged, apart from calling `dual_exp` and friends where numpy's ufuncs would be.
- **Tradeoff accepted.** Forward mode costs one pass per input, so it is the wrong tool for a gradient of a loss with millions of parameters. It is used here as an oracle on small cases, where exactness matters and cost does not.

### Explain it back

1. Why does `__rsub__` not just call `__sub__` the way `__radd__` calls `__add__`?

   <details><summary>Answer</summary>

   Subtraction does not commute. `1 - d` reaches `d.__rsub__(1)`, and `d.__sub__(1)` would compute $d - 1$, whose tangent is $+b$ instead of $-b$. Addition and multiplication commute, so their reflected forms can delegate.

   </details>

2. What goes wrong without `__array_ufunc__ = None`?

   <details><summary>Answer</summary>

   `np.float64(2) * d` is handled by numpy first. It treats `d` as an element of an object array, so the result is an object array instead of a `Dual`, or the tangent is lost (section 2). With the attribute set to `None`, numpy refuses, and Python calls `Dual.__rmul__`, which applies the product rule.

   </details>

3. Why does `__rmatmul__` multiply both fields by `w`, while `__matmul__` has two terms in the tangent?

   <details><summary>Answer</summary>

   In `W @ x` with a constant `W`, $W$ has tangent 0, so the product rule $W'x + Wx'$ collapses to $Wx'$. `__matmul__` handles `x @ y` where both may carry tangents, so it needs both terms. A version that maps only the value would leave the tangent unmapped, and every layer's JVP would be wrong while scalar cases still pass.

   </details>

4. Why does `dual_tanh` use $1 - t^2$ with $t = \tanh(a)$ computed for the value, rather than $1/\cosh^2 a$?

   <details><summary>Answer</summary>

   They are equal, but $1 - t^2$ reuses the forward output, costs one multiply, and stays finite: $\cosh^2 a$ overflows for $|a| > 355$ or so, while $t$ saturates at $\pm 1$ and the tangent goes smoothly to 0.

   </details>

5. Why does `derivative` return 0.0 when `f` returns a plain number, instead of raising?

   <details><summary>Answer</summary>

   A function that ignores its input (a constant) is a legitimate function with derivative 0. Its result never touched a `Dual`, so it comes back as a plain number; treating that as "tangent 0" is exactly the lifting rule for constants.

   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. keeping one term of the product rule | $f'(1) = e$ instead of $2e$ in the worked example | `__mul__` adds both $a\,d$ and $b\,c$ |
| 2. reflected operators that swap the order | $4 - x$ gets slope $+1$; $6/x$ gets the wrong sign | `__rsub__` computes `o - self`; `__rtruediv__` lifts `other` to the numerator |
| 3. no `__array_ufunc__ = None` | `np.float64(2) * d` silently loses the tangent | the class attribute sends numpy operands to the reflected methods |
| 4. `W @ x` that maps the value but not the tangent | JVPs of layers are wrong while scalar cases pass | `__rmatmul__` applies `w` to both `val` and `eps` |
| 5. a sign in the quotient rule, $1 + \tanh^2$, $\mathrm{erf}' = e^{-x^2}/\sqrt\pi$ | wrong slopes for division, tanh, and GELU | $(bc - ad)/c^2$; $1 - t^2$; the constant `TWO_OVER_SQRT_PI` |
| 6. $k x^k$ instead of $k x^{k-1}$; $c^x$ without $\ln c$ | power and exponential slopes off | separate rules for constant exponent, constant base, and both dual |
| 7. seeding the input tangent with 0 | every derivative is 0 | `derivative` seeds `Dual(x, 1.0)`; `jvp` seeds `Dual(x, v)` |
| 8. indexing the value but not the tangent | slices carry another entry's derivative | `__getitem__` indexes both fields with the same `idx` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M01.3` | its closed-form derivatives are what the dual derivatives must match |
| Back | `M04.2` | its `jvp_numeric` approximates the same JVP by central differences |
| Back | `M02.1` | Taylor's theorem is why $f(a + b\varepsilon) = f(a) + f'(a)b\varepsilon$; `erf` as a series |
| Forward | `M08.2` | reverse mode is checked against `Dual` on the same expressions |
| Forward | `L0.2` | each elementwise op's backward is checked against its dual derivative |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Dual` | PyTorch forward-mode AD | dual tensors (`fwAD.make_dual`, `unpack_dual`) and `torch.func.jvp` over every op | `torch/autograd/forward_ad.py` |
| `jvp` | JAX `jax.jvp` | forward mode by tracing, composable with `vmap` and with reverse mode (forward-over-reverse Hessian-vector products, `M08.4`) | `jax/_src/interpreters/ad.py` |
| arrays of duals | Julia ForwardDiff.jl | chunked tangents: up to 12 directions per pass, so a gradient costs $n / 12$ passes | `src/dual.jl` |
