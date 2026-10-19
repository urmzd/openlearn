# Dual numbers and forward-mode autodiff

An automatic differentiation engine is a pile of derivative rules: one per operation, each easy to get subtly wrong. Before you build one, you need a way to check those rules that is better than the tools you have so far. The closed forms of [Activation Functions and Their Derivatives](../01-calculus-1/03-activation-functions.md) cover only the functions someone differentiated by hand. The central differences of [Partial derivatives, gradients, and gradcheck](../04-calculus-3/01-gradients-and-gradcheck.md) and [Jacobians, the multivariable chain rule, and numeric JVP/VJP](../04-calculus-3/02-jacobians-and-the-chain-rule.md) work for anything, but they carry a step-size error near $10^{-10}$ and cannot tell a correct rule from one that is off by $10^{-11}$. Neither is good enough to serve as an oracle.

Dual numbers give a third way. You write the forward computation once, run it on a number that carries its own derivative, and read off the exact derivative of the whole composition. No step size, no symbolic algebra: the derivative falls out of ordinary arithmetic. In *Build Your Own LLM System*, [The op library and gradcheck_all](../../build-an-llm/p00-foundations/02-op-library-and-gradcheck.md) uses this to check the backward of every elementwise op, on the way to a bigram trained by the book's own autograd engine, and the next section, [Scalar reverse mode: the Value graph](02-scalar-reverse-mode.md), checks reverse mode against it.

This section builds dual numbers from first principles, works three examples by hand, and then walks through a complete implementation, [`dual.py`](../../../../course/solutions/python/tinyllm/autograd/dual.py), whose interface is fixed by the contract [`course/contracts/py/tinyllm/autograd/dual.pyi`](../../../../course/contracts/py/tinyllm/autograd/dual.pyi). It leans on [Taylor series, remainder bounds, and range reduction](../02-calculus-2/01-taylor-series.md) for the reason the method works, and on the closed-form activation derivatives that the dual derivatives must match.

> **Learning objectives**
> - Derive the dual-number rules for sum, product, quotient, and the elementwise functions from $\varepsilon^2 = 0$ and Taylor's theorem.
> - Compute a derivative and a Jacobian-vector product by hand by propagating tangents.
> - Implement a `Dual` class that interoperates with numpy on either side of an operator, including `@` and indexing.
> - Explain why dual derivatives are exact to rounding, and why forward mode costs one pass per input direction.

## The algebra of dual numbers

Start with a formal symbol $\varepsilon$ that is not zero but whose square is: $\varepsilon \ne 0$, $\varepsilon^2 = 0$. A dual number is $a + b\varepsilon$, with a value $a$ and a tangent $b$; in code it is `Dual(val, eps)`. Throughout this section, $f$ and $f'$ are a differentiable function and its derivative, $x \in \mathbb{R}^n$ is an input point and $v \in \mathbb{R}^n$ a tangent direction (both `float64[n]`), $J_f(x) \in \mathbb{R}^{m \times n}$ is the Jacobian of $f: \mathbb{R}^n \to \mathbb{R}^m$ with $J_{ij} = \partial f_i / \partial x_j$, $J_f(x)\, v \in \mathbb{R}^m$ is the Jacobian-vector product (JVP), $W$ is a constant $m \times n$ matrix, and $u \approx 1.1 \times 10^{-16}$ is the unit roundoff of float64.

Dual numbers add and multiply like polynomials in $\varepsilon$, after which you drop every $\varepsilon^2$:

$$(a + b\varepsilon) + (c + d\varepsilon) = (a + c) + (b + d)\varepsilon,$$
$$(a + b\varepsilon)(c + d\varepsilon) = ac + (ad + bc)\varepsilon + bd\,\varepsilon^2 = ac + (ad + bc)\varepsilon.$$

Look at the tangent of the product: it is the product rule. Division follows by multiplying top and bottom by the conjugate $c - d\varepsilon$, since $(c + d\varepsilon)(c - d\varepsilon) = c^2$:

$$\frac{a + b\varepsilon}{c + d\varepsilon} = \frac{(a + b\varepsilon)(c - d\varepsilon)}{c^2} = \frac{a}{c} + \frac{bc - ad}{c^2}\varepsilon,$$

and the tangent is the quotient rule. A plain number $c$ is the dual number $c + 0\varepsilon$: a constant has tangent 0.

## Why the tangent is the derivative

That the arithmetic reproduces the product and quotient rules is not a coincidence. Taylor's theorem expands a smooth $f$ around $a$:

$$f(a + h) = f(a) + f'(a)\,h + \tfrac12 f''(a)\,h^2 + \dots$$

Put $h = b\varepsilon$. Every term from $h^2$ on contains $\varepsilon^2 = 0$, so

$$f(a + b\varepsilon) = f(a) + f'(a)\, b\, \varepsilon \quad \text{exactly.}$$

There is no step size and no truncation: the tangent is the derivative times the input tangent, and floating point adds only the rounding of each operation.

This tells you how to extend the algebra to any primitive: read its dual rule straight off its derivative. The table collects the rules this section implements.

| Function | Dual rule |
|---|---|
| $e^x$ | $e^a + e^a\, b\,\varepsilon$ |
| $\log x$ | $\log a + (b / a)\,\varepsilon$ |
| $\tanh x$ | $\tanh a + (1 - \tanh^2 a)\, b\,\varepsilon$ |
| $\mathrm{erf}\, x$ | $\mathrm{erf}\, a + \tfrac{2}{\sqrt\pi} e^{-a^2}\, b\,\varepsilon$ |
| $x^k$, constant $k$ | $a^k + k a^{k-1}\, b\,\varepsilon$ |
| $c^x$, constant $c > 0$ | $c^a + c^a \ln c\; b\,\varepsilon$ |
| $x^y$, both dual | $a^c + a^c\left(d \ln a + c\,b/a\right)\varepsilon$ for $x = a + b\varepsilon$, $y = c + d\varepsilon$ |

You need one rule per primitive and nothing more, because the chain rule comes for free. If $f(a + \varepsilon) = f(a) + f'(a)\varepsilon$, then feeding that into $g$ gives $g(f(a)) + g'(f(a))\, f'(a)\,\varepsilon$, which is the derivative of $g \circ f$. Every program built from the primitives is differentiated simply by running it.

## From derivatives to Jacobian-vector products

So far each number carried a scalar tangent. Give every input its own tangent instead: $x + v\varepsilon$ with $v \in \mathbb{R}^n$. The multivariable Taylor expansion has the same shape, $f(x + v\varepsilon) = f(x) + J_f(x)\, v\,\varepsilon$, so the output tangent is the JVP. A linear map passes tangents through itself: $W(x + v\varepsilon) = Wx + (Wv)\varepsilon$.

To build the whole Jacobian, run once per basis vector $e_j$, getting column $j$ each time. Forward mode therefore costs one pass per input. It is ideal for few inputs and many outputs. A loss with millions of parameters and one output is the opposite case, and it is the reason the next section builds reverse mode.

In code, a vector of dual numbers is stored as two arrays of one shape, `val` and `eps`, and every rule is applied elementwise. Indexing and slicing take the same entries from both arrays; `@` maps the tangent by the same matrix.

## Exact versus approximate

It is worth being precise about how much better this is than finite differences. A central difference $(f(x + h) - f(x - h)) / 2h$ has truncation error about $h^2 |f'''|/6$ and rounding error about $u |f| / h$; as [Derivatives and Finite Differences](../01-calculus-1/01-derivatives-and-finite-differences.md) shows, these balance near $h = 10^{-5}$ at an error near $10^{-10}$. A dual derivative is as accurate as evaluating $f$ itself, about $u$ relative. That is why dual derivatives can be compared with the closed-form activation derivatives at a relative tolerance of $10^{-12}$, but with the numeric JVP of *Calculus 3* only at $10^{-6}$.

## Working it by hand

With the rules in place, trace a few evaluations by hand. First, $f(x) = x e^x + 3$ at $x = 1$. Seed the input with tangent 1, so $x = 1 + \varepsilon$, and propagate:

| step | value | tangent | rule |
|---|---|---|---|
| $x$ | 1 | 1 | seed |
| $e^x$ | $e = 2.718282$ | $e \cdot 1 = 2.718282$ | exp |
| $x \cdot e^x$ | $1 \cdot e = 2.718282$ | $1 \cdot e + 1 \cdot e = 5.436564$ | product |
| $+ 3$ | $5.718282$ | $5.436564$ | constant |

So $f(1) = e + 3 = 5.718282$ and $f'(1) = 2e = 5.436564$. By hand, $f'(x) = e^x + x e^x$, which is $2e$ at 1: the same number, from one evaluation. This example becomes the first test of the implementation.

Next, a quotient: $(x + 1)/(x - 1)$ at $x = 3$. The numerator is $4 + \varepsilon$ and the denominator $2 + \varepsilon$; by the quotient rule the tangent is $(1 \cdot 2 - 4 \cdot 1)/2^2 = -0.5$. The function is 2 there, and its derivative $-2/(x-1)^2 = -0.5$.

Finally, a JVP. Take $f(x_1, x_2) = (x_1 x_2,\; x_1 + x_2^2)$ at $(2, 3)$ along $v = (1, 0)$, so $x_1 = 2 + \varepsilon$ and $x_2 = 3 + 0\varepsilon$. Then $x_1 x_2 = 6 + 3\varepsilon$ and $x_1 + x_2^2 = 11 + \varepsilon$, so $J v = (3, 1)$: the first column of $J = \begin{bmatrix} x_2 & x_1 \\ 1 & 2x_2 \end{bmatrix} = \begin{bmatrix} 3 & 2 \\ 1 & 6 \end{bmatrix}$.

## Designing the interface

The implementation exposes one class and a handful of functions:

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

`derivative` seeds `Dual(x, 1.0)` and returns the output's tangent (0.0 when `f` ignores its input). `jvp` seeds `Dual(x, v)` and returns `(f(x), J v)` as float64 arrays. numpy has no `erf`, so the solution uses `math.erf`, applied elementwise for arrays through `np.vectorize`.

One design point needs care before any code: numpy on the left of an operator. `np.float64(2.0) * d` asks numpy first. numpy's scalar and array types try to treat `d` as an element of an object array, and the tangent is lost (or an object array comes back). Setting the class attribute `__array_ufunc__ = None` tells numpy to return `NotImplemented` for any operation with a `Dual`, and Python then calls `Dual.__rmul__`. Every reflected method (`__radd__`, `__rsub__`, `__rmul__`, `__rtruediv__`, `__rpow__`, `__rmatmul__`) must keep the operand order: $c - x$ has tangent $-b$, and $c / x$ has tangent $-cb/a^2$.

## Building it step by step

The reasoning that leads to the code runs in six steps.

1. A dual number is a pair (value, tangent), and $f(a + b\varepsilon) = f(a) + f'(a)b\varepsilon$ exactly. So the representation is two fields, `val` and `eps`, and every operation must produce a new pair whose tangent is the derivative rule for that primitive applied to the input tangents.
2. Every binary operation can meet a plain number. A constant is $c + 0\varepsilon$, so the code lifts any non-`Dual` operand to `Dual(c, 0.0)` once (`_lift`) and then writes each rule only for two duals.
3. Commutative operations (`+`, `*`) can reuse the forward method for the reflected one. Non-commutative ones (`-`, `/`, `**`, `@`) cannot: $c - x$ and $x - c$ have opposite tangents, so each reflected method computes `other op self` explicitly.
4. A vector of duals is the same pair with arrays in both fields, and every rule above is already elementwise in numpy. The only array-specific operations are those that mix entries: indexing takes the same entries from both arrays, and `@` applies the same linear map to both, $W(x + v\varepsilon) = Wx + (Wv)\varepsilon$.
5. numpy must not get the first chance at `ndarray op Dual`, or it builds an object array. Setting `__array_ufunc__ = None` makes numpy return `NotImplemented`, so Python falls through to the reflected method.
6. The entry points only seed and read: `derivative` seeds tangent 1 and reads `eps`; `jvp` seeds tangent `v` and returns both fields. A function that ignores its input returns a plain number, whose derivative is 0.

The class itself has two fields, with scalars kept as Python floats and arrays copied to one broadcast shape:

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

The arithmetic rules come next. Each tangent line is one of the derivative rules from earlier in the section, with the operand order kept in the reflected forms:

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

`__add__`, `__radd__`, `__rmul__`, and `__neg__` are the one-line versions of the same pattern (addition and multiplication commute, so the reflected forms call the forward ones). Notice that three separate rules sit behind `**`, because $x^k$, $c^x$, and $x^y$ have different derivatives. Using the general $x^y$ rule for a constant exponent would call `log` on a negative base.

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

The entry points only seed and read:

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

To connect the code back to the hand computation: `derivative(lambda x: x * dual_exp(x) + 3, 1.0)` runs `dual_exp` to get $(e, e)$, `__mul__` to get $(e, 1 \cdot e + 1 \cdot e)$, and `__radd__` with $3 + 0\varepsilon$, returning $2e$. The full solution is [`dual.py`](../../../../course/solutions/python/tinyllm/autograd/dual.py).

## Why this design

Each operation does a constant number of extra multiplies, so a forward pass with tangents costs a small constant factor (about 2 to 3) times the plain evaluation, and memory is doubled. A JVP is one pass; a full $m \times n$ Jacobian is $n$ passes.

Compare the alternatives. Finite differences need a step size and give about 5 to 10 correct digits; dual numbers give the derivative to rounding, which is what an oracle must do. Symbolic differentiation suffers from expression swell, since symbolic derivatives of compositions grow in size, and it needs a separate evaluator; dual numbers differentiate the program as it runs, through loops and branches, with no expression tree. Operator overloading is the right delivery mechanism because the functions being checked (the activations of *Calculus 1*, the forward code of the op library) are written in numpy-style arithmetic. Overloading lets the same code run on duals unchanged, apart from calling `dual_exp` and friends where numpy's ufuncs would be.

The tradeoff you accept is the cost model: forward mode costs one pass per input, so it is the wrong tool for the gradient of a loss with millions of parameters. Here it is used as an oracle on small cases, where exactness matters and cost does not.

## What a correct implementation must handle

The tests that pin the implementation down fall into a few groups. The hand example fixes what a dual number carries: $f(1) = e + 3$ and $f'(1) = 2e$ for $x e^x + 3$. Unit tests check 13 primitives and their reflected forms against known derivatives, so each rule is right on its own, and the power rules separately, with $x^3$, $x^{0.5}$, $x^{-1}$, $2^x$, and $x^x$, because three different rules hide behind `**`. Indexing and matmul tests check that slices, `W @ x`, and `x @ W.T` move value and tangent together, since layers are matrices.

Differential tests compare against independent references. Sigmoid, tanh, SiLU, both GELUs, and softplus are differentiated at 1001 points in $[-40, 40]$ and compared with the closed forms of *Calculus 1*: this is the exact oracle the op library relies on. A JVP $J v$ of a map $\mathbb{R}^4 \to \mathbb{R}^3$ is compared with `jvp_numeric` from *Calculus 3*, confirming that vector forward mode is right.

Boundary tests cover the cases real code throws at you. A numpy operand on the left (`np.float64(2) * d`, `1 - d`, `array + d`) must stay `Dual`, because numpy scalars appear everywhere. A constant function must give derivative 0, `jvp` must give zeros, and no nested `Dual` may appear. Finally, two property tests: the chain rule, checked through $\frac{d}{dx}\tanh(x e^x)$ at 40 points, and linearity of the JVP in $v$, with basis vectors giving the Jacobian's columns, which is the cost model of forward mode.

Most mistakes in this code produce a plausible but wrong number rather than an error, so it helps to know the usual ones.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Keeping one term of the product rule | $f'(1) = e$ instead of $2e$ in the worked example | `__mul__` adds both $a\,d$ and $b\,c$ |
| Reflected operators that swap the order | $4 - x$ gets slope $+1$; $6/x$ gets the wrong sign | `__rsub__` computes `o - self`; `__rtruediv__` lifts `other` to the numerator |
| No `__array_ufunc__ = None` | `np.float64(2) * d` silently loses the tangent | the class attribute sends numpy operands to the reflected methods |
| `W @ x` that maps the value but not the tangent | JVPs of layers are wrong while scalar cases pass | `__rmatmul__` applies `w` to both `val` and `eps` |
| A sign in the quotient rule, $1 + \tanh^2$, $\mathrm{erf}' = e^{-x^2}/\sqrt\pi$ | wrong slopes for division, tanh, and GELU | $(bc - ad)/c^2$; $1 - t^2$; the constant `TWO_OVER_SQRT_PI` |
| $k x^k$ instead of $k x^{k-1}$; $c^x$ without $\ln c$ | power and exponential slopes off | separate rules for constant exponent, constant base, and both dual |
| Seeding the input tangent with 0 | every derivative is 0 | `derivative` seeds `Dual(x, 1.0)`; `jvp` seeds `Dual(x, v)` |
| Indexing the value but not the tangent | slices carry another entry's derivative | `__getitem__` indexes both fields with the same `idx` |

> **Warning:** The `W @ x` pitfall is the dangerous one. Every scalar test passes, and only JVPs through a layer come out wrong.

## Where this leads

This section stands on three earlier ones: the closed-form derivatives of [Activation Functions and Their Derivatives](../01-calculus-1/03-activation-functions.md) are what the dual derivatives must match; `jvp_numeric` from [Jacobians, the multivariable chain rule, and numeric JVP/VJP](../04-calculus-3/02-jacobians-and-the-chain-rule.md) approximates the same JVP by central differences; and [Taylor series, remainder bounds, and range reduction](../02-calculus-2/01-taylor-series.md) explains why $f(a + b\varepsilon) = f(a) + f'(a)b\varepsilon$ and gives `erf` as a series. Going forward, [Scalar reverse mode: the Value graph](02-scalar-reverse-mode.md) checks reverse mode against `Dual` on the same expressions, and [The op library and gradcheck_all](../../build-an-llm/p00-foundations/02-op-library-and-gradcheck.md) checks each elementwise op's backward against its dual derivative.

Production systems carry the same idea further. PyTorch's forward-mode AD provides dual tensors (`fwAD.make_dual`, `unpack_dual`) and `torch.func.jvp` over every op (`torch/autograd/forward_ad.py`). JAX's `jax.jvp` does forward mode by tracing, composable with `vmap` and with reverse mode, which gives forward-over-reverse Hessian-vector products as in [Hessian-vector products and the recompute-versus-memory schedule](04-hessian-vector-products-and-recompute.md) (`jax/_src/interpreters/ad.py`). Julia's ForwardDiff.jl uses chunked tangents, up to 12 directions per pass, so a gradient costs $n / 12$ passes (`src/dual.jl`).

Further reading:

- Baydin, Pearlmutter, Radul, and Siskind, [*Automatic Differentiation in Machine Learning: a Survey*](https://www.jmlr.org/papers/v18/17-468.html) (JMLR 2018), sections 2 and 3.1.
- Griewank and Walther, *Evaluating Derivatives* (SIAM, 2nd ed.), chapter 3.

## Key terms

- **Dual number**: a pair $a + b\varepsilon$ with $\varepsilon^2 = 0$, whose tangent $b$ propagates derivatives through arithmetic.
- **Tangent**: the $\varepsilon$ coefficient of a dual number; the derivative of the value along the seeded direction.
- **Seed**: the tangent given to the inputs before evaluation (1 for a derivative, $v$ for a JVP).
- **Jacobian-vector product (JVP)**: $J_f(x)\,v$, the directional derivative of a vector function, computed by one forward-mode pass.
- **Forward mode**: automatic differentiation that carries tangents alongside values, costing one pass per input direction.
- **Reflected operator**: a method such as `__rsub__` that Python calls when the left operand does not handle the operation.

## Summary

- A dual number $a + b\varepsilon$ with $\varepsilon^2 = 0$ turns Taylor's theorem into an identity: $f(a + b\varepsilon) = f(a) + f'(a)\, b\, \varepsilon$, so evaluating a function on $x + \varepsilon$ returns its value and its exact derivative together.
- One rule per primitive is enough; the chain rule happens by itself when operations compose.
- With a vector tangent, one forward pass computes a Jacobian-vector product $J v$; a full Jacobian costs one pass per input.
- Dual derivatives have no step size and match closed-form derivatives to rounding, which makes them the oracle for every elementwise op in the autograd library.
- Interoperating with numpy requires `__array_ufunc__ = None` and reflected methods that preserve operand order.

## Review questions

1. Why does `__rsub__` not just call `__sub__` the way `__radd__` calls `__add__`?

   <details><summary>Answer</summary>

   Subtraction does not commute. `1 - d` reaches `d.__rsub__(1)`, and `d.__sub__(1)` would compute $d - 1$, whose tangent is $+b$ instead of $-b$. Addition and multiplication commute, so their reflected forms can delegate.

   </details>

2. What goes wrong without `__array_ufunc__ = None`?

   <details><summary>Answer</summary>

   `np.float64(2) * d` is handled by numpy first. It treats `d` as an element of an object array, so the result is an object array instead of a `Dual`, or the tangent is lost. With the attribute set to `None`, numpy refuses, and Python calls `Dual.__rmul__`, which applies the product rule.

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

6. Using dual numbers, compute the derivative of $(x + 1)/(x - 1)$ at $x = 3$.

   <details><summary>Answer</summary>

   The numerator is $4 + \varepsilon$ and the denominator $2 + \varepsilon$. The quotient rule on duals gives tangent $(1 \cdot 2 - 4 \cdot 1)/2^2 = -0.5$, matching $-2/(x-1)^2 = -0.5$.

   </details>
