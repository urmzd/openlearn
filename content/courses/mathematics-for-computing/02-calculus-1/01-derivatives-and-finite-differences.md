# Derivatives and Finite Differences

Training a model by gradient descent needs the derivative of a loss with respect to every weight. The autograd engine built in [Tensor, broadcasting backward, and no_grad](../../build-an-llm/01-foundations/02-tensor-and-broadcasting-backward.md) and [The op library and gradcheck_all](../../build-an-llm/01-foundations/03-op-library-and-gradcheck.md) in *Build Your Own LLM System* computes those derivatives by the chain rule, and every backward rule there could be wrong in ways that still train a little: a transposed gradient, a missing factor of 2, a flipped sign. The only independent check is the definition of the derivative itself, evaluated numerically: nudge one input, watch the output move, divide. That check is `gradcheck`, built in [Gradients and gradcheck](../05-calculus-3/01-gradients-and-gradcheck.md), and it rests on the functions you build in this section.

A numerical derivative is only as trustworthy as its step size. Too large, and the formula is inaccurate; too small, and floating-point rounding swamps the answer. This section derives the right step from first principles, then implements the three difference formulas the rest of the book uses: the forward difference, the central difference, and Richardson extrapolation. The finished code lives in [`course/solutions/python/tinyllm/num/diff.py`](../../../../course/solutions/python/tinyllm/num/diff.py) (functions `forward_diff`, `central_diff`, and `richardson`), and its contract is [`course/contracts/py/tinyllm/num/diff.pyi`](../../../../course/contracts/py/tinyllm/num/diff.pyi).

There is nothing to call from earlier sections, but the arguments lean on [Exponents, logs, and units of information](../01-precalculus/01-exponents-logs-and-units-of-information.md) for $e^x$ and on [Polynomials, Horner, and stable roots](../01-precalculus/04-polynomials-horner-and-stable-roots.md) for the shape of a truncated polynomial expansion. As you read the implementation, try to predict each step before you see it.

> **Learning objectives**
> - Derive the truncation error of the forward and central differences from Taylor's formula and state their orders.
> - Explain why rounding error grows as the step shrinks, and compute the step that balances the two errors for each formula.
> - Scale a default step with the size of $x$ and divide by the step the hardware actually took.
> - Build a Richardson extrapolation table and explain why it uses weights $4^k$.
> - Implement `forward_diff`, `central_diff`, and `richardson` with the validation the gradient checker relies on.

## The derivative is a limit

Throughout this section, $f$ is a function of one real variable evaluated in float64 (a `Callable[[float], float]`), $x$ is the point where you want the derivative, and $h > 0$ is the step. The **slope** of the straight line through $(x, f(x))$ and $(x + h, f(x + h))$ is the difference quotient $\frac{f(x+h) - f(x)}{h}$: rise over run. As $h$ shrinks, the second point slides toward the first, and if the slopes settle on one number, that number is the **derivative** $f'(x)$, the slope of the tangent line:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} .$$

"Settles on" has a precise meaning, the limit: for every tolerance you name, there is a step size below which every quotient is within that tolerance of $f'(x)$. For $f(x) = x^2$ you get $\frac{(x+h)^2 - x^2}{h} = \frac{2xh + h^2}{h} = 2x + h$, which tends to $2x$. The usual rules follow from the definition the same way: $(x^n)' = n x^{n-1}$, $(e^x)' = e^x$ (the property that defines $e$), $(\sin x)' = \cos x$, and the chain rule $(f(g(x)))' = f'(g(x))\, g'(x)$, which [Jacobians and the chain rule](../05-calculus-3/02-jacobians-and-the-chain-rule.md) generalizes and every backward pass applies.

## A computer cannot take a limit

A program picks one $h$ and computes one quotient. Two questions follow: how wrong is the quotient for a given $h$, and which $h$ makes it least wrong? Both answers come from comparing $f$ near $x$ with a polynomial. If $f$ is smooth (has enough continuous derivatives), then for small $h$

$$f(x + h) = f(x) + f'(x)\, h + \tfrac{1}{2} f''(x)\, h^2 + \tfrac{1}{6} f'''(x)\, h^3 + \cdots$$

This is **Taylor's formula**; [Taylor series](../03-calculus-2/01-taylor-series.md) proves it and bounds the dots. Here only the first few terms matter, and you can check them on $x^3$: $(x+h)^3 = x^3 + 3x^2 h + 3x h^2 + h^3$, and indeed $f' = 3x^2$, $\frac{1}{2} f'' = 3x$, $\frac{1}{6} f''' = 1$.

With Taylor's formula in hand, you can measure exactly how far each difference quotient sits from the true derivative.

## Truncation error

Write $D_+(h)$ for the forward difference and $D_0(h)$ for the central difference, and use $O(h^p)$ to mean "at most a constant times $h^p$" as $h \to 0$. Subtract $f(x)$ from Taylor's formula and divide by $h$:

$$D_+(h) = \frac{f(x+h) - f(x)}{h} = f'(x) + \tfrac{1}{2} f''(x)\, h + O(h^2) .$$

The forward difference is off by about $\frac{1}{2} f''(x) h$: it is **first order**, with error proportional to $h$. Halve $h$ and the error halves. Now write the expansion at $-h$ as well,

$$f(x - h) = f(x) - f'(x)\, h + \tfrac{1}{2} f''(x)\, h^2 - \tfrac{1}{6} f'''(x)\, h^3 + \cdots,$$

and subtract it from the one at $+h$. The even powers cancel:

$$D_0(h) = \frac{f(x+h) - f(x-h)}{2h} = f'(x) + \tfrac{1}{6} f'''(x)\, h^2 + O(h^4) .$$

The central difference is **second order**: halve $h$ and the error drops by a factor of 4. On a log-log plot of error against $h$, the forward difference is a line of slope 1 and the central difference a line of slope 2; that slope is the signature of each formula's order. The error from the dropped terms is called **truncation error**, because it comes from truncating the Taylor series. The central difference is exact on quadratics ($f''' = 0$), while the forward difference is exact only on straight lines.

If truncation were the whole story, you would simply take $h$ as small as possible. Floating point says otherwise.

## Rounding error

Every float64 operation rounds its exact result to the nearest representable number, with relative error at most $\varepsilon / 2$, where $\varepsilon = 2^{-52} \approx 2.2 \times 10^{-16}$ is float64 machine epsilon, the gap between 1 and the next float. So a computed $f(x + h)$ is off by about $\varepsilon |f(x)|$, more if $f$ itself is a long computation. When $h$ is small, the numerator $f(x+h) - f(x-h)$ is a difference of two nearly equal numbers: their leading digits cancel and their rounding errors do not. That error, about $\varepsilon |f|$, is then divided by $2h$. The total error of the central difference is

$$E_0(h) \approx \underbrace{\tfrac{1}{6} |f'''|\, h^2}_{\text{truncation}} + \underbrace{\varepsilon\, |f| / h}_{\text{rounding}} .$$

The first term falls as $h$ shrinks, and the second rises. The following table shows the measured error of $D_0$ on $e^x$ at $x = 1$, where $e = 2.71828\ldots$:

| $h$ | $10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-4}$ | $10^{-5}$ | $10^{-6}$ | $10^{-8}$ | $10^{-10}$ | $10^{-12}$ |
|---|---|---|---|---|---|---|---|---|---|
| error of $D_0$ | $4.5 \times 10^{-3}$ | $4.5 \times 10^{-5}$ | $4.5 \times 10^{-7}$ | $4.5 \times 10^{-9}$ | $5.6 \times 10^{-11}$ | $9.1 \times 10^{-11}$ | $5.2 \times 10^{-9}$ | $9.0 \times 10^{-7}$ | $1.2 \times 10^{-4}$ |

Down to $10^{-4}$ the error falls by a factor of 100 per decade (slope 2); below about $10^{-5}$ rounding takes over and the error rises again. A step of $10^{-12}$ is worse than a step of $10^{-2}$.

## Choosing the step

Since one error falls and the other rises, there is a best step in between. Minimize $E_0(h) = a h^2 + b / h$ with $a = |f'''|/6$ and $b = \varepsilon |f|$: the derivative $2ah - b/h^2$ is zero at $h^\star = (b / 2a)^{1/3}$. When $f$ and its derivatives are of size 1, that is $h^\star \approx \sqrt[3]{\varepsilon} \approx 6.1 \times 10^{-6}$, and the error there is about $\varepsilon^{2/3} \approx 3.7 \times 10^{-11}$: two thirds of the 16 digits survive. The same argument for the forward difference, $E_+(h) \approx \frac{1}{2}|f''| h + \varepsilon |f| / h$, gives $h^\star \approx \sqrt{\varepsilon} \approx 1.5 \times 10^{-8}$ and an error of about $\sqrt{\varepsilon}$, half the digits.

The step must also follow the **scale of $x$**. Floats near $x$ are about $\varepsilon |x|$ apart, so at $x = 10^8$ a step of $6 \times 10^{-6}$ moves $x$ by only $6 \times 10^{-14}$ relative, and the quotient is mostly rounding. A step proportional to $|x|$ keeps the relative perturbation fixed, and using $\max(1, |x|)$ instead of $|x|$ keeps it from vanishing at $x = 0$. The contract's defaults are therefore

$$h_+ = \sqrt{\varepsilon}\, \max(1, |x|), \qquad h_0 = \sqrt[3]{\varepsilon}\, \max(1, |x|) .$$

One more floating-point detail matters. $x + h$ is itself rounded, so the step the hardware took is $(x + h) - x$, not $h$. Dividing by the step actually taken makes $f(x) = x$ come out as exactly 1. With $x = 0.1$ and $h = 10^{-5}$, dividing by $2h$ gives $0.9999999999996$ instead.

## Richardson extrapolation

The best single central difference keeps about 11 digits. You can do better by removing truncation error with algebra instead of a smaller step. The central difference has an error series with only even powers, $D_0(h) = f' + c_2 h^2 + c_4 h^4 + \cdots$, where the $c$'s do not depend on $h$. Then $D_0(h/2) = f' + c_2 h^2/4 + c_4 h^4/16 + \cdots$, and the combination

$$R_1(h) = \frac{4\, D_0(h/2) - D_0(h)}{4 - 1} = f' + O(h^4)$$

cancels the $h^2$ term exactly. Repeating with weight $4^k$ at level $k$ cancels $h^{2k}$. Writing $D[0][j] = D_0(h / 2^j)$ and

$$D[k][j] = \frac{4^k D[k-1][j+1] - D[k-1][j]}{4^k - 1},$$

the value $R_L(h)$ after $L$ levels has error $O(h^{2L+2})$ and is exact on polynomials of degree up to $2L + 2$. Because the truncation error shrinks so fast, Richardson can use a large $h$ (like 0.1), where rounding is negligible, and still reach about 12 digits. This is the method behind adaptive differentiation libraries.

## Working it by hand

Before writing code, check every formula on an example small enough to compute by hand. Take $f(x) = x^3$ at $x = 2$, so $f'(2) = 3 \cdot 4 = 12$, with $h = 0.1$. The function values are $f(2.1) = 9.261$, $f(2) = 8$, $f(1.9) = 6.859$, $f(2.05) = 8.615125$, and $f(1.95) = 7.414875$. The table collects the four estimates:

| Quantity | Computation | Value | Error |
|---|---|---|---|
| $D_+(0.1)$ | $(9.261 - 8) / 0.1$ | 12.61 | $0.61 = 3 \cdot 2 \cdot 0.1 + 0.1^2$ |
| $D_0(0.1)$ | $(9.261 - 6.859) / 0.2$ | 12.01 | $0.01 = h^2$ (here $\frac{1}{6} f''' = 1$) |
| $D_0(0.05)$ | $(8.615125 - 7.414875) / 0.1$ | 12.0025 | $0.0025 = (h/2)^2$ |
| $R_1(0.1)$ | $(4 \cdot 12.0025 - 12.01) / 3 = 36 / 3$ | 12 | 0 |

The forward error $0.61$ is the $\frac{1}{2} f'' h = 3xh = 0.6$ predicted by the truncation analysis, plus the next term $h^2 = 0.01$. The central error is exactly $h^2$ because $x^3$ has no terms beyond $h^3$, and quartering it at $h/2$ is what lets Richardson remove it completely.

## Designing the interface

The three functions share one shape:

```python
def forward_diff(f: Callable[[float], float], x: float, h: Optional[float] = None) -> float: ...
def central_diff(f: Callable[[float], float], x: float, h: Optional[float] = None) -> float: ...
def richardson(f: Callable[[float], float], x: float, h: float, levels: int = 2) -> float: ...
```

`h = None` means the default step derived above; a given `h` is used as is, because the gradient checker passes its own. Each function returns a Python `float`, divides by the step actually taken, and raises `ValueError` for a non-finite `x`, a step that is not finite and positive, or `levels < 0`. Richardson's default is two levels.

## Building it step by step

The design follows from the analysis in seven decisions.

1. **Pick the formula by its truncation order.** The forward difference errs by $\tfrac12 f'' h$, the central one by $\tfrac16 f''' h^2$. Both are provided because callers differ (an optimizer that already has $f(x)$ saves one evaluation with the forward one), but gradcheck uses the central one.
2. **Pick the default step by balancing errors.** Minimizing $a h^p + b/h$ gives $h^\star \propto \varepsilon^{1/(p+1)}$, so $\sqrt{\varepsilon}$ for the forward difference ($p = 1$) and $\sqrt[3]{\varepsilon}$ for the central one ($p = 2$). These are separate constants because each is wrong for the other formula.
3. **Scale the step with $x$, but never to zero.** The spacing of floats near $x$ is about $\varepsilon |x|$, so a fixed relative step means $h \propto |x|$. Multiplying by $\max(1, |x|)$ instead of $|x|$ keeps $h$ from vanishing at $x = 0$.
4. **Treat a caller's step as sacred.** gradcheck chooses `eps` deliberately, so an explicit `h` is validated (finite and positive) and used unchanged; only `None` triggers the default.
5. **Divide by the step the hardware took.** $x + h$ is rounded, so the true run of the secant is `xp - x` (or `xp - xm`), not $h$ (or $2h$). Dividing by it makes the quotient the exact slope of the secant through the two evaluated points, which is why $f(x) = x$ gives exactly 1.
6. **Build Richardson as a table, column by column.** Column 0 holds $D_0(h/2^j)$ for $j = 0, \ldots, L$. Each later column applies $\frac{4^k D[k-1][j+1] - D[k-1][j]}{4^k - 1}$, which shortens the column by one, so after $L$ rounds a single value remains. Only the current column needs to be kept.
7. **Validate at the boundary.** NaN or infinite $x$, a bad step, or a negative or fractional `levels` raises immediately, so a NaN never travels silently into a gradient check.

The code starts with the validators and the machine epsilon constant:

```python
EPS = 2.0**-52  # float64 machine epsilon: the gap between 1 and the next float


def _check_x(x: float) -> float:
    x = float(x)
    if not math.isfinite(x):          # nan and +-inf have no derivative to estimate
        raise ValueError(f"x must be finite, got {x}")
    return x


def _check_h(h: float) -> float:
    h = float(h)
    if not math.isfinite(h) or h <= 0:   # "not isfinite" first, so nan is rejected too
        raise ValueError(f"h must be finite and > 0, got {h}")
    return h
```

The two difference formulas differ only in their default step and their second point:

```python
def forward_diff(
    f: Callable[[float], float], x: float, h: Optional[float] = None
) -> float:
    x = _check_x(x)
    # Default: sqrt(eps), the balance point for an O(h) formula, relative to |x|.
    h = math.sqrt(EPS) * max(1.0, abs(x)) if h is None else _check_h(h)
    xp = x + h
    # Divide by the step the hardware actually took: x + h is rounded, so
    # (x + h) - x is usually not h. Using it makes f(x) = x exact.
    return (float(f(xp)) - float(f(x))) / (xp - x)


def central_diff(
    f: Callable[[float], float], x: float, h: Optional[float] = None
) -> float:
    x = _check_x(x)
    # Default: cbrt(eps), the balance point for an O(h^2) formula.
    h = math.cbrt(EPS) * max(1.0, abs(x)) if h is None else _check_h(h)
    xp, xm = x + h, x - h
    # xp - xm is the run actually taken, not 2h.
    return (float(f(xp)) - float(f(xm))) / (xp - xm)
```

`float(f(...))` turns a numpy scalar returned by `f` into a Python `float` before the subtraction, so the result is always a plain float. At $x = 10^8$ the default central step is $6.1 \times 10^{-6} \cdot 10^8 = 610$, which perturbs $x$ by the same relative amount as $6.1 \times 10^{-6}$ perturbs 1.

Richardson builds on `central_diff` with explicit steps:

```python
def richardson(
    f: Callable[[float], float], x: float, h: float, levels: int = 2
) -> float:
    x = _check_x(x)
    h = _check_h(h)                     # no default: Richardson wants a large h
    if int(levels) != levels or levels < 0:
        raise ValueError(f"levels must be an integer >= 0, got {levels}")
    levels = int(levels)
    # Column 0: central differences at h, h/2, h/4, ... Each later column
    # cancels the next even power of h: D(h) = f' + c2 h^2 + c4 h^4 + ...,
    # so (4^k D(h/2) - D(h)) / (4^k - 1) removes the h^(2k) term.
    col = [central_diff(f, x, h / 2.0**j) for j in range(levels + 1)]
    for k in range(1, levels + 1):
        p = 4.0**k
        # Combine neighbours: entry j uses steps h/2^j and h/2^(j+1).
        col = [(p * col[j + 1] - col[j]) / (p - 1.0) for j in range(len(col) - 1)]
    return col[0]                        # the single value of the last column
```

On the hand-worked example with `levels=1`, `col` starts as $[12.01, 12.0025]$; with $p = 4$ the next column is $[(4 \cdot 12.0025 - 12.01)/3] = [12]$, exact. `levels=0` returns the plain central difference $D_0(h)$. The full solution is in [`diff.py`](../../../../course/solutions/python/tinyllm/num/diff.py).

### Why this design

Consider cost first. `forward_diff` evaluates $f$ twice, `central_diff` twice, and `richardson` $2(L + 1)$ times, plus $O(L^2)$ arithmetic for the table. For gradcheck on $n$ parameters that is $2n$ evaluations of the loss, which is why gradcheck runs on small models only.

The central difference beats the forward one for checking gradients. At its best step it keeps about $\varepsilon^{2/3}$ (11 digits), against $\sqrt{\varepsilon}$ (8 digits) for the forward one, for the same two evaluations. gradcheck compares against an analytic gradient with a tolerance, so the extra digits decide whether real bugs stand out from noise.

Richardson beats a smaller step. A smaller $h$ eventually makes rounding dominate, as the error table for $e^x$ showed. Richardson reduces truncation error by algebra instead, so it can keep $h$ large where rounding is negligible and reach about 12 digits with three pairs of evaluations.

Two alternatives were not taken. Complex-step differentiation has no subtraction and is exact to rounding, but it needs $f$ to accept complex input, which most of the book's code does not. Automatic differentiation is exact, but it is precisely what these functions exist to check, so it cannot be its own referee.

The design also accepts two tradeoffs. The default step assumes $f$ and its derivatives have size near 1 relative to $\max(1,|x|)$; a badly scaled $f$ (huge $f'''$) would want a different step, and callers that know better pass `h`. And `richardson` divides each column by the step actually taken too, so its steps are only approximately in ratio 2; the difference is of rounding size and does not affect the cancellation.

## What can go wrong

A correct implementation must reproduce the hand-worked values 12.61, 12.01, and 12, so that everyone agrees on the definitions, and it must return a Python `float` even when `f` returns numpy scalars, because gradcheck stores results in float64 arrays. Its error behavior is tested as properties: the log-log slope of the central error must be 2 and that of the forward error 1, confirming the truncation orders. At the default steps, the central error must be at most $10^{-9}$ on five smooth functions at 26 points (the tolerance in [Gradients and gradcheck](../05-calculus-3/01-gradients-and-gradcheck.md) assumes this accuracy), and the forward error at most $10^{-7}$, the $\sqrt{\varepsilon}$ rule at work.

The boundaries matter just as much. The default step must scale with $x$: $\frac{d}{dx}\log x$ at $x = 10^8$ must come out to 7 digits, because weights and logits are not all near 1. The default step at $x = 0$ must be nonzero, because activations are checked at their kink. $f(x) = x$ must give exactly 1, which only dividing by the step actually taken achieves. A given step must be used as is: the central difference of $x^3$ with `h = 0.5` at $x = 100$ gives $3 \cdot 100^2 + 0.5^2 = 30000.25$, not the default-step answer, so gradcheck's `eps` means what it says. For Richardson, on $e^x$ at 0.5 with $h = 0.1$, the errors for levels 0 to 3 must fall as $2.7 \times 10^{-3}$, $3.4 \times 10^{-7}$, $5 \times 10^{-12}$, and $10^{-14}$; it must be exact on a quartic with one level and a sextic with two, confirming the $O(h^{2L+2})$ claim. Finally, $h = 0$, a negative $h$, NaN, or infinity must raise `ValueError`, as must a NaN or infinite $x$ or negative levels, so bugs surface at the call and NaN never travels into a gradient check.

The common mistakes each have a recognizable symptom, and the table maps them to the line of the solution that prevents them:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| a one-sided difference where the central one was meant | error $10^{-6}$ instead of $10^{-11}$; slope 1 on the log-log plot | `central_diff` evaluates at `x + h` and `x - h` |
| an absolute step, ignoring the size of $x$ | the derivative of $\log$ at $10^8$ is off by 3 percent | default step multiplied by `max(1.0, abs(x))` |
| a step proportional to $\lvert x \rvert$ alone | $h = 0$ at $x = 0$: division by zero | the `max(1.0, ...)` floor |
| dividing by $2h$ instead of the step actually taken | $f(x) = x$ differentiates to $0.9999999999996$ | divides by `xp - xm` (or `xp - x`) |
| the wrong step rule for the formula ($\sqrt{\varepsilon}$ for central, $\sqrt[3]{\varepsilon}$ for forward) | central error $6 \times 10^{-9}$ instead of $10^{-10}$; forward error $3 \times 10^{-6}$ instead of $10^{-8}$ | `math.sqrt(EPS)` in `forward_diff`, `math.cbrt(EPS)` in `central_diff` |
| Richardson weights $2^k$ instead of $4^k$ | the extrapolation removes nothing; worse than level 0 | `p = 4.0**k` |
| Richardson steps that grow ($h \cdot 2^j$) instead of shrink | the hand example gives 12.05 | column 0 uses `h / 2.0**j` |
| rescaling a step the caller gave | gradcheck's `eps` silently becomes $100 \cdot$ `eps` at $x = 100$ | the default applies only when `h is None` |
| accepting a NaN or infinite step | a NaN derivative instead of an error | `_check_h` rejects non-finite and non-positive steps |

> **Warning:** The most damaging of these is silent: rescaling a caller's step. Nothing crashes, but gradcheck's tolerance stops meaning what its author intended.

## Where this leads

These functions are used throughout the book. [Gradients and gradcheck](../05-calculus-3/01-gradients-and-gradcheck.md) builds `numerical_grad` by calling `central_diff(along, x_i, eps)` once per coordinate of every input: the frozen-step gradient behind `gradcheck`. The next section, [Newton's Method](02-newtons-method.md), uses `central_diff` as its slope when it is given no derivative. [Taylor series](../03-calculus-2/01-taylor-series.md) proves Taylor's formula with a remainder, making the truncation errors here rigorous, and [Error analysis, condition numbers, and tolerance budgets](../10-numerical-methods-and-floating-point/03-error-analysis-condition-numbers-and-tolerance-budgets.md) generalizes the rounding analysis. In *Build Your Own LLM System*, `F.gradcheck_all()` in [The op library and gradcheck_all](../../build-an-llm/01-foundations/03-op-library-and-gradcheck.md) checks every op of the autograd library through gradcheck, and so through this file. Exact derivatives with no step at all come from automatic differentiation: the autograd engine and the dual numbers of [Dual numbers and forward-mode autodiff](../09-matrix-calculus-and-autodiff/01-dual-numbers-forward-mode.md).

### Further reading

- OpenStax, *Calculus Volume 1* (free), sections 2.2 and 3.1 to 3.2: limits and the derivative.
- Sauer, *Numerical Analysis*, section 5.1: numerical differentiation and its rounding error.
- Nocedal and Wright, *Numerical Optimization*, section 8.1: finite-difference gradients.
- PyTorch `torch.autograd.gradcheck` (`torch/autograd/gradcheck.py`, `_compute_numerical_gradient`): the same central difference per element in float64 with eps $10^{-6}$, plus checks of complex inputs and forward-mode gradients.
- SciPy `scipy.optimize.approx_fprime` and `scipy.differentiate.derivative` (`scipy/optimize/_numdiff.py`): forward steps for optimizers, with adaptive step and order selection.
- numdifftools (`numdifftools/limits.py`): Richardson tables with automatic step search and error estimates.
- Martins, Sturdza, and Alonso, "The complex-step derivative approximation" (2003): $f'(x) \approx \operatorname{Im} f(x + ih)/h$ has no subtraction, so $h = 10^{-200}$ works and the result is exact to rounding.

## Key terms

- **derivative**: the limit of the difference quotient $\frac{f(x+h) - f(x)}{h}$ as $h \to 0$; the slope of the tangent line.
- **forward difference**: $D_+(h) = \frac{f(x+h) - f(x)}{h}$, a first-order approximation of $f'(x)$.
- **central difference**: $D_0(h) = \frac{f(x+h) - f(x-h)}{2h}$, a second-order approximation of $f'(x)$.
- **truncation error**: the error from dropping the higher terms of the Taylor series.
- **rounding error**: the error from floating-point rounding, about $\varepsilon |f| / h$ in a difference quotient.
- **machine epsilon**: $\varepsilon = 2^{-52}$, the gap between 1 and the next float64.
- **Richardson extrapolation**: combining differences at $h, h/2, h/4, \ldots$ with weights $4^k$ to cancel successive even powers of $h$.

## Summary

- The derivative is a limit; a computer evaluates the quotient at one $h > 0$, and the result is an approximation whose error you can predict.
- The forward difference has truncation error proportional to $h$; the central difference cancels the $h$ term and has error proportional to $h^2$.
- Rounding adds an error of about $\varepsilon |f| / h$ that grows as $h$ shrinks, so the best step balances the two: $h \approx \sqrt{\varepsilon}$ for forward, $h \approx \sqrt[3]{\varepsilon}$ for central, times $\max(1, |x|)$.
- Divide by the step actually taken, $(x+h) - (x-h)$, not by $2h$, because $x + h$ is rounded.
- Richardson extrapolation combines central differences at $h, h/2, h/4, \ldots$ to cancel $h^2, h^4, \ldots$ in turn, reaching about 12 digits from a large step.

## Review questions

1. Why is the default step $\sqrt[3]{\varepsilon}$ for the central difference but $\sqrt{\varepsilon}$ for the forward one?

   <details><summary>Answer</summary>

   Each step minimizes truncation plus rounding, $a h^p + \varepsilon |f| / h$. Setting the derivative to zero gives $h \propto \varepsilon^{1/(p+1)}$. The central difference has $p = 2$, so $h \approx \varepsilon^{1/3} \approx 6 \times 10^{-6}$; the forward one has $p = 1$, so $h \approx \varepsilon^{1/2} \approx 1.5 \times 10^{-8}$. Using the other formula's step leaves either too much truncation or too much rounding.

   </details>

2. What happens to `central_diff(math.log, 1e8)` if the step is $\sqrt[3]{\varepsilon}$ without the $\max(1, |x|)$ factor?

   <details><summary>Answer</summary>

   $x + 6 \times 10^{-6}$ near $10^8$ moves $x$ by less than one part in $10^{13}$, a handful of float spacings (about $1.5 \times 10^{-8}$ each). The numerator $\log(x+h) - \log(x-h)$ is then mostly rounding, and the derivative comes out several percent wrong. Scaling the step to $6 \times 10^{-6} \cdot 10^8$ restores the relative perturbation of the $x = 1$ case.

   </details>

3. Why divide by `xp - xm` rather than `2 * h`?

   <details><summary>Answer</summary>

   `xp` and `xm` are rounded, so the points where $f$ was evaluated are not exactly $2h$ apart. The quotient is meant to be the slope of the secant through the two points actually used; dividing by their actual distance makes it so, which is why $f(x) = x$ gives exactly 1. Dividing by $2h$ adds a relative error of up to about $\varepsilon |x| / h$; at $x = 0.1$, $h = 10^{-5}$ it gives $0.9999999999996$ instead of 1.

   </details>

4. Why does Richardson combine levels with $4^k$ and not $2^k$?

   <details><summary>Answer</summary>

   The central difference's error series has only even powers, $c_2 h^2 + c_4 h^4 + \cdots$. Halving the step divides the $h^{2k}$ term by $2^{2k} = 4^k$, so $4^k D(h/2) - D(h)$ cancels it exactly. With $2^k$ the leading term does not cancel and the extrapolation is worse than not extrapolating.

   </details>

5. Why does `richardson` require `h` while the other two functions default it?

   <details><summary>Answer</summary>

   The defaults are tuned to make a single difference quotient as accurate as possible, which puts $h$ where rounding and truncation are equal. Richardson's point is the opposite regime: a large $h$ (like 0.1) where rounding is negligible, letting extrapolation remove the truncation. Its later columns also amplify rounding by the weights $4^k/(4^k-1)$, so starting from a tiny default step would undo its advantage.

   </details>
