# Derivative as a limit, finite differences, step-size choice

## Overview

| | |
|---|---|
| **Module** | `M01.1` · build · Python · Pass 2 · 1 to 2 h reading |
| **Solution** | [`course/solutions/python/tinyllm/num/diff.py`](../../../../course/solutions/python/tinyllm/num/diff.py): `forward_diff`, `central_diff`, `richardson` |
| **Contract** | [`course/contracts/py/tinyllm/num/diff.pyi`](../../../../course/contracts/py/tinyllm/num/diff.pyi) |
| **Needs** | nothing to call. Reading: `M00.1` (exponents and logs), `M00.4` (polynomials) |
| **Used by** | `M04.1` builds `numerical_grad` (the heart of `gradcheck`) from one `central_diff` per coordinate · `M01.2` differentiates with `central_diff` when Newton is given no derivative · later `M02.1` and `M09.3` reuse the error analysis of section 2 |
| **Optional depth** | OpenStax, *Calculus Volume 1* (free), sections 2.2 and 3.1 to 3.2 (limits and the derivative); Sauer, *Numerical Analysis*, section 5.1 (numerical differentiation and its rounding error); Nocedal and Wright, *Numerical Optimization*, section 8.1 (finite-difference gradients) |

## Key Takeaways

- The **derivative** $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$ is a limit, and a computer evaluates the quotient at one $h > 0$: the result is an approximation whose error you can predict.
- The **forward difference** has truncation error proportional to $h$; the **central difference** $\frac{f(x+h) - f(x-h)}{2h}$ cancels the $h$ term and has error proportional to $h^2$.
- **Rounding** adds an error of about $\varepsilon |f| / h$ that grows as $h$ shrinks, so the best step balances the two: $h \approx \sqrt{\varepsilon}$ for forward, $h \approx \sqrt[3]{\varepsilon}$ for central, times $\max(1, |x|)$.
- **Divide by the step actually taken**, $(x+h) - (x-h)$, not by $2h$: $x + h$ is rounded.
- **Richardson extrapolation** combines central differences at $h, h/2, h/4, \ldots$ to cancel $h^2, h^4, \ldots$ in turn.

## How to read this chapter

Sections 1 and 2 derive the difference formulas, their truncation and rounding errors, and the optimal step from first principles. Section 3 works them by hand on $x^3$. Section 4 fixes the interface, and section 5 walks through the solution line by line and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

Pass 2 replaces the tracer's counted bigram with a model trained by gradient descent, and gradient descent needs derivatives of a loss with respect to every weight. In `L0.1` and `L0.2` the solution builds an autograd engine that computes those derivatives by the chain rule, and every backward rule there could be wrong in ways that still train a little: a transposed gradient, a missing factor of 2, a sign. The only independent check is the definition of the derivative itself, evaluated numerically: nudge one input, watch the output move, divide. That check is `gradcheck` (`M04.1`), and it is built on the function this module provides. It is only as trustworthy as its step size: too large and the formula is inaccurate, too small and floating-point rounding swamps the answer. This module derives the right step from first principles and implements the three difference formulas the rest of the course uses.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f$ | a function of one real variable, evaluated in float64 | `Callable[[float], float]` |
| $x$ | the point where we want the derivative | `float` |
| $h$ | the step, $h > 0$ | `float` |
| $f'(x)$ | the derivative of $f$ at $x$ | `float` |
| $D_+(h)$ | forward difference $\frac{f(x+h) - f(x)}{h}$ | `float` |
| $D_0(h)$ | central difference $\frac{f(x+h) - f(x-h)}{2h}$ | `float` |
| $O(h^p)$ | "at most a constant times $h^p$" as $h \to 0$ | |
| $\varepsilon$ | float64 machine epsilon, $2^{-52} \approx 2.2 \times 10^{-16}$: the gap between 1 and the next float | |
| $R_k(h)$ | Richardson value after $k$ levels | `float` |

### 2.1 The derivative is a limit

The **slope** of the straight line through $(x, f(x))$ and $(x + h, f(x + h))$ is the difference quotient $\frac{f(x+h) - f(x)}{h}$: rise over run. As $h$ shrinks the second point slides toward the first, and if the slopes settle on one number, that number is the **derivative** $f'(x)$, the slope of the tangent line:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} .$$

"Settles on" has a precise meaning (the limit): for every tolerance you name, there is a step size below which every quotient is within that tolerance of $f'(x)$. For $f(x) = x^2$: $\frac{(x+h)^2 - x^2}{h} = \frac{2xh + h^2}{h} = 2x + h$, which tends to $2x$. The usual rules follow from the definition the same way: $(x^n)' = n x^{n-1}$, $(e^x)' = e^x$ (the property that defines $e$, `M00.1`), $(\sin x)' = \cos x$, and the chain rule $(f(g(x)))' = f'(g(x))\, g'(x)$, which `M04.2` generalizes and every backward pass applies.

### 2.2 A computer cannot take a limit

A program picks one $h$ and computes one quotient. Two questions follow: how wrong is the quotient for a given $h$, and which $h$ makes it least wrong? Both answers come from comparing $f$ near $x$ with a polynomial. If $f$ is smooth (has enough continuous derivatives), then for small $h$

$$f(x + h) = f(x) + f'(x)\, h + \tfrac{1}{2} f''(x)\, h^2 + \tfrac{1}{6} f'''(x)\, h^3 + \cdots$$

This is **Taylor's formula**; `M02.1` proves it and bounds the dots. Here only the first few terms matter, which you can check on $x^3$: $(x+h)^3 = x^3 + 3x^2 h + 3x h^2 + h^3$, and indeed $f' = 3x^2$, $\frac{1}{2} f'' = 3x$, $\frac{1}{6} f''' = 1$.

### 2.3 Truncation error

Subtract $f(x)$ and divide by $h$:

$$D_+(h) = \frac{f(x+h) - f(x)}{h} = f'(x) + \tfrac{1}{2} f''(x)\, h + O(h^2) .$$

The forward difference is off by about $\frac{1}{2} f''(x) h$: **first order**, error proportional to $h$. Halve $h$ and the error halves. Now write the expansion at $-h$ as well, $f(x - h) = f(x) - f'(x)\, h + \frac{1}{2} f''(x)\, h^2 - \frac{1}{6} f'''(x)\, h^3 + \cdots$, and subtract it from the one at $+h$. The even powers cancel:

$$D_0(h) = \frac{f(x+h) - f(x-h)}{2h} = f'(x) + \tfrac{1}{6} f'''(x)\, h^2 + O(h^4) .$$

The central difference is **second order**: halve $h$ and the error drops by 4. On a log-log plot of error against $h$, the forward difference is a line of slope 1 and the central difference a line of slope 2; that slope is the signature of each formula's order. The error formula dropped is called **truncation error**, because it comes from truncating the Taylor series. The central difference is exact on quadratics ($f''' = 0$), and the forward difference is exact only on straight lines.

### 2.4 Rounding error

Every float64 operation rounds its exact result to the nearest representable number, with relative error at most $\varepsilon / 2$. So a computed $f(x + h)$ is off by about $\varepsilon |f(x)|$ (more if $f$ itself is a long computation). The numerator $f(x+h) - f(x-h)$ is a difference of two nearly equal numbers when $h$ is small: their leading digits cancel and the rounding errors do not. That error, about $\varepsilon |f|$, is then divided by $2h$. Total error of the central difference:

$$E_0(h) \approx \underbrace{\tfrac{1}{6} |f'''|\, h^2}_{\text{truncation}} + \underbrace{\varepsilon\, |f| / h}_{\text{rounding}} .$$

The first term falls as $h$ shrinks, the second rises. Measured on $e^x$ at $x = 1$, where $e = 2.71828\ldots$:

| $h$ | $10^{-1}$ | $10^{-2}$ | $10^{-3}$ | $10^{-4}$ | $10^{-5}$ | $10^{-6}$ | $10^{-8}$ | $10^{-10}$ | $10^{-12}$ |
|---|---|---|---|---|---|---|---|---|---|
| error of $D_0$ | $4.5 \times 10^{-3}$ | $4.5 \times 10^{-5}$ | $4.5 \times 10^{-7}$ | $4.5 \times 10^{-9}$ | $5.6 \times 10^{-11}$ | $9.1 \times 10^{-11}$ | $5.2 \times 10^{-9}$ | $9.0 \times 10^{-7}$ | $1.2 \times 10^{-4}$ |

Down to $10^{-4}$ the error falls by 100 per decade (slope 2); below about $10^{-5}$ rounding takes over and it rises again. A step of $10^{-12}$ is worse than a step of $10^{-2}$.

### 2.5 Choosing h

Minimize $E_0(h) = a h^2 + b / h$ with $a = |f'''|/6$ and $b = \varepsilon |f|$: the derivative $2ah - b/h^2$ is zero at $h^\star = (b / 2a)^{1/3}$. When $f$ and its derivatives are of size 1, that is $h^\star \approx \sqrt[3]{\varepsilon} \approx 6.1 \times 10^{-6}$, and the error there is about $\varepsilon^{2/3} \approx 3.7 \times 10^{-11}$: two thirds of the 16 digits survive. The same argument for the forward difference, $E_+(h) \approx \frac{1}{2}|f''| h + \varepsilon |f| / h$, gives $h^\star \approx \sqrt{\varepsilon} \approx 1.5 \times 10^{-8}$ and an error of about $\sqrt{\varepsilon}$, half the digits. The step must also follow the **scale of $x$**. Floats near $x$ are about $\varepsilon |x|$ apart, so at $x = 10^8$ a step of $6 \times 10^{-6}$ moves $x$ by only $6 \times 10^{-14}$ relative and the quotient is mostly rounding. A step proportional to $|x|$ keeps the relative perturbation fixed; $\max(1, |x|)$ keeps it from vanishing at $x = 0$. The contract's defaults are therefore

$$h_+ = \sqrt{\varepsilon}\, \max(1, |x|), \qquad h_0 = \sqrt[3]{\varepsilon}\, \max(1, |x|) .$$

One more floating-point detail: $x + h$ is itself rounded, so the step the hardware took is $(x + h) - x$, not $h$. Dividing by the step actually taken makes $f(x) = x$ come out as exactly 1. With $x = 0.1$ and $h = 10^{-5}$, dividing by $2h$ gives $0.9999999999996$ instead.

### 2.6 Richardson extrapolation

The central difference has an error series with only even powers: $D_0(h) = f' + c_2 h^2 + c_4 h^4 + \cdots$, where the $c$'s do not depend on $h$. Then $D_0(h/2) = f' + c_2 h^2/4 + c_4 h^4/16 + \cdots$, and the combination

$$R_1(h) = \frac{4\, D_0(h/2) - D_0(h)}{4 - 1} = f' + O(h^4)$$

cancels the $h^2$ term exactly. Repeating with weight $4^k$ at level $k$ cancels $h^{2k}$: with $D[0][j] = D_0(h / 2^j)$ and $D[k][j] = \frac{4^k D[k-1][j+1] - D[k-1][j]}{4^k - 1}$, level $L$ has error $O(h^{2L+2})$ and is exact on polynomials of degree up to $2L + 2$. Because the truncation error shrinks so fast, Richardson can use a large $h$ (like 0.1), where rounding is negligible, and still reach about 12 digits. This is the method behind adaptive differentiation libraries.

## 3. Worked example by hand

Take $f(x) = x^3$ at $x = 2$, so $f'(2) = 3 \cdot 4 = 12$, with $h = 0.1$. The function values: $f(2.1) = 9.261$, $f(2) = 8$, $f(1.9) = 6.859$, $f(2.05) = 8.615125$, $f(1.95) = 7.414875$.

| Quantity | Computation | Value | Error |
|---|---|---|---|
| $D_+(0.1)$ | $(9.261 - 8) / 0.1$ | 12.61 | $0.61 = 3 \cdot 2 \cdot 0.1 + 0.1^2$ |
| $D_0(0.1)$ | $(9.261 - 6.859) / 0.2$ | 12.01 | $0.01 = h^2$ (here $\frac{1}{6} f''' = 1$) |
| $D_0(0.05)$ | $(8.615125 - 7.414875) / 0.1$ | 12.0025 | $0.0025 = (h/2)^2$ |
| $R_1(0.1)$ | $(4 \cdot 12.0025 - 12.01) / 3 = 36 / 3$ | 12 | 0 |

The forward error $0.61$ is the $\frac{1}{2} f'' h = 3xh = 0.6$ of section 2.3 plus the next term $h^2 = 0.01$. The central error is exactly $h^2$ because $x^3$ has no terms beyond $h^3$, and quartering it at $h/2$ is what lets Richardson remove it completely.

## 4. The interface

```python
def forward_diff(f: Callable[[float], float], x: float, h: Optional[float] = None) -> float: ...
def central_diff(f: Callable[[float], float], x: float, h: Optional[float] = None) -> float: ...
def richardson(f: Callable[[float], float], x: float, h: float, levels: int = 2) -> float: ...
```

`h = None` means the default step of section 2.5; a given `h` is used as is (gradcheck passes its own). Each function returns a Python `float`, divides by the step actually taken, and raises `ValueError` for a non-finite `x`, a step that is not finite and positive, or `levels < 0`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit, smoke | section 3: 12.61, 12.01, 12 | you and the tests agree on the definitions |
| `test_returns_python_float` | unit | a Python `float` even when `f` returns numpy scalars | gradcheck stores results in float64 arrays |
| `test_central_error_slope_is_two` | property | log-log slope of the central error is 2 | the truncation order of section 2.3 |
| `test_forward_error_slope_is_one` | property | log-log slope of the forward error is 1 | the step rule differs per formula |
| `test_default_step_is_accurate` | golden | default central error at most $10^{-9}$ on five smooth functions at 26 points | `M04.1`'s tolerance assumes this accuracy |
| `test_forward_default_step_is_accurate` | golden | default forward error at most $10^{-7}$ | the $\sqrt{\varepsilon}$ rule |
| `test_default_step_scales_with_x` | boundary | $\frac{d}{dx}\log x$ at $x = 10^8$ to 7 digits | weights and logits are not all near 1 |
| `test_default_step_at_zero` | boundary | a nonzero step at $x = 0$ | activations are checked at their kink |
| `test_divides_by_the_step_actually_taken` | boundary | $f(x) = x$ gives exactly 1 | exactness on linear maps |
| `test_given_step_is_used_as_is` | unit | `h = 0.5` at $x = 100$ gives 30000.25 | gradcheck's `eps` means what it says |
| `test_richardson_levels_raise_the_order` | property | on $e^x$ at 0.5 with $h = 0.1$: errors $2.7 \times 10^{-3}$, $3.4 \times 10^{-7}$, $5 \times 10^{-12}$, $10^{-14}$ for levels 0 to 3 | extrapolation works level by level |
| `test_richardson_exact_on_polynomials` | property | exact on a quartic (1 level) and a sextic (2 levels) | the $O(h^{2L+2})$ claim |
| `test_richardson_default_levels_is_two` | unit | the default is two levels | contract defaults |
| `test_rejects_bad_steps` | boundary | $h = 0$, negative, nan, inf raise `ValueError` | bugs surface at the call |
| `test_rejects_bad_points_and_levels` | boundary | nan or inf $x$, negative levels raise `ValueError` | nan never travels into a gradient check |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. a one-sided difference where the central one was meant | error $10^{-6}$ instead of $10^{-11}$; slope 1 on the log-log plot | `test_central_error_slope_is_two`, `test_hand_example` (mutant `s01`) |
| 2. an absolute step, ignoring the size of $x$ | the derivative of $\log$ at $10^8$ is off by 3 percent | `test_default_step_scales_with_x` (mutant `s02`) |
| 3. a step proportional to $\lvert x \rvert$ alone | $h = 0$ at $x = 0$: division by zero | `test_default_step_at_zero` (mutant `s03`) |
| 4. dividing by $2h$ instead of the step actually taken | $f(x) = x$ differentiates to $0.9999999999996$ | `test_divides_by_the_step_actually_taken` (mutant `s05`) |
| 5. the wrong step rule for the formula ($\sqrt{\varepsilon}$ for central, $\sqrt[3]{\varepsilon}$ for forward) | central error $6 \times 10^{-9}$ instead of $10^{-10}$; forward error $3 \times 10^{-6}$ instead of $10^{-8}$ | `test_default_step_is_accurate` (mutant `s04`), `test_forward_default_step_is_accurate` (mutant `s08`) |
| 6. Richardson weights $2^k$ instead of $4^k$ | the extrapolation removes nothing; worse than level 0 | `test_richardson_exact_on_polynomials` (mutant `s06`) |
| 7. Richardson steps that grow ($h \cdot 2^j$) instead of shrink | the hand example gives 12.05 | `test_hand_example` (mutant `s07`) |
| rescaling a step the caller gave | gradcheck's `eps` silently becomes $100 \cdot$ `eps` at $x = 100$ | `test_given_step_is_used_as_is` (mutant `s09`) |
| accepting a nan or inf step | a nan derivative instead of an error | `test_rejects_bad_steps` (mutant `s10`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.1` | exponents and logarithms, $e^x$ and its defining property (reading) |
| Back | `M00.4` | polynomials, the shape of a truncated Taylor series (reading) |
| Forward | `M04.1` | `numerical_grad` calls `central_diff(along, x_i, eps)` once per coordinate of every input: the frozen-step gradient behind `gradcheck` |
| Forward | `M01.2` | `newton(f, None, x0)` uses `central_diff` as its slope |
| Forward | `M02.1` | Taylor's formula with a proven remainder: the error terms of section 2.3 made rigorous |
| Forward | `M09.3` | condition numbers and tolerance budgets generalize the rounding analysis of section 2.4 |
| Forward | `L0.2` | `F.gradcheck_all()` checks every op of the autograd library through `M04.1`, and so through this file |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `central_diff` with $h = 10^{-6}$ | PyTorch `torch.autograd.gradcheck` | the same central difference per element, float64, eps $10^{-6}$; also checks complex inputs and forward-mode gradients | `torch/autograd/gradcheck.py` (`_compute_numerical_gradient`) |
| `forward_diff` with $\sqrt{\varepsilon}$ | SciPy `scipy.optimize.approx_fprime`, `scipy.differentiate.derivative` | forward steps for optimizers; adaptive step and order selection | `scipy/optimize/_numdiff.py` |
| `richardson` | numdifftools | Richardson tables with automatic step search and error estimates | `numdifftools/limits.py` |
| (not built) | complex-step differentiation | $f'(x) \approx \operatorname{Im} f(x + ih)/h$ has no subtraction, so $h = 10^{-200}$ works and the result is exact to rounding | Martins, Sturdza, and Alonso, "The complex-step derivative approximation" (2003) |
| finite differences | automatic differentiation | exact derivatives by the chain rule, no step at all: what the `L0.1` autograd and `M08.1` dual numbers do | `M08.1`, `L0.1` |
