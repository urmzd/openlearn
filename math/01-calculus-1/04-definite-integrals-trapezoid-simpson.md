<!-- ol:module M01.4 -->
# Definite integrals, trapezoid, Simpson

## Overview

| | |
|---|---|
| **Module** | `M01.4` · build · Python · Pass 5 · 2 to 3 h |
| **You build** | `python/tinyllm/num/integrate.py`: `trapezoid`, `trapezoid_rule`, `simpson` |
| **Contract** | [`course/contracts/py/tinyllm/num/integrate.pyi`](../../course/contracts/py/tinyllm/num/integrate.pyi) |
| **Tests** | `course/tests/M01.4/test_integrate.py` (what they check: section 4) |
| **Needs** | nothing to call. Reading: [`M01.1` finite differences](01-derivatives-and-finite-differences.md) (Richardson extrapolation, section 2.6 here) |
| **Used by** | `M07.7` computes ROC-AUC as `trapezoid(tpr, fpr)` · later `ethics.04` integrates the safety report's ROC and reliability curves the same way |
| **Milestone** | `MS-P5` (the Pass 5 gate) |
| **Optional depth** | OpenStax, *Calculus Volume 1* (free), sections 5.1 to 5.3 (the definite integral and the fundamental theorem) and 3.6 of *Volume 2* (numerical integration); Sauer, *Numerical Analysis*, sections 5.2 and 5.3 (Newton-Cotes rules and Romberg) |

## Key Takeaways

- The **definite integral** $\int_a^b f(x)\,dx$ is the signed area under $f$, the limit of sums of thin slices; a computer stops at $n$ slices of width $h$ (`test_hand_example`).
- The **trapezoid rule** joins neighbouring points by straight lines: error proportional to $h^2$, exact on lines (`test_trapezoid_error_order_two`, `test_trapezoid_rule_exact_on_lines`).
- **Simpson's rule** fits a parabola through each three points: error proportional to $h^4$ and, by a symmetry bonus, exact on cubics (`test_simpson_error_order_four`, `test_simpson_exact_on_cubics`).
- Simpson is **Richardson extrapolation of the trapezoid rule**: $(4T(h/2) - T(h))/3$, the same error-cancelling step as `M01.1` (`test_simpson_is_richardson_of_trapezoid`).
- Samples that are not on a grid, such as the corners of an ROC curve, are integrated **step by step with each step's own signed width** (`test_trapezoid_samples_match_scipy`).

## How to work this chapter

```bash
ol start M01.4              # stubs python/tinyllm/num/integrate.py into your repo
ol tests M01.4              # read the test catalog first: rung R0, you write no tests here
ol check M01.4              # exit code is the verdict
ol diff  M01.4              # after passing: your code against the reference
```

---

## 1. Why now

Pass 5 trains classifiers for the first time: BERT and ELECTRA heads in `L6.3` and `L6.5`, and the usage-policy head the gateway will run (D33). A classifier outputs a score, and the question "how good is this score at separating the two classes?" has a standard answer that does not depend on any one threshold: the area under the ROC curve, which `M07.7` computes. That curve is not a formula; it is a list of corner points, one per threshold, and its area is a definite integral over samples. The same integral computes the expected calibration area in the safety report (`ethics.04`). Get the integration rule subtly wrong (count each step at its left height, or sort points that were deliberately ordered) and every AUC in the model zoo is biased, with nothing visibly broken. This module builds the area rule for samples and, for functions you can evaluate anywhere, the two classic rules with predictable errors.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $f$ | a function of one real variable, vectorized over float64 arrays | `Callable[[NDArray], NDArray]` |
| $[a, b]$ | the interval of integration; $b < a$ is allowed | `float`, `float` |
| $\int_a^b f(x)\,dx$ | the definite integral: the signed area between $f$ and the $x$ axis | `float` |
| $F$ | an antiderivative of $f$: $F' = f$ | function |
| $n$ | the number of equal steps | `int` |
| $h$ | the step, $(b - a)/n$ | `float` |
| $x_i$ | the nodes $a + ih$, $i = 0, \ldots, n$ (`numpy.linspace(a, b, n + 1)`) | `float64[n + 1]` |
| $f_i$ | $f(x_i)$ | `float64[n + 1]` |
| $T(h)$, $S(h)$ | the composite trapezoid and Simpson values with step $h$ | `float` |
| $O(h^p)$ | at most a constant times $h^p$ as $h \to 0$ | |

### 2.1 The definite integral

Cut $[a, b]$ into $n$ slices of width $h$, stand a rectangle on each slice with the height of $f$ somewhere in it, and add the areas: $\sum_i f(\xi_i)\, h$. This is a **Riemann sum**. As $h \to 0$ the sums for a continuous $f$ settle on one number whatever points $\xi_i$ you chose, and that number is the definite integral $\int_a^b f(x)\,dx$. Area below the axis counts as negative, which is why it is a *signed* area. Running from $b$ to $a$ makes every width $-h$, so $\int_b^a f = -\int_a^b f$, and an interval of length 0 has integral 0.

The **fundamental theorem of calculus** links this area to derivatives (`M01.1`): if $F' = f$, then $\int_a^b f(x)\,dx = F(b) - F(a)$. For $f(x) = x^3$, $F(x) = x^4/4$, so $\int_0^2 x^3\,dx = 16/4 - 0 = 4$. Most integrals you meet in practice have no $F$ in closed form ($e^{-x^2/2}$, the normal density, is the famous one) or $f$ is only known at sample points. Then you compute the area numerically, which is called **quadrature**.

### 2.2 The trapezoid

Replace $f$ on one step $[x_i, x_{i+1}]$ by the straight line through its two end values. The region under that line is a trapezoid with parallel sides $f_i$ and $f_{i+1}$ and width $h$, so its area is $h\,(f_i + f_{i+1})/2$: the width times the average height. A straight line is integrated exactly, because the "approximation" is the function itself.

### 2.3 The composite rules

Add one trapezoid per step. Every inner node belongs to two trapezoids and is counted twice at half weight; the two end nodes belong to one each:

$$T(h) = h\left(\tfrac{1}{2} f_0 + f_1 + f_2 + \cdots + f_{n-1} + \tfrac{1}{2} f_n\right).$$

Simpson's rule instead takes the steps in pairs and fits the parabola through the three points $(x_{2j}, x_{2j+1}, x_{2j+2})$. The area under that parabola, over a panel of width $2h$, is $\frac{h}{3}(f_{2j} + 4 f_{2j+1} + f_{2j+2})$ (integrate the parabola through $(-h, f_-), (0, f_0), (h, f_+)$ term by term to check it). Adding the panels, every even inner node is shared by two panels:

$$S(h) = \frac{h}{3}\left(f_0 + 4 f_1 + 2 f_2 + 4 f_3 + 2 f_4 + \cdots + 2 f_{n-2} + 4 f_{n-1} + f_n\right).$$

The pattern 1, 4, 2, 4, ..., 2, 4, 1 only closes when $n$ is **even**. An odd $n$ is not a smaller Simpson's rule; it is a different, wrong one, so the contract raises instead of rounding.

### 2.4 Integrating samples

An ROC curve is a list of points $(x_i, y_i)$ with uneven gaps, produced by sweeping a threshold. Its area is the trapezoid rule applied step by step with each step's own width: $\sum_i (x_{i+1} - x_i)(y_i + y_{i+1})/2$. The widths keep their **sign**, so the order of the points is the direction of travel: the same points in decreasing order give the negative area, exactly as running an integral from $b$ to $a$. The function never sorts, because sorting would silently reconnect the points in a different order (the curve the caller drew is the curve that gets measured). Fewer than two points enclose nothing: the area is 0.

### 2.5 Error and order

Expand $f$ around the middle of one step with Taylor's formula (`M01.1` section 2.2, proved in `M02.1`). The straight line misses the curvature term, and integrating the difference over one step gives an error of $-\frac{h^3}{12} f''(\xi)$ for some $\xi$ in the step. There are $n = (b - a)/h$ steps, so the composite error is

$$\int_a^b f - T(h) = -\frac{(b - a)\, h^2}{12}\, f''(\xi), \qquad \int_a^b f - S(h) = -\frac{(b - a)\, h^4}{180}\, f^{(4)}(\xi).$$

The trapezoid rule is **second order**: halve $h$ and the error drops by 4. Simpson is **fourth order**: halve $h$ and it drops by 16. A parabola should only be exact on quadratics, yet the error involves $f^{(4)}$, so Simpson is exact on **cubics** too: the $x^3$ error on the left half of a panel cancels the one on the right half by symmetry. On $\int_0^\pi \sin x\,dx = 2$:

| $n$ | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| trapezoid error | $4.3 \times 10^{-1}$ | $1.0 \times 10^{-1}$ | $2.6 \times 10^{-2}$ | $6.4 \times 10^{-3}$ | $1.6 \times 10^{-3}$ |
| Simpson error | $9.4 \times 10^{-2}$ | $4.6 \times 10^{-3}$ | $2.7 \times 10^{-4}$ | $1.7 \times 10^{-5}$ | $1.0 \times 10^{-6}$ |

Each column divides the trapezoid error by 4 and the Simpson error by 16: the slopes 2 and 4 the order tests measure. The error formulas assume a smooth $f$ at the scale of $h$. On Runge's peak $1/(1 + 25x^2)$ over $[-1, 1]$ with $n = 8$, Simpson's error is 0.026 and the trapezoid's 0.0075: the parabolas overshoot a peak that is narrower than two steps. Higher order pays off only once $h$ resolves the function.

### 2.6 Simpson is Richardson

The trapezoid error is not just $O(h^2)$; it is a series in even powers, $T(h) = I + c_2 h^2 + c_4 h^4 + \cdots$ (the Euler-Maclaurin formula). That is the same shape as the central difference of `M01.1`, so the same step removes the leading term:

$$\frac{4\,T(h) - T(2h)}{3} = I + O(h^4).$$

Write it out node by node and the weights come out as 1, 4, 2, 4, ..., 1 over 3: it *is* Simpson's rule on the finer grid. Repeating the step (weights $16, 64, \ldots$) is Romberg integration. The test `test_simpson_is_richardson_of_trapezoid` checks this identity to rounding.

## 3. Worked example by hand

Take $\int_0^2 x^3\,dx = 4$.

| Rule | Nodes and values | Computation | Value | Error |
|---|---|---|---|---|
| $T$, $n = 2$, $h = 1$ | $f(0, 1, 2) = 0, 1, 8$ | $1 \cdot (0/2 + 1 + 8/2)$ | 5 | 1 |
| $T$, $n = 4$, $h = 0.5$ | $f(0, 0.5, 1, 1.5, 2) = 0, 0.125, 1, 3.375, 8$ | $0.5 \cdot (0 + 0.125 + 1 + 3.375 + 4)$ | 4.25 | 0.25 |
| $S$, $n = 2$, $h = 1$ | $0, 1, 8$ | $\frac{1}{3}(0 + 4 \cdot 1 + 8)$ | 4 | 0 |
| Richardson | $T(0.5)$, $T(1)$ | $(4 \cdot 4.25 - 5)/3 = 12/3$ | 4 | 0 |

Halving $h$ divided the trapezoid error by exactly 4 (here $f'' = 6x$ and the error formula is exact up to the mean value). Simpson is exact because $x^3$ is a cubic, and Richardson's combination of the two trapezoid values gives the same 4.

Now three samples, the corners of an ROC curve: $(0, 0)$, $(0.5, 0.75)$, $(1, 1)$. Two trapezoids: $0.5 \cdot (0 + 0.75)/2 = 0.1875$ and $0.5 \cdot (0.75 + 1)/2 = 0.4375$, total **0.625**. Counting each step at its left height only (a left Riemann sum) gives $0.5 \cdot 0 + 0.5 \cdot 0.75 = 0.375$. These numbers are the first two tests, `test_hand_example` and `test_hand_example_samples`.

## 4. The interface

```python
def trapezoid(y: ArrayLike, x: ArrayLike) -> float: ...
def trapezoid_rule(f: Callable[[NDArray], NDArray], a: float, b: float, n: int) -> float: ...
def simpson(f: Callable[[NDArray], NDArray], a: float, b: float, n: int) -> float: ...
```

`trapezoid` integrates samples in the order given (numpy's `trapezoid(y, x)`). The two rules evaluate `f` once on `numpy.linspace(a, b, n + 1)`, so both end nodes are exact and every implementation agrees with scipy to the last bit. All three return a Python `float` and raise `ValueError` for non-finite input, mismatched samples, a wrongly shaped or non-finite `f`, `n < 1`, or (for Simpson) an odd `n`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit, smoke | section 3: 5, 4.25, 4, 4 | you and the tests agree on the rules |
| `test_hand_example_samples` | unit, smoke | the ROC corners give 0.625 | `roc_auc` in `M07.7` is this call |
| `test_rules_match_scipy` | golden | `scipy.integrate.trapezoid` and `simpson` on the same nodes, six functions and a reversed interval | the rules are the standard ones |
| `test_trapezoid_samples_match_scipy` | golden | uneven, unsorted, and decreasing samples | ROC and reliability curves |
| `test_trapezoid_error_order_two` | property | log-log error slope 2 | section 2.5 |
| `test_simpson_error_order_four` | property | log-log error slope 4 | section 2.5 |
| `test_simpson_exact_on_cubics` | property | 40 random cubics, $n = 2, 4, 10$ | the symmetry bonus |
| `test_trapezoid_rule_exact_on_lines` | property | exact on lines, not on $x^2$ | section 2.2 |
| `test_simpson_is_richardson_of_trapezoid` | property | $S = (4T(h) - T(2h))/3$ to rounding | section 2.6 |
| `test_reversed_and_empty_intervals` | boundary | $\int_b^a = -\int_a^b$, $\int_a^a = 0$, decreasing samples negative | signed areas |
| `test_f_called_once_on_linspace_nodes` | unit | one vectorized call on the contract's nodes | bit-identical results across languages |
| `test_returns_python_float` | unit | a `float`, not a numpy scalar | callers compare and serialize it |
| `test_trapezoid_short_inputs` | boundary, smoke | zero or one sample gives 0.0 | a one-threshold ROC curve |
| `test_rejects_bad_arguments` | boundary | odd or non-positive `n`, non-finite limits, bad `f`, bad samples raise | bugs surface at the call |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. a Riemann sum (left heights only) for samples | the hand ROC area is 0.375, not 0.625; AUCs biased low | `test_hand_example_samples` (mutant `s01`) |
| 2. Simpson weights 2 and 4 swapped | order 2 instead of 4; the hand example gives 10/3 | `test_hand_example`, `test_simpson_exact_on_cubics` (mutant `s02`) |
| 3. accepting an odd number of Simpson steps | a plausible but wrong number | `test_rejects_bad_arguments` (mutant `s03`) |
| 4. counting the end points fully in the trapezoid rule | the hand example gives 9, order 1 | `test_hand_example` (mutant `s04`) |
| 5. sorting samples or using $\lvert \Delta x \rvert$ | a decreasing curve has positive area; unsorted samples are reconnected | `test_trapezoid_samples_match_scipy` (mutant `s05`) |
| 6. $n$ nodes instead of $n + 1$ | the last step is lost; the rules disagree with scipy | `test_rules_match_scipy`, `test_f_called_once_on_linspace_nodes` (mutant `s06`) |
| 7. Simpson with the panel width $2h$ in place of the step $h$ | every result doubled | `test_hand_example` (mutant `s07`) |
| 8. assuming equal gaps between samples | wrong area on any real ROC curve | `test_trapezoid_samples_match_scipy` (mutant `s08`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M01.1` | Richardson extrapolation and the Taylor error terms of section 2.5 (reading) |
| Forward | `M07.7` | `roc_auc(scores, labels)` builds the ROC corners and returns `trapezoid(tpr, fpr)` |
| Forward | `ethics.04` | the safety report's ROC area and reliability curve |
| Forward | `S-M07d` | q8 computes an AUC by hand, the same area as pairs ranked correctly |

If you skip this module, `ol check M07.7` stops with `BLOCKED ... needs M01.4`: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `trapezoid` | `numpy.trapezoid`, `scipy.integrate.trapezoid` | integration along any axis of an N-d array | `numpy/lib/_function_base_impl.py` |
| `simpson` | `scipy.integrate.simpson` | uneven sample spacing and an odd number of intervals (a corrected last panel) | `scipy/integrate/_quadrature.py` |
| `simpson` with Richardson | Romberg integration, `scipy.integrate.quad` | adaptive Gauss-Kronrod: more nodes only where the error estimate is large | QUADPACK (`qags`) |
| `roc_auc` via `trapezoid` | `sklearn.metrics.roc_auc_score` | the same corners and trapezoids, plus multiclass averaging | `sklearn/metrics/_ranking.py` (`auc`) |
