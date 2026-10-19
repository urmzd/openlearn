# Definite Integrals: Trapezoid and Simpson

Once you train classifiers, such as the BERT and ELECTRA heads in [ELECTRA replaced-token detection](../../build-an-llm/p06-objectives/03-electra-replaced-token-detection.md) and [Fine-tuning heads and the linear-head export](../../build-an-llm/p06-objectives/05-fine-tuning-heads-and-the-linear-head-export.md) in *Build Your Own LLM System*, or the usage-policy head a serving gateway runs, you need to know how good a score is at separating two classes. The standard answer that does not depend on any one threshold is the area under the ROC curve, computed in [Logistic regression, ROC-AUC, and calibration](../07-probability-statistics/07-logistic-regression-roc-auc-calibration.md). That curve is not a formula. It is a list of corner points, one per threshold, and its area is a definite integral over samples. The same integral computes the reliability-curve area in a safety report.

Get the integration rule subtly wrong, by counting each step at its left height or by sorting points that were deliberately ordered, and every AUC you report is biased, with nothing visibly broken. This section builds the area rule for samples and, for functions you can evaluate anywhere, the two classic rules with predictable errors: the trapezoid rule and Simpson's rule. The finished code is [`course/solutions/python/tinyllm/num/integrate.py`](../../../../course/solutions/python/tinyllm/num/integrate.py), with `trapezoid`, `trapezoid_rule`, and `simpson` (and the helpers `_samples` and `_nodes`); the contract is [`course/contracts/py/tinyllm/num/integrate.pyi`](../../../../course/contracts/py/tinyllm/num/integrate.pyi).

There is nothing to call from earlier sections, but the error analysis reuses the Taylor expansion and Richardson extrapolation from [Derivatives and finite differences](01-derivatives-and-finite-differences.md).

> **Learning objectives**
> - Define the definite integral as a limit of Riemann sums and use the fundamental theorem of calculus to evaluate one exactly.
> - Derive the composite trapezoid and Simpson rules and state their error orders.
> - Explain why Simpson's rule is exact on cubics and why it is Richardson extrapolation of the trapezoid rule.
> - Integrate unevenly spaced samples with signed widths, without sorting.
> - Implement the three integrators with validation that keeps NaN areas from escaping.

## The definite integral

Cut $[a, b]$ into $n$ slices of width $h$, stand a rectangle on each slice with the height of $f$ somewhere in it, and add the areas: $\sum_i f(\xi_i)\, h$. This is a **Riemann sum**. As $h \to 0$ the sums for a continuous $f$ settle on one number whatever points $\xi_i$ you chose, and that number is the **definite integral** $\int_a^b f(x)\,dx$. Area below the axis counts as negative, which is why it is a *signed* area. Running from $b$ to $a$ makes every width $-h$, so $\int_b^a f = -\int_a^b f$, and an interval of length 0 has integral 0. A computer stops at $n$ slices.

The **fundamental theorem of calculus** links this area to derivatives: if $F$ is an antiderivative of $f$, meaning $F' = f$, then $\int_a^b f(x)\,dx = F(b) - F(a)$. For $f(x) = x^3$, $F(x) = x^4/4$, so $\int_0^2 x^3\,dx = 16/4 - 0 = 4$. Most integrals met in practice have no $F$ in closed form ($e^{-x^2/2}$, the shape of the normal density, is the famous one), or $f$ is known only at sample points. Then the area is computed numerically, which is called **quadrature**.

## The trapezoid

In what follows, $n$ is the number of equal steps, $h = (b - a)/n$ is the step, $x_i = a + ih$ for $i = 0, \ldots, n$ are the nodes (`numpy.linspace(a, b, n + 1)`), and $f_i = f(x_i)$. Replace $f$ on one step $[x_i, x_{i+1}]$ by the straight line through its two end values. The region under that line is a trapezoid with parallel sides $f_i$ and $f_{i+1}$ and width $h$, so its area is $h\,(f_i + f_{i+1})/2$: the width times the average height. A straight line is integrated exactly, because the "approximation" is the function itself.

## The composite rules

Add one trapezoid per step. Every inner node belongs to two trapezoids and is counted twice at half weight; the two end nodes belong to one each. The composite trapezoid value with step $h$ is

$$T(h) = h\left(\tfrac{1}{2} f_0 + f_1 + f_2 + \cdots + f_{n-1} + \tfrac{1}{2} f_n\right).$$

Simpson's rule instead takes the steps in pairs and fits the parabola through the three points at $x_{2j}, x_{2j+1}, x_{2j+2}$. The area under that parabola, over a panel of width $2h$, is $\frac{h}{3}(f_{2j} + 4 f_{2j+1} + f_{2j+2})$; to check it, integrate the parabola through $(-h, f_-), (0, f_0), (h, f_+)$ term by term. Adding the panels, every even inner node is shared by two panels:

$$S(h) = \frac{h}{3}\left(f_0 + 4 f_1 + 2 f_2 + 4 f_3 + 2 f_4 + \cdots + 2 f_{n-2} + 4 f_{n-1} + f_n\right).$$

The pattern 1, 4, 2, 4, ..., 2, 4, 1 closes only when $n$ is **even**. An odd $n$ is not a smaller Simpson's rule; it is a different, wrong one, so the contract raises instead of rounding.

## Integrating samples

An ROC curve is a list of points $(x_i, y_i)$ with uneven gaps, produced by sweeping a threshold. Its area is the trapezoid rule applied step by step with each step's own width: $\sum_i (x_{i+1} - x_i)(y_i + y_{i+1})/2$. The widths keep their **sign**, so the order of the points is the direction of travel: the same points in decreasing order give the negative area, exactly as running an integral from $b$ to $a$. The function never sorts, because sorting would silently reconnect the points in a different order; the curve the caller drew is the curve that gets measured. Fewer than two points enclose nothing, so the area is 0.

You now have three rules. The next question is how wrong each one is.

## Error and order

Expand $f$ around the middle of one step with Taylor's formula, which [Taylor series](../02-calculus-2/01-taylor-series.md) proves. The straight line misses the curvature term, and integrating the difference over one step gives an error of $-\frac{h^3}{12} f''(\xi)$ for some $\xi$ in the step. There are $n = (b - a)/h$ steps, so the composite errors are

$$\int_a^b f - T(h) = -\frac{(b - a)\, h^2}{12}\, f''(\xi), \qquad \int_a^b f - S(h) = -\frac{(b - a)\, h^4}{180}\, f^{(4)}(\xi).$$

The trapezoid rule is **second order**: halve $h$ and the error drops by 4. Simpson is **fourth order**: halve $h$ and it drops by 16. A parabola should only be exact on quadratics, yet the error involves $f^{(4)}$, so Simpson is exact on **cubics** too: the $x^3$ error on the left half of a panel cancels the one on the right half by symmetry. The table shows both rules on $\int_0^\pi \sin x\,dx = 2$:

| $n$ | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| trapezoid error | $4.3 \times 10^{-1}$ | $1.0 \times 10^{-1}$ | $2.6 \times 10^{-2}$ | $6.4 \times 10^{-3}$ | $1.6 \times 10^{-3}$ |
| Simpson error | $9.4 \times 10^{-2}$ | $4.6 \times 10^{-3}$ | $2.7 \times 10^{-4}$ | $1.7 \times 10^{-5}$ | $1.0 \times 10^{-6}$ |

Each column divides the trapezoid error by 4 and the Simpson error by 16: log-log slopes of 2 and 4. The error formulas assume $f$ is smooth at the scale of $h$. On Runge's peak $1/(1 + 25x^2)$ over $[-1, 1]$ with $n = 8$, Simpson's error is 0.026 and the trapezoid's 0.0075: the parabolas overshoot a peak that is narrower than two steps. Higher order pays off only once $h$ resolves the function.

## Simpson is Richardson

The trapezoid error is not just $O(h^2)$; it is a series in even powers, $T(h) = I + c_2 h^2 + c_4 h^4 + \cdots$, where $I$ is the exact integral (this is the Euler-Maclaurin formula). That is the same shape as the central difference's error series, so the same Richardson step removes the leading term:

$$\frac{4\,T(h) - T(2h)}{3} = I + O(h^4).$$

Write it out node by node and the weights come out as 1, 4, 2, 4, ..., 1 over 3: it *is* Simpson's rule on the finer grid. Repeating the step (weights $16, 64, \ldots$) is Romberg integration. The solution's `simpson` and `trapezoid_rule` satisfy this identity to rounding.

## Working it by hand

Take $\int_0^2 x^3\,dx = 4$ and apply each rule:

| Rule | Nodes and values | Computation | Value | Error |
|---|---|---|---|---|
| $T$, $n = 2$, $h = 1$ | $f(0, 1, 2) = 0, 1, 8$ | $1 \cdot (0/2 + 1 + 8/2)$ | 5 | 1 |
| $T$, $n = 4$, $h = 0.5$ | $f(0, 0.5, 1, 1.5, 2) = 0, 0.125, 1, 3.375, 8$ | $0.5 \cdot (0 + 0.125 + 1 + 3.375 + 4)$ | 4.25 | 0.25 |
| $S$, $n = 2$, $h = 1$ | $0, 1, 8$ | $\frac{1}{3}(0 + 4 \cdot 1 + 8)$ | 4 | 0 |
| Richardson | $T(0.5)$, $T(1)$ | $(4 \cdot 4.25 - 5)/3 = 12/3$ | 4 | 0 |

Halving $h$ divided the trapezoid error by exactly 4 (here $f'' = 6x$ and the error formula is exact up to the mean value). Simpson is exact because $x^3$ is a cubic, and Richardson's combination of the two trapezoid values gives the same 4.

Now take three samples, the corners of an ROC curve: $(0, 0)$, $(0.5, 0.75)$, $(1, 1)$. The two trapezoids have areas $0.5 \cdot (0 + 0.75)/2 = 0.1875$ and $0.5 \cdot (0.75 + 1)/2 = 0.4375$, for a total of **0.625**. Counting each step at its left height only (a left Riemann sum) gives $0.5 \cdot 0 + 0.5 \cdot 0.75 = 0.375$.

## Designing the interface

There is one sample integrator and two function integrators:

```python
def trapezoid(y: ArrayLike, x: ArrayLike) -> float: ...
def trapezoid_rule(f: Callable[[NDArray], NDArray], a: float, b: float, n: int) -> float: ...
def simpson(f: Callable[[NDArray], NDArray], a: float, b: float, n: int) -> float: ...
```

`trapezoid` integrates samples in the order given, like numpy's `trapezoid(y, x)`. The two rules take a function $f$ vectorized over float64 arrays and evaluate it once on `numpy.linspace(a, b, n + 1)`, so both end nodes are exact and every implementation agrees with scipy to the last bit; $b < a$ is allowed. All three return a Python `float` and raise `ValueError` for non-finite input, mismatched samples, a wrongly shaped or non-finite `f`, `n < 1`, or (for Simpson) an odd `n`.

## Building it step by step

Seven decisions produce the solution.

1. **For samples, use one trapezoid per step with its signed width.** Step $i$ contributes $(x_{i+1} - x_i)(y_i + y_{i+1})/2$. In array form that is `np.diff(x) * (y[:-1] + y[1:])`, summed, divided by 2. Nothing is sorted and no absolute value is taken, so the given order is the direction of travel.
2. **Give fewer than two samples no area.** With 0 or 1 points there are no steps; return 0.0 rather than indexing an empty difference.
3. **For functions, evaluate once on fixed nodes.** The nodes are `np.linspace(a, b, n + 1)`: $n$ steps need $n + 1$ nodes, and `linspace` puts the first and last exactly at $a$ and $b$. One vectorized call to `f` returns all $f_i$, which keeps every implementation of the contract (Python, C, scipy) on the same nodes and so on the same rounding.
4. **Let the step carry the direction.** $h = (b - a)/n$ is negative when $b < a$ and zero when $a = b$, so the composite formulas give $-\int_a^b$ and $0$ with no special cases.
5. **Apply the trapezoid weights.** The end nodes get $\tfrac12$, inner nodes 1, all times $h$.
6. **Apply the Simpson weights.** $\tfrac{h}{3}$ times $1, 4, 2, 4, \ldots, 2, 4, 1$. Odd-indexed nodes are the parabola midpoints (weight 4), even inner nodes are shared by two panels (weight 2). This closes only for even $n$, so odd $n$ raises.
7. **Validate everything that could make the area meaningless.** Non-finite limits or samples, mismatched lengths, an `f` that returns the wrong shape or a non-finite value, and bad `n` all raise `ValueError` rather than producing a NaN area.

The sample rule comes with its validator:

```python
def _samples(y: ArrayLike, x: ArrayLike) -> tuple[NDArray, NDArray]:
    ya = np.asarray(y, dtype=np.float64)
    xa = np.asarray(x, dtype=np.float64)
    if ya.ndim != 1 or xa.ndim != 1:          # a curve is a 1-D list of points
        raise ValueError(f"x and y must be 1-D, got shapes {xa.shape} and {ya.shape}")
    if ya.shape != xa.shape:                  # every height needs its abscissa
        raise ValueError(
            f"x and y must have the same length, got {xa.size} and {ya.size}"
        )
    if not (np.all(np.isfinite(ya)) and np.all(np.isfinite(xa))):
        raise ValueError("x and y must be finite")
    return ya, xa


def trapezoid(y: ArrayLike, x: ArrayLike) -> float:
    ya, xa = _samples(y, x)
    if ya.size < 2:
        return 0.0                            # no step, no area
    # One trapezoid per step: width times the mean of its two heights. The
    # widths keep their sign, so the order of the samples is the direction.
    return float(np.sum(np.diff(xa) * (ya[:-1] + ya[1:])) / 2.0)
```

On the ROC corners, `np.diff(xa)` is $[0.5, 0.5]$ and `ya[:-1] + ya[1:]` is $[0.75, 1.75]$; the products sum to $1.25$, and halving gives $0.625$. Note the argument order, `y` first, as in numpy and in the `trapezoid(tpr, fpr)` call that computes ROC-AUC.

The two function rules share node construction and checking:

```python
def _nodes(
    f: Callable[[NDArray], NDArray], a: float, b: float, n: int
) -> tuple[NDArray, float]:
    a, b = float(a), float(b)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"a and b must be finite, got {a} and {b}")
    xs = np.linspace(a, b, n + 1)             # n steps, n + 1 nodes, ends exact
    fx = np.asarray(f(xs), dtype=np.float64)  # one vectorized call
    if fx.shape != xs.shape:
        raise ValueError(f"f must return shape {xs.shape}, got {fx.shape}")
    if not np.all(np.isfinite(fx)):
        raise ValueError("f returned a non-finite value on the nodes")
    return fx, (b - a) / n                    # h is signed: negative when b < a


def _steps(n: int) -> int:
    if int(n) != n:                           # 4.0 is fine, 4.5 is not
        raise ValueError(f"n must be an integer, got {n!r}")
    return int(n)


def trapezoid_rule(
    f: Callable[[NDArray], NDArray], a: float, b: float, n: int
) -> float:
    n = _steps(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    fx, h = _nodes(f, a, b, n)
    # The two ends belong to one trapezoid each, every inner node to two.
    return float(h * (fx[0] / 2.0 + np.sum(fx[1:-1]) + fx[-1] / 2.0))


def simpson(f: Callable[[NDArray], NDArray], a: float, b: float, n: int) -> float:
    n = _steps(n)
    if n < 2 or n % 2 != 0:
        raise ValueError(f"Simpson needs an even number of steps n >= 2, got {n}")
    fx, h = _nodes(f, a, b, n)
    # Weights 1, 4, 2, 4, ..., 2, 4, 1: odd nodes are parabola midpoints,
    # even inner nodes are shared by two parabolas.
    odd = np.sum(fx[1:-1:2])                  # indices 1, 3, ..., n - 1
    even = np.sum(fx[2:-1:2])                 # indices 2, 4, ..., n - 2 (empty when n = 2)
    return float(h / 3.0 * (fx[0] + 4.0 * odd + 2.0 * even + fx[-1]))
```

On $\int_0^2 x^3$ with $n = 2$, the nodes are $0, 1, 2$, `fx` is $[0, 1, 8]$, `odd` is 1, `even` is the empty sum 0, and the result is $\tfrac13(0 + 4 + 0 + 8) = 4$. With `trapezoid_rule` and $n = 2$: $1 \cdot (0 + 1 + 4) = 5$. The Richardson identity holds node by node: $4T(h) - T(2h)$ gives $h$ times $1, 4, 2, 4, \ldots, 1$, and dividing by 3 is exactly the `simpson` expression. The full solution is in [`integrate.py`](../../../../course/solutions/python/tinyllm/num/integrate.py).

### Why this design

All three functions are $O(n)$ in time and memory: one pass of `np.diff` and a sum for samples, and one call to `f` on $n + 1$ nodes for the rules. To reach an error $\delta$ the trapezoid rule needs $n \propto \delta^{-1/2}$ nodes and Simpson $n \propto \delta^{-1/4}$, so Simpson is far cheaper for smooth $f$.

The sample rule neither sorts nor takes absolute widths. Sorting would make a decreasing curve's area positive and silently reconnect unsorted points into a different curve; `np.abs(np.diff(x))` does the first. The caller's order is meaningful (it is the threshold sweep), so the function measures exactly the polyline it was given.

The nodes come from `linspace` rather than `a + i * h`. With a rounded $h$, $a + nh$ need not equal $b$, so the last node drifts. `linspace` pins both ends, and because scipy and the C port use the same nodes, results agree bit for bit.

Simpson raises on odd $n$. scipy's `simpson` accepts an odd number of intervals with a corrected last panel. That is a different rule with a different error constant; the contract keeps the classic composite rule and rejects what it does not implement.

One tradeoff is accepted. Fixed-step rules spend nodes uniformly, so for a function with a narrow peak, like Runge's example, Simpson can be worse than the trapezoid until $h$ resolves the peak. Adaptive quadrature solves that at the cost of an unpredictable number of evaluations, which an ROC area or a reference test does not need.

## What can go wrong

A correct solution reproduces the hand-worked values 5, 4.25, 4, and 4, and the ROC corners must give 0.625, since `roc_auc` is exactly this call. The rules must agree with `scipy.integrate.trapezoid` and `simpson` on the same nodes, for six functions and a reversed interval, and the sample rule must agree with scipy on uneven, unsorted, and decreasing samples, which is what ROC and reliability curves look like. The error orders are tested as properties: log-log slope 2 for the trapezoid and 4 for Simpson. Simpson must be exact on 40 random cubics with $n = 2, 4, 10$; the trapezoid must be exact on lines but not on $x^2$; and $S = (4T(h) - T(2h))/3$ must hold to rounding. Reversed and empty intervals must give $\int_b^a = -\int_a^b$ and $\int_a^a = 0$, and decreasing samples a negative area. The rules make one vectorized call to `f` on `linspace(a, b, n + 1)`, which keeps results bit-identical across languages, and return a Python `float` rather than a numpy scalar, because callers compare and serialize it. Zero or one sample gives 0.0, which covers a one-threshold ROC curve. Odd or non-positive `n`, non-finite limits, a bad `f`, and bad samples all raise.

The usual mistakes look like this:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| a Riemann sum (left heights only) for samples | the hand ROC area is 0.375, not 0.625; AUCs biased low | `ya[:-1] + ya[1:]`, halved: the mean of both heights |
| Simpson weights 2 and 4 swapped | order 2 instead of 4; the hand example gives 10/3 | `4.0 * odd` on indices `1:-1:2`, `2.0 * even` on `2:-1:2` |
| accepting an odd number of Simpson steps | a plausible but wrong number | raises unless `n >= 2` and `n % 2 == 0` |
| counting the end points fully in the trapezoid rule | the hand example gives 9, order 1 | `fx[0] / 2.0` and `fx[-1] / 2.0` |
| sorting samples or using $\lvert \Delta x \rvert$ | a decreasing curve has positive area; unsorted samples are reconnected | plain `np.diff(xa)`, no sort |
| $n$ nodes instead of $n + 1$ | the last step is lost; the rules disagree with scipy | `np.linspace(a, b, n + 1)` |
| Simpson with the panel width $2h$ in place of the step $h$ | every result doubled | `h / 3.0` with `h = (b - a) / n` |
| assuming equal gaps between samples | wrong area on any real ROC curve | each step uses its own width from `np.diff` |

> **Warning:** The left-Riemann-sum mistake is the one that ships. It produces a number between 0 and 1 that looks like an AUC, and it is biased low on every rising curve.

## Where this leads

The main consumer is `roc_auc(scores, labels)` in [Logistic regression, ROC-AUC, and calibration](../07-probability-statistics/07-logistic-regression-roc-auc-calibration.md), which builds the ROC corners and returns `trapezoid(tpr, fpr)`. In *Responsible AI*, [Bias and safety evals](../../responsible-ai/04-bias-and-safety-evals/01-bias-and-safety-evals.md) integrates the safety report's ROC area and reliability curve the same way. Question 8 of the [probability problem set on hypothesis tests and regression](../07-probability-statistics/93-problem-set-d.md) computes an AUC by hand, the same area seen as the fraction of pairs ranked correctly.

### Further reading

- OpenStax, *Calculus Volume 1* (free), sections 5.1 to 5.3 (the definite integral and the fundamental theorem), and *Calculus Volume 2*, section 3.6 (numerical integration).
- Sauer, *Numerical Analysis*, sections 5.2 and 5.3: Newton-Cotes rules and Romberg integration.
- `numpy.trapezoid` and `scipy.integrate.trapezoid` (`numpy/lib/_function_base_impl.py`): integration along any axis of an N-d array.
- `scipy.integrate.simpson` (`scipy/integrate/_quadrature.py`): uneven sample spacing and an odd number of intervals with a corrected last panel.
- Romberg integration and `scipy.integrate.quad` (QUADPACK `qags`): adaptive Gauss-Kronrod quadrature that adds nodes only where the error estimate is large.
- `sklearn.metrics.roc_auc_score` (`sklearn/metrics/_ranking.py`, `auc`): the same corners and trapezoids, plus multiclass averaging.

## Key terms

- **Riemann sum**: $\sum_i f(\xi_i)\, h$, the sum of thin rectangles under $f$.
- **definite integral**: the limit of Riemann sums, the signed area between $f$ and the $x$ axis.
- **fundamental theorem of calculus**: if $F' = f$, then $\int_a^b f = F(b) - F(a)$.
- **quadrature**: numerical computation of a definite integral.
- **trapezoid rule**: joining neighboring points by straight lines; second order, exact on lines.
- **Simpson's rule**: fitting a parabola through each three points; fourth order, exact on cubics, needs an even number of steps.
- **Romberg integration**: repeated Richardson extrapolation of the trapezoid rule.

## Summary

- The definite integral $\int_a^b f(x)\,dx$ is the signed area under $f$, the limit of sums of thin slices; a computer stops at $n$ slices of width $h$.
- The trapezoid rule joins neighboring points by straight lines: error proportional to $h^2$, exact on lines.
- Simpson's rule fits a parabola through each three points: error proportional to $h^4$ and, by a symmetry bonus, exact on cubics.
- Simpson is Richardson extrapolation of the trapezoid rule, $(4T(h/2) - T(h))/3$, the same error-cancelling step used for finite differences.
- Samples that are not on a grid, such as the corners of an ROC curve, are integrated step by step with each step's own signed width, in the order given.

## Review questions

1. Why does `trapezoid` use `np.diff(xa)` without an absolute value or a sort?

   <details><summary>Answer</summary>

   The widths are signed so that the order of the points is the direction of integration: decreasing $x$ gives a negative area, matching $\int_b^a f = -\int_a^b f$. Sorting or taking $|\Delta x|$ would change the curve being measured (or its orientation); an ROC curve's corners come in threshold order, and that order is the curve.

   </details>

2. Why does the hand ROC area come out as 0.625 and not 0.375?

   <details><summary>Answer</summary>

   Each step's height is the mean of its two end heights, $(y_i + y_{i+1})/2$: the straight segment between corners, which is what an ROC curve is. 0.375 uses only the left height of each step, a left Riemann sum, which assumes the curve stays flat until the next corner and underestimates every rising curve.

   </details>

3. Why must Simpson's $n$ be even, and why does `simpson` raise rather than drop a step?

   <details><summary>Answer</summary>

   Simpson fits one parabola per *pair* of steps, so the steps must pair up: the weight pattern $1, 4, 2, \ldots, 4, 1$ only ends on 1 when $n$ is even. Dropping or patching a step would integrate a different interval or use a different rule with a different error, returning a plausible but wrong number. Raising makes the caller choose.

   </details>

4. Simpson fits parabolas, so why is it exact on cubics?

   <details><summary>Answer</summary>

   On a panel $[-h, h]$ around its midpoint, write the cubic as a quadratic plus $c\,x^3$. The quadratic part is integrated exactly by the parabola. $x^3$ is odd: its integral over $[-h, h]$ is 0, and Simpson's weights $\tfrac h3(1, 4, 1)$ at $-h, 0, h$ also give $\tfrac h3(-h^3 + 0 + h^3) = 0$. The error formula reflects this: it involves $f^{(4)}$, which is 0 for a cubic.

   </details>

5. What does $h = (b - a)/n$ being signed buy, and what happens for $a = b$?

   <details><summary>Answer</summary>

   With $b < a$, `linspace` runs downward and $h < 0$, so both rules return the negative of the forward integral with no extra branch. With $a = b$, every node is $a$ and $h = 0$, so the result is exactly 0. Both are the signed-area rules of the definite integral, obtained by the same expression.

   </details>
