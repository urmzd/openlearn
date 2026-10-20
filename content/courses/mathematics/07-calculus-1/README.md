# Calculus 1

Calculus is the mathematics of change and accumulation. Differentiation measures an instantaneous rate of change; integration measures a total accumulated amount. The fundamental theorem of calculus shows that the two are inverse operations, one of the most powerful ideas in all of mathematics. For a programmer working on machine learning, both halves are everyday tools: every training step follows a derivative downhill, and every probability, area under a curve, and averaged metric is an integral.

This chapter teaches Calculus 1 the way a numerical programmer meets it. It opens with the derivative as a limit and asks what happens when a computer, which cannot take a limit, evaluates the difference quotient at one step size; the answer gives the finite differences that later check every gradient in the book. It then uses the derivative to find roots with Newton's method, including the division-free reciprocal square root that RMSNorm kernels rely on. Next it derives and implements the activation functions of neural networks together with their derivatives, written so that no input can overflow. It closes with the definite integral and the trapezoid and Simpson rules that compute ROC areas. A problem set at the end works through the pen-and-paper side of the subject: limits, derivative rules, related rates, optimization, integrals, and L'Hôpital's rule.

> **Learning objectives**
> - Explain the derivative as a limit and predict the truncation and rounding error of a finite-difference approximation.
> - Differentiate any elementary formula with the product, quotient, and chain rules, and use derivatives to optimize and to find roots.
> - Evaluate activation functions and their derivatives without overflow or loss of relative accuracy.
> - Compute definite integrals exactly with the fundamental theorem and numerically with the trapezoid and Simpson rules, knowing each rule's error order.
> - Recognize where these ideas reappear in gradient checking, optimization, and evaluation metrics.

## Before you begin

This chapter assumes precalculus: functions, exponents and logarithms, trigonometry, and polynomials, as covered in the [Precalculus](../06-precalculus/README.md) chapter, in particular [Exponents, logs, and units of information](../06-precalculus/01-exponents-logs-and-units-of-information.md) and [Polynomials, Horner, and stable roots](../06-precalculus/04-polynomials-horner-and-stable-roots.md). The code sections use Python and numpy. No earlier calculus is required.

## In this chapter

1. [Derivatives and finite differences](01-derivatives-and-finite-differences.md). The derivative as a limit, the forward and central differences, the step size that balances truncation against rounding, and Richardson extrapolation.
2. [Newton's Method](02-newtons-method.md). Root finding by tangent lines, quadratic convergence and its failure modes, and the division-free iteration for $1/\sqrt{x}$.
3. [Activation Functions and Their Derivatives](03-activation-functions.md). Sigmoid, tanh, ReLU, softplus, both GELUs, and SiLU, with derivatives derived from the rules of calculus and evaluated without overflow.
4. [Definite integrals: trapezoid and Simpson](04-definite-integrals-trapezoid-simpson.md). The definite integral, the trapezoid and Simpson rules with their error orders, Simpson as Richardson extrapolation, and the signed-width rule for ROC curves.
5. [Calculus 1 problem set](05-problem-set.md). Fifty-seven problems with worked answers on limits, derivatives, related rates, optimization, integrals, and L'Hôpital's rule.

## A map of Calculus 1

The sections above build what the rest of the book needs. Here, for reference, is the classical Calculus 1 core they rest on, with one worked example per topic. The section numbers refer to OpenStax, *Calculus Volume 1*.

### Limits and continuity

The limit $\lim_{x \to a} f(x) = L$ means that for every $\varepsilon > 0$ there exists $\delta > 0$ such that $0 < |x - a| < \delta$ implies $|f(x) - L| < \varepsilon$. A function is **continuous** at $a$ if $\lim_{x \to a} f(x) = f(a)$, and the **one-sided limits** $\lim_{x \to a^+} f(x)$ and $\lim_{x \to a^-} f(x)$ restrict $x$ to one side. Two theorems do much of the work. The **squeeze theorem** says that if $g(x) \le f(x) \le h(x)$ near $a$ and $\lim g = \lim h = L$, then $\lim f = L$: a function trapped between two functions that converge to the same value must converge there too. The **intermediate value theorem** says that if $f$ is continuous on $[a, b]$ and $N$ lies between $f(a)$ and $f(b)$, then $f(c) = N$ for some $c$ in $(a, b)$: a continuous function cannot jump over a value.

For example, $\lim_{x \to 0} \frac{\sin x}{x}$ cannot be evaluated by substitution, but for $0 < x < \pi/2$ you have $\cos x \le \frac{\sin x}{x} \le 1$. As $x \to 0$, $\cos x \to 1$, so by the squeeze theorem the limit is 1. (OpenStax sections 2.1 to 2.5.)

### Derivatives

The derivative $f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$ is the instantaneous rate of change; $f$ is differentiable at $a$ if $f'(a)$ exists, and differentiability implies continuity. The rules are the power rule $\frac{d}{dx} x^n = n x^{n-1}$, the product rule $(fg)' = f'g + fg'$, the quotient rule $(f/g)' = (f'g - fg')/g^2$, and the chain rule $\frac{d}{dx} f(g(x)) = f'(g(x))\, g'(x)$. Rates of change multiply through composition: if gear A turns gear B, which turns gear C, the total rate is the product of the individual rates. **Implicit differentiation** differentiates both sides of an equation in $x$, treating $y$ as a function of $x$, and solves for $dy/dx$.

For example, on the circle $x^2 + y^2 = 25$, differentiating gives $2x + 2y\,\frac{dy}{dx} = 0$, so $\frac{dy}{dx} = -x/y$. At $(3, 4)$ the slope of the tangent is $-3/4$. (OpenStax sections 3.1 to 3.9.)

### Applications of derivatives

A **critical point** is a $c$ where $f'(c) = 0$ or $f'(c)$ is undefined. A local maximum or minimum there is identified by a sign change of $f'$ (the first derivative test) or by the sign of $f''(c)$ (the second derivative test). An **inflection point** is where $f''$ changes sign, so the concavity changes. The **mean value theorem** says that if $f$ is continuous on $[a, b]$ and differentiable on $(a, b)$, then $f'(c) = \frac{f(b) - f(a)}{b - a}$ for some $c$ in $(a, b)$: a car that averages 60 mph must at some instant be going exactly 60 mph. **L'Hôpital's rule** says that if $\lim f/g$ is a $0/0$ or $\infty/\infty$ form, then $\lim f/g = \lim f'/g'$, provided the latter exists: when numerator and denominator both vanish, their rates of vanishing determine the limit.

For example, a farmer has 200 m of fencing to enclose a rectangle against a river, with no fence on the river side. With width $x$, the length is $200 - 2x$ and the area is $A(x) = x(200 - 2x) = 200x - 2x^2$. Then $A'(x) = 200 - 4x = 0$ at $x = 50$, and $A''(50) = -4 < 0$ confirms a maximum of $50 \cdot 100 = 5000$ m². (OpenStax sections 4.1 to 4.7.)

### Integration

A **Riemann sum** $\sum_i f(x_i^*)\,\Delta x$ over a partition approximates the area under a curve, and the **definite integral** $\int_a^b f(x)\,dx$ is its limit as the partition is refined. $F$ is an **antiderivative** of $f$ if $F' = f$. The fundamental theorem has two parts. Part 1: if $F(x) = \int_a^x f(t)\,dt$, then $F'(x) = f(x)$, so differentiation undoes integration. Part 2: $\int_a^b f(x)\,dx = F(b) - F(a)$ for any antiderivative $F$, so total accumulation is the antiderivative evaluated at the endpoints. **Substitution** is the chain rule in reverse: $\int f(g(x))\,g'(x)\,dx = \int f(u)\,du$ with $u = g(x)$.

For example, to evaluate $\int_0^2 x\,e^{x^2}\,dx$, let $u = x^2$, so $du = 2x\,dx$ and $x\,dx = du/2$; the bounds become $u(0) = 0$ and $u(2) = 4$. The integral is $\frac12 \int_0^4 e^u\,du = \frac12(e^4 - 1)$. (OpenStax sections 5.1 to 5.7.)

### Applications of integration

The area between curves with $f(x) \ge g(x)$ is $\int_a^b [f(x) - g(x)]\,dx$. Rotating a region about the $x$ axis gives a volume by the **disk method**, $V = \pi \int_a^b [R(x)]^2\,dx$, and rotating about the $y$ axis gives one by the **shell method**, $V = 2\pi \int_a^b x\,f(x)\,dx$.

For example, rotating $y = \sqrt{x}$ for $0 \le x \le 4$ about the $x$ axis gives, by the disk method, $V = \pi \int_0^4 (\sqrt{x})^2\,dx = \pi \int_0^4 x\,dx = \pi \left[x^2/2\right]_0^4 = 8\pi$. (OpenStax sections 6.1 to 6.4.)

### Choosing a technique

The following table matches each standard technique to the situation that calls for it:

| Technique | When to use | Key idea |
|---|---|---|
| Squeeze theorem | limit of an oscillating or bounded function | bound $f$ between $g$ and $h$ with the same limit |
| L'Hôpital's rule | $0/0$ or $\infty/\infty$ indeterminate forms | differentiate numerator and denominator |
| Power rule | polynomial derivatives | $\frac{d}{dx} x^n = n x^{n-1}$ |
| Chain rule | composite functions | multiply derivatives through the composition |
| Substitution | integrals of compositions | reverse the chain rule |
| Disk or shell method | volume of revolution | choose by the axis of rotation versus the variable |
| Optimization | maximum or minimum of a quantity | set $f' = 0$; check endpoints and critical points |

### Where Calculus 1 shows up in computing

Limits underlie asymptotic analysis (Big-O and Big-Theta) in *Algorithms*. Derivatives drive gradient descent in machine learning, developed in [Gradient descent and line search](../16-optimization/01-gradient-descent-and-line-search.md). Optimization connects to greedy algorithms and linear-programming relaxations. Integration defines probability densities, the subject of [Probability and statistics](../13-probability-statistics/README.md). Rates of change reappear in the amortized analysis of data structures. Interviews and quantitative roles test the same ideas: optimization problems, the mathematics behind PageRank, derivative pricing and rate-of-change models, and the derivative intuition gradient descent requires.

## Key terms

- **Continuous function / continuity**: $f$ is continuous at $a$ when $\lim_{x \to a} f(x) = f(a)$, so the graph has no jump or hole there; many theorems of calculus, such as the intermediate value theorem, require it.
- **One-sided limit**: $\lim_{x \to a^+} f(x)$ or $\lim_{x \to a^-} f(x)$, the limit as $x$ approaches $a$ from the right or the left only; the two-sided limit exists exactly when both one-sided limits exist and agree.
- **Squeeze theorem**: if $g(x) \le f(x) \le h(x)$ near $a$ and $\lim g = \lim h = L$, then $\lim f = L$; it settles limits such as $\lim_{x \to 0} \sin x / x = 1$ that substitution cannot.
- **Intermediate value theorem**: if $f$ is continuous on $[a, b]$ and $N$ lies between $f(a)$ and $f(b)$, then $f(c) = N$ for some $c$ in $(a, b)$; a continuous function cannot jump over a value, which is why bisection finds roots.
- **Product rule / quotient rule**: $(fg)' = f'g + fg'$ and $(f/g)' = (f'g - fg')/g^2$, the rules for differentiating a product or a ratio of two functions.
- **Power rule**: $\frac{d}{dx} x^n = n x^{n-1}$, the derivative of a power, and with linearity the derivative of any polynomial.
- **Implicit differentiation**: differentiating both sides of an equation in $x$ and $y$, treating $y$ as a function of $x$, then solving for $dy/dx$; it gives slopes of curves such as $x^2 + y^2 = 25$ that are not written as $y = f(x)$.
- **First derivative test**: a critical point $c$ is a local maximum if $f'$ changes from positive to negative at $c$ and a local minimum if it changes from negative to positive.
- **Second derivative test**: at a critical point $c$ with $f'(c) = 0$, $f''(c) < 0$ means a local maximum and $f''(c) > 0$ a local minimum; when $f''(c) = 0$ the test is inconclusive.
- **Inflection point**: a point where $f''$ changes sign, so the graph switches between curving up (concave up) and curving down (concave down).
- **Mean value theorem**: if $f$ is continuous on $[a, b]$ and differentiable on $(a, b)$, then $f'(c) = (f(b) - f(a))/(b - a)$ for some $c$ in $(a, b)$; somewhere the instantaneous rate equals the average rate.
- **Antiderivative**: a function $F$ with $F' = f$; by the fundamental theorem of calculus, $\int_a^b f(x)\,dx = F(b) - F(a)$ for any antiderivative $F$ of $f$.
- **Integration by substitution / u-substitution**: the chain rule in reverse, $\int f(g(x))\,g'(x)\,dx = \int f(u)\,du$ with $u = g(x)$; it turns an integral of a composition into a simpler one, with the bounds changed to $g(a)$ and $g(b)$.
- **Disk method / shell method**: formulas for the volume of a solid of revolution; the disk method $V = \pi \int_a^b R(x)^2\,dx$ slices perpendicular to the axis of rotation, and the shell method $V = 2\pi \int_a^b x\,f(x)\,dx$ wraps cylinders around the $y$ axis.

## Further reading

- OpenStax, [*Calculus Volume 1*](https://openstax.org/details/books/calculus-volume-1) (free, CC BY-NC-SA 4.0), chapters 2 to 6. Useful exercises: section 2.2 #65 to 72, 2.3 #95 to 106, 2.4 #131 to 140, and the challenge set 2.5 #191 to 196; section 3.2 #55 to 68, 3.3 #100 to 110, 3.4 #140 to 155, 3.6 #210 to 222, and the challenge set 3.8 #308 to 315; section 4.3 #101 to 110, 4.5 #198 to 210, 4.7 #300 to 310, and the challenge set 4.7 #316 to 320; section 5.2 #67 to 76, 5.3 #148 to 160, 5.5 #256 to 268, and the challenge set 5.7 #340 to 348; section 6.1 #1 to 12, 6.2 #55 to 70, 6.3 #105 to 115, and the challenge set 6.4 #140 to 148.
- [Paul's Online Math Notes](https://tutorial.math.lamar.edu): additional worked examples on tricky topics.
- 3Blue1Brown, [*Essence of Calculus*](https://www.3blue1brown.com/topics/calculus): geometric intuition, best watched before or after reading.
