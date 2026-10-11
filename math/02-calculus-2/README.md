# Calculus 2

## Overview
- **Textbook**: *Calculus Volume 2* -- https://openstax.org/details/books/calculus-volume-2 (CC BY-NC-SA 4.0)
- **Supplementary**: [Paul's Online Math Notes](https://tutorial.math.lamar.edu), [3Blue1Brown](https://www.3blue1brown.com/topics/calculus)
- **Prerequisites**: [Calculus 1](../01-calculus-1/)
- **Estimated time**: 4 weeks at 10-12 hrs/week

## Key Takeaways
- Master advanced integration techniques that appear in probability and physics
- Understand infinite series and convergence -- the foundation of numerical computation
- Approximate functions with Taylor/Maclaurin series (how computers actually compute sin, cos, exp)
- Model dynamic systems with introductory differential equations
- Build intuition for when series converge and how fast -- critical for algorithm analysis

## How to Study
- Read textbook sections listed under each concept
- Work through essential problems with pencil and paper
- Practice choosing the right integration technique before computing
- For series, focus on building a decision tree for which convergence test to apply
- Review connections to CS topics in the algorithms track

---

# Concepts & Techniques

## Core Insight

Calculus 2 extends integration to harder problems and then pivots to the profound idea that
functions can be represented as infinite sums (series). This is how computers approximate
transcendental functions and how we analyze the long-run behavior of algorithms.

## 1. Integration Techniques

**Textbook sections**: Ch 1, Sections 1.1-1.5

**Key techniques**:
- **Integration by parts**: integral of u dv = uv - integral of v du. Choose u by LIATE (Log, Inverse trig, Algebraic, Trig, Exponential).
- **Trigonometric integrals**: Use identities like sin^2(x) = (1 - cos(2x))/2 to reduce powers.
- **Trigonometric substitution**: For sqrt(a^2 - x^2), let x = a*sin(theta). For sqrt(a^2 + x^2), let x = a*tan(theta). For sqrt(x^2 - a^2), let x = a*sec(theta).
- **Partial fractions**: Decompose P(x)/Q(x) into simpler fractions when degree(P) < degree(Q).

**Worked example**:
> Evaluate integral of x * e^x dx. By parts: u = x, dv = e^x dx, so du = dx, v = e^x. Result: x*e^x - integral of e^x dx = x*e^x - e^x + C = e^x(x - 1) + C.

> Evaluate integral of 1/(x^2 - 1) dx. Partial fractions: 1/(x^2-1) = 1/[(x-1)(x+1)] = A/(x-1) + B/(x+1). Solving: A = 1/2, B = -1/2. Integral = (1/2)ln|x-1| - (1/2)ln|x+1| + C.

**Essential problems**: OpenStax 1.1 Exercises: #1-15; 1.3 Exercises: #115-128; 1.4 Exercises: #155-170
**Challenge problems**: OpenStax 1.5 Exercises: #200-210

## 2. Applications of Integration

**Textbook sections**: Ch 2, Sections 2.1-2.4

**Key definitions**:
- **Arc length**: L = integral from a to b of sqrt(1 + [f'(x)]^2) dx
- **Surface area of revolution**: S = 2*pi * integral from a to b of f(x) * sqrt(1 + [f'(x)]^2) dx
- **Center of mass**: x_bar = M_y / m where M_y = integral of x * rho(x) * f(x) dx

**Worked example**:
> Find the arc length of y = x^(3/2) from x=0 to x=4. f'(x) = (3/2)x^(1/2). L = integral from 0 to 4 of sqrt(1 + 9x/4) dx. Let u = 1 + 9x/4, du = 9/4 dx. L = (4/9) * (2/3) * [u^(3/2)] from 1 to 10 = (8/27)(10*sqrt(10) - 1).

**Essential problems**: OpenStax 2.1 Exercises: #1-10; 2.2 Exercises: #50-60; 2.4 Exercises: #100-110
**Challenge problems**: OpenStax 2.4 Exercises: #115-120

## 3. Parametric Curves and Polar Coordinates

**Textbook sections**: Ch 3, Sections 3.1-3.4

**Key definitions**:
- **Parametric curve**: x = f(t), y = g(t); derivative dy/dx = (dy/dt)/(dx/dt)
- **Polar coordinates**: x = r*cos(theta), y = r*sin(theta)
- **Polar area**: A = (1/2) * integral from alpha to beta of [r(theta)]^2 d(theta)

**Key theorems**:
- **Parametric arc length**: L = integral from a to b of sqrt([dx/dt]^2 + [dy/dt]^2) dt. *Intuition*: at each instant, the speed is the hypotenuse of the velocity components.

**Worked example**:
> Find the area enclosed by one petal of r = cos(2*theta). One petal spans theta in [-pi/4, pi/4]. A = (1/2) integral from -pi/4 to pi/4 of cos^2(2*theta) d(theta) = (1/2) * (pi/4) = pi/8.

**Essential problems**: OpenStax 3.1 Exercises: #1-12; 3.3 Exercises: #110-122; 3.4 Exercises: #150-162
**Challenge problems**: OpenStax 3.4 Exercises: #168-175

## 4. Sequences and Series

**Textbook sections**: Ch 5, Sections 5.1-5.6

**Key definitions**:
- **Sequence**: {a_n} converges if lim(n->inf) a_n exists and is finite
- **Series**: Sum from n=1 to inf of a_n; converges if the sequence of partial sums converges
- **Power series**: Sum of c_n * (x - a)^n with radius of convergence R
- **Taylor series**: f(x) = Sum from n=0 to inf of f^(n)(a)/n! * (x-a)^n

**Key theorems / convergence tests**:
- **Divergence Test**: If lim a_n != 0, the series diverges. *Intuition*: terms must shrink to zero (necessary but not sufficient).
- **Ratio Test**: If lim |a_(n+1)/a_n| = L < 1, converges absolutely; L > 1, diverges. *Intuition*: compare to a geometric series.
- **Root Test**: If lim |a_n|^(1/n) = L, same conclusion as ratio test.
- **Integral Test**: If f is positive, continuous, decreasing and a_n = f(n), the series and integral from 1 to inf of f(x) dx converge/diverge together.
- **Comparison Test**: Compare with a known series (geometric, p-series).
- **Alternating Series Test**: If |a_n| decreases to 0, the alternating series converges.

**Important series**:
- Geometric: Sum r^n = 1/(1-r) for |r| < 1
- p-series: Sum 1/n^p converges iff p > 1
- Harmonic: Sum 1/n diverges (p = 1)
- e^x = Sum x^n/n!, sin(x) = Sum (-1)^n * x^(2n+1)/(2n+1)!, cos(x) = Sum (-1)^n * x^(2n)/(2n)!

**Worked example**:
> Determine convergence of Sum from n=1 to inf of n^2/2^n. Apply the ratio test: |a_(n+1)/a_n| = (n+1)^2 / (2*n^2). As n->inf, this approaches 1/2 < 1. The series converges.

**Essential problems**: OpenStax 5.1 Exercises: #1-14; 5.3 Exercises: #120-138; 5.4 Exercises: #175-190; 5.6 Exercises: #260-275
**Challenge problems**: OpenStax 5.6 Exercises: #280-288

## 5. Introduction to Differential Equations

**Textbook sections**: Ch 4, Sections 4.1-4.3

**Key definitions**:
- **ODE**: An equation involving a function and its derivatives
- **Separable equation**: dy/dx = f(x)*g(y); solve by separating variables and integrating both sides
- **First-order linear**: dy/dx + P(x)*y = Q(x); solve using integrating factor mu(x) = e^(integral P(x) dx)

**Worked example**:
> Solve dy/dx = 2xy with y(0) = 1. Separate: dy/y = 2x dx. Integrate: ln|y| = x^2 + C. With y(0)=1: C=0. Solution: y = e^(x^2).

**Essential problems**: OpenStax 4.1 Exercises: #1-10; 4.3 Exercises: #100-115
**Challenge problems**: OpenStax 4.3 Exercises: #120-125

---

## Technique Catalog

| Technique | When to Use | Key Formula/Idea |
|-----------|-------------|------------------|
| Integration by parts | Product of two function types | LIATE priority for choosing u |
| Trig substitution | sqrt(a^2 +/- x^2) or sqrt(x^2 - a^2) | Substitute x = a*sin/tan/sec |
| Partial fractions | Rational functions P(x)/Q(x) | Factor denominator, decompose |
| Ratio test | Series with factorials or exponentials | Compute lim |a_(n+1)/a_n| |
| Comparison test | Series similar to known p-series/geometric | Bound by known convergent/divergent series |
| Taylor expansion | Approximate f(x) near a point | f(x) = Sum f^(n)(a)/n! * (x-a)^n |
| Separable ODE | dy/dx = f(x)*g(y) | Separate and integrate both sides |

## Connections to CS & Algorithms

| Math Concept | CS Application | Repo Link |
|-------------|----------------|-----------|
| Harmonic series | Average-case hashing analysis, coupon collector | algorithms track |
| Taylor series | How computers compute sin/cos/exp, floating-point | systems track |
| Power series / radius | Convergence of iterative algorithms | algorithms track |
| Differential equations | System modeling, PID controllers, epidemics | systems track |
| Convergence tests | Analyzing whether iterative methods terminate | ML track |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Quantitative Finance | Series pricing models, stochastic DEs | Hard |
| Google/DeepMind | Numerical stability of ML training loops | Hard |
| Signal Processing (Apple, Qualcomm) | Fourier series from Taylor series foundations | Medium |
| Robotics (Boston Dynamics, Tesla) | Differential equations for control systems | Hard |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `M02.1` | [Taylor series, remainder bounds, range reduction](01-taylor-series.md) | build | 2 |
| 2 | `M02.2` | [Series convergence, EMA as a geometric series, bias correction](02-series-and-the-ema.md) | build | 2 |
| 3 | `S-M02` | [Calculus 2 problem set: integration techniques, the Gaussian, series, Taylor, polar, ODEs](90-problem-set.md) | solve | 2 |
<!-- /ol:chapters -->
