# Calculus 1

## Overview
- **Textbook**: *Calculus Volume 1* -- https://openstax.org/details/books/calculus-volume-1 (CC BY-NC-SA 4.0)
- **Supplementary**: [Paul's Online Math Notes](https://tutorial.math.lamar.edu), [3Blue1Brown Essence of Calculus](https://www.3blue1brown.com/topics/calculus)
- **Prerequisites**: None (precalculus assumed)
- **Estimated time**: 4 weeks at 10-12 hrs/week

## Key Takeaways
- Understand limits as the foundation for all of calculus
- Compute derivatives using formal rules and apply them to real-world optimization
- Connect differentiation and integration through the Fundamental Theorem of Calculus
- Set up and evaluate integrals for area, volume, and accumulated quantities
- Recognize asymptotic behavior and rate-of-change reasoning used throughout CS

## How to Study
- Read textbook sections listed under each concept
- Work through essential problems with pencil and paper -- no shortcuts
- Watch 3Blue1Brown videos for geometric intuition before or after reading
- Use Paul's Online Math Notes for additional worked examples on tricky topics
- Review connections to CS topics in the algorithms track

---

# Concepts & Techniques

## Core Insight

Calculus is the mathematics of change and accumulation. Differentiation measures instantaneous
rate of change; integration measures total accumulation. The Fundamental Theorem of Calculus
reveals these are inverse operations -- one of the most powerful ideas in all of mathematics.

## 1. Limits and Continuity

**Textbook sections**: Ch 2, Sections 2.1-2.5

**Key definitions**:
- **Limit**: lim(x->a) f(x) = L means for every epsilon > 0 there exists delta > 0 such that 0 < |x - a| < delta implies |f(x) - L| < epsilon
- **Continuity**: f is continuous at a if lim(x->a) f(x) = f(a)
- **One-sided limits**: lim(x->a+) f(x) and lim(x->a-) f(x)

**Key theorems**:
- **Squeeze Theorem**: If g(x) <= f(x) <= h(x) near a and lim g(x) = lim h(x) = L, then lim f(x) = L. *Intuition*: if f is trapped between two functions that converge to the same value, f must converge there too.
- **Intermediate Value Theorem (IVT)**: If f is continuous on [a,b] and N is between f(a) and f(b), then there exists c in (a,b) with f(c) = N. *Intuition*: a continuous function cannot "jump over" a value -- it must pass through every intermediate value.

**Worked example**:
> Evaluate lim(x->0) sin(x)/x. Since -1 <= sin(x)/x is not directly evaluable, use the Squeeze Theorem: for 0 < x < pi/2, cos(x) <= sin(x)/x <= 1. As x->0, cos(x)->1, so by the Squeeze Theorem, lim(x->0) sin(x)/x = 1.

**Essential problems**: OpenStax 2.2 Exercises: #65-72; 2.3 Exercises: #95-106; 2.4 Exercises: #131-140
**Challenge problems**: OpenStax 2.5 Exercises: #191-196

## 2. Derivatives

**Textbook sections**: Ch 3, Sections 3.1-3.9

**Key definitions**:
- **Derivative**: f'(x) = lim(h->0) [f(x+h) - f(x)] / h -- the instantaneous rate of change
- **Differentiable**: f is differentiable at a if f'(a) exists; differentiability implies continuity

**Key theorems / rules**:
- **Power Rule**: d/dx [x^n] = n*x^(n-1)
- **Product Rule**: (fg)' = f'g + fg'
- **Quotient Rule**: (f/g)' = (f'g - fg') / g^2
- **Chain Rule**: d/dx [f(g(x))] = f'(g(x)) * g'(x). *Intuition*: rates of change multiply through composition -- if gear A turns gear B which turns gear C, the total rate is the product of each individual rate.
- **Implicit Differentiation**: Differentiate both sides of an equation with respect to x, treating y as a function of x, then solve for dy/dx.

**Worked example**:
> Find dy/dx for x^2 + y^2 = 25. Differentiate: 2x + 2y(dy/dx) = 0, so dy/dx = -x/y. At (3,4): dy/dx = -3/4. This gives the slope of the tangent to the circle at that point.

**Essential problems**: OpenStax 3.2 Exercises: #55-68; 3.3 Exercises: #100-110; 3.4 Exercises: #140-155; 3.6 Exercises: #210-222
**Challenge problems**: OpenStax 3.8 Exercises: #308-315

## 3. Applications of Derivatives

**Textbook sections**: Ch 4, Sections 4.1-4.7

**Key definitions**:
- **Critical point**: c where f'(c) = 0 or f'(c) is undefined
- **Local extremum**: a local max/min at c identified by sign changes in f' (First Derivative Test) or by f''(c) (Second Derivative Test)
- **Inflection point**: where f'' changes sign (concavity changes)

**Key theorems**:
- **Mean Value Theorem (MVT)**: If f is continuous on [a,b] and differentiable on (a,b), there exists c in (a,b) with f'(c) = [f(b)-f(a)]/(b-a). *Intuition*: at some point, the instantaneous rate equals the average rate. A car averaging 60 mph must have been going exactly 60 mph at some instant.
- **L'Hopital's Rule**: If lim f(x)/g(x) gives 0/0 or inf/inf, then lim f(x)/g(x) = lim f'(x)/g'(x) (provided the latter exists). *Intuition*: when both numerator and denominator vanish, their rates of vanishing determine the limit.

**Worked example**:
> A farmer has 200m of fencing to enclose a rectangular area against a river (no fence needed on the river side). Maximize the area. Let x = width, then length = 200 - 2x. Area A(x) = x(200 - 2x) = 200x - 2x^2. A'(x) = 200 - 4x = 0 gives x = 50. A''(50) = -4 < 0, confirming maximum. Max area = 50 * 100 = 5000 m^2.

**Essential problems**: OpenStax 4.3 Exercises: #101-110; 4.5 Exercises: #198-210; 4.7 Exercises: #300-310
**Challenge problems**: OpenStax 4.7 Exercises: #316-320

## 4. Integration

**Textbook sections**: Ch 5, Sections 5.1-5.7

**Key definitions**:
- **Riemann sum**: Sum of f(x_i*) * Delta_x over a partition -- approximates area under the curve
- **Definite integral**: integral from a to b of f(x) dx = lim(n->inf) of the Riemann sum
- **Antiderivative**: F is an antiderivative of f if F'(x) = f(x)

**Key theorems**:
- **FTC Part 1**: If F(x) = integral from a to x of f(t) dt, then F'(x) = f(x). *Intuition*: the derivative of the accumulation function recovers the original function -- differentiation undoes integration.
- **FTC Part 2**: integral from a to b of f(x) dx = F(b) - F(a) where F is any antiderivative of f. *Intuition*: to compute total accumulation, just evaluate the antiderivative at the endpoints.
- **Substitution** (u-substitution): integral of f(g(x))g'(x) dx = integral of f(u) du where u = g(x). The chain rule in reverse.

**Worked example**:
> Evaluate integral from 0 to 2 of x * e^(x^2) dx. Let u = x^2, du = 2x dx, so x dx = du/2. Bounds: u(0)=0, u(2)=4. Integral becomes (1/2) integral from 0 to 4 of e^u du = (1/2)(e^4 - 1).

**Essential problems**: OpenStax 5.2 Exercises: #67-76; 5.3 Exercises: #148-160; 5.5 Exercises: #256-268
**Challenge problems**: OpenStax 5.7 Exercises: #340-348

## 5. Applications of Integration

**Textbook sections**: Ch 6, Sections 6.1-6.4

**Key definitions**:
- **Area between curves**: integral from a to b of [f(x) - g(x)] dx where f(x) >= g(x)
- **Disk method**: V = pi * integral from a to b of [R(x)]^2 dx (rotation about x-axis)
- **Shell method**: V = 2*pi * integral from a to b of x * f(x) dx (rotation about y-axis)

**Worked example**:
> Find the volume of the solid obtained by rotating y = sqrt(x) from x=0 to x=4 about the x-axis. By the disk method: V = pi * integral from 0 to 4 of (sqrt(x))^2 dx = pi * integral from 0 to 4 of x dx = pi * [x^2/2] from 0 to 4 = 8*pi.

**Essential problems**: OpenStax 6.1 Exercises: #1-12; 6.2 Exercises: #55-70; 6.3 Exercises: #105-115
**Challenge problems**: OpenStax 6.4 Exercises: #140-148

---

## Technique Catalog

| Technique | When to Use | Key Formula/Idea |
|-----------|-------------|------------------|
| Squeeze Theorem | Limit of oscillating/bounded function | Bound f between g and h with same limit |
| L'Hopital's Rule | 0/0 or inf/inf indeterminate forms | Differentiate numerator and denominator |
| Power Rule | Polynomial derivatives | d/dx [x^n] = nx^(n-1) |
| Chain Rule | Composite functions | Multiply derivatives through the composition |
| u-Substitution | Integrals of compositions | Reverse the chain rule |
| Disk/Shell Method | Volume of revolution | Choose based on axis of rotation vs. variable |
| Optimization | Find max/min of a quantity | Set f'=0, check endpoints and critical points |

## Connections to CS & Algorithms

| Math Concept | CS Application | Repo Link |
|-------------|----------------|-----------|
| Limits | Asymptotic analysis (Big-O, Big-Theta) | algorithms track |
| Derivatives | Gradient descent in machine learning | ML track |
| Optimization | Greedy algorithms, LP relaxations | algorithms track |
| Integration | Probability density functions | [07-probability-statistics](../07-probability-statistics/) |
| Rate of change | Amortized analysis of data structures | algorithms track |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Google | Optimization problems in interviews, understanding PageRank math | Medium |
| Quantitative Finance | Derivative pricing, rate-of-change models | Hard |
| ML/AI Startups | Gradient descent requires derivative intuition | Medium |
| Amazon | Optimization of logistics/scheduling functions | Medium |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `M01.1` | [Derivative as a limit, finite differences, step-size choice](01-derivatives-and-finite-differences.md) | build | 2 |
| 2 | `M01.2` | [Newton's method](02-newtons-method.md) | build | 2 |
| 3 | `M01.3` | [Activation functions and their derivatives](03-activation-functions.md) | build | 2 |
| 4 | `M01.4` | [Definite integrals, trapezoid, Simpson](04-definite-integrals-trapezoid-simpson.md) | build | 5 |
| 5 | `S-M01` | [Calculus 1 problem set: limits, derivatives, rates, optimization, integrals, L'Hôpital](90-problem-set.md) | solve | 2 |
<!-- /ol:chapters -->
