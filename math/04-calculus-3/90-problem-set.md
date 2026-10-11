<!-- ol:module S-M04 -->
# Calculus 3 problem set: vectors, gradients, the chain rule, Hessians, multiple integrals, Lagrange

## Overview

| | |
|---|---|
| **Module** | `S-M04` · solve · none · Pass 2 · 6 to 8 h |
| **You build** | answers in `solve/S-M04.toml` (56 checked by SymPy) and 2 proofs in `solve/S-M04/q34.md` and `q58.md` (self-graded against their rubrics) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M04/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M04/problems.md` and in section 4 |
| **Needs** | no module. Reading: `S-M01` (one-variable derivatives and integrals) and the [Calculus 3 topic](README.md) |
| **Used by** | no call site (a solve set). Take it after `M04.1` (gradcheck) and `M04.2` (Jacobians, VJP, JVP) in Pass 2. It is also the whole of the solve-only `M04.3` (Hessian), `M04.4` (multiple integrals), and `M04.5` (Lagrange), which `M10.1`, `M07.3`, and `M11.5` read |
| **Milestone** | `MS-P2` (the Pass 2 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | OpenStax, *Calculus Volume 3* (free), ch. 2, 4, and 5; Boyd and Vandenberghe, *Convex Optimization*, section 5.1, for Lagrange duality |

## Key Takeaways

- The gradient collects the partial derivatives; it points uphill fastest, and its length is the steepest slope (q11, q30, q34).
- The multivariable chain rule multiplies Jacobians, and reverse mode multiplies them from the loss backward, one vector-Jacobian product at a time (q23, q27, q28).
- At a critical point the Hessian's eigenvalues decide minimum, maximum, or saddle, and their ratio is the condition number that slows gradient descent (q37, q42).
- A change of variables multiplies the area element by $|\det J|$: the $r$ in polar coordinates is why $\iint e^{-(x^2 + y^2)} = \pi$ (q26, q46, q50).
- At a constrained optimum the gradient is parallel to the constraint's gradient; maximum entropy under an energy constraint is a softmax (q51 to q58).

## How to work this chapter

```bash
ol start S-M04             # writes solve/S-M04.toml and the two proof files
ol check S-M04             # SymPy checks the answers, then asks each proof rubric (y/n)
ol check S-M04 --regrade   # ask the rubrics again after you change a proof
```

---

## 1. Why now

Your model has many parameters, and training moves all of them at once along the gradient. `M04.1` builds `gradcheck`, which compares your analytic gradients with central differences coordinate by coordinate, and `M04.2` computes Jacobians and the vector-Jacobian products that reverse-mode autodiff (`M08.2`, `L0.1`) chains together. `M10.1` then descends along gradients, and its convergence rate is set by the Hessian's condition number. `M07.3` sizes initial weights with Gaussian integrals in several variables, and `M11.5` reads the softmax as the solution of a constrained maximization. This set checks the hand computations behind all of them.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $u \cdot v$, $u \times v$ | dot product; cross product (in $\mathbb{R}^3$) | scalar; vector |
| $\lVert v \rVert$ | Euclidean length, $\sqrt{v \cdot v}$ | real |
| $\partial f / \partial x_i$ | partial derivative: vary $x_i$, hold the others fixed | function |
| $\nabla f$ | gradient, the vector of partials | vector |
| $D_u f$ | directional derivative along a unit vector $u$, $\nabla f \cdot u$ | real |
| $J_g$ | Jacobian of $g: \mathbb{R}^n \to \mathbb{R}^m$, $(J_g)_{ij} = \partial g_i / \partial x_j$ | $m \times n$ |
| $H_f$ | Hessian, $(H_f)_{ij} = \partial^2 f / \partial x_i \partial x_j$ | $n \times n$, symmetric |
| $\kappa$ | condition number of a positive definite $H$, $\lambda_{\max}/\lambda_{\min}$ | real |
| $\iint_D f\,dA$ | double integral over a region $D$ | real |
| $\lambda$ | a Lagrange multiplier | real |

### 2.1 Vectors and geometry

$u \cdot v = \sum_i u_i v_i = \lVert u \rVert \lVert v \rVert \cos\theta$, so $u \perp v$ exactly when $u \cdot v = 0$. The cross product $u \times v$ is perpendicular to both, with length the area of the parallelogram they span. A plane is $n \cdot x = d$ with normal $n$ (a cross product of two edge vectors), and the distance from a point $p$ to it is $|n \cdot p - d| / \lVert n \rVert$.

### 2.2 Partial derivatives and the gradient

A partial derivative differentiates in one variable with the rest held fixed, by the one-variable rules. Mixed second partials agree for smooth functions: $\partial^2 f/\partial x\,\partial y = \partial^2 f/\partial y\,\partial x$. Two gradients recur in machine learning: $\nabla_x \frac12 \lVert Ax - b \rVert^2 = A^\top(Ax - b)$, and the gradient of log-sum-exp, $\nabla \ln \sum_j e^{x_j}$, is the softmax of $x$.

### 2.3 The multivariable chain rule

If $z = f(x(t), y(t))$, then $\frac{dz}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}$: one term per path from $t$ to $z$. In general the Jacobian of a composite is the product of Jacobians, $J_{f \circ g} = J_f\, J_g$. A loss is a scalar at the end of a long chain, and **reverse mode** evaluates the product from the left: start with $u^\top = \partial L / \partial(\text{output})$ and multiply by one Jacobian at a time, $u^\top \leftarrow u^\top J$. Each step is a **vector-Jacobian product** (VJP), and backpropagation is nothing more.

### 2.4 Directional derivatives

The rate of change of $f$ at $x$ along a unit vector $u$ is $D_u f = \frac{d}{ds} f(x + s u)\big|_{s=0} = \nabla f \cdot u$, by the chain rule. By Cauchy-Schwarz it is at most $\lVert \nabla f \rVert$, reached at $u = \nabla f / \lVert \nabla f \rVert$: the gradient is the direction of steepest ascent, $-\nabla f$ of steepest descent, and directions perpendicular to $\nabla f$ follow a level curve.

### 2.5 The Hessian and the second-derivative test

At a **critical point** ($\nabla f = 0$), the second-order Taylor expansion is $f(x + h) \approx f(x) + \frac12 h^\top H h$. If every eigenvalue of $H$ is positive (positive definite), the point is a local minimum; all negative, a maximum; mixed signs, a saddle. For two variables: $\det H > 0$ and $f_{xx} > 0$ is a minimum, $\det H < 0$ is a saddle. For the quadratic $\frac12 x^\top H x$, gradient descent's error shrinks by about $1 - 1/\kappa$ per step with $\kappa = \lambda_{\max}/\lambda_{\min}$ (`M10.1`).

### 2.6 Multiple integrals and change of variables

A double integral over a rectangle is an iterated integral in either order; over other regions, the limits of the inner integral depend on the outer variable. Changing variables $(u, v) \mapsto (x, y)$ multiplies the area element by the absolute Jacobian determinant: $dx\,dy = |\det J|\,du\,dv$. Polar coordinates have $|\det J| = r$, so $dA = r\,dr\,d\theta$.

### 2.7 Lagrange multipliers

To optimize $f$ subject to $g = 0$, look for points where $\nabla f = \lambda \nabla g$ and $g = 0$: at a constrained optimum, moving along the constraint surface cannot change $f$ to first order, so $\nabla f$ has no component along the surface. With several constraints, $\nabla f = \sum_k \lambda_k \nabla g_k$. Solve the system, then compare the candidates' values.

## 3. Worked example by hand

This is a sibling of q27 and q52, not one of the graded problems.

**A backward pass by hand.** $h = 2x - 1$, $y = \sigma(h)$, $L = -\ln y$ (the cross-entropy of one positive example). At $x = 1/2$: forward, $h = 0$, $y = \sigma(0) = 1/2$, $L = \ln 2$. Backward, one factor at a time: $\frac{dL}{dy} = -\frac{1}{y} = -2$; $\frac{dy}{dh} = \sigma(h)(1 - \sigma(h)) = \frac14$; $\frac{dh}{dx} = 2$. So $\frac{dL}{dx} = (-2)\cdot\frac14\cdot 2 = -1$. The middle two factors combine to $\frac{dL}{dh} = -(1 - y) = y - 1 = -\frac12$, the familiar "prediction minus label" of a logistic output. In `solve/` the answer would be `answer = "-1"`.

**A constrained minimum.** Minimize $x^2 + 2y^2$ subject to $x + y = 3$. Lagrange: $\nabla f = (2x, 4y) = \lambda (1, 1)$, so $x = \lambda/2$, $y = \lambda/4$, and $x + y = 3\lambda/4 = 3$ gives $\lambda = 4$, $(x, y) = (2, 1)$, value $4 + 2 = 6$. Check by substitution: $f = x^2 + 2(3 - x)^2$, $f' = 2x - 4(3 - x) = 6x - 12 = 0$ at $x = 2$.

## 4. The problem set

Write each answer in `solve/S-M04.toml`:

```toml
[q5]
answer = "x + y + z = 1"
[q11]
answer = "[8, 7]"
[q35]
answer = "[[2, 3], [3, 4]]"
[q37]
answer = "saddle"
[q34]
proof = "S-M04/q34.md"
```

Vectors are flat lists, matrices lists of rows, and entries are exact (`-1/sqrt(5)`). For q33 either sign of the direction passes.

<!-- ol:problems S-M04 -->

### Vectors and geometry

**q1.** $(1, 2, 3) \cdot (4, -5, 6)$. `[number]`

**q2.** $(1, 2, 3) \times (4, 5, 6)$. `[vector]`

**q3.** Give the angle between $(1, 0)$ and $(1, 1)$, in radians. `[number]`

**q4.** $\lVert (2, 3, 6) \rVert$. `[number]`

**q5.** Give the equation of the plane through $(1, 0, 0)$, $(0, 1, 0)$, and $(0, 0, 1)$. `[equation in x, y, z]`

**q6.** Give the distance from the point $(1, 2, 3)$ to the plane $x + y + z = 0$. `[number]`

**q7.** Give the area of the triangle with vertices $(1, 0, 0)$, $(0, 1, 0)$, $(0, 0, 1)$. `[number]`

**q8.** Are $(1, 2, -1)$ and $(3, -1, 1)$ orthogonal? `[bool]`

### Partial derivatives and the gradient

**q9.** $f(x, y) = x^2 y + \sin(xy)$. Give $\partial f / \partial x$. `[expr in x, y]`

**q10.** Give $\partial f / \partial y$ for the $f$ of q9. `[expr in x, y]`

**q11.** $f(x, y) = x^2 + 3xy + y^2$. Give $\nabla f(1, 2)$. `[vector]`

**q12.** $f(a, b, c) = a^2 + b^2 + c^2$. Give $\nabla f$. `[vector]`

**q13.** $f(x, y) = 3x - 2y + 7$. Give $\nabla f$. `[vector]`

**q14.** $\dfrac{\partial^2}{\partial x\, \partial y}\, x^3 y^2$. `[expr in x, y]`

**q15.** $f(x, y) = \ln(e^x + e^y)$ (log-sum-exp). Give $\nabla f$. `[vector]`

**q16.** $f(x) = \frac{1}{2}\lVert A x - b \rVert^2$ with $A = \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}$ and $b = (1, 1)$. Give $\nabla f(0, 0)$. `[vector]`

**q17.** $f(x, y) = x e^y$. Give $\partial f / \partial y$ at $(2, 0)$. `[number]`

**q18.** The squared error of a one-weight model is $L(w) = (w x - y)^2$. Give $\partial L / \partial w$. `[expr in w, x, y]`

### The multivariable chain rule

**q19.** $z = x^2 + y^2$ with $x = \cos t$, $y = \sin t$. Give $dz/dt$. `[number]`

**q20.** $z = xy$ with $x = t^2$, $y = t^3$. Give $dz/dt$. `[expr in t]`

**q21.** $f(u, v) = uv$ with $u = x + y$, $v = x - y$. Give $\partial f / \partial x$. `[expr in x, y]`

**q22.** Give $\partial f / \partial y$ for the $f$ of q21. `[expr in x, y]`

**q23.** $L(w) = (\sigma(w x) - 1)^2$ with the sigmoid $\sigma$. Give $dL/dw$ at $w = 0$, $x = 2$. `[number]`

**q24.** $f(x, y) = x^2 + y^2$ with $x = r\cos\theta$, $y = r\sin\theta$. Give $\partial f / \partial r$. `[expr in r]`

**q25.** Give the Jacobian matrix of $(x, y) \mapsto (x + y,\ xy)$ at $(1, 2)$. `[matrix]`

**q26.** Give the determinant of the Jacobian of $(r, \theta) \mapsto (r\cos\theta,\ r\sin\theta)$. `[expr in r]`

**q27.** $h = 3x + 1$, $y = h^2$, $L = (y - 4)^2$. Give $dL/dx$ at $x = 0$ (backpropagate one factor at a time). `[number]`

**q28.** $f(x) = A x$ with $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$. Give the vector-Jacobian product $u^\top J_f$ for $u = (1, 1)$, as a column. `[vector]`

### Directional derivatives

**q29.** $f(x, y) = x^2 + y^2$. Give the directional derivative at $(1, 1)$ in the direction $u = (3/5, 4/5)$. `[number]`

**q30.** $f(x, y) = xy$. Give the largest directional derivative at $(2, 3)$ over all unit directions. `[number]`

**q31.** $f(x, y) = x^2 + 2y^2$. Give the unit direction of steepest descent at $(1, 1)$. `[vector]`

**q32.** $f(x, y) = e^x \sin y$. Give the directional derivative at $(0, \pi/2)$ in the direction $(1, 0)$. `[number]`

**q33.** $f(x, y) = x^2 - y^2$. Give a unit direction at $(1, 1)$ along which the directional derivative is 0 (either sign). `[vector]`

**q34.** Prove that at a point where $\nabla f \ne 0$, the unit direction $u$ that maximizes the directional derivative $D_u f = \nabla f \cdot u$ is $u = \nabla f / \lVert \nabla f \rVert$, and that the maximum is $\lVert \nabla f \rVert$. `[proof]`

### The Hessian and the second-derivative test

**q35.** Give the Hessian of $f(x, y) = x^2 + 3xy + 2y^2$. `[matrix]`

**q36.** Give the Hessian of $f(x, y) = x^3 + y^3$ at $(1, 1)$. `[matrix]`

**q37.** Classify the critical point $(0, 0)$ of $f(x, y) = x^2 - y^2$: `minimum`, `maximum`, or `saddle`. `[choice]`

**q38.** Classify the critical point $(0, 0)$ of $f(x, y) = x^2 + xy + y^2$: `minimum`, `maximum`, or `saddle`. `[choice]`

**q39.** Give the critical point of $f(x, y) = x^2 + y^2 - 2x + 4y$. `[vector]`

**q40.** Classify the critical point $(1, 0)$ of $f(x, y) = x^3 - 3x + y^2$: `minimum`, `maximum`, or `saddle`. `[choice]`

**q41.** Is the Hessian of q35 positive definite? `[bool]`

**q42.** $f(x) = \frac{1}{2} x^\top H x$ with $H = \operatorname{diag}(1, 100)$. Give the condition number $\lambda_{\max} / \lambda_{\min}$ of its Hessian. `[number]`

### Multiple integrals and change of variables

**q43.** $\displaystyle\int_0^1 \int_0^2 x y\, dy\, dx$. `[number]`

**q44.** Give the area of the triangle $0 \le y \le x \le 1$ as a double integral. `[number]`

**q45.** $\displaystyle\iint_{x^2 + y^2 \le 1} 1\, dA$. `[number]`

**q46.** $\displaystyle\iint_{\mathbb{R}^2} e^{-(x^2 + y^2)}\, dA$. `[number]`

**q47.** $\displaystyle\int_0^1 \int_0^x 2y\, dy\, dx$. `[number]`

**q48.** Give the volume under $z = 1 - x^2 - y^2$ and above the unit disk $x^2 + y^2 \le 1$. `[number]`

**q49.** $\displaystyle\iint_{\mathbb{R}^2} \frac{1}{2\pi} e^{-(x^2 + y^2)/2}\, dA$, the total mass of a 2-D standard normal. `[number]`

**q50.** With $u = x + y$, $v = x - y$, give the area of the region $|x + y| \le 1$, $|x - y| \le 1$. `[number]`

### Lagrange multipliers

**q51.** Give the maximum of $x + y$ subject to $x^2 + y^2 = 1$. `[number]`

**q52.** Give the minimum of $x^2 + y^2$ subject to $x + y = 1$. `[number]`

**q53.** Give the maximum of $xy$ subject to $x + y = 10$. `[number]`

**q54.** Maximize the entropy $-\sum_{i=1}^{3} p_i \ln p_i$ subject to $p_1 + p_2 + p_3 = 1$. Give the optimal $p_1$. `[number]`

**q55.** Give the maximum of $xyz$ subject to $x + y + z = 3$ with $x, y, z > 0$. `[number]`

**q56.** Give the minimum of $2x + 3y$ subject to $xy = 6$ with $x, y > 0$. `[number]`

**q57.** Give the point of the plane $x + 2y + 2z = 9$ closest to the origin. `[vector]`

**q58.** With Lagrange multipliers, show that the distribution $p$ over states $1, \dots, n$ with energies $E_i$ that maximizes the entropy $-\sum_i p_i \ln p_i$ subject to $\sum_i p_i = 1$ and $\sum_i p_i E_i = U$ has the form $p_i = e^{-\beta E_i} / \sum_j e^{-\beta E_j}$ (a softmax of $-\beta E$). `[proof]`

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Dropping an inner derivative in a partial | $\partial_x \sin(xy)$ given as $\cos(xy)$; gradcheck fails on every product inside a nonlinearity | q9, q10, q18 (canaries) |
| Forgetting the $A^\top$ in a least-squares gradient | descent along the wrong direction for any non-identity $A$ | q16 (canary) |
| Multiplying the derivatives of the parts instead of summing over paths | $dz/dt$ for $z = xy$ given as $x'(t)\,y'(t)$ | q20 (canary 6*t^3) |
| Skipping a factor in a chain of local derivatives | a backward pass off by the sigmoid's slope or a weight | q23, q27 (canaries) |
| Computing $J u$ when the backward pass needs $u^\top J$ | the gradient of a non-symmetric layer comes out transposed | q28 (canary) |
| Using an unnormalized direction | a directional derivative scaled by the direction's length | q29 (canary 4), q31 (canary) |
| Halving the mixed term of a Hessian | a saddle misread as a minimum | q35 (canary) |
| Forgetting the Jacobian factor in a change of variables | polar integrals off by the missing $r$ | q45 (canary 2*pi), q50 (canary 4) |
| Reporting the optimizer instead of the optimum | the argmax given as the max | q52, q53 (canaries) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M01` | one-variable derivatives, the chain rule, and integrals |
| Forward | `M04.1` | `gradcheck` compares your partials (q9 to q18) with central differences |
| Forward | `M04.2` | `jacobian`, `vjp_numeric`, `jvp_numeric`: the products of q25 and q28 |
| Forward | `M08.2` | reverse-mode autodiff is the chain of VJPs of q27 |
| Forward | `M10.1` | gradient descent, Armijo steps, and the condition number of q42 |
| Forward | `M07.3` | Gaussian integrals in two variables (q46, q49) behind initialization variances |
| Forward | `M11.5` | softmax as maximum entropy under an energy constraint (q58) |
