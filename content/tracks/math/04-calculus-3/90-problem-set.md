# Calculus 3 problem set: vectors, gradients, the chain rule, Hessians, multiple integrals, Lagrange

## Overview

| | |
|---|---|
| **Module** | `S-M04` · solve · none · Pass 2 · 6 to 8 h |
| **Solution** | a worked answer after every problem in section 4: 56 exact answers and 2 proofs (q34, q58; model proofs also in [`course/solutions/solve/S-M04/`](../../../../course/solutions/solve/S-M04/)) |
| **Contract** | none: a pen and paper set |
| **Needs** | no module. Reading: `S-M01` (one-variable derivatives and integrals) and the [Calculus 3 topic](README.md) |
| **Used by** | no call site (a solve set). Take it after `M04.1` (gradcheck) and `M04.2` (Jacobians, VJP, JVP) in Pass 2. It is also the whole of the solve-only `M04.3` (Hessian), `M04.4` (multiple integrals), and `M04.5` (Lagrange), which `M10.1`, `M07.3`, and `M11.5` read |
| **Optional depth** | OpenStax, *Calculus Volume 3* (free), ch. 2, 4, and 5; Boyd and Vandenberghe, *Convex Optimization*, section 5.1, for Lagrange duality |

## Key Takeaways

- The gradient collects the partial derivatives; it points uphill fastest, and its length is the steepest slope (q11, q30, q34).
- The multivariable chain rule multiplies Jacobians, and reverse mode multiplies them from the loss backward, one vector-Jacobian product at a time (q23, q27, q28).
- At a critical point the Hessian's eigenvalues decide minimum, maximum, or saddle, and their ratio is the condition number that slows gradient descent (q37, q42).
- A change of variables multiplies the area element by $|\det J|$: the $r$ in polar coordinates is why $\iint e^{-(x^2 + y^2)} = \pi$ (q26, q46, q50).
- At a constrained optimum the gradient is parallel to the constraint's gradient; maximum entropy under an energy constraint is a softmax (q51 to q58).

## How to read this chapter

Sections 1 to 3 review the ideas and work two examples by hand. Section 4 is the problem set: every problem is followed by a collapsed worked answer with the final result, the derivation, and the common wrong answers and why they are wrong. Try each problem before opening its answer.

---

## 1. Why now

Your model has many parameters, and training moves all of them at once along the gradient. `M04.1` builds `gradcheck`, which compares analytic gradients with central differences coordinate by coordinate, and `M04.2` computes Jacobians and the vector-Jacobian products that reverse-mode autodiff (`M08.2`, `L0.1`) chains together. `M10.1` then descends along gradients, and its convergence rate is set by the Hessian's condition number. `M07.3` sizes initial weights with Gaussian integrals in several variables, and `M11.5` reads the softmax as the solution of a constrained maximization. This set works the hand computations behind all of them.

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

This is a sibling of q27 and q52, not one of the problems of the set.

**A backward pass by hand.** $h = 2x - 1$, $y = \sigma(h)$, $L = -\ln y$ (the cross-entropy of one positive example). At $x = 1/2$: forward, $h = 0$, $y = \sigma(0) = 1/2$, $L = \ln 2$. Backward, one factor at a time: $\frac{dL}{dy} = -\frac{1}{y} = -2$; $\frac{dy}{dh} = \sigma(h)(1 - \sigma(h)) = \frac14$; $\frac{dh}{dx} = 2$. So $\frac{dL}{dx} = (-2)\cdot\frac14\cdot 2 = -1$. The middle two factors combine to $\frac{dL}{dh} = -(1 - y) = y - 1 = -\frac12$, the familiar "prediction minus label" of a logistic output.

**A constrained minimum.** Minimize $x^2 + 2y^2$ subject to $x + y = 3$. Lagrange: $\nabla f = (2x, 4y) = \lambda (1, 1)$, so $x = \lambda/2$, $y = \lambda/4$, and $x + y = 3\lambda/4 = 3$ gives $\lambda = 4$, $(x, y) = (2, 1)$, value $4 + 2 = 6$. Check by substitution: $f = x^2 + 2(3 - x)^2$, $f' = 2x - 4(3 - x) = 6x - 12 = 0$ at $x = 2$.

## 4. The problem set, with worked answers

Answers are exact (for example $-1/\sqrt5$, not $-0.447$). Vectors are written as tuples and matrices row by row. The tag after each problem says what kind of answer is expected. For q33 either sign of the direction is correct.

### Vectors and geometry

**q1.** $(1, 2, 3) \cdot (4, -5, 6)$. `[number]`

<details><summary>Answer</summary>

**$12$.** Multiply entry by entry and add: $1 \cdot 4 + 2 \cdot (-5) + 3 \cdot 6 = 4 - 10 + 18 = 12$. A common mistake is 32, which drops the sign of $-5$.

</details>

**q2.** $(1, 2, 3) \times (4, 5, 6)$. `[vector]`

<details><summary>Answer</summary>

**$(-3, 6, -3)$.** $u \times v = (u_2 v_3 - u_3 v_2,\ u_3 v_1 - u_1 v_3,\ u_1 v_2 - u_2 v_1) = (12 - 15,\ 12 - 6,\ 5 - 8)$. Check: $(-3, 6, -3) \cdot (1, 2, 3) = -3 + 12 - 9 = 0$, so it is perpendicular to $u$. Answering $(3, -6, 3)$ computes $v \times u$: the cross product is antisymmetric, so the order matters.

</details>

**q3.** Give the angle between $(1, 0)$ and $(1, 1)$, in radians. `[number]`

<details><summary>Answer</summary>

**$\pi/4$.** $\cos\theta = \frac{u \cdot v}{\lVert u \rVert \lVert v \rVert} = \frac{1}{1 \cdot \sqrt2}$, so $\theta = \arccos(1/\sqrt2) = \pi/4$. A common mistake is to stop at $1/\sqrt2$, which is the cosine, not the angle.

</details>

**q4.** $\lVert (2, 3, 6) \rVert$. `[number]`

<details><summary>Answer</summary>

**$7$.** $\sqrt{4 + 9 + 36} = \sqrt{49} = 7$. Answering 49 forgets the square root: that is $v \cdot v$.

</details>

**q5.** Give the equation of the plane through $(1, 0, 0)$, $(0, 1, 0)$, and $(0, 0, 1)$. `[equation in x, y, z]`

<details><summary>Answer</summary>

**$x + y + z = 1$.** Two edge vectors from $(1, 0, 0)$ are $(-1, 1, 0)$ and $(-1, 0, 1)$; their cross product is the normal $n = (1, 1, 1)$. Then $d = n \cdot (1, 0, 0) = 1$. Any nonzero multiple, such as $2x + 2y + 2z = 2$, is the same plane. $x + y + z = 0$ is the parallel plane through the origin, which contains none of the three points.

</details>

**q6.** Give the distance from the point $(1, 2, 3)$ to the plane $x + y + z = 0$. `[number]`

<details><summary>Answer</summary>

**$2\sqrt3$.** $\frac{|n \cdot p - d|}{\lVert n \rVert} = \frac{|1 + 2 + 3 - 0|}{\sqrt3} = \frac{6}{\sqrt3} = 2\sqrt3$. A common mistake is 6, which forgets to divide by the normal's length: $n \cdot p - d$ measures distance in units of $\lVert n \rVert$.

</details>

**q7.** Give the area of the triangle with vertices $(1, 0, 0)$, $(0, 1, 0)$, $(0, 0, 1)$. `[number]`

<details><summary>Answer</summary>

**$\sqrt3/2$.** With the edges of q5, $\lVert (-1, 1, 0) \times (-1, 0, 1) \rVert = \lVert (1, 1, 1) \rVert = \sqrt3$ is the area of the parallelogram; the triangle is half of it. Answering $\sqrt3$ reports the parallelogram.

</details>

**q8.** Are $(1, 2, -1)$ and $(3, -1, 1)$ orthogonal? `[bool]`

<details><summary>Answer</summary>

**True.** $(1, 2, -1) \cdot (3, -1, 1) = 3 - 2 - 1 = 0$, and a zero dot product means orthogonal.

</details>

### Partial derivatives and the gradient

**q9.** $f(x, y) = x^2 y + \sin(xy)$. Give $\partial f / \partial x$. `[expr in x, y]`

<details><summary>Answer</summary>

**$2xy + y\cos(xy)$.** Hold $y$ fixed. $\partial_x(x^2 y) = 2xy$; for $\sin(xy)$ the chain rule gives $\cos(xy)$ times the inner derivative $\partial_x(xy) = y$. A common mistake is $2xy + \cos(xy)$, which drops the inner factor $y$.

</details>

**q10.** Give $\partial f / \partial y$ for the $f$ of q9. `[expr in x, y]`

<details><summary>Answer</summary>

**$x^2 + x\cos(xy)$.** Hold $x$ fixed: $\partial_y(x^2 y) = x^2$, and $\partial_y \sin(xy) = \cos(xy) \cdot x$. Dropping the inner $x$ gives the wrong $x^2 + \cos(xy)$.

</details>

**q11.** $f(x, y) = x^2 + 3xy + y^2$. Give $\nabla f(1, 2)$. `[vector]`

<details><summary>Answer</summary>

**$(8, 7)$.** $\nabla f = (2x + 3y,\ 3x + 2y)$; at $(1, 2)$ that is $(2 + 6,\ 3 + 4)$. The order matters: $(7, 8)$ swaps the components.

</details>

**q12.** $f(a, b, c) = a^2 + b^2 + c^2$. Give $\nabla f$. `[vector]`

<details><summary>Answer</summary>

**$(2a, 2b, 2c)$.** Each partial differentiates one square: $\partial_a a^2 = 2a$, and the other terms are constants in $a$. Answering $(a, b, c)$ forgets the factor 2 from the power rule.

</details>

**q13.** $f(x, y) = 3x - 2y + 7$. Give $\nabla f$. `[vector]`

<details><summary>Answer</summary>

**$(3, -2)$.** A linear function has a constant gradient, its coefficient vector; the constant 7 has derivative 0 and is not a component. The gradient of a function of two variables has two entries, so $(3, -2, 7)$ is wrong.

</details>

**q14.** $\dfrac{\partial^2}{\partial x\, \partial y}\, x^3 y^2$. `[expr in x, y]`

<details><summary>Answer</summary>

**$6x^2 y$.** $\partial_y(x^3 y^2) = 2x^3 y$, then $\partial_x(2x^3 y) = 6x^2 y$ (the other order gives the same, as it must for a smooth function). A common mistake is $3x^2 y^2$, which takes only the $x$ derivative.

</details>

**q15.** $f(x, y) = \ln(e^x + e^y)$ (log-sum-exp). Give $\nabla f$. `[vector]`

<details><summary>Answer</summary>

**$\left(\frac{e^x}{e^x + e^y},\ \frac{e^y}{e^x + e^y}\right)$.** $\partial_x \ln(e^x + e^y) = \frac{1}{e^x + e^y} \cdot e^x$ by the chain rule, and the same for $y$. This is the softmax of $(x, y)$, and it is why the backward pass of cross-entropy (`L0.3`) starts from the softmax. Two common mistakes: $(e^x, e^y)$ drops the outer $1/(\cdot)$, and $\left(\frac{1}{e^x + e^y}, \frac{1}{e^x + e^y}\right)$ drops the inner derivative.

</details>

**q16.** $f(x) = \frac{1}{2}\lVert A x - b \rVert^2$ with $A = \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}$ and $b = (1, 1)$. Give $\nabla f(0, 0)$. `[vector]`

<details><summary>Answer</summary>

**$(-1, -2)$.** $\nabla f = A^\top(Ax - b)$; at $x = 0$, $Ax - b = (-1, -1)$ and $A^\top(-1, -1) = (-1, -2)$. This is the direction gradient descent (`M10.1`) follows. A common mistake is $(-1, -1)$, which forgets $A^\top$: the chain rule through $Ax$ contributes the transpose. $(1, 2)$ has the sign of $Ax - b$ flipped.

</details>

**q17.** $f(x, y) = x e^y$. Give $\partial f / \partial y$ at $(2, 0)$. `[number]`

<details><summary>Answer</summary>

**$2$.** $\partial_y(x e^y) = x e^y$, and at $(2, 0)$ that is $2 \cdot 1$. Answering 1 evaluates $e^y$ and forgets the factor $x$.

</details>

**q18.** The squared error of a one-weight model is $L(w) = (w x - y)^2$. Give $\partial L / \partial w$. `[expr in w, x, y]`

<details><summary>Answer</summary>

**$2x(wx - y)$.** The outer square gives $2(wx - y)$, and the inner derivative $\partial_w(wx - y) = x$ multiplies it. Dropping $x$ gives $2(wx - y)$, the gradient with respect to the prediction, not the weight.

</details>

### The multivariable chain rule

**q19.** $z = x^2 + y^2$ with $x = \cos t$, $y = \sin t$. Give $dz/dt$. `[number]`

<details><summary>Answer</summary>

**$0$.** One term per path: $\frac{dz}{dt} = 2x \cdot (-\sin t) + 2y \cdot \cos t = -2\cos t \sin t + 2\sin t \cos t = 0$. Directly, $z = \cos^2 t + \sin^2 t = 1$ is constant.

</details>

**q20.** $z = xy$ with $x = t^2$, $y = t^3$. Give $dz/dt$. `[expr in t]`

<details><summary>Answer</summary>

**$5t^4$.** $\frac{dz}{dt} = y \cdot x'(t) + x \cdot y'(t) = t^3 \cdot 2t + t^2 \cdot 3t^2 = 5t^4$; directly, $z = t^5$. A common mistake is $6t^3 = (2t)(3t^2)$, which multiplies the two inner derivatives instead of summing one term per path.

</details>

**q21.** $f(u, v) = uv$ with $u = x + y$, $v = x - y$. Give $\partial f / \partial x$. `[expr in x, y]`

<details><summary>Answer</summary>

**$2x$.** $\frac{\partial f}{\partial x} = v \cdot \frac{\partial u}{\partial x} + u \cdot \frac{\partial v}{\partial x} = (x - y) + (x + y) = 2x$. Check: $f = (x + y)(x - y) = x^2 - y^2$.

</details>

**q22.** Give $\partial f / \partial y$ for the $f$ of q21. `[expr in x, y]`

<details><summary>Answer</summary>

**$-2y$.** $\frac{\partial f}{\partial y} = v \cdot 1 + u \cdot (-1) = (x - y) - (x + y) = -2y$. The sign comes from $\partial v / \partial y = -1$; dropping it gives $2y$.

</details>

**q23.** $L(w) = (\sigma(w x) - 1)^2$ with the sigmoid $\sigma$. Give $dL/dw$ at $w = 0$, $x = 2$. `[number]`

<details><summary>Answer</summary>

**$-1/2$.** With $z = wx$: at $w = 0$, $z = 0$ and $\sigma(0) = 1/2$. Three factors: $\frac{dL}{d\sigma} = 2(\sigma - 1) = -1$, $\frac{d\sigma}{dz} = \sigma(1 - \sigma) = \frac14$, $\frac{dz}{dw} = x = 2$. Product: $-1 \cdot \frac14 \cdot 2 = -\frac12$. Common mistakes: $-1/4$ drops the factor $x$, and $-2$ drops the sigmoid's slope $1/4$.

</details>

**q24.** $f(x, y) = x^2 + y^2$ with $x = r\cos\theta$, $y = r\sin\theta$. Give $\partial f / \partial r$. `[expr in r]`

<details><summary>Answer</summary>

**$2r$.** $\frac{\partial f}{\partial r} = 2x\cos\theta + 2y\sin\theta = 2r\cos^2\theta + 2r\sin^2\theta = 2r$; directly, $f = r^2$. $2r^2$ is not a derivative of anything here.

</details>

**q25.** Give the Jacobian matrix of $(x, y) \mapsto (x + y,\ xy)$ at $(1, 2)$. `[matrix]`

<details><summary>Answer</summary>

**$\begin{pmatrix} 1 & 1 \\ 2 & 1 \end{pmatrix}$.** Row $i$ is the gradient of output $i$: $\nabla(x + y) = (1, 1)$ and $\nabla(xy) = (y, x) = (2, 1)$ at $(1, 2)$. The transpose $\begin{pmatrix} 1 & 2 \\ 1 & 1 \end{pmatrix}$ puts inputs in rows, against the convention $(J)_{ij} = \partial g_i / \partial x_j$.

</details>

**q26.** Give the determinant of the Jacobian of $(r, \theta) \mapsto (r\cos\theta,\ r\sin\theta)$. `[expr in r]`

<details><summary>Answer</summary>

**$r$.** $J = \begin{pmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{pmatrix}$, so $\det J = r\cos^2\theta + r\sin^2\theta = r$. This $r$ is the extra factor in $dA = r\,dr\,d\theta$ (q45, q46). Answering 1 treats polar coordinates as if they preserved area.

</details>

**q27.** $h = 3x + 1$, $y = h^2$, $L = (y - 4)^2$. Give $dL/dx$ at $x = 0$ (backpropagate one factor at a time). `[number]`

<details><summary>Answer</summary>

**$-36$.** Forward at $x = 0$: $h = 1$, $y = 1$, $L = 9$. Backward, one factor at a time: $\frac{dL}{dy} = 2(y - 4) = -6$; $\frac{dy}{dh} = 2h = 2$, so $\frac{dL}{dh} = -12$; $\frac{dh}{dx} = 3$, so $\frac{dL}{dx} = -36$. Common mistakes: $-12$ is $dL/dh$, which stops before the last factor $dh/dx = 3$, and $36$ loses the sign of $y - 4$.

</details>

**q28.** $f(x) = A x$ with $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$. Give the vector-Jacobian product $u^\top J_f$ for $u = (1, 1)$, as a column. `[vector]`

<details><summary>Answer</summary>

**$(4, 6)$.** $J_f = A$, and $u^\top A = (1 \cdot 1 + 1 \cdot 3,\ 1 \cdot 2 + 1 \cdot 4) = (4, 6)$: the column sums of $A$ weighted by $u$. This is the product `M04.2`'s `vjp_numeric` and every reverse-mode backward compute. A common mistake is $(3, 7) = Au$, the row sums: that is a JVP, and for a non-symmetric $A$ it is a different vector.

</details>

### Directional derivatives

**q29.** $f(x, y) = x^2 + y^2$. Give the directional derivative at $(1, 1)$ in the direction $u = (3/5, 4/5)$. `[number]`

<details><summary>Answer</summary>

**$14/5$.** $\nabla f(1, 1) = (2, 2)$, and $(2, 2) \cdot (3/5, 4/5) = 6/5 + 8/5 = 14/5$. A common mistake is 4, which dots the gradient with $(1, 1)$, a vector of length $\sqrt2$ that is not the given direction.

</details>

**q30.** $f(x, y) = xy$. Give the largest directional derivative at $(2, 3)$ over all unit directions. `[number]`

<details><summary>Answer</summary>

**$\sqrt{13}$.** The largest directional derivative is $\lVert \nabla f \rVert$ (q34). $\nabla f = (y, x) = (3, 2)$, with length $\sqrt{9 + 4}$. Answering 5 adds the components instead of taking the norm.

</details>

**q31.** $f(x, y) = x^2 + 2y^2$. Give the unit direction of steepest descent at $(1, 1)$. `[vector]`

<details><summary>Answer</summary>

**$(-1/\sqrt5,\ -2/\sqrt5)$.** $\nabla f = (2x, 4y) = (2, 4)$ at $(1, 1)$, with length $2\sqrt5$. Steepest descent is $-\nabla f / \lVert \nabla f \rVert$. Common mistakes: $(1/\sqrt5, 2/\sqrt5)$ is steepest ascent, and $(-2, -4)$ is the right direction but not a unit vector.

</details>

**q32.** $f(x, y) = e^x \sin y$. Give the directional derivative at $(0, \pi/2)$ in the direction $(1, 0)$. `[number]`

<details><summary>Answer</summary>

**$1$.** $\nabla f = (e^x \sin y,\ e^x \cos y) = (1, 0)$ at $(0, \pi/2)$, and $(1, 0) \cdot (1, 0) = 1$.

</details>

**q33.** $f(x, y) = x^2 - y^2$. Give a unit direction at $(1, 1)$ along which the directional derivative is 0 (either sign). `[vector]`

<details><summary>Answer</summary>

**$(1/\sqrt2,\ 1/\sqrt2)$** (or its negative). $\nabla f = (2x, -2y) = (2, -2)$; a direction with zero derivative is perpendicular to it, and $(1, 1)/\sqrt2$ is. Along it you follow the level curve. A common mistake is $(1, -1)/\sqrt2$, which points along the gradient, where the derivative is largest.

</details>

**q34.** Prove that at a point where $\nabla f \ne 0$, the unit direction $u$ that maximizes the directional derivative $D_u f = \nabla f \cdot u$ is $u = \nabla f / \lVert \nabla f \rVert$, and that the maximum is $\lVert \nabla f \rVert$. `[proof]`

<details><summary>Answer</summary>

**Claim.** Let $f$ be differentiable at $x$ with $\nabla f(x) \ne 0$. Over unit vectors $u$, the directional derivative $D_u f(x) = \nabla f(x) \cdot u$ is largest at $u^* = \nabla f(x) / \lVert \nabla f(x) \rVert$, and its maximum value is $\lVert \nabla f(x) \rVert$.

**Proof (direct).** By the chain rule applied to $s \mapsto f(x + su)$, $D_u f(x) = \frac{d}{ds} f(x + su)\big|_{s=0} = \nabla f(x) \cdot u$. For every unit vector $u$, the Cauchy-Schwarz inequality gives $\nabla f(x) \cdot u \le \lVert \nabla f(x) \rVert\,\lVert u \rVert = \lVert \nabla f(x) \rVert$. Since $\nabla f(x) \ne 0$, $u^*$ is defined and $\lVert u^* \rVert = 1$, and $\nabla f(x) \cdot u^* = \lVert \nabla f(x) \rVert^2 / \lVert \nabla f(x) \rVert = \lVert \nabla f(x) \rVert$, so the bound is attained. Hence the maximizing direction is $u^*$ and the maximum is $\lVert \nabla f(x) \rVert$.

A complete proof derives $D_u f = \nabla f \cdot u$ from the chain rule, bounds it with Cauchy-Schwarz, shows $u^*$ is a unit vector that attains the bound (this is where $\nabla f \ne 0$ is used), and states both the direction and the value. Model proof also in [`q34.md`](../../../../course/solutions/solve/S-M04/q34.md).

</details>

### The Hessian and the second-derivative test

**q35.** Give the Hessian of $f(x, y) = x^2 + 3xy + 2y^2$. `[matrix]`

<details><summary>Answer</summary>

**$\begin{pmatrix} 2 & 3 \\ 3 & 4 \end{pmatrix}$.** $f_{xx} = 2$, $f_{yy} = 4$, and $f_{xy} = \partial_y(2x + 3y) = 3$. A common mistake is $3/2$ off the diagonal: in $\frac12 x^\top H x$ the cross term $3xy$ is split as $\frac12(H_{12} + H_{21})xy$, so each off-diagonal entry is the full 3, not half of it. $\begin{pmatrix} 1 & 3 \\ 3 & 2 \end{pmatrix}$ reads the coefficients off without differentiating.

</details>

**q36.** Give the Hessian of $f(x, y) = x^3 + y^3$ at $(1, 1)$. `[matrix]`

<details><summary>Answer</summary>

**$\begin{pmatrix} 6 & 0 \\ 0 & 6 \end{pmatrix}$.** $f_{xx} = 6x$, $f_{yy} = 6y$, $f_{xy} = 0$; at $(1, 1)$ the diagonal is 6. Answering 3 on the diagonal takes only the first derivative's coefficient.

</details>

**q37.** Classify the critical point $(0, 0)$ of $f(x, y) = x^2 - y^2$: `minimum`, `maximum`, or `saddle`. `[choice]`

<details><summary>Answer</summary>

**Saddle.** $H = \operatorname{diag}(2, -2)$ has one positive and one negative eigenvalue: $f$ rises along $x$ and falls along $y$. $\det H = -4 < 0$.

</details>

**q38.** Classify the critical point $(0, 0)$ of $f(x, y) = x^2 + xy + y^2$: `minimum`, `maximum`, or `saddle`. `[choice]`

<details><summary>Answer</summary>

**Minimum.** $H = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$, with $\det H = 4 - 1 = 3 > 0$ and $f_{xx} = 2 > 0$ (eigenvalues 1 and 3). A cross term alone does not make a saddle; only $\det H < 0$ does.

</details>

**q39.** Give the critical point of $f(x, y) = x^2 + y^2 - 2x + 4y$. `[vector]`

<details><summary>Answer</summary>

**$(1, -2)$.** $\nabla f = (2x - 2,\ 2y + 4) = 0$ gives $x = 1$, $y = -2$. Completing the square, $f = (x - 1)^2 + (y + 2)^2 - 5$, confirms it. $(-1, 2)$ has both signs flipped.

</details>

**q40.** Classify the critical point $(1, 0)$ of $f(x, y) = x^3 - 3x + y^2$: `minimum`, `maximum`, or `saddle`. `[choice]`

<details><summary>Answer</summary>

**Minimum.** $\nabla f = (3x^2 - 3,\ 2y)$ vanishes at $(1, 0)$, and $H = \begin{pmatrix} 6x & 0 \\ 0 & 2 \end{pmatrix} = \operatorname{diag}(6, 2)$ there, positive definite. (The other critical point, $(-1, 0)$, is a saddle.)

</details>

**q41.** Is the Hessian of q35 positive definite? `[bool]`

<details><summary>Answer</summary>

**False.** $\det\begin{pmatrix} 2 & 3 \\ 3 & 4 \end{pmatrix} = 8 - 9 = -1 < 0$, so the eigenvalues have opposite signs: the matrix is indefinite and its critical point is a saddle, even though both diagonal entries are positive.

</details>

**q42.** $f(x) = \frac{1}{2} x^\top H x$ with $H = \operatorname{diag}(1, 100)$. Give the condition number $\lambda_{\max} / \lambda_{\min}$ of its Hessian. `[number]`

<details><summary>Answer</summary>

**$100$.** The Hessian of $\frac12 x^\top H x$ is $H$ itself, with eigenvalues 1 and 100. Gradient descent shrinks the error by about $1 - 1/100$ per step, so $\kappa = 100$ means slow convergence along the flat direction (`M10.1`). Common mistakes: $1/100$ inverts the ratio, and 10 is the ratio of the axes' lengths (the square root), not of the eigenvalues.

</details>

### Multiple integrals and change of variables

**q43.** $\displaystyle\int_0^1 \int_0^2 x y\, dy\, dx$. `[number]`

<details><summary>Answer</summary>

**$1$.** Inner: $\int_0^2 xy\,dy = x \cdot \frac{y^2}{2}\Big|_0^2 = 2x$. Outer: $\int_0^1 2x\,dx = 1$. Answering 2 keeps the inner result and forgets the outer integral over $x$.

</details>

**q44.** Give the area of the triangle $0 \le y \le x \le 1$ as a double integral. `[number]`

<details><summary>Answer</summary>

**$1/2$.** $\int_0^1 \int_0^x 1\,dy\,dx = \int_0^1 x\,dx = \frac12$. The inner limit depends on $x$; integrating over the whole unit square gives 1.

</details>

**q45.** $\displaystyle\iint_{x^2 + y^2 \le 1} 1\, dA$. `[number]`

<details><summary>Answer</summary>

**$\pi$.** In polar coordinates, $\int_0^{2\pi} \int_0^1 r\,dr\,d\theta = 2\pi \cdot \frac12 = \pi$, the area of the unit disk. A common mistake is $2\pi$, which integrates $dr\,d\theta$ without the Jacobian $r$.

</details>

**q46.** $\displaystyle\iint_{\mathbb{R}^2} e^{-(x^2 + y^2)}\, dA$. `[number]`

<details><summary>Answer</summary>

**$\pi$.** $\int_0^{2\pi} \int_0^\infty e^{-r^2} r\,dr\,d\theta$; substitute $s = r^2$, $ds = 2r\,dr$: the inner integral is $\frac12 \int_0^\infty e^{-s}\,ds = \frac12$, so the total is $2\pi \cdot \frac12 = \pi$. It is also $\left(\int e^{-x^2}dx\right)^2$, which is how $\int e^{-x^2} = \sqrt\pi$ is proved. $\sqrt\pi$ is the one-variable value.

</details>

**q47.** $\displaystyle\int_0^1 \int_0^x 2y\, dy\, dx$. `[number]`

<details><summary>Answer</summary>

**$1/3$.** Inner: $\int_0^x 2y\,dy = x^2$. Outer: $\int_0^1 x^2\,dx = \frac13$. Answering $1/2$ uses the wrong inner limit.

</details>

**q48.** Give the volume under $z = 1 - x^2 - y^2$ and above the unit disk $x^2 + y^2 \le 1$. `[number]`

<details><summary>Answer</summary>

**$\pi/2$.** $\int_0^{2\pi} \int_0^1 (1 - r^2)\, r\,dr\,d\theta = 2\pi\left(\frac12 - \frac14\right) = \frac{\pi}{2}$. Answering $\pi$ forgets the $r^2$ term (that is the volume of the cylinder of height 1).

</details>

**q49.** $\displaystyle\iint_{\mathbb{R}^2} \frac{1}{2\pi} e^{-(x^2 + y^2)/2}\, dA$, the total mass of a 2-D standard normal. `[number]`

<details><summary>Answer</summary>

**$1$.** $\frac{1}{2\pi}\int_0^{2\pi} \int_0^\infty e^{-r^2/2} r\,dr\,d\theta = \frac{1}{2\pi} \cdot 2\pi \cdot 1$, since $\int_0^\infty r e^{-r^2/2}dr = 1$ by $s = r^2/2$. A density integrates to 1, which is what makes $\frac{1}{2\pi}$ the right constant (`M07.0`). $2\pi$ forgets the normalizer.

</details>

**q50.** With $u = x + y$, $v = x - y$, give the area of the region $|x + y| \le 1$, $|x - y| \le 1$. `[number]`

<details><summary>Answer</summary>

**$2$.** In $(u, v)$ the region is the square $[-1, 1]^2$, of area 4. The map $(x, y) \mapsto (u, v)$ has $\det \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} = -2$, so $du\,dv = 2\,dx\,dy$ and the area in $(x, y)$ is $4 / 2 = 2$. (The region is a square with diagonals of length 2.) A common mistake is 4, which forgets the Jacobian factor; 8 multiplies by it instead of dividing.

</details>

### Lagrange multipliers

**q51.** Give the maximum of $x + y$ subject to $x^2 + y^2 = 1$. `[number]`

<details><summary>Answer</summary>

**$\sqrt2$.** $\nabla(x + y) = (1, 1) = \lambda(2x, 2y)$ gives $x = y$, and $x^2 + y^2 = 1$ gives $x = y = 1/\sqrt2$, so the maximum is $2/\sqrt2 = \sqrt2$ (the minimum $-\sqrt2$ is at the opposite point). 1 is the value at $(1, 0)$, not an optimum; 2 ignores the constraint.

</details>

**q52.** Give the minimum of $x^2 + y^2$ subject to $x + y = 1$. `[number]`

<details><summary>Answer</summary>

**$1/2$.** $(2x, 2y) = \lambda(1, 1)$ gives $x = y = \frac12$, and $f = \frac14 + \frac14 = \frac12$. A common mistake is $1/\sqrt2$, which is the distance from the origin to the line, not its square.

</details>

**q53.** Give the maximum of $xy$ subject to $x + y = 10$. `[number]`

<details><summary>Answer</summary>

**$25$.** $(y, x) = \lambda(1, 1)$ gives $x = y = 5$, and $xy = 25$. Answering 5 reports the optimizer $x = y = 5$ instead of the optimum value.

</details>

**q54.** Maximize the entropy $-\sum_{i=1}^{3} p_i \ln p_i$ subject to $p_1 + p_2 + p_3 = 1$. Give the optimal $p_1$. `[number]`

<details><summary>Answer</summary>

**$1/3$.** $\partial_{p_i}\left(-\sum p_j \ln p_j - \lambda(\sum p_j - 1)\right) = -\ln p_i - 1 - \lambda = 0$, so every $p_i$ has the same value, and the constraint makes it $1/3$. With only the normalization constraint, maximum entropy is uniform; adding an energy constraint (q58) turns it into a softmax (`M11.5`).

</details>

**q55.** Give the maximum of $xyz$ subject to $x + y + z = 3$ with $x, y, z > 0$. `[number]`

<details><summary>Answer</summary>

**$1$.** $(yz, xz, xy) = \lambda(1, 1, 1)$ forces $x = y = z$ (all positive), so each is 1 and $xyz = 1$; AM-GM confirms $xyz \le ((x + y + z)/3)^3 = 1$. A common mistake is 27, which is $(x + y + z)^3$.

</details>

**q56.** Give the minimum of $2x + 3y$ subject to $xy = 6$ with $x, y > 0$. `[number]`

<details><summary>Answer</summary>

**$12$.** $(2, 3) = \lambda(y, x)$ gives $y = 2/\lambda$, $x = 3/\lambda$, and $xy = 6/\lambda^2 = 6$ gives $\lambda = 1$: $x = 3$, $y = 2$, value $6 + 6 = 12$. A common mistake is 13 from $x = 2$, $y = 3$, which pairs the multiplier equations the wrong way round.

</details>

**q57.** Give the point of the plane $x + 2y + 2z = 9$ closest to the origin. `[vector]`

<details><summary>Answer</summary>

**$(1, 2, 2)$.** Minimize $x^2 + y^2 + z^2$: $(2x, 2y, 2z) = \lambda(1, 2, 2)$, so the point is $t(1, 2, 2)$, on the normal through the origin. The constraint gives $t + 4t + 4t = 9$, $t = 1$. The distance is 3. $(1/3, 2/3, 2/3)$ is the unit normal, a direction, not the point.

</details>

**q58.** With Lagrange multipliers, show that the distribution $p$ over states $1, \dots, n$ with energies $E_i$ that maximizes the entropy $-\sum_i p_i \ln p_i$ subject to $\sum_i p_i = 1$ and $\sum_i p_i E_i = U$ has the form $p_i = e^{-\beta E_i} / \sum_j e^{-\beta E_j}$ (a softmax of $-\beta E$). `[proof]`

<details><summary>Answer</summary>

**Claim.** Among distributions $p = (p_1, \dots, p_n)$ with $p_i > 0$, $\sum_i p_i = 1$, and $\sum_i p_i E_i = U$, a stationary point of the entropy $H(p) = -\sum_i p_i \ln p_i$ has $p_i = e^{-\beta E_i} / \sum_j e^{-\beta E_j}$ for a constant $\beta$.

**Proof (Lagrange multipliers).** Introduce multipliers $\alpha$ for the normalization and $\beta$ for the energy constraint, and form
$$\mathcal{L}(p, \alpha, \beta) = -\sum_i p_i \ln p_i - \alpha\Big(\sum_i p_i - 1\Big) - \beta\Big(\sum_i p_i E_i - U\Big).$$
At a constrained stationary point, $\partial \mathcal{L} / \partial p_i = 0$ for each $i$: $-\ln p_i - 1 - \alpha - \beta E_i = 0$, so $p_i = e^{-1 - \alpha}\, e^{-\beta E_i}$. The factor $e^{-1-\alpha}$ is the same for every $i$. The normalization $\sum_i p_i = 1$ gives $e^{-1 - \alpha} \sum_j e^{-\beta E_j} = 1$, so $e^{-1-\alpha} = 1 / \sum_j e^{-\beta E_j}$. Therefore
$$p_i = \frac{e^{-\beta E_i}}{\sum_j e^{-\beta E_j}},$$
the softmax of $-\beta E$, with $\beta$ the value for which $\sum_i p_i E_i = U$. (Since $H$ is strictly concave and the constraints are linear, this stationary point is the maximum.)

A complete proof names both multipliers in the Lagrangian, sets each partial in $p_i$ to zero, uses the normalization to fix the common constant, and concludes the softmax form with $\beta$ fixed by the energy constraint. Model proof also in [`q58.md`](../../../../course/solutions/solve/S-M04/q58.md).

</details>

## 5. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Dropping an inner derivative in a partial | $\partial_x \sin(xy)$ given as $\cos(xy)$; gradcheck fails on every product inside a nonlinearity | applies the chain rule to every inner function and writes its derivative as a separate factor (q9, q10, q18) |
| Forgetting the $A^\top$ in a least-squares gradient | descent along the wrong direction for any non-identity $A$ | differentiates through $Ax$ with the chain rule, which brings in $A^\top$ (q16) |
| Multiplying the derivatives of the parts instead of summing over paths | $dz/dt$ for $z = xy$ given as $x'(t)\,y'(t)$ | writes one term per path from $t$ to $z$ and adds them (q20) |
| Skipping a factor in a chain of local derivatives | a backward pass off by the sigmoid's slope or a weight | backpropagates one named local derivative at a time and multiplies them all (q23, q27) |
| Computing $J u$ when the backward pass needs $u^\top J$ | the gradient of a non-symmetric layer comes out transposed | computes the row vector $u^\top J$, a weighted sum of rows of $J$ (q28) |
| Using an unnormalized direction | a directional derivative scaled by the direction's length | uses the given unit vector, or divides by the norm first (q29, q31) |
| Halving the mixed term of a Hessian | a saddle misread as a minimum | takes $H_{12} = \partial_x\partial_y f$ directly, which is the full coefficient of $xy$ (q35) |
| Forgetting the Jacobian factor in a change of variables | polar integrals off by the missing $r$ | writes $dA = \lvert\det J\rvert\,du\,dv$ before setting limits (q45, q50) |
| Reporting the optimizer instead of the optimum | the argmax given as the max | substitutes the optimizer back into $f$ and reports that value (q52, q53) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M01` | one-variable derivatives, the chain rule, and integrals |
| Forward | `M04.1` | `gradcheck` compares analytic partials (q9 to q18) with central differences |
| Forward | `M04.2` | `jacobian`, `vjp_numeric`, `jvp_numeric`: the products of q25 and q28 |
| Forward | `M08.2` | reverse-mode autodiff is the chain of VJPs of q27 |
| Forward | `M10.1` | gradient descent, Armijo steps, and the condition number of q42 |
| Forward | `M07.3` | Gaussian integrals in two variables (q46, q49) behind initialization variances |
| Forward | `M11.5` | softmax as maximum entropy under an energy constraint (q58) |
