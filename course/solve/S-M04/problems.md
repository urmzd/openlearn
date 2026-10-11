# S-M04 problems: vectors, gradients, the chain rule, directional derivatives, Hessians, multiple integrals, Lagrange

Answer every question in `solve/S-M04.toml` (written by `ol start S-M04`).
The tag after each question is its answer type. `[number]` is an exact
value (`14/5`, `sqrt(13)`, `pi/2`). `[expr in x, y]` is a formula in the
named variables. `[vector]` is a flat list `[8, 7]`, entries possibly formulas
(`[2*a, 2*b, 2*c]`). `[matrix]` is a list of rows. `[equation]` is one
equation with a single `=`. `[choice]` is one word from the list given.
`[bool]` is `true` or `false`. `[proof]` is a file `solve/S-M04/qN.md`, graded
against its rubric. Write $e$ as `E` or `exp(1)` and $\pi$ as `pi`.

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
