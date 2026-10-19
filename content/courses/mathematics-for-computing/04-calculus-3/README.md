# Calculus 3

Multivariable calculus generalizes the single-variable ideas of derivative and integral to functions of several variables. The gradient replaces the derivative, multiple integrals replace single integrals, and the classical theorems of Green, Stokes, and Gauss reveal that integration over a region and integration over its boundary are deeply related. For computing, the payoff is direct: the gradient is the engine of optimization (gradient descent), the Jacobian is the engine of backpropagation, and multiple integrals compute joint probabilities.

This chapter has two parts. Its two build sections turn partial derivatives and Jacobians into code that later chapters depend on: a gradient checker that referees every backward rule you will ever write, and numeric oracles for the Jacobian-vector and vector-Jacobian products that forward and reverse mode compute. Its problem set then works the hand computations behind the rest of the subject: vectors and geometry, gradients, the chain rule, directional derivatives, Hessians, multiple integrals, and Lagrange multipliers. The survey below the section list covers the wider territory of a Calculus 3 course, including vector-valued functions and vector calculus, with a worked example for each topic.

> **Learning objectives**
> - Compute partial derivatives, gradients, and directional derivatives, and explain why $-\nabla f$ is the direction of steepest descent.
> - Build and use a numerical gradient check with a mixed absolute and relative tolerance.
> - Form Jacobians, apply the multivariable chain rule as a matrix product, and compute JVPs and VJPs without forming the matrix.
> - Classify critical points with the Hessian, and solve constrained problems with Lagrange multipliers.
> - Evaluate double and triple integrals with a change of variables, and state Green's, Stokes', and the Divergence theorems.

## Before you begin

You should be comfortable with one-variable derivatives and finite differences from [Calculus 1](../01-calculus-1/), especially [Derivative as a limit, finite differences, step-size choice](../01-calculus-1/01-derivatives-and-finite-differences.md), whose `central_diff` the first section calls; with Taylor series from [Calculus 2](../02-calculus-2/); and with matrices, row-major layout, and matrix products from [Linear Algebra](../03-linear-algebra/), especially [Vectors, matrices, row-major layout, and matmul](../03-linear-algebra/01-vectors-matrices-and-matmul.md).

## In this chapter

1. [Partial derivatives, gradients, and gradcheck](01-gradients-and-gradcheck.md): builds the gradient of a scalar function of several arrays from one central difference per element, and a `gradcheck` that rejects real backward-pass bugs and points at the element furthest past its tolerance.
2. [Jacobians, the multivariable chain rule, and numeric JVP/VJP](02-jacobians-and-the-chain-rule.md): defines the Jacobian and the chain rule as a matrix product, and builds numeric oracles for $Jv$ and $u^\top J$ that the autodiff chapter checks against.
3. [Calculus 3 problem set](90-problem-set.md): 58 problems with worked answers on vectors, gradients, the chain rule, directional derivatives, Hessians, multiple integrals, and Lagrange multipliers. It is the book's treatment of the Hessian, multiple integrals, and Lagrange multipliers.

## A survey of the subject

The sections above focus on the parts of multivariable calculus that code depends on. The rest of this opening surveys the whole subject, following *Calculus Volume 3* from OpenStax, so you can see where those parts sit. Each topic gives the key definitions and theorems, a worked example, and the textbook exercises to practice on.

### Vectors in space

Textbook chapter 2, sections 2.1 to 2.5. The **dot product** is $u \cdot v = u_1 v_1 + u_2 v_2 + u_3 v_3 = \lVert u \rVert \lVert v \rVert \cos\theta$. It measures how much two vectors point in the same direction; zero means perpendicular. The **cross product** is

$$u \times v = (u_2 v_3 - u_3 v_2,\; u_3 v_1 - u_1 v_3,\; u_1 v_2 - u_2 v_1),$$

perpendicular to both $u$ and $v$, with magnitude $\lVert u \times v \rVert = \lVert u \rVert \lVert v \rVert \sin\theta$, the area of the parallelogram they span. So the cross product encodes both a direction (normal to the plane of $u$ and $v$) and a magnitude (area). The **triple scalar product** $u \cdot (v \times w) = \det([u; v; w])$ is the signed volume of the parallelepiped: determinants measure volume, and the triple product is exactly a $3 \times 3$ determinant.

A **line** in space is $r(t) = r_0 + t d$, a point plus a direction. A **plane** is $n \cdot (r - r_0) = 0$, or $ax + by + cz = d$ where $(a, b, c)$ is the normal vector.

For example, find the plane through $(1,2,3)$, $(2,0,1)$, and $(0,1,2)$. Two vectors in the plane are $v = (1,-2,-2)$ and $w = (-1,-1,-1)$. The normal is

$$v \times w = \bigl((-2)(-1)-(-2)(-1),\; (-2)(-1)-1\cdot(-1),\; 1\cdot(-1)-(-2)(-1)\bigr) = (0, 3, -3),$$

which simplifies to $(0,1,-1)$. The plane is $(y-2) - (z-3) = 0$, or $y - z + 1 = 0$.

Practice: OpenStax 2.3 exercises 123 to 135, 2.4 exercises 161 to 175, and 2.5 exercises 215 to 228; for a challenge, 2.5 exercises 237 to 242.

### Vector-valued functions

Textbook chapter 3, sections 3.1 to 3.4. A **vector-valued function** $r(t) = \langle x(t), y(t), z(t) \rangle$ describes a curve in space. Its derivative $r'(t) = \langle x'(t), y'(t), z'(t) \rangle$ is the tangent vector, the **arc length** is $L = \int_a^b \lVert r'(t) \rVert\,dt$, and the **unit tangent** is $T(t) = r'(t) / \lVert r'(t) \rVert$. Every smooth curve can be reparameterized by arc length $s$ so that $\lVert r'(s) \rVert = 1$ everywhere: traveling at unit speed, the parameter directly measures distance along the curve.

**Curvature** measures how sharply a curve bends:

$$\kappa = \frac{\lVert T'(t) \rVert}{\lVert r'(t) \rVert} = \frac{\lVert r'(t) \times r''(t) \rVert}{\lVert r'(t) \rVert^3}.$$

A straight line has $\kappa = 0$; a circle of radius $R$ has $\kappa = 1/R$. For the helix $r(t) = \langle \cos t, \sin t, t \rangle$: $r'(t) = \langle -\sin t, \cos t, 1 \rangle$ with $\lVert r' \rVert = \sqrt2$, $r''(t) = \langle -\cos t, -\sin t, 0 \rangle$, and $r' \times r'' = \langle \sin t, -\cos t, 1 \rangle$ with $\lVert r' \times r'' \rVert = \sqrt2$. So $\kappa = \sqrt2 / (\sqrt2)^3 = 1/2$: the helix has constant curvature $1/2$.

Practice: OpenStax 3.2 exercises 60 to 72, 3.3 exercises 100 to 112, and 3.4 exercises 130 to 140; for a challenge, 3.4 exercises 145 to 150.

### Partial derivatives

Textbook chapter 4, sections 4.1 to 4.8. The **partial derivative** $f_x(x,y) = \lim_{h \to 0} [f(x+h,y) - f(x,y)]/h$ differentiates with respect to $x$ holding $y$ constant. The **gradient** $\nabla f = (f_x, f_y, f_z)$ points in the direction of steepest ascent, and the **directional derivative** $D_u f = \nabla f \cdot u$ is the rate of change in direction $u$. The **tangent plane** at $(x_0, y_0)$ is $z - z_0 = f_x(x_0,y_0)(x - x_0) + f_y(x_0,y_0)(y - y_0)$. A **critical point** is where $\nabla f = 0$ or $\nabla f$ is undefined, and the **Hessian** $H = \begin{pmatrix} f_{xx} & f_{xy} \\ f_{yx} & f_{yy} \end{pmatrix}$ encodes second-order behavior.

Four results organize the topic. The **multivariable chain rule**: if $z = f(x,y)$ with $x = x(t)$, $y = y(t)$, then $dz/dt = f_x\,dx/dt + f_y\,dy/dt$; each variable contributes its own rate of change, weighted by how the function responds to it. **Clairaut's theorem**: if $f_{xy}$ and $f_{yx}$ are continuous, they are equal, so the order of mixed partials does not matter under mild smoothness. The **second derivative test**: at a critical point $(a,b)$, let $D = f_{xx} f_{yy} - f_{xy}^2$; if $D > 0$ and $f_{xx} > 0$ it is a local minimum, if $D > 0$ and $f_{xx} < 0$ a local maximum, and if $D < 0$ a saddle. $D$ is the determinant of the Hessian, and it checks whether the surface curves the same way in all directions. **Lagrange multipliers**: to optimize $f$ subject to $g = c$, solve $\nabla f = \lambda \nabla g$ and $g = c$. At a constrained extremum the gradient of $f$ must be parallel to the gradient of $g$; otherwise you could improve $f$ by moving along the constraint.

Two worked examples. First, minimize $f(x,y) = x^2 + y^2$ subject to $x + y = 4$. Lagrange gives $(2x, 2y) = \lambda(1, 1)$, so $2x = \lambda = 2y$ and $x = y$. The constraint gives $x = y = 2$, and the minimum is $f(2,2) = 8$.

Second, classify the critical points of $f(x,y) = x^3 - 3xy + y^3$. Setting $f_x = 3x^2 - 3y = 0$ and $f_y = -3x + 3y^2 = 0$, the first gives $y = x^2$; substituting, $-3x + 3x^4 = 0$, so $x(x^3 - 1) = 0$, giving $(0,0)$ and $(1,1)$. At $(1,1)$, $f_{xx} = 6$, $f_{yy} = 6$, $f_{xy} = -3$, so $D = 36 - 9 = 27 > 0$ with $f_{xx} > 0$: a local minimum. At $(0,0)$, $D = 0 - 9 = -9 < 0$: a saddle.

Practice: OpenStax 4.3 exercises 115 to 128, 4.5 exercises 200 to 215, 4.6 exercises 240 to 252, and 4.8 exercises 340 to 355; for a challenge, 4.8 exercises 360 to 368.

### Multiple integrals

Textbook chapter 5, sections 5.1 to 5.7. A **double integral** $\iint_R f(x,y)\,dA$ is computed as iterated integrals over $x$ and $y$, and a **triple integral** $\iiint_E f(x,y,z)\,dV$ likewise. **Fubini's theorem** says that if $f$ is continuous on a rectangle $R = [a,b] \times [c,d]$, the double integral equals either iterated integral: you can slice horizontally or vertically and get the same answer.

Other coordinate systems simplify circular and spherical regions. **Polar**: $x = r\cos\theta$, $y = r\sin\theta$, $dA = r\,dr\,d\theta$. **Cylindrical**: $(r, \theta, z)$ with $dV = r\,dr\,d\theta\,dz$. **Spherical**: $x = \rho\sin\phi\cos\theta$, $y = \rho\sin\phi\sin\theta$, $z = \rho\cos\phi$, $dV = \rho^2 \sin\phi\,d\rho\,d\phi\,d\theta$. All are instances of the **change of variables** formula. For a transformation $(x,y) = T(u,v)$ with Jacobian determinant $J = \det\begin{pmatrix} \partial x/\partial u & \partial x/\partial v \\ \partial y/\partial u & \partial y/\partial v \end{pmatrix}$, $dA = |J|\,du\,dv$ and

$$\iint_R f(x,y)\,dA = \iint_S f(T(u,v))\,|J|\,du\,dv.$$

The Jacobian determinant measures how much the transformation stretches or compresses area, just as $|dx/du|$ does in single-variable substitution.

For example, integrate $x^2 + y^2$ over the disk $x^2 + y^2 \le 4$. In polar coordinates,

$$\int_0^{2\pi}\!\int_0^2 r^2 \cdot r\,dr\,d\theta = \int_0^{2\pi} d\theta \int_0^2 r^3\,dr = 2\pi \left[\frac{r^4}{4}\right]_0^2 = 2\pi \cdot 4 = 8\pi.$$

Practice: OpenStax 5.1 exercises 1 to 14, 5.3 exercises 130 to 145, 5.5 exercises 240 to 255, and 5.7 exercises 340 to 352; for a challenge, 5.7 exercises 358 to 365.

### Vector calculus

Textbook chapter 6, sections 6.1 to 6.8. A **vector field** is $F(x,y,z) = \langle P, Q, R \rangle$. The **line integral of a scalar** is $\int_C f\,ds = \int_a^b f(r(t))\,\lVert r'(t) \rVert\,dt$, and the **line integral of a vector field**, the work done by $F$ along $C$, is $\int_C F \cdot dr = \int_a^b F(r(t)) \cdot r'(t)\,dt$. A field is **conservative** if $F = \nabla f$ for some scalar potential $f$. **Curl**, $\nabla \times F = (R_y - Q_z,\; P_z - R_x,\; Q_x - P_y)$, measures local rotation; **divergence**, $\nabla \cdot F = P_x + Q_y + R_z$, measures local expansion. A **surface integral** is $\iint_S F \cdot dS = \iint_D F \cdot (r_u \times r_v)\,dA$.

Four theorems relate integrals over a region to integrals over its boundary:

- **Fundamental theorem for line integrals**: if $F = \nabla f$, then $\int_C F \cdot dr = f(B) - f(A)$. For gradient fields, work depends only on the endpoints, not the path, just like the fundamental theorem of calculus.
- **Green's theorem**: $\oint_C (P\,dx + Q\,dy) = \iint_D (Q_x - P_y)\,dA$. The circulation around a closed curve equals the total curl inside.
- **Stokes' theorem**: $\oint_C F \cdot dr = \iint_S (\nabla \times F) \cdot dS$. It generalizes Green's theorem to 3D: the circulation of $F$ around a closed curve equals the flux of the curl through any surface bounded by that curve.
- **Divergence theorem**: $\iint_S F \cdot dS = \iiint_E (\nabla \cdot F)\,dV$. The total outward flux through a closed surface equals the total divergence inside: what flows out must be produced within.

For example, use Green's theorem to evaluate $\oint_C (y^2\,dx + x^2\,dy)$ around the unit circle, counterclockwise. With $P = y^2$ and $Q = x^2$, $Q_x - P_y = 2x - 2y$, so the integral is $\iint_D (2x - 2y)\,dA$. In polar coordinates this is $\int_0^{2\pi}\!\int_0^1 (2r\cos\theta - 2r\sin\theta)\,r\,dr\,d\theta$, and since $\cos\theta$ and $\sin\theta$ each integrate to 0 over $[0, 2\pi]$, the result is 0.

Practice: OpenStax 6.2 exercises 55 to 70, 6.3 exercises 115 to 130, 6.4 exercises 155 to 168, 6.7 exercises 285 to 298, and 6.8 exercises 325 to 338; for a challenge, 6.8 exercises 345 to 352.

### Choosing a technique

Most problems in the subject reduce to recognizing which tool applies. This table pairs each technique with when to reach for it:

| Technique | When to use | Key formula or idea |
|---|---|---|
| Gradient | find the direction of steepest ascent | $\nabla f = (f_x, f_y, f_z)$ |
| Lagrange multipliers | constrained optimization | $\nabla f = \lambda \nabla g$ |
| Second derivative test | classify critical points in 2D | check $\det H$ and $f_{xx}$ |
| Polar, cylindrical, spherical coordinates | integrate over circular or spherical regions | include the Jacobian factor ($r$, $r$, $\rho^2 \sin\phi$) |
| Green's theorem | convert a line integral to a double integral (2D) | circulation = integral of curl over the region |
| Stokes' theorem | convert a line integral to a surface integral (3D) | circulation = flux of curl |
| Divergence theorem | convert a surface integral to a volume integral | flux = integral of divergence |
| Change of variables | simplify the integration domain | $dA = \lvert J \rvert\,du\,dv$ |

## Where this leads

The gradient and directional derivative drive gradient descent in [Optimization](../10-optimization/). The Jacobian is the chain rule for vector functions, which is backpropagation, and [Matrix calculus and autodiff](../08-matrix-calculus-and-autodiff/) builds it three ways. Lagrange multipliers underlie constrained optimization such as SVMs and regularization, and the reading of softmax as a maximum-entropy distribution in [Information theory](../11-information-theory/). Multiple integrals compute joint probability distributions and expected values in [Probability and statistics](../07-probability-statistics/). The change of variables formula is the core of normalizing flows in generative models, and divergence and curl appear in physics simulation and fluid dynamics for games.

### Further reading

- OpenStax, [*Calculus Volume 3*](https://openstax.org/details/books/calculus-volume-3) (CC BY-NC-SA 4.0), the textbook this chapter follows.
- [Paul's Online Math Notes, Calculus III](https://tutorial.math.lamar.edu/Classes/CalcIII/CalcIII.aspx), for additional worked examples and practice problems.
