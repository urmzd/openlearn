# Calculus 3

## Overview
- **Textbook**: *Calculus Volume 3* -- https://openstax.org/details/books/calculus-volume-3 (CC BY-NC-SA 4.0)
- **Supplementary**: [Paul's Online Math Notes](https://tutorial.math.lamar.edu/Classes/CalcIII/CalcIII.aspx)
- **Prerequisites**: [Calculus 2](../02-calculus-2/), [Linear Algebra](../03-linear-algebra/)
- **Estimated time**: 4 weeks at 10-12 hrs/week

## Key Takeaways
- Extend differentiation and integration to functions of several variables
- Master partial derivatives, the gradient, and directional derivatives -- the language of optimization in higher dimensions
- Understand Lagrange multipliers as the standard method for constrained optimization
- Compute double and triple integrals using coordinate transformations and the Jacobian
- Connect line integrals, surface integrals, and the classical integral theorems (Green's, Stokes', Divergence)

## How to Study
- Read textbook sections and work through all inline examples before attempting exercises
- Draw pictures constantly -- multivariable calculus is deeply geometric
- Use Paul's Online Math Notes for additional worked examples and practice problems
- Practice setting up integrals from word problems -- choosing the right coordinate system is half the battle
- Connect gradient descent to partial derivatives early; this motivates the entire course for CS students

---

# Concepts & Techniques

## Core Insight

Multivariable calculus generalizes the single-variable ideas of derivative and integral to
functions of several variables. The gradient replaces the derivative, multiple integrals replace
single integrals, and the classical theorems of Green, Stokes, and Gauss reveal that integration
over a region and integration over its boundary are deeply related. For CS, the gradient is the
engine of optimization (gradient descent), the Jacobian is the engine of backpropagation, and
multiple integrals compute joint probabilities.

## 1. Vectors in Space

**Textbook sections**: Ch 2, Sections 2.1-2.5

**Key definitions**:
- **Dot product**: u . v = u_1*v_1 + u_2*v_2 + u_3*v_3 = ||u|| ||v|| cos(theta)
- **Cross product**: u x v = (u_2*v_3 - u_3*v_2, u_3*v_1 - u_1*v_3, u_1*v_2 - u_2*v_1); result is perpendicular to both u and v with magnitude ||u|| ||v|| sin(theta)
- **Line in space**: r(t) = r_0 + t*d (point + direction)
- **Plane**: n . (r - r_0) = 0, or ax + by + cz = d where (a,b,c) is the normal vector

**Key theorems**:
- **Geometric Interpretation of Dot Product**: u . v = ||u|| ||v|| cos(theta). *Intuition*: the dot product measures how much two vectors point in the same direction. Zero means perpendicular.
- **Cross Product Magnitude**: ||u x v|| = ||u|| ||v|| sin(theta) = area of the parallelogram spanned by u and v. *Intuition*: the cross product encodes both a direction (normal to the plane of u,v) and a magnitude (area).
- **Triple Scalar Product**: u . (v x w) = det([u; v; w]) = signed volume of the parallelepiped. *Intuition*: determinants measure volume, and the triple product is exactly a 3x3 determinant.

**Worked example**:
> Find the equation of the plane through (1,2,3), (2,0,1), and (0,1,2). Two vectors in the plane: v = (1,-2,-2), w = (-1,-1,-1). Normal: v x w = ((-2)(-1)-(-2)(-1), (-2)(-1)-1(-1), 1(-1)-(-2)(-1)) = (0, 3, -3). Simplify normal to (0,1,-1). Plane: (y-2) - (z-3) = 0, or y - z + 1 = 0.

**Essential problems**: OpenStax 2.3 Exercises: #123-135; 2.4 Exercises: #161-175; 2.5 Exercises: #215-228
**Challenge problems**: OpenStax 2.5 Exercises: #237-242

## 2. Vector-Valued Functions

**Textbook sections**: Ch 3, Sections 3.1-3.4

**Key definitions**:
- **Vector-valued function**: r(t) = <x(t), y(t), z(t)>; describes a curve in space
- **Derivative**: r'(t) = <x'(t), y'(t), z'(t)>; tangent vector to the curve
- **Arc length**: L = integral from a to b of ||r'(t)|| dt
- **Unit tangent vector**: T(t) = r'(t) / ||r'(t)||
- **Curvature**: kappa = ||T'(t)|| / ||r'(t)|| = ||r'(t) x r''(t)|| / ||r'(t)||^3

**Key theorems**:
- **Arc Length Parameterization**: Every smooth curve can be reparameterized by arc length s, so that ||r'(s)|| = 1 everywhere. *Intuition*: arc length parameterization means traveling at unit speed -- the parameter directly measures distance along the curve.
- **Curvature Formula**: kappa = ||r' x r''|| / ||r'||^3. *Intuition*: curvature measures how sharply a curve bends. A straight line has kappa = 0; a circle of radius R has kappa = 1/R.

**Worked example**:
> Find the curvature of r(t) = <cos(t), sin(t), t> (helix). r'(t) = <-sin(t), cos(t), 1>, ||r'|| = sqrt(2). r''(t) = <-cos(t), -sin(t), 0>. r' x r'' = <sin(t), -cos(t), 1>. ||r' x r''|| = sqrt(2). kappa = sqrt(2) / (sqrt(2))^3 = 1/2. The helix has constant curvature 1/2.

**Essential problems**: OpenStax 3.2 Exercises: #60-72; 3.3 Exercises: #100-112; 3.4 Exercises: #130-140
**Challenge problems**: OpenStax 3.4 Exercises: #145-150

## 3. Partial Derivatives

**Textbook sections**: Ch 4, Sections 4.1-4.8

**Key definitions**:
- **Partial derivative**: f_x(x,y) = lim(h->0) [f(x+h,y) - f(x,y)] / h; differentiate with respect to x holding y constant
- **Gradient**: grad f = (f_x, f_y, f_z); points in the direction of steepest ascent
- **Directional derivative**: D_u f = grad f . u; rate of change in direction u
- **Tangent plane**: z - z_0 = f_x(x_0,y_0)(x - x_0) + f_y(x_0,y_0)(y - y_0)
- **Critical point**: A point where grad f = 0 or grad f is undefined
- **Hessian matrix**: H = [[f_xx, f_xy],[f_yx, f_yy]]; encodes second-order behavior

**Key theorems**:
- **Chain Rule (Multivariable)**: If z = f(x,y) where x = x(t), y = y(t), then dz/dt = f_x * dx/dt + f_y * dy/dt. *Intuition*: each variable contributes its own rate of change, weighted by how the function responds to that variable.
- **Clairaut's Theorem**: If f_xy and f_yx are continuous, then f_xy = f_yx. *Intuition*: the order of mixed partial derivatives does not matter (under mild smoothness conditions).
- **Second Derivative Test**: At a critical point (a,b), let D = f_xx * f_yy - (f_xy)^2. If D > 0 and f_xx > 0: local min. If D > 0 and f_xx < 0: local max. If D < 0: saddle point. *Intuition*: D is the determinant of the Hessian -- it checks whether the surface curves the same way in all directions.
- **Lagrange Multipliers**: To optimize f subject to g = c, solve grad f = lambda * grad g and g = c. *Intuition*: at a constrained extremum, the gradient of f must be parallel to the gradient of g -- otherwise you could improve f by moving along the constraint.

**Worked example**:
> Find the minimum of f(x,y) = x^2 + y^2 subject to x + y = 4. Lagrange: grad f = lambda * grad g gives (2x, 2y) = lambda(1, 1). So 2x = lambda and 2y = lambda, thus x = y. Constraint: x + y = 4 gives x = y = 2. Minimum value: f(2,2) = 8.

> Second Derivative Test: f(x,y) = x^3 - 3xy + y^3. Critical points: f_x = 3x^2 - 3y = 0, f_y = -3x + 3y^2 = 0. From the first: y = x^2. Substituting: -3x + 3x^4 = 0, so x(x^3 - 1) = 0, giving (0,0) and (1,1). At (1,1): f_xx = 6, f_yy = 6, f_xy = -3, D = 36 - 9 = 27 > 0 and f_xx > 0, so local minimum. At (0,0): D = 0 - 9 = -9 < 0, so saddle point.

**Essential problems**: OpenStax 4.3 Exercises: #115-128; 4.5 Exercises: #200-215; 4.6 Exercises: #240-252; 4.8 Exercises: #340-355
**Challenge problems**: OpenStax 4.8 Exercises: #360-368

## 4. Multiple Integrals

**Textbook sections**: Ch 5, Sections 5.1-5.7

**Key definitions**:
- **Double integral**: integral integral_R f(x,y) dA = iterated integrals over x and y
- **Triple integral**: integral integral integral_E f(x,y,z) dV
- **Polar coordinates**: x = r*cos(theta), y = r*sin(theta), dA = r dr d(theta)
- **Cylindrical coordinates**: (r, theta, z), dV = r dr d(theta) dz
- **Spherical coordinates**: x = rho*sin(phi)*cos(theta), y = rho*sin(phi)*sin(theta), z = rho*cos(phi), dV = rho^2 * sin(phi) d(rho) d(phi) d(theta)
- **Jacobian**: For a transformation (x,y) = T(u,v), J = det([[dx/du, dx/dv],[dy/du, dy/dv]]); dA = |J| du dv

**Key theorems**:
- **Fubini's Theorem**: If f is continuous on a rectangle R = [a,b] x [c,d], then the double integral equals either iterated integral. *Intuition*: you can integrate in either order -- slice horizontally or vertically and get the same answer.
- **Change of Variables**: integral integral_R f(x,y) dA = integral integral_S f(T(u,v)) |J| du dv. *Intuition*: the Jacobian determinant measures how much the transformation stretches or compresses area, just as |dx/du| does in single-variable substitution.

**Worked example**:
> Compute integral integral over the disk x^2 + y^2 <= 4 of (x^2 + y^2) dA. Switch to polar: integral from 0 to 2*pi integral from 0 to 2 of r^2 * r dr d(theta) = integral from 0 to 2*pi d(theta) * integral from 0 to 2 of r^3 dr = 2*pi * [r^4/4] from 0 to 2 = 2*pi * 4 = 8*pi.

**Essential problems**: OpenStax 5.1 Exercises: #1-14; 5.3 Exercises: #130-145; 5.5 Exercises: #240-255; 5.7 Exercises: #340-352
**Challenge problems**: OpenStax 5.7 Exercises: #358-365

## 5. Vector Calculus

**Textbook sections**: Ch 6, Sections 6.1-6.8

**Key definitions**:
- **Vector field**: F(x,y,z) = <P(x,y,z), Q(x,y,z), R(x,y,z)>
- **Line integral of a scalar**: integral_C f ds = integral from a to b of f(r(t)) * ||r'(t)|| dt
- **Line integral of a vector field**: integral_C F . dr = integral from a to b of F(r(t)) . r'(t) dt (work done by F along C)
- **Conservative field**: F = grad f for some scalar potential f; then integral_C F . dr = f(end) - f(start)
- **Curl**: curl F = (R_y - Q_z, P_z - R_x, Q_x - P_y); measures local rotation
- **Divergence**: div F = P_x + Q_y + R_z; measures local expansion
- **Surface integral**: integral integral_S F . dS = integral integral_D F . (r_u x r_v) dA

**Key theorems**:
- **Fundamental Theorem for Line Integrals**: If F = grad f, then integral_C F . dr = f(B) - f(A). *Intuition*: for gradient fields, work depends only on endpoints, not the path -- just like the FTC.
- **Green's Theorem**: integral_C (P dx + Q dy) = integral integral_D (Q_x - P_y) dA. *Intuition*: the circulation around a closed curve equals the total curl inside. Boundary behavior encodes interior behavior.
- **Stokes' Theorem**: integral_C F . dr = integral integral_S (curl F) . dS. *Intuition*: generalizes Green's theorem to 3D; the circulation of F around a closed curve equals the flux of curl F through any surface bounded by that curve.
- **Divergence Theorem**: integral integral_S F . dS = integral integral integral_E (div F) dV. *Intuition*: the total outward flux through a closed surface equals the total divergence inside. What flows out must be produced within.

**Worked example**:
> Use Green's Theorem to evaluate integral_C (y^2 dx + x^2 dy) where C is the unit circle traversed counterclockwise. P = y^2, Q = x^2. Q_x - P_y = 2x - 2y. By Green's: integral integral_D (2x - 2y) dA. Switch to polar: integral from 0 to 2*pi integral from 0 to 1 of (2r*cos(theta) - 2r*sin(theta)) * r dr d(theta). The integral of cos(theta) and sin(theta) over [0, 2*pi] are both 0, so the result is 0.

**Essential problems**: OpenStax 6.2 Exercises: #55-70; 6.3 Exercises: #115-130; 6.4 Exercises: #155-168; 6.7 Exercises: #285-298; 6.8 Exercises: #325-338
**Challenge problems**: OpenStax 6.8 Exercises: #345-352

---

## Technique Catalog

| Technique | When to Use | Key Formula/Idea |
|-----------|-------------|------------------|
| Gradient | Find direction of steepest ascent | grad f = (f_x, f_y, f_z) |
| Lagrange Multipliers | Constrained optimization | grad f = lambda * grad g |
| Second Derivative Test | Classify critical points in 2D | Check det(Hessian) and f_xx |
| Polar/Cylindrical/Spherical | Integrate over circular/spherical regions | Include Jacobian factor (r, r, rho^2 sin phi) |
| Green's Theorem | Convert line integral to double integral (2D) | Circulation = integral of curl over region |
| Stokes' Theorem | Convert line integral to surface integral (3D) | Circulation = flux of curl |
| Divergence Theorem | Convert surface integral to volume integral | Flux = integral of divergence |
| Change of Variables | Simplify integration domain | dA = |J| du dv |

## Connections to CS & Algorithms

| Math Concept | CS Application | Repo Link |
|-------------|----------------|-----------|
| Gradient & directional derivative | Gradient descent in ML and deep learning | ML track |
| Jacobian | Backpropagation in neural networks (chain rule for vector functions) | ML track |
| Lagrange multipliers | Constrained optimization (SVMs, regularization) | ML track |
| Multiple integrals | Joint probability distributions, expected values | [07-probability-statistics](../07-probability-statistics/) |
| Divergence / curl | Physics simulation, fluid dynamics in games | systems track |
| Change of variables | Normalizing flows in generative models | ML track |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| ML/AI Roles (OpenAI, DeepMind, Meta) | Gradient descent, backprop Jacobians, constrained optimization | Hard |
| Quantitative Finance | Multivariate optimization for portfolio theory | Hard |
| Graphics/Gaming (NVIDIA, Unity, Epic) | Surface normals, physics simulation, fluid dynamics | Medium |
| Robotics (Boston Dynamics, Tesla) | 3D kinematics, trajectory optimization | Hard |
| A/B Testing (any tech company) | Joint distributions require multiple integration | Medium |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `M04.1` | [Partial derivatives, gradients, gradcheck](01-gradients-and-gradcheck.md) | build | 2 |
| 2 | `M04.2` | [Jacobians, multivariable chain rule, numeric JVP/VJP](02-jacobians-and-the-chain-rule.md) | build | 2 |
| 3 | `S-M04` | [Calculus 3 problem set: vectors, gradients, the chain rule, Hessians, multiple integrals, Lagrange](90-problem-set.md) | solve | 2 |
<!-- /ol:chapters -->
