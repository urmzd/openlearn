# Linear Algebra

## Overview
- **Textbook**: *Linear Algebra* by Jim Hefferon -- https://hefferon.net/linearalgebra/ (GFDL)
- **Supplementary**: [3Blue1Brown Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra), [MIT OCW 18.06 (Strang)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- **Prerequisites**: None
- **Estimated time**: 4 weeks at 10-12 hrs/week

## Key Takeaways
- Solve linear systems systematically via Gauss's method and understand solution geometry
- Grasp vector spaces abstractly -- subspaces, basis, and dimension unify seemingly different structures
- Represent linear maps as matrices and understand the deep link between transformations and matrix multiplication
- Compute eigenvalues/eigenvectors and use diagonalization for efficient repeated computation
- Apply inner products, orthogonality, and projections -- the backbone of least squares and modern ML

## How to Study
- Watch 3Blue1Brown Essence of Linear Algebra before or alongside each chapter for geometric intuition
- Read Hefferon's text and work the in-chapter exercises before moving to the problem sets
- Verify computations by hand first, then check with a tool (Python/NumPy, MATLAB, or SageMath)
- Focus on understanding why theorems hold, not just memorizing procedures
- Strang's MIT lectures are excellent for a second perspective on tricky topics

---

# Concepts & Techniques

## Core Insight

Linear algebra is the study of linear maps between vector spaces. Every matrix represents a
linear transformation, and every linear transformation can be represented by a matrix (once you
choose bases). This duality -- between abstract maps and concrete arrays of numbers -- is the
central organizing idea. Eigenvalues reveal the intrinsic scaling behavior of a transformation,
and inner products let us measure angles and distances, enabling projection and approximation.

## 1. Linear Systems and Gauss's Method

**Textbook sections**: Ch 1, Sections I.1-I.3

**Key definitions**:
- **Linear system**: A set of linear equations a_11*x_1 + ... + a_1n*x_n = b_1, etc.
- **Echelon form**: A matrix where each leading entry is to the right of the leading entry in the row above
- **Reduced echelon form**: Echelon form where each leading entry is 1 and is the only nonzero entry in its column
- **Free variable**: A variable corresponding to a non-pivot column; it parameterizes the solution set

**Key theorems**:
- **Gauss's Method**: Every matrix can be brought to echelon form by a sequence of row operations (swap, scale, replacement). *Intuition*: systematically eliminate variables one at a time, just as you would by hand, but in a disciplined order.
- **Uniqueness of Reduced Echelon Form**: Every matrix has exactly one reduced echelon form. *Intuition*: while the path of row operations may differ, the destination is the same -- the reduced form captures intrinsic information about the system.
- **Solution Set Structure**: The general solution to Ax = b is a particular solution plus the general solution to Ax = 0. *Intuition*: shift the homogeneous solution set to pass through one particular solution.

**Worked example**:
> Solve: x + 2y + z = 4, 2x + 5y + 3z = 10, x + 3y + 2z = 7. Augmented matrix: [[1,2,1|4],[2,5,3|10],[1,3,2|7]]. R2 <- R2-2R1: [[1,2,1|4],[0,1,1|2],[1,3,2|7]]. R3 <- R3-R1: [[1,2,1|4],[0,1,1|2],[0,1,1|3]]. R3 <- R3-R2: [[1,2,1|4],[0,1,1|2],[0,0,0|1]]. Last row says 0 = 1, so the system is inconsistent (no solution).

**Essential problems**: Hefferon Ch 1, Section I.1: #1.1-1.8; Section I.2: #2.10-2.16; Section I.3: #3.18-3.22
**Challenge problems**: Hefferon Ch 1, Section I.3: #3.25-3.28

## 2. Vector Spaces

**Textbook sections**: Ch 2, Sections I-III

**Key definitions**:
- **Vector space**: A set V with addition and scalar multiplication satisfying ten axioms (closure, associativity, commutativity, identity, inverses for addition; closure, associativity, distributivity, identity for scalar multiplication)
- **Subspace**: A nonempty subset of V that is closed under addition and scalar multiplication
- **Linear combination**: v = c_1*v_1 + c_2*v_2 + ... + c_k*v_k
- **Span**: The set of all linear combinations of a collection of vectors
- **Linear independence**: {v_1, ..., v_k} is linearly independent if c_1*v_1 + ... + c_k*v_k = 0 implies all c_i = 0
- **Basis**: A linearly independent set that spans V
- **Dimension**: The number of vectors in any basis for V (well-defined by the next theorem)

**Key theorems**:
- **Basis Theorem**: Any two bases for a finite-dimensional vector space have the same number of elements. *Intuition*: dimension is an intrinsic property of the space, not an artifact of a particular basis choice.
- **Dimension Formula**: If W is a subspace of V, then dim(W) <= dim(V), with equality iff W = V. *Intuition*: subspaces cannot be "bigger" than the ambient space.
- **Expansion/Reduction**: Any linearly independent set can be extended to a basis; any spanning set can be reduced to a basis. *Intuition*: you can always find a basis by adding or removing vectors from what you have.

**Worked example**:
> Show that {(1,0,1), (0,1,1), (1,1,0)} is a basis for R^3. Check independence: c_1(1,0,1) + c_2(0,1,1) + c_3(1,1,0) = (0,0,0) gives c_1+c_3=0, c_2+c_3=0, c_1+c_2=0. From the first two: c_1=c_2. Substituting into the third: 2c_1=0, so c_1=c_2=c_3=0. Three independent vectors in R^3 form a basis.

**Essential problems**: Hefferon Ch 2, Section I: #1.18-1.24; Section II: #1.1-1.12; Section III: #2.10-2.18
**Challenge problems**: Hefferon Ch 2, Section III: #3.14-3.18

## 3. Maps Between Spaces

**Textbook sections**: Ch 3, Sections I-IV

**Key definitions**:
- **Linear transformation (map)**: T: V -> W such that T(c_1*v_1 + c_2*v_2) = c_1*T(v_1) + c_2*T(v_2)
- **Kernel (null space)**: ker(T) = {v in V : T(v) = 0}
- **Range (image)**: range(T) = {T(v) : v in V}
- **Matrix representation**: Rep_{B,D}(T) is the matrix whose columns are the coordinate vectors of T(b_i) with respect to basis D

**Key theorems**:
- **Rank-Nullity Theorem**: dim(ker(T)) + dim(range(T)) = dim(V). *Intuition*: the input space is partitioned into the part that collapses to zero (nullity) and the part that survives (rank). Every dimension is accounted for.
- **Matrix Representation Theorem**: Every linear map between finite-dimensional spaces is represented by a matrix (given a choice of bases), and conversely. *Intuition*: matrices are not just arrays of numbers -- they are linear maps written in coordinates.
- **Change of Basis**: If P is the change-of-basis matrix from B' to B, then the matrix of T in basis B' is P^(-1)AP. *Intuition*: the same transformation looks different in different coordinate systems, but these representations are related by conjugation.

**Worked example**:
> Let T: R^2 -> R^2 be defined by T(x,y) = (x+y, x-y). Find the matrix representation with respect to the standard basis. T(e_1) = T(1,0) = (1,1). T(e_2) = T(0,1) = (1,-1). The matrix is [[1,1],[1,-1]]. Kernel: solve (x+y, x-y) = (0,0), so x = -y and x = y, giving x = y = 0. The kernel is trivial, so T is injective. By Rank-Nullity: rank = 2, so T is also surjective.

**Essential problems**: Hefferon Ch 3, Section I: #1.14-1.20; Section II: #2.10-2.18; Section IV: #1.5-1.12
**Challenge problems**: Hefferon Ch 3, Section IV: #4.1-4.6

## 4. Determinants

**Textbook sections**: Ch 4, Sections I-III

**Key definitions**:
- **Determinant**: For a square matrix A, det(A) is the unique scalar satisfying multilinearity and alternation in the rows, with det(I) = 1
- **Cofactor expansion**: det(A) = Sum over j of (-1)^(i+j) * a_ij * M_ij (expand along row i)
- **Minor**: M_ij is the determinant of the matrix obtained by deleting row i and column j

**Key theorems**:
- **Determinant and Invertibility**: A is invertible iff det(A) != 0. *Intuition*: the determinant measures the signed volume scaling factor of the transformation. Zero volume means collapse -- information is lost.
- **Product Formula**: det(AB) = det(A) * det(B). *Intuition*: composing two transformations multiplies their volume-scaling factors.
- **Determinant of Transpose**: det(A^T) = det(A). *Intuition*: the column picture and row picture of a matrix contain the same volume information.
- **Cramer's Rule**: x_i = det(A_i)/det(A) where A_i replaces column i with b. *Intuition*: each variable can be isolated by a ratio of determinants -- elegant but computationally expensive for large systems.

**Worked example**:
> Compute det([[2,1,3],[0,4,1],[1,0,2]]) by cofactor expansion along the first column. det = 2*det([[4,1],[0,2]]) - 0*det([[1,3],[0,2]]) + 1*det([[1,3],[4,1]]) = 2*(8-0) - 0 + 1*(1-12) = 16 - 11 = 5. Since det != 0, the matrix is invertible.

**Essential problems**: Hefferon Ch 4, Section I: #1.1-1.8; Section II: #1.1-1.6; Section III: #1.1-1.8
**Challenge problems**: Hefferon Ch 4, Section III: #2.1-2.4

## 5. Eigenvalues and Eigenvectors

**Textbook sections**: Ch 5, Sections I-II

**Key definitions**:
- **Eigenvalue**: A scalar lambda such that Av = lambda*v for some nonzero v
- **Eigenvector**: A nonzero vector v such that Av = lambda*v
- **Characteristic polynomial**: p(lambda) = det(A - lambda*I); roots are eigenvalues
- **Eigenspace**: E_lambda = ker(A - lambda*I), the set of all eigenvectors for lambda (plus zero)
- **Diagonalizable**: A matrix A is diagonalizable if there exists an invertible P and diagonal D with A = PDP^(-1)

**Key theorems**:
- **Diagonalization Theorem**: An n x n matrix is diagonalizable iff it has n linearly independent eigenvectors. *Intuition*: if we can find a basis of eigenvectors, the matrix acts as pure scaling along those directions.
- **Distinct Eigenvalues**: If A has n distinct eigenvalues, then A is diagonalizable. *Intuition*: eigenvectors from different eigenvalues are automatically independent -- no coincidence is needed.
- **Cayley-Hamilton Theorem**: Every square matrix satisfies its own characteristic polynomial: p(A) = 0. *Intuition*: the matrix, when "plugged into" the polynomial that defines its eigenvalues, annihilates itself.
- **Spectral Theorem (Real Symmetric)**: A real symmetric matrix has real eigenvalues and orthogonal eigenvectors. *Intuition*: symmetric matrices are the "nicest" -- they stretch along perpendicular directions.

**Worked example**:
> Diagonalize A = [[4,1],[2,3]]. Characteristic polynomial: det(A - lambda*I) = (4-lambda)(3-lambda) - 2 = lambda^2 - 7*lambda + 10 = (lambda-5)(lambda-2). Eigenvalues: lambda_1 = 5, lambda_2 = 2. For lambda = 5: (A-5I)v = 0 gives [[-1,1],[2,-2]]v = 0, so v_1 = (1,1). For lambda = 2: (A-2I)v = 0 gives [[2,1],[2,1]]v = 0, so v_2 = (1,-2). Then A = PDP^(-1) where P = [[1,1],[1,-2]], D = [[5,0],[0,2]].

**Essential problems**: Hefferon Ch 5, Section I: #1.1-1.10; Section II: #1.1-1.8, #2.1-2.6
**Challenge problems**: Hefferon Ch 5, Section II: #3.1-3.6

## 6. Inner Product Spaces

**Textbook sections**: Ch 3, Section VI (Hefferon) and supplementary from Strang Ch 4

**Key definitions**:
- **Inner product (dot product)**: <u, v> = u_1*v_1 + ... + u_n*v_n (in R^n); more generally, a positive-definite symmetric bilinear form
- **Norm**: ||v|| = sqrt(<v, v>)
- **Orthogonal**: u and v are orthogonal if <u, v> = 0
- **Orthonormal basis**: A basis {q_1, ..., q_n} where <q_i, q_j> = 0 for i != j and ||q_i|| = 1
- **Orthogonal projection**: proj_W(v) = Sum of <v, q_i> * q_i for an orthonormal basis {q_i} of W
- **Least squares solution**: The x_hat that minimizes ||Ax - b||^2, given by A^T A x_hat = A^T b

**Key theorems**:
- **Cauchy-Schwarz Inequality**: |<u, v>| <= ||u|| * ||v||, with equality iff u and v are parallel. *Intuition*: the projection of u onto v cannot exceed the length of u -- the cosine factor is at most 1.
- **Gram-Schmidt Process**: Given a basis {v_1, ..., v_k}, produce an orthonormal basis by: q_1 = v_1/||v_1||, then subtract projections and normalize at each step. *Intuition*: at each step, strip away the components along previously found directions, leaving only the genuinely new direction.
- **Best Approximation Theorem**: The projection proj_W(v) is the closest point in W to v. *Intuition*: the error vector (v - proj_W(v)) is orthogonal to W -- you cannot reduce the error by moving within W.
- **Normal Equations**: The least squares solution satisfies A^T A x_hat = A^T b. *Intuition*: project b onto the column space of A; the residual must be orthogonal to every column.

**Worked example**:
> Apply Gram-Schmidt to {(1,1,0), (1,0,1), (0,1,1)} in R^3. Step 1: q_1 = (1,1,0)/sqrt(2). Step 2: v_2' = (1,0,1) - <(1,0,1), q_1>*q_1 = (1,0,1) - (1/sqrt(2))*(1/sqrt(2),1/sqrt(2),0) = (1,0,1) - (1/2,1/2,0) = (1/2,-1/2,1). q_2 = (1/2,-1/2,1)/||(1/2,-1/2,1)|| = (1/2,-1/2,1)/sqrt(3/2) = (1,-1,2)/sqrt(6). Step 3: project (0,1,1) onto q_1 and q_2, subtract, and normalize.

**Essential problems**: Strang 18.06 Problem Set 4: #4.1-4.4; Hefferon Ch 3, Section VI: #1.1-1.10
**Challenge problems**: Strang 18.06 Problem Set 4: #4.5-4.8; Least squares fitting exercises

---

## Technique Catalog

| Technique | When to Use | Key Formula/Idea |
|-----------|-------------|------------------|
| Gaussian Elimination | Solve linear systems | Row reduce to echelon form |
| Back Substitution | Read off solutions from echelon form | Work bottom to top |
| Rank-Nullity | Relate kernel and image dimensions | dim(ker) + dim(range) = dim(domain) |
| Cofactor Expansion | Compute determinants | Expand along row/column with most zeros |
| Characteristic Polynomial | Find eigenvalues | Solve det(A - lambda*I) = 0 |
| Diagonalization | Simplify matrix powers A^k | A^k = P D^k P^(-1) |
| Gram-Schmidt | Orthonormalize a basis | Subtract projections, normalize |
| Least Squares | Solve overdetermined systems | Solve A^T A x_hat = A^T b |

## Connections to CS & Algorithms

| Math Concept | CS Application | Repo Link |
|-------------|----------------|-----------|
| Matrix multiplication | Computer graphics (rotation, scaling, projection) | systems track |
| Eigenvalues | Google PageRank (dominant eigenvector of link matrix) | algorithms track |
| SVD / PCA | Dimensionality reduction, recommendation systems | ML track |
| Least squares | Linear regression, curve fitting | [07-probability-statistics](../07-probability-statistics/) |
| Linear independence | Feature selection in ML, avoiding multicollinearity | ML track |
| Change of basis | Coordinate transforms, Fourier/wavelet bases | systems track |
| Sparse matrices | Graph algorithms (adjacency matrices), databases | algorithms track |

## Company Relevance

| Company | How This Appears | Difficulty |
|---------|-----------------|------------|
| Google | PageRank is the dominant eigenvector of a stochastic matrix | Hard |
| Quantitative Finance | Portfolio optimization, covariance matrices, PCA for risk | Hard |
| ML/AI Roles (OpenAI, DeepMind) | Embeddings, attention as matrix ops, backprop is chain of linear maps | Hard |
| Graphics (Pixar, NVIDIA, Unity) | Transformation matrices, projection, quaternions | Medium |
| Robotics (Boston Dynamics, Tesla) | Kinematics use rotation matrices and coordinate transforms | Medium |
| Amazon/Netflix | Recommendation engines use matrix factorization | Medium |

## Chapters

<!-- ol:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `M03.1` | [Vectors, matrices, row-major layout, and matmul in Python](01-vectors-matrices-and-matmul.md) | build | 1 |
| 2 | `M03.2` | [Gaussian elimination and LU with partial pivoting](02-gaussian-elimination-and-lu.md) | build | 2 |
| 3 | `M03.3` | [Orthogonality, Householder QR, and orthogonal initialization](03-orthogonality-and-householder-qr.md) | build | 2 |
| 4 | `M03.4` | [Eigenvalues, power iteration, and the spectral radius](04-eigenvalues-and-power-iteration.md) | build | 2 |
| 5 | `M03.5` | [SVD, Eckart-Young low rank, and least squares](05-svd-low-rank-and-least-squares.md) | build | 3 |
| 6 | `M03.6` | [Inner products, projections, cosine similarity, and top-k](06-inner-products-projections-cosine-and-top-k.md) | build | 3 |
| 7 | `S-M03a` | [Linear algebra problem set, part a: elimination, LU, bases, rank, determinants, eigenvalues, QR](90-problem-set-a.md) | solve | 2 |
| 8 | `S-M03b` | [Linear algebra problem set, part b: inner products, SVD and low rank, matmul accounting and the roofline](91-problem-set-b.md) | solve | 3 |
<!-- /ol:chapters -->
