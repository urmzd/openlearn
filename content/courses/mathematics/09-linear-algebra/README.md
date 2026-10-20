# Linear Algebra

Linear algebra is the study of linear maps between vector spaces. Every matrix represents a linear transformation, and every linear transformation can be represented by a matrix once you choose bases. That duality, between abstract maps and concrete arrays of numbers, is the central organizing idea of the subject. Eigenvalues reveal the intrinsic scaling behaviour of a transformation, and inner products let you measure angles and distances, which is what makes projection and approximation possible.

For anyone building machine learning systems, this is the mathematics that everything else runs on. A neural network layer is a matrix product. An embedding is a vector, and similarity between embeddings is an angle. Whether a recurrent network's gradients explode is a question about eigenvalues; how to compress a weight matrix or start a low-rank adapter is a question about singular values; how to fit a scaling law to a handful of training runs is a least-squares problem. This chapter builds each of those tools from first principles, in code you write yourself, and explains why the numerically careful version is the right one.

The arc of the chapter follows the tools in the order they build on each other. It starts with the matrix product and how matrices are laid out in memory. It then learns to undo a product by Gaussian elimination and LU, and to build orthogonal matrices by Householder QR. QR becomes the engine for eigenvalues and the spectral radius, which in turn leads to the singular value decomposition, low-rank approximation, and least squares. The last section returns to the geometry of inner products, cosine similarity, and nearest-neighbour search. Two problem sets let you work the same ideas by hand.

> **Learning objectives**
> - Solve linear systems systematically by elimination, read off an LU factorization with partial pivoting, and understand the geometry of solution sets.
> - Work with vector spaces abstractly: subspaces, bases, and dimension, and the rank-nullity theorem that ties a map's kernel to its image.
> - Represent linear maps as matrices, compose them by multiplication, and explain the connection between transformations and matrix products.
> - Compute eigenvalues and eigenvectors, use them to predict the behaviour of repeated multiplication, and compute the spectral radius of matrices with complex eigenvalues.
> - Apply inner products, orthogonality, projections, the SVD, and least squares, the backbone of low-rank methods and of modern machine learning.

## Before you begin

The chapter assumes no previous linear algebra. You should be comfortable with high-school algebra and with Python and NumPy arrays, shapes, and views, covered in [Python and NumPy](../../programming-and-algorithms/01-language-primers/01-python-and-numpy.md) in *Programming and Algorithms*. Complex numbers and 2D rotations from [Trigonometry, rotations, complex numbers, and Euler's formula](../06-precalculus/02-trig-rotations-and-eulers-formula.md) help with the eigenvalue section, and the orthogonal initializer draws normal random numbers from [Random variables, expectation, variance, and normal draws by Box-Muller](../13-probability-statistics/01-random-variables-and-the-normal.md). Throughout, it pays to verify computations by hand first and then check them with NumPy, and to focus on why a theorem holds rather than memorizing the procedure.

## In this chapter

1. [Vectors, matrices, row-major layout, and matmul](01-vectors-matrices-and-matmul.md). The matrix product from its definition, how strides and transposes decide which stored element is read, and why `beta == 0` must never read the old output. The result is the reference that faster kernels are checked against.
2. [Gaussian elimination and LU with partial pivoting](02-gaussian-elimination-and-lu.md). Undoing a matrix product: elimination as a factorization $PA = LU$, why pivoting is required for correct answers, and why you factor once and solve many times.
3. [Orthogonality, Householder QR, and orthogonal initialization](03-orthogonality-and-householder-qr.md). Matrices that preserve lengths, the Householder construction that stays orthogonal to machine precision, the sign rule that makes QR unique, and an orthogonal initializer for recurrent weights.
4. [Eigenvalues, power iteration, and the spectral radius](04-eigenvalues-and-power-iteration.md). Power iteration, inverse iteration with one LU factorization, and the QR algorithm for rotations, where power iteration never converges. The spectral radius decides whether gradients explode or vanish.
5. [SVD, Eckart-Young low rank, and least squares](05-svd-low-rank-and-least-squares.md). The singular value decomposition by one-sided Jacobi, the best low-rank approximation and the balanced split used by PiSSA, and least squares by QR instead of the normal equations.
6. [Inner products, projections, cosine similarity, and top-k](06-inner-products-projections-cosine-and-top-k.md). Angles between vectors, safe handling of zero vectors, projection onto a line, and exact top-$k$ search that is reproducible across runs and languages.
7. [Linear algebra problem set, part a](07-problem-set-a.md). Forty-eight problems with worked answers on elimination, LU, bases, rank, determinants, eigenvalues, and QR. Best after sections 2 to 4.
8. [Linear algebra problem set, part b](08-problem-set-b.md). Problems with worked answers on inner products, SVD and low rank, and the FLOP and byte accounting behind the roofline model. Best after sections 5 and 6.

## The subject at a glance

The sections above build specific tools. This overview collects the classical ideas they rest on, each with a short worked example, so you have the whole subject in view. The textbook sections named in each part refer to Hefferon's free *Linear Algebra*, listed under further reading, which covers each topic in full with exercises.

### Linear systems and Gauss's method

A **linear system** is a set of linear equations $a_{11}x_1 + \dots + a_{1n}x_n = b_1$, and so on. A matrix is in **echelon form** when each leading entry is to the right of the leading entry in the row above; in **reduced echelon form** each leading entry is also 1 and the only nonzero entry in its column. A **free variable** corresponds to a non-pivot column and parameterizes the solution set.

Three results organize the topic. Gauss's method says every matrix can be brought to echelon form by row operations (swap, scale, replacement): you eliminate variables one at a time, as you would by hand, but in a disciplined order. The reduced echelon form is unique: the path of row operations may differ, but the destination captures intrinsic information about the system. And the general solution of $Ax = b$ is one particular solution plus the general solution of $Ax = 0$: the homogeneous solution set, shifted to pass through a particular solution.

For example, solve $x + 2y + z = 4$, $2x + 5y + 3z = 10$, $x + 3y + 2z = 7$. The augmented matrix is $\left(\begin{smallmatrix} 1 & 2 & 1 & | & 4 \\ 2 & 5 & 3 & | & 10 \\ 1 & 3 & 2 & | & 7 \end{smallmatrix}\right)$. $R_2 \leftarrow R_2 - 2R_1$ gives $(0, 1, 1 \mid 2)$; $R_3 \leftarrow R_3 - R_1$ gives $(0, 1, 1 \mid 3)$; $R_3 \leftarrow R_3 - R_2$ gives $(0, 0, 0 \mid 1)$. The last row says $0 = 1$, so the system is inconsistent: it has no solution. (Hefferon ch. 1, sections I.1 to I.3; essential exercises I.1 #1.1 to 1.8, I.2 #2.10 to 2.16, I.3 #3.18 to 3.22; challenge I.3 #3.25 to 3.28.)

### Vector spaces

A **vector space** is a set $V$ with addition and scalar multiplication satisfying ten axioms: closure, associativity, commutativity, identity, and inverses for addition; closure, associativity, distributivity, and identity for scalar multiplication. A **subspace** is a nonempty subset closed under both operations. A **linear combination** is $c_1v_1 + \dots + c_kv_k$, and the **span** of a set is all its linear combinations. Vectors are **linearly independent** if $c_1v_1 + \dots + c_kv_k = 0$ implies every $c_i = 0$. A **basis** is an independent spanning set, and the **dimension** is the number of vectors in any basis.

That definition of dimension is well defined because of the basis theorem: any two bases of a finite-dimensional space have the same number of elements, so dimension is intrinsic to the space, not an artifact of a basis choice. If $W$ is a subspace of $V$, then $\dim W \le \dim V$, with equality exactly when $W = V$. And any independent set can be extended to a basis, while any spanning set can be reduced to one.

For example, $\{(1,0,1), (0,1,1), (1,1,0)\}$ is a basis of $\mathbb{R}^3$. Independence: $c_1(1,0,1) + c_2(0,1,1) + c_3(1,1,0) = 0$ gives $c_1 + c_3 = 0$, $c_2 + c_3 = 0$, $c_1 + c_2 = 0$. The first two give $c_1 = c_2$; the third then gives $2c_1 = 0$, so $c_1 = c_2 = c_3 = 0$. Three independent vectors in $\mathbb{R}^3$ form a basis. (Hefferon ch. 2, sections I to III; essential I #1.18 to 1.24, II #1.1 to 1.12, III #2.10 to 2.18; challenge III #3.14 to 3.18.)

### Maps between spaces

A **linear map** $T: V \to W$ satisfies $T(c_1v_1 + c_2v_2) = c_1T(v_1) + c_2T(v_2)$. Its **kernel** is $\ker T = \{v : T(v) = 0\}$ and its **range** is $\{T(v) : v \in V\}$. Its **matrix representation** $\operatorname{Rep}_{B,D}(T)$ has as columns the coordinate vectors of $T(b_i)$ with respect to the basis $D$.

The **rank-nullity theorem**, $\dim \ker T + \dim \operatorname{range} T = \dim V$, says the input space splits into the part that collapses to zero and the part that survives, with every dimension accounted for. Every linear map between finite-dimensional spaces is represented by a matrix, given bases, and conversely: matrices are linear maps written in coordinates. And if $P$ is the change-of-basis matrix from $B'$ to $B$, the matrix of $T$ in $B'$ is $P^{-1}AP$: the same transformation looks different in different coordinates, and the representations are related by conjugation.

For example, $T(x, y) = (x + y, x - y)$ sends $e_1 \mapsto (1, 1)$ and $e_2 \mapsto (1, -1)$, so its matrix is $\left(\begin{smallmatrix} 1 & 1 \\ 1 & -1 \end{smallmatrix}\right)$. Its kernel solves $x + y = 0$ and $x - y = 0$, so $x = y = 0$: the kernel is trivial and $T$ is injective. By rank-nullity the rank is 2, so $T$ is also surjective. (Hefferon ch. 3, sections I to IV; essential I #1.14 to 1.20, II #2.10 to 2.18, IV #1.5 to 1.12; challenge IV #4.1 to 4.6.)

### Determinants

The **determinant** of a square matrix is the unique scalar function that is multilinear and alternating in the rows, with $\det I = 1$. **Cofactor expansion** along row $i$ computes it as $\sum_j (-1)^{i+j} a_{ij} M_{ij}$, where the **minor** $M_{ij}$ is the determinant of the matrix with row $i$ and column $j$ deleted.

$A$ is invertible exactly when $\det A \ne 0$: the determinant is the signed volume scaling factor of the transformation, and zero volume means collapse, with information lost. $\det(AB) = \det A \det B$, because composing transformations multiplies their volume factors. $\det A^\top = \det A$: the row and column pictures carry the same volume information. And Cramer's rule, $x_i = \det A_i / \det A$ with column $i$ replaced by $b$, isolates each variable by a ratio of determinants, elegant but computationally expensive for large systems.

For example, expand $\det\left(\begin{smallmatrix} 2 & 1 & 3 \\ 0 & 4 & 1 \\ 1 & 0 & 2 \end{smallmatrix}\right)$ along the first column: $2 \cdot (8 - 0) - 0 + 1 \cdot (1 - 12) = 16 - 11 = 5$. Since the determinant is nonzero, the matrix is invertible. (Hefferon ch. 4, sections I to III; essential I #1.1 to 1.8, II #1.1 to 1.6, III #1.1 to 1.8; challenge III #2.1 to 2.4.)

### Eigenvalues and eigenvectors

An **eigenvalue** is a scalar $\lambda$ with $Av = \lambda v$ for some nonzero **eigenvector** $v$. The eigenvalues are the roots of the **characteristic polynomial** $p(\lambda) = \det(A - \lambda I)$, and the **eigenspace** $E_\lambda = \ker(A - \lambda I)$ is the set of eigenvectors for $\lambda$ together with zero. $A$ is **diagonalizable** if $A = PDP^{-1}$ with $P$ invertible and $D$ diagonal.

An $n \times n$ matrix is diagonalizable exactly when it has $n$ independent eigenvectors: with a basis of eigenvectors, the matrix acts as pure scaling along those directions. If it has $n$ distinct eigenvalues, it is diagonalizable, because eigenvectors for different eigenvalues are automatically independent. The Cayley-Hamilton theorem says every square matrix satisfies its own characteristic polynomial, $p(A) = 0$. And the spectral theorem says a real symmetric matrix has real eigenvalues and orthogonal eigenvectors: symmetric matrices stretch along perpendicular directions. Diagonalization makes powers cheap, $A^k = PD^kP^{-1}$.

For example, diagonalize $A = \left(\begin{smallmatrix} 4 & 1 \\ 2 & 3 \end{smallmatrix}\right)$. $\det(A - \lambda I) = (4 - \lambda)(3 - \lambda) - 2 = \lambda^2 - 7\lambda + 10 = (\lambda - 5)(\lambda - 2)$. For $\lambda = 5$, $\left(\begin{smallmatrix} -1 & 1 \\ 2 & -2 \end{smallmatrix}\right)v = 0$ gives $v_1 = (1, 1)$; for $\lambda = 2$, $\left(\begin{smallmatrix} 2 & 1 \\ 2 & 1 \end{smallmatrix}\right)v = 0$ gives $v_2 = (1, -2)$. So $A = PDP^{-1}$ with $P = \left(\begin{smallmatrix} 1 & 1 \\ 1 & -2 \end{smallmatrix}\right)$ and $D = \operatorname{diag}(5, 2)$. (Hefferon ch. 5, sections I and II; essential I #1.1 to 1.10, II #1.1 to 1.8 and #2.1 to 2.6; challenge II #3.1 to 3.6.)

### Inner product spaces

The **inner product** in $\mathbb{R}^n$ is $\langle u, v \rangle = u_1v_1 + \dots + u_nv_n$; more generally it is a positive-definite symmetric bilinear form. The **norm** is $\lVert v \rVert = \sqrt{\langle v, v \rangle}$, and $u$ and $v$ are **orthogonal** when $\langle u, v \rangle = 0$. An **orthonormal basis** has $\langle q_i, q_j \rangle = 0$ for $i \ne j$ and $\lVert q_i \rVert = 1$. The **orthogonal projection** onto a subspace $W$ with orthonormal basis $\{q_i\}$ is $\operatorname{proj}_W v = \sum_i \langle v, q_i \rangle q_i$, and the **least-squares solution** $\hat{x}$ minimizes $\lVert Ax - b \rVert^2$.

The Cauchy-Schwarz inequality $\lvert\langle u, v \rangle\rvert \le \lVert u \rVert \lVert v \rVert$, with equality exactly when $u$ and $v$ are parallel, says the projection of one vector onto another cannot exceed its length. Gram-Schmidt turns a basis into an orthonormal one: at each step, strip away the components along the directions already found and normalize what is genuinely new. The projection $\operatorname{proj}_W v$ is the closest point in $W$ to $v$, because the error $v - \operatorname{proj}_W v$ is orthogonal to $W$. And the least-squares solution satisfies the normal equations $A^\top A\hat{x} = A^\top b$: project $b$ onto the column space, and the residual is orthogonal to every column. (Section 5 explains why, in floating point, you solve them through QR instead.)

For example, apply Gram-Schmidt to $\{(1,1,0), (1,0,1), (0,1,1)\}$. First, $q_1 = (1,1,0)/\sqrt{2}$. Next, $v_2' = (1,0,1) - \langle (1,0,1), q_1 \rangle q_1 = (1,0,1) - (1/2, 1/2, 0) = (1/2, -1/2, 1)$, and $q_2 = (1/2, -1/2, 1)/\sqrt{3/2} = (1, -1, 2)/\sqrt{6}$. The third step projects $(0,1,1)$ onto $q_1$ and $q_2$, subtracts, and normalizes. (Hefferon ch. 3, section VI, exercises #1.1 to 1.10, and Strang ch. 4; MIT 18.06 problem set 4, #4.1 to 4.4, challenge #4.5 to 4.8, and least-squares fitting exercises.)

### Techniques at a glance

The classical techniques, side by side:

| Technique | When to use | Key formula or idea |
|---|---|---|
| Gaussian elimination | solve linear systems | row reduce to echelon form |
| Back substitution | read off solutions from echelon form | work bottom to top |
| Rank-nullity | relate kernel and image dimensions | $\dim\ker + \dim\operatorname{range} = \dim(\text{domain})$ |
| Cofactor expansion | compute determinants | expand along the row or column with the most zeros |
| Characteristic polynomial | find eigenvalues | solve $\det(A - \lambda I) = 0$ |
| Diagonalization | simplify matrix powers $A^k$ | $A^k = PD^kP^{-1}$ |
| Gram-Schmidt | orthonormalize a basis | subtract projections, normalize |
| Least squares | solve overdetermined systems | $A^\top A\hat{x} = A^\top b$, solved stably by QR |

## Where linear algebra shows up

These ideas reach well beyond this book. Matrix multiplication drives computer graphics (rotation, scaling, projection) and every layer of a neural network, and attention is a sequence of matrix operations. Google's PageRank is the dominant eigenvector of a stochastic link matrix. The SVD and PCA underlie dimensionality reduction and recommendation systems, where matrix factorization predicts ratings. Least squares is linear regression and curve fitting, developed further in the [Probability and Statistics](../13-probability-statistics/README.md) chapter. Linear independence is the issue behind multicollinearity in feature selection. Change of basis is coordinate transforms and the Fourier and wavelet bases of [Signals and Images](../18-signals-and-images/README.md). Sparse matrices represent graphs as adjacency matrices. In quantitative finance, covariance matrices and PCA drive portfolio optimization and risk models; in robotics, kinematics is a chain of rotation matrices and coordinate transforms.

## Further reading

- Jim Hefferon, [*Linear Algebra*](https://hefferon.net/linearalgebra/) (free, GFDL): the textbook the overview above follows, with exercises for every topic.
- 3Blue1Brown, [*Essence of Linear Algebra*](https://www.3blue1brown.com/topics/linear-algebra): short videos that build geometric intuition; worth watching before or alongside each section.
- Gilbert Strang, [MIT OCW 18.06](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/), with *Introduction to Linear Algebra*: an excellent second perspective on tricky topics.
- Lloyd Trefethen and David Bau, *Numerical Linear Algebra*: the numerical side, from QR and pivoting to eigenvalue algorithms.
