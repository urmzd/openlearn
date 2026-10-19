# Linear algebra problem set, part a: elimination, LU, bases, rank, determinants, eigenvalues, QR

The sections before this one write code that factors matrices. [Gaussian elimination and LU with partial pivoting](02-gaussian-elimination-and-lu.md) builds LU, [Orthogonality, Householder QR, and orthogonal initialization](03-orthogonality-and-householder-qr.md) builds QR and an orthogonal initializer, and [Eigenvalues, power iteration, and the spectral radius](04-eigenvalues-and-power-iteration.md) builds power iteration and the spectral radius that [Vanilla RNN with manual BPTT and truncation](../../build-an-llm/p03-recurrent/01-vanilla-rnn-manual-bptt.md) in *Build Your Own LLM System* uses to diagnose exploding gradients. Comparing those factors with NumPy's is only meaningful if you know what the factors are supposed to be and why they exist: why pivoting is needed, what rank and null space mean for a least-squares fit, why an eigenvalue above 1 makes a recurrent network blow up.

This problem set works those ideas by hand, on matrices small enough to check on paper. It is a pen-and-paper set with no code to write. It assumes the vectors, matrices, and matrix product of [Vectors, matrices, row-major layout, and matmul](01-vectors-matrices-and-matmul.md) and the overview in the [chapter opening](README.md), and it fits best after the LU, QR, and power-iteration sections. Its sequel, the [linear algebra problem set, part b](91-problem-set-b.md), covers SVD, projections, and the roofline, and follows the SVD and inner-product sections.

Every problem is followed by a collapsed worked answer with the final result, the derivation, and the common wrong answers and why they are wrong. There are 44 exact answers and 4 proofs (q10, q20, q28, q44); model proofs for those four are also in [`course/solutions/solve/S-M03a/`](../../../../course/solutions/solve/S-M03a/). Try each problem before opening its answer.

> **Learning objectives**
> - Carry out elimination by hand, read off $L$ and $U$, and apply partial pivoting.
> - Test vectors for independence, find bases of column and null spaces, and use rank-nullity.
> - Build the matrix of a linear map from the images of the basis vectors and compose maps in the right order.
> - Compute determinants and eigenvalues of small matrices and connect them to volume, singularity, and repeated multiplication.
> - Project onto a line, orthonormalize with Gram-Schmidt, and solve a small least-squares problem.

## The ideas in brief

These are the facts the problems draw on, gathered in one place. The notation is standard:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $A \in \mathbb{R}^{m \times n}$ | a matrix with $m$ rows and $n$ columns | matrix |
| $x, b$ | the unknown and the right-hand side of $Ax = b$ | vectors |
| $P, L, U$ | permutation, unit lower triangular, upper triangular: $PA = LU$ | $n \times n$ |
| $\operatorname{span}\{v_i\}$ | all linear combinations $\sum c_i v_i$ | subspace |
| $\dim V$ | the number of vectors in any basis of $V$ | integer |
| $\operatorname{rank} A$ | the dimension of the column space (equal to that of the row space) | integer |
| $N(A)$ | the null space $\{x : Ax = 0\}$; its dimension is the nullity | subspace |
| $\det A$ | the determinant of a square matrix | real |
| $\lambda, v$ | an eigenvalue and an eigenvector: $Av = \lambda v$, $v \ne 0$ | scalar, vector |
| $\rho(A)$ | the spectral radius, $\max |\lambda|$ | real |
| $Q, R$ | orthonormal columns and upper triangular: $A = QR$ | matrices |

### Elimination and LU

Gaussian elimination subtracts multiples of a pivot row from the rows below it until the matrix is upper triangular, $U$. Recording each multiplier $\ell_{ij}$ (the amount of row $j$ subtracted from row $i$) in a unit lower triangular $L$ gives $A = LU$, so solving $Ax = b$ becomes two triangular solves. If a pivot is 0 the rows must be exchanged, and if it is merely small, dividing by it amplifies rounding errors; **partial pivoting** swaps the row with the largest $|a_{ij}|$ in the current column into the pivot position, giving $PA = LU$ with every $|\ell_{ij}| \le 1$. The **reduced row echelon form** continues until each pivot is 1 and the only nonzero entry in its column. A system has no solution (inconsistent), exactly one, or infinitely many (a free variable).

### Span, independence, basis, dimension

Vectors are **linearly independent** when $\sum c_i v_i = 0$ forces every $c_i = 0$. A **basis** of a subspace is an independent spanning set, and every basis has the same size, the **dimension**: a basis is a minimal spanning set, and any $n + 1$ vectors in $\mathbb{R}^n$ are dependent. The pivot columns of $A$ are a basis of its column space; the special solutions (one per free variable) are a basis of its null space. **Coordinates** in a basis are the unique coefficients $c$ with $\sum c_i v_i = x$: solve a linear system.

### Linear maps and rank-nullity

A map $T$ is **linear** when $T(x + y) = T(x) + T(y)$ and $T(cx) = cT(x)$; then $T(x) = Ax$ where column $j$ of $A$ is $T(e_j)$. Composition is multiplication: $S \circ T$ has matrix $S\,T$ (apply $T$ first, so it is on the right). For $A$ with $n$ columns, **rank-nullity** says $\operatorname{rank} A + \dim N(A) = n$: every column either adds a direction to the image or a direction to the null space.

### Determinants

The determinant is the signed volume scale factor of $x \mapsto Ax$: $|\det [u\ v]|$ is the area of the parallelogram spanned by $u$ and $v$. It is multiplicative, $\det(AB) = \det A \det B$, so $\det(A^{-1}) = 1/\det A$; it scales as $\det(cA) = c^n \det A$ for $n \times n$ $A$; a triangular matrix's determinant is the product of its diagonal; and $\det A = 0$ exactly when $A$ is singular.

### Eigenvalues and eigenvectors

$\lambda$ is an eigenvalue when $A - \lambda I$ is singular, that is, a root of the **characteristic polynomial** $\det(tI - A)$; the product of the eigenvalues is $\det A$ and their sum is the trace. A real matrix may have complex eigenvalues (a rotation moves every real vector). If $v$ is an eigenvector, $A^k v = \lambda^k v$, so the largest $|\lambda|$, the **spectral radius** $\rho(A)$, decides whether repeated application grows ($\rho > 1$) or shrinks ($\rho < 1$). **Power iteration** repeats $v \leftarrow Av / \lVert Av \rVert$; the component along the second eigenvector shrinks by $|\lambda_2/\lambda_1|$ per step.

### Orthogonality and QR

The projection of $b$ onto the line through $u$ is $\frac{u \cdot b}{u \cdot u} u$. **Gram-Schmidt** subtracts from each column its projections onto the previous orthonormal vectors and normalizes the rest, giving $A = QR$; requiring a positive diagonal of $R$ makes the factorization unique. Least squares minimizes $\lVert Ax - b \rVert^2$; its solution satisfies the **normal equations** $A^\top A x = A^\top b$, which QR solves stably as $Rx = Q^\top b$, as in [SVD, Eckart-Young low rank, and least squares](05-svd-low-rank-and-least-squares.md).

## Two examples worked by hand

Before the set itself, here are two siblings of q6, q7, and q36. They are not problems of the set, but they show the level of detail the answers use.

**LU with partial pivoting.** $A = \begin{pmatrix} 2 & 1 \\ 6 & 4 \end{pmatrix}$. The largest entry in column 1 is 6, in row 2, so swap: $P = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, $PA = \begin{pmatrix} 6 & 4 \\ 2 & 1 \end{pmatrix}$. The multiplier is $\ell_{21} = 2/6 = 1/3$, and row 2 becomes $(2, 1) - \frac13 (6, 4) = (0, -1/3)$. So $L = \begin{pmatrix} 1 & 0 \\ 1/3 & 1 \end{pmatrix}$, $U = \begin{pmatrix} 6 & 4 \\ 0 & -1/3 \end{pmatrix}$. Check: $LU = \begin{pmatrix} 6 & 4 \\ 2 & 4/3 - 1/3 \end{pmatrix} = PA$.

**Eigenvalues and an eigenvector.** $B = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$ is triangular, so $\det(tI - B) = (t - 3)(t - 2)$ and the eigenvalues are 3 and 2. For $\lambda = 2$: $(B - 2I)v = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix} v = 0$ gives $v = (1, -1)$, or any nonzero multiple such as $(2, -2)$. The product of the eigenvalues is $6 = \det B$.

## How the answers are graded

The problems are grouped by topic, in the order of the summary above. Answers are exact ($1/\sqrt2$, not 0.7071). A vector is one column; a basis answer is correct if it is any basis of the right space; an eigenvector is correct up to a nonzero scalar. The tag after each problem says what kind of answer is expected.

## Gaussian elimination and LU

**q1.** Solve $x + 2y = 5$, $3x + 4y = 6$. Give $(x, y)$. `[vector]`

<details><summary>Answer</summary>

**$(x, y) = (-4, \frac92)$.** Subtract 3 times the first equation from the second: $-2y = -9$, so $y = \frac92$, and back substitution gives $x = 5 - 2 \cdot \frac92 = -4$. Check: $-4 + 9 = 5$, $-12 + 18 = 6$. Writing $(\frac92, -4)$ swaps the unknowns.

</details>

**q2.** Solve $x + y + z = 6$, $2x - y + z = 3$, $x + 2y - z = 2$. Give $(x, y, z)$. `[vector]`

<details><summary>Answer</summary>

**$(1, 2, 3)$.** Eliminate $x$: row 2 minus 2 row 1 gives $-3y - z = -9$; row 3 minus row 1 gives $y - 2z = -4$. From the second, $y = 2z - 4$; substituting, $-3(2z - 4) - z = -9$, so $z = 3$, $y = 2$, $x = 1$. Check all three: $6$, $2 - 2 + 3 = 3$, $1 + 4 - 3 = 2$. $(3, 2, 1)$ reverses the order.

</details>

**q3.** Give the reduced row echelon form of $\begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 7 \end{pmatrix}$. `[matrix]`

<details><summary>Answer</summary>

**$\begin{pmatrix} 1 & 2 & 0 \\ 0 & 0 & 1 \end{pmatrix}$.** Row 2 minus 2 row 1 gives $(0, 0, 1)$; then row 1 minus 3 row 2 clears the entry above that pivot. Stopping at $\begin{pmatrix} 1 & 2 & 3 \\ 0 & 0 & 1 \end{pmatrix}$ gives an echelon form, not the reduced one: the reduced form needs zeros above every pivot.

</details>

**q4.** $A = \begin{pmatrix} 2 & 1 \\ 4 & 5 \end{pmatrix} = LU$ with $L$ unit lower triangular and $U$ upper triangular, no row exchanges. Give $L$. `[matrix]`

<details><summary>Answer</summary>

**$L = \begin{pmatrix} 1 & 0 \\ 2 & 1 \end{pmatrix}$.** The multiplier is $\ell_{21} = 4/2 = 2$ (row 2 minus 2 row 1), and $L$ stores it as is. The `lu` of [Gaussian elimination and LU with partial pivoting](02-gaussian-elimination-and-lu.md) would swap rows here (4 is the larger pivot candidate); without row exchanges, as asked, these are the factors. A common mistake is $\ell_{21} = \frac12$, the reciprocal, which gives $LU \ne A$.

</details>

**q5.** Give the $U$ of q4. `[matrix]`

<details><summary>Answer</summary>

**$U = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix}$.** Row 2 becomes $(4, 5) - 2(2, 1) = (0, 3)$. Check: $LU = \begin{pmatrix} 2 & 1 \\ 4 & 2 + 3 \end{pmatrix} = A$. Writing $5$ in the corner forgets to subtract $2 \times$ row 1 from the second column.

</details>

**q6.** Factor $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ as $PA = LU$ with **partial pivoting** (swap the row with the largest pivot candidate to the top). Give $L$. `[matrix]`

<details><summary>Answer</summary>

**$L = \begin{pmatrix} 1 & 0 \\ \frac13 & 1 \end{pmatrix}$.** The pivot candidates in column 1 are 1 and 3; the larger is 3, so swap ($P = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, $PA = \begin{pmatrix} 3 & 4 \\ 1 & 2 \end{pmatrix}$). The multiplier is $\frac13$, at most 1 in size, which is the point of partial pivoting. Without pivoting the multiplier would be 3.

</details>

**q7.** Give the $U$ of q6. `[matrix]`

<details><summary>Answer</summary>

**$U = \begin{pmatrix} 3 & 4 \\ 0 & \frac23 \end{pmatrix}$.** Row 2 of $PA$ becomes $(1, 2) - \frac13(3, 4) = (0, \frac23)$. Without pivoting one gets $\begin{pmatrix} 1 & 2 \\ 0 & -2 \end{pmatrix}$, a valid $A = LU$ but not the pivoted factorization asked for.

</details>

**q8.** How many solutions has the system $x + y = 1$, $x + y = 2$? `[number]`

<details><summary>Answer</summary>

**$0$.** Subtracting the equations gives $0 = 1$: the two lines $x + y = 1$ and $x + y = 2$ are parallel and distinct, so the system is inconsistent.

</details>

**q9.** Give the set of $k$ for which $x + k y = 1$, $k x + y = 1$ has infinitely many solutions. `[set]`

<details><summary>Answer</summary>

**$\{1\}$.** The determinant $1 - k^2$ is nonzero for $k \ne \pm 1$, giving exactly one solution. At $k = 1$ both equations are $x + y = 1$: infinitely many solutions. At $k = -1$ they are $x - y = 1$ and $-x + y = 1$, whose sum is $0 = 2$: no solution. A common mistake is $\{1, -1\}$, which treats every singular case as underdetermined.

</details>

**q10.** Prove that if the square matrix $A$ is invertible, then $Ax = b$ has exactly one solution for every $b$. `[proof]`

<details><summary>Answer</summary>

**Claim.** If the $n \times n$ matrix $A$ is invertible, then for every $b$ the system $Ax = b$ has exactly one solution.

**Proof.** Let $A^{-1}$ satisfy $A^{-1}A = AA^{-1} = I$. *Existence:* put $x = A^{-1}b$. Then $Ax = A(A^{-1}b) = (AA^{-1})b = b$ by associativity. *Uniqueness:* if $Ax = b$ and $Ay = b$, then $A(x - y) = 0$, and multiplying on the left by $A^{-1}$ gives $x - y = (A^{-1}A)(x - y) = A^{-1}0 = 0$. So for every $b$ there is exactly one solution, $x = A^{-1}b$.

A proof needs both halves (existence and uniqueness) and should use only the definition of the inverse and associativity. Model proof also in [`q10.md`](../../../../course/solutions/solve/S-M03a/q10.md).

</details>

## Span, linear independence, basis, dimension

**q11.** Is $(1, 2, 3)$ in the span of $(1, 0, 1)$ and $(0, 1, 1)$? `[bool]`

<details><summary>Answer</summary>

**Yes.** $a(1, 0, 1) + b(0, 1, 1) = (a, b, a + b)$. Matching $(1, 2, 3)$ forces $a = 1$, $b = 2$, and then $a + b = 3$ holds.

</details>

**q12.** Are $(1, 2)$ and $(2, 4)$ linearly independent? `[bool]`

<details><summary>Answer</summary>

**No.** $(2, 4) = 2(1, 2)$, so $2(1, 2) - (2, 4) = 0$ is a nontrivial combination.

</details>

**q13.** Give the dimension of the span of $(1, 0, 1)$, $(0, 1, 1)$, $(1, 1, 2)$. `[number]`

<details><summary>Answer</summary>

**$2$.** $(1, 1, 2) = (1, 0, 1) + (0, 1, 1)$, so the third vector adds nothing, and the first two are independent. Answering 3 counts vectors rather than independent directions.

</details>

**q14.** Give a basis of the column space of $\begin{pmatrix} 1 & 2 \\ 2 & 4 \\ 3 & 6 \end{pmatrix}$. `[basis]`

<details><summary>Answer</summary>

**$\{(1, 2, 3)\}$.** The second column is twice the first, so the column space is the line through $(1, 2, 3)$. Listing both columns is wrong because a basis must be independent.

</details>

**q15.** Give a basis of the null space of $\begin{pmatrix} 1 & 1 & 1 \end{pmatrix}$ (vectors in $\mathbb{R}^3$). `[basis]`

<details><summary>Answer</summary>

**$\{(-1, 1, 0), (-1, 0, 1)\}$** (any basis of the plane $x + y + z = 0$ is correct). The pivot is $x$; $y$ and $z$ are free. Setting $(y, z) = (1, 0)$ and $(0, 1)$ gives the two special solutions. Common mistakes: $(1, 1, 1)$, which spans the row space (orthogonal to the null space), and giving only one vector when the nullity is $3 - 1 = 2$.

</details>

**q16.** Give the coordinates of $(3, 5)$ in the basis $(1, 1)$, $(1, -1)$. `[vector]`

<details><summary>Answer</summary>

**$(4, -1)$.** Solve $a(1, 1) + b(1, -1) = (3, 5)$: $a + b = 3$, $a - b = 5$, so $a = 4$, $b = -1$. Check: $(4 - 1, 4 + 1) = (3, 5)$. $(-1, 4)$ swaps the coordinates.

</details>

**q17.** Give the dimension of the space of symmetric $2 \times 2$ real matrices. `[number]`

<details><summary>Answer</summary>

**$3$.** A symmetric matrix $\begin{pmatrix} a & b \\ b & c \end{pmatrix}$ is fixed by three free numbers, and the three matrices with a single 1 in position $a$, $c$, or the pair $b$ form a basis. Answering 4 counts all $2 \times 2$ matrices.

</details>

**q18.** Give the set of $c$ for which $(1, c)$ and $(c, 4)$ are linearly dependent. `[set]`

<details><summary>Answer</summary>

**$\{-2, 2\}$.** Two vectors in $\mathbb{R}^2$ are dependent exactly when $\det\begin{pmatrix} 1 & c \\ c & 4 \end{pmatrix} = 4 - c^2 = 0$. Forgetting the negative root is the common mistake: $(1, -2)$ and $(-2, 4)$ are also parallel.

</details>

**q19.** Give the dimension of the null space of $\begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{pmatrix}$. `[number]`

<details><summary>Answer</summary>

**$2$.** Row 2 is twice row 1, so the rank is 1, and rank-nullity gives $3 - 1 = 2$. Answering 1 reports the rank.

</details>

**q20.** Prove that any $n + 1$ vectors in $\mathbb{R}^n$ are linearly dependent. `[proof]`

<details><summary>Answer</summary>

**Claim.** Any $n + 1$ vectors $v_1, \dots, v_{n+1}$ in $\mathbb{R}^n$ are linearly dependent.

**Proof.** Let $A$ be the $n \times (n + 1)$ matrix with columns $v_1, \dots, v_{n+1}$. The vectors are dependent exactly when $Ac = 0$ for some $c \ne 0$, since $Ac = \sum_i c_i v_i$. Reduce $A$ to echelon form; row operations do not change the solutions of $Ac = 0$. Each pivot occupies its own row and there are only $n$ rows, so at most $n$ of the $n + 1$ columns have a pivot: some $c_j$ is a free variable. Set $c_j = 1$, every other free variable to 0, and solve for the pivot variables by back substitution. This gives $Ac = 0$ with $c_j = 1$, so $c \ne 0$, and the vectors are dependent.

The key steps are translating dependence into $Ac = 0$, counting pivots against columns, and building an explicit nonzero solution from the free variable. Model proof also in [`q20.md`](../../../../course/solutions/solve/S-M03a/q20.md).

</details>

## Linear maps and rank-nullity

**q21.** Give the matrix of the rotation of $\mathbb{R}^2$ by $90°$ counterclockwise. `[matrix]`

<details><summary>Answer</summary>

**$\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$.** Column $j$ is the image of $e_j$: $e_1 = (1, 0) \mapsto (0, 1)$ and $e_2 = (0, 1) \mapsto (-1, 0)$. `rotate_pairs` in [Trigonometry, rotations, complex numbers, and Euler's formula](../00-precalculus/02-trig-rotations-and-eulers-formula.md) and [RoPE](../../build-an-llm/p07-modern-block/03-rope.md) in *Build Your Own LLM System* apply this matrix to pairs of coordinates. The transpose $\begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$ rotates clockwise.

</details>

**q22.** Give the matrix of $T(x, y) = (x + y,\ 2x,\ y)$ from $\mathbb{R}^2$ to $\mathbb{R}^3$. `[matrix]`

<details><summary>Answer</summary>

**$\begin{pmatrix} 1 & 1 \\ 2 & 0 \\ 0 & 1 \end{pmatrix}$.** The columns are $T(e_1) = (1, 2, 0)$ and $T(e_2) = (1, 0, 1)$, giving a $3 \times 2$ matrix for a map from $\mathbb{R}^2$ to $\mathbb{R}^3$. Writing the images as rows gives the $2 \times 3$ transpose, which does not even have the right shape.

</details>

**q23.** Give the rank of $\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}$. `[number]`

<details><summary>Answer</summary>

**$2$.** Row 3 minus row 2 equals row 2 minus row 1 (both $(3, 3, 3)$), so the rows are dependent; rows 1 and 2 are not parallel. Elimination leaves two pivots. (q31 changes the last entry to 10, which makes the matrix invertible.)

</details>

**q24.** Give the nullity (dimension of the null space) of the matrix of q23. `[number]`

<details><summary>Answer</summary>

**$1$.** Rank-nullity: $3 - 2 = 1$. The null space is spanned by $(1, -2, 1)$: $1 - 4 + 3 = 0$, $4 - 10 + 6 = 0$, $7 - 16 + 9 = 0$.

</details>

**q25.** A $5 \times 7$ matrix has rank 4. Give the dimension of its null space. `[number]`

<details><summary>Answer</summary>

**$3$.** The null space lives in $\mathbb{R}^7$ (the number of columns), so its dimension is $7 - 4 = 3$. Answering 1 computes $5 - 4$, subtracting from the number of rows.

</details>

**q26.** Is $T(x) = x + 1$, from $\mathbb{R}$ to $\mathbb{R}$, a linear map? `[bool]`

<details><summary>Answer</summary>

**No.** A linear map sends 0 to 0, but $T(0) = 1$. (Also $T(2x) = 2x + 1 \ne 2T(x)$.) It is an affine map.

</details>

**q27.** $T(x, y) = (y, x)$ and $S(x, y) = (2x, y)$. Give the matrix of $S \circ T$ (first $T$, then $S$). `[matrix]`

<details><summary>Answer</summary>

**$\begin{pmatrix} 0 & 2 \\ 1 & 0 \end{pmatrix}$.** $(x, y) \xrightarrow{T} (y, x) \xrightarrow{S} (2y, x)$. As matrices, $ST = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$: the map applied first is on the right. Multiplying in the order of application gives $TS = \begin{pmatrix} 0 & 1 \\ 2 & 0 \end{pmatrix}$, which sends $(x, y)$ to $(y, 2x)$.

</details>

**q28.** Prove that the null space $\{x : Ax = 0\}$ of an $m \times n$ matrix $A$ is a subspace of $\mathbb{R}^n$. `[proof]`

<details><summary>Answer</summary>

**Claim.** For an $m \times n$ matrix $A$, $N(A) = \{x \in \mathbb{R}^n : Ax = 0\}$ is a subspace of $\mathbb{R}^n$.

**Proof.** Matrix multiplication is linear: $A(x + y) = Ax + Ay$ and $A(cx) = c(Ax)$. *Zero:* $A0 = 0$, so $0 \in N(A)$. *Addition:* if $Ax = 0$ and $Ay = 0$, then $A(x + y) = 0 + 0 = 0$. *Scaling:* if $Ax = 0$, then $A(cx) = c \cdot 0 = 0$. So $N(A)$ is a nonempty subset of $\mathbb{R}^n$ closed under addition and scalar multiplication: a subspace.

All three conditions are needed, and linearity of $x \mapsto Ax$ is the reason each holds. Model proof also in [`q28.md`](../../../../course/solutions/solve/S-M03a/q28.md).

</details>

## Determinants

**q29.** $\det \begin{pmatrix} 2 & 1 \\ 7 & 4 \end{pmatrix}$. `[number]`

<details><summary>Answer</summary>

**$1$.** $ad - bc = 2 \cdot 4 - 1 \cdot 7 = 1$. Answering 15 adds the products instead of subtracting.

</details>

**q30.** $\det \begin{pmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 0 & 0 & 6 \end{pmatrix}$. `[number]`

<details><summary>Answer</summary>

**$24$.** The matrix is upper triangular, so the determinant is the product of the diagonal, $1 \cdot 4 \cdot 6$. Answering 11 gives the trace.

</details>

**q31.** $\det \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 10 \end{pmatrix}$. `[number]`

<details><summary>Answer</summary>

**$-3$.** Expand along row 1: $1(5 \cdot 10 - 6 \cdot 8) - 2(4 \cdot 10 - 6 \cdot 7) + 3(4 \cdot 8 - 5 \cdot 7) = 2 + 4 - 9 = -3$. Common mistakes: 3 (a sign slip in the cofactor pattern $+, -, +$) and 0, which is the determinant of the matrix with a 9 in the corner (q23's).

</details>

**q32.** $A$ is $3 \times 3$ with $\det A = 5$. Give $\det(2A)$. `[number]`

<details><summary>Answer</summary>

**$40$.** Scaling every entry by 2 scales each of the 3 rows by 2, and the determinant is linear in each row: $\det(2A) = 2^3 \det A = 40$. Answering 10 scales only once.

</details>

**q33.** $\det A = 4$. Give $\det(A^{-1})$. `[number]`

<details><summary>Answer</summary>

**$\frac14$.** $\det(A)\det(A^{-1}) = \det(AA^{-1}) = \det I = 1$. Answering $-4$ confuses the inverse with the negative.

</details>

**q34.** Give the area of the parallelogram spanned by $(3, 1)$ and $(1, 2)$. `[number]`

<details><summary>Answer</summary>

**$5$.** The area is $|\det\begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}| = |3 \cdot 2 - 1 \cdot 1| = 5$. Answering 7 adds the products.

</details>

## Eigenvalues and eigenvectors

**q35.** Give the eigenvalues of $\begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$. `[set]`

<details><summary>Answer</summary>

**$\{2, 3\}$.** A diagonal matrix has its diagonal entries as eigenvalues, with eigenvectors $e_1$ and $e_2$. Answering $\{5\}$ adds them (that is the trace).

</details>

**q36.** Give the eigenvalues of $\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$. `[set]`

<details><summary>Answer</summary>

**$\{1, 3\}$.** $\det(tI - A) = (t - 2)^2 - 1 = (t - 1)(t - 3)$. Check: the sum $1 + 3 = 4$ is the trace and the product $3$ is the determinant. Reading $\{2\}$ off the diagonal works only for triangular matrices.

</details>

**q37.** Give an eigenvector of the matrix of q36 for its largest eigenvalue. `[vector]`

<details><summary>Answer</summary>

**$(1, 1)$** (or any nonzero multiple). $(A - 3I)v = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}v = 0$ gives $v_1 = v_2$. Check: $A(1, 1) = (3, 3)$. $(1, -1)$ is the eigenvector of the other eigenvalue, 1.

</details>

**q38.** Give the characteristic polynomial $\det(tI - A)$ of $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$. `[expr in t]`

<details><summary>Answer</summary>

**$t^2 - 5t - 2$.** $\det\begin{pmatrix} t - 1 & -2 \\ -3 & t - 4 \end{pmatrix} = (t - 1)(t - 4) - 6 = t^2 - 5t - 2$. Check: the constant term is $\det A = 4 - 6 = -2$ and the $t$ coefficient is $-\operatorname{tr} A$. Common mistakes: $t^2 - 5t + 4$ (the off-diagonal product dropped) and $t^2 + 5t - 2$ (the sign of the trace term).

</details>

**q39.** Give the (complex) eigenvalues of the rotation $\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$. `[set]`

<details><summary>Answer</summary>

**$\{i, -i\}$.** $\det(tI - A) = t^2 + 1$. No real vector is mapped onto its own line by a $90°$ rotation, so the eigenvalues cannot be real. $\{1, -1\}$ are the eigenvalues of a reflection.

</details>

**q40.** Give the product of the eigenvalues of $\begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$. `[number]`

<details><summary>Answer</summary>

**$10$.** The product of the eigenvalues is $\det A = 4 \cdot 3 - 1 \cdot 2 = 10$ (the eigenvalues are 5 and 2). Answering 7 gives their sum, the trace.

</details>

**q41.** With $A$ from q36, give $A^{10} (1, 1)$. `[vector]`

<details><summary>Answer</summary>

**$(59049, 59049)$.** $(1, 1)$ is the eigenvector for $\lambda = 3$ (q37), so $A^{10}(1, 1) = 3^{10}(1, 1)$ and $3^{10} = 59049$. Answering $(30, 30)$ multiplies by $10 \cdot 3$ instead of raising to the power.

</details>

**q42.** Give the spectral radius (largest $|\lambda|$) of $\begin{pmatrix} 1/2 & 2/5 \\ 2/5 & 1/2 \end{pmatrix}$. `[number]`

<details><summary>Answer</summary>

**$\frac{9}{10}$.** For $\begin{pmatrix} a & b \\ b & a \end{pmatrix}$ the eigenvalues are $a \pm b$ (eigenvectors $(1, \pm 1)$): here $\frac12 \pm \frac25 = \frac{9}{10}, \frac{1}{10}$. A spectral radius below 1 makes repeated multiplication shrink, which is the vanishing-gradient story of [Vanilla RNN with manual BPTT and truncation](../../build-an-llm/p03-recurrent/01-vanilla-rnn-manual-bptt.md). Common mistakes: $\frac12$ (the diagonal entry) and $\frac{1}{10}$ (the smaller eigenvalue).

</details>

**q43.** Power iteration on the matrix of q36 from a generic start shrinks its error by the factor $|\lambda_2 / \lambda_1|$ per step. Give that factor. `[number]`

<details><summary>Answer</summary>

**$\frac13$.** $|\lambda_2/\lambda_1| = 1/3$ from q36: the component along $(1, -1)$ is multiplied by 1 per step while the one along $(1, 1)$ is multiplied by 3, so their ratio shrinks by a factor 3 each step. The `power_iteration` of [Eigenvalues, power iteration, and the spectral radius](04-eigenvalues-and-power-iteration.md) converges at this rate. Answering 3 inverts the ratio.

</details>

**q44.** Prove that eigenvectors $v_1, v_2$ of $A$ with eigenvalues $\lambda_1 \ne \lambda_2$ are linearly independent. `[proof]`

<details><summary>Answer</summary>

**Claim.** If $Av_1 = \lambda_1 v_1$, $Av_2 = \lambda_2 v_2$, $v_1, v_2 \ne 0$, and $\lambda_1 \ne \lambda_2$, then $v_1$ and $v_2$ are linearly independent.

**Proof.** Suppose $c_1 v_1 + c_2 v_2 = 0$ (1). Apply $A$: $c_1\lambda_1 v_1 + c_2\lambda_2 v_2 = 0$ (2). Subtract $\lambda_2$ times (1) from (2): $c_1(\lambda_1 - \lambda_2)v_1 = 0$. Since $\lambda_1 \ne \lambda_2$ and $v_1 \ne 0$, $c_1 = 0$. Then (1) reads $c_2 v_2 = 0$, and $v_2 \ne 0$ gives $c_2 = 0$. So only the trivial combination is 0.

The essential move is getting a second equation by applying $A$ and eliminating one vector; both hypotheses ($\lambda_1 \ne \lambda_2$ and $v_1 \ne 0$) must be used. Model proof also in [`q44.md`](../../../../course/solutions/solve/S-M03a/q44.md).

</details>

## Orthogonality and QR

**q45.** Give the orthogonal projection of $(1, 2, 3)$ onto the line spanned by $(1, 1, 1)$. `[vector]`

<details><summary>Answer</summary>

**$(2, 2, 2)$.** $\frac{u \cdot b}{u \cdot u}u = \frac{1 + 2 + 3}{3}(1, 1, 1)$. The residual $(-1, 0, 1)$ is orthogonal to $(1, 1, 1)$, as it must be. This is the inner-product step of [Inner products, projections, cosine similarity, and top-k](06-inner-products-projections-cosine-and-top-k.md) and of Gram-Schmidt. Answering $(6, 6, 6)$ forgets to divide by $\lVert u\rVert^2 = 3$.

</details>

**q46.** $A = \begin{pmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{pmatrix}$. Gram-Schmidt on its columns gives $A = QR$ with $Q$ having orthonormal columns and $R$ upper triangular **with a positive diagonal**. Give $Q$. `[matrix]`

<details><summary>Answer</summary>

**$Q = \begin{pmatrix} \frac{1}{\sqrt2} & \frac{1}{\sqrt6} \\ \frac{1}{\sqrt2} & -\frac{1}{\sqrt6} \\ 0 & \frac{2}{\sqrt6} \end{pmatrix}$.** Normalize $a_1 = (1, 1, 0)$: $q_1 = (1, 1, 0)/\sqrt2$. Then $q_1 \cdot a_2 = \frac{1}{\sqrt2}$, and $a_2 - \frac{1}{\sqrt2}q_1 = (1, 0, 1) - (\frac12, \frac12, 0) = (\frac12, -\frac12, 1)$, with length $\sqrt{3/2} = \frac{\sqrt6}{2}$, so $q_2 = (1, -1, 2)/\sqrt6$. Common mistakes: leaving the second column unnormalized, and flipping the sign of $q_2$, which makes $R_{22}$ negative and violates the uniqueness convention.

</details>

**q47.** Give the $R$ of q46. `[matrix]`

<details><summary>Answer</summary>

**$R = \begin{pmatrix} \sqrt2 & \frac{1}{\sqrt2} \\ 0 & \frac{\sqrt6}{2} \end{pmatrix}$.** $R = Q^\top A$: $R_{11} = \lVert a_1\rVert = \sqrt2$, $R_{12} = q_1 \cdot a_2 = \frac{1}{\sqrt2}$, $R_{22}$ is the length of what was left of $a_2$, $\frac{\sqrt6}{2}$. Check the second column: $\frac{1}{\sqrt2}q_1 + \frac{\sqrt6}{2}q_2 = (\frac12, \frac12, 0) + (\frac12, -\frac12, 1) = (1, 0, 1)$. Writing $\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ forgets that $Q$'s columns are normalized.

</details>

**q48.** Fit $y = c\,x$ to the points $(1, 1)$, $(2, 2)$, $(3, 2)$ by least squares. Give $c$. `[number]`

<details><summary>Answer</summary>

**$c = \frac{11}{14}$.** Minimize $\sum (y_i - cx_i)^2$: the derivative is zero at $c = \frac{\sum x_i y_i}{\sum x_i^2} = \frac{1 + 4 + 6}{1 + 4 + 9}$. This is the normal equation $A^\top A c = A^\top y$ for the one-column $A = (1, 2, 3)^\top$; the `lstsq` of [SVD, Eckart-Young low rank, and least squares](05-svd-low-rank-and-least-squares.md) solves it through QR. Common mistakes: $\frac56 = \sum y / \sum x$, and 1, which fits only the first two points.

</details>

## Common mistakes

The wrong answers discussed above fall into a handful of patterns. Each one has a symptom you can spot and a habit that prevents it:

| Mistake | Symptom | The habit that avoids it |
|---|---|---|
| Storing the reciprocal of the multiplier in $L$ | $LU \ne A$ | store $\ell_{21} = a_{21}/a_{11}$ itself (q4) |
| Skipping the pivot search | multipliers larger than 1 and unstable factors; $P$ differs from NumPy's | pick the largest $\lvert\text{candidate}\rvert$ as pivot before eliminating (q6, q7) |
| Calling an echelon form "reduced" | entries above the pivots left nonzero | clear the entries above each pivot too (q3) |
| Counting vectors instead of independent vectors | dimension and rank too large | check each vector against the span of the previous ones (q13, q14) |
| Confusing rank with nullity, or rows with columns in rank-nullity | the wrong null space dimension | subtract the rank from the number of columns (q19, q25) |
| Writing images of basis vectors as rows | the transpose of the map's matrix | write $T(e_j)$ as column $j$ (q22) |
| Multiplying maps in the order they are applied | $TS$ instead of $ST$ | put the map applied first on the right (q27) |
| $\det(cA) = c \det A$ | volume scale off by $c^{n-1}$ | scale once per row: $c^n$ (q32) |
| Reading eigenvalues off the diagonal of a non-triangular matrix | wrong spectral radius; an RNN diagnosed as stable when it is not | solve the characteristic polynomial (q36, q42) |
| Forgetting to divide by $u \cdot u$ in a projection | projections scaled by $\lVert u \rVert^2$ | divide by $u \cdot u$ (q45) |

## Where this leads

Each group of problems is the hand version of code in this chapter. q4 to q7 are the factors that `lu` and `lu_solve` compute in [Gaussian elimination and LU with partial pivoting](02-gaussian-elimination-and-lu.md). The uniqueness convention of q46 and q47 is the sign rule of [Orthogonality, Householder QR, and orthogonal initialization](03-orthogonality-and-householder-qr.md). The rate in q43 is the convergence rate of `power_iteration` and `spectral_radius` in [Eigenvalues, power iteration, and the spectral radius](04-eigenvalues-and-power-iteration.md), and q48 is least squares through QR in [SVD, Eckart-Young low rank, and least squares](05-svd-low-rank-and-least-squares.md). The [linear algebra problem set, part b](91-problem-set-b.md) continues with inner products and projections, SVD and low rank, and FLOP and byte counts.

### Further reading

- Strang, *Introduction to Linear Algebra*, ch. 2 to 6, and MIT 18.06 lectures 1 to 21.
- Hefferon, *Linear Algebra* (free), ch. 1 to 5.

## Summary

- Elimination is LU: the multipliers are $L$, the result is $U$, and partial pivoting keeps every multiplier at most 1 in size (q4 to q7).
- A basis is a minimal spanning set; its size, the dimension, is the same for every basis, and any $n + 1$ vectors in $\mathbb{R}^n$ are dependent (q13, q20).
- Rank plus nullity equals the number of columns: every column either adds a direction to the image or a direction to the null space (q23 to q25).
- The determinant measures how a matrix scales volume; it is the product of the eigenvalues and is 0 exactly when the matrix is singular (q31, q34, q40).
- Repeated multiplication by $A$ is governed by its eigenvalues: $A^k v = \lambda^k v$, power iteration converges at rate $|\lambda_2/\lambda_1|$, and a spectral radius below 1 shrinks everything (q41 to q43).
