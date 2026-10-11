<!-- ol:module S-M03a -->
# Linear algebra problem set, part a: elimination, LU, bases, rank, determinants, eigenvalues, QR

## Overview

| | |
|---|---|
| **Module** | `S-M03a` · solve · none · Pass 2 · 6 to 8 h |
| **You build** | answers in `solve/S-M03a.toml` (44 checked by SymPy) and 4 proofs in `solve/S-M03a/q10.md`, `q20.md`, `q28.md`, `q44.md` (self-graded against their rubrics) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M03a/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M03a/problems.md` and in section 4 |
| **Needs** | no module. Reading: `M03.1` (vectors, matrices, and the matrix product) and the [Linear Algebra topic](README.md) |
| **Used by** | no call site (a solve set). Take it after `M03.2` (LU), `M03.3` (Householder QR), and `M03.4` (power iteration) in Pass 2; `S-M03b` (SVD, projections, the roofline) follows `M03.5` and `M03.6` in Pass 3 |
| **Milestone** | `MS-P2` (the Pass 2 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | Strang, *Introduction to Linear Algebra*, ch. 2 to 6 (and MIT 18.06 lectures 1 to 21); Hefferon, *Linear Algebra* (free), ch. 1 to 5 |

## Key Takeaways

- Elimination is LU: the multipliers are $L$, the result is $U$, and partial pivoting keeps every multiplier at most 1 in size (q4 to q7).
- A basis is a minimal spanning set; its size, the dimension, is the same for every basis, and any $n + 1$ vectors in $\mathbb{R}^n$ are dependent (q13, q20).
- Rank plus nullity equals the number of columns: every column either adds a direction to the image or a direction to the null space (q23 to q25).
- The determinant measures how a matrix scales volume; it is the product of the eigenvalues and is 0 exactly when the matrix is singular (q31, q34, q40).
- Repeated multiplication by $A$ is governed by its eigenvalues: $A^k v = \lambda^k v$, power iteration converges at rate $|\lambda_2/\lambda_1|$, and a spectral radius below 1 shrinks everything (q41 to q43).

## How to work this chapter

```bash
ol start S-M03a             # writes solve/S-M03a.toml and the four proof files
ol check S-M03a             # SymPy checks the answers, then asks each proof rubric (y/n)
ol check S-M03a --regrade   # ask the rubrics again after you change a proof
```

---

## 1. Why now

Your Pass 2 code factors matrices: `M03.2` writes LU with partial pivoting, `M03.3` Householder QR and orthogonal initialization, `M03.4` power iteration and the spectral radius that `L3.1` uses to diagnose exploding gradients. Their tests compare your factors with numpy's, but a factorization that "matches numpy" is only meaningful if you know what the factors are supposed to be and why they exist: why pivoting is needed, what rank and null space mean for a least-squares fit, why an eigenvalue above 1 makes a recurrent network blow up. This set works those ideas by hand on matrices small enough to check on paper.

## 2. Principles

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

### 2.1 Elimination and LU

Gaussian elimination subtracts multiples of a pivot row from the rows below it until the matrix is upper triangular, $U$. Recording each multiplier $\ell_{ij}$ (the amount of row $j$ subtracted from row $i$) in a unit lower triangular $L$ gives $A = LU$, so solving $Ax = b$ becomes two triangular solves. If a pivot is 0 the rows must be exchanged, and if it is merely small, dividing by it amplifies rounding errors; **partial pivoting** swaps the row with the largest $|a_{ij}|$ in the current column into the pivot position, giving $PA = LU$ with every $|\ell_{ij}| \le 1$. The **reduced row echelon form** continues until each pivot is 1 and the only nonzero entry in its column. A system has no solution (inconsistent), exactly one, or infinitely many (a free variable).

### 2.2 Span, independence, basis, dimension

Vectors are **linearly independent** when $\sum c_i v_i = 0$ forces every $c_i = 0$. A **basis** of a subspace is an independent spanning set, and every basis has the same size, the **dimension**. The pivot columns of $A$ are a basis of its column space; the special solutions (one per free variable) are a basis of its null space. **Coordinates** in a basis are the unique coefficients $c$ with $\sum c_i v_i = x$: solve a linear system.

### 2.3 Linear maps and rank-nullity

A map $T$ is **linear** when $T(x + y) = T(x) + T(y)$ and $T(cx) = cT(x)$; then $T(x) = Ax$ where column $j$ of $A$ is $T(e_j)$. Composition is multiplication: $S \circ T$ has matrix $S\,T$ (apply $T$ first, so it is on the right). For $A$ with $n$ columns, **rank-nullity** says $\operatorname{rank} A + \dim N(A) = n$.

### 2.4 Determinants

The determinant is the signed volume scale factor of $x \mapsto Ax$: $|\det [u\ v]|$ is the area of the parallelogram spanned by $u$ and $v$. It is multiplicative, $\det(AB) = \det A \det B$, so $\det(A^{-1}) = 1/\det A$; it scales as $\det(cA) = c^n \det A$ for $n \times n$ $A$; a triangular matrix's determinant is the product of its diagonal; and $\det A = 0$ exactly when $A$ is singular.

### 2.5 Eigenvalues and eigenvectors

$\lambda$ is an eigenvalue when $A - \lambda I$ is singular, that is, a root of the **characteristic polynomial** $\det(tI - A)$; the product of the eigenvalues is $\det A$ and their sum is the trace. A real matrix may have complex eigenvalues (a rotation moves every real vector). If $v$ is an eigenvector, $A^k v = \lambda^k v$, so the largest $|\lambda|$, the **spectral radius** $\rho(A)$, decides whether repeated application grows ($\rho > 1$) or shrinks ($\rho < 1$). **Power iteration** repeats $v \leftarrow Av / \lVert Av \rVert$; the component along the second eigenvector shrinks by $|\lambda_2/\lambda_1|$ per step.

### 2.6 Orthogonality and QR

The projection of $b$ onto the line through $u$ is $\frac{u \cdot b}{u \cdot u} u$. **Gram-Schmidt** subtracts from each column its projections onto the previous orthonormal vectors and normalizes the rest, giving $A = QR$; requiring a positive diagonal of $R$ makes the factorization unique. Least squares minimizes $\lVert Ax - b \rVert^2$; its solution satisfies the **normal equations** $A^\top A x = A^\top b$, which QR solves stably as $Rx = Q^\top b$ (`M03.5`).

## 3. Worked example by hand

This is a sibling of q6, q7, and q36, not one of the graded problems.

**LU with partial pivoting.** $A = \begin{pmatrix} 2 & 1 \\ 6 & 4 \end{pmatrix}$. The largest entry in column 1 is 6, in row 2, so swap: $P = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$, $PA = \begin{pmatrix} 6 & 4 \\ 2 & 1 \end{pmatrix}$. The multiplier is $\ell_{21} = 2/6 = 1/3$, and row 2 becomes $(2, 1) - \frac13 (6, 4) = (0, -1/3)$. So $L = \begin{pmatrix} 1 & 0 \\ 1/3 & 1 \end{pmatrix}$, $U = \begin{pmatrix} 6 & 4 \\ 0 & -1/3 \end{pmatrix}$. Check: $LU = \begin{pmatrix} 6 & 4 \\ 2 & 4/3 - 1/3 \end{pmatrix} = PA$. In `solve/` the factor $L$ would be `answer = "[[1, 0], [1/3, 1]]"`.

**Eigenvalues and an eigenvector.** $B = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$ is triangular, so $\det(tI - B) = (t - 3)(t - 2)$ and the eigenvalues are 3 and 2. For $\lambda = 2$: $(B - 2I)v = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix} v = 0$ gives $v = (1, -1)$, or any nonzero multiple; the checker accepts `[2, -2]` too when the key says `up_to = "scalar"`. The product of the eigenvalues is $6 = \det B$.

## 4. The problem set

Write each answer in `solve/S-M03a.toml`:

```toml
[q1]
answer = "[-4, 9/2]"
[q4]
answer = "[[1, 0], [2, 1]]"
[q15]
answer = "[[-1, 1, 0], [-1, 0, 1]]"
[q36]
answer = "{1, 3}"
[q10]
proof = "S-M03a/q10.md"
```

A vector is a flat list (one column); a matrix is a list of rows; a basis is a list of vectors, and any basis of the right space passes. Entries are exact: `1/sqrt(2)`, not `0.7071`.

<!-- ol:problems S-M03a -->

### Gaussian elimination and LU

**q1.** Solve $x + 2y = 5$, $3x + 4y = 6$. Give $(x, y)$. `[vector]`

**q2.** Solve $x + y + z = 6$, $2x - y + z = 3$, $x + 2y - z = 2$. Give $(x, y, z)$. `[vector]`

**q3.** Give the reduced row echelon form of $\begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 7 \end{pmatrix}$. `[matrix]`

**q4.** $A = \begin{pmatrix} 2 & 1 \\ 4 & 5 \end{pmatrix} = LU$ with $L$ unit lower triangular and $U$ upper triangular, no row exchanges. Give $L$. `[matrix]`

**q5.** Give the $U$ of q4. `[matrix]`

**q6.** Factor $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ as $PA = LU$ with **partial pivoting** (swap the row with the largest pivot candidate to the top). Give $L$. `[matrix]`

**q7.** Give the $U$ of q6. `[matrix]`

**q8.** How many solutions has the system $x + y = 1$, $x + y = 2$? `[number]`

**q9.** Give the set of $k$ for which $x + k y = 1$, $k x + y = 1$ has infinitely many solutions. `[set]`

**q10.** Prove that if the square matrix $A$ is invertible, then $Ax = b$ has exactly one solution for every $b$. `[proof]`

### Span, linear independence, basis, dimension

**q11.** Is $(1, 2, 3)$ in the span of $(1, 0, 1)$ and $(0, 1, 1)$? `[bool]`

**q12.** Are $(1, 2)$ and $(2, 4)$ linearly independent? `[bool]`

**q13.** Give the dimension of the span of $(1, 0, 1)$, $(0, 1, 1)$, $(1, 1, 2)$. `[number]`

**q14.** Give a basis of the column space of $\begin{pmatrix} 1 & 2 \\ 2 & 4 \\ 3 & 6 \end{pmatrix}$. `[basis]`

**q15.** Give a basis of the null space of $\begin{pmatrix} 1 & 1 & 1 \end{pmatrix}$ (vectors in $\mathbb{R}^3$). `[basis]`

**q16.** Give the coordinates of $(3, 5)$ in the basis $(1, 1)$, $(1, -1)$. `[vector]`

**q17.** Give the dimension of the space of symmetric $2 \times 2$ real matrices. `[number]`

**q18.** Give the set of $c$ for which $(1, c)$ and $(c, 4)$ are linearly dependent. `[set]`

**q19.** Give the dimension of the null space of $\begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{pmatrix}$. `[number]`

**q20.** Prove that any $n + 1$ vectors in $\mathbb{R}^n$ are linearly dependent. `[proof]`

### Linear maps and rank-nullity

**q21.** Give the matrix of the rotation of $\mathbb{R}^2$ by $90°$ counterclockwise. `[matrix]`

**q22.** Give the matrix of $T(x, y) = (x + y,\ 2x,\ y)$ from $\mathbb{R}^2$ to $\mathbb{R}^3$. `[matrix]`

**q23.** Give the rank of $\begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}$. `[number]`

**q24.** Give the nullity (dimension of the null space) of the matrix of q23. `[number]`

**q25.** A $5 \times 7$ matrix has rank 4. Give the dimension of its null space. `[number]`

**q26.** Is $T(x) = x + 1$, from $\mathbb{R}$ to $\mathbb{R}$, a linear map? `[bool]`

**q27.** $T(x, y) = (y, x)$ and $S(x, y) = (2x, y)$. Give the matrix of $S \circ T$ (first $T$, then $S$). `[matrix]`

**q28.** Prove that the null space $\{x : Ax = 0\}$ of an $m \times n$ matrix $A$ is a subspace of $\mathbb{R}^n$. `[proof]`

### Determinants

**q29.** $\det \begin{pmatrix} 2 & 1 \\ 7 & 4 \end{pmatrix}$. `[number]`

**q30.** $\det \begin{pmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 0 & 0 & 6 \end{pmatrix}$. `[number]`

**q31.** $\det \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 10 \end{pmatrix}$. `[number]`

**q32.** $A$ is $3 \times 3$ with $\det A = 5$. Give $\det(2A)$. `[number]`

**q33.** $\det A = 4$. Give $\det(A^{-1})$. `[number]`

**q34.** Give the area of the parallelogram spanned by $(3, 1)$ and $(1, 2)$. `[number]`

### Eigenvalues and eigenvectors

**q35.** Give the eigenvalues of $\begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$. `[set]`

**q36.** Give the eigenvalues of $\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$. `[set]`

**q37.** Give an eigenvector of the matrix of q36 for its largest eigenvalue. `[vector]`

**q38.** Give the characteristic polynomial $\det(tI - A)$ of $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$. `[expr in t]`

**q39.** Give the (complex) eigenvalues of the rotation $\begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$. `[set]`

**q40.** Give the product of the eigenvalues of $\begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$. `[number]`

**q41.** With $A$ from q36, give $A^{10} (1, 1)$. `[vector]`

**q42.** Give the spectral radius (largest $|\lambda|$) of $\begin{pmatrix} 1/2 & 2/5 \\ 2/5 & 1/2 \end{pmatrix}$. `[number]`

**q43.** Power iteration on the matrix of q36 from a generic start shrinks its error by the factor $|\lambda_2 / \lambda_1|$ per step. Give that factor. `[number]`

**q44.** Prove that eigenvectors $v_1, v_2$ of $A$ with eigenvalues $\lambda_1 \ne \lambda_2$ are linearly independent. `[proof]`

### Orthogonality and QR

**q45.** Give the orthogonal projection of $(1, 2, 3)$ onto the line spanned by $(1, 1, 1)$. `[vector]`

**q46.** $A = \begin{pmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{pmatrix}$. Gram-Schmidt on its columns gives $A = QR$ with $Q$ having orthonormal columns and $R$ upper triangular **with a positive diagonal**. Give $Q$. `[matrix]`

**q47.** Give the $R$ of q46. `[matrix]`

**q48.** Fit $y = c\,x$ to the points $(1, 1)$, $(2, 2)$, $(3, 2)$ by least squares. Give $c$. `[number]`

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Storing the reciprocal of the multiplier in $L$ | $LU \ne A$ | q4 (canary with 1/2) |
| Skipping the pivot search | multipliers larger than 1 and unstable factors; numpy's $P$ differs from yours | q6, q7 (canaries) |
| Calling an echelon form "reduced" | entries above the pivots left nonzero | q3 (canary) |
| Counting vectors instead of independent vectors | dimension and rank too large | q13 (canary 3), q14 (canary) |
| Confusing rank with nullity, or rows with columns in rank-nullity | the wrong null space dimension | q19, q25 (canaries) |
| Writing images of basis vectors as rows | the transpose of the map's matrix | q22 (canary) |
| Multiplying maps in the order they are applied | $TS$ instead of $ST$ | q27 (canary) |
| $\det(cA) = c \det A$ | volume scale off by $c^{n-1}$ | q32 (canary 10) |
| Reading eigenvalues off the diagonal of a non-triangular matrix | wrong spectral radius; an RNN diagnosed as stable when it is not | q36 (canary {2}), q42 (canary 1/2) |
| Forgetting to divide by $u \cdot u$ in a projection | projections scaled by $\lVert u \rVert^2$ | q45 (canary) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M03.1` | vectors, matrices, row-major layout, and the product you implemented in C |
| Forward | `M03.2` | `lu` and `lu_solve` with partial pivoting, the factors of q4 to q7 |
| Forward | `M03.3` | Householder QR; the uniqueness convention of q46 and q47 |
| Forward | `M03.4` | power iteration and `spectral_radius`, at the rate of q43 |
| Forward | `M03.5` | least squares through QR, the problem of q48 |
| Forward | `S-M03b` | inner products and projections, SVD and low rank, FLOP and byte counts |
