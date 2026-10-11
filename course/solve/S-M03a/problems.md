# S-M03a problems: elimination and LU, span and basis, linear maps, determinants, eigenvalues, QR

Answer every question in `solve/S-M03a.toml` (written by `ol start S-M03a`).
The tag after each question is its answer type. `[number]` is an exact
value (`9/10`, `-3`). `[vector]` is a column written as a flat list
`[1, 2, 3]`. `[matrix]` is a list of rows `[[1, 0], [2, 1]]`. `[basis]` is a
list of vectors `[[1, 0, 1], [0, 1, 1]]`: any basis of the same space passes.
`[set]` is `{a, b}`. `[bool]` is `true` or `false`. `[expr in t]` is a
formula in `t`. `[proof]` is a file `solve/S-M03a/qN.md`, graded against its
rubric. Write $i$ as `I` and $\sqrt{2}$ as `sqrt(2)`.

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
