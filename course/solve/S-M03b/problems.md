# S-M03b problems: inner products and projections, SVD and low rank, matmul accounting and the roofline

Answer every question in `solve/S-M03b.toml` (written by `ol start S-M03b`).
The tag after each question is its answer type. `[number]` is an exact
value (`4/3`, `sqrt(10)`, `(1 + sqrt(5))/2`). `[vector]` is a column written
as a flat list `[1, 2, 3]`. `[expr in M, N, K]` is a formula in those
variables. `[proof]` is a file `solve/S-M03b/qN.md`, graded against its
rubric. Lettered parts are answered separately (`[q1.a]`, `[q1.b]`).

### Inner products and projections

**q1.** Let $u = (1, 2, 2)$ and $v = (2, 0, 1)$. Give (a) the inner product $u \cdot v$ and (b) the cosine of the angle between $u$ and $v$. `[number]`

**q2.** Give the orthogonal projection of $b = (1, 2, 3)$ onto the line spanned by $a = (1, 1, 1)$. `[vector]`

**q3.** Prove the Cauchy-Schwarz inequality $\lvert a \cdot b \rvert \le \lVert a \rVert \, \lVert b \rVert$ for vectors $a, b \in \mathbb{R}^n$, with equality exactly when one is a multiple of the other. Conclude that cosine similarity lies in $[-1, 1]$. `[proof]`

### SVD and low rank

**q4.** $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$. Give (a) its largest singular value $\sigma_1$ and (b) its smallest $\sigma_2$. `[number]`

**q5.** Give the singular values of $A = \begin{pmatrix} 0 & 2 \\ 3 & 0 \\ 0 & 0 \end{pmatrix}$ in descending order. `[vector]`

**q6.** A $3 \times 3$ matrix $A$ has singular values $5, 3, 1$, and $A_r$ is its truncated SVD of rank $r$. Give (a) $\lVert A - A_1 \rVert_2$, (b) $\lVert A - A_1 \rVert_F$, (c) the smallest $r$ with $\lVert A - A_r \rVert_F \le 3/2$. `[number]`

**q7.** A weight matrix of shape $4096 \times 4096$ is replaced by a product $BC$ with $B$ of shape $4096 \times r$ and $C$ of shape $r \times 4096$ (a rank-$r$ adapter). Give the number of parameters in $B$ and $C$ together. `[expr in r]`

**q8.** Let $A = U \Sigma V^\top$ be an SVD of a real $m \times n$ matrix. Prove that (i) the columns of $V$ are eigenvectors of $A^\top A$ with eigenvalues $\sigma_i^2$, so the singular values are the square roots of the eigenvalues of $A^\top A$, and (ii) $\max_{\lVert x \rVert = 1} \lVert A x \rVert = \sigma_1$. `[proof]`

### Matmul accounting and the roofline

**q9.** (a) Give the number of floating-point operations (one multiply and one add per term) of $C = AB$ with $A$ of shape $M \times K$ and $B$ of shape $K \times N$. `[expr in M, N, K]` (b) A matrix-vector product $y = Wx$ with $W$ of shape $N \times N$ in fp32 (4 bytes per number) reads $W$ and $x$ once and writes $y$ once. Give its arithmetic intensity, floating-point operations per byte moved. `[expr in N]`

**q10.** A square $N \times N$ fp32 matmul reads $A$ and $B$ once and writes $C$ once. (a) Give its arithmetic intensity in FLOP per byte. `[expr in N]` (b) A machine has a peak of $10$ TFLOP/s and a memory bandwidth of $100$ GB/s. Give the smallest $N$ at which this matmul reaches the compute-bound side of the roofline (intensity at least the ridge point). `[number]`
