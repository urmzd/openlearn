# S-M08 problems: shapes, matrix differentials, the trace trick, VJPs, SDPA

Answer every question in `solve/S-M08.toml` (written by `ol start S-M08`).
The tag after each question is its answer type: `[vector]` is a list
`[1/4, -1/8, -1/8]` (also used for shapes, `[2, 3, 5]`), `[matrix]` a list of
rows `[[1, 3], [2, 4]]`, `[expr]` a formula in the named variables, `[number]`
an exact value (`log(5/2)`, `2*E/(1 + E)^2`), `[bool]` `true` or `false`,
`[choice]` one letter, and `[proof]` a file `solve/S-M08/qN.md` graded against
its rubric. Gradients have the shape of the variable (denominator layout).

### Layouts and shapes

A linear layer computes $Y = X W^\top + b$ for a batch $X$ of shape $(B, T, d_{in}) = (2, 3, 4)$, stored row-major in float32, with $W$ of shape $(d_{out}, d_{in}) = (5, 4)$ and $b$ of shape $(5)$. $L$ is a scalar loss.

**q1.** Give the shape of (a) $Y$, (b) $\partial L / \partial W$, (c) $\partial L / \partial b$. `[vector]`

**q2.** (a) At which element offset (counting from 0) is $X[1, 2, 1]$ stored? `[number]` (b) Give the strides of $X$ in elements, one per axis. `[vector]` (c) Give the strides of the view that swaps the last two axes of $X$ (shape $(2, 4, 3)$, no copy). `[vector]`

**q3.** $b$ is broadcast over the batch and time axes. How many entries of $\partial L / \partial Y$ are summed into each entry of $\partial L / \partial b$? `[number]`

**q4.** Flatten $X$ and $Y$ row-major into vectors. What is the shape of the Jacobian $\partial\, \mathrm{vec}(Y) / \partial\, \mathrm{vec}(X)$ (rows, columns)? `[vector]`

### Differentials of matrix expressions

**q5.** $f(x) = x^\top A x$ with $A = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix}$ and $x = (x_1, x_2)$. Give $\nabla f$. `[vector in x1, x2]`

**q6.** $f(X) = \operatorname{tr}(AX)$ with $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$. Give $\nabla_X f$. `[matrix]`

**q7.** $f(x) = \lVert Ax - b \rVert^2$ with $A = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}$, $b = (1, 0)$. Give $\nabla f$ at $x = (1, 2)$. `[vector]`

**q8.** $X(t) = \begin{pmatrix} 2 & t \\ 1 & 1 \end{pmatrix}$. Give $\frac{d}{dt} X(t)^{-1}$ at $t = 0$. `[matrix]`

**q9.** Give $\nabla_X \log \det X$ at $X = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix}$. `[matrix]`

**q10.** $f(W) = \lVert W x \rVert^2$ with $x = (1, 2)$. Give $\nabla_W f$ at $W = I_2$. `[matrix]`

**q11.** Give the derivative of the sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$. `[expr in z]`

**q12.** Give the derivative of softplus, $f(z) = \log(1 + e^z)$. `[expr in z]`

**q13.** Give the gradient of $\mathrm{LSE}(x_1, x_2) = \log(e^{x_1} + e^{x_2})$. `[vector in x1, x2]`

**q14.** Give the gradient of $f(x) = \lVert x \rVert_2$ at $x = (3, 4)$. `[vector]`

**q15.** Give the Jacobian of elementwise $\mathrm{ReLU}$ at $x = (-1, 2, 1/2)$. `[matrix]`

**q16.** Prove $d(X^{-1}) = -X^{-1} (dX) X^{-1}$ for an invertible square matrix $X$. `[proof]`

### The trace trick

**q17.** For conformable matrices: (a) is $\operatorname{tr}(ABC) = \operatorname{tr}(CAB)$ always? (b) Is $\operatorname{tr}(ABC) = \operatorname{tr}(BAC)$ always? `[bool]`

**q18.** $Y = XW$ with $X = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, $W = \begin{pmatrix} 1 & 1 \\ 0 & 2 \end{pmatrix}$, and upstream gradient $G = \partial L / \partial Y = I_2$. Give (a) $\partial L / \partial W$ and (b) $\partial L / \partial X$. `[matrix]`

**q19.** The Frobenius inner product is $\langle A, B \rangle = \operatorname{tr}(A^\top B)$. Give it for $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, $B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$. `[number]`

**q20.** Let $Y = XW$ and $L$ a scalar with $dL = \operatorname{tr}(G^\top dY)$, $G = \partial L / \partial Y$. Prove $\partial L / \partial W = X^\top G$ and $\partial L / \partial X = G W^\top$. `[proof]`

### VJPs by hand

**q21.** $y = \mathrm{softmax}(x) = (1/2, 1/4, 1/4)$ and the upstream gradient is $g = (1, 0, 0)$. Give $\partial L / \partial x$. `[vector]`

**q22.** Give the Jacobian $\partial y / \partial x$ of a 2-class softmax at $y = (1/4, 3/4)$. `[matrix]`

**q23.** $\ell = \mathrm{log\_softmax}(x)$ with $\mathrm{softmax}(x) = (1/2, 1/4, 1/4)$ and upstream gradient $g = (0, 1, 0)$. Give $\partial L / \partial x$. `[vector]`

**q24.** Cross-entropy with logits $z = (0, \log 2, \log 2)$ and target class 1 (counting from 0): $L = -\log \mathrm{softmax}(z)_1$. (a) Give $\partial L / \partial z$. `[vector]` (b) Give $L$. `[number]`

**q25.** A batch has 4 positions; one has the target `ignore_index` and the loss is the mean over the other positions. By what factor is each kept position's per-position gradient $\mathrm{softmax}(z) - \mathrm{onehot}(y)$ scaled? `[number]`

**q26.** RMSNorm with $\epsilon = 0$: $y = w \odot x / r$, $r = \sqrt{\frac{1}{n} \sum_i x_i^2}$. (a) For $x = (3, 4)$, $w = (1, 1)$, and upstream $g = (1, 0)$, give $\partial L / \partial x$. `[vector]` (b) Is $x \cdot \partial L / \partial x = 0$ for every $g$ and every $w$? `[bool]`

**q27.** LayerNorm with $\epsilon = 0$, $\gamma = 1$, $\beta = 0$: $y = (x - \mu)/\sigma$ with $\mu$ the mean and $\sigma^2 = \frac{1}{n}\sum_i (x_i - \mu)^2$. For $x = (0, 1, 2)$ and upstream $g = (1, 0, 0)$, give $\partial L / \partial x$. `[vector]`

**q28.** Prove the softmax VJP: if $y = \mathrm{softmax}(x)$ and $g = \partial L / \partial y$, then $\partial L / \partial x = y \odot (g - \langle g, y \rangle \mathbf{1})$. `[proof]`

### Forward versus reverse mode

**q29.** A loss $f: \mathbb{R}^n \to \mathbb{R}$ has $n = 10^6$ parameters. How many (a) Jacobian-vector products (forward mode) and (b) vector-Jacobian products (reverse mode) give the full gradient? `[number]`

**q30.** You need the full Jacobian of $f: \mathbb{R}^3 \to \mathbb{R}^{1000}$. Which mode needs fewer passes? (a) forward (b) reverse `[choice]`

**q31.** Backward of a 16-layer MLP stores each layer's input: a batch of 32 rows of width 1024 in float32. How many bytes of activations does it keep? `[number]`

**q32.** $Y = XW$ with $X$ of shape $m \times k$ and $W$ of shape $k \times n$. (a) How many FLOPs does backward take to compute both $\partial L/\partial X$ and $\partial L/\partial W$? `[expr in m, k, n]` (b) What is the ratio (forward + backward) / forward? `[number]`

### The SDPA Jacobian

Scaled dot-product attention: $S = Q K^\top / \sqrt{d}$, $P = \mathrm{softmax}(S)$ row by row, $O = P V$. Take $T = 2$ positions, $d = 1$, $Q = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$, $K = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$, $V = \begin{pmatrix} 1 \\ 3 \end{pmatrix}$, no mask. Write $e$ as `E`.

**q33.** (a) Give $O$. `[matrix]` With upstream gradient $\partial L / \partial O = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$, give (b) $\partial L / \partial V$ and (c) $\partial L / \partial Q$. `[matrix]`

**q34.** Prove the SDPA backward rules: with $dO = \partial L / \partial O$, $dV = P^\top dO$, $dP = dO\, V^\top$, $dS_{ij} = P_{ij} (dP_{ij} - \sum_k dP_{ik} P_{ik})$, $dQ = dS\, K / \sqrt{d}$, and $dK = dS^\top Q / \sqrt{d}$. `[proof]`
