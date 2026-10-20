# Matrix calculus problem set: differentials, trace trick, VJPs, SDPA Jacobian

By now you have a scalar autograd engine, [Scalar reverse mode: the Value graph](02-scalar-reverse-mode.md), that differentiates one number at a time. It is correct and hopeless at scale: applying one $576 \times 576$ weight to one vector would create a third of a million scalar multiplication nodes. [Matrix differentials, the trace trick, and closed-form VJPs](03-matrix-differentials-and-vjps.md) replaces it with closed-form vector-Jacobian products, one per operation, that take the upstream gradient as an array and return the input gradients as arrays, and [The op library and gradcheck_all](../../ai-engineering/06-numerics-and-autograd/03-op-library-and-gradcheck.md) in *AI Engineering* registers them as the ops of the tensor autograd. Each closed form is a derivation to get right on paper first, because a gradient with the right shape and the wrong value trains, slowly and badly, rather than crashing.

This problem set drills exactly those derivations: the shape bookkeeping, matrix differentials, the trace trick, the VJPs of softmax, log-softmax, cross-entropy, LayerNorm, and RMSNorm, the cost model of forward and reverse mode, and the attention backward that returns in [Scaled dot-product attention (forward and backward)](../../ai-engineering/11-transformer/01-scaled-dot-product-attention.md) and in FlashAttention. It is a pen-and-paper set with no code to write. Every problem is followed by a collapsed worked answer with the final result, the derivation, and the common wrong answers and why they are wrong; 42 answers in all, counting lettered parts, including four proofs (q16, q20, q28, q34) whose model derivations are also in [`course/solutions/solve/S-M08/`](../../../../course/solutions/solve/S-M08/). Try each problem before opening its answer.

The proofs assume the habits of the [Discrete math 1 problem set](../11-discrete-math-1/03-problem-set.md), and the set leans on the gradients and chain rule of the [Calculus 3 problem set](../10-calculus-3/03-problem-set.md). The opening of this chapter, [Matrix Calculus and Autodiff](README.md), gives the wider picture.

> **Learning objectives**
> - Give the shape of any gradient and the strides of any row-major view, and turn broadcasting into a backward sum.
> - Differentiate matrix expressions with the product rule and read off gradients with the trace trick.
> - Compute the VJPs of matmul, softmax, log-softmax, cross-entropy, LayerNorm, and RMSNorm by hand, and check them with their invariances.
> - Count the passes, bytes, and FLOPs of forward and reverse mode.
> - Derive the backward of scaled dot-product attention from the matmul and softmax rules.

## The ideas you will need

The problems use a small set of symbols, collected here.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $L$ | a scalar loss | scalar |
| $X \in \mathbb{R}^{m \times k}$, $W \in \mathbb{R}^{k \times n}$ | matrices; $Y = XW \in \mathbb{R}^{m \times n}$ | |
| $G = \partial L / \partial Y$ | upstream gradient, same shape as $Y$ | $m \times n$ |
| $dX$ | differential: first-order change of $X$ | like $X$ |
| $\operatorname{tr}(A)$ | trace, $\sum_i A_{ii}$ | scalar |
| $\langle A, B \rangle$ | Frobenius inner product, $\operatorname{tr}(A^\top B) = \sum_{ij} A_{ij} B_{ij}$ | scalar |
| $\odot$ | elementwise product | |
| $J_f$ | Jacobian of $f: \mathbb{R}^n \to \mathbb{R}^m$, $(J_f)_{ij} = \partial f_i / \partial x_j$ | $m \times n$ |
| $\mathrm{softmax}(x)_i$ | $e^{x_i} / \sum_k e^{x_k}$ | vector |
| $\delta_{ij}$ | 1 if $i = j$, else 0 | |
| $\mathbf{1}$ | the all-ones vector | |

### Shapes, layouts, and strides

The convention in this book (and in PyTorch) is the **denominator layout**: $\partial L/\partial X$ has the shape of $X$, entry by entry $(\partial L/\partial X)_{ij} = \partial L/\partial X_{ij}$. A row-major array of shape $(n_0, n_1, n_2)$ stores index $(i, j, k)$ at element offset $i\,s_0 + j\,s_1 + k\,s_2$ with **strides** $s = (n_1 n_2, n_2, 1)$. A **view** reinterprets the same memory with other strides: swapping two axes swaps their strides and copies nothing. **Broadcasting** reuses one value along an axis; its backward is a sum over that axis, which the `unbroadcast` of the tensor engine in [Tensor, broadcasting backward, and no_grad](../../ai-engineering/06-numerics-and-autograd/02-tensor-and-broadcasting-backward.md) performs. A **Jacobian** of a map from $\mathbb{R}^n$ to $\mathbb{R}^m$ is $m \times n$, which for a layer is enormous; reverse mode never builds it.

### Differentials

With shapes settled, the next tool is the differential. The **differential** of $F$ at $X$ is the linear part of $F(X + dX) - F(X)$. It obeys the rules you know for scalars, with the order of factors kept: $d(A + B) = dA + dB$, $d(AB) = (dA)B + A(dB)$, $d(A^\top) = (dA)^\top$, $d\operatorname{tr}(A) = \operatorname{tr}(dA)$, and $d(X^{-1}) = -X^{-1}(dX)X^{-1}$ (q16). For $\log \det X$ the result is $\operatorname{tr}(X^{-1} dX)$. For an elementwise function $y = f(x)$, $dy = f'(x) \odot dx$, so its Jacobian is diagonal.

### The trace trick

A differential becomes a gradient through the trace. For a scalar $L$ of a matrix $M$, $dL = \sum_{ij} \frac{\partial L}{\partial M_{ij}}\, dM_{ij} = \langle \partial L/\partial M, dM \rangle = \operatorname{tr}((\partial L/\partial M)^\top dM)$. So **once $dL$ is written as $\operatorname{tr}(N^\top dM)$, the gradient is $N$**. The tool for getting there is the **cyclic property**: $\operatorname{tr}(ABC) = \operatorname{tr}(BCA) = \operatorname{tr}(CAB)$ whenever the products are defined (a cyclic shift, never an arbitrary swap), together with $\operatorname{tr}(A^\top) = \operatorname{tr}(A)$. A scalar is its own trace, so any scalar expression may be wrapped in one.

### VJPs of the ops the library registers

A **vector-Jacobian product** maps an upstream gradient $g = \partial L/\partial y$ to $\partial L/\partial x = J^\top g$ without forming $J$. The table lists the ones implemented in [Matrix differentials, the trace trick, and closed-form VJPs](03-matrix-differentials-and-vjps.md).

| Op | Forward | VJP |
|---|---|---|
| matmul | $Y = XW$ | $\partial L/\partial X = G W^\top$, $\partial L/\partial W = X^\top G$ |
| softmax | $y = \mathrm{softmax}(x)$ | $y \odot (g - \langle g, y \rangle \mathbf{1})$ |
| log-softmax | $\ell = x - \mathrm{LSE}(x)$ | $g - \mathrm{softmax}(x) \sum_i g_i$ |
| cross-entropy | $L = -\log \mathrm{softmax}(z)_t$ | $\mathrm{softmax}(z) - e_t$; a mean over $N$ kept positions divides by $N$ |
| RMSNorm | $y = w \odot x / r$, $r = \sqrt{\tfrac1n \sum x_i^2 + \epsilon}$ | $\frac{w \odot g}{r} - x\,\frac{\langle w \odot g, x\rangle}{n r^3}$ |
| LayerNorm | $\hat x = (x - \mu)/\sigma$, $y = \gamma \odot \hat x + \beta$ | with $\hat g = \gamma \odot g$: $\frac{1}{\sigma}\left(\hat g - \overline{\hat g} - \hat x\, \overline{\hat g \odot \hat x}\right)$ |

Here $\overline{v}$ is the mean of the entries of $v$, $\mu$ the mean of $x$, $\sigma = \sqrt{\tfrac1n \sum (x_i - \mu)^2 + \epsilon}$, and $e_t$ the one-hot vector of the target. **LayerNorm** and **RMSNorm** are functions of a whole row: LayerNorm centers and scales it to mean 0 and variance 1, RMSNorm only scales it to root-mean-square 1, and both then apply a learned gain. Because each normalizes by a statistic of its own input, the gradient has a correction term that a "treat the statistic as constant" derivation misses. Two checks catch it: LayerNorm's input gradient sums to zero (shifting $x$ changes nothing), and RMSNorm's is orthogonal to $x$ when $\epsilon = 0$ (scaling $x$ changes nothing).

### Forward versus reverse mode

Which mode to run is a question of counting. For $f: \mathbb{R}^n \to \mathbb{R}^m$, forward mode (the dual numbers of [Dual numbers and forward-mode autodiff](01-dual-numbers-forward-mode.md)) computes one Jacobian-vector product $J v$ per pass, so the full Jacobian takes $n$ passes. Reverse mode computes one $J^\top u$ per pass, so it takes $m$. A loss has $m = 1$: one backward pass gives the gradient with respect to every parameter, at the price of storing the forward activations it needs. For a matrix product the backward computes two products of the same size as the forward one, so backward costs about twice the forward FLOPs, and a training step about three times: the $6N$ per token of q13 in the [Discrete math 1 problem set](../11-discrete-math-1/03-problem-set.md).

### Attention

Everything above comes together in attention. **Scaled dot-product attention** (SDPA) for $T$ positions computes scores $S = QK^\top/\sqrt{d}$, weights $P = \mathrm{softmax}(S)$ along each row, and output $O = PV$. Its backward chains the three rules above: matmul for $O = PV$, softmax row by row, matmul again for $S$ (q34).

## Two examples worked by hand

Before the set itself, two warm-ups. They are siblings of q21 and q24, not problems of the set.

**Softmax VJP.** Let $x = (0, 0)$, so $y = \mathrm{softmax}(x) = (1/2, 1/2)$, and let the upstream gradient be $g = (1, 3)$. Then $\langle g, y \rangle = 1 \cdot \tfrac12 + 3 \cdot \tfrac12 = 2$, so $g - 2 \cdot \mathbf{1} = (-1, 1)$ and $\partial L/\partial x = y \odot (-1, 1) = (-1/2, 1/2)$.

Check against the Jacobian: for two classes $J = \begin{pmatrix} y_1(1-y_1) & -y_1 y_2 \\ -y_1 y_2 & y_2(1 - y_2)\end{pmatrix} = \begin{pmatrix} 1/4 & -1/4 \\ -1/4 & 1/4 \end{pmatrix}$, and $J^\top g = (1/4 - 3/4,\ -1/4 + 3/4) = (-1/2, 1/2)$. The entries sum to 0, as every softmax input gradient must.

**Cross-entropy.** Logits $z = (\log 3, 0)$, target class 0. Then $\mathrm{softmax}(z) = (3/4, 1/4)$, $L = -\log(3/4) = \log(4/3)$, and $\partial L/\partial z = (3/4 - 1, 1/4 - 0) = (-1/4, 1/4)$: lower the wrong logit, raise the right one, by how far the prediction is from the target.

## The problems

Vectors are flat lists; matrices are lists of rows. A gradient answer has the variable's shape. Answers are exact ($\sqrt2$, $e$), not decimals. Lettered parts are answered separately. The tag after each problem names the form the answer takes.

### Layouts and shapes

The first four problems are bookkeeping: before any derivative, you need to know what shape each gradient has and how arrays sit in memory.

A linear layer computes $Y = X W^\top + b$ for a batch $X$ of shape $(B, T, d_{in}) = (2, 3, 4)$, stored row-major in float32, with $W$ of shape $(d_{out}, d_{in}) = (5, 4)$ and $b$ of shape $(5)$. $L$ is a scalar loss.

**q1.** Give the shape of (a) $Y$, (b) $\partial L / \partial W$, (c) $\partial L / \partial b$. `[vector]`

<details><summary>Answer</summary>

**(a) $(2, 3, 5)$. (b) $(5, 4)$. (c) $(5)$.** (a) The last axis $d_{in} = 4$ is contracted against $W^\top$ of shape $(4, 5)$, leaving $(B, T, d_{out})$. (b) A gradient has its variable's shape, so $\partial L/\partial W$ is $(5, 4)$ like $W$. (c) Likewise $(5)$ like $b$: the forward pass broadcast $b$ over $B$ and $T$, so the backward sums those axes away (q3). Common mistakes: $(2, 3, 4)$ for (a), the input width; $(4, 5)$ for (b), the transpose; $(2, 3, 5)$ for (c), forgetting to sum over the broadcast axes.

</details>

**q2.** (a) At which element offset (counting from 0) is $X[1, 2, 1]$ stored? `[number]` (b) Give the strides of $X$ in elements, one per axis. `[vector]` (c) Give the strides of the view that swaps the last two axes of $X$ (shape $(2, 4, 3)$, no copy). `[vector]`

<details><summary>Answer</summary>

**(a) $21$. (b) $(12, 4, 1)$. (c) $(12, 1, 4)$.** (b) Row-major strides are the products of the trailing dimensions: $(3 \cdot 4, 4, 1)$. (a) Offset $= 1 \cdot 12 + 2 \cdot 4 + 1 \cdot 1 = 21$. (c) A view that swaps the last two axes swaps their strides and copies nothing: $(12, 1, 4)$, now non-contiguous. Common mistakes: $11$ for (a), the column-major offset $1 + 2 \cdot 2 + 1 \cdot 6$; $(48, 16, 4)$ for (b), strides in bytes rather than elements; $(1, 2, 6)$, column-major strides; $(12, 3, 1)$ for (c), the strides a fresh contiguous copy of shape $(2, 4, 3)$ would have.

</details>

**q3.** $b$ is broadcast over the batch and time axes. How many entries of $\partial L / \partial Y$ are summed into each entry of $\partial L / \partial b$? `[number]`

<details><summary>Answer</summary>

**$6$.** $b_j$ is added to $Y[i, t, j]$ for every $(i, t)$, $B \cdot T = 2 \cdot 3$ entries, so $\partial L/\partial b_j = \sum_{i,t} \partial L/\partial Y[i, t, j]$: six terms. Common mistakes: $30$, every entry of $Y$; $2$, summing over the batch axis only.

</details>

**q4.** Flatten $X$ and $Y$ row-major into vectors. What is the shape of the Jacobian $\partial\, \mathrm{vec}(Y) / \partial\, \mathrm{vec}(X)$ (rows, columns)? `[vector]`

<details><summary>Answer</summary>

**$(30, 24)$.** A Jacobian has one row per output and one column per input: $\mathrm{vec}(Y)$ has $2 \cdot 3 \cdot 5 = 30$ entries and $\mathrm{vec}(X)$ has $2 \cdot 3 \cdot 4 = 24$. Nobody builds this matrix (for a real layer it is enormous and mostly zeros); VJPs exist so that you never do. Common mistake: $(24, 30)$, (inputs, outputs).

</details>

### Differentials of matrix expressions

The next group practices the differential rules on small expressions, ending with the proof of the inverse rule that q8 uses.

**q5.** $f(x) = x^\top A x$ with $A = \begin{pmatrix} 1 & 2 \\ 0 & 3 \end{pmatrix}$ and $x = (x_1, x_2)$. Give $\nabla f$. `[vector in x1, x2]`

<details><summary>Answer</summary>

**$(2x_1 + 2x_2,\ 2x_1 + 6x_2)$.** $df = dx^\top A x + x^\top A\,dx = x^\top(A^\top + A)\,dx$, so $\nabla f = (A + A^\top)x$, and $A + A^\top = \begin{pmatrix} 2 & 2 \\ 2 & 6 \end{pmatrix}$. Common mistakes: $2Ax = (2x_1 + 4x_2, 6x_2)$, the formula that holds only for symmetric $A$; $Ax$, dropping one of the two terms of the product rule.

</details>

**q6.** $f(X) = \operatorname{tr}(AX)$ with $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$. Give $\nabla_X f$. `[matrix]`

<details><summary>Answer</summary>

**$\begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix} = A^\top$.** $df = \operatorname{tr}(A\,dX) = \operatorname{tr}((A^\top)^\top dX)$, so the trace trick reads off $\nabla_X f = A^\top$. Entry by entry: $\operatorname{tr}(AX) = \sum_{ij} A_{ij}X_{ji}$, so $\partial f/\partial X_{ji} = A_{ij}$. Common mistake: $A$ itself.

</details>

**q7.** $f(x) = \lVert Ax - b \rVert^2$ with $A = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}$, $b = (1, 0)$. Give $\nabla f$ at $x = (1, 2)$. `[vector]`

<details><summary>Answer</summary>

**$(6, 6)$.** With $r = Ax - b$, $f = r^\top r$, $df = 2r^\top A\,dx$, so $\nabla f = 2A^\top(Ax - b)$. At $x = (1, 2)$: $Ax = (1, 3)$, $r = (0, 3)$, $A^\top r = (0 + 3, 0 + 3) = (3, 3)$, times 2. Common mistakes: $(3, 3)$, dropping the 2 from the square; $(0, 6) = 2(Ax - b)$, forgetting the $A^\top$ that the chain rule brings in.

</details>

**q8.** $X(t) = \begin{pmatrix} 2 & t \\ 1 & 1 \end{pmatrix}$. Give $\frac{d}{dt} X(t)^{-1}$ at $t = 0$. `[matrix]`

<details><summary>Answer</summary>

**$\begin{pmatrix} 1/4 & -1/2 \\ -1/4 & 1/2 \end{pmatrix}$.** Use $d(X^{-1}) = -X^{-1}(dX)X^{-1}$ (q16) with $X(0) = \begin{pmatrix} 2 & 0 \\ 1 & 1 \end{pmatrix}$, $X(0)^{-1} = \begin{pmatrix} 1/2 & 0 \\ -1/2 & 1 \end{pmatrix}$, and $\dot X = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$. Then $X^{-1}\dot X = \begin{pmatrix} 0 & 1/2 \\ 0 & -1/2 \end{pmatrix}$, times $X^{-1}$ gives $\begin{pmatrix} -1/4 & 1/2 \\ 1/4 & -1/2 \end{pmatrix}$, and the minus sign flips it. Common mistakes: the same matrix with the sign lost; $-\dot X = \begin{pmatrix} 0 & -1 \\ 0 & 0 \end{pmatrix}$, as if $X^{-1}$ were $I$.

</details>

**q9.** Give $\nabla_X \log \det X$ at $X = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix}$. `[matrix]`

<details><summary>Answer</summary>

**$X^{-\top} = \begin{pmatrix} 1/2 & 0 \\ -1/6 & 1/3 \end{pmatrix}$.** $d\log\det X = \operatorname{tr}(X^{-1}dX) = \operatorname{tr}((X^{-\top})^\top dX)$, so the gradient is $X^{-\top}$. Here $\det X = 6$ and $X^{-1} = \frac16\begin{pmatrix} 3 & -1 \\ 0 & 2 \end{pmatrix} = \begin{pmatrix} 1/2 & -1/6 \\ 0 & 1/3 \end{pmatrix}$; transpose it. Common mistake: $X^{-1}$, not transposed (the two differ whenever $X$ is not symmetric).

</details>

**q10.** $f(W) = \lVert W x \rVert^2$ with $x = (1, 2)$. Give $\nabla_W f$ at $W = I_2$. `[matrix]`

<details><summary>Answer</summary>

**$\begin{pmatrix} 2 & 4 \\ 4 & 8 \end{pmatrix}$.** $f = (Wx)^\top(Wx)$, so $df = 2(Wx)^\top dW\,x = \operatorname{tr}(2x(Wx)^\top dW)$ and $\nabla_W f = 2(Wx)x^\top$. At $W = I$ this is $2xx^\top$ with $x = (1, 2)$. Common mistake: $xx^\top$, dropping the 2.

</details>

**q11.** Give the derivative of the sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$. `[expr in z]`

<details><summary>Answer</summary>

**$\sigma'(z) = \dfrac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z)(1 - \sigma(z))$.** Chain rule on $(1 + e^{-z})^{-1}$: $-(1 + e^{-z})^{-2} \cdot (-e^{-z})$. The second form follows from $1 - \sigma = e^{-z}/(1 + e^{-z})$, and is the one [The op library and gradcheck_all](../../ai-engineering/06-numerics-and-autograd/03-op-library-and-gradcheck.md) uses because it reuses the forward output. Common mistakes: $\sigma(z)$ itself; the same fraction with a minus sign, losing the sign of the inner derivative $-e^{-z}$.

</details>

**q12.** Give the derivative of softplus, $f(z) = \log(1 + e^z)$. `[expr in z]`

<details><summary>Answer</summary>

**$\sigma(z) = \dfrac{1}{1 + e^{-z}}$.** $\frac{d}{dz}\log(1 + e^z) = \frac{e^z}{1 + e^z}$; divide top and bottom by $e^z$. Softplus is the integral of the sigmoid. Common mistake: $1/(1 + e^z) = \sigma(-z)$.

</details>

**q13.** Give the gradient of $\mathrm{LSE}(x_1, x_2) = \log(e^{x_1} + e^{x_2})$. `[vector in x1, x2]`

<details><summary>Answer</summary>

**$\left(\dfrac{e^{x_1}}{e^{x_1} + e^{x_2}},\ \dfrac{e^{x_2}}{e^{x_1} + e^{x_2}}\right) = \mathrm{softmax}(x)$.** $\partial/\partial x_i$ of $\log Z$ is $(\partial Z/\partial x_i)/Z = e^{x_i}/Z$. The gradient of log-sum-exp is softmax, which [Stable numerics: logsumexp, softmax, and compensated sums](../15-numerical-methods-and-floating-point/02-stable-numerics.md) and [Losses with a fused backward](../../ai-engineering/06-numerics-and-autograd/04-losses-with-fused-backward.md) rest on. Common mistake: $(e^{x_1}, e^{x_2})$, forgetting the derivative of the outer log.

</details>

**q14.** Give the gradient of $f(x) = \lVert x \rVert_2$ at $x = (3, 4)$. `[vector]`

<details><summary>Answer</summary>

**$(3/5, 4/5)$.** $\lVert x\rVert = (x^\top x)^{1/2}$, so $\nabla\lVert x\rVert = \frac12(x^\top x)^{-1/2} \cdot 2x = x/\lVert x\rVert$: the unit vector, with $\lVert(3, 4)\rVert = 5$. Common mistake: $(6, 8) = 2x$, the gradient of the squared norm.

</details>

**q15.** Give the Jacobian of elementwise $\mathrm{ReLU}$ at $x = (-1, 2, 1/2)$. `[matrix]`

<details><summary>Answer</summary>

**$\operatorname{diag}(0, 1, 1)$.** An elementwise function has a diagonal Jacobian with $f'(x_i)$ on the diagonal; $\mathrm{ReLU}'$ is 0 for negative inputs and 1 for positive ones. Common mistakes: $\operatorname{diag}(0, 2, 1/2)$, the outputs on the diagonal instead of the slopes; a matrix of ones, as if each output depended on every input.

</details>

**q16.** Prove $d(X^{-1}) = -X^{-1} (dX) X^{-1}$ for an invertible square matrix $X$. `[proof]`

<details><summary>Answer</summary>

**Claim.** For an invertible $n \times n$ matrix $X$, $d(X^{-1}) = -X^{-1}(dX)X^{-1}$.

**Proof.** The differential obeys the product rule $d(AB) = (dA)B + A(dB)$ (expand $(A + dA)(B + dB)$ and drop the second-order $dA\,dB$), with the order of the factors kept because matrices do not commute. $XX^{-1} = I$ for every invertible $X$ near the current one, and $dI = 0$, so
$$(dX)X^{-1} + X\,d(X^{-1}) = 0 .$$
Hence $X\,d(X^{-1}) = -(dX)X^{-1}$, and multiplying on the left by $X^{-1}$,
$$d(X^{-1}) = -X^{-1}(dX)X^{-1} .$$
For $n = 1$ this is $d(1/x) = -dx/x^2$. Entry by entry, $\partial(X^{-1})_{ij}/\partial X_{kl} = -(X^{-1})_{ik}(X^{-1})_{lj}$.

A complete derivation starts from $XX^{-1} = I$, uses the product rule and $dI = 0$, and keeps the non-commuting order when solving. Model derivation also in [`q16.md`](../../../../course/solutions/solve/S-M08/q16.md).

</details>

### The trace trick

These problems turn the differential into a gradient with the cyclic property, ending with the proof of the matmul VJP.

**q17.** For conformable matrices: (a) is $\operatorname{tr}(ABC) = \operatorname{tr}(CAB)$ always? (b) Is $\operatorname{tr}(ABC) = \operatorname{tr}(BAC)$ always? `[bool]`

<details><summary>Answer</summary>

**(a) Yes. (b) No.** (a) $CAB$ is a cyclic shift of $ABC$, and $\operatorname{tr}(MN) = \operatorname{tr}(NM)$ with $M = AB$, $N = C$ gives it. (b) $BAC$ swaps two factors without rotating; for example $A = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$, $B = A^\top$, $C = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ give $\operatorname{tr}(ABC) = 1$ but $\operatorname{tr}(BAC) = 0$. The trace is invariant under cyclic shifts only.

</details>

**q18.** $Y = XW$ with $X = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, $W = \begin{pmatrix} 1 & 1 \\ 0 & 2 \end{pmatrix}$, and upstream gradient $G = \partial L / \partial Y = I_2$. Give (a) $\partial L / \partial W$ and (b) $\partial L / \partial X$. `[matrix]`

<details><summary>Answer</summary>

**(a) $\begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix}$. (b) $\begin{pmatrix} 1 & 0 \\ 1 & 2 \end{pmatrix}$.** By q20, $\partial L/\partial W = X^\top G = X^\top$ and $\partial L/\partial X = GW^\top = W^\top$ when $G = I$. The `matmul_vjp` of [Matrix differentials, the trace trick, and closed-form VJPs](03-matrix-differentials-and-vjps.md) returns exactly these two. Common mistakes: $GX = X$ for (a), the factors in the wrong order and untransposed; the $\partial L/\partial X$ answer given for (a); $GW = W$ for (b), forgetting the transpose.

</details>

**q19.** The Frobenius inner product is $\langle A, B \rangle = \operatorname{tr}(A^\top B)$. Give it for $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, $B = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$. `[number]`

<details><summary>Answer</summary>

**$5$.** $\langle A, B\rangle = \sum_{ij}A_{ij}B_{ij}$; $B$ picks out the off-diagonal entries, $2 + 3$. Common mistakes: $0 = \operatorname{tr}(A)\operatorname{tr}(B)$, which is not the inner product; $10$, the sum of $A$'s entries.

</details>

**q20.** Let $Y = XW$ and $L$ a scalar with $dL = \operatorname{tr}(G^\top dY)$, $G = \partial L / \partial Y$. Prove $\partial L / \partial W = X^\top G$ and $\partial L / \partial X = G W^\top$. `[proof]`

<details><summary>Answer</summary>

**Claim.** If $Y = XW$ and $dL = \operatorname{tr}(G^\top dY)$ with $G = \partial L/\partial Y$, then $\partial L/\partial W = X^\top G$ and $\partial L/\partial X = GW^\top$.

**Proof.** For a scalar $L$ of a matrix $M$, $dL = \sum_{ij}\frac{\partial L}{\partial M_{ij}}dM_{ij} = \operatorname{tr}((\partial L/\partial M)^\top dM)$, so once $dL$ is written as $\operatorname{tr}(N^\top dM)$ the gradient is $N$ (unique, because the $dM_{ij}$ are independent). By the product rule $dY = (dX)W + X(dW)$, and by linearity of the trace
$$dL = \operatorname{tr}(G^\top(dX)W) + \operatorname{tr}(G^\top X\,dW).$$
Second term: $\operatorname{tr}(G^\top X\,dW) = \operatorname{tr}((X^\top G)^\top dW)$, so $\partial L/\partial W = X^\top G$. First term: by the cyclic property, $\operatorname{tr}(G^\top(dX)W) = \operatorname{tr}(WG^\top dX) = \operatorname{tr}((GW^\top)^\top dX)$, so $\partial L/\partial X = GW^\top$.

Shapes check: $X$ is $m \times k$, $W$ is $k \times n$, $G$ is $m \times n$; $X^\top G$ is $k \times n$ like $W$, and $GW^\top$ is $m \times k$ like $X$. A complete derivation writes $dY$ by the product rule, splits the trace, rotates each term into $\operatorname{tr}(N^\top dM)$ form, and reads off both gradients. Model derivation also in [`q20.md`](../../../../course/solutions/solve/S-M08/q20.md).

</details>

### VJPs by hand

Now the rules the op library registers, each on numbers small enough to check by hand, ending with the proof of the softmax VJP.

**q21.** $y = \mathrm{softmax}(x) = (1/2, 1/4, 1/4)$ and the upstream gradient is $g = (1, 0, 0)$. Give $\partial L / \partial x$. `[vector]`

<details><summary>Answer</summary>

**$(1/4, -1/8, -1/8)$.** $\langle g, y\rangle = 1/2$, $g - \frac12\mathbf 1 = (\frac12, -\frac12, -\frac12)$, and multiplying by $y$ gives the answer. The entries sum to 0, as every softmax input gradient must. Common mistakes: $(1/2, 0, 0) = y \odot g$, dropping the $\langle g, y\rangle$ term; $(1/2, -1/4, -1/4) = g - y$, which confuses softmax with cross-entropy.

</details>

**q22.** Give the Jacobian $\partial y / \partial x$ of a 2-class softmax at $y = (1/4, 3/4)$. `[matrix]`

<details><summary>Answer</summary>

**$\begin{pmatrix} 3/16 & -3/16 \\ -3/16 & 3/16 \end{pmatrix}$.** $J = \operatorname{diag}(y) - yy^\top$, entry $y_i(\delta_{ij} - y_j)$: diagonal $y_i(1 - y_i) = \frac14 \cdot \frac34$, off-diagonal $-y_1y_2 = -\frac3{16}$. Rows sum to 0 because softmax ignores a common shift. Common mistakes: $\operatorname{diag}(y)$, dropping $yy^\top$; all entries $+3/16$, losing the off-diagonal minus.

</details>

**q23.** $\ell = \mathrm{log\_softmax}(x)$ with $\mathrm{softmax}(x) = (1/2, 1/4, 1/4)$ and upstream gradient $g = (0, 1, 0)$. Give $\partial L / \partial x$. `[vector]`

<details><summary>Answer</summary>

**$(-1/2, 3/4, -1/4)$.** The log-softmax VJP is $g - \mathrm{softmax}(x)\sum_i g_i$; $\sum_i g_i = 1$, so it is $(0, 1, 0) - (\frac12, \frac14, \frac14)$. Common mistakes: $(0, 3/4, 0) = g - y \odot g$, mixing the softmax and log-softmax rules; the same vector with the sign flipped.

</details>

**q24.** Cross-entropy with logits $z = (0, \log 2, \log 2)$ and target class 1 (counting from 0): $L = -\log \mathrm{softmax}(z)_1$. (a) Give $\partial L / \partial z$. `[vector]` (b) Give $L$. `[number]`

<details><summary>Answer</summary>

**(a) $(1/5, -3/5, 2/5)$. (b) $\log(5/2)$.** $e^z = (1, 2, 2)$, sum 5, so $\mathrm{softmax}(z) = (1/5, 2/5, 2/5)$. (a) Softmax minus the one-hot of class 1: $(1/5, 2/5 - 1, 2/5)$. (b) $L = -\log(2/5) = \log(5/2) \approx 0.916$. Common mistakes: $(1/5, 2/5, 2/5)$, forgetting the one-hot; $(-1/5, 3/5, -2/5)$, one-hot minus softmax, which is the gradient of the log-likelihood and makes descent climb the loss; $\log(2/5)$, the minus sign missing (a negative loss); $2/5$, the probability rather than the loss.

</details>

**q25.** A batch has 4 positions; one has the target `ignore_index` and the loss is the mean over the other positions. By what factor is each kept position's per-position gradient $\mathrm{softmax}(z) - \mathrm{onehot}(y)$ scaled? `[number]`

<details><summary>Answer</summary>

**$1/3$.** The mean is over the $N = 3$ kept positions, so each kept per-position gradient is divided by 3, and the ignored position contributes nothing. Common mistake: $1/4$, dividing by all positions including the ignored one, which scales the gradient down by the padding fraction.

</details>

**q26.** RMSNorm with $\epsilon = 0$: $y = w \odot x / r$, $r = \sqrt{\frac{1}{n} \sum_i x_i^2}$. (a) For $x = (3, 4)$, $w = (1, 1)$, and upstream $g = (1, 0)$, give $\partial L / \partial x$. `[vector]` (b) Is $x \cdot \partial L / \partial x = 0$ for every $g$ and every $w$? `[bool]`

<details><summary>Answer</summary>

**(a) $\left(\dfrac{16\sqrt2}{125}, -\dfrac{12\sqrt2}{125}\right)$. (b) Yes.** (a) $r = \sqrt{(9 + 16)/2} = 5/\sqrt2$, $w \odot g = (1, 0)$, $\langle w \odot g, x\rangle = 3$, and $nr^3 = 2 \cdot \frac{125}{2\sqrt2} = \frac{125}{\sqrt2}$. First term: $(1, 0)/r = (\frac{\sqrt2}5, 0) = (\frac{25\sqrt2}{125}, 0)$. Correction: $x \cdot \frac{3\sqrt2}{125} = (\frac{9\sqrt2}{125}, \frac{12\sqrt2}{125})$. Subtract. (b) $x \cdot \partial L/\partial x = \frac{\langle w \odot g, x\rangle}{r} - \lVert x\rVert^2\frac{\langle w \odot g, x\rangle}{nr^3}$, and $\lVert x\rVert^2 = nr^2$ when $\epsilon = 0$, so the two terms cancel for every $g$ and $w$. This is scale invariance: $y(cx) = y(x)$ for $c > 0$, so the derivative along $x$ is 0; check (a): $3 \cdot 16 - 4 \cdot 12 = 0$. Common mistakes: $(\sqrt2/5, 0)$, treating $r$ as a constant; $(-9\sqrt2/125, -12\sqrt2/125)$, keeping only the $r$-dependent term.

</details>

**q27.** LayerNorm with $\epsilon = 0$, $\gamma = 1$, $\beta = 0$: $y = (x - \mu)/\sigma$ with $\mu$ the mean and $\sigma^2 = \frac{1}{n}\sum_i (x_i - \mu)^2$. For $x = (0, 1, 2)$ and upstream $g = (1, 0, 0)$, give $\partial L / \partial x$. `[vector]`

<details><summary>Answer</summary>

**$\left(\dfrac{\sqrt6}{12}, -\dfrac{\sqrt6}{6}, \dfrac{\sqrt6}{12}\right)$.** $\mu = 1$, $\sigma^2 = \frac{1 + 0 + 1}3 = \frac23$, so $1/\sigma = \sqrt{3/2} = \sqrt6/2$ and $\hat x = (-\sqrt{3/2}, 0, \sqrt{3/2})$. With $\hat g = g = (1, 0, 0)$: $\overline{\hat g} = \frac13$ and $\overline{\hat g \odot \hat x} = -\frac{\sqrt{3/2}}3$, so $\hat x\,\overline{\hat g \odot \hat x} = (\frac12, 0, -\frac12)$. Then $\hat g - \overline{\hat g} - \hat x\,\overline{\hat g \odot \hat x} = (\frac23 - \frac12, -\frac13, -\frac13 + \frac12) = (\frac16, -\frac13, \frac16)$; multiply by $\sqrt6/2$. The entries sum to 0 (shift invariance) and the gradient is orthogonal to $\hat x$ (scale invariance). Common mistakes: $(\sqrt6/2, 0, 0) = g/\sigma$, treating both $\mu$ and $\sigma$ as constants; $(\sqrt6/3, -\sqrt6/6, -\sqrt6/6)$, keeping the mean's dependence but dropping the variance's.

</details>

**q28.** Prove the softmax VJP: if $y = \mathrm{softmax}(x)$ and $g = \partial L / \partial y$, then $\partial L / \partial x = y \odot (g - \langle g, y \rangle \mathbf{1})$. `[proof]`

<details><summary>Answer</summary>

**Claim.** For $y = \mathrm{softmax}(x)$ and $g = \partial L/\partial y$, $\partial L/\partial x = y \odot (g - \langle g, y\rangle\mathbf 1)$.

**Proof.** Let $Z = \sum_k e^{x_k}$, so $\partial Z/\partial x_j = e^{x_j}$. By the quotient rule,
$$\frac{\partial y_i}{\partial x_j} = \frac{\delta_{ij}e^{x_i}Z - e^{x_i}e^{x_j}}{Z^2} = y_i(\delta_{ij} - y_j).$$
By the chain rule,
$$\Bigl(\frac{\partial L}{\partial x}\Bigr)_j = \sum_i g_i\frac{\partial y_i}{\partial x_j} = \sum_i g_iy_i(\delta_{ij} - y_j) = g_jy_j - y_j\sum_i g_iy_i = y_j(g_j - \langle g, y\rangle).$$
In vector form, $y \odot (g - \langle g, y\rangle\mathbf 1)$. It costs $O(V)$ time and memory and never forms the $V \times V$ Jacobian $\operatorname{diag}(y) - yy^\top$. Its entries sum to $\langle y, g\rangle - \langle g, y\rangle\sum_j y_j = 0$, as they must, since softmax is unchanged by adding a constant to every $x_j$.

A complete derivation gets the Jacobian from the quotient rule, contracts it with $g$, and states the vector form and its cost. Model derivation also in [`q28.md`](../../../../course/solutions/solve/S-M08/q28.md).

</details>

### Forward versus reverse mode

Four short problems on what each mode costs, in passes, bytes, and FLOPs.

**q29.** A loss $f: \mathbb{R}^n \to \mathbb{R}$ has $n = 10^6$ parameters. How many (a) Jacobian-vector products (forward mode) and (b) vector-Jacobian products (reverse mode) give the full gradient? `[number]`

<details><summary>Answer</summary>

**(a) $10^6$. (b) $1$.** Forward mode gives one column $Jv$ per pass, so the $1 \times 10^6$ Jacobian needs one pass per parameter. Reverse mode gives one row $J^\top u$ per pass, and a scalar loss has one row. This is why training uses reverse mode.

</details>

**q30.** You need the full Jacobian of $f: \mathbb{R}^3 \to \mathbb{R}^{1000}$. Which mode needs fewer passes? (a) forward (b) reverse `[choice]`

<details><summary>Answer</summary>

**(a) forward.** Forward mode needs one pass per input (3); reverse mode needs one per output (1000). The rule: pick the mode whose count of passes is the smaller dimension.

</details>

**q31.** Backward of a 16-layer MLP stores each layer's input: a batch of 32 rows of width 1024 in float32. How many bytes of activations does it keep? `[number]`

<details><summary>Answer</summary>

**$2\,097\,152$ bytes** (2 MiB). $16 \text{ layers} \times 32 \text{ rows} \times 1024 \text{ columns} \times 4 \text{ bytes}$. Memory that grows with depth and batch is what the checkpointing of [Hessian-vector products and the recompute-versus-memory schedule](04-hessian-vector-products-and-recompute.md) trades for recompute. Common mistake: $524\,288$, counting floats rather than bytes.

</details>

**q32.** $Y = XW$ with $X$ of shape $m \times k$ and $W$ of shape $k \times n$. (a) How many FLOPs does backward take to compute both $\partial L/\partial X$ and $\partial L/\partial W$? `[expr in m, k, n]` (b) What is the ratio (forward + backward) / forward? `[number]`

<details><summary>Answer</summary>

**(a) $4mkn$. (b) $3$.** Forward: $XW$ costs $2mkn$ FLOPs ($mn$ outputs, $k$ multiply-adds each). Backward computes two products of the same size: $GW^\top$ ($m \times n$ times $n \times k$) and $X^\top G$ ($k \times m$ times $m \times n$), $2mkn$ each. (b) $(2 + 4)/2 = 3$: the $6N$ FLOPs per token of q13 in the [Discrete math 1 problem set](../11-discrete-math-1/03-problem-set.md). Common mistakes: $2mkn$, counting only one of the two products; $mkn$, counting multiply-adds as one FLOP; $2$ for (b), the backward-to-forward ratio rather than the total.

</details>

### The SDPA Jacobian

The set closes by chaining the matmul and softmax rules through scaled dot-product attention.

Scaled dot-product attention: $S = Q K^\top / \sqrt{d}$, $P = \mathrm{softmax}(S)$ row by row, $O = P V$. Take $T = 2$ positions, $d = 1$, $Q = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$, $K = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$, $V = \begin{pmatrix} 1 \\ 3 \end{pmatrix}$, no mask.

**q33.** (a) Give $O$. `[matrix]` With upstream gradient $\partial L / \partial O = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$, give (b) $\partial L / \partial V$ and (c) $\partial L / \partial Q$. `[matrix]`

<details><summary>Answer</summary>

**(a) $\begin{pmatrix} (1 + 3e)/(1 + e) \\ 2 \end{pmatrix}$. (b) $\begin{pmatrix} 1/(1 + e) \\ e/(1 + e) \end{pmatrix}$. (c) $\begin{pmatrix} 2e/(1 + e)^2 \\ 0 \end{pmatrix}$.** (a) $S = QK^\top = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$, so row 0 of $P$ is $(\frac1{1 + e}, \frac e{1 + e})$ and row 1 is $(\frac12, \frac12)$. $O = PV$: row 0 is $\frac{1 + 3e}{1 + e} \approx 2.46$, row 1 is $2$. (b) $dV = P^\top dO$, which with $dO = (1, 0)^\top$ is row 0 of $P$ as a column. (c) $dP = dO\,V^\top = \begin{pmatrix} 1 & 3 \\ 0 & 0 \end{pmatrix}$. Row 0 through the softmax VJP: $\langle dP_0, P_0\rangle = O_0 = \frac{1 + 3e}{1 + e}$, so $dP_0 - O_0 = (\frac{-2e}{1 + e}, \frac{2}{1 + e})$ and $dS_0 = P_0 \odot$ that $= (\frac{-2e}{(1 + e)^2}, \frac{2e}{(1 + e)^2})$; row 1 is 0. $dQ = dS\,K/\sqrt d$ picks column 1 of $dS$ (since $K = (0, 1)^\top$): $(\frac{2e}{(1 + e)^2}, 0)$. Common mistakes: $(3 + e)/(1 + e)$ for (a), the softmax weights put on the wrong keys; $(3, 2)$, argmax instead of softmax; $(\frac1{1 + e}, \frac12)$ for (b), $P\,dO$ instead of $P^\top dO$; $(3, 0)$ for (c), $dP\,K$ with the softmax VJP skipped; the right magnitude with the sign lost.

</details>

**q34.** Prove the SDPA backward rules: with $dO = \partial L / \partial O$, $dV = P^\top dO$, $dP = dO\, V^\top$, $dS_{ij} = P_{ij} (dP_{ij} - \sum_k dP_{ik} P_{ik})$, $dQ = dS\, K / \sqrt{d}$, and $dK = dS^\top Q / \sqrt{d}$. `[proof]`

<details><summary>Answer</summary>

**Claim.** For $S = QK^\top/\sqrt d$, $P = \mathrm{softmax}(S)$ by rows, $O = PV$, and $dO = \partial L/\partial O$: $dV = P^\top dO$, $dP = dO\,V^\top$, $dS_{ij} = P_{ij}(dP_{ij} - \sum_k dP_{ik}P_{ik})$, $dQ = dS\,K/\sqrt d$, $dK = dS^\top Q/\sqrt d$.

Shapes: $Q, K$ are $T \times d$, $V$ is $T \times d_v$, $S$ and $P$ are $T \times T$, $O$ and $dO$ are $T \times d_v$.

**Proof.** Reverse mode through the three steps, each a rule already proved.

1. $O = PV$ is a matrix product. By q20 with $X = P$, $W = V$: $dV = P^\top dO$ ($T \times d_v$, like $V$) and $dP = dO\,V^\top$ ($T \times T$, like $P$).
2. Row $i$ of $P$ is the softmax of row $i$ of $S$, and rows do not interact. By q28 on row $i$ with upstream $dP_{i,:}$: $dS_{ij} = P_{ij}(dP_{ij} - \langle dP_{i,:}, P_{i,:}\rangle)$.
3. $S = QK^\top/\sqrt d$ is a scaled product. By q20 with $X = Q$, $W = K^\top$, and the constant $1/\sqrt d$ carried through: $dQ = dS\,(K^\top)^\top/\sqrt d = dS\,K/\sqrt d$ ($T \times d$, like $Q$), and $d(K^\top) = Q^\top dS/\sqrt d$, so $dK = dS^\top Q/\sqrt d$ ($T \times d$, like $K$).

Every gradient has its variable's shape. A causal mask sets masked $S_{ij}$ to $-\infty$, so $P_{ij} = 0$ there and step 2 gives $dS_{ij} = 0$: no gradient flows through masked entries. FlashAttention's backward recomputes $P$ tile by tile and applies these rules, with $\sum_k dP_{ik}P_{ik} = \sum_k dO_{ik}O_{ik}$ precomputed per row.

A complete derivation gets $dV$ and $dP$ from the matmul rule, applies the softmax VJP row by row, gets $dQ$ and $dK$ from the scaled product, and checks every shape. Model derivation also in [`q34.md`](../../../../course/solutions/solve/S-M08/q34.md).

</details>

## Common mistakes across the set

The wrong answers listed under each problem cluster into a handful of recurring errors. The table collects them, with the problems where each one shows up.

| Mistake | Symptom | The fix |
|---|---|---|
| Gradient transposed relative to its variable | shape error in the optimizer, or a silently wrong square gradient | give every gradient its variable's shape (q1 b); $X^{-\top}$, not $X^{-1}$ (q9) |
| Not summing over broadcast axes | bias gradient of shape (B, T, d) | sum $\partial L/\partial Y$ over $B$ and $T$, six terms per entry (q1 c, q3) |
| Counting bytes as strides, or using a contiguous copy's strides for a view | wrong element read through a view | strides in elements; a transpose view swaps strides (q2) |
| $\nabla x^\top A x = 2Ax$ for a non-symmetric $A$ | wrong gradient for asymmetric forms | $(A + A^\top)x$ from both terms of the product rule (q5) |
| Dropping the factor 2 of a square | gradients half as large; learning rates tuned to hide it | keep the 2 from $d(r^\top r) = 2r^\top dr$ (q7, q10) |
| Swapping a non-cyclic order inside a trace | an identity that holds only for commuting matrices | rotate cyclically only (q17, q20) |
| Softmax VJP without the $\langle g, y\rangle$ term | input gradients that do not sum to zero | $y \odot (g - \langle g, y\rangle\mathbf 1)$ (q21, q28) |
| One-hot minus softmax | gradient ascent on the loss | softmax minus one-hot (q24 a) |
| Mean over all positions including ignored ones | loss and gradient scaled down by the padding fraction | divide by the kept count (q25) |
| Treating a normalization's statistics as constants | RMSNorm or LayerNorm gradients without the correction term | differentiate through $r$, $\mu$, and $\sigma$; check orthogonality and zero sum (q26, q27) |
| Using $P\,dO$ for $dV$ | value gradients from the wrong positions | $dV = P^\top dO$ from the matmul rule (q33 b, q34) |

## Where this leads

Every derivation here becomes code. The dual numbers of [Dual numbers and forward-mode autodiff](01-dual-numbers-forward-mode.md) compute the JVPs of the forward-versus-reverse discussion; [Scalar reverse mode: the Value graph](02-scalar-reverse-mode.md) is the oracle against which the closed-form VJPs are tested; and [Matrix differentials, the trace trick, and closed-form VJPs](03-matrix-differentials-and-vjps.md) turns the VJP table into code, gradchecked rule by rule. In *AI Engineering*, [The op library and gradcheck_all](../../ai-engineering/06-numerics-and-autograd/03-op-library-and-gradcheck.md) registers those VJPs as tensor ops, with the `unbroadcast` of q1 and q3; [Losses with a fused backward](../../ai-engineering/06-numerics-and-autograd/04-losses-with-fused-backward.md) implements fused cross-entropy with `ignore_index` (q24, q25); [Pre-LN and RMSNorm](../../ai-engineering/13-decoder-block/01-pre-ln-rmsnorm.md) puts RMSNorm in the modern block (q26); the attention of the [Attention Origins](../../ai-engineering/10-attention/README.md) chapter and [Scaled dot-product attention (forward and backward)](../../ai-engineering/11-transformer/01-scaled-dot-product-attention.md) use the backward of q33 and q34; and FlashAttention, whose forward pass is built in [FlashAttention forward in C](../../ai-engineering/22-kernels/05-flash-attention-forward.md), recomputes $P$ in its backward and applies q34's rules tile by tile.

Further reading:

- Parr and Howard, "The Matrix Calculus You Need for Deep Learning" (2018).
- Minka, "Old and New Matrix Algebra Useful for Statistics" (2000).
- Baydin et al., "Automatic Differentiation in Machine Learning: a Survey" (2018), sections 2 and 3.
- Dao et al., "FlashAttention" (2022), appendix B.

## Summary

- A gradient has the shape of its variable, and broadcasting in the forward pass becomes a sum over the broadcast axes in the backward pass (q1, q3).
- Write the differential $dL = \operatorname{tr}(G^\top dY)$, push $dY$ through the product rule, rotate with the cyclic trace, and read off the gradient: $\partial L/\partial W = X^\top G$, $\partial L/\partial X = G W^\top$ (q18, q20).
- Softmax's VJP is $y \odot (g - \langle g, y\rangle)$ and cross-entropy's gradient with respect to logits is $\mathrm{softmax}(z) - \mathrm{onehot}(y)$, both $O(V)$ (q21, q24, q28).
- Normalizations differentiate through their statistics: LayerNorm's input gradient sums to zero, and RMSNorm's is orthogonal to $x$ (q26, q27).
- Reverse mode costs one pass per output, so one backward pass gives a million-parameter gradient, at about twice the forward FLOPs (q29, q32).
