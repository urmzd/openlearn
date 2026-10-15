<!-- ol:module S-M08 -->
# Matrix calculus problem set: differentials, trace trick, VJPs, SDPA Jacobian

## Overview

| | |
|---|---|
| **Module** | `S-M08` · solve · none · Pass 2 · 8 to 10 h |
| **You build** | answers in `solve/S-M08.toml` (42 checked by SymPy) and 4 derivations in `solve/S-M08/qN.md` (self-graded against their rubrics) |
| **Contract** | none: a pen and paper set |
| **Tests** | `course/solve/S-M08/key.toml` (hidden): typed answers plus reject canaries; the problems are in `course/solve/S-M08/problems.md` and in section 4 |
| **Needs** | `S-M05` (proof habits). Reading: the [Matrix Calculus and Autodiff topic](README.md), and the gradients and chain rule of `S-M04` |
| **Used by** | no call site (a solve set). It checks the derivations behind `M08.3` (`matmul_vjp`, `softmax_vjp`, `log_softmax_vjp`, `layernorm_vjp`, `rmsnorm_vjp`, `cross_entropy_vjp`), which `L0.2`, `L0.3`, and `L7.1` register as ops; q33 and q34 are the attention backward of `L4.*` and `L9.4` |
| **Milestone** | `MS-P2` (the Pass 2 gate runs `ol check` on every solve part of the pass) |
| **Optional depth** | Parr and Howard, "The Matrix Calculus You Need for Deep Learning" (2018); Minka, "Old and New Matrix Algebra Useful for Statistics" (2000); Baydin et al., "Automatic Differentiation in Machine Learning: a Survey" (2018), sections 2 and 3; Dao et al., "FlashAttention" (2022), appendix B |

## Key Takeaways

- A gradient has the shape of its variable, and broadcasting in the forward pass becomes a sum over the broadcast axes in the backward pass (q1, q3).
- Write the differential $dL = \operatorname{tr}(G^\top dY)$, push $dY$ through the product rule, rotate with the cyclic trace, and read off the gradient: $\partial L/\partial W = X^\top G$, $\partial L/\partial X = G W^\top$ (q18, q20).
- Softmax's VJP is $y \odot (g - \langle g, y\rangle)$ and cross-entropy's gradient with respect to logits is $\mathrm{softmax}(z) - \mathrm{onehot}(y)$, both $O(V)$ (q21, q24, q28).
- Normalizations differentiate through their statistics: LayerNorm's input gradient sums to zero, and RMSNorm's is orthogonal to $x$ (q26, q27).
- Reverse mode costs one pass per output, so one backward pass gives a million-parameter gradient, at about twice the forward FLOPs (q29, q32).

## How to work this chapter

```bash
ol start S-M08              # writes solve/S-M08.toml and one file per derivation
ol check S-M08              # SymPy checks the answers, then asks each rubric (y/n)
ol check S-M08 --regrade    # ask the rubrics again after you change a derivation
```

---

## 1. Why now

By this point in Pass 2 you have a scalar autograd engine (`M08.2`) that differentiates one number at a time. It is correct and hopeless at scale: applying one $576 \times 576$ weight to one vector would create a third of a million scalar multiplication nodes. `M08.3` replaces it with closed-form vector-Jacobian products, one per operation, that take the upstream gradient as an array and return the input gradients as arrays, and `L0.2` registers them as the ops of your tensor autograd. Each closed form is a derivation you must get right on paper first, because a gradient with the right shape and the wrong value trains, slowly and badly, rather than crashing. This set drills the shape bookkeeping, matrix differentials, the trace trick, the VJPs of softmax, log-softmax, cross-entropy, LayerNorm, and RMSNorm, the cost model of forward and reverse mode, and the attention backward you will meet again in `L9.4`.

## 2. Principles

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

### 2.1 Shapes, layouts, and strides

The convention in this course (and in PyTorch) is the **denominator layout**: $\partial L/\partial X$ has the shape of $X$, entry by entry $(\partial L/\partial X)_{ij} = \partial L/\partial X_{ij}$. A row-major array of shape $(n_0, n_1, n_2)$ stores index $(i, j, k)$ at element offset $i\,s_0 + j\,s_1 + k\,s_2$ with **strides** $s = (n_1 n_2, n_2, 1)$. A **view** reinterprets the same memory with other strides: swapping two axes swaps their strides and copies nothing. **Broadcasting** reuses one value along an axis; its backward is a sum over that axis, which `L0.1`'s `unbroadcast` performs. A **Jacobian** of a map from $\mathbb{R}^n$ to $\mathbb{R}^m$ is $m \times n$, which for a layer is enormous; reverse mode never builds it.

### 2.2 Differentials

The **differential** of $F$ at $X$ is the linear part of $F(X + dX) - F(X)$. It obeys the rules you know for scalars, with the order of factors kept: $d(A + B) = dA + dB$, $d(AB) = (dA)B + A(dB)$, $d(A^\top) = (dA)^\top$, $d\operatorname{tr}(A) = \operatorname{tr}(dA)$, and $d(X^{-1}) = -X^{-1}(dX)X^{-1}$ (q16). For $\log \det X$ the result is $\operatorname{tr}(X^{-1} dX)$. For an elementwise function $y = f(x)$, $dy = f'(x) \odot dx$, so its Jacobian is diagonal.

### 2.3 The trace trick

For a scalar $L$ of a matrix $M$, $dL = \sum_{ij} \frac{\partial L}{\partial M_{ij}}\, dM_{ij} = \langle \partial L/\partial M, dM \rangle = \operatorname{tr}((\partial L/\partial M)^\top dM)$. So **once $dL$ is written as $\operatorname{tr}(N^\top dM)$, the gradient is $N$**. The tool for getting there is the **cyclic property**: $\operatorname{tr}(ABC) = \operatorname{tr}(BCA) = \operatorname{tr}(CAB)$ whenever the products are defined (a cyclic shift, never an arbitrary swap), together with $\operatorname{tr}(A^\top) = \operatorname{tr}(A)$. A scalar is its own trace, so any scalar expression may be wrapped in one.

### 2.4 VJPs of the ops you will register

A **vector-Jacobian product** maps an upstream gradient $g = \partial L/\partial y$ to $\partial L/\partial x = J^\top g$ without forming $J$. The ones `M08.3` implements:

| Op | Forward | VJP |
|---|---|---|
| matmul | $Y = XW$ | $\partial L/\partial X = G W^\top$, $\partial L/\partial W = X^\top G$ |
| softmax | $y = \mathrm{softmax}(x)$ | $y \odot (g - \langle g, y \rangle \mathbf{1})$ |
| log-softmax | $\ell = x - \mathrm{LSE}(x)$ | $g - \mathrm{softmax}(x) \sum_i g_i$ |
| cross-entropy | $L = -\log \mathrm{softmax}(z)_t$ | $\mathrm{softmax}(z) - e_t$; a mean over $N$ kept positions divides by $N$ |
| RMSNorm | $y = w \odot x / r$, $r = \sqrt{\tfrac1n \sum x_i^2 + \epsilon}$ | $\frac{w \odot g}{r} - x\,\frac{\langle w \odot g, x\rangle}{n r^3}$ |
| LayerNorm | $\hat x = (x - \mu)/\sigma$, $y = \gamma \odot \hat x + \beta$ | with $\hat g = \gamma \odot g$: $\frac{1}{\sigma}\left(\hat g - \overline{\hat g} - \hat x\, \overline{\hat g \odot \hat x}\right)$ |

Here $\overline{v}$ is the mean of the entries of $v$, $\mu$ the mean of $x$, $\sigma = \sqrt{\tfrac1n \sum (x_i - \mu)^2 + \epsilon}$, and $e_t$ the one-hot vector of the target. **LayerNorm** and **RMSNorm** are functions of a whole row: LayerNorm centers and scales it to mean 0 and variance 1, RMSNorm only scales it to root-mean-square 1, and both then apply a learned gain. Because each normalizes by a statistic of its own input, the gradient has a correction term that a "treat the statistic as constant" derivation misses. Two checks catch it: LayerNorm's input gradient sums to zero (shifting $x$ changes nothing), and RMSNorm's is orthogonal to $x$ when $\epsilon = 0$ (scaling $x$ changes nothing).

### 2.5 Forward versus reverse mode

For $f: \mathbb{R}^n \to \mathbb{R}^m$, forward mode (`M08.1`'s dual numbers) computes one Jacobian-vector product $J v$ per pass, so the full Jacobian takes $n$ passes. Reverse mode computes one $J^\top u$ per pass, so it takes $m$. A loss has $m = 1$: one backward pass gives the gradient with respect to every parameter, at the price of storing the forward activations it needs. For a matrix product the backward computes two products of the same size as the forward one, so backward costs about twice the forward FLOPs, and a training step about three times: the $6N$ per token of `S-M05` q13.

### 2.6 Attention

**Scaled dot-product attention** (SDPA) for $T$ positions computes scores $S = QK^\top/\sqrt{d}$, weights $P = \mathrm{softmax}(S)$ along each row, and output $O = PV$. Its backward chains the three rules above: matmul for $O = PV$, softmax row by row, matmul again for $S$ (q34).

## 3. Worked example by hand

This is a sibling of q21 and q24, not one of the graded problems.

**Softmax VJP.** Let $x = (0, 0)$, so $y = \mathrm{softmax}(x) = (1/2, 1/2)$, and let the upstream gradient be $g = (1, 3)$.

- $\langle g, y \rangle = 1 \cdot \tfrac12 + 3 \cdot \tfrac12 = 2$.
- $g - 2 \cdot \mathbf{1} = (-1, 1)$.
- $\partial L/\partial x = y \odot (-1, 1) = (-1/2, 1/2)$.

Check against the Jacobian: for two classes $J = \begin{pmatrix} y_1(1-y_1) & -y_1 y_2 \\ -y_1 y_2 & y_2(1 - y_2)\end{pmatrix} = \begin{pmatrix} 1/4 & -1/4 \\ -1/4 & 1/4 \end{pmatrix}$, and $J^\top g = (1/4 - 3/4,\ -1/4 + 3/4) = (-1/2, 1/2)$. The entries sum to 0, as every softmax input gradient must. In `solve/` this is `answer = "[-1/2, 1/2]"`.

**Cross-entropy.** Logits $z = (\log 3, 0)$, target class 0. Then $\mathrm{softmax}(z) = (3/4, 1/4)$, $L = -\log(3/4) = \log(4/3)$, and $\partial L/\partial z = (3/4 - 1, 1/4 - 0) = (-1/4, 1/4)$: lower the wrong logit, raise the right one, by how far the prediction is from the target.

## 4. The problem set

Write each answer in `solve/S-M08.toml`; lettered parts are their own tables:

```toml
[q1.a]
answer = "[2, 3, 5]"
[q5]
answer = "[2*x1 + 2*x2, 2*x1 + 6*x2]"
[q18.a]
answer = "[[1, 3], [2, 4]]"
[q20]
proof = "S-M08/q20.md"
```

Vectors are flat lists; matrices are lists of rows; write $e$ as `E` and $\sqrt{\cdot}$ as `sqrt(...)`. A gradient answer must have the variable's shape.

<!-- ol:problems S-M08 -->

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

<!-- /ol:problems -->

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Gradient transposed relative to its variable | shape error in the optimizer, or a silently wrong square gradient | q1 (canary [4, 5]), q9 (canary: the inverse) |
| Not summing over broadcast axes | bias gradient of shape (B, T, d) | q1, q3 (canaries 30 and 2) |
| Counting bytes as strides, or using a contiguous copy's strides for a view | wrong element read through a view | q2 (canaries in bytes and [12, 3, 1]) |
| $\nabla x^\top A x = 2Ax$ for a non-symmetric $A$ | wrong gradient for asymmetric forms | q5 (canary 2Ax) |
| Dropping the factor 2 of a square | gradients half as large; learning rates tuned to hide it | q7, q10 (canaries) |
| Swapping a non-cyclic order inside a trace | an identity that holds only for commuting matrices | q17 (canary true) |
| Softmax VJP without the $\langle g, y\rangle$ term | input gradients that do not sum to zero | q21 (canary y * g) |
| One-hot minus softmax | gradient ascent on the loss | q24 (canary with the sign flipped) |
| Mean over all positions including ignored ones | loss and gradient scaled down by the padding fraction | q25 (canary 1/4) |
| Treating a normalization's statistics as constants | RMSNorm or LayerNorm gradients without the correction term | q26 and q27 (canaries) |
| Using $P\,dO$ for $dV$ | value gradients from the wrong positions | q33 (canary) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `S-M05` | proof structure for q16, q20, q28, q34 |
| Forward | `M08.1` | dual numbers compute the JVPs of section 2.5 |
| Forward | `M08.2` | scalar reverse mode, the oracle against which `M08.3` is tested |
| Forward | `M08.3` | the VJP table of section 2.4 as code, gradchecked rule by rule |
| Forward | `L0.2` | registers those VJPs as tensor ops, with `unbroadcast` from q1 and q3 |
| Forward | `L0.3` | fused cross-entropy with `ignore_index` (q24, q25) |
| Forward | `L7.1` | RMSNorm in the modern block (q26) |
| Forward | `L9.4` | FlashAttention backward recomputes $P$ and applies q34's rules tile by tile |
