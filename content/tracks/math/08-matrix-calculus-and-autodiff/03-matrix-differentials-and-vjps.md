<!-- ol:module M08.3 -->
# Matrix differentials, the trace trick, and closed-form VJPs

## Overview

| | |
|---|---|
| **Module** | `M08.3` · build · Python · Pass 2 · 3 to 4 h |
| **You build** | `python/tinyllm/autograd/vjp.py`: `unbroadcast`, `matmul_vjp`, `softmax_vjp`, `log_softmax_vjp`, `layernorm_vjp`, `rmsnorm_vjp`, `cross_entropy_vjp` |
| **Contract** | [`course/contracts/py/tinyllm/autograd/vjp.pyi`](../../../../course/contracts/py/tinyllm/autograd/vjp.pyi) |
| **Tests** | `course/tests/M08.3/` (what they check: section 4) |
| **Needs** | `M09.2` stable softmax · `M11.1` cross-entropy (the tests differentiate it) · `M04.2` numeric VJP · reading: `M03.1` matrices and matmul shapes (or `--ref-deps`) |
| **Used by** | `L0.1` `unbroadcast` in your Tensor's backward · `L0.2` op VJPs · `L0.3` fused cross-entropy · later: `L3.1` backpropagation through time by hand, `L7.1` RMSNorm |
| **Milestone** | `MS-P2` (Pass 2 gate: every math module of the pass checks green, then your autograd bigram trains) |
| **Optional depth** | Parr and Howard, [*The Matrix Calculus You Need for Deep Learning*](https://arxiv.org/abs/1802.01528); Minka, "Old and New Matrix Algebra Useful for Statistics" (the differential method); Petersen and Pedersen, *The Matrix Cookbook*, sections 2 and 4 |

## Key Takeaways

- For a scalar loss, $dL = \operatorname{tr}(G^\top dY)$ with $G = \partial L / \partial Y$; write $dY$ in terms of $dX$, move $dX$ to the right with the cyclic property of the trace, and the matrix in front of it is $\partial L / \partial X$ (`test_trace_identity`).
- That gives every rule in a few lines: $\bar A = G B^\top$ and $\bar B = A^\top G$ for a matmul, $y \odot (g - \langle g, y\rangle)$ for softmax, $(\mathrm{softmax} - \mathrm{onehot}) / n$ for cross-entropy (`test_hand_example`, `test_matmul_vjp_gradcheck`).
- LayerNorm and RMSNorm divide by a statistic of every input, so their VJPs carry correction terms that a "treat the statistic as a constant" derivation drops (`test_layernorm_vjp_gradcheck`, `test_rmsnorm_vjp_gradcheck`).
- A broadcast input receives the sum of its copies' gradients (`test_unbroadcast_shapes`), and padded positions receive exactly zero (`test_cross_entropy_ignore_index`).

## How to work this chapter

```bash
ol start M08.3              # stubs vjp.py into your repo, contract alongside
ol tests M08.3              # read the test catalog first: rung R0, you write no tests here
ol check M08.3              # exit code is the verdict
ol check M08.3 --ref-deps   # only if your M09.2, M11.1, or M04.2 is not passing yet
ol diff  M08.3              # after passing: your code against the reference
```

---

## 1. Why now

`M08.2` gave you reverse mode one scalar at a time. Your bigram's forward pass is a $[T, 256] \times [256, 256]$ matmul followed by a softmax over 256 entries per row, about 16 million multiply-adds per step; spelled out as Python `Value` objects that is minutes per step instead of milliseconds. `L0.2` builds a tensor op library instead, where each op (matmul, softmax, log-softmax, LayerNorm, RMSNorm, cross-entropy) has one backward function written in numpy. Each needs a closed-form vector-Jacobian product, derived once on paper and proven by gradcheck. This module derives and implements them, so that `L0.2` only wires them into the graph, `L0.3` fuses softmax with cross-entropy, `L3.1` reuses them to backpropagate through time by hand, and `L7.1`'s RMSNorm has its backward ready.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $L$ | a scalar loss | float |
| $X, Y = f(X)$ | an op's input and output | arrays |
| $G = \bar Y = \partial L / \partial Y$ | the upstream gradient: same shape as $Y$ | array |
| $\bar X = \partial L / \partial X$ | the VJP's result: same shape as $X$ | array |
| $dX$ | a differential: an arbitrary small change of $X$ | same shape as $X$ |
| $\langle A, B\rangle = \sum_{ij} A_{ij} B_{ij} = \operatorname{tr}(A^\top B)$ | the inner product of two same-shape arrays | float |
| $\odot$ | elementwise product | |
| $\mathbf{1}$ | the vector of ones | `float[D]` |
| $D$ | the size of the normalized (last) axis | `int` |
| $\mu, \sigma^2$ | mean and variance of a row over its $D$ entries | float per row |
| $\epsilon$ | small constant added to the variance ($10^{-5}$ for LayerNorm, $10^{-6}$ for RMSNorm in this course's models) | float |
| $r$ (`rstd`) | reciprocal standard deviation, $1/\sqrt{\sigma^2 + \epsilon}$ (LayerNorm) or $1/\sqrt{\mathrm{mean}(x^2) + \epsilon}$ (RMSNorm) | float per row |
| $\hat x = (x - \mu)\, r$ | the normalized row | `float[D]` |
| $\gamma, \beta, w$ | learned scale and shift (LayerNorm), learned scale (RMSNorm) | `float[D]` |
| $y = \mathrm{softmax}(x)$, $\ell = \mathrm{logsoftmax}(x)$ | probabilities and log-probabilities of a row | `float[V]` |
| $t$, $n$ | a row's target class; the number of rows whose target is not `ignore_index` | `int` |

**Gradients have the shape of their variable.** Whatever layout convention a textbook uses, here $\bar X$ is stored with exactly the shape of $X$, and the entry $\bar X_{ij}$ is $\partial L / \partial X_{ij}$. Then the first-order change of the loss is

$$dL = \sum_{ij} \bar X_{ij}\, dX_{ij} = \langle \bar X, dX\rangle.$$

**The recipe.** For $Y = f(X)$, the chain rule says $dL = \langle G, dY\rangle$. Write $dY$ as a linear expression in $dX$, then rearrange $\langle G, dY\rangle$ into the form $\langle \text{something}, dX\rangle$. Since this holds for every $dX$, the something is $\bar X$. Two tools do the rearranging:

- the **cyclic property** of the trace: $\operatorname{tr}(ABC) = \operatorname{tr}(CAB) = \operatorname{tr}(BCA)$, and $\operatorname{tr}(A^\top) = \operatorname{tr}(A)$;
- moving factors across the inner product: $\langle A, BC\rangle = \langle B^\top A, C\rangle = \langle A C^\top, B\rangle$.

**Matmul.** $Y = AB$ with $A \in \mathbb{R}^{m \times k}$, $B \in \mathbb{R}^{k \times n}$. The product rule holds for matrices (keep the order): $dY = dA\, B + A\, dB$. Then

$$\langle G, dA\, B\rangle = \operatorname{tr}(G^\top dA\, B) = \operatorname{tr}(B G^\top dA) = \langle G B^\top, dA\rangle, \qquad \langle G, A\, dB\rangle = \langle A^\top G, dB\rangle,$$

so $\bar A = G B^\top$ and $\bar B = A^\top G$. Check the shapes: $G$ is $m \times n$ and $B^\top$ is $n \times k$, so $\bar A$ is $m \times k$ like $A$. Shape-checking catches many mistakes but not all: if $B$ is square, $G B$ has the right shape and the wrong values.

**Broadcasting.** numpy's matmul broadcasts batch axes: a weight $B$ of shape $[k, n]$ times activations $[T, m, k]$ is the same $B$ used $T$ times. Each use contributes a gradient, so $\bar B$ is the sum over the batch axis. In general, if the forward pass broadcast $X$ from shape $s$ to a larger shape, `unbroadcast` sums the gradient over every axis that was added in front and over every axis where $s$ has size 1 (keeping it as size 1).

**Softmax.** For one row, $y_i = e^{x_i} / \sum_j e^{x_j}$. Differentiating the quotient gives $dy_i = y_i\,(dx_i - \sum_j y_j\, dx_j)$, that is, $dy = y \odot (dx - \langle y, dx\rangle \mathbf{1})$; the Jacobian is $\mathrm{diag}(y) - y y^\top$. Then

$$\langle g, dy\rangle = \sum_i g_i y_i\, dx_i - \Big(\sum_i g_i y_i\Big)\Big(\sum_j y_j\, dx_j\Big) = \big\langle y \odot (g - \langle g, y\rangle \mathbf{1}),\; dx\big\rangle,$$

so $\bar x = y \odot (g - \langle g, y\rangle)$. The VJP needs only the saved output $y$; the diagonal term alone, $y \odot g$, is a common half-derivation.

**Log-softmax.** $\ell = x - \mathrm{LSE}(x)\mathbf{1}$ and $d\,\mathrm{LSE} = \langle y, dx\rangle$ (the gradient of log-sum-exp is softmax), so $d\ell = dx - \langle y, dx\rangle \mathbf{1}$ and

$$\bar x = g - y \sum_i g_i, \qquad y = e^{\ell}.$$

The function receives the saved log-probabilities $\ell$ and exponentiates them.

**Cross-entropy, fused.** For logits $z$ (one row per position) and targets $t$, the loss is the mean over the $n$ valid rows of $-\ell_{t}$. For a valid row the upstream gradient of $\ell$ is $g = -\tfrac1n e_t$ (a one-hot vector scaled), with $\sum_i g_i = -\tfrac1n$. Plug into the log-softmax VJP:

$$\bar z = -\tfrac1n e_t + \tfrac1n y = \frac{\mathrm{softmax}(z) - \mathrm{onehot}(t)}{n}.$$

Rows whose target is `ignore_index` (padding, $-100$ as in PyTorch) are not in the loss, so their gradient is exactly 0 and they do not count in $n$. Use `M09.2`'s softmax, so logits near $10^4$ give a finite gradient. A target outside $[0, V)$ that is not `ignore_index` is a data bug, not a class: numpy would quietly wrap $-1$ to the last column.

**LayerNorm, defined.** LayerNorm (Ba, Kiros, and Hinton, 2016) standardizes each row of features, then applies a learned scale and shift:

$$\mu = \tfrac1D \textstyle\sum_i x_i,\quad \sigma^2 = \tfrac1D \sum_i (x_i - \mu)^2,\quad r = \frac{1}{\sqrt{\sigma^2 + \epsilon}},\quad \hat x = (x - \mu)\,r,\quad y = \gamma \odot \hat x + \beta.$$

Every token's features come out with mean 0 and variance about 1 before the scale, which keeps activations in a range where training is stable (the 2017 transformer of `L5` uses it). The forward pass saves $\hat x$ and $r$.

**LayerNorm's VJP.** The parameter gradients are immediate from $y = \gamma \odot \hat x + \beta$: summing over every row, $\bar\gamma = \sum_{\text{rows}} g \odot \hat x$ and $\bar\beta = \sum_{\text{rows}} g$. For $x$, let $d = g \odot \gamma$ be the gradient reaching $\hat x$. Both $\mu$ and $r$ depend on every $x_i$:

$$d\mu = \tfrac1D \langle \mathbf{1}, dx\rangle, \qquad d\sigma^2 = \tfrac2D \langle x - \mu, dx\rangle \;(\text{because } \textstyle\sum_i (x_i - \mu) = 0), \qquad dr = -\tfrac12 r^3\, d\sigma^2.$$

So $d\hat x = r\,(dx - d\mu\,\mathbf{1}) + (x - \mu)\,dr = r\,(dx - \tfrac1D\langle\mathbf{1}, dx\rangle\mathbf{1}) - \tfrac rD\, \hat x\, \langle \hat x, dx\rangle$. Taking $\langle d, \cdot\rangle$ and moving $dx$ to the right:

$$\bar x = r\left(d - \mathrm{mean}(d)\,\mathbf{1} - \hat x\; \mathrm{mean}(d \odot \hat x)\right).$$

Three terms: the direct path, the path through the mean, and the path through the variance. Dropping either correction is "treating $\mu$ (or $r$) as a constant", and gradcheck catches it immediately. A consequence worth noticing: $\langle \mathbf{1}, \bar x\rangle = 0$, because adding a constant to $x$ does not change $\hat x$.

**RMSNorm, defined.** RMSNorm (Zhang and Sennrich, 2019) drops the mean and the shift: it divides by the root mean square,

$$r = \frac{1}{\sqrt{\tfrac1D \sum_i x_i^2 + \epsilon}}, \qquad y = w \odot x\, r.$$

It is cheaper and works as well in practice; Llama-family models (`L7.1`, SmolLM2) use it before every attention and MLP block.

**RMSNorm's VJP.** With $d = g \odot w$ and $dr = -\tfrac12 r^3 \cdot \tfrac2D \langle x, dx\rangle = -\tfrac{r^3}{D} \langle x, dx\rangle$:

$$\langle d, r\,dx + x\,dr\rangle = r\langle d, dx\rangle - \tfrac{r^3}{D}\langle d, x\rangle \langle x, dx\rangle \;\Rightarrow\; \bar x = r\left(d - x\, r^2\, \mathrm{mean}(d \odot x)\right),$$

and $\bar w = \sum_{\text{rows}} g \odot x\, r$. Here $\langle x, \bar x\rangle = 0$ when $\epsilon = 0$: scaling $x$ does not change the output.

**Pure functions.** A VJP reads the saved values and the upstream gradient and returns new arrays. Writing into its arguments (subtracting the one-hot from a softmax buffer you were handed, updating `g` in place) corrupts values the graph still needs.

## 3. Worked example by hand

**Matmul.** $A = \begin{bmatrix}1&2\\3&4\end{bmatrix}$, $B = \begin{bmatrix}1&0\\2&1\end{bmatrix}$, and $G = \begin{bmatrix}1&0\\0&0\end{bmatrix}$, so $L = Y_{11} = a_{11}b_{11} + a_{12}b_{21}$. Directly: $\partial L/\partial a_{11} = b_{11} = 1$, $\partial L/\partial a_{12} = b_{21} = 2$, $\partial L/\partial b_{11} = a_{11} = 1$, $\partial L/\partial b_{21} = a_{12} = 2$, everything else 0. The formulas agree: $G B^\top = \begin{bmatrix}1&2\\0&0\end{bmatrix}$ and $A^\top G = \begin{bmatrix}1&0\\2&0\end{bmatrix}$. Using $GB$ instead gives $\begin{bmatrix}1&0\\0&0\end{bmatrix}$: the right shape, the wrong gradient.

**Softmax.** $x = [0, \ln 3]$: $e^x = [1, 3]$, $y = [1/4, 3/4]$. With $g = [1, 0]$, $\langle g, y\rangle = 1/4$, so $\bar x = [\tfrac14(1 - \tfrac14), \tfrac34(0 - \tfrac14)] = [3/16, -3/16]$. The entries sum to 0, as they must: adding a constant to $x$ does not change $y$.

**Cross-entropy.** The same logits with target 1 and $n = 1$: $\bar z = y - e_1 = [1/4, -1/4]$. The loss is $-\ln(3/4) = 0.2877$.

**LayerNorm** of $x = [1, 2, 6]$ with $\gamma = \mathbf{1}$, $\epsilon = 0$, $g = [1, 0, 0]$:

| quantity | value |
|---|---|
| $\mu$, $x - \mu$ | 3, $[-2, -1, 3]$ |
| $\sigma^2$, $r$ | $14/3$, $\sqrt{3/14} = 0.462910$ |
| $\hat x$ | $[-0.925820, -0.462910, 1.388730]$ |
| $d = g \odot \gamma$, $\mathrm{mean}(d)$ | $[1, 0, 0]$, $1/3$ |
| $\mathrm{mean}(d \odot \hat x)$ | $-0.925820 / 3 = -0.308607$ |
| $\hat x \cdot \mathrm{mean}(d \odot \hat x)$ | $[2/7, 1/7, -3/7]$ |
| $d - \mathrm{mean}(d) - $ that | $[8/21, -10/21, 2/21]$ |
| $\bar x = r \cdot$ that | $[0.176347, -0.220433, 0.044087]$ |

The entries of $\bar x$ sum to 0. The parameter gradients are $\bar\gamma = g \odot \hat x = [-0.925820, 0, 0]$ and $\bar\beta = [1, 0, 0]$.

**RMSNorm** of $x = [3, 4]$ with $w = \mathbf{1}$, $\epsilon = 0$, $g = [1, 0]$: $\mathrm{mean}(x^2) = 12.5$, $r = 0.282843$, $y = [0.848528, 1.131371]$. Then $d = [1, 0]$, $\mathrm{mean}(d \odot x) = 1.5$, $r^2 = 0.08$, $x r^2 \cdot 1.5 = [0.36, 0.48]$, and $\bar x = r\,[0.64, -0.48] = [0.181019, -0.135765]$. Check: $\langle x, \bar x\rangle \propto 3(0.64) + 4(-0.48) = 0$. And $\bar w = g \odot x\,r = [0.848528, 0]$.

All of these are the first test in section 4, `test_hand_example`.

## 4. The interface

```python
# python/tinyllm/autograd/vjp.py
def unbroadcast(g, shape: tuple[int, ...]) -> NDArray
def matmul_vjp(g, A, B) -> tuple[NDArray, NDArray]               # A [..., m, k], B [..., k, n]
def softmax_vjp(g, y, axis: int = -1) -> NDArray                  # y = softmax(x)
def log_softmax_vjp(g, y, axis: int = -1) -> NDArray              # y = log_softmax(x)
def layernorm_vjp(g, xhat, rstd, gamma) -> tuple[NDArray, NDArray, NDArray]   # dx, dgamma, dbeta
def rmsnorm_vjp(g, x, rstd, w) -> tuple[NDArray, NDArray]                     # dx, dw
def cross_entropy_vjp(logits, targets, ignore_index: int = -100) -> NDArray
```

LayerNorm and RMSNorm normalize the last axis; `rstd` may be passed with shape `x.shape[:-1]` or `x.shape[:-1] + (1,)`, and the parameter gradients sum over every leading axis. `matmul_vjp` needs at least 2-D operands and unbroadcasts both gradients. The forward passes are not part of this module: `L0.2` writes them and saves `y`, `xhat`, and `rstd`; the tests here compute their own.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | every section 3 number | you and the test agree on each rule |
| `test_matches_torch_golden` | golden | torch.autograd's gradients for all seven rules, in float64 | the framework `L0.2` is compared against |
| `test_unbroadcast_shapes` | unit | seven broadcast patterns, counted with a gradient of ones | biases and shared weights |
| `test_unbroadcast_rejects_impossible_shapes` | boundary | shapes that could not have broadcast raise `ValueError` | a wrong-shaped gradient is a bug |
| `test_matmul_vjp_gradcheck` | gradcheck | 2-D, square, and batched products with either operand broadcast | every linear layer |
| `test_matmul_vjp_rejects_vectors` | boundary | 1-D operands raise `ValueError` | vectors must be reshaped explicitly |
| `test_softmax_vjps_gradcheck` | gradcheck | softmax and log-softmax along the last axis and axis 0 | attention weights, the loss |
| `test_softmax_vjp_matches_numeric_vjp` | differential | against `M04.2`'s `vjp_numeric` on one row | the same $u^\top J$ two ways |
| `test_layernorm_vjp_gradcheck` | gradcheck | $x$, $\gamma$, $\beta$ on `[2, 3, 8]`, `rstd` in both shapes | the 2017 transformer (`L5`) |
| `test_rmsnorm_vjp_gradcheck` | gradcheck | $x$ and $w$, `rstd` in both shapes | `L7.1` |
| `test_cross_entropy_vjp_gradcheck` | gradcheck | the gradient of `M11.1`'s cross-entropy, with ignored rows | `L0.3`'s fused loss |
| `test_cross_entropy_ignore_index` | boundary | ignored rows get 0, the mean counts valid rows, out-of-range targets raise | padded batches |
| `test_cross_entropy_large_logits` | boundary | logits near $10^4$ give a finite gradient | a confident model |
| `test_trace_identity` | property | $\operatorname{tr}(G^\top dY) = \langle \bar A, dA\rangle + \langle \bar B, dB\rangle$ on 20 directions | the definition of a VJP |
| `test_vjps_do_not_mutate_inputs` | unit | saved values and upstream gradients are unchanged | the graph reuses them |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. $G B$ instead of $G B^\top$ | correct shape for square $B$, wrong values | `test_matmul_vjp_gradcheck` (mutant `s01`) |
| 2. a transposed gradient | $\bar B^\top$ where $\bar B$ belongs | `test_trace_identity` (mutant `s02`) |
| 3. not summing over broadcast axes | a bias or shared weight gets a batch of gradients | `test_matmul_vjp_gradcheck` (mutant `s03`), `test_unbroadcast_shapes` (mutants `s14`, `s15`) |
| 4. the softmax Jacobian's diagonal only | $\bar x = y \odot g$ | `test_softmax_vjps_gradcheck` (mutant `s04`) |
| 5. log-softmax VJP with $\ell$ where $e^\ell$ belongs | log-probabilities used as probabilities | `test_softmax_vjps_gradcheck` (mutant `s05`) |
| 6. treating $\mu$ or $r$ as a constant | missing correction terms in LayerNorm or RMSNorm | `test_layernorm_vjp_gradcheck` (mutants `s06`, `s07`), `test_rmsnorm_vjp_gradcheck` (mutants `s09`, `s10`) |
| 7. $\bar\gamma = \sum g$ | the scale learns like a shift | `test_layernorm_vjp_gradcheck` (mutant `s08`) |
| 8. averaging over every row, or letting padding through | gradients scaled by the padding ratio; padding tokens trained | `test_cross_entropy_ignore_index` (mutants `s11`, `s12`, `s13`) |
| 9. `exp(z) / sum(exp(z))` inside the loss gradient | NaN once a logit passes 709 | `test_cross_entropy_large_logits` (mutant `s16`) |
| 10. updating the upstream gradient in place | the caller's `g` changes under it | `test_vjps_do_not_mutate_inputs` (mutant `s17`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M09.2` | `softmax` inside `cross_entropy_vjp` |
| Back | `M11.1` | its `cross_entropy` is the loss whose gradient `cross_entropy_vjp` is |
| Back | `M04.2` | `vjp_numeric` checks the softmax VJP numerically |
| Back | `M03.1` | row-major matrices and matmul shapes |
| Forward | `L0.1` | your Tensor's backward sums broadcast gradients back to each input's shape with `unbroadcast` |
| Forward | `L0.2` | each op of the tensor library registers one of these VJPs |
| Forward | `L0.3` | the fused softmax cross-entropy, $(\mathrm{softmax} - \mathrm{onehot})/n$ |
| Forward | `L3.1` | backpropagation through time by hand chains `matmul_vjp` over steps |
| Forward | `L7.1` | `rmsnorm_vjp` is RMSNorm's backward |

If you skip this module, `ol check L0.2` stops with `L0.2 needs M08.3`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| the closed-form rules | PyTorch's derivative table | one backward formula per op, from which the autograd code is generated | `tools/autograd/derivatives.yaml` |
| `layernorm_vjp` | llm.c | the same three-term backward in plain C, fused over a batch | `train_gpt2.c` (`layernorm_backward`) |
| `cross_entropy_vjp` | Liger Kernel | the linear layer, softmax, and cross-entropy fused in chunks, so the $[T, V]$ logits never exist in memory | `src/liger_kernel/ops/fused_linear_cross_entropy.py` |
| `rmsnorm_vjp` | PyTorch `F.rms_norm` | fused kernels with the saved `rstd`, the same formula | `aten/src/ATen/native/layer_norm.cpp` |
