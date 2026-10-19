# Convolution as a linear map

## Overview

| | |
|---|---|
| **Module** | `M12.1` · build · Python · Pass 12 · 3 to 4 h |
| **Solution** | [`python/tinyllm/sig/conv.py`](../../../../course/solutions/python/tinyllm/sig/conv.py): `conv_out_size`, `conv1d_direct` and `conv2d_direct` (stride, padding, dilation, groups), `correlate_vs_convolve`, `im2col` and its adjoint `col2im`, and `conv_matrix` (the doubly block Toeplitz form) |
| **Contract** | [`course/contracts/py/tinyllm/sig/conv.pyi`](../../../../course/contracts/py/tinyllm/sig/conv.pyi) |
| **Needs** | no module code. Reading: `M03.2` and `M03.3` (a linear map is a matrix; its transpose), `M08.2` (a backward pass is a transposed Jacobian), and the naive loop in [`ml/06-neural-architectures/code/conv2d.c`](../../machine-learning/06-neural-architectures/code/conv2d.c) |
| **Used by** | `M12.3` (`resample_audio` is one strided `conv1d_direct`) · later `L13.1` Conv2d and Conv1d on autograd (im2col, col2im) · `L14.2` the Whisper stem (`conv_out_size`). Part of Pass 12, which teaches the system to see and hear |
| **Optional depth** | Dumoulin and Visin, "A guide to convolution arithmetic for deep learning" (2016, free); Chellapilla, Puri, Simard, "High performance convolutional neural networks for document processing" (2006), the im2col paper; Goodfellow, Bengio, Courville, *Deep Learning*, ch. 9 (free) |

## Key Takeaways

- A deep-learning "convolution" is a cross-correlation: no kernel flip. Flipping the kernel gives the true convolution of signal processing.
- One formula sizes every conv output: $\lfloor (n + 2p - d(k-1) - 1)/s \rfloor + 1$.
- A conv is linear, so it is a matrix: the doubly block Toeplitz $T$ with $\operatorname{vec}(y) = T \operatorname{vec}(x)$. im2col turns it into one matmul with the flattened weights.
- col2im is the adjoint of im2col (it sums overlapping windows), so the backward pass of a conv is built from the same two functions.

## How to read this chapter

Sections 1 to 3 build convolution up from a sliding dot product to a matrix, im2col and col2im, and the backward pass, and work one example by hand. Section 4 fixes the interface and lists what a correct implementation must handle. Section 5 walks through the solution, [`conv.py`](../../../../course/solutions/python/tinyllm/sig/conv.py), and explains why every index is what it is. Try to predict each step before reading it.

---

## 1. Why now

Pass 12 teaches the system to see and hear. Every vision tower in `L13` starts with a convolution (a ViT's patch embedding is a conv whose kernel equals its stride), and Whisper's encoder in `L14` starts with two 1-D convolutions over the log-mel frames. Without this module, `L13.1` has no way to build `Conv2d` on the autograd engine: a Python loop over output pixels is far too slow, and a vectorized version needs im2col for the forward pass and col2im for the backward pass. Get the output size wrong and the Whisper stem hands 1501 positions to an encoder whose position table has 1500; get the row order of im2col wrong and the weights multiply the wrong pixels while every shape still lines up.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | input batch, NCHW (2-D) or NCL (1-D) | `float64[N, C, H, W]` |
| $w$ | weights, KCRS (2-D) or KCR (1-D) | `float64[K, C/g, R, S]` |
| $b$ | bias, one per output channel | `float64[K]` |
| $s$, $p$, $d$ | stride, zero padding, dilation (per axis) | ints |
| $g$ | groups: channels split into $g$ independent convs | int |
| $P$, $Q$ | output height and width | ints |
| $\operatorname{vec}(\cdot)$ | row-major flattening | |
| $T$ | doubly block Toeplitz matrix of a single-channel conv | `float64[P*Q, H*W]` |

### 2.1 Correlation and convolution

The cross-correlation of an image $x$ with a kernel $w$ slides $w$ over $x$ and takes a dot product at each position:
$$y[i, j] = \sum_{r, s} w[r, s]\, x[i + r, j + s].$$
Signal processing's convolution flips the kernel first, $y[i,j] = \sum_{r,s} w[r,s]\,x[i - r, j - s]$, which is the same as correlating with $w$ rotated by 180 degrees. Flipping is what makes convolution commutative and makes the convolution theorem (`M12.4`) hold. A learned kernel does not care: whatever it would learn flipped, it learns unflipped. So every framework computes the correlation and calls it a convolution, and `conv2d_direct` does the same. `correlate_vs_convolve` returns both to make the difference visible: on one row the true convolution is exactly `numpy.convolve(..., 'valid')`.

### 2.2 Output size, stride, padding, dilation, groups

A kernel of size $k$ with dilation $d$ places its taps $d$ samples apart, so it spans $d(k-1) + 1$ input samples. Padding adds $p$ zeros at each end, and a stride $s$ moves the window $s$ samples per output. The first window starts at 0; the last must end inside the padded input. The number of outputs is
$$\text{out} = \left\lfloor \frac{n + 2p - d(k - 1) - 1}{s} \right\rfloor + 1 .$$
Output position $o$, tap $r$ reads padded index $o s + r d$, which is input index $o s + r d - p$. Groups split the $C$ input channels into $g$ blocks of $C/g$ and the $K$ output channels into $g$ blocks of $K/g$; output channel $k$ only sees input block $\lfloor k / (K/g) \rfloor$. That is why the weight's second axis is $C/g$, not $C$. Depthwise convolution is $g = C$. Whisper's stem uses $k = 3$, $p = 1$, which keeps 3000 frames at $s = 1$ and halves them to 1500 at $s = 2$.

### 2.3 A conv is a matrix

Each output is a fixed linear combination of inputs, so for a single-channel image there is a matrix $T$ with $\operatorname{vec}(y) = T \operatorname{vec}(x)$. Row $(p, q)$ of $T$ holds the kernel's taps in the columns of the pixels that window touches, and zeros elsewhere: each block of rows repeats the same pattern shifted by one column (Toeplitz), and the blocks repeat shifted by one image row (block Toeplitz), hence "doubly block Toeplitz". Taps that land in the zero padding have no column. `conv_matrix` builds $T$ explicitly; it is $PQ \times HW$, mostly zeros, and only for understanding. The forward pass is $T x$; the backward pass with respect to $x$ is $T^\top g$ (`M08.2`: a VJP is a transposed Jacobian).

### 2.4 im2col and col2im

The practical form keeps the matrix structure but moves it to the data. im2col copies every receptive field of $x$ into a column: row index $(c, r, s)$, column index $(p, q)$, shape $[N, C R S, P Q]$. Flattening the weights in the same $(c, r, s)$ order gives $W_{\text{flat}} \in \mathbb{R}^{K \times CRS}$, and
$$Y = W_{\text{flat}} \cdot \operatorname{im2col}(x), \qquad Y \in \mathbb{R}^{N \times K \times PQ},$$
one large matmul that BLAS runs at full speed. The row order matters: any other order still produces the right shapes, but pairs each weight with the wrong pixel. im2col is a linear map that copies; its adjoint, col2im, sends every column entry back to the pixel it came from and **adds** where windows overlap. Adjoint means $\langle \operatorname{im2col}(x), y \rangle = \langle x, \operatorname{col2im}(y) \rangle$ for all $x, y$. With stride smaller than the kernel, a pixel sits in several windows; assigning instead of adding loses all but the last contribution.

### 2.5 The gradient of a conv is a conv

With $Y = W_{\text{flat}} X_c$ where $X_c = \operatorname{im2col}(x)$ and upstream gradient $G$ (shaped like $Y$), the matrix calculus of `M08` gives
$$\frac{\partial L}{\partial W_{\text{flat}}} = \sum_n G_n X_{c,n}^\top, \qquad \frac{\partial L}{\partial x} = \operatorname{col2im}\Big(W_{\text{flat}}^\top G\Big), \qquad \frac{\partial L}{\partial b_k} = \sum_{n, p, q} G[n, k, p, q].$$
The weight gradient is a correlation of the input with the upstream gradient, and the input gradient is a "transposed convolution" of the upstream gradient with the weights: both are convolutions again. `L13.1` implements Conv2d's backward exactly this way.

## 3. Worked example by hand

Image and kernel:
$$x = \begin{pmatrix} 1 & 2 & 0 & 1 \\ 0 & 1 & 3 & 1 \\ 2 & 1 & 0 & 0 \\ 1 & 0 & 1 & 2 \end{pmatrix}, \qquad w = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}.$$
The output is $4 - 2 + 1 = 3$ on each axis (`conv_out_size(4, 2) = 3`). The correlation is $y[i, j] = x[i, j] - x[i+1, j+1]$:

| | $j = 0$ | $j = 1$ | $j = 2$ |
|---|---|---|---|
| $i = 0$ | $1 - 1 = 0$ | $2 - 3 = -1$ | $0 - 1 = -1$ |
| $i = 1$ | $0 - 1 = -1$ | $1 - 0 = 1$ | $3 - 0 = 3$ |
| $i = 2$ | $2 - 0 = 2$ | $1 - 1 = 0$ | $0 - 2 = -2$ |

The flipped kernel is $\begin{pmatrix} -1 & 0 \\ 0 & 1 \end{pmatrix}$, so the true convolution is $-x[i, j] + x[i+1, j+1]$: exactly the negative of the table. In im2col form, column $(0, 1)$ of `im2col(x, 2, 2)` is the window $(2, 0, 1, 3)$ in $(r, s)$ order, and $W_{\text{flat}} = (1, 0, 0, -1)$ gives $2 - 3 = -1$, the same entry.

## 4. The interface

```python
def conv_out_size(n, k, stride=1, pad=0, dilation=1) -> int: ...
def conv2d_direct(x, w, b, stride=(1, 1), pad=(0, 0), dilation=(1, 1), groups=1) -> NDArray: ...
def im2col(x, kh, kw, stride=(1, 1), pad=(0, 0), dilation=(1, 1)) -> NDArray:   # [N, C*kh*kw, P*Q]
def col2im(cols, x_shape, kh, kw, stride=(1, 1), pad=(0, 0), dilation=(1, 1)) -> NDArray:   # adjoint, sums overlaps
def conv_matrix(in_shape, w, stride=(1, 1), pad=(0, 0)) -> NDArray:   # T with vec(y) = T vec(x)
```

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand-example correlation | unit | section 3's correlation and its negated convolution | everyone agrees on "no flip" |
| the output-size table | unit | Whisper stem, ViT patch conv, dilation, exact fit, and bad arguments | every L13 and L14 layer sizes itself with it |
| conv2d matches PyTorch | golden | four conv2d cases against `F.conv2d`: plain, strided and dilated with groups, depthwise, patch embedding | `L13.1` must equal PyTorch |
| conv1d matches PyTorch | golden | three conv1d cases including Whisper's stride-2 stem | `L14.2`'s stem |
| im2col and col2im match unfold and fold | golden | im2col equals `F.unfold`, col2im equals `F.fold` | row order and summed overlaps |
| a conv is one matmul | differential | direct loop, grouped im2col matmul, and the Toeplitz matrix agree | the three views of one map |
| the backward pass is a conv | golden | dW, dx, db from im2col and col2im equal PyTorch's autograd | `L13.1`'s backward |
| col2im is the adjoint of im2col | property | $\langle \operatorname{im2col} x, y\rangle = \langle x, \operatorname{col2im} y\rangle$ with overlapping windows | why col2im is the right backward |
| flipping turns correlation into convolution | property | flipped-kernel correlation is convolution; one row equals `numpy.convolve` | the definition, independently |
| shapes that do not fit are rejected | boundary | groups that do not divide, wrong channel count, oversized kernel, wrong cols; inputs untouched | model-definition bugs fail loudly |

## 5. The solution, step by step

### 5.1 From first principles

1. **Size first, because everything else indexes into it.** Every function needs $P$ and $Q$ before it can allocate, so one function, `conv_out_size`, owns the formula of section 2.2, and every other function calls it. An oversized kernel is an error, not an empty output, while an exact fit (span equal to the padded input) is one output.
2. **Pad once, then forget about padding.** Zero-padding the input by $p$ on each side turns "input index $os + rd - p$" into "padded index $os + rd$", so no inner loop needs a bounds check. Symmetric padding is what PyTorch does.
3. **Loop over taps, not over outputs.** For a fixed tap $(i, j)$, the pixels read by all outputs form a strided slice of the padded input: rows $i d_h, i d_h + s_h, \dots$ (exactly $P$ of them) and likewise for columns. So the conv is $R \cdot S$ slice-and-accumulate steps, each a small matrix product over channels, instead of $P \cdot Q$ Python iterations. This is the same loop that im2col uses to fill its columns, which is why the two agree.
4. **Groups are independent convs on channel blocks.** Group $g$ reads input channels $[g C/g, (g+1) C/g)$ and writes output channels $[g K/g, (g+1) K/g)$. Slicing both and running the ungrouped computation per group is the definition; both $C$ and $K$ must be divisible by $g$ and the weight's second axis must be $C/g$.
5. **No flip.** The direct conv reads $x[os + rd]$ with weight $w[r]$, the same index direction for both, which is correlation. True convolution is that same function applied to `w[::-1, ::-1]`.
6. **im2col is the tap loop with storage.** Instead of multiplying each tap's slice by the weights immediately, store it at row $(c, i, j)$: the result is a `[N, C, R, S, P, Q]` array, reshaped to `[N, CRS, PQ]`. Row-major reshaping makes the row index $c \cdot RS + i \cdot S + j$, the same order in which a `[K, C, R, S]` weight flattens to `[K, CRS]`.
7. **col2im is the transpose of a copy, so it adds.** im2col copies pixel $x_k$ into every column entry whose window contains it. The transpose of "copy $x_k$ to positions $A_k$" is "sum the entries at $A_k$ back into $x_k$". So col2im is the same tap loop with `+=` in place of `=`, followed by cropping the padding away (gradients that land in padding belong to no pixel).
8. **The Toeplitz matrix is the same index map written into a matrix.** Row $(o_p, o_q)$, tap $(i, j)$ reads input pixel $(o_p s_h + i - p_h,\ o_q s_w + j - p_w)$; if it is inside the image, put $w[i, j]$ at that pixel's column. Taps in the padding have no column, because the padding is zeros.

### 5.2 The code, annotated

In the quotes below, signatures are shortened (type hints dropped) and argument validation is elided as `...`; the bodies are as in the solution.

The size formula, with the two validity conditions:

```python
def conv_out_size(n, k, stride=1, pad=0, dilation=1) -> int:
    if n < 1 or k < 1 or stride < 1 or dilation < 1 or pad < 0:
        raise ValueError(...)
    # A dilated kernel spans d (k - 1) + 1 input samples; the first window
    # starts at 0 and the last one must end inside the padded input.
    span = dilation * (k - 1) + 1
    if n + 2 * pad < span:          # "<", not "<=": an exact fit gives one output
        raise ValueError(...)
    return (n + 2 * pad - span) // stride + 1   # the "+1" counts the first window
```

Written with `span`, the formula is $\lfloor (n + 2p - \text{span})/s \rfloor + 1$, the same as section 2.2's. Check against Whisper's stem: $n = 3000$, $k = 3$, $p = 1$, $s = 2$ gives $\lfloor 2999/2 \rfloor + 1 = 1500$.

The direct 2-D conv: pad once, then for each group and each tap, one strided slice and one `einsum`.

```python
def conv2d_direct(x, w, b, stride=(1, 1), pad=(0, 0), dilation=(1, 1), groups=1):
    # ... validate ndim, normalize stride/pad/dilation to pairs ...
    n, c, h, wd = x.shape
    kout, cg, r, s = w.shape
    _check_groups(c, kout, cg, groups, "conv2d_direct")   # g | C, g | K, cg == C/g
    p = conv_out_size(h, r, sh, ph, dh)
    q = conv_out_size(wd, s, sw, pw, dw)
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))  # same p on both sides
    kg = kout // groups
    y = np.zeros((n, kout, p, q))
    for g in range(groups):
        xg = xp[:, g * cg : (g + 1) * cg]      # this group's input channels
        wg = w[g * kg : (g + 1) * kg]          # this group's output channels
        for i in range(r):
            for j in range(s):
                # Every output (op, oq) reads padded pixel (op*sh + i*dh, oq*sw + j*dw)
                # for tap (i, j): one strided slice with exactly P rows and Q columns.
                seg = xg[
                    :,
                    :,
                    i * dh : i * dh + sh * (p - 1) + 1 : sh,
                    j * dw : j * dw + sw * (q - 1) + 1 : sw,
                ]  # [N, C/g, P, Q]
                # Contract over input channels: y[n,k,p,q] += w[k,c,i,j] * seg[n,c,p,q].
                y[:, g * kg : (g + 1) * kg] += np.einsum(
                    "kc,ncpq->nkpq", wg[:, :, i, j], seg
                )
    if b is not None:
        y += np.asarray(b, dtype=np.float64)[None, :, None, None]   # one bias per output channel
    return y
```

The slice end `start + s * (P - 1) + 1` is the index just past the last pixel the last output reads, so the slice has exactly $P$ elements whatever the remainder in the size formula. `conv1d_direct` is the same code with one spatial axis.

im2col stores those same slices instead of contracting them:

```python
def im2col(x, kh, kw, stride=(1, 1), pad=(0, 0), dilation=(1, 1)):
    # ... validate, normalize pairs ...
    p = conv_out_size(h, kh, sh, ph, dh)
    q = conv_out_size(w, kw, sw, pw, dw)
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    cols = np.empty((n, c, kh, kw, p, q))
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = xp[          # the same strided slice as conv2d_direct
                :,
                :,
                i * dh : i * dh + sh * (p - 1) + 1 : sh,
                j * dw : j * dw + sw * (q - 1) + 1 : sw,
            ]
    # Rows ordered (c, r, s), columns (p, q): the layout of F.unfold, and the
    # order in which a KCRS weight flattens to [K, C*R*S].
    return cols.reshape(n, c * kh * kw, p * q)
```

With it, an ungrouped conv is `w.reshape(K, -1) @ im2col(x, R, S)` (a `[K, CRS]` matrix times `[N, CRS, PQ]`, broadcast over $N$), reshaped to `[N, K, P, Q]`; the agreement of that product with `conv2d_direct` and with the Toeplitz matrix is one of the cases in section 4.

col2im is the same loop with the arrow reversed and `+=`:

```python
def col2im(cols, x_shape, kh, kw, stride=(1, 1), pad=(0, 0), dilation=(1, 1)):
    # ... normalize pairs, compute p and q from x_shape ...
    if cols.shape != (n, c * kh * kw, p * q):
        raise ValueError(...)
    blocks = cols.reshape(n, c, kh, kw, p, q)          # undo im2col's reshape
    xp = np.zeros((n, c, h + 2 * ph, w + 2 * pw))      # accumulate into the padded frame
    for i in range(kh):
        for j in range(kw):
            # Add, never assign: overlapping windows each send their share
            # back to the same pixel (the adjoint of a copy is a sum).
            xp[
                :,
                :,
                i * dh : i * dh + sh * (p - 1) + 1 : sh,
                j * dw : j * dw + sw * (q - 1) + 1 : sw,
            ] += blocks[:, :, i, j]
    # Contributions that landed in the padding belong to no input pixel.
    return xp[:, :, ph : ph + h, pw : pw + w].copy()
```

Within one `+=`, the strided slice never hits the same pixel twice (consecutive outputs are $s \ge 1$ apart), so NumPy's buffered `+=` is safe; overlaps come from different taps $(i, j)$, which are separate statements.

The Toeplitz matrix, one entry per (output, tap) pair inside the image:

```python
def conv_matrix(in_shape, w, stride=(1, 1), pad=(0, 0)):
    # ... validate, normalize pairs ...
    t = np.zeros((p * q, h * wd))
    for op in range(p):
        for oq in range(q):
            row = op * q + oq                          # row-major vec(y)
            for i in range(r):
                for j in range(s):
                    a, bcol = op * sh + i - ph, oq * sw + j - pw   # unpadded input index
                    # Taps that land in the zero padding have no column.
                    if 0 <= a < h and 0 <= bcol < wd:
                        t[row, a * wd + bcol] = w[i, j]            # row-major vec(x)
    return t
```

Here the padding offset $-p$ is explicit because there is no padded array: the matrix acts on the original image.

Full solution: [`conv.py`](../../../../course/solutions/python/tinyllm/sig/conv.py).

### 5.3 Why this solution

- **Cost.** The direct conv does $N K (C/g) R S P Q$ multiply-adds, the minimum for a direct method, but in $g R S$ vectorized steps rather than a Python loop per output. im2col uses $R S$ times the input's memory (each pixel is copied once per window it appears in); that is the price of turning the conv into one BLAS matmul, and the reason production libraries use implicit GEMM, which never materializes the columns.
- **Why loop over taps.** A loop over output pixels is $PQ$ Python iterations (2.25 million for a 1500x1500 output); a loop over taps is $RS$ (9 for a 3x3 kernel). Both compute the same sums; only the second is fast in NumPy.
- **Why `conv_matrix` exists but is never used for compute.** $T$ is $PQ \times HW$, quadratic in image size and almost all zeros. It exists to make "a conv is a linear map" concrete and to check the other two forms against it, and it gives the backward pass for free in theory: $T^\top$.
- **Why col2im rather than a separate transposed-conv routine.** The adjoint of im2col is exactly what the input gradient needs ($\partial L/\partial x = \operatorname{col2im}(W_{\text{flat}}^\top G)$), so one pair of functions covers forward and backward, and their adjointness is a property one inner product can check.
- **Tradeoff accepted.** Everything is float64 and single-threaded; this module is about getting the index map right once, so `L13.1` can build on it.

### 5.4 Explain it back

1. Why is the weight's second axis $C/g$ and not $C$?
   <details><summary>Answer</summary>
   Each output channel only sees the input channels of its own group, and a group has $C/g$ of them. Storing $C$ would mean weights for channels the output never reads. For depthwise convolution ($g = C$) the axis is 1.
   </details>
2. Why does col2im add where im2col copies?
   <details><summary>Answer</summary>
   im2col is a linear map that copies each pixel into every window containing it. Its transpose sends each column entry back to its pixel, and when a pixel was copied to several places, the transpose sums all of their contributions. That is what $\langle \operatorname{im2col}(x), y\rangle = \langle x, \operatorname{col2im}(y)\rangle$ requires, and it is the input gradient of a conv. Assigning keeps only the last window's contribution.
   </details>
3. What goes wrong, visibly and invisibly, if im2col's rows are ordered $(r, s, c)$?
   <details><summary>Answer</summary>
   Visibly, nothing: the shape is still `[N, CRS, PQ]` and the matmul with `w.reshape(K, -1)` runs. Invisibly, the weights flatten in $(c, r, s)$ order, so for $C > 1$ weight $(c, r, s)$ multiplies the pixel of some other channel and tap. Only a value check (against `F.unfold` or the direct conv) catches it.
   </details>
4. Why does `conv_matrix` subtract the padding while `conv2d_direct` does not?
   <details><summary>Answer</summary>
   `conv2d_direct` works on an explicitly padded copy of the input, so padded index $os + rd$ is directly usable. `conv_matrix` acts on the unpadded image (its columns are the original $HW$ pixels), so it converts to the input index $os + rd - p$ and drops taps that land outside, where the padding's zeros contribute nothing.
   </details>
5. With $n = 3000$, $k = 3$, $s = 2$ and $p = 0$ instead of 1, how many frames does Whisper's stem produce, and why does it matter?
   <details><summary>Answer</summary>
   $\lfloor (3000 - 3)/2 \rfloor + 1 = 1499$, one short of the 1500 positions the encoder's positional table expects; the stem's $p = 1$ gives exactly 1500. An off-by-one here makes the shapes disagree downstream (or silently drops a frame if the code slices to fit).
   </details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. flipping the kernel (true convolution) | every output differs from PyTorch; pretrained weights give garbage | reads `x[o*s + r*d]` with `w[r]`, same direction; only `correlate_vs_convolve` flips; see step 5 |
| 2. leaving dilation out of the kernel span | dilated layers get too many outputs | `span = dilation * (k - 1) + 1` in `conv_out_size` |
| 3. col2im assigning instead of adding | wrong input gradients wherever windows overlap | `xp[...] += blocks[:, :, i, j]`; see step 7 |
| 4. im2col rows in $(r, s, c)$ order | shapes fine, values wrong for $C > 1$ | builds `[N, C, R, S, P, Q]` and reshapes, giving $(c, r, s)$ rows; see step 6 |
| 5. every group reading the first group's channels | grouped and depthwise convs wrong | slices `xp[:, g*cg:(g+1)*cg]` and `w[g*kg:(g+1)*kg]` per group |
| 6. all the padding on one side | outputs shifted by $p$ | `np.pad(..., (ph, ph), (pw, pw))`, symmetric |
| 7. Toeplitz columns without the padding offset | $T x$ shifted against the direct conv | `conv_matrix` uses `op*sh + i - ph`; see step 8 |
| 8. dropping the $+1$ in the size formula | one output too few everywhere | `(n + 2*pad - span) // stride + 1` |
| 9. rejecting an exact fit | a kernel as large as the input raises | raises only when `n + 2*pad < span` |
| 10. not checking that groups divide the output channels | silent zero channels | `_check_groups` tests `c % groups`, `kout % groups`, and `cg == c // groups` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M03.2`, `M03.3` | a linear map is a matrix and its adjoint is the transpose (reading) |
| Back | `M08.2` | a backward pass multiplies by the transposed Jacobian (reading) |
| Forward | `M12.3` | `resample_audio` applies its sinc filter bank as one strided `conv1d_direct` |
| Forward | `L13.1` | `Conv2d`, `Conv1d`, and pooling on autograd: im2col forward, col2im backward |
| Forward | `L13.3` | `PatchEmbed` is a conv with kernel = stride |
| Forward | `L14.2` | the Whisper stem sizes its output with `conv_out_size` |
| Forward | `S-M12` | output sizes, receptive fields, the Toeplitz form, and the adjoint by hand |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `im2col` + matmul | PyTorch `aten/src/ATen/native/ConvolutionMM2d.cpp` (`slow_conv2d`) | the same unfold-then-GEMM path, threaded, for CPU fallbacks | `slow_conv2d_forward_out_cpu` |
| `conv2d_direct` | oneDNN / cuDNN | Winograd and FFT algorithms, implicit GEMM without materializing columns, autotuning per shape | cuDNN `cudnnFindConvolutionForwardAlgorithm` |
| `col2im` | `F.fold` / `conv_transpose2d` | the adjoint as a first-class layer (upsampling in decoders) | `aten/src/ATen/native/Col2Im.cpp` |
| the C loop in `conv2d.c` | optional `L13.8` | im2col in C with a blocked matmul, benchmarked against the naive loop | `c/src/kernels/conv.c` |
