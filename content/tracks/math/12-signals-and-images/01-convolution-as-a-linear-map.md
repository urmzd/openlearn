<!-- ol:module M12.1 -->
# Convolution as a linear map

## Overview

| | |
|---|---|
| **Module** | `M12.1` · build · Python · Pass 12 · 3 to 4 h |
| **You build** | `python/tinyllm/sig/conv.py`: `conv_out_size`, `conv1d_direct` and `conv2d_direct` (stride, padding, dilation, groups), `correlate_vs_convolve`, `im2col` and its adjoint `col2im`, and `conv_matrix` (the doubly block Toeplitz form) |
| **Contract** | [`course/contracts/py/tinyllm/sig/conv.pyi`](../../../../course/contracts/py/tinyllm/sig/conv.pyi) |
| **Tests** | `course/tests/M12.1/test_conv.py` (what they check: section 4), golden values from `course/oracle/M12.1/conv_torch.py` (PyTorch `F.conv1d`, `F.conv2d`, `F.unfold`, `F.fold`, autograd) in `course/fixtures/M12.1/conv_torch.npz` |
| **Needs** | no module code. Reading: `M03.2` and `M03.3` (a linear map is a matrix; its transpose), `M08.2` (a backward pass is a transposed Jacobian), and the naive loop in [`ml/06-neural-architectures/code/conv2d.c`](../../ml/06-neural-architectures/code/conv2d.c) |
| **Used by** | `M12.3` (`resample_audio` is one strided `conv1d_direct`) · later `L13.1` Conv2d and Conv1d on autograd (im2col, col2im) · `L14.2` the Whisper stem (`conv_out_size`) |
| **Milestone** | `MS-P12` (the multimodal gate) |
| **Optional depth** | Dumoulin and Visin, "A guide to convolution arithmetic for deep learning" (2016, free); Chellapilla, Puri, Simard, "High performance convolutional neural networks for document processing" (2006), the im2col paper; Goodfellow, Bengio, Courville, *Deep Learning*, ch. 9 (free) |

## Key Takeaways

- A deep-learning "convolution" is a cross-correlation: no kernel flip. Flipping the kernel gives the true convolution of signal processing (`test_hand_example_correlation`, `test_flip_turns_correlation_into_convolution`).
- One formula sizes every conv output: $\lfloor (n + 2p - d(k-1) - 1)/s \rfloor + 1$ (`test_conv_out_size_table`).
- A conv is linear, so it is a matrix: the doubly block Toeplitz $T$ with $\operatorname{vec}(y) = T \operatorname{vec}(x)$. im2col turns it into one matmul with the flattened weights (`test_conv_is_one_matmul`).
- col2im is the adjoint of im2col (it sums overlapping windows), so the backward pass of a conv is built from the same two functions (`test_col2im_is_the_adjoint_of_im2col`, `test_backward_is_a_conv`).

## How to work this chapter

```bash
ol start M12.1              # stubs python/tinyllm/sig/conv.py into your repo
ol tests M12.1              # read the test catalog first: rung R0, you write no tests here
ol check M12.1              # exit code is the verdict
ol diff  M12.1              # after passing: your code against the reference
```

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
Signal processing's convolution flips the kernel first, $y[i,j] = \sum_{r,s} w[r,s]\,x[i - r, j - s]$, which is the same as correlating with $w$ rotated by 180 degrees. Flipping is what makes convolution commutative and makes the convolution theorem (`M12.4`) hold. A learned kernel does not care: whatever it would learn flipped, it learns unflipped. So every framework computes the correlation and calls it a convolution, and `conv2d_direct` does the same. `correlate_vs_convolve` returns both so you can see the difference: on one row the true convolution is exactly `numpy.convolve(..., 'valid')`.

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

The flipped kernel is $\begin{pmatrix} -1 & 0 \\ 0 & 1 \end{pmatrix}$, so the true convolution is $-x[i, j] + x[i+1, j+1]$: exactly the negative of the table. This is `test_hand_example_correlation`. In im2col form, column $(0, 1)$ of `im2col(x, 2, 2)` is the window $(2, 0, 1, 3)$ in $(r, s)$ order, and $W_{\text{flat}} = (1, 0, 0, -1)$ gives $2 - 3 = -1$, the same entry.

## 4. The interface

```python
def conv_out_size(n, k, stride=1, pad=0, dilation=1) -> int: ...
def conv2d_direct(x, w, b, stride=(1, 1), pad=(0, 0), dilation=(1, 1), groups=1) -> NDArray: ...
def im2col(x, kh, kw, stride=(1, 1), pad=(0, 0), dilation=(1, 1)) -> NDArray:   # [N, C*kh*kw, P*Q]
def col2im(cols, x_shape, kh, kw, stride=(1, 1), pad=(0, 0), dilation=(1, 1)) -> NDArray:   # adjoint, sums overlaps
def conv_matrix(in_shape, w, stride=(1, 1), pad=(0, 0)) -> NDArray:   # T with vec(y) = T vec(x)
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_correlation` | unit, smoke | section 3's correlation and its negated convolution | you and the tests agree on "no flip" |
| `test_conv_out_size_table` | unit, smoke | Whisper stem, ViT patch conv, dilation, exact fit, and bad arguments | every L13 and L14 layer sizes itself with it |
| `test_conv2d_matches_torch` | golden | four conv2d cases against `F.conv2d`: plain, strided and dilated with groups, depthwise, patch embedding | `L13.1` must equal PyTorch |
| `test_conv1d_matches_torch` | golden | three conv1d cases including Whisper's stride-2 stem | `L14.2`'s stem |
| `test_im2col_col2im_match_unfold_fold` | golden | im2col equals `F.unfold`, col2im equals `F.fold` | row order and summed overlaps |
| `test_conv_is_one_matmul` | differential | direct loop, grouped im2col matmul, and the Toeplitz matrix agree | the three views of one map |
| `test_backward_is_a_conv` | golden | dW, dx, db from im2col and col2im equal PyTorch's autograd | `L13.1`'s backward |
| `test_col2im_is_the_adjoint_of_im2col` | property | $\langle \operatorname{im2col} x, y\rangle = \langle x, \operatorname{col2im} y\rangle$ with overlapping windows | why col2im is the right backward |
| `test_flip_turns_correlation_into_convolution` | property | flipped-kernel correlation is convolution; one row equals `numpy.convolve` | the definition, independently |
| `test_rejects_shapes_that_do_not_fit` | boundary | groups that do not divide, wrong channel count, oversized kernel, wrong cols; inputs untouched | model-definition bugs fail loudly |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. flipping the kernel (true convolution) | every output differs from PyTorch; pretrained weights give garbage | `test_hand_example_correlation`, `test_conv2d_matches_torch` (mutant `s01`) |
| 2. leaving dilation out of the kernel span | dilated layers get too many outputs | `test_conv_out_size_table`, `test_conv1d_matches_torch` (mutant `s02`) |
| 3. col2im assigning instead of adding | wrong input gradients wherever windows overlap | `test_col2im_is_the_adjoint_of_im2col`, `test_backward_is_a_conv` (mutant `s03`) |
| 4. im2col rows in $(r, s, c)$ order | shapes fine, values wrong for $C > 1$ | `test_im2col_col2im_match_unfold_fold`, `test_conv_is_one_matmul` (mutant `s04`) |
| 5. every group reading the first group's channels | grouped and depthwise convs wrong | `test_conv2d_matches_torch` (mutant `s05`) |
| 6. all the padding on one side | outputs shifted by $p$ | `test_conv2d_matches_torch` (mutant `s06`) |
| 7. Toeplitz columns without the padding offset | $T x$ shifted against the direct conv | `test_conv_is_one_matmul` (mutant `s07`) |
| dropping the $+1$ in the size formula | one output too few everywhere | `test_conv_out_size_table` (mutant `m01`) |
| rejecting an exact fit | a kernel as large as the input raises | `test_conv_out_size_table` (mutant `m02`) |
| not checking that groups divide the output channels | silent zero channels | `test_rejects_shapes_that_do_not_fit` (mutant `m03`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M03.2`, `M03.3` | a linear map is a matrix and its adjoint is the transpose (reading) |
| Back | `M08.2` | a backward pass multiplies by the transposed Jacobian (reading) |
| Forward | `M12.3` | `resample_audio` applies its sinc filter bank as one strided `conv1d_direct` |
| Forward | `L13.1` | `Conv2d`, `Conv1d`, and pooling on autograd: im2col forward, col2im backward (with B14's L13 group) |
| Forward | `L13.3` | `PatchEmbed` is a conv with kernel = stride |
| Forward | `L14.2` | the Whisper stem sizes its output with `conv_out_size` |
| Forward | `S-M12` | output sizes, receptive fields, the Toeplitz form, and the adjoint by hand |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `im2col` + matmul | PyTorch `aten/src/ATen/native/ConvolutionMM2d.cpp` (`slow_conv2d`) | the same unfold-then-GEMM path, threaded, for CPU fallbacks | `slow_conv2d_forward_out_cpu` |
| `conv2d_direct` | oneDNN / cuDNN | Winograd and FFT algorithms, implicit GEMM without materializing columns, autotuning per shape | cuDNN `cudnnFindConvolutionForwardAlgorithm` |
| `col2im` | `F.fold` / `conv_transpose2d` | the adjoint as a first-class layer (upsampling in decoders) | `aten/src/ATen/native/Col2Im.cpp` |
| the C loop in `conv2d.c` | optional `L13.8` | im2col in C with a blocked matmul, benchmarked against the naive loop | `c/src/kernels/conv.c` |
