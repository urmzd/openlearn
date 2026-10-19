# Conv layers and pooling on autograd

[Convolution as a linear map](../../mathematics-for-computing/13-signals-and-images/01-convolution-as-a-linear-map.md) in *Mathematics for Computing* proved that a convolution is a matrix and built `im2col` and its adjoint `col2im`, but only as numpy functions. Nothing in the system can train a kernel yet. The autograd engine from the foundations chapter has `Linear` and no layer that shares weights across positions, so a model that reads an image must flatten it and learn every pixel position separately. This section turns those two functions into layers with hand-written backward passes on the `Tensor` from [Tensor, broadcasting backward, and no_grad](../01-foundations/02-tensor-and-broadcasting-backward.md).

Everything visual or audible in the rest of the book starts here. A vision transformer's patch embedding in [Vision transformer encoder](03-vision-transformer.md) is a `Conv2d` whose kernel equals its stride, and Whisper's stem in [Whisper encoder](../15-speech/02-whisper-encoder.md) is two `Conv1d` layers. A wrong backward pass here passes every shape check and shows up much later as a vision tower that does not learn, so the gradients are proven three ways: against PyTorch, against central differences, and by training LeNet-5 to read digits, which is the first end-to-end evidence that the vision layers train.

Besides the conv chapter of the mathematics book, you will use the reshapes and `relu` from [The op library and gradcheck_all](../01-foundations/03-op-library-and-gradcheck.md), `Module` and `Linear` from [Module system and basic layers](../01-foundations/05-module-system-and-layers.md), `normal_init` from [Variance propagation and initialization](../../mathematics-for-computing/08-probability-statistics/04-variance-propagation-and-initialization.md), and `PCG32` from [Modular arithmetic, hashing, and PCG32](../../mathematics-for-computing/07-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md). Training LeNet also uses the cross-entropy of [Losses with a fused backward](../01-foundations/04-losses-with-fused-backward.md) and [Adam and AdamW](../../mathematics-for-computing/11-optimization/03-adam-and-adamw.md); for comparison, a plain-loop LeNet lives in [`ml/06-neural-architectures/code/lenet.rs`](../../machine-learning/06-neural-architectures/code/lenet.rs). The full solution is [`conv.py`](../../../../course/solutions/python/tinyllm/vision/conv.py) and its contract is [`course/contracts/py/tinyllm/vision/conv.pyi`](../../../../course/contracts/py/tinyllm/vision/conv.pyi).

> **Learning objectives**
> - Derive the backward pass of a convolution as $dW = \sum_n G_n C_n^\top$, $dx_n = \text{col2im}(W^\top G_n)$, $db = \sum G$, and extend it to grouped convolution.
> - Implement `Conv1d` as `Conv2d` on an image one row tall, with no new backward code.
> - Implement max and average pooling with deterministic gradients, including the tie rule and overlapping windows.
> - Initialize a conv layer with the right fan-in and assemble LeNet-5 so a PyTorch checkpoint loads.

## Notation

The derivations below use a small set of symbols, summarized here with their shapes.

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | input batch (NCHW) | `float[N, C, H, W]` |
| $W$ | kernel bank (KCRS), $\gamma$ groups | `float[K, C/γ, R, S]` |
| $b$ | bias | `float[K]` |
| $P, Q$ | output height and width (`conv_out_size`) | int |
| $C_n$ | `im2col` of image $n$, rows ordered $(c, r, s)$ | `float64[C·R·S, P·Q]` |
| $G$ | upstream gradient $\partial L/\partial y$ | `float[N, K, P, Q]` |

## Conv2d: one matmul forward, its transpose backward

Take one group first ($\gamma = 1$). Flatten the kernels to $\bar W \in \mathbb{R}^{K \times CRS}$. Column $l$ of $C_n$ is the receptive field of output position $l$, so the layer is
$$Y_n = \bar W C_n + b\,\mathbf{1}^\top \in \mathbb{R}^{K \times PQ},$$
reshaped to $[K, P, Q]$. As in `torch.nn.Conv2d`, this is cross-correlation: no kernel flip.

For the backward pass, let $G_n$ be $\partial L/\partial Y_n$ and perturb each input. The loss changes by $dL = \sum_n \langle G_n, dY_n \rangle$ with $dY_n = d\bar W\,C_n + \bar W\,dC_n + db\,\mathbf{1}^\top$. Using $\langle A, BC \rangle = \langle A C^\top, B \rangle = \langle B^\top A, C \rangle$:
$$dL = \Big\langle \sum_n G_n C_n^\top,\ d\bar W \Big\rangle + \sum_n \langle \bar W^\top G_n,\ dC_n \rangle + \Big\langle \sum_n G_n \mathbf{1},\ db \Big\rangle.$$
The first and last terms read off $\partial L/\partial \bar W = \sum_n G_n C_n^\top$ and $\partial L/\partial b = \sum_{n,l} G_{n,\cdot,l}$. For the middle term, `im2col` is linear, so $dC_n = \text{im2col}(dx_n)$, and the mathematics book proved $\langle A, \text{im2col}(v) \rangle = \langle \text{col2im}(A), v \rangle$. Hence
$$\frac{\partial L}{\partial x_n} = \text{col2im}(\bar W^\top G_n).$$
You can see why this is right without the algebra: col2im sums where windows overlap, which is exactly the chain rule for a pixel that feeds several outputs. Stride, padding, and dilation change only which pixels `im2col` copies; the three formulas do not change.

Groups are a small extension. With $\gamma$ groups, output channels $gK/\gamma, \dots, (g+1)K/\gamma - 1$ read only input channels $gC/\gamma, \dots, (g+1)C/\gamma - 1$. Because im2col's rows are ordered $(c, r, s)$, group $g$ owns the contiguous rows $g\,\tfrac{C}{\gamma}RS, \dots, (g+1)\tfrac{C}{\gamma}RS - 1$, and the layer is $\gamma$ independent copies of the equations above on those row and channel slices: $\bar W$ is block diagonal. $\gamma$ must divide both $C$ and $K$.

The remaining question for the layer is how to start its weights. One output is a sum of $f = (C/\gamma) R S$ products $w_j x_j$. With independent zero-mean weights of variance $\sigma^2$, $\operatorname{Var}(y) = f \sigma^2\,\mathbb{E}[x^2]$, so $\sigma = 1/\sqrt{f}$ keeps the scale of the signal. The fan-in counts one group's channels, not all $C$. Biases start at zero, and weight then bias are drawn from one `PCG32` stream. The forward and backward run in float64 and the output is cast back to $x$'s dtype.

## Conv1d: an image one row tall

With `Conv2d` in hand, the 1-D layer costs almost nothing. A sequence `[N, C, L]` is an image `[N, C, 1, L]`, and a kernel `[K, C/γ, k]` is `[K, C/γ, 1, k]`. A 1-D conv with stride $s$, padding $p$, dilation $d$ is then the 2-D op with stride $(1, s)$, padding $(0, p)$, dilation $(1, d)$, and the output `[N, K, 1, Lout]` drops its row axis. The reshapes are existing autograd ops, so gradients reach $x$, $W$, and $b$ with no new backward code. The row axis must get padding 0: padding it by $p$ would give $1 + 2p$ output rows, all but one of them reading only zeros.

## Pooling

Pooling reuses the same machinery with no parameters at all. It treats every channel independently. A $k \times k$ window slides with stride $s$ (default $s = k$) and no padding, so $P = \lfloor (H - k)/s \rfloor + 1$ by `conv_out_size`. Folding channels into the batch (`[N·C, 1, H, W]`) makes `im2col` return each window as a column of $k^2$ taps in row-major order.

Average pooling computes $y = \tfrac{1}{k^2} \sum_j c_j$ over the column $c$. Every tap has derivative $1/k^2$, so the backward writes $g/k^2$ into all $k^2$ taps and col2im adds the overlaps.

Max pooling computes $y = c_{j^*}$ with $j^* = \arg\max_j c_j$. Near a point with a unique maximum, $y$ is a copy of one input, so the Jacobian is a selection: the backward writes $g$ at tap $j^*$ and zero elsewhere, then col2im sums, so a pixel that is the maximum of two overlapping windows receives both gradients. At a tie the function is not differentiable and any split of $g$ among the tied taps is a valid subgradient. The contract fixes one, as PyTorch does: the first tied tap in row-major window order, which is the lowest flat index in $x$. That makes gradients deterministic.

## LeNet-5

The layers are only worth something if they learn together, and LeNet-5 is the classic test. The composition is same-padded so any `in_hw` that is a multiple of 4 works: `conv1 = Conv2d(1, 6, 5, pad=2)`, ReLU, `max_pool2d` 2; `conv2 = Conv2d(6, 16, 5, pad=2)`, ReLU, `max_pool2d` 2; flatten in $(c, h, w)$ order (as `torch.flatten`); `fc1 = Linear(16 (in_hw/4)^2, 120)`, ReLU; `fc2 = Linear(120, 84)`, ReLU; `fc3 = Linear(84, n_classes)`. Parameters register in that order and all draw from one rng. For 8 x 8 digits the shapes are $[1, 8, 8] \to [6, 8, 8] \to [6, 4, 4] \to [16, 4, 4] \to [16, 2, 2] \to 64 \to 120 \to 84 \to 10$, with $156 + 2416 + 7800 + 10164 + 850 = 21386$ parameters. Trained with cross-entropy and Adam (batch 32, learning rate $3 \cdot 10^{-3}$), it reaches about 0.97 on held-out digits in 400 steps.

## Working pooling by hand

Before writing code, it helps to pin the tie rule with numbers you can check. Pool this $4 \times 4$ image with $2 \times 2$ windows and stride 2 ($P = Q = \lfloor (4-2)/2 \rfloor + 1 = 2$):

$$X = \begin{pmatrix} 1 & 4 & 2 & 2 \\ 3 & 2 & 0 & 1 \\ 0 & 3 & 5 & 1 \\ 1 & 2 & 4 & 1 \end{pmatrix}$$

The four windows reduce as follows.

| Window | Taps (row-major) | Max, at pixel | Mean |
|---|---|---|---|
| top-left | 1, 4, 3, 2 | 4 at $(0, 1)$ | $10/4 = 2.5$ |
| top-right | 2, 2, 0, 1 | 2 at $(0, 2)$ (tie with $(0, 3)$: the first wins) | $5/4 = 1.25$ |
| bottom-left | 0, 3, 1, 2 | 3 at $(2, 1)$ | $6/4 = 1.5$ |
| bottom-right | 5, 1, 4, 1 | 5 at $(2, 2)$ | $11/4 = 2.75$ |

So max pooling gives $\begin{pmatrix} 4 & 2 \\ 3 & 5 \end{pmatrix}$ and average pooling $\begin{pmatrix} 2.5 & 1.25 \\ 1.5 & 2.75 \end{pmatrix}$. Now backpropagate $L = \sum y$, so every $g = 1$. Max pooling puts ones at $(0,1)$, $(0,2)$, $(2,1)$, $(2,2)$ and zeros elsewhere; pixel $(0, 3)$ gets nothing although it also holds a 2. Average pooling puts $1/4$ at every pixel. These are the exact numbers a correct solution must produce, and they are the first test case listed later.

## Designing the API

The public surface is four layers and two functions:

```python
class Conv2d(Module):                       # weight [c_out, c_in/groups, kh, kw], bias [c_out] or None
    def __init__(self, c_in, c_out, k, stride=1, pad=0, dilation=1, groups=1, bias=True, rng=None): ...
    def forward(self, x: Tensor) -> Tensor: ...          # [N, C, H, W] -> [N, K, P, Q]
class Conv1d(Module):                       # weight [c_out, c_in/groups, k]
    def __init__(self, c_in, c_out, k, stride=1, pad=0, dilation=1, groups=1, bias=True, rng=None): ...
    def forward(self, x: Tensor) -> Tensor: ...          # [N, C, L] -> [N, K, Lout]
def max_pool2d(x: Tensor, k: int, stride: Optional[int] = None) -> Tensor: ...   # [N, C, H, W] -> [N, C, P, Q]
def avg_pool2d(x: Tensor, k: int, stride: Optional[int] = None) -> Tensor: ...
class LeNet5(Module):                       # conv1, conv2, fc1, fc2, fc3
    def __init__(self, n_classes=10, in_hw=8, rng=None): ...
    def forward(self, x: Tensor) -> Tensor: ...          # [N, 1, in_hw, in_hw] -> [N, n_classes]
```

For `Conv2d`, `k`, `stride`, `pad`, and `dilation` are an int or a (row, column) pair; for `Conv1d` they are ints. `rng=None` means `PCG32(0).substream("init")`. Each layer is a single autograd node built with `from_op`. Everything invalid raises `ValueError`: a size, stride, or dilation below 1, negative padding, groups that do not divide both channel counts, an input of the wrong rank or channel count, a kernel or window larger than the (padded) input, and `in_hw` not a multiple of 4.

## Building it step by step

The design follows from six observations, each of which removes code rather than adding it.

1. **Reuse the proven matrix.** The mathematics book showed $Y_n = \bar W C_n$ with $C_n = \text{im2col}(x_n)$, and that col2im is im2col's adjoint. So a conv layer needs no new indexing code: one call to `im2col` forward, one matmul per group, and the three formulas above backward.
2. **One autograd node, not many.** Building the conv out of `Tensor` ops (gathers, matmuls) would record thousands of nodes and be slow. A single `from_op` node with a hand-written VJP records one, and the VJP is exactly the derivation above.
3. **Groups are slices, not a different algorithm.** Because im2col's rows are ordered $(c, r, s)$, group $g$'s input rows and output channels are contiguous ranges. The forward and the VJP loop over groups and apply the one-group formulas to those slices.
4. **Conv1d should not be a second implementation.** A sequence is an image one row tall, so `Conv1d` reshapes and calls the 2-D op with stride, padding, and dilation 1, 0, 1 on the row axis; the reshape carries the gradient.
5. **Pooling is im2col with one channel.** Folding channels into the batch makes each column one window of $k^2$ taps. The forward is a reduction over the column; the backward writes into the column and col2im adds the overlaps.
6. **Compute in float64, return the input's dtype.** Matching torch within $10^{-10}$ needs float64 accumulation; casting back keeps a float32 model float32.

Start with the conv op. The forward builds the columns once and keeps them for the backward, because they are $dW$'s right-hand factor:

```python
def _conv2d_op(x, w, b, stride, pad, dilation, groups) -> Tensor:
    xd, wd = x.data, w.data
    n, c, h, wid = xd.shape
    kout, cg, r, s = wd.shape                       # cg = C / groups input channels per group
    # ... raise if c != cg * groups
    p = conv_out_size(h, r, stride[0], pad[0], dilation[0])
    q = conv_out_size(wid, s, stride[1], pad[1], dilation[1])
    kg = kout // groups                              # output channels per group
    dt = xd.dtype
    cols = im2col(xd, r, s, stride, pad, dilation)  # [N, C*R*S, P*Q], rows ordered (c, r, s)
    crs = cg * r * s                                 # rows owned by one group
    wflat = wd.reshape(kout, crs).astype(np.float64) # W-bar, one row per output channel
    y = np.empty((n, kout, p * q))
    for g in range(groups):                          # Y_g = W_g C_g: block-diagonal W, one block per group
        y[:, g * kg : (g + 1) * kg] = np.einsum(
            "kj,njl->nkl", wflat[g * kg : (g + 1) * kg], cols[:, g * crs : (g + 1) * crs]
        )
    if b is not None:
        y += b.data.astype(np.float64)[None, :, None] # b 1^T, broadcast over positions
    out = y.reshape(n, kout, p, q).astype(dt)
```

The VJP is the three formulas, applied per group:

```python
    def vjp(gout):
        gy = gout.reshape(n, kout, p * q).astype(np.float64)   # G_n
        dcols = np.empty_like(cols)
        dw = np.empty((kout, crs))
        for g in range(groups):
            gg = gy[:, g * kg : (g + 1) * kg]          # this group's output gradient
            cg_ = cols[:, g * crs : (g + 1) * crs]     # this group's columns
            dw[g * kg : (g + 1) * kg] = np.einsum("nkl,njl->kj", gg, cg_)        # sum_n G_n C_n^T
            dcols[:, g * crs : (g + 1) * crs] = np.einsum(
                "kj,nkl->njl", wflat[g * kg : (g + 1) * kg], gg                  # W^T G_n
            )
        dx = col2im(dcols, xd.shape, r, s, stride, pad, dilation)  # the adjoint: sums overlapping windows
        grads = [dx.astype(dt), dw.reshape(wd.shape).astype(wd.dtype)]
        if b is not None:
            grads.append(gy.sum(axis=(0, 2)).astype(b.data.dtype))  # over the batch and every position
        return tuple(grads)

    parents = (x, w) if b is None else (x, w, b)
    return from_op(out, parents, vjp, "conv2d")
```

`Conv2d.__init__` validates every argument and initializes with the group's fan-in:

```python
fan_in = (c_in // groups) * kh * kw                 # one group's channels: what one output actually sums
self.weight = Tensor(
    normal_init((c_out, c_in // groups, kh, kw), 1.0 / math.sqrt(fan_in), r),
    requires_grad=True,
)
self.bias = Tensor(np.zeros(c_out), requires_grad=True) if bias else None
```

`Conv1d.forward` is the one-row reshape:

```python
n, c, length = x.shape
x4 = F.reshape(x, (n, c, 1, length))                # an image one row tall
k_out, cg, k = self.weight.shape
w4 = F.reshape(self.weight, (k_out, cg, 1, k))
# row axis: stride 1, padding 0 (padding it would add rows of zeros), dilation 1
y = _conv2d_op(x4, w4, self.bias, (1, self.stride), (0, self.pad), (1, self.dilation), self.groups)
return F.reshape(y, (n, k_out, y.shape[3]))
```

The two pooling functions share `_pool_cols`, which defaults the stride to the window size and folds channels into the batch:

```python
s = k if stride is None else stride                 # non-overlapping windows by default
# ...
cols = im2col(x.data.reshape(n * c, 1, h, w), k, k, (s, s))   # [N*C, k*k, P*Q]
```

Max pooling keeps the argmax for the backward:

```python
def max_pool2d(x, k, stride=None):
    cols, s, p, q = _pool_cols(x, k, stride, "max_pool2d")
    n, c, h, w = x.shape
    # argmax returns the first maximum: the lowest (r, s) in the window,
    # which is the lowest flat index of x
    arg = np.argmax(cols, axis=1)                   # [N*C, P*Q]
    y = np.take_along_axis(cols, arg[:, None, :], axis=1)[:, 0]
    out = y.reshape(n, c, p, q).astype(x.data.dtype)

    def vjp(g):
        dcols = np.zeros_like(cols)                 # zero at every tap that was not the max
        np.put_along_axis(dcols, arg[:, None, :], g.reshape(n * c, 1, p * q).astype(np.float64), axis=1)
        dx = col2im(dcols, (n * c, 1, h, w), k, k, (s, s))    # overlapping windows add
        return (dx.reshape(n, c, h, w).astype(x.data.dtype),)

    return from_op(out, (x,), vjp, "max_pool2d")
```

Average pooling's backward spreads $g/k^2$ to every tap:

```python
def vjp(g):
    gg = g.reshape(n * c, 1, p * q).astype(np.float64) / (k * k)
    dcols = np.broadcast_to(gg, cols.shape).copy()  # copy: col2im needs a writable, real array
    dx = col2im(dcols, (n * c, 1, h, w), k, k, (s, s))
    return (dx.reshape(n, c, h, w).astype(x.data.dtype),)
```

Finally, `LeNet5` is the composition described earlier, with every layer drawing from the same `rng` in registration order, and a flatten that keeps NCHW's $(c, h, w)$ order:

```python
def forward(self, x):
    h = max_pool2d(F.relu(self.conv1(x)), 2)
    h = max_pool2d(F.relu(self.conv2(h)), 2)
    h = F.reshape(h, (h.shape[0], -1))  # (c, h, w) order, as torch.flatten
    h = F.relu(self.fc1(h))
    h = F.relu(self.fc2(h))
    return self.fc3(h)
```

The full solution is [`conv.py`](../../../../course/solutions/python/tinyllm/vision/conv.py).

### Why this design

The forward and backward are each $O(N \cdot K \cdot (C/\gamma) R S \cdot PQ)$ multiply-adds, the same as a direct convolution, done as dense matmuls. The price is memory: the columns hold every input pixel up to $RS$ times ($CRS \cdot PQ$ per image), and they are kept until the backward pass. The alternative, six nested loops over $(n, k, c, r, s, l)$ in Python, would be orders of magnitude slower and need its own backward indexing; im2col turns the whole layer into `einsum` calls and reuses an adjoint that is already proven.

A hand-written VJP also beats composing autograd ops. Composition would be correct but would record a node per gather and keep many intermediates; one node keeps exactly `cols` and the weights. And the deterministic tie rule is a choice, not a mathematical necessity: any subgradient is valid at a tie, but fixing the first index matches PyTorch, so golden comparisons and checkpoints behave identically, and repeated runs give the same gradients.

## What can go wrong

A correct solution is checked on the cases below. Several of them are golden comparisons against PyTorch, because later sections load pretrained conv weights into these layers.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: pooling | unit, smoke | the hand-worked maxima, means, and both gradients, exactly | the tie rule is pinned |
| Conv2d matches torch | golden | forward, $dx$, $dW$, $db$ within $10^{-10}$ of torch on three layers: LeNet's (5 x 5, pad 2), a grouped, strided, dilated one, and a patch embedding (kernel 4, stride 4) | the vision transformer and the planned vision-language models load pretrained conv weights |
| Conv1d matches torch | golden | the same for the two Whisper stem shapes (k 3 pad 1; k 3 stride 2 pad 1) | Whisper's stem halves the frames |
| pooling matches torch | golden | max and average, forward and $dx$ within $10^{-12}$, including k 3 stride 2 (overlap) and integer inputs full of ties | gradients add where windows overlap |
| gradient check, Conv2d | gradcheck | the hand-written vjp for $x$, $W$, $b$ against central differences, with unequal strides, padding, dilation, and 2 groups | the backward is correct beyond the fixture cases |
| gradient check, Conv1d | gradcheck | gradients reach $x$, $W$, $b$ through the reshapes (stride 2, pad 1) | training Whisper's stem |
| max-pool ties route to the lowest index | property | on a constant image each window's gradient lands on its top-left pixel | deterministic gradients |
| shapes, init, and dtype | unit | names (`weight`, `bias`), KCRS shape with $C/\gamma$ channels, std near $1/\sqrt{f}$, `bias=False`, float32 in gives float32 out | checkpoints load by name and shape |
| LeNet is the composition | unit | the ten parameter names in order, `fc1` of width 64 (144 for 12 x 12), and the forward equal to the layers composed by hand | a torch LeNet checkpoint loads |
| bad arguments rejected | boundary | every `ValueError` listed in the API | wrong layouts fail at the line that is wrong |
| LeNet learns digits | learning | 400 Adam steps, held-out accuracy about 0.97 | the layers learn together |

The mistakes that most often slip past shape checks are subtle indexing errors. Routing max-pool ties to the last maximum makes gradients differ from torch on flat or integer images; `np.argmax` returns the first maximum in row-major window order, which avoids it. In grouped convolution, two related bugs are every group reading the first group's columns (the loop must slice `cols[:, g * crs : (g + 1) * crs]`) and building $dx$ from the first group's weights only (`dcols` must use `wflat[g * kg : (g + 1) * kg]` for group `g`); either makes later groups compute on, or learn from, the wrong channels.

> **Warning:** Summing the bias gradient over one image instead of the batch lets biases learn from only a fraction of the signal. `gy.sum(axis=(0, 2))` sums the batch and the positions together.

Several pitfalls are off-by-a-factor errors. An average-pool backward divided by $k$ instead of $k^2$ makes gradients $k$ times too large; the VJP divides by `k * k`. A fan-in that counts every input channel instead of one group's starts grouped layers $\sqrt{\gamma}$ times too small; the solution uses `fan_in = (c_in // groups) * kh * kw`. A pooling stride that defaults to 1 instead of $k$ gives outputs nearly as large as the input; `_pool_cols` uses `s = k` when `stride is None`.

The remaining ones are layout errors. If `Conv1d` pads the dummy row axis too, it produces extra rows of zeros and the final reshape fails or is wrong, so `Conv1d.forward` passes padding `(0, self.pad)`. If LeNet flattens in $(h, w, c)$ order, it still trains, but a torch checkpoint no longer loads; `F.reshape(h, (N, -1))` on NCHW keeps $(c, h, w)$ order. And if the constructors did not check that groups divide `c_out`, you could build a weight whose groups have unequal sizes; they raise unless `groups` divides both channel counts.

## Where this leads

These layers are used immediately. In [Vision transformer encoder](03-vision-transformer.md), `PatchEmbed` is `Conv2d(3, d, p, stride=p)`: non-overlapping windows, one token per patch. In [Whisper encoder](../15-speech/02-whisper-encoder.md), the stem is `Conv1d(n_mels, d, 3, pad=1)` followed by `Conv1d(d, d, 3, stride=2, pad=1)`. The planned vision-language model section will load pretrained vision towers' patch-embedding weights into `Conv2d`.

Production libraries go further in several directions. `torch.nn.Conv2d` on cuDNN selects an algorithm per shape (im2col plus GEMM, Winograd, FFT) and supports a channels-last memory format; see `torch/nn/modules/conv.py` and cuDNN's `cudnnConvolutionForward`. On CPU, oneDNN and XNNPACK run direct convolutions with JIT-generated kernels and avoid im2col's $k^2$ memory blow-up (`oneapi-src/oneDNN`). `F.max_pool2d` in `torch/nn/functional.py` adds padding, dilation, `ceil_mode`, and returned indices for unpooling. And modern architectures such as ConvNeXt and ViT add normalization, residual streams, and depthwise (groups equal to channels) convolutions; a ViT keeps only the first conv.

### Further reading

- LeCun, Bottou, Bengio, Haffner, "Gradient-based learning applied to document recognition" (1998), section II.
- Chellapilla, Puri, Simard, "High performance convolutional neural networks for document processing" (2006), the im2col paper.
- Dumoulin and Visin, "A guide to convolution arithmetic for deep learning" (2016, free).
- Liu et al., "A ConvNet for the 2020s" (2022).

## Key terms

- **im2col**: the linear map that copies each receptive field of an image into a column, so a convolution becomes a matmul.
- **col2im**: the adjoint of im2col; it adds columns back into image positions, summing where windows overlap.
- **grouped convolution**: a convolution whose channels split into $\gamma$ independent groups, giving a block-diagonal weight matrix.
- **fan-in**: the number of products one output sums, $(C/\gamma) R S$ for a conv, which sets the init scale $1/\sqrt{f}$.
- **subgradient**: a valid gradient choice at a non-differentiable point, such as a tie in max pooling.

## Summary

- A conv layer is one matmul per group on im2col's columns, so its backward pass is three short formulas: $dW = \sum_n G_n C_n^\top$, $dx_n = \text{col2im}(W^\top G_n)$, $db = \sum G$.
- `Conv1d` is `Conv2d` on an image one row tall; stride, padding, and dilation apply to the time axis only.
- Max pooling's backward routes each output's gradient to its window's maximum; a tie goes to the lowest flat index, and overlapping windows add.
- Average pooling's backward spreads $g/k^2$ over the window.
- The layers learn together: LeNet-5 on 8 x 8 digits reaches about 0.97 held-out accuracy in 400 Adam steps.

## Review questions

1. Why is $\partial L/\partial x_n = \text{col2im}(\bar W^\top G_n)$ and not $\text{im2col}$ of something?

   <details><summary>Answer</summary>

   The forward reads $x$ through im2col, a linear map. The gradient through a linear map is its adjoint applied to the upstream gradient, and col2im is im2col's adjoint. Concretely, col2im adds every copy of a pixel back together, which is the chain rule for a pixel that fed several outputs.

   </details>

2. Why does `Conv1d` pass padding `(0, self.pad)` and not `(self.pad, self.pad)`?

   <details><summary>Answer</summary>

   The row axis has height 1 and a kernel of height 1. Padding it by $p$ would produce $1 + 2p$ output rows, all but one reading only zeros, and the final reshape to `[N, K, Lout]` would fail or mix rows into time steps.

   </details>

3. In the hand-worked pooling example, why does pixel $(0, 3)$ get no gradient even though it equals the window's maximum?

   <details><summary>Answer</summary>

   The window's taps are $2, 2, 0, 1$ in row-major order; `np.argmax` returns the first maximum, tap 0, which is pixel $(0, 2)$. The VJP writes $g$ only at the argmax, so $(0, 3)$ gets zero. Any split would be a valid subgradient, but only one choice matches PyTorch.

   </details>

4. Why is the fan-in $(C/\gamma) R S$ and not $C R S$?

   <details><summary>Answer</summary>

   Each output of a grouped conv sums only its own group's $C/\gamma$ input channels. Its variance is $(C/\gamma) R S \sigma^2 \mathbb{E}[x^2]$, so $\sigma = 1/\sqrt{(C/\gamma) R S}$ keeps the scale. Using $C R S$ would start a grouped layer $\sqrt{\gamma}$ times too small.

   </details>

5. Why does the max-pool VJP not need the input values, only `arg`?

   <details><summary>Answer</summary>

   Near a point with a unique maximum the output is a copy of one input, so the Jacobian is a selection matrix determined entirely by which tap won. Keeping the argmax indices is enough to route each gradient.

   </details>
