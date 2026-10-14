<!-- ol:module L13.1 -->
# Conv layers and pooling on autograd

## Overview

| | |
|---|---|
| **Module** | `L13.1` · build · Python · Pass 12 · 4 to 6 h |
| **You build** | `python/tinyllm/vision/conv.py`: `Conv2d` and `Conv1d` (stride, padding, dilation, groups; im2col forward, col2im backward), `max_pool2d`, `avg_pool2d`, `LeNet5` |
| **Contract** | [`course/contracts/py/tinyllm/vision/conv.pyi`](../../../course/contracts/py/tinyllm/vision/conv.pyi) |
| **Tests** | `course/tests/L13.1/test_conv_layers.py` (what they check: section 4), golden values from `course/oracle/L13.1/conv_layers_torch.py` (`torch.nn.Conv2d`, `Conv1d`, `F.max_pool2d`, `F.avg_pool2d`, forward and autograd backward in float64) in `course/fixtures/L13.1/conv_layers_torch.npz`; digits from `course/oracle/L13.1/digits_synth.py` in `course/fixtures/digits/digits.npz` |
| **Needs** | `M12.1` `im2col`, `col2im`, `conv_out_size` · `L0.1` `Tensor`, `from_op` · `L0.2` `reshape`, `relu` · `L0.4` `Module`, `Linear` · `M07.3` `normal_init` · `M06.3` `PCG32` (or `--ref-deps`). Reading: `L0.3` cross-entropy, `M10.3` Adam, and the plain-loop LeNet in [`ml/06-neural-architectures/code/lenet.rs`](../../06-neural-architectures/code/lenet.rs) |
| **Used by** | `L13.3` (`PatchEmbed` is a `Conv2d` whose kernel equals its stride) · later `L14.2` (the Whisper stem is two `Conv1d` layers) |
| **Milestone** | `MS-L13` step 1 (LeNet learns: `test_acc` at or above the calibrated threshold in 400 steps) |
| **Optional depth** | LeCun, Bottou, Bengio, Haffner, "Gradient-based learning applied to document recognition" (1998), section II; Chellapilla, Puri, Simard, "High performance convolutional neural networks for document processing" (2006), the im2col paper; Dumoulin and Visin, "A guide to convolution arithmetic for deep learning" (2016, free) |

## Key Takeaways

- A conv layer is one matmul per group on im2col's columns, so its backward pass is three short formulas: $dW = \sum_n G_n C_n^\top$, $dx_n = \text{col2im}(W^\top G_n)$, $db = \sum G$ (`test_conv2d_matches_torch`, `test_gradcheck_conv2d`).
- `Conv1d` is `Conv2d` on an image one row tall; stride, padding, and dilation apply to the time axis only (`test_conv1d_matches_torch`, `test_gradcheck_conv1d`).
- Max pooling's backward routes each output's gradient to its window's maximum; a tie goes to the lowest flat index, and overlapping windows add (`test_hand_example_pooling`, `test_max_pool_ties_route_to_lowest_index`).
- Average pooling's backward spreads $g/k^2$ over the window (`test_pooling_matches_torch`).
- The layers learn together: LeNet-5 on 8 x 8 digits reaches about 0.97 held-out accuracy in 400 Adam steps (`test_lenet_learns_digits`).

## How to work this chapter

```bash
ol start L13.1              # stubs python/tinyllm/vision/conv.py into your repo
ol tests L13.1              # read the test catalog first: rung R0, you write no tests here
ol check L13.1              # exit code is the verdict
ol check L13.1 --ref-deps   # only if one of your Needs modules is not passing yet
ol diff  L13.1              # after passing: your code against the reference
```

---

## 1. Why now

`M12.1` proved that a convolution is a matrix and gave you `im2col` and its adjoint `col2im`, but only as numpy functions: nothing in the system can train a kernel yet. The autograd engine from Part 0 has `Linear` and no layer that shares weights across positions, so a model that reads an image must flatten it and learn every pixel position separately. This module turns the two functions into layers with hand-written backward passes on `L0.1`'s `Tensor`. Everything visual or audible in Pass 12 starts here: a ViT's patch embedding (`L13.3`) is a `Conv2d` whose kernel equals its stride, and Whisper's stem (`L14.2`) is two `Conv1d` layers. A wrong backward pass here passes every shape check and shows up weeks later as a vision tower that does not learn, so the gradients are proven three ways: against PyTorch, against central differences, and by training LeNet-5.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x$ | input batch (NCHW) | `float[N, C, H, W]` |
| $W$ | kernel bank (KCRS), $\gamma$ groups | `float[K, C/γ, R, S]` |
| $b$ | bias | `float[K]` |
| $P, Q$ | output height and width (`conv_out_size`) | int |
| $C_n$ | `im2col` of image $n$, rows ordered $(c, r, s)$ | `float64[C·R·S, P·Q]` |
| $G$ | upstream gradient $\partial L/\partial y$ | `float[N, K, P, Q]` |

### 2.1 Conv2d: one matmul forward, its transpose backward

Take one group first ($\gamma = 1$). Flatten the kernels to $\bar W \in \mathbb{R}^{K \times CRS}$. Column $l$ of $C_n$ is the receptive field of output position $l$, so the layer is
$$Y_n = \bar W C_n + b\,\mathbf{1}^\top \in \mathbb{R}^{K \times PQ},$$
reshaped to $[K, P, Q]$. As in `torch.nn.Conv2d`, this is cross-correlation: no kernel flip.

For the backward pass, let $G_n$ be $\partial L/\partial Y_n$ and perturb each input. The loss changes by $dL = \sum_n \langle G_n, dY_n \rangle$ with $dY_n = d\bar W\,C_n + \bar W\,dC_n + db\,\mathbf{1}^\top$. Using $\langle A, BC \rangle = \langle A C^\top, B \rangle = \langle B^\top A, C \rangle$:
$$dL = \Big\langle \sum_n G_n C_n^\top,\ d\bar W \Big\rangle + \sum_n \langle \bar W^\top G_n,\ dC_n \rangle + \Big\langle \sum_n G_n \mathbf{1},\ db \Big\rangle.$$
The first and last terms read off $\partial L/\partial \bar W = \sum_n G_n C_n^\top$ and $\partial L/\partial b = \sum_{n,l} G_{n,\cdot,l}$. For the middle term, `im2col` is linear, so $dC_n = \text{im2col}(dx_n)$, and `M12.1` proved $\langle A, \text{im2col}(v) \rangle = \langle \text{col2im}(A), v \rangle$. Hence
$$\frac{\partial L}{\partial x_n} = \text{col2im}(\bar W^\top G_n).$$
col2im sums where windows overlap, which is exactly the chain rule for a pixel that feeds several outputs. Stride, padding, and dilation change only which pixels `im2col` copies; the three formulas do not change.

**Groups.** With $\gamma$ groups, output channels $gK/\gamma, \dots, (g+1)K/\gamma - 1$ read only input channels $gC/\gamma, \dots, (g+1)C/\gamma - 1$. Because im2col's rows are ordered $(c, r, s)$, group $g$ owns the contiguous rows $g\,\tfrac{C}{\gamma}RS, \dots, (g+1)\tfrac{C}{\gamma}RS - 1$, and the layer is $\gamma$ independent copies of the equations above on those row and channel slices: $\bar W$ is block diagonal. $\gamma$ must divide both $C$ and $K$.

**Initialization.** One output is a sum of $f = (C/\gamma) R S$ products $w_j x_j$. With independent zero-mean weights of variance $\sigma^2$, $\operatorname{Var}(y) = f \sigma^2\,\mathbb{E}[x^2]$, so $\sigma = 1/\sqrt{f}$ keeps the scale of the signal (`M07.3`). The fan-in counts one group's channels, not all $C$. Biases start at zero; weight then bias are drawn from one `PCG32` stream.

The forward and backward run in float64 and the output is cast back to $x$'s dtype.

### 2.2 Conv1d: an image one row tall

A sequence `[N, C, L]` is an image `[N, C, 1, L]`, and a kernel `[K, C/γ, k]` is `[K, C/γ, 1, k]`. A 1-D conv with stride $s$, padding $p$, dilation $d$ is then the 2-D op with stride $(1, s)$, padding $(0, p)$, dilation $(1, d)$, and the output `[N, K, 1, Lout]` drops its row axis. The reshapes are `L0.2` ops, so gradients reach $x$, $W$, and $b$ with no new backward code. The row axis must get padding 0: padding it by $p$ would give $1 + 2p$ output rows, all but one of them reading only zeros.

### 2.3 Pooling

Pooling treats every channel independently and has no parameters. A $k \times k$ window slides with stride $s$ (default $s = k$) and no padding, so $P = \lfloor (H - k)/s \rfloor + 1$ by `conv_out_size`. Folding channels into the batch (`[N·C, 1, H, W]`) makes `im2col` return each window as a column of $k^2$ taps in row-major order.

**Average.** $y = \tfrac{1}{k^2} \sum_j c_j$ over the column $c$. Every tap has derivative $1/k^2$, so the backward writes $g/k^2$ into all $k^2$ taps and col2im adds the overlaps.

**Max.** $y = c_{j^*}$ with $j^* = \arg\max_j c_j$. Near a point with a unique maximum, $y$ is a copy of one input, so the Jacobian is a selection: the backward writes $g$ at tap $j^*$ and zero elsewhere, then col2im sums, so a pixel that is the maximum of two overlapping windows receives both gradients. At a tie the function is not differentiable and any split of $g$ among the tied taps is a valid subgradient. The contract fixes one, as PyTorch does: the first tied tap in row-major window order, which is the lowest flat index in $x$. That makes gradients deterministic.

### 2.4 LeNet-5

The composition, same-padded so any `in_hw` that is a multiple of 4 works: `conv1 = Conv2d(1, 6, 5, pad=2)`, ReLU, `max_pool2d` 2; `conv2 = Conv2d(6, 16, 5, pad=2)`, ReLU, `max_pool2d` 2; flatten in $(c, h, w)$ order (as `torch.flatten`); `fc1 = Linear(16 (in_hw/4)^2, 120)`, ReLU; `fc2 = Linear(120, 84)`, ReLU; `fc3 = Linear(84, n_classes)`. Parameters register in that order and all draw from one rng. For 8 x 8 digits the shapes are $[1, 8, 8] \to [6, 8, 8] \to [6, 4, 4] \to [16, 4, 4] \to [16, 2, 2] \to 64 \to 120 \to 84 \to 10$, with $156 + 2416 + 7800 + 10164 + 850 = 21386$ parameters. Trained with `L0.3`'s cross-entropy and `M10.3`'s Adam (batch 32, learning rate $3 \cdot 10^{-3}$), it reaches about 0.97 on held-out digits in 400 steps.

## 3. Worked example by hand

Pool this $4 \times 4$ image with $2 \times 2$ windows and stride 2 ($P = Q = \lfloor (4-2)/2 \rfloor + 1 = 2$):

$$X = \begin{pmatrix} 1 & 4 & 2 & 2 \\ 3 & 2 & 0 & 1 \\ 0 & 3 & 5 & 1 \\ 1 & 2 & 4 & 1 \end{pmatrix}$$

| Window | Taps (row-major) | Max, at pixel | Mean |
|---|---|---|---|
| top-left | 1, 4, 3, 2 | 4 at $(0, 1)$ | $10/4 = 2.5$ |
| top-right | 2, 2, 0, 1 | 2 at $(0, 2)$ (tie with $(0, 3)$: the first wins) | $5/4 = 1.25$ |
| bottom-left | 0, 3, 1, 2 | 3 at $(2, 1)$ | $6/4 = 1.5$ |
| bottom-right | 5, 1, 4, 1 | 5 at $(2, 2)$ | $11/4 = 2.75$ |

So max pooling gives $\begin{pmatrix} 4 & 2 \\ 3 & 5 \end{pmatrix}$ and average pooling $\begin{pmatrix} 2.5 & 1.25 \\ 1.5 & 2.75 \end{pmatrix}$. Backpropagate $L = \sum y$ (every $g = 1$). Max: ones at $(0,1)$, $(0,2)$, $(2,1)$, $(2,2)$ and zeros elsewhere; pixel $(0, 3)$ gets nothing although it also holds a 2. Average: $1/4$ at every pixel. These are the numbers in `test_hand_example_pooling`, checked exactly.

## 4. The interface

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

For `Conv2d`, `k`, `stride`, `pad`, and `dilation` are an int or a (row, column) pair; for `Conv1d` they are ints. `rng=None` means `PCG32(0).substream("init")`. Each layer is a single autograd node built with `from_op`. Everything invalid raises `ValueError`: a size, stride, or dilation below 1, negative padding, groups that do not divide both channel counts, an input of the wrong rank or channel count, a kernel or window larger than the (padded) input, `in_hw` not a multiple of 4.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_pooling` | unit, smoke | section 3's maxima, means, and both gradients, exactly | you and the test agree on the tie rule |
| `test_conv2d_matches_torch` | golden | forward, $dx$, $dW$, $db$ within $10^{-10}$ of torch on three layers: LeNet's (5 x 5, pad 2), a grouped, strided, dilated one, and a patch embedding (kernel 4, stride 4) | `L13.3` and `L13.7` load pretrained conv weights |
| `test_conv1d_matches_torch` | golden | the same for the two Whisper stem shapes (k 3 pad 1; k 3 stride 2 pad 1) | `L14.2`'s stem halves the frames |
| `test_pooling_matches_torch` | golden | max and average, forward and $dx$ within $10^{-12}$, including k 3 stride 2 (overlap) and integer inputs full of ties | gradients add where windows overlap |
| `test_gradcheck_conv2d` | gradcheck | the hand-written vjp for $x$, $W$, $b$ against frozen central differences, with unequal strides, padding, dilation, and 2 groups | the backward is correct beyond the fixture cases |
| `test_gradcheck_conv1d` | gradcheck | gradients reach $x$, $W$, $b$ through the reshapes (stride 2, pad 1) | training Whisper's stem |
| `test_max_pool_ties_route_to_lowest_index` | property | on a constant image each window's gradient lands on its top-left pixel | deterministic gradients |
| `test_layers_shapes_init_and_dtype` | unit | names (`weight`, `bias`), KCRS shape with $C/\gamma$ channels, std near $1/\sqrt{f}$, `bias=False`, float32 in gives float32 out | checkpoints load by name and shape |
| `test_lenet_is_the_composition` | unit | the ten parameter names in order, `fc1` of width 64 (144 for 12 x 12), and the forward equal to the layers composed by hand | a torch LeNet checkpoint loads |
| `test_rejects_bad_arguments` | boundary | every `ValueError` listed above | wrong layouts fail at the line that is wrong |
| `test_lenet_learns_digits` | learning | 400 Adam steps (the test's own frozen copy), held-out accuracy at or above the calibrated threshold | `MS-L13` step 1 |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. max-pool ties routed to the last maximum | gradients differ from torch on flat or integer images | `test_hand_example_pooling`, `test_pooling_matches_torch`, `test_max_pool_ties_route_to_lowest_index` (mutant `s01`) |
| 2. every group reading the first group's columns | grouped convs compute on the wrong channels | `test_conv2d_matches_torch`, `test_gradcheck_conv2d` (mutant `s02`) |
| 3. the bias gradient summed over one image, not the batch | biases learn from a fraction of the signal | `test_conv2d_matches_torch`, `test_conv1d_matches_torch`, both gradchecks (mutant `s03`) |
| 4. average-pool backward divided by $k$, not $k^2$ | gradients $k$ times too large | `test_hand_example_pooling`, `test_pooling_matches_torch` (mutant `s04`) |
| 5. `Conv1d` padding the dummy row axis too | extra output rows of zeros; the final reshape fails or is wrong | `test_conv1d_matches_torch`, `test_gradcheck_conv1d` (mutant `s05`) |
| 6. LeNet flattening in $(h, w, c)$ order | trains, but a torch checkpoint no longer loads | `test_lenet_is_the_composition` (mutant `s06`) |
| 7. fan-in counting every input channel, not one group | grouped layers start $\sqrt{\gamma}$ too small | `test_layers_shapes_init_and_dtype` (mutant `s07`) |
| pooling stride defaulting to 1 | outputs nearly as large as the input | `test_hand_example_pooling`, `test_pooling_matches_torch` (mutant `m01`) |
| $dx$ built from the first group's weights only | inputs of later groups get the wrong gradient | `test_conv2d_matches_torch`, `test_gradcheck_conv2d` (mutant `m02`) |
| not checking that groups divide `c_out` | a weight whose groups have unequal sizes | `test_rejects_bad_arguments` (mutant `m03`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M12.1` | `im2col` forward, `col2im` backward, `conv_out_size` for every output shape |
| Back | `L0.1` | each layer is one `from_op` node on `Tensor` with a hand-written vjp |
| Back | `L0.2` | `reshape` (the one-row image, the flatten) and `relu` |
| Back | `L0.4` | `Module` registration and LeNet's three `Linear` layers |
| Back | `M07.3`, `M06.3` | `normal_init` with std $1/\sqrt{f}$ from one `PCG32` stream |
| Back | `L0.3`, `M10.3` | cross-entropy and Adam train LeNet (reading) |
| Forward | `L13.3` | `PatchEmbed` is `Conv2d(3, d, p, stride=p)`: non-overlapping windows, one token per patch |
| Forward | `L14.2` | `Conv1d(n_mels, d, 3, pad=1)` then `Conv1d(d, d, 3, stride=2, pad=1)` |
| Forward | `L13.7` | pretrained vision towers load their patch-embedding weights into `Conv2d` (with B14's VLM group) |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Conv2d` | `torch.nn.Conv2d` on cuDNN | algorithm selection per shape (im2col + GEMM, Winograd, FFT), channels-last memory format | `torch/nn/modules/conv.py`, cuDNN `cudnnConvolutionForward` |
| `Conv2d` on CPU | oneDNN, XNNPACK | direct convs with JIT-generated kernels; im2col's $k^2$ memory blow-up avoided | `oneapi-src/oneDNN` |
| `max_pool2d` | `F.max_pool2d` | padding, dilation, `ceil_mode`, returned indices for unpooling | `torch/nn/functional.py` |
| `LeNet5` | ConvNeXt, ViT | normalization, residual streams, depthwise (groups = channels) convs; a ViT keeps only the first conv | Liu et al., "A ConvNet for the 2020s" (2022) |
