# contracts/py/tinyllm/vision/conv.pyi (L13.1): conv layers and pooling on autograd
# chapter: content/courses/build-an-llm/p13-vision/01-conv-layers-and-pooling.md
#
# Shapes are PyTorch's: Conv2d takes x [N, C, H, W] (NCHW) and holds
# weight [K, C/groups, R, S] (KCRS) and bias [K]; Conv1d takes x [N, C, L]
# and holds weight [K, C/groups, R]. "Convolution" is cross-correlation (no
# kernel flip), as torch.nn.Conv2d. stride, pad, and dilation are an int or
# a (row, column) pair for Conv2d and an int for Conv1d; padding is zeros.
#
# The forward pass is M12.1's im2col then one matmul per group; the backward
# pass is a vector-Jacobian product built from the same two functions:
#     dW_g = sum_n G_{n,g} im2col(x)_{n,g}^T
#     dx   = col2im(W_g^T G_{n,g}) per group
#     db   = sum of G over n and every output position
# Outputs keep x's dtype (float32 or float64).
#
# Parameters, in registration order (weight, then bias), drawn from one rng
# (PCG32, M06.3; None means PCG32(0).substream("init")) with
# normal_init(std = 1 / sqrt(C/groups * R * S)) (M07.3); bias zeros.
#
# Pooling has no parameters. max_pool2d and avg_pool2d slide a k x k window
# with stride (default k) and no padding over every channel independently;
# the output size is M12.1's conv_out_size. max_pool2d's backward routes each
# output's gradient to its window's maximum; on a tie, to the element with the
# lowest flat index in x (the first in row-major window order), and a pixel
# that is the maximum of several overlapping windows receives their sum.
from typing import Any, Optional

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module

class Conv2d(Module):
    weight: Tensor  # [c_out, c_in / groups, kh, kw]
    bias: Optional[Tensor]  # [c_out]; None when bias=False
    stride: tuple[int, int]
    pad: tuple[int, int]
    dilation: tuple[int, int]
    groups: int

    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        pad: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        rng: Any = None,
    ) -> None:
        """ValueError for a size below 1, pad below 0, or groups that do not
        divide both c_in and c_out."""
    def forward(self, x: Tensor) -> Tensor:
        """[N, C, H, W] -> [N, K, P, Q], equal to F.conv2d(x, weight, bias,
        stride, pad, dilation, groups). ValueError for x not 4-D, a channel
        count other than c_in, or a kernel that does not fit the padded input."""

class Conv1d(Module):
    weight: Tensor  # [c_out, c_in / groups, k]
    bias: Optional[Tensor]  # [c_out]
    stride: int
    pad: int
    dilation: int
    groups: int

    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        rng: Any = None,
    ) -> None:
        """As Conv2d. Whisper's stem (L14.2) is Conv1d(n_mels, d, 3, pad=1)
        then Conv1d(d, d, 3, stride=2, pad=1)."""
    def forward(self, x: Tensor) -> Tensor:
        """[N, C, L] -> [N, K, Lout], equal to F.conv1d. ValueError for x not
        3-D or a channel count other than c_in."""

def max_pool2d(x: Tensor, k: int, stride: Optional[int] = None) -> Tensor:
    """[N, C, H, W] -> [N, C, P, Q], the maximum of each k x k window
    (F.max_pool2d(x, k, stride)); backward as described above. ValueError for
    x not 4-D, k or stride below 1, or a window larger than the input."""

def avg_pool2d(x: Tensor, k: int, stride: Optional[int] = None) -> Tensor:
    """[N, C, H, W] -> [N, C, P, Q], the mean of each k x k window
    (F.avg_pool2d(x, k, stride)); backward spreads g / k^2 over the window
    and sums where windows overlap. ValueError as max_pool2d."""

class LeNet5(Module):
    """LeCun et al. (1998) for small inputs, same-padded so any in_hw that is
    a multiple of 4 works: conv1 = Conv2d(1, 6, 5, pad=2), ReLU, max_pool2d 2;
    conv2 = Conv2d(6, 16, 5, pad=2), ReLU, max_pool2d 2; flatten to
    16 (in_hw/4)^2 in (c, h, w) order; fc1 = Linear(., 120), ReLU,
    fc2 = Linear(120, 84), ReLU, fc3 = Linear(84, n_classes). Parameters
    registered in that order (conv1, conv2, fc1, fc2, fc3), all drawn from one
    rng."""

    conv1: Conv2d
    conv2: Conv2d
    fc1: Linear
    fc2: Linear
    fc3: Linear

    def __init__(self, n_classes: int = 10, in_hw: int = 8, rng: Any = None) -> None:
        """ValueError unless in_hw >= 4 is a multiple of 4 and n_classes >= 2."""
    def forward(self, x: Tensor) -> Tensor:
        """[N, 1, in_hw, in_hw] -> logits [N, n_classes]."""
