"""tinyllm.vision.conv (L13.1): conv layers and pooling on autograd.

M12.1 showed that a convolution is a matrix: im2col copies every receptive
field into a column and the conv becomes one matmul, and col2im (the adjoint
of that copy) is the backward pass. Here the two become layers: Conv2d and
Conv1d are single autograd ops whose vector-Jacobian product is written by
hand with im2col and col2im, and pooling routes gradients through the same
pair. LeNet-5 puts them together. L13.3's PatchEmbed is a Conv2d whose kernel
equals its stride, and L14.2's Whisper stem is two Conv1d layers.

Contract: contracts/py/tinyllm/vision/conv.pyi.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor, from_op
from tinyllm.nn.init import normal_init
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.sig.conv import col2im, conv_out_size, im2col


def _pair(v: Any, who: str) -> tuple[int, int]:
    """(row, column) from an int or a pair of ints."""
    # SOLUTION-BEGIN L13.1
    t = (v, v) if isinstance(v, (int, np.integer)) else tuple(v)
    if len(t) != 2 or not all(isinstance(a, (int, np.integer)) for a in t):
        raise ValueError(f"{who}: need an int or a pair of ints, got {v!r}")
    return int(t[0]), int(t[1])
    # SOLUTION-END


def _conv2d_op(
    x: Tensor,
    w: Tensor,
    b: Optional[Tensor],
    stride: tuple[int, int],
    pad: tuple[int, int],
    dilation: tuple[int, int],
    groups: int,
) -> Tensor:
    """One autograd op: y = W_g im2col(x)_g per group (+ b), with the hand-written vjp."""
    # SOLUTION-BEGIN L13.1
    xd, wd = x.data, w.data
    n, c, h, wid = xd.shape
    kout, cg, r, s = wd.shape
    if c != cg * groups:
        raise ValueError(
            f"conv2d: x has {c} channels, the weight expects {cg} x {groups} groups = {cg * groups}"
        )
    p = conv_out_size(h, r, stride[0], pad[0], dilation[0])
    q = conv_out_size(wid, s, stride[1], pad[1], dilation[1])
    kg = kout // groups
    dt = xd.dtype
    # Columns of the whole input, rows ordered (c, r, s): group g owns the
    # contiguous rows g*cg*r*s .. (g+1)*cg*r*s - 1.
    cols = im2col(xd, r, s, stride, pad, dilation)  # [N, C*R*S, P*Q]
    crs = cg * r * s
    wflat = wd.reshape(kout, crs).astype(np.float64)
    y = np.empty((n, kout, p * q))
    for g in range(groups):
        y[:, g * kg : (g + 1) * kg] = np.einsum(
            "kj,njl->nkl", wflat[g * kg : (g + 1) * kg], cols[:, g * crs : (g + 1) * crs]
        )
    if b is not None:
        y += b.data.astype(np.float64)[None, :, None]
    out = y.reshape(n, kout, p, q).astype(dt)

    def vjp(gout: NDArray) -> tuple:
        gy = gout.reshape(n, kout, p * q).astype(np.float64)
        dcols = np.empty_like(cols)
        dw = np.empty((kout, crs))
        for g in range(groups):
            gg = gy[:, g * kg : (g + 1) * kg]  # [N, K/g, PQ]
            cg_ = cols[:, g * crs : (g + 1) * crs]  # [N, CRS/g, PQ]
            dw[g * kg : (g + 1) * kg] = np.einsum("nkl,njl->kj", gg, cg_)
            dcols[:, g * crs : (g + 1) * crs] = np.einsum(
                "kj,nkl->njl", wflat[g * kg : (g + 1) * kg], gg
            )
        dx = col2im(dcols, xd.shape, r, s, stride, pad, dilation)
        grads = [dx.astype(dt), dw.reshape(wd.shape).astype(wd.dtype)]
        if b is not None:
            grads.append(gy.sum(axis=(0, 2)).astype(b.data.dtype))
        return tuple(grads)

    parents = (x, w) if b is None else (x, w, b)
    return from_op(out, parents, vjp, "conv2d")
    # SOLUTION-END


class Conv2d(Module):
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
        # SOLUTION-BEGIN L13.1
        super().__init__()
        kh, kw = _pair(k, "k")
        self.stride = _pair(stride, "stride")
        self.pad = _pair(pad, "pad")
        self.dilation = _pair(dilation, "dilation")
        if min(c_in, c_out, kh, kw, *self.stride, *self.dilation) < 1 or min(self.pad) < 0:
            raise ValueError(
                f"Conv2d: sizes, stride, and dilation must be >= 1 and pad >= 0 "
                f"(c_in={c_in} c_out={c_out} k={k} stride={stride} pad={pad} dilation={dilation})"
            )
        if groups < 1 or c_in % groups or c_out % groups:
            raise ValueError(f"Conv2d: groups={groups} must divide c_in={c_in} and c_out={c_out}")
        self.groups = int(groups)
        r = rng if rng is not None else PCG32(0).substream("init")
        fan_in = (c_in // groups) * kh * kw
        self.weight = Tensor(
            normal_init((c_out, c_in // groups, kh, kw), 1.0 / math.sqrt(fan_in), r),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(c_out), requires_grad=True) if bias else None
        # SOLUTION-END

    def forward(self, x: Tensor) -> Tensor:
        # SOLUTION-BEGIN L13.1
        if x.ndim != 4:
            raise ValueError(f"Conv2d: need x [N, C, H, W], got shape {x.shape}")
        return _conv2d_op(x, self.weight, self.bias, self.stride, self.pad, self.dilation, self.groups)
        # SOLUTION-END


class Conv1d(Module):
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
        # SOLUTION-BEGIN L13.1
        super().__init__()
        if min(c_in, c_out, k, stride, dilation) < 1 or pad < 0:
            raise ValueError(
                f"Conv1d: sizes, stride, and dilation must be >= 1 and pad >= 0 "
                f"(c_in={c_in} c_out={c_out} k={k} stride={stride} pad={pad} dilation={dilation})"
            )
        if groups < 1 or c_in % groups or c_out % groups:
            raise ValueError(f"Conv1d: groups={groups} must divide c_in={c_in} and c_out={c_out}")
        self.stride, self.pad, self.dilation, self.groups = int(stride), int(pad), int(dilation), int(groups)
        r = rng if rng is not None else PCG32(0).substream("init")
        fan_in = (c_in // groups) * k
        self.weight = Tensor(
            normal_init((c_out, c_in // groups, k), 1.0 / math.sqrt(fan_in), r), requires_grad=True
        )
        self.bias = Tensor(np.zeros(c_out), requires_grad=True) if bias else None
        # SOLUTION-END

    def forward(self, x: Tensor) -> Tensor:
        # SOLUTION-BEGIN L13.1
        if x.ndim != 3:
            raise ValueError(f"Conv1d: need x [N, C, L], got shape {x.shape}")
        # A 1-D conv is a 2-D conv over an image one row tall.
        n, c, length = x.shape
        x4 = F.reshape(x, (n, c, 1, length))
        k_out, cg, k = self.weight.shape
        w4 = F.reshape(self.weight, (k_out, cg, 1, k))
        y = _conv2d_op(
            x4, w4, self.bias, (1, self.stride), (0, self.pad), (1, self.dilation), self.groups
        )
        return F.reshape(y, (n, k_out, y.shape[3]))
        # SOLUTION-END


def _pool_cols(x: Tensor, k: int, stride: Optional[int], who: str) -> tuple[NDArray, int, int, int]:
    """Every k x k window of every channel as a column: [N*C, k*k, P*Q], with stride, P, Q."""
    # SOLUTION-BEGIN L13.1
    if x.ndim != 4:
        raise ValueError(f"{who}: need x [N, C, H, W], got shape {x.shape}")
    s = k if stride is None else stride
    if k < 1 or s < 1:
        raise ValueError(f"{who}: k and stride must be >= 1, got k={k} stride={stride}")
    n, c, h, w = x.shape
    p = conv_out_size(h, k, s)
    q = conv_out_size(w, k, s)
    # One channel at a time: fold the channels into the batch so im2col's
    # rows are only the window's k*k taps, in row-major (r, s) order.
    cols = im2col(x.data.reshape(n * c, 1, h, w), k, k, (s, s))
    return cols, s, p, q
    # SOLUTION-END


def max_pool2d(x: Tensor, k: int, stride: Optional[int] = None) -> Tensor:
    # SOLUTION-BEGIN L13.1
    cols, s, p, q = _pool_cols(x, k, stride, "max_pool2d")
    n, c, h, w = x.shape
    # argmax returns the first maximum: the lowest (r, s) in the window, which
    # is the lowest flat index of x.
    arg = np.argmax(cols, axis=1)  # [N*C, P*Q]
    y = np.take_along_axis(cols, arg[:, None, :], axis=1)[:, 0]
    out = y.reshape(n, c, p, q).astype(x.data.dtype)

    def vjp(g: NDArray) -> tuple[NDArray]:
        dcols = np.zeros_like(cols)
        np.put_along_axis(dcols, arg[:, None, :], g.reshape(n * c, 1, p * q).astype(np.float64), axis=1)
        dx = col2im(dcols, (n * c, 1, h, w), k, k, (s, s))
        return (dx.reshape(n, c, h, w).astype(x.data.dtype),)

    return from_op(out, (x,), vjp, "max_pool2d")
    # SOLUTION-END


def avg_pool2d(x: Tensor, k: int, stride: Optional[int] = None) -> Tensor:
    # SOLUTION-BEGIN L13.1
    cols, s, p, q = _pool_cols(x, k, stride, "avg_pool2d")
    n, c, h, w = x.shape
    out = cols.mean(axis=1).reshape(n, c, p, q).astype(x.data.dtype)

    def vjp(g: NDArray) -> tuple[NDArray]:
        gg = g.reshape(n * c, 1, p * q).astype(np.float64) / (k * k)
        dcols = np.broadcast_to(gg, cols.shape).copy()
        dx = col2im(dcols, (n * c, 1, h, w), k, k, (s, s))
        return (dx.reshape(n, c, h, w).astype(x.data.dtype),)

    return from_op(out, (x,), vjp, "avg_pool2d")
    # SOLUTION-END


class LeNet5(Module):
    def __init__(self, n_classes: int = 10, in_hw: int = 8, rng: Any = None) -> None:
        # SOLUTION-BEGIN L13.1
        super().__init__()
        if in_hw < 4 or in_hw % 4:
            raise ValueError(f"LeNet5: in_hw must be a multiple of 4 and >= 4, got {in_hw}")
        if n_classes < 2:
            raise ValueError(f"LeNet5: n_classes must be >= 2, got {n_classes}")
        r = rng if rng is not None else PCG32(0).substream("init")
        self.in_hw = int(in_hw)
        self.conv1 = Conv2d(1, 6, 5, pad=2, rng=r)
        self.conv2 = Conv2d(6, 16, 5, pad=2, rng=r)
        self.fc1 = Linear(16 * (in_hw // 4) ** 2, 120, rng=r)
        self.fc2 = Linear(120, 84, rng=r)
        self.fc3 = Linear(84, n_classes, rng=r)
        # SOLUTION-END

    def forward(self, x: Tensor) -> Tensor:
        # SOLUTION-BEGIN L13.1
        h = max_pool2d(F.relu(self.conv1(x)), 2)
        h = max_pool2d(F.relu(self.conv2(h)), 2)
        h = F.reshape(h, (h.shape[0], -1))  # (c, h, w) order, as torch.flatten
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)
        # SOLUTION-END
