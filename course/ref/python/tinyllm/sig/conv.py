"""tinyllm.sig.conv (M12.1): convolution as a linear map.

A deep-learning "convolution" is a cross-correlation: slide the kernel over
the input and take a dot product at every position, with no flip. It is
linear in the input, so it is a matrix: a doubly block Toeplitz matrix for a
2-D image (`conv_matrix`). Nobody builds that matrix in practice. im2col
copies every receptive field into a column, and the convolution becomes one
matmul of the weights against those columns; col2im is the adjoint of that
copy (it sums overlapping patches back), which is exactly what the backward
pass of a conv needs. L13.1 builds its Conv2d and Conv1d layers on these two
functions, and L14.2's Whisper stem sizes its output with `conv_out_size`.

Contract: contracts/py/tinyllm/sig/conv.pyi.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _pair(v, who: str) -> tuple[int, int]:
    """A (row, column) pair from an int or a 2-sequence of ints."""
    # SOLUTION-BEGIN M12.1
    t = (v, v) if isinstance(v, (int, np.integer)) else tuple(v)
    if len(t) != 2 or not all(isinstance(a, (int, np.integer)) for a in t):
        raise ValueError(f"{who}: need an int or a pair of ints, got {v!r}")
    return int(t[0]), int(t[1])
    # SOLUTION-END


def conv_out_size(
    n: int, k: int, stride: int = 1, pad: int = 0, dilation: int = 1
) -> int:
    # SOLUTION-BEGIN M12.1
    if n < 1 or k < 1 or stride < 1 or dilation < 1 or pad < 0:
        raise ValueError(
            f"conv_out_size: need n, k, stride, dilation >= 1 and pad >= 0, "
            f"got n={n} k={k} stride={stride} pad={pad} dilation={dilation}"
        )
    # A dilated kernel spans d (k - 1) + 1 input samples; the first window
    # starts at 0 and the last one must end inside the padded input.
    span = dilation * (k - 1) + 1
    if n + 2 * pad < span:
        raise ValueError(
            f"conv_out_size: the kernel spans {span} samples but the padded input has {n + 2 * pad}"
        )
    return (n + 2 * pad - span) // stride + 1
    # SOLUTION-END


def _check_groups(c: int, kout: int, cg: int, groups: int, who: str) -> None:
    # SOLUTION-BEGIN M12.1
    if groups < 1 or c % groups or kout % groups:
        raise ValueError(
            f"{who}: groups={groups} must divide both the {c} input and {kout} output channels"
        )
    if cg != c // groups:
        raise ValueError(
            f"{who}: the weight has {cg} input channels per group, expected {c} / {groups} = {c // groups}"
        )
    # SOLUTION-END


def conv1d_direct(
    x: NDArray, w: NDArray, b: NDArray | None, stride=1, pad=0, dilation=1, groups=1
) -> NDArray:
    # SOLUTION-BEGIN M12.1
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x.ndim != 3 or w.ndim != 3:
        raise ValueError(
            f"conv1d_direct: need x [N, C, L] and w [K, C/g, R], got {x.shape}, {w.shape}"
        )
    n, c, length = x.shape
    kout, cg, r = w.shape
    _check_groups(c, kout, cg, groups, "conv1d_direct")
    lo = conv_out_size(length, r, stride, pad, dilation)
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    kg = kout // groups
    y = np.zeros((n, kout, lo))
    for g in range(groups):
        xg = xp[:, g * cg : (g + 1) * cg]  # [N, C/g, Lp]
        wg = w[g * kg : (g + 1) * kg]  # [K/g, C/g, R]
        for t in range(r):
            # Tap t of every output position reads input sample o*s + t*d.
            start = t * dilation
            seg = xg[
                :, :, start : start + stride * (lo - 1) + 1 : stride
            ]  # [N, C/g, Lo]
            y[:, g * kg : (g + 1) * kg] += np.einsum("kc,ncl->nkl", wg[:, :, t], seg)
    if b is not None:
        y += np.asarray(b, dtype=np.float64)[None, :, None]
    return y
    # SOLUTION-END


def conv2d_direct(
    x: NDArray,
    w: NDArray,
    b: NDArray | None,
    stride=(1, 1),
    pad=(0, 0),
    dilation=(1, 1),
    groups=1,
) -> NDArray:
    # SOLUTION-BEGIN M12.1
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x.ndim != 4 or w.ndim != 4:
        raise ValueError(
            f"conv2d_direct: need x NCHW and w KCRS, got {x.shape}, {w.shape}"
        )
    (sh, sw), (ph, pw), (dh, dw) = (
        _pair(stride, "stride"),
        _pair(pad, "pad"),
        _pair(dilation, "dilation"),
    )
    n, c, h, wd = x.shape
    kout, cg, r, s = w.shape
    _check_groups(c, kout, cg, groups, "conv2d_direct")
    p = conv_out_size(h, r, sh, ph, dh)
    q = conv_out_size(wd, s, sw, pw, dw)
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    kg = kout // groups
    y = np.zeros((n, kout, p, q))
    for g in range(groups):
        xg = xp[:, g * cg : (g + 1) * cg]
        wg = w[g * kg : (g + 1) * kg]
        for i in range(r):
            for j in range(s):
                seg = xg[
                    :,
                    :,
                    i * dh : i * dh + sh * (p - 1) + 1 : sh,
                    j * dw : j * dw + sw * (q - 1) + 1 : sw,
                ]  # [N, C/g, P, Q]
                y[:, g * kg : (g + 1) * kg] += np.einsum(
                    "kc,ncpq->nkpq", wg[:, :, i, j], seg
                )
    if b is not None:
        y += np.asarray(b, dtype=np.float64)[None, :, None, None]
    return y
    # SOLUTION-END


def correlate_vs_convolve(x: NDArray, w: NDArray) -> tuple[NDArray, NDArray]:
    # SOLUTION-BEGIN M12.1
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError(
            f"correlate_vs_convolve: need 2-D x and w, got {x.shape}, {w.shape}"
        )
    corr = conv2d_direct(x[None, None], w[None, None], None)[0, 0]
    # True convolution flips the kernel on both axes before sliding it.
    conv = conv2d_direct(x[None, None], w[::-1, ::-1][None, None], None)[0, 0]
    return corr, conv
    # SOLUTION-END


def im2col(
    x: NDArray, kh: int, kw: int, stride=(1, 1), pad=(0, 0), dilation=(1, 1)
) -> NDArray:
    # SOLUTION-BEGIN M12.1
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 4:
        raise ValueError(f"im2col: need x NCHW, got shape {x.shape}")
    (sh, sw), (ph, pw), (dh, dw) = (
        _pair(stride, "stride"),
        _pair(pad, "pad"),
        _pair(dilation, "dilation"),
    )
    n, c, h, w = x.shape
    p = conv_out_size(h, kh, sh, ph, dh)
    q = conv_out_size(w, kw, sw, pw, dw)
    xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    cols = np.empty((n, c, kh, kw, p, q))
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = xp[
                :,
                :,
                i * dh : i * dh + sh * (p - 1) + 1 : sh,
                j * dw : j * dw + sw * (q - 1) + 1 : sw,
            ]
    # Rows ordered (c, r, s), columns (p, q): the layout of F.unfold, and the
    # order in which a KCRS weight flattens to [K, C*R*S].
    return cols.reshape(n, c * kh * kw, p * q)
    # SOLUTION-END


def col2im(
    cols: NDArray,
    x_shape: tuple[int, int, int, int],
    kh: int,
    kw: int,
    stride=(1, 1),
    pad=(0, 0),
    dilation=(1, 1),
) -> NDArray:
    # SOLUTION-BEGIN M12.1
    cols = np.asarray(cols, dtype=np.float64)
    (sh, sw), (ph, pw), (dh, dw) = (
        _pair(stride, "stride"),
        _pair(pad, "pad"),
        _pair(dilation, "dilation"),
    )
    n, c, h, w = x_shape
    p = conv_out_size(h, kh, sh, ph, dh)
    q = conv_out_size(w, kw, sw, pw, dw)
    if cols.shape != (n, c * kh * kw, p * q):
        raise ValueError(
            f"col2im: cols has shape {cols.shape}, expected {(n, c * kh * kw, p * q)}"
        )
    blocks = cols.reshape(n, c, kh, kw, p, q)
    xp = np.zeros((n, c, h + 2 * ph, w + 2 * pw))
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
    return xp[:, :, ph : ph + h, pw : pw + w].copy()
    # SOLUTION-END


def conv_matrix(
    in_shape: tuple[int, int], w: NDArray, stride=(1, 1), pad=(0, 0)
) -> NDArray:
    # SOLUTION-BEGIN M12.1
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"conv_matrix: need a 2-D kernel, got shape {w.shape}")
    (sh, sw), (ph, pw) = _pair(stride, "stride"), _pair(pad, "pad")
    h, wd = in_shape
    r, s = w.shape
    p = conv_out_size(h, r, sh, ph)
    q = conv_out_size(wd, s, sw, pw)
    t = np.zeros((p * q, h * wd))
    for op in range(p):
        for oq in range(q):
            row = op * q + oq
            for i in range(r):
                for j in range(s):
                    a, bcol = op * sh + i - ph, oq * sw + j - pw
                    # Taps that land in the zero padding have no column.
                    if 0 <= a < h and 0 <= bcol < wd:
                        t[row, a * wd + bcol] = w[i, j]
    return t
    # SOLUTION-END
