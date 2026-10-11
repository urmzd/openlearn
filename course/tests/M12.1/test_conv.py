"""Course tests for M12.1: conv_out_size, conv1d_direct, conv2d_direct,
correlate_vs_convolve, im2col, col2im, and conv_matrix (tinyllm/sig/conv.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M12.1), and the chapter section it comes from.

The chapter's worked example (section 3) slides the 2 x 2 kernel
w = [[1, 0], [0, -1]] over the 4 x 4 image below: the cross-correlation is
x[i, j] - x[i+1, j+1], and the true convolution (kernel flipped) is its
negative. The torch golden values in $TINYLLM_FIXTURES/M12.1/conv_torch.npz
come from course/oracle/M12.1/conv_torch.py (F.conv1d, F.conv2d, F.unfold,
F.fold, and autograd, in float64).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.sig.conv import (
    col2im,
    conv1d_direct,
    conv2d_direct,
    conv_matrix,
    conv_out_size,
    correlate_vs_convolve,
    im2col,
)

X_HAND = np.array(
    [
        [1.0, 2.0, 0.0, 1.0],
        [0.0, 1.0, 3.0, 1.0],
        [2.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 2.0],
    ]
)
W_HAND = np.array([[1.0, 0.0], [0.0, -1.0]])
GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M12.1" / "conv_torch.npz"
CONV2D = ["conv2d.plain", "conv2d.strided", "conv2d.depthwise", "conv2d.patch"]
CONV1D = ["conv1d.plain", "conv1d.dilated", "conv1d.stem"]


def rng(seq: int) -> PCG32:
    return PCG32(int(os.environ.get("SS_SEED", "0")), seq=seq)


def golden():
    return np.load(GOLDEN)


def params2d(z, name):
    p = [int(v) for v in z[f"{name}.params"]]
    return (p[0], p[1]), (p[2], p[3]), (p[4], p[5]), p[6]


def test_hand_example_correlation():
    # WHY: section 3's worked example. Deep learning's "convolution" does
    #      not flip the kernel: y[i, j] = x[i, j] - x[i+1, j+1]. Flipping it
    #      gives the true convolution, here exactly the negative.
    # KIND: unit, smoke
    # CATCHES: s01, m01
    # CHAPTER: M12.1 section 3
    corr = np.array([[0.0, -1.0, -1.0], [-1.0, 1.0, 3.0], [2.0, 0.0, -2.0]])
    got_corr, got_conv = correlate_vs_convolve(X_HAND, W_HAND)
    assert_close(got_corr, corr, rtol=0.0, atol=0.0)
    assert_close(got_conv, -corr, rtol=0.0, atol=0.0)
    y = conv2d_direct(X_HAND[None, None], W_HAND[None, None], None)
    assert_close(y[0, 0], corr, rtol=0.0, atol=0.0)


def test_conv_out_size_table():
    # WHY: every layer in L13 and L14 sizes its output with this formula.
    #      Whisper's stem keeps 3000 frames (k 3, pad 1) then halves them
    #      (stride 2); a ViT-B/16 patch conv turns 224 into 14; a dilated
    #      kernel spans d (k - 1) + 1 samples; an exact fit gives one output.
    # KIND: unit, smoke
    # CATCHES: s02, m01, m02
    # CHAPTER: M12.1 section 2.2
    table = [
        ((3000, 3, 1, 1, 1), 3000),
        ((3000, 3, 2, 1, 1), 1500),
        ((224, 16, 16, 0, 1), 14),
        ((10, 3, 1, 0, 2), 6),
        ((7, 3, 2, 1, 1), 4),
        ((5, 5, 1, 0, 1), 1),
        ((9, 3, 3, 0, 4), 1),
    ]
    for args, want in table:
        assert conv_out_size(*args) == want, args
    for bad in (
        (4, 5, 1, 0, 1),
        (5, 3, 0, 0, 1),
        (5, 3, 1, -1, 1),
        (5, 3, 1, 0, 0),
        (5, 3, 1, 0, 3),
    ):
        with pytest.raises(ValueError):
            conv_out_size(*bad)


@pytest.mark.parametrize("name", CONV2D)
def test_conv2d_matches_torch(name):
    # WHY: L13.1's Conv2d must agree with PyTorch on stride, padding,
    #      dilation, and groups, including a depthwise conv (groups = C) and
    #      a ViT patch embedding (kernel = stride).
    # KIND: golden
    # CATCHES: s01, s02, s05, s06, m01
    # CHAPTER: M12.1 section 2.2
    z = golden()
    stride, pad, dil, groups = params2d(z, name)
    b = z[f"{name}.b"] if f"{name}.b" in z.files else None
    y = conv2d_direct(z[f"{name}.x"], z[f"{name}.w"], b, stride, pad, dil, groups)
    assert_close(y, z[f"{name}.y"], rtol=1e-12, atol=1e-12, msg=name)


@pytest.mark.parametrize("name", CONV1D)
def test_conv1d_matches_torch(name):
    # WHY: Whisper's stem (L14.2) is two conv1d layers; the dilated grouped
    #      case pins the tap index o * stride + r * dilation - pad.
    # KIND: golden
    # CATCHES: s02, m01
    # CHAPTER: M12.1 section 2.2
    z = golden()
    s, p, d, g = (int(v) for v in z[f"{name}.params"])
    b = z[f"{name}.b"] if f"{name}.b" in z.files else None
    y = conv1d_direct(z[f"{name}.x"], z[f"{name}.w"], b, s, p, d, g)
    assert_close(y, z[f"{name}.y"], rtol=1e-12, atol=1e-12, msg=name)


@pytest.mark.parametrize("name", ["unfold.a", "unfold.b"])
def test_im2col_col2im_match_unfold_fold(name):
    # WHY: im2col must lay rows out in (c, r, s) order, the order a KCRS
    #      weight flattens in, or the matmul pairs weights with the wrong
    #      pixels; col2im must SUM overlapping windows, as F.fold does.
    # KIND: golden
    # CATCHES: s02, s03, s04, m01
    # CHAPTER: M12.1 section 2.4
    z = golden()
    kh, kw, sh, sw, ph, pw, dh, dw = (int(v) for v in z[f"{name}.params"])
    x = z[f"{name}.x"]
    cols = im2col(x, kh, kw, (sh, sw), (ph, pw), (dh, dw))
    assert_close(cols, z[f"{name}.cols"], rtol=0.0, atol=0.0, msg=name)
    back = col2im(z[f"{name}.c"], x.shape, kh, kw, (sh, sw), (ph, pw), (dh, dw))
    assert_close(back, z[f"{name}.folded"], rtol=1e-13, atol=1e-13, msg=name)


def test_conv_is_one_matmul():
    # WHY: the three views of one linear map agree: the direct loop, im2col
    #      followed by a matmul with the flattened weights (per group), and
    #      the doubly block Toeplitz matrix times the flattened image. This
    #      is how L13.1 computes Conv2d.
    # KIND: differential
    # CATCHES: s01, s02, s04, s05, s06, s07
    # CHAPTER: M12.1 section 2.3
    g = rng(11)
    x = g.normal_array((2, 4, 7, 6))
    w = g.normal_array((6, 2, 3, 2))
    stride, pad, dil, groups = (2, 1), (1, 0), (1, 2), 2
    y = conv2d_direct(x, w, None, stride, pad, dil, groups)
    cols = (
        im2col(x[:, :2], 3, 2, stride, pad, dil),
        im2col(x[:, 2:], 3, 2, stride, pad, dil),
    )
    parts = [w[3 * i : 3 * i + 3].reshape(3, -1) @ cols[i] for i in range(2)]
    via = np.concatenate(parts, axis=1).reshape(y.shape)
    assert_close(via, y, rtol=1e-12, atol=1e-12)
    img, k = g.normal_array((5, 6)), g.normal_array((3, 2))
    for s, p in (((1, 1), (0, 0)), ((2, 1), (1, 1)), ((1, 2), (2, 0))):
        t = conv_matrix(img.shape, k, s, p)
        direct = conv2d_direct(img[None, None], k[None, None], None, s, p)
        assert t.shape == (direct.size, img.size)
        assert_close(
            t @ img.ravel(), direct.ravel(), rtol=1e-12, atol=1e-12, msg=f"{s} {p}"
        )


@pytest.mark.parametrize("name", ["conv2d.plain", "conv2d.patch", "conv1d.stem"])
def test_backward_is_a_conv(name):
    # WHY: L13.1's backward is built from these two functions. With
    #      Y = W_flat @ im2col(x): dW = g_flat @ im2col(x)^T, dx =
    #      col2im(W_flat^T @ g_flat), db = g summed over batch and positions.
    #      They must equal PyTorch's autograd gradients.
    # KIND: golden
    # CATCHES: s03, s04, m01, m02
    # CHAPTER: M12.1 section 2.5
    z = golden()
    x, w, g = z[f"{name}.x"], z[f"{name}.w"], z[f"{name}.g"]
    if x.ndim == 3:  # a conv1d is a conv2d with one row
        x, w, g = x[:, :, None, :], w[:, :, None, :], g[:, :, None, :]
        s, p, d, _ = (int(v) for v in z[f"{name}.params"])
        stride, pad, dil = (1, s), (0, p), (1, d)
    else:
        stride, pad, dil, _ = params2d(z, name)
    k, _, kh, kw = w.shape
    n = x.shape[0]
    cols = im2col(x, kh, kw, stride, pad, dil)  # [N, CRS, PQ]
    gf = g.reshape(n, k, -1)  # [N, K, PQ]
    dw = np.einsum("nkp,ncp->kc", gf, cols).reshape(w.shape)
    dx = col2im(
        np.einsum("kc,nkp->ncp", w.reshape(k, -1), gf),
        x.shape,
        kh,
        kw,
        stride,
        pad,
        dil,
    )
    want_dx, want_dw = z[f"{name}.dx"], z[f"{name}.dw"]
    assert_close(dx.reshape(want_dx.shape), want_dx, rtol=1e-12, atol=1e-12, msg=name)
    assert_close(dw.reshape(want_dw.shape), want_dw, rtol=1e-12, atol=1e-12, msg=name)
    assert_close(g.sum(axis=(0, 2, 3)), z[f"{name}.db"], rtol=1e-12, atol=1e-12)


def test_col2im_is_the_adjoint_of_im2col():
    # WHY: <im2col(x), y> == <x, col2im(y)> for every x and y is what makes
    #      col2im the right backward for im2col. Overlapping windows
    #      (stride < kernel) are where an assigning col2im loses mass.
    # KIND: property
    # CATCHES: s02, s03, s04
    # CHAPTER: M12.1 section 2.4
    g = rng(12)
    for kh, kw, stride, pad, dil in (
        (3, 3, (1, 1), (1, 1), (1, 1)),
        (2, 3, (2, 1), (0, 2), (1, 1)),
        (3, 2, (1, 2), (2, 1), (2, 1)),
    ):
        x = g.normal_array((2, 3, 7, 8))
        cols = im2col(x, kh, kw, stride, pad, dil)
        y = g.normal_array(cols.shape)
        lhs = float(np.sum(cols * y))
        rhs = float(np.sum(x * col2im(y, x.shape, kh, kw, stride, pad, dil)))
        assert_close(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_flip_turns_correlation_into_convolution():
    # WHY: correlating with a flipped kernel is convolving. On one row the
    #      true convolution is numpy.convolve's 'valid' mode, an independent
    #      definition with the flip built in.
    # KIND: property
    # CATCHES: s01, m01, m02
    # CHAPTER: M12.1 section 2.1
    g = rng(13)
    x, w = g.normal_array((5, 9)), g.normal_array((2, 3))
    corr, conv = correlate_vs_convolve(x, w)
    assert_close(conv, correlate_vs_convolve(x, w[::-1, ::-1])[0], rtol=0.0, atol=1e-14)
    row, k = g.normal_array((1, 11)), g.normal_array((1, 4))
    _, conv1 = correlate_vs_convolve(row, k)
    assert_close(
        conv1[0], np.convolve(row[0], k[0], mode="valid"), rtol=1e-13, atol=1e-13
    )
    assert corr.shape == (4, 7)


def test_rejects_shapes_that_do_not_fit():
    # WHY: a wrong channel count or a group count that does not divide the
    #      channels is a model-definition bug; it must fail loudly, not
    #      broadcast into a plausible tensor.
    # KIND: boundary
    # CATCHES: m03
    # CHAPTER: M12.1 section 4
    g = rng(14)
    x = g.normal_array((1, 4, 6, 6))
    with pytest.raises(ValueError):
        conv2d_direct(x, g.normal_array((6, 4, 3, 3)), None, groups=4)
    with pytest.raises(ValueError):
        conv2d_direct(x, g.normal_array((6, 3, 3, 3)), None)
    with pytest.raises(ValueError):
        conv2d_direct(x, g.normal_array((2, 4, 7, 7)), None)
    with pytest.raises(ValueError):
        conv1d_direct(
            g.normal_array((1, 4, 6)), g.normal_array((3, 2, 3)), None, groups=2
        )
    with pytest.raises(ValueError):
        col2im(np.zeros((1, 36, 15)), (1, 4, 6, 6), 3, 3)
    before = x.copy()
    conv2d_direct(x, g.normal_array((2, 4, 3, 3)), np.ones(2), 1, 1)
    im2col(x, 3, 3)
    assert (x == before).all()
