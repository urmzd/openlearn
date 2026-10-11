"""Course tests for M08.3: closed-form VJPs (tinyllm/autograd/vjp.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M08.3), and the chapter section it comes from.

The worked examples of the chapter (section 3): a 2 x 2 matmul with
G = [[1, 0], [0, 0]]; softmax and cross-entropy of x = [0, ln 3]
(y = [1/4, 3/4]); LayerNorm of [1, 2, 6] and RMSNorm of [3, 4] with
g = e_1.

Every gradient is checked by the frozen float64 gradcheck (never your
M04.1). Two tests call earlier modules through the overlay: M04.2's
numeric VJP and M11.1's cross-entropy as the loss being differentiated.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from tinyllm.autograd.vjp import (
    cross_entropy_vjp,
    layernorm_vjp,
    log_softmax_vjp,
    matmul_vjp,
    rmsnorm_vjp,
    softmax_vjp,
    unbroadcast,
)

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M08.3" / "torch_vjp.npz"


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


# The test's own forward passes (float64), independent of your code.
def softmax_np(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax_np(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


def layernorm_np(x, gamma, beta, eps):
    mu = x.mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps)
    xhat = (x - mu) * rstd
    return gamma * xhat + beta, xhat, rstd


def rmsnorm_np(x, w, eps):
    rstd = 1.0 / np.sqrt((x**2).mean(-1, keepdims=True) + eps)
    return w * x * rstd, rstd


# --- the worked examples -------------------------------------------------------


def test_hand_example():
    # WHY: the chapter's worked examples, number for number. Matmul: L = Y11
    #      = a11 b11 + a12 b21, so dA = G B^T = [[1, 2], [0, 0]] and dB =
    #      A^T G = [[1, 0], [2, 0]]. Softmax of [0, ln 3] is [1/4, 3/4]; with
    #      g = [1, 0], dx = y (g - <g, y>) = [3/16, -3/16]. Cross-entropy at
    #      target 1: softmax - onehot = [1/4, -1/4]. LayerNorm of [1, 2, 6]
    #      with g = e_1: dx = sqrt(3/14) [8, -10, 2] / 21. RMSNorm of [3, 4]
    #      with g = e_1: dx = [0.64, -0.48] / sqrt(12.5).
    # KIND: unit
    # CATCHES: s01, s02, s04, s06, s07, s09, s10
    # CHAPTER: M08.3 section 3, Worked example by hand
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[1.0, 0.0], [2.0, 1.0]])
    G = np.array([[1.0, 0.0], [0.0, 0.0]])
    dA, dB = matmul_vjp(G, A, B)
    assert_close(dA, [[1.0, 2.0], [0.0, 0.0]])
    assert_close(dB, [[1.0, 0.0], [2.0, 0.0]])

    y = np.array([0.25, 0.75])
    assert_close(softmax_vjp(np.array([1.0, 0.0]), y), [3 / 16, -3 / 16])
    assert_close(
        cross_entropy_vjp(np.array([[0.0, math.log(3)]]), np.array([1])),
        [[0.25, -0.25]],
    )

    x = np.array([[1.0, 2.0, 6.0]])
    _, xhat, rstd = layernorm_np(x, 1.0, 0.0, 0.0)
    dx, dgamma, dbeta = layernorm_vjp(
        np.array([[1.0, 0.0, 0.0]]), xhat, rstd, np.ones(3)
    )
    assert_close(
        dx,
        [
            [
                8 / 21 * math.sqrt(3 / 14),
                -10 / 21 * math.sqrt(3 / 14),
                2 / 21 * math.sqrt(3 / 14),
            ]
        ],
    )
    assert_close(dgamma, [-2 * math.sqrt(3 / 14), 0.0, 0.0])
    assert_close(dbeta, [1.0, 0.0, 0.0])

    x = np.array([[3.0, 4.0]])
    _, rstd = rmsnorm_np(x, 1.0, 0.0)
    dx, dw = rmsnorm_vjp(np.array([[1.0, 0.0]]), x, rstd, np.ones(2))
    assert_close(dx, [[0.64 / math.sqrt(12.5), -0.48 / math.sqrt(12.5)]])
    assert_close(dw, [3 / math.sqrt(12.5), 0.0])


def test_matches_torch_golden():
    # WHY: torch.autograd's gradients for every rule (matmul with a broadcast
    #      batch, softmax and log-softmax on two axes, LayerNorm, RMSNorm,
    #      cross-entropy with ignored rows), recorded in float64 by
    #      course/oracle/M08.3/torch_vjp_golden.py. Your rules match the
    #      framework L0.2's ops will be compared against.
    # KIND: golden
    # CATCHES: s02, s03, s04, s05, s06, s07, s08, s09, s10, s11, s12, m01
    # CHAPTER: M08.3 section 2, Principles
    d = np.load(GOLDEN, allow_pickle=False)
    for name in ("matmul", "matmul_batched"):
        dA, dB = matmul_vjp(d[f"{name}/g"], d[f"{name}/A"], d[f"{name}/B"])
        assert_close(dA, d[f"{name}/dA"], msg=f"{name} dA")
        assert_close(dB, d[f"{name}/dB"], msg=f"{name} dB")
    for name in ("softmax_last", "softmax_axis0"):
        axis, g = int(d[f"{name}/axis"]), d[f"{name}/g"]
        assert_close(
            softmax_vjp(g, d[f"{name}/softmax/y"], axis=axis),
            d[f"{name}/softmax/dx"],
            msg=name,
        )
        assert_close(
            log_softmax_vjp(g, d[f"{name}/log_softmax/y"], axis=axis),
            d[f"{name}/log_softmax/dx"],
            msg=name,
        )
    dx, dgamma, dbeta = layernorm_vjp(
        d["layernorm/g"], d["layernorm/xhat"], d["layernorm/rstd"], d["layernorm/gamma"]
    )
    assert_close(dx, d["layernorm/dx"], msg="layernorm dx")
    assert_close(dgamma, d["layernorm/dgamma"], msg="layernorm dgamma")
    assert_close(dbeta, d["layernorm/dbeta"], msg="layernorm dbeta")
    dx, dw = rmsnorm_vjp(
        d["rmsnorm/g"], d["rmsnorm/x"], d["rmsnorm/rstd"], d["rmsnorm/w"]
    )
    assert_close(dx, d["rmsnorm/dx"], msg="rmsnorm dx")
    assert_close(dw, d["rmsnorm/dw"], msg="rmsnorm dw")
    assert_close(
        cross_entropy_vjp(d["cross_entropy/logits"], d["cross_entropy/targets"]),
        d["cross_entropy/dlogits"],
    )


# --- one gradcheck per rule ----------------------------------------------------


@pytest.mark.parametrize(
    "g_shape, shape, want",
    [
        ((2, 3), (2, 3), [[1, 1, 1], [1, 1, 1]]),
        ((2, 3), (3,), [2, 2, 2]),
        ((2, 3), (1, 3), [[2, 2, 2]]),
        ((2, 3), (2, 1), [[3], [3]]),
        ((4, 2, 3), (2, 3), [[4, 4, 4], [4, 4, 4]]),
        ((4, 2, 3), (1, 1), [[24]]),
        ((2, 3), (), 6),
    ],
)
def test_unbroadcast_shapes(g_shape, shape, want):
    # WHY: if x was broadcast from `shape` to `g_shape` in the forward pass,
    #      every copy of an entry got the same upstream gradient, so x's
    #      gradient is their sum: over leading axes, and over size-1 axes
    #      kept as size 1. With g all ones the sum counts the copies.
    # KIND: unit
    # CATCHES: s14, s15
    # CHAPTER: M08.3 section 2, Principles (broadcasting)
    out = unbroadcast(np.ones(g_shape), shape)
    assert out.shape == shape
    assert_close(out, np.array(want, dtype=np.float64))


def test_unbroadcast_rejects_impossible_shapes():
    # WHY: (2, 3) cannot be broadcast from (4,) or (2, 3, 1); a gradient of
    #      the wrong shape is a bug in the op, and summing it into some
    #      shape would hide it.
    # KIND: boundary
    # CATCHES: s18
    # CHAPTER: M08.3 section 4, The interface
    for shape in ((4,), (2, 3, 1), (3, 3)):
        with pytest.raises(ValueError):
            unbroadcast(np.ones((2, 3)), shape)


def test_matmul_vjp_gradcheck():
    # WHY: dA = G B^T and dB = A^T G, derived by the trace trick, against
    #      central differences of L = sum(G * (A @ B)): a plain 2-D product,
    #      and batched products where A or B is broadcast over the batch
    #      (the summed copies come back through unbroadcast).
    # KIND: gradcheck
    # CATCHES: s01, s02, s03
    # CHAPTER: M08.3 section 2, Principles (the trace trick)
    rng = PCG32(seed=seed())
    for sa, sb in (
        ((3, 4), (4, 4)),
        ((3, 4), (4, 2)),
        ((2, 3, 4), (4, 2)),
        ((3, 4), (2, 4, 2)),
        ((2, 1, 2, 3), (4, 3, 2)),
    ):
        A, B = rng.normal_array(sa), rng.normal_array(sb)
        G = rng.normal_array(np.matmul(A, B).shape)
        dA, dB = matmul_vjp(G, A, B)
        gradcheck(
            lambda a, b: float(np.sum(G * (a @ b))), [A, B], [dA, dB], names=["A", "B"]
        )


def test_matmul_vjp_rejects_vectors():
    # WHY: a 1-D operand makes numpy drop an axis in the product, and the
    #      transposes in G B^T no longer mean anything. The contract asks
    #      for 2-D (or batched) operands; reshape vectors to [1, k] or [k, 1].
    # KIND: boundary
    # CHAPTER: M08.3 section 4, The interface
    with pytest.raises(ValueError):
        matmul_vjp(np.ones(2), np.ones(3), np.ones((3, 2)))


def test_softmax_vjps_gradcheck():
    # WHY: the softmax Jacobian is diag(y) - y y^T, so its VJP is
    #      y * (g - <g, y>); log-softmax's is I - 1 y^T, so its VJP is
    #      g - y * sum(g). Checked along the last axis and along axis 0.
    # KIND: gradcheck
    # CATCHES: s04, s05, m01
    # CHAPTER: M08.3 section 2, Principles (softmax)
    rng = PCG32(seed=seed())
    for shape, axis in (((3, 5), -1), ((4, 3), 0)):
        x, g = rng.normal_array(shape, scale=2.0), rng.normal_array(shape)
        y = softmax_np(x, axis)
        gradcheck(
            lambda a: float(np.sum(g * softmax_np(a, axis))),
            [x],
            [softmax_vjp(g, y, axis=axis)],
        )
        ly = log_softmax_np(x, axis)
        gradcheck(
            lambda a: float(np.sum(g * log_softmax_np(a, axis))),
            [x],
            [log_softmax_vjp(g, ly, axis=axis)],
        )


def test_softmax_vjp_matches_numeric_vjp():
    # WHY: M04.2's numeric VJP (central differences on the Jacobian rows)
    #      and the closed form agree for a single softmax vector: the same
    #      row vector u^T J two ways.
    # KIND: differential
    # CATCHES: s04
    # CHAPTER: M08.3 section 2, Principles (softmax)
    from tinyllm.num.jacobian import vjp_numeric

    rng = PCG32(seed=seed())
    x, u = rng.normal_array(6), rng.normal_array(6)
    assert_close(
        softmax_vjp(u, softmax_np(x)),
        vjp_numeric(softmax_np, x, u),
        rtol=1e-6,
        atol=1e-8,
    )


def test_layernorm_vjp_gradcheck():
    # WHY: LayerNorm's mean and variance both depend on every input, so dx
    #      has three terms: d, minus its mean (through mu), minus xhat times
    #      mean(d * xhat) (through the variance). Checked on [2, 3, 8] with
    #      rstd passed both as [2, 3] and as [2, 3, 1].
    # KIND: gradcheck
    # CATCHES: s06, s07, s08, m02
    # CHAPTER: M08.3 section 2, Principles (LayerNorm)
    rng = PCG32(seed=seed())
    x = rng.normal_array((2, 3, 8), scale=1.5)
    gamma, beta, g = (
        rng.normal_array(8),
        rng.normal_array(8),
        rng.normal_array((2, 3, 8)),
    )
    eps = 1e-5
    _, xhat, rstd = layernorm_np(x, gamma, beta, eps)
    for r in (rstd, rstd[..., 0]):
        dx, dgamma, dbeta = layernorm_vjp(g, xhat, r, gamma)
        gradcheck(
            lambda a, gm, bt: float(np.sum(g * layernorm_np(a, gm, bt, eps)[0])),
            [x, gamma, beta],
            [dx, dgamma, dbeta],
            names=["x", "gamma", "beta"],
        )


def test_rmsnorm_vjp_gradcheck():
    # WHY: RMSNorm divides by the root mean square, which depends on every
    #      input: dx = rstd (d - x rstd^2 mean(d x)). Treating rstd as a
    #      constant drops the second term. This is the backward of L7.1.
    # KIND: gradcheck
    # CATCHES: s09, s10, m02
    # CHAPTER: M08.3 section 2, Principles (RMSNorm)
    rng = PCG32(seed=seed())
    x, w, g = (
        rng.normal_array((5, 6), scale=1.5),
        rng.normal_array(6),
        rng.normal_array((5, 6)),
    )
    eps = 1e-6
    _, rstd = rmsnorm_np(x, w, eps)
    for r in (rstd, rstd[..., 0]):
        dx, dw = rmsnorm_vjp(g, x, r, w)
        gradcheck(
            lambda a, ww: float(np.sum(g * rmsnorm_np(a, ww, eps)[0])),
            [x, w],
            [dx, dw],
            names=["x", "w"],
        )


# --- cross-entropy -------------------------------------------------------------


def test_cross_entropy_vjp_gradcheck():
    # WHY: the fused softmax + cross-entropy gradient (softmax - onehot) / n
    #      is the gradient of M11.1's cross-entropy between the one-hot
    #      target and softmax(logits), averaged over the valid rows. The
    #      loss here calls your M11.1 cross_entropy, so the two modules agree
    #      on what is being differentiated.
    # KIND: gradcheck
    # CATCHES: s11, s12
    # CHAPTER: M08.3 section 2, Principles (cross-entropy)
    from tinyllm.info.entropy import cross_entropy

    rng = PCG32(seed=seed())
    z = rng.normal_array((6, 4), scale=2.0)
    t = np.array([3, -100, 0, 1, -100, 2])
    valid = t != -100
    onehot = np.zeros_like(z)
    onehot[np.nonzero(valid)[0], t[valid]] = 1.0

    def loss(a):
        return float(
            np.sum(cross_entropy(onehot[valid], softmax_np(a)[valid], axis=-1))
            / valid.sum()
        )

    gradcheck(loss, [z], [cross_entropy_vjp(z, t)])


def test_cross_entropy_ignore_index():
    # WHY: padded positions carry ignore_index (-100, as in PyTorch). They get
    #      exactly zero gradient, the mean divides by the number of valid
    #      rows only, and a batch with no valid row gives zeros, not 0 / 0.
    #      A target outside [0, V) that is not -100 is a data bug: numpy would
    #      wrap -1 to the last class silently.
    # KIND: boundary
    # CATCHES: s11, s12, s13
    # CHAPTER: M08.3 section 5, Pitfalls, item 4
    z = np.array([[0.0, math.log(3)], [5.0, 1.0], [2.0, 2.0]])
    out = cross_entropy_vjp(z, np.array([1, -100, 0]))
    assert out[1].tolist() == [0.0, 0.0]
    assert_close(out[0], [0.125, -0.125])
    assert_close(out[2], [-0.25, 0.25])
    assert (
        cross_entropy_vjp(z, np.array([-100, -100, -100])).tolist() == [[0.0, 0.0]] * 3
    )
    assert (
        cross_entropy_vjp(z, np.array([7, 7, 7]), ignore_index=7).tolist()
        == [[0.0, 0.0]] * 3
    )
    for bad in ([1, 2, 0], [1, -1, 0]):
        with pytest.raises(ValueError):
            cross_entropy_vjp(z, np.array(bad))


def test_cross_entropy_large_logits():
    # WHY: logits near 1e4 overflow exp; the gradient still exists and is
    #      softmax - onehot computed with M09.2's stable softmax. A model
    #      that is confidently right gets a gradient near 0, not nan.
    # KIND: boundary
    # CATCHES: s16
    # CHAPTER: M08.3 section 5, Pitfalls, item 5
    z = np.array([[1e4, 0.0, -1e4], [1e4, 1e4, 0.0]])
    out = cross_entropy_vjp(z, np.array([0, 1]))
    assert np.isfinite(out).all()
    assert_close(out, [[0.0, 0.0, 0.0], [0.25, -0.25, 0.0]])


# --- laws --------------------------------------------------------------------


def test_trace_identity():
    # WHY: the trace trick in one line: for Y = A B and any directions dA, dB,
    #      tr(G^T dY) with dY = dA B + A dB equals <dA_bar, dA> + <dB_bar, dB>.
    #      A VJP is exactly the linear map that makes this hold for every
    #      direction, so it is checked on 20 random ones.
    # KIND: property
    # CATCHES: s02
    # CHAPTER: M08.3 section 2, Principles (the trace trick)
    rng = PCG32(seed=seed())
    A, B = rng.normal_array((3, 5)), rng.normal_array((5, 2))
    G = rng.normal_array((3, 2))
    Abar, Bbar = matmul_vjp(G, A, B)
    for _ in range(20):
        dA, dB = rng.normal_array((3, 5)), rng.normal_array((5, 2))
        lhs = float(np.trace(G.T @ (dA @ B + A @ dB)))
        rhs = float(np.sum(Abar * dA) + np.sum(Bbar * dB))
        assert_close(lhs, rhs)


def test_vjps_do_not_mutate_inputs():
    # WHY: L0.2 hands the saved forward values to the VJP and may reuse them
    #      (the same logits feed the loss and the sampler); an in-place
    #      update such as subtracting the onehot from a softmax buffer you
    #      were given corrupts them.
    # KIND: unit
    # CATCHES: s17
    # CHAPTER: M08.3 section 4, The interface
    rng = PCG32(seed=seed())
    z = rng.normal_array((4, 3))
    t = np.array([0, 2, -100, 1])
    y = softmax_np(z)
    g = rng.normal_array((4, 3))
    keep = [a.copy() for a in (z, t, y, g)]
    cross_entropy_vjp(z, t)
    softmax_vjp(g, y)
    log_softmax_vjp(g, np.log(y))
    for a, b in zip((z, t, y, g), keep):
        assert np.array_equal(a, b)
