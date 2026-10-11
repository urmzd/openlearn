"""Course tests for L5.1: scaled dot-product attention, forward and backward
(tinyllm/xfmr/sdpa.py).

Rung R0 for these (your own graded tests are rung R5, see the chapter): read
them before you write code. Each test names why it exists (WHY), what kind of
check it is (KIND), the planted bugs it kills (CATCHES, mutants in
course/mutants/L5.1), and the chapter section it comes from.

The worked example of the chapter (section 3), d = 4 so the scale is 1/2:
q = (2, 0, 0, 0); keys (1, 0, 0, 0), (0, 1, 0, 0), (2, 0, 0, 0); values
(1, 0), (0, 1), (1, 1). Scores (1, 0, 2), weights (e, 1, e^2) / (1 + e + e^2)
= (0.244728, 0.090031, 0.665241), output (0.909969, 0.755272). With
dO = (1, 0): dV rows (0.244728, 0), (0.090031, 0), (0.665241, 0); dP =
(1, 0, 1); rowsum(dP * P) = 0.909969 = dO . O; dS = (0.022033, -0.081925,
0.059892); dQ = (0.070909, -0.040963, 0, 0); dK rows (dS_j, 0, 0, 0).

Masks are plain bool arrays here (True = may attend), so this module needs
none of L5.2's code. The golden fixture (course/fixtures/L5.1/sdpa_torch.npz)
holds torch 2.14.1's scaled_dot_product_attention outputs, weights, and
float64 gradients, from course/oracle/L5.1/sdpa_torch.py.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _lib.close import assert_close, assert_close_bounded
from _lib.gradcheck import gradcheck
from _lib.pcg32 import PCG32
from tinyllm.autograd.tensor import Tensor
from tinyllm.xfmr.sdpa import scaled_dot_product_attention, sdpa_backward, sdpa_forward

FIX = os.path.join(os.environ.get("TINYLLM_FIXTURES", ""), "L5.1", "sdpa_torch.npz")
SEED = int(os.environ.get("OL_SEED", "0"))
CASES = ("plain", "causal", "cross", "scaled", "three_d")

HQ = np.array([[2.0, 0.0, 0.0, 0.0]])
HK = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
HV = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


class Rng:
    """The frozen PCG32 behind the uniforms() of M06.3, counting draws."""

    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)
        self.drawn = 0

    def uniform(self) -> float:
        self.drawn += 1
        return self.g.uniform()

    def uniforms(self, n: int) -> np.ndarray:
        return np.array([self.uniform() for _ in range(n)], dtype=np.float64)


def leaf(a, dtype=np.float64):
    return Tensor(np.asarray(a), requires_grad=True, dtype=dtype)


def golden_case(f, name):
    c = {k[len(name) + 1 :]: f[k] for k in f.files if k.startswith(name + ".")}
    sc = float(c["scale"])
    c["scale"] = None if math.isnan(sc) else sc
    if name in ("causal", "three_d"):
        T = c["q"].shape[-2]
        c["mask"] = np.tril(np.ones((T, T), dtype=bool))
    c.setdefault("mask", None)
    return c


def rand(g: PCG32, *shape):
    return g.normal_array(shape)


# --- the worked example ----------------------------------------------------------------------


def test_hand_example_forward():
    # WHY: section 3, number for number: scores q.k / sqrt(4) = (1, 0, 2),
    #      softmax over the KEYS (0.244728, 0.090031, 0.665241), and the
    #      weighted average of the values (0.909969, 0.755272).
    # KIND: unit, smoke
    # CATCHES: s01, s02, m01
    # CHAPTER: L5.1 section 3, Worked example by hand
    e = math.e
    z = 1 + e + e * e
    out, p = sdpa_forward(HQ, HK, HV)
    assert_close(p, [[e / z, 1 / z, e * e / z]], rtol=1e-14, atol=1e-15)
    assert_close(out, [[(e + e * e) / z, (1 + e * e) / z]], rtol=1e-14, atol=1e-15)


def test_hand_example_backward():
    # WHY: section 3's backward with dO = (1, 0): dV = P^T dO, dP = dO V^T =
    #      (1, 0, 1), rowsum(dP * P) = dO . O = 0.909969, dS = P * (dP - that)
    #      = (0.022033, -0.081925, 0.059892), dQ = dS K / 2, dK = dS^T Q / 2.
    # KIND: unit, smoke
    # CATCHES: s04, s05, s06, m02
    # CHAPTER: L5.1 section 3, Worked example by hand
    out, p = sdpa_forward(HQ, HK, HV)
    dq, dk, dv = sdpa_backward(HQ, HK, HV, p, np.array([[1.0, 0.0]]))
    D = float(out[0, 0])
    ds = p[0] * (np.array([1.0, 0.0, 1.0]) - D)
    assert_close(ds, [0.02203304, -0.08192507, 0.05989202], rtol=1e-6, atol=1e-9)
    assert_close(
        dv, [[p[0, 0], 0.0], [p[0, 1], 0.0], [p[0, 2], 0.0]], rtol=1e-14, atol=1e-15
    )
    assert_close(
        dq, [[0.5 * (ds[0] + 2 * ds[2]), 0.5 * ds[1], 0.0, 0.0]], rtol=1e-12, atol=1e-15
    )
    assert_close(
        dk,
        [[ds[0], 0, 0, 0], [ds[1], 0, 0, 0], [ds[2], 0, 0, 0]],
        rtol=1e-12,
        atol=1e-15,
    )


# --- against torch ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", CASES)
def test_golden_torch(name):
    # WHY: torch's scaled_dot_product_attention on the same float64 inputs:
    #      self and cross attention (Tq != Tk, dv != d), causal, key padding,
    #      a custom scale with a random mask, and a 3-D input. The output,
    #      the weights, and the gradients of sum(out * g) must agree to float64
    #      rounding, through the Tensor op and its backward.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s05, s06, m01, m02
    # CHAPTER: L5.1 section 2.3, The backward pass
    c = golden_case(np.load(FIX), name)
    q, k, v = leaf(c["q"]), leaf(c["k"]), leaf(c["v"])
    out, w = scaled_dot_product_attention(q, k, v, mask=c["mask"], scale=c["scale"])
    assert_close(out.data, c["out"], rtol=1e-10, atol=1e-12, msg="out")
    assert_close(w.data, c["weights"], rtol=1e-10, atol=1e-12, msg="weights")
    out.backward(c["g"])
    for name_, t in (("q", q), ("k", k), ("v", v)):
        assert_close(
            t.grad, c[f"grad.{name_}"], rtol=1e-9, atol=1e-11, msg=f"grad {name_}"
        )


def test_golden_torch_float32():
    # WHY: models run in float32: inputs cast to float32 must give float32
    #      outputs and gradients within the float32 bound for a reduction over
    #      the keys (the 5.11 tolerance scaled by sqrt(Tk)).
    # KIND: golden
    # CATCHES: s01, s02, m04
    # CHAPTER: L5.1 section 4, The interface
    f = np.load(FIX)
    for name in CASES:
        c = golden_case(f, name)
        q, k, v = (leaf(c[x], np.float32) for x in ("q", "k", "v"))
        out, w = scaled_dot_product_attention(q, k, v, mask=c["mask"], scale=c["scale"])
        assert out.dtype == np.float32 and w.dtype == np.float32
        Tk = c["k"].shape[-2]
        assert_close_bounded(out.data, c["out"], k=Tk * c["q"].shape[-1], msg=name)
        out.backward(c["g"].astype(np.float32))
        assert q.grad.dtype == np.float32
        assert_close_bounded(q.grad, c["grad.q"], k=Tk * c["q"].shape[-1] * 4, msg=name)


# --- the gradient --------------------------------------------------------------------------------


def test_gradcheck_float64():
    # WHY: the hand-written backward against central differences (the frozen
    #      check, D35) on random inputs with a causal mask and with a key
    #      padding mask: every element of dQ, dK, dV.
    # KIND: gradcheck
    # CATCHES: s04, s06, m02
    # CHAPTER: L5.1 section 2.3, The backward pass
    g = PCG32(seed=SEED)
    for mask in (
        np.tril(np.ones((4, 4), dtype=bool)),
        np.array([[True, True, True, False]]),
    ):
        q, k, v = rand(g, 2, 4, 3), rand(g, 2, 4, 3), rand(g, 2, 4, 2)
        w = rand(g, 2, 4, 2)

        def f(q_, k_, v_):
            return float(np.sum(sdpa_forward(q_, k_, v_, mask)[0] * w))

        tq, tk, tv = leaf(q), leaf(k), leaf(v)
        out, _ = scaled_dot_product_attention(tq, tk, tv, mask=mask)
        out.backward(w)
        gradcheck(f, [q, k, v], [tq.grad, tk.grad, tv.grad], names=["q", "k", "v"])


def test_gradcheck_through_dropout():
    # WHY: with attention dropout the forward uses P * keep / (1 - p); the
    #      backward must use the SAME mask and scale for dV and dP. Replaying
    #      the seed makes the dropped set fixed, so central differences apply.
    # KIND: gradcheck
    # CATCHES: s07, s08
    # CHAPTER: L5.1 section 2.5, Attention dropout
    g = PCG32(seed=SEED + 1)
    q, k, v = rand(g, 1, 3, 4), rand(g, 1, 5, 4), rand(g, 1, 5, 3)
    w = rand(g, 1, 3, 3)

    def f(q_, k_, v_):
        out, _ = scaled_dot_product_attention(
            Tensor(q_, dtype=np.float64),
            Tensor(k_, dtype=np.float64),
            Tensor(v_, dtype=np.float64),
            dropout_p=0.4,
            rng=Rng(9),
        )
        return float(np.sum(out.data * w))

    tq, tk, tv = leaf(q), leaf(k), leaf(v)
    out, _ = scaled_dot_product_attention(tq, tk, tv, dropout_p=0.4, rng=Rng(9))
    out.backward(w)
    gradcheck(f, [q, k, v], [tq.grad, tk.grad, tv.grad], names=["q", "k", "v"])


# --- properties ------------------------------------------------------------------------------------


def test_key_permutation_equivariance():
    # WHY: attention is a weighted average over a SET of keys: permuting the
    #      keys together with their values (and the mask's columns) leaves the
    #      output unchanged and permutes the weights' columns. Position
    #      information must come from elsewhere (L5.4).
    # KIND: property, smoke
    # CHAPTER: L5.1 section 2.1, Attention as a soft lookup
    g = PCG32(seed=SEED + 2)
    for _ in range(5):
        q, k, v = rand(g, 2, 3, 4), rand(g, 2, 6, 4), rand(g, 2, 6, 5)
        mask = np.array([[g.uniform() < 0.7 for _ in range(6)] for _ in range(3)])
        mask[:, 0] = True
        perm = list(np.argsort([g.uniform() for _ in range(6)], kind="stable"))
        o1, p1 = sdpa_forward(q, k, v, mask)
        o2, p2 = sdpa_forward(q, k[:, perm], v[:, perm], mask[:, perm])
        assert_close(o2, o1, rtol=1e-12, atol=1e-14)
        assert_close(p2, p1[..., perm], rtol=1e-12, atol=1e-15)


def test_queries_are_independent():
    # WHY: each query's output depends only on that query: permuting the
    #      queries permutes the outputs, and computing one batch item or one
    #      row alone gives the same numbers (what lets L8.2 decode one query
    #      at a time against cached keys).
    # KIND: property
    # CATCHES: s02
    # CHAPTER: L5.1 section 2.1, Attention as a soft lookup
    g = PCG32(seed=SEED + 3)
    q, k, v = rand(g, 2, 2, 5, 4), rand(g, 2, 2, 7, 4), rand(g, 2, 2, 7, 3)
    full, _ = sdpa_forward(q, k, v)
    perm = list(np.argsort([g.uniform() for _ in range(5)], kind="stable"))
    o2, _ = sdpa_forward(q[..., perm, :], k, v)
    assert_close(o2, full[..., perm, :], rtol=1e-12, atol=1e-14)
    one, _ = sdpa_forward(q[1, 0, 3:4], k[1, 0], v[1, 0])
    assert_close(one, full[1, 0, 3:4], rtol=1e-12, atol=1e-14)


def test_masked_keys_get_no_weight_or_gradient():
    # WHY: a blocked key gets weight exactly 0 and, through P = 0 in dS and
    #      dV, exactly zero gradient: padding never learns from the loss. The
    #      open weights of each row still sum to 1.
    # KIND: boundary
    # CATCHES: s03
    # CHAPTER: L5.1 section 2.2, Masking
    g = PCG32(seed=SEED + 4)
    q, k, v = leaf(rand(g, 1, 3, 4)), leaf(rand(g, 1, 5, 4)), leaf(rand(g, 1, 5, 2))
    mask = np.array([True, True, True, False, False])
    out, w = scaled_dot_product_attention(q, k, v, mask=mask)
    assert np.all(w.data[..., 3:] == 0.0)
    assert_close(w.data.sum(-1), np.ones((1, 3)), rtol=1e-14, atol=1e-15)
    out.backward(rand(g, 1, 3, 2))
    assert np.all(k.grad[:, 3:] == 0.0) and np.all(v.grad[:, 3:] == 0.0)
    assert np.any(k.grad[:, :3] != 0.0)


def test_fully_masked_row_gives_zeros():
    # WHY: a query with every key blocked (a padded query of an empty
    #      sequence) must get zero weights, a zero output, and zero gradients,
    #      never NaN: the stable softmax (M09.2) defines that row as zeros.
    # KIND: boundary
    # CATCHES: s10
    # CHAPTER: L5.1 section 2.2, Masking
    g = PCG32(seed=SEED + 5)
    q, k, v = leaf(rand(g, 2, 4)), leaf(rand(g, 3, 4)), leaf(rand(g, 3, 2))
    mask = np.array([[True, False, True], [False, False, False]])
    out, w = scaled_dot_product_attention(q, k, v, mask=mask)
    assert (
        np.all(np.isfinite(out.data))
        and np.all(out.data[1] == 0.0)
        and np.all(w.data[1] == 0.0)
    )
    out.backward(np.ones((2, 2)))
    assert np.all(np.isfinite(q.grad)) and np.all(q.grad[1] == 0.0)


def test_large_scores_do_not_overflow():
    # WHY: scores of 1e4 overflow exp in float32 and float64 alike; the
    #      softmax subtracts each row's maximum (M09.2), so attention to the
    #      largest-scoring key is a clean one-hot, never inf / inf.
    # KIND: boundary
    # CATCHES: s02
    # CHAPTER: L5.1 section 5, Pitfalls
    q = np.array([[1e4, 0.0]], dtype=np.float32)
    k = np.array([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32)
    v = np.array([[3.0], [7.0]], dtype=np.float32)
    out, p = sdpa_forward(q, k, v)
    assert p.dtype == np.float32
    assert p.tolist() == [[1.0, 0.0]] and out.tolist() == [[3.0]]


def test_default_scale_is_inverse_sqrt_d():
    # WHY: the default scale is 1 / sqrt(d) (d = the query width), which keeps
    #      the scores' variance at 1 for unit-variance q and k; passing it
    #      explicitly changes nothing, and another scale changes the weights.
    # KIND: unit
    # CATCHES: s01, m01
    # CHAPTER: L5.1 section 2.1, Attention as a soft lookup
    g = PCG32(seed=SEED + 6)
    q, k, v = rand(g, 3, 9), rand(g, 4, 9), rand(g, 4, 2)
    o1, p1 = sdpa_forward(q, k, v)
    o2, p2 = sdpa_forward(q, k, v, scale=1.0 / 3.0)
    assert np.array_equal(p1, p2) and np.array_equal(o1, o2)
    _, p3 = sdpa_forward(q, k, v, scale=1.0)
    assert np.max(np.abs(p3 - p1)) > 1e-3


def test_score_variance_is_one():
    # WHY: section 2.1's argument, measured: with unit-normal q and k of width
    #      d, q . k has variance d; times 1 / sqrt(d) it has variance 1 for
    #      every d, so the softmax neither saturates nor flattens as heads
    #      grow. Recovered from the weights: log P0 - log P1 is the scaled
    #      score difference, whose variance is 2 for fresh keys per query.
    # KIND: statistical
    # CATCHES: s01
    # CHAPTER: L5.1 section 2.1, Attention as a soft lookup
    g = PCG32(seed=SEED + 7)
    for d in (4, 64):
        n = 1500
        q, k = rand(g, n, 1, d), rand(g, n, 2, d)
        _, p = sdpa_forward(q, k, np.zeros((n, 2, 1)))
        diff = np.log(p[:, 0, 0]) - np.log(p[:, 0, 1])  # s0 - s1: variance 1 + 1
        assert 1.6 < np.var(diff) < 2.4, (d, np.var(diff))


def test_dropout_draws_and_scaling():
    # WHY: attention dropout draws exactly one uniform per weight (C order),
    #      keeps u >= p, and scales the kept weights by 1 / (1 - p); p = 0
    #      draws nothing (eval mode), p = 1 gives zeros. The returned weights
    #      are P before dropout.
    # KIND: unit
    # CATCHES: s09
    # CHAPTER: L5.1 section 2.5, Attention dropout
    g = PCG32(seed=SEED + 8)
    q, k, v = rand(g, 2, 3, 4), rand(g, 2, 5, 4), rand(g, 2, 5, 3)
    r = Rng(4)
    out, w = scaled_dot_product_attention(
        Tensor(q, dtype=np.float64),
        Tensor(k, dtype=np.float64),
        Tensor(v, dtype=np.float64),
        dropout_p=0.25,
        rng=r,
    )
    assert r.drawn == 2 * 3 * 5
    p = sdpa_forward(q, k, v)[1]
    assert_close(w.data, p, rtol=1e-14, atol=1e-16)
    u = Rng(4).uniforms(30).reshape(2, 3, 5)
    want = (p * (u >= 0.25) / 0.75) @ v
    assert_close(out.data, want, rtol=1e-13, atol=1e-15)
    r0 = Rng(4)
    o0, _ = scaled_dot_product_attention(
        Tensor(q, dtype=np.float64),
        Tensor(k, dtype=np.float64),
        Tensor(v, dtype=np.float64),
        dropout_p=0.0,
        rng=r0,
    )
    assert r0.drawn == 0 and np.array_equal(o0.data, sdpa_forward(q, k, v)[0])
    o1, _ = scaled_dot_product_attention(
        Tensor(q, dtype=np.float64),
        Tensor(k, dtype=np.float64),
        Tensor(v, dtype=np.float64),
        dropout_p=1.0,
        rng=Rng(4),
    )
    assert np.all(o1.data == 0.0)


def test_weights_are_a_constant():
    # WHY: out is one node of the graph with q, k, v as parents; the weights
    #      come back for inspection only (heat maps, tests, the KV cache), as a
    #      constant: no gradient flows through them. Constant inputs give a
    #      constant output.
    # KIND: unit
    # CATCHES: m06
    # CHAPTER: L5.1 section 4, The interface
    q, k, v = leaf(HQ), Tensor(HK, dtype=np.float64), Tensor(HV, dtype=np.float64)
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.requires_grad and not w.requires_grad
    out.backward(np.array([[1.0, 0.0]]))
    assert q.grad.shape == (1, 4) and k.grad is None
    c, _ = scaled_dot_product_attention(HQ, HK, HV)
    assert not c.requires_grad


def test_batched_equals_per_item():
    # WHY: leading (batch, head) axes are independent problems: a [B, H, T, d]
    #      call equals B * H separate 2-D calls, including the gradients.
    # KIND: property
    # CATCHES: s05
    # CHAPTER: L5.1 section 2.4, Batches and heads
    g = PCG32(seed=SEED + 9)
    q, k, v = rand(g, 2, 3, 4, 5), rand(g, 2, 3, 6, 5), rand(g, 2, 3, 6, 2)
    dout = rand(g, 2, 3, 4, 2)
    out, p = sdpa_forward(q, k, v)
    dq, dk, dv = sdpa_backward(q, k, v, p, dout)
    for b in range(2):
        for h in range(3):
            o1, p1 = sdpa_forward(q[b, h], k[b, h], v[b, h])
            g1 = sdpa_backward(q[b, h], k[b, h], v[b, h], p1, dout[b, h])
            assert_close(out[b, h], o1, rtol=1e-12, atol=1e-14)
            for a1, a2 in zip((dq, dk, dv), g1):
                assert_close(a1[b, h], a2, rtol=1e-12, atol=1e-14)


def test_rejects_bad_arguments():
    # WHY: mismatched widths or lengths, a float or 0/1 mask (the convention
    #      mix-up), a mask that would broadcast the scores to a new shape, a
    #      bad scale, a dropout rate outside [0, 1], or dropout with no rng
    #      are wiring bugs: they raise instead of attending to the wrong thing.
    # KIND: boundary
    # CATCHES: m03, m05, m07
    # CHAPTER: L5.1 section 4, The interface
    q, k, v = np.zeros((2, 3, 4)), np.zeros((2, 5, 4)), np.zeros((2, 5, 2))
    for args in (
        (q, np.zeros((2, 5, 3)), v),
        (q, k, np.zeros((2, 4, 2))),
        (q, k, np.zeros((3, 5, 2))),
        (np.zeros(4), k, v),
    ):
        with pytest.raises(ValueError):
            sdpa_forward(*args)
    for mask in (
        np.ones((3, 5)),
        np.ones((3, 5), dtype=np.int64),
        np.ones((4, 5), dtype=bool),
        np.ones((7, 2, 3, 5), dtype=bool),
    ):
        with pytest.raises(ValueError):
            sdpa_forward(q, k, v, mask)
    for scale in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            sdpa_forward(q, k, v, scale=scale)
    for p in (-0.1, 1.5):
        with pytest.raises(ValueError):
            scaled_dot_product_attention(
                Tensor(q), Tensor(k), Tensor(v), dropout_p=p, rng=Rng(0)
            )
    with pytest.raises(ValueError):
        scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v), dropout_p=0.1)
