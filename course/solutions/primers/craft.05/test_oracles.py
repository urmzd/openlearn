"""My oracle tests for the craft.05 kata (rung R5).

Three oracles, none of them the code under test:
  golden        torch's forward values and autograd gradients in
                course/fixtures/craft.05/oracles.npz (TINYLLM_FIXTURES)
  gradcheck     central differences in float64 of sum(f(x) * g), for a
                random upstream g, against every hand-written backward
  differential  the causal mask equals running each query on its prefix
"""

import os
from pathlib import Path

import numpy as np
import pytest

import kernels as K

FIX = np.load(Path(os.environ["TINYLLM_FIXTURES"]) / "craft.05" / "oracles.npz")
RNG = np.random.Generator(np.random.PCG64(5))


def gradcheck(f, xs, grads, eps=1e-6, tol=1e-6):
    """Central differences of the scalar f(*xs) against grads, element by element."""
    for i, (x, g) in enumerate(zip(xs, grads)):
        assert g.shape == x.shape, f"input {i}: gradient shape {g.shape} != {x.shape}"
        for idx in np.ndindex(x.shape):
            old = x[idx]
            x[idx] = old + eps
            hi = f(*xs)
            x[idx] = old - eps
            lo = f(*xs)
            x[idx] = old
            num = (hi - lo) / (2 * eps)
            assert abs(num - g[idx]) <= tol * (1 + abs(num)), f"input {i} at {idx}: {g[idx]} vs {num}"


@pytest.mark.parametrize("p", ["linear", "linear_sq"])
def test_linear_golden(p):
    x, W, b, g = FIX[p + ".x"], FIX[p + ".W"], FIX[p + ".b"], FIX[p + ".g"]
    np.testing.assert_allclose(K.linear(x, W, b), FIX[p + ".y"], atol=1e-12)
    for got, name in zip(K.linear_backward(x, W, g), ("gx", "gW", "gb")):
        np.testing.assert_allclose(got, FIX[f"{p}.{name}"], atol=1e-12, err_msg=name)


def test_rmsnorm_golden():
    x, w, g = FIX["rmsnorm.x"], FIX["rmsnorm.w"], FIX["rmsnorm.g"]
    np.testing.assert_allclose(K.rmsnorm(x, w), FIX["rmsnorm.y"], atol=1e-12)
    gx, gw = K.rmsnorm_backward(x, w, g)
    np.testing.assert_allclose(gx, FIX["rmsnorm.gx"], atol=1e-12)
    np.testing.assert_allclose(gw, FIX["rmsnorm.gw"], atol=1e-12)


def test_swiglu_golden():
    a, b, g = FIX["swiglu.a"], FIX["swiglu.b"], FIX["swiglu.g"]
    np.testing.assert_allclose(K.swiglu(a, b), FIX["swiglu.y"], atol=1e-12)
    ga, gb = K.swiglu_backward(a, b, g)
    np.testing.assert_allclose(ga, FIX["swiglu.ga"], atol=1e-12)
    np.testing.assert_allclose(gb, FIX["swiglu.gb"], atol=1e-12)


@pytest.mark.parametrize("p,causal", [("attn", False), ("attn_causal", True)])
def test_attention_golden(p, causal):
    q, k, v, g = (FIX[f"{p}.{n}"] for n in ("q", "k", "v", "g"))
    np.testing.assert_allclose(K.attention(q, k, v, causal), FIX[p + ".y"], atol=1e-12)
    for got, name in zip(K.attention_backward(q, k, v, g, causal), ("gq", "gk", "gv")):
        np.testing.assert_allclose(got, FIX[f"{p}.{name}"], atol=1e-12, err_msg=name)


def test_linear_gradcheck_square_and_not():
    for o in (4, 3):
        x, W, b, g = RNG.normal(size=(2, 4)), RNG.normal(size=(o, 4)), RNG.normal(size=o), RNG.normal(size=(2, o))
        gx, gW, gb = K.linear_backward(x, W, g)
        gradcheck(lambda x, W, b: float(np.sum(K.linear(x, W, b) * g)), [x, W, b], [gx, gW, gb])


def test_rmsnorm_gradcheck():
    x, w, g = RNG.normal(size=(3, 5)), RNG.normal(size=5), RNG.normal(size=(3, 5))
    gx, gw = K.rmsnorm_backward(x, w, g)
    gradcheck(lambda x, w: float(np.sum(K.rmsnorm(x, w) * g)), [x, w], [gx, gw])


def test_swiglu_gradcheck():
    a, b, g = RNG.normal(size=(2, 6)) * 3, RNG.normal(size=(2, 6)), RNG.normal(size=(2, 6))
    ga, gb = K.swiglu_backward(a, b, g)
    gradcheck(lambda a, b: float(np.sum(K.swiglu(a, b) * g)), [a, b], [ga, gb])


@pytest.mark.parametrize("causal", [False, True])
def test_attention_gradcheck(causal):
    q, k, v = RNG.normal(size=(3, 4)), RNG.normal(size=(3, 4)), RNG.normal(size=(3, 2))
    g = RNG.normal(size=(3, 2))
    grads = K.attention_backward(q, k, v, g, causal)
    gradcheck(lambda q, k, v: float(np.sum(K.attention(q, k, v, causal) * g)), [q, k, v], list(grads))


def test_causal_equals_prefix_attention():
    q, k, v = RNG.normal(size=(5, 3)), RNG.normal(size=(5, 3)), RNG.normal(size=(5, 2))
    full = K.attention(q, k, v, causal=True)
    for i in range(5):
        np.testing.assert_allclose(full[i], K.attention(q[i : i + 1], k[: i + 1], v[: i + 1])[0], atol=1e-12)


def test_large_scores_stay_finite():
    q = np.full((2, 3), 300.0)
    k = np.full((4, 3), 300.0)
    v = RNG.normal(size=(4, 2))
    o = K.attention(q, k, v)
    assert np.isfinite(o).all()
    np.testing.assert_allclose(o, np.tile(v.mean(axis=0), (2, 1)), atol=1e-12)
