"""My tests for L7.1 (rung R5: oracles and my own gradient check). The oracle
is the RMSNorm formula written out in numpy float64. They import only the
contract."""

import numpy as np
import pytest
import tinyllm.autograd.functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.norm import RMSNorm, pre_norm_residual
from tinyllm.nn.layers import Linear


def oracle(x, w, eps, offset):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * (offset + w)


def fd(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        a, b = x.copy(), x.copy()
        a[i] += eps
        b[i] -= eps
        g[i] = (f(a) - f(b)) / (2 * eps)
    return g


def test_hand_example():
    """(3, 4, 0, 0) has rms 2.5: (1.2, 1.6, 0, 0); a constant vector stays constant."""
    n = RMSNorm(4, eps=0.0)
    np.testing.assert_allclose(n(Tensor([[3.0, 4.0, 0.0, 0.0]])).data, [[1.2, 1.6, 0, 0]], rtol=1e-6)
    np.testing.assert_allclose(n(Tensor([[2.0, 2.0, 2.0, 2.0]])).data, [[1, 1, 1, 1]], rtol=1e-6)


@pytest.mark.parametrize("offset,eps,scale", [(0.0, 1e-5, 1.0), (1.0, 1e-6, 1.0), (0.0, 1e-5, 1e-3)])
def test_matches_the_formula(offset, eps, scale):
    r = np.random.Generator(np.random.PCG64(1))
    x = scale * r.normal(size=(2, 3, 8))
    w = r.uniform(-1, 1, size=8)
    n = RMSNorm(8, eps=eps, offset=offset)
    n.load_state_dict({"weight": w})
    np.testing.assert_allclose(n(Tensor(x)).data, oracle(x, w, eps, offset), rtol=1e-5, atol=1e-7)


def test_init():
    assert np.all(RMSNorm(3).weight.data == 1) and np.all(RMSNorm(3, offset=1.0).weight.data == 0)


def test_gradients_by_finite_differences():
    r = np.random.Generator(np.random.PCG64(2))
    x0, w0, gy = r.normal(size=(3, 5)), r.normal(size=5), r.normal(size=(3, 5))
    for offset in (0.0, 1.0):
        n = RMSNorm(5, eps=1e-3, offset=offset)
        n.weight.data = w0.copy()
        x = Tensor(x0, requires_grad=True, dtype=np.float64)
        F.sum(n(x) * gy).backward()
        np.testing.assert_allclose(x.grad, fd(lambda a: np.sum(oracle(a, w0, 1e-3, offset) * gy), x0), rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(n.weight.grad, fd(lambda a: np.sum(oracle(x0, a, 1e-3, offset) * gy), w0), rtol=1e-5, atol=1e-8)


def test_pre_norm_residual():
    r = np.random.Generator(np.random.PCG64(3))
    x0 = 30 * r.normal(size=(2, 4))
    lin = Linear(4, 4, bias=False)
    n = RMSNorm(4)
    out = pre_norm_residual(Tensor(x0), n, lin)
    W = lin.weight.data.astype(np.float64)
    want = x0 + oracle(x0, np.ones(4), 1e-6, 0.0) @ W.T
    np.testing.assert_allclose(out.data, want, rtol=1e-5, atol=1e-4)
