"""My tests for L0.3 (rung R2: the names and docstrings come from the chapter,
section 4; the bodies are mine). They import only the contract."""

import math

import numpy as np
import pytest
from tinyllm.autograd.losses import bce_with_logits, cross_entropy, mse
from tinyllm.autograd.tensor import Tensor

LN2, LN3, LN6 = math.log(2.0), math.log(3.0), math.log(6.0)
ROW = [0.0, LN2, LN3]  # softmax = [1/6, 1/3, 1/2]


def leaf(x):
    return Tensor(np.asarray(x, dtype=np.float64), requires_grad=True, dtype=np.float64)


def fd_grad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        a, b = x.copy(), x.copy()
        a[i] += eps
        b[i] -= eps
        g[i] = (f(a) - f(b)) / (2 * eps)
    return g


def test_hand_example_matches_section_3():
    """logits [0, ln 2, ln 3], target 2: loss ln 2, gradient [1/6, 1/3, -1/2]."""
    z = leaf([ROW])
    loss = cross_entropy(z, np.array([2]))
    np.testing.assert_allclose(loss.data, LN2, rtol=1e-12)
    loss.backward()
    np.testing.assert_allclose(z.grad, [[1 / 6, 1 / 3, -1 / 2]], rtol=1e-12)


def test_ignored_rows_get_no_loss_and_no_gradient():
    """A row whose target is ignore_index adds no loss and gets a zero gradient."""
    z = leaf([ROW, [9.0, -3.0, 1.0]])
    loss = cross_entropy(z, np.array([2, -100]))
    np.testing.assert_allclose(loss.data, LN2, rtol=1e-12)
    loss.backward()
    np.testing.assert_array_equal(z.grad[1], [0.0, 0.0, 0.0])
    z2 = leaf([ROW, ROW])
    cross_entropy(z2, np.array([1, 1]), ignore_index=1).backward()
    np.testing.assert_array_equal(z2.grad, np.zeros((2, 3)))


def test_mean_divides_by_kept_rows():
    """Two kept rows and one ignored row: the mean is the kept sum over 2."""
    z = leaf([ROW, ROW, ROW])
    loss = cross_entropy(z, np.array([2, 0, -100]))
    np.testing.assert_allclose(loss.data, (LN2 + LN6) / 2, rtol=1e-12)
    total = cross_entropy(leaf([ROW, ROW, ROW]), np.array([2, 0, -100]), reduction="sum")
    np.testing.assert_allclose(total.data, LN2 + LN6, rtol=1e-12)


def test_all_ignored_batch_is_zero():
    """Only padding: loss 0 and a zero gradient, never nan."""
    z = leaf([ROW])
    loss = cross_entropy(z, np.array([-100]))
    assert float(loss.data) == 0.0
    loss.backward()
    np.testing.assert_array_equal(z.grad, [[0.0, 0.0, 0.0]])


def test_label_smoothing_hand_value():
    """eps 0.3 on V = 3: q = [0.1, 0.1, 0.8]; loss 0.1 ln 6 + 0.1 ln 3 + 0.8 ln 2; grad p - q."""
    z = leaf([ROW])
    loss = cross_entropy(z, np.array([2]), label_smoothing=0.3)
    np.testing.assert_allclose(loss.data, 0.1 * LN6 + 0.1 * LN3 + 0.8 * LN2, rtol=1e-12)
    loss.backward()
    np.testing.assert_allclose(z.grad, [[1 / 6 - 0.1, 1 / 3 - 0.1, -0.3]], rtol=1e-12, atol=1e-15)


def test_large_logits_are_finite():
    """Logits of 1e4 and bce logits of 1000 give finite losses and gradients."""
    z = leaf([[1e4, 0.0, -1e4]])
    loss = cross_entropy(z, np.array([2]))
    assert np.isfinite(loss.data)
    np.testing.assert_allclose(loss.data, 2e4, rtol=1e-12)
    b = leaf([-1000.0, 1000.0])
    lb = bce_with_logits(b, np.array([1.0, 0.0]))
    np.testing.assert_allclose(lb.data, 1000.0, rtol=1e-12)


def test_out_of_range_target_raises():
    """Targets outside [0, V) (other than ignore_index) are a ValueError."""
    for t in ([3], [-1]):
        with pytest.raises(ValueError):
            cross_entropy(leaf([ROW]), np.array(t))


def test_mse_and_bce_gradients_by_finite_differences():
    """Both gradients match central differences; bce with pos_weight 2 at x = 0, y = 1 is 2 ln 2."""
    x0 = np.array([[0.3, -1.2], [2.0, 0.5]])
    y = np.array([[1.0, 0.0], [0.25, 1.0]])
    p = leaf(x0)
    mse(p, y).backward()
    np.testing.assert_allclose(p.grad, fd_grad(lambda a: float(mse(Tensor(a, dtype=np.float64), y).data), x0), rtol=1e-6)
    q = leaf(x0)
    bce_with_logits(q, y).backward()
    np.testing.assert_allclose(q.grad, fd_grad(lambda a: float(bce_with_logits(Tensor(a, dtype=np.float64), y).data), x0), rtol=1e-6)
    w = bce_with_logits(leaf([0.0]), np.array([1.0]), pos_weight=np.array([2.0]))
    np.testing.assert_allclose(w.data, 2 * LN2, rtol=1e-12)
