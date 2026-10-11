"""My tests for L3.1 (rung R3: each test written first and seen failing with
`ol tdd red`, then made to pass with `ol tdd green`). They import only the
contract (tinyllm.rnn.manual)."""

import numpy as np
import pytest
from tinyllm.rnn.manual import gradient_flow, rnn_backward, rnn_forward, tbptt_grads, tbptt_windows

KEYS = ("x", "h0", "Wxh", "Whh", "bh")


def problem(seed, T=4, B=2, D=3, H=3):
    r = np.random.default_rng(seed)
    return (
        r.normal(size=(T, B, D)),
        0.5 * r.normal(size=(B, H)),
        0.6 * r.normal(size=(D, H)),
        0.6 * r.normal(size=(H, H)),
        0.2 * r.normal(size=H),
        r.normal(size=(T, B, H)),
    )


def numeric(f, arrs, eps=1e-6):
    out = []
    for k in range(len(arrs)):
        g = np.zeros_like(arrs[k])
        for i in np.ndindex(arrs[k].shape):
            a = [x.copy() for x in arrs]
            a[k][i] += eps
            hi = f(*a)
            a[k][i] -= 2 * eps
            g[i] = (hi - f(*a)) / (2 * eps)
        out.append(g)
    return out


def test_hand_example_two_steps():
    """Wxh = 0.5, Whh = 0.8, x = [1, 0], h0 = 0, L = h_1: dWxh = 0.550438, dWhh = 0.404297, dh0 = 0.440350."""
    h, cache = rnn_forward(np.array([[[1.0]], [[0.0]]]), [[0.0]], [[0.5]], [[0.8]], [0.0])
    np.testing.assert_allclose(h[:, 0, 0], [np.tanh(0.5), np.tanh(0.8 * np.tanh(0.5))])
    g = rnn_backward(np.array([[[0.0]], [[1.0]]]), cache)
    np.testing.assert_allclose(g["Wxh"], [[0.5504375878410427]])
    np.testing.assert_allclose(g["Whh"], [[0.4042968189107551]])
    np.testing.assert_allclose(g["bh"], [1.425317069286649])
    np.testing.assert_allclose(g["h0"], [[0.4403500702728342]])
    np.testing.assert_allclose(g["x"][:, 0, 0], [0.5504375878410427 * 0.5, 0.8748794814456065 * 0.5])


def test_gradients_match_central_differences():
    """Every gradient against central differences of the forward, with dh_next."""
    x, h0, Wxh, Whh, bh, G = problem(0)
    gn = np.random.default_rng(1).normal(size=h0.shape)

    def loss(*a):
        h, _ = rnn_forward(*a)
        return float((G * h).sum() + (gn * h[-1]).sum())

    _, cache = rnn_forward(x, h0, Wxh, Whh, bh)
    g = rnn_backward(G, cache, dh_next=gn)
    for k, want in zip(KEYS, numeric(loss, [x, h0, Wxh, Whh, bh])):
        np.testing.assert_allclose(g[k], want, rtol=1e-6, atol=1e-8, err_msg=k)


def test_tbptt_windows():
    """Cuts every k1 steps and at T, each reaching back k2 steps."""
    assert tbptt_windows(10, 4, 6) == [(0, 4), (2, 8), (4, 10)]
    assert tbptt_windows(10, 4, 4) == [(0, 4), (4, 8), (6, 10)]
    assert tbptt_windows(0, 1, 1) == []
    with pytest.raises(ValueError):
        tbptt_windows(5, 3, 2)


def test_tbptt_long_reach_is_full_bptt():
    """k2 >= T: every loss counted once over its whole history, for any k1."""
    x, h0, Wxh, Whh, bh, G = problem(2, T=7)
    _, cache = rnn_forward(x, h0, Wxh, Whh, bh)
    full = rnn_backward(G, cache)
    for k1 in (1, 3):
        got = tbptt_grads(x, h0, Wxh, Whh, bh, G, k1, 7)
        for k in KEYS:
            np.testing.assert_allclose(got[k], full[k], rtol=1e-9, atol=1e-12, err_msg=f"{k1} {k}")


def test_tbptt_one_step_windows():
    """k1 = k2 = 1: each loss only reaches its own step."""
    x, h0, Wxh, Whh, bh, G = problem(3, T=3)
    h, _ = rnn_forward(x, h0, Wxh, Whh, bh)
    got = tbptt_grads(x, h0, Wxh, Whh, bh, G, 1, 1)
    da = G * (1 - h * h)
    prev = np.concatenate([h0[None], h[:-1]])
    np.testing.assert_allclose(got["Whh"], np.einsum("tbi,tbj->ij", prev, da))
    np.testing.assert_allclose(got["h0"], da[0] @ Whh.T)


def test_gradient_flow_non_normal():
    """Whh = [[0.5, 2], [0, 0.5]]: rho is 0.5 (not the spectral norm), steps back use Whh^T."""
    Whh = np.array([[0.5, 2.0], [0.0, 0.5]])
    _, cache = rnn_forward(np.zeros((3, 1, 1)), np.zeros((1, 2)), np.zeros((1, 2)), Whh, np.zeros(2))
    norms, rho = gradient_flow(cache, np.array([[1.0, 0.0]]))
    np.testing.assert_allclose(rho, 0.5, rtol=1e-6)
    np.testing.assert_allclose(norms, [0.25, 0.5, 1.0])


def test_shape_errors():
    """A wrong Wxh or bh shape is rejected."""
    with pytest.raises(ValueError):
        rnn_forward(np.zeros((2, 1, 3)), np.zeros((1, 4)), np.zeros((4, 3)), np.zeros((4, 4)), np.zeros(4))
    with pytest.raises(ValueError):
        rnn_forward(np.zeros((2, 1, 3)), np.zeros((1, 4)), np.zeros((3, 4)), np.zeros((4, 4)), np.zeros(3))
