"""My tests for L7.3 (rung R5). The oracle is RoPE written as complex
multiplication in numpy float64: pair (a, b) is a + ib, rotated by e^{i angle}.
They import only the contract."""

import numpy as np
import pytest
import tinyllm.autograd.functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.rope import apply_rope, rope_cos_sin


def ladder(r, base=10000.0):
    return base ** (-np.arange(0, r, 2) / r)


def oracle(x, pos, inv, layout, r):
    ang = pos[:, None] * inv
    rot = x[..., :r]
    if layout == "half":
        z = rot[..., : r // 2] + 1j * rot[..., r // 2 :]
        z = z * np.exp(1j * ang)
        out = np.concatenate([z.real, z.imag], axis=-1)
    else:
        z = rot[..., 0::2] + 1j * rot[..., 1::2]
        z = z * np.exp(1j * ang)
        out = np.stack([z.real, z.imag], axis=-1).reshape(rot.shape)
    return np.concatenate([out, x[..., r:]], axis=-1)


@pytest.mark.parametrize("layout,r", [("half", 8), ("interleaved", 8), ("half", 4), ("interleaved", 6)])
def test_matches_complex_rotation(layout, r):
    rng = np.random.Generator(np.random.PCG64(1))
    x = rng.normal(size=(2, 5, 8))
    pos = np.array([0, 1, 5, 100, 3000])
    cos, sin = rope_cos_sin(pos, ladder(r))
    got = apply_rope(Tensor(x, dtype=np.float64), cos, sin, layout, r).data
    np.testing.assert_allclose(got, oracle(x, pos, ladder(r), layout, r), atol=3e-3)
    np.testing.assert_allclose(got[:, :2], oracle(x, pos, ladder(r), layout, r)[:, :2], atol=1e-6)


def test_hand_example():
    cos, sin = rope_cos_sin(np.array([1]), np.array([1.0, 0.01]))
    y = apply_rope(Tensor([[1.0, 0.0, 0.0, 1.0]]), cos, sin, "half").data
    np.testing.assert_allclose(y, [[np.cos(1), -np.sin(0.01), np.sin(1), np.cos(0.01)]], atol=1e-6)


def test_scaling_multiplies_both():
    c1, s1 = rope_cos_sin(np.arange(4), ladder(4))
    c2, s2 = rope_cos_sin(np.arange(4), ladder(4), 1.5)
    np.testing.assert_allclose(c2, 1.5 * c1, rtol=1e-6)
    np.testing.assert_allclose(s2, 1.5 * s1, rtol=1e-6)


def test_large_positions_are_precise():
    cos, sin = rope_cos_sin(np.array([4097, 60001]), np.array([1.0]))
    np.testing.assert_allclose(cos[:, 0], np.cos([4097.0, 60001.0]), atol=1e-3)


def test_gradient_is_inverse_rotation():
    rng = np.random.Generator(np.random.PCG64(2))
    x0, g = rng.normal(size=(3, 6)), rng.normal(size=(3, 6))
    cos, sin = rope_cos_sin(np.arange(3), ladder(6))
    x = Tensor(x0, requires_grad=True, dtype=np.float64)
    F.sum(apply_rope(x, cos, sin) * g).backward()
    np.testing.assert_allclose(x.grad, apply_rope(Tensor(g, dtype=np.float64), cos, -sin).data, atol=1e-12)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        rope_cos_sin(np.array([-1]), ladder(4))
    cos, sin = rope_cos_sin(np.arange(2), ladder(4))
    with pytest.raises(ValueError):
        apply_rope(Tensor(np.ones((2, 4))), cos, sin, "neox")
