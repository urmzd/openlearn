"""My tests for L5.4 (rung R5: the sinusoid formula in float64 numpy, and
the rotation law, are the oracles). They import only the contract."""

import math

import numpy as np
import pytest
import tinyllm.autograd.functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.num.rng import PCG32
from tinyllm.xfmr.pos import (
    LearnedPE,
    SinusoidalPE,
    shift_pe,
    sinusoidal_freqs,
    sinusoidal_pe,
)


def formula(positions, d, base=10000.0):
    out = np.zeros((len(positions), d))
    for r, p in enumerate(positions):
        for i in range(d // 2):
            a = p / base ** (2 * i / d)
            out[r, 2 * i], out[r, 2 * i + 1] = math.sin(a), math.cos(a)
    return out


def test_hand_example():
    np.testing.assert_allclose(sinusoidal_freqs(4), [1.0, 0.01], rtol=1e-12)
    np.testing.assert_allclose(
        sinusoidal_pe(3, 4), formula([0, 1, 2], 4), rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("d,base", [(16, 10000.0), (6, 77.0)])
def test_matches_formula_at_large_positions(d, base):
    ps = [0, 3, 999, 9999, 30001]
    table = sinusoidal_pe(30002, d, base)
    np.testing.assert_allclose(table[ps], formula(ps, d, base), rtol=1e-5, atol=2e-6)


def test_shift_is_rotation():
    t = sinusoidal_pe(100, 8).astype(np.float64)
    for p, k in [(0, 1), (10, 7), (40, 33)]:
        np.testing.assert_allclose(shift_pe(t[p], k, 8), t[p + k], atol=2e-6)
    np.testing.assert_allclose(
        shift_pe(np.array(formula([1], 4)[0]), 1, 4), formula([2], 4)[0], atol=1e-12
    )


def test_offsets():
    np.testing.assert_array_equal(
        sinusoidal_pe(3, 8, offset=5), sinusoidal_pe(8, 8)[5:]
    )
    m = SinusoidalPE(10, 4)
    assert m.state_dict() == {}
    y = m(Tensor(np.zeros((1, 2, 4))), offset=3)
    np.testing.assert_allclose(y.data[0], formula([3, 4], 4), atol=1e-6)


def test_learned_rows_offset_and_gradient():
    m = LearnedPE(8, 3, rng=PCG32(1))
    x = Tensor(np.zeros((2, 3, 3)))
    y = m(x, offset=4)
    np.testing.assert_array_equal(y.data[1], m.weight.data[4:7])
    F.sum(y).backward()
    g = np.zeros((8, 3))
    g[4:7] = 2.0
    np.testing.assert_array_equal(m.weight.grad, g)


def test_learned_init_std():
    w = LearnedPE(128, 64, std=0.1, rng=PCG32(2)).weight.data
    assert abs(w.std() - 0.1) < 0.005


def test_validation():
    with pytest.raises(ValueError):
        sinusoidal_pe(4, 7)
    with pytest.raises(ValueError):
        LearnedPE(4, 2)(Tensor(np.zeros((1, 3, 2))), offset=2)
