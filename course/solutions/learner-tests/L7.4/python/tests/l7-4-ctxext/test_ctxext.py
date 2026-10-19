"""My tests for L7.4 (rung R5). Oracles: each paper's rule written out per
pair in plain Python floats. They import only the contract."""

import math

import numpy as np
import pytest
from tinyllm.modern.ctxext import alibi_bias, rope_inv_freq_scaled


def theta(r, base):
    return [base ** (-2 * i / r) for i in range(r // 2)]


def yarn(r, base, f, L0, bf=32, bs=1):
    c = lambda n: r * math.log(L0 / (n * 2 * math.pi)) / (2 * math.log(base))  # noqa: E731
    lo, hi = max(math.floor(c(bf)), 0), min(math.ceil(c(bs)), r - 1)
    out = []
    for i, t in enumerate(theta(r, base)):
        ramp = min(max((i - lo) / (hi - lo), 0.0), 1.0)
        out.append(t * (1 - ramp) + t / f * ramp)
    return out, 0.1 * math.log(f) + 1


def llama3(r, base, f, L0, lo=1.0, hi=4.0):
    out = []
    for t in theta(r, base):
        w = 2 * math.pi / t
        if w < L0 / hi:
            out.append(t)
        elif w > L0 / lo:
            out.append(t / f)
        else:
            s = (L0 / w - lo) / (hi - lo)
            out.append((1 - s) * t / f + s * t)
    return out


def test_linear():
    inv, s = rope_inv_freq_scaled(64, 1e4, "linear", 4.0)
    np.testing.assert_allclose(inv, np.array(theta(64, 1e4)) / 4, rtol=1e-12)
    assert s == 1.0


def test_ntk():
    inv, _ = rope_inv_freq_scaled(32, 1e4, "ntk", 8.0)
    np.testing.assert_allclose(inv, theta(32, 1e4 * 8.0 ** (32 / 30)), rtol=1e-12)


@pytest.mark.parametrize("r,base,f,L0", [(8, 1e4, 4.0, 2048), (128, 1e6, 8.0, 32768), (64, 1e5, 4.0, 2048)])
def test_yarn(r, base, f, L0):
    inv, s = rope_inv_freq_scaled(r, base, "yarn", f, L0)
    want, ws = yarn(r, base, f, L0)
    np.testing.assert_allclose(inv, want, rtol=1e-12)
    assert abs(s - ws) < 1e-12


def test_llama3():
    inv, _ = rope_inv_freq_scaled(128, 5e5, "llama3", 8.0, 8192)
    np.testing.assert_allclose(inv, llama3(128, 5e5, 8.0, 8192), rtol=1e-12)


def test_alibi():
    np.testing.assert_allclose(alibi_bias(2, 2, 3)[0], [[-1 / 16, 0, 1 / 16], [-2 / 16, -1 / 16, 0]])


@pytest.mark.parametrize("kw", [
    dict(d_rot=64, base=1e4, kind="dynamic", factor=2.0, original_max_pos=4096),
    dict(d_rot=64, base=1e4, kind="ntk", factor=0.5),
    dict(d_rot=64, base=1e4, kind="llama3", factor=8.0, original_max_pos=8192, low_freq_factor=4, high_freq_factor=1),
])
def test_rejects_bad_config(kw):
    with pytest.raises(ValueError):
        rope_inv_freq_scaled(**kw)


def test_alibi_rejects_more_queries_than_keys():
    with pytest.raises(ValueError):
        alibi_bias(2, 4, 3)
