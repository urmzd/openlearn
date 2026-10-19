"""My tests for L5.2 (rung R5: an oracle written as loops from the
definitions, plus the causality property through a small numpy attention).
They import only the contract."""

import numpy as np
import pytest
from tinyllm.xfmr.masks import (
    causal_mask,
    combine,
    padding_mask,
    sliding_window_mask,
    to_additive,
)


def oracle_causal(tq, tk, off):
    return np.array([[j <= off + i for j in range(tk)] for i in range(tq)])


def oracle_window(tq, tk, w, off):
    return np.array(
        [[j <= off + i and off + i - j < w for j in range(tk)] for i in range(tq)]
    )


def attend(x, add):
    s = x @ x.T + add
    m = s.max(axis=-1, keepdims=True)
    m = np.where(np.isneginf(m), 0.0, m)
    e = np.exp(s - m)
    t = e.sum(axis=-1, keepdims=True)
    return (e / np.where(t == 0, 1.0, t)) @ x


@pytest.mark.parametrize(
    "tq,tk,off", [(1, 1, 0), (3, 3, 0), (2, 5, 3), (4, 6, 0), (1, 8, 7), (3, 9, 6)]
)
def test_causal_matches_loops(tq, tk, off):
    assert (
        causal_mask(tq, tk, q_offset=off).tolist()
        == oracle_causal(tq, tk, off).tolist()
    )


def test_default_tk():
    assert causal_mask(2, q_offset=4).tolist() == oracle_causal(2, 6, 4).tolist()


@pytest.mark.parametrize(
    "tq,tk,w,off",
    [(5, 5, 1, 0), (5, 5, 2, 0), (6, 6, 3, 0), (2, 7, 3, 5), (4, 4, 9, 0)],
)
def test_window_matches_loops(tq, tk, w, off):
    assert (
        sliding_window_mask(tq, tk, w, q_offset=off).tolist()
        == oracle_window(tq, tk, w, off).tolist()
    )


def test_padding():
    assert padding_mask(np.array([0, 2, 3]), 3).tolist() == [
        [False] * 3,
        [True, True, False],
        [True] * 3,
    ]


def test_combine_is_and():
    c = causal_mask(4)
    p = padding_mask(np.array([2, 4]), 4)[:, None, :]
    assert combine(c, p).tolist() == (c & p).tolist()


def test_additive():
    a = to_additive(np.array([True, False]), np.float64)
    assert a[0] == 0.0 and np.isneginf(a[1])
    assert to_additive(np.array([True])).dtype == np.float32
    row = to_additive(np.array([[False, False]]), np.float64)
    assert np.all(np.isneginf(row))


def test_causality():
    r = np.random.default_rng(0)
    x = r.normal(size=(6, 3))
    add = to_additive(causal_mask(6), np.float64)
    base = attend(x, add)
    y = x.copy()
    y[4:] += 5.0
    assert np.array_equal(attend(y, add)[:4], base[:4])


def test_rejects_float_masks():
    with pytest.raises(ValueError):
        to_additive(np.array([1.0, 0.0]))
    with pytest.raises(ValueError):
        combine(np.array([1, 0]))
