"""My tests for L3.4 (rung R4: the chapter's properties, written as property
tests with Hypothesis). They import only the contract."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from tinyllm.autograd.tensor import Tensor
from tinyllm.rnn.bi import bidirectional, reversal_index, reverse_padded
from tinyllm.rnn.gru import GRU


class Gen:
    def __init__(self, k=0):
        self.k = k

    def uniform(self):
        self.k += 1
        return (self.k * 0.6180339887) % 1.0

    def next_u32(self):
        return int(self.uniform() * 2**32)

    def uniforms(self, n):
        return np.array([self.uniform() for _ in range(n)])


batches = st.integers(1, 6).flatmap(lambda T: st.tuples(st.just(T), st.lists(st.integers(1, T), min_size=1, max_size=4)))


@given(batches)
def test_reversal_is_an_involution_and_keeps_padding(tb):
    """Reversing twice is the identity; padding positions never move; real steps are mirrored."""
    T, lengths = tb
    n = np.array(lengths)
    x = np.arange(T * len(n), dtype=np.float64).reshape(T, len(n))
    r = reverse_padded(x, n)
    np.testing.assert_array_equal(reverse_padded(r, n), x)
    for b, nb in enumerate(n):
        np.testing.assert_array_equal(r[nb:, b], x[nb:, b])
        np.testing.assert_array_equal(r[:nb, b], x[:nb, b][::-1])
    assert reversal_index(n, T).shape == (T, len(n))


@given(st.integers(1, 5), st.integers(1, 3))
def test_no_lengths_is_a_flip(T, B):
    """lengths None is x[::-1] along time."""
    x = np.arange(T * B, dtype=np.float64).reshape(T, B)
    np.testing.assert_array_equal(reverse_padded(x, None), x[::-1])


def test_matches_per_sequence_loop():
    """The batch equals each sequence alone: forward half from fwd, backward half from bwd read backward."""
    fwd, bwd = GRU(2, 3, rng=Gen(1)), GRU(2, 3, rng=Gen(2))
    lengths = np.array([4, 1, 3])
    x = np.random.default_rng(0).normal(size=(4, 3, 2)).astype(np.float32)
    out = bidirectional(fwd, bwd, Tensor(x), lengths).data
    assert out.shape == (4, 3, 6)
    for b, n in enumerate(lengths):
        of, _ = fwd(x[:n, b : b + 1])
        ob, _ = bwd(x[:n, b : b + 1][::-1].copy())
        np.testing.assert_allclose(out[:n, b, :3], of.data[:, 0], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(out[:n, b, 3:], ob.data[::-1, 0], rtol=1e-5, atol=1e-6)
        assert (out[n:, b] == 0).all()


def test_padding_values_do_not_matter():
    """Changing the padding changes no output."""
    fwd, bwd = GRU(2, 3, rng=Gen(3)), GRU(2, 3, rng=Gen(4))
    lengths = np.array([2, 4])
    x = np.random.default_rng(1).normal(size=(4, 2, 2)).astype(np.float32)
    y = x.copy()
    y[2:, 0] = 100.0
    a = bidirectional(fwd, bwd, Tensor(x), lengths).data
    b = bidirectional(fwd, bwd, Tensor(y), lengths).data
    np.testing.assert_allclose(a, b)


def test_bad_lengths():
    """Lengths outside 1..T raise."""
    with pytest.raises(ValueError):
        reverse_padded(np.zeros((3, 2)), np.array([1, 4]))
