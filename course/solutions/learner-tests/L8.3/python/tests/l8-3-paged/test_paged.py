"""My tests for L8.3 (rung R4: properties). They import only the contract."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tinyllm.infer.paged import OutOfBlocks, PagedKVCache


def chunk(t0, T, H=2, D=3):
    """Distinct, float16-exact values per position."""
    return (
        np.arange(t0, t0 + T, dtype=np.float32)[None, :, None]
        + np.zeros((H, 1, D), np.float32)
        + np.arange(H, dtype=np.float32)[:, None, None] * 0.5
    )


@settings(max_examples=40)
@given(st.lists(st.integers(1, 7), min_size=1, max_size=8))
def test_gather_is_the_concatenation(sizes):
    with PagedKVCache(32, 4, 2, 2, 3) as c:
        c.add_seq(0)
        t = 0
        for T in sizes:
            for layer in range(2):
                c.append(0, layer, chunk(t, T), -chunk(t, T))
            t += T
        for layer in range(2):
            K, V = c.gather(0, layer)
            want = chunk(0, t).astype(np.float16)
            assert (K == want).all() and (V == -want).all()
        assert len(c.block_table(0)) == (t + 3) // 4


def test_fork_then_write_never_changes_the_parent():
    with PagedKVCache(8, 2, 1, 2, 3) as c:
        c.add_seq(0)
        c.append(0, 0, chunk(0, 3), chunk(0, 3))
        used = c.stats()["used"]
        c.fork(0, 1)
        assert c.stats()["used"] == used
        c.append(1, 0, chunk(100, 1), chunk(100, 1))
        c.append(
            0, 0, chunk(200, 1), chunk(200, 1)
        )  # both write position 3 of the shared block
        assert c.gather(0, 0)[0][0, :, 0].tolist() == [0, 1, 2, 200]
        assert c.gather(1, 0)[0][0, 3, 0] == 100
        c.free(0)
        assert c.gather(1, 0)[0].shape == (2, 4, 3)
        c.free(1)
        assert c.stats()["free"] == 8


def test_layers_keep_independent_lengths():
    with PagedKVCache(4, 4, 2, 2, 3) as c:
        c.add_seq(0)
        c.append(0, 0, chunk(0, 5), chunk(0, 5))
        c.append(0, 1, chunk(0, 2), chunk(0, 2))
        assert c.seq_len(0, 0) == 5 and c.seq_len(0, 1) == 2
        assert len(c.block_table(0)) == 2
        assert c.gather(0, 1)[0].shape == (2, 2, 3)


def test_freeing_everything_returns_every_block():
    with PagedKVCache(6, 2, 2, 2, 3) as c:
        for s in range(3):
            c.add_seq(s)
            for layer in range(2):
                c.append(s, layer, chunk(0, 3), chunk(0, 3))
        c.fork(1, 9)
        for s in (0, 1, 2, 9):
            c.free(s)
        assert c.stats() == {"free": 6, "used": 0, "cached": 0, "evictions": 0}


def test_out_of_blocks_changes_nothing():
    with PagedKVCache(3, 2, 1, 2, 3) as c:
        c.add_seq(0)
        c.append(0, 0, chunk(0, 2), chunk(0, 2))
        before = c.stats()
        with pytest.raises(OutOfBlocks):
            c.append(0, 0, chunk(2, 5), chunk(2, 5))
        assert c.seq_len(0) == 2 and c.stats() == before
        assert (c.gather(0, 0)[0] == chunk(0, 2).astype(np.float16)).all()
        with pytest.raises(ValueError):
            c.append(0, 0, np.ones((2, 0, 3)), np.ones((2, 0, 3)))
