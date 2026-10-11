"""Course tests for L0.6: the token-stream reader (tinyllm/io/tokens.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L0.6), and the chapter section it comes from.

The worked example of the chapter (section 3): one shard holding the ids
10, 11, ..., 19 (n = 10), seq_len T = 3, batch B = 2, and a generator whose
phase draws come out 1, then 0. Pass 1 starts at offset 1: windows at 1 and
4 (the next, at 7, would need ids up to 10). Batch 1 is inputs
[[11, 12, 13], [14, 15, 16]], targets [[12, 13, 14], [15, 16, 17]]. Batch 2
starts pass 2 at phase 0: windows at 0 and 3, inputs [[10, 11, 12],
[13, 14, 15]]; the cursor is then {shard 0, offset 6}. Each phase is drawn
with below(min(T, n - T)) = below(3).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from _lib.pcg32 import PCG32
from tinyllm.io.tokens import TokenStream, open_tokens, read_tokens_header

MAGIC = 20240520


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def write_bin(
    path,
    ids,
    version: int = 1,
    vocab: int = 256,
    n: int | None = None,
    magic: int = MAGIC,
) -> str:
    """A formats/tokens-bin.md file, written by hand (the writer is data.07's)."""
    head = np.zeros(256, dtype="<i4")
    head[:4] = [magic, version, len(ids) if n is None else n, vocab]
    width = "<u2" if version == 1 else "<u4"
    with open(path, "wb") as f:
        f.write(head.tobytes())
        f.write(np.asarray(ids, dtype=width).tobytes())
    return str(path)


class Rng:
    """The frozen PCG32 behind the generator API TokenStream uses (M06.3:
    below, state, set_state), so no verdict here depends on your PCG32."""

    def __init__(self, s: int) -> None:
        self.g = PCG32(seed=s)

    def below(self, n: int) -> int:
        return self.g.below(n)

    def state(self) -> tuple[int, int]:
        return (self.g.state, self.g.inc)

    def set_state(self, s) -> None:
        self.g.state, self.g.inc = int(s[0]), int(s[1])


class Scripted:
    """A generator whose below() answers come from a list: the hand example."""

    def __init__(self, answers: list[int]) -> None:
        self.answers, self.asked, self.i = list(answers), [], 0

    def below(self, n: int) -> int:
        self.asked.append(n)
        v = self.answers[self.i]
        self.i += 1
        assert 0 <= v < n
        return v

    def state(self) -> tuple[int, int]:
        return (self.i, 1)

    def set_state(self, s) -> None:
        self.i = int(s[0])


def test_hand_example_token_windows(tmp_path):
    # WHY: the chapter's worked example, window by window: a random phase
    #      per pass, windows T apart that share their boundary token, a new
    #      pass when the next window would run off the end, and the cursor
    #      naming the next window.
    # KIND: unit
    # CATCHES: s25, s26, s27, s30
    # CHAPTER: L0.6 section 3, Worked example by hand
    p = write_bin(tmp_path / "a.bin", range(10, 20))
    rng = Scripted([1, 0])
    s = TokenStream([p], seq_len=3, batch=2, rng=rng)
    x, y = s.next_batch()
    assert x.tolist() == [[11, 12, 13], [14, 15, 16]] and y.tolist() == [
        [12, 13, 14],
        [15, 16, 17],
    ]
    assert x.dtype == np.int64 and y.dtype == np.int64
    x, y = s.next_batch()
    assert x.tolist() == [[10, 11, 12], [13, 14, 15]] and y.tolist() == [
        [11, 12, 13],
        [14, 15, 16],
    ]
    c = s.cursor()
    assert (c["shard"], c["offset"]) == (0, 6)
    assert rng.asked == [3, 3]


def test_header_worked_example(tmp_path):
    # WHY: formats/tokens-bin.md's worked example: the ids [1, 2, 3], version
    #      1, vocab 256, is exactly 1030 bytes, and the reader returns its
    #      header fields and the ids as uint16.
    # KIND: unit
    # CHAPTER: L0.6 section 3, Worked example by hand (the .bin header)
    p = write_bin(tmp_path / "w.bin", [1, 2, 3])
    assert os.path.getsize(p) == 1030
    with open(p, "rb") as f:
        assert f.read(16) == bytes.fromhex(
            "88d83401 01000000 03000000 00010000".replace(" ", "")
        )
    assert read_tokens_header(p) == {"version": 1, "n_tokens": 3, "vocab_size": 256}
    ids = open_tokens(p)
    assert ids.tolist() == [1, 2, 3] and ids.dtype == np.dtype("<u2")


def test_version_2_reads_uint32(tmp_path):
    # WHY: version 2 stores uint32 ids for vocabularies above 65536; reading
    #      them as uint16 would split every id in two.
    # KIND: unit
    # CATCHES: s31
    # CHAPTER: L0.6 section 2, Principles (the .bin layout)
    p = write_bin(tmp_path / "v2.bin", [70000, 5, 99999], version=2, vocab=100000)
    assert os.path.getsize(p) == 1024 + 12
    assert open_tokens(p).tolist() == [70000, 5, 99999]


def test_open_tokens_maps_the_file(tmp_path):
    # WHY: a training shard can be larger than memory: the ids are a
    #      read-only np.memmap over the file, not a copy, so opening a
    #      100M-token shard costs nothing until a window is read.
    # KIND: unit
    # CATCHES: s32
    # CHAPTER: L0.6 section 2, Principles (memory mapping)
    p = write_bin(tmp_path / "m.bin", list(range(100)))
    ids = open_tokens(p)
    assert isinstance(ids, np.memmap) and not ids.flags.writeable


def test_rejects_bad_files(tmp_path):
    # WHY: a reader never trusts a header: wrong magic, an unknown version, a
    #      size that disagrees with n_tokens (truncated, or trailing bytes), an
    #      id at or above vocab_size, or a file shorter than the header are
    #      all errors before training starts, not garbage windows.
    # KIND: boundary
    # CATCHES: s33, s34
    # CHAPTER: L0.6 section 5, Pitfalls, item 8
    bad = [
        write_bin(tmp_path / "magic.bin", [1, 2], magic=20240801),
        write_bin(tmp_path / "ver.bin", [1, 2], version=3),
        write_bin(tmp_path / "short.bin", [1, 2], n=3),
        write_bin(tmp_path / "long.bin", [1, 2, 3], n=2),
        write_bin(tmp_path / "vocab.bin", [1, 256], vocab=256),
    ]
    (tmp_path / "tiny.bin").write_bytes(b"\x88\xd8\x34\x01")
    bad.append(str(tmp_path / "tiny.bin"))
    for p in bad:
        with pytest.raises(ValueError):
            open_tokens(p)
    assert open_tokens(write_bin(tmp_path / "zero.bin", [1, 70], vocab=0)).tolist() == [
        1,
        70,
    ]


def test_windows_tile_each_pass(tmp_path):
    # WHY: within a pass the windows start T apart, so every token after the
    #      phase is a target exactly once: no overlap (double weight) and no
    #      gap (data never seen). Across passes the shards come in order and
    #      wrap around.
    # KIND: property
    # CATCHES: s25, s28
    # CHAPTER: L0.6 section 2, Principles (windows and passes)
    g = PCG32(seed())
    a = write_bin(tmp_path / "a.bin", [g.below(256) for _ in range(203)])
    b = write_bin(tmp_path / "b.bin", [g.below(256) for _ in range(97)])
    T = 8
    s = TokenStream([a, b], seq_len=T, batch=1, rng=Rng(seed()))
    starts: list[tuple[int, int]] = []
    for _ in range(80):
        s.next_batch()
        c = s.cursor()  # the window just read starts T before the next one
        starts.append((c["shard"], c["offset"] - T))
    # Group consecutive windows by shard: each run is one pass.
    runs: list[list[tuple[int, int]]] = []
    for sh, off in starts:
        if runs and runs[-1][-1][0] == sh and off == runs[-1][-1][1] + T:
            runs[-1].append((sh, off))
        else:
            runs.append([(sh, off)])
    sizes = {0: 203, 1: 97}
    for run in runs[:-1]:
        sh, first = run[0]
        assert 0 <= first < T
        last = run[-1][1]
        assert last + T + 1 <= sizes[sh] < last + 2 * T + 1, (sh, first, last)
    assert [r[0][0] for r in runs[:4]] == [0, 1, 0, 1]


def test_cursor_restore_is_bitwise(tmp_path):
    # WHY: the data half of a bitwise resume: N batches, save the cursor,
    #      restore it into a NEW stream (its own generator seeded
    #      differently), M batches: equal, element for element, to batches
    #      N + 1 to N + M of one uninterrupted stream. The generator state is
    #      part of the cursor because the next pass's phase comes from it.
    # KIND: property
    # CATCHES: s29
    # CHAPTER: L0.6 section 2, Principles (the cursor)
    g = PCG32(seed() + 1)
    a = write_bin(tmp_path / "a.bin", [g.below(300) for _ in range(150)], vocab=300)
    b = write_bin(tmp_path / "b.bin", [g.below(300) for _ in range(61)], vocab=300)
    one = TokenStream([a, b], seq_len=8, batch=4, rng=Rng(seed()))
    want = [one.next_batch() for _ in range(16)]
    two = TokenStream([a, b], seq_len=8, batch=4, rng=Rng(seed()))
    for _ in range(7):
        two.next_batch()
    c = two.cursor()
    assert set(c) == {"shard", "offset", "rng"}
    three = TokenStream([a, b], seq_len=8, batch=4, rng=Rng(seed() + 99))
    three.restore(c)
    for k in range(7, 16):
        x, y = three.next_batch()
        assert (x == want[k][0]).all() and (y == want[k][1]).all(), k


def test_targets_are_inputs_shifted(tmp_path):
    # WHY: language-model targets are the next token: targets[b, t] ==
    #      inputs[b, t + 1] inside every window, and the arrays are fresh
    #      int64 copies (a view into the memmap would pin the file and break
    #      when the batch is modified).
    # KIND: property
    # CATCHES: s26, m05
    # CHAPTER: L0.6 section 2, Principles (windows and passes)
    g = PCG32(seed() + 2)
    p = write_bin(tmp_path / "a.bin", [g.below(256) for _ in range(500)])
    s = TokenStream([p], seq_len=16, batch=5, rng=Rng(seed()))
    for _ in range(10):
        x, y = s.next_batch()
        assert x.shape == y.shape == (5, 16) and x.flags.writeable and x.flags.owndata
        assert (y[:, :-1] == x[:, 1:]).all()


def test_stream_rejects_bad_args(tmp_path):
    # WHY: errors at construction, where the cause is visible: no shards, a
    #      bare path string (iterating it gives characters), seq_len or batch
    #      below 1, a shard too short for one window, or a shard whose
    #      vocabulary does not match the model's.
    # KIND: boundary
    # CATCHES: s24
    # CHAPTER: L0.6 section 4, The interface
    p = write_bin(tmp_path / "a.bin", list(range(20)))
    for shards, T, B in (([], 4, 1), (p, 4, 1), ([p], 0, 1), ([p], 4, 0), ([p], 20, 1)):
        with pytest.raises(ValueError):
            TokenStream(shards, seq_len=T, batch=B, rng=Rng(0))
    one = TokenStream([p], seq_len=19, batch=1, rng=Rng(0))  # exactly one window fits:
    for _ in range(3):  # the phase must come from below(min(T, n - T)) = below(1)
        assert one.next_batch()[0].tolist() == [list(range(19))]
    with pytest.raises(ValueError):
        TokenStream([p], seq_len=4, batch=1, rng=Rng(0), vocab_size=512)
    q = write_bin(tmp_path / "q.bin", [1, 2, 300, 4, 5, 6], vocab=0)
    with pytest.raises(ValueError):
        TokenStream([q], seq_len=2, batch=1, rng=Rng(0), vocab_size=256)


def test_restore_rejects_bad_cursor(tmp_path):
    # WHY: a cursor from another run (more shards, a longer shard) must fail
    #      loudly instead of reading past the end of a shard.
    # KIND: boundary
    # CATCHES: m06
    # CHAPTER: L0.6 section 4, The interface
    p = write_bin(tmp_path / "a.bin", list(range(20)))
    s = TokenStream([p], seq_len=4, batch=1, rng=Rng(0))
    good = s.cursor()
    for bad in (
        {**good, "shard": 1},
        {**good, "shard": -1},
        {**good, "offset": 21},
        {**good, "offset": -1},
    ):
        with pytest.raises(ValueError):
            s.restore(bad)
    s.restore({**good, "offset": 20})  # at the end: the next batch starts a new pass
    assert s.next_batch()[0].shape == (1, 4)
