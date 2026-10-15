# contracts/py/tinyllm/io/tokens.pyi (L0.6): the token-stream reader
# chapter: content/courses/build-an-llm/p00-foundations/06-safetensors-checkpoints-and-token-streams.md
#
# Shards are formats/tokens-bin.md files (llm.c .bin): a 1024-byte header of
# 256 little-endian int32 [magic 20240520, version, n_tokens, vocab_size],
# then n_tokens ids, uint16 (version 1) or uint32 (version 2). data.07 writes
# them; this module reads them through np.memmap, without loading a shard.
#
# The stream. T = seq_len. A window is T + 1 consecutive ids at some offset:
# inputs are its first T, targets its last T. The stream reads shard 0, then
# 1, ..., then wraps to 0. Each pass over a shard (including the first, at
# construction) starts at offset rng.below(min(T, n - T)) of that shard, and
# successive windows start T apart (each window's last id is the next one's
# first), until a window would run past the end of the shard; then the next
# pass begins. One batch is B windows in this order. rng is a PCG32 (M06.3):
# below(n), state(), set_state(s); the stream advances it only at the start
# of a pass.
#
# The cursor is the whole position: {"shard": i, "offset": the offset of the
# next window in shard i (it may be past the last full window: the pass
# ends at the next draw), "rng": rng.state()}. Restoring a cursor into a
# stream over the same shards continues bit for bit: N batches, cursor,
# restore in a new stream, M batches equals N + M batches of one stream.
from typing import Any, Optional, Sequence

from numpy.typing import NDArray

MAGIC: int  # 20240520
HEADER_BYTES: int  # 1024

def read_tokens_header(path: str) -> dict[str, int]:
    """{"version", "n_tokens", "vocab_size"}. ValueError for a file shorter
    than the header, a wrong magic, a version other than 1 or 2, a negative
    count, or a file size other than 1024 + n_tokens * (2 or 4)."""

def open_tokens(path: str) -> NDArray:
    """The ids as a read-only np.memmap (dtype <u2 or <u4, shape [n_tokens]);
    an empty array when n_tokens is 0. ValueError as read_tokens_header,
    and for an id >= vocab_size when the header's vocab_size is not 0."""

class TokenStream:
    def __init__(
        self,
        shards: Sequence[str],
        seq_len: int,
        batch: int,
        rng: Any,
        vocab_size: Optional[int] = None,
    ) -> None:
        """Open every shard (open_tokens) and draw the first pass's phase.
        With vocab_size, a shard whose header names another non-zero
        vocab_size, or holding an id >= vocab_size, is a ValueError.
        ValueError for no shards, seq_len or batch < 1, or a shard shorter
        than seq_len + 1 ids."""

    def next_batch(self) -> tuple[NDArray, NDArray]:
        """(inputs, targets), int64 [batch, seq_len] each, fresh arrays;
        targets[b, t] == inputs[b, t + 1] within a window."""

    def cursor(self) -> dict:
        """{"shard": int, "offset": int, "rng": (state, inc)}."""

    def restore(self, cursor: dict) -> None:
        """Continue from a cursor of a stream over the same shards and
        seq_len. ValueError for a shard index or offset outside the shards."""
