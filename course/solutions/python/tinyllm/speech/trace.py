"""Strict recorded-logit playback for Python/Rust ASR policy parity (L14.4)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


class RecordedLogits:
    """Rows keyed by (10 ms seek frame, complete decoder prefix).

    NPZ v1 stores prefix_offsets[R+1], prefix_ids[N], optional seeks[R], and
    either logits[R,V] or token_ids[R,K], values[R,K], vocab_size[]. Missing
    sparse entries have logit -inf. No absent prefix is silently extrapolated.
    """

    def __init__(self, path: str, *, context_size: int = 448) -> None:
        if type(context_size) is not int or context_size < 2:
            raise ValueError("trace context_size must be an integer >= 2")
        self.context_size = context_size
        self.rows = {}
        with np.load(Path(path), allow_pickle=False) as data:
            if "prefix_offsets" not in data or "prefix_ids" not in data:
                raise ValueError("trace requires prefix_offsets and prefix_ids")
            offsets, ids = data["prefix_offsets"], data["prefix_ids"]
            if (
                offsets.ndim != 1
                or ids.ndim != 1
                or len(offsets) < 2
                or not np.issubdtype(offsets.dtype, np.integer)
                or not np.issubdtype(ids.dtype, np.integer)
            ):
                raise ValueError("trace prefixes must be integer vectors")
            count = len(offsets) - 1
            if (
                offsets[0] != 0
                or offsets[-1] != len(ids)
                or np.any(offsets[1:] <= offsets[:-1])
            ):
                raise ValueError(
                    "trace prefix offsets must partition nonempty prefixes"
                )
            seeks = (
                data["seeks"] if "seeks" in data else np.zeros(count, dtype=np.int64)
            )
            if (
                seeks.shape != (count,)
                or not np.issubdtype(seeks.dtype, np.integer)
                or np.any(seeks < 0)
            ):
                raise ValueError("trace seeks must be nonnegative integer frames")
            if "logits" in data:
                if any(key in data for key in ("token_ids", "values", "vocab_size")):
                    raise ValueError("trace cannot mix dense and sparse logits")
                values = data["logits"]
                if values.ndim != 2 or values.shape[0] != count or values.shape[1] < 1:
                    raise ValueError("dense trace logits must have shape [rows, vocab]")
                vocab = values.shape[1]
                sparse_ids = None
            else:
                if not all(
                    key in data for key in ("token_ids", "values", "vocab_size")
                ):
                    raise ValueError(
                        "sparse trace requires token_ids, values and vocab_size"
                    )
                scalar = data["vocab_size"]
                if (
                    scalar.shape != ()
                    or not np.issubdtype(scalar.dtype, np.integer)
                    or not 1 <= int(scalar) <= 1000000
                ):
                    raise ValueError(
                        "trace vocabulary must be an integer in [1, 1000000]"
                    )
                vocab = int(scalar)
                sparse_ids, values = data["token_ids"], data["values"]
                if (
                    values.ndim != 2
                    or values.shape[0] != count
                    or values.shape[1] < 1
                    or sparse_ids.shape != values.shape
                    or not np.issubdtype(sparse_ids.dtype, np.integer)
                    or np.any(sparse_ids < 0)
                    or np.any(sparse_ids >= vocab)
                ):
                    raise ValueError(
                        "sparse trace ids and values must have matching [rows, k] shapes"
                    )
                if any(len(set(row.tolist())) != len(row) for row in sparse_ids):
                    raise ValueError("sparse trace rows cannot repeat token ids")
            if (
                not np.issubdtype(values.dtype, np.floating)
                or np.any(np.isnan(values))
                or np.any(np.isposinf(values))
                or not np.isfinite(values).any(axis=1).all()
            ):
                raise ValueError(
                    "trace logits must be floating, without NaN/+inf, with a finite choice per row"
                )
            if (
                np.any(ids < 0)
                or np.any(ids >= vocab)
                or np.any(np.diff(offsets) > context_size)
            ):
                raise ValueError("trace prefix exceeds the vocabulary or context")
            self.vocab_size = vocab
            for row in range(count):
                key = (
                    int(seeks[row]),
                    tuple(int(t) for t in ids[offsets[row] : offsets[row + 1]]),
                )
                if key in self.rows:
                    raise ValueError("duplicate seek/prefix in trace")
                # Keep top-k recordings sparse in memory; expand only queried rows.
                self.rows[key] = (
                    None if sparse_ids is None else sparse_ids[row].copy(),
                    values[row].copy(),
                )

    def at_seek(self, seek: int):
        if type(seek) is not int or seek < 0:
            raise ValueError("seek must be a nonnegative integer frame")
        recording = self

        class Window:
            context_size = recording.context_size

            def step(self, prefix: Sequence[int]):
                if any(not isinstance(t, (int, np.integer)) for t in prefix):
                    raise ValueError("trace prefix must contain integer ids")
                key = (seek, tuple(int(t) for t in prefix))
                if key not in recording.rows:
                    raise ValueError(
                        f"trace has no logits for seek={seek}, prefix={key[1]}"
                    )
                ids, values = recording.rows[key]
                if ids is None:
                    return values.copy()
                row = np.full(recording.vocab_size, -np.inf, dtype=np.float64)
                row[ids] = values
                return row

        return Window()
