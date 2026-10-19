"""Data loader, train step, eval loop (L0.5).

The smallest training loop that every later model reuses: a seeded loader
(PCG32 shuffle, M06.3), a step that zeroes, backpropagates, clips (M10.4),
and updates (an M10.2 optimizer), and an evaluation that runs in eval mode
under no_grad and restores the mode afterwards.

Contract: contracts/py/tinyllm/train/loop.pyi.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterator, Mapping, Optional

import numpy as np
from numpy.typing import NDArray

from tinyllm.autograd.mode import no_grad
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module
from tinyllm.num.ema import EMA
from tinyllm.optim.schedule import clip_grad_norm_


class DataLoader:
    def __init__(
        self,
        arrays: Mapping[str, NDArray],
        batch_size: int,
        shuffle: bool,
        rng: Any,
        drop_last: bool = True,
    ) -> None:
        if not arrays:
            raise ValueError("DataLoader needs at least one array")
        self.arrays = {k: np.asarray(v) for k, v in arrays.items()}
        lengths = {len(v) for v in self.arrays.values()}
        if len(lengths) != 1:
            raise ValueError(f"arrays have different lengths: {sorted(lengths)}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if shuffle and rng is None:
            raise ValueError("shuffle needs an rng (a PCG32)")
        self.n = lengths.pop()
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.rng = rng
        self.drop_last = bool(drop_last)

    def __iter__(self) -> Iterator[dict[str, NDArray]]:
        order = list(range(self.n))
        if self.shuffle:
            # Drawn when the epoch starts: each epoch advances the generator,
            # so epoch 2 differs from epoch 1 and a seeded run repeats exactly.
            self.rng.shuffle(order)
        idx = np.asarray(order, dtype=np.int64)
        b = self.batch_size
        stop = (self.n // b) * b if self.drop_last else self.n
        for i in range(0, stop, b):
            rows = idx[i : i + b]
            yield {k: v[rows] for k, v in self.arrays.items()}

    def __len__(self) -> int:
        return self.n // self.batch_size if self.drop_last else math.ceil(self.n / self.batch_size)


def _split(out: Any) -> tuple[Tensor, dict[str, float]]:
    """loss_fn's result as (loss, extra metrics)."""
    loss, extra = (out[0], dict(out[1])) if isinstance(out, tuple) else (out, {})
    if not isinstance(loss, Tensor):
        raise TypeError(f"loss_fn must return a Tensor (or a (Tensor, dict) pair), got {type(loss).__name__}")
    if loss.data.size != 1:
        raise ValueError(f"the loss must have one element, got shape {loss.shape}")
    return loss, {k: float(v) for k, v in extra.items()}


def train_step(
    model: Module,
    batch: Mapping[str, NDArray],
    loss_fn: Callable[..., Any],
    opt: Any,
    clip: Optional[float] = None,
    ema: Optional[EMA] = None,
) -> dict[str, float]:
    # Zero first: gradients accumulate across backward calls (L0.1).
    opt.zero_grad()
    loss, extra = _split(loss_fn(model, batch))
    value = float(np.asarray(loss.data).reshape(()))
    if not math.isfinite(value):
        # Stop before the update: one nan step would poison every weight.
        raise FloatingPointError(f"the loss is {value}; no parameter was updated")
    loss.backward()
    stats = {"loss": value}
    if clip is not None:
        stats["grad_norm"] = float(clip_grad_norm_(model.parameters(), clip))
    opt.step()
    stats.update(extra)
    if ema is not None:
        # One step's loss is noisy; the curve to read is its EMA (M02.2),
        # debiased so the first steps are not pulled toward m_0 = 0.
        ema.update(value)
        stats["loss_ema"] = float(ema.value_debiased())
    return stats


def evaluate(model: Module, loader: Any, loss_fn: Callable[..., Any]) -> dict[str, float]:
    was_training = model.training
    model.eval()
    sums: dict[str, float] = {}
    n = 0
    try:
        with no_grad():
            for batch in loader:
                loss, extra = _split(loss_fn(model, batch))
                rows = len(next(iter(batch.values())))
                # Weight by rows: a smaller last batch must count less.
                for k, v in {"loss": float(np.asarray(loss.data).reshape(())), **extra}.items():
                    sums[k] = sums.get(k, 0.0) + v * rows
                n += rows
    finally:
        model.train(was_training)
    if n == 0:
        raise ValueError("evaluate: the loader produced no batches")
    out = {k: v / n for k, v in sums.items()}
    out["n"] = float(n)
    return out
