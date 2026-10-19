"""Emulated mixed precision, dynamic loss scaling, and gradient accumulation (L11.1).

bf16 and fp16 are emulated by rounding float arrays to the values those
formats can hold (M09.1). `autocast` makes every Tensor matmul round its
operands and its output while the parameters stay full-precision master
weights. The loss scaler keeps fp16 gradients out of the underflow range and
skips steps that overflow. Accumulation weights each micro-batch by its share
of the rows, so k micro-batches give the gradient of one big batch.

Contract: contracts/py/tinyllm/train/precision.pyi.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

from tinyllm.autograd.tensor import Tensor, from_op
from tinyllm.nn.module import Module
from tinyllm.num.fp import round_to_bf16, round_to_fp16
from tinyllm.optim.schedule import clip_grad_norm_

_ROUND = {"bf16": round_to_bf16, "fp16": round_to_fp16}
_stack: list[str] = []  # open autocast blocks, innermost last
_orig: dict[str, Any] = {}  # Tensor's own matmul methods while patched


def _round(a: np.ndarray, dtype: str) -> np.ndarray:
    """a rounded to dtype, in a's own float dtype."""
    return np.asarray(_ROUND[dtype](a)).astype(a.dtype, copy=False)


def _check(dtype: str) -> None:
    if dtype not in _ROUND:
        raise ValueError(f"dtype must be 'bf16' or 'fp16', got {dtype!r}")


def cast(x: Tensor, dtype: str) -> Tensor:
    _check(dtype)
    return from_op(_round(x.data, dtype), (x,), lambda g: (_round(np.asarray(g), dtype),), f"cast_{dtype}")


def _as_tensor(v: Any, like: Tensor) -> Tensor:
    return v if isinstance(v, Tensor) else Tensor(v, dtype=like.data.dtype)


def _matmul(a: Tensor, b: Any) -> Tensor:
    """a @ b with rounded operands and output (the patched __matmul__)."""
    dt = _stack[-1]
    b = _as_tensor(b, a)
    return cast(_orig["mm"](cast(a, dt), cast(b, dt)), dt)


def _rmatmul(a: Tensor, b: Any) -> Tensor:
    """b @ a with rounded operands and output (the patched __rmatmul__)."""
    dt = _stack[-1]
    b = _as_tensor(b, a)
    return cast(_orig["mm"](cast(b, dt), cast(a, dt)), dt)


@contextmanager
def autocast(dtype: str) -> Iterator[None]:
    _check(dtype)
    if not _stack:
        _orig["mm"], _orig["rmm"] = Tensor.__matmul__, Tensor.__rmatmul__
        Tensor.__matmul__, Tensor.__rmatmul__ = _matmul, _rmatmul
    _stack.append(dtype)
    try:
        yield
    finally:
        _stack.pop()
        if not _stack:
            Tensor.__matmul__, Tensor.__rmatmul__ = _orig.pop("mm"), _orig.pop("rmm")


def autocast_bf16():
    return autocast("bf16")


def autocast_dtype() -> Optional[str]:
    return _stack[-1] if _stack else None


class DynamicLossScaler:
    def __init__(
        self, init: float = 2.0**16, growth: float = 2.0, backoff: float = 0.5, interval: int = 2000
    ) -> None:
        if not (init > 0 and growth > 1 and 0 < backoff < 1 and int(interval) >= 1):
            raise ValueError("need init > 0, growth > 1, 0 < backoff < 1, interval >= 1")
        self.loss_scale = float(init)
        self.growth, self.backoff, self.interval = float(growth), float(backoff), int(interval)
        self.good_steps = 0
        self.last_grad_norm: Optional[float] = None

    def scale(self, loss: Tensor) -> Tensor:
        return loss * self.loss_scale

    def step(self, opt: Any, params: Iterable[Any], clip: Optional[float] = None) -> bool:
        ps = [p for p in params if p.grad is not None]
        if not all(np.isfinite(p.grad).all() for p in ps):
            # Overflow: an inf or nan anywhere poisons the update. Skip it,
            # drop the bad gradients, and try a smaller scale next time.
            for p in ps:
                p.grad = None
            self.loss_scale *= self.backoff
            self.good_steps = 0
            return False
        inv = 1.0 / self.loss_scale
        for p in ps:
            p.grad *= inv  # unscale in place before clipping: the clip bound is in true units
        self.last_grad_norm = float(clip_grad_norm_(ps, clip)) if clip is not None else None
        opt.step()
        self.good_steps += 1
        if self.good_steps == self.interval:
            self.loss_scale *= self.growth
            self.good_steps = 0
        return True

    def state_dict(self) -> dict:
        return {
            "loss_scale": self.loss_scale,
            "growth": self.growth,
            "backoff": self.backoff,
            "interval": self.interval,
            "good_steps": self.good_steps,
        }

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        self.loss_scale = float(sd["loss_scale"])
        self.growth, self.backoff = float(sd["growth"]), float(sd["backoff"])
        self.interval, self.good_steps = int(sd["interval"]), int(sd["good_steps"])


def _split(out: Any) -> Tensor:
    """The loss of loss_fn's result: a Tensor, or the first of a (loss, metrics) pair."""
    loss = out[0] if isinstance(out, tuple) else out
    if np.asarray(loss.data).size != 1:
        raise ValueError(f"loss_fn must return a one-element loss, got shape {loss.shape}")
    return loss


def _size(mb: Mapping[str, ArrayLike]) -> int:
    return len(np.asarray(next(iter(mb.values()))))


def grad_accumulate(
    model: Module,
    micro_batches: Sequence[Mapping[str, ArrayLike]],
    loss_fn: Callable[..., Any],
    scaler: Optional[DynamicLossScaler] = None,
) -> float:
    mbs = list(micro_batches)
    if not mbs:
        raise ValueError("grad_accumulate needs at least one micro-batch")
    sizes = [_size(mb) for mb in mbs]
    if min(sizes) < 1:
        raise ValueError(f"every micro-batch needs at least one row, got sizes {sizes}")
    total = sum(sizes)
    out = 0.0
    for mb, n in zip(mbs, sizes):
        loss = _split(loss_fn(model, mb))
        w = n / total  # a mean over n rows times n / N is that rows' share of the big mean
        part = loss * w
        (scaler.scale(part) if scaler is not None else part).backward()
        out += w * float(np.asarray(loss.data).reshape(()))
    return out


@contextmanager
def _precision(precision: str) -> Iterator[None]:
    if precision == "fp32":
        yield
        return
    with autocast(precision):
        yield


def train_step_mixed(
    model: Module,
    micro_batches: Sequence[Mapping[str, ArrayLike]],
    loss_fn: Callable[..., Any],
    opt: Any,
    precision: str = "fp32",
    scaler: Optional[DynamicLossScaler] = None,
    clip: Optional[float] = None,
) -> dict[str, float]:
    if precision not in ("fp32", "bf16", "fp16"):
        raise ValueError(f"precision must be fp32, bf16, or fp16, got {precision!r}")
    opt.zero_grad()
    with _precision(precision):
        loss = grad_accumulate(model, micro_batches, loss_fn, scaler)
    stats = {"loss": loss, "skipped": 0.0, "scale": 1.0}
    if scaler is not None:
        ok = scaler.step(opt, model.parameters(), clip)
        stats["skipped"] = 0.0 if ok else 1.0
        stats["scale"] = scaler.loss_scale
        if ok and clip is not None:
            stats["grad_norm"] = scaler.last_grad_norm
        return stats
    if not math.isfinite(loss):
        raise FloatingPointError(f"the loss is {loss}; no parameter was updated")
    if clip is not None:
        stats["grad_norm"] = float(clip_grad_norm_(model.parameters(), clip))
    opt.step()
    return stats
