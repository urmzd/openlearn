"""Data parallelism (DDP) and ZeRO stages 1 to 3 over L11.2's collectives (L11.3, optional).

DDP averages the gradients of identical replicas with one all-reduce. ZeRO
keeps one flat vector of all parameters, split into chunks by chunk_bounds:
rank r updates chunk r only (stage 1), receives only chunk r's gradient
(stage 2), and between steps stores only chunk r's parameters (stage 3).

Contract: contracts/py/tinyllm/dist/zero.pyi.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from tinyllm.autograd.tensor import Tensor
from tinyllm.dist.comm import Comm, chunk_bounds
from tinyllm.nn.module import Module


def _flat_grads(params: list) -> np.ndarray:
    """Every parameter's gradient (zeros when None), flattened in order, float64."""
    return np.concatenate(
        [
            np.zeros(p.data.size) if p.grad is None else np.asarray(p.grad, dtype=np.float64).reshape(-1)
            for p in params
        ]
    )


def _write_back(params: list, shapes: list, dtypes: list, flat: np.ndarray, attr: str) -> None:
    """Split flat into the parameters' shapes and set each one's `attr` (data or grad)."""
    at = 0
    for p, shape, dt in zip(params, shapes, dtypes):
        size = int(np.prod(shape)) if shape else 1
        piece = flat[at : at + size].reshape(shape).astype(dt)
        if attr == "data" and p.data.shape == shape:
            p.data[...] = piece  # in place: the arrays other code holds stay the same objects
        else:
            setattr(p, attr, piece)
        at += size


class DDP(Module):
    def __init__(self, model: Module, comm: Comm) -> None:
        super().__init__()
        self.module = model
        self._comm = comm
        for p in model.parameters():
            p.data[...] = comm.broadcast(p.data, 0)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def sync_grads(self) -> None:
        ps = list(self.module.parameters())
        mean = self._comm.all_reduce(_flat_grads(ps), op="mean")
        _write_back(ps, [p.data.shape for p in ps], [p.data.dtype for p in ps], mean, "grad")


class _Chunk:
    """The flat parameter the wrapped optimizer updates: this rank's chunk."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.grad: Optional[np.ndarray] = None


class ZeroOptimizer:
    def __init__(self, opt_cls: Any, params: Iterable[Tensor], comm: Comm, stage: int, **kw: Any) -> None:
        if stage not in (1, 2, 3):
            raise ValueError(f"ZeRO stage must be 1, 2, or 3, got {stage}")
        self.params = list(params)
        if not self.params:
            raise ValueError("ZeroOptimizer needs at least one parameter")
        self.stage, self.comm = stage, comm
        self.shapes = [p.data.shape for p in self.params]
        self.dtypes = [p.data.dtype for p in self.params]
        n = sum(p.data.size for p in self.params)
        self.bounds = chunk_bounds(n, comm.world)
        a, b = self.bounds[comm.rank]
        full = np.concatenate([np.asarray(p.data, dtype=np.float64).reshape(-1) for p in self.params])
        self.chunk = _Chunk(full[a:b].copy())
        self.opt = opt_cls([self.chunk], **kw)
        if stage == 3:
            self._release()

    def _release(self) -> None:
        for p, dt in zip(self.params, self.dtypes):
            p.data = np.zeros(0, dtype=dt)

    def gather(self) -> None:
        if self.stage != 3:
            return
        full = self.comm.all_gather(self.chunk.data)
        _write_back(self.params, self.shapes, self.dtypes, full, "data")

    def step(self) -> None:
        a, b = self.bounds[self.comm.rank]
        g = _flat_grads(self.params)
        if self.stage == 1:
            self.chunk.grad = self.comm.all_reduce(g, op="mean")[a:b].copy()
        else:
            self.chunk.grad = self.comm.reduce_scatter(g) / self.comm.world
            for p in self.params:
                p.grad = None  # the full gradient is not kept: only the chunk's
        self.opt.step()
        if self.stage == 3:
            self._release()
            return
        full = self.comm.all_gather(self.chunk.data)
        _write_back(self.params, self.shapes, self.dtypes, full, "data")

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None
        self.chunk.grad = None

    def memory_bytes(self) -> dict[str, int]:
        def nbytes(x: Any) -> int:
            if isinstance(x, np.ndarray):
                return x.nbytes
            if isinstance(x, dict):
                return sum(nbytes(v) for v in x.values())
            if isinstance(x, (list, tuple)):
                return sum(nbytes(v) for v in x)
            return 0

        params = sum(p.data.nbytes for p in self.params)
        if self.stage == 3:
            params += self.chunk.data.nbytes
        grads = sum(p.grad.nbytes for p in self.params if p.grad is not None)
        grads += self.chunk.grad.nbytes if self.chunk.grad is not None else 0
        optimizer = nbytes(self.opt.state_dict()) + (self.chunk.data.nbytes if self.stage < 3 else 0)
        return {"params": params, "grads": grads, "optimizer": optimizer}
