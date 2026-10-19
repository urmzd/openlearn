"""The Optimizer protocol and SGD with momentum, Nesterov, weight decay (M10.2).

An optimizer owns references to the parameters and whatever state it needs
between steps (here one momentum buffer per parameter). The protocol keeps
training loops independent of the optimizer, and state_dict makes a run
resumable bit for bit.

Contract: contracts/py/tinyllm/optim/sgd.pyi.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class Param(Protocol):
    data: NDArray
    grad: NDArray | None


@runtime_checkable
class Optimizer(Protocol):
    def step(self) -> None: ...

    def zero_grad(self) -> None: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, sd: dict) -> None: ...


class SGD:
    """torch.optim.SGD semantics with dampening 0 (see the contract)."""

    def __init__(
        self,
        params: Iterable[Any],
        lr: float,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if momentum < 0:
            raise ValueError(f"momentum must be >= 0, got {momentum}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
        if nesterov and momentum == 0:
            raise ValueError("nesterov needs momentum > 0")
        self.params = list(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.weight_decay = float(weight_decay)
        self.buffers: dict[int, NDArray] = {}

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = np.asarray(p.grad, dtype=p.data.dtype)
            if self.weight_decay != 0:
                g = g + self.weight_decay * p.data
            if self.momentum != 0:
                buf = self.buffers.get(i)
                if buf is None:
                    buf = np.array(g, copy=True)
                else:
                    buf *= self.momentum
                    buf += g
                self.buffers[i] = buf
                g = g + self.momentum * buf if self.nesterov else buf
            p.data -= self.lr * g

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def state_dict(self) -> dict:
        return {
            "state": {i: {"momentum_buffer": b.copy()} for i, b in self.buffers.items()},
            "param_groups": [
                {
                    "lr": self.lr,
                    "momentum": self.momentum,
                    "nesterov": self.nesterov,
                    "weight_decay": self.weight_decay,
                    "params": list(range(len(self.params))),
                }
            ],
        }

    def load_state_dict(self, sd: dict) -> None:
        (group,) = sd["param_groups"]
        if len(group["params"]) != len(self.params):
            raise ValueError(
                f"state_dict has {len(group['params'])} parameters, this optimizer {len(self.params)}"
            )
        buffers = {}
        for i, st in sd["state"].items():
            i = int(i)
            b = np.array(st["momentum_buffer"], copy=True)
            if not 0 <= i < len(self.params) or b.shape != self.params[i].data.shape:
                raise ValueError(f"momentum buffer {i} does not match parameter shapes")
            buffers[i] = b
        self.lr = float(group["lr"])
        self.momentum = float(group["momentum"])
        self.nesterov = bool(group["nesterov"])
        self.weight_decay = float(group["weight_decay"])
        self.buffers = buffers
