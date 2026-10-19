"""Activation checkpointing (L11.1).

`checkpoint` runs a function without recording its graph and keeps only its
inputs; the node it returns reruns the function with the graph on when the
backward pass reaches it. Replaying each random generator from the state it
had in the forward pass makes the rerun draw the same dropout masks, so the
gradients are bit for bit those of the plain run. `checkpoint_sequential`
applies M08.4's schedule to a stack of layers.

Contract: contracts/py/tinyllm/train/recompute.pyi.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

from tinyllm.autograd.hvp import checkpoint_schedule
from tinyllm.autograd.mode import is_grad_enabled, no_grad
from tinyllm.autograd.tensor import Tensor, from_op
from tinyllm.nn.module import Module


def checkpoint(
    fn: Callable[..., Tensor],
    *args: Any,
    params: Sequence[Tensor] = (),
    rngs: Sequence[Any] = (),
) -> Tensor:
    # each distinct Tensor once: a tensor passed twice gets one summed gradient
    tensors = list({id(a): a for a in args if isinstance(a, Tensor)}.values())
    parents = tensors + list(params)
    if not is_grad_enabled():
        return fn(*args)
    before = [r.state() for r in rngs]
    with no_grad():
        out = fn(*args)  # no graph: the activations inside fn are freed now

    def vjp(g: np.ndarray) -> list[Optional[np.ndarray]]:
        if not is_grad_enabled():
            raise RuntimeError("checkpoint: backward through a checkpoint needs grad mode on")
        after = [r.state() for r in rngs]
        for r, s in zip(rngs, before):
            r.set_state(s)  # replay the same draws (dropout masks) as the forward pass
        leaves = {
            id(a): Tensor(a.data, requires_grad=a.requires_grad, dtype=a.data.dtype) for a in tensors
        }
        try:
            y = fn(*[leaves[id(a)] if isinstance(a, Tensor) else a for a in args])
            if y.requires_grad:
                y.backward(g)  # parameters inside fn get their .grad here
        finally:
            for r, s in zip(rngs, after):
                r.set_state(s)  # the stream continues where the forward pass left it
        grads: list[Optional[np.ndarray]] = []
        for a in tensors:
            leaf = leaves[id(a)]
            grads.append(None if leaf.grad is None else leaf.grad)
        return grads + [None] * len(params)

    # from_op records the node only when some parent requires grad: a
    # parameter fn reads but `params` omits gets no gradient when the input
    # is data, the classic checkpointing bug.
    return from_op(out.data, parents, vjp, "checkpoint")


def _segment(layers: Sequence[Module]) -> Callable[[Tensor], Tensor]:
    """The function applying layers in order."""
    def run(x: Tensor) -> Tensor:
        for layer in layers:
            x = layer(x)
        return x

    return run


def checkpoint_sequential(
    layers: Sequence[Module], x: Tensor, mem_budget_layers: int, rngs: Sequence[Any] = ()
) -> Tensor:
    layers = list(layers)
    if not layers:
        raise ValueError("checkpoint_sequential needs at least one layer")
    starts = checkpoint_schedule(len(layers), mem_budget_layers)
    if not is_grad_enabled():
        return _segment(layers)(x)
    ends = starts[1:] + [len(layers)]
    for i, (a, b) in enumerate(zip(starts, ends)):
        seg = layers[a:b]
        if i == len(starts) - 1:
            x = _segment(seg)(x)  # the last segment keeps its activations: no rerun
        else:
            params = [p for m in seg for p in m.parameters()]
            x = checkpoint(_segment(seg), x, params=params, rngs=rngs)
    return x
