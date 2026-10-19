# contracts/py/tinyllm/train/recompute.pyi (L11.1): activation checkpointing
# chapter: content/courses/build-an-llm/12-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md
#
# Trade compute for memory (M08.4's cost model): run a piece of the network
# without recording its graph, keep only its inputs, and rerun it with the
# graph on during backward. Gradients come out bit for bit the same as
# without checkpointing, because the recomputation performs the same
# operations on the same values, random draws included.
from typing import Any, Callable, Sequence

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module

def checkpoint(
    fn: Callable[..., Tensor],
    *args: Any,
    params: Sequence[Tensor] = (),
    rngs: Sequence[Any] = (),
) -> Tensor:
    """fn(*args), keeping only the args. Forward: record each rng's state()
    (a PCG32, M06.3), then run fn under no_grad. The result is one op node
    (from_op, L0.1) whose parents are the Tensor args followed by `params`,
    the parameters fn reads: listing them records the node even when no arg
    requires grad (the first layer's input is data). Backward of that node:
    save each rng's current state, set the recorded one, rerun fn with grad
    mode on over fresh leaf copies of the Tensor args (requires_grad as the
    originals), backpropagate the upstream gradient through the rerun (which
    adds the parameters' gradients straight into their .grad), set each
    rng's saved state back, and return the args' gradients (None for the
    params, which already have theirs). A Tensor passed twice is one parent.
    Non-Tensor args are passed through. When grad mode is off it is plain
    fn(*args). When no parent requires grad the result is a constant (no
    node), so a parameter fn reads that `params` does not list gets no
    gradient when no arg requires grad: list them. RuntimeError when the
    node's backward runs with grad mode off."""

def checkpoint_sequential(
    layers: Sequence[Module], x: Tensor, mem_budget_layers: int, rngs: Sequence[Any] = ()
) -> Tensor:
    """Apply layers in order, as M08.4's checkpoint_schedule(len(layers),
    mem_budget_layers) plans: every segment but the last through
    checkpoint(..., params=the segment's parameters, rngs=rngs), the last
    one normally. So layers outside the last segment run forward twice
    (checkpoint_cost's recomputed count) and the peak of saved layer inputs
    stays within the budget. Under no_grad, every layer runs once.
    ValueError for no layers, or as checkpoint_schedule."""
