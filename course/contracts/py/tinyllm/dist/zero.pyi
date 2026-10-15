# contracts/py/tinyllm/dist/zero.pyi (L11.3, optional): data parallelism and ZeRO stages 1 to 3
# chapter: content/courses/build-an-llm/p11-training-at-scale/03-ddp-and-zero.md
#
# Every rank (L11.2's Comm) runs the same model on its own slice of the batch.
# DDP keeps a full replica per rank and averages the gradients with one
# all-reduce. ZeRO (Rajbhandari et al. 2020) removes the redundancy: the
# parameters, flattened in parameters() order into one vector of n entries,
# are split by L11.2's chunk_bounds(n, world), and rank r owns chunk r:
#
#   stage 1  optimizer state sharded: all-reduce (mean) the full gradient,
#            update the owned chunk, all-gather the parameters
#   stage 2  + gradients sharded: reduce-scatter (sum, then / world) gives
#            each rank only its chunk's mean gradient; the full .grad
#            arrays are dropped (None) at the step
#   stage 3  + parameters sharded: between steps a rank holds only its
#            chunk; gather() rebuilds the full parameters before forward
#
# The update itself is the wrapped optimizer's, run on one flat
# "parameter" per rank: an object with `data` (the owned chunk, the rank's
# master copy) and `grad`. Elementwise optimizers (SGD, AdamW) then give
# exactly the update a single process computes for those entries.
from typing import Any, Iterable, Literal

from tinyllm.autograd.tensor import Tensor
from tinyllm.dist.comm import Comm
from tinyllm.nn.module import Module

class DDP(Module):
    module: Module

    def __init__(self, model: Module, comm: Comm) -> None:
        """Wrap model (registered as the child `module`) and broadcast every
        parameter from rank 0 in place, so all replicas start equal."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """module(*args, **kwargs)."""

    def sync_grads(self) -> None:
        """Replace every parameter's .grad with the mean over ranks: the
        gradients, flattened in parameters() order into one bucket (a
        parameter without a gradient contributes zeros), all-reduced with
        op "mean", and written back with each parameter's shape and dtype.
        Every rank ends with bit-identical gradients."""

class ZeroOptimizer:
    stage: int

    def __init__(
        self, opt_cls: Any, params: Iterable[Tensor], comm: Comm, stage: Literal[1, 2, 3], **kw: Any
    ) -> None:
        """Flatten params (float64 master, the parameters' common dtype for
        the write-back), keep this rank's chunk as the master copy, and build
        opt_cls([that flat parameter], **kw). Stage 3 then releases the
        parameters (see step). ValueError for a stage outside 1..3 or no params."""

    def gather(self) -> None:
        """Stage 3: all-gather the chunks and give every parameter its full
        data again (a new array of its shape and dtype). A no-op in stages 1
        and 2. Call it before each forward pass."""

    def step(self) -> None:
        """Reduce the gradients as the stage says (mean over ranks), update
        the owned chunk with the wrapped optimizer, then: stages 1 and 2
        all-gather the parameters into every rank's model; stage 3 keeps
        only the chunk and releases every parameter's data (a zero-size
        array of its dtype) until the next gather(). Stage 2 and 3 set every
        parameter's .grad to None."""

    def zero_grad(self) -> None:
        """Every parameter's .grad and the chunk's grad to None."""

    def memory_bytes(self) -> dict[str, int]:
        """What this rank holds now, in bytes: {"params": the parameters'
        data arrays (in stage 3 between steps, the owned chunk instead),
        "grads": the parameters' .grad arrays plus the chunk's gradient,
        "optimizer": every ndarray in the wrapped optimizer's state_dict()
        plus, in stages 1 and 2, the chunk's master copy}."""
