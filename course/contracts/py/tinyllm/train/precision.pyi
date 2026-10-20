# contracts/py/tinyllm/train/precision.pyi (L11.1): emulated mixed precision,
# dynamic loss scaling, and gradient accumulation
# chapter: content/courses/ai-engineering/17-training-at-scale/01-mixed-precision-accumulation-and-checkpointing.md
#
# numpy has no bfloat16, so low precision is emulated: a value "stored in
# bf16" is a float array whose every entry is a bf16 value, made by M09.1's
# round_to_bf16 (round_to_fp16 for fp16). The parameters stay full-precision
# master weights; only the matmuls run on rounded operands, as under
# torch.autocast. `dtype` is "bf16" or "fp16" throughout.
#
# Words used below:
#   cast        the differentiable rounding op below: forward rounds the
#               data to dtype, backward rounds the incoming gradient to dtype
#               (the gradient of a cast is the cast of the gradient)
#   micro-batch one of the pieces a batch is split into; its size is the
#               length of the first array of its dict (as L0.5's evaluate)
from contextlib import AbstractContextManager
from typing import Any, Callable, Iterable, Literal, Mapping, Optional, Sequence

from numpy.typing import ArrayLike

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module

def cast(x: Tensor, dtype: Literal["bf16", "fp16"]) -> Tensor:
    """A Tensor of x's shape and dtype whose data is round_to_bf16(x.data)
    (or round_to_fp16) converted back to x's dtype; its vjp returns the
    upstream gradient rounded the same way. An op on x (from_op, L0.1), so
    it records a node only when x requires grad. ValueError for another dtype."""

def autocast(dtype: Literal["bf16", "fp16"]) -> AbstractContextManager[None]:
    """Context manager. Inside the block every Tensor matmul (a @ b, the
    reflected b @ a, F.matmul, Linear, attention) computes
        cast(cast(a, dtype) @ cast(b, dtype), dtype)
    with the float32 (or float64) product of the rounded operands: the
    inputs are rounded, the products accumulate in full precision, and the
    output is rounded, as a bf16 GEMM with an fp32 accumulator. A non-Tensor
    operand is a constant of the other operand's dtype. It works by wrapping
    tinyllm.autograd.tensor.Tensor's __matmul__ and __rmatmul__ when the
    outermost block is entered and putting the originals back when it exits,
    also when the block raises. Blocks nest; the innermost dtype applies and
    the outer one is restored on exit. Process-wide, not per thread.
    ValueError for another dtype."""

def autocast_bf16() -> AbstractContextManager[None]:
    """autocast("bf16")."""

def autocast_dtype() -> Optional[str]:
    """The dtype of the innermost open autocast block, None outside any."""

class DynamicLossScaler:
    """Dynamic loss scaling for fp16 (Micikevicius et al. 2018; torch's
    GradScaler without the per-device bookkeeping):

        loss = scaler.scale(loss); loss.backward()        # gradients times S
        stepped = scaler.step(opt, params, clip=1.0)      # unscale, check, clip, step, update

    A step with any non-finite gradient is skipped: no parameter changes,
    every gradient is set to None, S is multiplied by `backoff`, and the run
    of good steps restarts at 0. A finite step divides every gradient by S in
    place, clips (M10.4's clip_grad_norm_) when `clip` is given, calls
    opt.step(), and counts one good step; after `interval` good steps in a
    row S is multiplied by `growth` and the count restarts."""

    loss_scale: float  # the current S
    growth: float
    backoff: float
    interval: int
    good_steps: int  # good steps since the last change of S
    last_grad_norm: Optional[float]  # the norm clip_grad_norm_ returned on the last finite step with clip, else None

    def __init__(
        self, init: float = 2.0**16, growth: float = 2.0, backoff: float = 0.5, interval: int = 2000
    ) -> None:
        """ValueError unless init > 0, growth > 1, 0 < backoff < 1, interval >= 1."""

    def scale(self, loss: Tensor) -> Tensor:
        """loss * S (a Tensor op, so backward carries the factor into every gradient)."""

    def step(self, opt: Any, params: Iterable[Any], clip: Optional[float] = None) -> bool:
        """The update above over params (objects with `grad`, ndarray or
        None; the parameters `opt` updates). True when opt.step() ran, False
        when the step was skipped."""

    def state_dict(self) -> dict:
        """{"loss_scale", "growth", "backoff", "interval", "good_steps"}."""

    def load_state_dict(self, sd: Mapping[str, Any]) -> None:
        """Restore every key of state_dict(), so a resumed run scales and
        grows exactly as the uninterrupted one."""

def grad_accumulate(
    model: Module,
    micro_batches: Sequence[Mapping[str, ArrayLike]],
    loss_fn: Callable[..., Any],
    scaler: Optional[DynamicLossScaler] = None,
) -> float:
    """Backward every micro-batch into the parameters' .grad (added to what
    is there: nothing is zeroed and no step is taken). With N the total size
    and n_i the size of micro-batch i, the loss of i (loss_fn(model, mb),
    a one-element Tensor or a (loss, metrics) pair as in L0.5) is weighted
    by n_i / N before its backward, scaled by the scaler when given, so the
    gradients equal those of one batch of all N rows when loss_fn takes a
    mean over rows. Returns sum_i (n_i / N) loss_i, the big batch's loss.
    ValueError for no micro-batches or a micro-batch of size 0."""

def train_step_mixed(
    model: Module,
    micro_batches: Sequence[Mapping[str, ArrayLike]],
    loss_fn: Callable[..., Any],
    opt: Any,
    precision: Literal["fp32", "bf16", "fp16"] = "fp32",
    scaler: Optional[DynamicLossScaler] = None,
    clip: Optional[float] = None,
) -> dict[str, float]:
    """One optimizer step from accumulated micro-batches: opt.zero_grad();
    grad_accumulate under autocast(precision) (no autocast for "fp32"); then
    scaler.step(opt, model.parameters(), clip) when a scaler is given, else
    (as L0.5's train_step) a non-finite loss raises FloatingPointError before
    any update, clip_grad_norm_ when clip is set, and opt.step(). Returns
    {"loss", "skipped" (1.0 when the scaler skipped the step, else 0.0),
    "scale" (S, or 1.0 without a scaler), and "grad_norm" when clip is set
    and the step ran}. ValueError for an unknown precision."""
