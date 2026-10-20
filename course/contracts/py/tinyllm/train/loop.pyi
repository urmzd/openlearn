# contracts/py/tinyllm/train/loop.pyi (L0.5): data loader, train step, eval loop
# chapter: content/courses/ai-engineering/06-numerics-and-autograd/06-training-loop-and-the-autograd-bigram.md
#
# The three pieces every training run in the course is made of:
#
#   loader = DataLoader({"x": X, "y": Y}, batch_size=32, shuffle=True, rng=rng)
#   for epoch in range(E):
#       for batch in loader:
#           stats = train_step(model, batch, loss_fn, opt, clip=1.0)
#   metrics = evaluate(model, val_loader, loss_fn)
#
# loss_fn(model, batch) returns a one-element loss Tensor (L0.1), or a pair
# (loss, {name: float}) of the loss and extra metrics (an accuracy). `opt` is
# an optimizer of the M10.2 protocol: zero_grad(), step(), over the model's
# parameters. `rng` is a PCG32 (M06.3), whose shuffle(xs) is spec/pcg32.md's
# Fisher-Yates.
from typing import Any, Callable, Iterator, Mapping, Optional

from numpy.typing import NDArray

from tinyllm.nn.module import Module
from tinyllm.num.ema import EMA

class DataLoader:
    def __init__(
        self,
        arrays: Mapping[str, NDArray],
        batch_size: int,
        shuffle: bool,
        rng: Any,
        drop_last: bool = True,
    ) -> None:
        """Batches of rows of every array, which share their first dimension
        n. ValueError for no arrays, arrays of different lengths,
        batch_size < 1, or shuffle without an rng."""

    def __iter__(self) -> Iterator[dict[str, NDArray]]:
        """One epoch. The order is 0..n-1, or, with shuffle, that list after
        rng.shuffle (a new permutation each epoch, so the epochs of a seeded
        run are reproducible). Each batch is {name: array[rows]}; a final
        partial batch is dropped when drop_last."""

    def __len__(self) -> int:
        """Batches per epoch: n // batch_size, or ceil(n / batch_size) without drop_last."""

def train_step(
    model: Module,
    batch: Mapping[str, NDArray],
    loss_fn: Callable[..., Any],
    opt: Any,
    clip: Optional[float] = None,
    ema: Optional[EMA] = None,
) -> dict[str, float]:
    """opt.zero_grad(); loss = loss_fn(model, batch); loss.backward(); with
    clip, clip_grad_norm_(model.parameters(), clip) (M10.4); opt.step().
    Returns {"loss": float, plus "grad_norm" (the norm before clipping) when
    clip is set, plus the extra metrics, plus "loss_ema" when an M02.2 EMA
    is passed: the step's loss goes into ema.update and "loss_ema" is
    ema.value_debiased(), the smoothed curve a training log plots}. A loss
    that is not finite raises FloatingPointError before any parameter
    changes (and before the EMA sees it); a loss with more than one element
    is a ValueError."""

def evaluate(model: Module, loader: Any, loss_fn: Callable[..., Any]) -> dict[str, float]:
    """Run loss_fn over every batch of loader in eval mode and under no_grad,
    then put the model back in the mode it was in. Returns the mean loss and
    the mean of each extra metric, weighted by batch size (the length of the
    batch's first array), and "n", the number of rows seen.
    ValueError for an empty loader."""
