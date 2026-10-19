# contracts/py/tinyllm/autograd/losses.pyi (L0.3): losses with a fused backward
# chapter: content/courses/build-an-llm/01-foundations/04-losses-with-fused-backward.md
#
# Each loss is ONE graph node: the forward computes the loss from the logits
# with stable numerics (tinyllm.num.stable, M09.2), and the backward is the
# closed-form gradient (M08.3 derives it), never a chain of log, softmax, and
# gather nodes. Values and gradients match torch.nn.functional (golden tests).
# float32 logits give a float32 loss and float32 gradients.
#
# Reductions: "mean" (default), "sum", or "none" (one loss per row or element).
from typing import Literal, Optional

from numpy.typing import ArrayLike

from tinyllm.autograd.tensor import Tensor

def cross_entropy(
    logits: Tensor,
    targets: ArrayLike,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """Softmax cross-entropy over the last axis. logits [..., V], targets
    integers of shape logits.shape[:-1]. With eps = label_smoothing, row i's
    loss is (1 - eps) * (-log p_i[t_i]) + eps * mean_j(-log p_i[j]), where
    p_i = softmax(logits_i); its gradient is p_i - q_i with
    q_i = (1 - eps) * onehot(t_i) + eps / V.
    Rows whose target equals ignore_index contribute nothing: no loss, no
    gradient, and they do not count in the mean (which divides by the number
    of kept rows). When every row is ignored the mean is 0.0 with a zero
    gradient (torch returns nan). "none" returns targets.shape, ignored rows 0.
    ValueError for a target outside [0, V) other than ignore_index, a
    non-integer target, a shape mismatch, eps outside [0, 1], or an unknown
    reduction."""

def mse(pred: Tensor, target: ArrayLike) -> Tensor:
    """mean((pred - target)^2) over every element; target has pred's shape.
    ValueError for a shape mismatch."""

def bce_with_logits(
    logits: Tensor, targets: ArrayLike, pos_weight: Optional[ArrayLike] = None
) -> Tensor:
    """Mean binary cross-entropy on logits x with targets y in [0, 1]:
    -(w y log sigmoid(x) + (1 - y) log(1 - sigmoid(x))), w = pos_weight
    (broadcast over the last axis, default 1), computed as
    (1 - y) x + (1 + (w - 1) y) softplus(-x) so no exp overflows at |x| = 1000.
    Gradient (1 - y) - (1 + (w - 1) y) sigmoid(-x), divided by the count.
    ValueError for a shape mismatch."""
