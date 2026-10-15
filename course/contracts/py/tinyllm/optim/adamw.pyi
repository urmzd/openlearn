# contracts/py/tinyllm/optim/adamw.pyi (M10.3)
# chapter: content/tracks/math/10-optimization/03-adam-and-adamw.md
#
# Adam and AdamW over a list of parameters. A parameter is any object with
# two attributes: `data`, a float ndarray the optimizer updates IN PLACE, and
# `grad`, an ndarray of the same shape or None (an autograd Tensor from L0.1
# is one). Both classes follow the Optimizer protocol of M10.2: step,
# zero_grad, state_dict, load_state_dict. Formulas are torch's single-tensor
# Adam and AdamW (torch 2.14), with one global step counter t:
#
#   t += 1, for every p whose grad g is not None:
#     Adam : g = g + weight_decay * p                (L2 coupled into g)
#     AdamW: p = p * (1 - lr * weight_decay)          (decoupled, before the update)
#     m = beta1 * m + (1 - beta1) * g
#     v = beta2 * v + (1 - beta2) * g * g
#     p = p - (lr / (1 - beta1**t)) * m / (sqrt(v) / sqrt(1 - beta2**t) + eps)
#
# The moments live in the parameter's dtype. formats/checkpoint.md stores them
# as <name>.exp_avg and <name>.exp_avg_sq.
from typing import Any, Iterable

class Adam:
    lr: float  # read on every step: a schedule (M10.4) sets it between steps
    betas: tuple[float, float]
    eps: float
    weight_decay: float

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """Keep the parameters in the given order and allocate both moments as
        zeros of each parameter's shape and dtype; t = 0. ValueError when
        lr < 0, eps < 0, weight_decay < 0, a beta is outside [0, 1), params is
        empty, or a parameter's data is not a floating-point ndarray."""

    def step(self) -> None:
        """One update of every parameter whose grad is not None (formulas
        above), in place: p.data keeps its identity and dtype. Parameters with
        grad None are not decayed and their moments do not change; t still
        advances once per call."""

    def zero_grad(self) -> None:
        """Set every parameter's grad to None."""

    def state_dict(self) -> dict:
        """A snapshot that later steps never change:
        {"step": t, "lr", "betas": [beta1, beta2], "eps", "weight_decay",
         "exp_avg": [m per parameter], "exp_avg_sq": [v per parameter]}
        (lists of copies, in parameter order)."""

    def load_state_dict(self, sd: dict) -> None:
        """Restore t, the hyperparameters, and copies of both moments, so that
        steps after a save and load equal the uninterrupted run bit for bit.
        ValueError when the number of moments or a shape differs from the
        parameters."""

class AdamW(Adam):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """Adam with decoupled weight decay (Loshchilov and Hutter, 2019):
        p is multiplied by (1 - lr * weight_decay) before the Adam update,
        and the gradient is not changed."""
