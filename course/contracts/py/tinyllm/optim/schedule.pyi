# contracts/py/tinyllm/optim/schedule.pyi (M10.4)
# chapter: content/courses/mathematics-for-computing/11-optimization/04-schedules-and-gradient-clipping.md
#
# Learning-rate schedules are pure functions of `step`, the number of
# optimizer updates already taken (0 for the first update). The train loop
# (L0.5) sets opt.lr = schedule(step, ...) before each opt.step(), which is
# the value HF's LambdaLR schedulers give after `step` scheduler.step() calls.
# Every warmup ramps linearly from 0: lr_max * step / warmup.
from typing import Any, Iterable

def cosine_with_warmup(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    """lr_max * step / warmup while step < warmup; then
    lr_min + (lr_max - lr_min) * (1 + cos(pi * (step - warmup) / (total - warmup))) / 2
    up to step = total; lr_min for every step >= total.
    ValueError unless 0 <= step, 0 <= warmup <= total, 1 <= total, and
    0 <= lr_min <= lr_max."""

def wsd(step: int, warmup: int, stable: int, decay: int, lr_max: float, lr_min: float) -> float:
    """Warmup-stable-decay: lr_max * step / warmup while step < warmup;
    lr_max while step < warmup + stable; then a linear decay
    lr_max - (lr_max - lr_min) * (step - warmup - stable) / decay while
    step < warmup + stable + decay; lr_min after. ValueError unless
    0 <= step, warmup >= 0, stable >= 0, decay >= 1, and 0 <= lr_min <= lr_max."""

def noam(step: int, d_model: int, warmup: int) -> float:
    """Vaswani et al. (2017), equation 3: d_model**-0.5 * min(step**-0.5, step * warmup**-1.5),
    and 0.0 at step 0 (the limit of the formula). The peak, at step = warmup,
    is (d_model * warmup)**-0.5. ValueError unless step >= 0, d_model >= 1, warmup >= 1."""

def clip_grad_norm_(params: Iterable[Any], max_norm: float) -> float:
    """Global-norm clipping as torch.nn.utils.clip_grad_norm_. Over every
    parameter whose grad is not None: total = sqrt(sum of all squared grad
    entries), computed in float64. When total is finite and
    max_norm / (total + 1e-6) < 1, every grad is multiplied IN PLACE by that
    coefficient; otherwise the grads are not changed (a non-finite total leaves
    them for the caller to skip the step). Returns total, the norm before
    clipping (0.0 when no parameter has a grad). ValueError when max_norm <= 0."""
