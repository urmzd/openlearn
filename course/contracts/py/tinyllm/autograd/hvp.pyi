# contracts/py/tinyllm/autograd/hvp.pyi (M08.4): Hessian-vector products and
# the recompute-versus-memory schedule of activation checkpointing
# chapter: content/courses/mathematics/14-matrix-calculus-and-autodiff/04-hessian-vector-products-and-recompute.md
#
# Curvature without the Hessian: for f: R^n -> R with gradient g(x), the
# Hessian-vector product H v is the directional derivative of g along v, so a
# central difference of two gradient calls gives it with O(eps^2) error and
# never forms the n x n matrix.
#
# Checkpoint schedules (the cost model of the chapter, section 2): a stack of
# n layers runs forward and keeps one saved input per layer for its backward,
# 1 memory unit each. A schedule is the list `starts` of segment start layers
# (starts[0] == 0, strictly increasing, every entry < n); segment i covers
# layers starts[i] .. starts[i+1] - 1 and has s_i layers. As in
# torch.utils.checkpoint.checkpoint_sequential, every segment but the last
# keeps only its input during the forward pass and is recomputed during
# backward; the last segment keeps all its activations. Then
#     peak       = max over i of (i + s_i)     units held at once
#     recomputed = n - s_last                  layers run forward twice
# (during the backward of segment i the inputs of segments 0 .. i-1 are still
# held, plus segment i's s_i recomputed activations).
from typing import Callable, Sequence

from numpy.typing import ArrayLike, NDArray

def hvp_fd(
    grad_fn: Callable[[NDArray], ArrayLike], x: ArrayLike, v: ArrayLike, eps: float = 1e-4
) -> NDArray:
    """H(x) v by the central difference of the gradient along v:
        (grad_fn(x + eps v) - grad_fn(x - eps v)) / (2 eps),
    float64 with x's shape. grad_fn is called exactly twice, each time with a
    new float64 array (never x itself), and its results are copied before
    the second call, so a grad_fn that returns (a view of) its argument is
    fine. Exact up to rounding when g is affine (f quadratic); otherwise the
    error is (eps^2 / 6) D^3 g(x)[v, v, v] + O(eps^4). The step is eps * v,
    not normalized. ValueError when x and v differ in shape, eps is not a
    positive finite number, or grad_fn returns a different shape."""

def hessian_fd(grad_fn: Callable[[NDArray], ArrayLike], x: ArrayLike, eps: float = 1e-4) -> NDArray:
    """The dense n x n Hessian of a 1-D x (n = len(x)): column j is
    hvp_fd(grad_fn, x, e_j, eps), then the result is symmetrized,
    (H + H^T) / 2, so it is exactly symmetric. 2n gradient calls.
    ValueError when x is not 1-D or is empty."""

def checkpoint_cost(n_layers: int, starts: Sequence[int]) -> tuple[int, int]:
    """(peak, recomputed) of a schedule, by the cost model above.
    [0] (no checkpointing) gives (n, 0); every layer its own segment gives
    (n, n - 1). ValueError when n_layers < 1 or starts is not a valid
    schedule (empty, starts[0] != 0, not strictly increasing, an entry >= n)."""

def min_checkpoint_memory(n_layers: int) -> int:
    """The smallest peak any schedule reaches: the least P >= 1 with
    P (P + 1) / 2 >= n_layers (segments of P, P - 1, ..., 1 layers), about
    sqrt(2 n). Exact integer arithmetic. ValueError when n_layers < 1."""

def checkpoint_schedule(n_layers: int, mem_budget_layers: int) -> list[int]:
    """The schedule with the fewest recomputed layers whose peak is at most
    mem_budget_layers (B). Use the fewest segments k that can fit; the last
    segment gets L = min(B - k + 1, n - k + 1) layers; the first k - 1
    segments cover the other n - L layers greedily, largest first:
    s_i = min(B - i, rest - (k - 2 - i)), where rest is what is still
    uncovered. [0] when n <= B. ValueError when n_layers < 1 or
    B < min_checkpoint_memory(n_layers)."""
