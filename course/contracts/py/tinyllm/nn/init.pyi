# contracts/py/tinyllm/nn/init.pyi (M07.3)
# chapter: content/tracks/math/07-probability-statistics/03-variance-propagation-and-initialization.md
#
# Weight initializers that keep activation variance constant through depth.
# A weight of shape (out, in, *kernel) has fan_in = in * prod(kernel) and
# fan_out = out * prod(kernel) (torch's rule; a Linear weight is (out, in)).
# Every initializer returns a new float32 array of the given shape, filled in
# C (row-major) order, one draw per element, from `rng` (a PCG32 of M06.3,
# spec/pcg32.md): uniforms are rng.uniform(); normals are
# tinyllm.prob.rv.normal(rng, n) (M07.0), so a seed fixes the weights in
# every language.
from typing import Any, Literal

from numpy.typing import NDArray

def fans(shape: tuple[int, ...]) -> tuple[int, int]:
    """(fan_in, fan_out) of a weight shape: (shape[1] * r, shape[0] * r) with
    r = prod(shape[2:]). ValueError for fewer than 2 dimensions or a fan of 0."""

def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    """torch.nn.init.calculate_gain: 1 for linear, conv1d/2d/3d,
    conv_transpose1d/2d/3d, and sigmoid; 5/3 for tanh; sqrt(2) for relu;
    sqrt(2 / (1 + s^2)) for leaky_relu with slope s = param (0.01 when None);
    3/4 for selu. ValueError for any other name or a non-numeric param."""

def xavier_uniform(shape: tuple[int, ...], gain: float, rng: Any) -> NDArray:
    """U(-a, a) with a = gain * sqrt(6 / (fan_in + fan_out)), so
    Var = gain^2 * 2 / (fan_in + fan_out). Element k is -a + 2 a u_k with u_k
    the k-th rng.uniform(). ValueError for gain < 0 or a bad shape."""

def xavier_normal(shape: tuple[int, ...], gain: float, rng: Any) -> NDArray:
    """N(0, std^2) with std = gain * sqrt(2 / (fan_in + fan_out)):
    std * normal(rng, n). ValueError for gain < 0 or a bad shape."""

def kaiming_normal(
    shape: tuple[int, ...],
    fan_mode: Literal["fan_in", "fan_out"],
    nonlinearity: str,
    rng: Any,
) -> NDArray:
    """N(0, std^2) with std = calculate_gain(nonlinearity) / sqrt(fan), fan
    the chosen fan: fan_in keeps forward activations at constant variance,
    fan_out keeps backward gradients. std * normal(rng, n). ValueError for a
    bad fan_mode, nonlinearity, or shape."""

def normal_init(shape: tuple[int, ...], std: float, rng: Any) -> NDArray:
    """std * normal(rng, n) reshaped to shape (any number of dimensions,
    including 1-D biases and embeddings). ValueError for std < 0."""

def scaled_residual_std(base_std: float, n_layers: int) -> float:
    """GPT-2's init for projections that write into the residual stream:
    base_std / sqrt(2 * n_layers), since each of the n_layers blocks adds two
    such outputs. ValueError for base_std < 0 or n_layers < 1."""
