# contracts/py/tinyllm/num/rotation.pyi (M00.2)
# chapter: content/courses/mathematics-for-computing/00-precalculus/02-trig-rotations-and-eulers-formula.md
#
# Angles are in radians. Pairs are INTERLEAVED: pair i of a vector of length
# 2k is (x[2i], x[2i+1]), the layout of the original RoPE paper and Meta's
# Llama reference code (L7.3 also implements the "half" layout of HF, which
# pairs x[i] with x[i + k]).
from numpy.typing import ArrayLike, NDArray

def rotation_matrix(theta: float) -> NDArray:
    """The 2x2 float64 matrix [[cos t, -sin t], [sin t, cos t]] that rotates
    the plane counterclockwise by theta radians."""

def rotate_pairs(x: ArrayLike, theta: ArrayLike) -> NDArray:
    """Rotate every pair (x[..., 2i], x[..., 2i+1]) by theta[..., i]:
        out[..., 2i]   = x[..., 2i] cos t - x[..., 2i+1] sin t
        out[..., 2i+1] = x[..., 2i] sin t + x[..., 2i+1] cos t
    theta broadcasts to x.shape[:-1] + (k,) where k = x.shape[-1] // 2 (a [k]
    theta rotates every row alike; a [T, k] theta gives each row its own
    angles). Computed in float64; returns a new array of x's shape, float32
    when x is float32 and float64 otherwise. x is never modified.
    ValueError when x is 0-d or its last axis is odd, or theta does not
    broadcast."""

def as_complex(x: ArrayLike) -> NDArray:
    """[..., 2k] real -> [..., k] complex with z[..., i] = x[..., 2i] + 1j * x[..., 2i+1]
    (numpy's analogue of torch.view_as_complex). complex64 for float32 input,
    complex128 otherwise. ValueError when x is 0-d or its last axis is odd."""

def as_real(z: ArrayLike) -> NDArray:
    """The inverse of as_complex: [..., k] complex -> [..., 2k] real with
    out[..., 2i] = z.real and out[..., 2i+1] = z.imag. float32 for complex64
    input, float64 otherwise. ValueError when z is 0-d."""
