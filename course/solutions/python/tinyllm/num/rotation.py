"""Trig, the unit circle, 2D rotations, and Euler's formula (M00.2).

A rotation of the plane by theta sends (x, y) to
(x cos theta - y sin theta, x sin theta + y cos theta). Reading the pair as the
complex number x + iy, the same rotation is one multiplication by
e^{i theta} = cos theta + i sin theta (Euler's formula). RoPE (L7.3) and the
sinusoidal encoding (L5.4) rotate the pairs (x[2i], x[2i+1]) of a vector.

Contract: contracts/py/tinyllm/num/rotation.pyi.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def rotation_matrix(theta: float) -> NDArray:
    c, s = math.cos(theta), math.sin(theta)
    # Columns are where the basis vectors land: e0 -> (c, s), e1 -> (-s, c).
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotate_pairs(x: ArrayLike, theta: ArrayLike) -> NDArray:
    a = np.asarray(x)
    # float32 stays float32; every other input is computed and returned in float64.
    out_dtype = np.float32 if a.dtype == np.float32 else np.float64
    if a.ndim == 0 or a.shape[-1] % 2 != 0:
        raise ValueError(
            f"the last axis must have even length (pairs), got shape {a.shape}"
        )
    k = a.shape[-1] // 2
    t = np.asarray(theta, dtype=np.float64)
    try:
        t = np.broadcast_to(t, a.shape[:-1] + (k,))
    except ValueError:
        raise ValueError(
            f"theta of shape {t.shape} does not broadcast to {a.shape[:-1] + (k,)}"
        ) from None
    a64 = a.astype(np.float64)
    even, odd = a64[..., 0::2], a64[..., 1::2]  # interleaved pairs (x[2i], x[2i+1])
    c, s = np.cos(t), np.sin(t)
    out = np.empty(a.shape, dtype=np.float64)  # a new array: x is never modified
    # Both outputs read the OLD even and odd values.
    out[..., 0::2] = even * c - odd * s
    out[..., 1::2] = even * s + odd * c
    return out.astype(out_dtype)


def as_complex(x: ArrayLike) -> NDArray:
    a = np.asarray(x)
    if a.ndim == 0 or a.shape[-1] % 2 != 0:
        raise ValueError(
            f"the last axis must have even length (pairs), got shape {a.shape}"
        )
    ctype = np.complex64 if a.dtype == np.float32 else np.complex128
    a = a.astype(np.float64)
    z = a[..., 0::2] + 1j * a[..., 1::2]
    return z.astype(ctype)


def as_real(z: ArrayLike) -> NDArray:
    c = np.asarray(z)
    if c.ndim == 0:
        raise ValueError("as_real needs at least one axis")
    rtype = np.float32 if c.dtype == np.complex64 else np.float64
    c = c.astype(np.complex128)
    out = np.empty(c.shape[:-1] + (2 * c.shape[-1],), dtype=np.float64)
    out[..., 0::2] = c.real
    out[..., 1::2] = c.imag
    return out.astype(rtype)
