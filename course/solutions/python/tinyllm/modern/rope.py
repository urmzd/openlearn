"""Rotary position embeddings (L7.3).

Each pair of query and key entries is rotated by an angle proportional to the
token's position. Rotations preserve length and compose by adding angles, so
the score between a query at position m and a key at position n depends only
on m - n: relative position, for free, inside the dot product.

Contract: contracts/py/tinyllm/modern/rope.pyi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor


def rope_cos_sin(
    positions: ArrayLike, inv_freq: ArrayLike, attention_scaling: float = 1.0
) -> tuple[NDArray, NDArray]:
    p = np.asarray(positions)
    if p.size and (p.dtype.kind not in "iuf" or np.any(p != np.round(p))):
        raise ValueError(f"positions must be integers, got {p.dtype}")
    if p.size and p.min() < 0:
        raise ValueError(f"positions must be >= 0, got min {p.min()}")
    f = np.asarray(inv_freq)
    if f.ndim != 1:
        raise ValueError(f"inv_freq must be 1-D, got shape {f.shape}")
    # One float32 product per (position, pair), as HF's rotary embedding does
    # (positions.float() * inv_freq.float()); then cos and sin of it.
    angle = p.astype(np.float32)[..., None] * f.astype(np.float32)
    s = np.float32(attention_scaling)
    return (np.cos(angle) * s).astype(np.float32), (np.sin(angle) * s).astype(np.float32)


def _rotary_dim(x: Tensor, cos: NDArray, sin: NDArray, rotary_dim: Optional[int]) -> int:
    dh = x.shape[-1]
    r = dh if rotary_dim is None else int(rotary_dim)
    if r < 2 or r % 2 or r > dh:
        raise ValueError(f"rotary_dim must be even and in [2, {dh}], got {rotary_dim!r}")
    for name, t in (("cos", cos), ("sin", sin)):
        if t.ndim < 1 or t.shape[-1] != r // 2:
            raise ValueError(f"{name} must end in r/2 = {r // 2}, got shape {t.shape}")
        try:
            np.broadcast_shapes(t.shape, x.shape[:-1] + (r // 2,))
        except ValueError:
            raise ValueError(f"{name} {t.shape} does not broadcast to {x.shape[:-1] + (r // 2,)}") from None
    return r


def apply_rope(
    x: Tensor,
    cos: ArrayLike,
    sin: ArrayLike,
    layout: Literal["half", "interleaved"] = "half",
    rotary_dim: Optional[int] = None,
) -> Tensor:
    if layout not in ("half", "interleaved"):
        raise ValueError(f"layout must be 'half' or 'interleaved', got {layout!r}")
    c, s = np.asarray(cos), np.asarray(sin)
    r = _rotary_dim(x, c, s, rotary_dim)
    dh, h = x.shape[-1], r // 2
    if layout == "half":
        # pair i is (x[i], x[i + r/2])
        a, b = x[..., :h], x[..., h:r]
        rot = F.concat([a * c - b * s, b * c + a * s], axis=-1)
    else:
        # pair i is (x[2i], x[2i + 1]); rotate, then interleave back
        a, b = x[..., 0:r:2], x[..., 1:r:2]
        pairs = F.stack([a * c - b * s, a * s + b * c], axis=-1)
        rot = F.reshape(pairs, x.shape[:-1] + (r,))
    if r == dh:
        return rot
    # partial rotary: entries r.. pass through untouched
    return F.concat([rot, x[..., r:]], axis=-1)


@dataclass
class RopeSpec:
    """How an attention layer rotates its queries and keys (L7.5, L7.6)."""

    inv_freq: NDArray
    attention_scaling: float
    layout: str
    rotary_dim: int
