"""lang.01 broadcasting worksheet (a primer exercise, not part of the system).

Three functions the rest of the course leans on:

  broadcast_shape  the broadcasting rule, by hand
  unbroadcast      its adjoint: sum a gradient back down to an operand's shape (L0.1)
  as_c_float32     a C-contiguous float32 array, copied only when needed (M03.1)
"""

from __future__ import annotations

import numpy as np


def broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """The shape numpy gives `x + y` when x.shape == a and y.shape == b.

    Align the two shapes on the right and pad the shorter one with 1s on the
    left. In every aligned pair the sizes must be equal or one of them must
    be 1; the result takes the other size. Raise ValueError otherwise.
    """
    n = max(len(a), len(b))
    pa = (1,) * (n - len(a)) + tuple(a)
    pb = (1,) * (n - len(b)) + tuple(b)
    out = []
    for axis, (x, y) in enumerate(zip(pa, pb)):
        if x == y or y == 1:
            out.append(x)
        elif x == 1:
            out.append(y)
        else:
            raise ValueError(f"shapes {tuple(a)} and {tuple(b)} do not broadcast: "
                             f"axis {axis - n} has sizes {x} and {y}")
    return tuple(out)


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum `grad`, shaped like a broadcast result, back down to `shape`.

    Broadcasting copies each element of an operand to several output
    positions, so the gradient of that element is the sum over its copies:
    sum away the leading axes broadcasting added, then sum (keeping the axis)
    every axis where `shape` has size 1. Raise ValueError when `shape` could
    not have broadcast to grad.shape.
    """
    g = np.asarray(grad)
    shape = tuple(shape)
    extra = g.ndim - len(shape)
    if extra < 0:
        raise ValueError(f"grad has {g.ndim} axes, fewer than the target shape {shape}")
    if extra:
        g = g.sum(axis=tuple(range(extra)))
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and g.shape[i] != 1)
    if axes:
        g = g.sum(axis=axes, keepdims=True)
    if g.shape != shape:
        raise ValueError(f"{shape} does not broadcast to {np.shape(grad)}")
    return g


def as_c_float32(x) -> np.ndarray:
    """x as a C-contiguous float32 array: the layout a C function taking
    `const float *` with explicit dims reads. Share memory with x when it
    already is one; copy (and convert) only when it is not."""
    # Not np.ascontiguousarray: it returns at least 1-D, so a 0-d scalar
    # would come back with shape (1,).
    return np.asarray(x, dtype=np.float32, order="C")
