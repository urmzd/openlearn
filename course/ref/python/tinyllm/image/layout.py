"""tinyllm.image.layout (M12.2): image tensors and memory layouts.

An image is a 3-D array, and which axis is fastest in memory is a choice.
Decoders hand out uint8 HWC (row, column, channel: the three bytes of a pixel
sit together); convolution and the vision towers in L13 want float32 CHW
(each channel a contiguous plane), batched as NCHW. The same logical tensor
can live in either order: the strides say how many bytes to step for one
index on each axis. This module converts between layouts, turns pixels into
the normalized floats a model takes, reproduces Pillow's grayscale rule
bit for bit (data.10 hashes grayscale thumbnails), and cuts an image into
the flattened patches a ViT embeds (L13.3).

Contract: contracts/py/tinyllm/image/layout.pyi.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Layout = Literal["NCHW", "NHWC"]
_PERM = {("NCHW", "NHWC"): (0, 2, 3, 1), ("NHWC", "NCHW"): (0, 3, 1, 2)}


def to_layout(x: NDArray, src: Layout, dst: Layout) -> NDArray:
    # SOLUTION-BEGIN M12.2
    if src not in ("NCHW", "NHWC") or dst not in ("NCHW", "NHWC"):
        raise ValueError(
            f"to_layout: layouts are 'NCHW' or 'NHWC', got {src!r}, {dst!r}"
        )
    x = np.asarray(x)
    if x.ndim != 4:
        raise ValueError(f"to_layout: need a 4-D batch, got shape {x.shape}")
    if src == dst:
        return np.ascontiguousarray(x).copy()
    # transpose only relabels the axes (a view with permuted strides); the
    # contiguous copy is what actually moves the bytes into the new order.
    return np.ascontiguousarray(x.transpose(_PERM[(src, dst)]))
    # SOLUTION-END


def strides_of(
    shape: tuple[int, ...], layout: Layout, itemsize: int
) -> tuple[int, ...]:
    # SOLUTION-BEGIN M12.2
    if layout not in ("NCHW", "NHWC"):
        raise ValueError(f"strides_of: layout is 'NCHW' or 'NHWC', got {layout!r}")
    if len(shape) not in (3, 4) or any(d < 1 for d in shape) or itemsize < 1:
        raise ValueError(
            f"strides_of: need a (N,) C, H, W shape and itemsize >= 1, got {shape}, {itemsize}"
        )
    dims = tuple(shape) if len(shape) == 4 else (1, *shape)
    n, c, h, w = dims
    if layout == "NCHW":
        st = (c * h * w, h * w, w, 1)
    else:
        # Channels last: memory order N, H, W, C, reported per logical axis.
        st = (h * w * c, 1, w * c, c)
    st = tuple(itemsize * s for s in st)
    return st if len(shape) == 4 else st[1:]
    # SOLUTION-END


def rescale_normalize(
    img_u8_hwc: NDArray, scale: float, mean: Sequence[float], std: Sequence[float]
) -> NDArray:
    # SOLUTION-BEGIN M12.2
    img = np.asarray(img_u8_hwc)
    if img.ndim != 3:
        raise ValueError(f"rescale_normalize: need an HWC image, got shape {img.shape}")
    c = img.shape[2]
    m = np.asarray(mean, dtype=np.float64).reshape(-1)
    s = np.asarray(std, dtype=np.float64).reshape(-1)
    if m.shape != (c,) or s.shape != (c,):
        raise ValueError(
            f"rescale_normalize: need {c} means and stds, got {m.shape}, {s.shape}"
        )
    if not (s > 0).all():
        raise ValueError("rescale_normalize: every std must be positive")
    y = (img.astype(np.float64) * scale - m) / s  # float64, then one rounding
    return np.ascontiguousarray(y.transpose(2, 0, 1)).astype(np.float32)
    # SOLUTION-END


def to_grayscale_pil(img_u8_hwc: NDArray) -> NDArray:
    # SOLUTION-BEGIN M12.2
    img = np.asarray(img_u8_hwc)
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(
            f"to_grayscale_pil: need uint8 HWC RGB, got {img.dtype} {img.shape}"
        )
    r, g, b = (img[..., i].astype(np.int64) for i in range(3))
    # ITU-R 601-2 luma in 16-bit fixed point, rounded: Pillow's L24 macro.
    return ((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16).astype(np.uint8)
    # SOLUTION-END


def patchify(x_chw: NDArray, p: int) -> NDArray:
    # SOLUTION-BEGIN M12.2
    x = np.asarray(x_chw)
    if x.ndim != 3 or p < 1:
        raise ValueError(f"patchify: need CHW and p >= 1, got {x.shape}, p={p}")
    c, h, w = x.shape
    if h % p or w % p:
        raise ValueError(f"patchify: {h}x{w} is not a multiple of the patch size {p}")
    gh, gw = h // p, w // p
    # [C, gh, p, gw, p] -> [gh, gw, C, p, p]: patches row-major over the grid,
    # each flattened in (c, i, j) order like a KCRS conv weight.
    return (
        x.reshape(c, gh, p, gw, p).transpose(1, 3, 0, 2, 4).reshape(gh * gw, c * p * p)
    )
    # SOLUTION-END


def unpatchify(patches: NDArray, c: int, h: int, w: int, p: int) -> NDArray:
    # SOLUTION-BEGIN M12.2
    a = np.asarray(patches)
    if p < 1 or h % p or w % p:
        raise ValueError(f"unpatchify: {h}x{w} is not a multiple of the patch size {p}")
    gh, gw = h // p, w // p
    if a.shape != (gh * gw, c * p * p):
        raise ValueError(
            f"unpatchify: patches have shape {a.shape}, expected {(gh * gw, c * p * p)}"
        )
    return a.reshape(gh, gw, c, p, p).transpose(2, 0, 3, 1, 4).reshape(c, h, w)
    # SOLUTION-END
