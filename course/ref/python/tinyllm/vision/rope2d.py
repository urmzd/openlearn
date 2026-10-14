"""tinyllm.vision.rope2d (L13.2): 2D RoPE for patches and Qwen2-VL's M-RoPE.

L7.3 rotated query and key pairs by an angle proportional to a token's
position, so attention scores see only relative position. An image has two
coordinates. Axial 2D RoPE gives half the pairs to the row and half to the
column, so a score depends on (dh, dw) only. M-RoPE goes one step further for
a language model that reads images: every token carries three ids (time,
row, column), text tokens set all three to their position, and image tokens
spread over a grid; the text after an image resumes past the image's largest
position. L13.3's ViT uses rope2d_cos_sin for pos='rope2d', and L13.7's
qwen2vl architecture uses both halves.

Contract: contracts/py/tinyllm/vision/rope2d.pyi.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.rope import apply_rope, rope_cos_sin


def patch_coords(grid_hw: tuple[int, int], merge: int = 1) -> NDArray:
    # SOLUTION-BEGIN L13.2
    h, w = (int(v) for v in grid_hw)
    if h < 1 or w < 1 or merge < 1 or h % merge or w % merge:
        raise ValueError(f"patch_coords: need H, W >= 1 divisible by merge >= 1, got {grid_hw}, merge {merge}")
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    if merge > 1:
        # Blocks of m x m patches, block-major: [H/m, m, W/m, m] -> [H/m, W/m, m, m].
        shape = (h // merge, merge, w // merge, merge)
        rows = rows.reshape(shape).transpose(0, 2, 1, 3)
        cols = cols.reshape(shape).transpose(0, 2, 1, 3)
    return np.stack([rows.ravel(), cols.ravel()], axis=-1).astype(np.int64)
    # SOLUTION-END


def rope2d_cos_sin(
    grid_hw: tuple[int, int], d_rot: int, base: float = 10000.0, merge: int = 1
) -> tuple[NDArray, NDArray]:
    # SOLUTION-BEGIN L13.2
    if d_rot < 4 or d_rot % 4:
        raise ValueError(f"rope2d_cos_sin: d_rot must be a multiple of 4 and >= 4, got {d_rot}")
    xy = patch_coords(grid_hw, merge)
    half = d_rot // 2  # pairs; each axis owns half of them
    # One ladder for both axes, over the half-width: base^(-2i / half).
    inv = 1.0 / (base ** (np.arange(0, half, 2, dtype=np.float32) / np.float32(half)))
    ch, sh = rope_cos_sin(xy[:, 0], inv)
    cw, sw = rope_cos_sin(xy[:, 1], inv)
    return np.concatenate([ch, cw], axis=-1), np.concatenate([sh, sw], axis=-1)
    # SOLUTION-END


def mrope_position_ids(
    input_ids: ArrayLike, image_grid_thw: ArrayLike, image_token_id: int, spatial_merge: int = 2
) -> NDArray:
    # SOLUTION-BEGIN L13.2
    ids = np.asarray(input_ids)
    if ids.ndim != 1:
        raise ValueError(f"mrope_position_ids: need 1-D ids, got shape {ids.shape}")
    grids = np.asarray(image_grid_thw, dtype=np.int64).reshape(-1, 3)
    m = int(spatial_merge)
    pos = np.zeros((3, len(ids)), dtype=np.int64)
    cur, i, img = 0, 0, 0
    while i < len(ids):
        if ids[i] != image_token_id:
            pos[:, i] = cur
            cur += 1
            i += 1
            continue
        if img >= len(grids):
            raise ValueError(f"mrope_position_ids: image tokens at {i} but only {len(grids)} grid(s)")
        t, gh, gw = (int(v) for v in grids[img])
        if gh % m or gw % m:
            raise ValueError(f"mrope_position_ids: merge {m} does not divide grid {grids[img].tolist()}")
        lh, lw = gh // m, gw // m
        n = t * lh * lw
        if i + n > len(ids) or np.any(ids[i : i + n] != image_token_id):
            raise ValueError(f"mrope_position_ids: image {img} needs {n} image tokens from index {i}")
        tt, hh, ww = np.meshgrid(np.arange(t), np.arange(lh), np.arange(lw), indexing="ij")
        pos[0, i : i + n] = cur + tt.ravel()
        pos[1, i : i + n] = cur + hh.ravel()
        pos[2, i : i + n] = cur + ww.ravel()
        # Text resumes past the image: its widest spatial extent.
        cur += max(lh, lw)
        i += n
        img += 1
    if img != len(grids):
        raise ValueError(f"mrope_position_ids: {len(grids)} grids but {img} image runs")
    return pos
    # SOLUTION-END


def mrope_cos_sin(
    pos3: ArrayLike, inv_freq: ArrayLike, mrope_section: Sequence[int]
) -> tuple[NDArray, NDArray]:
    # SOLUTION-BEGIN L13.2
    p = np.asarray(pos3)
    f = np.asarray(inv_freq)
    if p.ndim != 2 or p.shape[0] != 3:
        raise ValueError(f"mrope_cos_sin: need pos3 [3, T], got shape {p.shape}")
    sec = [int(s) for s in mrope_section]
    if sum(sec) != len(f) or min(sec, default=0) < 0:
        raise ValueError(f"mrope_cos_sin: sections {sec} must sum to {len(f)} pairs")
    cos, sin = rope_cos_sin(p, f)  # [3, T, r/2]: every axis at every frequency
    axis = np.repeat(np.arange(len(sec)) % 3, sec)  # which axis drives pair i
    cols = np.arange(len(f))
    return cos[axis, :, cols].T.copy(), sin[axis, :, cols].T.copy()
    # SOLUTION-END


def apply_mrope(
    x: Tensor, pos3: ArrayLike, inv_freq: ArrayLike, mrope_section: Sequence[int]
) -> Tensor:
    # SOLUTION-BEGIN L13.2
    f = np.asarray(inv_freq)
    if x.shape[-1] != 2 * len(f):
        raise ValueError(f"apply_mrope: x ends in {x.shape[-1]}, need 2 x {len(f)} rotary entries")
    cos, sin = mrope_cos_sin(pos3, f, mrope_section)
    return apply_rope(x, cos, sin, "half")
    # SOLUTION-END
