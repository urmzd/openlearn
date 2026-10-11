"""tinyllm.vision.tiling (L13.4): dynamic resolution and tiling.

A vision tower is trained on one square size, but photos come in every size
and shape. Squeezing a 4000 x 1000 receipt into 512 x 512 destroys the text,
so current VLMs cut a large image into tower-sized tiles and add one small
view of the whole image. Two recipes are here: Idefics3 / SmolVLM (grow the
image to a multiple of the tile, crop a grid, add a global image, mark every
tile with a row and column tag) and LLaVA-NeXT AnyRes (fit the image into the
best of a few allowed resolutions, pad, crop, and later strip the padding
from the feature grid). The token count of an image follows from the grid;
L13.6 runs the split, L13.7 expands the placeholders, and the Rust engine
(L15.1) and the Go gateway (gw.10) re-implement image_token_count against the
fixture table.

Contract: contracts/py/tinyllm/vision/tiling.pyi.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tinyllm.sig.resample import resize

MAX_IMAGE_SIZE = 4096


@dataclass
class TilingConfig:
    kind: Literal["idefics3", "llava_next", "single"]
    tile: int
    longest_edge: int = 0
    image_seq_len: int = 0
    pinpoints: tuple = ()
    patch_size: int = 14
    base_extra: int = 0
    split: bool = True


def select_best_resolution(hw: tuple[int, int], pinpoints: Sequence[tuple[int, int]]) -> tuple[int, int]:
    # SOLUTION-BEGIN L13.4
    h, w = (int(v) for v in hw)
    if h < 1 or w < 1 or not len(pinpoints):
        raise ValueError(f"select_best_resolution: need a size >= 1 and candidates, got {hw}, {pinpoints}")
    best, best_eff, best_waste = None, 0, math.inf
    for ph, pw in pinpoints:
        scale = min(pw / w, ph / h)
        dw, dh = int(w * scale), int(h * scale)
        eff = min(dw * dh, w * h)
        waste = pw * ph - eff
        # Strictly better only: an exact tie keeps the earlier candidate.
        if eff > best_eff or (eff == best_eff and waste < best_waste):
            best, best_eff, best_waste = (int(ph), int(pw)), eff, waste
    return best
    # SOLUTION-END


def fit_size(hw: tuple[int, int], target: tuple[int, int]) -> tuple[int, int]:
    # SOLUTION-BEGIN L13.4
    h, w = hw
    th, tw = target
    sw, sh = tw / w, th / h
    if sw < sh:
        return min(math.ceil(h * sw), th), tw
    return th, min(math.ceil(w * sh), tw)
    # SOLUTION-END


def _rgb(img: NDArray, who: str) -> NDArray:
    # SOLUTION-BEGIN L13.4
    a = np.asarray(img)
    if a.dtype != np.uint8 or a.ndim != 3:
        raise ValueError(f"{who}: need a uint8 HWC image, got {a.dtype} {a.shape}")
    return a
    # SOLUTION-END


def anyres_tiles(
    img: NDArray, pinpoints: Sequence[tuple[int, int]], tile: int
) -> tuple[list[NDArray], NDArray, tuple[int, int]]:
    # SOLUTION-BEGIN L13.4
    a = _rgb(img, "anyres_tiles")
    if any(ph % tile or pw % tile for ph, pw in pinpoints):
        raise ValueError(f"anyres_tiles: every pinpoint must be a multiple of tile {tile}: {pinpoints}")
    h, w, c = a.shape
    th, tw = select_best_resolution((h, w), pinpoints)
    nh, nw = fit_size((h, w), (th, tw))
    fitted = resize(a, (nh, nw), "bicubic")
    py, ry = divmod(th - nh, 2)
    px, rx = divmod(tw - nw, 2)
    canvas = np.pad(fitted, ((py, py + ry), (px, px + rx), (0, 0)))
    rows, cols = th // tile, tw // tile
    tiles = [canvas[i * tile : (i + 1) * tile, j * tile : (j + 1) * tile].copy() for i in range(rows) for j in range(cols)]
    base = resize(a, (tile, tile), "bicubic")
    return tiles, base, (rows, cols)
    # SOLUTION-END


def unpad_shape(orig_hw: tuple[int, int], cur_hw: tuple[int, int]) -> tuple[int, int]:
    # SOLUTION-BEGIN L13.4
    h, w = orig_hw
    ch, cw = cur_hw
    if w / h > cw / ch:
        new_h = int(round(h * (cw / w), 7))
        pad = (ch - new_h) // 2
        return ch - 2 * pad, cw
    new_w = int(round(w * (ch / h), 7))
    pad = (cw - new_w) // 2
    return ch, cw - 2 * pad
    # SOLUTION-END


def _longest_edge_size(h: int, w: int, max_len: int) -> tuple[int, int]:
    """Resize so the longest edge is max_len (the other rounded up to even), then cap at 4096."""
    # SOLUTION-BEGIN L13.4
    aspect = w / h
    if w >= h:
        w = max_len
        h = int(w / aspect)
        h += h % 2
    else:
        h = max_len
        w = int(h * aspect)
        w += w % 2
    h, w = max(h, 1), max(w, 1)
    aspect = w / h
    if w >= h and w > MAX_IMAGE_SIZE:
        w = MAX_IMAGE_SIZE
        h = int(w / aspect)
    elif h > w and h > MAX_IMAGE_SIZE:
        h = MAX_IMAGE_SIZE
        w = int(h * aspect)
    return max(h, 1), max(w, 1)
    # SOLUTION-END


def idefics3_resize_size(hw: tuple[int, int], longest_edge: int, tile: int, split: bool = True) -> tuple[int, int]:
    # SOLUTION-BEGIN L13.4
    h, w = (int(v) for v in hw)
    if h < 1 or w < 1:
        raise ValueError(f"idefics3_resize_size: need a size >= 1, got {hw}")
    if not split:
        return tile, tile
    h, w = _longest_edge_size(h, w, longest_edge)
    aspect = w / h
    if w >= h:
        w = math.ceil(w / tile) * tile
        h = math.ceil(int(w / aspect) / tile) * tile
    else:
        h = math.ceil(h / tile) * tile
        w = math.ceil(int(h * aspect) / tile) * tile
    return h, w
    # SOLUTION-END


def idefics3_split(
    img: NDArray, longest_edge: int, tile: int, split: bool = True
) -> tuple[list[NDArray], NDArray, tuple[int, int]]:
    # SOLUTION-BEGIN L13.4
    a = _rgb(img, "idefics3_split")
    h, w = a.shape[:2]
    if not split:
        # Two Lanczos passes, as the processor: to the longest edge, then square.
        a = resize(a, _longest_edge_size(h, w, longest_edge), "lanczos3")
        return [], resize(a, (tile, tile), "lanczos3"), (0, 0)
    a = resize(a, _longest_edge_size(h, w, longest_edge), "lanczos3")
    big = resize(a, idefics3_resize_size((h, w), longest_edge, tile), "lanczos3")
    bh, bw = big.shape[:2]
    if bh <= tile and bw <= tile:
        return [], resize(big, (tile, tile), "lanczos3"), (0, 0)
    rows, cols = bh // tile, bw // tile
    tiles = [big[i * tile : (i + 1) * tile, j * tile : (j + 1) * tile].copy() for i in range(rows) for j in range(cols)]
    return tiles, resize(big, (tile, tile), "lanczos3"), (rows, cols)
    # SOLUTION-END


def idefics3_prompt(
    rows: int,
    cols: int,
    image_seq_len: int,
    image_token: str = "<image>",
    fake_token: str = "<fake_token_around_image>",
    global_tag: str = "<global-img>",
) -> str:
    # SOLUTION-BEGIN L13.4
    run = image_token * image_seq_len
    if rows == 0 and cols == 0:
        return fake_token + global_tag + run + fake_token
    out = ""
    for r in range(rows):
        for c in range(cols):
            out += fake_token + f"<row_{r + 1}_col_{c + 1}>" + run
        out += "\n"
    return out + "\n" + fake_token + global_tag + run + fake_token
    # SOLUTION-END


def pixel_shuffle(x: NDArray, r: int) -> NDArray:
    # SOLUTION-BEGIN L13.4
    a = np.asarray(x)
    if a.ndim != 3:
        raise ValueError(f"pixel_shuffle: need [N, S*S, D], got {a.shape}")
    n, seq, d = a.shape
    s = math.isqrt(seq)
    if s * s != seq or r < 1 or s % r:
        raise ValueError(f"pixel_shuffle: {seq} tokens are not an S x S grid with r={r} dividing S")
    # Hugging Face's Idefics3Connector, op for op.
    y = a.reshape(n, s, s, d).reshape(n, s, s // r, d * r)
    y = y.transpose(0, 2, 1, 3).reshape(n, s // r, s // r, d * r * r)
    return y.transpose(0, 2, 1, 3).reshape(n, seq // (r * r), d * r * r)
    # SOLUTION-END


def pixel_unshuffle(y: NDArray, r: int) -> NDArray:
    # SOLUTION-BEGIN L13.4
    a = np.asarray(y)
    if a.ndim != 3 or r < 1 or a.shape[2] % (r * r):
        raise ValueError(f"pixel_unshuffle: need [N, T, D r^2], got {a.shape} with r={r}")
    n, t, dd = a.shape
    q = math.isqrt(t)
    if q * q != t:
        raise ValueError(f"pixel_unshuffle: {t} tokens are not a square grid")
    d, s = dd // (r * r), q * r
    x = a.reshape(n, q, q, d * r * r).transpose(0, 2, 1, 3).reshape(n, q, s, d * r)
    return x.transpose(0, 2, 1, 3).reshape(n, s, s, d).reshape(n, s * s, d)
    # SOLUTION-END


def image_token_count(h: int, w: int, cfg: TilingConfig, detail: Literal["low", "high", "auto"] = "auto") -> int:
    # SOLUTION-BEGIN L13.4
    if h < 1 or w < 1:
        raise ValueError(f"image_token_count: need a size >= 1, got {h}x{w}")
    if detail not in ("low", "high", "auto"):
        raise ValueError(f"image_token_count: detail must be low, high, or auto, got {detail!r}")
    split = cfg.split if detail == "auto" else detail == "high"
    if cfg.kind == "single":
        return cfg.image_seq_len
    if cfg.kind == "idefics3":
        if not split:
            return cfg.image_seq_len
        bh, bw = idefics3_resize_size((h, w), cfg.longest_edge, cfg.tile)
        if bh <= cfg.tile and bw <= cfg.tile:
            return cfg.image_seq_len
        return ((bh // cfg.tile) * (bw // cfg.tile) + 1) * cfg.image_seq_len
    if cfg.kind == "llava_next":
        side = cfg.tile // cfg.patch_size
        base = side * side + cfg.base_extra
        if not split:
            return base + 1
        th, tw = select_best_resolution((h, w), cfg.pinpoints)
        uh, uw = unpad_shape((h, w), (th // cfg.tile * side, tw // cfg.tile * side))
        return base + uh * uw + uh
    raise ValueError(f"image_token_count: unknown kind {cfg.kind!r}")
    # SOLUTION-END
