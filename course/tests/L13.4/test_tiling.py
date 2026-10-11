"""Course tests for L13.4: select_best_resolution, fit_size, anyres_tiles,
unpad_shape, idefics3_resize_size, idefics3_split, idefics3_prompt,
pixel_shuffle, pixel_unshuffle, and image_token_count
(tinyllm/vision/tiling.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/L13.4), and the chapter section it comes from.

The chapter's worked example (section 3) is a 1000 x 1500 photo through
SmolVLM's processor (longest edge 2048, tiles of 512, 64 tokens per tile):
the width becomes 2048 and the height int(2048 / 1.5) = 1365, rounded up to
the even 1366; growing to multiples of 512 gives 1536 x 2048, so 3 rows by 4
columns of tiles plus the global image, (12 + 1) * 64 = 832 tokens. The
golden values in $TINYLLM_FIXTURES/L13.4/tiling_hf.npz come from
course/oracle/L13.4/tiling_hf.py (transformers 5.19.0's
Idefics3ImageProcessorPil, Idefics3Processor, LlavaNextImageProcessorPil,
LlavaNextProcessor, select_best_resolution, and Idefics3Connector, on
Pillow 12.3.0).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from _lib.pcg32 import PCG32
from tinyllm.vision.tiling import (
    TilingConfig,
    anyres_tiles,
    fit_size,
    idefics3_prompt,
    idefics3_resize_size,
    idefics3_split,
    image_token_count,
    pixel_shuffle,
    pixel_unshuffle,
    select_best_resolution,
    unpad_shape,
)

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "L13.4" / "tiling_hf.npz"
SMOLVLM = TilingConfig("idefics3", 512, longest_edge=2048, image_seq_len=64)
SMALL_PINS = [(16, 32), (32, 16), (32, 32), (48, 16)]


def seed() -> int:
    return int(os.environ.get("OL_SEED", "0"))


def llava(z) -> TilingConfig:
    return TilingConfig(
        "llava_next",
        336,
        pinpoints=tuple(tuple(int(v) for v in p) for p in z["table.pinpoints"]),
    )


def test_hand_example():
    # WHY: section 3's worked example: a 1000 x 1500 photo becomes a 3 x 4
    #      grid of 512 tiles plus a global image, 832 tokens; and LLaVA-NeXT
    #      fits a 500 x 1000 image best into 336 x 672 (no waste).
    # KIND: unit, smoke
    # CATCHES: s01, s07
    # CHAPTER: L13.4 section 3
    assert idefics3_resize_size((1000, 1500), 2048, 512) == (1536, 2048)
    assert image_token_count(1000, 1500, SMOLVLM) == 832
    assert image_token_count(1000, 1500, SMOLVLM, detail="low") == 64
    pins = [(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]
    assert select_best_resolution((500, 1000), pins) == (336, 672)


def test_token_table_matches_hf():
    # WHY: the 40-size table is the parity fixture the Rust engine (L15.1)
    #      and the Go gateway (gw.10) also check; image tokens are metered
    #      and reserved before the image is ever decoded, so every size,
    #      tiny, huge, and extreme aspect, must match the processors.
    # KIND: golden
    # CATCHES: s03, s08
    # CHAPTER: L13.4 section 2.4
    z = np.load(GOLDEN)
    lv = llava(z)
    for k, (h, w) in enumerate(z["table.hw"].tolist()):
        assert idefics3_resize_size((h, w), 2048, 512) == tuple(
            z["table.idefics3_resized"][k]
        ), (h, w)
        assert image_token_count(h, w, SMOLVLM) == int(z["table.idefics3_tokens"][k]), (
            h,
            w,
        )
        assert select_best_resolution((h, w), lv.pinpoints) == tuple(
            z["table.llava_best"][k]
        ), (h, w)
        assert image_token_count(h, w, lv) == int(z["table.llava_tokens"][k]), (h, w)


def test_prompt_strings_match_hf():
    # WHY: the text the chat template puts where an image was: a row and
    #      column tag before each tile, a newline after each row, the global
    #      image last. L13.7 expands placeholders with it.
    # KIND: golden
    # CATCHES: s06
    # CHAPTER: L13.4 section 2.2
    z = np.load(GOLDEN)
    for r, c in ((0, 0), (1, 2), (2, 3)):
        assert idefics3_prompt(r, c, 3) == bytes(z[f"prompt.{r}x{c}"]).decode(), (r, c)


@pytest.mark.parametrize("i", [0, 1, 2])
def test_idefics3_tiles_match_hf(i):
    # WHY: the tiles themselves, pixel for pixel: two Lanczos resizes
    #      (longest edge, then up to the tile grid), the crops in row-major
    #      order, and the global image last.
    # KIND: golden
    # CATCHES: s02, s07
    # CHAPTER: L13.4 section 2.2
    z = np.load(GOLDEN)
    tiles, glob, grid = idefics3_split(z[f"idef.{i}.img"], 48, 16)
    assert grid == tuple(z[f"idef.{i}.grid"])
    got = np.stack([t.transpose(2, 0, 1) for t in tiles + [glob]])
    assert got.dtype == np.uint8
    assert got.tolist() == z[f"idef.{i}.frames"].tolist()


@pytest.mark.parametrize("i", [0, 1])
def test_anyres_tiles_match_hf(i):
    # WHY: AnyRes: best resolution, an aspect-preserving bicubic fit,
    #      centered zero padding, the crops, and the base image first.
    # KIND: golden
    # CATCHES: s04
    # CHAPTER: L13.4 section 2.3
    z = np.load(GOLDEN)
    img = z[f"anyres.{i}.img"]
    tiles, base, grid = anyres_tiles(img, SMALL_PINS, 16)
    best = tuple(int(v) for v in z[f"anyres.{i}.best"])
    assert grid == (best[0] // 16, best[1] // 16)
    got = np.stack([t.transpose(2, 0, 1) for t in [base] + tiles])
    assert got.tolist() == z[f"anyres.{i}.frames"].tolist()


def test_pixel_shuffle_matches_hf_and_inverts():
    # WHY: Idefics3's connector trades 4x fewer tokens for 4x wider ones
    #      with exactly Hugging Face's reshape order (the projector weights
    #      were trained on it); pixel_unshuffle undoes it exactly.
    # KIND: golden
    # CATCHES: s05
    # CHAPTER: L13.4 section 2.5
    z = np.load(GOLDEN)
    y = pixel_shuffle(z["shuffle.x"], 2)
    assert y.shape == (2, 4, 24)
    assert y.tolist() == z["shuffle.y"].tolist()
    assert pixel_unshuffle(y, 2).tolist() == z["shuffle.x"].tolist()
    x = PCG32(seed=seed() + 1).normal_array((1, 36, 5))
    assert np.array_equal(pixel_unshuffle(pixel_shuffle(x, 3), 3), x)


def test_grow_to_tiles_uses_the_grown_width():
    # WHY: after the width grows to a multiple of the tile, the height is
    #      recomputed from the grown width (int(W' / aspect)), then grown. A
    #      505 x 1000 image at longest edge 1000 is 506 x 1000, the width
    #      grows to 1024, so the height becomes 518 and needs two rows.
    # KIND: unit
    # CATCHES: s02
    # CHAPTER: L13.4 section 2.2
    assert idefics3_resize_size((505, 1000), 1000, 512) == (1024, 1024)
    assert idefics3_resize_size((1000, 505), 1000, 512) == (1024, 1024)
    assert idefics3_resize_size((300, 200), 1000, 512, split=False) == (512, 512)
    cfg = TilingConfig("idefics3", 512, longest_edge=1000, image_seq_len=10)
    assert image_token_count(505, 1000, cfg) == 50


def test_unpad_shape_and_fit_size():
    # WHY: AnyRes pads to the chosen resolution; the language model must see
    #      only the rows and columns that hold the image, one newline per
    #      remaining row.
    # KIND: unit
    # CATCHES: s08, s03
    # CHAPTER: L13.4 section 2.3
    assert fit_size((500, 1000), (672, 672)) == (336, 672)
    assert fit_size((1000, 500), (672, 336)) == (672, 336)
    assert unpad_shape((500, 1000), (48, 48)) == (24, 48)
    assert unpad_shape((1000, 500), (48, 48)) == (48, 24)
    assert unpad_shape((480, 640), (24, 48)) == (24, 32)
    cfg = TilingConfig(
        "llava_next", 336, pinpoints=((336, 672), (672, 672)), patch_size=14
    )
    # 500 x 1000 -> 336 x 672 -> a 24 x 48 grid, nothing to unpad.
    assert image_token_count(500, 1000, cfg) == 576 + 24 * 48 + 24
    assert image_token_count(500, 1000, cfg, detail="low") == 577
    assert image_token_count(7, 9, TilingConfig("single", 224, image_seq_len=49)) == 49


def test_select_best_resolution_ties():
    # WHY: the order of the rule matters: more effective pixels first, then
    #      less waste, and an exact tie keeps the earlier candidate.
    # KIND: unit
    # CATCHES: s01
    # CHAPTER: L13.4 section 2.3
    assert select_best_resolution((100, 100), [(200, 200), (400, 400)]) == (200, 200)
    assert select_best_resolution((100, 100), [(200, 300), (300, 200)]) == (200, 300)
    assert select_best_resolution((100, 100), [(300, 200), (200, 300)]) == (300, 200)
    assert select_best_resolution((300, 300), [(100, 100), (200, 400)]) == (200, 400)


def test_tokens_monotone_in_area():
    # WHY: a bigger copy of the same image never costs fewer tokens within
    #      one processor; rate limits and the encoder budget rely on it.
    # KIND: property
    # CATCHES: s03
    # CHAPTER: L13.4 section 2.4
    z = np.load(GOLDEN)
    lv = llava(z)
    for a, b in ((3, 4), (1, 1), (9, 16), (1, 5), (7, 2)):
        for cfg in (SMOLVLM, lv):
            counts = [
                image_token_count(a * k, b * k, cfg)
                for k in (16, 32, 64, 128, 256, 512)
            ]
            assert counts == sorted(counts), (a, b, cfg.kind, counts)


def test_rejects_bad_inputs():
    # WHY: a zero-size image, a shuffle factor that does not divide the
    #      grid, or pinpoints that are not tile multiples are caller bugs;
    #      they fail loudly instead of producing a wrong token count.
    # KIND: boundary
    # CATCHES: m01, m02, m03
    # CHAPTER: L13.4 section 4
    with pytest.raises(ValueError):
        image_token_count(0, 10, TilingConfig("single", 224, image_seq_len=49))
    with pytest.raises(ValueError):
        image_token_count(10, 10, SMOLVLM, detail="medium")
    with pytest.raises(ValueError):
        select_best_resolution((10, 10), [])
    with pytest.raises(ValueError):
        pixel_shuffle(np.zeros((1, 16, 2)), 3)
    with pytest.raises(ValueError):
        pixel_shuffle(np.zeros((1, 15, 2)), 1)
    with pytest.raises(ValueError):
        anyres_tiles(np.zeros((20, 20, 3), np.uint8), [(16, 24)], 16)
    with pytest.raises(ValueError):
        idefics3_split(np.zeros((20, 20, 3), np.float32), 48, 16)
