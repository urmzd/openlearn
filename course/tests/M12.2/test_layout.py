"""Course tests for M12.2: to_layout, strides_of, rescale_normalize,
to_grayscale_pil, patchify, and unpatchify (tinyllm/image/layout.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M12.2), and the chapter section it comes from.

The chapter's worked example (section 3) is a float32 image with C, H, W =
2, 3, 4: stored CHW its byte strides are (48, 16, 4); stored HWC (channels
last) the same logical axes step (4, 32, 8) bytes. The Pillow golden pixels
in $TINYLLM_FIXTURES/M12.2/grayscale_pil.npz come from
course/oracle/M12.2/grayscale_pil.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.image.layout import (
    patchify,
    rescale_normalize,
    strides_of,
    to_grayscale_pil,
    to_layout,
    unpatchify,
)

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M12.2" / "grayscale_pil.npz"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def rng(seq: int) -> PCG32:
    return PCG32(int(os.environ.get("OL_SEED", "0")), seq=seq)


def pixels(g: PCG32, shape) -> np.ndarray:
    n = int(np.prod(shape))
    return np.array([g.below(256) for _ in range(n)], dtype=np.uint8).reshape(shape)


def test_hand_example_strides():
    # WHY: section 3's table. The logical axes are always (C, H, W); the
    #      layout only decides how far apart neighbours are in memory. In
    #      HWC the channel axis is the fastest (4 bytes) and a row step
    #      skips W * C values. numpy's own strides of the same views agree.
    # KIND: unit, smoke
    # CATCHES: s02
    # CHAPTER: M12.2 section 3
    assert strides_of((2, 3, 4), "NCHW", 4) == (48, 16, 4)
    assert strides_of((2, 3, 4), "NHWC", 4) == (4, 32, 8)
    assert strides_of((5, 2, 3, 4), "NCHW", 4) == (96, 48, 16, 4)
    assert strides_of((5, 2, 3, 4), "NHWC", 4) == (96, 4, 32, 8)
    hwc = np.zeros((3, 4, 2), dtype=np.float32)  # memory order H, W, C
    assert hwc.transpose(2, 0, 1).strides == strides_of((2, 3, 4), "NHWC", 4)
    assert np.zeros((2, 3, 4), dtype=np.float32).strides == strides_of(
        (2, 3, 4), "NCHW", 4
    )


def test_strides_match_numpy_for_any_shape():
    # WHY: channels-last strides for random shapes and item sizes must be the
    #      strides of an NHWC array viewed as NCHW, which is what
    #      torch.channels_last reports and what a C kernel indexes with.
    # KIND: property
    # CATCHES: s02
    # CHAPTER: M12.2 section 2.2
    g = rng(21)
    for dtype in (np.uint8, np.float32, np.float64):
        for _ in range(5):
            n, c, h, w = (1 + g.below(5) for _ in range(4))
            nhwc = np.zeros((n, h, w, c), dtype=dtype).transpose(0, 3, 1, 2)
            nchw = np.zeros((n, c, h, w), dtype=dtype)
            size = np.dtype(dtype).itemsize
            assert strides_of((n, c, h, w), "NHWC", size) == nhwc.strides
            assert strides_of((n, c, h, w), "NCHW", size) == nchw.strides


def test_to_layout_moves_the_bytes():
    # WHY: a transpose is only a view with permuted strides; the C kernels
    #      of L13.8 and the safetensors writer read raw memory, so the result
    #      must be C-contiguous with the new order in memory. Converting to
    #      the same layout must still copy: the caller may write to it.
    # KIND: property
    # CATCHES: s01, s07
    # CHAPTER: M12.2 section 2.2
    g = rng(22)
    x = g.normal_array((2, 3, 4, 5)).astype(np.float32)
    nhwc = to_layout(x, "NCHW", "NHWC")
    assert nhwc.shape == (2, 4, 5, 3) and nhwc.flags["C_CONTIGUOUS"]
    assert nhwc.dtype == np.float32
    assert_close(nhwc[1, 2, 3], x[1, :, 2, 3], rtol=0.0, atol=0.0)
    back = to_layout(nhwc, "NHWC", "NCHW")
    assert back.flags["C_CONTIGUOUS"] and (back == x).all()
    same = to_layout(x, "NCHW", "NCHW")
    same[0, 0, 0, 0] += 1.0
    assert same[0, 0, 0, 0] != x[0, 0, 0, 0]


def test_rescale_normalize_hand_values():
    # WHY: L13.6's processors scale u8 pixels to [0, 1], subtract the
    #      channel mean, divide by the channel std, and hand the model CHW
    #      float32. Order matters: (x * scale - mean) / std, not
    #      (x - mean) / std * scale.
    # KIND: unit
    # CATCHES: s04, s06
    # CHAPTER: M12.2 section 2.3
    img = np.array([[[0, 128, 255], [255, 255, 255]]], dtype=np.uint8)  # 1 x 2 x 3
    out = rescale_normalize(img, 1 / 255, CLIP_MEAN, CLIP_STD)
    assert out.shape == (3, 1, 2) and out.dtype == np.float32
    want = (img.astype(np.float64) / 255 - np.array(CLIP_MEAN)) / np.array(CLIP_STD)
    assert_close(out, want.transpose(2, 0, 1), rtol=1e-6, atol=1e-7)
    assert_close(out[0, 0, 0], -0.48145466 / 0.26862954, rtol=1e-6, atol=0.0)


def test_hand_example_grayscale():
    # WHY: Pillow's luma is fixed point in 1/65536ths. Pure blue at 250
    #      has exact luma 0.114 * 250 = 28.5: the thousandths rule rounds it
    #      up to 29, Pillow's 7471/65536 = 0.11399 gives 28.4996, so 28.
    # KIND: unit
    # CATCHES: s03
    # CHAPTER: M12.2 section 3
    img = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 250], [255, 255, 255]]],
        dtype=np.uint8,
    )
    assert to_grayscale_pil(img).tolist() == [[76, 150, 29, 28, 255]]


def test_grayscale_matches_pil():
    # WHY: data.10 hashes grayscale thumbnails; a one-level difference flips
    #      hash bits. The fixture's last rows are pixels on which the two
    #      rounding rules disagree, so only Pillow's rule passes.
    # KIND: golden
    # CATCHES: s03, m03
    # CHAPTER: M12.2 section 2.4
    z = np.load(GOLDEN)
    got = to_grayscale_pil(z["img"])
    assert got.dtype == np.uint8
    assert (got == z["gray"]).all(), (
        f"{int((got != z['gray']).sum())} pixels differ from Pillow"
    )
    with pytest.raises(ValueError):
        to_grayscale_pil(z["img"].astype(np.float32))


def test_hand_example_patchify():
    # WHY: a ViT reads patches row-major over the grid, each flattened in
    #      (c, i, j) order. For the 2 x 4 x 4 image arange(32), patch 1 is
    #      the top-right 2 x 2 block of channel 0, then of channel 1.
    # KIND: unit
    # CATCHES: s05
    # CHAPTER: M12.2 section 3
    x = np.arange(32).reshape(2, 4, 4)
    pt = patchify(x, 2)
    assert pt.shape == (4, 8)
    assert pt[1].tolist() == [2, 3, 6, 7, 18, 19, 22, 23]
    assert pt[2].tolist() == [8, 9, 12, 13, 24, 25, 28, 29]


def test_patchify_is_a_stride_p_conv():
    # WHY: L13.3's PatchEmbed is a conv with kernel = stride = p; flattening
    #      patches in (c, i, j) order makes it patchify followed by one matmul
    #      with the KCRS weight reshaped to [K, C*p*p]. unpatchify undoes it.
    # KIND: differential
    # CATCHES: s05
    # CHAPTER: M12.2 section 2.5
    g = rng(23)
    x = g.normal_array((3, 8, 12))
    w = g.normal_array((5, 3, 4, 4))
    y = patchify(x, 4) @ w.reshape(5, -1).T  # [6 patches, 5]
    for gi in range(2):
        for gj in range(3):
            block = x[:, 4 * gi : 4 * gi + 4, 4 * gj : 4 * gj + 4]
            want = np.einsum("kcij,cij->k", w, block)
            assert_close(y[gi * 3 + gj], want, rtol=1e-12, atol=1e-12)
    assert (unpatchify(patchify(x, 4), 3, 8, 12, 4) == x).all()
    assert (unpatchify(patchify(x, 2), 3, 8, 12, 2) == x).all()


def test_rejects_bad_inputs():
    # WHY: a wrong layout name, a patch size that does not tile the image,
    #      a zero std, or a zero-size dimension is a pipeline bug that must
    #      fail at the boundary, not as a NaN in the model.
    # KIND: boundary
    # CATCHES: m01, m02
    # CHAPTER: M12.2 section 4
    g = rng(24)
    x = pixels(g, (6, 4, 3))
    with pytest.raises(ValueError):
        to_layout(np.zeros((2, 3, 4)), "NCHW", "NHWC")
    with pytest.raises(ValueError):
        to_layout(np.zeros((1, 2, 3, 4)), "NCHW", "CHWN")
    with pytest.raises(ValueError):
        strides_of((0, 3, 4), "NCHW", 4)
    with pytest.raises(ValueError):
        rescale_normalize(x, 1 / 255, CLIP_MEAN, (0.2, 0.0, 0.3))
    with pytest.raises(ValueError):
        rescale_normalize(x, 1 / 255, CLIP_MEAN[:2], CLIP_STD[:2])
    with pytest.raises(ValueError):
        patchify(np.zeros((3, 6, 7)), 3)
    with pytest.raises(ValueError):
        unpatchify(np.zeros((4, 11)), 3, 4, 4, 2)
    before = x.copy()
    rescale_normalize(x, 1 / 255, CLIP_MEAN, CLIP_STD)
    to_grayscale_pil(x)
    assert (x == before).all()
