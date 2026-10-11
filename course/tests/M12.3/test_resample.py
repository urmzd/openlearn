"""Course tests for M12.3: kernel_weights, resize, resize_planar_f32, and
resample_audio (tinyllm/sig/resample.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M12.3), and the chapter section it comes from.

The chapter's worked example (section 3) shrinks the row [10, 20, 30, 40]
to 2 pixels with bilinear antialiasing: the kernel stretches to support 2,
the left output's taps are pixels 0, 1, 2 with weights 3/7, 3/7, 1/7 after
renormalizing at the border, and Pillow's fixed point gives [17, 33].
Golden values: $TINYLLM_FIXTURES/M12.3/resize_pil.npz (Pillow 12.3),
resize_torch.npz (torchvision and F.interpolate), and
resample_torchaudio.npz (torchaudio 2.11), from course/oracle/M12.3/.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.sig.resample import (
    kernel_weights,
    resample_audio,
    resize,
    resize_planar_f32,
)

FIX = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M12.3"
IMAGES = ["noise", "gradient", "checker", "waves", "tiny", "gray"]
KERNELS = ["nearest", "bilinear", "bicubic", "lanczos3"]


def rng(seq: int) -> PCG32:
    return PCG32(int(os.environ.get("SS_SEED", "0")), seq=seq)


def test_hand_example_bilinear_antialias():
    # WHY: section 3. Shrinking 4 samples to 2 halves the sampling rate, so
    #      the triangle stretches to support 2: output 0 (center 1.0) takes
    #      pixels 0, 1, 2 at offsets -0.25, 0.25, 0.75 (in output units),
    #      weights 0.75, 0.75, 0.25, renormalized to 3/7, 3/7, 1/7 because
    #      the window is cut at the border. Pillow's fixed point then gives
    #      17 and 33 for the row [10, 20, 30, 40]. Without antialiasing the
    #      kernel keeps support 1 and averages only two pixels.
    # KIND: unit, smoke
    # CATCHES: s01, s02, s05
    # CHAPTER: M12.3 section 3
    start, w = kernel_weights(4, 2, "bilinear")
    assert start.tolist() == [0, 1]
    assert_close(
        w[:, :3], [[3 / 7, 3 / 7, 1 / 7], [1 / 7, 3 / 7, 3 / 7]], rtol=1e-15, atol=1e-16
    )
    assert (w[:, 3:] == 0).all()
    row = np.array([[10, 20, 30, 40]], dtype=np.uint8)
    assert resize(row, (1, 2), "bilinear").tolist() == [[17, 33]]
    s0, w0 = kernel_weights(4, 2, "bilinear", antialias=False)
    assert s0.tolist() == [0, 2]
    assert_close(w0[:, :2], [[0.5, 0.5], [0.5, 0.5]], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("kernel", KERNELS)
def test_pil_mode_is_pillow_bit_for_bit(kernel):
    # WHY: L13.6's preprocessing must reproduce Pillow, which rounds to
    #      uint8 between the horizontal and the vertical pass and computes
    #      in 22-bit fixed point. Six images, four sizes each (half, an odd
    #      shrink, an odd enlargement, width only), every pixel equal.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s05, s06, m01
    # CHAPTER: M12.3 section 2.4
    z = np.load(FIX / "resize_pil.npz")
    for name in IMAGES:
        img = z[f"{name}.in"]
        for i in range(4):
            h, w = (int(v) for v in z[f"{name}.size{i}"])
            want = z[f"{name}.size{i}.{kernel}"]
            got = resize(img, (h, w), kernel)
            assert got.dtype == np.uint8 and got.shape == want.shape, (name, i)
            diff = np.abs(got.astype(int) - want.astype(int))
            assert diff.max() == 0, (
                f"{name} size{i}: {int((diff > 0).sum())} pixels differ, max {diff.max()}"
            )


@pytest.mark.parametrize("kernel", ["bilinear", "bicubic"])
def test_float_mode_matches_torchvision(kernel):
    # WHY: the same separable weights without rounding are torchvision's
    #      antialiased float resize, the path a float pipeline takes.
    # KIND: golden
    # CATCHES: s01, s02, s04, s05, m01
    # CHAPTER: M12.3 section 2.3
    z = np.load(FIX / "resize_torch.npz")
    for name in IMAGES[:-1]:
        img = z[f"float.{name}.in"]
        for i in range(4):
            h, w = (int(v) for v in z[f"float.{name}.size{i}"])
            got = resize(img, (h, w), kernel, mode="float")
            assert_close(
                got,
                z[f"float.{name}.size{i}.{kernel}"],
                rtol=0.0,
                atol=1e-4,
                msg=f"{name} {i}",
            )


@pytest.mark.parametrize(
    "name,kernel",
    [
        ("posembed", "bicubic"),
        ("bicubic_ac", "bicubic"),
        ("bilinear", "bilinear"),
        ("bilinear_ac", "bilinear"),
        ("nearest", "nearest"),
        ("bilinear_aa", "bilinear"),
        ("bicubic_aa", "bicubic"),
    ],
)
def test_planar_matches_interpolate(name, kernel):
    # WHY: L13.3 resizes ViT position tables with F.interpolate(bicubic,
    #      align_corners=False), which samples at (o + 0.5) in/out - 0.5 with
    #      a = -0.75 and clamps reads at the border: a different convention
    #      from Pillow's, and the one HF checkpoints were trained with.
    # KIND: golden
    # CATCHES: s01, s02, s04, s05, s08, m01
    # CHAPTER: M12.3 section 2.5
    z = np.load(FIX / "resize_torch.npz")
    oh, ow, ac, aa = (int(v) for v in z[f"planar.{name}.args"])
    got = resize_planar_f32(
        z[f"planar.{name}.in"],
        (oh, ow),
        kernel,
        align_corners=bool(ac),
        antialias=bool(aa),
    )
    assert got.dtype == np.float32
    assert_close(got, z[f"planar.{name}.out"], rtol=0.0, atol=1e-5, msg=name)


@pytest.mark.parametrize("case", list("abcdef"))
def test_resample_audio_matches_torchaudio(case):
    # WHY: L14.1 brings every clip to 16 kHz with this function before the
    #      log-mel; Whisper was trained on torchaudio-style resampling.
    #      Rates 44.1k, 48k, 22.05k, 8k, a wider filter, and a stereo clip.
    # KIND: golden
    # CATCHES: s07
    # CHAPTER: M12.3 section 2.6
    z = np.load(FIX / "resample_torchaudio.npz")
    sr_in, sr_out, width, rolloff = (float(v) for v in z[f"{case}.args"])
    got = resample_audio(z[f"{case}.x"], int(sr_in), int(sr_out), int(width), rolloff)
    assert_close(got, z[f"{case}.y"], rtol=0.0, atol=1e-5, msg=case)


def test_weights_sum_to_one():
    # WHY: a row that sums to 1 maps a flat image to the same flat image,
    #      including next to the border, where taps fall off the edge and
    #      the survivors must be renormalized.
    # KIND: property
    # CATCHES: s02
    # CHAPTER: M12.3 section 2.2
    g = rng(31)
    for _ in range(12):
        n_in, n_out = 1 + g.below(40), 1 + g.below(40)
        for kernel in KERNELS:
            for aa in (True, False):
                start, w = kernel_weights(n_in, n_out, kernel, aa)
                assert start.shape == (n_out,) and w.shape[0] == n_out
                assert_close(
                    w.sum(axis=1),
                    np.ones(n_out),
                    rtol=0.0,
                    atol=1e-12,
                    msg=f"{kernel} {n_in}->{n_out}",
                )
                assert (start >= 0).all() and (start < n_in).all()
    flat = np.full((7, 9, 3), 77, dtype=np.uint8)
    for kernel in KERNELS:
        assert (resize(flat, (5, 13), kernel) == 77).all()


def test_identity_at_equal_size():
    # WHY: resizing to the same size must not blur. Pillow skips a pass
    #      whose size does not change, and the audio resampler returns a
    #      copy at equal rates.
    # KIND: property
    # CATCHES: m02
    # CHAPTER: M12.3 section 2.4
    g = rng(32)
    img = np.array([g.below(256) for _ in range(6 * 7 * 3)], dtype=np.uint8).reshape(
        6, 7, 3
    )
    for kernel in KERNELS:
        assert (resize(img, (6, 7), kernel) == img).all()
        out = resize(img, (6, 7), kernel, mode="float")
        assert_close(out, img.astype(np.float64), rtol=0.0, atol=0.0)
    x = g.normal_array((50,))
    y = resample_audio(x, 16000, 16000)
    assert (y == x).all() and y is not x
    with pytest.raises(ValueError):
        kernel_weights(4, 0, "bilinear")


def test_band_limited_round_trip():
    # WHY: a signal with nothing above the lower Nyquist limit survives
    #      16 kHz -> 48 kHz -> 16 kHz almost exactly: the sinc filters only
    #      remove what the coarser grid could not hold. Edges are skipped,
    #      where the zero padding enters the filters.
    # KIND: property
    # CATCHES: s07
    # CHAPTER: M12.3 section 2.6
    t = np.arange(1600) / 16000
    x = np.sin(2 * np.pi * 300 * t) + 0.5 * np.sin(2 * np.pi * 2100 * t + 0.3)
    up = resample_audio(x, 16000, 48000)
    assert up.shape == (4800,)
    back = resample_audio(up, 48000, 16000)
    assert back.shape == x.shape
    err = np.abs(back - x)[100:-100].max()
    assert err < 2e-3, err


def test_antialias_prevents_aliasing():
    # WHY: a 1-pixel checkerboard is all Nyquist frequency. Shrunk by 3
    #      without antialiasing, bilinear picks single pixels and the
    #      pattern aliases into pure black and white; with antialiasing the
    #      stretched kernel averages it to mid gray, the right answer.
    # KIND: property
    # CATCHES: s01, s05
    # CHAPTER: M12.3 section 2.3
    yy, xx = np.mgrid[0:24, 0:24]
    checker = ((yy + xx) % 2 * 255).astype(np.uint8)
    aliased = resize(checker, (8, 8), "bilinear", antialias=False)
    assert set(np.unique(aliased).tolist()) <= {0, 255}
    smooth = resize(checker, (8, 8), "bilinear")
    assert np.abs(smooth.astype(int) - 128).max() <= 30


def test_rejects_bad_arguments():
    # WHY: an unknown kernel, an empty output, float pixels in 'pil' mode,
    #      Lanczos without antialias in F.interpolate mode (PyTorch has no
    #      such mode), or a zero rolloff are caller bugs.
    # KIND: boundary
    # CATCHES: m03
    # CHAPTER: M12.3 section 4
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    for bad in (
        lambda: resize(img, (2, 2), "area"),
        lambda: resize(img, (0, 2), "bilinear"),
        lambda: resize(img.astype(np.float32), (2, 2), "bilinear"),
        lambda: resize(img, (2, 2), "bilinear", mode="exact"),
        lambda: resize_planar_f32(np.zeros((1, 4, 4)), (2, 2), "lanczos3"),
        lambda: resize_planar_f32(
            np.zeros((1, 4, 4)), (2, 2), "nearest", align_corners=True
        ),
        lambda: resample_audio(np.zeros(10), 16000, 8000, rolloff=0.0),
        lambda: resample_audio(np.zeros(10), 0, 8000),
    ):
        with pytest.raises(ValueError):
            bad()
    assert math.isfinite(float(resize(img, (2, 2), "lanczos3").sum()))
