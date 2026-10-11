"""Course tests for M12.5: hann, frame_count, stft, istft, and power
(tinyllm/sig/stft.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M12.5), and the chapter section it comes from.

The chapter's worked example (section 3) takes x = [1, 2, 3, 4, 5], n_fft 4,
hop 2, a periodic Hann window [0, 0.5, 1, 0.5], and center reflect padding
to [3, 2, 1, 2, 3, 4, 5, 4, 3]: three frames, and
S = [[3, 6, 9], [-1, -3 + i, -5], [-1, 0, 1]]. Golden values from
torch.stft are in $TINYLLM_FIXTURES/M12.5/stft_torch.npz, written by
course/oracle/M12.5/stft_torch.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.sig.stft import frame_count, hann, istft, power, stft

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M12.5" / "stft_torch.npz"


def rng(seq: int) -> PCG32:
    return PCG32(int(os.environ.get("OL_SEED", "0")), seq=seq)


def assert_complex_close(a, b, atol: float, msg: str = "") -> None:
    a, b = np.asarray(a), np.asarray(b)
    assert_close(
        np.stack([a.real, a.imag]),
        np.stack([b.real, b.imag]),
        rtol=0.0,
        atol=atol,
        msg=msg,
    )


def test_hand_example_stft():
    # WHY: section 3 by hand. Reflect padding mirrors without repeating
    #      the edge sample, the periodic Hann window zeroes each frame's
    #      first sample, and each column is the rfft of one windowed frame.
    # KIND: unit, smoke
    # CATCHES: s01, s02, s04, s06
    # CHAPTER: M12.5 section 3
    w = hann(4)
    assert_close(w, [0.0, 0.5, 1.0, 0.5], rtol=0.0, atol=1e-15)
    assert_close(hann(4, periodic=False), [0.0, 0.75, 0.75, 0.0], rtol=0.0, atol=1e-15)
    S = stft(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 4, 2, w)
    want = np.array([[3, 6, 9], [-1, -3 + 1j, -5], [-1, 0, 1]])
    assert S.shape == (3, 3)
    assert_complex_close(S, want, atol=1e-14)


def test_frame_count_table():
    # WHY: Whisper pads 30 s of 16 kHz audio (480000 samples) to 3001
    #      frames at hop 160 and drops the last, so its encoder sees 3000.
    #      With center=True an even n_fft gives 1 + n // hop; without
    #      padding the last frame must fit: 1 + (n - n_fft) // hop.
    # KIND: unit, smoke
    # CATCHES: s03, m03
    # CHAPTER: M12.5 section 2.4
    assert frame_count(480000, 400, 160) == 3001
    assert frame_count(1600, 400, 160) == 11
    assert frame_count(1600, 400, 160, center=False) == 8
    assert frame_count(100, 16, 5) == 21
    assert frame_count(5, 4, 2) == 3
    assert frame_count(16, 16, 4, center=False) == 1
    for bad in ((15, 16, 4, False), (0, 16, 4, True), (100, 16, 0, True)):
        with pytest.raises(ValueError):
            frame_count(*bad)


def test_hann_matches_torch():
    # WHY: torch.hann_window is periodic by default (period n, the window an
    #      FFT frame repeats); the symmetric window has period n - 1 and ends
    #      at 0 again. Whisper uses the periodic one.
    # KIND: golden
    # CATCHES: s01
    # CHAPTER: M12.5 section 2.2
    z = np.load(GOLDEN)
    for n in (1, 2, 8, 400):
        for p in (0, 1):
            assert_close(
                hann(n, bool(p)),
                z[f"hann.{n}.{p}"],
                rtol=0.0,
                atol=1e-15,
                msg=f"{n} {p}",
            )
    assert hann(0).shape == (0,)


@pytest.mark.parametrize("name", ["whisper", "small", "constant", "hop1"])
def test_stft_matches_torch(name):
    # WHY: L14.1 computes Whisper's spectrogram with exactly these
    #      semantics: center reflect padding, periodic Hann, n_fft // 2 + 1
    #      bins by frames. Zero padding and a symmetric window are covered too.
    # KIND: golden
    # CATCHES: s01, s02, s03, s04, s06
    # CHAPTER: M12.5 section 2.4
    z = np.load(GOLDEN)
    n_fft, hop, periodic, reflect = (int(v) for v in z[f"{name}.args"])
    x = z[f"{name}.x"]
    S = stft(
        x,
        n_fft,
        hop,
        hann(n_fft, bool(periodic)),
        True,
        "reflect" if reflect else "constant",
    )
    assert S.shape == z[f"{name}.S"].shape
    assert S.shape[1] == frame_count(x.size, n_fft, hop)
    assert_complex_close(S, z[f"{name}.S"], atol=1e-5, msg=name)


def test_istft_reconstructs_under_cola():
    # WHY: a periodic Hann window at hop n_fft / 4 has a constant
    #      overlap-added square (1.5), so weighted overlap-add inverts the
    #      STFT exactly: the frames lose nothing.
    # KIND: property
    # CATCHES: s04, s05, s06
    # CHAPTER: M12.5 section 2.3
    g = rng(51)
    for n_fft in (16, 64, 400):
        x = g.normal_array((n_fft * 5 + 3,))
        w = hann(n_fft)
        S = stft(x, n_fft, n_fft // 4, w)
        back = istft(S, n_fft, n_fft // 4, w, x.size)
        assert np.abs(back - x).max() < 1e-6, n_fft


def test_tone_above_nyquist_folds():
    # WHY: sampling at 8 kHz cannot hold 5 kHz: the samples of
    #      cos(2 pi 5000 t) are the samples of cos(2 pi 3000 t), so the
    #      STFT peak sits at bin 3000 * 256 / 8000 = 96, not at 160.
    # KIND: property
    # CATCHES: s06
    # CHAPTER: M12.5 section 2.1
    sr, n_fft = 8000, 256
    t = np.arange(4096) / sr
    S = stft(np.cos(2 * np.pi * 5000 * t), n_fft, 64, hann(n_fft))
    p = power(S).mean(axis=1)
    assert int(np.argmax(p)) == 96
    alias = np.cos(2 * np.pi * 3000 * t)
    assert_close(np.cos(2 * np.pi * 5000 * t), alias, rtol=0.0, atol=1e-9)


def test_power_and_periodic_hann_leakage():
    # WHY: power is |S|^2 with no square root. A tone exactly on bin 20 seen
    #      through a periodic Hann window, a sum of three cosines, lands in
    #      bins 19, 20, 21 only, with powers in the ratio 1 : 4 : 1.
    # KIND: property
    # CATCHES: s01, s04, s06, s07
    # CHAPTER: M12.5 section 2.2
    n = 128
    x = np.cos(2 * np.pi * 20 * np.arange(n) / n)
    S = stft(np.tile(x, 4), n, n, hann(n), center=False)
    p = power(S)[:, 1]
    assert_close(p, np.abs(S[:, 1]) ** 2, rtol=1e-12, atol=1e-12)
    others = np.delete(p, [19, 20, 21])
    assert others.max() < 1e-20 * p[20] + 1e-18
    assert_close([p[19] / p[20], p[21] / p[20]], [0.25, 0.25], rtol=1e-12, atol=0.0)


def test_rejects_bad_frames():
    # WHY: a window whose length is not n_fft, a signal too short to
    #      reflect-pad (torch raises too), hop 0, or a spectrogram of the
    #      wrong height cannot be framed or inverted.
    # KIND: boundary
    # CATCHES: m01, m02
    # CHAPTER: M12.5 section 4
    for bad in (
        lambda: stft(np.ones(100), 16, 4, hann(15)),
        lambda: stft(np.ones(8), 16, 4, hann(16)),
        lambda: stft(np.ones(100), 16, 0, hann(16)),
        lambda: stft(np.ones(100), 16, 4, hann(16), pad_mode="edge"),
        lambda: stft(np.ones(10), 16, 4, hann(16), center=False),
        lambda: istft(np.ones((8, 5), dtype=complex), 16, 4, hann(16), 20),
        lambda: hann(-1),
    ):
        with pytest.raises(ValueError):
            bad()
    x = np.arange(40.0)
    before = x.copy()
    stft(x, 8, 2, hann(8))
    assert (x == before).all()
