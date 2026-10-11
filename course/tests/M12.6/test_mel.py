"""Course tests for M12.6: hz_to_mel, mel_to_hz, mel_filterbank,
power_to_db, and log_mel_whisper (tinyllm/sig/mel.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M12.6), and the chapter section it comes from.

The chapter's worked example (section 3): on the Slaney scale 1000 Hz is
15 mel and 6400 Hz is 15 + 27 = 42 mel; the powers 1, 10, 1e-12 are 0, 10,
and -100 dB (amin 1e-10 floors the last), and -70 dB once top_db = 80
clips 80 dB below the loudest value. Golden filterbanks and decibels from
librosa 1.0 are in $TINYLLM_FIXTURES/M12.6/mel_librosa.npz, written by
course/oracle/M12.6/mel_librosa.py.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.sig.mel import (
    hz_to_mel,
    log_mel_whisper,
    mel_filterbank,
    mel_to_hz,
    power_to_db,
)

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M12.6" / "mel_librosa.npz"


def rng(seq: int) -> PCG32:
    return PCG32(int(os.environ.get("SS_SEED", "0")), seq=seq)


def test_hand_example_mel_scale():
    # WHY: section 3. Slaney's scale is linear at 200/3 Hz per mel up to
    #      1000 Hz (15 mel) and logarithmic above, 27 mel per factor 6.4;
    #      HTK's is 2595 log10(1 + f / 700) everywhere.
    # KIND: unit, smoke
    # CATCHES: s01
    # CHAPTER: M12.6 section 3
    assert_close(
        hz_to_mel(np.array([0.0, 500.0, 1000.0, 6400.0])),
        [0.0, 7.5, 15.0, 42.0],
        rtol=1e-14,
        atol=1e-14,
    )
    assert_close(
        mel_to_hz(np.array([7.5, 42.0])), [500.0, 6400.0], rtol=1e-14, atol=0.0
    )
    assert_close(
        hz_to_mel(np.array([700.0]), "htk"),
        [2595 * math.log10(2)],
        rtol=1e-14,
        atol=0.0,
    )


def test_hand_example_decibels():
    # WHY: 10 log10 of a power ratio; amin floors zeros and tiny values
    #      before the log, and top_db keeps only 80 dB below the loudest.
    # KIND: unit, smoke
    # CATCHES: s04, s05
    # CHAPTER: M12.6 section 3
    p = np.array([1.0, 10.0, 1e-12])
    assert_close(power_to_db(p, top_db=None), [0.0, 10.0, -100.0], rtol=0.0, atol=1e-12)
    assert_close(power_to_db(p), [0.0, 10.0, -70.0], rtol=0.0, atol=1e-12)
    assert_close(
        power_to_db(p, ref=10.0, top_db=None),
        [-10.0, 0.0, -110.0],
        rtol=0.0,
        atol=1e-12,
    )


@pytest.mark.parametrize("n_mels", [80, 128])
def test_whisper_filterbanks_match_librosa(n_mels):
    # WHY: Whisper ships librosa's filters (sr 16000, n_fft 400, Slaney
    #      scale and norm) as float32; every input feature of L14 is a dot
    #      product with them, so they must agree to 1e-7.
    # KIND: golden
    # CATCHES: s01, s02, s03
    # CHAPTER: M12.6 section 2.2
    z = np.load(GOLDEN)
    fb = mel_filterbank(16000, 400, n_mels)
    assert fb.shape == (n_mels, 201)
    assert_close(fb, z[f"whisper{n_mels}.f64"], rtol=0.0, atol=1e-12)
    assert_close(fb, z[f"whisper{n_mels}.f32"], rtol=0.0, atol=1e-7)


def test_other_filterbanks_and_scales_match_librosa():
    # WHY: the HTK scale without normalization and a band-limited bank
    #      (fmin 300, fmax 8000) pin the edges, and the scale conversions
    #      are checked on both sides of the 1000 Hz break.
    # KIND: golden
    # CATCHES: s01, s02, s03
    # CHAPTER: M12.6 section 2.1
    z = np.load(GOLDEN)
    assert_close(
        mel_filterbank(8000, 64, 10, scale="htk", norm=None),
        z["htk"],
        rtol=0.0,
        atol=1e-12,
    )
    assert_close(
        mel_filterbank(22050, 512, 24, fmin=300.0, fmax=8000.0),
        z["band"],
        rtol=0.0,
        atol=1e-12,
    )
    assert_close(hz_to_mel(z["hz"]), z["hz_to_mel.slaney"], rtol=1e-13, atol=1e-13)
    assert_close(hz_to_mel(z["hz"], "htk"), z["hz_to_mel.htk"], rtol=1e-13, atol=1e-13)
    assert_close(mel_to_hz(z["mels"]), z["mel_to_hz.slaney"], rtol=1e-13, atol=1e-12)
    assert_close(
        mel_to_hz(z["mels"] * 60.0, "htk"), z["mel_to_hz.htk"], rtol=1e-13, atol=1e-12
    )
    assert_close(
        mel_filterbank(8000, 64, 10, fmax=4000.0),
        mel_filterbank(8000, 64, 10),
        rtol=0.0,
        atol=0.0,
    )


def test_power_to_db_matches_librosa():
    # WHY: librosa.power_to_db with its defaults and with a reference
    #      level, a larger amin, and no top_db, on powers from 0 to 1e3.
    # KIND: golden
    # CATCHES: s04, s05
    # CHAPTER: M12.6 section 2.3
    z = np.load(GOLDEN)
    assert_close(power_to_db(z["power"]), z["db.default"], rtol=0.0, atol=1e-10)
    got = power_to_db(z["power"], ref=2.5, amin=1e-6, top_db=None)
    assert_close(got, z["db.ref"], rtol=0.0, atol=1e-10)


def test_triangles_have_the_declared_edges():
    # WHY: filter i must be the triangle rising from edge i to a peak at
    #      edge i + 1 and falling to edge i + 2, edges equally spaced in mel;
    #      without normalization its peak value at a bin is at most 1.
    # KIND: property
    # CATCHES: s03
    # CHAPTER: M12.6 section 2.2
    sr, n_fft, n_mels = 16000, 512, 20
    fb = mel_filterbank(sr, n_fft, n_mels, norm=None)
    edges = mel_to_hz(np.linspace(0.0, hz_to_mel(np.array(sr / 2.0)), n_mels + 2))
    f = np.arange(n_fft // 2 + 1) * sr / n_fft
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        want = np.clip(
            np.minimum((f - lo) / (mid - lo), (hi - f) / (hi - mid)), 0.0, None
        )
        assert_close(fb[i], want, rtol=0.0, atol=1e-12, msg=str(i))
        assert fb[i].max() <= 1.0 and (fb[i][(f <= lo) | (f >= hi)] == 0).all()


def test_slaney_rows_have_unit_area():
    # WHY: Slaney normalization divides each triangle by half its width in
    #      Hz, so every band integrates to 1 over frequency and a wide high
    #      band does not collect more energy than a narrow low one. On a fine
    #      frequency grid the sampled area is within a bin of 1.
    # KIND: property
    # CATCHES: s01, s02
    # CHAPTER: M12.6 section 2.2
    sr, n_fft = 16000, 16384
    fb = mel_filterbank(sr, n_fft, 40)
    area = fb.sum(axis=1) * sr / n_fft
    assert_close(area, np.ones(40), rtol=0.0, atol=0.02)


def test_whisper_log_mel_scaling():
    # WHY: Whisper's features are log10 of the mel power, floored at 1e-10,
    #      clipped to 8 units (80 dB) below the loudest value, then mapped by
    #      (x + 4) / 4. Silence becomes -1.5 everywhere; any input spans at
    #      most 2 units.
    # KIND: unit
    # CATCHES: s06, s07
    # CHAPTER: M12.6 section 2.4
    eye = np.eye(2)
    spec = np.array([[1e-12, 100.0], [1.0, 1e4]])
    assert_close(
        log_mel_whisper(spec, eye), [[0.0, 1.5], [1.0, 2.0]], rtol=0.0, atol=1e-14
    )
    fb = mel_filterbank(16000, 400, 80)
    silent = log_mel_whisper(np.zeros((201, 7)), fb)
    assert silent.shape == (80, 7)
    assert_close(silent, np.full((80, 7), -1.5), rtol=0.0, atol=1e-14)
    g = rng(61)
    loud = log_mel_whisper(g.uniform_array((201, 9), 0.0, 1e3) ** 3, fb)
    assert loud.max() - loud.min() <= 2.0 + 1e-12


def test_rejects_bad_arguments():
    # WHY: amin <= 0 would take the log of 0, a negative top_db clips
    #      everything, fmin >= fmax has no bands, and a complex spectrum
    #      must go through power() first.
    # KIND: boundary
    # CATCHES: m01, m02, m03
    # CHAPTER: M12.6 section 4
    for bad in (
        lambda: power_to_db(np.ones(3), amin=0.0),
        lambda: power_to_db(np.ones(3), top_db=-1.0),
        lambda: power_to_db(np.ones(3) + 1j),
        lambda: mel_filterbank(16000, 400, 80, fmin=8000.0, fmax=8000.0),
        lambda: mel_filterbank(16000, 400, 80, norm="area"),
        lambda: hz_to_mel(np.ones(2), "bark"),
        lambda: log_mel_whisper(np.ones((5, 3)), np.ones((4, 6))),
    ):
        with pytest.raises(ValueError):
            bad()
    p = np.array([0.0, 1.0, 5.0])
    before = p.copy()
    power_to_db(p)
    assert (p == before).all()
