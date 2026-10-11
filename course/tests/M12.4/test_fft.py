"""Course tests for M12.4: dft_matrix, fft, ifft, rfft, irfft, fft_convolve,
and dct2_ortho (tinyllm/sig/fft.py).

Rung R0: read these before you write code. Each test names why it exists
(WHY), what kind of check it is (KIND), the planted bugs it kills (CATCHES,
mutants in course/mutants/M12.4), and the chapter section it comes from.

The chapter's worked example (section 3) is the DFT of x = [1, 2, 0, -1]:
X = [2, 1 - 3i, 0, 1 + 3i], conjugate symmetric because x is real, with
sum |x|^2 = 6 = sum |X|^2 / 4 (Parseval). Golden values from numpy.fft and
scipy.fft.dct are in $TINYLLM_FIXTURES/M12.4/fft_numpy.npz, written by
course/oracle/M12.4/fft_numpy.py. The frozen accuracy bound for a length-n
transform of x is 8 log2(max(n, 2)) eps64 ||x||_2 in every entry.
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import numpy as np
import pytest
from _lib.close import assert_close
from _lib.pcg32 import PCG32
from tinyllm.sig.fft import (
    dct2_ortho,
    dft_matrix,
    fft,
    fft_convolve,
    ifft,
    irfft,
    rfft,
)

GOLDEN = Path(os.environ.get("TINYLLM_FIXTURES", "")) / "M12.4" / "fft_numpy.npz"
EPS = np.finfo(np.float64).eps


def bound(n: int, x) -> float:
    return 8.0 * math.log2(max(n, 2)) * EPS * float(np.linalg.norm(np.ravel(x)))


def rng(seq: int) -> PCG32:
    return PCG32(int(os.environ.get("OL_SEED", "0")), seq=seq)


def cnormal(g: PCG32, shape) -> np.ndarray:
    return g.normal_array(shape) + 1j * g.normal_array(shape)


def test_hand_example_dft():
    # WHY: section 3 by hand. With W = exp(-2 pi i / 4) = -i, X[1] =
    #      1 + 2(-i) + 0 - 1(i) = 1 - 3i. A real input gives X[n - k] =
    #      conj(X[k]), which is why rfft keeps only n // 2 + 1 bins.
    # KIND: unit, smoke
    # CATCHES: s02
    # CHAPTER: M12.4 section 3
    x = np.array([1.0, 2.0, 0.0, -1.0])
    want = np.array([2, 1 - 3j, 0, 1 + 3j])
    assert_close(fft(x).real, want.real, rtol=0.0, atol=1e-15)
    assert_close(fft(x).imag, want.imag, rtol=0.0, atol=1e-15)
    r = rfft(x)
    assert r.shape == (3,)
    assert_close(
        np.stack([r.real, r.imag]),
        np.stack([want[:3].real, want[:3].imag]),
        rtol=0.0,
        atol=1e-15,
    )
    assert_close(np.sum(np.abs(fft(x)) ** 2) / 4, 6.0, rtol=1e-15, atol=0.0)


def test_dft_matrix_is_unitary():
    # WHY: W / sqrt(n) is unitary: the DFT is a rotation of C^n, so it
    #      keeps lengths (Parseval) and its inverse is its conjugate
    #      transpose. Entries are exp(-2 pi i (k j mod n) / n).
    # KIND: property
    # CATCHES: m01
    # CHAPTER: M12.4 section 2.1
    for n in (1, 2, 5, 8, 12):
        u = dft_matrix(n, unitary=True)
        assert u.shape == (n, n) and u.dtype == np.complex128
        prod = u.conj().T @ u
        assert_close(prod.real, np.eye(n), rtol=0.0, atol=1e-14)
        assert_close(prod.imag, np.zeros((n, n)), rtol=0.0, atol=1e-14)
    w = dft_matrix(6)
    assert_close(
        w[2, 5].imag, math.sin(-2 * math.pi * (10 % 6) / 6), rtol=0.0, atol=1e-15
    )
    with pytest.raises(ValueError):
        dft_matrix(0)


def test_fft_matches_numpy():
    # WHY: every supported size, including 400 = 4 * 4 * 5 * 5 (Whisper's
    #      frame) and 240 = 4 * 4 * 3 * 5, against numpy's pocketfft within
    #      the frozen bound; a batch of three frames transforms row by row.
    # KIND: golden
    # CATCHES: s01, s07
    # CHAPTER: M12.4 section 2.3
    z = np.load(GOLDEN)
    for n in (int(v) for v in z["sizes"]):
        x = z[f"fft.{n}.x"]
        got = fft(x)
        assert got.shape == (n,)
        err = float(np.abs(got - z[f"fft.{n}.y"]).max())
        assert err <= bound(n, x), f"n={n}: error {err:.3e} > bound {bound(n, x):.3e}"
    xb = z["batch.x"]
    err = float(np.abs(fft(xb) - z["batch.y"]).max())
    assert err <= bound(400, xb[0]) * 2, err


def test_fft_equals_the_dft_matrix():
    # WHY: the FFT is only a fast way to multiply by W. Mixed radix sizes
    #      6, 9, 10, 20, 45 exercise every radix and their combinations.
    # KIND: differential
    # CATCHES: s01, s07
    # CHAPTER: M12.4 section 2.3
    g = rng(41)
    for n in (1, 2, 3, 6, 9, 10, 20, 45):
        x = cnormal(g, (n,))
        d = dft_matrix(n) @ x
        assert np.abs(fft(x) - d).max() <= bound(n, x) * 4, n


def test_rfft_and_irfft_match_numpy():
    # WHY: M12.5's STFT runs rfft on every frame: one complex FFT of half
    #      the size on z[m] = x[2m] + i x[2m + 1], then an untangling step.
    #      irfft must ignore the imaginary parts of the DC and Nyquist bins,
    #      which a real signal cannot have.
    # KIND: golden
    # CATCHES: s01, s02, s03, s06, s07
    # CHAPTER: M12.4 section 2.4
    z = np.load(GOLDEN)
    for n in (int(v) for v in z["sizes"] if v % 2 == 0):
        x = z[f"rfft.{n}.x"]
        got = rfft(x)
        assert got.shape == (n // 2 + 1,)
        assert np.abs(got - z[f"rfft.{n}.y"]).max() <= bound(n, x), n
        half = z[f"irfft.{n}.X"]
        back = irfft(half, n)
        assert back.shape == (n,) and back.dtype == np.float64
        assert np.abs(back - z[f"irfft.{n}.y"]).max() <= bound(n, half), n


def test_inverse_and_parseval():
    # WHY: ifft undoes fft (it is the conjugate transform divided by n),
    #      and the unitary DFT keeps energy: sum |x|^2 = sum |X|^2 / n.
    # KIND: property
    # CATCHES: s01, s02, s03, s07
    # CHAPTER: M12.4 section 2.1
    g = rng(42)
    for n in (4, 12, 50, 400):
        x = cnormal(g, (3, n))
        X = fft(x)
        assert np.abs(ifft(X) - x).max() <= 2 * bound(n, x[0])
        assert_close(
            np.sum(np.abs(X) ** 2, axis=-1) / n,
            np.sum(np.abs(x) ** 2, axis=-1),
            rtol=1e-12,
            atol=0.0,
        )
        r = g.normal_array((n,))
        assert np.abs(irfft(rfft(r), n) - r).max() <= 2 * bound(n, r)


def test_fft_convolve_is_linear_convolution():
    # WHY: the convolution theorem turns a product of spectra into a
    #      CIRCULAR convolution; zero-padding to at least len(a) + len(b) - 1
    #      makes it the linear one numpy.convolve computes. Lengths 7 and 5
    #      need 11, padded to the 5-smooth size 12.
    # KIND: differential
    # CATCHES: s01, s03, s04, s07
    # CHAPTER: M12.4 section 2.2
    g = rng(43)
    for la, lb in ((7, 5), (1, 1), (16, 3), (30, 31), (2, 9)):
        a, b = g.normal_array((la,)), g.normal_array((lb,))
        got = fft_convolve(a, b)
        assert got.shape == (la + lb - 1,) and got.dtype == np.float64
        assert_close(got, np.convolve(a, b), rtol=1e-12, atol=1e-12, msg=f"{la} {lb}")


def test_dct_matches_scipy_and_is_orthogonal():
    # WHY: data.10's pHash takes an orthonormal DCT-II of a 32 x 32 block
    #      along both axes. norm='ortho' scales bin 0 by sqrt(1/(4n)) and the
    #      rest by sqrt(1/(2n)), which makes the DCT matrix orthogonal.
    # KIND: golden
    # CATCHES: s01, s05, s07
    # CHAPTER: M12.4 section 2.5
    z = np.load(GOLDEN)
    for n in (1, 2, 8, 15, 32, 64):
        assert_close(
            dct2_ortho(z[f"dct.{n}.x"]),
            z[f"dct.{n}.y"],
            rtol=0.0,
            atol=1e-13,
            msg=str(n),
        )
    assert_close(
        dct2_ortho(z["dct2d.x"], axis=0), z["dct2d.axis0"], rtol=0.0, atol=1e-13
    )
    assert_close(
        dct2_ortho(z["dct2d.x"], axis=1), z["dct2d.axis1"], rtol=0.0, atol=1e-13
    )
    m = dct2_ortho(np.eye(16), axis=0)
    assert_close(m.T @ m, np.eye(16), rtol=0.0, atol=1e-14)


def test_unsupported_sizes_raise():
    # WHY: a size with a prime factor other than 2, 3, 5 has no radix here;
    #      it must fail loudly instead of falling back to a slow O(n^2) path
    #      that hides a misconfigured frame length. rfft needs an even real
    #      input; irfft needs exactly n // 2 + 1 bins.
    # KIND: boundary
    # CATCHES: m02, m03
    # CHAPTER: M12.4 section 4
    for n in (7, 14, 22):
        with pytest.raises(ValueError):
            fft(np.ones(n))
    with pytest.raises(ValueError):
        fft(np.ones(0))
    with pytest.raises(ValueError):
        rfft(np.ones(9))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            rfft(np.ones(8) + 1j)
    with pytest.raises(ValueError):
        irfft(np.ones(6, dtype=complex), 8)
    x = np.arange(8.0)
    before = x.copy()
    fft(x)
    rfft(x)
    dct2_ortho(x)
    assert (x == before).all()
