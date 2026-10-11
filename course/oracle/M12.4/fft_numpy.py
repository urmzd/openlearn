# /// script
# requires-python = ">=3.12"
# dependencies = ["scipy==1.17.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the M12.4 FFT and DCT fixture (numpy.fft, scipy.fft.dct).

Complex inputs for every length n in 1, 2, 3, 4, 5, 6, 8, 9, 12, 15, 16, 25,
30, 45, 64, 100, 240, 400, 512 with numpy.fft.fft of each; real inputs for
the even lengths with numpy.fft.rfft, and numpy.fft.irfft of a random
half-spectrum (imaginary DC and Nyquist parts included: irfft must drop
them); a batch [3, 400] for the batched FFT; scipy.fft.dct(type=2,
norm='ortho') of vectors of length 1, 2, 8, 15, 32, 64 and of a 32 x 32
block along each axis (the pHash case, data.10).

    uv run --offline --python 3.12 --script course/oracle/M12.4/fft_numpy.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import scipy
import scipy.fft

OUT = Path("course/fixtures/M12.4/fft_numpy.npz")
gen = np.random.default_rng(20261204)
SIZES = [1, 2, 3, 4, 5, 6, 8, 9, 12, 15, 16, 25, 30, 45, 64, 100, 240, 400, 512]


def main() -> None:
    arrs: dict[str, np.ndarray] = {"sizes": np.array(SIZES)}
    for n in SIZES:
        z = gen.standard_normal(n) + 1j * gen.standard_normal(n)
        arrs[f"fft.{n}.x"] = z
        arrs[f"fft.{n}.y"] = np.fft.fft(z)
        if n % 2 == 0:
            r = gen.standard_normal(n)
            arrs[f"rfft.{n}.x"] = r
            arrs[f"rfft.{n}.y"] = np.fft.rfft(r)
            half = gen.standard_normal(n // 2 + 1) + 1j * gen.standard_normal(
                n // 2 + 1
            )
            arrs[f"irfft.{n}.X"] = half
            arrs[f"irfft.{n}.y"] = np.fft.irfft(half, n)
    batch = gen.standard_normal((3, 400)) + 1j * gen.standard_normal((3, 400))
    arrs["batch.x"] = batch
    arrs["batch.y"] = np.fft.fft(batch, axis=-1)
    for n in (1, 2, 8, 15, 32, 64):
        v = gen.standard_normal(n)
        arrs[f"dct.{n}.x"] = v
        arrs[f"dct.{n}.y"] = scipy.fft.dct(v, type=2, norm="ortho")
    block = gen.standard_normal((32, 32))
    arrs["dct2d.x"] = block
    arrs["dct2d.axis0"] = scipy.fft.dct(block, type=2, norm="ortho", axis=0)
    arrs["dct2d.axis1"] = scipy.fft.dct(block, type=2, norm="ortho", axis=1)
    meta = {
        "generator": "course/oracle/M12.4/fft_numpy.py",
        "scipy": scipy.__version__,
        "numpy": np.__version__,
    }
    arrs["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    data = buf.getvalue()
    OUT.write_bytes(data)
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/M12.4/fft_numpy.py\t"
        f"scipy=={scipy.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
