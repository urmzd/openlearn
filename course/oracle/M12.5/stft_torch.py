# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the M12.5 STFT fixture (torch.stft).

float64 signals (a chirp plus noise) and torch.stft(..., center=True,
return_complex=True) of each, with the window passed explicitly:

  whisper   1600 samples, n_fft 400, hop 160, hann_window(400, periodic=True), reflect
  small     257 samples,  n_fft 64,  hop 16,  hann_window(64, periodic=True),  reflect
  constant  300 samples,  n_fft 32,  hop 8,   hann_window(32, periodic=False), constant (zero) pad
  hop1      40 samples,   n_fft 16,  hop 5,   hann_window(16, periodic=True),  reflect
  window    torch.hann_window(n, periodic) for n in 1, 2, 8, 400 and both flags

    uv run --offline --python 3.12 --script course/oracle/M12.5/stft_torch.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch

OUT = Path("course/fixtures/M12.5/stft_torch.npz")
gen = np.random.default_rng(20261205)


def main() -> None:
    arrs: dict[str, np.ndarray] = {}
    cases = [
        ("whisper", 1600, 400, 160, True, "reflect"),
        ("small", 257, 64, 16, True, "reflect"),
        ("constant", 300, 32, 8, False, "constant"),
        ("hop1", 40, 16, 5, True, "reflect"),
    ]
    for name, n, n_fft, hop, periodic, pad in cases:
        t = np.arange(n) / n
        x = np.sin(2 * np.pi * (5 + 40 * t) * t * 8) + 0.1 * gen.standard_normal(n)
        win = torch.hann_window(n_fft, periodic=periodic, dtype=torch.float64)
        s = torch.stft(
            torch.from_numpy(x),
            n_fft,
            hop_length=hop,
            window=win,
            center=True,
            pad_mode=pad,
            return_complex=True,
        )
        arrs[f"{name}.x"] = x
        arrs[f"{name}.S"] = s.numpy()
        arrs[f"{name}.args"] = np.array(
            [n_fft, hop, int(periodic), int(pad == "reflect")]
        )
    for n in (1, 2, 8, 400):
        for periodic in (True, False):
            w = torch.hann_window(n, periodic=periodic, dtype=torch.float64)
            arrs[f"hann.{n}.{int(periodic)}"] = w.numpy()
    meta = {
        "generator": "course/oracle/M12.5/stft_torch.py",
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    arrs["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    data = buf.getvalue()
    OUT.write_bytes(data)
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/M12.5/stft_torch.py\t"
        f"torch=={torch.__version__.split('+')[0]},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
