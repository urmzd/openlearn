# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.11.0", "torchaudio==2.11.0", "numpy==2.5.3"]
# ///
"""Maintainer generator for the M12.3 audio resampling fixture (torchaudio).

torchaudio's last release is 2.11.0 and it requires torch 2.11.0, so this
oracle pins that pair instead of the torch 2.14.1 the other oracles use
(course/DEVIATIONS.md, B14M12-03). torchaudio.functional.resample with its
default sinc_interp_hann method, on float64 signals (two tones plus noise,
0.03 s at the input rate):

  44100 -> 16000, 48000 -> 16000, 22050 -> 16000, 8000 -> 16000 (width 6, rolloff 0.99)
  44100 -> 16000 with lowpass_filter_width 16, rolloff 0.945
  a stereo [2, T] signal at 48000 -> 16000

    uv run --offline --python 3.12 --script course/oracle/M12.3/resample_torchaudio.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.functional as AF

OUT = Path("course/fixtures/M12.3/resample_torchaudio.npz")
gen = np.random.default_rng(20261213)


def signal(sr: int, channels: int = 1) -> np.ndarray:
    t = np.arange(int(0.03 * sr)) / sr
    x = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 3100 * t)
    x = x[None].repeat(channels, axis=0) + 0.05 * gen.standard_normal(
        (channels, t.size)
    )
    return x if channels > 1 else x[0]


def main() -> None:
    arrs: dict[str, np.ndarray] = {}
    cases = [
        ("a", 44100, 16000, 6, 0.99, 1),
        ("b", 48000, 16000, 6, 0.99, 1),
        ("c", 22050, 16000, 6, 0.99, 1),
        ("d", 8000, 16000, 6, 0.99, 1),
        ("e", 44100, 16000, 16, 0.945, 1),
        ("f", 48000, 16000, 6, 0.99, 2),
    ]
    for name, sr_in, sr_out, width, rolloff, ch in cases:
        x = signal(sr_in, ch)
        y = AF.resample(
            torch.from_numpy(x),
            sr_in,
            sr_out,
            lowpass_filter_width=width,
            rolloff=rolloff,
        )
        arrs[f"{name}.x"] = x
        arrs[f"{name}.y"] = y.numpy()
        arrs[f"{name}.args"] = np.array([sr_in, sr_out, width, rolloff])
    meta = {
        "generator": "course/oracle/M12.3/resample_torchaudio.py",
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "numpy": np.__version__,
    }
    arrs["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    data = buf.getvalue()
    OUT.write_bytes(data)
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/M12.3/resample_torchaudio.py\t"
        f"torch=={torch.__version__.split('+')[0]},torchaudio=={torchaudio.__version__.split('+')[0]},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
