# /// script
# requires-python = ">=3.12"
# dependencies = ["librosa==1.0.0", "numpy==2.5.3"]
# ///
"""Maintainer generator for the M12.6 mel fixture (librosa).

librosa.filters.mel for Whisper's two banks (sr 16000, n_fft 400, n_mels 80
and 128, Slaney scale and norm) in float64 and in librosa's default float32
(the dtype Whisper's mel_filters.npz ships); an HTK bank without
normalization (sr 8000, n_fft 64, n_mels 10); a band-limited Slaney bank
(sr 22050, n_fft 512, n_mels 24, fmin 300, fmax 8000); librosa.hz_to_mel
and mel_to_hz on a frequency grid in both scales; and librosa.power_to_db
on a spectrum spanning 1e-14 to 1e3 with the defaults (ref 1, amin 1e-10,
top_db 80) and with ref 2.5, amin 1e-6, top_db None.

    uv run --offline --python 3.12 --script course/oracle/M12.6/mel_librosa.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import librosa
import numpy as np

OUT = Path("course/fixtures/M12.6/mel_librosa.npz")
gen = np.random.default_rng(20261206)


def main() -> None:
    arrs: dict[str, np.ndarray] = {}
    for n_mels in (80, 128):
        arrs[f"whisper{n_mels}.f64"] = librosa.filters.mel(
            sr=16000, n_fft=400, n_mels=n_mels, dtype=np.float64
        )
        arrs[f"whisper{n_mels}.f32"] = librosa.filters.mel(
            sr=16000, n_fft=400, n_mels=n_mels
        )
    arrs["htk"] = librosa.filters.mel(
        sr=8000, n_fft=64, n_mels=10, htk=True, norm=None, dtype=np.float64
    )
    arrs["band"] = librosa.filters.mel(
        sr=22050, n_fft=512, n_mels=24, fmin=300.0, fmax=8000.0, dtype=np.float64
    )
    hz = np.array(
        [0.0, 50.0, 200.0, 700.0, 999.0, 1000.0, 1001.0, 2500.0, 8000.0, 11025.0]
    )
    arrs["hz"] = hz
    arrs["hz_to_mel.slaney"] = librosa.hz_to_mel(hz)
    arrs["hz_to_mel.htk"] = librosa.hz_to_mel(hz, htk=True)
    mels = np.linspace(0.0, 45.0, 19)
    arrs["mels"] = mels
    arrs["mel_to_hz.slaney"] = librosa.mel_to_hz(mels)
    arrs["mel_to_hz.htk"] = librosa.mel_to_hz(mels * 60.0, htk=True)
    p = 10.0 ** gen.uniform(-14.0, 3.0, size=(6, 9))
    p[0, 0] = 0.0
    arrs["power"] = p
    arrs["db.default"] = librosa.power_to_db(p)
    arrs["db.ref"] = librosa.power_to_db(p, ref=2.5, amin=1e-6, top_db=None)
    meta = {
        "generator": "course/oracle/M12.6/mel_librosa.py",
        "librosa": librosa.__version__,
        "numpy": np.__version__,
    }
    arrs["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    data = buf.getvalue()
    OUT.write_bytes(data)
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/M12.6/mel_librosa.py\t"
        f"librosa=={librosa.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
