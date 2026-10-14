# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Maintainer generator for course/fixtures/digits/digits.npz: synthetic 8x8 digits.

DESIGN's L13.1 learning test names "digits 8x8" (the UCI handwritten digits
that scikit-learn ships). The course does not re-host that set; this script
draws a stand-in of the same shape from scratch (DEVIATIONS B14L-04): each
digit 0..9 is a 5x7 bitmap glyph placed at a random offset in an 8x8 canvas
(columns 0..3, rows 0..1), each stroke pixel gets a random intensity, about
8% of stroke pixels drop out, about 8% of background pixels light
up, and Gaussian noise (sigma 20) is added before clipping to uint8.

Keys: x_train uint8 [2000, 8, 8], y_train int64 [2000], x_test uint8
[400, 8, 8], y_test int64 [400]; classes are balanced in both splits.

    uv run --offline --python 3.12 --script course/oracle/L13.1/digits_synth.py
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

OUT = Path("course/fixtures/digits/digits.npz")
GLYPHS = {
    0: ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    1: ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    2: ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    3: ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    4: ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    5: ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    6: ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    7: ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    8: ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    9: ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}


def draw(gen, d: int) -> np.ndarray:
    g = np.array([[c == "1" for c in row] for row in GLYPHS[d]], dtype=bool)
    img = np.zeros((8, 8))
    r0, c0 = gen.integers(0, 2), gen.integers(0, 4)
    stroke = g & (gen.random(g.shape) > 0.08)
    img[r0 : r0 + 7, c0 : c0 + 5][stroke] = gen.uniform(150, 255, stroke.sum())
    speck = gen.random((8, 8)) < 0.08
    img[speck] += gen.uniform(40, 140, speck.sum())
    img += gen.normal(0, 20, (8, 8))
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def split(gen, per_class: int) -> tuple[np.ndarray, np.ndarray]:
    ys = np.repeat(np.arange(10), per_class)
    gen.shuffle(ys)
    return np.stack([draw(gen, int(y)) for y in ys]), ys.astype(np.int64)


def main() -> None:
    gen = np.random.default_rng(20261311)
    xtr, ytr = split(gen, 200)
    xte, yte = split(gen, 40)
    meta = {"generator": "course/oracle/L13.1/digits_synth.py", "numpy": np.__version__}
    arrs = {
        "x_train": xtr,
        "y_train": ytr,
        "x_test": xte,
        "y_test": yte,
        "__meta__": np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    b = buf.getvalue()
    OUT.write_bytes(b)
    print(
        f"{OUT}\t{hashlib.sha256(b).hexdigest()}\t{len(b)}\tcourse/oracle/L13.1/digits_synth.py\t"
        f"numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
