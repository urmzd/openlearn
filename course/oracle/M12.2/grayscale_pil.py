# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow==12.3.0", "numpy==2.5.3"]
# ///
"""Maintainer generator for the M12.2 Pillow grayscale fixture.

Pillow's "L" conversion is ITU-R 601-2 luma rounded in 16-bit fixed point,
not in thousandths: the two rules disagree on about 0.05% of RGB triples (Pillow 12).
The fixture holds a 24 x 48 RGB image whose first 16 rows are random pixels
and whose last 8 rows are triples on which the thousandths rule
(R*299 + G*587 + B*114 + 500) // 1000 differs from Pillow, found by a seeded
search, plus Image.fromarray(img).convert("L") for it.

    uv run --offline --python 3.12 --script course/oracle/M12.2/grayscale_pil.py

Run from the repo root, then update the MANIFEST.tsv row it prints.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

OUT = Path("course/fixtures/M12.2/grayscale_pil.npz")
gen = np.random.default_rng(20261202)


def main() -> None:
    top = gen.integers(0, 256, size=(16, 48, 3), dtype=np.uint8)
    cand = gen.integers(0, 256, size=(1000000, 3), dtype=np.int64)
    r, g, b = cand.T
    thousandths = (r * 299 + g * 587 + b * 114 + 500) // 1000
    pil = np.asarray(Image.fromarray(cand.astype(np.uint8)[None], "RGB").convert("L"))[
        0
    ]
    hard = cand[thousandths != pil][: 8 * 48]
    assert len(hard) == 8 * 48, len(hard)
    img = np.concatenate([top, hard.astype(np.uint8).reshape(8, 48, 3)], axis=0)
    gray = np.asarray(Image.fromarray(img, "RGB").convert("L"))
    arrs = {"img": img, "gray": gray}
    meta = {
        "generator": "course/oracle/M12.2/grayscale_pil.py",
        "pillow": PIL.__version__,
        "numpy": np.__version__,
    }
    arrs["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    data = buf.getvalue()
    OUT.write_bytes(data)
    print(
        f"{OUT}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/M12.2/grayscale_pil.py\t"
        f"pillow=={PIL.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
