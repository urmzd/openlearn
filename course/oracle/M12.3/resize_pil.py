# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow==12.3.0", "torch==2.14.1", "torchvision==0.29.1", "numpy==2.5.3"]
# ///
"""Maintainer generator for the M12.3 resize fixtures (Pillow, torchvision, F.interpolate).

Six small uint8 images (inputs stored as arrays, so no decoder is involved):
noise 13x17x3, gradient 32x32x3, checker 24x40x3 (a 1-pixel checkerboard:
the worst case for aliasing), waves 50x30x3 (sums of sinusoids), tiny 3x5x3,
and gray 20x26 (a single-channel "L" image, stored as HW). Each is resized
by Pillow (Image.resize, reducing_gap None) with NEAREST, BILINEAR, BICUBIC,
and LANCZOS to four sizes: half, an odd shrink (0.37 x 0.61), an odd
enlargement (1.7 x 2.3), and width-only (height unchanged, so Pillow skips
the vertical pass).

Float mode: the same images scaled to [0, 1] float32 CHW, resized by
torchvision.transforms.v2.functional.resize with antialias=True (BILINEAR,
BICUBIC), stored HWC. NEAREST_EXACT is left out: torch computes its source
index in float32, so it picks a different neighbour than Pillow's running
double sum on some pixels; float mode follows Pillow.

Planar mode: torch.nn.functional.interpolate on float32 CHW tables
(nearest, bilinear and bicubic with and without align_corners, and the
antialiased bilinear and bicubic), including the ViT position-table case
(a 7 x 7 grid to 10 x 10, bicubic, align_corners False).

    uv run --offline --python 3.12 --script course/oracle/M12.3/resize_pil.py

Run from the repo root, then update the MANIFEST.tsv rows it prints.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import PIL
import torch
import torch.nn.functional as Fn
import torchvision
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TF

OUT_PIL = Path("course/fixtures/M12.3/resize_pil.npz")
OUT_FLOAT = Path("course/fixtures/M12.3/resize_torch.npz")
gen = np.random.default_rng(20261203)
KERNELS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos3": Image.Resampling.LANCZOS,
}


def images() -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:32, 0:32]
    grad = np.stack([yy * 8, xx * 8, (yy + xx) * 4], axis=-1) + gen.integers(
        0, 6, (32, 32, 3)
    )
    cy, cx = np.mgrid[0:24, 0:40]
    checker = ((cy + cx) % 2 * 255)[..., None].repeat(3, axis=-1)
    wy, wx = np.mgrid[0:50, 0:30]
    waves = np.stack(
        [
            127 + 120 * np.sin(wx * 0.9) * np.cos(wy * 0.4),
            127 + 120 * np.sin(wy * 1.3 + 0.5),
            127 + 100 * np.cos((wx + wy) * 0.7),
        ],
        axis=-1,
    )
    return {
        "noise": gen.integers(0, 256, (13, 17, 3)),
        "gradient": np.clip(grad, 0, 255),
        "checker": checker,
        "waves": np.round(waves),
        "tiny": gen.integers(0, 256, (3, 5, 3)),
        "gray": gen.integers(0, 256, (20, 26)),
    }


def sizes(h: int, w: int) -> list[tuple[int, int]]:
    return [
        (max(h // 2, 1), max(w // 2, 1)),
        (max(round(h * 0.37), 1), max(round(w * 0.61), 1)),
        (round(h * 1.7), round(w * 2.3)),
        (h, max(round(w * 0.5), 1)),
    ]


def save(path: Path, arrs: dict, deps: str) -> None:
    meta = {
        "generator": "course/oracle/M12.3/resize_pil.py",
        "pillow": PIL.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
    }
    arrs["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    data = buf.getvalue()
    path.write_bytes(data)
    print(
        f"{path}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/M12.3/resize_pil.py\t{deps}\t-\tApache-2.0"
    )


def main() -> None:
    imgs = {k: v.astype(np.uint8) for k, v in images().items()}
    pil: dict[str, np.ndarray] = {}
    flt: dict[str, np.ndarray] = {}
    for name, img in imgs.items():
        pil[f"{name}.in"] = img
        im = Image.fromarray(img, "L" if img.ndim == 2 else "RGB")
        for i, (h, w) in enumerate(sizes(*img.shape[:2])):
            pil[f"{name}.size{i}"] = np.array([h, w])
            for kname, k in KERNELS.items():
                pil[f"{name}.size{i}.{kname}"] = np.asarray(
                    im.resize((w, h), k, reducing_gap=None)
                )
            if img.ndim == 3:
                t = (
                    torch.from_numpy(img.astype(np.float32) / 255.0)
                    .permute(2, 0, 1)
                    .contiguous()
                )
                for kname, mode in (
                    ("bilinear", InterpolationMode.BILINEAR),
                    ("bicubic", InterpolationMode.BICUBIC),
                ):
                    out = TF.resize(t, [h, w], interpolation=mode, antialias=True)
                    flt[f"float.{name}.size{i}.{kname}"] = out.permute(1, 2, 0).numpy()
    for name, img in imgs.items():
        if img.ndim == 3:
            flt[f"float.{name}.in"] = img.astype(np.float32) / 255.0
            for i, hw in enumerate(sizes(*img.shape[:2])):
                flt[f"float.{name}.size{i}"] = np.array(hw)
    planar = [
        ("posembed", (4, 7, 7), (10, 10), "bicubic", False, False),
        ("bicubic_ac", (3, 9, 6), (5, 11), "bicubic", True, False),
        ("bilinear", (2, 5, 8), (9, 3), "bilinear", False, False),
        ("bilinear_ac", (2, 6, 6), (11, 4), "bilinear", True, False),
        ("nearest", (3, 10, 7), (4, 15), "nearest", False, False),
        ("bilinear_aa", (2, 17, 23), (6, 9), "bilinear", False, True),
        ("bicubic_aa", (2, 17, 23), (5, 31), "bicubic", False, True),
    ]
    for name, shape, size, mode, ac, aa in planar:
        x = gen.standard_normal(shape).astype(np.float32)
        kw = {} if mode == "nearest" else {"align_corners": ac}
        y = Fn.interpolate(
            torch.from_numpy(x)[None], size=size, mode=mode, antialias=aa, **kw
        )[0]
        flt[f"planar.{name}.in"] = x
        flt[f"planar.{name}.out"] = y.numpy()
        flt[f"planar.{name}.args"] = np.array([*size, int(ac), int(aa)])
    save(OUT_PIL, pil, f"pillow=={PIL.__version__},numpy=={np.__version__}")
    save(
        OUT_FLOAT,
        flt,
        f"torch=={torch.__version__.split('+')[0]},torchvision=={torchvision.__version__.split('+')[0]},numpy=={np.__version__}",
    )


if __name__ == "__main__":
    main()
