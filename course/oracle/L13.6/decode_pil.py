# /// script
# requires-python = ">=3.12"
# dependencies = ["pillow==12.3.0", "numpy==2.5.3"]
# ///
"""Synthetic decode/orientation oracle; imports no learner implementation."""

import hashlib
import io
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageOps


def main():
    arrays = {}
    rgb = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3) * 5

    def save(name, image, fmt="PNG", **kwargs):
        buf = io.BytesIO()
        image.save(buf, format=fmt, **kwargs)
        data = buf.getvalue()
        arrays[name + ".bytes"] = np.frombuffer(data, dtype=np.uint8)
        with Image.open(io.BytesIO(data)) as loaded:
            oriented = ImageOps.exif_transpose(loaded).convert("RGBA")
            white = Image.new("RGBA", oriented.size, (255, 255, 255, 255))
            arrays[name + ".rgb"] = np.asarray(
                Image.alpha_composite(white, oriented).convert("RGB")
            )

    for orientation in range(1, 9):
        exif = Image.Exif()
        exif[0x0112] = orientation
        save(f"exif{orientation}", Image.fromarray(rgb), exif=exif)
    invalid_exif = Image.Exif()
    invalid_exif[0x0112] = 9
    save("invalid_exif", Image.fromarray(rgb), exif=invalid_exif)
    save("rgb", Image.fromarray(rgb))
    save("jpeg", Image.fromarray(rgb), "JPEG", quality=96, subsampling=0)
    save("gray", Image.fromarray(rgb[..., 0]))
    save("binary", Image.fromarray(rgb[..., 0] > 100))
    alpha = np.array([0, 1, 127, 128, 255], dtype=np.uint8)[None, :, None]
    rgba = np.concatenate([rgb, np.broadcast_to(alpha, (3, 5, 1))], axis=-1)
    save("rgba", Image.fromarray(rgba))
    save("la", Image.fromarray(np.stack([rgb[..., 0], rgba[..., 3]], axis=-1), "LA"))
    palette = Image.fromarray(np.arange(15, dtype=np.uint8).reshape(3, 5), "P")
    palette.putpalette(np.arange(768, dtype=np.uint16).astype(np.uint8))
    save("palette", palette)
    save("palette_alpha", palette, transparency=bytes([0, 1, 127, 128, 255]))
    save("palette_key", palette, transparency=3)
    save("rgb_key", Image.fromarray(rgb), transparency=tuple(int(x) for x in rgb[1, 1]))
    save("gray_key", Image.fromarray(rgb[..., 0]), transparency=int(rgb[1, 1, 0]))
    for name, image, fmt in [
        ("cmyk", Image.fromarray(rgb).convert("CMYK"), "JPEG"),
        ("png16", Image.fromarray(rgb[..., 0].astype(np.uint16) * 257), "PNG"),
    ]:
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        arrays[name + ".bytes"] = np.frombuffer(buf.getvalue(), dtype=np.uint8)

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    # True-color 16-bit PNG: Pillow otherwise silently drops low bytes.
    raw = b"".join(
        b"\0" + (row.astype(">u2") * 257).astype(">u2").tobytes() for row in rgb
    )
    png16 = b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", 5, 3, 16, 2, 0, 0, 0)
    )
    png16 += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    arrays["rgb16.bytes"] = np.frombuffer(png16, dtype=np.uint8)
    arrays["__meta__"] = np.frombuffer(
        json.dumps({"pillow": PIL.__version__, "numpy": np.__version__}).encode(),
        dtype=np.uint8,
    )
    out = Path("course/fixtures/L13.6/decode_pil.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    data = out.read_bytes()
    print(
        f"{out}\t{hashlib.sha256(data).hexdigest()}\t{len(data)}\tcourse/oracle/L13.6/decode_pil.py\tpillow=={PIL.__version__},numpy=={np.__version__}\t-\tApache-2.0"
    )


if __name__ == "__main__":
    main()
