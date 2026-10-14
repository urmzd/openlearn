"""L13.6: bounded image decoding and learner-owned pixel transformations.

Pillow parses containers and decompresses pixels only (D46). Orientation,
palette expansion, alpha compositing, resizing and normalization belong here.
"""

from __future__ import annotations

import io
import warnings
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tinyllm.num.rng import fnv1a64


def image_header(data: bytes) -> tuple[str, int, int]:
    """Read format and un-oriented dimensions without decompressing pixels."""
    # SOLUTION-BEGIN L13.6
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                return image.format or "", image.width, image.height
    except (
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError(f"invalid image header: {exc}") from exc
    # SOLUTION-END


def exif_orientation(data: bytes) -> int | None:
    """Read tag 0x0112; reject invalid orientation values instead of guessing."""
    # SOLUTION-BEGIN L13.6
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                value = image.getexif().get(0x0112)
        if value is not None and (
            not isinstance(value, int) or value not in range(1, 9)
        ):
            raise ValueError(f"invalid EXIF orientation: {value!r}")
        return value
    except (
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError(f"invalid image EXIF: {exc}") from exc
    # SOLUTION-END


def _orient(pixels: NDArray, orientation: int | None) -> NDArray:
    # SOLUTION-BEGIN L13.6
    if orientation in (None, 1):
        result = pixels
    elif orientation == 2:
        result = pixels[:, ::-1]
    elif orientation == 3:
        result = pixels[::-1, ::-1]
    elif orientation == 4:
        result = pixels[::-1]
    elif orientation == 5:
        result = pixels.transpose(1, 0, 2)
    elif orientation == 6:
        result = np.rot90(pixels, -1, axes=(0, 1))
    elif orientation == 7:
        result = pixels.transpose(1, 0, 2)[::-1, ::-1]
    elif orientation == 8:
        result = np.rot90(pixels, 1, axes=(0, 1))
    else:
        raise ValueError(f"invalid EXIF orientation: {orientation!r}")
    return np.array(result, dtype=np.uint8, order="C", copy=True)
    # SOLUTION-END


def _rgb(image: Image.Image) -> NDArray:
    # SOLUTION-BEGIN L13.6
    a = np.asarray(image)
    alpha = None
    if image.mode == "P":
        palette = image.getpalette("RGBA")
        if palette is None:
            raise ValueError("palette image has no palette")
        table = np.asarray(palette, dtype=np.uint8).reshape(-1, 4)
        if int(a.max()) >= len(table):
            raise ValueError("pixel index exceeds palette")
        rgb, alpha = table[a, :3], table[a, 3]
        transparency = image.info.get("transparency")
        if isinstance(transparency, int):
            alpha = np.where(a == transparency, 0, alpha)
        elif isinstance(transparency, bytes):
            lut = np.full(256, 255, dtype=np.uint8)
            lut[: len(transparency)] = np.frombuffer(transparency, dtype=np.uint8)
            alpha = lut[a]
    elif image.mode in ("1", "L", "LA"):
        gray = a[..., 0] if image.mode == "LA" else a
        if image.mode == "1":
            gray = gray.astype(np.uint8) * 255
        rgb = np.repeat(gray[..., None], 3, axis=2)
        if image.mode == "LA":
            alpha = a[..., 1]
        elif "transparency" in image.info:
            alpha = np.where(gray == image.info["transparency"], 0, 255)
    elif image.mode in ("RGB", "RGBA"):
        rgb = a[..., :3]
        if image.mode == "RGBA":
            alpha = a[..., 3]
        elif "transparency" in image.info:
            alpha = np.where(np.all(rgb == image.info["transparency"], axis=-1), 0, 255)
    else:
        raise ValueError(f"unsupported image mode: {image.mode}")
    if alpha is not None:
        # Exact integer round-half-up of compositing over white. uint32
        # avoids overflow for both products and their sum.
        opacity = np.asarray(alpha, dtype=np.uint32)[..., None]
        rgb = (rgb.astype(np.uint32) * opacity + 255 * (255 - opacity) + 127) // 255
    return np.asarray(rgb, dtype=np.uint8)
    # SOLUTION-END


def decode_image(data: bytes, max_pixels: int) -> NDArray:
    """Decode one image to oriented, contiguous uint8 HWC RGB on white.

    Reject oversized headers and unsupported modes before Image.load. The first
    frame is used for animated containers. The caller owns the pixel budget.
    """
    # SOLUTION-BEGIN L13.6
    if (
        not isinstance(max_pixels, int)
        or isinstance(max_pixels, bool)
        or max_pixels < 1
    ):
        raise ValueError("max_pixels must be a positive integer")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > max_pixels:
                    raise ValueError(f"image exceeds max_pixels={max_pixels}")
                # Pillow exposes 16-bit RGB PNG as mode RGB after discarding
                # low bits, so inspect the IHDR depth before decompression.
                if image.format == "PNG" and len(data) > 24 and data[24] == 16:
                    raise ValueError("unsupported 16-bit PNG")
                if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
                    raise ValueError(f"unsupported image mode: {image.mode}")
                orientation = image.getexif().get(0x0112)
                if orientation is not None and (
                    not isinstance(orientation, int) or orientation not in range(1, 9)
                ):
                    raise ValueError(f"invalid EXIF orientation: {orientation!r}")
                image.load()
                return _orient(_rgb(image), orientation)
    except (
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError(f"invalid image: {exc}") from exc
    # SOLUTION-END


def media_hash(kind: Literal["image", "audio"], data: bytes) -> int:
    """D44: FNV-1a-64 of kind byte (1=image, 2=audio) followed by bytes."""
    # SOLUTION-BEGIN L13.6
    if kind not in ("image", "audio"):
        raise ValueError(f"unsupported media kind: {kind!r}")
    return fnv1a64(bytes([1 if kind == "image" else 2]) + data)
    # SOLUTION-END
