# L13.6 image preprocessing: D44 identity and D46 bounded decoding.
from typing import Literal

from numpy.typing import NDArray

def image_header(data: bytes) -> tuple[str, int, int]:
    """Container format, width and height before EXIF orientation, without
    loading pixels. Invalid headers raise ValueError."""

def exif_orientation(data: bytes) -> int | None:
    """Parsed EXIF tag 0x0112, or None when absent. Present values outside
    integer 1..8 raise ValueError. Does not apply the transform."""

def decode_image(data: bytes, max_pixels: int) -> NDArray:
    """First frame, uint8 [height,width,3] RGB, owned and C-contiguous.
    The budget is a positive integer and is checked against header width *
    height before decoding. Reject CMYK, 16-bit PNG (including true-color),
    and other modes except 1/L/LA/P/RGB/RGBA. Apply EXIF orientation 1..8;
    expand gray/palette channels and composite alpha over white, rounding
    half up. Invalid, truncated, oversized or unsupported images raise
    ValueError. Pillow may impose a stricter decompression-bomb limit.
    Pillow is used only for container parsing and decompression (D46)."""

def media_hash(kind: Literal["image", "audio"], data: bytes) -> int:
    """FNV-1a-64(kind byte || original media bytes); 1=image, 2=audio.
    Unsupported kinds raise ValueError. This is a content identifier,
    not a cryptographic integrity check."""
