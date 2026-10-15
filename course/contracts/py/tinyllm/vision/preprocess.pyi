# L13.6 image preprocessing: D44 identity and D46 bounded decoding.
# chapter: content/courses/build-an-llm/p13-vision/06-image-preprocessing.md
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

# Pipeline additions: one ImageInputs per original image, with no padding to
# another image's frame count. Idefics3 puts the global frame last; AnyRes first.
from collections.abc import Sequence
from dataclasses import dataclass

from tinyllm.sig.resample import Kernel
from tinyllm.vision.tiling import TilingConfig

@dataclass
class ProcessorConfig:
    kind: Literal["single", "idefics3", "llava_next"] = "single"
    do_resize: bool = True
    size: dict = ...
    resample: Kernel = "bicubic"
    do_center_crop: bool = True
    crop_size: dict = ...
    do_rescale: bool = True
    rescale_factor: float = ...
    do_normalize: bool = True
    image_mean: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073)
    image_std: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711)
    do_image_splitting: bool = False
    max_image_size: dict = ...
    tile_size: int = 224
    scale_factor: int = 1
    image_seq_len: int = 256
    image_grid_pinpoints: tuple[tuple[int, int], ...] = ()
    patch_size: int = 14
    base_extra: int = 0
    preprocess_rev: int = 1
    max_pixels: int = 40_000_000

    def validate(self) -> None:
        """Validate processor kind, booleans, positive dimensions, finite
        normalization with positive std, supported kernels and AnyRes tile
        multiples. preprocess_rev is a positive u32. Invalid configs fail."""

    @classmethod
    def from_hf(cls, preprocessor_config_json: str) -> ProcessorConfig:
        """Read a JSON file or a directory's preprocessor_config.json.
        Defaults follow CLIP, Siglip, Idefics3, SmolVLM, Llava or LlavaNext
        image_processor_type; optional Pil/Fast suffixes are accepted.
        Use the PIL reference recipe even for a Fast config. Unknown keys
        are ignored, unsupported values raise ValueError. HF resample enum:
        0 nearest, 1 lanczos3, 2 bilinear, 3 bicubic; reject BOX/HAMMING.
        Scalars mean/std expand to RGB. Read patch_size, scale_factor and
        image_seq_len from adjacent processor_config.json when absent here.
        Explicit image_seq_len wins; otherwise derive it from patch/merge
        geometry. Full feature selection retains additional base tokens."""

    def tiling(self) -> TilingConfig:
        """Validated L13.4 token geometry; longest_edge=0 disables the
        initial Idefics3 resize when do_resize is false."""

@dataclass
class ImageInputs:
    pixel_values: NDArray  # contiguous float32 [frames,3,H,W]
    pixel_mask: NDArray  # bool [frames,H,W]; active frames, all pixels valid
    grid: tuple[int, int]  # (0,0) for a single image
    n_tokens: int  # normative L13.4 count, including AnyRes newline tokens
    media_hash: int
    original_size: tuple[int, int] = (0, 0)  # oriented (H,W), needed for unpadding

def preprocess(
    images: Sequence[bytes],
    cfg: ProcessorConfig,
    detail: Literal["low", "high", "auto"] = "auto",
) -> list[ImageInputs]:
    """Decode under cfg.max_pixels, then process pixels. Hash original
    encoded bytes with D44. Inputs are unchanged. Empty input gives []."""

def preprocess_pixels(
    pixels: Sequence[NDArray],
    cfg: ProcessorConfig,
    detail: Literal["low", "high", "auto"] = "auto",
) -> list[ImageInputs]:
    """Nonempty uint8 HWC RGB arrays; preserve input storage. Resize via
    M12.3, crop (zero-pad small images, extra pixel top/left), split via
    L13.4, rescale then normalize via M12.2 to float32 CHW. Flags disable
    their individual steps. low disables splitting; high enables it;
    auto uses config. Single-image processors ignore splitting.
    No file bytes exist here: hash b'RGB8' || little-endian u32 H,W ||
    contiguous RGB bytes with D44. Geometry is included to prevent aliasing.
    Encoded and pixel-only identities intentionally differ. Reject invalid
    pixels/detail and intermediate/output allocations beyond max_pixels.
    Idefics3 also bounds its configured longest-edge square as a conservative
    intermediate allocation limit. preprocess_rev belongs in cache keys
    separately from the media hash (formats/image-preprocess.md)."""
