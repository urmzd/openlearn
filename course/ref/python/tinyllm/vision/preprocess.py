"""L13.6: bounded image decoding and learner-owned pixel transformations.

Pillow parses containers and decompresses pixels only (D46). Orientation,
palette expansion, alpha compositing, resizing and normalization belong here.
"""

from __future__ import annotations

import io
import json
import math
import struct
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tinyllm.image.layout import rescale_normalize
from tinyllm.num.rng import fnv1a64
from tinyllm.sig.resample import Kernel, resize
from tinyllm.vision.tiling import (
    TilingConfig,
    anyres_tiles,
    idefics3_resize_size,
    idefics3_split,
    image_token_count,
    select_best_resolution,
)


@dataclass
class ProcessorConfig:
    kind: Literal["single", "idefics3", "llava_next"] = "single"
    do_resize: bool = True
    size: dict = field(default_factory=lambda: {"shortest_edge": 224})
    resample: Kernel = "bicubic"
    do_center_crop: bool = True
    crop_size: dict = field(default_factory=lambda: {"height": 224, "width": 224})
    do_rescale: bool = True
    rescale_factor: float = 1 / 255
    do_normalize: bool = True
    image_mean: tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073)
    image_std: tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711)
    do_image_splitting: bool = False
    max_image_size: dict = field(default_factory=lambda: {"longest_edge": 224})
    tile_size: int = 224
    scale_factor: int = 1
    image_seq_len: int = 256
    image_grid_pinpoints: tuple[tuple[int, int], ...] = ()
    patch_size: int = 14
    base_extra: int = 0
    preprocess_rev: int = 1
    max_pixels: int = 40_000_000

    def validate(self) -> None:
        # SOLUTION-BEGIN L13.6
        if self.kind not in ("single", "idefics3", "llava_next"):
            raise ValueError(f"unsupported processor kind: {self.kind!r}")
        for name in (
            "do_resize",
            "do_center_crop",
            "do_rescale",
            "do_normalize",
            "do_image_splitting",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")  # noqa: TRY004 - malformed config value
        for name in (
            "tile_size",
            "scale_factor",
            "image_seq_len",
            "patch_size",
            "max_pixels",
            "preprocess_rev",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.preprocess_rev > 0xFFFFFFFF
            or type(self.base_extra) is not int
            or self.base_extra < 0
        ):
            raise ValueError("invalid preprocess_rev or base_extra")
        if self.resample not in ("nearest", "bilinear", "bicubic", "lanczos3"):
            raise ValueError(f"unsupported resample: {self.resample!r}")
        if not math.isfinite(self.rescale_factor):
            raise ValueError("rescale_factor must be finite")
        if len(self.image_mean) != 3 or len(self.image_std) != 3:
            raise ValueError("image_mean and image_std must have three channels")
        if (
            not all(math.isfinite(x) for x in (*self.image_mean, *self.image_std))
            or min(self.image_std) <= 0
        ):
            raise ValueError(
                "normalization requires finite means and positive finite standard deviations"
            )
        for name in ("size", "crop_size", "max_image_size"):
            value = getattr(self, name)
            if (
                not isinstance(value, dict)
                or not value
                or any(type(v) is not int or v < 1 for v in value.values())
            ):
                raise ValueError(f"{name} must contain positive integer dimensions")
        if self.kind == "idefics3":
            if set(self.size) != {"longest_edge"} or self.max_image_size != {
                "longest_edge": self.tile_size
            }:
                raise ValueError(
                    "Idefics3 requires longest_edge sizes and max_image_size equal to tile_size"
                )
            if self.do_center_crop:
                raise ValueError("Idefics3 does not center-crop")
        elif set(self.size) not in (
            {"height", "width"},
            {"shortest_edge"},
            {"shortest_edge", "longest_edge"},
        ):
            raise ValueError(
                "size requires height/width or shortest_edge with optional longest_edge"
            )
        if (
            "shortest_edge" in self.size
            and "longest_edge" in self.size
            and self.size["longest_edge"] <= self.size["shortest_edge"]
        ):
            raise ValueError("longest_edge must exceed shortest_edge")
        if set(self.crop_size) != {"height", "width"}:
            raise ValueError("crop_size requires height and width")
        if self.kind == "llava_next":
            if not self.image_grid_pinpoints or any(
                len(p) != 2
                or any(type(v) is not int or v < 1 or v % self.tile_size for v in p)
                for p in self.image_grid_pinpoints
            ):
                raise ValueError("AnyRes pinpoints must be positive tile multiples")
            if self.tile_size % self.patch_size:
                raise ValueError("patch_size must divide the AnyRes tile")
        # SOLUTION-END

    @classmethod
    def from_hf(cls, preprocessor_config_json: str) -> ProcessorConfig:
        # SOLUTION-BEGIN L13.6
        try:
            return cls._from_hf(preprocessor_config_json)
        except (TypeError, KeyError, AttributeError, OverflowError) as exc:
            raise ValueError(f"invalid processor configuration: {exc}") from exc
        # SOLUTION-END

    @classmethod
    def _from_hf(cls, preprocessor_config_json: str) -> ProcessorConfig:
        # SOLUTION-BEGIN L13.6
        path = Path(preprocessor_config_json)
        if path.is_dir():
            path = path / "preprocessor_config.json"
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError("preprocessor config must be an object")  # noqa: TRY004 - JSON config
        name = (
            raw.get("image_processor_type", "CLIPImageProcessor")
            .removesuffix("Pil")
            .removesuffix("Fast")
        )
        if name not in (
            "CLIPImageProcessor",
            "SiglipImageProcessor",
            "Idefics3ImageProcessor",
            "SmolVLMImageProcessor",
            "LlavaImageProcessor",
            "LlavaNextImageProcessor",
        ):
            raise ValueError(f"unsupported image_processor_type: {name!r}")
        defaults = {}
        if name == "SiglipImageProcessor":
            defaults = {
                "size": {"height": 224, "width": 224},
                "do_center_crop": False,
                "image_mean": (0.5,) * 3,
                "image_std": (0.5,) * 3,
            }
        elif name in ("Idefics3ImageProcessor", "SmolVLMImageProcessor"):
            tile = 512 if name == "SmolVLMImageProcessor" else 364
            defaults = {
                "kind": "idefics3",
                "size": {"longest_edge": tile * 4},
                "max_image_size": {"longest_edge": tile},
                "tile_size": tile,
                "scale_factor": 4 if tile == 512 else 2,
                "patch_size": 16 if tile == 512 else 14,
                "do_center_crop": False,
                "do_image_splitting": True,
                "resample": "lanczos3",
                "image_mean": (0.5,) * 3,
                "image_std": (0.5,) * 3,
            }
        elif name == "LlavaNextImageProcessor":
            defaults = {
                "kind": "llava_next",
                "size": {"shortest_edge": 336},
                "crop_size": {"height": 336, "width": 336},
                "tile_size": 336,
                "do_image_splitting": True,
                "image_grid_pinpoints": (
                    (336, 672),
                    (672, 336),
                    (672, 672),
                    (1008, 336),
                    (336, 1008),
                ),
            }
        # Token geometry lives in processor_config.json for several HF models.
        companion = path.with_name("processor_config.json")
        if companion.is_file() and companion != path:
            extra = json.loads(companion.read_text())
            if not isinstance(extra, dict):
                raise ValueError("processor config must be an object")
            for key in ("image_seq_len", "scale_factor", "patch_size"):
                if key in extra and key not in raw:
                    raw[key] = extra[key]
            if (
                "base_extra" not in raw
                and extra.get("vision_feature_select_strategy") == "full"
            ):
                raw["base_extra"] = extra.get("num_additional_image_tokens", 1)
        known = {f.name for f in fields(cls)}
        cfg = cls(**(defaults | {k: v for k, v in raw.items() if k in known}))
        if isinstance(cfg.resample, int) and not isinstance(cfg.resample, bool):
            cfg.resample = {
                0: "nearest",
                1: "lanczos3",
                2: "bilinear",
                3: "bicubic",
            }.get(cfg.resample, "unsupported")
        for attr in ("image_mean", "image_std"):
            value = getattr(cfg, attr)
            setattr(
                cfg,
                attr,
                (float(value),) * 3
                if isinstance(value, (float, int))
                else tuple(value),
            )
        if type(cfg.size) is int:
            cfg.size = (
                {"longest_edge": cfg.size}
                if cfg.kind == "idefics3"
                else {"shortest_edge": cfg.size}
            )
        if type(cfg.crop_size) is int:
            cfg.crop_size = {"height": cfg.crop_size, "width": cfg.crop_size}
        if "tile_size" not in raw:
            cfg.tile_size = (
                cfg.max_image_size.get("longest_edge", cfg.tile_size)
                if cfg.kind == "idefics3"
                else cfg.crop_size["height"]
                if cfg.do_center_crop or cfg.kind == "llava_next"
                else cfg.size.get(
                    "height", cfg.size.get("shortest_edge", cfg.tile_size)
                )
            )
        cfg.image_grid_pinpoints = tuple(tuple(p) for p in cfg.image_grid_pinpoints)
        if "image_seq_len" not in raw:
            if (
                type(cfg.patch_size) is not int
                or cfg.patch_size < 1
                or type(cfg.scale_factor) is not int
                or cfg.scale_factor < 1
            ):
                raise ValueError(
                    "patch_size and scale_factor must be positive integers"
                )
            feature_h = cfg.tile_size
            feature_w = cfg.tile_size
            if cfg.kind == "single":
                geometry = cfg.crop_size if cfg.do_center_crop else cfg.size
                feature_h = geometry.get("height", cfg.tile_size)
                feature_w = geometry.get("width", cfg.tile_size)
            stride = cfg.patch_size * cfg.scale_factor
            cfg.image_seq_len = (feature_h // stride) * (feature_w // stride)
        cfg.validate()
        return cfg
        # SOLUTION-END

    def tiling(self) -> TilingConfig:
        # SOLUTION-BEGIN L13.6
        self.validate()
        return TilingConfig(
            self.kind,
            self.tile_size,
            longest_edge=self.size.get("longest_edge", 0) if self.do_resize else 0,
            image_seq_len=self.image_seq_len,
            pinpoints=self.image_grid_pinpoints,
            patch_size=self.patch_size,
            base_extra=self.base_extra,
            split=self.do_image_splitting,
        )
        # SOLUTION-END


@dataclass
class ImageInputs:
    pixel_values: NDArray
    pixel_mask: NDArray
    grid: tuple[int, int]
    n_tokens: int
    media_hash: int
    original_size: tuple[int, int] = (0, 0)


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


def _resize_size(hw: tuple[int, int], size: dict) -> tuple[int, int]:
    # SOLUTION-BEGIN L13.6
    if "height" in size:
        return size["height"], size["width"]
    h, w = hw
    short, long = min(h, w), max(h, w)
    new_short = size["shortest_edge"]
    new_long = int(new_short * long / short)
    maximum = size.get("longest_edge")
    if maximum is not None and new_long > maximum:
        new_short, new_long = max(1, int(maximum * new_short / new_long)), maximum
    return (new_short, new_long) if h <= w else (new_long, new_short)
    # SOLUTION-END


def _center_crop(a: NDArray, size: tuple[int, int]) -> NDArray:
    # SOLUTION-BEGIN L13.6
    h, w = a.shape[:2]
    oh, ow = size
    top, left = (h - oh) // 2, (w - ow) // 2
    out = np.zeros((oh, ow, 3), dtype=np.uint8)
    sy, sx = max(top, 0), max(left, 0)
    dy, dx = max(-top, 0), max(-left, 0)
    nh, nw = min(h - sy, oh - dy), min(w - sx, ow - dx)
    out[dy : dy + nh, dx : dx + nw] = a[sy : sy + nh, sx : sx + nw]
    return out
    # SOLUTION-END


def _single(a: NDArray, cfg: ProcessorConfig) -> NDArray:
    # SOLUTION-BEGIN L13.6
    if cfg.do_resize:
        hw = _resize_size(a.shape[:2], cfg.size)
        if hw[0] * hw[1] > cfg.max_pixels:
            raise ValueError("resized image exceeds max_pixels")
        a = resize(a, hw, cfg.resample)
    if cfg.do_center_crop:
        hw = (cfg.crop_size["height"], cfg.crop_size["width"])
        if hw[0] * hw[1] > cfg.max_pixels:
            raise ValueError("crop exceeds max_pixels")
        a = _center_crop(a, hw)
    return a
    # SOLUTION-END


def preprocess_pixels(
    pixels: Sequence[NDArray],
    cfg: ProcessorConfig,
    detail: Literal["low", "high", "auto"] = "auto",
) -> list[ImageInputs]:
    # SOLUTION-BEGIN L13.6
    cfg.validate()
    if detail not in ("low", "high", "auto"):
        raise ValueError("detail must be low, high, or auto")
    split = cfg.do_image_splitting if detail == "auto" else detail == "high"
    tiling = cfg.tiling()
    result = []
    for pixels_i in pixels:
        a = np.asarray(pixels_i)
        if (
            a.dtype != np.uint8
            or a.ndim != 3
            or a.shape[-1] != 3
            or min(a.shape[:2]) < 1
        ):
            raise ValueError("pixels must be nonempty uint8 HWC RGB")
        h, w = a.shape[:2]
        if h * w > cfg.max_pixels:
            raise ValueError("image exceeds max_pixels")
        grid = (0, 0)
        if cfg.kind == "idefics3":
            target = idefics3_resize_size(
                (h, w), tiling.longest_edge, cfg.tile_size, split
            )
            # Include the global tile in the allocation bound, before resizing.
            output_area = target[0] * target[1] + cfg.tile_size**2
            if (
                output_area > cfg.max_pixels
                or min(4096, tiling.longest_edge) ** 2 > cfg.max_pixels
            ):
                raise ValueError("Idefics3 tiles exceed max_pixels")
            tiles, base, grid = idefics3_split(
                a, tiling.longest_edge, cfg.tile_size, split, kernel=cfg.resample
            )
            frames = tiles + [base]
        elif cfg.kind == "llava_next":
            bh, bw = (
                (cfg.size["height"], cfg.size["width"])
                if "height" in cfg.size
                else (cfg.size["shortest_edge"],) * 2
            )
            th, tw = (
                select_best_resolution((h, w), cfg.image_grid_pinpoints)
                if split
                else (0, 0)
            )
            if th * tw + bh * bw > cfg.max_pixels:
                raise ValueError("AnyRes tiles exceed max_pixels")
            tiles = []
            if split:
                tiles, _, grid = anyres_tiles(
                    a, cfg.image_grid_pinpoints, cfg.tile_size, kernel=cfg.resample
                )
            base = resize(a, (bh, bw), cfg.resample)
            frames = [_single(frame, cfg) for frame in [base] + tiles]
        else:
            frames = [_single(a, cfg)]
        values = np.stack(
            [
                rescale_normalize(
                    frame,
                    cfg.rescale_factor if cfg.do_rescale else 1.0,
                    cfg.image_mean if cfg.do_normalize else (0.0,) * 3,
                    cfg.image_std if cfg.do_normalize else (1.0,) * 3,
                )
                for frame in frames
            ]
        )
        # Pixel-only fixtures have no encoded file. Frame the canonical bytes
        # with dimensions so reshapes of the same flat bytes cannot alias.
        identity = b"RGB8" + struct.pack("<II", h, w) + a.tobytes(order="C")
        result.append(
            ImageInputs(
                values,
                np.ones((len(frames), *values.shape[-2:]), dtype=np.bool_),
                grid,
                image_token_count(h, w, tiling, detail),
                media_hash("image", identity),
                (h, w),
            )
        )
    return result
    # SOLUTION-END


def preprocess(
    images: Sequence[bytes],
    cfg: ProcessorConfig,
    detail: Literal["low", "high", "auto"] = "auto",
) -> list[ImageInputs]:
    # SOLUTION-BEGIN L13.6
    cfg.validate()
    if detail not in ("low", "high", "auto"):
        raise ValueError("detail must be low, high, or auto")
    result = []
    for data in images:
        item = preprocess_pixels([decode_image(data, cfg.max_pixels)], cfg, detail)[0]
        item.media_hash = media_hash("image", data)
        result.append(item)
    return result
    # SOLUTION-END
