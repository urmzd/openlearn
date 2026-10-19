# contracts/py/tinyllm/vision/tiling.pyi (L13.4): dynamic resolution and tiling
# chapter: content/courses/build-an-llm/14-vision/04-dynamic-resolution-and-tiling.md
#
# A vision tower sees fixed tiles (SmolVLM 512 x 512, LLaVA-NeXT 336 x 336);
# a large or oddly shaped image is cut into several tiles plus one
# low-resolution view of the whole image, and the language model receives
# the features of all of them. Images here are uint8 HWC arrays; every resize
# is M12.3's resize(.., mode='pil'), so tiles equal Pillow's (and Hugging
# Face's PIL image processors') bit for bit.
#
# Idefics3 / SmolVLM ("idefics3"): resize so the longest edge is
# longest_edge (the other edge int(edge / aspect), rounded up to even; both
# capped at 4096), then grow each edge to a multiple of `tile` (width first
# when width >= height: W' = ceil(W / tile) tile, H' = ceil(int(W' / aspect)
# / tile) tile; mirrored otherwise), all with Lanczos. If H' or W' exceeds
# tile, crop rows = H' / tile by cols = W' / tile tiles in row-major order
# and append a global image (the resized image squeezed to tile x tile);
# otherwise the image alone, squeezed to tile x tile, with rows = cols = 0.
# Every tile becomes image_seq_len tokens after pixel shuffle.
# longest_edge=0 skips the initial resize (HF do_resize=False).
# The split functions accept a kernel override; defaults preserve the
# canonical recipes above.
#
# LLaVA-NeXT AnyRes ("llava_next"): choose the pinpoint resolution (H, W)
# with select_best_resolution, resize the image to fit inside it keeping the
# aspect ratio (bicubic; the fitted edge is the target, the other
# min(ceil(edge * scale), target)), pad it centered with zeros (the odd pixel
# goes bottom or right), and cut H/tile x W/tile tiles; the base image is
# the original resized to tile x tile (bicubic). The language model gets the
# base image's (tile/patch)^2 features, then the tiles' feature grid
# (H/tile * tile/patch rows by W/tile * tile/patch columns) with its padding
# rows or columns removed (unpad_shape) and one image_newline per row.
#
# "single": one fixed-size image, image_seq_len tokens (LLaVA-1.5, CLIP).
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from numpy.typing import NDArray
from tinyllm.sig.resample import Kernel

@dataclass
class TilingConfig:
    """What image_token_count needs from a processor (L13.6's
    ProcessorConfig.tiling() builds one)."""
    kind: Literal["idefics3", "llava_next", "single"]
    tile: int
    longest_edge: int = 0
    image_seq_len: int = 0
    pinpoints: tuple = ()
    patch_size: int = 14
    base_extra: int = 0
    split: bool = True

def select_best_resolution(hw: tuple[int, int], pinpoints: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """LLaVA-NeXT: for each candidate (H, W), scale = min(W / w, H / h), the
    effective resolution is min(int(w scale) int(h scale), w h) and the waste
    H W - effective; pick the largest effective resolution, ties broken by
    the smallest waste, then by the earlier candidate. ValueError for an
    empty list or a size < 1."""

def fit_size(hw: tuple[int, int], target: tuple[int, int]) -> tuple[int, int]:
    """The (h, w) an image of size hw is resized to inside target before
    AnyRes padding, as described above."""

def anyres_tiles(img: NDArray, pinpoints: Sequence[tuple[int, int]], tile: int, *, kernel: Kernel = "bicubic") -> tuple[list[NDArray], NDArray, tuple[int, int]]:
    """(tiles row-major, base image, (rows, cols)) for uint8 HWC img, each
    tile and the base uint8 [tile, tile, C]. ValueError unless every
    pinpoint is a multiple of tile."""

def unpad_shape(orig_hw: tuple[int, int], cur_hw: tuple[int, int]) -> tuple[int, int]:
    """The feature grid left after removing AnyRes padding from a cur_hw
    grid for an image of size orig_hw: if w / h > W / H, keep
    H - 2 ((H - int(round(h W / w, 7))) // 2) rows and every column; else
    mirrored on the columns."""

def idefics3_resize_size(hw: tuple[int, int], longest_edge: int, tile: int, split: bool = True) -> tuple[int, int]:
    """The (H', W') the image is resized to before cutting (above). Without
    split, (tile, tile)."""

def idefics3_split(img: NDArray, longest_edge: int, tile: int, split: bool = True, *, kernel: Kernel = "lanczos3") -> tuple[list[NDArray], NDArray, tuple[int, int]]:
    """(tiles row-major, global image, (rows, cols)) for uint8 HWC img, as
    above; with rows = cols = 0 the tile list is empty and the global image
    is the only one. Tiles and global are uint8 [tile, tile, C]."""

def idefics3_prompt(rows: int, cols: int, image_seq_len: int, image_token: str = "<image>",
                    fake_token: str = "<fake_token_around_image>", global_tag: str = "<global-img>") -> str:
    """The text that replaces one image in an Idefics3 / SmolVLM prompt.
    rows = cols = 0: fake + global_tag + image_token * n + fake. Otherwise,
    for each row r and column c: fake + '<row_{r+1}_col_{c+1}>' +
    image_token * n, a '\\n' after each row, then '\\n' + fake + global_tag +
    image_token * n + fake."""

def pixel_shuffle(x: NDArray, r: int) -> NDArray:
    """Idefics3's connector: [N, S*S, D] (a row-major S x S grid) ->
    [N, (S/r)^2, D r^2]; output token (i, j) concatenates the r x r block
    at rows i r .. i r + r - 1 and columns j r .. in Hugging Face's order
    (each output row of the block contributes r D channels, columns inner).
    ValueError unless S*S is a square and r divides S."""

def pixel_unshuffle(y: NDArray, r: int) -> NDArray:
    """The inverse: pixel_unshuffle(pixel_shuffle(x, r), r) == x."""

def image_token_count(h: int, w: int, cfg: TilingConfig, detail: Literal["low", "high", "auto"] = "auto") -> int:
    """How many image placeholder tokens (feature rows) an h x w image
    becomes. 'auto' follows cfg.split, 'high' splits, 'low' does not.
    idefics3: (rows cols + 1) image_seq_len when split produces tiles, else
    image_seq_len. llava_next: (tile/patch)^2 + base_extra, plus for
    'auto'/'high' the unpadded tile grid's cells and one newline per row of
    it; 'low' adds one newline. single: image_seq_len. Wrapper tokens (the
    fake token, row tags, the global tag, newlines in the text) are not
    counted: their token count depends on the tokenizer. ValueError for h or w < 1."""
