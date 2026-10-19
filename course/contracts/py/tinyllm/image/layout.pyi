# contracts/py/tinyllm/image/layout.pyi (M12.2): image tensors and memory layouts
# chapter: content/courses/mathematics-for-computing/12-signals-and-images/02-image-tensors-and-layouts.md
#
# Images arrive as uint8 [H, W, C] (HWC, RGB) from a decoder and leave as
# float32 [C, H, W] (CHW; batched NCHW) for a model. A layout names the
# order of the axes in memory; the logical shape stays (N, C, H, W).
# Inputs are never modified.
from collections.abc import Sequence
from typing import Literal

from numpy.typing import NDArray

Layout = Literal["NCHW", "NHWC"]

def to_layout(x: NDArray, src: Layout, dst: Layout) -> NDArray:
    """Reorder a 4-D batch from src axis order to dst axis order (NCHW
    [N, C, H, W] <-> NHWC [N, H, W, C]) and return a C-contiguous copy, even
    when src == dst, so the bytes really are in dst order. Same dtype.
    ValueError for another layout name or a non-4-D x."""

def strides_of(shape: tuple[int, ...], layout: Layout, itemsize: int) -> tuple[int, ...]:
    """Byte strides, one per LOGICAL axis of shape (N, C, H, W), or (C, H, W)
    for a single image, of a contiguous tensor stored in `layout` order:
    NCHW gives (C H W, H W, W, 1) * itemsize; NHWC (channels last) gives
    (H W C, 1, W C, C) * itemsize, the strides torch reports for
    memory_format=torch.channels_last. ValueError for another layout, a
    shape that is not 3-D or 4-D, a dimension < 1, or itemsize < 1."""

def rescale_normalize(img_u8_hwc: NDArray, scale: float, mean: Sequence[float], std: Sequence[float]) -> NDArray:
    """((img * scale) - mean[c]) / std[c] per channel, computed in float64,
    returned as contiguous float32 CHW [C, H, W] (one rounding at the end).
    CLIP: scale 1/255, mean (0.4815, 0.4578, 0.4082), std (0.2686, 0.2613,
    0.2758). ValueError unless img is 3-D, mean and std have C entries each,
    and every std is > 0."""

def to_grayscale_pil(img_u8_hwc: NDArray) -> NDArray:
    """Pillow's convert("L") bit for bit: ITU-R 601-2 luma rounded in 16-bit
    fixed point, L = (R*19595 + G*38470 + B*7471 + 32768) >> 16, uint8
    [H, W]. (The textbook rule in thousandths, (R*299 + G*587 + B*114 + 500)
    // 1000, differs from Pillow on about 0.05% of pixels.) ValueError unless
    img is uint8 [H, W, 3]."""

def patchify(x_chw: NDArray, p: int) -> NDArray:
    """Cut [C, H, W] into non-overlapping p x p patches and flatten each:
    [(H/p)*(W/p), C*p*p]. Patches are numbered row-major over the grid
    (patch (gi, gj) is row gi * (W/p) + gj) and each is flattened in
    (c, i, j) order, the order a KCRS conv weight flattens in, so a stride-p
    conv equals patchify then a matmul. ValueError unless p >= 1 divides H
    and W."""

def unpatchify(patches: NDArray, c: int, h: int, w: int, p: int) -> NDArray:
    """The inverse of patchify: unpatchify(patchify(x, p), C, H, W, p) == x.
    ValueError unless patches is [(h/p)*(w/p), c*p*p]."""
