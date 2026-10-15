# contracts/py/tinyllm/vision/rope2d.pyi (L13.2): 2D RoPE and M-RoPE
# chapter: content/courses/build-an-llm/p13-vision/02-rope-2d-and-m-rope.md
#
# Axial 2D RoPE (vision patches). A patch at grid row h, column w rotates
# the first half of its rotary pairs by h * inv and the second half by
# w * inv, with inv = base^(-2i / (r/2)) for i < r/4 (r = d_rot, a multiple
# of 4): one ladder of r/4 frequencies shared by both axes. The tables use
# L7.3's convention (one value per pair, "half" layout), so
# apply_rope(x, cos, sin, "half") rotates a [..., N, r] head (Qwen2-VL's
# vision tower). Scores then depend on (h_q - h_k, w_q - w_k) only.
#
# Patch order. merge = 1: row-major over the grid. merge = m > 1: m x m
# blocks in row-major order, row-major inside each block (Qwen2-VL orders
# its patches this way so that one block's m^2 patches are adjacent and
# PatchMerger can concatenate them).
#
# M-RoPE (Qwen2-VL). Every token has three position ids (t, h, w). Text
# token: t = h = w = its 1D position. Image token (one per m x m block of a
# t x H x W patch grid; m = spatial_merge): (s + ti, s + hi, s + wi) over the
# t x (H/m) x (W/m) grid in (t, h, w) row-major order, with s the position
# the image starts at; the next token starts at s + max(H/m, W/m) (for
# t = 1, the image's largest position + 1). The head's r/2 rotary pairs are
# split into consecutive sections of mrope_section[0], [1], [2], ... pairs
# (sum = r/2); section j uses axis j % 3 (t, h, w, t, ...). When t = h = w
# for every token this is exactly L7.3's 1D RoPE.
from collections.abc import Sequence

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor

def patch_coords(grid_hw: tuple[int, int], merge: int = 1) -> NDArray:
    """int64 [H*W, 2]: (row, column) of each patch in the patch order above.
    ValueError unless H, W >= 1 and merge >= 1 divides both."""

def rope2d_cos_sin(grid_hw: tuple[int, int], d_rot: int, base: float = 10000.0, merge: int = 1) -> tuple[NDArray, NDArray]:
    """cos and sin, float32 [H*W, d_rot/2], of the 2D angles above:
    columns 0 .. d_rot/4 - 1 from the row coordinate, the rest from the
    column coordinate, each computed as L7.3's rope_cos_sin. ValueError
    unless d_rot >= 4 is a multiple of 4 (and as patch_coords)."""

def mrope_position_ids(input_ids: ArrayLike, image_grid_thw: ArrayLike, image_token_id: int, spatial_merge: int = 2) -> NDArray:
    """int64 [3, T] (t, h, w) position ids of a 1-D id sequence, as above.
    image_grid_thw is [n_images, 3] (t, H, W in patches); the i-th run of
    t*H*W/m^2 image tokens belongs to image i (two images may be adjacent).
    ValueError when the image tokens do not split into runs of exactly those
    lengths, or m does not divide H and W."""

def mrope_cos_sin(pos3: ArrayLike, inv_freq: ArrayLike, mrope_section: Sequence[int]) -> tuple[NDArray, NDArray]:
    """cos and sin, float32 [T, r/2] with r/2 = len(inv_freq): pair i takes
    its angle from axis j % 3 of pos3 [3, T], where section j contains pair
    i. ValueError unless pos3 is [3, T] and sum(mrope_section) == r/2."""

def apply_mrope(x: Tensor, pos3: ArrayLike, inv_freq: ArrayLike, mrope_section: Sequence[int]) -> Tensor:
    """Rotate x [..., T, r] (typically [B, H, T, dh] with r = dh) with
    mrope_cos_sin in L7.3's "half" layout; differentiable in x. ValueError
    as mrope_cos_sin, or when x's last axis is not 2 len(inv_freq)."""
