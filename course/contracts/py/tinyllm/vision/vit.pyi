# contracts/py/tinyllm/vision/vit.pyi (L13.3): the vision transformer encoder
# chapter: content/courses/build-an-llm/p13-vision/03-vision-transformer.md
#
# Dosovitskiy et al. (2021): cut the image into p x p patches, embed each
# with one linear map (a Conv2d with kernel = stride = p, L13.1), add
# positions, and run pre-LN transformer blocks over the patch sequence.
#
# Shapes: pixel_values float [N, C, H, W]; the patch grid is
# (Hp, Wp) = (H / p, W / p), patches in row-major order; with cls_token the
# sequence is T = 1 + Hp Wp (CLS first), else T = Hp Wp. Hidden states are
# [N, T, d].
#
# Parameter names follow Hugging Face's CLIP and SigLIP vision towers, so
# their checkpoints (and Idefics3's) load by name; registration order:
#   embeddings.patch_embedding.weight [d, C, p, p]  (.bias [d] if patch_bias)
#   embeddings.class_embedding [d]                  (cls_token)
#   embeddings.position_embedding.weight [P, d]     (pos 'learned'; P = (image_size/p)^2 (+1 with cls))
#   pre_layrnorm.weight, .bias                      (pre_norm; HF's spelling)
#   encoder.layers.<i>.self_attn.{q,k,v,out}_proj.{weight,bias}   (L5.3's MultiHeadAttention)
#   encoder.layers.<i>.layer_norm1.{weight,bias}, .mlp.fc1, .mlp.fc2, .layer_norm2
#   post_layernorm.weight, .bias                    (post_layernorm)
# Block i: h = h + attn(layer_norm1(h)); h = h + fc2(act(fc1(layer_norm2(h)))).
# Activations: 'gelu' (erf), 'gelu_tanh', 'quick_gelu' (x sigmoid(1.702 x)).
#
# Positions:
#   'learned'   a table of P rows. For a grid other than (image_size/p)^2 the
#               patch rows are resized with interpolate_pos_embed (a constant,
#               no gradient) unless `bucketed`; with `bucketed` (Idefics3,
#               NaViT) patch (i, j) of an image whose valid grid is h x w rows
#               table row bucketed_position_ids(mask, image_size/p)[i Wp + j].
#   'sincos2d'  fixed: the first d/2 entries encode the row r, the rest the
#               column c, each [sin(x w_0..w_{d/4-1}), cos(x w_0..)] with
#               w_k = 10000^(-k / (d/4)); added to the patches (CLS gets 0).
#   'rope2d'    no additive table: queries and keys of every block are
#               rotated with L13.2's rope2d_cos_sin(grid, d_head, rope_base,
#               rope_merge) (the CLS token, if any, is not rotated). With
#               rope_merge m > 1 the patch sequence is in L13.2's block-major
#               order (Qwen2-VL), which the caller of encode provides.
# patch_mask (bool [N, Hp, Wp], True = a real patch): masked patches are
# hidden as keys from every query, so they never change the outputs of real
# patches; their own outputs are unspecified.
from dataclasses import dataclass
from typing import Literal, Optional

from numpy.typing import NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import LayerNorm
from tinyllm.nn.module import Module
from tinyllm.vision.conv import Conv2d

@dataclass
class ViTConfig:
    image_size: int
    patch_size: int
    n_channels: int
    d: int
    n_layers: int
    n_heads: int
    d_ff: int
    pos: Literal["learned", "sincos2d", "rope2d"] = "learned"
    cls_token: bool = False
    pre_norm: bool = False
    norm: Literal["layernorm"] = "layernorm"
    act: Literal["gelu", "gelu_tanh", "quick_gelu"] = "gelu"
    ln_eps: float = 1e-6
    post_layernorm: bool = True
    patch_bias: bool = True
    bucketed: bool = False
    rope_base: float = 10000.0
    rope_merge: int = 1

    @classmethod
    def from_hf(cls, config: dict) -> "ViTConfig":
        """From a Hugging Face vision config dict: model_type 'vit'
        (cls_token, learned, gelu, post layernorm), 'clip_vision_model'
        (cls_token, no patch bias, pre_norm, no post layernorm on the hidden
        states, quick_gelu), 'siglip_vision_model' (no cls, gelu_tanh, post
        layernorm), or 'idefics3_vision' / 'smolvlm_vision' (as SigLIP, with
        bucketed positions). Keys hidden_size, num_hidden_layers,
        num_attention_heads, intermediate_size, image_size, patch_size,
        num_channels (default 3), layer_norm_eps, hidden_act override the
        defaults. ValueError for another model_type or a missing size."""

class PatchEmbed(Conv2d):
    """Conv2d(n_channels, d, k = p, stride = p, bias = patch_bias): one
    linear map per patch, so forward(x) equals M12.2's patchify(x, p) times
    weight.reshape(d, -1)^T plus bias, arranged [N, d, Hp, Wp]."""

    def __init__(self, cfg: ViTConfig, rng: object = None) -> None: ...

class VisionBlock(Module):
    """One pre-LN block with the parameter names above (self_attn,
    layer_norm1, mlp.fc1, mlp.fc2, layer_norm2). L13.5's text tower stacks
    these too."""

    def __init__(self, cfg: ViTConfig, rng: object) -> None: ...
    def forward(self, x: Tensor, mask: Optional[NDArray] = None, rope: object = None) -> Tensor:
        """x [N, T, d] -> [N, T, d]. mask: bool, True = may attend,
        broadcastable to [N, heads, T, T]. rope: None or (cos, sin, s) to
        rotate queries and keys from token s on (L7.3's 'half' layout)."""

class ViT(Module):
    cfg: ViTConfig
    pre_layrnorm: Optional[LayerNorm]
    post_layernorm: Optional[LayerNorm]

    def __init__(self, cfg: ViTConfig, rng: object = None) -> None:
        """Parameters as above, drawn from one rng (PCG32; None means
        PCG32(0).substream("init")); LayerNorms start at ones and zeros, the
        CLS vector and the position table from normal_init(std 0.02).
        ValueError when n_heads does not divide d, the patch size does not
        divide image_size, or (sincos2d, rope2d) d or d_head is not a
        multiple of 4."""
    def embed(self, pixel_values: Tensor, patch_mask: Optional[NDArray] = None) -> Tensor:
        """Patches, CLS, and additive positions (then pre_layrnorm): the
        input of block 0, [N, T, d]. ValueError when H or W is not a multiple
        of p, the channel count is wrong, or patch_mask is not [N, Hp, Wp]."""
    def encode(self, h: Tensor, grid_hw: tuple[int, int], patch_mask: Optional[NDArray] = None,
               output_hidden_states: bool = False) -> tuple[Tensor, list[Tensor]]:
        """Run the blocks over h [N, T, d] (the output of embed, or any
        embedded patch sequence with that grid), then post_layernorm. Returns
        (last [N, T, d], hidden) where hidden is [] or, with
        output_hidden_states, the L + 1 block inputs and outputs: hidden[0]
        is h, hidden[i] the output of block i - 1 (before post_layernorm).
        L13.7's vision_feature_layer indexes this list."""
    def forward(self, pixel_values: Tensor, patch_mask: Optional[NDArray] = None,
                output_hidden_states: bool = False) -> tuple[Tensor, list[Tensor]]:
        """encode(embed(pixel_values, patch_mask), (Hp, Wp), patch_mask, ...)."""

def interpolate_pos_embed(table: NDArray, old_grid: tuple[int, int], new_grid: tuple[int, int]) -> NDArray:
    """Resize a position table to a new patch grid, as Hugging Face's
    interpolate_pos_encoding: table is [oh*ow, d] or, with a CLS row first,
    [1 + oh*ow, d]; the patch rows, seen as a [d, oh, ow] image, go through
    M12.3's resize_planar_f32(.., new_grid, 'bicubic') (torch's
    F.interpolate, align_corners False, a = -0.75) and come back row-major;
    the CLS row is kept. float32 [(1 +) nh*nw, d]. ValueError when the row
    count matches neither layout."""

def bucketed_position_ids(patch_mask: NDArray, n_side: int) -> NDArray:
    """Idefics3 / NaViT positions for bool patch_mask [Hp, Wp] or
    [N, Hp, Wp] (True = real patch), int64 [..., Hp*Wp]: with h and w the
    numbers of real rows and columns, patch (i, j) gets
    bucket(i / h) * n_side + bucket(j / w), where i / h is computed in
    float32 and clamped to 1 - 1e-6, and bucket(x) counts the boundaries
    k / n_side (k = 1 .. n_side - 1, float32) that are <= x; masked patches
    get 0. A full n_side x n_side mask gives 0 .. n_side^2 - 1."""
