# contracts/py/tinyllm/vision/contrastive.pyi (L13.5): contrastive image-text learning
# chapter: content/courses/build-an-llm/p13-vision/05-contrastive-image-text.md
#
# A dual encoder embeds images and texts into one space where matching pairs
# are close. Both losses take a batch of B matching pairs (row i of img_emb
# goes with row i of txt_emb), L2-normalize every row, and form the cosine
# matrix C[i, j] = <txt_i, img_j> (texts on rows, as Hugging Face's
# logits_per_text).
#
# CLIP (Radford et al. 2021): logits = s C with s = exp(logit_scale),
# clamped at 100 (when exp(logit_scale) > 100, s = 100 and logit_scale gets
# no gradient); loss = (CE(logits, diag) + CE(logits^T, diag)) / 2, the mean
# cross-entropy of picking the matching image for each text and the matching
# text for each image. Hugging Face's CLIPModel loss without the clamp.
#
# SigLIP (Zhai et al. 2023): logits = exp(t) C + b; z[i, j] = +1 on the
# diagonal and -1 off it; loss = -(1/B) sum_i sum_j log sigmoid(z_ij
# logits_ij), every pair an independent binary decision (SiglipModel's loss;
# init t = log 10, b = -10).
#
# Every input is an L0.1 Tensor; the losses are scalar Tensors whose
# backward reaches the embeddings, logit_scale / t, and b.
from collections.abc import Sequence
from typing import Any, Literal

from numpy.typing import NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module
from tinyllm.vision.preprocess import ProcessorConfig
from tinyllm.vision.vit import ViT

def clip_loss(img_emb: Tensor, txt_emb: Tensor, logit_scale: Tensor) -> Tensor:
    """The CLIP loss above. img_emb and txt_emb are [B, d]; logit_scale is a
    one-element Tensor. ValueError for shapes that differ or are not 2-D."""

def siglip_loss(img_emb: Tensor, txt_emb: Tensor, t: Tensor, b: Tensor) -> Tensor:
    """The SigLIP loss above, defined for any B >= 1 (one positive pair and
    no negatives at B = 1). ValueError as clip_loss."""

class TextEncoder(Module):
    """A small text tower: token_embedding [vocab, d] and
    position_embedding [max_len, d] (L0.4 Embedding), n_layers of L13.3's
    VisionBlock (gelu, LayerNorm eps 1e-5, no mask between real tokens; pad
    id 0 is hidden as a key), then final_layer_norm. forward(ids [N, T]) ->
    [N, T, d]."""

    d: int

    def __init__(self, vocab: int, d: int, n_layers: int, n_heads: int, d_ff: int, max_len: int, rng: Any = None) -> None: ...
    def forward(self, ids: NDArray) -> Tensor: ...

class DualEncoder(Module):
    """Parameters: vision (the ViT), text (a Module with attribute d whose
    forward maps ids [N, T] to [N, T, d]), visual_projection
    Linear(vision.cfg.d, d_embed, bias False), text_projection
    Linear(text.d, d_embed, bias False), logit_scale [1] = t_init, and with
    loss 'siglip' logit_bias [1] = b_init. Image pooling: the CLS token when
    the ViT has one, else the mean over patches. Text pooling: the mean over
    positions whose id is not 0."""

    vision: ViT
    text: Module
    visual_projection: Linear
    text_projection: Linear
    logit_scale: Tensor

    def __init__(self, vision: ViT, text: Module, d_embed: int, loss: Literal["clip", "siglip"], t_init: float, b_init: float = -10.0, rng: Any = None) -> None:
        """ValueError for an unknown loss."""
    def encode_image(self, pixel_values: Tensor) -> Tensor:
        """[N, d_embed], every row L2-normalized."""
    def encode_images(self, images: Sequence[bytes], cfg: ProcessorConfig) -> Tensor:
        """L13.6 decode/preprocess then encode_image, preserving input order.
        Require a nonempty batch, kind single, no splitting, and each
        output [1, n_channels, image_size, image_size]. Invalid input raises
        ValueError; no tiling or silent averaging of image embeddings."""
    def encode_text(self, ids: NDArray) -> Tensor:
        """[N, d_embed], every row L2-normalized. ValueError for a row of
        only padding."""
    def loss(self, pixel_values: Tensor, ids: NDArray) -> Tensor:
        """clip_loss or siglip_loss of the two encodings with this model's
        logit_scale (and logit_bias)."""

def zero_shot_classifier(model: DualEncoder, tok: Any, class_names: Sequence[str], templates: Sequence[str]) -> NDArray:
    """float [n_classes, d_embed]: for each class, the text embeddings of
    template.format(name) for every template, averaged and L2-normalized
    again (Radford et al.'s prompt ensembling). tok is a callable or has
    .encode(text) -> list[int]; prompts are right-padded with 0."""

def zero_shot_predict(model: DualEncoder, pixel_values: Tensor, W: NDArray) -> NDArray:
    """int64 [N]: argmax over classes of encode_image(pixel_values) W^T
    (ties to the lowest class index)."""
