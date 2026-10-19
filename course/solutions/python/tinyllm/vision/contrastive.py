"""tinyllm.vision.contrastive (L13.5): contrastive image-text learning.

Pair every image in a batch with its caption and every other caption is a
negative: CLIP turns that into two softmax classification problems (which
caption goes with this image, which image with this caption), SigLIP into
B^2 independent yes-or-no questions, which needs no batch-wide softmax and is
defined even for one pair. A dual encoder trained this way can classify
images it never saw labels for: embed "a photo of a {class}" for every class
and pick the closest. data.10 filters image-text pairs with this score, and
ethics.07 probes it for disparities.

Contract: contracts/py/tinyllm/vision/contrastive.pyi.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor, from_op
from tinyllm.nn.layers import Embedding, LayerNorm, Linear, ModuleList
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.vision.preprocess import ProcessorConfig, preprocess
from tinyllm.vision.vit import ViT, ViTConfig, VisionBlock

LN100 = math.log(100.0)


def _normalize(x: Tensor) -> Tensor:
    return x * (F.sum(x * x, axis=-1, keepdims=True) ** -0.5)


def _pairs(img_emb: Tensor, txt_emb: Tensor, who: str) -> Tensor:
    """The cosine matrix C[i, j] = <txt_i, img_j>, [B, B]."""
    if img_emb.ndim != 2 or txt_emb.ndim != 2 or img_emb.shape != txt_emb.shape:
        raise ValueError(f"{who}: need img_emb and txt_emb of one shape [B, d], got {img_emb.shape}, {txt_emb.shape}")
    return F.matmul(_normalize(txt_emb), F.transpose(_normalize(img_emb), 0, 1))


def _diag_ce(logits: Tensor) -> Tensor:
    """Mean cross-entropy of row i against class i."""
    b = logits.shape[0]
    lp = F.log_softmax(logits, axis=-1)
    return -F.mean(lp[np.arange(b), np.arange(b)])


def clip_loss(img_emb: Tensor, txt_emb: Tensor, logit_scale: Tensor) -> Tensor:
    c = _pairs(img_emb, txt_emb, "clip_loss")
    t = F.reshape(logit_scale, ())
    # Clamp the temperature at 100: past it the scale is a constant.
    scale = F.exp(t) if float(t.data) <= LN100 else Tensor(np.array(100.0), dtype=c.data.dtype)
    logits = c * scale
    return (_diag_ce(logits) + _diag_ce(F.transpose(logits, 0, 1))) * 0.5


def _logsigmoid(x: Tensor) -> Tensor:
    """log sigmoid(x) = min(x, 0) - log(1 + exp(-|x|)), stable for any x."""
    a = x.data
    y = np.minimum(a, 0) - np.log1p(np.exp(-np.abs(a)))
    # d/dx log sigmoid(x) = 1 - sigmoid(x) = sigmoid(-x)
    sneg = np.exp(np.minimum(-a, 0)) / (1.0 + np.exp(-np.abs(a)))
    return from_op(y.astype(a.dtype), (x,), lambda g: (g * sneg.astype(a.dtype),), "logsigmoid")


def siglip_loss(img_emb: Tensor, txt_emb: Tensor, t: Tensor, b: Tensor) -> Tensor:
    c = _pairs(img_emb, txt_emb, "siglip_loss")
    n = c.shape[0]
    logits = c * F.exp(F.reshape(t, ())) + F.reshape(b, ())
    z = 2.0 * np.eye(n) - 1.0  # +1 for the matching pair, -1 for every other
    return -F.sum(_logsigmoid(logits * z)) * (1.0 / n)


class TextEncoder(Module):
    def __init__(self, vocab: int, d: int, n_layers: int, n_heads: int, d_ff: int, max_len: int, rng: Any = None) -> None:
        super().__init__()
        r = rng if rng is not None else PCG32(0).substream("init")
        self.d, self.max_len = int(d), int(max_len)
        self.token_embedding = Embedding(vocab, d, rng=r)
        self.position_embedding = Embedding(max_len, d, rng=r)
        cfg = ViTConfig(image_size=1, patch_size=1, n_channels=1, d=d, n_layers=n_layers, n_heads=n_heads,
                        d_ff=d_ff, act="gelu", ln_eps=1e-5)
        self.layers = ModuleList([VisionBlock(cfg, r) for _ in range(n_layers)])
        self.final_layer_norm = LayerNorm(d, eps=1e-5)

    def forward(self, ids: NDArray) -> Tensor:
        a = np.asarray(ids)
        if a.ndim != 2 or a.shape[1] > self.max_len:
            raise ValueError(f"TextEncoder: need ids [N, T <= {self.max_len}], got {a.shape}")
        h = self.token_embedding(a) + self.position_embedding(np.arange(a.shape[1]))
        mask = (a != 0)[:, None, None, :]  # padding is never a key
        for layer in self.layers:
            h = layer(h, mask)
        return self.final_layer_norm(h)


class DualEncoder(Module):
    def __init__(
        self,
        vision: ViT,
        text: Module,
        d_embed: int,
        loss: Literal["clip", "siglip"],
        t_init: float,
        b_init: float = -10.0,
        rng: Any = None,
    ) -> None:
        super().__init__()
        if loss not in ("clip", "siglip"):
            raise ValueError(f"DualEncoder: loss must be 'clip' or 'siglip', got {loss!r}")
        r = rng if rng is not None else PCG32(0).substream("init")
        self.loss_kind = loss
        self.vision = vision
        self.text = text
        self.visual_projection = Linear(vision.cfg.d, d_embed, bias=False, rng=r)
        self.text_projection = Linear(text.d, d_embed, bias=False, rng=r)
        self.logit_scale = Tensor(np.array([t_init]), requires_grad=True)
        self.logit_bias = Tensor(np.array([b_init]), requires_grad=True) if loss == "siglip" else None

    def encode_image(self, pixel_values: Tensor) -> Tensor:
        last, _ = self.vision(pixel_values)
        pooled = last[:, 0] if self.vision.cfg.cls_token else F.mean(last, axis=1)
        return _normalize(self.visual_projection(pooled))

    def encode_text(self, ids: NDArray) -> Tensor:
        a = np.asarray(ids)
        keep = (a != 0).astype(np.float32)
        counts = keep.sum(axis=1, keepdims=True)
        if np.any(counts == 0):
            raise ValueError("encode_text: a row holds only padding")
        h = self.text(a)
        pooled = F.sum(h * keep[:, :, None], axis=1) * (1.0 / counts)
        return _normalize(self.text_projection(pooled))

    def loss(self, pixel_values: Tensor, ids: NDArray) -> Tensor:
        img, txt = self.encode_image(pixel_values), self.encode_text(ids)
        if self.loss_kind == "clip":
            return clip_loss(img, txt, self.logit_scale)
        return siglip_loss(img, txt, self.logit_scale, self.logit_bias)


    def encode_images(self, images: Sequence[bytes], cfg: ProcessorConfig) -> Tensor:
        """Encode a batch of image files with one frame per image."""
        if not images:
            raise ValueError("encode_images: need at least one image")
        cfg.validate()
        if cfg.kind != "single" or cfg.do_image_splitting:
            raise ValueError("encode_images: contrastive models require a single-frame processor")
        inputs = preprocess(images, cfg)
        size = self.vision.cfg.image_size
        expected = (1, self.vision.cfg.n_channels, size, size)
        if any(item.pixel_values.shape != expected for item in inputs):
            raise ValueError("encode_images: processor output must match the vision tower")
        pixels = np.concatenate([item.pixel_values for item in inputs], axis=0)
        return self.encode_image(Tensor(pixels))

def _encode_prompts(tok: Any, prompts: Sequence[str]) -> NDArray:
    enc = tok if callable(tok) and not hasattr(tok, "encode") else tok.encode
    rows = [list(enc(p)) for p in prompts]
    t = max(len(r) for r in rows)
    out = np.zeros((len(rows), t), dtype=np.int64)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def zero_shot_classifier(model: DualEncoder, tok: Any, class_names: Sequence[str], templates: Sequence[str]) -> NDArray:
    out = []
    for name in class_names:
        e = model.encode_text(_encode_prompts(tok, [t.format(name) for t in templates])).data
        m = e.mean(axis=0)
        out.append(m / np.linalg.norm(m))
    return np.stack(out)


def zero_shot_predict(model: DualEncoder, pixel_values: Tensor, W: NDArray) -> NDArray:
    scores = model.encode_image(pixel_values).data @ np.asarray(W).T
    return np.argmax(scores, axis=1).astype(np.int64)
