"""tinyllm.vision.vit (L13.3): the vision transformer encoder.

An image becomes a sequence: cut it into p x p patches, map each patch to a
d-vector with one linear map (a stride-p Conv2d, L13.1), add where each patch
sits, and run the same pre-LN transformer blocks a language model uses. The
three position schemes of current vision towers are here: a learned table
(ViT, CLIP, SigLIP, resized with bicubic interpolation for a new grid, or
bucketed for any aspect ratio as Idefics3 does), a fixed 2D sin-cos table,
and 2D RoPE inside attention (L13.2). L13.5's dual encoder and L13.7's
vision-language models use this tower, and L13.7 taps its hidden states.

Contract: contracts/py/tinyllm/vision/vit.pyi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.modern.rope import apply_rope
from tinyllm.nn.init import normal_init
from tinyllm.nn.layers import Embedding, LayerNorm, Linear, ModuleList
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.sig.resample import resize_planar_f32
from tinyllm.vision.conv import Conv2d
from tinyllm.vision.rope2d import rope2d_cos_sin
from tinyllm.xfmr.mha import MultiHeadAttention
from tinyllm.xfmr.sdpa import scaled_dot_product_attention

HF_KINDS = {
    "vit": dict(cls_token=True, pre_norm=False, post_layernorm=True, patch_bias=True, act="gelu", ln_eps=1e-12),
    "clip_vision_model": dict(cls_token=True, pre_norm=True, post_layernorm=False, patch_bias=False, act="quick_gelu", ln_eps=1e-5),
    "siglip_vision_model": dict(cls_token=False, pre_norm=False, post_layernorm=True, patch_bias=True, act="gelu_tanh", ln_eps=1e-6),
    "idefics3_vision": dict(cls_token=False, pre_norm=False, post_layernorm=True, patch_bias=True, act="gelu_tanh", ln_eps=1e-6, bucketed=True),
    "smolvlm_vision": dict(cls_token=False, pre_norm=False, post_layernorm=True, patch_bias=True, act="gelu_tanh", ln_eps=1e-6, bucketed=True),
}
HF_ACTS = {"gelu": "gelu", "gelu_pytorch_tanh": "gelu_tanh", "gelu_new": "gelu_tanh", "quick_gelu": "quick_gelu"}


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
        kind = config.get("model_type", "")
        if kind not in HF_KINDS:
            raise ValueError(f"ViTConfig.from_hf: unsupported model_type {kind!r} (one of {sorted(HF_KINDS)})")
        need = ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "image_size", "patch_size")
        missing = [k for k in need if k not in config]
        if missing:
            raise ValueError(f"ViTConfig.from_hf: config lacks {missing}")
        kw = dict(HF_KINDS[kind])
        if "layer_norm_eps" in config:
            kw["ln_eps"] = float(config["layer_norm_eps"])
        if "hidden_act" in config:
            if config["hidden_act"] not in HF_ACTS:
                raise ValueError(f"ViTConfig.from_hf: hidden_act {config['hidden_act']!r} is not supported")
            kw["act"] = HF_ACTS[config["hidden_act"]]
        return cls(
            image_size=int(config["image_size"]),
            patch_size=int(config["patch_size"]),
            n_channels=int(config.get("num_channels", 3)),
            d=int(config["hidden_size"]),
            n_layers=int(config["num_hidden_layers"]),
            n_heads=int(config["num_attention_heads"]),
            d_ff=int(config["intermediate_size"]),
            pos="learned",
            **kw,
        )


def _act(x: Tensor, kind: str) -> Tensor:
    if kind == "gelu":
        return F.gelu(x)
    if kind == "gelu_tanh":
        return F.gelu(x, approximate="tanh")
    if kind == "quick_gelu":
        return x * F.sigmoid(x * 1.702)
    raise ValueError(f"unknown activation {kind!r}")


def sincos2d_table(grid_hw: tuple[int, int], d: int) -> NDArray:
    """Fixed 2D sin-cos positions, float32 [Hp*Wp, d] (rows then columns)."""
    gh, gw = grid_hw
    q = d // 4
    omega = 1.0 / 10000.0 ** (np.arange(q) / q)
    r, c = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    ar = r.reshape(-1, 1) * omega
    ac = c.reshape(-1, 1) * omega
    return np.concatenate([np.sin(ar), np.cos(ar), np.sin(ac), np.cos(ac)], axis=1).astype(np.float32)


def interpolate_pos_embed(table: NDArray, old_grid: tuple[int, int], new_grid: tuple[int, int]) -> NDArray:
    t = np.asarray(table, dtype=np.float32)
    oh, ow = old_grid
    nh, nw = new_grid
    if t.ndim != 2 or t.shape[0] not in (oh * ow, oh * ow + 1):
        raise ValueError(f"interpolate_pos_embed: {t.shape[0]} rows fit neither {oh}x{ow} nor 1 + {oh}x{ow}")
    has_cls = t.shape[0] == oh * ow + 1
    cls_row, grid = (t[:1], t[1:]) if has_cls else (t[:0], t)
    d = t.shape[1]
    img = grid.reshape(oh, ow, d).transpose(2, 0, 1)  # [d, oh, ow]
    out = resize_planar_f32(img, (nh, nw), "bicubic")
    rows = out.transpose(1, 2, 0).reshape(nh * nw, d)
    return np.concatenate([cls_row, rows], axis=0).astype(np.float32)


def bucketed_position_ids(patch_mask: NDArray, n_side: int) -> NDArray:
    m = np.asarray(patch_mask, dtype=bool)
    single = m.ndim == 2
    if single:
        m = m[None]
    if m.ndim != 3:
        raise ValueError(f"bucketed_position_ids: need [Hp, Wp] or [N, Hp, Wp], got {m.shape}")
    n, hp, wp = m.shape
    # float32 throughout, as Idefics3 computes them; arange's values are
    # start + k step formed in double, then rounded once to float32.
    bounds = (np.arange(1, n_side) * (1.0 / n_side)).astype(np.float32)
    out = np.zeros((n, hp * wp), dtype=np.int64)
    for b in range(n):
        h = int(m[b, :, 0].sum())
        w = int(m[b, 0, :].sum())
        if h == 0 or w == 0:
            continue
        step_h = np.float32(1.0) / np.float32(h)
        step_w = np.float32(1.0) / np.float32(w)
        fh = np.minimum(np.arange(hp, dtype=np.float32) * step_h, np.float32(1.0 - 1e-6))
        fw = np.minimum(np.arange(wp, dtype=np.float32) * step_w, np.float32(1.0 - 1e-6))
        bh = np.searchsorted(bounds, fh, side="right")
        bw = np.searchsorted(bounds, fw, side="right")
        ids = (bh[:, None] * n_side + bw[None, :]).reshape(-1)
        keep = m[b].reshape(-1)
        out[b, keep] = ids[keep]
    return out[0] if single else out


class PatchEmbed(Conv2d):
    def __init__(self, cfg: ViTConfig, rng: Any = None) -> None:
        p = cfg.patch_size
        super().__init__(cfg.n_channels, cfg.d, p, stride=p, bias=cfg.patch_bias, rng=rng)


class VisionEmbeddings(Module):
    """Holds the embedding parameters under HF's names (embeddings.*)."""

    def __init__(self, cfg: ViTConfig, rng: Any) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbed(cfg, rng)
        if cfg.cls_token:
            self.class_embedding = Tensor(normal_init((cfg.d,), 0.02, rng), requires_grad=True)
        else:
            self.class_embedding = None
        if cfg.pos == "learned":
            n = (cfg.image_size // cfg.patch_size) ** 2 + (1 if cfg.cls_token else 0)
            self.position_embedding = Embedding(n, cfg.d, rng=rng)
            self.position_embedding.weight.data *= np.float32(0.02)  # normal_init std 0.02, same draws
        else:
            self.position_embedding = None


class VisionMLP(Module):
    def __init__(self, cfg: ViTConfig, rng: Any) -> None:
        super().__init__()
        self.act = cfg.act
        self.fc1 = Linear(cfg.d, cfg.d_ff, rng=rng)
        self.fc2 = Linear(cfg.d_ff, cfg.d, rng=rng)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(_act(self.fc1(x), self.act))


class VisionBlock(Module):
    def __init__(self, cfg: ViTConfig, rng: Any) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d, cfg.n_heads, rng=rng)
        self.layer_norm1 = LayerNorm(cfg.d, eps=cfg.ln_eps)
        self.mlp = VisionMLP(cfg, rng)
        self.layer_norm2 = LayerNorm(cfg.d, eps=cfg.ln_eps)

    def attend(self, x: Tensor, mask: Optional[NDArray], rope: Optional[tuple[NDArray, NDArray, int]]) -> Tensor:
        """Self-attention with an optional key mask and optional 2D RoPE on the
        patch positions (rope = (cos, sin, index of the first patch))."""
        a = self.self_attn
        q = a.split_heads(a.q_proj(x))
        k = a.split_heads(a.k_proj(x))
        v = a.split_heads(a.v_proj(x))
        if rope is not None:
            cos, sin, s = rope

            def rot(t: Tensor) -> Tensor:
                # A CLS token in front keeps its vector: rotate patches only.
                r = apply_rope(t[:, :, s:], cos, sin, "half")
                return F.concat([t[:, :, :s], r], axis=2) if s else r

            q, k = rot(q), rot(k)
        ctx, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        return a.out_proj(a.merge_heads(ctx))

    def forward(self, x: Tensor, mask: Optional[NDArray] = None, rope: Any = None) -> Tensor:
        h = x + self.attend(self.layer_norm1(x), mask, rope)
        return h + self.mlp(self.layer_norm2(h))


class VisionEncoder(Module):
    def __init__(self, cfg: ViTConfig, rng: Any) -> None:
        super().__init__()
        self.layers = ModuleList([VisionBlock(cfg, rng) for _ in range(cfg.n_layers)])


class ViT(Module):
    def __init__(self, cfg: ViTConfig, rng: Any = None) -> None:
        super().__init__()
        if cfg.d % cfg.n_heads:
            raise ValueError(f"ViT: n_heads {cfg.n_heads} does not divide d {cfg.d}")
        if cfg.image_size % cfg.patch_size:
            raise ValueError(f"ViT: patch_size {cfg.patch_size} does not divide image_size {cfg.image_size}")
        if cfg.pos not in ("learned", "sincos2d", "rope2d"):
            raise ValueError(f"ViT: unknown pos {cfg.pos!r}")
        if cfg.pos == "sincos2d" and cfg.d % 4:
            raise ValueError(f"ViT: sincos2d needs d % 4 == 0, got {cfg.d}")
        if cfg.pos == "rope2d" and (cfg.d // cfg.n_heads) % 4:
            raise ValueError(f"ViT: rope2d needs d_head % 4 == 0, got {cfg.d // cfg.n_heads}")
        r = rng if rng is not None else PCG32(0).substream("init")
        self.cfg = cfg
        self.embeddings = VisionEmbeddings(cfg, r)
        self.pre_layrnorm = LayerNorm(cfg.d, eps=cfg.ln_eps) if cfg.pre_norm else None
        self.encoder = VisionEncoder(cfg, r)
        self.post_layernorm = LayerNorm(cfg.d, eps=cfg.ln_eps) if cfg.post_layernorm else None

    def embed(self, pixel_values: Tensor, patch_mask: Optional[NDArray] = None) -> Tensor:
        cfg = self.cfg
        x = pixel_values
        if x.ndim != 4 or x.shape[1] != cfg.n_channels:
            raise ValueError(f"ViT: need pixel_values [N, {cfg.n_channels}, H, W], got {x.shape}")
        n, _, hh, ww = x.shape
        p = cfg.patch_size
        if hh % p or ww % p:
            raise ValueError(f"ViT: image {hh}x{ww} is not a multiple of the patch size {p}")
        gh, gw = hh // p, ww // p
        if patch_mask is not None and np.asarray(patch_mask).shape != (n, gh, gw):
            raise ValueError(f"ViT: patch_mask must be [{n}, {gh}, {gw}], got {np.asarray(patch_mask).shape}")
        e = self.embeddings
        pe = e.patch_embedding(x)  # [N, d, Hp, Wp]
        h = F.transpose(F.reshape(pe, (n, cfg.d, gh * gw)), 1, 2)  # [N, Hp*Wp, d], row-major
        n_side = cfg.image_size // p
        if cfg.pos == "learned":
            w = e.position_embedding.weight
            s = 1 if cfg.cls_token else 0
            if cfg.bucketed:
                mask = np.ones((n, gh, gw), bool) if patch_mask is None else np.asarray(patch_mask, bool)
                h = h + F.embedding(w[s:], bucketed_position_ids(mask, n_side))
            elif (gh, gw) == (n_side, n_side):
                h = h + w[s:]
            else:
                h = h + interpolate_pos_embed(w.data[s:], (n_side, n_side), (gh, gw))
        elif cfg.pos == "sincos2d":
            h = h + sincos2d_table((gh, gw), cfg.d)
        if cfg.cls_token:
            cls = F.reshape(e.class_embedding, (1, 1, cfg.d)) + np.zeros((n, 1, cfg.d), dtype=h.data.dtype)
            if cfg.pos == "learned":
                cls = cls + e.position_embedding.weight[0:1]
            h = F.concat([cls, h], axis=1)
        if self.pre_layrnorm is not None:
            h = self.pre_layrnorm(h)
        return h

    def encode(
        self,
        h: Tensor,
        grid_hw: tuple[int, int],
        patch_mask: Optional[NDArray] = None,
        output_hidden_states: bool = False,
    ) -> tuple[Tensor, list[Tensor]]:
        cfg = self.cfg
        n, t = h.shape[0], h.shape[1]
        s = 1 if cfg.cls_token else 0
        mask = None
        if patch_mask is not None:
            keep = np.asarray(patch_mask, bool).reshape(n, -1)
            if s:
                keep = np.concatenate([np.ones((n, 1), bool), keep], axis=1)
            if keep.shape[1] != t:
                raise ValueError(f"ViT: patch_mask covers {keep.shape[1]} tokens, h has {t}")
            # Keys only: a padded patch is invisible to every query.
            mask = keep[:, None, None, :]
        rope = None
        if cfg.pos == "rope2d":
            cos, sin = rope2d_cos_sin(grid_hw, cfg.d // cfg.n_heads, cfg.rope_base, cfg.rope_merge)
            rope = (cos, sin, s)
        hidden = [h] if output_hidden_states else []
        for layer in self.encoder.layers:
            h = layer(h, mask, rope)
            if output_hidden_states:
                hidden.append(h)
        if self.post_layernorm is not None:
            h = self.post_layernorm(h)
        return h, hidden

    def forward(
        self, pixel_values: Tensor, patch_mask: Optional[NDArray] = None, output_hidden_states: bool = False
    ) -> tuple[Tensor, list[Tensor]]:
        p = self.cfg.patch_size
        grid = (pixel_values.shape[2] // p, pixel_values.shape[3] // p)
        h = self.embed(pixel_values, patch_mask)
        return self.encode(h, grid, patch_mask, output_hidden_states)
