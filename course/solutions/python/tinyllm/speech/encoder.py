"""tinyllm.speech.encoder (L14.2): the Whisper encoder.

Whisper's encoder turns a 30 s log-mel picture (80 bands by 3000 frames)
into 1500 vectors of width d. A small convolutional stem does the
listening at frame level: a width-3 Conv1d mixes neighbouring frames and
lifts 80 bands to d channels, and a second width-3 Conv1d with stride 2
halves the time axis, so each output position covers 20 ms. Fixed
sinusoids (sines in the first half of each vector, cosines in the second)
say where each position is. Then a stack of pre-LN transformer blocks with
full, unmasked self-attention: the encoder sees the whole window at once.
Every piece is a layer the course already built (L13.1's Conv1d, L5.3's
attention, L0.4's LayerNorm and Linear); Whisper's only quirk is that its
key projection has no bias.

Contract: contracts/py/tinyllm/speech/encoder.pyi.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import LayerNorm, Linear, ModuleList
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.sig.conv import conv_out_size
from tinyllm.vision.conv import Conv1d
from tinyllm.xfmr.mha import MultiHeadAttention


@dataclass
class WhisperConfig:
    n_mels: int = 80
    n_audio_ctx: int = 1500
    d: int = 384
    n_heads: int = 6
    enc_layers: int = 4
    dec_layers: int = 4
    n_text_ctx: int = 448
    vocab: int = 51865
    d_ff: int = 1536

    @classmethod
    def from_hf(cls, config_json: str) -> "WhisperConfig":
        p = Path(config_json)
        if p.is_dir():
            p = p / "config.json"
        c = json.loads(p.read_text())
        if c.get("model_type", "whisper") != "whisper":
            raise ValueError(f"WhisperConfig.from_hf: model_type {c['model_type']!r}, need 'whisper'")
        heads = c.get("encoder_attention_heads", cls.n_heads)
        ffn = c.get("encoder_ffn_dim", cls.d_ff)
        if c.get("decoder_attention_heads", heads) != heads:
            raise ValueError("WhisperConfig.from_hf: decoder heads differ from encoder heads")
        if c.get("decoder_ffn_dim", ffn) != ffn:
            raise ValueError("WhisperConfig.from_hf: decoder ffn size differs from the encoder's")
        if c.get("activation_function", "gelu") != "gelu":
            raise ValueError(
                f"WhisperConfig.from_hf: activation {c['activation_function']!r}, need 'gelu'"
            )
        if c.get("scale_embedding", False):
            raise ValueError("WhisperConfig.from_hf: scale_embedding is not supported")
        cfg = cls(
            n_mels=int(c.get("num_mel_bins", cls.n_mels)),
            n_audio_ctx=int(c.get("max_source_positions", cls.n_audio_ctx)),
            d=int(c.get("d_model", cls.d)),
            n_heads=int(heads),
            enc_layers=int(c.get("encoder_layers", cls.enc_layers)),
            dec_layers=int(c.get("decoder_layers", cls.dec_layers)),
            n_text_ctx=int(c.get("max_target_positions", cls.n_text_ctx)),
            vocab=int(c.get("vocab_size", cls.vocab)),
            d_ff=int(ffn),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        sizes = (
            self.n_mels,
            self.n_audio_ctx,
            self.d,
            self.n_heads,
            self.enc_layers,
            self.dec_layers,
            self.n_text_ctx,
            self.vocab,
            self.d_ff,
        )
        if min(sizes) < 1:
            raise ValueError(f"WhisperConfig: every size must be >= 1, got {self}")
        if self.d % self.n_heads:
            raise ValueError(f"WhisperConfig: d {self.d} not divisible by n_heads {self.n_heads}")
        if self.d % 2:
            raise ValueError(f"WhisperConfig: d {self.d} must be even (sin and cos halves)")


def whisper_sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> NDArray:
    if channels % 2 or channels < 4:
        raise ValueError(f"whisper_sinusoids: channels must be even and >= 4, got {channels}")
    if length < 1:
        raise ValueError(f"whisper_sinusoids: length must be >= 1, got {length}")
    half = channels // 2
    # The angles in float32, as Whisper computes them; sin and cos of those
    # angles in float64, rounded once. Libraries' float32 exp can differ by
    # one ulp, which at angle 1499 moves a table entry by about 3e-5, so
    # L14.3's loader uses the table a checkpoint stores after checking it
    # against this one.
    inc = np.float32(math.log(max_timescale) / (half - 1))
    inv = np.exp(-inc * np.arange(half, dtype=np.float32))
    t = (np.arange(length, dtype=np.float32)[:, None] * inv[None, :]).astype(np.float64)
    return np.concatenate([np.sin(t), np.cos(t)], axis=1).astype(np.float32)


class WhisperAttention(MultiHeadAttention):
    def __init__(self, d: int, n_heads: int, rng: Any = None) -> None:
        r = rng if rng is not None else PCG32(0).substream("init")
        super().__init__(d, n_heads, bias=True, rng=r)
        # Re-assigning keeps k_proj's place in the registration order.
        self.k_proj = Linear(d, d, bias=False, rng=r)


class WhisperEncoderLayer(Module):
    def __init__(self, cfg: WhisperConfig, rng: Any = None) -> None:
        super().__init__()
        r = rng if rng is not None else PCG32(0).substream("init")
        self.self_attn = WhisperAttention(cfg.d, cfg.n_heads, rng=r)
        self.self_attn_layer_norm = LayerNorm(cfg.d)
        self.fc1 = Linear(cfg.d, cfg.d_ff, rng=r)
        self.fc2 = Linear(cfg.d_ff, cfg.d, rng=r)
        self.final_layer_norm = LayerNorm(cfg.d)

    def forward(self, x: Tensor) -> Tensor:
        h = self.self_attn_layer_norm(x)
        x = x + self.self_attn(h, h)
        h = self.final_layer_norm(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


class WhisperEncoder(Module):
    def __init__(self, cfg: WhisperConfig, rng: Any = None) -> None:
        super().__init__()
        cfg.validate()
        r = rng if rng is not None else PCG32(0).substream("init")
        self.cfg = cfg
        self.conv1 = Conv1d(cfg.n_mels, cfg.d, 3, pad=1, rng=r)
        self.conv2 = Conv1d(cfg.d, cfg.d, 3, stride=2, pad=1, rng=r)
        self.layers = ModuleList([WhisperEncoderLayer(cfg, r) for _ in range(cfg.enc_layers)])
        self.layer_norm = LayerNorm(cfg.d)
        self.positions = whisper_sinusoids(cfg.n_audio_ctx, cfg.d)

    def forward(self, mel: Tensor) -> Tensor:
        x = mel if isinstance(mel, Tensor) else Tensor(np.asarray(mel, dtype=np.float32))
        cfg = self.cfg
        if x.ndim != 3 or x.shape[1] != cfg.n_mels:
            raise ValueError(f"WhisperEncoder: need mel [N, {cfg.n_mels}, frames], got {x.shape}")
        frames = x.shape[2]
        a = conv_out_size(conv_out_size(frames, 3, 1, 1), 3, 2, 1)
        if frames != 2 * cfg.n_audio_ctx:
            raise ValueError(
                f"WhisperEncoder: {frames} mel frames give {a} positions; the window must be "
                f"{2 * cfg.n_audio_ctx} frames ({cfg.n_audio_ctx} positions)"
            )
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))  # [N, d, A]
        x = F.transpose(x, 1, 2) + self.positions  # [N, A, d]
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)
