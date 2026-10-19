"""Whisper causal decoder, cached cross-attention, and complete model (L14.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from tinyllm.autograd import functional as F
from tinyllm.autograd.tensor import Tensor
from tinyllm.infer.kvcache import KVCache
from tinyllm.nn.layers import Embedding, LayerNorm, Linear, ModuleList
from tinyllm.nn.module import Module
from tinyllm.num.rng import PCG32
from tinyllm.speech.encoder import WhisperAttention, WhisperConfig, WhisperEncoder
from tinyllm.xfmr.masks import causal_mask
from tinyllm.xfmr.pos import LearnedPE
from tinyllm.xfmr.sdpa import scaled_dot_product_attention


@dataclass
class CrossKV:
    k: list[NDArray]
    v: list[NDArray]


class WhisperDecoderLayer(Module):
    def __init__(self, cfg: WhisperConfig, rng: Any = None) -> None:
        super().__init__()
        r = rng if rng is not None else PCG32(0).substream("init")
        self.self_attn = WhisperAttention(cfg.d, cfg.n_heads, r)
        self.self_attn_layer_norm = LayerNorm(cfg.d)
        self.encoder_attn = WhisperAttention(cfg.d, cfg.n_heads, r)
        self.encoder_attn_layer_norm = LayerNorm(cfg.d)
        self.fc1 = Linear(cfg.d, cfg.d_ff, rng=r)
        self.fc2 = Linear(cfg.d_ff, cfg.d, rng=r)
        self.final_layer_norm = LayerNorm(cfg.d)

    def forward(
        self,
        x: Tensor,
        xk: NDArray,
        xv: NDArray,
        mask: NDArray,
        cache: KVCache | None = None,
        layer: int = 0,
    ) -> Tensor:
        h = self.self_attn_layer_norm(x)
        a = self.self_attn
        q, k, v = [a.split_heads(p(h)) for p in (a.q_proj, a.k_proj, a.v_proj)]
        if cache is not None:
            keys, values = cache.update(layer, k.data, v.data)
            k, v = Tensor(keys), Tensor(values)
        context, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        x = x + a.out_proj(a.merge_heads(context))
        a = self.encoder_attn
        q = a.split_heads(a.q_proj(self.encoder_attn_layer_norm(x)))
        context, _ = scaled_dot_product_attention(q, Tensor(xk), Tensor(xv))
        x = x + a.out_proj(a.merge_heads(context))
        return x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))


class WhisperDecoder(Module):
    def __init__(self, cfg: WhisperConfig, rng: Any = None) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        r = rng if rng is not None else PCG32(0).substream("init")
        self.embed_tokens = Embedding(cfg.vocab, cfg.d, r)
        self.embed_positions = LearnedPE(cfg.n_text_ctx, cfg.d, rng=r)
        self.layers = ModuleList(
            [WhisperDecoderLayer(cfg, r) for _ in range(cfg.dec_layers)]
        )
        self.layer_norm = LayerNorm(cfg.d)

    def cross_kv(self, audio: Tensor) -> CrossKV:
        audio = (
            audio
            if isinstance(audio, Tensor)
            else Tensor(np.asarray(audio, dtype=np.float32))
        )
        if audio.ndim != 3 or audio.shape[-1] != self.cfg.d or audio.shape[1] < 1:
            raise ValueError(f"cross_kv needs [N, A, {self.cfg.d}], got {audio.shape}")
        keys, values = [], []
        for layer in self.layers:
            a = layer.encoder_attn
            keys.append(
                a.split_heads(a.k_proj(audio)).data.astype(np.float32, copy=True)
            )
            values.append(
                a.split_heads(a.v_proj(audio)).data.astype(np.float32, copy=True)
            )
        return CrossKV(keys, values)

    def forward(
        self, ids: ArrayLike, xkv: CrossKV, cache: KVCache | None = None
    ) -> Tensor:
        ids = np.asarray(ids)
        one = ids.ndim == 1
        if one:
            ids = ids[None, :]
        if (
            ids.ndim != 2
            or ids.shape[1] < 1
            or not np.issubdtype(ids.dtype, np.integer)
        ):
            raise ValueError(
                "decoder ids must be a nonempty integer [T] or [N, T] array"
            )
        if np.any(ids < 0) or np.any(ids >= self.cfg.vocab):
            raise ValueError("decoder token id outside the vocabulary")
        n, t = ids.shape
        cfg = self.cfg
        if len(xkv.k) != cfg.dec_layers or len(xkv.v) != cfg.dec_layers:
            raise ValueError("cross-KV layer count differs from the decoder")
        for k, v in zip(xkv.k, xkv.v):
            if (
                k.ndim != 4
                or k.shape != v.shape
                or k.shape[:2] != (n, cfg.n_heads)
                or k.shape[2] < 1
                or k.shape[3] != cfg.d // cfg.n_heads
            ):
                raise ValueError(
                    "cross-KV shape differs from the decoder batch or heads"
                )
        offset = 0 if cache is None else cache.seq_len()
        if offset + t > cfg.n_text_ctx:
            raise ValueError("decoder positions exceed n_text_ctx")
        if cache is not None:
            if (cache.n_layers, cache.batch, cache.n_kv_heads, cache.d_head) != (
                cfg.dec_layers,
                n,
                cfg.n_heads,
                cfg.d // cfg.n_heads,
            ):
                raise ValueError("self-KV cache shape differs from the decoder")
            if offset + t > cache.max_len or any(
                cache.seq_len(i) != offset for i in range(cfg.dec_layers)
            ):
                raise ValueError(
                    "self-KV cache is full or has inconsistent layer lengths"
                )
        mask = (
            causal_mask(t, offset + t, q_offset=offset)
            if cache is None
            else cache.mask(t)
        )
        x = self.embed_positions(self.embed_tokens(ids), offset)
        for i, layer in enumerate(self.layers):
            x = layer(x, xkv.k[i], xkv.v[i], mask, cache, i)
        logits = F.matmul(
            self.layer_norm(x), F.transpose(self.embed_tokens.weight, 0, 1)
        )
        return F.reshape(logits, (t, cfg.vocab)) if one else logits


class Whisper(Module):
    def __init__(self, cfg: WhisperConfig, rng: Any = None) -> None:
        super().__init__()
        cfg.validate()
        r = rng if rng is not None else PCG32(0).substream("init")
        self.config = cfg
        self.encoder = WhisperEncoder(cfg, r)
        self.decoder = WhisperDecoder(cfg, r)

    def encode(self, mel: Tensor) -> CrossKV:
        return self.decoder.cross_kv(self.encoder(mel))

    def encode_wav(self, data: bytes) -> CrossKV:
        """Decode one WAV and encode its first model-sized audio window."""
        from tinyllm.io.wav import read_wav
        from tinyllm.speech.frontend import (
            FeatureConfig, pad_or_trim, to_mono_16k, whisper_log_mel,
        )

        cfg = FeatureConfig(feature_size=self.config.n_mels)
        samples = to_mono_16k(read_wav(data), cfg)
        window = pad_or_trim(samples, 2 * self.config.n_audio_ctx * cfg.hop_length)
        mel = whisper_log_mel(window, cfg)
        return self.encode(Tensor(mel[None].astype(np.float32)))

    def forward(self, mel: Tensor, ids: ArrayLike) -> Tensor:
        return self.decoder(ids, self.encode(mel))

    def new_cache(self, batch: int = 1) -> KVCache:
        c = self.config
        return KVCache(c.dec_layers, c.n_heads, c.d // c.n_heads, c.n_text_ctx, batch)

    def param_count(self) -> int:
        return (
            sum(p.data.size for p in self.parameters())
            + self.config.n_audio_ctx * self.config.d
        )

    @classmethod
    def from_pretrained(cls, dir: str) -> Whisper:
        from tinyllm.io.hf_whisper import read_whisper

        cfg, state, positions = read_whisper(dir)
        model = cls(cfg)
        model.load_state_dict(state, strict=True)
        if positions is not None:
            model.encoder.positions = positions.copy()
        return model.eval()
