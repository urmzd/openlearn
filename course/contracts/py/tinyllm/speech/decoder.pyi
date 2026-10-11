# contracts/py/tinyllm/speech/decoder.pyi (L14.3): the Whisper decoder and model
# chapter: ml/08-tinyllm/p14-speech/03-whisper-decoder-and-weights.md
#
# Hugging Face WhisperDecoder semantics, float32. For token ids at absolute
# positions o .. o + T - 1 (o = the cache's committed length, 0 without a
# cache):
#
#     x = embed_tokens(ids) + embed_positions[o : o + T]      learned positions
#     for each layer:
#         x = x + self_attn(ln1(x))         causal: query t sees keys <= o + t
#         x = x + encoder_attn(ln2(x), audio)                 cross-attention
#         x = x + fc2(gelu(fc1(ln3(x))))
#     logits = layer_norm(x) @ embed_tokens^T                 tied output
#
# Attention is L14.2's WhisperAttention (no k_proj bias). Cross-attention
# keys and values depend only on the encoder output, so cross_kv computes
# them ONCE per audio window for every layer; each decoding step reuses
# them. Self-attention keys and values go through an L8.2 KVCache when one
# is passed: the step's new keys and values are appended and the queries
# attend to everything cached (the arrays the cache returns are constants,
# so a cached forward is for inference only).
#
# Parameter names are HF's without "model.decoder.": embed_tokens,
# embed_positions (L5.4's LearnedPE, weight [n_text_ctx, d]),
# layers.{i}.{self_attn, self_attn_layer_norm, encoder_attn,
# encoder_attn_layer_norm, fc1, fc2, final_layer_norm}, layer_norm. Whisper
# holds `encoder` and `decoder`, so its names are HF's without "model.".
from dataclasses import dataclass
from typing import Any, Optional

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.infer.kvcache import KVCache
from tinyllm.nn.layers import Embedding, LayerNorm, Linear, ModuleList
from tinyllm.nn.module import Module
from tinyllm.speech.encoder import WhisperAttention, WhisperConfig, WhisperEncoder
from tinyllm.xfmr.pos import LearnedPE

@dataclass
class CrossKV:
    k: list[NDArray]  # per decoder layer, float32 [N, H, A, d_head]
    v: list[NDArray]

class WhisperDecoderLayer(Module):
    self_attn: WhisperAttention
    self_attn_layer_norm: LayerNorm
    encoder_attn: WhisperAttention
    encoder_attn_layer_norm: LayerNorm
    fc1: Linear
    fc2: Linear
    final_layer_norm: LayerNorm

    def __init__(self, cfg: WhisperConfig, rng: Any = None) -> None: ...
    def forward(self, x: Tensor, xk: NDArray, xv: NDArray, mask: NDArray, cache: Optional[KVCache] = None, layer: int = 0) -> Tensor:
        """One block as above; xk, xv are this layer's cross keys and values,
        mask the bool self-attention mask [T, o + T]. With a cache, the new
        keys and values are appended to its `layer`."""

class WhisperDecoder(Module):
    embed_tokens: Embedding
    embed_positions: LearnedPE
    layers: ModuleList
    layer_norm: LayerNorm

    def __init__(self, cfg: WhisperConfig, rng: Any = None) -> None:
        """cfg.validate() first."""
    def cross_kv(self, audio: Tensor) -> CrossKV:
        """Every layer's encoder_attn k_proj and v_proj of the encoder output
        audio [N, A, d] (a Tensor or an array), split into heads, as float32
        arrays. ValueError for a shape other than [N, A, d]."""
    def forward(self, ids: ArrayLike, xkv: CrossKV, cache: Optional[KVCache] = None) -> Tensor:
        """Logits [N, T, vocab] for int ids [N, T] ([T, vocab] for 1-D ids).
        With a cache, positions start at cache.seq_len() and the mask is
        cache.mask(T), both read before any layer appends. ValueError for
        non-integer ids or ids outside [0, vocab), a batch size other than
        xkv's, a layer count other than xkv's, or positions past
        n_text_ctx."""

class Whisper(Module):
    config: WhisperConfig
    encoder: WhisperEncoder
    decoder: WhisperDecoder

    def __init__(self, cfg: WhisperConfig, rng: Any = None) -> None: ...
    def encode(self, mel: Tensor) -> CrossKV:
        """decoder.cross_kv(encoder(mel)): run once per 30 s window."""
    def forward(self, mel: Tensor, ids: ArrayLike) -> Tensor:
        """decoder(ids, encode(mel)): logits [N, T, vocab]."""
    def new_cache(self, batch: int = 1) -> KVCache:
        """An L8.2 KVCache for the decoder: dec_layers layers, n_heads heads
        of d / n_heads, max_len n_text_ctx."""
    def param_count(self) -> int:
        """Parameters plus the n_audio_ctx * d sinusoid table, which HF
        stores as a (frozen) parameter: equals HF's num_parameters()."""
    @classmethod
    def from_pretrained(cls, dir: str) -> "Whisper":
        """config.json and model.safetensors of an HF Whisper checkpoint
        (io/hf_whisper.py), loaded strictly; the encoder uses the stored
        sinusoid table when the checkpoint has one."""
