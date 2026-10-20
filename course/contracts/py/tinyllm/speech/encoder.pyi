# contracts/py/tinyllm/speech/encoder.pyi (L14.2): the Whisper encoder
# chapter: content/courses/ai-engineering/15-speech/02-whisper-encoder.md
#
# Hugging Face WhisperEncoder semantics (Radford et al. 2022, section 2.2;
# transformers/models/whisper/modeling_whisper.py), float32:
#
#     x = gelu(conv1(mel))            Conv1d(n_mels, d, k=3, pad=1)    [N, d, 2 A]
#     x = gelu(conv2(x))              Conv1d(d, d, k=3, stride=2, pad=1) [N, d, A]
#     x = x^T + whisper_sinusoids(A, d)                                  [N, A, d]
#     for each layer: x = x + attn(ln1(x)); x = x + fc2(gelu(fc1(ln2(x))))
#     out = layer_norm(x)
#
# with A = n_audio_ctx (1500 for every released model: 3000 mel frames,
# 30 s), gelu the exact erf form, LayerNorm eps 1e-5, and pre-LN blocks with
# no attention mask (every frame sees every frame). Attention is L5.3's
# multi-head attention with Whisper's one quirk: k_proj has no bias.
#
# Parameter names are HF's without the "model.encoder." prefix: conv1,
# conv2, layers.{i}.self_attn.{q_proj,k_proj,v_proj,out_proj},
# layers.{i}.self_attn_layer_norm, layers.{i}.fc1, layers.{i}.fc2,
# layers.{i}.final_layer_norm, layer_norm. The sinusoid table is a constant,
# not a parameter (HF stores a copy as embed_positions.weight; L14.3's loader
# drops it). rng: PCG32 (M06.3), None means PCG32(0).substream("init").
from dataclasses import dataclass
from typing import Any

from numpy.typing import NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import LayerNorm, Linear, ModuleList
from tinyllm.nn.module import Module
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
    d_ff: int = 1536  # whisper-tiny's sizes are the defaults

    @classmethod
    def from_hf(cls, config_json: str) -> "WhisperConfig":
        """From a config.json path (or a directory holding one):
        num_mel_bins, max_source_positions, d_model, encoder_attention_heads,
        encoder_layers, decoder_layers, max_target_positions, vocab_size,
        encoder_ffn_dim. ValueError when model_type is present and is not
        "whisper", when the decoder's heads or ffn size differ from the
        encoder's, when activation_function is not "gelu", or when
        scale_embedding is true."""

    def validate(self) -> None:
        """ValueError for a size below 1 or d not divisible by n_heads, or an
        odd d (the sinusoids need sin and cos halves)."""

def whisper_sinusoids(length: int, channels: int, max_timescale: float = 10000.0) -> NDArray:
    """float32 [length, channels], computed in float32 as Whisper does:
    inc = ln(max_timescale) / (channels / 2 - 1), inv[i] = exp(-inc i) for
    i < channels / 2, t[p, i] = p inv[i], and the table is concat(sin(t),
    cos(t)) along the last axis: sines in the first half, cosines in the
    second, NOT interleaved as L5.4's sinusoidal_pe. Agrees with the tables
    HF checkpoints store within 1e-4 (float32 exp differs by an ulp between
    libraries, and angles reach 1499 rad). ValueError for an odd channels or
    channels < 4, or length < 1."""

class WhisperAttention(MultiHeadAttention):
    def __init__(self, d: int, n_heads: int, rng: Any = None) -> None:
        """L5.3's attention with q, v, out biases and k_proj = Linear(d, d,
        bias=False)."""

class WhisperEncoderLayer(Module):
    self_attn: WhisperAttention
    self_attn_layer_norm: LayerNorm
    fc1: Linear  # d -> d_ff
    fc2: Linear  # d_ff -> d
    final_layer_norm: LayerNorm

    def __init__(self, cfg: WhisperConfig, rng: Any = None) -> None: ...
    def forward(self, x: Tensor) -> Tensor:
        """The pre-LN block above, [N, A, d] -> [N, A, d]."""

class WhisperEncoder(Module):
    conv1: Conv1d
    conv2: Conv1d
    layers: ModuleList
    layer_norm: LayerNorm
    positions: NDArray  # whisper_sinusoids(n_audio_ctx, d): a constant

    def __init__(self, cfg: WhisperConfig, rng: Any = None) -> None:
        """cfg.validate() first."""
    def forward(self, mel: Tensor) -> Tensor:
        """[N, n_mels, 2 A] (a Tensor or an array) -> [N, A, d]. The output
        length is M12.1's conv_out_size applied for each conv. ValueError
        for mel not 3-D, n_mels channels other than cfg.n_mels, or a length
        other than 2 A (HF's check: a window is always 30 s)."""
