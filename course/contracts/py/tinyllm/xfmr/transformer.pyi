# contracts/py/tinyllm/xfmr/transformer.pyi (L5.5): the encoder-decoder Transformer
# chapter: content/courses/ai-engineering/11-transformer/05-encoder-decoder-transformer.md
#
# Vaswani et al. (2017), sections 3 and 5. With d = d_model:
#
#   embed(ids)  = dropout(table(ids) * sqrt(d) + PE)          SinusoidalPE (L5.4)
#   encoder layer, post-LN (norm="post", the paper):
#       x = norm1(x + drop(self_attn(x, x, src_keys)))
#       x = norm2(x + drop(ff(x)))
#   encoder layer, pre-LN (norm="pre", Xiong et al. 2020):
#       x = x + drop(self_attn(norm1(x), norm1(x), src_keys))
#       x = x + drop(ff(norm2(x)))
#   decoder layer: the same with three sublayers, masked self-attention
#       (causal AND target padding), cross-attention (queries from the decoder,
#       keys and values from the encoder output `memory`, source padding),
#       and ff; norm1, norm2, norm3 in that order.
#   ff(x) = linear2(drop(relu(linear1(x))))
#   pre-LN only: enc_norm after the last encoder layer, dec_norm after the last
#   decoder layer (post-LN has no final norm: its last sublayer already ends
#   in one).
#   logits = decoder output @ out.weight^T                     [B, T, tgt_vocab]
#
# Padding masks are bool [B, S] / [B, T], True = a real token; the model makes
# the attention masks from them (L5.2's causal_mask and combine; key padding
# as [B, 1, 1, S]). Every attention module is L5.3's MultiHeadAttention.
#
# Parameters, in registration order (= state_dict order), float32, drawn from
# one rng (PCG32; None means PCG32(0).substream("init")):
#   src_emb.weight [Vs, d] ~ N(0, 1/d)
#   tgt_emb.weight [Vt, d] ~ N(0, 1/d); with tie_embeddings and Vs == Vt it IS
#                  src_emb.weight (listed once, as src_emb.weight)
#   encoder.{i}.self_attn.*, .linear1.*, .linear2.*, .norm1.*, .norm2.*
#   decoder.{i}.self_attn.*, .cross_attn.*, .linear1.*, .linear2.*, .norm1.*, .norm2.*, .norm3.*
#   enc_norm.*, dec_norm.*            pre-LN only
#   out.weight [Vt, d] (no bias); with tie_embeddings it IS tgt_emb.weight
# torch's TransformerEncoderLayer / DecoderLayer use the same sublayer names
# (multihead_attn is cross_attn here; in_proj is split, L5.3).
#
# Training (section 5.3, 5.4): Adam (beta1 0.9, beta2 0.98, eps 1e-9), the Noam
# rate of M10.4 with the paper's step_num = update index + 1, and label
# smoothing through L0.3's cross_entropy (the eps mass spread over all V ids).
#
# The model directory (save_transformer, load_transformer) for the L6.7 zoo:
# config.json {"tl_arch": "transformer", "tl_tokenizer": "file", "vocab_size": Vt,
#   "tl_src_vocab": Vs, "tl_format": 1, "d_model", "n_heads", "d_ff", "n_enc",
#   "n_dec", "tl_norm": "post" | "pre", "tie_word_embeddings",
#   "max_position_embeddings", "layer_norm_eps"}
# and model.safetensors (F32, metadata {"format": "tinyllm", "tl_arch": "transformer"}).
from dataclasses import dataclass
from typing import Any, Callable, Literal, NamedTuple, Optional

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Embedding
from tinyllm.nn.module import Module
from tinyllm.optim.adamw import Adam

@dataclass
class TransformerConfig:
    src_vocab: int
    tgt_vocab: int
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_enc: int = 6
    n_dec: int = 6
    dropout: float = 0.1
    norm: Literal["post", "pre"] = "post"
    tie_embeddings: bool = True
    max_len: int = 256
    ln_eps: float = 1e-5

class DecodeState(NamedTuple):
    prefix: NDArray  # int64 [B, t]: the target tokens fed so far, bos first
    src_mask: NDArray  # bool [B, S]

class EncoderLayer(Module):
    def __init__(self, cfg: TransformerConfig, rng: Any) -> None: ...
    def ff(self, x: Tensor) -> Tensor: ...
    def forward(self, x: Tensor, mask: NDArray) -> Tensor:
        """One encoder layer above; mask broadcasts to [B, H, S, S]."""

class DecoderLayer(Module):
    def __init__(self, cfg: TransformerConfig, rng: Any) -> None: ...
    def ff(self, x: Tensor) -> Tensor: ...
    def forward(self, y: Tensor, memory: Tensor, self_mask: NDArray, cross_mask: NDArray) -> Tensor:
        """One decoder layer above."""

class Transformer(Module):
    # The L6.7 zoo loads it (tl_arch "transformer") and decodes with beam search.
    cfg: TransformerConfig

    def __init__(self, cfg: TransformerConfig, rng: Any = None) -> None:
        """ValueError for a norm other than "post" or "pre" or a size below 1
        (and as L5.3 for d_model and n_heads)."""

    def embed(self, ids: ArrayLike, table: Embedding, offset: int = 0) -> Tensor:
        """table(ids) * sqrt(d_model) + positions offset.., then dropout."""

    def encode(self, src: ArrayLike, src_mask: ArrayLike) -> Tensor:
        """memory [B, S, d_model]. ValueError for src not int [B, S] or a
        src_mask that is not bool [B, S]."""

    def decode(self, tgt_in: ArrayLike, memory: Tensor, src_mask: ArrayLike, tgt_mask: ArrayLike) -> Tensor:
        """logits [B, T, tgt_vocab] for every position of tgt_in int [B, T];
        position t reads targets 0..t that are real, and every real source
        position. ValueError for bad shapes or dtypes."""

    def forward(self, src: ArrayLike, tgt_in: ArrayLike, src_mask: ArrayLike, tgt_mask: ArrayLike) -> Tensor:
        """decode(tgt_in, encode(src, src_mask), src_mask, tgt_mask)."""

    def init_state(self, src_mask: ArrayLike) -> DecodeState:
        """An empty prefix [B, 0] for these sources."""

    def decode_step(self, y_prev: ArrayLike, memory: Tensor, state: DecodeState) -> tuple[Tensor, DecodeState]:
        """Append y_prev int [B] to the prefix and return (the logits of the
        LAST position [B, tgt_vocab], the new state). No cache: the whole
        prefix is decoded again (L8.2 adds a KV cache). ValueError when the
        batch of the prefix and of memory differ."""

def label_smoothed_loss(logits: Tensor, targets: ArrayLike, eps: float, pad_id: int) -> Tensor:
    """L0.3's cross_entropy with label_smoothing = eps, where every target
    equal to pad_id is ignored (no loss, no gradient, not counted in the
    mean)."""

def noam_rate(step: int, d_model: int, warmup: int, factor: float = 1.0) -> float:
    """The learning rate of update number `step` (0 for the first):
    factor * noam(step + 1, d_model, warmup) (M10.4), so the first update
    already has a positive rate and the peak is at update warmup - 1."""

def make_optimizer(model: Module) -> Adam:
    """M10.3's Adam over model.parameters() with betas (0.9, 0.98), eps 1e-9,
    no weight decay, lr 0 until a schedule sets it."""

def fit(
    model: Transformer,
    src: ArrayLike,
    tgt: ArrayLike,
    steps: int,
    batch: int,
    pad_id: int,
    warmup: int = 4000,
    factor: float = 1.0,
    smoothing: float = 0.1,
    lr: Optional[float] = None,
    clip: Optional[float] = None,
    rng: Any = None,
    on_step: Optional[Callable[[int, dict], None]] = None,
) -> list[float]:
    """Train for exactly `steps` updates and return each update's loss.
    src int [N, S] and tgt int [N, T + 1] (bos y_1 .. y_T, then pad), padded
    with pad_id. Batches come from L0.5's DataLoader(shuffle, rng; None means
    PCG32(0).substream("shuffle")), epoch after epoch; each update is L0.5's
    train_step (with clip) on label_smoothed_loss(model(src, tgt[:, :-1],
    src != pad, tgt[:, :-1] != pad), tgt[:, 1:], smoothing, pad_id), after
    setting the optimizer's lr to `lr` when given, else
    noam_rate(step, d_model, warmup, factor). The model is put in training
    mode. on_step(step, stats) is called after every update. A non-finite
    loss raises FloatingPointError (from train_step). ValueError for
    mismatched or too short arrays."""

def translate(
    model: Transformer,
    src: ArrayLike,
    bos: int,
    eos: int,
    pad_id: int,
    max_len: int,
    beam_size: int = 1,
    length_penalty: float = 1.0,
) -> list[list[int]]:
    """For each row of src int [B, S] (padded with pad_id), L4.4's
    beam_search over decode_step in eval mode under no_grad, and the best
    hypothesis's tokens without its final eos. The model's mode is restored
    afterwards. beam_size 1 is greedy decoding."""

def save_transformer(model: Transformer, dir: str) -> None:
    """Write the model directory above into dir (created if missing)."""

def load_transformer(dir: str) -> Transformer:
    """The Transformer a save_transformer directory holds, in eval mode,
    dropout 0. ValueError when tl_arch is not "transformer"."""
