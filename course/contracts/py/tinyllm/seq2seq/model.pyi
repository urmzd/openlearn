# contracts/py/tinyllm/seq2seq/model.pyi (L4.1): an encoder-decoder with teacher forcing
# chapter: content/courses/ai-engineering/10-attention/01-encoder-decoder-teacher-forcing.md
#
# Sutskever, Vinyals, and Le (2014); Cho et al. (2014). An encoder reads the
# source; a decoder writes the target one token at a time, conditioned on the
# encoder through its initial state and, optionally, attention (L4.2, L4.3).
#
# Encoder: src_emb, then a bidirectional GRU (L3.4's bidirectional over two
# L3.3 GRUs of hidden size d_h / 2, so d_h must be even), lengths-aware:
#     keys [B, S, d_h] = bidirectional(enc_fwd, enc_bwd, src_emb(src)^T, src_lens)^T
#     s_0  [B, d_h]    = tanh(bridge([fwd state after the last real token ;
#                                     bwd state after reading back to t = 0]))
#
# Decoder, one step from token y_prev with state s (an LSTM cell keeps (s, c),
# c_0 = 0), by attention.query_from:
#   no attention:  s' = cell(tgt_emb(y), s);            logits = out(s')
#   "previous"     (Bahdanau, L4.2):
#                  ctx, a = attention(s, keys, mask)       # reads with the OLD state
#                  s' = cell([tgt_emb(y) ; ctx], s);       logits = out([s' ; ctx])
#   "current"      (Luong, L4.3, input feeding):
#                  s' = cell([tgt_emb(y) ; feed], s)       # feed = h~ of the step before, 0 at first
#                  ctx, a = attention(s', keys, mask)      # reads with the NEW state
#                  feed' = attention.attentional(s', ctx); logits = out(feed')
#
# Parameters, in registration order (= state_dict order = safetensors keys),
# float32, all drawn from one rng (PCG32; None means PCG32(0).substream("init"))
# in this order, except the attention module, which is built by the caller:
#   src_emb.weight [Vs, d_emb]           L0.4 Embedding
#   enc_fwd.*, enc_bwd.*                 L3.3 GRU(d_emb, d_h / 2) each (torch names)
#   bridge.weight [d_h, d_h], bridge.bias [d_h]
#   tgt_emb.weight [Vt, d_emb]
#   cell.*                               L3.3 GRUCell or L3.2 LSTMCell, input
#                                        d_emb (no attention) or d_emb + d_h
#   attention.*                          the given module (absent when None)
#   out.weight [Vt, d_h or 2 d_h], out.bias [Vt]   2 d_h for "previous"
#
# Token arrays are batch first: src int [B, S] padded after src_lens[b]
# (any id there; it is never read), tgt_in int [B, T] (bos, y_1, ..., y_{T-1}).
#
# The model directory (save_seq2seq, load_seq2seq) for the L6.7 zoo:
# config.json {"tl_arch": "seq2seq", "tl_tokenizer": "file", "vocab_size": Vt,
#   "tl_src_vocab": Vs, "tl_format": 1, "tl_cell": cell, "tl_d_emb": d_emb,
#   "hidden_size": d_h, "tl_seq2seq_attention": "none" | "bahdanau" |
#   "luong-dot" | "luong-general" | "luong-concat", "tl_d_attn": A (bahdanau)}
# and model.safetensors (F32, metadata {"format": "tinyllm", "tl_arch": "seq2seq"}).
from typing import Any, Literal, NamedTuple, Optional

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module

class EncoderState(NamedTuple):
    keys: Tensor  # [B, S, d_h], every encoder output
    mask: NDArray  # bool [B, S], L4.2's length_mask(src_lens, S)
    proj: Optional[Tensor]  # attention.project_keys(keys), or None without attention
    init: Tensor  # [B, d_h], s_0

class DecoderState(NamedTuple):
    h: Tensor  # [B, d_h]
    c: Optional[Tensor]  # [B, d_h] for an LSTM cell, else None
    feed: Optional[Tensor]  # [B, d_h] h~ of the last step (Luong), else None
    keys: Tensor
    mask: NDArray
    proj: Optional[Tensor]

class Seq2Seq(Module):
    src_vocab: int
    tgt_vocab: int
    d_emb: int
    d_h: int
    cell_type: str  # "gru" | "lstm"
    attention: Optional[Module]

    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_emb: int,
        d_h: int,
        cell: Literal["gru", "lstm"] = "gru",
        attention: Optional[Module] = None,
        rng: Any = None,
    ) -> None:
        """The parameters above. attention is an L4.2 AdditiveAttention(d_h,
        d_h, A) or an L4.3 LuongAttention(d_h, score), or None. ValueError
        for a size below 1, an odd d_h, another cell, or an attention whose
        query_from is neither "previous" nor "current"."""

    def encode(self, src: ArrayLike, src_lens: ArrayLike) -> EncoderState:
        """The encoder above. ValueError for src not int [B, S], lengths outside
        1..S, or an id outside [0, src_vocab)."""

    def init_state(self, enc: EncoderState) -> DecoderState:
        """The decoder state before the first step: h = enc.init, c = 0 for
        an LSTM cell, feed = 0 for Luong attention."""

    def decode_step(
        self, y_prev: ArrayLike, state: DecoderState
    ) -> tuple[Tensor, DecoderState, Optional[Tensor]]:
        """One decoder step above for y_prev int [B]: (logits [B, Vt], the new
        state, attention weights [B, S] or None without attention)."""

    def forward(
        self,
        src: ArrayLike,
        src_lens: ArrayLike,
        tgt_in: ArrayLike,
        teacher_forcing: float = 1.0,
        rng: Any = None,
    ) -> Tensor:
        """Logits [B, T, Vt]: encode, then decode_step for t = 0 .. T-1. The
        input at t = 0 is tgt_in[:, 0]. At t >= 1 it is tgt_in[:, t] (teacher
        forcing) when teacher_forcing >= 1, which draws nothing; otherwise
        one u = rng.uniform() per step for the whole batch, and the input is
        tgt_in[:, t] when u < teacher_forcing, else the argmax of step t-1's
        logits (ties to the lowest id, no gradient through the choice).
        ValueError for tgt_in not int [B, T >= 1] with the batch of src, an
        id outside [0, tgt_vocab), teacher_forcing outside [0, 1], or
        teacher_forcing < 1 without an rng."""

    def greedy(
        self, src: ArrayLike, src_lens: ArrayLike, bos: int, eos: int, max_len: int
    ) -> list[list[int]]:
        """Greedy decoding under no_grad, all rows at once: per row, the argmax
        token (ties to the lowest id) at each step after bos, up to and
        including its first eos, at most max_len tokens. ValueError for
        max_len < 1."""

def save_seq2seq(model: Seq2Seq, dir: str) -> None:
    """Write the model directory above into dir (created if missing)."""

def load_seq2seq(dir: str) -> Seq2Seq:
    """The Seq2Seq a save_seq2seq directory holds, attention included,
    parameters equal to the saved ones. ValueError when tl_arch is not
    "seq2seq" or a key is missing."""
