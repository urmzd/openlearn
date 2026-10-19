# contracts/py/tinyllm/obj/bert.pyi (L6.2): BERT and masked-LM masking
# chapter: content/courses/build-an-llm/07-objectives/02-bert-masked-lm.md
#
# Devlin et al. (2019). A post-LN encoder (L5.5's norm="post") with exact
# GELU, learned positions (L5.4), and token type (segment) embeddings, read in
# both directions (padding is the only mask):
#
#   x = drop(emb_norm(word_emb(ids) + pos_emb(0..T-1) + type_emb(types)))
#   layer: x = attn_norm(x + drop(attn(x, x, padding)))
#          x = ff_norm(x + drop(ff2(gelu(ff1(x)))))
#   MLM head: logits = transform_norm(gelu(transform(x))) @ word_emb.weight^T + decoder_bias
#
# Parameters, in registration order (= state_dict order), float32, drawn from
# one rng (PCG32; None means PCG32(0).substream("init")), N(0, 0.02^2) for
# embeddings and Linear weights, biases 0, LayerNorms 1, 0:
#   BertEncoder: word_emb.weight [V, d], pos_emb.weight [max_len, d],
#     type_emb.weight [type_vocab, d], emb_norm.*,
#     layers.{i}.attn.{q,k,v,out}_proj.*, layers.{i}.attn_norm.*,
#     layers.{i}.ff1.* [d_ff, d], layers.{i}.ff2.* [d, d_ff], layers.{i}.ff_norm.*
#   BertForMLM: decoder_bias [V] (its own parameter, so first, L0.4), then
#     bert.* (the encoder), transform.* [d, d], transform_norm.*; the
#     decoder's weight IS bert.word_emb.weight
#
# Masking (mlm_mask), the op order every implementation follows so that one
# PCG32 seed gives one result. Over the positions of ids in C order:
#   special position: skip, draw nothing, label -100
#   u = rng.uniform(); u >= p: not picked, label -100
#   picked: label = the original id; a = sample_categorical([0.8, 0.1, 0.1],
#     rng.uniform()) (M07.1); a = 0: input mask_id; a = 1: input
#     rng.below(vocab); a = 2: input unchanged
#
# The model directory (save_bert, load_bert) for the L6.7 zoo:
# config.json {"tl_arch": "bert", "tl_tokenizer", "tl_format": 1, "vocab_size",
#   "max_position_embeddings", "hidden_size", "num_attention_heads",
#   "num_hidden_layers", "intermediate_size", "type_vocab_size", "layer_norm_eps"}
# and model.safetensors (F32, metadata {"format": "tinyllm", "tl_arch": "bert"}).
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module
from tinyllm.tok.wordpiece import WordPieceTokenizer

INIT_STD: float  # 0.02
MASK: int  # 0, the index of each action in SPLIT
RANDOM: int  # 1
KEEP: int  # 2
SPLIT: NDArray  # [0.8, 0.1, 0.1]
SPECIALS: tuple[str, ...]  # ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")

@dataclass
class BertConfig:
    vocab: int
    max_len: int = 512
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    type_vocab: int = 2
    dropout: float = 0.1
    ln_eps: float = 1e-12

def mlm_mask(
    ids: ArrayLike, special_mask: ArrayLike, mask_id: int, vocab: int, p: float, rng: Any
) -> tuple[NDArray, NDArray]:
    """(inputs, labels), int64 arrays of ids's shape, by the op order above;
    ids is not modified. rng has uniform() and below(n) (PCG32, M06.3).
    ValueError for non-integer ids, a special_mask of another shape, p
    outside [0, 1], or mask_id outside [0, vocab)."""

def special_ids(tok: WordPieceTokenizer) -> list[int]:
    """The ids of the SPECIALS the tokenizer has, ascending."""

def mlm_batch(
    tok: WordPieceTokenizer, texts: Sequence[str], max_len: int, p: float, rng: Any
) -> dict[str, NDArray]:
    """One MLM batch from raw text with L1.3's tokenizer: each text is
    tok.encode(text, add_special=True) ([CLS] ... [SEP]), cut to max_len
    keeping [SEP] last, padded with [PAD] to the longest row; then mlm_mask
    with special_mask = special_ids or padding, mask_id = [MASK], vocab =
    the largest id + 1. Returns {"input_ids", "token_type_ids" (zeros),
    "attention_mask" (bool, True = real), "labels"}. ValueError without
    [PAD] and [MASK] or for max_len < 2."""

class BertLayer(Module):
    def __init__(self, cfg: BertConfig, rng: Any) -> None: ...
    def forward(self, x: Tensor, mask: NDArray) -> Tensor:
        """One post-LN layer above; mask broadcasts to [B, H, T, T]."""

class BertEncoder(Module):
    # L6.3's ELECTRA discriminator and L6.5's classification heads read it.
    cfg: BertConfig

    def __init__(self, cfg: BertConfig, rng: Any = None) -> None:
        """ValueError for a size below 1 (and as L5.3 for d_model, n_heads)."""

    def forward(
        self, ids: ArrayLike, token_type_ids: Optional[ArrayLike] = None, attn_mask: Optional[ArrayLike] = None
    ) -> Tensor:
        """Hidden states [B, T, d]. token_type_ids default 0; attn_mask
        (bool or 0/1, True = a real token) default all real; a padded key is
        never read. ValueError for ids not int [B, T] or masks of another
        shape."""

class BertForMLM(Module):
    bert: BertEncoder

    def __init__(self, cfg: BertConfig, rng: Any = None) -> None: ...
    def forward(
        self,
        ids: ArrayLike,
        token_type_ids: Optional[ArrayLike] = None,
        attn_mask: Optional[ArrayLike] = None,
        labels: Optional[ArrayLike] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """(logits [B, T, V], loss): loss is None without labels, else L0.3's
        mean cross_entropy over the positions whose label is not -100."""

def hf_bert_encoder_sd(sd: Mapping[str, Any], n_layers: int) -> dict[str, NDArray]:
    """A Hugging Face BertModel / BertForMaskedLM state dict (keys with or
    without "bert.") renamed to BertEncoder's keys. HF's BERT uses
    nn.Linear, so no weight is transposed. KeyError naming a missing key."""

def load_hf_bert(model: Module, sd: Mapping[str, Any]) -> None:
    """Load an HF state dict into a BertEncoder, or into a BertForMLM (also
    cls.predictions.transform.* and cls.predictions.bias), in place.
    TypeError for another module."""

def save_bert(model: BertForMLM, dir: str, tokenizer: str = "file") -> None:
    """Write the model directory above into dir (created if missing)."""

def load_bert(dir: str) -> BertForMLM:
    """The BertForMLM a save_bert directory holds, in eval mode, dropout 0.
    ValueError when tl_arch is not "bert"."""
