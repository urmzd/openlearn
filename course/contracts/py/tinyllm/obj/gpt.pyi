# contracts/py/tinyllm/obj/gpt.pyi (L6.1): GPT, the causal LM loss, GPT-2 weights
# chapter: content/courses/ai-engineering/12-objectives-and-adaptation/01-gpt-decoder-only.md
#
# Radford et al. (2019), GPT-2. One stack of pre-LN blocks over token plus
# learned position embeddings (L5.4's LearnedPE), causal self-attention
# (L5.2's causal_mask, L5.3's MultiHeadAttention), a final LayerNorm, and an
# output layer tied to the token embedding:
#
#   x = drop(wte(ids) + wpe(positions 0..T-1))
#   block: x = x + drop(attn(ln_1(x), ln_1(x), causal))
#          x = x + drop(mlp_proj(gelu_tanh(mlp_fc(ln_2(x)))))
#   logits = ln_f(x) @ wte.weight^T                     (tie=True)
#
# GELU is the tanh approximation (HF's "gelu_new", L0.2's approximate="tanh").
#
# Parameters, in registration order (= state_dict order), float32, drawn from
# one rng (PCG32; None means PCG32(0).substream("init")), with GPT-2's init:
# N(0, 0.02^2) for wte, wpe, and every Linear weight, except attn.out_proj and
# mlp_proj, which write into the residual stream and get
# std = scaled_residual_std(0.02, n_layers) (M07.3); biases 0, LayerNorms 1, 0:
#   wte.weight [V, d], wpe.weight [n_ctx, d]
#   h.{i}.ln_1.*, h.{i}.attn.{q,k,v,out}_proj.*, h.{i}.ln_2.*,
#   h.{i}.mlp_fc.* [d_ff, d], h.{i}.mlp_proj.* [d, d_ff]
#   ln_f.*;  lm_head.weight [V, d] only when tie=False
#
# Hugging Face's GPT-2 keeps its projections in Conv1D modules, whose weight
# is [in, out] (y = x W + b), the transpose of a Linear's; attn.c_attn packs q,
# k, v as three d-wide column blocks of one [d, 3 d] weight.
#
# The model directory (save_gpt, load_gpt) for the L6.7 zoo:
# config.json {"tl_arch": "gpt", "tl_tokenizer", "tl_format": 1, "vocab_size",
#   "n_positions", "n_embd", "n_head", "n_layer", "n_inner",
#   "layer_norm_epsilon", "tie_word_embeddings"}
# and model.safetensors (F32, metadata {"format": "tinyllm", "tl_arch": "gpt"}).
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module

INIT_STD: float  # 0.02

@dataclass
class GPTConfig:
    vocab: int
    n_ctx: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float = 0.0
    ln_eps: float = 1e-5
    tie: bool = True

class Block(Module):
    def __init__(self, cfg: GPTConfig, rng: Any) -> None: ...
    def forward(self, x: Tensor, mask: NDArray) -> Tensor:
        """One pre-LN block above."""

class GPT(Module):
    # The L6.7 zoo (tl_arch "gpt"), L6.5's heads, L6.6's LoRA, and L8.2's
    # first KV cache build on it.
    cfg: GPTConfig

    def __init__(self, cfg: GPTConfig, rng: Any = None) -> None:
        """ValueError for a size below 1 (and as L5.3 for d_model, n_heads)."""

    def hidden(self, ids: ArrayLike) -> Tensor:
        """ln_f output [B, T, d]. ValueError for ids not int [B, T] or
        T > n_ctx."""

    def forward(self, ids: ArrayLike, targets: Optional[ArrayLike] = None) -> tuple[Tensor, Optional[Tensor]]:
        """(logits [B, T, V], loss): loss is None without targets, else L0.3's
        mean cross_entropy of logits against targets int [B, T] (already
        shifted by the caller; -100 is ignored)."""

def clm_loss(logits: Tensor, ids: ArrayLike, mask: Optional[ArrayLike] = None) -> Tensor:
    """The causal LM loss: position t predicts ids[:, t + 1], so the mean
    cross-entropy of logits[:, :-1] against ids[:, 1:]. mask bool [B, T]
    marks real tokens: a target whose own position is padding is ignored.
    ValueError for logits not [B, T, V] matching ids [B, T], T < 2, or a
    mask of another shape."""

def load_hf_gpt2(model: GPT, sd: Mapping[str, Any]) -> None:
    """Copy a Hugging Face GPT-2 state dict (GPT2LMHeadModel or GPT2Model:
    keys with or without the "transformer." prefix; HF's attn.bias and
    attn.masked_bias buffers are ignored) into model, in place: Conv1D
    weights transposed, c_attn split into q, k, v. lm_head.weight is read only
    when tie=False. KeyError naming a missing key; ValueError for a shape
    that does not match the model's config."""

def fit_gpt(
    model: GPT,
    stream: Any,
    steps: int,
    lr: float,
    warmup: int = 0,
    weight_decay: float = 0.1,
    clip: Optional[float] = 1.0,
) -> list[float]:
    """Train for exactly `steps` updates and return each update's loss.
    Each update: (x, y) = stream.next_batch() (L0.6's TokenStream), the lr
    cosine_with_warmup(step, warmup, steps, lr, 0.1 * lr) (M10.4), then
    L0.5's train_step on model(x, y)'s loss with M10.3's AdamW (betas 0.9,
    0.95, weight_decay) and clip. ValueError for steps < 1."""

def save_gpt(model: GPT, dir: str, tokenizer: str = "bytes") -> None:
    """Write the model directory above into dir (created if missing)."""

def load_gpt(dir: str) -> GPT:
    """The GPT a save_gpt directory holds, in eval mode. ValueError when
    tl_arch is not "gpt"."""
