# contracts/py/tinyllm/obj/electra.pyi (L6.3): ELECTRA replaced-token detection
# chapter: content/courses/build-an-llm/07-objectives/03-electra-replaced-token-detection.md
#
# Clark et al. (2020). A generator G (a small masked LM, L6.2's BertForMLM)
# and a discriminator D (L6.2's BertEncoder plus a token head). One step on
# ids [B, T] with real-token mask m (attn_mask, True = real):
#
#   1. (inputs, labels) = mlm_mask(ids, special | ~m, mask_id, V, p, rng)     L6.2
#   2. (gen_logits, L_G) = G(inputs, types, m, labels)                          MLM loss
#   3. corrupt = ids; at each picked position (label != -100), in C order:
#        corrupt = sample_categorical(softmax(gen_logits), rng.uniform())      M07.1
#      is_replaced = (corrupt != ids): a sample equal to the original is ORIGINAL
#   4. L_D = mean over real tokens of BCE(D(corrupt, types, m), is_replaced)
#   5. L = L_G + lam * L_D           (lam = 50, the paper's)
#
# The sample is data: no gradient flows from L_D into G. The draws, in order:
# mlm_mask's (L6.2's op order), then one uniform per picked position.
#
# Parameters, in registration order (= state_dict order), float32, from one
# rng (PCG32; None means PCG32(0).substream("init")):
#   ElectraDiscriminator: electra.* (a BertEncoder, L6.2's names),
#     discriminator_predictions.dense.* [d, d], discriminator_predictions.dense_prediction.* [1, d]
#     (Hugging Face's ElectraForPreTraining names; weights N(0, 0.02^2), biases 0)
#   ELECTRA: discriminator.*, then generator.* (a BertForMLM); with
#     tie_embeddings the generator's word, position, and type embeddings ARE
#     the discriminator's Tensors (listed once, under discriminator.*).
#
# The model directory (save_electra, load_electra) for the L6.7 zoo:
# config.json {"tl_arch": "electra", "tl_tokenizer", "tl_format": 1,
#   the discriminator's "vocab_size", "max_position_embeddings", "hidden_size",
#   "num_attention_heads", "num_hidden_layers", "intermediate_size",
#   "type_vocab_size", "layer_norm_eps", "tl_generator": {the same keys},
#   "tl_tie_embeddings", "tl_mask_id", "tl_special_ids"}
# and model.safetensors (F32, metadata {"format": "tinyllm", "tl_arch": "electra"}).
from typing import Any, Optional, Sequence

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module
from tinyllm.obj.bert import BertConfig, BertEncoder, BertForMLM

INIT_STD: float  # 0.02, the head's init std
HF_KEYS: dict[str, str]  # BertConfig field -> config.json key

class ElectraDiscriminatorHead(Module):
    dense: Linear
    dense_prediction: Linear

    def __init__(self, d_model: int, rng: Any = None) -> None: ...
    def forward(self, h: Tensor) -> Tensor:
        """[..., d] -> [...]: dense_prediction(gelu(dense(h))), the last axis
        dropped. GELU is the exact (erf) one."""

class ElectraDiscriminator(Module):
    cfg: BertConfig
    electra: BertEncoder
    discriminator_predictions: ElectraDiscriminatorHead

    def __init__(self, cfg: BertConfig, rng: Any = None) -> None: ...
    def forward(
        self, ids: ArrayLike, token_type_ids: Optional[ArrayLike] = None, attn_mask: Optional[ArrayLike] = None
    ) -> Tensor:
        """Replaced-token logits [B, T] (> 0 means "replaced")."""

class ELECTRA(Module):
    tie_embeddings: bool
    discriminator: ElectraDiscriminator
    generator: BertForMLM

    def __init__(
        self, gen_cfg: BertConfig, disc_cfg: BertConfig, rng: Any = None, tie_embeddings: bool = True
    ) -> None:
        """Discriminator first, then generator, from one rng. ValueError when
        the vocabularies differ, or when tie_embeddings and d_model, max_len,
        or type_vocab differ."""

    def forward(self, ids: ArrayLike, rng: Any, **kw: Any) -> dict[str, Any]:
        """electra_step(self.generator, self.discriminator, ids, rng, **kw)."""

def replace_tokens(
    ids: ArrayLike, labels: ArrayLike, gen_logits: ArrayLike, rng: Any, ignore_index: int = -100
) -> tuple[NDArray, NDArray]:
    """Step 3: (corrupt int64 [B, T], is_replaced float32 [B, T] of 0 and 1).
    Positions whose label is ignore_index are copied and draw nothing; the
    others draw one rng.uniform() each, in C order, and sample from the
    softmax of their logits row in float64. ValueError for ids not integer,
    labels of another shape, or logits not [*ids.shape, V]."""

def rtd_loss(disc_logits: Tensor, is_replaced: ArrayLike, attn_mask: Optional[ArrayLike] = None) -> Tensor:
    """Step 4: L0.3's bce_with_logits over the real positions only (all
    positions when attn_mask is None), the mean over them. ValueError for a
    shape mismatch."""

def electra_step(
    gen: Module,
    disc: Module,
    ids: ArrayLike,
    rng: Any,
    lam: float = 50.0,
    *,
    mask_id: int,
    vocab: Optional[int] = None,
    special_mask: Optional[ArrayLike] = None,
    token_type_ids: Optional[ArrayLike] = None,
    attn_mask: Optional[ArrayLike] = None,
    p: float = 0.15,
) -> dict[str, Any]:
    """Steps 1 to 5. gen is called gen(inputs, token_type_ids, attn_mask,
    labels) -> (logits, loss) and disc as disc(ids, token_type_ids,
    attn_mask) -> logits [B, T]; vocab defaults to gen.bert.cfg.vocab.
    Returns {"loss", "gen_loss", "disc_loss", "disc_logits": Tensors;
    "inputs", "labels", "corrupt", "is_replaced": arrays}."""

def rtd_accuracy(
    model: ELECTRA,
    ids: ArrayLike,
    rng: Any,
    *,
    mask_id: int,
    special_mask: Optional[ArrayLike] = None,
    token_type_ids: Optional[ArrayLike] = None,
    attn_mask: Optional[ArrayLike] = None,
    p: float = 0.15,
) -> NDArray:
    """The zoo's ELECTRA metric: one electra_step under no_grad, then per real
    token (C order) 1.0 when the discriminator's call (logit > 0 means
    replaced) matches is_replaced, else 0.0. float64 [number of real tokens]."""

def save_electra(
    model: ELECTRA, dir: str, mask_id: int, special_ids: Sequence[int] = (), tokenizer: str = "file"
) -> None:
    """Write the model directory above into dir (created if missing)."""

def load_electra(dir: str) -> tuple[ELECTRA, dict[str, Any]]:
    """(the ELECTRA a save_electra directory holds, in eval mode with dropout
    0, its config.json as a dict). ValueError when tl_arch is not "electra"."""
