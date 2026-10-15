# contracts/py/tinyllm/obj/heads.pyi (L6.5): fine-tuning heads and the linear-head export
# chapter: content/courses/build-an-llm/p06-objectives/05-fine-tuning-heads-and-the-linear-head-export.md
#
# A head reads a backbone's hidden states h [B, T, d]:
#   backbone with a hidden(ids) method (L6.1's GPT): h = backbone.hidden(ids)
#   otherwise (L6.2's BertEncoder, L6.3's discriminator body):
#                                  h = backbone(ids, token_type_ids, attn_mask)
# attn_mask is bool or 0/1 [B, T], True = a real token; None = all real.
# Padding is on the right. Pooling one vector per row:
#   "cls"   h[:, 0]                                  BERT's [CLS]
#   "mean"  sum_t m_t h_t / sum_t m_t                  real tokens only
#   "last"  h[b, last real position of row b]          a decoder's summary
#
# Parameters (after the backbone's, whose names start "backbone."):
#   SequenceClassifier, TokenClassifier: classifier.weight [C, d], classifier.bias [C]
#   RewardHead: score.weight [1, d] (no bias)
# each an L0.4 Linear built from rng (PCG32; None means PCG32(0).substream("init")).
#
# The classifier directory (save_classifier, load_classifier) for the L6.7 zoo:
# config.json {"tl_arch": "bert" | "electra" | "gpt", "tl_tokenizer",
#   "tl_format": 1, "tl_backbone": the backbone's BertConfig or GPTConfig as a
#   dict (field names), "tl_head": {"kind": "sequence", "n_classes", "pool",
#   "d_model", "labels": [class names]}} and model.safetensors (F32, metadata
#   {"format": "tinyllm", "tl_arch": <arch>}).
#
# The linear policy head (D33, formats/linear-head.schema.json), scored by the
# gateway (gw.08) on an engine's /v1/embeddings output e:
#   u = e / ||e|| (a zero vector stays zero);  p = softmax(W u + b)
# fit_linear_head fits a binary head with M07.7's logistic_regression_fit
# (IRLS, intercept last) on the unit rows and stores it as two softmax rows,
# class 0 the reference: W = [0; w], b = [0, c], so p[1] = sigmoid(w u + c).
from typing import Any, Callable, Literal, Mapping, Optional, Sequence

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Dropout, Linear
from tinyllm.nn.module import Module

POOLS: tuple[str, ...]  # ("cls", "mean", "last")
ARCHS: tuple[str, ...]  # ("bert", "electra", "gpt")

def hidden_states(
    backbone: Module, ids: ArrayLike, token_type_ids: Optional[ArrayLike] = None, attn_mask: Optional[ArrayLike] = None
) -> Tensor:
    """h [B, T, d] as above."""

def pool(h: Tensor, attn_mask: Optional[ArrayLike], how: str) -> Tensor:
    """[B, d] as above. ValueError for an unknown pool, a mask of another
    shape than h's first two axes, or a row with no real token."""

class SequenceClassifier(Module):
    pool: str
    d_model: int
    n_classes: int
    backbone: Module
    dropout: Dropout
    classifier: Linear

    def __init__(
        self,
        backbone: Module,
        d_model: int,
        n_classes: int,
        pool: Literal["cls", "mean", "last"] = "cls",
        dropout: float = 0.0,
        rng: Any = None,
    ) -> None:
        """ValueError for an unknown pool, n_classes < 2, or d_model < 1."""

    def forward(
        self,
        ids: ArrayLike,
        token_type_ids: Optional[ArrayLike] = None,
        attn_mask: Optional[ArrayLike] = None,
        labels: Optional[ArrayLike] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """(logits [B, C] = classifier(dropout(pool(h))), loss): loss is None
        without labels, else L0.3's mean cross_entropy against labels [B]."""

class TokenClassifier(Module):
    d_model: int
    n_labels: int
    backbone: Module
    dropout: Dropout
    classifier: Linear

    def __init__(self, backbone: Module, d_model: int, n_labels: int, dropout: float = 0.0, rng: Any = None) -> None:
        """ValueError for n_labels < 2 or d_model < 1."""

    def forward(
        self,
        ids: ArrayLike,
        token_type_ids: Optional[ArrayLike] = None,
        attn_mask: Optional[ArrayLike] = None,
        labels: Optional[ArrayLike] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """(logits [B, T, C], loss): loss is the mean cross_entropy over the
        real positions whose label is not -100 (padding never counts)."""

class RewardHead(Module):
    d_model: int
    backbone: Module
    score: Linear

    def __init__(self, backbone: Module, d_model: int, rng: Any = None) -> None: ...
    def forward(self, ids: ArrayLike, attn_mask: Optional[ArrayLike] = None) -> Tensor:
        """Rewards [B]: score(pool(h, attn_mask, "last"))."""

def pairwise_reward_loss(r_chosen: Tensor, r_rejected: Tensor) -> Tensor:
    """Bradley-Terry: mean over pairs of -log sigmoid(r_chosen - r_rejected),
    computed as L0.3's bce_with_logits of the margin against 1. ValueError
    unless both are [B]."""

def lora_classifier(
    clf: Module,
    r: int,
    alpha: float,
    target: Optional[Callable[[str, Module], bool]] = None,
    init: Literal["default", "pissa"] = "default",
    rng: Any = None,
) -> list[str]:
    """L6.6's inject_lora(clf, target, r, alpha, init=init, rng=rng), then
    every parameter of the head (classifier, or score for a RewardHead)
    trainable again. The default target is every Linear named
    "backbone.<...>.q_proj" or "backbone.<...>.v_proj" (attention queries and
    values). Returns the adapted names."""

def train_classifier(
    clf: Module,
    ids: ArrayLike,
    labels: ArrayLike,
    steps: int,
    batch_size: int,
    lr: float,
    rng: Any,
    attn_mask: Optional[ArrayLike] = None,
    weight_decay: float = 0.0,
) -> list[float]:
    """Exactly `steps` updates of M10.3's AdamW(lr, weight_decay) over the
    parameters with requires_grad, in training mode; each batch is
    batch_size rows drawn with replacement, row = rng.below(n), in order.
    Leaves clf in eval mode; returns each step's loss. ValueError for steps
    or batch_size < 1 or a label count that differs from the rows."""

def predict(clf: Module, ids: ArrayLike, attn_mask: Optional[ArrayLike] = None, batch_size: int = 64) -> NDArray:
    """int64 [n]: the argmax class of each row (ties to the lowest), in eval
    mode under no_grad, batch_size rows at a time."""

def save_classifier(
    clf: "SequenceClassifier", dir: str, arch: str, tokenizer: str = "file", labels: Sequence[str] = ()
) -> None:
    """Write the classifier directory above. labels default to "0", "1", ...
    ValueError for an unknown arch, a classifier that still holds LoRA
    adapters (merge_lora first), or a wrong number of labels."""

def load_classifier(dir: str) -> tuple["SequenceClassifier", dict[str, Any]]:
    """(the classifier, in eval mode with dropout 0, config.json as a dict):
    the backbone is GPT(GPTConfig(**tl_backbone)) for "gpt", else
    BertEncoder(BertConfig(**tl_backbone)). ValueError when config.json is
    not a sequence classifier."""

def fit_linear_head(
    embeddings: ArrayLike,
    labels: ArrayLike,
    classes: Sequence[str],
    l2: float,
    iters: int = 50,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """{"dim", "classes", "W" float64 [2, dim], "b" float64 [2], "threshold",
    "l2"} as above; labels are 0/1 (1 = classes[1]). ValueError unless there
    are exactly 2 classes and 0 <= threshold <= 1 (and as M07.7)."""

def head_probs(head: Mapping[str, Any], embeddings: ArrayLike) -> NDArray:
    """float64 [n, C]: softmax(W u + b) per row in float64, the max logit
    subtracted first. ValueError for a dimension mismatch or non-finite
    embeddings."""

def head_metrics(head: Mapping[str, Any], embeddings: ArrayLike, labels: ArrayLike) -> dict[str, float]:
    """Over p = head_probs(...)[:, -1] with flagged = p >= threshold:
    {"accuracy", "precision", "recall" (0.0 when undefined), "auc": M07.7's
    roc_auc(p, labels), "ece": M07.7's ece(p, labels)}."""

def export_linear_head(
    head: Mapping[str, Any], embedding_model: str, path: str, metrics: Optional[Mapping[str, float]] = None
) -> None:
    """Write the formats/linear-head.schema.json document: embedding_model,
    dim, classes, W, b, threshold (and metrics when given), every number a
    JSON float that reads back to the same float64. ValueError for an empty
    embedding_model, shapes that do not match the classes and dim, or a
    non-finite weight."""

def load_linear_head(path: str) -> dict[str, Any]:
    """The exported document with W and b as float64 arrays. ValueError when
    W's shape does not match classes and dim."""
