# contracts/py/tinyllm/rnn/elmo.pyi (L3.5, optional): ELMo's biLM, ScalarMix, and linear probes
# chapter: content/courses/ai-engineering/09-recurrent-networks/05-elmo-bilm-scalarmix-probes.md
#
# Peters et al. (2018). Two language models read the same sentence: a forward
# LM predicts token t + 1 from tokens 0..t, a backward LM predicts token t - 1
# from tokens t..n-1. Each is a stack of L single-direction LSTMs (L3.2); the
# backward stack reads each sequence reversed inside its own length (L3.4's
# reverse_padded). The two share the token embedding and the softmax.
#
#   e_t        = E[ids_t]                                     [d]
#   f^k_t      = forward LSTM k over f^{k-1} (f^0 = e)         [d]
#   b^k_t      = backward LSTM k over b^{k-1} (b^0 = e)        [d], in source order
#   R^0_t      = [e_t ; e_t],  R^k_t = [f^k_t ; b^k_t]         [2d], k = 1..L
#   fwd logits = out(f^L_t) predicts ids_{t+1};  bwd logits = out(b^L_t) predicts ids_{t-1}
#
# ScalarMix turns the L + 1 layers into one task-specific vector:
#
#   ELMo_t = gamma * sum_k softmax(s)_k R^k_t
#
# with s [L + 1] (zeros at init: every layer weighted equally) and gamma
# [1] (ones), the task's own parameters (AllenNLP's ScalarMix without
# layer norm).
#
# A linear probe asks what a frozen representation already encodes: a
# softmax regression from features to labels, nothing deeper, so its
# accuracy measures the representation, not the probe.
#
# Parameters of BiLM, in registration order (= state_dict order), float32,
# from one rng (PCG32; None means PCG32(0).substream("init")) in this order:
#   emb.weight [V, d]                L0.4 Embedding
#   fwd.<k>.*  for k = 0..L-1        L3.2 LSTM(d, d), torch names (weight_ih_l0, ...)
#   bwd.<k>.*  for k = 0..L-1        L3.2 LSTM(d, d)
#   out.weight [V, d], out.bias [V]  L0.4 Linear, shared by both directions
# Token arrays are batch first: ids int [B, T], lengths int [B] in 1..T.
from typing import Any, Optional, Sequence

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Embedding, Linear, ModuleList
from tinyllm.nn.module import Module

class ScalarMix(Module):
    scalar_parameters: Tensor  # [n_layers], zeros
    gamma: Tensor  # [1], ones
    n_layers: int

    def __init__(self, n_layers: int) -> None:
        """ValueError for n_layers < 1."""

    def weights(self) -> NDArray:
        """softmax(scalar_parameters) as float64 [n_layers] (a read-out)."""

    def forward(self, layers: Sequence[Tensor]) -> Tensor:
        """gamma * sum_k softmax(s)_k layers[k], every layer the same shape.
        Gradients reach s, gamma, and every layer. ValueError for a wrong
        number of layers or mismatched shapes."""

class BiLM(Module):
    vocab: int
    d: int
    n_layers: int
    emb: Embedding
    fwd: ModuleList
    bwd: ModuleList
    out: Linear

    def __init__(self, vocab: int, d: int, n_layers: int = 2, rng: Any = None) -> None:
        """The parameters above. ValueError for a size below 1."""

    def layers(
        self, ids: ArrayLike, lengths: Optional[ArrayLike] = None
    ) -> list[Tensor]:
        """[R^0, R^1, ..., R^L], each [B, T, 2d], batch first. Padded positions
        (t >= lengths[b]) are 0 in every R^k with k >= 1; R^0 holds the
        embedding of whatever id sits there. lengths None means every row
        has length T. ValueError for ids not int [B, T], an id outside
        [0, V), or lengths outside 1..T."""

    def forward(
        self, ids: ArrayLike, lengths: Optional[ArrayLike] = None
    ) -> tuple[Tensor, Tensor]:
        """(fwd_logits, bwd_logits), each [B, T, V]: out(f^L) and out(b^L) at
        every position. Same validation as layers."""

def bilm_loss(
    model: BiLM, ids: ArrayLike, lengths: Optional[ArrayLike] = None
) -> Tensor:
    """(forward CE + backward CE) / 2, each the mean cross-entropy (L0.3) over
    the valid predictions: forward at t = 0..len-2 predicting ids[t + 1],
    backward at t = 1..len-1 predicting ids[t - 1]. A row of length 1 has no
    prediction. ValueError when no row has a prediction."""

def fit_linear_probe(
    features: ArrayLike,
    labels: ArrayLike,
    n_classes: int,
    l2: float = 1e-3,
    steps: int = 200,
    lr: float = 0.5,
) -> tuple[NDArray, NDArray]:
    """Multinomial logistic regression by full-batch gradient descent in
    float64, from W = 0, b = 0: for `steps` steps, P = softmax(X W + b),
    W -= lr * (X^T (P - Y) / N + l2 W), b -= lr * sum(P - Y) / N, with X the
    features standardized by their own mean and std (std 0 counts as 1) and
    Y one-hot. Returns (W [D, C], b [C]) for the standardized features,
    with the mean and std folded in: predict with features @ W + b directly.
    Deterministic. ValueError for features not [N, D], labels not N
    integers in [0, n_classes), n_classes < 2, or steps < 0."""

def probe_accuracy(
    W: ArrayLike, b: ArrayLike, features: ArrayLike, labels: ArrayLike
) -> float:
    """Fraction of rows whose argmax of features @ W + b (ties to the lowest
    class) equals the label."""
