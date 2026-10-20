# contracts/py/tinyllm/seq2seq/additive.pyi (L4.2): Bahdanau additive attention
# chapter: content/courses/ai-engineering/10-attention/02-bahdanau-additive-attention.md
#
# Bahdanau, Cho, and Bengio (2015). A decoder state (the query q) reads the
# encoder outputs (the keys k_1 .. k_S) through a weighted average whose
# weights it computes itself, one score per source position:
#
#     e_s     = v . tanh(W q + U k_s + b)                 score, a scalar
#     e_s     = -inf where mask[s] is False               padding is never read
#     a       = softmax(e)                                 weights over s, sum 1
#     context = sum_s a_s k_s                              [Dk]
#
# Shapes, batch first:
#   query  [B, Dq]     the decoder state that asks
#   keys   [B, S, Dk]  the encoder outputs that answer (also the values)
#   mask   bool [B, S], True where a position may be read (length_mask)
#   proj   [B, S, A]   U k_s + b for every s: project_keys(keys), computed once
#                      per source sentence and reused at every decoder step
#
# Parameters, in registration order (= state_dict order = safetensors keys),
# float32, from L0.4's Linear, drawn from one rng (PCG32, M06.3; None means
# PCG32(0).substream("init")) in this order:
#   query.weight  W  [A, Dq]   Linear(Dq, A, bias=False)
#   key.weight    U  [A, Dk]   Linear(Dk, A)
#   key.bias      b  [A]
#   v.weight      v  [1, A]    Linear(A, 1, bias=False)
#
# Every forward is built from the op library (L0.2), so backward reaches the
# query, the keys, and every parameter. A row whose mask is all False (an
# empty source) gets weights 0 and context 0, not NaN (M09.2's softmax of an
# all -inf row), and passes no gradient.
from typing import Any, Literal, Optional

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module

def length_mask(lengths: ArrayLike, S: int) -> NDArray:
    """bool [B, S]: mask[b, s] = s < lengths[b]. ValueError for a length
    outside [0, S] or lengths that are not a 1-D integer array."""

class AdditiveAttention(Module):
    # The seq2seq decoder (L4.1) reads this: Bahdanau scores with the state
    # BEFORE the recurrent step, and the context feeds the step's input.
    query_from: Literal["previous"]
    d_query: int
    d_key: int
    d_attn: int
    query: Linear
    key: Linear
    v: Linear

    def __init__(self, d_query: int, d_key: int, d_attn: int, rng: Any = None) -> None:
        """The parameters above. ValueError when a size is below 1."""

    def project_keys(self, keys: Tensor) -> Tensor:
        """U k_s + b for every position: [B, S, Dk] -> [B, S, A]."""

    def scores(
        self, query: Tensor, keys: Tensor, proj: Optional[Tensor] = None
    ) -> Tensor:
        """e [B, S] before masking: v . tanh(W q + proj), proj =
        project_keys(keys) when None."""

    def forward(
        self,
        query: Tensor,
        keys: Tensor,
        mask: ArrayLike,
        proj: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """(context [B, Dk], weights [B, S]) as above. ValueError when query is
        not [B, Dq], keys not [B, S, Dk], mask not bool-like [B, S], or proj
        not [B, S, A]."""
