# contracts/py/tinyllm/seq2seq/luong.pyi (L4.3): Luong attention and input feeding
# chapter: content/courses/build-an-llm/05-attention-origins/03-luong-attention-input-feeding.md
#
# Luong, Pham, and Manning (2015). Same reading as Bahdanau's (L4.2): scores,
# a masked softmax, a weighted average of the keys. Two differences: the
# query is the decoder state AFTER this step's recurrence (h_t, not h_{t-1}),
# and the score is one of three simpler functions, all of query dimension d
# and key dimension d:
#
#     dot      e_s = h . k_s
#     general  e_s = h . (W_a k_s)                       W_a: Linear(d, d, bias=False)
#     concat   e_s = v . tanh(W_a [h ; k_s])            W_a: Linear(2d, d, bias=False),
#                                                          v: Linear(d, 1, bias=False)
#
# then  a = softmax(e masked to -inf where mask is False),  c = sum_s a_s k_s,
# and the attentional state that the output layer reads:
#
#     h~ = tanh(W_c [c ; h])                             W_c: Linear(2d, d, bias=False)
#
# Input feeding: the decoder's next input is [embedding(y_t) ; h~_t], so the
# next step knows where the model just looked (h~_0 = 0). The seq2seq decoder
# (L4.1) does that concatenation; this module provides h~.
#
# Parameters, in registration order (= state_dict order), float32, from L0.4's
# Linear, drawn from one rng (PCG32; None means PCG32(0).substream("init")):
#   dot:      combine.weight [d, 2d]
#   general:  score_proj.weight [d, d], combine.weight [d, 2d]
#   concat:   score_proj.weight [d, 2d], v.weight [1, d], combine.weight [d, 2d]
# For concat, score_proj.weight[:, :d] multiplies h and [:, d:] multiplies k_s.
#
# Shapes as L4.2: query [B, d], keys [B, S, d], mask bool [B, S] (True =
# may be read, L4.2's length_mask), proj [B, S, d] from project_keys.
# All forwards are built from the op library (L0.2).
from typing import Any, Literal, Optional

from numpy.typing import ArrayLike

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.layers import Linear
from tinyllm.nn.module import Module

class LuongAttention(Module):
    # The seq2seq decoder (L4.1) reads this: Luong scores with the state AFTER
    # the recurrent step and feeds h~ into the next step's input.
    query_from: Literal["current"]
    d: int
    score: str  # "dot" | "general" | "concat"
    combine: Linear

    def __init__(
        self, d: int, score: Literal["dot", "general", "concat"], rng: Any = None
    ) -> None:
        """The parameters above for this score. ValueError for d < 1 or
        another score name."""

    def project_keys(self, keys: Tensor) -> Tensor:
        """What the score needs from the keys alone, computed once per source:
        dot: keys itself; general: W_a k_s; concat: score_proj.weight[:, d:]
        applied to k_s. [B, S, d] -> [B, S, d]."""

    def scores(
        self, query: Tensor, keys: Tensor, proj: Optional[Tensor] = None
    ) -> Tensor:
        """e [B, S] before masking, for the configured score."""

    def forward(
        self,
        query: Tensor,
        keys: Tensor,
        mask: ArrayLike,
        proj: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """(context [B, d], weights [B, S]), masked positions weight exactly 0,
        a fully masked row weights and context 0. ValueError for shapes as
        L4.2's forward."""

    def attentional(self, query: Tensor, context: Tensor) -> Tensor:
        """h~ = tanh(W_c [context ; query]): [B, d]. ValueError unless both
        are [B, d]."""
