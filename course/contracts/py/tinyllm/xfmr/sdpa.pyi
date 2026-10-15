# contracts/py/tinyllm/xfmr/sdpa.pyi (L5.1): scaled dot-product attention
# chapter: content/courses/build-an-llm/p05-transformer-2017/01-scaled-dot-product-attention.md
#
# Vaswani et al. (2017), equation 1, with a hand-written backward:
#
#     S = (Q K^T) * scale            [..., Tq, Tk]   scale defaults to 1 / sqrt(d)
#     S = where(mask, S, -inf)                       mask: bool, True = may attend (L5.2)
#     P = softmax(S, axis=-1)                        tinyllm.num.stable.softmax (M09.2):
#                                                    a fully masked row is zeros
#     O = P V                        [..., Tq, dv]
#
# Backward, for the upstream gradient dO (the shape of O):
#
#     dV = P^T dO
#     dP = dO V^T
#     dS = P * (dP - rowsum(dP * P))                 the softmax VJP (M08.3)
#     dQ = (dS K) * scale            dK = (dS^T Q) * scale
#
# q [..., Tq, d], k [..., Tk, d], v [..., Tk, dv]: the leading (batch, head)
# dimensions must be equal in all three. mask broadcasts to [..., Tq, Tk].
# Arrays keep their float dtype (float32 in, float32 out; float64 for
# gradient checks). L5.3's multi-head attention calls the Tensor version once
# per layer with q, k, v of shape [B, H, T, d_head]; L9.3's C kernel is
# proven against sdpa_forward.
from typing import Any, Optional

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor

def sdpa_forward(
    q: ArrayLike, k: ArrayLike, v: ArrayLike, mask: Optional[ArrayLike] = None, scale: Optional[float] = None
) -> tuple[NDArray, NDArray]:
    """(O, P) in numpy, by the forward formulas above. ValueError when the
    shapes disagree (see the header), the mask is not bool or does not
    broadcast to [..., Tq, Tk], or scale is not finite and > 0."""

def sdpa_backward(
    q: ArrayLike,
    k: ArrayLike,
    v: ArrayLike,
    p: ArrayLike,
    dout: ArrayLike,
    scale: Optional[float] = None,
    dropout_mult: Optional[ArrayLike] = None,
) -> tuple[NDArray, NDArray, NDArray]:
    """(dQ, dK, dV) by the backward formulas above, given the forward's P.
    With dropout_mult (the [..., Tq, Tk] factor keep / (1 - p) applied to P in
    the forward, so O = (P * dropout_mult) V): dV = (P * dropout_mult)^T dO and
    dP = (dO V^T) * dropout_mult; the rest is unchanged. Masked positions have
    P = 0, so they get no gradient. ValueError when the shapes disagree."""

def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Optional[ArrayLike] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
    rng: Any = None,
) -> tuple[Tensor, Tensor]:
    """(out, weights). out is one op on the autograd graph (tensor.from_op,
    L0.1) whose vjp is sdpa_backward; it requires grad when q, k, or v does.
    weights is P, a constant Tensor (no gradient flows through it), before
    dropout. An ndarray for q, k, or v is a constant.
    Attention dropout: when dropout_p > 0, draw u = rng.uniforms(P.size) (one
    uniform per weight, C order of [..., Tq, Tk], from a PCG32, M06.3), keep
    where u >= dropout_p, and out = (P * keep / (1 - dropout_p)) V; dropout_p
    = 1 gives zeros. dropout_p = 0 draws nothing (the caller passes 0 in eval
    mode). ValueError unless 0 <= dropout_p <= 1, and when dropout_p > 0 with
    no rng; the checks of sdpa_forward otherwise."""
