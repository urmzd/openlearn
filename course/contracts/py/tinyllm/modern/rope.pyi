# contracts/py/tinyllm/modern/rope.pyi (L7.3): rotary position embeddings
# chapter: content/courses/build-an-llm/08-modern-block/03-rope.md
#
# RoPE (Su et al. 2021) rotates pair i of a query or key vector at position
# p by the angle p * inv_freq[i] (inv_freq from M00.3's ladder, or from
# L7.4's scaled ladders). A rotation preserves length, and the dot product
# of a query rotated by angle a with a key rotated by angle b depends only on
# a - b: attention scores see relative position only.
#
#   r           rotary_dim: the first r entries of the head rotate, the other
#               dh - r pass through unchanged (partial rotary: GPT-NeoX,
#               Phi, gpt-oss uses full); r is even, r <= dh; None means dh
#   inv_freq    float [r / 2], one frequency per pair
#   cos, sin    float32 [..., r / 2]: ONE value per pair (not repeated to r)
#
# Which entries form pair i (the layout):
#   "half"         (x[i], x[i + r/2])     HF Llama / Mistral / Qwen / SmolLM2 / gpt-oss
#   "interleaved"  (x[2i], x[2i + 1])     the RoPE paper, Meta's Llama code, M00.2
# Rotating pair (a, b) by t gives (a cos t - b sin t, a sin t + b cos t).
# The two layouts hold the same rotation on a permuted vector: Meta's
# checkpoints become HF's by permuting the rows of q_proj and k_proj.
from dataclasses import dataclass
from typing import Literal, Optional

from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor

def rope_cos_sin(
    positions: ArrayLike, inv_freq: ArrayLike, attention_scaling: float = 1.0
) -> tuple[NDArray, NDArray]:
    """cos and sin of the angles positions[..., None] * inv_freq, each
    float32 of shape positions.shape + (len(inv_freq),), multiplied by
    attention_scaling (YaRN's factor, L7.4; 1 otherwise). The angle is
    computed as HF does: float32(position) * float32(inv_freq), one float32
    product, then cos and sin of that float32 angle. positions are integers
    >= 0 (any shape). ValueError for negative or non-integer positions or an
    inv_freq that is not 1-D."""

def apply_rope(
    x: Tensor,
    cos: ArrayLike,
    sin: ArrayLike,
    layout: Literal["half", "interleaved"] = "half",
    rotary_dim: Optional[int] = None,
) -> Tensor:
    """Rotate the pairs of the first r entries of x's last axis; entries r
    and beyond pass through. x is [..., dh] (typically [B, H, T, dh]); cos
    and sin broadcast against x.shape[:-1] + (r / 2,) (a [T, r/2] table
    serves [B, H, T, dh]; per-row positions pass cos[:, None] for
    [B, 1, T, r/2]). Built from the op library (indexing, products,
    concatenation), so backward reaches x (RoPE is linear in x: the
    gradient is the inverse rotation of the upstream gradient).
    ValueError for an unknown layout, r odd, r < 2, r > dh, or a cos/sin
    whose last axis is not r / 2."""

@dataclass
class RopeSpec:
    """How an attention layer rotates its queries and keys (L7.5, L7.6)."""

    inv_freq: NDArray  # [rotary_dim / 2]
    attention_scaling: float  # passed to rope_cos_sin
    layout: str  # "half" or "interleaved"
    rotary_dim: int  # even, <= d_head
