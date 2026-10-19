# contracts/py/tinyllm/modern/ctxext.pyi (L7.4): context extension for RoPE, and ALiBi
# chapter: content/courses/build-an-llm/p07-modern-block/04-context-extension.md
#
# A model trained on L positions has never seen the angles p * inv_freq[i]
# for p >= L on its slow pairs. Context extension rescales inv_freq so a
# longer context L' = factor * L maps onto angles the model knows:
#
#   kind       inv_freq'[i], with theta_i = base^(-2i / r)              attention_scaling
#   default    theta_i                                                   1
#   linear     theta_i / factor              (position interpolation)    1
#   ntk        theta_i computed with base' = base * factor^(r / (r - 2)) 1
#              (NTK-aware: the fastest pair keeps its frequency, the slowest is
#              divided by exactly factor)
#   yarn       (1 - g_i) theta_i / factor + g_i theta_i, g_i = 1 - ramp_i  0.1 ln(factor) + 1
#              ramp_i = clamp((i - lo) / (hi - lo), 0, 1),                 (1 when factor <= 1)
#              lo = floor(c(beta_fast)), hi = ceil(c(beta_slow)),
#              clamped to [0, r - 1], hi = lo + 0.001 when equal,
#              c(n) = r ln(original_max_pos / (2 pi n)) / (2 ln base):
#              pairs that turn more than beta_fast times over the original
#              context keep theta_i, fewer than beta_slow times are divided by
#              factor, and the ones between blend linearly
#   llama3     wavelength w_i = 2 pi / theta_i; with L0 = original_max_pos   1
#              w_i < L0 / high_freq_factor: theta_i
#              w_i > L0 / low_freq_factor:  theta_i / factor
#              otherwise: (1 - s) theta_i / factor + s theta_i,
#              s = (L0 / w_i - low_freq_factor) / (high_freq_factor - low_freq_factor)
#
# These are Hugging Face's ROPE_INIT_FUNCTIONS (transformers 5.19.0) for
# "linear", "yarn", and "llama3"; "ntk" is the static NTK-aware rule, which
# HF's "dynamic" applies with an effective factor that grows with the
# sequence length. The attention_scaling multiplies cos and sin (L7.3), so
# YaRN's attention logits grow by attention_scaling^2.
#
# ALiBi (Press et al. 2022; optional part): no rotation at all,
# a bias added to the scores that falls linearly with distance, one slope
# per head (M00.3's alibi_slopes).
from typing import Literal

from numpy.typing import NDArray

def rope_inv_freq_scaled(
    d_rot: int,
    base: float,
    kind: Literal["default", "linear", "ntk", "yarn", "llama3"],
    factor: float = 1.0,
    original_max_pos: int = 0,
    beta_fast: float = 32,
    beta_slow: float = 1,
    low_freq_factor: float = 1,
    high_freq_factor: float = 4,
) -> tuple[NDArray, float]:
    """(inv_freq float64 [d_rot / 2], attention_scaling) for the kind above,
    computed in float64. "default" ignores factor; "linear" and "ntk" need
    factor >= 1; "yarn" and "llama3" need factor >= 1 and
    original_max_pos >= 1; "llama3" needs high_freq_factor > low_freq_factor > 0.
    ValueError for an unknown kind, d_rot not a positive even integer,
    base <= 1, or a parameter outside those ranges."""

def alibi_bias(n_heads: int, Tq: int, Tk: int) -> NDArray:
    """float32 [n_heads, Tq, Tk]: bias[h, i, j] = -slope_h * (Tk - Tq + i - j),
    the distance from query i (the last Tq of Tk positions) back to key j,
    times the head's slope (M00.3 alibi_slopes). Keys after the query get a
    positive bias; the causal mask removes them. Each row differs from
    BLOOM's slope_h * j by a constant, so their softmaxes are equal.
    ValueError when n_heads, Tq, or Tk < 1, or Tq > Tk."""
