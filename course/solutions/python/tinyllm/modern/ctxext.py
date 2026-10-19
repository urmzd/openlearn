"""Context extension for RoPE (PI, NTK, YaRN, Llama-3) and ALiBi (L7.4).

A model trained on L positions has learned what the angles p * theta_i look
like for p < L. Past L, the slow pairs reach angles it has never seen and
attention breaks down. Every method here changes theta_i (and maybe the
attention temperature) so a longer context maps onto familiar angles,
differing in WHICH pairs they slow down and by how much.

Contract: contracts/py/tinyllm/modern/ctxext.pyi.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from tinyllm.num.series import alibi_slopes, rope_inv_freq


def _yarn_ramp(d_rot: int, base: float, original_max_pos: int, beta_fast: float, beta_slow: float) -> NDArray:
    """ramp_i in [0, 1] per pair: 0 keeps theta_i, 1 divides it by factor."""
    def corr(n_rot: float) -> float:
        # the pair index whose wavelength fits n_rot times into the context
        return d_rot * math.log(original_max_pos / (n_rot * 2 * math.pi)) / (2 * math.log(base))

    lo = max(math.floor(corr(beta_fast)), 0)
    hi = min(math.ceil(corr(beta_slow)), d_rot - 1)
    if lo == hi:
        hi += 0.001
    i = np.arange(d_rot // 2, dtype=np.float64)
    return np.clip((i - lo) / (hi - lo), 0.0, 1.0)


def rope_inv_freq_scaled(
    d_rot: int, base: float, kind: Literal["default", "linear", "ntk", "yarn", "llama3"],
    factor: float = 1.0, original_max_pos: int = 0, beta_fast: float = 32, beta_slow: float = 1,
    low_freq_factor: float = 1, high_freq_factor: float = 4,
) -> tuple[NDArray, float]:
    if kind not in ("default", "linear", "ntk", "yarn", "llama3"):
        raise ValueError(f"unknown kind {kind!r}")
    if isinstance(d_rot, bool) or int(d_rot) != d_rot or d_rot < 2 or d_rot % 2:
        raise ValueError(f"d_rot must be a positive even integer, got {d_rot!r}")
    if not base > 1:
        raise ValueError(f"base must be > 1, got {base!r}")
    d_rot = int(d_rot)
    theta = rope_inv_freq(d_rot, base)
    if kind == "default":
        return theta, 1.0
    if not factor >= 1:
        raise ValueError(f"{kind} needs factor >= 1, got {factor!r}")
    if kind == "linear":
        # position interpolation: position p behaves like p / factor
        return theta / factor, 1.0
    if kind == "ntk":
        # NTK-aware: a larger base; the fastest pair keeps its frequency and
        # the slowest (i = r/2 - 1) is divided by exactly factor.
        return rope_inv_freq(d_rot, base * factor ** (d_rot / (d_rot - 2))), 1.0
    if isinstance(original_max_pos, bool) or int(original_max_pos) != original_max_pos or original_max_pos < 1:
        raise ValueError(f"{kind} needs original_max_pos >= 1, got {original_max_pos!r}")
    if kind == "yarn":
        ramp = _yarn_ramp(d_rot, base, int(original_max_pos), beta_fast, beta_slow)
        keep = 1.0 - ramp  # HF's inv_freq_extrapolation_factor
        inv = (theta / factor) * (1.0 - keep) + theta * keep
        scale = 1.0 if factor <= 1 else 0.1 * math.log(factor) + 1.0
        return inv, scale
    # llama3
    if not high_freq_factor > low_freq_factor > 0:
        raise ValueError(
            f"llama3 needs high_freq_factor > low_freq_factor > 0, got {high_freq_factor}, {low_freq_factor}"
        )
    L0 = float(original_max_pos)
    wavelen = 2 * math.pi / theta
    inv = np.where(wavelen > L0 / low_freq_factor, theta / factor, theta)
    smooth = (L0 / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * inv / factor + smooth * inv
    medium = ~(wavelen < L0 / high_freq_factor) & ~(wavelen > L0 / low_freq_factor)
    return np.where(medium, smoothed, inv), 1.0


def alibi_bias(n_heads: int, Tq: int, Tk: int) -> NDArray:
    if min(n_heads, Tq, Tk) < 1 or Tq > Tk:
        raise ValueError(f"need n_heads, Tq, Tk >= 1 and Tq <= Tk, got {n_heads}, {Tq}, {Tk}")
    slopes = alibi_slopes(n_heads)
    i = np.arange(Tq)[:, None] + (Tk - Tq)  # absolute index of each query
    j = np.arange(Tk)[None, :]
    dist = (i - j).astype(np.float64)
    return (-slopes[:, None, None] * dist[None]).astype(np.float32)
