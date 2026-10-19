"""Sampling and logit processors (L8.1), exactly as spec/sampling.md.

The model hands back one float32 logit per vocabulary id. Turning that into
one token is a fixed pipeline: widen to float64, penalize repeats, stop here
if greedy, divide by the temperature, keep the top k, keep the nucleus whose
mass reaches p, drop what falls below min_p of the most likely, normalize,
draw one uniform, walk the CDF. Every sum is a left-to-right loop in
ascending id order and every exp is math.exp, so the Rust engine (L10.1)
reproduces the same token and the same logprob from the same logits and
seed.

Contract: contracts/py/tinyllm/infer/sample.pyi.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyllm.info.entropy import entropy
from tinyllm.num.rng import PCG32
from tinyllm.prob.sampling import UniformSource, sample_categorical


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: Optional[int] = None
    max_tokens: int = 128
    stop: list[str] = field(default_factory=list)
    logprobs: int = 0

    def validate(self) -> None:
        def finite(name: str) -> float:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{name} must be a finite number, got {v!r}")
            return float(v)

        if finite("temperature") < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError(f"top_k must be an int >= 0, got {self.top_k!r}")
        if not 0.0 < finite("top_p") <= 1.0:
            raise ValueError(f"top_p must lie in (0, 1], got {self.top_p}")
        if not 0.0 <= finite("min_p") <= 1.0:
            raise ValueError(f"min_p must lie in [0, 1], got {self.min_p}")
        if finite("repetition_penalty") <= 0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")
        finite("presence_penalty")
        finite("frequency_penalty")
        if self.seed is not None and (not isinstance(self.seed, int) or self.seed < 0):
            raise ValueError(f"seed must be None or an int >= 0, got {self.seed!r}")
        if not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise ValueError(f"max_tokens must be an int >= 1, got {self.max_tokens!r}")
        if not isinstance(self.logprobs, int) or self.logprobs < 0:
            raise ValueError(f"logprobs must be an int >= 0, got {self.logprobs!r}")


def request_rng(seed: int) -> PCG32:
    return PCG32(seed).substream("sample")


def _seq_sum(xs: Sequence[float]) -> float:
    """Left-to-right float64 sum (numpy's cumsum is a plain running loop)."""
    if len(xs) == 0:
        return 0.0
    return float(np.cumsum(np.asarray(xs, dtype=np.float64))[-1])


def _softmax_over(l: NDArray, keep: Sequence[int]) -> list[float]:
    """Step 9 over the ids in `keep` (ascending): q_i = exp(l_i - M) / Z."""
    ids = sorted(keep)
    m = max(float(l[i]) for i in ids)
    e = [math.exp(float(l[i]) - m) for i in ids]
    z = _seq_sum(e)
    return [v / z for v in e]


def apply_penalties(
    logits: ArrayLike, p: SamplingParams, history: Sequence[int] = (), prompt: Sequence[int] = ()
) -> NDArray:
    l = np.array(logits, dtype=np.float64)  # step 1: a float64 copy
    if l.ndim != 1 or l.size == 0:
        raise ValueError(f"logits must be a non-empty 1-D array, got shape {l.shape}")
    if np.isnan(l).any() or np.isposinf(l).any():
        raise ValueError("logits must not contain NaN or +inf")
    if np.isneginf(l).all():
        raise ValueError("every logit is -inf: nothing can be sampled")
    V = l.size
    hist, pr = [int(t) for t in history], [int(t) for t in prompt]
    for t in hist + pr:
        if not 0 <= t < V:
            raise ValueError(f"token id {t} outside [0, {V})")
    r = p.repetition_penalty
    if r != 1.0:
        for i in sorted(set(hist) | set(pr)):  # step 2: once per distinct id
            l[i] = l[i] / r if l[i] > 0 else l[i] * r
    if p.presence_penalty != 0.0 or p.frequency_penalty != 0.0:
        for i, c in sorted(Counter(hist).items()):  # step 3: generated ids only
            l[i] = l[i] - p.frequency_penalty * c - p.presence_penalty
    return l


def token_logprobs(
    logits: ArrayLike, p: SamplingParams, history: Sequence[int] = (), prompt: Sequence[int] = ()
) -> NDArray:
    l = apply_penalties(logits, p, history, prompt)
    m = float(np.max(l))
    z = _seq_sum([math.exp(float(v) - m) for v in l])
    return l - m - math.log(z)


def _kept(l: NDArray, p: SamplingParams) -> list[int]:
    """Steps 5 to 8 on penalized logits: the kept ids, ascending."""
    t = l / p.temperature  # step 5
    # step 6: order by (logit desc, id asc); masked ids never enter
    order = [i for i in sorted(range(t.size), key=lambda i: (-t[i], i)) if t[i] != -np.inf]
    if 0 < p.top_k < len(order):
        order = order[: p.top_k]
    if p.top_p < 1.0:  # step 7: the nucleus, keeping the token that crosses p
        q = dict(zip(sorted(order), _softmax_over(t, order)))
        s, cut = 0.0, len(order)
        for n, i in enumerate(order):
            s += q[i]
            if s >= p.top_p:
                cut = n + 1
                break
        order = order[:cut]
    if p.min_p > 0.0:  # step 8: relative to the most likely kept token
        q = dict(zip(sorted(order), _softmax_over(t, order)))
        qmax = max(q.values())
        order = [i for i in order if q[i] >= p.min_p * qmax]
    return sorted(order)


def _greedy(l: NDArray) -> int:
    return int(np.argmax(l))  # first maximum: ties to the lowest id


def process_logits(
    logits: ArrayLike, p: SamplingParams, history: Sequence[int] = (), prompt: Sequence[int] = ()
) -> NDArray:
    p.validate()
    l = apply_penalties(logits, p, history, prompt)
    if p.temperature == 0.0:
        return l
    out = np.full(l.shape, -np.inf)
    keep = _kept(l, p)
    out[keep] = l[keep] / p.temperature
    return out


def sampling_distribution(
    logits: ArrayLike, p: SamplingParams, history: Sequence[int] = (), prompt: Sequence[int] = ()
) -> NDArray:
    p.validate()
    l = apply_penalties(logits, p, history, prompt)
    q = np.zeros(l.size)
    if p.temperature == 0.0:
        q[_greedy(l)] = 1.0
        return q
    keep = _kept(l, p)
    q[keep] = _softmax_over(l / p.temperature, keep)
    return q


def sample(
    logits: ArrayLike,
    p: SamplingParams,
    history: Sequence[int],
    rng: UniformSource,
    prompt: Sequence[int] = (),
) -> tuple[int, float]:
    lp = token_logprobs(logits, p, history, prompt)
    if p.temperature == 0.0:
        p.validate()
        tok = _greedy(apply_penalties(logits, p, history, prompt))
        return tok, float(lp[tok])
    q = sampling_distribution(logits, p, history, prompt)
    tok = sample_categorical(q, rng.uniform())  # step 10 and 11: one draw
    return tok, float(lp[tok])


def sampled_entropy(
    logits: ArrayLike, p: SamplingParams, history: Sequence[int] = (), prompt: Sequence[int] = ()
) -> float:
    return float(entropy(sampling_distribution(logits, p, history, prompt)))
