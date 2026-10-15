# contracts/py/tinyllm/info/entropy.pyi (M11.1): entropy and divergences, in nats
# chapter: content/tracks/math/11-information-theory/01-entropy-cross-entropy-and-kl.md
#
# Distributions are float arrays along `axis` (default the last), compared
# position by position; the result has the other axes (the reduced axis is
# dropped). Inputs are not renormalized: pass distributions that sum to 1.
# Every log is the natural log, so results are in nats (divide by ln 2 for
# bits). Conventions at zero, fixed so every caller agrees:
#
#     0 * log 0 = 0          (an outcome p never produces costs nothing)
#     p * log(p / 0) = +inf  for p > 0 (q rules out what p produces)
#
# A negative probability is a caller bug: ValueError.
from numpy.typing import ArrayLike, NDArray

def entropy(p: ArrayLike, axis: int = -1) -> NDArray:
    """H(p) = -sum_i p_i log p_i, with 0 log 0 = 0. 0 <= H(p) <= log n.
    ValueError when any p_i < 0."""

def cross_entropy(p: ArrayLike, q: ArrayLike, axis: int = -1) -> NDArray:
    """H(p, q) = -sum_i p_i log q_i = H(p) + KL(p || q). Terms with p_i = 0
    contribute 0 (even when q_i = 0); a term with p_i > 0 and q_i = 0 makes
    the result +inf. ValueError when any p_i or q_i < 0."""

def kl(p: ArrayLike, q: ArrayLike, axis: int = -1) -> NDArray:
    """KL(p || q) = sum_i p_i (log p_i - log q_i), with the zero conventions
    above. >= 0 (Gibbs), 0 when p == q, not symmetric. ValueError when any
    p_i or q_i < 0."""

def kl_from_logprobs(logp: ArrayLike, logq: ArrayLike, axis: int = -1) -> NDArray:
    """KL(p || q) from log-probabilities: sum_i exp(logp_i) (logp_i - logq_i).
    A term with logp_i = -inf contributes 0 (even when logq_i = -inf); one
    with finite logp_i and logq_i = -inf makes the result +inf. Never forms
    p / q, so it stays finite for log-probabilities near -1e4 (the output of
    M09.2's log_softmax on large logits)."""

def js(p: ArrayLike, q: ArrayLike) -> NDArray:
    """Jensen-Shannon divergence along the last axis:
    JS = KL(p || m) / 2 + KL(q || m) / 2 with m = (p + q) / 2.
    Symmetric, finite even for disjoint supports, 0 <= JS <= ln 2."""

def kl_k3(logp_ref: ArrayLike, logp: ArrayLike) -> NDArray:
    """Elementwise k3 estimator of KL(pi || pi_ref) (Schulman 2020): with
    r = logp_ref - logp at a sample x drawn from pi,
    k3 = exp(r) - r - 1 >= 0, and its mean over x ~ pi is exactly
    KL(pi || pi_ref). Computed as expm1(r) - r, so it keeps its relative
    precision for small |r| (r = 1e-6 gives 5.0000017e-13, not rounding
    noise). Shapes broadcast; no reduction."""

def entropy_from_logits(z: ArrayLike, axis: int = -1) -> NDArray:
    """H(softmax(z)) without forming log(softmax(z)): with
    l = log_softmax(z) (M09.2), H = -sum_i exp(l_i) l_i, masked entries
    (z_i = -inf) contributing 0. Finite for logits of any size."""
