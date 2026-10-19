# contracts/py/tinyllm/infer/spec.pyi (L8.6)
# chapter: content/courses/build-an-llm/09-inference/06-speculative-decoding.md
#
# Speculative decoding: a cheap draft proposes k tokens, the target model
# scores all of them in ONE forward pass, and a verification rule keeps the
# longest acceptable prefix plus one token of the target's own. With greedy
# decoding the rule is "keep while the draft equals the target's argmax", so
# the output is token for token the target's greedy output. With sampling the
# rule is M07.6's rejection step (accept x ~ q with probability
# min(1, p(x) / q(x)), else draw from the residual), so every emitted token
# is distributed exactly as the target's sampler (L8.1) would draw it.
#
# Randomness: one request generator (L8.1's request_rng(seed), seed 0 when
# p.seed is None) serves the draft and the verifier, in program order. For
# each verified draft position the verifier draws exactly two uniforms,
# u_accept then u_resample (both always, so the count does not depend on the
# outcome); the bonus token after a fully accepted draft is L8.1's sample
# (one uniform). Greedy (temperature 0) draws nothing. The Rust port (L10.8)
# follows the same order and is held to the same outputs on shared logits.
from typing import Any, Optional, Protocol, Sequence

from numpy.typing import ArrayLike, NDArray

from tinyllm.infer.generate import Generation
from tinyllm.infer.sample import SamplingParams
from tinyllm.lm.ngram import NGramLM
from tinyllm.prob.sampling import UniformSource

class DraftModel(Protocol):
    def propose(
        self, ctx: Sequence[int], k: int, rng: Optional[UniformSource]
    ) -> tuple[list[int], Optional[NDArray]]:
        """Up to k token ids that follow ctx, and the distributions they
        were drawn from: float64 [len(ids), V] with row i the draft's
        distribution for ids[i], or None for a deterministic draft (each
        row then counts as one-hot on its id). rng None means draft
        greedily (and return None). Fewer than k ids (even none) is
        allowed."""

class NGramDraft:
    """Drafts from an L2.1 n-gram model: each token from lm.logprobs of the
    context so far (the drafted ids appended). Greedy: the largest logprob,
    ties to the lowest id. Sampling: L8.1's sample on the logprobs with
    default SamplingParams (temperature 1), and the row is
    sampling_distribution of the same."""

    lm: NGramLM
    def __init__(self, lm: NGramLM) -> None: ...
    def propose(
        self, ctx: Sequence[int], k: int, rng: Optional[UniformSource]
    ) -> tuple[list[int], Optional[NDArray]]: ...

class PromptLookupDraft:
    """Drafts by copying from the context itself (prompt-lookup decoding):
    for n = max_ngram down to min_ngram, take the last n ids of ctx and find
    their most recent earlier occurrence (a start i < len(ctx) - n with
    ctx[i:i+n] equal to them); the draft is ctx[i+n : i+n+k], which may run
    into the suffix itself. The first n that finds a non-empty continuation
    wins; none gives []. Always deterministic (probs None); rng is
    ignored."""

    max_ngram: int
    min_ngram: int
    def __init__(self, max_ngram: int = 3, min_ngram: int = 1) -> None:
        """ValueError unless 1 <= min_ngram <= max_ngram."""
    def propose(
        self, ctx: Sequence[int], k: int, rng: Optional[UniformSource]
    ) -> tuple[list[int], Optional[NDArray]]: ...

class ModelDraft:
    """Drafts with a smaller CausalLM (L8.2's protocol) and its own KVCache.
    Between calls it keeps the ids its cache holds; a new ctx is synced by
    truncating the cache to the common prefix and feeding the rest. Greedy:
    the largest logit, ties to the lowest id. Sampling: L8.1's sample with
    SamplingParams(temperature=temperature); the row is
    sampling_distribution of the same."""

    model: Any
    temperature: float
    def __init__(self, model: Any, temperature: float = 1.0) -> None:
        """ValueError for temperature <= 0."""
    def propose(
        self, ctx: Sequence[int], k: int, rng: Optional[UniformSource]
    ) -> tuple[list[int], Optional[NDArray]]:
        """ValueError for an empty ctx."""

def verify_draft(
    target_logits: ArrayLike,
    draft_ids: Sequence[int],
    draft_probs: Optional[ArrayLike],
    p: SamplingParams,
    history: Sequence[int],
    rng: UniformSource,
    prompt: Sequence[int] = (),
) -> tuple[list[int], int]:
    """One verification step. target_logits is [m + 1, V]: row i is the
    target's logits after history + draft_ids[:i] (row m: after the whole
    draft). Returns (emitted, n_accepted) with emitted = draft_ids[:n] plus
    one more token, n = n_accepted in [0, m]; len(emitted) = n + 1.

    temperature == 0: position i is accepted while draft_ids[i] equals
    L8.1's greedy id of row i (penalties see history + the ids accepted so
    far); the first mismatch emits that greedy id instead. After m
    acceptances the extra token is the greedy id of row m.

    temperature > 0: for each position, P = L8.1's sampling_distribution of
    row i (same history rule), Q = draft_probs[i] (or one-hot on
    draft_ids[i]); u_accept, u_resample = two uniforms; then M07.6's
    speculative_step(P, Q, x, u_accept, u_resample) either accepts x or
    emits the residual draw and stops. After m acceptances the extra token
    is L8.1's sample of row m (one uniform).

    ValueError for target_logits that are not [len(draft_ids) + 1, V], or
    draft_probs not [len(draft_ids), V]."""

def speculative_generate(
    target: Any,
    draft: DraftModel,
    tok: Any,
    prompt: str,
    p: SamplingParams,
    k: int = 4,
    kv_dtype: Any = ...,
    eos_ids: Sequence[int] = (),
) -> Generation:
    """L8.2's generate, k + 1 tokens per target pass at best. Each round:
    the draft proposes up to k ids after prompt + generated (fewer near
    p.max_tokens or the cache's end); the target runs ONE forward over the
    tokens not yet in its KVCache (the previous round's extra token, or the
    prompt in the first round) followed by the draft; verify_draft keeps
    n_accepted drafts and adds one token; the cache is truncated to the
    accepted length (rejected drafts' keys and values are dropped). The
    target cache holds min(model max_len, len(prompt) + p.max_tokens + k)
    positions in kv_dtype (default float32).

    Stops at p.max_tokens, at an id in eos_ids (not part of ids), at a stop
    string (text is cut before it, finish_reason "stop"), or when the cache
    is full. logprobs[j] is L8.1's token_logprobs of the row that produced
    ids[j]. stats: prompt_tokens, completion_tokens, drafted (ids proposed),
    accepted (drafts kept), acceptance_rate (accepted / drafted, 0 when
    nothing was drafted), target_calls (forward passes), tokens_per_s.
    ValueError for k < 0, an empty prompt encoding, or a prompt longer than
    the model's max_len."""
