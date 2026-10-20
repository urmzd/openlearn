# contracts/py/tinyllm/infer/sample.pyi (L8.1)
# chapter: content/courses/ai-engineering/21-inference/01-sampling-and-logit-processors.md
#
# The sampler of spec/sampling.md, step for step, in IEEE float64: widen,
# repetition penalty (HF), presence and frequency penalties (OpenAI),
# greedy when temperature == 0, temperature, top-k, top-p, min-p, softmax,
# exactly one uniform per sampled token, inverse CDF. Given the same logits
# and the same generator, the Rust engine (L10.1) returns the same id and
# the same logprob, so every rule below is part of that parity contract:
#
#   * every sum runs over ids in ascending order with a plain left-to-right
#     loop (not Python's sum(), which compensates since 3.12, and not
#     numpy's pairwise np.sum);
#   * exp and log are the platform's float64 functions (math.exp, math.log);
#   * orderings are by (logit descending, id ascending), so ties go to the
#     lowest id;
#   * a logit of -inf (a mask from L8.7) is a token that cannot be sampled;
#     it never enters the kept set, so the inverse-CDF fallback of step 11
#     (rounding left the running sum below u) is the largest kept id, which
#     always has q > 0.
#
# `prompt` holds the request's prompt ids and `history` the ids generated so
# far (`out` in the spec): the repetition penalty looks at both, presence and
# frequency at `history` only. The generator is `rng` with a uniform()
# method (PCG32 of spec/pcg32.md); request_rng(seed) builds a request's.
from dataclasses import dataclass, field
from typing import Optional, Sequence

from numpy.typing import ArrayLike, NDArray

from tinyllm.num.rng import PCG32
from tinyllm.prob.sampling import UniformSource

@dataclass
class SamplingParams:
    """OpenAI request fields (openapi/openai-subset.v1.yaml); the defaults
    turn every processor off. max_tokens, stop, and logprobs are read by
    generate (L8.2), not by the sampler."""

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
        """ValueError unless temperature >= 0, top_k >= 0, 0 < top_p <= 1,
        0 <= min_p <= 1, repetition_penalty > 0, the two additive penalties
        are finite, seed is None or an int >= 0, max_tokens >= 1, and
        logprobs >= 0 (all numbers finite)."""

def request_rng(seed: int) -> PCG32:
    """The request's generator: stream(seed, "sample") of spec/pcg32.md, that
    is PCG32(seed).substream("sample") (M06.3). Its first uniform for seed 0
    is 0.80209... (the spec's worked example)."""

def apply_penalties(
    logits: ArrayLike, p: SamplingParams, history: Sequence[int] = (), prompt: Sequence[int] = ()
) -> NDArray:
    """Steps 1 to 3: float64 [V] copy of logits with the repetition penalty
    applied once per distinct id of prompt and history (l / r when l > 0,
    else l * r), then l - a_f * c - a_p for each id occurring c > 0 times in
    history. The input is not modified. ValueError for logits that are not
    1-D or contain NaN or +inf, all -inf logits, or an id outside [0, V)."""

def token_logprobs(
    logits: ArrayLike, p: SamplingParams, history: Sequence[int] = (), prompt: Sequence[int] = ()
) -> NDArray:
    """float64 [V] log_softmax of apply_penalties(...) (step 3: before
    temperature and filtering): l_i - M - log(Z) with M the maximum and Z
    the ascending sum of exp(l_i - M). What OpenAI reports as logprobs."""

def process_logits(
    logits: ArrayLike, p: SamplingParams, history: Sequence[int] = (), prompt: Sequence[int] = ()
) -> NDArray:
    """Steps 1 to 8: float64 [V], the tempered logits l_i / T of the kept
    ids and -inf for every id removed by top-k, top-p, or min-p (or masked).
    With temperature == 0 it returns apply_penalties(...) unchanged (greedy
    filters nothing). ValueError from validate() or apply_penalties."""

def sampling_distribution(
    logits: ArrayLike, p: SamplingParams, history: Sequence[int] = (), prompt: Sequence[int] = ()
) -> NDArray:
    """float64 [V]: the exact distribution sample() draws from. Step 9 over
    the kept set K: q_i = exp(l_i - M) / Z with M = max over K and Z the
    ascending sum over K; 0 outside K. With temperature == 0, one-hot on the
    greedy id. L8.6 uses it as the target distribution p of M07.6."""

def sample(
    logits: ArrayLike,
    p: SamplingParams,
    history: Sequence[int],
    rng: UniformSource,
    prompt: Sequence[int] = (),
) -> tuple[int, float]:
    """(id, logprob). temperature == 0: the largest penalized logit, ties to
    the lowest id, and no draw. Otherwise exactly one u = rng.uniform() and
    the inverse CDF of sampling_distribution(...) (M07.1's
    sample_categorical: ascending ids, first id with u < c). logprob is
    token_logprobs(...)[id]."""

def sampled_entropy(
    logits: ArrayLike, p: SamplingParams, history: Sequence[int] = (), prompt: Sequence[int] = ()
) -> float:
    """Entropy in nats (M11.1) of sampling_distribution(...): what generate
    logs per token. 0 for greedy; log(n) for n equally likely kept ids."""
