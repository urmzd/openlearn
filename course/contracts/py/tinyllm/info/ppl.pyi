# contracts/py/tinyllm/info/ppl.pyi (M11.2)
# chapter: content/courses/mathematics/17-information-theory/02-perplexity-bits-per-byte-nll-accumulator.md
#
# Evaluation numbers from per-token negative log-likelihoods. A model that
# gives the true next token probability q pays nll = -ln q nats. Over a text
# of n_tokens tokens covering n_bytes UTF-8 bytes, with total S nats:
#   nll_mean       = S / n_tokens                      nats per token
#   ppl            = exp(nll_mean)                     perplexity
#   bits_per_token = nll_mean / ln 2
#   bpb            = S / (n_bytes * ln 2)              bits per byte (M00.1)
# Only bits per byte compares models with different tokenizers.
from numpy.typing import ArrayLike

def perplexity(nll_sum: float, n_tokens: int) -> float:
    """exp(nll_sum / n_tokens); float('inf') when that overflows a float64
    (a mean above about 709.78 nats) or nll_sum is inf. ValueError when
    n_tokens < 1, or nll_sum is negative or NaN."""

class NLLAccumulator:
    """A running total over batches, so a corpus of 10^7 tokens is scored one
    batch at a time with the same result as one big batch. The total is
    float64 with compensated (Kahan/Neumaier) summation across add() calls,
    and each batch is summed exactly (math.fsum) after widening to float64."""

    def __init__(self) -> None:
        """An empty accumulator: no tokens, no bytes."""

    def add(self, nll: ArrayLike, mask: ArrayLike | None = None, n_bytes: int = 0) -> None:
        """Add a batch of per-token NLLs in nats, any shape. mask (same shape,
        bool or 0/1) marks the positions that count; None counts all. Masked
        positions are ignored whatever they hold, NaN and inf included.
        n_bytes is the number of UTF-8 bytes of text this batch covers.
        ValueError for a mask of another shape, a counted NLL that is
        negative or NaN, or n_bytes < 0. A counted +inf (an impossible token)
        is allowed and makes the total inf."""

    def merge(self, other: "NLLAccumulator") -> None:
        """Add another accumulator's tokens, bytes, and total into this one
        (results from several workers or shards). other is unchanged."""

    def result(self) -> dict[str, float]:
        """{"nll_sum", "n_tokens", "n_bytes", "nll_mean", "ppl",
        "bits_per_token", "bpb"} as floats. bpb is NaN when no bytes were
        added. ValueError when no token was counted."""
