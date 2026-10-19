# contracts/py/tinyllm/lm/bigram.pyi (L0.0 v0; L0.5 takes the unit over: adds BigramLogits, samples with PCG32)
# chapter: content/courses/build-an-llm/01-foundations/06-training-loop-and-the-autograd-bigram.md
# chapter: content/courses/build-an-llm/01-foundations/01-byte-bigram.md (v0)
#
# A bigram language model is a [V, V] table `weight`: row i holds the
# next-token logits after token i, and softmax(weight[i]) is the model's
# distribution over the next token. V is the vocabulary size (256 for the
# byte tokenizer, formats/tokenizer.md). Ids are integers in [0, V).
#
# L0.5 keeps every BigramLM signature of v0. What changes: `sample` draws from
# PCG32 (spec/pcg32.md, tinyllm.num.rng, M06.3) exactly as the tracer engine
# does (L10.0), so a seed gives the same ids in Python and in Rust; and
# BigramLogits trains the same table with the course's autograd (L0.1 to L0.4).
from numpy.typing import ArrayLike, NDArray

from tinyllm.autograd.tensor import Tensor
from tinyllm.nn.module import Module

class BigramLM:
    weight: NDArray  # float32 [V, V]; saved as `bigram.weight` (formats/safetensors.md)

    def __init__(self, weight: ArrayLike | None = None) -> None:
        """Empty model, or one built from a [V, V] float32 table (a loaded checkpoint).
        ValueError unless weight is 2-D, square, float32, and finite."""

    @property
    def vocab_size(self) -> int:
        """V. RuntimeError before fit_counts or a weight."""

    def fit_counts(self, ids: ArrayLike, vocab_size: int, alpha: float = 1.0) -> None:
        """Count every adjacent pair (ids[t], ids[t + 1]) into C[V, V] and store
        weight[i, j] = log((C[i, j] + alpha) / (sum_k C[i, k] + alpha * V)) as float32.
        ValueError when ids is not 1-D integer, an id is outside [0, vocab_size),
        or alpha <= 0."""

    def logits(self, ids: ArrayLike) -> NDArray:
        """float32 [T, V] = weight[ids], a NumPy row gather. Row t equals
        weight[ids[t]] exactly. ValueError for an id outside [0, V)."""

    def nll(self, ids: ArrayLike) -> float:
        """Mean negative log-likelihood in nats per predicted token:
        -(1 / (T - 1)) * sum_t log softmax(logits(ids[:-1]))[t, ids[t + 1]].
        ValueError when len(ids) < 2."""

    def sample(
        self, prefix: list[int], n: int, temperature: float, seed: int
    ) -> list[int]:
        """Continue prefix with exactly n new ids and return only the new ids.
        temperature == 0 is greedy (highest logit, ties to the lowest id) and
        draws nothing. Otherwise one generator PCG32(seed) (stream 54) per call,
        one uniform() per token, and the tracer engine's draw: weights
        w_j = exp(z_j - max z) of z = row / temperature in float64, summed in id
        order (a NaN logit has weight 0), and the next id is the first j whose
        running sum exceeds u * sum(w). Same seed, same ids as the engine.
        ValueError for an empty prefix, n < 0, or temperature < 0."""

class BigramLogits(Module):
    weight: Tensor  # float32 [vocab, vocab], requires grad, zeros at construction

    def __init__(self, vocab: int = 256) -> None:
        """One parameter, `weight` (state_dict key "weight"), all zeros: every
        next token starts equally likely, an NLL of ln(vocab). ValueError for
        vocab < 1."""

    def forward(self, ids: ArrayLike) -> Tensor:
        """Row ids[...] of weight for every id: shape ids.shape + (vocab,), the
        same values onehot(ids) @ weight gives, as an embedding lookup
        (F.embedding, L0.2) so backward adds each position's gradient into its
        row. ValueError for an id outside [0, vocab) or non-integer ids."""

    def to_lm(self) -> BigramLM:
        """A BigramLM holding a float32 copy of weight: the table the CLI saves
        as `bigram.weight` and the unchanged tracer engine serves."""
