# contracts/py/tinyllm/lm/ngram.pyi (L2.1)
# chapter: content/courses/build-an-llm/p02-statistical-lm/01-ngram-kneser-ney.md
#
# An n-gram language model over integer token ids 0 .. V-1 with interpolated
# Kneser-Ney smoothing, modified (three discounts per order, Chen and Goodman
# 1998) or with one fixed discount. Every probability is float64; every log
# is natural (nats).
#
# Sequences and the start symbol. Each training sequence is scored on its
# own: a start symbol <s> (not a token, never predicted) is put in front of
# it, so the first token is predicted from the context (<s>). A context
# passed to prob/logprobs is the list of tokens before the predicted one in
# the same sequence: when it has fewer than n - 1 tokens, it is the start of
# a sequence and <s> is put in front of it; otherwise only its last n - 1
# tokens are used and no <s>.
#
# Counts. For k = 1 .. n, c_k(g) is the number of times the k-gram g (a run
# of k symbols of a padded sequence that ends at a token, so <s> can only
# be its first symbol) occurs. The
# count a k-gram enters the model with, its adjusted count a_k(g), is
#   c_k(g)                       when k == n, or g starts with <s>,
#   N1+(. g) = #{v : c_{k+1}(v g) > 0}   otherwise (the number of distinct
#                                   symbols, <s> included, seen before g).
#
# Probabilities, for a history h of k - 1 symbols (h' = h without its first
# symbol) and a token w, from p_0(w) = 1 / V upward:
#   if no k-gram starts with h:  p_k(w | h) = p_{k-1}(w | h')
#   else, with T = sum over v of a_k(h v) and N_j(h) = #{v : a_k(h v) == j}:
#   p_k(w | h) = max(a_k(h w) - D_k(a_k(h w)), 0) / T + gamma_k(h) p_{k-1}(w | h')
#   gamma_k(h) = (D_k1 N_1(h) + D_k2 N_2(h) + D_k3 (N_3(h) + N_4(h) + ...)) / T
# with D_k(0) = 0, D_k(1) = D_k1, D_k(2) = D_k2, D_k(c >= 3) = D_k3.
# p(w | context) is p_K at the longest history the context gives
# (K = min(n, len(padded context) + 1)). It sums to 1 over the V tokens.
#
# Discounts. With n_j = the number of k-grams with a_k == j (all histories
# of order k), Y = n_1 / (n_1 + 2 n_2) (M07.2's ney_discount) and
#   D_k1 = 1 - 2 Y n_2 / n_1,  D_k2 = 2 - 3 Y n_3 / n_2,  D_k3 = 3 - 4 Y n_4 / n_3.
# When any of n_1 .. n_4 is 0, or some D_kj falls outside (0, j], order k
# uses the fallback (0.5, 1.0, 1.5) (KenLM's --discount_fallback). With a
# fixed discount d, D_k1 = D_k2 = D_k3 = d at every order.
from typing import Iterable, Literal, Sequence, Union

from numpy.typing import NDArray

FALLBACK_DISCOUNTS: tuple[float, float, float]  # (0.5, 1.0, 1.5)

class NGramLM:
    n: int  # the order
    vocab_size: int  # V; 0 before fit when not given

    def __init__(
        self,
        n: int,
        discount: Union[float, Literal["modified"]] = "modified",
        vocab_size: int | None = None,
    ) -> None:
        """An unfitted model of order n >= 1. discount is "modified" or a
        fixed d with 0 < d <= 1. vocab_size fixes V; None takes
        V = 1 + the largest id seen by fit. ValueError for n < 1, a bad
        discount, or vocab_size < 1."""

    def fit(self, sequences: Iterable[Sequence[int]]) -> None:
        """Count every sequence (each padded with <s>) and estimate the
        discounts; replaces anything fitted before. Empty sequences are
        skipped. ValueError when an id is negative or >= vocab_size, or when
        no sequence holds a token."""

    def discounts(self, order: int) -> tuple[float, float, float]:
        """(D_k1, D_k2, D_k3) of order k = order, 1 <= order <= n.
        ValueError before fit or for another order."""

    def prob(self, context: Sequence[int], token: int) -> float:
        """p(token | context) under the rules above, in (0, 1]. ValueError
        before fit or for an id outside 0 .. V-1."""

    def logprobs(self, context: Sequence[int]) -> NDArray:
        """float64 [V]: log p(w | context) for every token w (log of the
        same numbers prob gives); logsumexp is 0."""

    def nll(self, ids: Sequence[int]) -> NDArray:
        """float64 [len(ids)]: -log p(ids[t] | ids[:t]) for each token of one
        sequence, the first predicted from <s>."""

    def perplexity(self, ids: Sequence[int]) -> float:
        """exp(mean nll) over one sequence, summed with M11.2's
        NLLAccumulator. ValueError for an empty sequence."""

    def save(self, path: str) -> None:
        """Write the model to one safetensors file (formats/safetensors.md)
        with L0.6's save_safetensors: for each order k, the I32 tensors
        "order{k}.grams" [N_k, k] (every k-gram with c_k > 0, <s> written as
        -1, rows in ascending lexicographic order) and "order{k}.counts"
        [N_k] (their raw counts c_k), and the metadata
        {"tl_arch": "ngram", "n": str(n), "discount": "modified" or repr(d),
        "vocab_size": str(V)}. ValueError before fit."""

    @classmethod
    def load(cls, path: str) -> "NGramLM":
        """The model save wrote: the same order, discount mode, V, counts,
        discounts, and probabilities. save(load(p)) writes the same bytes."""
