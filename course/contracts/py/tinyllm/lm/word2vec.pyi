# contracts/py/tinyllm/lm/word2vec.pyi (L2.3): word2vec skip-gram with negative sampling, and PPMI-SVD
# chapter: content/courses/build-an-llm/03-statistical-lm/03-word2vec-and-ppmi-svd.md
#
# Two ways to turn co-occurrence into word vectors. Skip-gram with negative
# sampling (SGNS, Mikolov et al. 2013) trains two tables by manual sparse SGD
# (no autograd): W_in (one vector v_w per center word, the embeddings) and
# W_out (one vector u_c per context word). For a (center c, context o) pair
# and K negatives k_1..k_K drawn from the noise distribution P_n:
#
#     loss = -log sigma(u_o . v_c) - sum_j log sigma(-u_{k_j} . v_c)
#
# PPMI-SVD (Levy and Goldberg 2014) factorizes the positive PMI matrix of the
# same co-occurrences directly (M11.4's ppmi, M03.5's svd).
#
# Words are ids 0..V-1. All arrays are float64 unless noted. Every random
# choice comes from ONE generator `rng` (a PCG32, M06.3; None means
# PCG32(0).substream("init")) given to SkipGramNS, in call order:
#   __init__   V * dim uniforms (W_in, C order)
#   pairs      one uniform per input token, then one below(window) per kept token
#   negatives  one uniform per draw (M07.1's AliasTable.sample)
#   step       negatives(B * n_neg)
# so a seed fixes the whole run.
from typing import Any, Sequence

from numpy.typing import ArrayLike, NDArray

NOISE_POWER: float  # 0.75: P_n(w) is proportional to count(w) ** 0.75
LR_FLOOR: float  # 1e-4: fit's learning rate never falls below lr * LR_FLOOR

def word_vocab(words: Sequence[str], min_count: int = 1) -> tuple[list[str], NDArray]:
    """(vocab, ids): the distinct words seen at least min_count times, by
    count descending, ties by the word (ascending, Python str order), and
    int64 ids of the input words in that vocabulary, words below min_count
    dropped. ValueError for min_count < 1."""

def sgns_loss(
    v: ArrayLike, u_pos: ArrayLike, u_neg: ArrayLike
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """For B pairs: v [B, d] center vectors, u_pos [B, d] context vectors,
    u_neg [B, K, d] negative vectors. Returns (loss [B], g_v [B, d],
    g_pos [B, d], g_neg [B, K, d]): each pair's loss (above) and its
    gradients with respect to v, u_pos, u_neg. With s = u_pos . v and
    t_j = u_neg_j . v:
        g_v   = (sigma(s) - 1) u_pos + sum_j sigma(t_j) u_neg_j
        g_pos = (sigma(s) - 1) v
        g_neg_j = sigma(t_j) v
    -log sigma(x) is computed as log1p(exp(-|x|)) + max(-x, 0), finite for
    every finite x (|x| = 1000 included). ValueError for mismatched shapes."""

class SkipGramNS:
    vocab: int
    dim: int
    n_neg: int
    window: int
    subsample_t: float
    W_in: NDArray  # [V, dim]: (u - 0.5) / dim, u one uniform per entry in C order
    W_out: NDArray  # [V, dim]: zeros

    def __init__(
        self,
        vocab: int,
        dim: int,
        n_neg: int = 5,
        window: int = 5,
        subsample_t: float = 1e-5,
        rng: Any = None,
    ) -> None:
        """ValueError when vocab, dim, n_neg, or window is below 1, or
        subsample_t <= 0."""

    def set_counts(self, counts: ArrayLike) -> None:
        """Unigram counts [V] (non-negative, positive sum) of the training
        corpus: they define the noise distribution and the subsampling.
        ValueError for another shape, a negative or non-finite count, or a
        zero sum."""

    def noise_probs(self) -> NDArray:
        """[V]: P_n(w) = count(w)^0.75 / sum over w' of count(w')^0.75.
        RuntimeError before set_counts."""

    def keep_probs(self) -> NDArray:
        """[V]: the probability a token of w survives subsampling,
        min(1, sqrt(subsample_t / f(w))) with f(w) = count(w) / total count;
        1 for a word of count 0. RuntimeError before set_counts."""

    def pairs(self, ids: ArrayLike) -> tuple[NDArray, NDArray]:
        """(centers, contexts), int64 [P] each, from one token sequence.
        Subsampling: token i is kept when rng.uniform() < keep_probs()[ids[i]]
        (one draw per token, in order). Then for each kept position j, in
        order: r = 1 + rng.below(window), and the pairs (kept[j], kept[k])
        for k = j - r .. j + r, k != j, inside the kept sequence, k ascending.
        ValueError for ids that are not 1-D integers in [0, V);
        RuntimeError before set_counts."""

    def negatives(self, n: int) -> NDArray:
        """int64 [n] draws from P_n by M07.1's AliasTable (built in
        set_counts), one rng.uniform() each. ValueError for n < 0."""

    def step(self, centers: ArrayLike, contexts: ArrayLike, lr: float) -> float:
        """One sparse SGD step on B pairs: negs = negatives(B * n_neg)
        reshaped [B, n_neg] (C order); losses and gradients from sgns_loss on
        the current rows W_in[centers], W_out[contexts], W_out[negs]; then
        W_in[centers] -= lr * g_v and W_out[contexts] -= lr * g_pos,
        W_out[negs] -= lr * g_neg, each row receiving the SUM of its
        gradients (a word used twice in the batch is updated twice: scatter
        with np.add.at, never W[idx] -= ...). Returns the mean loss of the
        batch before the update. ValueError for mismatched or out-of-range
        ids, an empty batch, or lr < 0."""

    def fit(
        self, ids: ArrayLike, epochs: int, batch_size: int, lr: float
    ) -> list[float]:
        """Train on one token sequence: set_counts(bincount(ids, V)), then for
        each epoch e = 0..epochs-1: pairs(ids), and step() over its n
        consecutive batches of batch_size pairs in order (the last one may
        be shorter), batch b with the learning rate decayed linearly over
        the whole run, lr * max(LR_FLOOR, 1 - (e + b / n) / epochs).
        Returns the mean step loss of each epoch. ValueError for epochs < 1,
        batch_size < 1, lr <= 0, or a sequence that yields no pairs."""

    def embeddings(self) -> NDArray:
        """A copy of W_in [V, dim]."""

def cooccurrence(ids: ArrayLike, vocab: int, window: int) -> NDArray:
    """float64 [V, V]: cooc[a, b] = the number of positions i != j with
    |i - j| <= window, ids[i] = a, ids[j] = b (symmetric, unweighted, no
    subsampling). ValueError for window < 1 or an id outside [0, vocab)."""

def ppmi_svd_embeddings(cooc: ArrayLike, dim: int) -> NDArray:
    """[V, dim]: M = M11.4's ppmi(cooc) (context smoothing 0.75), U, S, Vt =
    M03.5's svd(M), and the embeddings U[:, :dim] * sqrt(S[:dim]) (the
    symmetric split of Levy, Goldberg, and Dagan 2015). ValueError unless
    1 <= dim <= V (and as ppmi)."""

def analogy(
    emb: ArrayLike, vocab: Sequence[str], a: str, b: str, c: str, k: int = 1
) -> list[str]:
    """ "a is to b as c is to ?" by 3CosAdd: the k words whose embeddings
    have the highest cosine similarity with normalize(b) - normalize(a) +
    normalize(c) (M03.6's normalize and topk_cosine, ties to the lowest
    id), a, b, and c excluded, best first. ValueError for a word not in
    vocab or k outside 1 .. V - 3."""

def spearman(x: ArrayLike, y: ArrayLike) -> float:
    """Spearman's rank correlation: the Pearson correlation of the ranks,
    tied values sharing their average rank. ValueError for fewer than two
    values, different lengths, or a constant input."""

def word_similarity(
    emb: ArrayLike, vocab: Sequence[str], pairs: Sequence[tuple[str, str, float]]
) -> float:
    """spearman(cosine similarity of each pair's embeddings, gold score) over
    the pairs whose two words are both in vocab (the zoo's word2vec metric).
    ValueError when fewer than two pairs are covered."""
