# contracts/py/tinyllm/tok/unigram.pyi (L1.4)
# chapter: content/courses/ai-engineering/07-tokenizers/04-unigram-lm.md
#
# The Unigram LM tokenizer (Kudo 2018), sentencepiece compatible: Viterbi
# ids equal Hugging Face tokenizers' and sentencepiece's on the same vocab
# (formats/tokenizer.md, Unigram). Uses M06.2 (Trie.prefixes) for the
# lattice and M07.2 (mle) to score the seed pieces from their integer
# counts; sample_encode draws from a PCG32 (M06.3; the course tests pass
# the frozen one).
from typing import Iterable, Optional, Sequence

from tinyllm.num.rng import PCG32
from tinyllm.tok.base import Tokenizer

def metaspace(text: str) -> list[str]:
    """Hugging Face Metaspace (replacement U+2581, prepend_scheme "always",
    split): U+0020 -> U+2581, prepend U+2581 unless the text starts with it,
    then split before every U+2581. "a  b" -> ["▁a", "▁", "▁b"]; "" -> []."""

class UnigramTokenizer(Tokenizer):
    pieces: list[tuple[str, float]]       # id -> (piece, log-probability)
    min_score: float                      # min over pieces; an unknown character scores min_score - 10
    def __init__(
        self,
        pieces: Sequence[tuple[str, float]],
        unk_id: int = 0,
        specials: Sequence[str] = (),
    ) -> None:
        """special_ids = the unk piece plus specials. ValueError for duplicate
        pieces, an unk_id outside pieces, or a special that is not a piece."""
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        seed_factor: int = 10,
        em_iters: int = 2,
        shrink: float = 0.75,
    ) -> "UnigramTokenizer":
        """Seed: every character plus the seed_factor * vocab_size substrings
        (2 to 16 code points, inside one Metaspace word) with the highest
        frequency x length (ties by string), scored log mle(freq) (M07.2). Then
        repeat: em_iters EM steps (pieces expected fewer than 0.5 times are
        dropped, characters never: they keep a count of at least 0.5); stop
        when at most vocab_size - 1 pieces remain; else keep the characters and
        the max(vocab_size - 1, int(n * shrink)) - #characters other pieces
        with the largest removal loss (Viterbi frequency x (log p(piece) - log p
        of the best segmentation of its text without it), ties by string).
        Result: id 0 = ("<unk>", 0.0), then pieces by score descending, ties
        by string; vocab_size at most the one asked. ValueError when the
        characters alone do not fit, em_iters < 1, or shrink is not in (0, 1)."""
    def viterbi(self, word: str) -> list[int]:
        """The most probable segmentation of one Metaspace word; ties go to the
        path whose last piece starts earliest; adjacent unknowns fuse."""
    def encode(self, text: str, add_special: bool = False) -> list[int]:
        """Specials in the raw text (leftmost longest, <unk> included) are one
        id each; the rest is metaspace + viterbi per word."""
    def sample_encode(self, text: str, alpha: float, rng: PCG32) -> list[int]:
        """A segmentation drawn with probability proportional to P(s)^alpha,
        word by word: forward sums of alpha * score, then backward from the
        end, one rng.uniform() per chosen piece, candidates (pieces ending
        here) by start position ascending, inverse CDF. alpha = 0 is uniform
        over segmentations. ValueError for alpha < 0."""
    def log_likelihood(self, texts: Iterable[str]) -> float:
        """sum over Metaspace words of log sum_s P(s) (natural log)."""
    def em_step(self, texts: Iterable[str]) -> "UnigramTokenizer":
        """One EM iteration on this piece set: expected counts (forward-
        backward), then log(count / total); pieces with expected count 0 are
        dropped. The log_likelihood of texts never decreases (up to rounding)."""
    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        """Pieces joined, U+2581 -> U+0020, the first token's leading space dropped."""
    def token_to_id(self, s: str) -> Optional[int]: ...
    def id_to_token(self, i: int) -> str: ...
    def save(self, dir: str) -> None:
        """dir/tokenizer.json: Unigram model, Metaspace pre_tokenizer and decoder."""
    @classmethod
    def load(cls, dir: str) -> "UnigramTokenizer": ...
    @classmethod
    def from_hf_json(cls, tokenizer_json: str) -> "UnigramTokenizer":
        """A Unigram tokenizer.json with normalizer null and the Metaspace above."""
