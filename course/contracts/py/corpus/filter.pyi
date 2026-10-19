# contracts/py/corpus/filter.pyi (data.02): normalization and quality filters, each a Stage
# chapter: content/courses/data-engineering/05-corpus-pipeline/02-extract-normalize-and-quality-filters.md
#
# Words used below:
#   words       text.split(): maximal runs of non-white-space characters
#   letters     characters c with c.isalpha()
#   lang words  maximal runs of letters in text.lower() (regex [^\W\d_]+)
#   line        a piece of text.split("\n"); blank when it is white space only
#
# The thresholds' names are the keys of the corpus config's `filters` table
# (formats/corpus-config.schema.json); its `gopher` table uses the keyword
# names of gopher_reasons.
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from corpus.stage import Doc, Stage

STOPWORDS: Mapping[str, frozenset[str]]  # ISO 639-1 code -> stop words; de en es fr it nl pt, as in the chapter
GOPHER_STOPWORDS: frozenset[str]  # the be to of and that have with
GOPHER_RULES: tuple[str, ...]  # words mean_word_len symbol_ratio bullet_lines ellipsis_lines alpha_words stopwords
BULLETS: tuple[str, ...]  # "•" "‣" "◦" "-" "*"
ELLIPSES: tuple[str, ...]  # "..." "…"

def normalize_text(text: str) -> str:
    """In order: CRLF and lone CR become LF; every character of category Cc
    other than LF and TAB, and U+FEFF, is removed; Unicode NFC; trailing
    spaces and tabs of each line are removed; runs of three or more LFs
    become two; leading and trailing white space is removed. Idempotent
    (removing before composing matters: a deleted U+FEFF can leave a
    combining mark next to the letter it belongs to)."""

def normalize_unicode(docs: Iterator[Doc]) -> Iterator[Doc]:
    """A Stage: normalize_text on each document's text; a document whose text
    becomes empty is dropped."""

def length_filter(min_chars: int = 50, max_chars: int = 100_000) -> Stage:
    """Keep documents with min_chars <= len(text) <= max_chars (code points)."""

def identify_lang(text: str) -> tuple[str, float]:
    """(code, confidence) by stop-word share. hits[l] is the number of lang
    words of text in STOPWORDS[l] (a word listed by two languages counts for
    both); H = sum of hits. H < 2 gives ("und", 0.0). Otherwise the code with
    the most hits, ties to the alphabetically first, and hits[code] / H."""

def lang_filter(min_conf: float = 0.65, langs: Sequence[str] = ("en",)) -> Stage:
    """Keep documents whose identify_lang code is in langs with confidence
    >= min_conf, with meta["lang"] and meta["lang_conf"] set."""

def gopher_reasons(
    text: str,
    *,
    min_words: int = 50,
    max_words: int = 100_000,
    min_mean_word_len: float = 3.0,
    max_mean_word_len: float = 10.0,
    max_symbol_ratio: float = 0.1,
    max_bullet_lines: float = 0.9,
    max_ellipsis_lines: float = 0.3,
    min_alpha_words: float = 0.8,
    min_stopwords: int = 2,
) -> list[str]:
    """The GOPHER_RULES text fails, in GOPHER_RULES order. With n = len(words):
    words        fails unless min_words <= n <= max_words
    mean_word_len fails unless n > 0 and the mean len(w) is in [min, max]
    symbol_ratio fails when n > 0 and text.count("#") / n, or
                 (text.count("...") + text.count("…")) / n, exceeds max_symbol_ratio
    bullet_lines fails when, among non-blank lines, the fraction whose
                 lstrip() starts with a BULLETS entry exceeds max_bullet_lines
    ellipsis_lines likewise for rstrip() ending with an ELLIPSES entry
    alpha_words  fails when n == 0 or the fraction of words holding a letter
                 is below min_alpha_words
    stopwords    fails when fewer than min_stopwords words, lowercased and
                 stripped of the characters "'.,;:!?()[]{} at both ends, are
                 in GOPHER_STOPWORDS (occurrences, not distinct words)"""

def gopher_rules(**thresholds: float) -> Stage:
    """Keep documents for which gopher_reasons(text, **thresholds) is empty.
    ValueError, raised by this call, for a name gopher_reasons does not take."""

def dup_ngram_char_frac(text: str, n: int) -> float:
    """With words w_0..w_{L-1}: an n-gram (w_i..w_{i+n-1}) is duplicated when
    it occurs at least twice; a word is covered when it lies inside some
    occurrence of a duplicated n-gram. Returns sum(len(w) for covered w) /
    sum(len(w) for all w), and 0.0 when L < n."""

def repetition_filter(n: int = 3, max_frac: float = 0.3) -> Stage:
    """Keep documents with dup_ngram_char_frac(text, n) <= max_frac.
    ValueError for n < 1."""

def ppl_filter(lm: Any, max_ppl: float) -> Stage:
    """Keep documents with lm.perplexity(text) <= max_ppl, with meta["ppl"]
    set. `lm` is any object with perplexity(str) -> float (the L2.1 model in C1)."""

def build_filters(filters: Mapping[str, Any], lm: Any = None) -> Stage:
    """The config's `filters` table as one Stage, schema defaults filled in:
    compose(normalize_unicode, length_filter(min_chars, max_chars),
    lang_filter(lang_min_conf, langs), gopher_rules(**gopher),
    repetition_filter(repetition.n, repetition.max_frac)), then
    ppl_filter(lm, ppl.max_ppl) when the table has ppl.max_ppl.
    ValueError for an unknown key or Gopher name, or ppl.max_ppl with lm None."""
