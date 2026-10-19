"""corpus.filter (data.02): normalization and quality filters, each a Stage.

    normalize_unicode         a Stage: NFC, \\n line ends, no control characters
    length_filter(...)        keep min_chars <= len(text) <= max_chars
    lang_filter(...)          stop-word language identification, keep `langs`
    gopher_rules(...)         the Gopher quality rules (Rae et al. 2021, A.1.1)
    repetition_filter(...)    drop text whose duplicated word n-grams cover too much
    ppl_filter(lm, max_ppl)   drop text a language model finds too surprising
    build_filters(filters)    the corpus config's `filters` table as one Stage

Chapter: content/tracks/data-engineering/05-corpus-pipeline/02-extract-normalize-and-quality-filters.md.
"""

from __future__ import annotations

import dataclasses
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from corpus.stage import Doc, compose

Stage = Callable[[Iterator[Doc]], Iterator[Doc]]  # corpus.stage.Stage

# Stop words per language (ISO 639-1): the 25 most frequent function words of
# each, chosen so a sentence of ordinary prose holds several of them. Some
# words are shared ("a" en/pt, "de" fr/es/nl/pt, "in" en/nl/de); a shared
# word counts for every language that lists it.
STOPWORDS: dict[str, frozenset[str]] = {
    "de": frozenset(
        "der die das und ist nicht ein eine zu den mit sich des auf für im dem von auch es er sie wir ich war".split()
    ),
    "en": frozenset(
        "the of and to a in is it that was he she for on with as his her they at be this have from but".split()
    ),
    "es": frozenset(
        "el la los las de y que en un una es por con para del se no lo al su como pero muy yo está".split()
    ),
    "fr": frozenset(
        "le la les de des et est un une du en que qui dans pour pas sur au ce il elle nous vous je avec".split()
    ),
    "it": frozenset(
        "il la di che e un una per non sono del della con si gli le ma come anche io questo nel è ha lo".split()
    ),
    "nl": frozenset(
        "de het een en van is dat op te in met niet zijn voor er maar ook als bij nog ik je wat hij ze".split()
    ),
    "pt": frozenset(
        "o a os as de e que do da em um uma para com não no na por mais se ao dos das ele ela".split()
    ),
}
# Gopher's stop-word rule: at least two occurrences of these.
GOPHER_STOPWORDS = frozenset("the be to of and that have with".split())
BULLETS = ("•", "‣", "◦", "-", "*")
ELLIPSES = ("...", "…")
GOPHER_RULES = (
    "words",
    "mean_word_len",
    "symbol_ratio",
    "bullet_lines",
    "ellipsis_lines",
    "alpha_words",
    "stopwords",
)
_WORD = re.compile(r"[^\W\d_]+")


def _with_meta(doc: Doc, **kv: Any) -> Doc:
    return dataclasses.replace(doc, meta={**doc.meta, **kv})


# ---------------------------------------------------------------------------
# normalization


def normalize_text(text: str) -> str:
    """In order: CRLF and CR become LF; control characters (category Cc)
    other than LF and TAB, and U+FEFF, removed; Unicode NFC; trailing spaces
    and tabs of every line removed; runs of 3 or more line feeds become 2;
    leading and trailing white space removed. Removing before composing
    keeps it idempotent: a deleted BOM can leave a combining mark next to
    the letter it belongs to."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        c
        for c in text
        if c in "\n\t" or (c != "\ufeff" and unicodedata.category(c) != "Cc")
    )
    text = unicodedata.normalize("NFC", text)
    text = "\n".join(line.rstrip(" \t") for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_unicode(docs: Iterator[Doc]) -> Iterator[Doc]:
    """A Stage: normalize_text on every document; documents that become
    empty are dropped."""
    for doc in docs:
        text = normalize_text(doc.text)
        if text:
            yield doc if text == doc.text else dataclasses.replace(doc, text=text)


def length_filter(min_chars: int = 50, max_chars: int = 100_000) -> Stage:
    """Keep documents with min_chars <= len(text) <= max_chars (code points)."""
    def stage(docs: Iterator[Doc]) -> Iterator[Doc]:
        for doc in docs:
            if min_chars <= len(doc.text) <= max_chars:
                yield doc

    return stage


# ---------------------------------------------------------------------------
# language


def identify_lang(text: str) -> tuple[str, float]:
    """(language, confidence) by stop-word share.

    Words are the maximal runs of letters in text.lower(). hits[l] counts the
    words that are in STOPWORDS[l]; H is the sum of hits over languages. When
    H < 2 the answer is ("und", 0.0). Otherwise the language with the most
    hits (ties: the alphabetically first code) and confidence hits / H."""
    words = _WORD.findall(text.lower())
    hits = {lang: 0 for lang in STOPWORDS}
    for w in words:
        for lang, sw in STOPWORDS.items():
            if w in sw:
                hits[lang] += 1
    total = sum(hits.values())
    if total < 2:
        return "und", 0.0
    best = min(hits, key=lambda lang: (-hits[lang], lang))
    return best, hits[best] / total


def lang_filter(min_conf: float = 0.65, langs: Sequence[str] = ("en",)) -> Stage:
    """Keep documents whose identified language is in `langs` with confidence
    >= min_conf; record meta["lang"] and meta["lang_conf"] on every kept
    document."""
    keep = frozenset(langs)

    def stage(docs: Iterator[Doc]) -> Iterator[Doc]:
        for doc in docs:
            lang, conf = identify_lang(doc.text)
            if lang in keep and conf >= min_conf:
                yield _with_meta(doc, lang=lang, lang_conf=conf)

    return stage


# ---------------------------------------------------------------------------
# Gopher rules


def gopher_reasons(
    text: str, *, min_words: int = 50, max_words: int = 100_000,
    min_mean_word_len: float = 3.0, max_mean_word_len: float = 10.0,
    max_symbol_ratio: float = 0.1, max_bullet_lines: float = 0.9,
    max_ellipsis_lines: float = 0.3, min_alpha_words: float = 0.8, min_stopwords: int = 2,
) -> list[str]:
    """The names (from GOPHER_RULES, in that order) of the rules `text` fails.

    words = text.split(). words: min_words <= len(words) <= max_words.
    mean_word_len: the mean of len(w) is in [min, max]. symbol_ratio: the
    count of "#", and separately of "..." plus "…", divided by len(words), are
    each <= max_symbol_ratio. bullet_lines / ellipsis_lines: among non-blank
    lines, the fraction that start with a bullet (after leading white space)
    or end with an ellipsis (before trailing white space) is <= the maximum.
    alpha_words: the fraction of words holding at least one letter is >=
    min_alpha_words. stopwords: at least min_stopwords words, lowercased and
    stripped of surrounding punctuation, are in GOPHER_STOPWORDS. A text with
    no words fails words, mean_word_len, alpha_words, and stopwords."""
    words = text.split()
    n = len(words)
    failed: list[str] = []
    if not min_words <= n <= max_words:
        failed.append("words")
    mean = sum(len(w) for w in words) / n if n else 0.0
    if not n or not min_mean_word_len <= mean <= max_mean_word_len:
        failed.append("mean_word_len")
    if n:
        hashes = text.count("#") / n
        dots = (text.count("...") + text.count("…")) / n
        if hashes > max_symbol_ratio or dots > max_symbol_ratio:
            failed.append("symbol_ratio")
    lines = [x.strip() for x in text.split("\n") if x.strip()]
    if lines:
        if sum(x.startswith(BULLETS) for x in lines) / len(lines) > max_bullet_lines:
            failed.append("bullet_lines")
        if sum(x.endswith(ELLIPSES) for x in lines) / len(lines) > max_ellipsis_lines:
            failed.append("ellipsis_lines")
    if not n or sum(any(c.isalpha() for c in w) for w in words) / n < min_alpha_words:
        failed.append("alpha_words")
    stops = sum(w.lower().strip("\"'.,;:!?()[]{}") in GOPHER_STOPWORDS for w in words)
    if stops < min_stopwords:
        failed.append("stopwords")
    return failed


def gopher_rules(**thresholds: float) -> Stage:
    """Keep documents that fail no Gopher rule. `thresholds` are the keyword
    arguments of gopher_reasons; an unknown name is a ValueError (raised here,
    not when the stage runs)."""
    known = set(gopher_reasons.__kwdefaults__)
    bad = sorted(set(thresholds) - known)
    if bad:
        raise ValueError(f"unknown Gopher threshold(s) {bad}; known: {sorted(known)}")

    def stage(docs: Iterator[Doc]) -> Iterator[Doc]:
        for doc in docs:
            if not gopher_reasons(doc.text, **thresholds):
                yield doc

    return stage


# ---------------------------------------------------------------------------
# repetition and perplexity


def dup_ngram_char_frac(text: str, n: int) -> float:
    """The fraction of characters covered by duplicated word n-grams.

    words = text.split(). An n-gram (n consecutive words) is duplicated when
    it occurs at least twice. A word is covered when some occurrence of a
    duplicated n-gram contains it. The result is the sum of len(w) over
    covered words divided by the sum over all words; 0.0 when there are
    fewer than n words."""
    words = text.split()
    if len(words) < n or n < 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    covered = [False] * len(words)
    for i, g in enumerate(grams):
        if counts[g] >= 2:
            for j in range(i, i + n):
                covered[j] = True
    total = sum(len(w) for w in words)
    return sum(len(w) for w, c in zip(words, covered) if c) / total


def repetition_filter(n: int = 3, max_frac: float = 0.3) -> Stage:
    """Keep documents with dup_ngram_char_frac(text, n) <= max_frac."""
    if n < 1:
        raise ValueError("n must be at least 1")

    def stage(docs: Iterator[Doc]) -> Iterator[Doc]:
        for doc in docs:
            if dup_ngram_char_frac(doc.text, n) <= max_frac:
                yield doc

    return stage


def ppl_filter(lm: Any, max_ppl: float) -> Stage:
    """Keep documents with lm.perplexity(text) <= max_ppl; record
    meta["ppl"]. `lm` is any object with a perplexity(str) -> float method
    (in C1, the L2.1 n-gram model)."""
    def stage(docs: Iterator[Doc]) -> Iterator[Doc]:
        for doc in docs:
            ppl = float(lm.perplexity(doc.text))
            if ppl <= max_ppl:
                yield _with_meta(doc, ppl=ppl)

    return stage


def build_filters(filters: Mapping[str, Any], lm: Any = None) -> Stage:
    """The `filters` table of a corpus config (corpus-config.schema.json) as
    one Stage, with the schema defaults:

        normalize_unicode
        length_filter(min_chars, max_chars)
        lang_filter(lang_min_conf, langs)
        gopher_rules(**gopher)
        repetition_filter(repetition.n, repetition.max_frac)
        ppl_filter(lm, ppl.max_ppl)      only when filters has ppl.max_ppl

    ValueError for an unknown key, an unknown Gopher threshold, or a ppl
    table without an `lm`."""
    allowed = {"min_chars", "max_chars", "langs", "lang_min_conf", "gopher", "repetition", "ppl"}
    bad = sorted(set(filters) - allowed)
    if bad:
        raise ValueError(f"unknown filters key(s) {bad}")
    rep = dict(filters.get("repetition", {}))
    stages: list[Stage] = [
        normalize_unicode,
        length_filter(int(filters.get("min_chars", 50)), int(filters.get("max_chars", 100_000))),
        lang_filter(float(filters.get("lang_min_conf", 0.65)), tuple(filters.get("langs", ("en",)))),
        gopher_rules(**dict(filters.get("gopher", {}))),
        repetition_filter(int(rep.get("n", 3)), float(rep.get("max_frac", 0.3))),
    ]
    ppl = filters.get("ppl", {})
    if "max_ppl" in ppl:
        if lm is None:
            raise ValueError("filters.ppl.max_ppl needs a language model: pass lm")
        stages.append(ppl_filter(lm, float(ppl["max_ppl"])))
    return compose(*stages)
