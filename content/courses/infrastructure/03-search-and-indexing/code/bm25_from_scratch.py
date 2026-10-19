#!/usr/bin/env python3
"""From-scratch tokenizer + inverted index + BM25 scorer (no external deps).

Run:  python3 bm25_from_scratch.py

This file is the math-to-code translation of an Apache-Lucene-style search core:

  1. ANALYSIS  : raw text -> tokens  (tokenize -> lowercase -> stopwords -> stem)
  2. INDEXING  : tokens   -> inverted index  (term -> postings list of (doc, tf))
  3. SCORING   : query    -> ranked docs via Okapi BM25

BM25 (Okapi), the dominant lexical ranker, scores a document D for query Q as:

  score(D, Q) = sum over q in Q of
        IDF(q) * ( f(q,D) * (k1 + 1) )
                 / ( f(q,D) + k1 * (1 - b + b * |D| / avgdl) )

where
  f(q,D)  = term frequency of q in D
  |D|     = length of D in tokens
  avgdl   = average document length over the corpus
  k1      = term-frequency saturation (typ. 1.2-2.0): extra occurrences of a
            term matter less and less -- a diminishing-returns curve. As
            f(q,D) -> infinity the TF factor -> (k1 + 1), a hard ceiling.
  b       = length normalization (0..1, typ. 0.75): b=0 ignores length, b=1
            fully penalizes long docs (they win term matches "for free").
  IDF(q)  = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1 )   [Lucene's smoothed form]
            N    = number of docs, n(q) = docs containing q.
            The "+1" inside ln keeps IDF >= 0 even for very common terms.

TF-IDF (the ancestor) is just  tf * ln(N / n(q))  -- BM25 adds (a) TF
saturation via k1 and (b) length normalization via b, both of which TF-IDF
lacks. That is the entire conceptual delta.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------
# 1. ANALYSIS PIPELINE  (Lucene calls this an Analyzer: Tokenizer + TokenFilters)
# ----------------------------------------------------------------------------

# A tiny stopword list. Lucene ships a far larger one; stopwords are dropped
# because they carry almost no discriminative signal (huge n(q) => ~0 IDF).
STOPWORDS = {"a", "an", "and", "the", "of", "to", "for", "in", "on", "is", "are"}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _porter_lite_stem(token: str) -> str:
    """A deliberately tiny suffix stripper standing in for a Porter stemmer.

    Real Lucene uses PorterStemFilter / KStem / Snowball. The point is only to
    show *where* stemming sits in the pipeline: it collapses 'orders',
    'ordering', 'ordered' toward a shared stem so they match 'order'.
    """
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def analyze(text: str) -> list[str]:
    """Full analysis chain: tokenize -> lowercase -> drop stopwords -> stem."""
    tokens = _TOKEN_RE.findall(text.lower())  # tokenizer + lowercase filter
    out: list[str] = []
    for tok in tokens:
        if tok in STOPWORDS:  # stopword filter
            continue
        out.append(_porter_lite_stem(tok))  # stemming filter
    return out


# ----------------------------------------------------------------------------
# 2. INVERTED INDEX  (term dictionary -> postings list)
# ----------------------------------------------------------------------------


@dataclass
class Posting:
    doc_id: int
    tf: int  # term frequency of this term in this doc


@dataclass
class InvertedIndex:
    # term -> postings list, sorted by doc_id (so AND/OR merges are linear).
    postings: dict[str, list[Posting]] = field(
        default_factory=lambda: defaultdict(list)
    )
    doc_len: dict[int, int] = field(default_factory=dict)  # |D| per doc
    docs: dict[int, str] = field(default_factory=dict)  # original text (for display)
    n_docs: int = 0

    def add(self, doc_id: int, text: str) -> None:
        terms = analyze(text)
        self.docs[doc_id] = text
        self.doc_len[doc_id] = len(terms)
        self.n_docs += 1
        tf: dict[str, int] = defaultdict(int)
        for t in terms:
            tf[t] += 1
        for term, count in tf.items():
            self.postings[term].append(Posting(doc_id, count))

    @property
    def avgdl(self) -> float:
        if not self.doc_len:
            return 0.0
        return sum(self.doc_len.values()) / len(self.doc_len)

    def idf(self, term: str) -> float:
        # Lucene's smoothed, non-negative IDF.
        n_q = len(self.postings.get(term, []))
        return math.log(1 + (self.n_docs - n_q + 0.5) / (n_q + 0.5))


# ----------------------------------------------------------------------------
# 3. BM25 SCORER
# ----------------------------------------------------------------------------


def bm25_search(
    index: InvertedIndex, query: str, k1: float = 1.2, b: float = 0.75
) -> list[tuple[int, float]]:
    """Rank docs for `query` by BM25. Returns [(doc_id, score), ...] desc."""
    q_terms = analyze(query)
    avgdl = index.avgdl
    scores: dict[int, float] = defaultdict(float)

    for term in q_terms:
        plist = index.postings.get(term)
        if not plist:
            continue
        idf = index.idf(term)
        for p in plist:  # walk the postings list (Lucene skips via skip-lists)
            dl = index.doc_len[p.doc_id]
            tf_component = (p.tf * (k1 + 1)) / (p.tf + k1 * (1 - b + b * dl / avgdl))
            scores[p.doc_id] += idf * tf_component

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ----------------------------------------------------------------------------
# DEMO: a tiny Streamflow corpus (orders / products)
# ----------------------------------------------------------------------------

STREAMFLOW_CORPUS = {
    0: "Streamflow order 8801 for two wireless keyboards shipped to Berlin",
    1: "Order 8802 cancelled: customer returned the wireless mouse and keyboard",
    2: "Product catalog: ergonomic wireless keyboard, mechanical keyboard, trackball mouse",
    3: "Refund issued for order 8801 after the keyboard arrived damaged in shipping",
    4: "New product launch: noise cancelling headphones now in the Streamflow catalog",
}


def main() -> None:
    idx = InvertedIndex()
    for doc_id, text in STREAMFLOW_CORPUS.items():
        idx.add(doc_id, text)

    print(f"Indexed {idx.n_docs} docs | avgdl = {idx.avgdl:.2f} tokens")
    print(f"Vocabulary size = {len(idx.postings)} terms\n")

    # Show a postings list (the heart of the inverted index).
    sample = idx.postings.get("keyboard", [])
    print("Postings list for term 'keyboard' (stemmed):")
    for p in sample:
        print(f"  doc {p.doc_id}: tf={p.tf}")
    print(f"  IDF('keyboard') = {idx.idf('keyboard'):.4f}\n")

    for query in ["wireless keyboard", "order 8801", "headphones"]:
        print(f"Query: {query!r}")
        results = bm25_search(idx, query)
        if not results:
            print("  (no matches)")
        for doc_id, score in results:
            print(f"  score={score:.4f}  doc {doc_id}: {idx.docs[doc_id]}")
        print()


if __name__ == "__main__":
    main()
