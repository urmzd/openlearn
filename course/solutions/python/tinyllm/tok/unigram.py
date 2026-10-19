"""The Unigram LM tokenizer (L1.4): Viterbi encode, sampled encode, EM training.

A Unigram model is a set of pieces with log-probabilities. A segmentation
s = (p_1, ..., p_k) of a word has probability prod_i P(p_i), and:

  encode         picks the most probable segmentation (Viterbi over the
                 lattice of pieces, built with the M06.2 trie)
  sample_encode  draws a segmentation with probability proportional to
                 P(s)^alpha (forward filtering, backward sampling)
  train          starts from many frequent substrings scored by the M07.2
                 maximum-likelihood estimate of their counts, re-estimates
                 the probabilities by EM (expected counts, then the same
                 estimate: count over total), and prunes the pieces whose
                 removal costs the least likelihood, until the vocabulary fits

Text is pre-tokenized like SentencePiece (Hugging Face `Metaspace`): every
U+0020 becomes U+2581 "▁", one "▁" is prepended, and the text is split
before each "▁", so pieces never cross a word start. A character no piece
covers is an unknown edge with log-probability min_score - 10, and adjacent
unknowns fuse into one <unk> (sentencepiece and Hugging Face agree on all of
this, so their tokenizer.json files load unchanged).

Ties in Viterbi go to the path whose last piece starts earliest (the
backpointer keeps the first candidate that reaches the best score, scanning
start positions left to right), which is sentencepiece's rule.

Contract: contracts/py/tinyllm/tok/unigram.pyi.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Iterable, Optional, Sequence

from tinyllm.prob.mle import mle
from tinyllm.tok.base import check_ids
from tinyllm.tok.trie import Trie

FILE = "tokenizer.json"
SPACE = "▁"
UNK = "<unk>"
UNK_PENALTY = 10.0
MAX_PIECE_LEN = 16


def _logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _m_step(counts: dict[str, float]) -> dict[str, float]:
    """log(count / total): the maximum-likelihood estimate of M07.2 on
    fractional (expected) counts, which its integer `mle` does not take."""
    total = math.fsum(counts.values())
    return {p: math.log(c / total) for p, c in counts.items()}


def metaspace(text: str) -> list[str]:
    """The Metaspace pre-tokenizer (replacement "▁", prepend "always", split)."""
    if not text:
        return []
    s = text.replace(" ", SPACE)
    if not s.startswith(SPACE):
        s = SPACE + s
    words: list[str] = []
    start = 0
    for i in range(1, len(s)):
        if s[i] == SPACE:
            words.append(s[start:i])
            start = i
    words.append(s[start:])
    return words


class UnigramTokenizer:
    """pieces[i] = (piece, log-probability); id i. Id unk_id is <unk>."""

    def __init__(
        self,
        pieces: Sequence[tuple[str, float]],
        unk_id: int = 0,
        specials: Sequence[str] = (),
    ) -> None:
        self.pieces = [(str(p), float(s)) for p, s in pieces]
        if not 0 <= unk_id < len(self.pieces):
            raise ValueError(f"unk_id {unk_id} is outside the vocab")
        texts = [p for p, _ in self.pieces]
        if len(set(texts)) != len(texts):
            raise ValueError("pieces must be unique")
        self.vocab_size = len(self.pieces)
        self.unk_id: Optional[int] = unk_id
        self._ids = {p: i for i, (p, _) in enumerate(self.pieces)}
        special = {self.pieces[unk_id][0], *specials}
        missing = [s for s in special if s not in self._ids]
        if missing:
            raise ValueError(f"special token {missing[0]!r} is not a piece")
        self.special_ids = {s: self._ids[s] for s in special}
        self.min_score = min(s for _, s in self.pieces)
        self._trie = Trie()
        for i, (p, _) in enumerate(self.pieces):
            if i not in self.special_ids.values():
                self._trie.insert(p, i)
        self._special_trie = Trie()  # specials, <unk> included, match as written
        for s, i in self.special_ids.items():
            self._special_trie.insert(s, i)

    # -- the lattice -------------------------------------------------------

    def _edges(self, word: str, scale: float = 1.0) -> list[list[tuple[int, int, float]]]:
        """edges[j] = every (start, id, score * scale) of a piece ending at j.
        An unknown edge (one character, min_score - 10) is added where no
        single-character piece starts."""
        n = len(word)
        edges: list[list[tuple[int, int, float]]] = [[] for _ in range(n + 1)]
        unk = (self.min_score - UNK_PENALTY) * scale
        for i in range(n):
            single = False
            for length, pid in self._trie.prefixes(word, i):
                edges[i + length].append((i, pid, self.pieces[pid][1] * scale))
                single = single or length == 1
            if not single:
                edges[i + 1].append((i, self.unk_id, unk))
        for e in edges:
            e.sort(key=lambda t: t[0])
        return edges

    def _fuse(self, path: list[tuple[int, int, int]]) -> list[int]:
        """Ids of a path of (start, end, id), adjacent unknowns fused."""
        ids: list[int] = []
        for _, _, pid in path:
            if pid == self.unk_id and ids and ids[-1] == self.unk_id:
                continue
            ids.append(pid)
        return ids

    def viterbi(self, word: str) -> list[int]:
        """The most probable segmentation of one pre-tokenized word."""
        n = len(word)
        if n == 0:
            return []
        edges = self._edges(word)
        best = [-math.inf] * (n + 1)
        back: list[Optional[tuple[int, int]]] = [None] * (n + 1)
        best[0] = 0.0
        for j in range(1, n + 1):
            for i, pid, s in edges[j]:  # start positions ascending
                if best[i] == -math.inf:
                    continue
                cand = best[i] + s
                if back[j] is None or cand > best[j]:
                    best[j], back[j] = cand, (i, pid)
        path = []
        j = n
        while j > 0:
            i, pid = back[j]
            path.append((i, j, pid))
            j = i
        path.reverse()
        return self._fuse(path)

    def _split_special(self, text: str) -> list[tuple[str, Optional[int]]]:
        out: list[tuple[str, Optional[int]]] = []
        start = i = 0
        while i < len(text):
            n, v = self._special_trie.longest_prefix(text, i)
            if n > 0 and v is not None:
                if start < i:
                    out.append((text[start:i], None))
                out.append(("", v))
                i += n
                start = i
            else:
                i += 1
        if start < len(text) or not out:
            out.append((text[start:], None))
        return out

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids: list[int] = []
        for seg, sid in self._split_special(text):
            if sid is not None:
                ids.append(sid)
            elif seg:
                for w in metaspace(seg):
                    ids += self.viterbi(w)
        return ids

    def sample_encode(self, text: str, alpha: float, rng) -> list[int]:
        if not alpha >= 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        ids: list[int] = []
        for seg, sid in self._split_special(text):
            if sid is not None:
                ids.append(sid)
                continue
            if not seg:
                continue
            for w in metaspace(seg):
                n = len(w)
                edges = self._edges(w, alpha)
                fwd = [-math.inf] * (n + 1)
                fwd[0] = 0.0
                for j in range(1, n + 1):
                    fwd[j] = _logsumexp([fwd[i] + s for i, _, s in edges[j]])
                path = []
                j = n
                while j > 0:
                    cands = [(i, pid, fwd[i] + s - fwd[j]) for i, pid, s in edges[j]]
                    u = rng.uniform()
                    acc = 0.0
                    pick = cands[-1]
                    for c in cands:
                        acc += math.exp(c[2])
                        if u < acc:
                            pick = c
                            break
                    path.append((pick[0], j, pick[1]))
                    j = pick[0]
                path.reverse()
                ids += self._fuse(path)
        return ids

    def log_likelihood(self, texts: Iterable[str]) -> float:
        """sum over the words of all texts of log sum_s P(s): the EM objective."""
        total = 0.0
        for t in texts:
            for seg, sid in self._split_special(t):
                if sid is None and seg:
                    for w in metaspace(seg):
                        n = len(w)
                        edges = self._edges(w)
                        fwd = [-math.inf] * (n + 1)
                        fwd[0] = 0.0
                        for j in range(1, n + 1):
                            fwd[j] = _logsumexp([fwd[i] + s for i, _, s in edges[j]])
                        total += fwd[n]
        return total

    # -- training ----------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        seed_factor: int = 10,
        em_iters: int = 2,
        shrink: float = 0.75,
    ) -> "UnigramTokenizer":
        if em_iters < 1:
            raise ValueError(f"em_iters must be >= 1, got {em_iters}")
        if not 0 < shrink < 1:
            raise ValueError(f"shrink must be in (0, 1), got {shrink}")
        words: Counter[str] = Counter()
        for t in texts:
            words.update(metaspace(t))
        chars: Counter[str] = Counter()
        for w, c in words.items():
            for ch in w:
                chars[ch] += c
        target = vocab_size - 1  # id 0 is <unk>
        if len(chars) > target:
            raise ValueError(f"vocab_size {vocab_size} cannot hold the {len(chars)} characters plus <unk>")
        # 1. seed: frequent substrings (score = frequency x length) and every character
        subs: Counter[str] = Counter()
        for w, c in words.items():
            for i in range(len(w)):
                for j in range(i + 2, min(len(w), i + MAX_PIECE_LEN) + 1):
                    subs[w[i:j]] += c
        ranked = sorted(subs, key=lambda s: (-subs[s] * len(s), s))[: seed_factor * vocab_size]
        freq = dict(chars)
        freq.update({s: subs[s] for s in ranked})
        probs = mle(freq)
        model = {p: math.log(q) for p, q in probs.items()}
        while True:
            # 2. EM: expected counts by forward-backward, then their MLE
            for _ in range(em_iters):
                counts = cls._from_scores(model)._expected_counts(words)
                # SentencePiece drops pieces expected fewer than 0.5 times;
                # a character always stays (count >= 0.5) so every text encodes.
                counts = {p: c for p, c in counts.items() if c >= 0.5 or p in chars}
                for ch in chars:
                    counts[ch] = max(counts.get(ch, 0.0), 0.5)
                model = _m_step(counts)
            if len(model) <= target:
                break
            # 3. prune the pieces that cost the least likelihood
            tok = cls._from_scores(model)
            keep = max(target, int(len(model) * shrink))
            loss = tok._removal_loss(words)
            optional = sorted((p for p in model if p not in chars), key=lambda p: (-loss.get(p, 0.0), p))
            kept = set(chars) | set(optional[: keep - len(chars)])
            model = {p: s for p, s in model.items() if p in kept}
        ordered = sorted(model.items(), key=lambda kv: (-kv[1], kv[0]))
        return cls([(UNK, 0.0)] + ordered, unk_id=0)

    def em_step(self, texts: Iterable[str]) -> "UnigramTokenizer":
        """One EM iteration on this piece set: expected counts over the words
        of texts (E), then their maximum-likelihood estimate (M). Pieces with
        expected count 0 are dropped; <unk> and specials are kept as they are."""
        words: Counter[str] = Counter()
        for t in texts:
            words.update(metaspace(t))
        counts = {p: c for p, c in self._expected_counts(words).items() if c > 0}
        new = _m_step(counts)
        keep = [(p, s) for i, (p, s) in enumerate(self.pieces) if i in self.special_ids.values()]
        pieces = keep + sorted(new.items(), key=lambda kv: (-kv[1], kv[0]))
        unk = self.pieces[self.unk_id][0]
        return type(self)(pieces, [p for p, _ in pieces].index(unk), list(self.special_ids))

    @classmethod
    def _from_scores(cls, model: dict) -> "UnigramTokenizer":
        return cls([(UNK, 0.0)] + sorted(model.items()), unk_id=0)

    def _expected_counts(self, words: Counter) -> dict[str, float]:
        """E-step: expected number of uses of each piece, summed over words."""
        counts: dict[str, float] = {
            p: 0.0 for i, (p, _) in enumerate(self.pieces) if i not in self.special_ids.values()
        }
        for w, c in words.items():
            n = len(w)
            edges = self._edges(w)
            fwd = [-math.inf] * (n + 1)
            bwd = [-math.inf] * (n + 1)
            fwd[0] = 0.0
            bwd[n] = 0.0
            for j in range(1, n + 1):
                fwd[j] = _logsumexp([fwd[i] + s for i, _, s in edges[j]])
            for j in range(n, 0, -1):
                for i, _, s in edges[j]:
                    bwd[i] = _logsumexp([bwd[i], s + bwd[j]])
            z = fwd[n]
            for j in range(1, n + 1):
                for i, pid, s in edges[j]:
                    if pid != self.unk_id:
                        counts[self.pieces[pid][0]] += c * math.exp(fwd[i] + s + bwd[j] - z)
        return counts

    def _removal_loss(self, words: Counter) -> dict[str, float]:
        """Per piece: Viterbi frequency x (its log-probability minus that of
        the best segmentation of its own text without it)."""
        freq: Counter[int] = Counter()
        for w, c in words.items():
            for pid in self.viterbi(w):
                freq[pid] += c
        loss: dict[str, float] = {}
        for pid, f in freq.items():
            piece, score = self.pieces[pid]
            if pid == self.unk_id or len(piece) == 1:
                continue
            n = len(piece)
            edges = self._edges(piece)
            best = [-math.inf] * (n + 1)
            best[0] = 0.0
            for j in range(1, n + 1):
                for i, q, s in edges[j]:
                    if q != pid:
                        best[j] = max(best[j], best[i] + s)
            loss[piece] = f * (score - best[n])
        return loss

    # -- the rest of the protocol ------------------------------------------

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        special = set(self.special_ids.values())
        out = []
        for i in check_ids(ids, self.vocab_size):
            if skip_special and i in special:
                continue
            t = self.pieces[i][0].replace(SPACE, " ")
            if not out and t.startswith(" "):
                t = t[1:]
            out.append(t)
        return "".join(out)

    def token_to_id(self, s: str) -> Optional[int]:
        return self._ids.get(s)

    def id_to_token(self, i: int) -> str:
        (v,) = check_ids([i], self.vocab_size)
        return self.pieces[v][0]

    def save(self, dir: str) -> None:
        os.makedirs(dir, exist_ok=True)
        meta = {"type": "Metaspace", "replacement": SPACE, "prepend_scheme": "always", "split": True}
        doc = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [
                {"id": i, "content": s, "single_word": False, "lstrip": False, "rstrip": False,
                 "normalized": False, "special": True}
                for s, i in sorted(self.special_ids.items(), key=lambda kv: kv[1])
            ],
            "normalizer": None,
            "pre_tokenizer": meta,
            "post_processor": None,
            "decoder": dict(meta),
            "model": {"type": "Unigram", "unk_id": self.unk_id,
                      "vocab": [[p, s] for p, s in self.pieces], "byte_fallback": False},
        }
        with open(os.path.join(dir, FILE), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
            f.write("\n")

    @classmethod
    def load(cls, dir: str) -> "UnigramTokenizer":
        return cls.from_hf_json(os.path.join(dir, FILE))

    @classmethod
    def from_hf_json(cls, tokenizer_json: str) -> "UnigramTokenizer":
        with open(tokenizer_json, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("version") != "1.0":
            raise ValueError(f"version: expected \"1.0\", got {doc.get('version')!r}")
        model = doc.get("model") or {}
        mtype = model.get("type", "Unigram" if isinstance(model.get("vocab"), list) else None)
        if mtype != "Unigram":
            raise ValueError(f"model.type: expected \"Unigram\", got {mtype!r}")
        if model.get("byte_fallback"):
            raise ValueError("model.byte_fallback must be false")
        if model.get("unk_id") is None:
            raise ValueError("model.unk_id is required")
        if doc.get("normalizer") is not None:
            raise ValueError("normalizer: only null is supported for Unigram")
        pre = doc.get("pre_tokenizer") or {}
        if (
            pre.get("type") != "Metaspace"
            or pre.get("replacement", SPACE) != SPACE
            or pre.get("prepend_scheme", "always") != "always"
            or pre.get("split", True) is not True
        ):
            raise ValueError("pre_tokenizer: expected Metaspace (▁, prepend always, split)")
        specials = [t["content"] for t in doc.get("added_tokens") or [] if t.get("special")]
        return cls([(p, s) for p, s in model["vocab"]], int(model["unk_id"]), specials)
