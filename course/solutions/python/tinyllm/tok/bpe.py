"""Byte-level BPE (L1.2): GPT-2 compatible encode and decode, the trainer, and
the loaders for GPT-2's vocab.json + merges.txt and Hugging Face tokenizer.json.

Pipeline for encode (formats/tokenizer.md, in this order):

  1. split out added tokens (`<|endoftext|>`, `<|im_start|>`, ...), leftmost
     longest match, through the M06.2 trie; they become one id each
  2. pre-tokenize the rest: the declared sequence (SmolLM2: Digits, then the
     GPT-2 regex; GPT-2: the regex alone)
  3. map each pre-token's UTF-8 bytes to printable code points (the M05.2
     byte map), so a token is an ordinary string
  4. apply merges by rank: repeatedly merge the adjacent pair with the lowest
     rank, the leftmost one on a tie, until no ranked pair remains
  5. look each symbol up in the vocabulary

Decode reverses 3 and 1 and decodes the bytes as UTF-8 with replacement.

The trainer is the Hugging Face BpeTrainer algorithm: count pairs once, then
keep a max-heap of (count, pair) with lazy updates, and after each merge
apply only the count changes the merge causes. Ties go to the pair whose ids
are smallest: specials first, then the 256 byte symbols in code point order,
then merged tokens in creation order.

Contract: contracts/py/tinyllm/tok/bpe.pyi.
"""

from __future__ import annotations

import heapq
import json
import os
from collections import Counter
from typing import Iterable, Mapping, Optional, Sequence

from tinyllm.tok.base import check_ids
from tinyllm.tok.bytes_unicode import bytes_to_unicode, unicode_to_bytes
from tinyllm.tok.pretok import pretokenize_gpt2, split_digits
from tinyllm.tok.trie import Trie

FILE = "tokenizer.json"
GPT2_PRE = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True}
GPT2_EOT = "<|endoftext|>"


def _check_pre_tokenizer(pre: object) -> list[dict]:
    """The declared pre-tokenizer as a flat list of steps; ValueError outside
    the subset this loader supports: Digits, ByteLevel, or a Sequence of them,
    with ByteLevel (use_regex true) last."""
    if not isinstance(pre, dict):
        raise ValueError("pre_tokenizer: a byte-level BPE needs a ByteLevel pre-tokenizer")
    steps = pre.get("pretokenizers") if pre.get("type") == "Sequence" else [pre]
    if not isinstance(steps, list) or not steps:
        raise ValueError("pre_tokenizer: Sequence needs a non-empty `pretokenizers` list")
    for k, s in enumerate(steps):
        t = s.get("type") if isinstance(s, dict) else None
        if t == "Digits":
            if not isinstance(s.get("individual_digits", False), bool):
                raise ValueError("pre_tokenizer: Digits.individual_digits must be a bool")
        elif t == "ByteLevel":
            if s.get("use_regex", True) is not True:
                raise ValueError("pre_tokenizer: ByteLevel.use_regex must be true")
            if k != len(steps) - 1:
                raise ValueError("pre_tokenizer: ByteLevel must be the last step")
        else:
            raise ValueError(f"pre_tokenizer: type {t!r} is not supported for BPE")
    if steps[-1].get("type") != "ByteLevel":
        raise ValueError("pre_tokenizer: the last step must be ByteLevel")
    return [dict(s) for s in steps]


def _merge_word(w: list[int], a: int, b: int, new: int) -> list[tuple[tuple[int, int], int]]:
    """Merge every non-overlapping (a, b) in w, left to right, in place.
    Returns the pair count changes the merge causes around each occurrence
    (Hugging Face `Word::merge`): the merged pair itself is not decremented."""
    changes: list[tuple[tuple[int, int], int]] = []
    i = 0
    while i < len(w):
        if w[i] == a and i + 1 < len(w) and w[i + 1] == b:
            if i > 0:
                changes.append(((w[i - 1], a), -1))
                changes.append(((w[i - 1], new), 1))
            w[i : i + 2] = [new]
            if i < len(w) - 1:
                changes.append(((b, w[i + 1]), -1))
                changes.append(((new, w[i + 1]), 1))
        i += 1
    return changes


class BPETokenizer:
    """A byte-level BPE: vocab (token string -> id), merges by rank, added tokens."""

    def __init__(
        self,
        vocab: Mapping[str, int],
        merges: Sequence[tuple[str, str]],
        added_tokens: Sequence[tuple[str, int, bool]] = (),
        pre_tokenizer: Optional[Mapping] = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = [(str(a), str(b)) for a, b in merges]
        self.ranks = {pair: r for r, pair in reversed(list(enumerate(self.merges)))}
        self.pre_tokenizer = dict(pre_tokenizer) if pre_tokenizer is not None else dict(GPT2_PRE)
        self._steps = _check_pre_tokenizer(self.pre_tokenizer)
        self.added = {content: int(i) for content, i, _ in added_tokens}
        self.special_ids = {content: int(i) for content, i, sp in added_tokens if sp}
        self.unk_id: Optional[int] = None
        self._id_to_tok: dict[int, str] = {i: t for t, i in self.vocab.items()}
        self._added_by_id = {i: t for t, i in self.added.items()}
        n = max([-1, *self._id_to_tok, *self._added_by_id]) + 1
        missing = [i for i in range(n) if i not in self._id_to_tok and i not in self._added_by_id]
        if missing:
            raise ValueError(f"ids must be contiguous: no token for id {missing[0]}")
        self.vocab_size = n
        self._b2u = bytes_to_unicode()
        self._u2b = unicode_to_bytes()
        self._trie = Trie()
        for content, i in self.added.items():
            self._trie.insert(content, i)
        self._cache: dict[str, list[int]] = {}

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_gpt2(cls, vocab_json: str, merges_txt: str) -> "BPETokenizer":
        with open(vocab_json, encoding="utf-8") as f:
            vocab = json.load(f)
        merges = []
        with open(merges_txt, encoding="utf-8") as f:
            for line in f.read().split("\n"):
                if not line or line.startswith("#version"):
                    continue
                parts = line.split(" ")
                if len(parts) != 2:
                    raise ValueError(f"merges.txt: bad line {line!r}")
                merges.append((parts[0], parts[1]))
        added = [(GPT2_EOT, vocab[GPT2_EOT], True)] if GPT2_EOT in vocab else []
        return cls(vocab, merges, added, GPT2_PRE)

    @classmethod
    def from_hf_json(cls, tokenizer_json: str) -> "BPETokenizer":
        with open(tokenizer_json, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("version") != "1.0":
            raise ValueError(f"version: expected \"1.0\", got {doc.get('version')!r}")
        if doc.get("normalizer") is not None:
            raise ValueError("normalizer: a byte-level BPE file has none (null)")
        model = doc.get("model") or {}
        # Older files (GPT-2's) omit model.type; a BPE model is the one with merges.
        mtype = model.get("type", "BPE" if "merges" in model else None)
        if mtype != "BPE":
            raise ValueError(f"model.type: expected \"BPE\", got {mtype!r}")
        for key, ok in (
            ("dropout", (None,)),
            ("unk_token", (None,)),
            ("continuing_subword_prefix", (None, "")),
            ("end_of_word_suffix", (None, "")),
            ("fuse_unk", (False, None)),
            ("byte_fallback", (False, None)),
            ("ignore_merges", (False, None)),
        ):
            if model.get(key) not in ok:
                raise ValueError(f"model.{key}: {model.get(key)!r} is outside the subset")
        merges = []
        for m in model.get("merges", []):
            pair = m.split(" ") if isinstance(m, str) else m
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"model.merges: bad entry {m!r}")
            merges.append((pair[0], pair[1]))
        added = []
        for t in doc.get("added_tokens") or []:
            for key in ("single_word", "lstrip", "rstrip"):
                if t.get(key):
                    raise ValueError(f"added_tokens: {key} must be false ({t.get('content')!r})")
            added.append((t["content"], t["id"], bool(t.get("special"))))
        pp = doc.get("post_processor")
        if pp is not None and pp.get("type") != "ByteLevel":
            raise ValueError(f"post_processor: {pp.get('type')!r} is not supported for BPE")
        dec = doc.get("decoder")
        if dec is not None and dec.get("type") != "ByteLevel":
            raise ValueError(f"decoder: {dec.get('type')!r} is not supported for BPE")
        return cls(model["vocab"], merges, added, doc.get("pre_tokenizer"))

    # -- training ----------------------------------------------------------

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        specials: Sequence[str] = (),
        min_freq: int = 2,
    ) -> "BPETokenizer":
        specials = list(dict.fromkeys(specials))
        if vocab_size < 256 + len(specials):
            raise ValueError(f"vocab_size must be >= 256 + {len(specials)} specials, got {vocab_size}")
        if min_freq < 1:
            raise ValueError(f"min_freq must be >= 1, got {min_freq}")
        b2u = bytes_to_unicode()
        trie = Trie()
        for k, s in enumerate(specials):
            trie.insert(s, k)
        # 1. word counts: pre-tokens in the byte alphabet, specials removed
        wc: Counter[str] = Counter()
        for text in texts:
            for seg, sid in _split_added(trie, text) if specials else [(text, None)]:
                if sid is None:
                    for p in pretokenize_gpt2(seg):
                        wc["".join(b2u[b] for b in p.encode("utf-8"))] += 1
        # 2. the initial vocabulary: specials, then the 256 symbols by code point
        id_to_tok = list(specials) + sorted(b2u.values(), key=ord)
        tok_to_id = {t: i for i, t in enumerate(id_to_tok)}
        words = [[tok_to_id[c] for c in w] for w in wc]
        counts = list(wc.values())
        # 3. pair counts and where each pair occurs
        pair_counts: dict[tuple[int, int], int] = {}
        where: dict[tuple[int, int], set[int]] = {}
        for wi, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + counts[wi]
                where.setdefault(pair, set()).add(wi)
        seq = 0
        heap: list[list] = []
        for pair, pos in where.items():
            if pair_counts[pair] > 0:
                heap.append([-pair_counts[pair], pair[0], pair[1], seq, pos])
                seq += 1
        heapq.heapify(heap)
        merges: list[tuple[str, str]] = []
        # 4. merge the most frequent pair until the vocabulary is full
        while len(id_to_tok) < vocab_size and heap:
            top = heapq.heappop(heap)
            pair = (top[1], top[2])
            cur = pair_counts.get(pair, 0)
            if -top[0] != cur:  # stale: requeue with the current count
                top[0] = -cur
                heapq.heappush(heap, top)
                continue
            if cur < 1 or cur < min_freq:
                break
            a, b = id_to_tok[pair[0]], id_to_tok[pair[1]]
            new_tok = a + b
            if new_tok in tok_to_id:
                new_id = tok_to_id[new_tok]
            else:
                new_id = len(id_to_tok)
                id_to_tok.append(new_tok)
                tok_to_id[new_tok] = new_id
            merges.append((a, b))
            update: dict[tuple[int, int], set[int]] = {}
            for wi in sorted(top[4]):
                for p, d in _merge_word(words[wi], pair[0], pair[1], new_id):
                    pair_counts[p] = pair_counts.get(p, 0) + d * counts[wi]
                    if d > 0:
                        update.setdefault(p, set()).add(wi)
            for p, pos in update.items():
                if pair_counts[p] > 0:
                    heapq.heappush(heap, [-pair_counts[p], p[0], p[1], seq, pos])
                    seq += 1
        vocab = {t: i for i, t in enumerate(id_to_tok)}
        added = [(s, k, True) for k, s in enumerate(specials)]
        return cls(vocab, merges, added, GPT2_PRE)

    # -- encode and decode -------------------------------------------------

    def _pieces(self, seg: str) -> list[str]:
        """Pre-tokens of text that holds no added token, as declared."""
        pieces = [seg]
        for step in self._steps:
            nxt: list[str] = []
            for p in pieces:
                if step["type"] == "Digits":
                    nxt += split_digits(p, bool(step.get("individual_digits", False)))
                else:
                    if step.get("add_prefix_space") and not p.startswith(" "):
                        p = " " + p
                    nxt += pretokenize_gpt2(p)
            pieces = nxt
        return pieces

    def _bpe(self, word: str) -> list[int]:
        """Ids of one pre-token already in the byte alphabet (cached)."""
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        # A byte symbol the vocab lacks is dropped, as Hugging Face does when
        # unk_token is null (SmolLM2 has no symbol for six control bytes).
        syms = [c for c in word if c in self.vocab]
        while len(syms) > 1:
            best_rank, best_k = None, -1
            for k in range(len(syms) - 1):
                r = self.ranks.get((syms[k], syms[k + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_k = r, k
            if best_rank is None:
                break
            syms[best_k : best_k + 2] = [syms[best_k] + syms[best_k + 1]]
        try:
            ids = [self.vocab[s] for s in syms]
        except KeyError as e:  # a merge whose result the vocab lacks: a broken file
            raise ValueError(f"merge result {e.args[0]!r} is not in the vocab") from None
        self._cache[word] = ids
        return ids

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids: list[int] = []
        segs = _split_added(self._trie, text) if self.added else [(text, None)]
        for seg, aid in segs:
            if aid is not None:
                ids.append(aid)
                continue
            for p in self._pieces(seg):
                ids += self._bpe("".join(self._b2u[b] for b in p.encode("utf-8")))
        return ids

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        buf = bytearray()
        for i in check_ids(ids, self.vocab_size):
            if i in self._added_by_id:
                content = self._added_by_id[i]
                if not (skip_special and content in self.special_ids):
                    buf += content.encode("utf-8")
            else:
                buf += bytes(self._u2b[c] for c in self._id_to_tok[i])
        return buf.decode("utf-8", "replace")

    def token_to_id(self, s: str) -> Optional[int]:
        if s in self.added:
            return self.added[s]
        return self.vocab.get(s)

    def id_to_token(self, i: int) -> str:
        (v,) = check_ids([i], self.vocab_size)
        return self._added_by_id.get(v) or self._id_to_tok[v]

    # -- files -------------------------------------------------------------

    def save(self, dir: str) -> None:
        os.makedirs(dir, exist_ok=True)
        added = [
            {
                "id": i,
                "content": t,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": t in self.special_ids,
            }
            for t, i in sorted(self.added.items(), key=lambda kv: kv[1])
        ]
        doc = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": added,
            "normalizer": None,
            "pre_tokenizer": self.pre_tokenizer,
            "post_processor": None,
            "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": None,
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "byte_fallback": False,
                "ignore_merges": False,
                "vocab": dict(sorted(self.vocab.items(), key=lambda kv: kv[1])),
                "merges": [[a, b] for a, b in self.merges],
            },
        }
        with open(os.path.join(dir, FILE), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
            f.write("\n")

    @classmethod
    def load(cls, dir: str) -> "BPETokenizer":
        return cls.from_hf_json(os.path.join(dir, FILE))


def _split_added(trie: Trie, text: str) -> list[tuple[str, Optional[int]]]:
    """text as (segment, None) runs and (content, id) added tokens, by
    leftmost-longest match; "".join(segments) == text."""
    out: list[tuple[str, Optional[int]]] = []
    start = i = 0
    while i < len(text):
        n, v = trie.longest_prefix(text, i)
        if n > 0 and v is not None:
            if start < i:
                out.append((text[start:i], None))
            out.append((text[i : i + n], v))
            i += n
            start = i
        else:
            i += 1
    if start < len(text):
        out.append((text[start:], None))
    return out
