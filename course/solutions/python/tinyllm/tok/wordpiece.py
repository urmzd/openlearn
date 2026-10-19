"""WordPiece (L1.3): the BERT basic tokenizer, then greedy longest match.

Encode, in order (the Hugging Face BertNormalizer, BertPreTokenizer, and
WordPiece model, so bert-base-uncased ids come out exactly):

  1. split out the special tokens ([CLS], [SEP], [MASK], ...) as written
  2. clean: drop U+0000, U+FFFD, and control characters (category C*, but
     tab, newline, and carriage return count as white space); every white
     space character becomes U+0020
  3. put a space on both sides of every CJK ideograph
  4. strip accents (NFD, then drop nonspacing marks, category Mn) when
     `strip_accents` is true, or when it is None and `lowercase` is true
  5. lowercase, one code point at a time
  6. split on white space, and make every punctuation character (Unicode P*
     or ASCII punctuation) its own word
  7. per word: the longest vocab entry that starts the word, then the
     longest `##`-prefixed entry that continues it, and so on; a word with
     no full segmentation, or longer than max_input_chars_per_word code
     points, is one [UNK]

The longest-prefix search walks the M06.2 trie once per piece.

Contract: contracts/py/tinyllm/tok/wordpiece.pyi.
"""

from __future__ import annotations

import json
import os
import unicodedata
from typing import Mapping, Optional, Sequence

from tinyllm.tok.base import check_ids
from tinyllm.tok.trie import Trie

FILE = "tokenizer.json"
SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
CJK = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B920, 0x2CEAF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
)
# White_Space (what Rust's char::is_whitespace tests); \t \n \r included.
WHITE_SPACE = frozenset(
    "\t\n\x0b\x0c\r \x85\xa0\u1680"
    + "".join(chr(c) for c in range(0x2000, 0x200B))
    + "\u2028\u2029\u202f\u205f\u3000"
)


def is_bert_whitespace(ch: str) -> bool:
    return ch in WHITE_SPACE


def is_bert_control(ch: str) -> bool:
    if ch in "\t\n\r":
        return False
    return unicodedata.category(ch).startswith("C")


def is_bert_punctuation(ch: str) -> bool:
    o = ord(ch)
    if 33 <= o <= 47 or 58 <= o <= 64 or 91 <= o <= 96 or 123 <= o <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in CJK)


class WordPieceTokenizer:
    """BERT's tokenizer: vocab (token -> id), [UNK], `##` continuations."""

    def __init__(
        self,
        vocab: Mapping[str, int],
        unk_token: str = "[UNK]",
        prefix: str = "##",
        max_input_chars_per_word: int = 100,
        lowercase: bool = True,
        strip_accents: Optional[bool] = None,
    ) -> None:
        self.vocab = dict(vocab)
        if unk_token not in self.vocab:
            raise ValueError(f"unk_token {unk_token!r} is not in the vocab")
        self._id_to_tok = {i: t for t, i in self.vocab.items()}
        n = max(self._id_to_tok) + 1
        if len(self._id_to_tok) != n:
            raise ValueError("vocab ids must be 0..N-1 with no gaps or duplicates")
        self.vocab_size = n
        self.unk_token = unk_token
        self.unk_id: Optional[int] = self.vocab[unk_token]
        self.prefix = prefix
        self.max_input_chars_per_word = max_input_chars_per_word
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.special_ids = {t: self.vocab[t] for t in SPECIALS if t in self.vocab}
        self._special_trie = Trie()
        for t, i in self.special_ids.items():
            self._special_trie.insert(t, i)
        self._word = Trie()  # pieces that may start a word
        self._cont = Trie()  # `##` pieces, stored without the prefix
        for t, i in self.vocab.items():
            if prefix and t.startswith(prefix) and len(t) > len(prefix):
                self._cont.insert(t[len(prefix) :], i)
            self._word.insert(t, i)

    @classmethod
    def from_vocab(
        cls, vocab_txt: str, lowercase: bool = True, strip_accents: Optional[bool] = None
    ) -> "WordPieceTokenizer":
        vocab: dict[str, int] = {}
        with open(vocab_txt, encoding="utf-8") as f:
            for line in f:
                tok = line.rstrip("\n").rstrip("\r")
                if tok in vocab:
                    raise ValueError(f"vocab.txt: duplicate token {tok!r}")
                vocab[tok] = len(vocab)
        return cls(vocab, lowercase=lowercase, strip_accents=strip_accents)

    @classmethod
    def from_hf_json(cls, tokenizer_json: str) -> "WordPieceTokenizer":
        with open(tokenizer_json, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("version") != "1.0":
            raise ValueError(f"version: expected \"1.0\", got {doc.get('version')!r}")
        model = doc.get("model") or {}
        mtype = model.get("type", "WordPiece" if "max_input_chars_per_word" in model else None)
        if mtype != "WordPiece":
            raise ValueError(f"model.type: expected \"WordPiece\", got {mtype!r}")
        norm = doc.get("normalizer") or {}
        if norm.get("type") != "BertNormalizer":
            raise ValueError("normalizer: expected BertNormalizer")
        if norm.get("clean_text") is not True or norm.get("handle_chinese_chars") is not True:
            raise ValueError("normalizer: clean_text and handle_chinese_chars must be true")
        if (doc.get("pre_tokenizer") or {}).get("type") != "BertPreTokenizer":
            raise ValueError("pre_tokenizer: expected BertPreTokenizer")
        tok = cls(
            model["vocab"],
            unk_token=model.get("unk_token", "[UNK]"),
            prefix=model.get("continuing_subword_prefix", "##"),
            max_input_chars_per_word=int(model.get("max_input_chars_per_word", 100)),
            lowercase=bool(norm.get("lowercase", True)),
            strip_accents=norm.get("strip_accents"),
        )
        added = {t["content"]: t["id"] for t in doc.get("added_tokens") or [] if t.get("special")}
        if added:
            tok.special_ids = added
            tok._special_trie = Trie()
            for t, i in added.items():
                tok._special_trie.insert(t, i)
        return tok

    # -- the basic tokenizer -----------------------------------------------

    def normalize(self, text: str) -> str:
        """Steps 2 to 5: clean, space CJK, strip accents, lowercase."""
        out = []
        for ch in text:
            o = ord(ch)
            if o == 0 or o == 0xFFFD or is_bert_control(ch):
                continue
            if is_bert_whitespace(ch):
                out.append(" ")
            elif is_cjk(ch):
                out.append(" " + ch + " ")
            else:
                out.append(ch)
        s = "".join(out)
        strip = self.strip_accents if self.strip_accents is not None else self.lowercase
        if strip:
            s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
        if self.lowercase:
            s = "".join(c.lower() for c in s)
        return s

    def basic_tokenize(self, text: str) -> list[str]:
        """Steps 2 to 6: the words WordPiece then splits."""
        words: list[str] = []
        cur: list[str] = []
        for ch in self.normalize(text):
            if is_bert_whitespace(ch):
                if cur:
                    words.append("".join(cur))
                    cur = []
            elif is_bert_punctuation(ch):
                if cur:
                    words.append("".join(cur))
                    cur = []
                words.append(ch)
            else:
                cur.append(ch)
        if cur:
            words.append("".join(cur))
        return words

    def wordpiece(self, word: str) -> list[int]:
        """Step 7 for one word: greedy longest match, or [UNK]."""
        if len(word) > self.max_input_chars_per_word:
            return [self.unk_id]
        ids: list[int] = []
        start = 0
        while start < len(word):
            trie = self._word if start == 0 else self._cont
            n, v = trie.longest_prefix(word, start)
            if n == 0 or v is None:
                return [self.unk_id]
            ids.append(v)
            start += n
        return ids

    # -- the protocol ------------------------------------------------------

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids: list[int] = []
        start = i = 0
        segs: list[tuple[str, Optional[int]]] = []
        while i < len(text):
            n, v = self._special_trie.longest_prefix(text, i)
            if n > 0 and v is not None:
                segs.append((text[start:i], None))
                segs.append(("", v))
                i += n
                start = i
            else:
                i += 1
        segs.append((text[start:], None))
        for seg, sid in segs:
            if sid is not None:
                ids.append(sid)
            else:
                for w in self.basic_tokenize(seg):
                    ids += self.wordpiece(w)
        if add_special:
            ids = [self.vocab["[CLS]"]] + ids + [self.vocab["[SEP]"]]
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        special = set(self.special_ids.values())
        toks = [
            self._id_to_tok[i]
            for i in check_ids(ids, self.vocab_size)
            if not (skip_special and i in special)
        ]
        out = []
        for k, t in enumerate(toks):
            if k > 0:
                t = t[len(self.prefix) :] if t.startswith(self.prefix) else " " + t
            for dirty, clean in (
                (" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"),
                (" n't", "n't"), (" 'm", "'m"), (" do not", " don't"), (" 's", "'s"),
                (" 've", "'ve"), (" 're", "'re"),
            ):
                t = t.replace(dirty, clean)
            out.append(t)
        return "".join(out)

    def token_to_id(self, s: str) -> Optional[int]:
        return self.vocab.get(s)

    def id_to_token(self, i: int) -> str:
        (v,) = check_ids([i], self.vocab_size)
        return self._id_to_tok[v]

    def save(self, dir: str) -> None:
        os.makedirs(dir, exist_ok=True)
        added = [
            {"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
             "normalized": False, "special": True}
            for t, i in sorted(self.special_ids.items(), key=lambda kv: kv[1])
        ]
        cls_, sep = "[CLS]", "[SEP]"
        doc = {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": added,
            "normalizer": {"type": "BertNormalizer", "clean_text": True, "handle_chinese_chars": True,
                           "strip_accents": self.strip_accents, "lowercase": self.lowercase},
            "pre_tokenizer": {"type": "BertPreTokenizer"},
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [{"SpecialToken": {"id": cls_, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                           {"SpecialToken": {"id": sep, "type_id": 0}}],
                "pair": [{"SpecialToken": {"id": cls_, "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                         {"SpecialToken": {"id": sep, "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}},
                         {"SpecialToken": {"id": sep, "type_id": 1}}],
                "special_tokens": {t: {"id": t, "ids": [self.vocab[t]], "tokens": [t]}
                                   for t in (cls_, sep) if t in self.vocab},
            },
            "decoder": {"type": "WordPiece", "prefix": self.prefix, "cleanup": True},
            "model": {"type": "WordPiece", "unk_token": self.unk_token,
                      "continuing_subword_prefix": self.prefix,
                      "max_input_chars_per_word": self.max_input_chars_per_word,
                      "vocab": dict(sorted(self.vocab.items(), key=lambda kv: kv[1]))},
        }
        with open(os.path.join(dir, FILE), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
            f.write("\n")

    @classmethod
    def load(cls, dir: str) -> "WordPieceTokenizer":
        return cls.from_hf_json(os.path.join(dir, FILE))
