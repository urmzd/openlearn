"""The char tokenizer (L1.1): one id per Unicode code point.

The vocabulary is `<unk>` (always id 0), then the special tokens given to
`train` (only the roles of formats/tokenizer-char.schema.json: `<bos>`,
`<eos>`, `<pad>`), then every code point seen in training, sorted by code
point. Encoding never normalizes: `"é"` (U+00E9) and `"é"` are
different texts and get different ids, so decode(encode(x)) == x exactly.

Saved as `tinyllm_char.json` (formats/tokenizer.md).

Contract: contracts/py/tinyllm/tok/char.pyi.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Mapping, Optional, Sequence

from tinyllm.tok.base import check_ids

FILE = "tinyllm_char.json"
ROLES = ("unk", "bos", "eos", "pad")


class CharTokenizer:
    """vocab[i] is the text of id i: one code point, or a special token."""

    def __init__(self, vocab: Sequence[str], specials: Mapping[str, int]) -> None:
        vocab = list(vocab)
        if len(set(vocab)) != len(vocab):
            raise ValueError("vocab entries must be unique")
        if "unk" not in specials:
            raise ValueError("specials must name the `unk` id")
        for role, i in specials.items():
            if role not in ROLES:
                raise ValueError(f"special role {role!r} is not one of {ROLES}")
            if not 0 <= i < len(vocab):
                raise ValueError(f"special {role} = {i} is outside the vocab")
        special_texts = {vocab[i] for i in specials.values()}
        for i, t in enumerate(vocab):
            if t not in special_texts and len(t) != 1:
                raise ValueError(f"vocab[{i}] = {t!r} is not one code point")
        self.vocab = vocab
        self.roles = dict(specials)
        self.vocab_size = len(vocab)
        self.special_ids = {vocab[i]: i for i in specials.values()}
        self.unk_id: Optional[int] = specials["unk"]
        self._ids = {t: i for i, t in enumerate(vocab) if t not in special_texts}

    @classmethod
    def train(cls, texts: Iterable[str], specials: Sequence[str] = ()) -> "CharTokenizer":
        vocab = ["<unk>"]
        roles = {"unk": 0}
        for s in specials:
            role = s[1:-1] if s.startswith("<") and s.endswith(">") else ""
            if role not in ROLES:
                raise ValueError(f"special {s!r} must be one of <unk>, <bos>, <eos>, <pad>")
            if role not in roles:
                roles[role] = len(vocab)
                vocab.append(s)
        seen: set[str] = set()
        for t in texts:
            seen.update(t)
        vocab += sorted(seen, key=ord)
        return cls(vocab, roles)

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids = [self._ids.get(ch, self.unk_id) for ch in text]
        if add_special:
            if "bos" in self.roles:
                ids.insert(0, self.roles["bos"])
            if "eos" in self.roles:
                ids.append(self.roles["eos"])
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = False) -> str:
        special = set(self.roles.values())
        out = []
        for i in check_ids(ids, self.vocab_size):
            if skip_special and i in special:
                continue
            out.append(self.vocab[i])
        return "".join(out)

    def token_to_id(self, s: str) -> Optional[int]:
        if s in self.special_ids:
            return self.special_ids[s]
        return self._ids.get(s)

    def id_to_token(self, i: int) -> str:
        (v,) = check_ids([i], self.vocab_size)
        return self.vocab[v]

    def save(self, dir: str) -> None:
        os.makedirs(dir, exist_ok=True)
        doc = {"type": "char", "vocab": self.vocab, "specials": self.roles}
        with open(os.path.join(dir, FILE), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
            f.write("\n")

    @classmethod
    def load(cls, dir: str) -> "CharTokenizer":
        with open(os.path.join(dir, FILE), encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict) or doc.get("type") != "char":
            raise ValueError(f"{FILE}: `type` must be \"char\"")
        vocab, specials = doc.get("vocab"), doc.get("specials")
        if not isinstance(vocab, list) or not all(isinstance(t, str) and t for t in vocab):
            raise ValueError(f"{FILE}: `vocab` must be a list of non-empty strings")
        if not isinstance(specials, dict) or not all(
            isinstance(v, int) and not isinstance(v, bool) for v in specials.values()
        ):
            raise ValueError(f"{FILE}: `specials` must map roles to integer ids")
        return cls(vocab, specials)
