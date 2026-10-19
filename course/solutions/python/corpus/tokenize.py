"""Tokenize the shards and pack llm.c .bin token streams (data.07).

The trainer never sees text: it memory-maps a flat array of token ids
(L0.6 TokenStream). This stage turns each split of the sharded corpus into
such arrays, in the llm.c layout (a 1024-byte header, then uint16 or uint32
ids), with a manifest that lets anyone check the token counts and hashes.

Contract: contracts/py/corpus/tokenize.pyi. Files: formats/tokens-bin.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from corpus.shard import read_shards
from tinyllm.tok.bpe import BPETokenizer
from tinyllm.tok.metrics import bytes_per_token

MAGIC = 20240520
HEADER_INTS = 256
MAX_FILE_TOKENS = 100_000_000
_HEADER_BYTES = HEADER_INTS * 4
_MAX_IDS = (1 << 31) - 1
_BATCH = 1024


def _width(vocab_size: int) -> tuple[int, np.dtype]:
    """(version, little-endian id dtype) for a vocabulary size."""
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
    return (1, np.dtype("<u2")) if vocab_size <= 65536 else (2, np.dtype("<u4"))


def _header(version: int, n_tokens: int, vocab_size: int) -> bytes:
    """The 256 little-endian int32 header."""
    h = np.zeros(HEADER_INTS, dtype="<i4")
    h[0], h[1], h[2], h[3] = MAGIC, version, n_tokens, vocab_size
    return h.tobytes()


def write_bin(path: Path, ids: Sequence[int], vocab_size: int) -> None:
    """One .bin file: header then ids."""
    version, dt = _width(vocab_size)
    arr = np.asarray(ids, dtype=np.int64).reshape(-1)
    if arr.size > _MAX_IDS:
        raise ValueError(f"{arr.size} ids do not fit one file (at most 2^31 - 1)")
    if arr.size and (arr.min() < 0 or arr.max() >= vocab_size):
        raise ValueError(f"an id is outside [0, {vocab_size})")
    with open(path, "wb") as f:
        f.write(_header(version, int(arr.size), vocab_size))
        f.write(arr.astype(dt).tobytes())


def read_bin(path: Path) -> tuple[dict[str, int], NDArray]:
    """(header fields, the ids as int64)."""
    raw = Path(path).read_bytes()
    if len(raw) < _HEADER_BYTES:
        raise ValueError(f"{path}: shorter than the 1024-byte header")
    h = np.frombuffer(raw[:_HEADER_BYTES], dtype="<i4")
    if int(h[0]) != MAGIC:
        raise ValueError(f"{path}: magic {int(h[0])}, want {MAGIC}")
    if int(h[1]) not in (1, 2):
        raise ValueError(f"{path}: unknown version {int(h[1])}")
    dt = np.dtype("<u2") if int(h[1]) == 1 else np.dtype("<u4")
    n = int(h[2])
    if n < 0 or len(raw) != _HEADER_BYTES + n * dt.itemsize:
        raise ValueError(f"{path}: {len(raw)} bytes do not hold {n} ids")
    ids = np.frombuffer(raw[_HEADER_BYTES:], dtype=dt).astype(np.int64)
    return {"version": int(h[1]), "n_tokens": n, "vocab_size": int(h[3])}, ids


@dataclass
class TokensManifest:
    tokenizer_id: str
    tokenizer_sha256: str
    vocab_size: int
    doc_sep_id: Optional[int]
    dataset: str
    version: str
    corpus_manifest_sha256: str
    files: list[dict[str, Any]]
    val_bytes_per_token: Optional[float]

    def to_json(self) -> dict[str, Any]:
        """The schema object, in schema order."""
        return {
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_sha256": self.tokenizer_sha256,
            "vocab_size": self.vocab_size,
            "doc_sep_id": self.doc_sep_id,
            "dataset": self.dataset,
            "version": self.version,
            "corpus_manifest_sha256": self.corpus_manifest_sha256,
            "files": self.files,
        }


class _Bytes:
    """The identity byte tokenizer (D32)."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [list(t.encode("utf-8")) for t in texts]

    vocab_size = 256


class _Packer:
    """Appends whole documents to <split>-NNNNN.bin, opening the next file
    before a document that would push the current one past the limit."""

    def __init__(self, out: Path, split: str, vocab_size: int, limit: int) -> None:
        self.out, self.split, self.vocab, self.limit = out, split, vocab_size, limit
        self.version, self.dtype = _width(vocab_size)
        self.files: list[dict[str, Any]] = []
        self.f = None
        self._open()

    def _open(self) -> None:
        name = f"{self.split}-{len(self.files):05d}.bin"
        self.f = open(self.out / name, "wb")
        self.f.write(_header(self.version, 0, self.vocab))
        self.files.append(
            {
                "name": name,
                "split": self.split,
                "n_tokens": 0,
                "n_docs": 0,
                "sha256": "",
            }
        )

    def _close(self) -> None:
        cur = self.files[-1]
        self.f.seek(0)
        self.f.write(_header(self.version, cur["n_tokens"], self.vocab))
        self.f.close()
        h = hashlib.sha256((self.out / cur["name"]).read_bytes())
        cur["sha256"] = h.hexdigest()

    def add(self, ids: list[int]) -> None:
        n = len(ids)
        cur = self.files[-1]
        if cur["n_docs"] and cur["n_tokens"] + n > self.limit:
            self._close()
            self._open()
            cur = self.files[-1]
        if cur["n_tokens"] + n > _MAX_IDS:
            raise ValueError("a .bin file cannot hold more than 2^31 - 1 ids")
        arr = np.asarray(ids, dtype=np.int64)
        if n and (arr.min() < 0 or arr.max() >= self.vocab):
            raise ValueError(f"the tokenizer produced an id outside [0, {self.vocab})")
        self.f.write(arr.astype(self.dtype).tobytes())
        cur["n_tokens"] += n
        cur["n_docs"] += 1

    def finish(self) -> list[dict[str, Any]]:
        self._close()
        return self.files


def tokenize_shards(
    manifest: Path,
    tokenizer_json: Optional[Path],
    out: Path,
    *,
    tokenizer_id: str,
    doc_sep_id: Optional[int] = None,
    max_file_tokens: int = MAX_FILE_TOKENS,
) -> TokensManifest:
    """Tokenize the corpus of `manifest` into out; return the manifest."""
    if max_file_tokens < 1:
        raise ValueError(f"max_file_tokens must be >= 1, got {max_file_tokens}")
    manifest = Path(manifest)
    corpus = json.loads(manifest.read_text())
    if tokenizer_json is None:
        tok: Any = _Bytes()
        tok_sha = hashlib.sha256(b"").hexdigest()
    else:
        tokenizer_json = Path(tokenizer_json)
        tok = BPETokenizer.from_hf_json(str(tokenizer_json))
        tok_sha = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
        gen = tokenizer_json.parent / "generation_config.json"
        if doc_sep_id is None and gen.is_file():
            eos = json.loads(gen.read_text()).get("eos_token_id")
            doc_sep_id = eos if isinstance(eos, int) else None
    vocab = int(tok.vocab_size)
    if doc_sep_id is not None and not 0 <= doc_sep_id < vocab:
        raise ValueError(
            f"doc_sep_id {doc_sep_id} is outside the vocabulary [0, {vocab})"
        )
    sep = [] if doc_sep_id is None else [doc_sep_id]
    out = Path(out)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    files: list[dict[str, Any]] = []
    val_texts: list[str] = []
    for split in ("train", "val"):
        packer = _Packer(tmp, split, vocab, max_file_tokens)
        batch: list[str] = []

        def flush() -> None:
            for ids in tok.encode_batch(batch):
                packer.add(sep + list(ids))
            batch.clear()

        for doc in read_shards(manifest.parent, split=split):
            batch.append(doc.text)
            if split == "val":
                val_texts.append(doc.text)
            if len(batch) >= _BATCH:
                flush()
        flush()
        files += packer.finish()
    files.sort(key=lambda f: f["name"])
    val_ids = sum(
        f["n_tokens"] - len(sep) * f["n_docs"] for f in files if f["split"] == "val"
    )
    val_bpt = bytes_per_token(tok, val_texts) if val_ids else None
    result = TokensManifest(
        tokenizer_id=tokenizer_id,
        tokenizer_sha256=tok_sha,
        vocab_size=vocab,
        doc_sep_id=doc_sep_id,
        dataset=corpus["dataset"],
        version=corpus["version"],
        corpus_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        files=files,
        val_bytes_per_token=val_bpt,
    )
    (tmp / "_MANIFEST.json").write_text(json.dumps(result.to_json(), indent=2) + "\n")
    if out.exists():
        shutil.rmtree(out)
    os.rename(tmp, out)
    return result
