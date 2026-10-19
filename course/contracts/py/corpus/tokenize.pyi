# contracts/py/corpus/tokenize.pyi (data.07): tokenize the shards and pack
# llm.c .bin token streams
# chapter: content/courses/data-engineering/05-corpus-pipeline/07-tokenize-and-pack.md
#
# Files: formats/tokens-bin.md (a 1024-byte header of 256 little-endian
# int32 [20240520, version, n_tokens, vocab_size, 0...], then the ids,
# uint16 for version 1, uint32 for version 2) and
# formats/tokens-manifest.schema.json.
#
# Tokenizers. tokenizer_json None is the byte tokenizer (D32: id = UTF-8
# byte, vocab_size 256, no separator, tokenizer_sha256 of b""). Otherwise
# the file loads through Python BPETokenizer (L1.2/L1.5) and documents are
# encoded with no special tokens added.
#
# Separator. doc_sep_id when given; else the int eos_token_id of
# generation_config.json in tokenizer_json's directory when that file
# exists; else none. With a separator every document is PRECEDED by it.
#
# Packing. For each split ("train", then "val") the documents of that split
# are read with corpus.shard.read_shards (data.06) in manifest order and
# appended to <split>-00000.bin; a document that would push the current
# file past max_file_tokens starts the next file (-00001, ...), so a file
# holds whole documents (one longer document fills a file alone). Every
# split gets at least its -00000 file, possibly with no tokens. Version 1
# when vocab_size <= 65536, else 2.
#
# Manifest. out/_MANIFEST.json with the schema's keys in its order
# (tokenizer_id, tokenizer_sha256, vocab_size, doc_sep_id, dataset, version,
# corpus_manifest_sha256, files [{name, split, n_tokens, n_docs, sha256}]),
# files sorted by name, written as json.dumps(m, indent=2) plus "\n".
# Output is byte-identical across runs. Written under out + ".tmp" and
# renamed, like data.06.
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from numpy.typing import NDArray

MAGIC: int  # 20240520
HEADER_INTS: int  # 256
MAX_FILE_TOKENS: int  # 100_000_000

def write_bin(path: Path, ids: Sequence[int], vocab_size: int) -> None:
    """One .bin file: header then ids, version 1 when vocab_size <= 65536,
    else 2. ValueError for an id outside [0, vocab_size), more than
    2^31 - 1 ids, or vocab_size < 1."""

def read_bin(path: Path) -> tuple[dict[str, int], NDArray]:
    """({"version", "n_tokens", "vocab_size"}, the ids as int64 [n_tokens]).
    ValueError for a wrong magic, an unknown version, or a file size that
    disagrees with n_tokens."""

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
    # Bytes per token of the val split (L1.6 bytes_per_token over its
    # texts, separators not counted): the factor that turns validation loss
    # in nats per token into bits per byte. None when val has no tokens.
    val_bytes_per_token: Optional[float]

    def to_json(self) -> dict[str, Any]:
        """The schema object: every field but val_bytes_per_token, in
        schema order."""

def tokenize_shards(
    manifest: Path,
    tokenizer_json: Optional[Path],
    out: Path,
    *,
    tokenizer_id: str,
    doc_sep_id: Optional[int] = None,
    max_file_tokens: int = ...,
) -> TokensManifest:
    """Tokenize the corpus whose _MANIFEST.json is `manifest` into out (the
    rules above); return the manifest. ValueError for max_file_tokens < 1 or
    a doc_sep_id outside the vocabulary."""
