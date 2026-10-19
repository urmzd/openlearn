# Tokenize and pack to llm.c .bin

## Overview

| | |
|---|---|
| **Module** | `data.07` · build · Python · Pass 3 · 2 to 3 h |
| **Solution** | [`python/corpus/tokenize.py`](../../../../course/solutions/python/corpus/tokenize.py): `write_bin`, `read_bin`, `TokensManifest` (with `to_json`), `tokenize_shards`, and the constants `MAGIC`, `HEADER_INTS`, `MAX_FILE_TOKENS` |
| **Contract** | [`course/contracts/py/corpus/tokenize.pyi`](../../../../course/contracts/py/corpus/tokenize.pyi) · files: [`formats/tokens-bin.md`](../../../../course/contracts/formats/tokens-bin.md), [`formats/tokens-manifest.schema.json`](../../../../course/contracts/formats/tokens-manifest.schema.json) |
| **Needs** | `L1.2` Python `BPETokenizer` · `L1.6` `bytes_per_token` · `data.06` `read_shards` · reading: `L0.6` (the reader of these files), `L1.5` (Rust parity implementation) |
| **Used by** | `data.08` reports the token counts in the datasheet · later: `C1` trains on these files through `L0.6`'s `TokenStream` |
| **Optional depth** | Karpathy, [llm.c](https://github.com/karpathy/llm.c) `dev/data/data_common.py` (the format, MIT); Hugging Face [tokenizers](https://github.com/huggingface/tokenizers) `encode_batch` (free) |

## Key Takeaways

- A token stream is a flat array of ids behind a 1024-byte header; the trainer memory-maps it and never sees text, so this file is the whole interface between data and training.
- Every token the tokenizer produces lands in exactly one file of its document's split, preceded by the separator when the tokenizer has one: train and val never mix, and the counts in the manifest are exact.
- Files hold whole documents, so a reader of one file never starts mid-document.
- The val split's bytes per token is the factor that turns validation loss per token into bits per byte, the unit `C1` compares tokenizers in.

## How to read this chapter

Sections 1 to 3 build the llm.c `.bin` layout, the separator rule, file rollover, and bits per byte from first principles and pack three documents by hand, byte by byte. Section 4 fixes the interface and lists the cases a correct solution must handle, including the model property tests. Section 5 walks through `tokenize.py` and explains why each choice is the right one. Try to predict each step of the solution before reading it.

---

## 1. Why now

After `data.06` the corpus is clean Parquet text, and the tokenizers (`L1.2` in Python, `L1.5` in Rust) can turn text into ids. The training loop does not want either: it wants to grab a random window of $T + 1$ consecutive ids in microseconds, millions of times, without parsing anything. Today `C1` would have to tokenize the corpus at every step, or someone would write a one-off script whose output nobody can check. This module turns each split into llm.c `.bin` files, the format `L0.6`'s `TokenStream` memory-maps, with a manifest that records every file's token count, document count, and hash, so a training run can prove which data it saw.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | vocabulary size of the tokenizer | `int` |
| $w$ | bytes per id: 2 (`uint16`) when $V \le 65536$, else 4 (`uint32`) | `int` |
| $n$ | number of ids in one file | `int`, at most $2^{31} - 1$ |
| $e$ | the document separator id, or none | `int` or `None` |
| $M$ | `max_file_tokens`, the rollover limit (100,000,000) | `int` |
| $\beta$ | bytes per token of the val split: total UTF-8 bytes over total tokens | `float` |

**The layout.** A file is a header of 256 little-endian `int32` values, `[20240520, version, n, V, 0, ..., 0]` (1024 bytes), then the $n$ ids, little-endian, $w$ bytes each. Version 1 means `uint16` ids and requires $V \le 65536$ (ids 0 to 65535); version 2 means `uint32`. The file size is exactly $1024 + n w$, which is how a reader detects truncation. Writing an id outside $[0, V)$ is an error: in `uint16`, 65536 would silently wrap to 0.

**Tokenizers.** With no `tokenizer.json`, the tokenizer is the byte tokenizer of the tracer (D32): ids are the UTF-8 bytes, $V = 256$, no separator. With one, it is the Python `BPETokenizer` from `L1.2`, which encodes each document; Parity with the Rust implementation is the subject of `L1.5`. No special tokens are added by the encoder.

**The separator precedes.** When the tokenizer defines an end-of-text id $e$ (the `eos_token_id` of the `generation_config.json` next to `tokenizer.json`, or `doc_sep_id` when one is passed), every document is written as $[e, \text{ids}\dots]$. Preceding, as llm.c does, means every window that starts at a document start sees the separator first, the same signal the model gets at generation time. The byte tokenizer has no separator.

**Splits and files.** For each split, train then val, documents come from `read_shards(corpus, split)` in manifest order and are appended to `<split>-00000.bin`. Before a document that would push the file past $M$ ids, the next file starts (`-00001`, ...): files hold whole documents, and a single document longer than $M$ fills a file alone. Every split has at least its `-00000` file, possibly with $n = 0$, so a consumer can always open it.

**Conservation.** For each split, the manifest's $n$ summed over files equals the sum over that split's documents of (its token count plus one separator), and its document counts sum to the split's documents. Nothing is lost, duplicated, or moved across splits.

**Bits per byte.** A model's validation loss is in nats per token, which depends on the tokenizer: a tokenizer with bigger tokens has a higher loss per token on the same text. Dividing by $\ln 2$ and by $\beta$ gives bits per byte of text, comparable across tokenizers (`M11.2`). $\beta$ is exactly `L1.6`'s `bytes_per_token` over the val texts (separators are not text), computed here once, while the val documents pass through.

**The manifest.** `_MANIFEST.json` names its inputs by hash: `tokenizer_sha256` (of `tokenizer.json`, or of the empty string for bytes) and `corpus_manifest_sha256` (of the corpus `_MANIFEST.json`). Files are listed sorted by name with their split, $n$, document count, and SHA-256. Like the shards, the output is written under `out.tmp` and renamed, and two runs give the same bytes.

## 3. Worked example by hand

Three documents through `data.06` with `val_permille` 100, then the byte tokenizer:

| id | text | SHA-256 first 8 bytes mod 1000 | split | ids |
|---|---|---|---|---|
| `a:0` | `Hi` | 147 | train | 72 105 |
| `a:1` | `é` | 51 (< 100) | val | 195 169 |
| `a:2` | `ok` | 814 | train | 111 107 |

`é` is one character but two UTF-8 bytes, `C3 A9`, so two tokens.

**train-00000.bin.** $n = 4$ ids, $V = 256$, version 1, so $1024 + 4 \cdot 2 = 1032$ bytes:

```
offset 0     88 d8 34 01     20240520 = 0x0134D888, little-endian
offset 4     01 00 00 00     version 1
offset 8     04 00 00 00     n = 4
offset 12    00 01 00 00     V = 256
offset 16    00 ... 00       1008 zero bytes, up to offset 1024
offset 1024  48 00 69 00 6f 00 6b 00     72 105 111 107 as uint16
```

**val-00000.bin.** $n = 2$: `c3 00 a9 00`. **$\beta$** for val: 2 bytes over 2 tokens $= 1.0$. With GPT-2 instead and separator 50256, "Hello world" and "Hi" become `[50256, 15496, 995, 50256, 17250]`: each document preceded by `<|endoftext|>`.

## 4. The interface

```python
# python/corpus/tokenize.py (the full contract is contracts/py/corpus/tokenize.pyi)
MAGIC: int                 # 20240520
HEADER_INTS: int           # 256
MAX_FILE_TOKENS: int       # 100_000_000

def write_bin(path: Path, ids: Sequence[int], vocab_size: int) -> None
def read_bin(path: Path) -> tuple[dict[str, int], NDArray]

@dataclass
class TokensManifest:      # the schema's fields, plus val_bytes_per_token
    def to_json(self) -> dict

def tokenize_shards(manifest: Path, tokenizer_json: Path | None, out: Path, *,
                    tokenizer_id: str, doc_sep_id: int | None = None,
                    max_file_tokens: int = MAX_FILE_TOKENS) -> TokensManifest
```

Each file is streamed: a header with $n = 0$, ids appended as documents arrive, then a seek back to write the real $n$. `BPETokenizer` is loaded only when a `tokenizer.json` is given, so the byte tokenizer needs no BPE files.

### What a correct solution must handle

| Case | Kind of check | What must hold | Why it matters downstream |
|---|---|---|---|
| The hand example | unit | the section 3 bytes, splits, counts, and $\beta$ | the format is exactly the one above |
| File layout | unit | the format page's 1030-byte example; version 2 above 65536; 65536 still version 1 | any llm.c-compatible reader opens the files |
| Bad ids | boundary | an id at or above $V$, a negative id, $V = 0$ raise | no silent wraparound in `uint16` |
| Bad files | boundary | wrong magic, version 3, short or long file raise | truncation is detected |
| GPT-2 golden ids | golden | 300 documents through the BPE tokenizer equal the Hugging Face ids, each after 50256 | `C1`'s tokens are GPT-2's tokens |
| Separator source | unit | the separator from `generation_config.json`; an explicit one wins; none without either; out-of-vocabulary raises | the model learns where documents start |
| Conservation | property | per split: token and document counts exact, the stream equals the split's documents in order, across file rollovers | no val text is trained on |
| Whole documents | boundary | rollover before the document that would cross the limit; a long document alone; no empty file first | a reader never starts mid-document |
| Manifest and empty split | conformance | schema-valid manifest, key order, input hashes, file hashes, an empty val file present | consumers can verify and always open both splits |
| Val bytes per token | unit | $\beta$ equals `L1.6`'s `bytes_per_token` on the val texts, separators excluded | bits per byte in `C1` |
| Determinism | property | repeated tokenization produces identical bytes | the corpus build compares output hashes |
| Atomic replace | fault | a stale `.tmp` is cleared; a failing run leaves the old output | retried activities are safe |

**The model property tests.** [`test_tokenize_props.py`](../../../../course/solutions/learner-tests/data.07/python/tests/data-07-tokenize/test_tokenize_props.py) covers the hand example; for any list of texts (Hypothesis), the byte streams of each split concatenate to exactly the split's documents, file by file and in order, with header counts that match; rollover keeps documents whole; `uint32` above 65536; header padding and the reader's checks; manifest fields; GPT-2 with a separator and with `generation_config.json`; a rerun that replaces the output. Each property pins down one pitfall of section 6.

## 5. The solution, step by step

### 5.1 From first principles

1. **The trainer memory-maps ids and never parses anything, so the file must be self-describing in a fixed-size header.** Magic, version, count, vocabulary size, padded to 1024 bytes, then raw little-endian ids. The size $1024 + nw$ is then a complete integrity check against truncation.
2. **The id width follows the vocabulary.** `uint16` holds 0 to 65535, exactly $V \le 65536$; anything larger needs `uint32`. Writing an id outside $[0, V)$ must fail, because `astype("<u2")` would wrap it silently.
3. **Files must be written as a stream, but the header holds the count.** So the packer writes a header with $n = 0$, appends ids as documents arrive, and seeks back to rewrite the header when the file closes.
4. **A reader must never start mid-document, and files must stay under $M$.** Roll over *before* a document that would cross the limit, but only if the current file already holds a document; otherwise a document longer than $M$ would produce an empty file and then still not fit.
5. **Train and val must never mix.** Each split is read separately with `read_shards(corpus, split)` into its own packer; no document is ever read for the wrong split.
6. **The separator goes before each document**, as in llm.c, so every document start in the stream is preceded by $e$, the signal the model sees at generation time.
7. **$\beta$ is a property of the text, not of the separators.** It is `bytes_per_token` over the val texts with the tokenizer itself, so it matches `L1.6` exactly.
8. **The output is identified by its inputs and its bytes.** The manifest names the tokenizer and corpus manifest by hash, lists files sorted by name with their hashes, and is swapped in from `out.tmp`.

### 5.2 The code, annotated

The header and the width rule ([`tokenize.py`](../../../../course/solutions/python/corpus/tokenize.py)):

```python
def _width(vocab_size: int) -> tuple[int, np.dtype]:
    if vocab_size < 1:
        raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
    # ids 0..65535 fit uint16 exactly when V <= 65536
    return (1, np.dtype("<u2")) if vocab_size <= 65536 else (2, np.dtype("<u4"))


def _header(version: int, n_tokens: int, vocab_size: int) -> bytes:
    h = np.zeros(HEADER_INTS, dtype="<i4")     # 256 little-endian int32 = 1024 bytes, zero padded
    h[0], h[1], h[2], h[3] = MAGIC, version, n_tokens, vocab_size
    return h.tobytes()
```

The reader checks everything the header promises:

```python
def read_bin(path: Path) -> tuple[dict[str, int], NDArray]:
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
    if n < 0 or len(raw) != _HEADER_BYTES + n * dt.itemsize:   # exact size: short AND long files fail
        raise ValueError(f"{path}: {len(raw)} bytes do not hold {n} ids")
    ids = np.frombuffer(raw[_HEADER_BYTES:], dtype=dt).astype(np.int64)
    return {"version": int(h[1]), "n_tokens": n, "vocab_size": int(h[3])}, ids
```

The packer streams whole documents and patches the header on close:

```python
class _Packer:
    def _open(self) -> None:
        name = f"{self.split}-{len(self.files):05d}.bin"
        self.f = open(self.out / name, "wb")
        self.f.write(_header(self.version, 0, self.vocab))   # n unknown yet: placeholder 0
        self.files.append({"name": name, "split": self.split,
                           "n_tokens": 0, "n_docs": 0, "sha256": ""})

    def _close(self) -> None:
        cur = self.files[-1]
        self.f.seek(0)
        self.f.write(_header(self.version, cur["n_tokens"], self.vocab))   # the real n
        self.f.close()
        cur["sha256"] = hashlib.sha256((self.out / cur["name"]).read_bytes()).hexdigest()

    def add(self, ids: list[int]) -> None:
        n = len(ids)
        cur = self.files[-1]
        # roll over BEFORE a document that would cross the limit, never before the first one
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
```

Every split's packer opens its `-00000` file in `__init__`, so an empty split still has a file with $n = 0$.

The driver: choose the tokenizer and separator, then one packer per split:

```python
    if tokenizer_json is None:
        tok: Any = _Bytes()                                      # ids = UTF-8 bytes, V = 256
        tok_sha = hashlib.sha256(b"").hexdigest()
    else:
        tokenizer_json = Path(tokenizer_json)
        tok = BPETokenizer.from_hf_json(str(tokenizer_json))     # loaded only when needed
        tok_sha = hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
        gen = tokenizer_json.parent / "generation_config.json"
        if doc_sep_id is None and gen.is_file():                 # an explicit doc_sep_id wins
            eos = json.loads(gen.read_text()).get("eos_token_id")
            doc_sep_id = eos if isinstance(eos, int) else None
    vocab = int(tok.vocab_size)
    if doc_sep_id is not None and not 0 <= doc_sep_id < vocab:
        raise ValueError(f"doc_sep_id {doc_sep_id} is outside the vocabulary [0, {vocab})")
    sep = [] if doc_sep_id is None else [doc_sep_id]
    # ... tmp directory as in data.06 ...
    for split in ("train", "val"):
        packer = _Packer(tmp, split, vocab, max_file_tokens)
        batch: list[str] = []

        def flush() -> None:
            for ids in tok.encode_batch(batch):                  # results in batch order
                packer.add(sep + list(ids))                      # separator PRECEDES the document
            batch.clear()

        for doc in read_shards(manifest.parent, split=split):    # only this split's documents
            batch.append(doc.text)                               # the text exactly as sharded
            if split == "val":
                val_texts.append(doc.text)
            if len(batch) >= _BATCH:
                flush()
        flush()
        files += packer.finish()
    files.sort(key=lambda f: f["name"])
    val_ids = sum(f["n_tokens"] - len(sep) * f["n_docs"] for f in files if f["split"] == "val")
    val_bpt = bytes_per_token(tok, val_texts) if val_ids else None   # text tokens only
```

The manifest is written as `json.dumps(result.to_json(), indent=2)` plus a newline, then `out.tmp` replaces `out`, exactly as the shard writer does. `to_json` leaves `val_bytes_per_token` out of the file (the schema has no such field); it is returned on the `TokensManifest` for the caller.

Full solution: [`tokenize.py`](../../../../course/solutions/python/corpus/tokenize.py).

### 5.3 Why this solution

- **Cost.** Tokenization dominates: one `encode_batch` call per 1,024 documents. Writing is linear in the ids. Memory is one batch of texts plus, for $\beta$, the val texts (a small share of the corpus by design, `val_permille`).
- **A raw flat array over a self-describing format** (Arrow, TFRecord) is what makes the trainer's window sampling a pointer offset into a memory map; the 1024-byte header is the minimum needed to make it checkable and compatible with llm.c.
- **Separator before, not after.** After-document separators give the same token counts but a different stream: the first window of each document never begins with the start signal, and the ids differ from llm.c's for the same data.
- **Seek-and-patch over buffering a file.** Holding up to 100M ids (200 to 400 MB) in memory to know $n$ first would work but is wasteful; patching the header costs one seek, and the `.tmp` directory hides the transient $n = 0$ header from readers.
- **Accepted tradeoff.** Separators make per-document boundaries visible but do not stop attention across documents inside a window; per-document attention masks (Going further) do, at the cost of an index file.

### 5.4 Explain it back

1. Why is version 1 (`uint16`) still correct at exactly $V = 65536$?
   <details><summary>Answer</summary>The largest id is $V - 1 = 65535$, the largest `uint16` value. Only $V \ge 65537$ produces an id that does not fit.</details>
2. Why does `_Packer.add` test `cur["n_docs"]` before rolling over?
   <details><summary>Answer</summary>So a file always receives at least one document before it can be closed. Without it, a document longer than $M$ arriving at an empty file would close that empty file and open another, leaving an empty file first and still not fitting; with it, the long document fills a file alone.</details>
3. How does `read_bin` detect a truncated file?
   <details><summary>Answer</summary>The header says $n$ and the version fixes $w$, so the file must be exactly $1024 + nw$ bytes. Any other length, shorter or longer, raises.</details>
4. Why is $\beta$ computed from the val texts and not from `n_tokens / bytes` of the val files?
   <details><summary>Answer</summary>The files include one separator per document, which is not text. Counting separators would make the token count too large, $\beta$ too small, and bits per byte too large. `bytes_per_token(tok, val_texts)` uses only the text's own tokens, matching `L1.6`.</details>
5. What would go wrong if a single `read_shards(corpus)` pass fed both packers by checking each row's split?
   <details><summary>Answer</summary>Done right it would still separate the splits, but it couples the two streams and makes the conservation property depend on that check; the common bug is to read every document for every split, which trains on val text. Reading each split with `split=` makes the separation structural.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. always `uint16` | a vocabulary above 65536 wraps ids silently | `_width` picks `<u4` and version 2 above 65536 |
| 2. rolling over after the limit, or before the first document | files exceed the limit, or an empty file appears first | rolls over before the crossing document, only when `n_docs > 0` |
| 3. writing in place | a crash leaves a file whose header says more ids than it holds | everything is written under `out.tmp` and renamed at the end |
| 4. the separator after each document | the first window of every document never sees the start signal; ids differ from llm.c's | `sep + list(ids)` |
| 5. reading every document for every split | val text is trained on | `read_shards(..., split=split)` per packer |
| 6. sorting a batch by length for speed and not restoring the order | documents interleave in the wrong order | `encode_batch` results are consumed in batch order; no reordering |
| 7. counting separators as tokens of the text | $\beta$ too small, bits per byte too large | `bytes_per_token(tok, val_texts)` on the texts alone |
| 8. stripping or normalizing the text again | ids differ from the corpus the ledger describes | encodes `doc.text` exactly as read from the shards |
| 9. running `python/corpus/__main__.py` as a script with this file named `tokenize.py` | the script's directory comes first on `sys.path`, this module shadows the standard library's `tokenize`, and `import asyncio` fails | the entry point ([`entry/python/corpus/__main__.py`](../../../../course/solutions/entry/python/corpus/__main__.py)) sets `sys.path[0]` to `python/` before other imports; `python -m corpus` also works |

## 7. Where it's used next


| Direction | Module | How it uses this |
|---|---|---|
| Back | `L1.2` | Python `BPETokenizer.from_hf_json` and `encode_batch` |
| Back | `L1.6` | `bytes_per_token` gives $\beta$ for the val split |
| Back | `data.06` | `read_shards(corpus, split)` is the input, in manifest order |
| Back | `L0.6` | the reader (`read_tokens_header`, `TokenStream`) these files are written for |
| Forward | `data.08` | the datasheet's Preprocessing section reports the tokenizer and the token count of each split |
| Forward | `C1` | the capstone trains on `train-*.bin` and evaluates on `val-*.bin`, in bits per byte |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `tokenize_shards` | llm.c `dev/data/*.py`, nanoGPT `prepare.py` | multiprocessing tokenization of FineWeb into 100M-token shards | llm.c `dev/data/fineweb.py` |
| the `.bin` format | Megatron-LM indexed datasets | an `.idx` file with document offsets, so samplers can respect document boundaries | Megatron `megatron/core/datasets/indexed_dataset.py` |
| the separator | packing with attention masks | per-document attention masks (no cross-document attention) instead of a separator | the document masking in Llama 3's training notes |
| `encode_batch` | Hugging Face tokenizers | Rayon-parallel batches with padding and truncation | `tokenizers/src/tokenizer/mod.rs` |
