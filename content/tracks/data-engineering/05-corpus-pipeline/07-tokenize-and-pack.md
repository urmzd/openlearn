<!-- ol:module data.07 -->
# Tokenize and pack to llm.c .bin

## Overview

| | |
|---|---|
| **Module** | `data.07` · build · Python · Pass 3 · 2 to 3 h |
| **You build** | `python/corpus/tokenize.py`: `write_bin`, `read_bin`, `TokensManifest` (with `to_json`), `tokenize_shards`, and the constants `MAGIC`, `HEADER_INTS`, `MAX_FILE_TOKENS` |
| **Contract** | [`course/contracts/py/corpus/tokenize.pyi`](../../../../course/contracts/py/corpus/tokenize.pyi) · files: [`formats/tokens-bin.md`](../../../../course/contracts/formats/tokens-bin.md), [`formats/tokens-manifest.schema.json`](../../../../course/contracts/formats/tokens-manifest.schema.json) |
| **Tests** | `course/tests/data.07/` (what they check: section 4); the GPT-2 oracle `course/fixtures/tok-gpt2/` (Hugging Face ids for 300 strings) |
| **Needs** | `L1.2` Python `BPETokenizer` · `L1.6` `bytes_per_token` · `data.06` `read_shards` (or `--ref-deps`) · reading: `L0.6` (the reader of these files), `L1.5` (Rust parity implementation) |
| **Used by** | `data.08` reports the token counts in the datasheet · later: `C1` trains on these files through `L0.6`'s `TokenStream` |
| **Milestone** | `MS-corpus` |
| **Optional depth** | Karpathy, [llm.c](https://github.com/karpathy/llm.c) `dev/data/data_common.py` (the format, MIT); Hugging Face [tokenizers](https://github.com/huggingface/tokenizers) `encode_batch` (free) |

## Key Takeaways

- A token stream is a flat array of ids behind a 1024-byte header; the trainer memory-maps it and never sees text, so this file is the whole interface between data and training (`test_hand_example`, `test_write_bin_layout`).
- Every token the tokenizer produces lands in exactly one file of its document's split, preceded by the separator when the tokenizer has one: train and val never mix, and the counts in the manifest are exact (`test_tokens_are_conserved_and_splits_never_mix`).
- Files hold whole documents, so a reader of one file never starts mid-document (`test_files_hold_whole_documents`).
- The val split's bytes per token is the factor that turns validation loss per token into bits per byte, the unit `C1` compares tokenizers in (`test_val_bytes_per_token`).

## How to work this chapter

```bash
ol start data.07              # stubs tokenize.py into python/corpus/
ol tests data.07              # the course test catalog
ol tdd red data.07            # rung R4: your property tests first
ol check data.07              # exit code is the verdict
ol check data.07 --ref-deps   # only if L1.5, L1.6, or data.06 is not passing yet
ol mutate data.07             # how many planted bugs your tests catch
ol diff  data.07              # after passing: your code against the reference
```

---

## 1. Why now

After `data.06` your corpus is clean Parquet text, and your tokenizers (`L1.2` in Python, `L1.5` in Rust) can turn text into ids. The training loop does not want either: it wants to grab a random window of $T + 1$ consecutive ids in microseconds, millions of times, without parsing anything. Today `C1` would have to tokenize the corpus at every step, or you would write a one-off script whose output nobody can check. This module turns each split into llm.c `.bin` files, the format `L0.6`'s `TokenStream` memory-maps, with a manifest that records every file's token count, document count, and hash, so a training run can prove which data it saw.

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

**Tokenizers.** With no `tokenizer.json`, the tokenizer is the byte tokenizer of the tracer (D32): ids are the UTF-8 bytes, $V = 256$, no separator. With one, it is the Python `BPETokenizer` from `L1.2`, which encodes each document; Rust parity is checked separately against frozen fixture rows. No special tokens are added by the encoder.

**The separator precedes.** When the tokenizer defines an end-of-text id $e$ (the `eos_token_id` of the `generation_config.json` next to `tokenizer.json`, or `doc_sep_id` when you pass one), every document is written as $[e, \text{ids}\dots]$. Preceding, as llm.c does, means every window that starts at a document start sees the separator first, the same signal the model gets at generation time. The byte tokenizer has no separator.

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

This is `test_hand_example`; the separator example is in `test_separator_from_generation_config`.

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

Stream each file: write a header with $n = 0$, append ids as documents arrive, then seek back and write the real $n$. Load `BPETokenizer` only when a `tokenizer.json` is given, so the byte tokenizer needs no BPE files.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example` | unit | the section 3 bytes, splits, counts, and $\beta$ | you and the tests agree on the format |
| `test_write_bin_layout` | unit | the format page's 1030-byte example; version 2 above 65536; 65536 still version 1 | any llm.c-compatible reader opens your files |
| `test_write_bin_rejects_bad_ids` | boundary | an id at or above $V$, a negative id, $V = 0$ | no silent wraparound in `uint16` |
| `test_read_bin_rejects_bad_files` | boundary | wrong magic, version 3, short or long file | truncation is detected |
| `test_gpt2_golden` | golden | 300 documents through your Rust BPE equal the Hugging Face ids, each after 50256 | `C1`'s tokens are GPT-2's tokens |
| `test_separator_from_generation_config` | unit | the separator from `generation_config.json`; an explicit one wins; none without either; out-of-vocabulary raises | the model learns where documents start |
| `test_tokens_are_conserved_and_splits_never_mix` | property | per split: token and document counts exact, the stream equals the split's documents in order, across file rollovers | no val text is trained on |
| `test_files_hold_whole_documents` | boundary | rollover before the document that would cross the limit; a long document alone; no empty file first | a reader never starts mid-document |
| `test_manifest_and_empty_split` | conformance | schema-valid manifest, key order, input hashes, file hashes, an empty val file present | consumers can verify and always open both splits |
| `test_val_bytes_per_token` | unit | $\beta$ equals `L1.6`'s `bytes_per_token` on the val texts, separators excluded | bits per byte in `C1` |
| `test_output_is_deterministic` | property | repeated tokenization produces identical bytes | MS-corpus compares output hashes |
| `test_atomic_replace` | fault | a stale `.tmp` is cleared; a failing run leaves the old output | retried activities are safe |

**Your tests (rung R4, properties).** Under `python/tests/data-07-tokenize/`, failing first against the stubs: the hand example; for any list of texts (Hypothesis), the byte streams of each split concatenate to exactly the split's documents, file by file and in order, with header counts that match; rollover keeps documents whole; `uint32` above 65536; header padding and the reader's checks; manifest fields; GPT-2 with a separator and with `generation_config.json`; a rerun that replaces the output. `ol mutate data.07` grades them: 0.80 of the mutants, and every Pitfall below.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. always `uint16` | a vocabulary above 65536 wraps ids silently | `test_write_bin_layout` (mutant `s04`) |
| 2. rolling over after the limit, or before the first document | files exceed the limit, or an empty file appears first | `test_files_hold_whole_documents` (mutants `s09`, `s10`) |
| 3. writing in place | a crash leaves a file whose header says more ids than it holds | `test_atomic_replace` (mutant `s15`) |
| 4. the separator after each document | the first window of every document never sees the start signal; ids differ from llm.c's | `test_gpt2_golden` (mutant `s02`) |
| 5. reading every document for every split | val text is trained on | `test_tokens_are_conserved_and_splits_never_mix` (mutant `s03`) |
| 6. sorting a batch by length for speed and not restoring the order | documents interleave in the wrong order | `test_gpt2_golden` (mutant `s14`) |
| 7. counting separators as tokens of the text | $\beta$ too small, bits per byte too large | `test_val_bytes_per_token` (mutant `s13`) |
| 8. stripping or normalizing the text again | ids differ from the corpus the ledger describes | `test_gpt2_golden` (mutant `s05`) |
| 9. running `python/corpus/__main__.py` as a script with this file named `tokenize.py` | the script's directory comes first on `sys.path`, your module shadows the standard library's `tokenize`, and `import asyncio` fails; put `python/` in `sys.path[0]` before other imports, or run `python -m corpus` | your CLI, in the MS-corpus steps (an entry point, not a unit) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L1.2` | Python `BPETokenizer.from_hf_json` and `encode_batch` |
| Back | `L1.6` | `bytes_per_token` gives $\beta$ for the val split |
| Back | `data.06` | `read_shards(corpus, split)` is the input, in manifest order |
| Back | `L0.6` | the reader (`read_tokens_header`, `TokenStream`) these files are written for |
| Forward | `data.08` | the datasheet's Preprocessing section reports the tokenizer and the token count of each split |
| Forward | `C1` | the capstone trains on `train-*.bin` and evaluates on `val-*.bin`, in bits per byte |

If you skip this module, `ol check data.08` stops with `data.08 needs data.07`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `tokenize_shards` | llm.c `dev/data/*.py`, nanoGPT `prepare.py` | multiprocessing tokenization of FineWeb into 100M-token shards | llm.c `dev/data/fineweb.py` |
| the `.bin` format | Megatron-LM indexed datasets | an `.idx` file with document offsets, so samplers can respect document boundaries | Megatron `megatron/core/datasets/indexed_dataset.py` |
| the separator | packing with attention masks | per-document attention masks (no cross-document attention) instead of a separator | the document masking in Llama 3's training notes |
| `encode_batch` | Hugging Face tokenizers | Rayon-parallel batches with padding and truncation | `tokenizers/src/tokenizer/mod.rs` |
