# Part 1: Tokenizers

From bytes to subwords. The tracer's byte tokenizer (256 ids, no training) gives way to the `Tokenizer` protocol and three trained families: byte-level BPE (GPT-2 compatible, loading Hugging Face files), WordPiece (BERT), and the Unigram LM (EM, Viterbi, subword sampling). Then BPE is implemented independently in Rust (`tl-tok`) with a streaming UTF-8 decoder; Python and Rust compare behavior using `tokenizer.json` and shared fixtures. The part ends with metrics that decide a vocabulary size for the capstone.

**Course passes**: L1.1 to L1.6 belong to [Pass 3](../paths/course-p03-tokens-and-data/milestone.md)

**Before you start**: the just-in-time math `M05.2`, `M06.2`, `M07.1`, `M07.2`, `M11.2` and the solve parts `S-M06b`, `S-M07b`; for `L1.5`, the [Rust primer](../../programming-languages/02-language-and-tool-primers/04-rust.md) and the Rust structures `ds.05` and `ds.06` in [Systems Data Structures](../../algorithms/16-systems-data-structures/).

## Key ideas

- **A tokenizer is a contract**: `encode(decode(ids)) == ids` and `decode(encode(text)) == text` for every string, including invalid UTF-8 split across a stream.
- **BPE** merges the most frequent adjacent pair until the vocabulary is full; encoding replays the merges by rank. A heap over pairs (`ds.06`) makes training fast.
- **Unigram** starts from a large vocabulary and prunes by likelihood loss; it can sample segmentations, which the capstone ablation (BPE vs Unigram) measures.
- **Compression is the first metric**: bytes per token and bits per byte on held-out text (`L1.6`).

## Modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `L1.1` | Tokenizer protocol, char tokenizer | build | 3 |
| `L1.2` | Byte-level BPE (GPT-2 compatible): hand-written pre-tokenizer, trainer, HF loader | build | 3 |
| `L1.3` | WordPiece (BERT basic tokenizer + greedy longest match) | build | 3 |
| `L1.4` | Unigram LM tokenizer (EM, Viterbi, subword sampling) | build | 3 |
| `L1.5` | Rust fast BPE (`tl-tok`) with a streaming UTF-8 decoder and fixture parity with Python | build | 3 |
| `L1.6` | Tokenizer metrics | build | 3 |

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `L1.1` | [Tokenizer protocol, char tokenizer](01-tokenizer-protocol-and-char.md) | build | 3 |
| 2 | `L1.2` | [Byte-level BPE (GPT-2 compatible): pre-tokenizer, trainer, HF loader](02-byte-level-bpe.md) | build | 3 |
| 3 | `L1.3` | [WordPiece (BERT basic tokenizer + greedy longest match)](03-wordpiece.md) | build | 3 |
| 4 | `L1.4` | [Unigram LM tokenizer (EM, Viterbi, subword sampling)](04-unigram-lm.md) | build | 3 |
| 5 | `L1.5` | [Rust fast BPE (tl-tok), byte tokenizer, streaming decoder](05-rust-fast-bpe.md) | build | 3 |
| 6 | `L1.6` | [Tokenizer metrics](06-tokenizer-metrics.md) | build | 3 |

## Going further

- Hugging Face [tokenizers](https://github.com/huggingface/tokenizers) (Rust BPE merges and the pre-tokenizer pipeline) and OpenAI [tiktoken](https://github.com/openai/tiktoken).
- Sennrich, Haddow, and Birch, [*Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909); Kudo, [*Subword Regularization*](https://arxiv.org/abs/1804.10959).
