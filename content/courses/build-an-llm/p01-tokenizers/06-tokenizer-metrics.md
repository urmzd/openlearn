<!-- ol:module L1.6 -->
# Tokenizer metrics

## Overview

| | |
|---|---|
| **Module** | `L1.6` · build · Python · Pass 3 · 2 h |
| **You build** | `python/tinyllm/tok/metrics.py`: `fertility`, `bytes_per_token`, `is_fallback`, `byte_fallback_rate` |
| **Contract** | [`course/contracts/py/tinyllm/tok/metrics.pyi`](../../../../course/contracts/py/tinyllm/tok/metrics.pyi) |
| **Tests** | `course/tests/L1.6/` (what they check: section 4) · your own tests in `python/tests/l1-6-metrics/`, rung R2, graded by mutation (threshold 0.60) |
| **Needs** | `L1.1` the protocol (the metrics take any `Tokenizer`) · the course tests also load real tokenizers with your `L1.2` (GPT-2, SmolLM2), `L1.3` (BERT), and `L1.4` (a Unigram), and turn bytes per token into bits per byte with `M11.2`'s `NLLAccumulator` · reading: `M00.1` units of information (or `--ref-deps`) |
| **Used by** | no call site registered yet: `C1` (the vocabulary ADR and the BPE vs Unigram ablation) and `data.07` (token statistics in the corpus manifest) call it when they arrive |
| **Milestone** | `MS-L1` (its pass line reports `bytes_per_token`) |
| **Optional depth** | Rust et al., *How Good is Your Tokenizer?* (ACL 2021), on fertility and continued words; Petrov et al., *Language Model Tokenizers Introduce Unfairness Between Languages* (NeurIPS 2023) |

## Key Takeaways

- Three numbers describe what a vocabulary does to text: fertility (tokens per word), bytes per token (compression), and the fallback rate (tokens that are not whole characters) (`test_hand_example_metrics`).
- Every one is a ratio of totals over the corpus, counted in UTF-8 bytes and without special tokens; a mean of per-text ratios or a count in code points is a different, wrong number (`test_ratio_of_totals_not_mean_of_ratios`, `test_bytes_are_utf8_not_code_points`, `test_no_special_tokens_are_added`).
- Bytes per token is the exchange rate between a model's loss per token and its bits per byte, which is why only bits per byte compares models with different tokenizers (`test_bits_per_byte_is_tokenizer_independent`).
- The metrics see only the `Tokenizer` protocol, so your BPE, WordPiece, and Unigram, and the Rust tokenizer of `L1.5`, are measured by the same code and agree with Hugging Face's numbers (`test_golden_metrics`, `test_any_protocol_object_is_measured`).

## How to work this chapter

```bash
ol start L1.6              # stubs metrics.py into your repo
ol tests L1.6              # read the test catalog first
ol check L1.6              # exit code is the verdict; then grades your tests by mutation
ol check L1.6 --ref-deps   # only if your L1.2, L1.3, L1.4, or M11.2 is not passing yet
ol diff  L1.6              # after passing: your code against the reference
```

---

## 1. Why now

You now have five tokenizers behind one protocol: bytes and char (`L1.1`), byte-level BPE (`L1.2`), WordPiece (`L1.3`), and Unigram (`L1.4`). The capstone must pick one, at one vocabulary size, and write the choice down in an ADR. "BPE felt better" is not a reason. The first evidence is cheap and needs no model: how many tokens a text costs (that is your context window and your training compute), how much a token carries, and how often the tokenizer falls back to pieces of characters or to `<unk>` (that is text the model cannot see properly). The second evidence, bits per byte after training, is only comparable across tokenizers because of one of these numbers. This module computes them, for any tokenizer, exactly as Hugging Face would.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $T$ | a tokenizer (the `L1.1` protocol) | `Tokenizer` |
| $t_1, \dots, t_m$ | the evaluation texts | `list[str]` |
| $w_1, \dots, w_k$ | the evaluation words, each encoded alone | `list[str]` |
| $\lvert T(x) \rvert$ | the number of ids of `T.encode(x)`, no special tokens | `int` |
| $B(x)$ | the number of UTF-8 bytes of $x$ | `int` |
| $\mathrm{fb}(i)$ | 1 when id $i$ is a fallback token, else 0 | `int` |
| $\ell$ | a model's total negative log-likelihood on the texts, in nats | `float` |

**Fertility** is the mean number of tokens per word, each word encoded on its own:

$$\mathrm{fertility} = \frac{1}{k}\sum_{j=1}^{k} \lvert T(w_j) \rvert.$$

A word that occurs twice in the list counts twice. Encoding the words joined by spaces would add the spaces' tokens and let BPE merge across the join, which is no longer "what does one word cost".

**Bytes per token** is the compression ratio:

$$\mathrm{bpt} = \frac{\sum_i B(t_i)}{\sum_i \lvert T(t_i) \rvert}.$$

It is a ratio of **totals**: averaging per-text ratios would weight a one-byte text like a thousand-byte one. It counts **bytes**, not code points: an emoji is one code point and four bytes, a CJK character one and three, so a code-point count understates what each token of non-ASCII text carries, and it is not the unit bits per byte divides by. The byte tokenizer scores exactly 1.0; a good English BPE scores around 4.

**Fallback tokens.** Id $i$ is a fallback when it is the unknown id, or when it is not a special token and `T.decode([i])` contains U+FFFD, the replacement character. For byte-level tokenizers that second case is a token holding only part of a multi-byte character: decoded alone, its bytes are not valid UTF-8. A special token is never a fallback, whatever its text. A vocabulary entry that is itself U+FFFD counts too: that text was already lost upstream. The rate is over token **occurrences**:

$$\mathrm{fallback\ rate} = \frac{\sum_i \sum_{j \in T(t_i)} \mathrm{fb}(j)}{\sum_i \lvert T(t_i) \rvert}.$$

A high rate means the model spends tokens spelling bytes; on a language the vocabulary barely covers, it can be most of the tokens.

**The exchange rate to bits per byte.** A model that pays $\ell$ nats over the texts has $\ell / \sum_i \lvert T(t_i) \rvert$ nats per token, a number that depends on how the tokenizer cut the text. Bits per byte (`M11.2`) divides by bytes instead, which every tokenizer agrees on:

$$\mathrm{bpb} = \frac{\ell}{\ln 2 \cdot \sum_i B(t_i)} = \frac{\text{bits per token}}{\mathrm{bpt}}.$$

A tokenizer with longer tokens has a higher loss per token and the same bpb for the same model quality. That is how `C1` compares its BPE and Unigram runs.

Each metric raises `ValueError` on an empty input (no words, or texts with no tokens): a metric over nothing has no value, and returning 0 or NaN would let an empty evaluation split pass a threshold.

**What real tokenizers score** on the 300 shared test strings of this part (emoji, CJK, accents, control characters, code), from `course/fixtures/L1.6/metrics.json`:

| Tokenizer | fertility | bytes per token | fallback rate |
|---|---|---|---|
| GPT-2 (50257, BPE) | 4.74 | 2.12 | 0.336 |
| SmolLM2 (49152, BPE) | 4.50 | 2.20 | 0.266 |
| bert-base-uncased (30522, WordPiece) | 3.62 | 2.91 | 0.046 |
| a Unigram trained on the 3 KB story (300) | 6.06 | 1.63 | 0.111 |

BERT looks best on compression, but its normalizer threw information away (case, accents, and every control character), and every unknown word is one `[UNK]`: compression alone is not the decision.

## 3. Worked example by hand

The text `"naïve café"`: 10 code points, and 12 UTF-8 bytes, because `ï` (U+00EF) is `c3 af` and `é` (U+00E9) is `c3 a9`. The words are `naïve` (6 bytes) and `café` (5 bytes).

**The byte tokenizer.** 12 tokens, one per byte:

- bytes per token $= 12 / 12 = 1.0$;
- fertility $= (6 + 5) / 2 = 5.5$;
- fallbacks: `c3`, `af`, `c3`, `a9` each decode alone to U+FFFD, so the rate is $4 / 12 \approx 0.333$.

**The char tokenizer trained on `["naïve"]`.** Its vocabulary is `<unk>`, `a`, `e`, `n`, `v`, `ï`. The text encodes to `n a ï v e <unk> <unk> a <unk> <unk>`: the space, `c`, `f`, and `é` were never seen. 10 tokens:

- bytes per token $= 12 / 10 = 1.2$;
- fertility: `naïve` is 5 tokens, `café` is `<unk> a <unk> <unk>`, 4 tokens, so $(5 + 4) / 2 = 4.5$;
- fallbacks: the four `<unk>`, so $4 / 10 = 0.4$.

**Bits per byte.** Suppose a model pays $12 \ln 2$ nats for this text, 12 bits over 12 bytes. Through the byte tokenizer that is $\ln 2$ nats per token, 1 bit per token, and $1 / 1.0 = 1$ bit per byte. Through the char tokenizer the same total is spread over 10 tokens: 1.2 bits per token, and $1.2 / 1.2 = 1$ bit per byte. Different loss per token, the same bits per byte.

These are the first case in section 4, `test_hand_example_metrics`, and the numbers of `test_bits_per_byte_is_tokenizer_independent`.

## 4. The interface

```python
# python/tinyllm/tok/metrics.py
def fertility(tok: Tokenizer, words: Sequence[str]) -> float
def bytes_per_token(tok: Tokenizer, texts: Sequence[str]) -> float
def is_fallback(tok: Tokenizer, i: int) -> bool
def byte_fallback_rate(tok: Tokenizer, texts: Sequence[str]) -> float
```

Every input is encoded with `tok.encode(x)`, never `add_special=True`. The functions use only `encode`, `decode`, `unk_id`, and `special_ids`, so any object with the `L1.1` protocol is measured, including one that is not a subclass of anything in `tinyllm`. Cache `is_fallback` per id inside `byte_fallback_rate`: a corpus repeats ids, and `decode` is the slow part.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_metrics` | unit | the section 3 numbers for bytes and char, exactly | you and the test agree on the definitions |
| `test_golden_metrics` | golden | GPT-2, SmolLM2, BERT, and a Unigram, loaded by your `L1.2` to `L1.4`, on 300 strings equal Hugging Face's numbers to 1e-12 | the `C1` ADR quotes numbers anyone can reproduce |
| `test_bytes_are_utf8_not_code_points` | boundary | an emoji is 4 bytes; emoji plus `日` over 2 tokens is 3.5 | non-English text is not flattered |
| `test_ratio_of_totals_not_mean_of_ratios` | unit | `["ab", "é"]` gives 4/3, not 1.5 | consistent with bits per byte, a ratio of totals |
| `test_words_are_encoded_one_at_a_time` | unit | `["ab", "c"]` in bytes is 1.5 | fertility is per word |
| `test_repeated_words_count_each_time` | unit | `["a", "a", "bb"]` is 4/3 | the word list is a sample, duplicates included |
| `test_no_special_tokens_are_added` | unit | a `<bos>`/`<eos>` tokenizer: `"hi"` costs 2, not 4 | templates are not text |
| `test_is_fallback_rules` | unit | partial bytes and `<unk>` are fallbacks; `a`, `<bos>`, GPT-2's whole `é`, and its end-of-text are not; a U+FFFD entry is | the definition the fallback rate counts |
| `test_fallback_counts_every_occurrence` | unit | `"éé"` in bytes is 1.0; `["aé", "é"]` is 4/5 | a rate over occurrences, not distinct ids |
| `test_empty_inputs_raise_value_error` | boundary | no words, no texts, texts with no tokens: `ValueError`; one token is enough | an empty split cannot pass a threshold |
| `test_bits_per_byte_is_tokenizer_independent` | property | the section 3 bpb through `M11.2`'s accumulator for both tokenizers, and bits per token over bytes per token | how `C1` compares tokenizers |
| `test_any_protocol_object_is_measured` | unit | a bare protocol object with no `tinyllm` base class | `L1.5`'s Rust tokenizer is measured the same way |

### Your tests (rung R2)

Write these in `python/tests/l1-6-metrics/test_metrics.py`, importing only names from the `tok` contracts:

```python
def test_bytes_per_token_counts_utf8_bytes():
    """ByteTokenizer on "naïve café" gives 1.0: 12 bytes, 12 tokens, not 10 code points."""
def test_bytes_per_token_is_a_ratio_of_totals():
    """Over ["ab", "é"] with a char tokenizer trained on "ab": 4 bytes / 3 tokens."""
def test_fertility_encodes_each_word_alone():
    """ByteTokenizer: fertility(["naïve", "café"]) is (6 + 5) / 2 = 5.5, no space tokens."""
def test_fertility_counts_repeated_words():
    """["a", "a", "bb"] in bytes is 4 / 3."""
def test_specials_are_not_added():
    """A char tokenizer with <bos> and <eos>: fertility(["hi"]) is 2, not 4."""
def test_unk_and_partial_bytes_are_fallbacks():
    """The unknown id and a lone UTF-8 lead byte are fallbacks; "a" and <bos> are not."""
def test_fallback_rate_counts_occurrences():
    """"éé" in bytes is 4 fallbacks of 4 tokens: rate 1.0; "naïve café" in chars is 0.4."""
def test_empty_inputs_raise():
    """No words, or texts with no tokens, are ValueError."""
```

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. `len(text)` for bytes | Chinese text looks three times less compressed than it is, and a bpb computed from it is wrong | `test_bytes_are_utf8_not_code_points` (mutant `s02`) |
| 2. encoding the words joined by spaces | fertility grows with the space tokens, or shrinks where BPE merges across words | `test_words_are_encoded_one_at_a_time` (mutant `s01`) |
| 3. `add_special=True` | every word costs two extra tokens; WordPiece looks worst for its template | `test_no_special_tokens_are_added` (mutant `s03`) |
| 4. a mean of per-text ratios | short texts dominate; the number disagrees with bits per byte | `test_ratio_of_totals_not_mean_of_ratios` (mutant `s08`) |
| 5. counting distinct fallback ids | a text full of one unknown character scores a low rate | `test_fallback_counts_every_occurrence` (mutant `s06`) |
| 6. forgetting `<unk>`, or counting specials | the char tokenizer's unknowns vanish from the rate; GPT-2's end-of-text counts as lost text | `test_is_fallback_rules` (mutants `s04`, `s05`) |
| 7. tokens per byte | the ratio is inverted and every comparison flips | `test_bits_per_byte_is_tokenizer_independent` (mutant `s07`) |
| 8. dividing by zero on empty input | a `ZeroDivisionError` deep in an evaluation job, or a silent 0 | `test_empty_inputs_raise_value_error` (mutants `s11`, `m01`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L1.1` | the protocol: `encode`, `decode`, `unk_id`, `special_ids` |
| Back | `L1.2` | the course tests load GPT-2 and SmolLM2 with `from_hf_json` |
| Back | `L1.3` | the course tests load bert-base-uncased with `from_hf_json` |
| Back | `L1.4` | the course tests load a Unigram with `from_hf_json` |
| Back | `M11.2` | `NLLAccumulator` turns a total loss and a byte count into bits per byte |
| Forward | `C1` | the vocabulary ADR and the BPE vs Unigram ablation report these numbers (arrives with B11) |
| Forward | `data.07` | the corpus manifest records bytes per token of each shard (arrives with B5) |

No call site is registered yet: `C1` and `data.07` add themselves to this module's `used_by` when their batches land, and until then `ol verify course L1.6` reports the missing call site.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `bytes_per_token` | tokenizer comparisons in model reports | characters per token and bytes per token across many languages, the "tokenizer tax" | Petrov et al. (2023), and the Llama 3 report's tokenizer section |
| `fertility` | multilingual tokenizer evaluation | fertility and the proportion of continued words per language | Rust et al. (2021), `tokenizer-eval` style suites |
| bits per byte | evaluation harnesses | bpb as the tokenizer-independent loss for comparing models | `lm-evaluation-harness` `bits_per_byte` metric |
