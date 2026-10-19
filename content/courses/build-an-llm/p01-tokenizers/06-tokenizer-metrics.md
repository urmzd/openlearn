# Tokenizer metrics

## Overview

| | |
|---|---|
| **Module** | `L1.6` · build · Python · Pass 3 · 1 h reading |
| **Solution** | [`metrics.py`](../../../../course/solutions/python/tinyllm/tok/metrics.py): `fertility`, `bytes_per_token`, `is_fallback`, `byte_fallback_rate` · model tests [`test_metrics.py`](../../../../course/solutions/learner-tests/L1.6/python/tests/l1-6-metrics/test_metrics.py) |
| **Contract** | [`course/contracts/py/tinyllm/tok/metrics.pyi`](../../../../course/contracts/py/tinyllm/tok/metrics.pyi) |
| **Needs** | `L1.1` the protocol (the metrics take any `Tokenizer`) · the golden numbers load real tokenizers with `L1.2` (GPT-2, SmolLM2), `L1.3` (BERT), and `L1.4` (a Unigram), and turn bytes per token into bits per byte with `M11.2`'s `NLLAccumulator` · reading: `M00.1` units of information |
| **Used by** | `C1` (the vocabulary ADR and the BPE vs Unigram ablation) and `data.07` (token statistics in the corpus manifest) · the tokenizer layer's end-to-end run reports `bytes_per_token` |
| **Optional depth** | Rust et al., *How Good is Your Tokenizer?* (ACL 2021), on fertility and continued words; Petrov et al., *Language Model Tokenizers Introduce Unfairness Between Languages* (NeurIPS 2023) |

## Key Takeaways

- Three numbers describe what a vocabulary does to text: fertility (tokens per word), bytes per token (compression), and the fallback rate (tokens that are not whole characters).
- Every one is a ratio of totals over the corpus, counted in UTF-8 bytes and without special tokens; a mean of per-text ratios or a count in code points is a different, wrong number.
- Bytes per token is the exchange rate between a model's loss per token and its bits per byte, which is why only bits per byte compares models with different tokenizers.
- The metrics see only the `Tokenizer` protocol, so the BPE, WordPiece, and Unigram tokenizers, and the Rust tokenizer of `L1.5`, are measured by the same code and agree with Hugging Face's numbers.

## How to read this chapter

Sections 1 to 3 define the three metrics from first principles and compute them by hand on one short text. Section 4 fixes the interface. Section 5 walks through the solution, which is short, and explains why each line is the right one. Try to predict each function's body from its definition before reading it.

---

## 1. Why now

The system now has five tokenizers behind one protocol: bytes and char (`L1.1`), byte-level BPE (`L1.2`), WordPiece (`L1.3`), and Unigram (`L1.4`). The capstone must pick one, at one vocabulary size, and write the choice down in an ADR. "BPE felt better" is not a reason. The first evidence is cheap and needs no model: how many tokens a text costs (that is the context window and the training compute), how much a token carries, and how often the tokenizer falls back to pieces of characters or to `<unk>` (that is text the model cannot see properly). The second evidence, bits per byte after training, is only comparable across tokenizers because of one of these numbers. This module computes them, for any tokenizer, exactly as Hugging Face would.

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

**What real tokenizers score** on the 300 shared test strings of this part (emoji, CJK, accents, control characters, code), from the frozen metrics fixture, which equals Hugging Face's numbers to 1e-12:

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

These are the first case in section 4, and the bits-per-byte numbers are the tokenizer-independence case.

## 4. The interface

```python
# python/tinyllm/tok/metrics.py
def fertility(tok: Tokenizer, words: Sequence[str]) -> float
def bytes_per_token(tok: Tokenizer, texts: Sequence[str]) -> float
def is_fallback(tok: Tokenizer, i: int) -> bool
def byte_fallback_rate(tok: Tokenizer, texts: Sequence[str]) -> float
```

Every input is encoded with `tok.encode(x)`, never `add_special=True`. The functions use only `encode`, `decode`, `unk_id`, and `special_ids`, so any object with the `L1.1` protocol is measured, including one that is not a subclass of anything in `tinyllm`. `byte_fallback_rate` caches `is_fallback` per id: a corpus repeats ids, and `decode` is the slow part.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand example | unit | the section 3 numbers for bytes and char, exactly | the definitions are the ones the chapter states |
| Golden metrics | golden | GPT-2, SmolLM2, BERT, and a Unigram, loaded by `L1.2` to `L1.4`, on 300 strings equal Hugging Face's numbers to 1e-12 | the `C1` ADR quotes numbers anyone can reproduce |
| Bytes are UTF-8, not code points | boundary | an emoji is 4 bytes; emoji plus `日` over 2 tokens is 3.5 | non-English text is not flattered |
| A ratio of totals, not a mean of ratios | unit | `["ab", "é"]` gives 4/3, not 1.5 | consistent with bits per byte, a ratio of totals |
| Words are encoded one at a time | unit | `["ab", "c"]` in bytes is 1.5 | fertility is per word |
| Repeated words count each time | unit | `["a", "a", "bb"]` is 4/3 | the word list is a sample, duplicates included |
| No special tokens are added | unit | a `<bos>`/`<eos>` tokenizer: `"hi"` costs 2, not 4 | templates are not text |
| The fallback rules | unit | partial bytes and `<unk>` are fallbacks; `a`, `<bos>`, GPT-2's whole `é`, and its end-of-text are not; a U+FFFD entry is | the definition the fallback rate counts |
| Fallbacks count every occurrence | unit | `"éé"` in bytes is 1.0; `["aé", "é"]` is 4/5 | a rate over occurrences, not distinct ids |
| Empty inputs raise `ValueError` | boundary | no words, no texts, texts with no tokens: `ValueError`; one token is enough | an empty split cannot pass a threshold |
| Bits per byte is tokenizer independent | property | the section 3 bpb through `M11.2`'s accumulator for both tokenizers, and bits per token over bytes per token | how `C1` compares tokenizers |
| Any protocol object is measured | unit | a bare protocol object with no `tinyllm` base class | `L1.5`'s Rust tokenizer is measured the same way |

The model tests in [`test_metrics.py`](../../../../course/solutions/learner-tests/L1.6/python/tests/l1-6-metrics/test_metrics.py) pin the hand numbers: `ByteTokenizer` on `"naïve café"` is 1.0 bytes per token and a char tokenizer trained on `"naïve"` is 1.2; `["ab", "é"]` under a char tokenizer trained on `"ab"` is 4/3; fertility of `["naïve", "café"]` in bytes is 5.5 and of `["a", "a", "bb"]` is 4/3; a `<bos>`/`<eos>` tokenizer costs 2 for `"hi"`; `"éé"` in bytes has fallback rate 1.0; empty inputs raise.

## 5. The solution, step by step

### 5.1 From first principles

1. **Measure what the model will see.** The model sees `tok.encode(x)` with no template. So every count is `len(tok.encode(x))`, never with `add_special=True`: a `<bos>` and `<eos>` are the template's cost, not the text's.
2. **A word's cost is the word alone.** "What does one word cost" means encoding that word in isolation; joining words adds space tokens and lets merges cross word boundaries. So fertility encodes each word separately and averages over the list, duplicates included, because the list is a sample of word occurrences.
3. **Compression is total bytes over total tokens.** Bits per byte is (total nats) / (ln 2 · total bytes), a ratio of totals; for bytes per token to be its exchange rate it must also be a ratio of totals, in the same unit, UTF-8 bytes. A mean of per-text ratios weights texts equally, not bytes, and breaks $\mathrm{bpb} = \text{bits per token} / \mathrm{bpt}$.
4. **A fallback is a token that is not whole characters.** The protocol exposes `decode`, `unk_id`, and `special_ids`, nothing about bytes, so the only tokenizer-independent test is: is it the unknown id, or does decoding it alone produce U+FFFD? Every other special token is excluded, because its text is a control symbol, not lost text.
5. **A rate over occurrences.** The question is "what share of the tokens the model sees are fallbacks", so the numerator counts every occurrence. Since `is_fallback` is a pure function of the id and `decode` is the expensive call, memoise it per id.
6. **No value without data.** Each ratio is undefined with a zero denominator; raising `ValueError` is the only answer that cannot let an empty evaluation split pass a threshold.

### 5.2 The code, annotated

```python
def fertility(tok: Tokenizer, words: Sequence[str]) -> float:
    if len(words) == 0:
        raise ValueError("fertility needs at least one word")
    # Each word alone, no specials; a repeated word counts every time.
    return sum(len(tok.encode(w)) for w in words) / len(words)
```

The denominator is the number of words, which is known before encoding, so the empty check comes first.

```python
def bytes_per_token(tok: Tokenizer, texts: Sequence[str]) -> float:
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)   # UTF-8 bytes, not len(t)
    n_tokens = sum(len(tok.encode(t)) for t in texts)      # totals, not a mean of ratios
    if n_tokens == 0:
        raise ValueError("bytes_per_token needs texts that encode to at least one token")
    return n_bytes / n_tokens                              # bytes per token, not tokens per byte
```

Here the denominator is the token total, so "empty" means "no tokens", which also covers `[]` and `[""]`. On `["ab", "é"]` with a char tokenizer trained on `"ab"`: 2 + 2 = 4 bytes over 2 + 1 = 3 tokens, 4/3; the mean of ratios would be $(1 + 2)/2 = 1.5$.

```python
def is_fallback(tok: Tokenizer, i: int) -> bool:
    if tok.unk_id is not None and i == tok.unk_id:   # <unk> is lost text, even though it is special
        return True
    if i in tok.special_ids.values():                # <bos>, <|endoftext|>: control, not text
        return False
    return "�" in tok.decode([i])               # a fragment of a multi-byte character
```

The order matters. The unknown id is usually also listed among the specials, so it must be tested first, or every `<unk>` would be excused as a special. A vocabulary entry that already decodes to U+FFFD is counted by the last line too, which is intended: that text was lost upstream.

```python
def byte_fallback_rate(tok: Tokenizer, texts: Sequence[str]) -> float:
    total = fallback = 0
    cache: dict[int, bool] = {}          # is_fallback depends only on the id
    for t in texts:
        for i in tok.encode(t):
            total += 1
            if i not in cache:
                cache[i] = is_fallback(tok, i)
            fallback += cache[i]         # every occurrence, not distinct ids
    if total == 0:
        raise ValueError("byte_fallback_rate needs texts that encode to at least one token")
    return fallback / total
```

On `"naïve café"` in bytes: 12 tokens, of which `c3 af c3 a9` decode alone to U+FFFD, so $4/12$. The cache turns $O(N)$ decodes into $O(\text{distinct ids})$.

Full solution: [`metrics.py`](../../../../course/solutions/python/tinyllm/tok/metrics.py).

### 5.3 Why this solution

- **Complexity.** Each function is one pass of encoding: $O(\sum_i B(t_i))$ for the byte counts plus the cost of `encode`. `byte_fallback_rate` adds one `decode` per distinct id and $O(\text{distinct ids})$ memory for the cache.
- **Why only the protocol?** Reaching into a BPE's byte table or a WordPiece's `##` prefix would give per-family definitions and exclude the Rust tokenizer. `decode([i])` containing U+FFFD is the one definition every tokenizer answers the same way, and it is exactly what Hugging Face computes.
- **Why not code points?** Bits per byte divides by bytes, so bytes per token in code points would not be its exchange rate, and it would understate what each token of CJK or emoji text carries by a factor of 3 or 4.
- **Why raise instead of returning 0 or NaN?** A 0 passes any "at most" threshold and a NaN fails comparisons silently. A `ValueError` stops an evaluation that measured nothing.
- **Tradeoff accepted.** `fertility` is "tokens per whitespace word supplied by the caller"; the solution does not segment words itself, so languages without spaces need a word list from elsewhere.

### 5.4 Explain it back

1. Why must `is_fallback` test `unk_id` before `special_ids`?
   <details><summary>Answer</summary>The unknown token is normally registered as a special token. Testing specials first would return `False` for every `<unk>`, and the char tokenizer's unknown characters would vanish from the rate.</details>
2. A corpus is one 1,000-byte English text (250 tokens) and one 3-byte `日` (3 byte tokens). What do the ratio of totals and the mean of ratios give, and which is right?
   <details><summary>Answer</summary>Totals: $1003 / 253 \approx 3.96$. Mean of ratios: $(4 + 1)/2 = 2.5$. The ratio of totals is right: it is the number that converts loss per token to bits per byte over this corpus; the mean lets a 3-byte text weigh as much as a 1,000-byte one.</details>
3. Why does a model with a higher loss per token not necessarily model text worse?
   <details><summary>Answer</summary>Loss per token depends on how much each token carries. $\mathrm{bpb} = \text{bits per token} / \mathrm{bpt}$: a tokenizer with longer tokens has higher bits per token and higher bpt, and the same bpb for the same model quality. Section 3 shows it: 1 bit per token at 1.0 bpt and 1.2 bits per token at 1.2 bpt are both 1 bit per byte.</details>
4. Why encode each word alone in `fertility`?
   <details><summary>Answer</summary>Joining words adds the separator's tokens and lets BPE merge across the join (GPT-2 attaches the space to the next word), so the count would no longer be the cost of a word. Each word alone is the definition.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. `len(text)` for bytes | Chinese text looks three times less compressed than it is, and a bpb computed from it is wrong | counts `len(t.encode("utf-8"))` |
| 2. encoding the words joined by spaces | fertility grows with the space tokens, or shrinks where BPE merges across words | encodes each word alone inside the sum |
| 3. `add_special=True` | every word costs two extra tokens; WordPiece looks worst for its template | calls `tok.encode(x)` with the default, no specials |
| 4. a mean of per-text ratios | short texts dominate; the number disagrees with bits per byte | sums bytes and tokens separately, divides once |
| 5. counting distinct fallback ids | a text full of one unknown character scores a low rate | adds `cache[i]` for every occurrence; the cache only saves `decode` calls |
| 6. forgetting `<unk>`, or counting specials | the char tokenizer's unknowns vanish from the rate; GPT-2's end-of-text counts as lost text | `is_fallback` tests `unk_id` first, then excludes `special_ids` |
| 7. tokens per byte | the ratio is inverted and every comparison flips | returns `n_bytes / n_tokens` |
| 8. dividing by zero on empty input | a `ZeroDivisionError` deep in an evaluation job, or a silent 0 | checks the denominator and raises `ValueError` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L1.1` | the protocol: `encode`, `decode`, `unk_id`, `special_ids` |
| Back | `L1.2` | the golden numbers load GPT-2 and SmolLM2 with `from_hf_json` |
| Back | `L1.3` | the golden numbers load bert-base-uncased with `from_hf_json` |
| Back | `L1.4` | the golden numbers load a Unigram with `from_hf_json` |
| Back | `M11.2` | `NLLAccumulator` turns a total loss and a byte count into bits per byte |
| Forward | `C1` | the vocabulary ADR and the BPE vs Unigram ablation report these numbers |
| Forward | `data.07` | the corpus manifest records bytes per token of each shard |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `bytes_per_token` | tokenizer comparisons in model reports | characters per token and bytes per token across many languages, the "tokenizer tax" | Petrov et al. (2023), and the Llama 3 report's tokenizer section |
| `fertility` | multilingual tokenizer evaluation | fertility and the proportion of continued words per language | Rust et al. (2021), `tokenizer-eval` style suites |
| bits per byte | evaluation harnesses | bpb as the tokenizer-independent loss for comparing models | `lm-evaluation-harness` `bits_per_byte` metric |
