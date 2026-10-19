# Tokenizer metrics

The system now has five tokenizers behind one protocol: bytes and char from [the tokenizer protocol](01-tokenizer-protocol-and-char.md), [byte-level BPE](02-byte-level-bpe.md), [WordPiece](03-wordpiece.md), and [Unigram](04-unigram-lm.md). The [TinyStories capstone](../capstones/01-tinystories.md) must pick one, at one vocabulary size, and write the choice down in an architecture decision record. "BPE felt better" is not a reason.

The first evidence is cheap and needs no model. How many tokens does a text cost? That is the context window and the training compute. How much does each token carry? And how often does the tokenizer fall back to pieces of characters or to `<unk>`? That is text the model cannot see properly. Three numbers answer these questions: fertility (tokens per word), bytes per token (compression), and the fallback rate (tokens that are not whole characters). The second kind of evidence, bits per byte after training, is only comparable across tokenizers because of one of these three numbers. This short section computes them for any tokenizer, exactly as Hugging Face would.

The metrics take any object that satisfies [the tokenizer protocol](01-tokenizer-protocol-and-char.md). The golden numbers load real tokenizers with the loaders of the three previous sections and turn bytes per token into bits per byte with the `NLLAccumulator` of [Perplexity, bits per byte, NLL accumulator](../../mathematics-for-computing/12-information-theory/02-perplexity-bits-per-byte-nll-accumulator.md) in *Mathematics for Computing*; [Exponents, logs, and units of information](../../mathematics-for-computing/01-precalculus/01-exponents-logs-and-units-of-information.md) is useful background on nats and bits. The solution is [`metrics.py`](../../../../course/solutions/python/tinyllm/tok/metrics.py) (`fertility`, `bytes_per_token`, `is_fallback`, `byte_fallback_rate`), and the contract is [`metrics.pyi`](../../../../course/contracts/py/tinyllm/tok/metrics.pyi).

> **Learning objectives**
> - Compute fertility, bytes per token, and the fallback rate by hand for a given tokenizer and text.
> - Explain why each metric is a ratio of totals counted in UTF-8 bytes, without special tokens.
> - Use bytes per token to convert loss per token into bits per byte, and explain why only bits per byte compares models with different tokenizers.
> - Define a fallback token using only the tokenizer protocol.

## Three numbers

The notation for this section:

| Symbol | Meaning | Type / shape |
|---|---|---|
| $T$ | a tokenizer (the protocol) | `Tokenizer` |
| $t_1, \dots, t_m$ | the evaluation texts | `list[str]` |
| $w_1, \dots, w_k$ | the evaluation words, each encoded alone | `list[str]` |
| $\lvert T(x) \rvert$ | the number of ids of `T.encode(x)`, no special tokens | `int` |
| $B(x)$ | the number of UTF-8 bytes of $x$ | `int` |
| $\mathrm{fb}(i)$ | 1 when id $i$ is a fallback token, else 0 | `int` |
| $\ell$ | a model's total negative log-likelihood on the texts, in nats | `float` |

### Fertility

**Fertility** is the mean number of tokens per word, each word encoded on its own:

$$\mathrm{fertility} = \frac{1}{k}\sum_{j=1}^{k} \lvert T(w_j) \rvert.$$

A word that occurs twice in the list counts twice. Encoding the words joined by spaces would add the spaces' tokens and let BPE merge across the join, which is no longer "what does one word cost".

### Bytes per token

**Bytes per token** is the compression ratio:

$$\mathrm{bpt} = \frac{\sum_i B(t_i)}{\sum_i \lvert T(t_i) \rvert}.$$

Two choices in that formula matter. It is a ratio of **totals**: averaging per-text ratios would weight a one-byte text like a thousand-byte one. And it counts **bytes**, not code points: an emoji is one code point and four bytes, a CJK character one and three, so a code-point count understates what each token of non-ASCII text carries, and it is not the unit bits per byte divides by. The byte tokenizer scores exactly 1.0; a good English BPE scores around 4.

### The fallback rate

Id $i$ is a **fallback** when it is the unknown id, or when it is not a special token and `T.decode([i])` contains U+FFFD, the replacement character. For byte-level tokenizers that second case is a token holding only part of a multi-byte character: decoded alone, its bytes are not valid UTF-8. A special token is never a fallback, whatever its text. A vocabulary entry that is itself U+FFFD counts too, because that text was already lost upstream. The rate is over token **occurrences**:

$$\mathrm{fallback\ rate} = \frac{\sum_i \sum_{j \in T(t_i)} \mathrm{fb}(j)}{\sum_i \lvert T(t_i) \rvert}.$$

A high rate means the model spends tokens spelling bytes; on a language the vocabulary barely covers, fallbacks can be most of the tokens.

Every metric raises `ValueError` on an empty input (no words, or texts with no tokens). A metric over nothing has no value, and returning 0 or NaN would let an empty evaluation split pass a threshold.

## The exchange rate to bits per byte

Why does compression matter beyond cost? Because it connects the loss of models with different tokenizers. A model that pays $\ell$ nats over the texts has $\ell / \sum_i \lvert T(t_i) \rvert$ nats per token, a number that depends on how the tokenizer cut the text. Bits per byte divides by bytes instead, which every tokenizer agrees on:

$$\mathrm{bpb} = \frac{\ell}{\ln 2 \cdot \sum_i B(t_i)} = \frac{\text{bits per token}}{\mathrm{bpt}}.$$

So bytes per token is the exchange rate between a model's loss per token and its bits per byte. A tokenizer with longer tokens has a higher loss per token and the same bpb for the same model quality. That is how the capstone compares its BPE and Unigram runs.

## What real tokenizers score

On the 300 shared test strings of this chapter (emoji, CJK, accents, control characters, code), the frozen metrics fixture, which equals Hugging Face's numbers to 1e-12, gives:

| Tokenizer | fertility | bytes per token | fallback rate |
|---|---|---|---|
| GPT-2 (50257, BPE) | 4.74 | 2.12 | 0.336 |
| SmolLM2 (49152, BPE) | 4.50 | 2.20 | 0.266 |
| bert-base-uncased (30522, WordPiece) | 3.62 | 2.91 | 0.046 |
| a Unigram trained on the 3 KB story (300) | 6.06 | 1.63 | 0.111 |

BERT looks best on compression, but its normalizer threw information away (case, accents, and every control character), and every unknown word is one `[UNK]`. Compression alone is not the decision.

## Working it by hand

Take the text `"naïve café"`: 10 code points and 12 UTF-8 bytes, because `ï` (U+00EF) is `c3 af` and `é` (U+00E9) is `c3 a9`. The words are `naïve` (6 bytes) and `café` (5 bytes).

The byte tokenizer gives 12 tokens, one per byte. Bytes per token is $12 / 12 = 1.0$. Fertility is $(6 + 5) / 2 = 5.5$. For fallbacks, `c3`, `af`, `c3`, and `a9` each decode alone to U+FFFD, so the rate is $4 / 12 \approx 0.333$.

Now the char tokenizer trained on `["naïve"]`. Its vocabulary is `<unk>`, `a`, `e`, `n`, `v`, `ï`. The text encodes to `n a ï v e <unk> <unk> a <unk> <unk>`, because the space, `c`, `f`, and `é` were never seen: 10 tokens. Bytes per token is $12 / 10 = 1.2$. For fertility, `naïve` is 5 tokens and `café` is `<unk> a <unk> <unk>`, 4 tokens, so $(5 + 4) / 2 = 4.5$. The fallbacks are the four `<unk>`, so the rate is $4 / 10 = 0.4$.

Finally, bits per byte. Suppose a model pays $12 \ln 2$ nats for this text, 12 bits over 12 bytes. Through the byte tokenizer that is $\ln 2$ nats per token, 1 bit per token, and $1 / 1.0 = 1$ bit per byte. Through the char tokenizer the same total is spread over 10 tokens: 1.2 bits per token, and $1.2 / 1.2 = 1$ bit per byte. Different loss per token, the same bits per byte.

## Designing the API

Four functions:

```python
# python/tinyllm/tok/metrics.py
def fertility(tok: Tokenizer, words: Sequence[str]) -> float
def bytes_per_token(tok: Tokenizer, texts: Sequence[str]) -> float
def is_fallback(tok: Tokenizer, i: int) -> bool
def byte_fallback_rate(tok: Tokenizer, texts: Sequence[str]) -> float
```

Every input is encoded with `tok.encode(x)`, never `add_special=True`. The functions use only `encode`, `decode`, `unk_id`, and `special_ids`, so any object with the protocol is measured, including one that is not a subclass of anything in `tinyllm`. `byte_fallback_rate` caches `is_fallback` per id, because a corpus repeats ids and `decode` is the slow part.

The model tests in [`test_metrics.py`](../../../../course/solutions/learner-tests/L1.6/python/tests/l1-6-metrics/test_metrics.py) pin the hand numbers: `ByteTokenizer` on `"naïve café"` is 1.0 bytes per token and a char tokenizer trained on `"naïve"` is 1.2; `["ab", "é"]` under a char tokenizer trained on `"ab"` is 4/3; fertility of `["naïve", "café"]` in bytes is 5.5 and of `["a", "a", "bb"]` is 4/3; a `<bos>`/`<eos>` tokenizer costs 2 for `"hi"`; `"éé"` in bytes has fallback rate 1.0; and empty inputs raise.

## Building it step by step

The solution is short, and each line follows from one principle.

1. **Measure what the model will see.** The model sees `tok.encode(x)` with no template. So every count is `len(tok.encode(x))`, never with `add_special=True`: a `<bos>` and `<eos>` are the template's cost, not the text's.
2. **A word's cost is the word alone.** "What does one word cost" means encoding that word in isolation; joining words adds space tokens and lets merges cross word boundaries. So fertility encodes each word separately and averages over the list, duplicates included, because the list is a sample of word occurrences.
3. **Compression is total bytes over total tokens.** Bits per byte is (total nats) / (ln 2 · total bytes), a ratio of totals; for bytes per token to be its exchange rate it must also be a ratio of totals, in the same unit, UTF-8 bytes. A mean of per-text ratios weights texts equally, not bytes, and breaks $\mathrm{bpb} = \text{bits per token} / \mathrm{bpt}$.
4. **A fallback is a token that is not whole characters.** The protocol exposes `decode`, `unk_id`, and `special_ids`, nothing about bytes, so the only tokenizer-independent test is: is it the unknown id, or does decoding it alone produce U+FFFD? Every other special token is excluded, because its text is a control symbol, not lost text.
5. **A rate over occurrences.** The question is "what share of the tokens the model sees are fallbacks", so the numerator counts every occurrence. Since `is_fallback` is a pure function of the id and `decode` is the expensive call, memoize it per id.
6. **No value without data.** Each ratio is undefined with a zero denominator; raising `ValueError` is the only answer that cannot let an empty evaluation split pass a threshold.

### Fertility

```python
def fertility(tok: Tokenizer, words: Sequence[str]) -> float:
    if len(words) == 0:
        raise ValueError("fertility needs at least one word")
    # Each word alone, no specials; a repeated word counts every time.
    return sum(len(tok.encode(w)) for w in words) / len(words)
```

The denominator is the number of words, which is known before encoding, so the empty check comes first.

### Bytes per token

```python
def bytes_per_token(tok: Tokenizer, texts: Sequence[str]) -> float:
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)   # UTF-8 bytes, not len(t)
    n_tokens = sum(len(tok.encode(t)) for t in texts)      # totals, not a mean of ratios
    if n_tokens == 0:
        raise ValueError("bytes_per_token needs texts that encode to at least one token")
    return n_bytes / n_tokens                              # bytes per token, not tokens per byte
```

Here the denominator is the token total, so "empty" means "no tokens", which also covers `[]` and `[""]`. On `["ab", "é"]` with a char tokenizer trained on `"ab"`, that is 2 + 2 = 4 bytes over 2 + 1 = 3 tokens, 4/3; the mean of ratios would be $(1 + 2)/2 = 1.5$.

### Is it a fallback?

```python
def is_fallback(tok: Tokenizer, i: int) -> bool:
    if tok.unk_id is not None and i == tok.unk_id:   # <unk> is lost text, even though it is special
        return True
    if i in tok.special_ids.values():                # <bos>, <|endoftext|>: control, not text
        return False
    return "�" in tok.decode([i])               # a fragment of a multi-byte character
```

The order matters. The unknown id is usually also listed among the specials, so it must be tested first, or every `<unk>` would be excused as a special. A vocabulary entry that already decodes to U+FFFD is counted by the last line too, which is intended: that text was lost upstream.

### The fallback rate

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

On `"naïve café"` in bytes there are 12 tokens, of which `c3 af c3 a9` decode alone to U+FFFD, so the rate is $4/12$. The cache turns $O(N)$ decodes into $O(\text{distinct ids})$.

The full solution is [`metrics.py`](../../../../course/solutions/python/tinyllm/tok/metrics.py).

### Why this design

Each function is one pass of encoding: $O(\sum_i B(t_i))$ for the byte counts plus the cost of `encode`. `byte_fallback_rate` adds one `decode` per distinct id and $O(\text{distinct ids})$ memory for the cache.

Why use only the protocol? Reaching into a BPE's byte table or a WordPiece's `##` prefix would give per-family definitions and exclude [the Rust tokenizer](05-rust-fast-bpe.md). `decode([i])` containing U+FFFD is the one definition every tokenizer answers the same way, and it is exactly what Hugging Face computes. Why not code points? Bits per byte divides by bytes, so bytes per token in code points would not be its exchange rate, and it would understate what each token of CJK or emoji text carries by a factor of 3 or 4. Why raise instead of returning 0 or NaN? A 0 passes any "at most" threshold, and a NaN fails comparisons silently; a `ValueError` stops an evaluation that measured nothing.

One tradeoff is accepted: `fertility` is "tokens per whitespace word supplied by the caller". The solution does not segment words itself, so languages without spaces need a word list from elsewhere.

## What can go wrong

A correct solution reproduces the hand numbers for bytes and char exactly, and the golden metrics for GPT-2, SmolLM2, BERT, and a Unigram on 300 strings to 1e-12, so the capstone's decision record quotes numbers anyone can reproduce. The boundary and unit checks spell out the definitions: an emoji is 4 bytes, and emoji plus `日` over 2 tokens is 3.5; `["ab", "é"]` gives 4/3, not 1.5; `["ab", "c"]` in bytes has fertility 1.5; `["a", "a", "bb"]` has fertility 4/3; a `<bos>`/`<eos>` tokenizer costs 2 for `"hi"`, not 4. Partial bytes and `<unk>` are fallbacks, while `a`, `<bos>`, GPT-2's whole `é`, and its end-of-text are not, and a U+FFFD entry is. `"éé"` in bytes has rate 1.0 and `["aé", "é"]` has 4/5. No words, no texts, and texts with no tokens raise `ValueError`, while one token is enough. Bits per byte through the `NLLAccumulator` must be tokenizer independent on the hand example, and a bare protocol object with no `tinyllm` base class must be measurable, so the Rust tokenizer is measured the same way.

> **Warning:** Counting `len(text)` instead of UTF-8 bytes makes Chinese text look three times less compressed than it is, and any bits per byte computed from it is wrong. The solution counts `len(t.encode("utf-8"))`.

The other pitfalls:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| encoding the words joined by spaces | fertility grows with the space tokens, or shrinks where BPE merges across words | encodes each word alone inside the sum |
| `add_special=True` | every word costs two extra tokens; WordPiece looks worst for its template | calls `tok.encode(x)` with the default, no specials |
| a mean of per-text ratios | short texts dominate; the number disagrees with bits per byte | sums bytes and tokens separately, divides once |
| counting distinct fallback ids | a text full of one unknown character scores a low rate | adds `cache[i]` for every occurrence; the cache only saves `decode` calls |
| forgetting `<unk>`, or counting specials | the char tokenizer's unknowns vanish from the rate; GPT-2's end-of-text counts as lost text | `is_fallback` tests `unk_id` first, then excludes `special_ids` |
| tokens per byte | the ratio is inverted and every comparison flips | returns `n_bytes / n_tokens` |
| dividing by zero on empty input | a `ZeroDivisionError` deep in an evaluation job, or a silent 0 | checks the denominator and raises `ValueError` |

## Where this leads

The [TinyStories capstone](../capstones/01-tinystories.md) reports these numbers in its vocabulary decision record and in the BPE versus Unigram ablation. [Tokenize and pack](../../data-engineering/05-corpus-pipeline/07-tokenize-and-pack.md) in *Data Engineering* records the bytes per token of each shard in the corpus manifest, and the tokenizer layer's end-to-end run reports `bytes_per_token`.

These metrics have production counterparts:

- `bytes_per_token` appears in tokenizer comparisons in model reports, measured as characters per token and bytes per token across many languages, the "tokenizer tax" (Petrov et al., 2023, and the Llama 3 report's tokenizer section).
- `fertility` appears in multilingual tokenizer evaluation, alongside the proportion of continued words per language (Rust et al., 2021, and `tokenizer-eval` style suites).
- Bits per byte is the tokenizer-independent loss evaluation harnesses use to compare models (the `lm-evaluation-harness` `bits_per_byte` metric).

### Further reading

- Rust et al., *How Good is Your Tokenizer?* (ACL 2021), on fertility and continued words.
- Petrov et al., *Language Model Tokenizers Introduce Unfairness Between Languages* (NeurIPS 2023).

## Key terms

- **Fertility**: the mean number of tokens per word, each word encoded alone.
- **Bytes per token**: total UTF-8 bytes over total tokens; the compression ratio.
- **Fallback token**: the unknown id, or a non-special token whose decoding alone contains U+FFFD.
- **Fallback rate**: the share of token occurrences that are fallbacks.
- **Bits per byte**: total loss in bits divided by total UTF-8 bytes; equal to bits per token over bytes per token.

## Summary

- Three numbers describe what a vocabulary does to text: fertility (tokens per word), bytes per token (compression), and the fallback rate (tokens that are not whole characters).
- Every one is a ratio of totals over the corpus, counted in UTF-8 bytes and without special tokens; a mean of per-text ratios or a count in code points is a different, wrong number.
- Bytes per token is the exchange rate between a model's loss per token and its bits per byte, which is why only bits per byte compares models with different tokenizers.
- The metrics see only the `Tokenizer` protocol, so the BPE, WordPiece, and Unigram tokenizers, and the Rust tokenizer, are measured by the same code and agree with Hugging Face's numbers.

## Review questions

1. Why must `is_fallback` test `unk_id` before `special_ids`?
   <details><summary>Answer</summary>The unknown token is normally registered as a special token. Testing specials first would return `False` for every `<unk>`, and the char tokenizer's unknown characters would vanish from the rate.</details>
2. A corpus is one 1,000-byte English text (250 tokens) and one 3-byte `日` (3 byte tokens). What do the ratio of totals and the mean of ratios give, and which is right?
   <details><summary>Answer</summary>Totals: $1003 / 253 \approx 3.96$. Mean of ratios: $(4 + 1)/2 = 2.5$. The ratio of totals is right: it is the number that converts loss per token to bits per byte over this corpus; the mean lets a 3-byte text weigh as much as a 1,000-byte one.</details>
3. Why does a model with a higher loss per token not necessarily model text worse?
   <details><summary>Answer</summary>Loss per token depends on how much each token carries. $\mathrm{bpb} = \text{bits per token} / \mathrm{bpt}$: a tokenizer with longer tokens has higher bits per token and higher bpt, and the same bpb for the same model quality. The hand example shows it: 1 bit per token at 1.0 bpt and 1.2 bits per token at 1.2 bpt are both 1 bit per byte.</details>
4. Why encode each word alone in `fertility`?
   <details><summary>Answer</summary>Joining words adds the separator's tokens and lets BPE merge across the join (GPT-2 attaches the space to the next word), so the count would no longer be the cost of a word. Each word alone is the definition.</details>
