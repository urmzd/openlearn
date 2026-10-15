<!-- ol:module L2.1 -->
# n-gram language model with interpolated modified Kneser-Ney

## Overview

| | |
|---|---|
| **Module** | `L2.1` · build · Python · Pass 3 · 5 to 6 h, plus your graded tests (rung R3) |
| **You build** | `python/tinyllm/lm/ngram.py`: `NGramLM(n, discount, vocab_size)` with `fit`, `prob`, `logprobs`, `nll`, `perplexity`, `discounts`, `save`, `load`: an n-gram model with interpolated Kneser-Ney smoothing (modified, or one fixed discount); and your own tests in `python/tests/l2-1-ngram/`, written first |
| **Contract** | [`course/contracts/py/tinyllm/lm/ngram.pyi`](../../../../course/contracts/py/tinyllm/lm/ngram.pyi) |
| **Tests** | `course/tests/L2.1/test_ngram.py` (what they check: section 4), golden values from an independent exact-fraction implementation, `course/oracle/L2.1/kn_golden.py`, in `course/fixtures/L2.1/kn_golden.json`; your tests are graded by mutation, threshold 0.70 plus one required fault, with a red-then-green journal |
| **Needs** | `M07.2` `ney_discount` · `M11.2` `NLLAccumulator` for perplexity · `L0.6` `save_safetensors` and `load_safetensors` (or `--ref-deps`). Reading: `S-M07a` (conditional probability), `L0.0` (the bigram this generalizes) |
| **Used by** | later `L6.7` the zoo's KN baseline, `L8.6` the n-gram draft model of speculative decoding, `C1` the corpus perplexity filter plugged into `data.02`'s `ppl_filter` (each joins the registry with its batch) |
| **Milestone** | `MS-L2` (`{tinyllm} lm train ngram --n 4`, then `{tinyllm} eval`: perplexity within 0.5% of the reference) |
| **Optional depth** | Chen and Goodman, "An empirical study of smoothing techniques for language modeling" (1998), sections 2.7, 3, and 4.1.6; Jurafsky and Martin, *Speech and Language Processing*, 3rd ed., ch. 3; Heafield et al., "Scalable modified Kneser-Ney language model estimation" (ACL 2013, KenLM) |

## Key Takeaways

- Maximum-likelihood n-grams give probability 0 to every unseen continuation, so their perplexity on new text is infinite; smoothing moves a little mass from seen counts to a lower-order model, at every context (`test_probabilities_sum_to_one`).
- Kneser-Ney's lower orders count how many different words a token **follows** (its continuation count), not how often it occurs: a frequent token that only ever follows one word gets little mass as a fresh continuation (`test_hand_example_bigram`).
- The start symbol `<s>` has nothing before it, so n-grams that begin with it keep their raw counts (`test_hand_example_sequence_start`, `test_golden_hand_cases`).
- Modified KN uses three discounts per order, $D_1, D_2, D_{3+}$, estimated from the counts of counts, with a fixed fallback when the data are too small for the formulas (`test_golden_byte_corpus`, `test_discount_fallback`).
- The same numbers come out of `prob`, the vectorized `logprobs`, and a reloaded file (`test_logprobs_agree_with_prob`, `test_save_load_roundtrip`).

## How to work this chapter

```bash
ol start L2.1              # stubs ngram.py; prints your test path and rung (R3)
ol tests L2.1              # the course tests
# write ONE test in python/tests/l2-1-ngram/ (start with section 4's), then:
ol tdd red L2.1            # must FAIL against your current code: records the red
# make it pass, then:
ol tdd green L2.1          # must PASS with the same test files: records the green
# repeat for each test; then:
ol check L2.1              # course tests, the red-then-green journal, the mutation grade
ol mutate L2.1             # the full grade, cached by your test files' hash
```

---

## 1. Why now

Your Pass 1 bigram (`L0.0`) predicts the next byte from one byte of history, and the Pass 2 trainer (`L0.5`) fits the same table by gradient descent. Neither can use more context: a table indexed by the last three bytes has $256^3 \approx 1.7 \times 10^7$ rows, almost all never seen, and an unseen row means probability 0 for the true next token and an infinite loss. This module is the classic answer, used in speech recognition and machine translation for twenty years before neural models: count n-grams, then smooth the counts so that every context still gives every token some probability. Interpolated modified Kneser-Ney is the strongest of those smoothers, the baseline every neural model in the zoo (`L6.7`) has to beat on bits per byte, a cheap draft model for speculative decoding (`L8.6`), and in the capstone the perplexity filter that throws out garbage documents before training.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $V$ | vocabulary size; tokens are ids $0 \dots V-1$ | `int` |
| $n$ | the order: an n-gram model conditions on $n - 1$ previous tokens | `int` |
| $w$, $v$ | tokens | ids |
| `<s>` | the start symbol, put before every sequence; a context symbol, never predicted (stored as $-1$) | |
| $h$ | a history: the $k - 1$ symbols before the predicted token at order $k$ | tuple |
| $h'$ | $h$ without its first symbol (the next lower order's history) | tuple |
| $c_k(g)$ | raw count of the $k$-gram $g$ | `int` |
| $N_{1+}(\bullet\, g)$ | continuation count: number of distinct symbols seen right before $g$ | `int` |
| $a_k(g)$ | adjusted count: $c_k(g)$ at the top order or when $g$ starts with `<s>`, else $N_{1+}(\bullet\, g)$ | `int` |
| $T_k(h) = \sum_v a_k(h v)$ | total adjusted count after $h$ | `int` |
| $N_j(h)$ | number of tokens $v$ with $a_k(hv) = j$ ($N_{3+}$: at least 3) | `int` |
| $D_{k1}, D_{k2}, D_{k3}$ | discounts for adjusted counts 1, 2, and 3 or more at order $k$ | `float` |
| $\gamma_k(h)$ | interpolation weight: the mass the discounts removed after $h$ | `float` |
| $p_k(w \mid h)$ | the order-$k$ estimate; $p_0(w) = 1/V$ | `float` |
| $n_j$ | counts of counts: number of $k$-grams with $a_k = j$ | `int` |
| $Y$ | $n_1 / (n_1 + 2 n_2)$, Ney's discount (`M07.2`) | `float` |

### 2.1 n-gram models and the zero problem

The chain rule writes the probability of a sequence as a product of next-token probabilities, $P(w_1 \dots w_T) = \prod_t P(w_t \mid w_{<t})$. An **n-gram model** assumes the next token depends only on the last $n - 1$: $P(w_t \mid w_{<t}) \approx P(w_t \mid w_{t-n+1} \dots w_{t-1})$. The maximum-likelihood estimate (`M07.2`) is a ratio of counts, $c(h w) / \sum_v c(h v)$. On a 2 KB corpus a byte 4-gram model has seen a few thousand of the possible $256^4$ four-byte strings; any test sentence contains n-grams it never saw, each gets probability 0, and the perplexity is infinite. Smoothing is the whole game.

### 2.2 Sequences, `<s>`, and contexts

Each training sequence (a line, a document) is padded with one start symbol: `a b c` becomes `<s> a b c`. The first real token is then predicted from the context `<s>`, which says "start of a sequence"; a model learns that sentences start with capitals and stories with "Once". The n-grams of a padded sequence are the runs of $k \le n$ symbols that **end at a token**, so `<s>` can only be first. `<s>` is never predicted, so every distribution is over the $V$ real tokens and sums to 1.

A query passes the tokens before the predicted one in the same sequence. Two rules decide what the model conditions on: a context with fewer than $n - 1$ tokens is the **start** of a sequence, so `<s>` goes in front of it; a longer one is cut to its **last** $n - 1$ tokens and gets no `<s>`. For a trigram, `[b]` means `<s> b` while `[a, b]` and `[c, c, a, b]` both mean `a b`. The order actually used is $K = \min(n, \text{len(padded context)} + 1)$.

### 2.3 Absolute discounting and interpolation

Absolute discounting subtracts a fixed $D \in (0, 1]$ from every seen count after a history $h$ and gives the removed mass to the next lower order, at every context (interpolation), not only when the count is zero (backoff):

$$p_k(w \mid h) = \frac{\max(c(hw) - D, 0)}{T(h)} + \gamma(h)\, p_{k-1}(w \mid h'), \qquad \gamma(h) = \frac{D \cdot \#\{v : c(hv) > 0\}}{T(h)} .$$

`M07.2`'s `absolute_discount` is this formula for one context. The recursion bottoms out at $p_0(w) = 1/V$, so every token has positive probability. Why it sums to 1: the first term sums to $(T(h) - D \cdot \#\{\text{seen}\})/T(h) = 1 - \gamma(h)$, and $\gamma(h)$ times a distribution that sums to 1 adds back exactly $\gamma(h)$. This needs $D \le$ every count it is subtracted from; for $D \le 1$ and integer counts that always holds.

### 2.4 Kneser-Ney: continuation counts

Absolute discounting still builds its lower orders from raw counts, and that is wrong in a specific way. The lower-order model is only consulted in proportion to $\gamma(h)$, that is, for histories where the higher order has little evidence. It should answer "how likely is $w$ to appear in a **new** context?", not "how frequent is $w$?". In English "Francisco" is frequent but almost always follows "San"; as a continuation of an unfamiliar word it deserves little probability. Kneser and Ney's fix: below the top order, replace the count of a $k$-gram $g$ by its **continuation count** $N_{1+}(\bullet\, g)$, the number of distinct symbols seen right before it.

Two exceptions keep raw counts: the top order $n$ (nothing is backed off to from above), and n-grams that start with `<s>`, since nothing ever precedes `<s>` and their continuation count would be 0 (KenLM's rule). Call the result the adjusted count $a_k(g)$. The model, from $p_0(w) = 1/V$ up to order $K$:

$$p_k(w \mid h) = \frac{\max\big(a_k(hw) - D_k(a_k(hw)), 0\big)}{T_k(h)} + \gamma_k(h)\, p_{k-1}(w \mid h'), \qquad \gamma_k(h) = \frac{D_{k1} N_1(h) + D_{k2} N_2(h) + D_{k3} N_{3+}(h)}{T_k(h)} ,$$

with $D_k(0) = 0$, $D_k(1) = D_{k1}$, $D_k(2) = D_{k2}$, $D_k(c \ge 3) = D_{k3}$. The weight $\gamma_k(h)$ is exactly the mass the discounts removed after $h$, bucket by bucket, so the distribution still sums to 1 as long as $D_{kj} \le j$. If no $k$-gram starts with $h$ (a history never seen), there is nothing to discount and $p_k(w \mid h) = p_{k-1}(w \mid h')$: the model falls through to the next lower order, never to the uniform distribution or to 0.

`logprobs` computes all $V$ probabilities at once with numpy, order by order; `prob` computes one with dictionary lookups. Both must give the same numbers.

### 2.5 Modified Kneser-Ney: three discounts

One discount is a compromise: a count of 1 is mostly noise and deserves a large discount relative to its size, a count of 50 almost none. Chen and Goodman estimate three discounts per order from the **counts of counts** of the adjusted counts at that order, $n_j = \#\{g : a_k(g) = j\}$:

$$Y = \frac{n_1}{n_1 + 2 n_2}, \qquad D_{k1} = 1 - 2Y\frac{n_2}{n_1}, \qquad D_{k2} = 2 - 3Y\frac{n_3}{n_2}, \qquad D_{k3} = 3 - 4Y\frac{n_4}{n_3} .$$

$Y$ is Ney's single discount, `M07.2`'s `ney_discount` applied to the adjusted counts. On small data the formulas break: some $n_j$ is 0 (division by zero), or a discount falls outside $(0, j]$ (negative, or larger than the count it is subtracted from, which would make probabilities negative). The contract then uses KenLM's fallback $(0.5, 1.0, 1.5)$ for that order. The worked example's trigram falls back at every order; the 2 KB byte corpus of the tests has enough data from order 2 up. With a fixed discount $d$, $D_{k1} = D_{k2} = D_{k3} = d$ everywhere.

### 2.6 Scoring and saving

`nll(ids)` scores one sequence token by token, $-\ln p(w_t \mid w_{<t})$, the first token from `<s>`; `perplexity(ids)` is $\exp$ of their mean, summed with `M11.2`'s `NLLAccumulator` so the result is exact over long texts. Bits per byte, the zoo's metric, is the same total over the byte count (`M11.2`).

`save` writes one safetensors file through `L0.6`'s writer: for every order $k$, an `I32` tensor `order{k}.grams` with every $k$-gram seen (`<s>` as $-1$, rows in ascending order) and `order{k}.counts` with their raw counts, plus metadata `tl_arch = "ngram"`, `n`, `discount`, `vocab_size`. Everything else (adjusted counts, discounts, $\gamma$) is recomputed by `load`, so the file holds only data and saving a loaded model writes the same bytes.

## 3. Worked example by hand

Three sequences over $V = 3$ tokens `a`, `b`, `c` (ids 0, 1, 2), a bigram model ($n = 2$), fixed discount $d = 1/2$:

```
<s> a b c
<s> a b a b
<s> c a b c
```

**Bigram counts** (the top order: raw counts): `<s> a` 2, `<s> c` 1, `a b` 4, `b c` 2, `b a` 1, `c a` 1.

**Unigram level: continuation counts.** `a` follows `<s>`, `b`, and `c`: 3. `b` follows only `a`: 1. `c` follows `b` and `<s>`: 2. So $T_1 = 6$, three seen types, $\gamma_1 = 3 \cdot \frac12 / 6 = \frac14$, and

$$p_1(a) = \frac{3 - \frac12}{6} + \frac14 \cdot \frac13 = \frac12, \qquad p_1(b) = \frac{1 - \frac12}{6} + \frac1{12} = \frac16, \qquad p_1(c) = \frac{2 - \frac12}{6} + \frac1{12} = \frac13 .$$

`b` is the most frequent token (4 raw occurrences, like `a`) but gets the least unigram mass: it only ever follows `a`. With raw counts (4, 4, 3) the same formula would give $p_1(b) = \frac{3.5}{11} + \frac{3}{22} \cdot \frac13 = \frac{4}{11} \approx 0.36$, more than twice as much.

**After `b`**: `b a` 1, `b c` 2, so $T = 3$ and $\gamma = 2 \cdot \frac12 / 3 = \frac13$:

| $w$ | discounted count / $T$ | $+\ \gamma\, p_1(w)$ | $p(w \mid b)$ |
|---|---|---|---|
| `a` | $0.5 / 3 = 1/6$ | $\frac13 \cdot \frac12 = \frac16$ | $1/3$ |
| `b` | $0$ | $\frac13 \cdot \frac16 = \frac1{18}$ | $1/18$ |
| `c` | $1.5 / 3 = 1/2$ | $\frac13 \cdot \frac13 = \frac19$ | $11/18$ |

They sum to $\frac{6 + 1 + 11}{18} = 1$. After `c` (only `c a`, $T = 1$, $\gamma = \frac12$): $p(\cdot \mid c) = (\frac34, \frac1{12}, \frac16)$. These are `test_hand_example_bigram`.

**At the start** (`<s> a` 2, `<s> c` 1, $T = 3$, $\gamma = \frac13$): $p(a \mid \texttt{<s>}) = \frac{1.5}{3} + \frac16 = \frac23$, $p(b \mid \texttt{<s>}) = \frac1{18}$, $p(c \mid \texttt{<s>}) = \frac{0.5}{3} + \frac19 = \frac5{18}$. This is `test_hand_example_sequence_start`, and the sequence `a b` costs $-\ln\frac23 - \ln p(b \mid a)$ nats with $p(b \mid a) = \frac{3.5}{4} + \frac18 \cdot \frac16 = \frac{43}{48}$.

## 4. The interface

```python
class NGramLM:
    def __init__(self, n: int, discount: float | Literal["modified"] = "modified",
                 vocab_size: int | None = None) -> None: ...
    def fit(self, sequences: Iterable[Sequence[int]]) -> None: ...      # replaces earlier counts
    def discounts(self, order: int) -> tuple[float, float, float]: ...
    def prob(self, context: Sequence[int], token: int) -> float: ...
    def logprobs(self, context: Sequence[int]) -> NDArray: ...         # [V]
    def nll(self, ids: Sequence[int]) -> NDArray: ...                  # [len(ids)], first token from <s>
    def perplexity(self, ids: Sequence[int]) -> float: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "NGramLM": ...
FALLBACK_DISCOUNTS = (0.5, 1.0, 1.5)
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_bigram` | unit, smoke | section 3's $p(\cdot \mid b)$ and $p(\cdot \mid c)$ to $10^{-15}$ | continuation counts at the unigram level |
| `test_hand_example_sequence_start` | unit, smoke | $p(\cdot \mid \texttt{<s>}) = (2/3, 1/18, 5/18)$ | every sequence starts here |
| `test_golden_hand_cases` | golden | every context of the example, as a bigram and as a modified-KN trigram, against exact fractions | `<s>`-initial grams keep raw counts at lower orders |
| `test_golden_byte_corpus` | golden | the MS-P1 corpus as bytes, $n = 1 \dots 4$, modified and $d = 0.75$: discounts, held-out perplexities, probabilities after "the " | the numbers MS-L2 compares within 0.5% |
| `test_probabilities_sum_to_one` | property | $\sum_v p(v \mid \text{ctx}) = 1$ for seen, unseen, and start contexts, $n = 1 \dots 4$ | a proper distribution, as `L8.6` needs for exact acceptance |
| `test_logprobs_agree_with_prob` | property | the vectorized path equals `prob` for every token; logsumexp 0 | `L6.7` and `L8.6` call `logprobs` |
| `test_unseen_history_backs_off` | boundary | an unseen history gives the KN unigram $(23, 7, 15, 3)/48$ | no jump to uniform or to 0 |
| `test_context_length_rules` | unit | short context means sequence start; long context keeps its last $n - 1$ tokens | the draft model passes whole prefixes |
| `test_discount_fallback` | boundary | zero counts of counts and a negative $D_3$ both give $(0.5, 1, 1.5)$, and the result still sums to 1 | small corpora in tests and filters |
| `test_nll_and_perplexity` | unit | per-token NLL from `<s>`, perplexity $= \exp(\text{mean})$, empty input raises | the corpus filter thresholds perplexity |
| `test_save_load_roundtrip` | unit | header `tl_arch = "ngram"`, I32 tensors, equal probabilities after reload, identical bytes on re-save | the zoo and the capstone load the file later |
| `test_fit_replaces_previous_counts` | boundary | a second `fit` forgets the first | one model object per corpus source |
| `test_input_validation` | boundary | bad $n$, bad discounts, ids outside $0 \dots V-1$, queries before `fit`, inferred $V$ | caller bugs fail early |

### Your graded tests (rung R3)

Rung R3 gives you the interface (above) and one test; you write the rest **before** the code that makes each pass. The given test:

```python
# python/tests/l2-1-ngram/test_ngram.py
import numpy as np
from tinyllm.lm.ngram import NGramLM

def test_hand_example_bigram():
    """Continuation counts a 3, b 1, c 2: p1 = (1/2, 1/6, 1/3); after b: (1/3, 1/18, 11/18)."""
    lm = NGramLM(2, 0.5, 3)
    lm.fit([[0, 1, 2], [0, 1, 0, 1], [2, 0, 1, 2]])
    np.testing.assert_allclose([lm.prob([1], w) for w in range(3)], [1 / 3, 1 / 18, 11 / 18], atol=1e-15)
```

Then, one at a time, red then green: the start-of-sequence probabilities; modified discounts of a unigram model from counts of counts you choose (counts 1, 1, 1, 1, 2, 2, 2, 3, 3, 4 give $Y = 0.4$ and $D = (0.4, 1.2, 2.2)$); the fallback; probabilities summing to 1 for several orders and contexts; a start context differing from the same tokens mid-sequence; an unseen history; `logprobs` against `prob`; `nll` of a two-token sequence; a save and load roundtrip; a second `fit` replacing the first. Import only contract names (`tinyllm.lm.ngram`). `ol check L2.1` requires, for every test file, a `ol tdd red` record before its last `ol tdd green`, a mutation score of at least 0.70, and the required fault killed: it is pitfall 1.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. lower orders from raw counts (absolute discounting, not Kneser-Ney) | frequent tokens get too much mass as fresh continuations; `p(b)` at the unigram level is 4/11 instead of 1/6 | `test_hand_example_bigram`, `test_golden_byte_corpus` (mutant `s01`) |
| 2. $\gamma$ charged with $D_1$ for every seen token | with three discounts, the distribution no longer sums to 1 | `test_probabilities_sum_to_one`, `test_golden_hand_cases` (mutant `s02`) |
| 3. continuation counts for n-grams that start with `<s>` | they become 0: start-of-sequence predictions fall to the lower order | `test_golden_hand_cases`, `test_context_length_rules` (mutant `s03`) |
| 4. no fallback when a discount leaves $(0, j]$ | a negative $D_3$ gives negative probabilities on small data | `test_discount_fallback` (mutant `s04`) |
| 5. an unseen history jumps to the uniform distribution | every rare context forgets what the unigram level knows | `test_unseen_history_backs_off` (mutant `s05`) |
| 6. keeping the first $n - 1$ tokens of a long context | the draft model conditions on the start of the prompt, not its end | `test_context_length_rules` (mutant `s06`) |
| counts of 3 or more discounted with $D_2$ | $\gamma$ and the numerators disagree: no longer sums to 1 | `test_probabilities_sum_to_one` (mutant `s07`) |
| the first token of a sequence not scored | perplexity skips the hardest prediction | `test_nll_and_perplexity`, `test_golden_byte_corpus` (mutant `s08`) |
| `<s>`-initial n-grams dropped by `save` | a reloaded model predicts sequence starts differently | `test_save_load_roundtrip` (mutant `s09`) |
| `logprobs` using $D_1$ for every count | the vectorized path disagrees with `prob` | `test_logprobs_agree_with_prob` (mutant `s10`) |
| $D_3$ computed from $n_3 / n_2$ | wrong discounts at every order with enough data | `test_golden_byte_corpus`, `test_discount_fallback` (mutant `s11`) |
| `fit` adding to the previous counts | a reused model mixes two corpora | `test_fit_replaces_previous_counts` (mutant `s12`) |
| inferred $V$ = the largest id instead of 1 + it | the largest token becomes unpredictable | `test_input_validation`, `test_save_load_roundtrip` (mutant `s13`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.2` | `ney_discount` gives $Y = n_1/(n_1 + 2n_2)$ for each order's discounts |
| Back | `M11.2` | `perplexity` sums the per-token NLLs with `NLLAccumulator` |
| Back | `L0.6` | `save` and `load` go through `save_safetensors` and `load_safetensors` |
| Back | `L0.0` | the byte bigram this generalizes (reading) |
| Forward | `L6.7` | the zoo trains KN-4 on TinyStories and reports its bits per byte, the floor the neural models must beat |
| Forward | `L8.6` | the n-gram draft model proposes tokens from `logprobs` that the target model verifies |
| Forward | `C1` | the corpus pipeline's `ppl_filter` (`data.02`) drops documents whose KN perplexity is too high, and quality checks score samples with it |

If you skip this module, `L6.7` and `L8.6` stop with `BLOCKED ... needs L2.1` once they land: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| counting in dicts | KenLM `lmplz` | disk-based streaming sorts, so a 100 GB corpus is counted in bounded memory; pruning of rare n-grams | `lm/builder/` (`corpus_count.cc`, `adjust_counts.cc`, `initial_probabilities.cc`) |
| `save` with every n-gram | KenLM's ARPA and binary formats | probabilities and backoff weights precomputed per n-gram; a trie or probing hash table with quantized values for fast queries | `lm/binary_format.cc`, `lm/trie.cc` |
| `NGramLM` as a quality filter | CCNet and RedPajama | a KenLM 5-gram trained on Wikipedia scores every web document; perplexity buckets decide what is kept | `facebookresearch/cc_net`, `cc_net/perplexity.py` |
| n-gram draft model | prompt-lookup and n-gram speculative decoding in vLLM | drafts from n-gram matches against the prompt, no extra model | `vllm/v1/spec_decode/ngram_proposer.py` |
| interpolated KN | the infini-gram index (Liu et al. 2024) | unbounded-order n-gram counts over trillions of tokens with suffix arrays | `liujch1998/infini-gram` |
