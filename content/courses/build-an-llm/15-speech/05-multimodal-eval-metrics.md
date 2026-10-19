# Multimodal eval metrics: WER, CER, VQA, captions, zero-shot

[Whisper decoding](04-whisper-decoding.md) produces transcripts, and the vision models of the previous chapter produce answers and captions. Nothing so far can say whether they are any good. This section builds the measuring instruments: word and character error rates for speech, VQA accuracy, caption scores and zero-shot accuracy for vision, and a bootstrap confidence interval for every one of them.

Several later parts of the system lean on these numbers. Release gates block a model on a WER ceiling, bias evaluations compare WER across speaker groups, and the audio corpus pipeline normalizes transcripts the same way. All of them need one implementation whose numbers match what papers report (jiwer, sacreBLEU), whose ambiguous cases are pinned down, and whose intervals are reproducible from a seed across languages. The system's `tinyllm eval asr` command reports WER with a bootstrap `wer_ci` using exactly these functions.

You will reuse `corpus_bleu` and `chrf` from [Sequence metrics: exact match, BLEU, chrF](../05-attention-origins/05-sequence-metrics.md), and from *Mathematics for Computing* the percentile bootstrap of [LLN, CLT, confidence intervals, bootstrap](../../mathematics-for-computing/08-probability-statistics/05-lln-clt-confidence-intervals-bootstrap.md) and the `PCG32` stream of [Modular arithmetic, hashing, and PCG32 in Python](../../mathematics-for-computing/07-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md). It helps to think of text as Unicode code points, as in [Injective, surjective, bijective: the GPT-2 byte map](../../mathematics-for-computing/06-discrete-math-1/02-injective-surjective-bijective-gpt2-byte-map.md). The solution is [`mm.py`](../../../../course/solutions/python/tinyllm/eval/mm.py), with `basic_normalize`, `edit_ops` (Levenshtein with a fixed tie rule), `wer_stats`, `wer`, `cer`, `vqa_accuracy`, `caption_scores`, `zero_shot_accuracy` and `bootstrap_ci`; the contract is [`mm.pyi`](../../../../course/contracts/py/tinyllm/eval/mm.pyi).

> **Learning objectives**
> - Compute a Levenshtein alignment by hand and recover its substitution, deletion and insertion counts with a fixed tie rule.
> - Explain why corpus WER pools edits and reference lengths before dividing, and why it can exceed 1.
> - Apply Whisper's basic normalizer and explain the order of its steps.
> - Define VQA accuracy, caption attribute recall and top-$k$ zero-shot accuracy, including their edge rules.
> - Build a bootstrap interval that resamples items, not words, and pools again on each resample.

## The notation

| Symbol | Meaning | Type / shape |
|---|---|---|
| $r, h$ | reference and hypothesis tokens | lists of str |
| $d_{i,j}$ | edits turning $r_{:i}$ into $h_{:j}$ | int |
| $S, D, I$ | substitutions, deletions, insertions | int |
| $N$ | reference length | int |
| $m$ | human answers equal to the prediction | int |

## Levenshtein alignment

The edit distance satisfies $d_{i,0} = i$, $d_{0,j} = j$, and

$$d_{i,j} = \min\big(d_{i-1,j-1} + [r_i \ne h_j],\ d_{i-1,j} + 1,\ d_{i,j-1} + 1\big),$$

the three moves being a match or substitution, a deletion of $r_i$, and an insertion of $h_j$. Then $d_{n,m}$ is the minimum $S + D + I$.

The table gives only the total. To count each kind of edit, you walk back from $(n, m)$ taking any move consistent with the table, and here is the catch: several paths can be consistent. "a b" against "b c" costs 2 either as two substitutions or as delete "a", keep "b", insert "c". The totals never depend on the choice, but the S, D, I split does, so the tie rule is part of the contract. The rule used throughout this book tries the diagonal first, then a deletion, then an insertion, so every implementation (including the Rust and Go ports) reports the same split. jiwer's backtrace (via rapidfuzz) tries a deletion first: its totals always equal these, but its split can differ.

## WER and CER

For a corpus, $\text{WER} = \sum_k (S_k + D_k + I_k) / \sum_k N_k$ over items $k$, with words from `str.split()`. Pooling weighs every reference word equally; the mean of per-item rates would let a one-word item count as much as a paragraph. An empty hypothesis is all deletions, giving WER 1, and a hypothesis with extra words can exceed 1, since insertions are unbounded while the denominator is fixed by the reference. CER is the same computation over the characters of the stripped strings, spaces included. When every reference is empty the rate is undefined, and the solution raises a `ValueError` rather than returning 0.

## Normalization

"Hello, World!" and "hello world" are the same transcript, so both sides must be normalized before scoring. Whisper's basic normalizer lowercases, deletes `<...>` and `[...]` tags and parenthesized asides, applies Unicode NFKC (folding ligatures and full-width forms), replaces every character whose category is a mark (M), symbol (S), or punctuation (P) with a space, and collapses white space. It keeps letters of every script and their accents, unless the accents are separate combining marks. Both sides must go through the same normalizer, or every comma is an error.

## Vision metrics

The vision metrics are each short once their edge rules are fixed. VQA collects ten human answers per question; an answer earns $\min(1, m/3)$, so three agreeing humans give full credit. Captions use corpus BLEU-4 and chrF with several references per item, plus attribute recall: the fraction of the attributes the references mention that the prediction also mentions. The attributes are the words of a given vocabulary or, without one, the words every reference shares. Zero-shot classification scores each class by similarity and counts an item when the true class is in the top $k$; equal scores rank by the lower class index.

## Intervals

A metric over a few hundred clips is uncertain, and a number without an interval invites overreading. `bootstrap_ci` resamples items with replacement (the percentile bootstrap, with uniforms from `PCG32(seed)`), recomputes the statistic on each resample, and reports the 2.5 and 97.5 percentiles. For WER the resampled unit is a row $(S, D, I, N)$, and each resample is pooled again, which gives the interval of a ratio rather than of per-item rates.

## Working it by hand

Take the reference "the cat sat on the mat" (6 words) and the hypothesis "the bat sat the mat too now" (7 words). Align them: the = the; cat to bat is a substitution; sat = sat; "on" has no partner, a deletion; the = the; mat = mat; "too" and "now" are insertions. So $S, D, I, N = 1, 1, 2, 6$ and $\text{WER} = 4/6 = 0.667$. No cheaper alignment exists: the hypothesis is one word longer, so $I - D = 1$, and "cat" cannot match anything.

Now the tie: "a b" against "b c". The table's last cell is 2. At $(2, 2)$, $r_2 = $ b and $h_2 = $ c differ and $d_{1,1} + 1 = 2$, so the diagonal is consistent and is taken, a substitution. At $(1, 1)$, a against b is again a substitution. The result is $(2, 0, 0, 2)$, where a deletion-first rule would give $(0, 1, 1, 2)$.

## Designing the API

```python
def basic_normalize(text: str) -> str: ...
def edit_ops(ref, hyp) -> tuple[int, int, int, int]: ...     # (S, D, I, N)
def wer_stats(refs, hyps) -> NDArray: ...                     # int64 [items, 4]
def wer(refs, hyps) -> float: ...
def cer(refs, hyps) -> float: ...
def vqa_accuracy(preds, answers) -> float: ...
def caption_scores(preds, refs, attributes=()) -> dict[str, float]: ...   # bleu4, chrf, attr_recall
def zero_shot_accuracy(pred, gold, k=1) -> float: ...
def bootstrap_ci(per_item, stat, n=1000, seed=0, alpha=0.05) -> tuple[float, float, float]: ...
```

A correct implementation reproduces the hand alignment, (1, 1, 2, 6) and WER 4/6, and the tie split, so every port gives identical breakdowns. It agrees with jiwer on corpus WER and CER, per-item totals and $N$, so its numbers are comparable with papers. It satisfies the WER invariants on random token lists: WER(x, x) = 0, an empty hypothesis gives 1, totals are symmetric, and bounds hold. It pools rather than averages, and CER strips and counts spaces. Bad corpora (all-empty references, mismatched lengths, a bare string) fail loudly. The normalizer agrees with Hugging Face's on tags, punctuation, NFKC, accents, CJK and emoji. VQA credit follows $\min(1, m/3)$ on normalized answers; BLEU-4 and chrF agree with sacreBLEU; attribute recall handles vocabulary attributes, skipped items and `nan`; zero-shot handles ties in top-$k$ and checks its arguments. The bootstrap interval can be recomputed by hand from PCG32 draws, which keeps the intervals reproducible in Python and Go.

The system's entry point, `tinyllm eval asr --pred SEGMENTS.json --ref TRANSCRIPT.txt` in [`cli_speech.py`](../../../../course/solutions/entry/python/tinyllm/cli_speech.py), joins generated segments before normalization and scoring. One substitution and one deletion in a six-word reference give WER 2/6, and malformed segment text is rejected.

## Building it step by step

The reasoning:

1. **WER must count the cheapest explanation.** Many edit sequences turn $r$ into $h$; the metric is the minimum, which the recurrence computes in an $(n+1) \times (m+1)$ table. Each cell depends only on its left, upper and upper-left neighbours, so a row-by-row fill is enough.
2. **The split needs a backtrace, and the backtrace needs a rule.** Walking back from $(n, m)$ along any move whose cost explains the cell recovers $S, D, I$; when several moves explain it, a fixed preference (diagonal, deletion, insertion) makes the split a function of the inputs, so every port agrees.
3. **The denominator is the reference.** A rate per reference word is defined for an empty hypothesis (all deletions, WER 1) and comparable across systems on the same test set. Dividing by the hypothesis length would reward deleting words.
4. **Pool, then divide.** Corpus WER is $\sum \text{edits} / \sum N$, so the per-item result must keep $(S, D, I, N)$ as integers, and the bootstrap must resample those rows and pool again.
5. **Normalize both sides identically.** Scoring is about words, not formatting. The normalizer must be deterministic Unicode processing, so it is written from category tests rather than language-specific rules.
6. **Small metrics are definitions.** VQA, attribute recall and top-$k$ accuracy are each a line or two once their edge rules (cap at 1, consensus set, tie order) are fixed.

### The normalizer

Tags go first, because they contain punctuation that would otherwise become spaces and leave the words inside. NFKC comes before the category test, so full-width and ligature forms fold first.

```python
def basic_normalize(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # <noise>, [laughter]
    s = re.sub(r"\(([^)]+?)\)", "", s)  # (inaudible)
    # Marks, symbols, and punctuation become spaces; letters and digits stay.
    s = "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKC", s)
    ).lower()                                 # lowercase again after NFKC folding
    return re.sub(r"\s+", " ", s)
```

### The alignment

The table fill is the recurrence verbatim, and the backtrace encodes the tie rule in the order of its branches.

```python
def edit_ops(ref, hyp):
    r, h = list(ref), list(hyp)
    n, m = len(r), len(h)
    # d[i][j]: edits to turn r[:i] into h[:j].
    d = np.zeros((n + 1, m + 1), dtype=np.int64)
    d[:, 0] = np.arange(n + 1)                # delete everything
    d[0, :] = np.arange(m + 1)                # insert everything
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = d[i - 1, j - 1] + (r[i - 1] != h[j - 1])   # match costs 0
            d[i, j] = min(sub, d[i - 1, j] + 1, d[i, j - 1] + 1)
    # Backtrace from the end; on equal cost prefer the diagonal, then a
    # deletion, then an insertion.
    S = D = I = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            S += int(r[i - 1] != h[j - 1])    # a match moves diagonally for free
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            D += 1
            i -= 1
        else:
            I += 1                            # the only remaining consistent move
            j -= 1
    return S, D, I, n
```

On "a b" against "b c", the first branch is consistent at $(2, 2)$ and at $(1, 1)$, giving the hand result $(2, 0, 0, 2)$.

### Pooled rates

`wer_stats` keeps one integer row per item; `_rate` sums before dividing and refuses an all-empty reference set. CER reuses `edit_ops` on character lists of the stripped strings.

```python
def wer_stats(refs, hyps):
    _pairs(refs, hyps)            # sequences of equal, nonzero length; not a bare str
    return np.array([edit_ops(r.split(), h.split()) for r, h in zip(refs, hyps)], dtype=np.int64)

def _rate(st):
    total = int(st[:, 3].sum())
    if total == 0:
        raise ValueError("every reference is empty: the error rate is undefined")
    return float(st[:, :3].sum()) / total     # pooled: sum of edits / sum of N

def cer(refs, hyps):
    _pairs(refs, hyps)
    st = np.array(
        [edit_ops(list(r.strip()), list(h.strip())) for r, h in zip(refs, hyps)],
        dtype=np.int64,
    )
    return _rate(st)
```

The `_pairs` check for a bare string matters more than it looks: `wer("abc", "abd")` would otherwise iterate characters and return a plausible number.

### Vision metrics

```python
def vqa_accuracy(preds, answers):
    _per_item(answers, len(preds), "answers")       # every item needs >= 1 answer
    total = 0.0
    for p, ans in zip(preds, answers):
        key = basic_normalize(p).strip()            # "Two." == "two"
        m = sum(basic_normalize(a).strip() == key for a in ans)
        total += min(1.0, m / 3.0)                  # three agreeing humans: full credit
    return total / len(preds)
```

In `caption_scores`, the attributes an item is scored on are the vocabulary words that *any* reference mentions (`set().union(*words) & vocab`), or, without a vocabulary, the words *every* reference shares (`set.intersection(*words)`). Items with no such attributes are skipped, and if none remain the recall is `nan` rather than 0 or 1. BLEU-4 and chrF come from the sequence metrics section.

For zero-shot accuracy, a stable sort does the tie-breaking:

```python
# A stable sort of -score ranks ties by the lower class index.
top = np.argsort(-s, axis=1, kind="stable")[:, :k]
return float((top == g[:, None]).any(axis=1).mean())
```

`zero_shot_accuracy` first requires `1 <= k <= classes` and integer gold labels in `[0, classes)`.

### The bootstrap

The percentile bootstrap from *Mathematics for Computing* resamples a 1-D array, so the solution bootstraps item *indices* and lets `stat` see the selected rows. For WER, `stat` is `_rate`-style pooling over the resampled $(S, D, I, N)$ rows.

```python
def bootstrap_ci(per_item, stat, n=1000, seed=0, alpha=0.05):
    a = np.asarray(per_item)
    idx = np.arange(len(a), dtype=np.float64)
    return _bootstrap(idx, lambda ix: float(stat(a[ix.astype(np.int64)])), n, alpha, PCG32(seed))
```

The full solution is [`mm.py`](../../../../course/solutions/python/tinyllm/eval/mm.py).

### Why this design

`edit_ops` is $O(nm)$ time and memory per item; for transcripts of tens to hundreds of words this is negligible next to decoding. The bootstrap costs $n$ recomputations of the statistic, $O(n \cdot \text{items})$ for pooled WER.

jiwer (rapidfuzz) is faster and reports the same totals, but its deletion-first backtrace gives a different split, and the Rust and Go ports need a rule they can reproduce exactly. Averaging per-item WER is simpler but weights a one-word clip like a paragraph. Bootstrapping words instead of items ignores that errors cluster within an utterance, and gives intervals that are too narrow.

Two tradeoffs are accepted. The basic normalizer does not spell out numbers or unify spellings (Whisper's English normalizer does), so "2" and "two" still differ. VQA uses the simple $\min(1, m/3)$ rule, not the official average over annotator subsets.

## What can go wrong

The alignment's failures change the split or the total. A backtrace that tries an insertion before the diagonal gives an S, D, I split that differs from the specification on ties; the branches must run diagonal, deletion, insertion. Swapping deletions and insertions reports missing words as hallucinated ones; an upward move (`i -= 1`, a reference word consumed) is a deletion. A substitution that costs 2 turns every substitution into a deletion plus an insertion; the diagonal adds `r[i-1] != h[j-1]`, at most 1.

> **Warning:** Averaging per-item rates lets short items dominate the corpus WER, and dividing by the hypothesis length leaves the WER of an empty hypothesis undefined. `_rate` sums edits and reference lengths before dividing, and divides by $N$, the reference length.

The remaining pitfalls, compactly:

| Mistake | Symptom | Fix in the solution |
|---|---|---|
| no NFKC before the category test | full-width and ligature text never matches | `unicodedata.normalize("NFKC", s)` first |
| keeping `[NOISE]` tags | tags count as words | the tag regexes run first |
| CER without strip | leading spaces count as errors | aligns `list(r.strip())` with `list(h.strip())` |
| CER over words | CER equals WER | passes character lists to `edit_ops` |
| VQA credit not capped | accuracy above 1 | `min(1.0, m / 3.0)` |
| VQA answers compared raw | "Two." never equals "two" | `basic_normalize(...).strip()` on both sides |
| ties ranked by the higher index | top-1 changes with class order | stable `argsort(-s)` |
| a bootstrap that never resamples | a zero-width interval | resamples indices with `PCG32(seed)` |
| consensus attributes as the union | recall measured on words only one annotator used | `set.intersection(*words)` without a vocabulary |
| vocabulary attributes as the intersection | attributes one reference mentions are missed | `set().union(*words) & vocab` |
| all-empty references | division by zero instead of a clear error | `_rate` raises `ValueError` when $\sum N = 0$ |
| accepting `k = 0` | accuracy 0 by construction | requires `1 <= k <= c` |
| a label equal to the class count | an impossible label scored as a miss | requires gold in `[0, c)` |
| an item with no answers | the question silently scores 0 | `_per_item` rejects empty answer lists |

## Where this leads

These functions are the shared scoring layer for the rest of the system. In *Responsible AI*, [Consent and biometric evidence in model cards](../../responsible-ai/02-privacy-and-pii/02-consent-and-biometric-evidence.md) calls them to compute WER and CER evidence. Later chapters build on them as well: the bias evaluations compute per-group WER and gaps with two-sample bootstrap intervals, the audio corpus pipeline applies `basic_normalize` to transcripts before deduplication and splits, and the durable evaluation suites score their `asr`, `vqa`, `caption` and `zeroshot` rows with these functions.

### Further reading

- Levenshtein, "Binary codes capable of correcting deletions, insertions, and reversals" (1966).
- Antol et al., "VQA: Visual Question Answering" (2015), section 4.
- Radford et al. (2022), appendix C, on text normalization.
- Efron and Tibshirani, *An Introduction to the Bootstrap* (1993), chapter 13.
- jiwer (rapidfuzz) adds bit-parallel Levenshtein and alignment visualization; see `jiwer/process.py`.
- Whisper's `EnglishTextNormalizer` adds spelled-out numbers, British to American spelling, and contractions; see `whisper/normalizers/english.py`.
- The official VQA evaluation averages over the 10 choose 9 annotator subsets and normalizes numbers and articles; see `VQA/PythonEvaluationTools/vqaEvaluation`.
- pycocoevalcap adds CIDEr and SPICE, consensus-based caption metrics; see `pycocoevalcap/cider`.

## Key terms

- **word error rate (WER)**: pooled substitutions, deletions and insertions divided by the total reference word count.
- **character error rate (CER)**: the same rate computed over the characters of the stripped strings, spaces included.
- **tie rule**: the fixed backtrace preference (diagonal, deletion, insertion) that makes the S, D, I split deterministic.
- **basic normalizer**: Whisper's lowercase, tag-removal, NFKC and category-based punctuation stripping applied to both sides before scoring.
- **attribute recall**: the fraction of reference-mentioned attributes that a caption also mentions.
- **item bootstrap**: resampling whole items with replacement and recomputing the pooled statistic on each resample.

## Summary

- WER is the minimum number of word substitutions, deletions and insertions, pooled over the corpus and divided by the reference word count; it can exceed 1.
- Equal-cost alignments exist; totals never depend on the choice, but the S, D, I split does, so the tie rule is part of the contract.
- Normalize before scoring: Whisper's basic normalizer drops bracketed tags and turns punctuation and symbols into spaces.
- VQA, caption attribute recall and zero-shot accuracy are short definitions whose edge rules (cap at 1, consensus set, tie order) must be fixed.
- Every metric gets a bootstrap interval that resamples items, not words.

## Review questions

1. Why can WER exceed 1?
   <details><summary>Answer</summary>

   Insertions are unbounded: a hypothesis with many extra words has $I$ larger than $N$. The denominator is the reference length, which does not grow with the hypothesis.

   </details>

2. Why does the order of branches in the backtrace not change the WER?
   <details><summary>Answer</summary>

   Every branch taken is consistent with the table, so each step subtracts exactly the cost of its move, and the path's total is $d_{n,m}$ whichever path is followed. Only the attribution of that total to S, D or I depends on the order.

   </details>

3. Why are tags removed before punctuation is turned into spaces?
   <details><summary>Answer</summary>

   After punctuation becomes spaces, `[laughter]` is the word `laughter`, which the tag pattern can no longer find. Removing tags first deletes the whole annotation.

   </details>

4. Why does the WER bootstrap resample rows of `(S, D, I, N)` rather than per-item WER values?
   <details><summary>Answer</summary>

   The statistic is a pooled ratio. Resampling rows and pooling again gives the interval of that ratio; resampling per-item rates would give the interval of their mean, a different (and differently weighted) statistic.

   </details>

5. Align "the cat sat on the mat" with "the bat sat the mat too now". What are $S, D, I, N$ and the WER?
   <details><summary>Answer</summary>

   One substitution (cat to bat), one deletion ("on"), two insertions ("too", "now"): $(1, 1, 2, 6)$, so WER $= 4/6 = 0.667$.

   </details>
