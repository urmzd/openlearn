# Multimodal eval metrics: WER, CER, VQA, captions, zero-shot

## Overview

| | |
|---|---|
| **Module** | `L14.5` · build · Python · Pass 12 · 1.5 to 2 h |
| **Solution** | [`mm.py`](../../../../course/solutions/python/tinyllm/eval/mm.py): `basic_normalize`, `edit_ops` (Levenshtein with a fixed tie rule), `wer_stats`, `wer`, `cer`, `vqa_accuracy`, `caption_scores`, `zero_shot_accuracy`, `bootstrap_ci` |
| **Contract** | [`course/contracts/py/tinyllm/eval/mm.pyi`](../../../../course/contracts/py/tinyllm/eval/mm.pyi) |
| **Needs** | `L4.5` `corpus_bleu`, `chrf` · `M07.4` `bootstrap_ci` · `M06.3` `PCG32`. Reading: `M05.2` (text as Unicode code points) |
| **Used by** | `ethics.08`; later `ethics.07` (per-group WER gaps) · `data.11` (the normalizer) · `dur.13` (multimodal `EvalSuite` rows); they join the registry with B14's ethics, data, and durable groups. The system's `tinyllm eval asr` command reports WER with a bootstrap `wer_ci` using these functions |
| **Optional depth** | Levenshtein, "Binary codes capable of correcting deletions, insertions, and reversals" (1966); Antol et al., "VQA: Visual Question Answering" (2015), section 4; Radford et al. (2022), appendix C (text normalization); Efron and Tibshirani, *An Introduction to the Bootstrap* (1993), chapter 13 |

## Key Takeaways

- WER is the minimum number of word substitutions, deletions, and insertions, pooled over the corpus and divided by the reference word count; it can exceed 1.
- Equal-cost alignments exist; the totals never depend on the choice, the S, D, I split does, so the tie rule is part of the contract.
- Normalize before scoring: Whisper's basic normalizer drops bracketed tags and turns punctuation and symbols into spaces.
- Every metric gets a bootstrap interval that resamples items, not words.

## How to read this chapter

Sections 1 to 3 define each metric from first principles and work an alignment and a tie by hand. Section 4 fixes the interface and the cases a correct implementation must handle. Section 5 walks through the solution function by function and explains why each convention (pooling, the tie rule, the normalizer, the bootstrap unit) is the right one. Try to predict each choice before reading it.

---

## 1. Why now

`L14.4` produces transcripts and Pass 12's vision models produce answers and captions; nothing in the course can yet say whether they are good. Release gates (`dur.13`) block a model on a WER ceiling, the bias evals (`ethics.07`) compare WER across speaker groups, and the audio corpus (`data.11`) normalizes transcripts the same way. All of them need one implementation whose numbers match what papers report (jiwer, sacreBLEU), whose ambiguous cases are pinned, and whose intervals are reproducible from a seed across languages.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $r, h$ | reference and hypothesis tokens | lists of str |
| $d_{i,j}$ | edits turning $r_{:i}$ into $h_{:j}$ | int |
| $S, D, I$ | substitutions, deletions, insertions | int |
| $N$ | reference length | int |
| $m$ | human answers equal to the prediction | int |

### 2.1 Levenshtein alignment

The edit distance satisfies $d_{i,0} = i$, $d_{0,j} = j$, and
$$d_{i,j} = \min\big(d_{i-1,j-1} + [r_i \ne h_j],\ d_{i-1,j} + 1,\ d_{i,j-1} + 1\big),$$
the three moves being a match or substitution, a deletion of $r_i$, and an insertion of $h_j$. $d_{n,m}$ is the minimum $S + D + I$. To count each kind, walk back from $(n, m)$ taking any move consistent with the table. Several paths can be consistent: "a b" against "b c" costs 2 as two substitutions or as delete "a", keep "b", insert "c". The course rule tries the diagonal first, then a deletion, then an insertion, so every implementation (and the Rust and Go ports) reports the same split. jiwer's backtrace (rapidfuzz) tries a deletion first: its totals always equal ours, its split can differ.

### 2.2 WER and CER

For a corpus, $\text{WER} = \sum_k (S_k + D_k + I_k) / \sum_k N_k$ over items $k$, with words from `str.split()`. Pooling weighs every reference word equally; the mean of per-item rates would let a one-word item count as much as a paragraph. An empty hypothesis is all deletions (WER 1); a hypothesis with extra words can exceed 1. CER is the same over the characters of the stripped strings, spaces included. With every reference empty the rate is undefined: a `ValueError`, not 0.

### 2.3 Normalization

"Hello, World!" and "hello world" are the same transcript. Whisper's basic normalizer lowercases, deletes `<...>` and `[...]` tags and parenthesized asides, applies Unicode NFKC (folding ligatures and full-width forms), replaces every character whose category is a mark (M), symbol (S), or punctuation (P) with a space, and collapses white space. It keeps letters of every script and their accents (unless they are separate combining marks). Both sides must go through the same normalizer, or every comma is an error.

### 2.4 Vision metrics

VQA collects ten human answers per question; an answer earns $\min(1, m/3)$, so three agreeing humans give full credit. Captions use `L4.5`'s corpus BLEU-4 and chrF with several references per item, plus attribute recall: the fraction of the attributes the references mention (words of a given vocabulary, or, without one, words every reference shares) that the prediction also mentions. Zero-shot classification scores each class by similarity and counts an item when the true class is in the top $k$; equal scores rank by the lower class index.

### 2.5 Intervals

A metric over a few hundred clips is uncertain. `bootstrap_ci` resamples items with replacement (`M07.4`'s percentile bootstrap, uniforms from `PCG32(seed)`), recomputes the statistic on each resample, and reports the 2.5 and 97.5 percentiles. For WER the resampled unit is a row $(S, D, I, N)$, and each resample is pooled again: the interval of a ratio, not of per-item rates.

## 3. Worked example by hand

Reference "the cat sat on the mat" (6 words), hypothesis "the bat sat the mat too now" (7 words). Align: the = the; cat to bat is a substitution; sat = sat; "on" has no partner, a deletion; the = the; mat = mat; "too" and "now" are insertions. $S, D, I, N = 1, 1, 2, 6$ and $\text{WER} = 4/6 = 0.667$. No cheaper alignment exists: the hypothesis is one word longer, so $I - D = 1$, and "cat" cannot match anything.

Tie: "a b" against "b c". The table's last cell is 2. At $(2, 2)$, $r_2 = $ b and $h_2 = $ c differ and $d_{1,1} + 1 = 2$: the diagonal is consistent, so it is taken (a substitution); at $(1, 1)$, a against b, again a substitution. Result $(2, 0, 0, 2)$, where a deletion-first rule would give $(0, 1, 1, 2)$.

## 4. The interface

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

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Section 3's alignment | unit, smoke | (1, 1, 2, 6) and WER 4/6 | the definition |
| Equal-cost alignments | unit | the S, D, I split on ties | identical breakdowns in every port |
| Agreement with jiwer | golden | corpus WER and CER, per-item totals and N | numbers comparable with papers |
| WER invariants | property | WER(x, x) = 0, empty hypothesis 1, symmetry of totals, bounds | sanity on random token lists |
| Pooling | unit | pooling vs averaging; CER strips and counts spaces | corpus WER for release gates |
| Bad corpora | boundary | all-empty references, mismatched lengths, a bare string | caller bugs fail loudly |
| Agreement with Whisper's normalizer | golden | tags, punctuation, NFKC, accents, CJK, emoji vs HF's normalizer | the normalizer `data.11` reuses |
| VQA credit | unit | $\min(1, m/3)$ with normalized answers | VQA suite scores |
| Agreement with sacreBLEU | golden | BLEU-4 and chrF; consensus attribute recall | caption suite scores |
| Attribute recall | unit | vocabulary attributes, skipped items, nan | attribute probes |
| Zero-shot ties | unit | top-k with ties, argument checks | zero-shot suite scores |
| `eval asr` on generated segments | golden | one substitution and one deletion over six words gives WER 2/6 | scoring generated transcript files |
| Bootstrap | property | the interval recomputed by hand from PCG32 draws | reproducible CIs in Python and Go |

The system's entry point `tinyllm eval asr --pred SEGMENTS.json --ref TRANSCRIPT.txt` ([`cli_speech.py`](../../../../course/solutions/entry/python/tinyllm/cli_speech.py)) joins generated segments before normalization and scoring. One substitution and one deletion in a six-word reference give WER 2/6, and malformed segment text is rejected.

## 5. The solution, step by step

### From first principles

1. **WER must count the cheapest explanation.** Many edit sequences turn $r$ into $h$; the metric is the minimum, which section 2.1's recurrence computes in an $(n+1) \times (m+1)$ table. Each cell depends only on its left, upper and upper-left neighbours, so a row-by-row fill is enough.
2. **The split needs a backtrace, and the backtrace needs a rule.** The table gives only the total. Walking back from $(n, m)$ along any move whose cost explains the cell recovers $S, D, I$; when several moves explain it, a fixed preference (diagonal, deletion, insertion) makes the split a function of the inputs, so every port agrees.
3. **The denominator is the reference.** A rate per reference word is defined for an empty hypothesis (all deletions, WER 1) and comparable across systems on the same test set. Dividing by the hypothesis length would reward deleting words.
4. **Pool, then divide.** Corpus WER is $\sum \text{edits} / \sum N$; so the per-item result must keep $(S, D, I, N)$ as integers, and the bootstrap must resample those rows and pool again.
5. **Normalize both sides identically.** Scoring is about words, not formatting. The normalizer must be deterministic Unicode processing, so it is written from category tests rather than language-specific rules.
6. **Small metrics are definitions.** VQA, attribute recall and top-$k$ accuracy are each a line or two once their edge rules (cap at 1, consensus set, tie order) are fixed.

### The code, annotated

**The normalizer.** Tags first (they contain punctuation that would otherwise become spaces and leave the words inside), then NFKC so full-width and ligature forms fold before the category test.

```python
def basic_normalize(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # <noise>, [laughter]
    s = re.sub(r"\(([^)]+?)\)", "", s)  # (inaudible)
    # Marks, symbols, and punctuation become spaces; letters and digits stay.
    s = "".join(
        " " if unicodedata.category(c)[0] in "MSP" else c
        for c in unicodedata.normalize("NFKC", s)
    ).lower()                                 # NFKC can produce new uppercase forms
    return re.sub(r"\s+", " ", s)
```

**The alignment.** The table fill is the recurrence verbatim; the backtrace encodes the tie rule in the order of its branches.

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

On "a b" against "b c", the first branch is consistent at $(2, 2)$ and at $(1, 1)$, giving section 3's $(2, 0, 0, 2)$.

**Pooled rates.** `wer_stats` keeps one integer row per item; `_rate` sums before dividing and refuses an all-empty reference set. CER reuses `edit_ops` on character lists of the stripped strings.

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

The `_pairs` check for a bare string matters: `wer("abc", "abd")` would otherwise iterate characters and return a plausible number.

**Vision metrics.**

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

In `caption_scores`, the attributes an item is scored on are the vocabulary words that *any* reference mentions (`set().union(*words) & vocab`), or, without a vocabulary, the words *every* reference shares (`set.intersection(*words)`). Items with no such attributes are skipped, and if none remain the recall is `nan` rather than 0 or 1. BLEU-4 and chrF come from L4.5.

```python
# A stable sort of -score ranks ties by the lower class index.
top = np.argsort(-s, axis=1, kind="stable")[:, :k]
return float((top == g[:, None]).any(axis=1).mean())
```

`zero_shot_accuracy` first requires `1 <= k <= classes` and integer gold labels in `[0, classes)`.

**The bootstrap.** M07.4's percentile bootstrap resamples a 1-D array, so the solution bootstraps item *indices* and lets `stat` see the selected rows. For WER, `stat` is `_rate`-style pooling over the resampled $(S, D, I, N)$ rows.

```python
def bootstrap_ci(per_item, stat, n=1000, seed=0, alpha=0.05):
    a = np.asarray(per_item)
    idx = np.arange(len(a), dtype=np.float64)
    return _bootstrap(idx, lambda ix: float(stat(a[ix.astype(np.int64)])), n, alpha, PCG32(seed))
```

Full solution: [`mm.py`](../../../../course/solutions/python/tinyllm/eval/mm.py).

### Why this solution

- **Cost.** `edit_ops` is $O(nm)$ time and memory per item; for transcripts of tens to hundreds of words this is negligible next to decoding. The bootstrap costs $n$ recomputations of the statistic, $O(n \cdot \text{items})$ for pooled WER.
- **Alternatives.** jiwer (rapidfuzz) is faster and reports the same totals, but its deletion-first backtrace gives a different split, and the Rust and Go ports need a rule they can reproduce exactly. Averaging per-item WER is simpler but weights a one-word clip like a paragraph. Bootstrapping words instead of items ignores that errors cluster within an utterance and gives intervals that are too narrow.
- **Tradeoffs accepted.** The basic normalizer does not spell out numbers or unify spellings (Whisper's English normalizer does), so "2" and "two" still differ. VQA uses the simple $\min(1, m/3)$ rule, not the official average over annotator subsets.

### Explain it back

<details><summary>Why can WER exceed 1?</summary>

Insertions are unbounded: a hypothesis with many extra words has $I$ larger than $N$. The denominator is the reference length, which does not grow with the hypothesis.

</details>

<details><summary>Why does the order of branches in the backtrace not change the WER?</summary>

Every branch taken is consistent with the table, so each step subtracts exactly the cost of its move, and the path's total is $d_{n,m}$ whichever path is followed. Only the attribution of that total to S, D or I depends on the order.

</details>

<details><summary>Why are tags removed before punctuation is turned into spaces?</summary>

After punctuation becomes spaces, `[laughter]` is the word `laughter`, which the tag pattern can no longer find. Removing tags first deletes the whole annotation.

</details>

<details><summary>Why does the WER bootstrap resample rows of <code>(S, D, I, N)</code> rather than per-item WER values?</summary>

The statistic is a pooled ratio. Resampling rows and pooling again gives the interval of that ratio; resampling per-item rates would give the interval of their mean, a different (and differently weighted) statistic.

</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. a backtrace that tries an insertion before the diagonal | S, D, I split differs from the spec on ties | branches in order: diagonal, deletion, insertion |
| 2. averaging per-item rates | short items dominate the corpus WER | `_rate` sums edits and reference lengths before dividing |
| 3. deletions and insertions swapped | missing words reported as hallucinated ones | an upward move (`i -= 1`, a reference word consumed) counts as D |
| 4. no NFKC before the category test | full-width and ligature text never matches | `unicodedata.normalize("NFKC", s)` before the category test |
| 5. keeping `[NOISE]` tags | tags count as words | the tag regexes run first |
| 6. CER without strip | leading spaces count as errors | `cer` aligns `list(r.strip())` with `list(h.strip())` |
| 7. VQA credit not capped | accuracy above 1 | `min(1.0, m / 3.0)` |
| 8. VQA answers compared raw | "Two." never equals "two" | both sides go through `basic_normalize(...).strip()` |
| 9. ties ranked by the higher index | top-1 changes with class order | stable `argsort(-s)` |
| 10. a bootstrap that never resamples | a zero-width interval | resamples item indices with `PCG32(seed)` via M07.4 |
| 11. CER over words | CER equals WER | `cer` passes character lists to `edit_ops` |
| 12. dividing by the hypothesis length | WER of an empty hypothesis undefined | divides by `N`, the reference length |
| 13. a substitution costing 2 | substitutions become a deletion and an insertion | the diagonal adds `r[i-1] != h[j-1]`, at most 1 |
| 14. consensus attributes as the union | recall measured on words only one annotator used | without a vocabulary, `set.intersection(*words)` |
| 15. vocabulary attributes as the intersection | attributes one reference mentions are missed | with a vocabulary, `set().union(*words) & vocab` |
| an all-empty reference set gives a division by zero | a crash instead of a clear error | `_rate` raises `ValueError` when $\sum N = 0$ |
| accepting k = 0 | accuracy 0 by construction | requires `1 <= k <= c` |
| accepting a label equal to the class count | an impossible label scored as a miss | requires gold in `[0, c)` |
| accepting an item with no answers | the question scores 0 silently | `_per_item` rejects empty answer lists |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L4.5` | `corpus_bleu` and `chrf` score captions |
| Back | `M07.4` | `bootstrap_ci` gives every interval |
| Back | `M06.3` | `PCG32(seed)` is the resampling stream |
| Back | `M05.2` | text as Unicode code points (reading) |
| Forward | `ethics.07` | per-group WER and gaps with two-sample bootstrap intervals (with B14's ethics group) |
| Forward | `data.11` | `basic_normalize` on transcripts before dedup and splits |
| Forward | `dur.13` | the `asr`, `vqa`, `caption`, `zeroshot` suites score with these functions |
| Forward | `ethics.08` | The consent and model-card evidence practice calls this implementation. |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `edit_ops`, `wer` | jiwer (rapidfuzz) | bit-parallel Levenshtein, alignment visualization | `jiwer/process.py` |
| `basic_normalize` | Whisper's `EnglishTextNormalizer` | spelled-out numbers, British to American spelling, contractions | `whisper/normalizers/english.py` |
| `vqa_accuracy` | the official VQA evaluation | averages over the 10 choose 9 annotator subsets, number and article normalization | `VQA/PythonEvaluationTools/vqaEvaluation` |
| `caption_scores` | pycocoevalcap | CIDEr and SPICE, consensus-based caption metrics | `pycocoevalcap/cider` |
