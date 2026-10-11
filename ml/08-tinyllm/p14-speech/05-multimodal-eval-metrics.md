<!-- ol:module L14.5 -->
# Multimodal eval metrics: WER, CER, VQA, captions, zero-shot

## Overview

| | |
|---|---|
| **Module** | `L14.5` · build · Python · Pass 12 · 3 to 4 h |
| **You build** | `python/tinyllm/eval/mm.py`: `basic_normalize`, `edit_ops` (Levenshtein with a fixed tie rule), `wer_stats`, `wer`, `cer`, `vqa_accuracy`, `caption_scores`, `zero_shot_accuracy`, `bootstrap_ci` |
| **Contract** | [`course/contracts/py/tinyllm/eval/mm.pyi`](../../../course/contracts/py/tinyllm/eval/mm.pyi) |
| **Tests** | `course/tests/L14.5/test_mm_metrics.py` (what they check: section 4), golden values from `course/oracle/L14.5/metrics_oracle.py` (jiwer 4.0, sacreBLEU 2.6, transformers' `BasicTextNormalizer`) |
| **Needs** | `L4.5` `corpus_bleu`, `chrf` · `M07.4` `bootstrap_ci` · `M06.3` `PCG32` (or `--ref-deps`). Reading: `M05.2` (text as Unicode code points) |
| **Used by** | later `ethics.07` (per-group WER gaps) · `data.11` (the normalizer) · `dur.13` (multimodal `EvalSuite` rows); they join the registry with B14's ethics, data, and durable groups |
| **Milestone** | `MS-L14` steps 7 and 8 (`eval asr`: WER within 0.01 of the oracle run, with `wer_ci`) |
| **Optional depth** | Levenshtein, "Binary codes capable of correcting deletions, insertions, and reversals" (1966); Antol et al., "VQA: Visual Question Answering" (2015), section 4; Radford et al. (2022), appendix C (text normalization); Efron and Tibshirani, *An Introduction to the Bootstrap* (1993), chapter 13 |

## Key Takeaways

- WER is the minimum number of word substitutions, deletions, and insertions, pooled over the corpus and divided by the reference word count; it can exceed 1 (`test_hand_example_alignment`, `test_corpus_wer_pools_edits`).
- Equal-cost alignments exist; the totals never depend on the choice, the S, D, I split does, so the tie rule is part of the contract (`test_tie_rule_prefers_substitution`).
- Normalize before scoring: Whisper's basic normalizer drops bracketed tags and turns punctuation and symbols into spaces (`test_basic_normalize_matches_whisper`).
- Every metric gets a bootstrap interval that resamples items, not words (`test_bootstrap_ci_resamples_items`).

## How to work this chapter

```bash
ol start L14.5              # stubs python/tinyllm/eval/mm.py into your repo
ol tests L14.5              # read the test catalog first: rung R0, you write no tests here
ol check L14.5              # exit code is the verdict
ol check L14.5 --ref-deps   # only if your L4.5, M07.4, or M06.3 is not passing yet
ol diff  L14.5              # after passing: your code against the reference
```

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

Reference "the cat sat on the mat" (6 words), hypothesis "the bat sat the mat too now" (7 words). Align: the = the; cat to bat is a substitution; sat = sat; "on" has no partner, a deletion; the = the; mat = mat; "too" and "now" are insertions. $S, D, I, N = 1, 1, 2, 6$ and $\text{WER} = 4/6 = 0.667$. No cheaper alignment exists: the hypothesis is one word longer, so $I - D = 1$, and "cat" cannot match anything. This is `test_hand_example_alignment`.

Tie: "a b" against "b c". The table's last cell is 2. At $(2, 2)$, $r_2 = $ b and $h_2 = $ c differ and $d_{1,1} + 1 = 2$: the diagonal is consistent, so it is taken (a substitution); at $(1, 1)$, a against b, again a substitution. Result $(2, 0, 0, 2)$, where a deletion-first rule would give $(0, 1, 1, 2)$ (`test_tie_rule_prefers_substitution`).

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_alignment` | unit, smoke | section 3: (1, 1, 2, 6) and WER 4/6 | the definition |
| `test_tie_rule_prefers_substitution` | unit | the S, D, I split on equal-cost alignments | identical breakdowns in every port |
| `test_wer_and_cer_match_jiwer` | golden | corpus WER and CER, per-item totals and N, against jiwer | numbers comparable with papers |
| `test_wer_invariants` | property | WER(x, x) = 0, empty hypothesis 1, symmetry of totals, bounds | sanity on random token lists |
| `test_corpus_wer_pools_edits` | unit | pooling vs averaging; CER strips and counts spaces | corpus WER for release gates |
| `test_rejects_bad_corpora` | boundary | all-empty references, mismatched lengths, a bare string | caller bugs fail loudly |
| `test_basic_normalize_matches_whisper` | golden | tags, punctuation, NFKC, accents, CJK, emoji vs HF's normalizer | the normalizer `data.11` reuses |
| `test_vqa_accuracy` | unit | $\min(1, m/3)$ with normalized answers | VQA suite scores |
| `test_caption_scores_match_sacrebleu` | golden | BLEU-4 and chrF vs sacreBLEU; consensus attribute recall | caption suite scores |
| `test_attribute_recall` | unit | vocabulary attributes, skipped items, nan | attribute probes |
| `test_zero_shot_topk_ties_to_lower_index` | unit | top-k with ties, argument checks | zero-shot suite scores |
| `test_bootstrap_ci_resamples_items` | property | the interval recomputed by hand from PCG32 draws | reproducible CIs in Python and Go |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. a backtrace that tries an insertion before the diagonal | S, D, I split differs from the spec on ties | `test_tie_rule_prefers_substitution` (mutant `s01`) |
| 2. averaging per-item rates | short items dominate the corpus WER | `test_corpus_wer_pools_edits`, `test_wer_and_cer_match_jiwer` (mutant `s02`) |
| 3. deletions and insertions swapped | missing words reported as hallucinated ones | `test_hand_example_alignment` (mutant `s03`) |
| 4. no NFKC before the category test | full-width and ligature text never matches | `test_basic_normalize_matches_whisper` (mutant `s04`) |
| 5. keeping `[NOISE]` tags | tags count as words | `test_basic_normalize_matches_whisper` (mutant `s05`) |
| 6. CER without strip | leading spaces count as errors | `test_corpus_wer_pools_edits` (mutant `s06`) |
| 7. VQA credit not capped | accuracy above 1 | `test_vqa_accuracy` (mutant `s07`) |
| 8. VQA answers compared raw | "Two." never equals "two" | `test_vqa_accuracy` (mutant `s08`) |
| 9. ties ranked by the higher index | top-1 changes with class order | `test_zero_shot_topk_ties_to_lower_index` (mutant `s09`) |
| 10. a bootstrap that never resamples | a zero-width interval | `test_bootstrap_ci_resamples_items` (mutant `s10`) |
| 11. CER over words | CER equals WER | `test_corpus_wer_pools_edits`, `test_wer_and_cer_match_jiwer` (mutant `s11`) |
| 12. dividing by the hypothesis length | WER of an empty hypothesis undefined | `test_hand_example_alignment` (mutant `s12`) |
| 13. a substitution costing 2 | substitutions become a deletion and an insertion | `test_hand_example_alignment`, `test_wer_and_cer_match_jiwer` (mutant `s13`) |
| 14. consensus attributes as the union | recall measured on words only one annotator used | `test_caption_scores_match_sacrebleu` (mutant `s14`) |
| 15. vocabulary attributes as the intersection | attributes one reference mentions are missed | `test_attribute_recall` (mutant `s15`) |
| an all-empty reference set gives a division by zero | a crash instead of a clear error | `test_rejects_bad_corpora` (mutant `m01`) |
| accepting k = 0 | accuracy 0 by construction | `test_zero_shot_topk_ties_to_lower_index` (mutant `m02`) |
| accepting a label equal to the class count | an impossible label scored as a miss | `test_zero_shot_topk_ties_to_lower_index` (mutant `m03`) |
| accepting an item with no answers | the question scores 0 silently | `test_vqa_accuracy` (mutant `m04`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L4.5` | `corpus_bleu` and `chrf` score captions |
| Back | `M07.4` | `bootstrap_ci` gives every interval |
| Back | `M06.3` | `PCG32(seed)` is the resampling stream |
| Back | `M05.2` | text as Unicode code points (reading) |
| Forward | `ethics.07` | per-group WER and gaps with two-sample bootstrap intervals (with B14's ethics group) |
| Forward | `data.11` | `basic_normalize` on transcripts before dedup and splits |
| Forward | `dur.13` | the `asr`, `vqa`, `caption`, `zeroshot` suites score with these functions |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `edit_ops`, `wer` | jiwer (rapidfuzz) | bit-parallel Levenshtein, alignment visualization | `jiwer/process.py` |
| `basic_normalize` | Whisper's `EnglishTextNormalizer` | spelled-out numbers, British to American spelling, contractions | `whisper/normalizers/english.py` |
| `vqa_accuracy` | the official VQA evaluation | averages over the 10 choose 9 annotator subsets, number and article normalization | `VQA/PythonEvaluationTools/vqaEvaluation` |
| `caption_scores` | pycocoevalcap | CIDEr and SPICE, consensus-based caption metrics | `pycocoevalcap/cider` |
