<!-- ol:module L4.5 -->
# Sequence metrics: exact match, BLEU, chrF

## Overview

| | |
|---|---|
| **Module** | `L4.5` · build · Python · Pass 4 · 2 to 3 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/eval/seqmetrics.py`: `exact_match`, `tokenize_13a`, `bleu_stats`, `bleu_from_stats`, `corpus_bleu`, `chrf_stats`, `chrf_from_stats`, `chrf`, `sentence_stats`, `score_from_stats`, `metric_ci`; and your own tests in `python/tests/l4-5-seqmetrics/` |
| **Contract** | [`course/contracts/py/tinyllm/eval/seqmetrics.pyi`](../../../../course/contracts/py/tinyllm/eval/seqmetrics.pyi) |
| **Tests** | `course/tests/L4.5/` (what they check: section 4), golden values from sacreBLEU 2.5.1 in `course/fixtures/L4.5/sacrebleu_golden.json`; your tests are graded by mutation, threshold 0.80 with every semantic fault required |
| **Needs** | `M07.4` `bootstrap_ci` (the interval every score carries) (or `--ref-deps`) |
| **Used by** | later: `L6.7` scores the model zoo's `seq2seq` and `transformer` rows with these functions; `ag.10` re-implements them in Go; the `translate` verb of your CLI reports them in `MS-L4` · `L14.5` caption BLEU-4 and chrF |
| **Milestone** | `MS-L4` |
| **Optional depth** | Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation" (ACL 2002); Popovic, "chrF: character n-gram F-score for automatic MT evaluation" (WMT 2015); Post, "A Call for Clarity in Reporting BLEU Scores" (WMT 2018); Koehn, "Statistical Significance Tests for Machine Translation Evaluation" (EMNLP 2004) |

## Key Takeaways

- BLEU counts word n-grams of orders 1 to 4 that the hypothesis shares with a reference, **clipped** by the reference counts, takes the geometric mean of the four precisions, and multiplies by a brevity penalty measured against the reference **closest** in length (`test_hand_example_bleu`, `test_clipping`, `test_closest_reference_length`).
- A corpus score pools sufficient statistics over sentences; it is not the mean of sentence scores (`test_corpus_pools_statistics`). Pooling is also what makes a bootstrap over sentences cheap.
- chrF counts character n-grams with white space removed, averages precision and recall over orders 1 to 6, and weights recall by $\beta^2 = 4$; it forgives "cats" for "cat" where BLEU sees a miss (`test_hand_example_chrf`, `test_chrf_ignores_white_space_and_forgives_inflection`).
- The tokenizer, the smoothing, the case rule, and the reference layout are part of the metric: change one and the number is no longer comparable. Matching sacreBLEU's defaults exactly makes yours comparable with published scores (`test_bleu_matches_sacrebleu`, `test_chrf_matches_sacrebleu`).
- A score without an interval is a guess: `metric_ci` resamples sentences with `M07.4`'s bootstrap and reproduces an independent sacreBLEU run seed for seed (`test_metric_ci_matches_independent_bootstrap`).

## How to work this chapter

```bash
ol start L4.5              # stubs seqmetrics.py; prints your test path and rung (R5)
ol tests L4.5              # the course tests
# compute the section 3 numbers by hand, write them as tests in python/tests/l4-5-seqmetrics/, then:
ol check L4.5              # course tests and the mutation grade of your tests
ol mutate L4.5             # the full grade, cached by your test files' hash
```

---

## 1. Why now

Your seq2seq models of `L4.1` to `L4.4` now write dates, and soon translations and stories. The training loss says how surprised a model is by the reference, not whether what it **generates** is right: beam search can produce a fluent output the loss never saw. So the system needs metrics over generated text. The simplest, exact match, is the `MS-L4` bar (a date is right or wrong). For free text, the field reports BLEU and chrF, and reports them in one exact configuration: sacreBLEU's. Two "BLEU" implementations that differ in tokenization or smoothing disagree by several points on the same output, which is larger than most of the improvements papers claim. This module builds all three metrics from their sufficient statistics, matches sacreBLEU to twelve digits, and attaches the bootstrap interval from `M07.4`, so every number the `L6.7` evaluation harness prints later is both comparable and honest about its noise.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $h$, $\{r_1, \dots, r_K\}$ | one hypothesis and its references | strings |
| $\mathrm{cnt}_h(g)$, $\mathrm{cnt}_{r}(g)$ | how many times n-gram $g$ occurs in $h$ or $r$ | int |
| $m_n = \sum_{g} \min(\mathrm{cnt}_h(g), \max_k \mathrm{cnt}_{r_k}(g))$ | clipped matches of order $n$ (sum over the $n$-grams $g$ of $h$) | int |
| $t_n$ | number of $n$-grams in $h$ | int |
| $c$, $r$ | hypothesis length in tokens; length of the reference closest to $c$ (ties to the shorter) | int |
| $p_n = m_n / t_n$ | modified precision of order $n$ | float |
| $\mathrm{BP}$ | brevity penalty, $1$ if $c \ge r$, else $e^{1 - r/c}$ | float |
| $P_k$, $R_k$ | chrF character precision and recall of order $k$ | float |
| $\beta$ | recall weight of chrF (2: recall counts $\beta^2 = 4$ times) | float |
| $N$ | number of sentences in the corpus | int |

### 2.1 Tokens

BLEU counts **words**, so it needs a tokenizer, and the tokenizer is part of the metric. sacreBLEU's default, `13a`, reproduces the WMT `mteval-v13a` script: it removes `<skipped>`, joins a word broken by `-` at a line end, unescapes `&quot; &amp; &lt; &gt;`, puts spaces around punctuation and symbols, splits a period or comma **except between two digits** (so `3.14` and `1,000` stay whole), and splits a dash only **after a digit** (`10-12` becomes `10 - 12`, `e-mail` stays). The line is right-stripped first. `tokenize="none"` splits on white space only.

### 2.2 BLEU

For one hypothesis, keep ten numbers: $[c, r, m_1..m_4, t_1..t_4]$. **Clipping** is the heart of BLEU: without the $\min$, "the the the the" would score a perfect unigram precision against "the cat". With several references, a hypothesis $n$-gram may match up to the largest count of that $n$-gram in any **one** reference. Then

$$\mathrm{BLEU} = 100 \cdot \mathrm{BP} \cdot \exp\Big(\tfrac{1}{4}\sum_{n=1}^{4} \log p_n\Big).$$

The geometric mean means one zero precision zeroes the score. sacreBLEU's default **"exp" smoothing** replaces the $j$-th zero precision by $1 / (2^j t_n)$; but if nothing matches at all the score is $0$, and an order with $t_n = 0$ (every hypothesis shorter than $n$ tokens) ends the loop and counts as $\log 0$, so such a corpus scores $0$. The brevity penalty exists because precision alone rewards short output: a one-word hypothesis that is correct has $p_1 = 1$.

### 2.3 chrF

chrF removes white space and counts **character** $n$-grams of orders $k = 1..6$. Per order it keeps $[$hypothesis count, reference count, matches$]$, where matches are clipped as in BLEU, and the hypothesis count is $0$ for an order the reference has no $n$-gram of (sacreBLEU's rule). Averaging over the orders where both counts are positive gives $\bar P$ and $\bar R$, and

$$\mathrm{chrF} = 100 \cdot \frac{(1 + \beta^2)\, \bar P \bar R}{\beta^2 \bar P + \bar R}.$$

With several references, a sentence keeps the statistics of the reference with the best sentence chrF, the first one on a tie. Characters make chrF kinder to morphology: "cats" shares "c", "a", "t", "ca", "at", "cat" with "cat".

### 2.4 Corpus statistics and intervals

Both metrics are functions of **sums** of per-sentence statistics, so a corpus score is computed once from the column sums. The mean of sentence BLEUs is a different, worse-behaved number (`test_corpus_pools_statistics`: 31.26 pooled against 36.96 averaged). Pooling also makes the bootstrap cheap: `sentence_stats` gives an $N \times S$ matrix, and `metric_ci` calls `M07.4`'s `bootstrap_ci` on the indices $0..N-1$ with the statistic "sum the resampled rows, then score". Each resample keeps a hypothesis with its own references. One surprise the tests pin: smoothing is not scale-invariant, so a sentence counted twice in a resample can score differently from the sentence alone (its zero-order precision becomes $1/(2 \cdot 2t_n)$).

### 2.5 Exact match

Exact match is the fraction of hypotheses equal to their reference after removing white space at **both ends** (a decoder's trailing newline is not an error). Case and inner white space count. Normalized variants (SQuAD lowercases and drops articles) are different metrics with different names.

## 3. Worked example by hand

**BLEU.** $h$ = "the cat sat on the mat", $r$ = "the cat is on the mat"; both have 6 tokens, so $c = r = 6$ and $\mathrm{BP} = 1$.

| $n$ | hypothesis $n$-grams | matched (clipped) | $p_n$ |
|---|---|---|---|
| 1 | the ×2, cat, sat, on, mat | the ×2, cat, on, mat | $5/6$ |
| 2 | the cat, cat sat, sat on, on the, the mat | the cat, on the, the mat | $3/5$ |
| 3 | the cat sat, cat sat on, sat on the, on the mat | on the mat | $1/4$ |
| 4 | the cat sat on, cat sat on the, sat on the mat | none: smoothed to $1/(2 \cdot 3)$ | $1/6$ |

So the statistics are $[6, 6, 5, 3, 1, 0, 6, 5, 4, 3]$ and $\mathrm{BLEU} = 100 \cdot (\tfrac{5}{6} \cdot \tfrac{3}{5} \cdot \tfrac{1}{4} \cdot \tfrac{1}{6})^{1/4} = 100 \cdot (1/48)^{1/4} = 37.991784$.

**chrF.** $h$ = "cat", $r$ = "cats".

| $k$ | hyp, ref, match | $P_k$ | $R_k$ |
|---|---|---|---|
| 1 | 3, 4, 3 | 1 | 3/4 |
| 2 | 2, 3, 2 | 1 | 2/3 |
| 3 | 1, 2, 1 | 1 | 1/2 |
| 4 | 0, 1, 0 (no hypothesis 4-gram) | skipped | skipped |
| 5, 6 | 0, 0, 0 | skipped | skipped |

$\bar P = 1$, $\bar R = (3/4 + 2/3 + 1/2)/3 = 23/36$, and $\mathrm{chrF} = 100 \cdot 5 \bar P \bar R / (4 \bar P + \bar R) = 100 \cdot 115/167 = 68.862275$.

**Exact match.** Hypotheses "2021-05-03", " 2021-05-04\n", "2021-5-3" against "2021-05-03", "2021-05-04", "2021-05-03": the first matches, the second matches after stripping, the third does not (`5` is not `05`). EM $= 2/3$.

These are the first three tests of section 4.

## 4. The interface

```python
TOKENIZERS = ("13a", "none")
def tokenize_13a(line: str) -> str
def exact_match(hyps: Sequence[str], refs: Sequence[str]) -> float              # 0..1
def bleu_stats(hyp: str, refs: Sequence[str], tokenize: str = "13a") -> list[int]  # 10 numbers
def bleu_from_stats(stats: Sequence[float]) -> float                              # 0..100
def corpus_bleu(hyps: Sequence[str], refs: Sequence[Sequence[str]], tokenize: str = "13a") -> float
def chrf_stats(hyp: str, refs: Sequence[str], n: int = 6, beta: float = 2.0) -> list[int]  # 3 n numbers
def chrf_from_stats(stats: Sequence[float], n: int = 6, beta: float = 2.0) -> float        # 0..100
def chrf(hyps: Sequence[str], refs: Sequence[Sequence[str]], n: int = 6, beta: float = 2.0) -> float
def sentence_stats(metric: str, hyps, refs) -> NDArray                            # [N, S] float64
def score_from_stats(metric: str, summed: NDArray, n_sentences: int) -> float
def metric_ci(metric, hyps, refs, n_boot=1000, alpha=0.05, rng=None) -> tuple[float, float, float]
```

`refs[i]` is the **list** of references of `hyps[i]` (sacreBLEU's own API takes reference streams, the transpose). A bare string there is a `TypeError`.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_bleu` | unit | the section 3 BLEU statistics and 37.991784 | you and the tests agree on the definition |
| `test_hand_example_chrf` | unit | the section 3 chrF statistics and $100 \cdot 115/167$ | the same for chrF |
| `test_hand_example_exact_match` | unit | 2/3, with the strip | the `MS-L4` bar |
| `test_clipping` | boundary | "the the the the" matches one unigram | repetition never pays |
| `test_closest_reference_length` | boundary | closest, ties to the shorter, longer when closer | the brevity penalty with several references |
| `test_brevity_penalty` | unit | $e^{1 - r/c}$ when short, nothing when long | dropping hard words is not rewarded |
| `test_smoothing_and_zero_orders` | boundary | "exp" smoothing values; no match and no 4-gram give 0 | short outputs of the dates task |
| `test_corpus_pools_statistics` | property | pooled sums, not the mean of sentence scores | the corpus number `L6.7` reports |
| `test_references_are_per_hypothesis` | boundary | per-hypothesis layout; variable counts; a bare string fails | the zoo passes one reference per row |
| `test_tokenize_13a_matches_sacrebleu` | golden | 15 lines through 13a | comparable token counts |
| `test_bleu_matches_sacrebleu` | golden | statistics and score on 26 corpora, both tokenizers | comparable with published BLEU |
| `test_case_counts` | unit | case-sensitive by default | no silent inflation |
| `test_chrf_matches_sacrebleu` | golden | statistics and score on 16 corpora | comparable with published chrF |
| `test_chrf_ignores_white_space_and_forgives_inflection` | unit | white space removed; "cats" vs "cat" | why chrF sits beside BLEU |
| `test_chrf_beta_weights_recall` | unit | $\beta^2$, and $\beta$ is a parameter | recall-weighted F |
| `test_chrf_uses_the_best_reference` | unit | the best reference, not an average or the last | several references per sentence |
| `test_exact_match_is_strict_inside` | boundary | case and inner spaces count | a date is right or wrong |
| `test_metric_ci_matches_independent_bootstrap` | golden | 6 intervals equal an independent sacreBLEU bootstrap | every `L6.7` number carries this CI |
| `test_metric_ci_brackets_the_score` | property | reproducible, in range, degenerate when every pair matches | resampling keeps pairs together |
| `test_validation` | boundary | mismatched lengths, empty corpus, unknown names, missing rng | a silent 0 hides a broken pipeline |

### Your graded tests (rung R5)

At rung R5 you write oracle tests: numbers you computed by hand from the definitions, never numbers your code printed. Cover at least: the section 3 examples; a clipping case; a closest-reference case where the closest is the **longer** reference; the brevity penalty both ways; a smoothed zero order; pooled statistics against a mean; the per-hypothesis layout; 13a keeping `3.14` whole; case; a chrF order that needs $n = 6$; white space in chrF; the best reference; exact match with a strip and with case; and an interval on a two-sentence corpus where you can list every resample by hand. Your tests run against the course's reference with one planted fault at a time: at least 0.80 of the faults, and every semantic one (one per pitfall below), must fail them.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. Counting n-gram matches without clipping | "the the the the" scores like "the cat" | `test_clipping` (mutant `s01`) |
| 2. Brevity penalty against the shortest reference, or applied to long hypotheses too | scores drop for outputs of the right length | `test_closest_reference_length` (mutant `s02`), `test_brevity_penalty` (mutant `s03`) |
| 3. Corpus BLEU as the mean of sentence BLEUs | 36.96 instead of 31.26 on two sentences | `test_corpus_pools_statistics` (mutant `s04`) |
| 4. The wrong reference layout: a bare string, or sacreBLEU's streams | each character becomes a reference; pairs scored against the wrong sentence | `test_references_are_per_hypothesis` (mutants `s05`, `s06`) |
| 5. Arithmetic mean of precisions, or no smoothing | scores far too high, or 0 for any output missing one 4-gram | `test_hand_example_bleu` (mutant `s07`), `test_smoothing_and_zero_orders` (mutant `s08`) |
| 6. A different tokenizer or case rule | `3.14` split in three; "The" matches "the" | `test_tokenize_13a_matches_sacrebleu` (mutant `s09`), `test_case_counts` (mutant `s10`) |
| 7. chrF with white space kept, $\beta$ not squared, per-order F averaged, or orders 1 to $n-1$ | chrF off by several points | `test_chrf_ignores_white_space_and_forgives_inflection` (mutant `s11`), `test_chrf_beta_weights_recall` (mutant `s12`), `test_hand_example_chrf` (mutant `s13`), `test_chrf_matches_sacrebleu` (mutant `s15`) |
| 8. chrF keeping the last reference instead of the best | a bad extra reference lowers the score | `test_chrf_uses_the_best_reference` (mutant `s14`) |
| 9. Exact match that ignores case, or does not strip | "Paris" equals "paris"; a trailing newline is an error | `test_exact_match_is_strict_inside` (mutants `s16`, `s17`) |
| 10. A bootstrap that averages sentence scores | an interval for a different metric than the one reported | `test_metric_ci_matches_independent_bootstrap` (mutant `s18`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.4` | `bootstrap_ci` and the percentile rule behind `metric_ci` |
| Forward | `L6.7` | the model zoo's EM and chrF columns, each with its interval (`run_task`) |
| Forward | `ag.10` | re-implements exact match in Go for the agent eval scorers |
| Forward | `craft.07` | your L4.5 suite meets its survivors: equivalent mutants and the ones it missed |
| Forward | `L14.5` | caption BLEU-4 and chrF through `corpus_bleu` and `chrf` |

If you skip this module, the `translate` verb of `MS-L4` has nothing to report, and `ol check L6.7` needs `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `corpus_bleu`, `chrf` | [sacreBLEU](https://github.com/mjpost/sacrebleu) | signatures that name every setting, other tokenizers (`intl`, `zh`, `ja-mecab`), chrF++ word n-grams, TER | `sacrebleu/metrics/bleu.py`, `chrf.py` |
| `metric_ci` | sacreBLEU paired bootstrap and approximate randomization | significance of the DIFFERENCE between two systems on the same sentences | `sacrebleu/significance.py` |
| exact match | SQuAD and HF `evaluate` | normalized EM and token F1 for extractive QA | `evaluate/metrics/squad` |
| string metrics | COMET, BLEURT | learned metrics that correlate better with human judgment | Rei et al. (2020), Sellam et al. (2020) |
