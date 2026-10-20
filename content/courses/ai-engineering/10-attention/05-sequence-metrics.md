# Sequence metrics: exact match, BLEU, chrF

The seq2seq models of this chapter now write dates, and soon translations and stories. The training loss says how surprised a model is by the reference, not whether what it **generates** is right: [beam search](04-beam-search.md) can produce a fluent output the loss never saw. So the system needs metrics over generated text. The simplest, exact match, is the bar for the dates task, because a date is right or wrong. For free text, the field reports BLEU and chrF, and reports them in one exact configuration: sacreBLEU's. Two "BLEU" implementations that differ in tokenization or smoothing disagree by several points on the same output, which is larger than most of the improvements papers claim.

This section builds all three metrics from their sufficient statistics, matches sacreBLEU to twelve digits, and attaches a bootstrap interval, so every number that [the LM evaluation harness](../12-objectives-and-adaptation/07-lm-evaluation-harness-and-the-model-zoo.md) prints later is both comparable and honest about its noise. The one prerequisite is `bootstrap_ci` from [LLN, CLT, confidence intervals, bootstrap](../../mathematics/13-probability-statistics/05-lln-clt-confidence-intervals-bootstrap.md) in *Mathematics*, which supplies the interval every score carries; the companion [problem set on LLN, CLT, and confidence intervals](../../mathematics/13-probability-statistics/11-problem-set-c.md) is good practice.

The solution is [`python/tinyllm/eval/seqmetrics.py`](../../../../course/solutions/python/tinyllm/eval/seqmetrics.py), with `exact_match`, `tokenize_13a`, `bleu_stats`, `bleu_from_stats`, `corpus_bleu`, `chrf_stats`, `chrf_from_stats`, `chrf`, `sentence_stats`, `score_from_stats`, and `metric_ci`. It implements the contract [`course/contracts/py/tinyllm/eval/seqmetrics.pyi`](../../../../course/contracts/py/tinyllm/eval/seqmetrics.pyi), and the oracle tests are in [`test_seqmetrics.py`](../../../../course/solutions/learner-tests/L4.5/python/tests/l4-5-seqmetrics/test_seqmetrics.py). The `translate` verb of the `{tinyllm}` CLI reports these metrics.

> **Learning objectives**
> - Compute BLEU by hand: clipped n-gram precisions, exp smoothing, the geometric mean, and the brevity penalty against the closest reference.
> - Compute chrF by hand from character n-gram counts, and explain why it forgives inflection.
> - Explain why a corpus score pools sufficient statistics rather than averaging sentence scores.
> - Attach a bootstrap interval that resamples whole sentences and rescores the pooled statistics.
> - Name the settings (tokenizer, smoothing, case, reference layout) that make a BLEU number comparable.

## Notation

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

## Exact match

Start with the easy one. Exact match is the fraction of hypotheses equal to their reference after removing white space at **both ends**; a decoder's trailing newline is a formatting artifact, not an error. Case and inner white space count, because they change the meaning of a date. Normalized variants (SQuAD lowercases and drops articles) are different metrics with different names.

## Tokens

BLEU counts **words**, so it needs a tokenizer, and the tokenizer is part of the metric. sacreBLEU's default, `13a`, reproduces the WMT `mteval-v13a` script. It removes `<skipped>`, joins a word broken by `-` at a line end, unescapes `&quot; &amp; &lt; &gt;`, puts spaces around punctuation and symbols, splits a period or comma **except between two digits** (so `3.14` and `1,000` stay whole), and splits a dash only **after a digit** (`10-12` becomes `10 - 12`, `e-mail` stays). The line is right-stripped first. `tokenize="none"` splits on white space only. To be comparable with published BLEU, `13a` must be reproduced regex for regex, including its digit exceptions.

## BLEU

BLEU's definition (Papineni et al.) is a corpus-level ratio of counts, not an average of sentence scores, so the unit of computation is a fixed-length vector of counts per sentence. For one hypothesis, keep ten numbers: $[c, r, m_1..m_4, t_1..t_4]$.

**Clipping** is the heart of BLEU. Without the $\min$ in $m_n$, "the the the the" would score a perfect unigram precision against "the cat". With several references, a hypothesis $n$-gram may match up to the largest count of that $n$-gram in any **one** reference: the references are merged by taking the per-gram maximum, not the sum. Then

$$\mathrm{BLEU} = 100 \cdot \mathrm{BP} \cdot \exp\Big(\tfrac{1}{4}\sum_{n=1}^{4} \log p_n\Big).$$

The geometric mean means one zero precision zeroes the score, so zeros must be smoothed, and smoothed in one specific way to be comparable. sacreBLEU's default **"exp" smoothing** replaces the $j$-th zero precision by $1 / (2^j t_n)$, halving the stand-in each time another zero order appears. Two exceptions remain: if nothing matches at all the score is $0$, and an order with $t_n = 0$ (every hypothesis shorter than $n$ tokens) ends the loop and counts as $\log 0$, so such a corpus scores $0$.

The brevity penalty exists because precision alone rewards short output: a one-word hypothesis that is correct has $p_1 = 1$. The penalty needs one reference length, and it compares $c$ with the reference length closest to $c$, ties to the shorter, as sacreBLEU does. It applies only when the hypothesis is shorter.

## chrF

chrF is the same pattern on characters. It removes white space and counts **character** $n$-grams of orders $k = 1..6$. Per order it keeps three numbers, [hypothesis count, reference count, matches], where matches are clipped as in BLEU, and the hypothesis count is $0$ for an order the reference has no $n$-gram of (sacreBLEU's rule). Averaging over the orders where both counts are positive gives $\bar P$ and $\bar R$, and a single $F_\beta$ is taken from the averages, not an average of per-order Fs:

$$\mathrm{chrF} = 100 \cdot \frac{(1 + \beta^2)\, \bar P \bar R}{\beta^2 \bar P + \bar R}.$$

With several references, a sentence keeps the statistics of the reference with the best sentence chrF, the first one on a tie. Characters make chrF kinder to morphology: "cats" shares "c", "a", "t", "ca", "at", "cat" with "cat", where BLEU sees a miss.

## Corpus statistics and intervals

Both metrics are functions of **sums** of per-sentence statistics, so a corpus score is computed once from the column sums. The mean of sentence BLEUs is a different, worse-behaved number: on a two-sentence corpus, 31.26 pooled against 36.96 averaged.

Pooling also makes the bootstrap cheap. `sentence_stats` gives an $N \times S$ matrix, and `metric_ci` calls `bootstrap_ci` on the indices $0..N-1$ with the statistic "sum the resampled rows, then score". Each resample keeps a hypothesis with its own references, so the bootstrap distribution is of the *same* metric that is reported. A score without an interval is a guess.

One surprise the tests pin down: smoothing is not scale-invariant, so a sentence counted twice in a resample can score differently from the sentence alone, because its zero-order precision becomes $1/(2 \cdot 2t_n)$.

## Working it by hand

Each metric gets one hand-worked example, and a correct solution reproduces all three numbers exactly.

### BLEU

Let $h$ = "the cat sat on the mat" and $r$ = "the cat is on the mat". Both have 6 tokens, so $c = r = 6$ and $\mathrm{BP} = 1$. The n-gram counts:

| $n$ | hypothesis $n$-grams | matched (clipped) | $p_n$ |
|---|---|---|---|
| 1 | the ×2, cat, sat, on, mat | the ×2, cat, on, mat | $5/6$ |
| 2 | the cat, cat sat, sat on, on the, the mat | the cat, on the, the mat | $3/5$ |
| 3 | the cat sat, cat sat on, sat on the, on the mat | on the mat | $1/4$ |
| 4 | the cat sat on, cat sat on the, sat on the mat | none: smoothed to $1/(2 \cdot 3)$ | $1/6$ |

So the statistics are $[6, 6, 5, 3, 1, 0, 6, 5, 4, 3]$ and $\mathrm{BLEU} = 100 \cdot (\tfrac{5}{6} \cdot \tfrac{3}{5} \cdot \tfrac{1}{4} \cdot \tfrac{1}{6})^{1/4} = 100 \cdot (1/48)^{1/4} = 37.991784$.

### chrF

Let $h$ = "cat" and $r$ = "cats". Per order:

| $k$ | hyp, ref, match | $P_k$ | $R_k$ |
|---|---|---|---|
| 1 | 3, 4, 3 | 1 | 3/4 |
| 2 | 2, 3, 2 | 1 | 2/3 |
| 3 | 1, 2, 1 | 1 | 1/2 |
| 4 | 0, 1, 0 (no hypothesis 4-gram) | skipped | skipped |
| 5, 6 | 0, 0, 0 | skipped | skipped |

$\bar P = 1$, $\bar R = (3/4 + 2/3 + 1/2)/3 = 23/36$, and $\mathrm{chrF} = 100 \cdot 5 \bar P \bar R / (4 \bar P + \bar R) = 100 \cdot 115/167 = 68.862275$.

### Exact match

Hypotheses "2021-05-03", " 2021-05-04\n", "2021-5-3" against "2021-05-03", "2021-05-04", "2021-05-03": the first matches, the second matches after stripping, the third does not (`5` is not `05`). EM $= 2/3$.

## Designing the API

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

Every metric is split into a statistics function and a scoring function, so the same rows serve the corpus score, any sub-corpus, and every bootstrap resample. `refs[i]` is the **list** of references of `hyps[i]`; sacreBLEU's own API takes reference streams, the transpose. A bare string there is a `TypeError`, because otherwise each character would silently become a reference.

## Building it step by step

### The 13a tokenizer

The tokenizer is four regexes over a padded line, then a white-space split and rejoin:

```python
_13A = [
    # symbols and punctuation: { | } ~ [ \ ] ^ _ ` space ! " # $ % & ( ) * + : ; < = > ? @ /
    (re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),
    # period and comma unless preceded by a digit
    (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),
    # period and comma unless followed by a digit
    (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),
    # dash when preceded by a digit
    (re.compile(r"([0-9])(-)"), r"\1 \2 "),
]

def tokenize_13a(line):
    line = line.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    if "&" in line:
        line = line.replace("&quot;", '"').replace("&amp;", "&")
        line = line.replace("&lt;", "<").replace("&gt;", ">")
    line = f" {line} "                     # so the "not a digit" rules see a neighbour at the ends
    for pattern, repl in _13A:
        line = pattern.sub(repl, line)
    return " ".join(line.split())
```

`3.14` survives because both period rules need a non-digit on one side; `10-12` splits because the dash follows a digit.

### BLEU statistics

```python
def bleu_stats(hyp, refs, tokenize="13a"):
    refs = _ref_list(refs)                 # a bare str raises TypeError
    max_ref = Counter()
    ref_lens = []
    for r in refs:
        toks = _tokens(r, tokenize)        # rstrip, then 13a or plain split
        ref_lens.append(len(toks))
        for g, c in _word_ngrams(toks).items():
            if c > max_ref[g]:
                max_ref[g] = c             # per-gram MAX over references
    htoks = _tokens(hyp, tokenize)
    correct = [0] * MAX_ORDER
    total = [0] * MAX_ORDER
    for g, c in _word_ngrams(htoks).items():
        total[len(g) - 1] += c
        correct[len(g) - 1] += min(c, max_ref[g])   # clipping
    return [len(htoks), _closest_ref_len(len(htoks), ref_lens)] + correct + total
```

For "the cat sat on the mat" against "the cat is on the mat" this returns $[6, 6, 5, 3, 1, 0, 6, 5, 4, 3]$, the row worked by hand. `_closest_ref_len` scans references keeping the smallest $|c - r|$ and, on equal distance, the shorter $r$.

### BLEU score

Scoring the statistics uses sacreBLEU's arithmetic, logs of percentages, so the result is already in 0..100:

```python
def bleu_from_stats(stats):
    # ... (exactly 10 numbers)
    sys_len, ref_len = stats[0], stats[1]
    correct, total = stats[2 : 2 + MAX_ORDER], stats[2 + MAX_ORDER :]
    if not any(correct):
        return 0.0                                   # nothing matched at all
    bp = 1.0
    if sys_len < ref_len:                            # only short output is penalized
        bp = math.exp(1.0 - ref_len / sys_len) if sys_len > 0 else 0.0
    logs = [LOG_ZERO] * MAX_ORDER                    # sacreBLEU's log(0)
    smooth = 1.0
    for n in range(MAX_ORDER):
        if total[n] == 0:
            break                                    # no n-grams of this order: stays log 0
        if correct[n] == 0:
            smooth *= 2.0                            # j-th zero order -> 1 / (2^j t_n)
            logs[n] = math.log(100.0 / (smooth * total[n]))
        else:
            logs[n] = math.log(100.0 * correct[n] / total[n])
    return bp * math.exp(sum(logs) / MAX_ORDER)      # geometric mean, times BP
```

In the hand example the single zero 4-gram order makes `smooth` 2, so the stand-in is $1/(2 \cdot 3)$ and the score is $100 \cdot (1/48)^{1/4} = 37.991784$. `corpus_bleu` sums `bleu_stats` over sentences element by element and calls `bleu_from_stats` once.

### chrF statistics and score

```python
def _char_ngrams(line, n):
    """Character n-gram counts of orders 1..n, white space removed."""
    s = "".join(line.split())
    return [Counter(s[i : i + k] for i in range(len(s) - k + 1)) for k in range(1, n + 1)]

def chrf_stats(hyp, refs, n=CHRF_N, beta=CHRF_BETA):
    # ... (n, beta checks; refs as a list)
    hyp_grams = _char_ngrams(hyp, n)
    best, best_f = None, -1.0
    for r in refs:
        stats = []
        for h, g in zip(hyp_grams, _char_ngrams(r, n)):
            match = sum(min(c, g[x]) for x, c in h.items() if x in g)   # clipped matches
            # sacreBLEU's rule: the hypothesis count is 0 when the reference has no k-gram.
            stats += [sum(h.values()) if g else 0, sum(g.values()), match]
        f = chrf_from_stats(stats, n, beta)
        if f > best_f:                     # strict: the first reference wins a tie
            best, best_f = stats, f
    return best

def chrf_from_stats(stats, n=CHRF_N, beta=CHRF_BETA):
    # ... (checks)
    factor = beta**2
    prec = rec = 0.0
    orders = 0
    for k in range(n):
        n_hyp, n_ref, n_match = stats[3 * k : 3 * k + 3]
        if n_hyp > 0 and n_ref > 0:        # only orders both sides have
            prec += n_match / n_hyp
            rec += n_match / n_ref
            orders += 1
    if orders == 0:
        return 0.0
    prec, rec = prec / orders, rec / orders  # average P and R first ...
    if prec + rec == 0:
        return 0.0
    return 100.0 * (1 + factor) * prec * rec / (factor * prec + rec)   # ... then one F_beta
```

For "cat" against "cats", orders 1 to 3 contribute and $\bar R = 23/36$, so the score is $100 \cdot 5 \cdot 23/36 / (4 + 23/36) = 100 \cdot 115/167$.

### Exact match and the interval

Exact match is `h.strip() == r.strip()` averaged. The interval wraps all three metrics in one shape:

```python
def metric_ci(metric, hyps, refs, n_boot=1000, alpha=0.05, rng=None):
    if rng is None:
        raise ValueError("metric_ci needs an rng (a PCG32): the interval must be reproducible")
    rows = sentence_stats(metric, hyps, refs)      # [N, S]: 1, 10 or 18 columns

    def stat(idx):
        picked = rows[np.asarray(idx, dtype=np.int64)]   # a resample of whole sentences
        return score_from_stats(metric, picked.sum(axis=0), len(picked))

    # bootstrap_ci resamples the indices 0..N-1; the statistic re-pools and rescores.
    point, lo, hi = bootstrap_ci(np.arange(len(rows), dtype=np.float64), stat, n_boot, alpha, rng)
    return float(point), float(lo), float(hi)
```

`metric_ci` reproduces an independent sacreBLEU bootstrap run seed for seed. The full solution is [`seqmetrics.py`](../../../../course/solutions/python/tinyllm/eval/seqmetrics.py).

### Why this design

BLEU statistics cost $O(L)$ per sentence for $L$ tokens (four orders), and chrF $O(n C)$ for $C$ characters. The bootstrap costs $O(B \cdot N S)$ for $B$ resamples, because each resample only sums precomputed rows; recomputing n-grams per resample would cost $O(B \cdot \text{corpus size})$.

Why match sacreBLEU instead of writing a "cleaner" BLEU? Any change in tokenization, smoothing, or case moves scores by more than typical improvements (Post 2018). Matching the de facto standard makes every number comparable; the cost is reproducing quirks such as `LOG_ZERO` and the `break` on empty orders.

Why sufficient statistics rather than a direct corpus function? The same rows give the corpus score, any sub-corpus score, and every bootstrap resample, where a direct function would have to be rerun on raw text each time.

Why does exact match strip only the ends? It is a strict metric by design. A trailing newline from a decoder is a formatting artifact, while case and inner spacing change the meaning of a date.

The tradeoff accepted is that smoothing is not scale-invariant, so a resample that duplicates a sentence can score differently from the sentence alone. The solution accepts this to stay identical to sacreBLEU's bootstrap.

## Testing it

The suite [`test_seqmetrics.py`](../../../../course/solutions/learner-tests/L4.5/python/tests/l4-5-seqmetrics/test_seqmetrics.py) uses oracle values: numbers computed by hand from the definitions, never numbers the code printed. It covers the three hand examples; a clipping case; a closest-reference case where the closest is the **longer** reference; the brevity penalty both ways; a smoothed zero order; pooled statistics against a mean; the per-hypothesis layout; 13a keeping `3.14` whole; case; a chrF order that needs $n = 6$; white space in chrF; the best reference; exact match with a strip and with case; and an interval on a two-sentence corpus where every resample can be listed by hand. Each pitfall in the next section makes at least one of these tests fail.

The full set of behaviors a correct solution must have:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: BLEU | unit | the hand-worked BLEU statistics and 37.991784 | pins down the definition |
| hand example: chrF | unit | the hand-worked chrF statistics and $100 \cdot 115/167$ | the same for chrF |
| hand example exact match | unit | 2/3, with the strip | the dates-task bar |
| clipping | boundary | "the the the the" matches one unigram | repetition never pays |
| closest reference length | boundary | closest, ties to the shorter, longer when closer | the brevity penalty with several references |
| brevity penalty | unit | $e^{1 - r/c}$ when short, nothing when long | dropping hard words is not rewarded |
| smoothing and zero orders | boundary | "exp" smoothing values; no match and no 4-gram give 0 | short outputs of the dates task |
| corpus pools statistics | property | pooled sums, not the mean of sentence scores | the corpus number the harness reports |
| references are per hypothesis | boundary | per-hypothesis layout; variable counts; a bare string fails | the zoo passes one reference per row |
| 13a tokenizer matches sacreBLEU | golden | 15 lines through 13a | comparable token counts |
| BLEU matches sacreBLEU | golden | statistics and score on 26 corpora, both tokenizers | comparable with published BLEU |
| case counts | unit | case-sensitive by default | no silent inflation |
| chrF matches sacreBLEU | golden | statistics and score on 16 corpora | comparable with published chrF |
| chrF ignores white space and forgives inflection | unit | white space removed; "cats" vs "cat" | why chrF sits beside BLEU |
| chrF $\beta$ weights recall | unit | $\beta^2$, and $\beta$ is a parameter | recall-weighted F |
| chrF uses the best reference | unit | the best reference, not an average or the last | several references per sentence |
| exact match is strict inside | boundary | case and inner spaces count | a date is right or wrong |
| `metric_ci` matches an independent bootstrap | golden | 6 intervals equal an independent sacreBLEU bootstrap | every harness number carries this CI |
| `metric_ci` brackets the score | property | reproducible, in range, degenerate when every pair matches | resampling keeps pairs together |
| validation | boundary | mismatched lengths, empty corpus, unknown names, missing rng | a silent 0 hides a broken pipeline |

## What can go wrong

The common mistakes all produce a plausible number that is simply not BLEU or chrF as anyone else computes it:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| counting n-gram matches without clipping | "the the the the" scores like "the cat" | `min(c, max_ref[g])` in `bleu_stats` |
| brevity penalty against the shortest reference, or applied to long hypotheses too | scores drop for outputs of the right length | `_closest_ref_len` (ties to the shorter); BP only `if sys_len < ref_len` |
| corpus BLEU as the mean of sentence BLEUs | 36.96 instead of 31.26 on two sentences | `corpus_bleu` sums the ten statistics, then scores once |
| the wrong reference layout: a bare string, or sacreBLEU's streams | each character becomes a reference; pairs scored against the wrong sentence | `_ref_list` rejects a bare `str`; `refs[i]` is the list for `hyps[i]` |
| arithmetic mean of precisions, or no smoothing | scores far too high, or 0 for any output missing one 4-gram | mean of `logs` (geometric mean); `smooth *= 2.0` for each zero order |
| a different tokenizer or case rule | `3.14` split in three; "The" matches "the" | the four `_13A` regexes exactly as sacreBLEU; no lowercasing |
| chrF with white space kept, $\beta$ not squared, per-order F averaged, or orders 1 to $n-1$ | chrF off by several points | `"".join(line.split())`; `factor = beta**2`; average $P$ and $R$ first, then one F; `range(1, n + 1)` |
| chrF keeping the last reference instead of the best | a bad extra reference lowers the score | `chrf_stats` keeps the stats with the highest sentence chrF, `>` so the first wins ties |
| exact match that ignores case, or does not strip | "Paris" equals "paris"; a trailing newline is an error | `h.strip() == r.strip()`, nothing more |
| a bootstrap that averages sentence scores | an interval for a different metric than the one reported | `stat` sums resampled rows and calls `score_from_stats` |

> **Warning:** The tokenizer, the smoothing, the case rule, and the reference layout are part of the metric. Change one and the number is no longer comparable with anything published, even though it is still called BLEU.

## Where this leads

[The LM evaluation harness and the model zoo](../12-objectives-and-adaptation/07-lm-evaluation-harness-and-the-model-zoo.md) scores the zoo's `seq2seq` and `transformer` rows with these functions, reporting EM and chrF columns each with its interval. [Multimodal eval metrics](../15-speech/05-multimodal-eval-metrics.md) computes caption BLEU-4 and chrF through `corpus_bleu` and `chrf`. Outside this book, [Agent evaluation scorers](../29-evaluation/02-scorers.md) in *AI Engineering* re-implements exact match in Go, and [Mutation testing in depth](../../software-engineering/03-testing/05-mutation-testing-in-depth.md) in *Software Engineering* uses this section's test suite as its example of surviving faults, both equivalent ones and missed ones.

Production tooling goes further:

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `corpus_bleu`, `chrf` | [sacreBLEU](https://github.com/mjpost/sacrebleu) | signatures that name every setting, other tokenizers (`intl`, `zh`, `ja-mecab`), chrF++ word n-grams, TER | `sacrebleu/metrics/bleu.py`, `chrf.py` |
| `metric_ci` | sacreBLEU paired bootstrap and approximate randomization | significance of the DIFFERENCE between two systems on the same sentences | `sacrebleu/significance.py` |
| exact match | SQuAD and HF `evaluate` | normalized EM and token F1 for extractive QA | `evaluate/metrics/squad` |
| string metrics | COMET, BLEURT | learned metrics that correlate better with human judgment | Rei et al. (2020), Sellam et al. (2020) |

### Further reading

- Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine Translation" (ACL 2002).
- Popovic, "chrF: character n-gram F-score for automatic MT evaluation" (WMT 2015).
- Post, "A Call for Clarity in Reporting BLEU Scores" (WMT 2018).
- Koehn, "Statistical Significance Tests for Machine Translation Evaluation" (EMNLP 2004).

## Key terms

- **exact match**: the fraction of hypotheses equal to their reference after stripping white space at both ends.
- **clipping**: crediting a hypothesis n-gram at most as many times as it occurs in any one reference.
- **modified precision**: $p_n = m_n / t_n$, clipped matches over hypothesis n-grams of order $n$.
- **brevity penalty**: $e^{1 - r/c}$ when the hypothesis is shorter than the closest reference, else 1.
- **exp smoothing**: sacreBLEU's replacement of the $j$-th zero precision by $1/(2^j t_n)$.
- **chrF**: a recall-weighted F score over character n-grams of orders 1 to 6, white space removed.
- **sufficient statistics**: the per-sentence count vectors whose column sums determine the corpus score.

## Summary

- BLEU counts word n-grams of orders 1 to 4 that the hypothesis shares with a reference, **clipped** by the reference counts, takes the geometric mean of the four precisions, and multiplies by a brevity penalty measured against the reference **closest** in length.
- A corpus score pools sufficient statistics over sentences; it is not the mean of sentence scores. Pooling is also what makes a bootstrap over sentences cheap.
- chrF counts character n-grams with white space removed, averages precision and recall over orders 1 to 6, and weights recall by $\beta^2 = 4$; it forgives "cats" for "cat" where BLEU sees a miss.
- The tokenizer, the smoothing, the case rule, and the reference layout are part of the metric: matching sacreBLEU's defaults exactly makes the numbers comparable with published scores.
- A score without an interval is a guess: `metric_ci` resamples sentences with a bootstrap and reproduces an independent sacreBLEU run seed for seed.

## Review questions

1. Why is the corpus BLEU of two sentences not the mean of their sentence BLEUs?
   <details><summary>Answer</summary>BLEU is a geometric mean of pooled precisions times a pooled brevity penalty, a nonlinear function of summed counts. Averaging sentence scores applies the nonlinearity per sentence, which weights short sentences and smoothed zeros very differently (31.26 pooled against 36.96 averaged on the example corpus).</details>
2. With two references, why is the clip the maximum count in one reference and not the sum over references?
   <details><summary>Answer</summary>Each reference is a complete valid output. A word that appears once in each of two references is still correct only once in a single sentence; summing would let a hypothesis repeat it and be credited twice.</details>
3. In the worked example, where does the $1/6$ for the 4-gram precision come from?
   <details><summary>Answer</summary>No 4-gram matched, so exp smoothing applies: it is the first zero order, so `smooth` becomes 2 and the stand-in precision is $1/(2 \cdot t_4) = 1/(2 \cdot 3)$.</details>
4. Why does chrF average $P$ and $R$ over orders before forming $F_\beta$?
   <details><summary>Answer</summary>That is chrF's definition in sacreBLEU. Averaging per-order F scores is a different metric and gives different numbers, so it would not be comparable with published chrF.</details>
5. Why does `metric_ci` resample row indices rather than resampling hypotheses and references separately?
   <details><summary>Answer</summary>A hypothesis is only meaningful against its own references. Resampling rows keeps each pair together, and summing the picked rows before scoring makes the interval describe the same pooled metric that is reported.</details>
