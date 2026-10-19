# Model evals as tests

The [first capstone](../../build-an-llm/capstones/01-tinystories.md) of *Build Your Own LLM System* trains for hours, and the release workflow in [ModelRelease: gates, canary, PromQL burn check, rollback](../../ai-platform-engineering/05-durable-orchestration-and-workers/12-model-release.md) gates on eval numbers. Until now, a model was right when its logits matched a fixture to $10^{-5}$, as in the transformer, modern-block, and inference chapters. That oracle is gone the moment you train a new model: there is no reference to match, and two correct runs differ. Yet the bugs are real and quiet. A RoPE base read from the wrong config key, a layer skipped by an off-by-one, a tokenizer that lost one merge: every one of them still trains, still produces plausible text, and costs a few tenths of a bit per byte. Without a regression test you find out from users.

This section builds that test: measured numbers, recorded from a good model, compared with statistics that know the difference between noise and a defect. A model eval **is a test**: a held-out metric and a regression rule, run on every change, that fails the build, and it is judged like any test, by the defects it catches. The section builds the statistics from first principles, works four candidates by hand, then walks through the solution function by function and the three-test suite. Try to compute each p-value before reading it.

It builds on judging tests by planted faults ([TDD, unit tests, and mutation testing](01-tdd-unit-tests-and-mutation-grading.md)) and reading survivors ([Mutation testing in depth](05-mutation-testing-in-depth.md)). From *Mathematics for Computing* it uses permutation tests ([Hypothesis tests](../../mathematics-for-computing/07-probability-statistics/05-hypothesis-tests.md)) and bits per byte ([Perplexity, bits per byte, NLL accumulator](../../mathematics-for-computing/11-information-theory/02-perplexity-bits-per-byte-nll-accumulator.md)); from *Build Your Own LLM System*, the [LM evaluation harness and the model zoo](../../build-an-llm/p06-objectives/07-lm-evaluation-harness-and-the-model-zoo.md) and [RoPE](../../build-an-llm/p07-modern-block/03-rope.md).

The solution is [`evals.py`](../../../../course/solutions/primers/craft.22/evals.py), a kata with held-out bits per byte `heldout_bpb`, a seeded sample-quality score `quality_score`, the paired sign-flip test `sign_flip_p`, and the regression rule `regressed`; [`test_model_evals.py`](../../../../course/solutions/primers/craft.22/test_model_evals.py), the eval suite; and [`baseline.json`](../../../../course/solutions/primers/craft.22/baseline.json), the good model, measured. The model under test is [`tinymodel.py`](../../../../course/solutions/primers/craft.22/tinymodel.py). The contract is the rules in the kata's docstring. The solution is used like this (the fixtures, `tinylm.npz`, `val.txt`, and `words.txt`, live under `$TINYLLM_FIXTURES/craft.22/`):

```bash
uv run --no-project --with numpy python course/solutions/primers/craft.22/evals.py --record   # writes baseline.json
(cd course/solutions/primers/craft.22 && uv run --no-project --with pytest --with numpy python -m pytest -q test_model_evals.py)
```

> **Learning objectives**
> - Measure held-out bits per byte and a seeded generation score as paired, per-seed metrics.
> - Compute an exact one-sided sign-flip p-value by hand and explain why five seeds is the minimum.
> - Decide a regression with a margin and a significance test together, and explain why neither alone works.
> - Explain why a suite needs both a loss metric and a generation metric to catch model mutants.

## Paired measurements

The notation for this section:

| Symbol | Meaning | Type |
|---|---|---|
| $s = 1, \dots, n$ | a seed: it picks the held-out windows and drives the sampler | `int` |
| $b_s$, $c_s$ | the baseline's and the candidate's metric at seed $s$ | `float` |
| $d_s$ | how much worse the candidate is at seed $s$: $c_s - b_s$ for bpb (lower is better), $b_s - c_s$ for quality | `float` |
| $T = \sum_s d_s$ | the observed statistic | `float` |
| $\epsilon \in \{+1, -1\}^n$ | a sign vector | |
| $p = \frac{1}{2^n} \#\{\epsilon : \sum_s \epsilon_s d_s \ge T\}$ | the exact one-sided sign-flip p-value | `float` |
| $\delta$ | the margin: the smallest regression worth failing a build for | `float` |
| $\alpha$ | the significance level, 0.05 | `float` |

Measure the baseline and the candidate on **the same** windows and with **the same** seeds, so that each seed gives one difference $d_s$. The windows' own difficulty (one held-out story is harder than another) then cancels in $d_s$ instead of swamping it. Held-out bits per byte pools the bits of every window over their bytes, which makes it independent of the tokenizer: a tokenizer change that splits text into more tokens cannot hide behind a per-token loss.

## The sign-flip test

If the candidate is as good as the baseline, each $d_s$ is as likely to be positive as negative: flipping its sign gives an equally likely world. Enumerate all $2^n$ flips and ask how often the flipped sum is at least the one you saw. That fraction is an exact p-value with no distribution assumed. When all $n$ differences are positive, only the unflipped vector reaches $T$, so $p = 1/2^n$: $1/32 = 0.031$ for five seeds, $1/16 = 0.0625$ for four. **A four-seed suite can never reject at 0.05**, which is why five seeds is the minimum.

## Margin and significance together

A model is never bit-identical to its baseline: float64 arithmetic, another BLAS, or a new sampling stream all move the numbers. So flag a regression only when **both** of two conditions hold: the mean difference exceeds $\delta$ (it matters) and $p < \alpha$ (it is not noise). Significance alone fails on tiny, real, harmless shifts: float64 arithmetic moves every seed by $10^{-7}$, and all in one direction gives $p = 1/32$. The margin alone fails on noise: another sampling stream can make quality 0.03 worse on average with signs going both ways.

## Two kinds of metric

Held-out bpb tests what the model computes under teacher forcing. A seeded generation score tests what it does when it runs: sample from fixed prompts with a seeded generator and score the text, here as the fraction of words that exist in the training vocabulary, times one minus the rate of repeated 4-grams. Decoding bugs live only in the second metric, and some forward bugs show up only there too. One metric is not enough: held-out loss is blind to decoding bugs, and a broken final norm can make bpb *better* while samples get worse.

## Working it by hand

Take five seeds, a baseline bpb of 1.00 at each, $\delta = 0.02$, and $\alpha = 0.05$, and judge four candidates.

**Candidate A** scores 1.03 at every seed. Then $d = (0.03, 0.03, 0.03, 0.03, 0.03)$, with mean $0.03 > 0.02$ and $T = 0.15$. Every flip of a positive term lowers the sum, so only $\epsilon = (+,+,+,+,+)$ reaches $0.15$: $p = 1/32 = 0.03125 < 0.05$. **Regressed.**

**Candidate B** scores 1.03, 1.03, 1.03, 1.03, 0.99. Then $d = (0.03, 0.03, 0.03, 0.03, -0.01)$, with mean $0.022 > 0.02$ and $T = 0.11$. The identity gives 0.11; flipping the last sign gives $0.13 \ge 0.11$; any flip of a $0.03$ lowers the sum below 0.11. So $p = 2/32 = 0.0625$. **Not a regression**: one seed disagrees.

**Candidate C** scores 1.01 at every seed. Then $p = 1/32$, but the mean $0.01 < 0.02$. **Not a regression**: real, but too small to fail a build for.

**Four seeds**, all worse, give $p = 1/16 = 0.0625$. Nothing can ever be flagged.

A correct `sign_flip_p` and `regressed` must reproduce these four verdicts exactly.

## The artifacts

The kata, `primers/craft.22/evals.py`, has its spec in its docstring:

```python
SEEDS = (0, 1, 2, 3, 4); PROMPTS = ("Once upon a time", "One day", "Mia and the", "At night")
def heldout_bpb(model, text, seed, windows=16, chars=80) -> float   # default_rng(seed) picks the windows
def words(text) -> list[str]
def repeat_ngram_rate(ws, n=4) -> float
def valid_word_rate(text, vocabulary) -> float
def quality_score(model, vocabulary, seed, tokens=48) -> float     # one rng for the four prompts, in order
def measure(model, text, vocabulary, seeds=SEEDS) -> dict          # {"seeds", "bpb", "quality"}
def sign_flip_p(diffs) -> float
def regressed(base, cand, margin, alpha=0.05, higher_is_better=False) -> bool
# python evals.py --record   writes baseline.json
```

The model, `primers/craft.22/tinymodel.py`, is given and is not part of the solution's design: `load()`, `encode`, `decode`, `log_probs(ids)`, `sample(prompt, tokens, rng, temperature=0.8)`. It is a two-layer Llama-style decoder (RMSNorm, RoPE, SwiGLU, tied embeddings) over a character BPE of 96 merges, trained on synthetic stories; the held-out stories and the training vocabulary are fixtures.

The suite, `primers/craft.22/test_model_evals.py`, has at least two tests, imports only `evals`, `tinymodel`, numpy, pytest, and the standard library, reads the fixtures through `TINYLLM_FIXTURES`, and fails when held-out bpb or sample quality regressed against `baseline.json` (margins 0.02 each).

A correct solution handles six cases:

| Case | Kind of check | Why it matters downstream |
|---|---|---|
| the four hand-worked candidates on `sign_flip_p` and `regressed` | unit | the decision rule every release gate uses |
| bpb, word, repeat, and quality functions follow the docstring exactly (window choice, pooling, the one rng) | golden | a baseline recorded with one definition and compared with another is meaningless |
| `baseline.json` has 5 or more seeds and equals the good model measured now | golden | a stale or typed-in baseline gates against a model that does not exist |
| the suite passes on the good model | conformance | a gate that fails the model it was recorded from is switched off |
| the suite passes on two harmless variants: float64 arithmetic, another sampling stream | conformance | no false alarms on changes that do not matter |
| the suite fails on at least 80% of seven model mutants, always on the wrong RoPE base, a dropped layer, and a tokenizer one merge short | fault | the defects that train and generate but are wrong |

## Building it step by step

Seven observations shape the kata.

1. **There is no exact oracle for a trained model,** so a test must compare a candidate with a recorded good model and decide whether the difference is a defect. That needs a measurement, a baseline, and a rule.
2. **Differences between windows are larger than differences between models.** So measure both models on the same windows with the same seeds, and test the per-seed differences: pairing cancels window difficulty. This forces the seed to choose the windows (`default_rng(seed)`), so the baseline and the candidate see the same text.
3. **The metric must not depend on the tokenizer,** or a tokenizer bug hides in it. Bits per byte, pooled as total bits over total bytes, has that property; a mean of per-window ratios would overweight short windows, and a per-token loss rewards a tokenizer that makes more, easier tokens.
4. **The first token of a window is never predicted,** so its bytes must not be in the denominator; otherwise bpb is biased low by a tokenizer-dependent amount.
5. **Generation bugs are invisible to teacher-forced loss.** So a second metric must run the sampler with a seeded generator and score the text. The score must be cheap and catch the two ways a broken model fails: non-words and loops.
6. **The rule needs a margin and a p-value,** and an exact test needs no distributional assumption. With $n = 5$ the sign-flip test enumerates 32 vectors, which is trivially cheap and exact.
7. **Two metrics, two directions.** bpb is lower-better, quality higher-better; one `regressed` with a flag keeps the "worse is positive" convention in one place.

### The measurements

```python
def heldout_bpb(model, text: str, seed: int, windows: int = 16, chars: int = 80) -> float:
    rng = np.random.default_rng(seed)                     # the seed picks the windows: paired across models
    starts = rng.integers(0, len(text) - chars, size=windows)
    bits = 0.0
    nbytes = 0
    for s in starts:
        w = text[int(s) : int(s) + chars]
        ids = model.encode(w)
        lp = model.log_probs(ids)                         # lp[t] predicts ids[t + 1]
        bits += -sum(float(lp[t, ids[t + 1]]) for t in range(len(ids) - 1)) / math.log(2)  # nats to bits
        nbytes += len(w.encode()) - len(model.decode(ids[:1]).encode())                     # the first token is never predicted
    return bits / nbytes                                  # pooled: total bits over total bytes
```

Windows are 80 characters at random offsets, so they cut words; that is fine because both models see the same cuts. Dividing once at the end pools the windows (step 3).

The generation score combines two failure detectors:

```python
def repeat_ngram_rate(ws: Sequence[str], n: int = 4) -> float:
    grams = [tuple(ws[i : i + n]) for i in range(len(ws) - n + 1)]
    if not grams:
        return 0.0
    seen: set[tuple[str, ...]] = set()
    repeats = 0
    for g in grams:
        repeats += g in seen           # a repeat of an n-gram that started earlier
        seen.add(g)
    return repeats / len(grams)


def quality_score(model, vocabulary: set[str], seed: int, tokens: int = 48) -> float:
    rng = np.random.default_rng(seed)                                     # one stream for all four prompts
    text = " ".join(model.sample(p, tokens, rng) for p in PROMPTS)        # in order: the stream is reproducible
    return valid_word_rate(text, vocabulary) * (1.0 - repeat_ngram_rate(words(text), 4))
```

The product is 1 only when every word is real and no 4-gram repeats; a model that loops scores low on the second factor, one that emits fragments on the first. Using one generator for the prompts in a fixed order means the score at seed $s$ is a deterministic function of the model, so it can be paired.

### The decision rule

```python
def sign_flip_p(diffs: Sequence[float]) -> float:
    d = np.asarray(diffs, dtype=np.float64)
    observed = d.sum()
    hits = sum(
        # every sign vector; ties count, with a 1e-12 tolerance so the identity always counts
        float(np.dot(s, d)) >= observed - 1e-12 for s in itertools.product((1.0, -1.0), repeat=len(d))
    )
    return hits / 2 ** len(d)                     # >= 1/2^n: the identity is always a hit


def regressed(base, cand, margin, alpha=0.05, higher_is_better=False) -> bool:
    b, c = np.asarray(base, dtype=np.float64), np.asarray(cand, dtype=np.float64)
    worse = b - c if higher_is_better else c - b  # positive means worse, for either metric
    return bool(worse.mean() > margin and sign_flip_p(worse) < alpha)   # it matters AND it is not noise
```

Check it against the worked example. Candidate A gives `worse = [0.03]*5`, mean 0.03 > 0.02, and only the identity reaches the sum, so $p = 1/32$: regressed. Candidate B's mean is 0.022, but two vectors reach 0.11, so $p = 2/32$: not regressed. The `1e-12` tolerance exists because a sum of floats in a different order can differ in the last bit. Without it, a sign vector whose sum equals the observed one mathematically (a tie, which the definition counts) could be dropped by rounding, making $p$ too small. The full kata is [`evals.py`](../../../../course/solutions/primers/craft.22/evals.py).

## Building the suite

```python
FIX = Path(os.environ["TINYLLM_FIXTURES"]) / "craft.22"   # the model and texts are fixtures, not code


@pytest.fixture(scope="module")
def current():                                            # measured once per run: the expensive part
    model = tinymodel.load(FIX / "tinylm.npz")
    text = (FIX / "val.txt").read_text()
    vocab = set((FIX / "words.txt").read_text().split())
    return evals.measure(model, text, vocab)


@pytest.fixture(scope="module")
def baseline():
    return json.loads((HERE / "baseline.json").read_text())


def test_seeds_match_the_baseline(current, baseline):
    # Pairing is only valid on the same seeds, and five is the minimum for p < 0.05.
    assert current["seeds"] == baseline["seeds"] and len(baseline["seeds"]) >= 5


def test_heldout_bpb_did_not_regress(current, baseline):
    assert not evals.regressed(baseline["bpb"], current["bpb"], evals.BPB_MARGIN), (
        f"bpb {current['bpb']} vs baseline {baseline['bpb']}"
    )


def test_sample_quality_did_not_regress(current, baseline):
    assert not evals.regressed(
        baseline["quality"], current["quality"], evals.QUALITY_MARGIN, higher_is_better=True
    ), f"quality {current['quality']} vs baseline {baseline['quality']}"
```

Each metric is its own test, so a failure names which property regressed. The baseline is a file written by `evals.py --record` from the good model, never typed in. The recorded bpb ranges from 0.97 to 1.20 across seeds, which is exactly the window-difficulty spread that pairing removes.

The suite is judged against seven model mutants, and each must be caught by at least one metric:

| Mutant | Which metric must catch it |
|---|---|
| RoPE base 500 instead of 10000 | either: the forward pass is wrong, so teacher-forced bpb moves |
| the last layer never runs | either, as above |
| the tokenizer is one merge short | bpb, measured per byte so the extra tokens cannot hide it |
| decoding samples from the previous position's logits | quality only: teacher forcing never calls `sample`, so bpb is unchanged |
| attention scores not scaled by $1/\sqrt{d}$ | either: the forward pass is wrong |
| the causal mask shows the next token | either: the forward pass sees what it must predict |
| the final norm loses its gain | quality only in the direction that matters: bpb *improves* by 0.16 while samples get worse |

Two of the seven are invisible to a bpb-only suite, which is why a single-metric suite cannot reach 80%. The full suite is [`test_model_evals.py`](../../../../course/solutions/primers/craft.22/test_model_evals.py).

## Why this design

The suite is cheap: $5 \text{ seeds} \times (16 \text{ forward passes} + 4 \text{ samples of } 48 \text{ tokens})$ on a two-layer model takes seconds. The sign-flip test is $O(2^n \cdot n)$, 160 multiply-adds at $n = 5$; beyond about 20 seeds one would sample sign vectors instead of enumerating them.

The sign-flip test beats a paired t-test here. The t-test assumes normal differences, which five seeds cannot check; the sign-flip test is exact under the symmetric null. Its price is coarse p-values ($k/32$).

It also beats a bootstrap interval at this size. A bootstrap over five values is unstable, and with small $n$ an exact test is the honest choice. The bootstrap fits better when there are hundreds of items, as in [Agent evals as tests](23-agent-evals-as-tests.md).

More seeds would give more power and finer p-values at linear cost; five is the minimum that can ever reject at 0.05, chosen to keep the gate fast.

The margins, 0.02 bits per byte and 0.02 quality, are set above the harmless variants' shifts and below the mutants' effects. A fixed margin is a decision someone signs, like the perf budget in [Benchmarks and perf gates](04-benchmarks-and-perf-gates.md).

## What can go wrong

Eight pitfalls account for the ways a model eval suite fails to protect anything:

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. checking only that the model runs | a wrong RoPE base, a dropped layer, a tokenizer one merge short, an unscaled attention, a mask that shows the next token: all train and all generate | a measured bpb and quality compared with a recorded baseline |
| 2. held-out loss as the only metric | a decoder that samples from the previous position's logits has a perfect bpb | a second, generation-based metric, `quality_score` |
| 3. "lower bpb is better, so it passed" | dropping the final norm's gain improves bpb by 0.16 and ruins the samples | the quality test fails independently of bpb |
| 4. exact equality with the baseline | float64 arithmetic or another machine fails the build | `regressed` requires the mean to exceed a margin |
| 5. a margin without a significance test | another sampling stream, 0.03 worse on average with mixed signs, fails the build | `regressed` also requires `sign_flip_p < 0.05` |
| 6. four seeds, or unpaired windows | nothing can ever reach $p < 0.05$; window difficulty drowns the difference | five seeds, asserted by `test_seeds_match_the_baseline`; the seed picks the windows |
| 7. a stale or typed-in baseline | the suite compares with a model that no longer exists | `baseline.json` is written only by `evals.py --record` |
| 8. bpb averaged per window, or per token | a tokenizer change hides behind more, easier tokens | bits and bytes are pooled before one division |

> **Warning:** A lower loss is not a pass. Dropping the final norm's gain improves bpb by 0.16 while the samples fall apart; only the generation metric sees it.

## Where this leads

This section reused judging by planted faults from [TDD, unit tests, and mutation testing](01-tdd-unit-tests-and-mutation-grading.md), the permutation tests of [Hypothesis tests](../../mathematics-for-computing/07-probability-statistics/05-hypothesis-tests.md) (the sign flip is their paired, one-sided, exact form), and bits per byte from [Perplexity, bits per byte, NLL accumulator](../../mathematics-for-computing/11-information-theory/02-perplexity-bits-per-byte-nll-accumulator.md). Going forward:

- The [first capstone](../../build-an-llm/capstones/01-tinystories.md) gates its release on held-out bpb and a seeded quality score with regression thresholds, measured this way; the [post-training chapter](../../build-an-llm/p12-post-training/README.md) and the [second capstone](../../build-an-llm/capstones/02-post-trained.md) use model evals too.
- [ModelRelease](../../ai-platform-engineering/05-durable-orchestration-and-workers/12-model-release.md) gates on the rows these evals produce.
- [Agent evals as tests](23-agent-evals-as-tests.md) applies the same discipline with scorers, a judge, and confidence intervals.

Each piece has a production equivalent:

- `heldout_bpb` corresponds to EleutherAI's lm-evaluation-harness, which adds rolling-window perplexity, bits per byte and per word, and hundreds of tasks (`lm_eval/api/task.py`, `loglikelihood_rolling`).
- `regressed` corresponds to Dror et al.'s significance testing for NLP, which adds test selection by metric, multiple comparisons (Holm), and effect sizes.
- `quality_score` corresponds to HELM and OpenAI evals, which add many scenarios and metrics per model and judges for open-ended outputs (`crfm-helm`).
- The model mutants correspond to mutation testing of ML systems (DeepMutation), with operators on weights and training data and killing by statistical tests (Ma et al., ISSRE 2018).

### Further reading

- Dror et al., ["The Hitchhiker's Guide to Testing Statistical Significance in NLP"](https://aclanthology.org/P18-1128/) (ACL 2018).
- Good, *Permutation, Parametric, and Bootstrap Tests of Hypotheses*, ch. 3.
- Biderman et al., ["Lessons from the Trenches on Reproducible Evaluation of Language Models"](https://arxiv.org/abs/2405.14782) (2024).

## Key terms

- **Model eval**: a held-out metric plus a regression rule, run on every change as a test.
- **Bits per byte (bpb)**: total predicted bits divided by total predicted bytes; independent of the tokenizer.
- **Paired measurement**: measuring baseline and candidate on the same windows and seeds, so per-seed differences cancel window difficulty.
- **Sign-flip test**: an exact permutation test that counts sign vectors whose flipped sum reaches the observed sum.
- **Margin**: the smallest regression worth failing a build for.
- **Model mutant**: a deliberately broken variant of a model (wrong RoPE base, dropped layer) used to judge the eval suite.

## Summary

- A model eval is a test: a held-out metric and a regression rule, run on every change, that fails the build, judged by the defects it catches.
- A model is never bit-identical to its baseline, so a margin and a significance test together keep the suite quiet on harmless changes.
- Five seeds is the minimum for $p < 0.05$ with an exact paired sign-flip test, because the smallest possible p-value is $1/2^n$ and $1/16 > 0.05$.
- One metric is not enough: held-out loss is blind to decoding bugs, and a broken final norm can make bpb better while samples get worse.

## Review questions

1. Why can a four-seed suite never flag a regression at $\alpha = 0.05$?

   <details><summary>Answer</summary>The identity sign vector always reaches the observed sum, so $p \ge 1/2^n$. For $n = 4$ that is $1/16 = 0.0625 > 0.05$, so `sign_flip_p(worse) < alpha` is never true.</details>

2. Why is the first token's byte length subtracted from the denominator in `heldout_bpb`?

   <details><summary>Answer</summary>The model never predicts the first token of a window (there is no context), so its bits are not in the numerator. Keeping its bytes in the denominator would lower bpb by an amount that depends on how long the tokenizer's first token is, which a tokenizer change would shift.</details>

3. Why does float64 arithmetic, which shifts every seed's bpb by about $10^{-7}$ in the same direction, not fail the suite?

   <details><summary>Answer</summary>All differences positive gives $p = 1/32 < 0.05$, so significance alone would flag it. The mean, $10^{-7}$, is far below the 0.02 margin, so `regressed` is false. This is why both conditions are required.</details>

4. Which mutant does held-out bpb miss, and why does the quality score catch it?

   <details><summary>Answer</summary>Sampling from the previous position's logits: teacher-forced `log_probs` never calls `sample`, so bpb is identical. The generated text is shifted by one step and turns into fragments and repeats, so the valid-word rate and the repeat factor drop.</details>

5. Why must `quality_score` use one generator for the four prompts in a fixed order?

   <details><summary>Answer</summary>So the score at seed $s$ is a deterministic function of the model and the seed. Then the baseline's and the candidate's scores at the same seed are paired measurements, and a difference is due to the model, not to which random numbers each prompt happened to draw.</details>
