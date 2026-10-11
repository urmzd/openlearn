<!-- ol:module craft.22 -->
# Model evals as tests (R8)

## Overview

| | |
|---|---|
| **Module** | `craft.22` · practice · Python · Pass 9 · 3 to 4 h |
| **You build** | `primers/craft.22/evals.py` (a kata: held-out bits per byte, a seeded sample-quality score, the paired sign-flip test, and the regression rule), `primers/craft.22/test_model_evals.py` (your eval suite), and `primers/craft.22/baseline.json` (the good model, measured) |
| **Contract** | the rules in the kata's docstring (`ol start craft.22` writes the stub; `ol check craft.22` writes the course's model `tinymodel.py` beside it) |
| **Tests** | `course/tests/craft.22/`: the grade of your suite by two harmless variants and seven model mutants (section 4), on the model in `course/fixtures/craft.22/` (`course/oracle/craft.22/train_tinylm.py`) |
| **Needs** | reading: [`craft.03`](01-tdd-unit-tests-and-mutation-grading.md) how tests are graded · [`craft.07`](05-mutation-testing-in-depth.md) reading survivors · `M07.5` permutation tests · `M11.2` bits per byte · `L6.7` the eval harness and zoo · `L7.3` RoPE |
| **Used by** | no call site (a practice): rung R8 grades the model evals of `C1` (and of `L12` and `C2`), whose release gates are these numbers |
| **Milestone** | `MS-C1` (the capstone's held-out loss and seeded quality are measured with your evals) |
| **Optional depth** | Dror et al., ["The Hitchhiker's Guide to Testing Statistical Significance in NLP"](https://aclanthology.org/P18-1128/) (ACL 2018, free); Good, *Permutation, Parametric, and Bootstrap Tests of Hypotheses*, ch. 3; Biderman et al., ["Lessons from the Trenches on Reproducible Evaluation of Language Models"](https://arxiv.org/abs/2405.14782) (2024, free) |

## Key Takeaways

- A model eval **is a test**: a held-out metric and a regression rule, run on every change, that fails the build. It is graded like any test, by the defects it catches.
- A model is never bit-identical to its baseline: float64 arithmetic, another BLAS, or a new sampling stream all move the numbers. A **margin and a significance test** keep the suite quiet on harmless changes (`test_no_false_alarms`).
- **Five seeds is the minimum** for $p < 0.05$ with an exact paired sign-flip test: the smallest possible p-value is $1/2^n$, and $1/16 > 0.05$ (`test_hand_example_five_seeds`).
- **One metric is not enough.** Held-out loss is blind to decoding bugs, and a broken final norm can make bpb *better* while samples get worse; the suite needs a loss and a seeded generation score (`test_model_mutants_are_flagged`).

## How to work this chapter

```bash
ol start craft.22           # writes primers/craft.22/evals.py (stubs)
ol check craft.22           # first run: also writes tinymodel.py, then fails
# write evals.py, record the baseline, write test_model_evals.py, run it yourself:
export TINYLLM_FIXTURES=$PWD/.ol/openlearn/course/fixtures
uv run --no-project --with numpy python primers/craft.22/evals.py --record
(cd primers/craft.22 && uv run --no-project --with pytest --with numpy python -m pytest -q test_model_evals.py)
ol check craft.22           # grades your suite: harmless variants and model mutants
```

(`ol check` sets `TINYLLM_FIXTURES` itself; point it at the `fixtures/` directory of your openlearn checkout when you run by hand.) A survivor prints only its one-line description, never the planted code.

---

## 1. Why now

Your capstone (`C1`) trains for hours, and your release workflow (`dur.12`) gates on eval numbers. Until now a model was right when its logits matched a fixture to $10^{-5}$ (Parts 5 to 8). That oracle is gone the moment you train your own model: there is no reference to match, and two correct runs differ. Yet the bugs are real and quiet. A RoPE base read from the wrong config key, a layer skipped by an off-by-one, a tokenizer that lost one merge: every one of them still trains, still produces plausible text, and costs a few tenths of a bit per byte. Without a regression test you find out from users. This module builds the test: measured numbers, recorded from a good model, compared with statistics that know the difference between noise and a defect.

## 2. Principles

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

### 2.1 Paired measurements

Measure the baseline and the candidate on **the same** windows and with **the same** seeds, so that each seed gives one difference $d_s$. The windows' own difficulty (one held-out story is harder than another) then cancels in $d_s$ instead of swamping it. Held-out bits per byte pools the bits of every window over their bytes, which makes it independent of the tokenizer: a tokenizer change that splits text into more tokens cannot hide behind a per-token loss.

### 2.2 The sign-flip test

If the candidate is as good as the baseline, each $d_s$ is as likely to be positive as negative: flipping its sign gives an equally likely world. Enumerate all $2^n$ flips and ask how often the flipped sum is at least the one you saw. That fraction is an exact p-value with no distribution assumed. When all $n$ differences are positive, only the unflipped vector reaches $T$, so $p = 1/2^n$: $1/32 = 0.031$ for five seeds, $1/16 = 0.0625$ for four. **A four-seed suite can never reject at 0.05.**

### 2.3 Margin and significance together

Flag a regression when **both** hold: the mean difference exceeds $\delta$ (it matters) and $p < \alpha$ (it is not noise). Significance alone fails on tiny, real, harmless shifts (float64 arithmetic moves every seed by $10^{-7}$, all in one direction would give $p = 1/32$); the margin alone fails on noise (another sampling stream can make quality 0.03 worse on average with signs going both ways).

### 2.4 Two kinds of metric

Held-out bpb tests what the model computes under teacher forcing. A seeded generation score tests what it does when it runs: sample from fixed prompts with a seeded generator and score the text (here: the fraction of words that exist in the training vocabulary, times one minus the rate of repeated 4-grams). Decoding bugs live only in the second; some forward bugs show up only there too.

## 3. Worked example by hand

Five seeds. The baseline's bpb is 1.00 at each, $\delta = 0.02$, $\alpha = 0.05$.

**Candidate A**: 1.03 at every seed. $d = (0.03, 0.03, 0.03, 0.03, 0.03)$, mean $0.03 > 0.02$. $T = 0.15$; every flip of a positive term lowers the sum, so only $\epsilon = (+,+,+,+,+)$ reaches $0.15$: $p = 1/32 = 0.03125 < 0.05$. **Regressed.**

**Candidate B**: 1.03, 1.03, 1.03, 1.03, 0.99. $d = (0.03, 0.03, 0.03, 0.03, -0.01)$, mean $0.022 > 0.02$, $T = 0.11$. The identity gives 0.11; flipping the last sign gives $0.13 \ge 0.11$; any flip of a $0.03$ lowers the sum below 0.11. So $p = 2/32 = 0.0625$. **Not a regression** (one seed disagrees).

**Candidate C**: 1.01 at every seed: $p = 1/32$, but the mean $0.01 < 0.02$. **Not a regression** (real, but too small to fail a build for).

**Four seeds**, all worse: $p = 1/16 = 0.0625$. Nothing can ever be flagged.

These numbers are `test_hand_example_five_seeds`.

## 4. The artifact and its check

**The kata**, `primers/craft.22/evals.py` (its docstring is the spec):

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

**The model**, `primers/craft.22/tinymodel.py` (the course's; read it, do not edit it): `load()`, `encode`, `decode`, `log_probs(ids)`, `sample(prompt, tokens, rng, temperature=0.8)`. It is a two-layer Llama-style decoder (RMSNorm, RoPE, SwiGLU, tied embeddings) over a character BPE of 96 merges, trained on synthetic stories; the held-out stories and the training vocabulary are fixtures.

**Your suite**, `primers/craft.22/test_model_evals.py`: at least two tests, importing only `evals`, `tinymodel`, numpy, pytest, and the stdlib, reading the fixtures through `TINYLLM_FIXTURES`, and failing when held-out bpb or sample quality regressed against `baseline.json` (margins 0.02 each work well).

**The check** (`ol check craft.22`, `course/tests/craft.22/check`) runs eight tests:

| Test | KIND | Checks |
|---|---|---|
| `test_hand_example_five_seeds` | unit | section 3 with your `sign_flip_p` and `regressed` |
| `test_your_evals_match_the_reference` | golden | your bpb, word, repeat, and quality functions equal the course's |
| `test_your_suite_is_an_eval_suite` | unit | the suite exists, at least 2 tests, allowed imports, `TINYLLM_FIXTURES`; `baseline.json` has 5 or more seeds |
| `test_your_baseline_is_the_good_model` | golden | `baseline.json` equals the good model measured now |
| `test_your_suite_passes_on_the_good_model` | conformance | baseline A: your suite passes on the model it was recorded from |
| `test_no_false_alarms` | conformance | your suite passes on `b01` (float64) and `b02` (another sampling stream) |
| `test_model_mutants_are_flagged` | fault | your suite fails on at least 80% of the 7 model mutants, always on `s01` to `s03` |

Each run copies your suite, your evals, and your baseline next to one version of the model in a scratch directory and runs pytest in its own process group with a timeout.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. checking only that the model runs | a wrong RoPE base, a dropped layer, a tokenizer one merge short, an unscaled attention, a mask that shows the next token: all train and all generate | `test_model_mutants_are_flagged` (mutants `s01`, `s02`, `s03`, `s05`, `s06`) |
| 2. held-out loss as the only metric | a decoder that samples from the previous position's logits has a perfect bpb | `test_model_mutants_are_flagged` (mutant `s04`) |
| 3. "lower bpb is better, so it passed" | dropping the final norm's gain improves bpb by 0.16 and ruins the samples | `test_model_mutants_are_flagged` (mutant `s07`) |
| 4. exact equality with the baseline | float64 arithmetic or another machine fails the build | `test_no_false_alarms` |
| 5. a margin without a significance test | another sampling stream, 0.03 worse on average with mixed signs, fails the build | `test_no_false_alarms` |
| 6. four seeds, or unpaired windows | nothing can ever reach $p < 0.05$; window difficulty drowns the difference | `test_your_suite_is_an_eval_suite`, `test_model_mutants_are_flagged` |
| 7. a stale or typed-in baseline | the suite compares with a model that no longer exists | `test_your_baseline_is_the_good_model` |
| 8. bpb averaged per window, or per token | a tokenizer change hides behind more, easier tokens | `test_your_evals_match_the_reference` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `craft.03` | mutation grading: baselines A and B, required faults |
| Back | `M07.5` | permutation tests; the sign flip is the paired, one-sided, exact form |
| Back | `M11.2` | bits per byte |
| Forward | `C1` | the capstone's held-out bpb and seeded quality score, with regression thresholds, gate its release |
| Forward | `dur.12` | `ModelRelease` gates on the rows these evals produce |
| Forward | `craft.23` | agent evals: the same discipline with scorers, a judge, and confidence intervals |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `heldout_bpb` | EleutherAI lm-evaluation-harness | rolling-window perplexity, bits per byte and per word, hundreds of tasks | `lm_eval/api/task.py` (`loglikelihood_rolling`) |
| `regressed` | Dror et al.'s significance testing for NLP | test selection by metric, multiple comparisons (Holm), effect sizes | the paper above |
| `quality_score` | HELM, OpenAI evals | many scenarios and metrics per model, judges for open-ended outputs | `crfm-helm` |
| model mutants | mutation testing of ML systems (DeepMutation) | operators on weights and training data, killing by statistical tests | Ma et al., ISSRE 2018 |
