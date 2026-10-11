<!-- ol:module L6.7 -->
# LM evaluation harness and the model zoo

## Overview

| | |
|---|---|
| **Module** | `L6.7` · build · Python · Pass 5 · 5 to 6 h, plus your graded tests (rung R5) |
| **You build** | `python/tinyllm/eval/lm.py`: `Result`, `ByteTokenizer`, `log_probs`, `token_nlls`, `eval_ppl`, `score_choices`, `run_task`, `compare`; `python/tinyllm/eval/zoo.py`: `read_config`, `kind_of`, `save_word2vec`, `load_model`, `lm_token_nlls`, `encode_batch`, `score_entry`, `run_zoo`, `write_report`; and your own oracle tests in `python/tests/l6-7-eval/` |
| **Contract** | [`course/contracts/py/tinyllm/eval/lm.pyi`](../../../course/contracts/py/tinyllm/eval/lm.pyi), [`course/contracts/py/tinyllm/eval/zoo.pyi`](../../../course/contracts/py/tinyllm/eval/zoo.pyi); the report: [`formats/eval-results.schema.json`](../../../course/contracts/formats/eval-results.schema.json) |
| **Tests** | `course/tests/L6.7/test_eval.py` (what they check: section 4); the perplexity oracle is Hugging Face's GPT-2; your tests are graded by mutation, threshold 0.80 with every required pitfall fault killed |
| **Needs** | [`M11.2` NLLAccumulator](../../../math/11-information-theory/02-perplexity-bits-per-byte-nll-accumulator.md) · [`M07.4` intervals](../../../math/07-probability-statistics/04-lln-clt-confidence-intervals-bootstrap.md) · [`M07.5` paired permutation test](../../../math/07-probability-statistics/05-hypothesis-tests.md) · every family: `L0.5` bigram, `L2.1` n-gram, `L2.2` NPLM, `L2.3` word2vec, `L3.6` RNN LM, `L4.1` seq2seq, `L5.5` Transformer, `L6.1` GPT, `L6.2` BERT, `L6.3` ELECTRA, `L6.5` classifiers · `L4.4` beam search · `L4.5` exact match · `L0.1` · `L0.6` · `M06.3` (or `--ref-deps`) |
| **Used by** | `L8.5` (`quant_ppl` reads a scheme's perplexity cost with `eval_ppl`) · later: `L7.9` parity, `C1` (the baseline table), `dur.11` `EvalSuite` (execs `{tinyllm} eval --suite zoo`), `L12.1` |
| **Milestone** | `MS-L6` (`eval ppl` reports an interval; `eval --suite zoo` reports one row per family) |
| **Optional depth** | Hugging Face, "Perplexity of fixed-length models" (docs); Gao et al., EleutherAI lm-evaluation-harness (2021 onward); Biderman et al., "Lessons from the Trenches on Reproducible Evaluation of Language Models" (2024); Dror et al., "The Hitchhiker's Guide to Testing Statistical Significance in NLP" (2018) |

## Key Takeaways

- Strided perplexity scores every token exactly once, each from the first window that reaches it, with at least `ctx_len - stride` tokens of context (`test_hand_example_strided_windows`, `test_every_token_is_scored_once`, `test_strided_nll_matches_hf`).
- Multiple choice is log-likelihood of each answer after the context; `acc_norm` divides by the answer's bytes, which can flip the winner (`test_hand_example_choices`).
- Every metric carries a 95% interval: Student t for mean NLL, bootstrap for task accuracy, Wilson for proportions, Fisher z for Spearman (`test_eval_ppl_bpb_and_interval`, `test_run_task_accuracy_with_bootstrap`).
- "Is B better than A?" is a **paired** question on the same items, answered by M07.5's sign-flip test (`test_compare_is_the_paired_permutation_test`).
- The zoo loads every family through its own checkpoint loader and scores it on its own task in one report; a broken entry becomes a row with a reason (`test_zoo_language_model_rows_are_each_familys_own_nll`, `test_zoo_reports_failures_as_rows`).

## How to work this chapter

```bash
ol start L6.7              # stubs lm.py and zoo.py; prints your test path and rung (R5)
ol tests L6.7              # the course tests
# write your oracle tests in python/tests/l6-7-eval/, then:
ol check L6.7              # course tests and the mutation grade of your tests
ol mutate L6.7             # the full grade, cached by your test files' hash
ol diff  L6.7              # after passing: your code against the reference
```

---

## 1. Why now

By now you have trained ten kinds of model: a byte bigram, a Kneser-Ney n-gram, an NPLM, word vectors, an RNN language model, an attention seq2seq, a Transformer, GPT, BERT, and ELECTRA, each with its own little evaluation in its own chapter. They cannot be compared. The n-gram reports perplexity per byte, GPT per window, the classifier an accuracy on whatever split was handy, and none says how sure it is. The capstone (`C1`) has to put your 10M-parameter Llama in a table next to all of them, and the durable workflow `EvalSuite` (`dur.11`) has to rerun that table on every release. This module writes the measuring instrument once: one definition of perplexity for fixed-context models, one of multiple-choice accuracy, an interval on every number, a paired test for "better", and a zoo that loads each family through the checkpoint it already saves.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x_0, \dots, x_{n-1}$ | the evaluation tokens | `int64[n]` |
| $\ell_t = -\ln p(x_t \mid \text{context})$ | the NLL of token $t$, in nats | `float64` |
| $C$, $s$ | `ctx_len` and `stride`, $1 \le s < C$ | `int` |
| $[a_k, e_k)$ | window $k$: $a_k = k s$, $e_k = \min(a_k + C, n)$ | `int` |
| $\bar\ell = \frac{1}{N} \sum_t \ell_t$ | mean NLL over the $N$ scored tokens | `float` |
| $\mathrm{ppl} = e^{\bar\ell}$ | perplexity | `float` |
| $\mathrm{bpb} = \sum_t \ell_t / (n_{\text{bytes}} \ln 2)$ | bits per byte | `float` |
| $\mathrm{LL}(c) = \sum_j \ln p(c_j \mid \text{ctx}, c_{<j})$ | a choice's log-likelihood | `float` |
| $a_i, b_i$ | two models' scores on item $i$ (`per_item`) | `float64[n]` |

### 2.1 Strided perplexity

A model with a fixed context of $C$ tokens cannot read a long text at once. Cutting the text into disjoint blocks of $C$ is cheap but unfair: the first tokens of every block are predicted from almost nothing. Sliding the window one token at a time is fair but costs $n$ forward passes. The strided compromise moves the window by $s$: window $k$ covers $[a_k, e_k)$ and scores only the tokens it **adds**, $t \in [\max(e_{k-1}, 1), e_k)$, each from its own log-probabilities at position $t - 1 - a_k$. The loop stops at the first window that reaches $n$. Three facts follow, and the tests check each:

1. Every token $1, \dots, n - 1$ is scored exactly once (token 0 has no context).
2. In every window after the first, a scored token has at least $C - s$ tokens of context.
3. $s = C$ is not allowed: each window's first token would be predicted with no context from its own window, which is the bug in many copies of the popular Hugging Face recipe. $s = C - 1$ is the cheapest valid stride and $s = 1$ the fairest.

The totals go through `M11.2`'s `NLLAccumulator`: compensated sums, then $\bar\ell$, $\mathrm{ppl}$, and, given the byte count, $\mathrm{bpb}$. Only bits per byte compares models with different tokenizers; for byte-level models (D32) one token is one byte.

### 2.2 Multiple choice by log-likelihood

A task item is a context and $k$ choices. Each choice is scored by its log-likelihood after the context: run the model on context plus choice and sum the log-probabilities of the choice's tokens, each read at the position before it. `acc` takes $\arg\max_c \mathrm{LL}(c)$; `acc_norm` takes $\arg\max_c \mathrm{LL}(c) / \mathrm{bytes}(c)$, because a sum of negative numbers favors short answers. Ties go to the first choice. This is the lm-evaluation-harness recipe, and the same function scores the capstone's held-out tasks.

### 2.3 Every metric with an interval

A metric from $n$ items is an estimate, and the zoo reports its 95% interval (`M07.4`):

| metric | interval | why that one |
|---|---|---|
| mean NLL, bpb, ppl | Student t on the per-token NLLs; ppl's ends are $e^{\text{ends}}$ | NLLs are many and roughly independent within a text |
| task accuracy (`run_task`) | percentile bootstrap of the per-item scores from the given rng | works for any statistic; reproducible in Go (ag.12) |
| zoo accuracy and EM | Wilson score interval | correct coverage for small $n$ and accuracies near 0 or 1 |
| Spearman | Fisher z: $\tanh(\operatorname{atanh}\rho \pm 1.96 / \sqrt{n - 3})$ | the standard interval for a rank correlation |

### 2.4 Paired comparison

Two models scored on the same items are not two independent samples: the items' difficulty is shared. `compare` runs `M07.5`'s paired permutation (sign-flip) test on $d_i = a_i - b_i$: under "no difference" each $d_i$ is as likely to have either sign, so flipping signs at random gives the null distribution of $\lvert\sum_i d_i\rvert$. A small p-value says the difference is not item noise. Comparing the two means with unpaired intervals throws the pairing away and misses real differences.

### 2.5 The model zoo

`load_model(dir)` reads `config.json` (the n-gram's safetensors file names its arch in its metadata) and dispatches on `tl_arch` to the family's own loader; a config with `tl_head` is an `L6.5` classifier. `run_zoo(manifest, rng)` scores each manifest entry by its kind:

| kind | families | task | metric |
|---|---|---|---|
| lm | bigram, ngram, nplm, rnnlm, gpt | a token file (`formats/tokens-bin.md`) | bpb from the family's own per-token NLLs (gpt: strided) |
| seq2seq | seq2seq, transformer | JSONL `{"src", "tgt"}` | exact match of the best beam hypothesis without its eos (`L4.4`, `L4.5`) |
| classifier | any `tl_head` | `split, sentence, label` rows | accuracy on the split |
| mlm | bert | the same rows | accuracy of the masked tokens |
| rtd | electra | the same rows | replaced-token detection accuracy (`L6.3`) |
| embeddings | word2vec | `w1, w2, score` lines | Spearman (`L2.3`) |

The report is `formats/eval-results.schema.json`, one row per entry in manifest order. A row that cannot be scored is not dropped: an exception becomes `status: "error"` with its type and message, an unknown family `status: "skipped"`, both with `value: null`, and the rest of the zoo is still scored. A table that silently loses its broken rows looks better than the system it measures.

## 3. Worked example by hand

**Strided windows.** Six tokens, $C = 4$, and a model that gives the true next token probability $1 / (j + 2)$ at window position $j$ (it has read the $j + 1$ tokens up to there), so each NLL reveals the context it was scored with.

| stride | windows | tokens scored by each | NLLs |
|---|---|---|---|
| 2 | $[0, 4)$, $[2, 6)$ | 1, 2, 3 at $j = 0, 1, 2$; then 4, 5 at $j = 1, 2$ | $\ln 2, \ln 3, \ln 4, \ln 3, \ln 4$ |
| 3 | $[0, 4)$, $[3, 6)$ | 1, 2, 3; then 4, 5 at $j = 0, 1$ | $\ln 2, \ln 3, \ln 4, \ln 2, \ln 3$ |

With stride 2 the mean is $(\ln 2 + 2 \ln 3 + 2 \ln 4) / 5$, so $\mathrm{ppl} = (2 \cdot 3^2 \cdot 4^2)^{1/5} = 288^{1/5} = 3.1037$. The larger stride is cheaper and gives tokens 4 and 5 less context.

**Multiple choice.** Context "a"; choices "bb" and "c"; $p(b \mid a) = 0.4$, $p(b \mid b) = 0.9$, $p(c \mid a) = 0.45$.

| choice | LL | bytes | LL per byte |
|---|---|---|---|
| "bb" | $\ln 0.4 + \ln 0.9 = -1.0217$ | 2 | $-0.5108$ |
| "c" | $\ln 0.45 = -0.7985$ | 1 | $-0.7985$ |

`acc` picks "c", `acc_norm` picks "bb".

**The zoo's unit.** A bigram whose table is all zeros gives every byte probability $1/256$: $\ell_t = \ln 256$ and $\mathrm{bpb} = \ln 256 / \ln 2 = 8$ exactly, with a zero-width interval because every NLL is the same.

These are `test_hand_example_strided_windows`, `test_hand_example_choices`, and `test_hand_example_uniform_bigram_is_eight_bits`.

## 4. The interface

```python
@dataclass
class Result: value: float; lo: float; hi: float; n: int; per_item: NDArray
def token_nlls(model, ids, ctx_len: int, stride: int) -> NDArray: ...          # [n - 1]
def eval_ppl(model, ids, ctx_len, stride, n_bytes=0) -> dict[str, float]: ...
def score_choices(model, tok, context, choices, normalize=True, max_len=None) -> NDArray: ...
def run_task(model, tok, task_jsonl, metric, rng, n_boot=1000, max_len=None) -> Result: ...
def compare(a: Result, b: Result, n_perm=10000, rng=None) -> float: ...
def load_model(dir) -> Any: ...                     # zoo dispatch on tl_arch
def run_zoo(manifest, rng, seed=0) -> dict: ...     # formats/eval-results report
```

A model is anything callable as `model(ids[None])` that returns logits `[1, T, V]`, or a tuple whose first item is (GPT returns `(logits, loss)`).

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_strided_windows` | unit | section 3: the windows the model is called on, the NLLs, $288^{1/5}$ | you and the test agree on the definition |
| `test_hand_example_choices` | unit | section 3: both sums, both normalizations, the two winners | acc and acc_norm |
| `test_hand_example_uniform_bigram_is_eight_bits` | unit | a uniform byte model scores exactly 8 bpb, zero-width interval | the unit every LM row is in |
| `test_strided_nll_matches_hf` | golden | per-token NLLs of HF's GPT-2 under four (ctx, stride) settings, loaded by `L6.1` | the definition on a real model |
| `test_every_token_is_scored_once` | property | 25 random settings: $n - 1$ tokens, context bounds; stride $\ge$ ctx refused | no token skipped or counted twice |
| `test_eval_ppl_bpb_and_interval` | differential | M11.2's totals, M07.4's interval, rescaled to ppl and bpb | numbers with error bars |
| `test_run_task_accuracy_with_bootstrap` | differential | per-item scores, ties to the first choice, M07.4's bootstrap from the same rng | task tables in C1 and L8.5 |
| `test_compare_is_the_paired_permutation_test` | differential | M07.5's p-value on the per-item scores; mismatched items refused | "better" means significant on the same items |
| `test_zoo_language_model_rows_are_each_familys_own_nll` | differential | five LM families: bpb and interval from each family's own NLL | the baseline table of C1 |
| `test_zoo_seq2seq_rows_decode_with_beam` | differential | EM of the beam decode without eos; transformer through `translate` | the seq2seq and transformer rows |
| `test_zoo_accuracy_rows` | unit | classifier, BERT, and ELECTRA rows with Wilson intervals; the val split only | the backbone comparison of MS-L6 |
| `test_zoo_word_vectors_row` | unit | Spearman from `L2.3` with the Fisher z interval | the word2vec row |
| `test_zoo_report_validates_against_the_schema` | conformance | `formats/eval-results.schema.json`, row order, seed | EvalSuite reads it |
| `test_zoo_reports_failures_as_rows` | fault | a missing directory is an error row, an unknown family a skipped row, the healthy row is scored | one broken family never hides the rest |

### Your graded tests (rung R5)

Your oracles: a toy model whose NLL encodes its context length, hand-computed log-likelihoods from a table, M07.4 and M07.5 called directly, and checkpoints whose score you know exactly (a uniform bigram is 8 bpb; a seq2seq model with a huge eos bias answers the empty string). Cover the windows and the refused stride, the interval's transform, both choice scores, the pairing, the bpb unit, and eos stripping. Import only the contract. `ol check L6.7` requires a mutation score of at least 0.80 with every required pitfall fault killed.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. rescoring the overlap, or allowing stride = ctx_len | tokens scored with less context than they had; window starts predicted from nothing | `test_hand_example_strided_windows` (mutant `s01`), `test_every_token_is_scored_once` (mutant `s02`) |
| 2. reading a choice's log-probabilities one position late | the score of the wrong tokens; accuracy near chance | `test_hand_example_choices` (mutant `s03`) |
| 3. acc and acc_norm swapped, or ties to the last choice | short-answer bias where none was asked for; results that depend on choice order | `test_hand_example_choices` (mutant `s04`), `test_run_task_accuracy_with_bootstrap` (mutant `s07`) |
| 4. bits per byte in nats, or the bigram table read transposed | LM rows off by $\ln 2$; a bigram scored on the reversed text | `test_hand_example_uniform_bigram_is_eight_bits` (mutant `s05`), `test_zoo_language_model_rows_are_each_familys_own_nll` (mutant `s09`) |
| 5. intervals left in log space, or Fisher z with $\sqrt n$ | a perplexity interval in nats; Spearman intervals too narrow | `test_eval_ppl_bpb_and_interval` (mutant `s06`), `test_zoo_word_vectors_row` (mutant `s12`) |
| 6. comparing unpaired items | a real difference reported as noise, or the reverse | `test_compare_is_the_paired_permutation_test` (mutant `s08`) |
| 7. exact match with the eos still on | every seq2seq row scores 0 | `test_zoo_seq2seq_rows_decode_with_beam` (mutant `s10`) |
| 8. classifier rows scored on every split | training sentences inflate the accuracy | `test_zoo_accuracy_rows` (mutant `s11`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M11.2` | `NLLAccumulator` turns per-token NLLs into ppl and bpb |
| Back | `M07.4` | `mean_ci`, `bootstrap_ci`, `wilson_interval`, `normal_ppf` |
| Back | `M07.5` | `paired_permutation_test` in `compare` |
| Back | `M06.3` | PCG32: `compare`'s default stream; the zoo's draws |
| Back | `L0.1` | `no_grad` while scoring; the Tensor of logits |
| Back | `L0.5` | `BigramLM`, the byte bigram family |
| Back | `L0.6` | safetensors and `open_tokens` for checkpoints and token files |
| Back | `L2.1` | `NGramLM.load` and `nll` |
| Back | `L2.2` | `load_nplm` and `nll` |
| Back | `L2.3` | `word_similarity` for the word2vec row |
| Back | `L3.6` | `load_rnnlm` and `nll` |
| Back | `L4.1` | `load_seq2seq`, `encode`, `decode_step` |
| Back | `L4.4` | `beam_search` decodes the seq2seq rows |
| Back | `L4.5` | `exact_match` scores them |
| Back | `L5.5` | `load_transformer` and `translate` |
| Back | `L6.1` | `load_gpt`; `load_hf_gpt2` in the golden test |
| Back | `L6.2` | `load_bert` and `mlm_mask` for the masked-token row |
| Back | `L6.3` | `load_electra` and `rtd_accuracy` for the ELECTRA row |
| Back | `L6.5` | `load_classifier` and `predict` for classifier rows |
| Forward | `L7.9` | SmolLM2 parity reports bpb through `eval_ppl` |
| Forward | `L8.5` | `quant_ppl` runs `eval_ppl` on a float model and its quantized copy: the perplexity cost MS-L8 budgets |
| Forward | `C1` | the capstone's baseline table is `run_zoo` over every family trained so far |
| Forward | `dur.11` | the `EvalSuite` workflow execs `{tinyllm} eval --suite zoo` and keeps the report |

`L8.5` is the registered call site; the others join with their batches. If you skip this module, the capstone's evaluation step fails: build it, or pass `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `token_nlls`, `eval_ppl` | HF "Perplexity of fixed-length models" recipe | batched windows on GPU (and an off-by-one in the token count you no longer have) | `transformers` docs, `perplexity.md` |
| `score_choices`, `run_task` | EleutherAI lm-evaluation-harness | hundreds of tasks, few-shot prompts, `loglikelihood_rolling`, caching, request reordering | `lm_eval/api/model.py`, `lm_eval/api/task.py` |
| intervals and `compare` | HELM, the harness's stderr | per-task standard errors; paired bootstrap across scenarios | Liang et al., "Holistic Evaluation of Language Models" (2022) |
| `run_zoo` | the LLM leaderboard, OpenCompass | many models by many tasks, with result stores and versioned task definitions | `open-llm-leaderboard`, `opencompass` |
