<!-- ol:module M11.2 -->
# Perplexity, bits per byte, NLL accumulator

## Overview

| | |
|---|---|
| **Module** | `M11.2` · build · Python · Pass 3 · 2 to 3 h |
| **You build** | `python/tinyllm/info/ppl.py`: `perplexity`, `NLLAccumulator` (`add`, `merge`, `result`) |
| **Contract** | [`course/contracts/py/tinyllm/info/ppl.pyi`](../../course/contracts/py/tinyllm/info/ppl.pyi) |
| **Tests** | `course/tests/M11.2/` (what they check: section 4) |
| **Needs** | `M00.1` `bits_per_byte` (`tinyllm/num/units.py`) · reading: `M11.1` cross-entropy, `M09.2` compensated sums (or `--ref-deps`) |
| **Used by** | `L1.6` tokenizer metrics · later: `L2.1`, `L2.2`, `L3.6` model perplexities, `L6.7` the model-zoo bpb table, `L8.5` quantization budgets, `C1` |
| **Milestone** | `MS-P3` (tokens and data) |
| **Optional depth** | Cover and Thomas, *Elements of Information Theory* (2nd ed.), ch. 2 and 5; Gao et al., "The Pile" (2020), section 4 (bits per byte); Higham, *Accuracy and Stability of Numerical Algorithms*, ch. 4 |

## Key Takeaways

- Perplexity is $e$ to the mean NLL per token: the geometric mean of $1/q$, so a model that is uniform over $k$ choices scores exactly $k$ (`test_hand_example_sentence`, `test_perplexity_of_a_uniform_model_is_k`).
- Bits per byte divides the same total by the text's UTF-8 length instead of its token count, which makes it the one number that compares models with different tokenizers (`test_bits_per_byte_compares_tokenizers`).
- A corpus mean is the total over all tokens divided by the token count, never the average of batch means, and padding is selected out, not multiplied by zero (`test_hand_example_two_batches_with_padding`, `test_masked_positions_are_ignored_whatever_they_hold`).
- Summing each batch exactly and carrying the rounding error across batches makes any split of the corpus give the same answer to float64 rounding (`test_streaming_equals_one_batch`, `test_compensated_sum_across_many_adds`).

## How to work this chapter

```bash
ol start M11.2              # stubs ppl.py into your repo
ol tests M11.2              # read the test catalog first
ol check M11.2              # exit code is the verdict
ol check M11.2 --ref-deps   # only if your M00.1 is not passing yet
ol diff  M11.2              # after passing: your code against the reference
```

---

## 1. Why now

Your tracer bigram reports one number, `nll`: the mean negative log-likelihood of one text, in nats per token. Pass 3 breaks that number in two ways. First, tokenizers arrive: the byte tokenizer, BPE with 4096 merges (`L1.2`), Unigram (`L1.4`). A BPE model and a byte model scoring the same text report nats per *their* token, and a BPE token is three or four bytes, so their perplexities are not comparable; `L1.6`'s tokenizer metrics and the `C1` ablation need a per-byte number. Second, evaluation stops fitting in one call: the zoo (`L6.7`) scores $10^7$ tokens in padded batches, possibly on several workers. Averaging batch means, counting padding, or letting float rounding drift all change the reported number while every individual line of code looks right. This module turns per-token losses into perplexity and bits per byte, correctly, one batch at a time.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $x_1, \dots, x_n$ | the tokens of an evaluation text | `int64[n]` |
| $q_t$ | the probability the model gave the true token $x_t$ | `float` in $(0, 1]$ |
| $\ell_t = -\ln q_t$ | that token's negative log-likelihood, in nats | `float` $\ge 0$ |
| $S = \sum_t \ell_t$ | the total NLL of the text, in nats | `float` |
| $n$ | the number of scored tokens | `int` |
| $B$ | the number of UTF-8 bytes the scored tokens cover | `int` |
| $\bar\ell = S / n$ | the mean NLL per token (the cross-entropy, `M11.1`) | `float` |
| $\mathrm{PPL} = e^{\bar\ell}$ | perplexity | `float` $\ge 1$ |
| $\mathrm{bpb} = S / (B \ln 2)$ | bits per byte | `float` |
| $s$, $c$ | a running sum and its compensation (the low bits it lost) | `float64` |

**From NLL to perplexity.** The model gives the text probability $\prod_t q_t$ (each prediction conditioned on what came before). Its log is $-S$. The mean $\bar\ell = S/n$ is the cross-entropy between the text and the model, in nats per token. Perplexity undoes the log:

$$\mathrm{PPL} = e^{\bar\ell} = \Big(\prod_t \frac{1}{q_t}\Big)^{1/n},$$

the geometric mean of $1/q_t$. If the model spreads its mass evenly over $k$ tokens every time, $q_t = 1/k$, $\ell_t = \ln k$, and $\mathrm{PPL} = k$ exactly: perplexity is the effective number of tokens the model is choosing between. A perfect model scores 1. A model that gives a real token probability 0 has $\ell_t = \infty$ and $\mathrm{PPL} = \infty$ (the zero problem of `M07.2`). In float64, $e^{\bar\ell}$ overflows once $\bar\ell > \ln(\text{DBL\_MAX}) \approx 709.78$; an untrained model can get there, so the function returns $\infty$ instead of raising.

**Units.** One bit is $\ln 2$ nats (`M00.1`), so bits per token is $\bar\ell / \ln 2$, and $\mathrm{PPL} = 2^{\text{bits per token}}$.

**Bits per byte.** $S$ is the length, in nats, of the shortest code for the whole text under the model: an arithmetic coder driven by the model spends $-\log q_t$ for each token (`M11.3`). That total does not care how the text was cut into tokens; only the model's skill at predicting it matters. Dividing by the number of tokens brings the tokenizer back in. Dividing by the number of bytes does not, because every tokenizer sees the same $B$ bytes:

$$\mathrm{bpb} = \frac{S}{B \ln 2} = \text{bits per token} \times \frac{n}{B}.$$

A BPE model whose tokens average 4 bytes and a byte model can have per-token perplexities that differ by a factor of hundreds while their bits per byte say which one actually predicts the text better. `L6.7` reports bpb for every model in the zoo for this reason.

**One mean, many batches.** The corpus mean is $S/n$ with $S$ and $n$ summed over every batch. The average of batch means, $\frac1m \sum_j S_j / n_j$, weights each batch equally, so a short last batch or heavily padded batches pull it away from $S/n$; the two agree only when every batch has the same token count. So the accumulator keeps the two totals, never a running mean.

**Masks select.** Padded batches have slots with no real token, whose loss is garbage: a loss against a pad id, $\infty$, or NaN. A mask marks the slots that count. Multiplying by the mask looks equivalent and is not, because $\mathrm{NaN} \times 0 = \mathrm{NaN}$ and $\infty \times 0 = \mathrm{NaN}$; selecting the masked values (`x[mask]`) never touches them. The token count is the number of selected slots, not the batch size.

**Summing $10^7$ numbers.** Adding a small number to a large float64 total rounds away its low bits: near 1.0 the spacing is $2.2 \times 10^{-16}$, so $1 + 10^{-16} = 1$ exactly. Over $10^7$ tokens the lost bits add up to changes in the last digits of a reported perplexity, and worse, the answer depends on batch size and order. Two tools fix it. Within a batch, `math.fsum` returns the exactly rounded sum of the batch (after widening float32 losses to float64: summing $10^5$ float32 values in float32 loses about seven digits). Across batches, Neumaier's compensated summation keeps a second float $c$: for each batch total $x$,

$$t = s + x, \qquad c \mathrel{+}= \begin{cases} (s - t) + x & \lvert s \rvert \ge \lvert x \rvert \\ (x - t) + s & \text{otherwise} \end{cases}, \qquad s = t,$$

and the total is $s + c$. When $\lvert s \rvert \ge \lvert x \rvert$, $s - t$ is computed exactly, and $(s - t) + x$ is exactly the part of $x$ that $t$ dropped. This is the same idea as Kahan's sum in `M09.2`, run as a stream. `merge` adds another accumulator's $s$ and then its $c$, so shards scored on different workers combine without losing their low bits.

## 3. Worked example by hand

Score `the cat sat.` as four BPE tokens with the model probabilities below. The text is 12 UTF-8 bytes.

| token | `the` | ` cat` | ` sat` | `.` | total |
|---|---|---|---|---|---|
| bytes | 3 | 4 | 4 | 1 | $B = 12$ |
| $q_t$ | 1/4 | 1/2 | 1/8 | 1/2 | |
| $\ell_t = -\ln q_t$ | $2 \ln 2$ | $\ln 2$ | $3 \ln 2$ | $\ln 2$ | $S = 7 \ln 2 = 4.8520$ nats |

- Mean: $\bar\ell = 7 \ln 2 / 4 = 1.75 \ln 2 = 1.2130$ nats per token, which is 1.75 bits per token.
- Perplexity: $e^{1.75 \ln 2} = 2^{1.75} = 3.3636$. The model is as unsure as a fair choice among about 3.4 tokens.
- Bits per byte: $7 \text{ bits} / 12 \text{ bytes} = 0.5833$.

**A byte model on the same text.** Suppose a byte-level model spends 0.7 bits on each of the 12 bytes: 8.4 bits in all. Its perplexity per (byte) token is $2^{0.7} = 1.62$, which looks better than 3.36. Its bits per byte, 0.7, is worse than 0.5833. The BPE model predicts this text better; only bpb says so.

**Batches.** Score the same four tokens as two batches, `[the]` and `[ cat,  sat, ., PAD]` with mask `[1, 1, 1, 0]` and NaN in the pad slot. Batch means: $2 \ln 2$ and $(1 + 3 + 1)/3 \cdot \ln 2 = 1.667 \ln 2$. Their average is $1.833 \ln 2$: wrong. The totals give $S = 2\ln 2 + 5 \ln 2 = 7 \ln 2$ over $n = 1 + 3 = 4$ tokens: $1.75 \ln 2$, the same as one batch.

**Compensation.** Start at $s = 1$, $c = 0$ and add $10^{-16}$ ten times. Plain addition: $1 + 10^{-16}$ rounds to 1 every time, so the total stays 1. Neumaier: each step $t = 1$, $c \mathrel{+}= (1 - 1) + 10^{-16}$, so after ten steps $c = 10^{-15}$ and $s + c = 1.000000000000001$, the right answer.

These numbers are the first cases in section 4: `test_hand_example_sentence` and `test_hand_example_two_batches_with_padding`.

## 4. The interface

```python
# python/tinyllm/info/ppl.py
def perplexity(nll_sum: float, n_tokens: int) -> float          # exp(nll_sum / n_tokens); inf on overflow
class NLLAccumulator:
    def __init__(self) -> None
    def add(self, nll: ArrayLike, mask: ArrayLike | None = None, n_bytes: int = 0) -> None
    def merge(self, other: "NLLAccumulator") -> None
    def result(self) -> dict[str, float]
    # {"nll_sum", "n_tokens", "n_bytes", "nll_mean", "ppl", "bits_per_token", "bpb"}
```

Call `bits_per_byte` from your `M00.1` for `bpb`; with no bytes added, `bpb` is NaN. A counted NaN or negative loss is a `ValueError`; a counted $+\infty$ is allowed and makes `ppl` infinite.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_sentence` | unit | $7 \ln 2$, 1.75 bits per token, $\mathrm{PPL} = 3.3636$, bpb 0.5833 | you and the test agree on every definition |
| `test_hand_example_two_batches_with_padding` | unit | two batches with a NaN pad give the one-batch numbers | the mean of means is wrong |
| `test_perplexity_of_a_uniform_model_is_k` | property | $\mathrm{PPL} = k$ for uniform-$k$ losses, up to $k = 50\,257$ | perplexity as a number of choices |
| `test_perplexity_overflow_is_inf` | boundary | 800 nats per token and an infinite total give $\infty$ | untrained models in the zoo |
| `test_perplexity_rejects_bad_arguments` | boundary | zero tokens, a negative or NaN total | sign bugs upstream |
| `test_masked_positions_are_ignored_whatever_they_hold` | boundary | NaN, $\infty$, and negative values in masked slots; 0/1 masks | padded batches in `L6.7` |
| `test_rejects_bad_batches` | boundary | mask shape, counted NaN or negative, negative bytes; a counted $\infty$ is allowed | caller bugs fail loudly |
| `test_result_needs_tokens_and_bpb_needs_bytes` | boundary | no tokens raises; no bytes gives NaN bpb | a table shows "missing", not "perfect" |
| `test_streaming_equals_one_batch` | property | random splits in reverse order match one batch to $10^{-13}$ | the number does not depend on batch size |
| `test_compensated_sum_across_many_adds` | boundary | $1 + 10^5 \times 10^{-15}$ to within 2 ulps | drift over $10^7$ tokens |
| `test_float32_batches_are_widened` | boundary | $10^5$ float32 losses summed to $10^{-15}$ | losses arrive as float32 |
| `test_merge_equals_a_single_accumulator` | property | merged shards equal one accumulator, low bits included; `other` unchanged | sharded evaluation in `dur.11` |
| `test_bits_per_byte_compares_tokenizers` | unit | the byte model has lower perplexity and higher bpb | `L1.6`, the `C1` tokenizer ablation |
| `test_golden_cases` | golden | three long cases against 50-digit totals from `course/oracle/M11.2` | independent of float64 rounding |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. averaging batch means | the reported loss changes with the batch size and the last batch's length | `test_hand_example_two_batches_with_padding` (mutant `s01`) |
| 2. multiplying by the mask | NaN in a pad slot turns the total into NaN; pad slots are counted as tokens | `test_masked_positions_are_ignored_whatever_they_hold` (mutants `s02`, `s03`) |
| 3. plain float64 addition across batches | $10^{-15}$-sized contributions vanish into a large total; the result depends on order | `test_compensated_sum_across_many_adds` (mutant `s04`) |
| 4. summing float32 losses in float32 | seven digits of a $10^5$-token batch lost | `test_float32_batches_are_widened` (mutant `s05`) |
| 5. `math.exp` of a huge mean | `OverflowError` in the middle of a zoo report | `test_perplexity_overflow_is_inf` (mutant `s06`) |
| 6. bits per byte divided by tokens, or left in nats | a bpb that is really bits per token, or off by $\ln 2$ | `test_hand_example_sentence` (mutants `s07`, `s08`) |
| 7. a merge that drops the other side's compensation or bytes | sharded and single runs disagree in the last digits, or bpb is wrong after merging | `test_merge_equals_a_single_accumulator` (mutants `s09`, `s10`) |
| 8. accepting a counted NaN, or reporting on zero tokens | NaN or a `ZeroDivisionError` reaches the report | `test_rejects_bad_batches`, `test_result_needs_tokens_and_bpb_needs_bytes` (mutants `s11`, `s12`) |

## 6. Where it's used next
| Forward | `L2.1` | Registered call site uses this module. |
| Forward | `L2.2` | Registered call site uses this module. |

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M00.1` | `bits_per_byte` converts the nats total to bits per byte |
| Back | `M11.1` | cross-entropy: the mean NLL is the cross-entropy of the text under the model |
| Back | `M09.2` | Kahan's compensated sum, here run as a stream across batches |
| Forward | `L1.6` | tokenizer metrics report bytes per token and the bpb of a reference model per tokenizer |
| Forward | `L2.1`, `L2.2`, `L3.6` | each statistical and recurrent model reports its validation perplexity and bpb through an accumulator |
| Forward | `L6.7` | the model zoo scores every family with one accumulator per model and prints bpb |
| Forward | `L8.5`, `C1` | quantization is allowed a fixed bpb increase; the capstone's training curve and ablations are in bpb |

If you skip this module, `ol check L1.6` stops with `L1.6 needs M11.2`: build it, or rerun with `--ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `result()["bpb"]` | EleutherAI lm-evaluation-harness | `bits_per_byte`, `byte_perplexity`, and `word_perplexity` aggregations over documents, weighted by length | `lm_eval/api/metrics.py` |
| `perplexity` over a corpus | llama.cpp `perplexity` | strided windows that score only the second half of each context, so every token has enough history | `tools/perplexity/perplexity.cpp` |
| batch sums with `fsum` | CPython `math.fsum` | Shewchuk's exact partial sums, correctly rounded whatever the order | `Modules/mathmodule.c` (`math_fsum`) |
