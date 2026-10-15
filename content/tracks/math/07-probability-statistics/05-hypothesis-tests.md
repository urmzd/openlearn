<!-- ol:module M07.5 -->
# Hypothesis tests: paired and two-sample permutation, McNemar, Holm

## Overview

| | |
|---|---|
| **Module** | `M07.5` · build · Python · Pass 5 · 3 to 4 h |
| **You build** | `python/tinyllm/prob/tests.py`: `mean_difference`, `paired_permutation_test`, `permutation_test`, `mcnemar`, `holm`, `holm_adjust` |
| **Contract** | [`course/contracts/py/tinyllm/prob/tests.pyi`](../../../../course/contracts/py/tinyllm/prob/tests.pyi) · the generator behind `rng`: [`spec/pcg32.md`](../../../../course/contracts/spec/pcg32.md) |
| **Tests** | `course/tests/M07.5/test_tests.py` (what they check: section 4), golden values from scipy 1.17.1 in `course/fixtures/M07.5/scipy_golden.json` |
| **Needs** | no code from earlier modules · reading: [`M07.4`](04-lln-clt-confidence-intervals-bootstrap.md) sampling distributions and intervals, [`S-M07c`](92-problem-set-c.md) |
| **Used by** | later: `L6.7` compares models in the zoo · `L8.5` and `L8.6` assert "no regression" after quantization and speculative decoding · `C1` and `C2` A/B ablations · `load.02` re-implements the two-sample test in Go |
| **Milestone** | `MS-P5` (the Pass 5 gate) |
| **Optional depth** | Wasserman, *All of Statistics*, ch. 10 (hypothesis testing and p-values); Good, *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (2005), ch. 3; Phipson and Smyth, "Permutation p-values should never be zero" (2010); Holm, "A simple sequentially rejective multiple test procedure" (1979) |

## Key Takeaways

- A **p-value** is the probability, if nothing really changed, of a difference at least as large as the one you saw; a valid test rejects a true null at rate $\alpha$ (`test_paired_type_one_rate`, `test_two_sample_type_one_rate`).
- A **permutation test** builds that "nothing changed" world by relabelling: flip which model each paired score belongs to, or reshuffle two pools; no normality assumption (`test_hand_example_paired_enumeration`, `test_paired_matches_exact_scipy`).
- Count the observed labelling as one more permutation: $p = (1 + \text{count})/(1 + B)$ is never 0 and keeps the type-I promise exactly (`test_add_one_and_degenerate_data`).
- **McNemar** compares two classifiers on the same examples using only the pairs they disagree on: an exact binomial test (`test_hand_example_mcnemar`, `test_mcnemar_matches_scipy`).
- Ten comparisons at 0.05 give a false alarm 40% of the time; **Holm** controls the chance of any false alarm and rejects at least as much as Bonferroni (`test_hand_example_holm`, `test_holm_steps_down_and_stops`).

## How to work this chapter

```bash
ol start M07.5              # stubs tests.py into your repo
ol tests M07.5              # read the test catalog first: rung R0, you write no tests here
ol check M07.5              # exit code is the verdict
ol diff  M07.5              # after passing: your code against the reference
```

---

## 1. Why now

From Pass 5 on, every change you make to a model is a claim: "pre-LN trains better", "this LoRA rank matches full fine-tuning", "int8 quantization costs nothing" (`L8.5`). The evidence is a score on a finite eval set, and two runs of the same model already differ by a few tenths of a point. `M07.4` put an interval on each number; this module answers the comparison question directly. Is the gap between model A and model B on the same 200 prompts larger than the gap chance alone produces? The model zoo (`L6.7`) prints a p-value next to every comparison, the quantization and speculative-decoding modules fail their "no regression" checks on one, and the zoo compares many models at once, which needs a correction or it will "discover" a winner by luck. All of these call the six functions you write here.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $H_0$ | the null hypothesis: "no difference" in a precise form (exchangeable labels) | |
| $T$ | the test statistic, larger meaning more extreme | `float` |
| $t_{\text{obs}}$ | $T$ on the data as observed | `float` |
| $p$ | the p-value, $P(T \ge t_{\text{obs}} \mid H_0)$ | `float` in $[0, 1]$ |
| $\alpha$ | the significance level: reject $H_0$ when $p \le \alpha$ | `float`, often 0.05 |
| $a_i, b_i$ | the scores of models A and B on prompt $i$ (paired) | `float64[n]` |
| $d_i = a_i - b_i$ | paired differences | `float64[n]` |
| $s_i \in \{-1, +1\}$ | a random sign per pair | |
| $B$ | the number of random permutations, `n_perm` | `int` |
| $b_{01}, b_{10}$ | counts of examples where only A is right, and only B is right | `int` |
| $m$, $p_{(k)}$ | the number of tests in a family, and the $k$-th smallest p-value ($k = 0, \ldots, m-1$) | |

### 2.1 Testing as a thought experiment

Suppose B scores 0.8 points above A on 50 prompts. Two stories explain it: B is better, or the prompts happened to favour B. A test makes the second story precise as a null hypothesis $H_0$, picks a statistic $T$ that is large when the data disagree with $H_0$, and asks how often $H_0$ would produce a $T$ at least as large as the observed one. That probability is the **p-value**. A small p-value says the data would be surprising if $H_0$ were true; it is not the probability that $H_0$ is true, and a large one does not prove $H_0$ (the test may just lack data).

### 2.2 What a p-value promises

Reject $H_0$ whenever $p \le \alpha$. If $H_0$ is true, a valid p-value satisfies $P(p \le \alpha) \le \alpha$: a rejection is a false alarm (a **type-I error**) at most a fraction $\alpha$ of the time. Not rejecting a false $H_0$ is a **type-II error**; the probability of rejecting when the effect is real is the **power**. A test that never rejects has a perfect type-I rate and no power, so the tests check both. "Different" is usually **two-sided**: B worse is as interesting as B better, so $T$ is a magnitude, $|\cdot|$, and swapping A and B must not change $p$.

### 2.3 The paired test

When both models answer the same prompts, compare them prompt by prompt: $d_i = a_i - b_i$. Under $H_0$ ("which model produced which score of a pair is arbitrary"), each $d_i$ was as likely to come out as $-d_i$. So the null world is: flip each sign with probability $\frac{1}{2}$. With $T(s) = |\sum_i s_i d_i|$ and $t_{\text{obs}} = T(1, \ldots, 1)$, the exact p-value is the fraction of all $2^n$ sign vectors with $T(s) \ge t_{\text{obs}}$. For $n = 50$ that is $10^{15}$ vectors, so draw $B$ of them at random. The natural estimate $\text{count}/B$ can be 0, which no true p-value is. Counting the observed labelling as one more draw,

$$p = \frac{1 + \#\{b : T(s^{(b)}) \ge t_{\text{obs}}\}}{1 + B},$$

fixes that, and it is exactly valid: under $H_0$ the observed $T$ is one of $B + 1$ exchangeable values, its rank is uniform, so $P(p \le \alpha) = \lfloor \alpha (B + 1) \rfloor / (B + 1) \le \alpha$. With $B = 99$, $P(p \le 0.1)$ is exactly 0.1: the rate the type-I tests measure. Pairing matters: prompt difficulty varies far more than the model gap, and pairing cancels it.

### 2.4 The two-sample test

When the two samples are not paired (latencies of two server builds, scores on different eval sets), $H_0$ says the group labels are arbitrary: all $N = n_a + n_b$ values come from one distribution. The null world deals the pooled values into two groups of the original sizes at random, and $T = |\text{stat}(a', b')|$, by default the difference of means. A uniformly random order of $N$ items comes from **Fisher-Yates**: for $i = N - 1$ down to 1, pick $j$ uniformly in $\{0, \ldots, i\}$ and swap positions $i$ and $j$. Each of the $N!$ orders comes out with probability $1/N!$. The tempting shortcut, $j$ uniform over all $N$ positions every time, produces $N^N$ equally likely swap sequences, which is not a multiple of $N!$, so some orders are more likely than others.

### 2.5 McNemar

Two classifiers on the same $n$ examples give a 2 by 2 table: both right, both wrong, only A right ($b_{01}$), only B right ($b_{10}$). The examples where they agree say nothing about which is better. Under $H_0$ each disagreement is a fair coin, so $b_{01} \sim \text{Binomial}(b_{01} + b_{10}, \frac{1}{2})$, and the exact two-sided p-value doubles the smaller tail:

$$p = \min\Bigl(1,\ 2 \sum_{i=0}^{k} \binom{n}{i} 2^{-n}\Bigr), \qquad n = b_{01} + b_{10},\ k = \min(b_{01}, b_{10}).$$

The doubling can exceed 1 when $b_{01} = b_{10}$, hence the cap. For large $n$ the classic approximation is $\chi^2 = (|b_{01} - b_{10}| - 1)^2 / n$ on one degree of freedom (the $-1$ is Edwards' continuity correction), with $p = \operatorname{erfc}(\sqrt{\chi^2/2})$. With no disagreements at all there is no evidence: $p = 1$.

### 2.6 Many comparisons

Test $m = 10$ true null hypotheses at $\alpha = 0.05$ and the chance of at least one false alarm is $1 - 0.95^{10} = 0.40$. The **family-wise error rate** (FWER) is that chance. **Bonferroni** tests each at $\alpha/m$, which bounds the FWER by $\alpha$ (a union bound) but is needlessly strict. **Holm** is uniformly better: sort the p-values, compare the smallest with $\alpha/m$, the next with $\alpha/(m-1)$, and so on, rejecting while each passes and **stopping at the first that does not**. Once a hypothesis is rejected, only $m - 1$ can still be true nulls, which is why the bar relaxes. The **adjusted p-values** $\tilde p_{(k)} = \max_{j \le k} \min(1, (m - j) p_{(j)})$ make Holm a table column: reject exactly where $\tilde p \le \alpha$. The running maximum keeps the order: a smaller raw p-value never gets a larger adjusted one.

## 3. Worked example by hand

**Paired.** Model B minus model A on three prompts: $d = (2, 1, 3)$, $t_{\text{obs}} = 6$. All eight sign patterns:

| $s$ | $+{+}{+}$ | $+{+}{-}$ | $+{-}{+}$ | $+{-}{-}$ | $-{+}{+}$ | $-{+}{-}$ | $-{-}{+}$ | $-{-}{-}$ |
|---|---|---|---|---|---|---|---|---|
| $\sum s_i d_i$ | 6 | 0 | 4 | $-2$ | 2 | $-4$ | 0 | $-6$ |
| $\lvert \cdot \rvert \ge 6$ | yes | | | | | | | yes |

The exact p-value is $2/8 = 1/4$: with three prompts, even a unanimous win is not convincing. Fed exactly these eight patterns as its $B = 8$ permutations, the add-one estimate is $(1 + 2)/(1 + 8) = 1/3$.

**McNemar.** On 40 prompts, A alone is right on $b_{01} = 1$, B alone on $b_{10} = 7$. Then $n = 8$, $k = 1$, the tail is $\binom{8}{0} + \binom{8}{1} = 9$ out of $2^8 = 256$, and $p = 2 \cdot 9/256 = 9/128 = 0.0703$. Not significant at 0.05, despite 7 to 1. The approximation gives $\chi^2 = (6 - 1)^2/8 = 3.125$ and $p = \operatorname{erfc}(1.25) = 0.0771$.

**Holm.** Four comparisons with p-values $(0.01, 0.04, 0.02, 0.005)$ at $\alpha = 0.05$:

| sorted $k$ | $p_{(k)}$ | bar $\alpha/(m-k)$ | reject? | $(m-k)\,p_{(k)}$ | adjusted |
|---|---|---|---|---|---|
| 0 | 0.005 | 0.0125 | yes | 0.02 | 0.02 |
| 1 | 0.01 | 0.0167 | yes | 0.03 | 0.03 |
| 2 | 0.02 | 0.025 | yes | 0.04 | 0.04 |
| 3 | 0.04 | 0.05 | yes | 0.04 | 0.04 |

Holm rejects all four; Bonferroni (every p against 0.0125) rejects only 0.005 and 0.01. In input order the adjusted values are $(0.03, 0.04, 0.04, 0.02)$. These are the first three tests.

## 4. The interface

```python
def mean_difference(a: NDArray, b: NDArray) -> float: ...
def paired_permutation_test(a: ArrayLike, b: ArrayLike, n_perm: int, rng: UniformSource) -> float: ...
def permutation_test(a: ArrayLike, b: ArrayLike, stat: Callable[[NDArray, NDArray], float],
                     n_perm: int, rng: UniformSource) -> float: ...
def mcnemar(b01: int, b10: int, exact: bool = True) -> float: ...
def holm(pvalues: ArrayLike, alpha: float = 0.05) -> NDArray: ...      # bool, True = rejected
def holm_adjust(pvalues: ArrayLike) -> NDArray: ...
```

The draw order is part of the contract, because `load.02` re-implements the two-sample test in Go and must reproduce your p-values from the same seed: the paired test draws one uniform per pair per permutation ($u < 0.5$ flips); the two-sample test restarts from the identity order each permutation and runs Fisher-Yates from the end with $j = \min(\lfloor u(i+1) \rfloor, i)$. A permuted statistic counts as "at least as extreme" when it is at least $t_{\text{obs}}(1 - 10^{-9})$, so rounding never splits a tie.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_paired_enumeration` | unit, smoke | section 3: $(1 + 2)/(1 + 8) = 1/3$ over the eight patterns | you and the tests agree on the statistic and the add-one rule |
| `test_hand_example_mcnemar` | unit, smoke | $9/128$ exact, $\operatorname{erfc}(1.25)$ approximate | classifier comparisons in `L6.7` |
| `test_hand_example_holm` | unit, smoke | all four rejected, adjusted $(0.03, 0.04, 0.04, 0.02)$ | the zoo's comparison table |
| `test_paired_matches_exact_scipy` | golden | Monte Carlo within 4.5 standard errors of scipy's exact enumeration | the estimate converges to the right number |
| `test_two_sample_matches_exact_scipy` | golden | the same for 6 + 6, 7 + 5, 8 + 8 splits | latency and eval comparisons |
| `test_replay_pins_the_draw_order` | golden | exact p-values for fixed seeds | `load.02` in Go reproduces them |
| `test_mcnemar_matches_scipy` | golden | `binomtest` and the chi-square, up to 360 pairs | exact arithmetic, no overflow |
| `test_holm_matches_reference` | golden | adjusted p and rejections on five families | ties, 0 and 1, a single test |
| `test_paired_type_one_rate` | statistical | $P(p \le 0.1) = 0.1$ under $H_0$ over 300 datasets | false alarms at the promised rate |
| `test_two_sample_type_one_rate` | statistical | the same with groups of 6 and 9 | the shuffle really mixes the groups |
| `test_tests_have_power` | statistical | real shifts are detected at 0.05 | a test that never rejects is useless |
| `test_draw_counts` | unit | $B n$ and $B(N - 1)$ uniforms | one stream in every language |
| `test_fisher_yates_order` | unit | $u \to 1$ is the identity, $u = 0$ deals $(5, 1 \mid 2, 3)$ | the shuffle of section 2.4 |
| `test_two_sided` | property | swapping A and B, or negating the statistic, leaves $p$ unchanged | "different", not "better" |
| `test_add_one_and_degenerate_data` | boundary | $p > 0$ always; no differences give $p = 1$ | no "p = 0" in a report |
| `test_rounding_never_splits_a_tie` | boundary | $d = (-0.7, 0.2, 0.7, 0.1)$ counts 12 of 16 ties | decimal scores tie in exact arithmetic |
| `test_holm_steps_down_and_stops` | property | stops at the first failure; equality rejects; superset of Bonferroni | section 2.6 |
| `test_holm_adjust_is_monotone_and_capped` | property | adjusted p sorted like raw p, at most 1 | a table column you can trust |
| `test_mcnemar_edges` | boundary | no disagreements or a tie give 1; symmetric | small eval sets |
| `test_rejects_bad_arguments` | boundary | bad lengths, NaN, `n_perm < 1`, bad counts, p outside $[0, 1]$ raise | bugs surface at the call |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. a one-sided comparison by accident (no $\lvert \cdot \rvert$) | $p$ near 1 when B is worse, too small when better | `test_two_sided` (mutant `s01`) |
| 2. $\text{count}/B$ without the add-one | "p = 0.000" in a report; type-I promise broken | `test_add_one_and_degenerate_data` (mutant `s02`) |
| 3. the naive shuffle ($j$ over the whole array) | some orders more likely than others; another stream than Go | `test_fisher_yates_order`, `test_replay_pins_the_draw_order` (mutant `s03`) |
| 4. shuffling on from the previous permutation | still a valid test, but not the contract's stream | `test_fisher_yates_order` (mutant `s04`) |
| 5. comparing permuted statistics with a bare $\ge$ | exact ties land an ulp below and are missed: $p$ too small | `test_rounding_never_splits_a_tie` (mutant `s05`) |
| 6. McNemar without doubling, without the cap, or without the continuity correction | $p$ halved, above 1, or too small | `test_hand_example_mcnemar`, `test_mcnemar_edges` (mutants `s06`, `s07`, `s12`) |
| 7. Bonferroni where Holm was meant | real improvements missed | `test_hand_example_holm` (mutant `s08`) |
| 8. Holm that does not stop at the first failure | a later hypothesis rejected past a failed bar: FWER broken | `test_holm_steps_down_and_stops` (mutant `s09`) |
| 9. adjusted p-values without the running maximum | a smaller raw p gets a larger adjusted one | `test_holm_adjust_is_monotone_and_capped` (mutant `s10`) |
| 10. a relabelling that never mixes the groups | every permutation equals the data: $p = 1$, no power | `test_two_sample_type_one_rate`, `test_tests_have_power` (mutant `s11`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M07.4` | the sampling distribution and the standard error; an interval and a test are two readings of it (reading) |
| Back | `S-M07c` | the CLT behind the chi-square approximation (reading) |
| Forward | `S-M07d` | q1 to q4 are these tests by hand |
| Forward | `L6.7` | the zoo prints a paired p-value for each comparison against the baseline, Holm-adjusted over the table |
| Forward | `L8.5` | "quantization costs nothing": a paired test on per-prompt losses must not reject |
| Forward | `L8.6` | speculative decoding must not change quality: McNemar on exact-match outcomes |
| Forward | `load.02` | the two-sample test in Go, for p50 and p99 latency comparisons, bit-identical to yours |
| Forward | `C1` | ablation verdicts over seeds |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `permutation_test` | `scipy.stats.permutation_test` | vectorized batches, exact enumeration when it is small enough, one- and two-sided alternatives | `scipy/stats/_resampling.py` |
| `mcnemar` | `statsmodels.stats.contingency_tables.mcnemar` | the same two variants, plus Cochran's Q for more than two classifiers | `statsmodels/stats/contingency_tables.py` |
| `holm` | `statsmodels.stats.multitest.multipletests` | Hochberg, Benjamini-Hochberg (false discovery rate), and others | `statsmodels/stats/multitest.py` |
| paired tests on eval scores | lm-evaluation-harness, HELM | bootstrap standard errors per task; paired comparisons across models | `lm_eval/api/metrics.py` (`stderr_for_metric`) |
| sign flips | sequential testing (always-valid p-values) | stop an A/B test early without inflating false alarms | Johari et al., "Peeking at A/B Tests" (2017) |
