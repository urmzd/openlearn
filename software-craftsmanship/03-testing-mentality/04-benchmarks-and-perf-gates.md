<!-- ol:module craft.06 -->
# Benchmarks and perf gates (R7)

## Overview

| | |
|---|---|
| **Module** | `craft.06` · practice · C and Python · Pass 6 · 3 to 4 h |
| **You build** | `primers/craft.06/kernels.c` (a kata: four C kernels whose speed depends on how they are written), `primers/craft.06/bench.c` (your benchmark of them with `ol_bench.h`), `primers/craft.06/budget.toml` (your performance budget), and `primers/craft.06/gate.py` (your regression gate) |
| **Contract** | the kata's spec in `primers/craft.06/kernels.h` (`ol check craft.06` writes it, and the kata with stubs, on its first run) and the timer [`ol_bench.h`](../../course/contracts/c/include/ol_bench.h) |
| **Tests** | `course/tests/craft.06/`: the grade of your gate by ten planted slowdowns (section 4), with `kata_check.c` for your kata's correctness |
| **Needs** | reading: [`craft.03`](01-tdd-unit-tests-and-mutation-grading.md) how tests are graded · [`craft.05`](03-oracles-golden-and-gradcheck.md) oracles: a fast kernel must first be a right one · [`L9.1`](../../ml/08-tinyllm/p09-kernels/04-tiled-batch-invariant-matmul.md) tiling and loop order · [`L9.6`](../../ml/08-tinyllm/p09-kernels/09-elementwise-kernels.md) RMSNorm |
| **Used by** | no call site (a practice): rung R7 applies the same gate to your kernels (`L9`) and engine (`L10`), and `ops.07` bisects a perf regression with it |
| **Milestone** | `MS-L9` (requires a passing `craft.06`) |
| **Optional depth** | Brendan Gregg, *Systems Performance*, 2nd ed., ch. 12 "Benchmarking"; Georges, Buytaert, Eeckhout, "Statistically Rigorous Java Performance Evaluation" (OOPSLA 2007); Chen and Revels, "Robust benchmarking in noisy environments" (2016) |

## Key Takeaways

- A benchmark is a **measurement with a stated error**: fixed inputs, a warm-up, repetitions, and the **minimum** of the repetitions, because noise only ever makes a run slower.
- A perf gate compares **base and head on the same machine, alternately, within seconds**, so it needs no stored numbers and no calibration, and a slow laptop and a fast CI runner give the same verdict.
- The **budget** is a fraction: head may be at most 30% slower than base. It must sit above the noise (`test_your_gate_accepts_the_reference`) and well below a real regression (`test_perf_faults_trip_your_gate`).
- The slowdowns that matter keep the answer right: loop order, tiles too small to vectorize, one accumulator, duplicated work, a linear search, a defensive copy. Only a timing test catches them.
- The grade: your gate fails at least 9 of 10 planted slowdowns, always the four marked required, and never fails the course's kata against itself.

## How to work this chapter

```bash
ol check craft.06           # first run: writes kernels.h and a stubbed kernels.c, then fails
# write bench.c, budget.toml, gate.py (section 4), then the kata; try your gate yourself:
python3 primers/craft.06/gate.py --base primers/craft.06/kernels.c --head primers/craft.06/kernels.c \
  --include .ol/openlearn/course/contracts/c/include
ol check craft.06           # grades your gate against the planted slowdowns
```

Run the check on an otherwise idle machine: it times things. Every build and benchmark it starts runs in its own process group with a timeout. The planted slowdowns are shown by their one-line description only.

---

## 1. Why now

Your kernels are correct: `L9.1` to `L9.6` matched their oracles, and `L9.7` runs a real model through them. Correct is not the same as fast, and fast does not stay fast. Swap two loops in `matmul.c` while fixing an edge case and the answer is unchanged, every test passes, and decode is 14 times slower. No unit test can notice, because a unit test checks values. From this pass on, rung R7 asks you to protect speed the way rungs R0 to R5 protect values: a benchmark that measures it, a budget that says how much it may move, and a gate that fails the change that breaks it. This chapter does it on a small kata first, so the noise, the budgets, and the failure modes are all in front of you before your own kernels are on the line.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $t_i$ | the time of repetition $i$, per call | seconds |
| $\hat t = \min_i t_i$ | the benchmark's estimate | seconds |
| $t_b, t_h$ | $\hat t$ for the base and the head kernels.c | seconds |
| $\Delta = t_h / t_b - 1$ | the change; $+1.0$ is twice as slow | fraction |
| $\beta$ | the budget of a metric: the largest $\Delta$ allowed | fraction in $(0, 1)$ |

### 2.1 What a benchmark measures

A call's time is the work plus noise: other processes, frequency scaling, cache and TLB state, interrupts. Noise is one-sided: it can only add time. So `ol_bench_best(fn, ctx, reps, min_s)` calls `fn` once to warm caches and branch predictors, then runs `reps` repetitions, each calling `fn` as often as it takes to fill at least `min_s` seconds (a 5 µs call is timed over thousands of calls, not one), and returns the **smallest** per-call time. The mean would absorb every hiccup; the minimum is the closest any run came to the work alone.

Three rules make a benchmark mean something:

- **Fixed inputs.** A formula of the index, never `rand()`: the same work on every run and every machine.
- **Keep the result.** Write it to a `volatile` variable. An optimizing compiler deletes a computation whose result is never used, and you time an empty loop.
- **Sizes that matter.** Big enough to take microseconds to milliseconds (timer resolution and call overhead vanish), small enough that a 1000x slower version still finishes in a second.

The benchmark prints its metrics as the last stdout line, one JSON object (`ol_bench_done`), so any tool can read them.

### 2.2 A gate compares, it does not remember

The obvious gate stores yesterday's number and fails when today's is slower. It breaks on the first machine change: a laptop on battery, a CI runner of a different generation, a busy neighbor. The robust gate measures **base and head together**: build the benchmark with each version, run them alternately (base, head, base, head, ...) a few times, keep each side's minimum, and compare. Drift over the run (a thermal throttle, a background job) hits both sides alike, and the machine's absolute speed cancels in $\Delta$. This is how `{loadgen} compare` (Pass 7) and production systems such as Go's `benchstat` work.

Absolute budgets still have a place: `ol bench <ID>` checks a module against `ol bench --calibrate`, a reference kernel timed on the same machine, for "at least half the calibrated GFLOP/s". That answers "is it fast enough?"; the gate answers "did this change make it slower?".

### 2.3 The budget

A metric regresses when $t_h > t_b (1 + \beta)$. Choosing $\beta$ is a trade between two errors:

| | Too small (5%) | Too large (100%) |
|---|---|---|
| Noise ($\lvert \Delta \rvert$ up to about 10% between identical builds) | false alarms: the gate is switched off within a week | fine |
| A 2x slowdown ($\Delta = +1.0$) | caught | missed: $+1.0$ is not more than $1.0$ |

$\beta = 0.3$ sits well above the noise and far below a doubling. A metric measured over microseconds (the binary search) is noisier and gets more room. The budget lives in a file next to the benchmark, reviewed like code: loosening it is a decision someone signs.

### 2.4 Slowdowns that keep the answer right

Each of these leaves every output the same and costs 2x to 1000x:

| Change | Why it is slow |
|---|---|
| matmul loop order $i, j, k$ | the inner loop walks $B$ down a column (stride $n$) and is a dot product the compiler may not vectorize (float addition is not associative, so it may not reorder the sum) |
| tiles 2 columns wide | the unit-stride inner loop is too short to vectorize and its overhead dominates |
| one accumulator in a sum | each add waits for the previous one (a 3 to 4 cycle dependency chain); eight independent partial sums keep the adder busy and vectorize |
| `volatile` partial sums | every add is a load and a store |
| a mean square recomputed per element | $O(d^2)$ instead of $O(d)$ per row |
| the work done twice | exactly 2x: the smallest regression your budget must catch |
| a linear search for a binary one | $O(n)$ instead of $O(\log n)$ |
| a defensive copy | an $O(n)$ `malloc` and `memcpy` around an $O(\log n)$ search |

"Tiling disabled" (the whole matrix as one tile) is the textbook regression, and on a machine whose cache holds the matrices it costs only 1.4x at $n = 256$. A perf fault is only a fault if it is slow on the machine that runs the gate; measure before you plant one.

## 3. Worked example by hand

A gate with $\beta = 0.25$ on one metric:

| Base $\hat t$ | Head $\hat t$ | $\Delta = t_h / t_b - 1$ | $t_h > t_b (1 + \beta)$? | Verdict |
|---|---|---|---|---|
| 8.0 ms | 9.6 ms | $9.6 / 8.0 - 1 = +0.20$ | $9.6 > 10.0$: no | ok (noise or a small cost) |
| 8.0 ms | 16.0 ms | $+1.00$ | $16.0 > 10.0$: yes | regressed (twice the work) |
| 8.0 ms | 10.0 ms | $+0.25$ | $10.0 > 10.0$: no | ok: the bound is inclusive |
| 8.0 ms | 10.01 ms | $+0.25125$ | yes | regressed |

And why the minimum: five repetitions of one build give 8.0, 8.1, 12.7 (a background job), 8.0, 8.3 ms. The mean is 9.02 ms, 13% above the work; the minimum is 8.0 ms. Compare a head whose five runs give 8.2, 8.1, 8.4, 8.2, 8.3 ms: by means, $8.24 / 9.02 - 1 = -9\%$, so the noisy base made the head look faster; by minima, $8.1 / 8.0 - 1 = +1.3\%$.

The first four rows are `test_hand_example_regression_arithmetic`, on your gate's `regressed`.

## 4. The artifact and its check

**The kata**, `primers/craft.06/kernels.c` (spec in `kernels.h`, given), float32, plain C11:

```c
void  kata_matmul(const float *A, const float *B, float *C, int n);   /* C = A @ B, n x n: loop order i, k, j, tiled over k and j */
float kata_sum(const float *x, int n);                                  /* eight partial sums */
void  kata_rmsnorm(const float *x, const float *w, float *y, int rows, int d, float eps);
int   kata_count_below(const float *sorted, int n, float t);           /* binary search: entries < t */
```

**Your benchmark**, `primers/craft.06/bench.c`: includes `"kernels.h"` and `"ol_bench.h"`, times every kata kernel with `ol_bench_best`, and ends with `ol_bench_done()`, so its last line is `{"matmul_ns": ..., "sum_ns": ..., "rmsnorm_ns": ..., "count_below_ns": ...}` (your metric names; lower is better). Fixed inputs only.

**Your budget**, `primers/craft.06/budget.toml`:

```toml
runs = 3            # alternating runs per side; each side keeps its best
[budget]            # one fraction in (0, 1) per metric of bench.c
matmul_ns = 0.30
sum_ns = 0.30
rmsnorm_ns = 0.30
count_below_ns = 0.50
```

**Your gate**, `primers/craft.06/gate.py` (Python standard library only):

```
python3 gate.py --base BASE.c --head HEAD.c [--include DIR] [--runs N]
```

It builds `bench.c` (next to `gate.py`) against each `kernels.c` with `cc -std=c11 -O2` and `-I` for the gate's own directory (`kernels.h`) and each `--include` (`ol_bench.h`), runs base and head alternately `runs` times, keeps each side's minimum per metric, and prints a table and a final JSON line. It defines `regressed(base, head, max_regression) -> bool` (head more than `max_regression` slower). Exit 0 when nothing regresses, 1 when something does, 2 when it cannot decide (a build failure, a crash, a metric missing). Start every build and benchmark in its own process group with a timeout, so a hung benchmark cannot hang your CI.

**The check** (`ol check craft.06`, `course/tests/craft.06/check`) runs six tests:

| Test | KIND | Checks |
|---|---|---|
| `test_hand_example_regression_arithmetic` | unit | section 3 on your `regressed` |
| `test_your_artifacts_are_a_benchmark_and_a_budget` | unit | `bench.c` uses `ol_bench.h`, times all four kernels, no `rand()`; `budget.toml` has `runs` of at least 2 and a fraction in $(0, 1)$ per kernel; `gate.py` defines `regressed` |
| `test_your_kata_is_correct` | golden | `kata_check.c` compares your kata with the plain kernels on awkward sizes |
| `test_your_gate_accepts_the_reference` | conformance | the course's kata as base and head: exit 0, twice in a row |
| `test_your_kata_keeps_pace` | unit | your kata as head against the course's as base: exit 0 |
| `test_perf_faults_trip_your_gate` | fault | the course's kata with each of 10 planted slowdowns as head: exit 1 on at least 9, always on `s01`, `s02`, `s04`, `s09` |

Each run copies the two `kernels.c` versions to a scratch directory; your tree is never touched.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. the matmul loop order $i, j, k$ | 14x slower, same values | `test_perf_faults_trip_your_gate` (mutant `s01`) |
| 2. tiles too narrow to vectorize | 24x slower, same values | `test_perf_faults_trip_your_gate` (mutant `s02`) |
| 3. the work done twice | exactly 2x: a budget of 1.0 or more lets it through | `test_perf_faults_trip_your_gate` (mutants `s03`, `s08`) |
| 4. one accumulator | 4x to 7x slower sums | `test_perf_faults_trip_your_gate` (mutants `s04`, `s06`) |
| 5. `volatile` where it is not needed | every add goes through memory | `test_perf_faults_trip_your_gate` (mutant `s05`) |
| 6. a quantity recomputed inside the loop that uses it | $O(d^2)$ | `test_perf_faults_trip_your_gate` (mutant `s07`) |
| 7. a linear search | 800x slower at $n = 2^{18}$ | `test_perf_faults_trip_your_gate` (mutant `s09`) |
| 8. a defensive copy | an $O(n)$ copy per $O(\log n)$ query | `test_perf_faults_trip_your_gate` (mutant `s10`) |
| a benchmark result nobody reads | the compiler deletes the work; every version is "fast" | `test_perf_faults_trip_your_gate` (every fault survives) |
| the mean of noisy runs | one background job fails a correct change | `test_your_gate_accepts_the_reference` |
| a budget below the noise | the gate fails the course's kata against itself | `test_your_gate_accepts_the_reference` |
| comparing with a number stored on another machine | the gate passes or fails by machine, not by change | `test_your_gate_accepts_the_reference` (base and head are built here, now) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `craft.03` | grading by planted faults, baselines A and B |
| Back | `craft.05` | a fast kernel is checked for values first (`kata_check.c`) |
| Back | `L9.1` | loop order and tiling, the kata's matmul |
| Forward | `L9.7` | rung R7: your backend's tests are graded at 0.90 |
| Forward | `L10.1` | criterion benchmarks of the Rust runner behind the same kind of gate |
| Forward | `ops.07` | a perf-regression drill: a seeded slowdown in a scratch copy of your repo, found by bisecting with your gate |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `ol_bench_best` | Google Benchmark, criterion.rs, Go `testing.B` | automatic iteration counts, statistics (confidence intervals, outlier classes) | criterion.rs user guide, "Analysis Process" |
| `gate.py` base-versus-head | `benchstat`, Bencher, CodSpeed | significance tests over many runs, history dashboards | `golang.org/x/perf/cmd/benchstat` |
| fixed-machine noise control | instruction counting with `perf stat`, Cachegrind | counts instead of times: deterministic, but blind to cache effects | CodSpeed's instrumentation mode; Valgrind Cachegrind manual |
| planted slowdowns | performance mutation testing (Delgado-Pérez et al., 2021), regression bisection | mutation operators designed to cost time, not change values; bisection finds which commit made it slow | `git bisect run` with your gate as the test |
