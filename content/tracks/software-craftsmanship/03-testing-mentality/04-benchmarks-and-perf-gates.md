# Benchmarks and perf gates (R7)

## Overview

| | |
|---|---|
| **Module** | `craft.06` · practice · C and Python · Pass 6 · 2 h reading |
| **Solution** | [`kernels.c`](../../../../course/solutions/primers/craft.06/kernels.c) (a kata: four C kernels whose speed depends on how they are written: `kata_matmul`, `kata_sum`, `kata_rmsnorm`, `kata_count_below`), [`bench.c`](../../../../course/solutions/primers/craft.06/bench.c) (the benchmark of them with `ol_bench.h`), [`budget.toml`](../../../../course/solutions/primers/craft.06/budget.toml) (the performance budget), and [`gate.py`](../../../../course/solutions/primers/craft.06/gate.py) (the regression gate: `regressed`, `run`, `build`, `metrics`, `main`) |
| **Contract** | the kata's spec in [`kernels.h`](../../../../course/solutions/primers/craft.06/kernels.h) and the timer [`ol_bench.h`](../../../../course/contracts/c/include/ol_bench.h) |
| **Needs** | reading: [`craft.03`](01-tdd-unit-tests-and-mutation-grading.md) grading tests by planted faults · [`craft.05`](03-oracles-golden-and-gradcheck.md) oracles: a fast kernel must first be a right one · [`L9.1`](../../../courses/build-an-llm/p09-kernels/04-tiled-batch-invariant-matmul.md) tiling and loop order · [`L9.6`](../../../courses/build-an-llm/p09-kernels/09-elementwise-kernels.md) RMSNorm |
| **Used by** | no call site (a practice): rung R7 applies the same gate to the kernels (`L9`) and engine (`L10`), and `ops.07` bisects a perf regression with it. Pass 6 ends with the kernels both correct and protected for speed. |
| **Optional depth** | Brendan Gregg, *Systems Performance*, 2nd ed., ch. 12 "Benchmarking"; Georges, Buytaert, Eeckhout, "Statistically Rigorous Java Performance Evaluation" (OOPSLA 2007); Chen and Revels, "Robust benchmarking in noisy environments" (2016) |

## Key Takeaways

- A benchmark is a **measurement with a stated error**: fixed inputs, a warm-up, repetitions, and the **minimum** of the repetitions, because noise only ever makes a run slower.
- A perf gate compares **base and head on the same machine, alternately, within seconds**, so it needs no stored numbers and no calibration, and a slow laptop and a fast CI runner give the same verdict.
- The **budget** is a fraction: head may be at most 30% slower than base. It must sit above the noise (the gate must accept a kata compared with itself) and well below a real regression (it must reject a 2x slowdown).
- The slowdowns that matter keep the answer right: loop order, tiles too small to vectorize, one accumulator, duplicated work, a linear search, a defensive copy. Only a timing test catches them.
- A good gate fails at least 9 of 10 such planted slowdowns, always the gross ones (loop order, narrow tiles, one accumulator, linear search), and never fails a kata against itself.

## How to read this chapter

Sections 1 to 3 build the idea of a benchmark and a gate from first principles and work the arithmetic by hand. Section 4 fixes the artifacts and their interfaces. Section 5 walks through the solution, kernel by kernel and line by line through the gate, and explains why each choice is the right one. Try to predict each step before reading it. The gate is run as:

```bash
python3 course/solutions/primers/craft.06/gate.py \
  --base course/solutions/primers/craft.06/kernels.c --head course/solutions/primers/craft.06/kernels.c \
  --include course/contracts/c/include
```

Run it on an otherwise idle machine: it times things. Every build and benchmark it starts runs in its own process group with a timeout.

---

## 1. Why now

The kernels are correct: `L9.1` to `L9.6` matched their oracles, and `L9.7` runs a real model through them. Correct is not the same as fast, and fast does not stay fast. Swap two loops in `matmul.c` while fixing an edge case and the answer is unchanged, every test passes, and decode is 14 times slower. No unit test can notice, because a unit test checks values. From this pass on, rung R7 protects speed the way rungs R0 to R5 protect values: a benchmark that measures it, a budget that says how much it may move, and a gate that fails the change that breaks it. This chapter does it on a small kata first, so the noise, the budgets, and the failure modes are all in front of you before the real kernels are on the line.

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

Absolute budgets still have a place: "at least half the GFLOP/s of a reference kernel timed on the same machine" answers "is it fast enough?"; the gate answers "did this change make it slower?".

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
| the work done twice | exactly 2x: the smallest regression the budget must catch |
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

The gate's `regressed` must reproduce the first four rows exactly.

## 4. The artifacts

**The kata**, `primers/craft.06/kernels.c` (spec in `kernels.h`), float32, plain C11:

```c
void  kata_matmul(const float *A, const float *B, float *C, int n);   /* C = A @ B, n x n: loop order i, k, j, tiled over k and j */
float kata_sum(const float *x, int n);                                  /* eight partial sums */
void  kata_rmsnorm(const float *x, const float *w, float *y, int rows, int d, float eps);
int   kata_count_below(const float *sorted, int n, float t);           /* binary search: entries < t */
```

**The benchmark**, `primers/craft.06/bench.c`: includes `"kernels.h"` and `"ol_bench.h"`, times every kata kernel with `ol_bench_best`, and ends with `ol_bench_done()`, so its last line is `{"matmul_ns": ..., "sum_ns": ..., "rmsnorm_ns": ..., "count_below_ns": ...}` (lower is better). Fixed inputs only.

**The budget**, `primers/craft.06/budget.toml`:

```toml
runs = 3            # alternating runs per side; each side keeps its best
[budget]            # one fraction in (0, 1) per metric of bench.c
matmul_ns = 0.30
sum_ns = 0.30
rmsnorm_ns = 0.30
count_below_ns = 0.50
```

**The gate**, `primers/craft.06/gate.py` (Python standard library only):

```
python3 gate.py --base BASE.c --head HEAD.c [--include DIR] [--runs N]
```

It builds `bench.c` (next to `gate.py`) against each `kernels.c` with `cc -std=c11 -O2` and `-I` for the gate's own directory (`kernels.h`) and each `--include` (`ol_bench.h`), runs base and head alternately `runs` times, keeps each side's minimum per metric, and prints a table and a final JSON line. It defines `regressed(base, head, max_regression) -> bool` (head more than `max_regression` slower). Exit 0 when nothing regresses, 1 when something does, 2 when it cannot decide (a build failure, a crash, a metric missing). Every build and benchmark starts in its own process group with a timeout, so a hung benchmark cannot hang CI.

### What a correct solution must handle

| Case | Kind of check | Why it matters downstream |
|---|---|---|
| the four hand rows of section 3 on `regressed` | unit | the inclusive bound is the contract every later gate inherits |
| `bench.c` uses `ol_bench.h`, times all four kernels, no `rand()`; `budget.toml` has `runs` of at least 2 and a fraction in $(0, 1)$ per kernel | unit | a benchmark with random inputs or a missing metric cannot gate anything |
| the kata agrees with the plain kernels on awkward sizes (not multiples of the tile) | golden | a fast kernel that is wrong is worthless; the tile edges are where it goes wrong |
| the kata as base and head: exit 0, twice in a row | conformance | a gate that fails a change against itself gets switched off |
| a candidate kata as head against the reference as base: exit 0 | unit | the fast version really is fast on this machine |
| ten planted slowdowns as head: exit 1 on at least 9, always on loop order, narrow tiles, one accumulator, linear search | fault | the gate catches what unit tests cannot |

## 5. The solution, step by step

### 5.1 From first principles

1. **A timing is work plus non-negative noise.** So the best estimate of the work is the smallest observation, not the mean (section 3: 8.0 ms versus 9.02 ms). This forces `min` everywhere: in `ol_bench_best` across repetitions and in the gate across alternating runs.
2. **One call can be shorter than the timer's resolution.** So each repetition loops the call until at least `min_s` seconds pass and divides by the count. This forces a per-call metric (nanoseconds per call), which is also what makes base and head comparable regardless of how many calls each repetition made.
3. **The compiler is allowed to delete unused work.** So every result is written to a `volatile` sink. Without it, every version is "fast" and every planted slowdown survives.
4. **Absolute numbers vary by machine; ratios on one machine within seconds do not.** So the gate builds both versions now and alternates them, and compares $t_h / t_b$. That forces the gate to own the build (it must compile both `kernels.c` files against the same `bench.c`) rather than read a stored number.
5. **The budget must separate two distributions:** identical builds ($\lvert\Delta\rvert \lesssim 0.1$) and the smallest real regression ($\Delta = 1.0$, work done twice). Any $\beta$ between them works; $0.3$ leaves room on both sides. The noisiest metric (a microsecond-scale search) gets $0.5$, still far below $1.0$.
6. **A gate that cannot decide must say so.** A build failure or a crash is neither "ok" nor "regressed", so it gets its own exit code (2). Otherwise a broken head would read as "not slower" and pass, or a flaky machine as a regression.
7. **A hung benchmark (an infinite loop in head) must not hang CI.** So each subprocess runs in its own session with a timeout, and the whole process group is killed, which also gets any children.

### 5.2 The code, annotated

**The kernels.** Each fast kernel is fast for one reason, and the planted slowdowns each remove exactly that reason.

```c
/* Tile sizes for kata_matmul: a BK x BJ block of B (64 x 256 floats, 64 KiB)
 * is reused by every row i before the next block is loaded. */
enum { BK = 64, BJ = 256 };

/* c[0..BJ) += a * b[0..BJ): a fixed trip count and restrict (no alias check
 * needed) are what let gcc vectorize at -O2, as clang already does. */
static void axpy_tile(float *restrict c, const float *restrict b, float a) {
    for (int j = 0; j < BJ; j++) c[j] += a * b[j];
}

void kata_matmul(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n * n; i++) C[i] = 0.0f;
    for (int k0 = 0; k0 < n; k0 += BK)
        for (int j0 = 0; j0 < n; j0 += BJ)
            for (int i = 0; i < n; i++)
                for (int k = k0; k < min_i(k0 + BK, n); k++) {
                    float a = A[i * n + k];        /* loop order i, k, j: a is a scalar for the inner loop */
                    if (j0 + BJ <= n) {            /* a whole tile: unit stride over j, vectorized */
                        axpy_tile(C + i * n + j0, B + k * n + j0, a);
                        continue;
                    }
                    for (int j = j0; j < n; j++) C[i * n + j] += a * B[k * n + j];  /* ragged edge */
                }
}
```

The inner loop is an axpy over a row of $B$ and a row of $C$: unit stride, no reduction, so no float reassociation is needed to vectorize it. That is why loop order $i, k, j$ beats $i, j, k$ (whose inner loop is a dot product down a column of $B$). The fixed `BJ` trip count and `restrict` are what make gcc vectorize; shrink the tile to 2 columns and the vector loop disappears. The ragged-edge branch exists only for $n$ not a multiple of 256, which is exactly where the golden check with awkward sizes looks.

```c
float kata_sum(const float *x, int n) {
    float s[8] = {0};                      /* eight independent chains: no add waits on the previous one */
    int i = 0;
    for (; i + 8 <= n; i += 8)
        for (int l = 0; l < 8; l++) s[l] += x[i + l];
    for (; i < n; i++) s[0] += x[i];       /* the tail, fewer than 8 elements */
    return ((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]));  /* a fixed tree: deterministic */
}
```

The compiler may not turn one accumulator into eight, because float addition is not associative; the source has to. The fixed combination tree makes the result the same on every run. `kata_rmsnorm` reuses the same eight-way sum for $\sum x^2$, computes `inv = 1.0f / sqrtf(ss / d + eps)` once per row, and then does one multiply pass: the $O(d^2)$ slowdown is recomputing that mean square inside the output loop.

```c
int kata_count_below(const float *sorted, int n, float t) {
    int lo = 0, hi = n; /* invariant: sorted[0..lo) < t <= sorted[hi..n) */
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;      /* no overflow of lo + hi */
        if (sorted[mid] < t)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;                             /* lo == hi: the count of entries < t */
}
```

$O(\log n)$ and touches no memory but `sorted`; a linear scan is 800x slower at $n = 2^{18}$, and a defensive copy adds an $O(n)$ `malloc` and `memcpy` per query.

Full kata: [`kernels.c`](../../../../course/solutions/primers/craft.06/kernels.c).

**The benchmark.** One program, built against whichever `kernels.c` the gate hands it.

```c
/* ol_bench.h comes first: it defines _POSIX_C_SOURCE for clock_gettime, and
 * glibc ignores that once any other system header is in. */
#include "ol_bench.h"
#include <stdlib.h>
#include "kernels.h"

#define NM 256        /* matmul: 256^3 = 16.8M multiply-adds */
#define NS (1 << 20)  /* sum: 4 MiB of floats */
#define NC (1 << 18)  /* count_below: 1 MiB sorted, 256 queries per call */
#define QC 256
// ...
static volatile float sink_f;            /* results land here, so the work cannot be deleted */
static volatile int sink_i;

static void run_matmul(void *ctx) { (void)ctx; kata_matmul(A, B, C, NM); sink_f = C[NM * NM - 1]; }
static void run_count(void *ctx) {
    (void)ctx;
    int acc = 0;                         /* 256 queries per call: one search alone is too short to time */
    for (int q = 0; q < QC; q++) acc += kata_count_below(SORTED, NC, (float)((q * 7919) % NC) + 0.5f);
    sink_i = acc;
}

int main(void) {
    // ... malloc the buffers; return 2 if any fails
    for (int i = 0; i < NM * NM; i++) {  /* fixed formulas, never rand() */
        A[i] = (float)(i % 7) - 3.0f;
        B[i] = (float)(i % 5) - 2.0f;
    }
    // ... X, W, SORTED likewise
    ol_bench_metric("matmul_ns", 1e9 * ol_bench_best(run_matmul, NULL, 5, 0.02));  /* best of 5, each >= 20 ms */
    // ... sum_ns, rmsnorm_ns, count_below_ns
    return ol_bench_done();              /* last stdout line: one JSON object */
}
```

The include order comment records a real portability bug: under `-std=c11`, a `_POSIX_C_SOURCE` defined after another system header leaves `CLOCK_MONOTONIC` undeclared on Linux. Full benchmark: [`bench.c`](../../../../course/solutions/primers/craft.06/bench.c).

**The gate.** The decision is one line; everything else is making the measurement trustworthy.

```python
def regressed(base: float, head: float, max_regression: float) -> bool:
    """True when head is more than max_regression slower than base."""
    return head > base * (1.0 + max_regression)   # strict >: Delta == budget is ok (section 3, row 3)
```

```python
def run(cmd: list[str], timeout: float = 120) -> tuple[int, str]:
    """One subprocess in its own process group, killed whole on timeout."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True)   # new session = new process group
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)                       # kill the group, not just the child
        out, _ = p.communicate()
        return 124, out + f"\n(timed out after {timeout:.0f} s)"
    return p.returncode, out
```

`build` compiles `bench.c` plus the given `kernels.c` with `cc -std=c11 -O2 -I<gate dir> -I<each include>` (and `-lm` except on macOS) and raises `RuntimeError` on failure. `metrics` runs the executable and parses the last line that starts with `{`, raising if the program exits non-zero or prints no JSON.

```python
def main(argv: list[str] | None = None) -> int:
    # ... argparse: --base, --head, --include (repeatable), --runs
    try:
        runs, budget = load_budget(HERE / "budget.toml")
        runs = a.runs or runs
        best: dict[str, dict[str, float]] = {"base": {}, "head": {}}
        with tempfile.TemporaryDirectory(prefix="gate-") as d:   # never write next to the sources
            exes = {side: Path(d) / f"bench-{side}" for side in best}
            build(a.base.resolve(), a.include, exes["base"])
            build(a.head.resolve(), a.include, exes["head"])
            for _ in range(runs):
                for side in ("base", "head"):  # alternate: drift hits both sides alike
                    for k, v in metrics(exes[side]).items():
                        best[side][k] = min(v, best[side].get(k, float("inf")))  # noise only adds
        missing = [k for k in budget if k not in best["base"] or k not in best["head"]]
        if missing:
            raise RuntimeError(f"the benchmark reports no {missing}")  # a budgeted metric must exist
    except (RuntimeError, ValueError, OSError) as e:
        print(f"gate: {e}", file=sys.stderr)
        return 2                                                  # cannot decide: neither pass nor fail
    change, bad = {}, []
    for k, limit in budget.items():                               # the budget drives the loop, not bench.c
        b, h = best["base"][k], best["head"][k]
        change[k] = round(h / b - 1.0, 4)
        slow = regressed(b, h, limit)
        bad += [k] if slow else []
        # ... print one table row
    print(json.dumps({"ok": not bad, "change": change}))          # machine-readable last line
    return 1 if bad else 0
```

Iterating over the budget rather than over what the benchmark printed means a metric nobody budgeted is ignored, and a budgeted metric the benchmark dropped is an error (exit 2), not a silent pass. Full gate: [`gate.py`](../../../../course/solutions/primers/craft.06/gate.py).

### 5.3 Why this solution

- **Cost.** The gate does two builds and $2 \cdot \text{runs}$ benchmark executions; each execution is $4 \times 5$ repetitions of at least 20 ms, so about 0.4 s plus the warm-ups. Three runs per side is a few seconds: cheap enough for every pull request.
- **Minimum versus mean or median.** The mean absorbs outliers (section 3). The median is robust but still biased upward by sustained noise; with one-sided noise the minimum is the estimator closest to the work. Its weakness is that it hides variance a user would feel (tail latency), which is why serving benchmarks in Pass 7 report percentiles, not minima.
- **Paired comparison versus stored baselines.** A stored baseline needs a pinned machine and recalibration; the paired run needs neither, at the cost of building and running base on every check.
- **Times versus instruction counts.** Counting instructions (Cachegrind) is deterministic but blind to the cache and vectorization effects that most of these slowdowns are about.
- **A significance test.** `benchstat` tests whether the two samples differ. With three runs per side a test has little power; a fixed budget between the noise floor and 2x is simpler and, for these gross regressions, just as reliable. The accepted tradeoff is that slowdowns smaller than 30% pass.

### 5.4 Explain it back

1. Why is the bound `head > base * (1 + budget)` strict, and what would change if it were `>=`?
   <details><summary>Answer</summary>A change of exactly the budget is allowed by definition (section 3, row 3). With `>=`, a head measured at exactly the boundary would fail, which only matters at the boundary, but the contract says "at most 30% slower" and the code should mean that.</details>
2. Why does the gate alternate base, head, base, head instead of running all base runs first?
   <details><summary>Answer</summary>Machine drift (thermal throttling, a background job starting) is correlated in time. Interleaving spreads it over both sides, so it cancels in the ratio; running base first would charge all of a late slowdown to head.</details>
3. What happens if a planted slowdown makes `bench.c` crash or never finish, and why is that the right outcome?
   <details><summary>Answer</summary>A crash raises in `metrics`, a hang is killed by `run`'s timeout (exit 124, then a raise); both make `main` return 2. The gate cannot say "slower" for something it could not measure, and it must not say "ok". CI treats 2 as a failure that needs a human.</details>
4. Why does `kata_sum` need eight partial sums in the source when `-O2` vectorizes integer sums on its own?
   <details><summary>Answer</summary>Float addition is not associative, so the compiler may not reorder a single accumulator's chain into vector lanes (without `-ffast-math`). Writing eight independent sums gives it permission, and a fixed combination tree keeps the result deterministic.</details>
5. Why would a budget of 1.0 miss the "work done twice" slowdown?
   <details><summary>Answer</summary>Doubled work gives $\Delta = +1.0$, and the gate fails only when $\Delta > \beta$. With $\beta = 1.0$, $+1.0$ is not more than $1.0$, so it passes. The budget must be well below 1.0 to catch the smallest real regression.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| 1. the matmul loop order $i, j, k$ | 14x slower, same values | loop order $i, k, j$: the inner loop is a unit-stride axpy (5.2) |
| 2. tiles too narrow to vectorize | 24x slower, same values | `BJ = 256` with a fixed-trip `axpy_tile` and `restrict` |
| 3. the work done twice | exactly 2x: a budget of 1.0 or more lets it through | budgets of 0.30 and 0.50, far below 1.0 (step 5) |
| 4. one accumulator | 4x to 7x slower sums | eight partial sums in `kata_sum` and `kata_rmsnorm` |
| 5. `volatile` where it is not needed | every add goes through memory | `volatile` only on the benchmark's sinks, never in the kernels |
| 6. a quantity recomputed inside the loop that uses it | $O(d^2)$ | `inv` computed once per row before the output loop |
| 7. a linear search | 800x slower at $n = 2^{18}$ | binary search with the `[0, lo) < t <= [hi, n)` invariant |
| 8. a defensive copy | an $O(n)$ copy per $O(\log n)$ query | `kata_count_below` reads `sorted` in place |
| a benchmark result nobody reads | the compiler deletes the work; every version is "fast" | every result goes to `sink_f` or `sink_i` (`volatile`) |
| the mean of noisy runs | one background job fails a correct change | `min` over repetitions and over alternating runs |
| a budget below the noise | the gate fails a kata against itself | 0.30, three times the observed noise |
| comparing with a number stored on another machine | the gate passes or fails by machine, not by change | base and head are built and run here, now, alternately |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `craft.03` | grading tests by planted faults, baselines A and B |
| Back | `craft.05` | a fast kernel is checked for values first (the golden check on awkward sizes) |
| Back | `L9.1` | loop order and tiling, the kata's matmul |
| Forward | `L9.7` | rung R7: the backend's tests must catch perf faults too |
| Forward | `L10.1` | criterion benchmarks of the Rust runner behind the same kind of gate |
| Forward | `ops.07` | a perf-regression drill: a seeded slowdown, found by bisecting with this gate |

## Going further

| This piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `ol_bench_best` | Google Benchmark, criterion.rs, Go `testing.B` | automatic iteration counts, statistics (confidence intervals, outlier classes) | criterion.rs user guide, "Analysis Process" |
| `gate.py` base-versus-head | `benchstat`, Bencher, CodSpeed | significance tests over many runs, history dashboards | `golang.org/x/perf/cmd/benchstat` |
| fixed-machine noise control | instruction counting with `perf stat`, Cachegrind | counts instead of times: deterministic, but blind to cache effects | CodSpeed's instrumentation mode; Valgrind Cachegrind manual |
| planted slowdowns | performance mutation testing (Delgado-Pérez et al., 2021), regression bisection | mutation operators designed to cost time, not change values; bisection finds which commit made it slow | `git bisect run` with the gate as the test |
