# K-Means Optimization

## Overview

- **Problem**: implement k-means from scratch, then improve it in the order a
  reviewer will push on it, naming the cost of each change.
- **Runnable**: [`kmeans_ladder.py`](kmeans_ladder.py) -- three implementations
  behind one contract, standard library only, `python kmeans_ladder.py`.
- **Prerequisites**: Euclidean distance, means, and big-O. No ML background.
- **Estimated time**: 1 day

## Key Takeaways

- **Initialisation dominates.** Lloyd's converges to a *local* optimum, so the
  seeding decides the answer. k-means++ cut inertia by 11x on the demo data --
  more than every constant-factor optimisation combined, by an order of magnitude.
- **Know which lever you are pulling.** `average` reduces the *number* of
  iterations; `best` reduces the *constant factor*. Both live inside the same
  O(iters · n · k · d), and conflating them produces confused optimisation.
- **Optimisations have a regime.** The distance-bound trick is worth 2x at d=20
  and nothing at all at d=2. An optimisation without a stated regime is folklore.
- **A shared contract makes tiers comparable.** One signature, one test suite,
  three implementations. A tier that fails the suite is a regression, not an
  optimisation.

## How to Study

- Write the baseline yourself first, then list its weaknesses before reading the
  `average` section. If your list matches, you already know what a reviewer will
  ask for.
- Run the file. The demo sweeps three shapes on purpose, and the d=2 row where
  `best` ties `average` is the honest part.
- Then explain, out loud, why dropping `sqrt` is safe. If the monotonicity
  argument does not come immediately, that is the gap worth closing.

---

# Concepts & Techniques

## The Problem

Given n points in R^d and an integer k, partition the points into k clusters
minimising **inertia**: the sum of squared distances from each point to the
centroid of its assigned cluster.

Exact k-means is NP-hard, so every practical answer is **Lloyd's algorithm**:

1. **ASSIGN** each point to its nearest centroid.
2. **UPDATE** each centroid to the mean of its assigned points.

Neither step can increase inertia, and there are finitely many partitions, so it
terminates. But it terminates at a *local* optimum -- which is the single fact
that determines where the effort should go.

## The Shared Contract

```python
fit(points, k, seed=...) -> (centroids, labels, inertia)
```

Three implementations, one signature, one test suite run against all of them.
This is what makes the ladder a ladder rather than three separate programs: any
tier that fails the shared contract is a regression wearing an optimisation's
clothes.

## 1. Slow: the baseline worth writing

Random init, `sqrt` in the distance, a fixed 100 iterations, every centroid
rebuilt from scratch each round. `O(iters · n · k · d)` time, `O(n + k·d)` space.

Its weaknesses, each of which a reviewer will reach for:

| Weakness | Consequence |
|----------|-------------|
| Random init | Can drop two centroids inside one true cluster and converge badly |
| `sqrt` in the metric | Pure waste: it cannot change an argmin |
| Fixed iteration count | Either burns work after settling, or stops before settling |
| Full centroid rebuild | O(n·d) every iteration regardless of how little changed |
| Silent empty clusters | An emptied cluster keeps a stale centroid forever |

It is correct, and everything after it is judged against it. Writing it is not a
formality: it is the only way the later tiers have something to be compared to.

## 2. Average: the four fixes

**k-means++ initialisation** (Arthur & Vassilvitskii, 2007) is the highest-value
change in the entire study. The first centroid is uniform at random; each
subsequent one is a data point drawn with probability proportional to D(p)², its
squared distance to the nearest already-chosen centroid.

```python
d2 = [min(_dist2(p, c) for c in centroids) for p in points]
r = rng.random() * sum(d2)          # inverse-CDF walk
cumulative = 0.0
for p, w in zip(points, d2):
    cumulative += w
    if cumulative >= r:
        centroids.append(list(p))
        break
```

Far regions are likely to receive a centroid, and a duplicate pick has D² = 0 so
it is never re-chosen. Expected O(log k)-competitive with the optimal clustering,
and in practice it slashes the iteration count too.

**Drop the sqrt.** `sqrt` is monotonic, so comparing squared distances gives the
same argmin. Free, and it removes a transcendental call from the innermost loop.

**A real stopping rule.** Assignments are the discrete state of Lloyd's: if none
changed, the update step is a no-op and the algorithm has converged. Stopping on
that is both faster and more correct than a fixed budget.

**Explicit empty-cluster repair.** Reseed a dead centroid at the point
contributing the most inertia. One extra O(n·d) scan, only on the rare empty
event, in exchange for removing a silent-wrong-answer failure mode.

## 3. Best: constant-factor engineering

**Incremental centroid maintenance.** Keep per-cluster running sums and counts.
A point that changes cluster patches two sums in O(d):

```python
def move(i, dst):
    src = labels[i]
    labels[i] = dst
    counts[src] -= 1; counts[dst] += 1
    for dim in range(d):
        sums[src][dim] -= pts[i][dim]
        sums[dst][dim] += pts[i][dim]
```

The update step falls from O(n·d) per iteration to O(moved · d) -- and `moved`
collapses toward zero exactly when the naive rebuild wastes the most work.

**Partial-distance abandonment.** While accumulating a squared distance, bail out
the moment the running sum exceeds the best candidate so far. Safe because
squared terms only ever add. Seeding the bound with the point's *current* cluster
means rivals abandon almost immediately near convergence.

**Free convergence check.** "No point moved" falls out of the move tracking.

## 4. Measured, including where it does not help

Deterministic blobs, seed 0, same machine:

| Shape | slow | average | best |
|-------|------|---------|------|
| d=2, k=3, n=900 | 44989 inertia, 0.069s | 4030, 0.0019s | 4030, **0.0019s** |
| d=20, k=8, n=2400 | 718562, 1.957s | 710131, 0.300s | 710131, **0.152s** |
| d=50, k=12, n=2400 | 2668359, 6.743s | 269036, 0.508s | 269036, **0.423s** |

Read this carefully, because two different things are happening:

- **The inertia column is k-means++.** An 11x better clustering at d=2 and a 10x
  better one at d=50, from initialisation alone. `average` and `best` produce
  *identical* inertia -- they are the same algorithm, so this is the algorithmic
  win, and it is much larger than the engineering win.
- **The timing column at d=2 is a tie.** The distance bound has almost nothing to
  abandon in two coordinates, so `best` buys nothing. It pays 2x at d=20 and
  1.2x at d=50.

That tie is the most useful row in the table. **An optimisation has a regime**,
and reporting only the shape where yours wins is how folklore gets made. The demo
sweeps all three deliberately.

## 5. The production ladder beyond this file

Everything above is constant-factor work inside O(iters · n · k · d). Asked to go
further, the real order is:

1. **Vectorise.** numpy/BLAS turns the assign step into one matrix expression.
   Banned in a from-scratch exercise, and the first thing to reach for in reality.
2. **Elkan / Hamerly.** Triangle-inequality bounds that *skip* whole distance
   computations rather than truncating them as partial-distance does.
3. **Mini-batch k-means.** When n stops fitting the time budget, update centroids
   from sampled batches.

Naming the ladder you are not climbing is how you show the constant-factor work
was a choice rather than the limit of what you knew.

## Build Log

1. **Restate the problem and the edge cases** before any code: k < 1 and k > n
   are errors; k == 1 (grand mean) and k == n (zero inertia) fall out naturally
   and need no special casing.
2. **Shared helpers first** -- `_dist2`, `_validate`, `_inertia`. A minute of
   work that de-risks all three implementations.
3. **The slow baseline**, with each weakness named in a comment as it is written.
4. **The contract tests**, written against the baseline: recovers separated
   clusters, k=1, k=n, invalid k, determinism under a fixed seed.
5. **`average`**, then re-run the same suite unchanged. The suite is the safety
   net that makes the rewrite safe.
6. **The "better init never loses" test**, averaged over eight seeds. A single
   lucky random init proves nothing, so the comparison has to be a mean.
7. **`best`**, again against the unchanged suite.
8. **The timing sweep across dimensions**, which is what exposed the d=2 tie.

Step 4 before step 5 is the whole method. The tests were written against the
implementation that was allowed to be wrong, so they encode the *contract*
rather than the current behaviour.

## Technique Catalog

| Technique | When to apply |
|-----------|---------------|
| Fix initialisation before iteration | Any local-search algorithm |
| Drop monotonic transforms in comparisons | Any argmin/argmax over a metric |
| Converge on discrete state | Any alternating-optimisation loop |
| Incremental aggregate maintenance | Any loop recomputing a sum that changed slightly |
| Partial-distance abandonment | High-dimensional nearest-neighbour search |
| Seed the bound with the current answer | Any best-so-far pruning near a fixed point |
| Shared contract across tiers | Any optimisation done as successive rewrites |
| Sweep the regime | Any performance claim |
| Handle degenerate cases explicitly | Any partitioning algorithm (empty clusters) |

## Connections to Other Tracks

| Concept | Connected Track | How |
|---------|-----------------|-----|
| Unsupervised learning, EM | [Statistical Learning](../../machine-learning/01-statistical-learning/) | Lloyd's is EM's hard-assignment cousin |
| Nearest-neighbour search, ANN | [Retrieval & RAG](../../ai-platform-engineering/07-retrieval-and-rag/) | The same distance-pruning ideas at index scale |
| Amortised analysis | [Algorithms](../../algorithms/) | Why incremental updates beat rebuilds |
| Property-based and contract testing | [The Testing Mentality](../../software-craftsmanship/03-testing-mentality/) | One suite, three implementations |

## Company Relevance

| Company | How This Appears | Focus |
|---------|-----------------|-------|
| ML and research roles | Implement a classic from scratch, then optimise | Naming the lever you pulled |
| Any quant or systems role | Constant-factor work with a stated regime | Measurement over assertion |
| Any senior interview | "Make it faster" after a correct baseline | Earning each improvement |
