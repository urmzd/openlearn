# Case Study: K-Means Optimization

"Implement k-means, then make it faster" is a classic from-scratch exercise, and how you answer the second half says more than the first. This case study implements k-means three times behind one contract, improving it in the order a reviewer will push on it, and names the cost and the regime of every change. The result is a lesson in knowing which lever you are pulling: one tier changes the algorithm and cuts inertia by an order of magnitude, the next only changes constant factors, and the measurements show exactly where each one helps and where it does not.

The runnable implementation is [`kmeans_ladder.py`](kmeans_ladder.py): three implementations behind one contract, standard library only. Run it with `python kmeans_ladder.py`; the demo sweeps three data shapes on purpose.

> **Learning objectives**
> - State the k-means objective and explain why Lloyd's algorithm terminates at a local optimum.
> - Explain k-means++ initialization and why it outweighs every constant-factor optimization.
> - Justify dropping `sqrt`, stopping on unchanged assignments, and repairing empty clusters.
> - Explain incremental centroid maintenance and partial-distance abandonment, and the regime in which each pays.
> - Use a shared contract and test suite to keep successive rewrites honest.

## Before you begin

You need Euclidean distance, means, and big-O; no machine-learning background is required. Write the baseline yourself first and list its weaknesses before reading the section on the second tier. If your list matches, you already know what a reviewer will ask for.

## The problem

Given $n$ points in $\mathbb{R}^d$ and an integer $k$, partition the points into $k$ clusters minimizing **inertia**: the sum of squared distances from each point to the centroid of its assigned cluster.

Exact k-means is NP-hard, so every practical answer is **Lloyd's algorithm**, which alternates two steps:

1. **ASSIGN** each point to its nearest centroid.
2. **UPDATE** each centroid to the mean of its assigned points.

Neither step can increase inertia, and there are finitely many partitions, so the algorithm terminates. But it terminates at a *local* optimum, and that single fact determines where the effort should go.

## The shared contract

All three implementations share one signature:

```python
fit(points, k, seed=...) -> (centroids, labels, inertia)
```

One test suite runs against all of them. That is what makes the ladder a ladder rather than three separate programs: any tier that fails the shared contract is a regression wearing an optimization's clothes.

## The slow baseline worth writing

The first tier uses random initialization, `sqrt` in the distance, a fixed 100 iterations, and rebuilds every centroid from scratch each round. It runs in $O(\text{iters} \cdot n \cdot k \cdot d)$ time and $O(n + k d)$ space. Each of its weaknesses is something a reviewer will reach for:

| Weakness | Consequence |
|----------|-------------|
| Random init | Can drop two centroids inside one true cluster and converge badly |
| `sqrt` in the metric | Pure waste: it cannot change an argmin |
| Fixed iteration count | Either burns work after settling, or stops before settling |
| Full centroid rebuild | O(n·d) every iteration regardless of how little changed |
| Silent empty clusters | An emptied cluster keeps a stale centroid forever |

It is correct, and everything after it is judged against it. Writing it is not a formality: it is the only way the later tiers have something to be compared to.

## The average tier: four fixes

The second tier addresses the weaknesses that change the answer or waste work for nothing.

**k-means++ initialization** (Arthur and Vassilvitskii, 2007) is the highest-value change in the entire study. The first centroid is chosen uniformly at random; each subsequent one is a data point drawn with probability proportional to $D(p)^2$, its squared distance to the nearest already-chosen centroid:

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

Far regions are likely to receive a centroid, and a duplicate pick has $D^2 = 0$, so it is never chosen again. The seeding is expected $O(\log k)$-competitive with the optimal clustering, and in practice it slashes the iteration count too. Because Lloyd's converges to a local optimum, the seeding largely decides the answer.

**Drop the `sqrt`.** Square root is monotonic, so comparing squared distances gives the same argmin. The change is free, and it removes a transcendental call from the innermost loop.

**A real stopping rule.** Assignments are the discrete state of Lloyd's algorithm: if none changed, the update step is a no-op and the algorithm has converged. Stopping on that is both faster and more correct than a fixed budget.

**Explicit empty-cluster repair.** A dead centroid is reseeded at the point contributing the most inertia. This costs one extra $O(nd)$ scan, only on the rare empty event, in exchange for removing a silent wrong-answer failure mode.

## The best tier: constant-factor engineering

The third tier keeps the algorithm and reduces the work per iteration. Note the distinction: the average tier reduces the *number* of iterations, and the best tier reduces the *constant factor*. Both live inside the same $O(\text{iters} \cdot n \cdot k \cdot d)$, and conflating them produces confused optimization.

**Incremental centroid maintenance.** Keep per-cluster running sums and counts. A point that changes cluster patches two sums in $O(d)$:

```python
def move(i, dst):
    src = labels[i]
    labels[i] = dst
    counts[src] -= 1; counts[dst] += 1
    for dim in range(d):
        sums[src][dim] -= pts[i][dim]
        sums[dst][dim] += pts[i][dim]
```

The update step falls from $O(nd)$ per iteration to $O(\text{moved} \cdot d)$, and `moved` collapses toward zero exactly when the naive rebuild wastes the most work.

**Partial-distance abandonment.** While accumulating a squared distance, bail out the moment the running sum exceeds the best candidate so far. This is safe because squared terms only ever add. Seeding the bound with the point's *current* cluster means rivals are abandoned almost immediately near convergence.

**A free convergence check.** "No point moved" falls out of the move tracking.

## Measured, including where it does not help

On deterministic blobs, seed 0, same machine, the three tiers give:

| Shape | slow | average | best |
|-------|------|---------|------|
| d=2, k=3, n=900 | 44989 inertia, 0.069s | 4030, 0.0019s | 4030, **0.0019s** |
| d=20, k=8, n=2400 | 718562, 1.957s | 710131, 0.300s | 710131, **0.152s** |
| d=50, k=12, n=2400 | 2668359, 6.743s | 269036, 0.508s | 269036, **0.423s** |

Read this carefully, because two different things are happening. The inertia column is k-means++: an 11x better clustering at d=2 and a 10x better one at d=50, from initialization alone. The average and best tiers produce *identical* inertia, because they are the same algorithm. That is the algorithmic win, and it is much larger than the engineering win, by an order of magnitude more than every constant-factor optimization combined.

The timing column at d=2 is a tie. The distance bound has almost nothing to abandon in two coordinates, so the best tier buys nothing there. It pays 2x at d=20 and 1.2x at d=50.

That tie is the most useful row in the table. **An optimization has a regime**, and reporting only the shape where yours wins is how folklore gets made. That is why the demo sweeps all three shapes.

## The production ladder beyond this file

Everything above is constant-factor work inside $O(\text{iters} \cdot n \cdot k \cdot d)$. Asked to go further, the real order is:

1. **Vectorize.** numpy and BLAS turn the assign step into one matrix expression. That is banned in a from-scratch exercise, and the first thing to reach for in reality.
2. **Elkan or Hamerly.** Triangle-inequality bounds that *skip* whole distance computations rather than truncating them as partial-distance abandonment does.
3. **Mini-batch k-means.** When $n$ stops fitting the time budget, update centroids from sampled batches.

Naming the ladder you are not climbing is how you show the constant-factor work was a choice rather than the limit of what you knew.

## The build log

The ladder was built in this order:

1. **Restate the problem and the edge cases** before any code: $k < 1$ and $k > n$ are errors; $k = 1$ (the grand mean) and $k = n$ (zero inertia) fall out naturally and need no special casing.
2. **Shared helpers first**: `_dist2`, `_validate`, `_inertia`. A minute of work that de-risks all three implementations.
3. **The slow baseline**, with each weakness named in a comment as it is written.
4. **The contract tests**, written against the baseline: it recovers separated clusters, handles $k = 1$, $k = n$, and invalid $k$, and is deterministic under a fixed seed.
5. **The average tier**, then the same suite re-run unchanged. The suite is the safety net that makes the rewrite safe.
6. **The "better init never loses" test**, averaged over eight seeds. A single lucky random initialization proves nothing, so the comparison has to be a mean.
7. **The best tier**, again against the unchanged suite.
8. **The timing sweep across dimensions**, which is what exposed the d=2 tie.

Step 4 before step 5 is the whole method. The tests were written against the implementation that was allowed to be wrong, so they encode the *contract* rather than the current behavior.

## Techniques to take away

| Technique | When to apply |
|-----------|---------------|
| Fix initialization before iteration | Any local-search algorithm |
| Drop monotonic transforms in comparisons | Any argmin/argmax over a metric |
| Converge on discrete state | Any alternating-optimization loop |
| Incremental aggregate maintenance | Any loop recomputing a sum that changed slightly |
| Partial-distance abandonment | High-dimensional nearest-neighbor search |
| Seed the bound with the current answer | Any best-so-far pruning near a fixed point |
| Shared contract across tiers | Any optimization done as successive rewrites |
| Sweep the regime | Any performance claim |
| Handle degenerate cases explicitly | Any partitioning algorithm (empty clusters) |

## Where this leads

Lloyd's algorithm is the hard-assignment cousin of EM, developed with the rest of unsupervised learning in [Statistical learning](../01-statistical-learning/README.md). The same distance-pruning ideas reappear at index scale in nearest-neighbor search, in [Retrieval and RAG](../27-retrieval-and-rag/README.md). Why incremental updates beat rebuilds is amortized analysis, from [Programming and Algorithms](../../programming-and-algorithms/README.md). And one suite across three implementations is contract testing, from [The testing mentality](../../software-engineering/03-testing/README.md).

Where the problem shows up: ML and research roles ask you to implement a classic from scratch and then optimize it, and listen for whether you name the lever you pulled; quant and systems roles want constant-factor work with a stated regime, measured rather than asserted; and any senior interview that says "make it faster" after a correct baseline is asking you to earn each improvement.

## Key terms

- **Inertia**: the sum of squared distances from each point to its assigned centroid.
- **Lloyd's algorithm**: alternating assignment to the nearest centroid and recomputation of centroids as means.
- **k-means++**: seeding that picks each new centroid with probability proportional to its squared distance from the nearest chosen one.
- **Partial-distance abandonment**: stopping a distance computation once it exceeds the best candidate so far.
- **Regime**: the range of inputs in which an optimization actually pays.

## Summary

- Lloyd's algorithm converges to a local optimum, so initialization dominates: k-means++ improved inertia 10 to 11 times, far more than any engineering.
- Dropping `sqrt`, stopping on unchanged assignments, and repairing empty clusters are cheap correctness and speed fixes.
- Incremental sums and partial-distance abandonment reduce the constant factor, paying 2x at d=20 and nothing at d=2.
- One contract and one test suite across all tiers turn rewrites into measured improvements.

## Review questions

1. Why does initialization matter more than every constant-factor optimization in this study?
   <details><summary>Answer</summary>Lloyd's algorithm terminates at a local optimum that depends on the seeding. k-means++ produced 11x better inertia at d=2 and 10x at d=50, while the constant-factor tier changed only speed and left inertia identical.</details>
2. Explain why dropping `sqrt` is safe.
   <details><summary>Answer</summary>Square root is monotonically increasing on non-negative numbers, so ordering by squared distance gives the same ordering as by distance, and the argmin is unchanged.</details>
3. Why does the best tier tie the average tier at d=2?
   <details><summary>Answer</summary>Partial-distance abandonment saves work by stopping a sum partway through its coordinates; with only two coordinates there is almost nothing to abandon, so the bound buys nothing.</details>
4. Why is the "better init never loses" test averaged over eight seeds?
   <details><summary>Answer</summary>A single lucky random initialization can beat k-means++ on one seed, so a single comparison proves nothing; the claim is about the mean.</details>
5. What is the difference between the levers pulled by the average and best tiers?
   <details><summary>Answer</summary>The average tier reduces the number of iterations (and improves the answer) through better seeding and a real stopping rule; the best tier reduces the constant factor per iteration through incremental sums and pruning. Both stay within $O(\text{iters} \cdot n \cdot k \cdot d)$.</details>
