"""K-means three times: slow -> average -> best, one optimisation at a time.

Standard library only, no numpy. Run me: ``python kmeans_ladder.py``.

THE PROBLEM
-----------
Given n points in R^d and an integer k, partition the points into k clusters
minimising *inertia* -- the sum of squared distances from each point to the
centroid of its assigned cluster.

Exact k-means is NP-hard, so every practical answer is **Lloyd's algorithm**:
alternate two steps until nothing changes.

    1. ASSIGN  each point to its nearest centroid
    2. UPDATE  each centroid to the mean of its assigned points

Neither step can increase inertia and there are finitely many partitions, so
it always terminates -- but at a *local* optimum. That single fact is why
initialisation, not iteration, is the highest-leverage thing to fix.

THE LADDER
----------
All three implementations share one contract, so they are interchangeable and
testable against each other:

    fit(points, k, seed=...) -> (centroids, labels, inertia)

    slow     random init, sqrt distances, fixed iteration count
    average  k-means++ init, squared distances, real stopping rule,
             explicit empty-cluster repair
    best     incremental centroid maintenance and partial-distance
             abandonment -- constant-factor work on both Lloyd's steps

The asymptotic cost per iteration, O(n * k * d), never goes away in Lloyd's.
`average` attacks the NUMBER of iterations; `best` attacks the CONSTANT. They
are different levers and it is worth being explicit about which one is being
pulled.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

Point = Sequence[float]
Result = tuple[list[list[float]], list[int], float]


# ---------------------------------------------------------------------------
# Shared helpers. Write these first: they cost a minute and de-risk the rest.
# ---------------------------------------------------------------------------


def _dist2(a: Point, b: Point) -> float:
    """Squared Euclidean distance.

    Squared on purpose: sqrt is monotonic, so "nearest centroid" is the same
    with or without it. `slow` below takes the sqrt anyway, to make the
    baseline mistake explicit; everything after it drops the sqrt.
    """
    total = 0.0
    for x, y in zip(a, b):
        total += (x - y) ** 2
    return total


def _validate(points: Sequence[Point], k: int) -> None:
    """The edge cases, stated up front: k < 1 and k > n are errors. k == n
    (every point its own cluster) and k == 1 (the grand mean) fall out of the
    algorithm naturally and need no special casing."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > len(points):
        raise ValueError(f"k={k} exceeds the number of points ({len(points)})")


def _inertia(
    points: Sequence[Point], centroids: Sequence[Point], labels: Sequence[int]
) -> float:
    return sum(_dist2(p, centroids[lab]) for p, lab in zip(points, labels))


# ---------------------------------------------------------------------------
# SLOW -- the baseline. Correct first; everything after is judged against it.
#
# Deliberate weaknesses, each one a thing a reviewer will reach for:
#   * random init           can drop two centroids inside one true cluster and
#                           converge somewhere bad
#   * sqrt in the metric     pure waste for an argmin
#   * fixed iteration count  no convergence check: either burns work after
#                            settling, or stops before settling
#   * full centroid rebuild  every centroid recomputed from scratch each round
#   * silent empty clusters  an emptied cluster keeps a stale centroid
#
# Complexity: O(iters * n * k * d) time, O(n + k*d) extra space.
# ---------------------------------------------------------------------------


def kmeans_slow(
    points: Sequence[Point], k: int, iters: int = 100, seed: int = 0
) -> Result:
    """Brute-force Lloyd's: random init, sqrt distances, fixed iterations."""
    _validate(points, k)
    pts = list(points)
    rng = random.Random(seed)

    # Sampling actual data points (rather than random coordinates in the
    # bounding box) at least guarantees every centroid starts where data is.
    centroids = [list(p) for p in rng.sample(pts, k)]

    labels = [0] * len(pts)
    for _ in range(iters):
        # ASSIGN: full scan, every point against every centroid.
        for i, p in enumerate(pts):
            best_j = 0
            best_dist = _dist2(p, centroids[0]) ** 0.5  # the wasted sqrt
            for j in range(1, k):
                dist = _dist2(p, centroids[j]) ** 0.5
                if dist < best_dist:
                    best_j, best_dist = j, dist
            labels[i] = best_j

        # UPDATE: rebuild every centroid from scratch.
        for j in range(k):
            members = [p for p, lab in zip(pts, labels) if lab == j]
            if members:  # else: silently keeps the stale centroid
                centroids[j] = [sum(dim) / len(members) for dim in zip(*members)]

    return centroids, labels, _inertia(pts, centroids, labels)


# ---------------------------------------------------------------------------
# AVERAGE -- the same Lloyd's core with the four fixes a reviewer asks for.
# ---------------------------------------------------------------------------


def _plus_plus_init(
    points: Sequence[Point], k: int, rng: random.Random
) -> list[list[float]]:
    """k-means++ seeding (Arthur & Vassilvitskii, 2007).

    First centroid uniform at random; each subsequent one is a data point drawn
    with probability proportional to D(p)^2, its squared distance to the
    nearest already-chosen centroid. Far-away regions are likely to receive a
    centroid, and a duplicate pick has D^2 = 0 so it is never re-chosen.

    Expected O(log k)-competitive with the optimal clustering, and in practice
    it slashes the iteration count. This is the single highest-value change in
    the whole ladder.
    """
    centroids = [list(rng.choice(points))]
    for _ in range(k - 1):
        d2 = [min(_dist2(p, c) for c in centroids) for p in points]
        total = sum(d2)
        if total == 0:
            # Every remaining point coincides with a centroid (duplicates).
            centroids.append(list(rng.choice(points)))
            continue
        # Weighted draw by inverse-CDF walk: r uniform in [0, total), take the
        # first point whose cumulative weight reaches r.
        r = rng.random() * total
        cumulative = 0.0
        for p, w in zip(points, d2):
            cumulative += w
            if cumulative >= r:
                centroids.append(list(p))
                break
    return centroids


def kmeans_average(
    points: Sequence[Point], k: int, max_iters: int = 100, seed: int = 0
) -> Result:
    """Lloyd's with k-means++ init, squared distances, and a real stop rule."""
    _validate(points, k)
    pts = list(points)
    rng = random.Random(seed)
    centroids = _plus_plus_init(pts, k, rng)  # fix 1

    labels: list[int] | None = None
    for _ in range(max_iters):
        # ASSIGN -- squared distances only (fix 2).
        new_labels = []
        for p in pts:
            best_j = 0
            best_d2 = _dist2(p, centroids[0])
            for j in range(1, k):
                d2 = _dist2(p, centroids[j])
                if d2 < best_d2:
                    best_j, best_d2 = j, d2
            new_labels.append(best_j)

        # STOP -- assignments are the discrete state of Lloyd's. If none
        # changed, the update step is a no-op too: converged (fix 3).
        if new_labels == labels:
            break
        labels = new_labels

        # UPDATE -- with explicit empty-cluster repair (fix 4).
        for j in range(k):
            members = [p for p, lab in zip(pts, labels) if lab == j]
            if members:
                centroids[j] = [sum(dim) / len(members) for dim in zip(*members)]
            else:
                # Reseed at the worst-served point: the one contributing the
                # most inertia. One O(n*d) scan, only on the rare empty event.
                worst = max(
                    range(len(pts)), key=lambda i: _dist2(pts[i], centroids[labels[i]])
                )
                centroids[j] = list(pts[worst])

    assert labels is not None  # max_iters >= 1 guarantees one assignment pass
    return centroids, labels, _inertia(pts, centroids, labels)


# ---------------------------------------------------------------------------
# BEST -- keep the algorithmic wins, attack the constant factor of both steps.
# ---------------------------------------------------------------------------


def _dist2_bounded(a: Point, b: Point, bound: float) -> float:
    """Squared distance with early abandonment.

    Returns the true squared distance when it is <= bound; otherwise returns
    *some* value greater than bound (the partial sum at the moment of
    abandonment). Safe because the caller only ever accepts a value that beats
    the bound, and because squared coordinate terms only ever add.
    """
    total = 0.0
    for x, y in zip(a, b):
        total += (x - y) ** 2
        if total > bound:
            return total
    return total


def kmeans_best(
    points: Sequence[Point], k: int, max_iters: int = 100, seed: int = 0
) -> Result:
    """Lloyd's with k-means++, move-based centroid maintenance, and pruning."""
    _validate(points, k)
    pts = [list(p) for p in points]
    n, d = len(pts), len(pts[0])
    rng = random.Random(seed)
    centroids = _plus_plus_init(pts, k, rng)

    # Initial ASSIGN, already using bounded distances.
    labels = []
    for p in pts:
        best_j = 0
        best_d2 = _dist2(p, centroids[0])
        for j in range(1, k):
            cand = _dist2_bounded(p, centroids[j], best_d2)
            if cand < best_d2:
                best_j, best_d2 = j, cand
        labels.append(best_j)

    # Running sums and counts: the state that makes UPDATE incremental.
    sums = [[0.0] * d for _ in range(k)]
    counts = [0] * k
    for p, lab in zip(pts, labels):
        counts[lab] += 1
        for dim in range(d):
            sums[lab][dim] += p[dim]

    def move(i: int, dst: int) -> None:
        """Reassign point i to cluster dst, patching both running sums in O(d)."""
        src = labels[i]
        labels[i] = dst
        counts[src] -= 1
        counts[dst] += 1
        for dim in range(d):
            sums[src][dim] -= pts[i][dim]
            sums[dst][dim] += pts[i][dim]

    for _ in range(max_iters):
        # UPDATE from running sums: O(k*d) per iteration, not O(n*d).
        for j in range(k):
            if counts[j]:
                centroids[j] = [s / counts[j] for s in sums[j]]
            else:
                worst = max(
                    range(n), key=lambda i: _dist2(pts[i], centroids[labels[i]])
                )
                move(worst, j)
                centroids[j] = list(pts[worst])

        # ASSIGN with pruning, tracking how many points actually moved.
        moved = 0
        for i, p in enumerate(pts):
            # Seed the bound with the CURRENT assignment: near convergence it
            # is usually already the winner, so every rival abandons almost
            # immediately. The bound is the optimisation, not the loop.
            best_j = labels[i]
            best_d2 = _dist2(p, centroids[best_j])
            for j in range(k):
                if j == best_j:
                    continue
                cand = _dist2_bounded(p, centroids[j], best_d2)
                if cand < best_d2:
                    best_j, best_d2 = j, cand
            if best_j != labels[i]:
                move(i, best_j)
                moved += 1

        # STOP: nothing moved means the next update is a no-op. Free, because
        # the move tracking already exists.
        if moved == 0:
            break

    return centroids, labels, _inertia(pts, centroids, labels)


# ---------------------------------------------------------------------------
# The production ladder beyond this file, in the order it is usually climbed:
#
#   1. Vectorise    numpy/BLAS turns ASSIGN into one matrix expression. Banned
#                   in a from-scratch exercise; the first thing to reach for
#                   in reality.
#   2. Elkan/Hamerly  triangle-inequality bounds SKIP whole distance
#                   computations rather than truncating them, as `best` does.
#   3. Mini-batch   when n stops fitting the time budget, update centroids
#                   from sampled batches instead of the full set.
# ---------------------------------------------------------------------------


IMPLEMENTATIONS = (kmeans_slow, kmeans_average, kmeans_best)


def make_blobs(
    n_per: int = 300, k: int = 3, d: int = 2, spread: float = 1.0, seed: int = 42
):
    """Deterministic well-separated clusters, so a correct implementation has
    an unambiguous right answer to recover."""
    rng = random.Random(seed)
    centres = [[rng.uniform(-20, 20) for _ in range(d)] for _ in range(k)]
    return [
        [c[dim] + rng.gauss(0, spread) for dim in range(d)]
        for c in centres
        for _ in range(n_per)
    ]


# ---------------------------------------------------------------------------
# Contract tests: one suite, run against all three implementations. Any tier
# that fails these is not an optimisation, it is a regression.
# ---------------------------------------------------------------------------


def test_shared_contract() -> None:
    points = make_blobs()
    for fit in IMPLEMENTATIONS:
        centroids, labels, inertia = fit(points, k=3, seed=0)
        assert len(centroids) == 3, fit.__name__
        assert len(labels) == len(points), fit.__name__
        assert set(labels) <= {0, 1, 2}, fit.__name__
        assert inertia > 0, fit.__name__


def test_recovers_separated_clusters() -> None:
    """Well-separated blobs: every implementation must find all three, and no
    cluster may be empty."""
    points = make_blobs(n_per=200, k=3, spread=0.5)
    for fit in IMPLEMENTATIONS:
        _, labels, _ = fit(points, k=3, seed=0)
        sizes = sorted(labels.count(j) for j in range(3))
        assert sizes == [200, 200, 200], f"{fit.__name__} got {sizes}"


def test_k_equals_one_is_the_grand_mean() -> None:
    points = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]
    for fit in IMPLEMENTATIONS:
        centroids, labels, _ = fit(points, k=1, seed=0)
        assert labels == [0, 0, 0, 0], fit.__name__
        assert centroids[0] == [1.0, 1.0], f"{fit.__name__} got {centroids[0]}"


def test_k_equals_n_gives_zero_inertia() -> None:
    """Every point its own cluster: inertia must be exactly zero. This is the
    case where a stale-centroid bug shows up immediately."""
    points = [[float(i), float(i)] for i in range(6)]
    for fit in (kmeans_average, kmeans_best):
        _, _, inertia = fit(points, k=6, seed=0)
        assert inertia < 1e-9, f"{fit.__name__} got inertia {inertia}"


def test_rejects_invalid_k() -> None:
    points = make_blobs(n_per=5, k=2)
    for fit in IMPLEMENTATIONS:
        for bad_k in (0, -1, len(points) + 1):
            try:
                fit(points, k=bad_k, seed=0)
            except ValueError:
                continue
            raise AssertionError(f"{fit.__name__} accepted k={bad_k}")


def test_deterministic_for_a_fixed_seed() -> None:
    points = make_blobs(n_per=100, k=3)
    for fit in IMPLEMENTATIONS:
        first = fit(points, k=3, seed=7)
        second = fit(points, k=3, seed=7)
        assert first == second, fit.__name__


def test_better_init_never_loses() -> None:
    """The point of the whole ladder: k-means++ tiers must not produce worse
    clustering than the random-init baseline. Averaged over seeds, because a
    single lucky random init proves nothing."""
    points = make_blobs(n_per=150, k=4, spread=1.5)
    seeds = range(8)
    mean_slow = sum(kmeans_slow(points, 4, seed=s)[2] for s in seeds) / len(list(seeds))
    for fit in (kmeans_average, kmeans_best):
        mean_fit = sum(fit(points, 4, seed=s)[2] for s in seeds) / len(list(seeds))
        assert mean_fit <= mean_slow * 1.001, (
            f"{fit.__name__}: {mean_fit} vs {mean_slow}"
        )


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(
        f"\n{len(tests)} contract tests passed across {len(IMPLEMENTATIONS)} implementations\n"
    )

    import time

    # Swept across dimensionality on purpose. At d=2 the two k-means++ tiers
    # are indistinguishable -- the partial-distance bound has almost nothing to
    # abandon in two coordinates. The constant-factor win only appears as d and
    # k grow, which is the honest shape of that optimisation.
    print(f"{'shape':<18}{'implementation':<18}{'inertia':>14}{'seconds':>10}")
    print("-" * 60)
    for d, k, n_per in ((2, 3, 300), (20, 8, 300), (50, 12, 200)):
        data = make_blobs(n_per=n_per, k=k, d=d, spread=1.5)
        shape = f"d={d} k={k} n={len(data)}"
        for fit in IMPLEMENTATIONS:
            start = time.perf_counter()
            *_, inertia = fit(data, k=k, seed=0)
            elapsed = time.perf_counter() - start
            print(f"{shape:<18}{fit.__name__:<18}{inertia:>14.1f}{elapsed:>10.4f}")
            shape = ""
        print()
