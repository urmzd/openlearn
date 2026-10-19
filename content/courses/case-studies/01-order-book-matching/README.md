# Order Book Matching

A limit order book with a matching engine is the canonical live-coding problem at trading firms, and a good test of data-structure judgment anywhere latency matters. The domain is small enough to state in three rules, yet the obvious implementation is slow in a way you have to name precisely before you can fix it. This case study builds the engine the way it was actually built under time pressure: the flat, obviously correct version first, then a heap of price levels, then a tick-indexed ladder, each earned by an explicit analysis of what the previous one wasted, and all three behind one interface so the matching loop never changes.

The runnable implementation is [`matching_engine.py`](matching_engine.py): three book layers behind one interface, standard library only, with its own tests. Run it with `python matching_engine.py`. No finance knowledge is needed; the three rules below are the entire domain model.

> **Learning objectives**
> - State the three matching rules and the invariant that makes the engine testable.
> - Explain why two independent orderings need two structures, and why the heap holds price levels rather than orders.
> - Justify each optimization with a bottleneck, unnecessary-work, and duplicated-work (BUD) analysis, and state what the ladder costs.
> - Name the small details that produce real bugs, and the test that catches each one.
> - Prove that a storage refactor did not change behavior with a cross-implementation equivalence test.

## Before you begin

You need heaps, hash maps, and amortized analysis; [Algorithms](../../algorithms/README.md) covers all three. Before reading past the rules, try writing the flat-list version yourself. It takes about fifteen minutes, and it makes everything after it easier to judge.

## The problem

Orders arrive one at a time, each with a side, a limit price, and a quantity. Three rules define the entire system:

1. **Match on arrival.** An incoming order matches against resting orders on the opposite side while prices are compatible. Whatever is left rests in the book.
2. **Best price first, then oldest first.** This is *price-time priority*.
3. **The execution prints at the resting order's price**, not the arriving one.

Rule 3 catches people out. The arriving order was willing to do worse; the resting order set the terms by waiting. Waiting is what earns the better price, and that is the economic content of the whole mechanism.

## The invariant

One property holds after every event:

> After every event, the best bid is strictly below the best ask.

If a buy and a sell could rest simultaneously at compatible prices, rule 1 was violated somewhere. This single assertion turns a deterministic replay into a genuine test: feed a known stream, check the invariant after every order, and compare the fill log against the expected output.

It is also the property that makes the three-layer refactor safe. Any layer that breaks price-time priority produces a different fill log, and the cross-layer equivalence test fails immediately.

## Two orderings, two structures

Rule 2 ranks by price, then by time. These are independent orderings, so the storage layer needs one structure for each:

| Ordering | Question it answers | Structure |
|----------|--------------------|-----------|
| Price | Which level is best? | Heap of prices, or an indexed array |
| Time | Which order in this level is oldest? | Insertion-ordered dict per level |

Trying to serve both from one container is the mistake that makes everything else hard. Python dicts preserve insertion order, so a level is a queue for free: adding is joining the back, and `next(iter(level.values()))` is the front. In a language without that guarantee, a deque or an intrusive linked list plays the same role.

## The heap holds levels, not orders

The detail that makes the heap version work at all is what goes in the heap. There is one entry per price *level*, pushed once when the level is created, so the heap stays bounded by the number of distinct live prices rather than the number of resting orders.

The consequence is that `fill()` never touches the heap. When a level empties, its heap entry is left behind as a stale price and swept lazily on the way down in `best()`:

```python
def best(self):
    while self.heap:
        price = -self.heap[0] if self.side == "buy" else self.heap[0]
        level = self.levels.get(price)
        if level:
            return next(iter(level.values()))   # first inserted = oldest
        heapq.heappop(self.heap)                # dead level: sweep and retry
    return None
```

Eager deletion would mean finding an arbitrary entry in a heap, which is O(n), exactly the cost being avoided. Lazy deletion is amortized O(1), because every stale entry is swept at most once.

## Earning each layer with BUD

Each jump from one layer to the next is justified by naming three things about the previous one: the **B**ottleneck, the **U**nnecessary work, and the **D**uplicated work. Say this analysis out loud before reading the heap layer. If you cannot name the bottleneck in the list version, the heap is a memorized answer rather than a derived one.

Going from the list to the heap, the bottleneck is that `best()` scans every resting order, O(n), once per fill iteration. The unnecessary work is scanning orders at prices that cannot possibly win. The duplicated work is that the best price rarely changes between calls, yet it is recomputed from scratch every single time.

Going from the heap to the ladder, the bottleneck is that `heappush` is O(log L) in the number of live levels, and it runs on every new price. The unnecessary work is comparisons at all: real venues quote a bounded price range on a fixed tick, so a price is a small integer, an array index rather than a sort key. The duplicated work is that the best price moves by small steps, so re-deriving it from a heap discards the locality of the previous answer.

The ladder therefore replaces the heap with a preallocated array plus a pointer that only walks away from the top of the book and gets pulled back by `add()`. That is amortized O(1), with no comparisons and no allocation on the hot path.

## What the ladder costs

State the cost out loud, because it is what makes the optimization legitimate rather than a micro-optimization. The ladder uses memory proportional to the price range, not to the number of orders, and it makes a hard assumption that prices stay inside that range. `LadderBook` raises on a price outside `[0, max_price]` instead of corrupting silently.

That trade is correct for an exchange with circuit breakers and a tick size. It is wrong for an arbitrary-precision or unbounded-price market. The structure encodes a domain assumption, and the honest move is to name it.

## The details that produce real bugs

The structures are the interesting part, but small details are where the real bugs live. Each row below has a named test in the implementation, and that is no coincidence: these are the failure modes worth encoding, and the test names are the documentation.

| Detail | What breaks without it |
|--------|------------------------|
| Integer prices | Float equality silently breaks both comparison and level bucketing |
| Remove on exact fill | A zero-quantity ghost at the front of a level blocks every future match at that price |
| Partial fill keeps position | An order that was filled, not resubmitted, must not lose its place in the queue |
| Zero-remainder guard | A fully filled arriving order must never be added to the book |
| Frozen `Fill` records | Later mutation of order quantities would otherwise rewrite fill history |
| Sequence stamped on arrival | Time priority needs a total order that survives equal timestamps |

> **Try it:** Remove the zero-quantity removal in `fill()` and run the tests. `test_exact_fill_leaves_no_ghost` catches the classic blocking-ghost bug.

## The build log

The order the engine was assembled in is the transferable part of this study:

1. **Restate the three rules and the invariant** before writing anything. If the rules are wrong, nothing downstream matters.
2. **Data first.** `Order` (mutable quantity, stamped sequence) and `Fill` (frozen). Integer prices, decided and justified up front.
3. **The flat-list book.** Correct by inspection, three methods, no cleverness.
4. **The engine.** The matching loop, written once and never touched again. This is where partial fills, the resting-price rule, and the zero-remainder guard get worked out, against a book simple enough that bugs are obviously bugs.
5. **The invariant check**, wired into a replay harness.
6. **Scenario tests**: level sweeping, FIFO within a price, a partial fill keeping its position, the exact-fill ghost, and the empty book.
7. **Only then, the heap layer.** BUD analysis first, implementation second.
8. **The cross-layer equivalence test.** A 400-order pseudo-random stream through every layer, asserting identical fill logs. This is what proves the interface was real rather than aspirational.
9. **The ladder layer**, with its assumption stated and enforced.

Steps 3 through 6 produce a complete, defensible system. Writing the O(n) version first is not a concession: it is correct by inspection, it gets the matching loop under test, and it gives you a working system to point at while discussing improvements. Skipping it to look clever costs more than it saves. Everything after step 6 is optional improvement on a working base, which is the position you want to be in when the clock runs out. And because the engine does not change when the storage does, the payoff of the interface is provable: the same order stream through all three layers produces byte-identical fill logs.

## Techniques to take away

The techniques in this build recur well beyond order books:

| Technique | When to apply |
|-----------|---------------|
| One structure per ordering | Any time a rule ranks by two independent keys |
| Heap of buckets, not items | When items cluster onto far fewer distinct keys |
| Lazy deletion with a sweep on read | When eager removal from a heap would be O(n) |
| Index-as-key (the ladder) | Bounded, discrete key range on a hot path |
| Invariant asserted after every event | Any stateful system with a "cannot happen" property |
| Deterministic replay | Validating stateful systems without a live environment |
| Cross-implementation equivalence | Any refactor that swaps a structure behind an interface |
| BUD analysis | Justifying an optimization instead of reciting one |

## Where this leads

The underlying heaps, ordering, and amortized analysis are in [Algorithms](../../algorithms/README.md). Changing storage without touching the engine is orthogonality, from [The Pragmatic Programmer](../../software-craftsmanship/01-pragmatic-programmer/README.md). The invariant is a property and the replay is its harness, which is the subject of [The testing mentality](../../software-craftsmanship/03-testing-mentality/README.md). Why the constant factor deserves this much attention is a question of latency budgets and tail behavior, covered in [System design](../../systems/01-system-design/README.md). And [Lessons from practice](../../software-craftsmanship/04-lessons-from-practice/README.md) catalogs the failure mode this build avoids by using named constants over magic numbers.

Where the problem shows up: trading firms use the order book as a live-coding problem that rewards correctness first and constant factors second; low-latency roles probe bounded key ranges, cache behavior, and allocation on the hot path; and any systems interview can ask you to refactor behind a stable interface under pressure.

## Key terms

- **Price-time priority**: matching the best price first and, within a price, the oldest order first.
- **Price level**: the set of resting orders on one side at one price, kept in arrival order.
- **Crossed book**: a book in which the best bid is at or above the best ask; it must never happen.
- **Lazy deletion**: leaving stale entries in a heap and discarding them when they reach the top.
- **Ladder**: a preallocated array indexed by price tick, with a pointer to the best level.
- **BUD analysis**: justifying an optimization by naming the bottleneck, the unnecessary work, and the duplicated work.

## Summary

- Two independent orderings in the rules mean two structures in the storage: a price structure of levels and a time-ordered queue per level.
- The heap holds one entry per level and sweeps empty levels lazily, so `fill()` never touches it.
- Each layer is earned by a BUD analysis of the previous one, and the ladder's cost (memory and a bounded price range) is stated and enforced.
- One invariant, that the book is never crossed, makes replay a real test.
- The same order stream through all three layers produces identical fill logs, which proves the interface.

## Review questions

1. Why does the execution print at the resting order's price?
   <details><summary>Answer</summary>The resting order set the terms by waiting, while the arriving order was willing to do worse. Waiting is what earns the better price; that is the economic content of the mechanism.</details>
2. Why does the heap hold price levels rather than orders, and what does that mean for `fill()`?
   <details><summary>Answer</summary>One entry per level keeps the heap bounded by the number of distinct live prices instead of the number of resting orders. `fill()` never touches the heap; an emptied level's entry stays as a stale price and is swept lazily by `best()`, which is amortized O(1) because each stale entry is swept at most once.</details>
3. Give the BUD analysis that justifies replacing the heap with a ladder.
   <details><summary>Answer</summary>Bottleneck: `heappush` is O(log L) on every new price. Unnecessary: comparisons at all, since a price on a fixed tick in a bounded range is an array index. Duplicated: the best price moves in small steps, and re-deriving it from a heap throws away the previous answer's locality.</details>
4. What does a zero-quantity ghost do, and which test catches it?
   <details><summary>Answer</summary>If an exactly filled order is not removed, it sits at the front of its level with zero quantity and blocks every future match at that price. `test_exact_fill_leaves_no_ghost` catches it.</details>
5. How do you prove that swapping the storage layer did not change behavior?
   <details><summary>Answer</summary>Run the same 400-order pseudo-random stream through every layer and assert that the fill logs are identical. Any layer that breaks price-time priority produces a different log.</details>
