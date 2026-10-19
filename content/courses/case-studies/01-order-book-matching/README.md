# Order Book Matching

## Overview

- **Problem**: build a limit order book with a matching engine, live, under time
  pressure, and be able to defend every data-structure choice.
- **Runnable**: [`matching_engine.py`](matching_engine.py) -- three book layers
  behind one interface, standard library only, `python matching_engine.py`.
- **Prerequisites**: heaps, hash maps, amortised analysis. No finance knowledge;
  the three rules below are the entire domain model.
- **Estimated time**: 1-2 days

## Key Takeaways

- **Two orderings in the rules means two structures in the storage.** Price and
  time are ranked independently, so trying to serve both from one container is
  the mistake that makes the rest hard.
- **Write the O(n) version first.** It is correct by inspection, it gets the
  matching loop under test, and it gives you a working system to point at while
  discussing improvements. Skipping it to look clever costs more than it saves.
- **One invariant makes the whole thing testable**: the book is never crossed.
  Assert it after every event and a replay becomes a real test.
- **The engine must not change when the storage does.** That is the payoff of
  the interface, and it is provable: the same order stream through all three
  layers produces byte-identical fill logs.

## How to Study

- Read the three rules, then write the flat-list version yourself before
  reading further. It takes about fifteen minutes.
- Run the implementation, then break it deliberately: remove the zero-quantity
  removal in `fill()` and watch `test_exact_fill_leaves_no_ghost` catch the
  classic blocking-ghost bug.
- Say the BUD analysis out loud before reading Layer 2. If you cannot name the
  bottleneck in the list version, the heap is a memorised answer rather than a
  derived one.

---

# Concepts & Techniques

## The Problem

Orders arrive one at a time, each with a side, a limit price, and a quantity.
Three rules define the entire system:

1. **Match on arrival.** An incoming order matches against resting orders on the
   opposite side while prices are compatible. Whatever is left rests in the book.
2. **Best price first, then oldest first.** This is *price-time priority*.
3. **The execution prints at the resting order's price**, not the arriving one.

Rule 3 catches people out. The arriving order was willing to do worse; the
resting order set the terms by waiting. Waiting is what earns the better price,
and that is the economic content of the whole mechanism.

## The Invariant

> After every event, the best bid is strictly below the best ask.

If a buy and a sell could rest simultaneously at compatible prices, rule 1 was
violated somewhere. This single assertion turns a deterministic replay into a
genuine test: feed a known stream, check the invariant after every order, and
compare the fill log against expected output.

It is also the property that makes the three-layer refactor safe. Any layer that
breaks price-time priority produces a different fill log, and the cross-layer
equivalence test fails immediately.

## 1. Two orderings, two structures

Rule 2 ranks by price, then by time. These are independent orderings, so the
storage layer needs one structure for each:

| Ordering | Question it answers | Structure |
|----------|--------------------|-----------|
| Price | Which level is best? | Heap of prices, or an indexed array |
| Time | Which order in this level is oldest? | Insertion-ordered dict per level |

Python dicts preserve insertion order, so a level is a queue for free: adding is
joining the back, and `next(iter(level.values()))` is the front. In a language
without that guarantee, a deque or an intrusive linked list plays the same role.

## 2. The heap holds levels, not orders

This is the detail that makes the heap version work at all. One entry per price
*level*, pushed once when the level is created, so the heap stays bounded by the
number of distinct live prices rather than the number of resting orders.

The consequence: `fill()` never touches the heap. When a level empties, its heap
entry is left behind as a stale price and swept lazily on the way down in
`best()`.

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

Eager deletion would mean finding an arbitrary entry in a heap, which is O(n) --
exactly the cost being avoided. Lazy deletion is amortised O(1) because every
stale entry is swept at most once.

## 3. Earning each layer with BUD

The jump from one layer to the next is justified by naming three things about
the previous one: the **B**ottleneck, the **U**nnecessary work, and the
**D**uplicated work.

**List to heap:**

- *Bottleneck*: `best()` scans every resting order, O(n), once per fill iteration.
- *Unnecessary*: it scans orders at prices that cannot possibly win.
- *Duplicated*: the best price rarely changes between calls, yet it is
  recomputed from scratch every single time.

**Heap to ladder:**

- *Bottleneck*: `heappush` is O(log L) in the number of live levels, and it runs
  on every new price.
- *Unnecessary*: comparisons at all. Real venues quote a bounded price range on a
  fixed tick, so a price is a small integer -- an array index, not a sort key.
- *Duplicated*: the best price moves by small steps, so re-deriving it from a
  heap discards the locality of the previous answer.

The ladder replaces the heap with a preallocated array plus a pointer that only
walks away from the top of the book, and gets pulled back by `add()`. Amortised
O(1), no comparisons, no allocation on the hot path.

## 4. What the ladder costs

State this out loud, because it is what makes the optimisation legitimate rather
than a micro-optimisation:

- **Memory** proportional to the price range, not to the number of orders.
- **A hard assumption** that prices stay inside that range. `LadderBook` raises
  on a price outside `[0, max_price]` instead of corrupting silently.

That trade is correct for an exchange with circuit breakers and a tick size. It
is wrong for an arbitrary-precision or unbounded-price market. The structure
encodes a domain assumption, and the honest move is to name it.

## 5. The details that produce real bugs

| Detail | What breaks without it |
|--------|------------------------|
| Integer prices | Float equality silently breaks both comparison and level bucketing |
| Remove on exact fill | A zero-quantity ghost at the front of a level blocks every future match at that price |
| Partial fill keeps position | An order that was filled, not resubmitted, must not lose its place in the queue |
| Zero-remainder guard | A fully filled arriving order must never be added to the book |
| Frozen `Fill` records | Later mutation of order quantities would otherwise rewrite fill history |
| Sequence stamped on arrival | Time priority needs a total order that survives equal timestamps |

Each of these has a named test. That is not coincidence: they are the failure
modes worth encoding, and the test names are the documentation.

## Build Log

The order this was assembled in, which is the transferable part:

1. **Restate the three rules and the invariant** before writing anything. If the
   rules are wrong, nothing downstream matters.
2. **Data first.** `Order` (mutable quantity, stamped sequence) and `Fill`
   (frozen). Integer prices, decided and justified up front.
3. **The flat-list book.** Correct by inspection, three methods, no cleverness.
4. **The engine.** The matching loop, written once and never touched again. This
   is where partial fills, the resting-price rule, and the zero-remainder guard
   get worked out, against a book simple enough that bugs are obviously bugs.
5. **The invariant check**, wired into a replay harness.
6. **Scenario tests**: level sweeping, FIFO within a price, partial fill keeping
   position, the exact-fill ghost, the empty book.
7. **Only then, the heap layer.** BUD analysis first, implementation second.
8. **Cross-layer equivalence test.** A 400-order pseudo-random stream through
   every layer, asserting identical fill logs. This is what proves the interface
   was real rather than aspirational.
9. **The ladder layer**, with its assumption stated and enforced.

Steps 3 through 6 produce a complete, defensible system. Everything after is
optional improvement on a working base, which is the position you want to be in
when the clock runs out.

## Technique Catalog

| Technique | When to apply |
|-----------|---------------|
| One structure per ordering | Any time a rule ranks by two independent keys |
| Heap of buckets, not items | When items cluster onto far fewer distinct keys |
| Lazy deletion with a sweep on read | When eager removal from a heap would be O(n) |
| Index-as-key (the ladder) | Bounded, discrete key range on a hot path |
| Invariant asserted after every event | Any stateful system with a "cannot happen" property |
| Deterministic replay | Validating stateful systems without a live environment |
| Cross-implementation equivalence | Any refactor that swaps a structure behind an interface |
| BUD analysis | Justifying an optimisation instead of reciting one |

## Connections to Other Tracks

| Concept | Connected Track | How |
|---------|-----------------|-----|
| Heaps, amortised analysis, ordering | [Algorithms](../../algorithms/) | The underlying data structures |
| Interface stability across implementations | [The Pragmatic Programmer](../../software-craftsmanship/01-pragmatic-programmer/) | Orthogonality: change storage without touching the engine |
| Invariants and property-based testing | [The Testing Mentality](../../software-craftsmanship/03-testing-mentality/) | The invariant is the property; replay is the harness |
| Latency budgets and tail behaviour | [System Design](../../systems/01-system-design/) | Why the constant factor is worth this much attention |
| Named constants over magic numbers | [Lessons from Practice](../../software-craftsmanship/04-lessons-from-practice/) | The failure mode this build avoids |

## Company Relevance

| Company | How This Appears | Focus |
|---------|-----------------|-------|
| Trading firms | The order book is the canonical live-coding problem | Correctness first, then constant factors |
| Any low-latency role | Bounded key ranges, cache behaviour, allocation on the hot path | Where the log factor goes |
| Any systems interview | Refactoring behind a stable interface | Orthogonality under pressure |
