# 08-greedy

## Summary
- Contains: `best-time-to-buy-and-sell-stock.js`, `jump-game.js`, `example-1.py`, `example-2.py`, `example-3.py`.
- Greedy strategy examples in JS and Python.

## Key takeaways
- Greedy solutions rely on proving a local choice leads to a global optimum.
- Track running minima/maxima to avoid extra passes.

## How to run
- JS files are solution functions (run with Node + a driver or an online judge).
- Python examples are runnable with `python3 <file>`.

---

# Greedy Patterns

## Core Insight

Greedy works when the **locally optimal choice is globally optimal**. The proof technique is usually an **exchange argument**: show that swapping any non-greedy choice for the greedy choice doesn't worsen the solution. If you can't convince yourself of correctness in 2 minutes, it's probably DP.

## Pattern 1: Sort Then Greedy

**When to use**: Interval scheduling, task assignment, meeting rooms — sort by some criterion, then process greedily.

**Template (Interval Scheduling)**:
```python
intervals.sort(key=lambda x: x[1])  # sort by END time
count, end = 0, float('-inf')
for start, finish in intervals:
    if start >= end:
        count += 1
        end = finish
```

**Interview problems**: Non-Overlapping Intervals, Meeting Rooms II, Merge Intervals, Minimum Number of Arrows

## Pattern 2: Greedy + Heap (Priority Queue)

**When to use**: Need to repeatedly pick the best available option.

**Template**:
```python
import heapq
heap = initial_items
while heap and condition:
    best = heapq.heappop(heap)
    process(best)
    if has_more:
        heapq.heappush(heap, next_item)
```

**Interview problems**: Task Scheduler, Reorganize String, K Closest Points, Meeting Rooms II

## Pattern 3: Greedy from Both Ends

**When to use**: Assign largest/smallest items to specific positions.

**Template**:
```python
arr.sort()
left, right = 0, len(arr) - 1
while left < right:
    pair(arr[left], arr[right])
    left += 1
    right -= 1
```

**Interview problems**: Boats to Save People, Two City Scheduling, Assign Cookies

## Pattern 4: Local Decision (No Sort Needed)

**When to use**: The answer builds up by making the obvious choice at each step.

**Interview problems and their greedy decisions**:
- **Jump Game**: Track farthest reachable index. `farthest = max(farthest, i + nums[i])`
- **Best Time to Buy/Sell Stock**: Track minimum price seen so far. `profit = max(profit, price - min_price)`
- **Gas Station**: If total gas >= total cost, a solution exists. Start from where running sum is lowest.
- **Container With Most Water**: Move the shorter pointer inward.

## Pattern 5: Huffman Coding (Optimal Prefix Codes)

**When to use**: Build optimal binary tree with minimum weighted path length.

**Template**:
```python
import heapq
heap = [(freq, char) for char, freq in freq_map.items()]
heapq.heapify(heap)
while len(heap) > 1:
    lo = heapq.heappop(heap)
    hi = heapq.heappop(heap)
    heapq.heappush(heap, (lo[0] + hi[0], Node(lo, hi)))
```

**Interview problems**: Huffman Coding, Minimum Cost to Merge Stones

## Greedy vs DP Decision Guide

| Signal | Greedy | DP |
|--------|--------|----|
| Optimal substructure | Yes | Yes |
| Greedy choice property | Yes | No — choices interact |
| "Maximum items" or "minimum cost" with sorting | Likely greedy | — |
| "Count all ways" or "min operations" | — | Likely DP |
| Can you construct counterexample to greedy? | If no → greedy | If yes → DP |

## Company Targeting

| Company | Favorite Variant | Difficulty |
|---------|-----------------|------------|
| Google | Interval + heap combos | Medium-Hard |
| Amazon | Meeting rooms, task scheduling | Medium |
| Meta | Stock problems, jump game | Medium |
| Netflix | Scheduling/resource allocation | Medium |
| Stripe | Transaction ordering problems | Medium |
