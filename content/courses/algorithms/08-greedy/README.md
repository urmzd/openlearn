# Greedy algorithms

A greedy algorithm makes the choice that looks best right now and never reconsiders it. That works only when the **locally optimal choice is globally optimal**, and the whole skill of greedy algorithms is knowing when that is true. The usual proof technique is an **exchange argument**: take any optimal solution that differs from the greedy one, swap one of its choices for the greedy choice, and show the result is no worse. Repeat until the optimal solution *is* the greedy one. A practical rule: if you cannot convince yourself a greedy choice is correct in two minutes, the problem is probably dynamic programming.

This chapter follows [Dynamic programming](../07-dynamic-programming/README.md) on purpose. Both rely on optimal substructure; what greedy adds is the greedy-choice property, which lets it commit to one choice instead of comparing them all. The chapter covers sorting then choosing, greedy with a heap, pairing from both ends, local decisions that need no sort, and Huffman coding, and ends with a guide for telling greedy problems from DP problems.

> **Learning objectives**
> - Prove a greedy choice correct with an exchange argument, or find the counterexample that rules it out.
> - Solve interval scheduling by sorting on end time.
> - Use a heap to repeatedly take the best available option.
> - Solve running-extreme problems (Jump Game, Best Time to Buy and Sell Stock) in one pass.
> - Build an optimal prefix code with Huffman's algorithm.

## Before you begin

Read [Dynamic programming](../07-dynamic-programming/README.md) first, so you can recognize when greedy fails and DP is needed. The heap-based patterns use a priority queue; [the binary heap for top-k](../13-systems-data-structures/04-binary-heap-top-k.md) builds one, and Huffman coding builds a binary tree as in [Trees](../05-trees/README.md).

## Sort, then choose

Interval scheduling, task assignment, and meeting rooms share a shape: sort by the right criterion, then make one pass taking whatever fits. For "the most non-overlapping intervals," the right criterion is the **end** time. The interval that ends first leaves the most room for everything after it, and an exchange argument shows that any optimal schedule can be rewritten to start with it:

```python
intervals.sort(key=lambda x: x[1])  # sort by END time
count, end = 0, float('-inf')
for start, finish in intervals:
    if start >= end:
        count += 1
        end = finish
```

Sorting by start time or by length both fail on small counterexamples, which is exactly why the criterion matters. Non-overlapping Intervals, Meeting Rooms II, Merge Intervals, and Minimum Number of Arrows to Burst Balloons use this pattern.

## Greedy with a heap

When you need to repeatedly pick the best available option, and taking an option can make new options available, keep the options in a priority queue:

```python
import heapq
heap = initial_items
while heap and condition:
    best = heapq.heappop(heap)
    process(best)
    if has_more:
        heapq.heappush(heap, next_item)
```

Task Scheduler, Reorganize String, K Closest Points to Origin, and Meeting Rooms II (a heap of end times, one per room in use) are the standard problems.

## Pairing from both ends

To assign the largest and smallest items to specific positions, such as fitting people into two-person boats under a weight limit, sort and pair from the two ends:

```python
arr.sort()
left, right = 0, len(arr) - 1
while left < right:
    pair(arr[left], arr[right])
    left += 1
    right -= 1
```

This is the opposite-end pointer pattern from [Two pointers and sliding window](../02-two-pointers-sliding-window/README.md), used for a greedy assignment. Boats to Save People, Two City Scheduling, and Assign Cookies are examples.

## Local decisions that need no sort

Some answers build up by making the obvious choice at each step, with no sorting at all. Each of these problems reduces to one line of greedy state:

- **Jump Game**: track the farthest reachable index, `farthest = max(farthest, i + nums[i])`.
- **Best Time to Buy and Sell Stock**: track the minimum price seen so far, `profit = max(profit, price - min_price)`.
- **Gas Station**: if total gas is at least total cost, a solution exists, and it starts just after the point where the running sum is lowest.
- **Container With Most Water**: move the shorter pointer inward.

## Huffman coding

Huffman's algorithm builds an optimal prefix code, the binary tree with the minimum weighted path length. The greedy choice is to merge the two least frequent symbols into one subtree, since the least frequent symbols should be deepest:

```python
import heapq
heap = [(freq, char) for char, freq in freq_map.items()]
heapq.heapify(heap)
while len(heap) > 1:
    lo = heapq.heappop(heap)
    hi = heapq.heappop(heap)
    heapq.heappush(heap, (lo[0] + hi[0], Node(lo, hi)))
```

With $k$ distinct symbols this takes $O(k \log k)$. Huffman Coding and Minimum Cost to Merge Stones are the classic problems; [Huffman coding](01-huffman-coding/README.md) in this chapter implements the full encoder and decoder.

## Greedy or DP?

The decision between the two comes down to a few signals, summarized here:

| Signal | Greedy | DP |
|--------|--------|----|
| optimal substructure | yes | yes |
| greedy-choice property | yes | no: choices interact |
| "maximum items" or "minimum cost" with sorting | likely greedy | |
| "count all ways" or "minimum operations" | | likely DP |
| can you construct a counterexample to the greedy choice? | if no, greedy | if yes, DP |

Coin change is the standard cautionary example: taking the largest coin first works for US coins but fails for coins `{1, 3, 4}` and amount 6 (greedy gives $4 + 1 + 1$, the optimum is $3 + 3$), which is why it lives in the DP chapter.

Interviewers favor different variants. As a rough guide:

| Company | Favorite variant | Difficulty |
|---------|-----------------|------------|
| Google | interval and heap combinations | Medium-Hard |
| Amazon | meeting rooms, task scheduling | Medium |
| Meta | stock problems, Jump Game | Medium |
| Netflix | scheduling and resource allocation | Medium |
| Stripe | transaction ordering problems | Medium |

## In this chapter

The JavaScript files are solution functions; run them with Node and a small driver, or in an online judge. The Python examples run with `python3 <file>` and take their input as command-line arguments.

- [Best Time to Buy and Sell Stock](best-time-to-buy-and-sell-stock.js): one pass that moves the buy day whenever a lower price appears and records the best profit from selling at each later price; the answer is $0$ if prices only fall.
- [Jump Game](jump-game.js): a backward greedy written as a tail recursion. It walks from the end toward the start, moving the target position left whenever index `i` can reach it (`nums[i] + i >= lastPosition`), and succeeds if the target reaches index $0$.
- [Answering the most questions](example-1.py): given a time budget `h` and the time each question takes, answers the shortest questions first, keeping every question whose cumulative sorted time fits within `h`. Run it as `python3 example-1.py <h> <times...>`.
- [Fewest pit stops](example-2.py): a diver who can cover distance `d` on one breath travels between pit stops separated by the given distances, and stops only when the next leg would exceed `d`. Going as far as possible before each stop minimizes the number of stops, by the classic refueling exchange argument. Run it as `python3 example-2.py <d> <distances...>`.
- [Greedy subsequence partition](example-3.py): sweeps the sequence's values in sorted order, appending each to the current subsequence while it also comes later in the original order, and starting a new subsequence otherwise.
- [Huffman coding](01-huffman-coding/README.md): a Java program that compresses a text file with a Huffman tree and decodes it again.
