# Binary search

Everyone learns binary search as "find a value in a sorted array." That is the smallest case of a much more useful idea. Binary search works whenever there is a **monotonic predicate**: a function $f(x)$ that flips from false to true (or the other way) at exactly one point. You do not need a sorted array; you need a search space in which one comparison lets you eliminate half. Each step halves the space, so the search takes $O(\log n)$ steps.

This chapter generalizes in stages: from exact-match search, to finding the boundary where a condition first holds, to rotated arrays where only one half is sorted, to searching over the space of possible *answers*, and finally to searching over real numbers. The pattern of searching on the answer is the one to take away; it converts optimization problems into decision problems and is the most important binary-search pattern for hard interviews.

> **Learning objectives**
> - Write exact-match and left-boundary binary searches without off-by-one errors, and explain each bound.
> - Search a rotated sorted array by identifying which half is sorted.
> - Turn "find the minimum $x$ such that a condition holds" into a binary search over answers with a feasibility check.
> - Binary search over real numbers with a fixed iteration count.

## Before you begin

You should be comfortable with arrays and loop invariants, as in [Two pointers and sliding window](../04-two-pointers-sliding-window/README.md). The patience-sorting solution at the end of the chapter computes a longest increasing subsequence, a problem that [Dynamic programming](../09-dynamic-programming/README.md) solves another way.

## Exact match

The classic search keeps a closed interval `[left, right]` and stops when it is empty:

```
left, right = 0, len(arr) - 1
while left <= right:
    mid = left + (right - left) // 2  # avoid overflow
    if arr[mid] == target: return mid
    elif arr[mid] < target: left = mid + 1
    else: right = mid - 1
return -1
```

Writing `left + (right - left) // 2` instead of `(left + right) // 2` avoids integer overflow in languages with fixed-width integers. Binary Search and Search Insert Position are the standard problems.

## The left boundary: first true

Often you want the first element that satisfies a condition, a lower bound. Change the interval to half-open `[left, right)`, with `right` starting at `len(arr)`, and never discard `mid` when it satisfies the condition, because it might be the answer:

```
left, right = 0, len(arr)  # note: right = len, not len-1
while left < right:         # note: strict <
    mid = left + (right - left) // 2
    if condition(mid):
        right = mid         # mid might be the answer
    else:
        left = mid + 1      # mid is definitely not
return left
```

When the loop ends, `left == right` is the first index where the condition holds, or `len(arr)` if it never does. First Bad Version, Find First and Last Position of Element, and Search Insert Position all fit this template.

## Rotated arrays

A sorted array that has been rotated (`[4, 5, 6, 7, 0, 1, 2]`) is no longer sorted, but one half around any `mid` always is. Compare `arr[left]` with `arr[mid]` to find out which, then check whether the target lies in the sorted half:

```
left, right = 0, len(arr) - 1
while left <= right:
    mid = left + (right - left) // 2
    if arr[mid] == target: return mid

    if arr[left] <= arr[mid]:  # left half is sorted
        if arr[left] <= target < arr[mid]:
            right = mid - 1
        else:
            left = mid + 1
    else:                       # right half is sorted
        if arr[mid] < target <= arr[right]:
            left = mid + 1
        else:
            right = mid - 1
```

The careful comparisons with `mid` are what preserve correctness. Search in Rotated Sorted Array and Find Minimum in Rotated Sorted Array are the canonical problems; the solutions below take two different approaches to them.

## Searching on the answer

Now drop the array entirely. For "find the minimum (or maximum) value $x$ such that some condition holds," the search space is the range of possible answers. If a `feasible(x)` check is monotone (once a capacity is large enough, every larger capacity is too), binary search finds the boundary:

```
left, right = min_possible, max_possible
while left < right:
    mid = left + (right - left) // 2
    if feasible(mid):
        right = mid       # try smaller
    else:
        left = mid + 1    # need larger
return left
```

This is the left-boundary template applied to answers. It converts an optimization problem ("what is the smallest speed?") into a decision problem ("can I finish at this speed?"), which is usually much easier to write. Koko Eating Bananas, Capacity to Ship Packages Within D Days, Split Array Largest Sum, and Magnetic Force Between Two Balls are the standard examples, and it is the pattern that most often separates hard-interview answers.

## Searching over real numbers

For a real-valued answer, such as a square root, there is no "next integer" to step to, so run a fixed number of halvings instead of looping until the interval is empty:

```
left, right = 0.0, upper_bound
for _ in range(100):  # ~100 iterations gives ~10^-30 precision
    mid = (left + right) / 2
    if condition(mid):
        right = mid
    else:
        left = mid
return left
```

One hundred halvings shrink the interval by a factor of $2^{100} \approx 10^{30}$. Sqrt(x) and Minimize Max Distance to Gas Station are the usual problems.

## Choosing a template

The choice between the templates comes down to two questions, which this decision tree captures:

```
Is it a sorted array?
├── Yes → exact match? → exact-match search
│       → first/last occurrence? → left boundary
│       → rotated? → rotated-array search
└── No → Can I binary search on the answer?
        ├── Yes → search on the answer (most common in hard problems)
        └── No → Probably not a binary search problem
```

Different companies favor different variants. As a rough guide:

| Company | Favorite variant | Difficulty |
|---------|-----------------|------------|
| Google | binary search on the answer | Hard |
| Amazon | rotated array variants | Medium |
| Meta | standard and boundary | Medium |
| Jane Street | binary search on real-valued continuous spaces | Hard |
| Two Sigma | binary search on the answer with complex feasibility | Hard |
| Databricks | distributed binary search | Hard |

## Key terms

- **Monotonic predicate**: a condition $f(x)$ over an ordered search space that flips from false to true (or true to false) exactly once; binary search finds the flip point in $O(\log n)$ evaluations.
- **Binary search**: a search that keeps an interval known to contain the answer and halves it with one comparison per step, taking $O(\log n)$ steps; it needs a monotonic predicate, not necessarily a sorted array.
- **Left-boundary binary search**: the binary search over a half-open interval `[left, right)` that returns the first index where a condition holds, or the length if it never does, by never discarding a `mid` that satisfies the condition.
- **Rotated sorted array**: a sorted array shifted cyclically, such as `[4, 5, 6, 7, 0, 1, 2]`; around any midpoint one half is still sorted, which lets binary search decide which half can contain the target.
- **Binary search on the answer**: binary searching over the range of possible answers with a monotone feasibility check, which turns an optimization problem ("the smallest capacity that works") into $O(\log n)$ decision problems ("does this capacity work?").
- **Patience sorting**: an $O(n \log n)$ method for the longest increasing subsequence that keeps the smallest possible tail for each subsequence length and binary searches where each new value belongs; the number of piles is the length of the subsequence.

## In this chapter

Each solution is a JavaScript function; run it with Node by adding a small driver, or paste it into an online judge.

- [Find Minimum in Rotated Sorted Array](find-minimum-in-rotated-sorted-array.js): returns early when the array is not rotated (`nums[low] < nums[high]`), then binary searches for the inflection point where a value is larger than its successor, comparing `mid` with `nums[0]` to decide which side the inflection is on.
- [Search in Rotated Sorted Array](search-in-rotated-sorted-array.js): a two-phase approach. A first binary search finds the index of the smallest value (the rotation offset); then an ordinary exact-match search runs over "virtual" indices, each mapped to a real one by `(index + offset) % n`.
- [Longest Increasing Subsequence](liss.js): the patience-sorting method. It keeps the smallest possible tail for each subsequence length in `piles`, and for each new number binary searches for the first pile whose top is at least that number, replacing it or starting a new pile. The number of piles is the length of the longest increasing subsequence, in $O(n \log n)$.
