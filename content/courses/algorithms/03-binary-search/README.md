# 03-binary-search

## Summary
- Contains: `find-minimum-in-rotated-sorted-array.js`, `liss.js`, `search-in-rotated-sorted-array.js`.
- JavaScript implementations of binary-search style problems (including rotated arrays).

## Key takeaways
- Binary search generalizes beyond exact match to boundary finding.
- Rotated arrays require careful mid comparisons to preserve sorted halves.

## How to run
- These are solution functions. Run with Node by adding a small driver, or paste into an online judge.

---

# Binary Search Patterns

## Core Insight

Binary search works whenever there's a **monotonic predicate** — a function f(x) that flips from false to true (or vice versa) at exactly one point. You don't need a sorted array; you need a search space where you can eliminate half at each step.

## Pattern 1: Classic Binary Search (Find Exact Value)

**Template**:
```
left, right = 0, len(arr) - 1
while left <= right:
    mid = left + (right - left) // 2  # avoid overflow
    if arr[mid] == target: return mid
    elif arr[mid] < target: left = mid + 1
    else: right = mid - 1
return -1
```

**Interview problems**: Binary Search, Search Insert Position

## Pattern 2: Left Boundary (First True)

**When to use**: Find the first element satisfying a condition, lower bound.

**Template**:
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

**Interview problems**: First Bad Version, Find First and Last Position, Search Insert Position

## Pattern 3: Rotated Array Search

**When to use**: Sorted array that's been rotated. One half is always sorted.

**Template**:
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

**Interview problems**: Search in Rotated Sorted Array, Find Minimum in Rotated Sorted Array

## Pattern 4: Binary Search on Answer

**When to use**: "Find the minimum/maximum value X such that some condition holds." The search space is the range of possible answers, not an array.

**Template**:
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

**Interview problems**: Koko Eating Bananas, Capacity to Ship Packages, Split Array Largest Sum, Magnetic Force Between Balls

**This is the most important pattern for hard interviews.** It converts optimization problems into decision problems.

## Pattern 5: Binary Search on Floating Point

**When to use**: Square root, finding a real-valued answer.

**Template**:
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

**Interview problems**: Sqrt(x), Minimize Max Distance to Gas Station

## Company Targeting

| Company | Favorite Variant | Difficulty |
|---------|-----------------|------------|
| Google | Binary search on answer | Hard |
| Amazon | Rotated array variants | Medium |
| Meta | Standard + boundary | Medium |
| Jane Street | BS on real-valued continuous spaces | Hard |
| Two Sigma | BS on answer with complex feasibility | Hard |
| Databricks | Distributed binary search | Hard |

## The Template Decision Tree

```
Is it a sorted array?
├── Yes → exact match? → Pattern 1
│       → first/last occurrence? → Pattern 2
│       → rotated? → Pattern 3
└── No → Can I binary search on the answer?
        ├── Yes → Pattern 4 (most common in hard problems)
        └── No → Probably not a binary search problem
```
