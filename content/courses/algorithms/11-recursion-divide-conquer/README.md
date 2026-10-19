# 11-recursion-divide-conquer

## Summary
- Contains: `example-1.py`, `example-3.py`.
- Python exercises showcasing recursion and divide-and-conquer patterns.

## Key takeaways
- Base cases and input reduction are the core of correct recursion.
- Divide-and-conquer trades depth for clarity and often improves asymptotics.

## How to run
- `python3 11-recursion-divide-conquer/example-1.py`
- `python3 11-recursion-divide-conquer/example-3.py`

---

# Recursion & Divide-and-Conquer Patterns

## Core Insight

Divide and conquer splits a problem into **independent subproblems**, solves them recursively, and **combines** the results. Unlike DP, subproblems don't overlap. The Master Theorem gives you the complexity.

## Pattern 1: Classic Divide and Conquer

**Template**:
```python
def solve(arr, left, right):
    if left == right:
        return base_case(arr[left])
    mid = (left + right) // 2
    left_result = solve(arr, left, mid)
    right_result = solve(arr, mid + 1, right)
    return combine(left_result, right_result)
```

**Interview problems**: Merge Sort, Maximum Subarray (D&C variant), Count Inversions

## Pattern 2: Merge Sort Framework

**When to use**: Problems that need a global property computable during the merge step.

**Template**:
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)  # counting happens here
```

**Interview problems**: Sort an Array, Count of Smaller Numbers After Self, Reverse Pairs, Count Inversions

**Count Inversions insight**: During merge, when `right[j] < left[i]`, all remaining elements in left (from i to end) form inversions with `right[j]`.

## Pattern 3: Quick Select (Kth Element)

**When to use**: Find kth smallest/largest element in O(n) average.

**Template**:
```python
def quickselect(arr, k):
    pivot = arr[random.randint(0, len(arr)-1)]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    if k <= len(left):
        return quickselect(left, k)
    elif k <= len(left) + len(mid):
        return pivot
    else:
        return quickselect(right, k - len(left) - len(mid))
```

**Interview problems**: Kth Largest Element, Top K Frequent Elements, K Closest Points to Origin

## Pattern 4: Binary Exponentiation

**When to use**: Compute x^n in O(log n).

**Template**:
```python
def power(x, n):
    if n == 0: return 1
    if n < 0: return 1 / power(x, -n)
    half = power(x, n // 2)
    if n % 2 == 0:
        return half * half
    else:
        return half * half * x
```

**Extends to**: Matrix exponentiation (Fibonacci in O(log n)), modular exponentiation.

## Pattern 5: Closest Pair / Geometric D&C

**When to use**: Geometric problems where brute force is O(n²).

**Template**:
```
Sort by x-coordinate
Split into left/right halves
Recursively find closest pair in each half
Check strip of width 2*min_dist around midline
Combine results
```

**Interview problems**: Closest Pair of Points

## The Master Theorem (Quick Reference)

For `T(n) = a * T(n/b) + O(n^d)`:

| Condition | Complexity |
|-----------|-----------|
| d > log_b(a) | O(n^d) |
| d = log_b(a) | O(n^d * log n) |
| d < log_b(a) | O(n^(log_b(a))) |

| Algorithm | a | b | d | Result |
|-----------|---|---|---|--------|
| Merge Sort | 2 | 2 | 1 | O(n log n) |
| Binary Search | 1 | 2 | 0 | O(log n) |
| Strassen | 7 | 2 | 2 | O(n^2.81) |
| Karatsuba | 3 | 2 | 1 | O(n^1.59) |

## Company Targeting

| Company | Favorite Variant | Difficulty |
|---------|-----------------|------------|
| Google | Count inversions, closest pair | Hard |
| Meta | Quick select, merge sort variants | Medium |
| Jane Street | Matrix exponentiation | Hard |
| Two Sigma | D&C optimization problems | Hard |
| Renaissance | Strassen-style optimizations | Hard |
