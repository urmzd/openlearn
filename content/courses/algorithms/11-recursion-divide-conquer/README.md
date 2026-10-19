# Recursion and divide-and-conquer

Correct recursion rests on two things: a base case that stops it, and an input that gets strictly smaller on every call. Divide-and-conquer is the recursive strategy that pushes this furthest. It splits a problem into **independent subproblems**, solves them recursively, and **combines** the results. Unlike dynamic programming, the subproblems do not overlap, so there is nothing to memoize; the payoff instead is that splitting often improves the asymptotics, as merge sort's $O(n \log n)$ improves on the $O(n^2)$ of simple sorts. The Master Theorem tells you exactly when and by how much.

This chapter covers the classic split-and-combine template, the merge-sort framework for counting global properties during the merge, quickselect, binary exponentiation, and geometric divide-and-conquer, then states the Master Theorem with worked cases. It ends with two Python examples.

> **Learning objectives**
> - Write a divide-and-conquer recursion with a correct base case, split, and combine step.
> - Count inversions and similar global properties during merge sort's merge.
> - Find the $k$th smallest element in expected $O(n)$ with quickselect.
> - Compute $x^n$ in $O(\log n)$ multiplications.
> - Solve recurrences of the form $T(n) = aT(n/b) + O(n^d)$ with the Master Theorem.

## Before you begin

You need to be comfortable writing recursive functions and reading big-O notation. Merge sort and binary search, both used here as examples, appear in [Binary search](../03-binary-search/README.md); the contrast with overlapping subproblems is developed in [Dynamic programming](../07-dynamic-programming/README.md).

## The classic template

Split the range at the midpoint, solve each half, and combine. The base case is a range of one element:

```python
def solve(arr, left, right):
    if left == right:
        return base_case(arr[left])
    mid = (left + right) // 2
    left_result = solve(arr, left, mid)
    right_result = solve(arr, mid + 1, right)
    return combine(left_result, right_result)
```

Merge Sort, the divide-and-conquer variant of Maximum Subarray, and Count Inversions all follow it. For maximum subarray, the combine step takes the best of the left answer, the right answer, and the best subarray crossing the midpoint, found by extending outward from `mid` in both directions.

## The merge sort framework

Some problems need a global property that can be computed during the merge step, while both halves are sorted:

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)  # counting happens here
```

Counting inversions is the model case. During the merge, when `right[j] < left[i]`, every remaining element of `left` (from `i` to the end) forms an inversion with `right[j]`, so you add `len(left) - i` in one step instead of counting pairs individually. Sort an Array, Count of Smaller Numbers After Self, Reverse Pairs, and Count Inversions use this framework.

## Quickselect

To find the $k$th smallest or largest element without sorting, partition around a random pivot and recurse into only the side that contains the answer:

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

Because only one side is explored, the expected work is $n + n/2 + n/4 + \dots = O(n)$. Kth Largest Element in an Array, Top K Frequent Elements, and K Closest Points to Origin are the standard problems.

## Binary exponentiation

To compute $x^n$, square the half power instead of multiplying $n$ times, which takes $O(\log n)$ multiplications:

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

The same idea extends to matrix exponentiation (Fibonacci numbers in $O(\log n)$) and to modular exponentiation.

## Geometric divide-and-conquer

Geometric problems whose brute force is $O(n^2)$ often split cleanly in space. The closest pair of points is the classic:

```
Sort by x-coordinate
Split into left/right halves
Recursively find closest pair in each half
Check strip of width 2*min_dist around midline
Combine results
```

Only a constant number of points in the strip need to be compared with each point, so the combine step is linear and the whole algorithm is $O(n \log n)$.

## The Master Theorem

The running time of a divide-and-conquer algorithm that makes $a$ recursive calls on inputs of size $n/b$ and does $O(n^d)$ work to split and combine satisfies

$$T(n) = a\,T(n/b) + O(n^d).$$

The answer depends on how $d$ compares with $\log_b a$, which measures how fast the number of subproblems grows:

| Condition | Complexity |
|-----------|-----------|
| $d > \log_b a$ | $O(n^d)$ |
| $d = \log_b a$ | $O(n^d \log n)$ |
| $d < \log_b a$ | $O(n^{\log_b a})$ |

Four familiar algorithms show each case:

| Algorithm | $a$ | $b$ | $d$ | Result |
|-----------|---|---|---|--------|
| merge sort | 2 | 2 | 1 | $O(n \log n)$ |
| binary search | 1 | 2 | 0 | $O(\log n)$ |
| Strassen | 7 | 2 | 2 | $O(n^{2.81})$ |
| Karatsuba | 3 | 2 | 1 | $O(n^{1.59})$ |

Strassen and Karatsuba both win by reducing $a$: seven half-size matrix products instead of eight, three half-size integer products instead of four.

Interviewers favor different variants. As a rough guide:

| Company | Favorite variant | Difficulty |
|---------|-----------------|------------|
| Google | count inversions, closest pair | Hard |
| Meta | quickselect, merge sort variants | Medium |
| Jane Street | matrix exponentiation | Hard |
| Two Sigma | divide-and-conquer optimization problems | Hard |
| Renaissance | Strassen-style optimizations | Hard |

## In this chapter

Both examples are Python programs; run them with `python3 11-recursion-divide-conquer/example-1.py` and `python3 11-recursion-divide-conquer/example-3.py`.

- [Longest monochromatic tree path](example-1.py): an outline, in Python, of a divide-and-conquer solution for the longest path whose nodes all share a color in a rooted tree. `divide` extends the current path when a node has its parent's color and restarts it otherwise, recurses into every subtree, and `conquer` joins the root with its two longest subtree paths (taking only two prevents revisiting nodes). The helper functions are left as stubs to fill in.
- [Maximum circular subarray](example-3.py): the divide-and-conquer maximum subarray, with the crossing-subarray step that extends left and right from the midpoint, applied to every rotation of the array to find the best circular subarray. Trying all $n$ rotations makes it $O(n^2 \log n)$; Kadane's algorithm in [Dynamic programming](../07-dynamic-programming/README.md) can solve the same problem in $O(n)$ by also tracking the minimum subarray.
