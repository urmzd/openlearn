# 02-two-pointers-sliding-window

## Summary
- Contains: `3sum.js`, `container-with-most-water.js`.
- JavaScript solutions emphasizing two-pointer sweeps and windowed scans.

## Key takeaways
- Sorted arrays + two pointers reduce nested loops to linear scans.
- Always reason about pointer movement invariants to avoid missing cases.

## How to run
- These are solution functions. Run with Node by adding a small driver, or paste into an online judge.

---

# Two Pointers & Sliding Window Patterns

## Core Insight

Two pointers exploit **sorted order** or **monotonic structure** to collapse O(n²) into O(n). Sliding window maintains a **valid window** over a contiguous subarray, expanding right and shrinking left.

## Pattern 1: Opposite-End Two Pointers

**When to use**: Sorted array, find pairs with some property (sum, area, palindrome check).

**Template**:
```
left, right = 0, len(arr) - 1
while left < right:
    val = f(arr[left], arr[right])
    if val == target: return answer
    elif val < target: left += 1
    else: right -= 1
```

**Interview problems**: Two Sum II (sorted), Container With Most Water, 3Sum, Trapping Rain Water

**Key insight for Container With Most Water**: Always move the shorter side — moving the taller side can never increase the area since width decreases and height is bounded by the shorter side.

## Pattern 2: Same-Direction Two Pointers (Fast/Slow)

**When to use**: Linked list cycle detection, finding middle, remove duplicates in-place.

**Template**:
```
slow, fast = head, head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
# slow is at midpoint (or meeting point in cycle)
```

**Interview problems**: Linked List Cycle, Middle of Linked List, Remove Duplicates from Sorted Array, Remove Element

## Pattern 3: Fixed-Size Sliding Window

**When to use**: "Find max/min/average of all subarrays of size k."

**Template**:
```
window_sum = sum(arr[:k])
best = window_sum
for i in range(k, len(arr)):
    window_sum += arr[i] - arr[i - k]  # slide: add right, remove left
    best = max(best, window_sum)
```

**Interview problems**: Maximum Average Subarray, Max Sum Subarray of Size K, Sliding Window Maximum (with deque)

## Pattern 4: Variable-Size Sliding Window

**When to use**: "Find shortest/longest subarray satisfying condition X."

**Template**:
```
left = 0
state = initial_state
for right in range(len(arr)):
    update state with arr[right]  # expand
    while state is invalid:
        update state removing arr[left]  # shrink
        left += 1
    update answer
```

**Interview problems**: Minimum Size Subarray Sum, Longest Substring Without Repeating Characters, Minimum Window Substring, Longest Repeating Character Replacement

**Critical distinction**: For "longest valid" — update answer after the while loop. For "shortest valid" — update answer inside the while loop.

## Pattern 5: Three Pointers (3Sum Reduction)

**When to use**: K-sum problems, reducing to two-pointer after fixing one element.

**Template**:
```
sort(arr)
for i in range(len(arr)):
    if i > 0 and arr[i] == arr[i-1]: continue  # skip duplicates
    left, right = i + 1, len(arr) - 1
    # standard two-pointer on remaining
```

**Interview problems**: 3Sum, 3Sum Closest, 4Sum

## Company Targeting

| Company | Favorite Variant | Difficulty |
|---------|-----------------|------------|
| Google | Sliding window with complex state | Hard |
| Meta | 3Sum, substring problems | Medium |
| Amazon | Min window substring | Medium-Hard |
| NVIDIA | Parallel-friendly window ops | Medium |
| Citadel | Optimized streaming aggregation | Hard |

## Common Mistakes

1. **Forgetting to sort** for opposite-end two pointers
2. **Off-by-one** in window boundaries (is it `[left, right]` or `[left, right)`?)
3. **Not handling duplicates** in 3Sum — always skip `arr[i] == arr[i-1]`
4. **Wrong shrink condition** — for "longest," shrink when invalid; for "shortest," shrink when valid
