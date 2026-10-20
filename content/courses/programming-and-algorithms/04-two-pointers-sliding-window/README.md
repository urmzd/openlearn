# Two pointers and sliding window

[Arrays and hashing](../03-arrays-hashing/README.md) removed nested loops by spending memory on a hash map. This chapter removes them by exploiting structure instead. Two pointers use **sorted order** or **monotonic structure** to collapse $O(n^2)$ into $O(n)$: because the array is ordered, moving one pointer in a known direction is guaranteed to move a quantity (a sum, an area) in a known direction, so whole rows of the brute-force search can be skipped. A sliding window maintains a **valid window** over a contiguous subarray, expanding on the right and shrinking on the left, so each element enters and leaves the window at most once.

The chapter covers five patterns: opposite-end pointers, same-direction (fast and slow) pointers, fixed-size windows, variable-size windows, and the reduction of $k$-sum problems to two pointers. The thread running through all of them is the pointer-movement invariant: before you write the loop, you should be able to say why moving this pointer cannot skip an answer.

> **Learning objectives**
> - Use opposite-end pointers on a sorted array and justify each pointer move with an invariant.
> - Detect cycles and find midpoints with fast and slow pointers.
> - Write fixed-size and variable-size sliding windows, and know where to update the answer for "longest" versus "shortest."
> - Reduce 3Sum and its relatives to a two-pointer sweep while skipping duplicates.

## Before you begin

Read [Arrays and hashing](../03-arrays-hashing/README.md) first; several problems here (Two Sum II, 3Sum) are sorted-array variants of problems solved there with a map. You should also be comfortable with sorting and its $O(n \log n)$ cost.

## Opposite-end pointers

On a sorted array where you need a pair with some property (a sum, an area, a palindrome check), start one pointer at each end and move them toward each other:

```
left, right = 0, len(arr) - 1
while left < right:
    val = f(arr[left], arr[right])
    if val == target: return answer
    elif val < target: left += 1
    else: right -= 1
```

The order is what makes this safe: if the sum is too small, no pair using `arr[left]` with anything left of `right` can be larger, so `left` can advance. Two Sum II (sorted), Container With Most Water, 3Sum, and Trapping Rain Water all use it.

Container With Most Water is the classic test of the invariant. The area between two lines is the shorter height times the width. Always move the shorter side: moving the taller side can never increase the area, since the width decreases and the height is still bounded by the shorter side.

## Fast and slow pointers

Two pointers can also move in the same direction at different speeds. Moving `slow` one step and `fast` two steps leaves `slow` at the midpoint when `fast` reaches the end, and if the list has a cycle, the two eventually meet inside it:

```
slow, fast = head, head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
# slow is at midpoint (or meeting point in cycle)
```

Linked List Cycle and Middle of Linked List use the two-speed form; Remove Duplicates from Sorted Array and Remove Element use a read pointer and a write pointer moving in the same direction over an array. [Linked lists](../06-linked-lists/README.md) returns to this pattern in detail.

## Fixed-size windows

For "find the max, min, or average of all subarrays of size $k$," keep a running aggregate and slide it: add the element entering on the right, remove the one leaving on the left:

```
window_sum = sum(arr[:k])
best = window_sum
for i in range(k, len(arr)):
    window_sum += arr[i] - arr[i - k]  # slide: add right, remove left
    best = max(best, window_sum)
```

Maximum Average Subarray and Max Sum Subarray of Size K follow the template directly. Sliding Window Maximum needs a deque to keep the window's maximum available in $O(1)$.

## Variable-size windows

For "find the shortest or longest subarray satisfying condition $X$," the window grows on the right every step and shrinks on the left while it is invalid:

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

Minimum Size Subarray Sum, Longest Substring Without Repeating Characters, Minimum Window Substring, and Longest Repeating Character Replacement are the standard problems. One distinction decides whether your answer is right. For "longest valid," shrink while the window is invalid and update the answer after the while loop, when the window is valid again. For "shortest valid," shrink while the window is still valid and update the answer inside the while loop, before each shrink.

## Three pointers: the 3Sum reduction

$k$-sum problems reduce to two pointers after fixing elements. Sort, fix one element, and run the opposite-end sweep on the rest, skipping repeated values so no triplet is reported twice:

```
sort(arr)
for i in range(len(arr)):
    if i > 0 and arr[i] == arr[i-1]: continue  # skip duplicates
    left, right = i + 1, len(arr) - 1
    # standard two-pointer on remaining
```

The total is $O(n^2)$, down from $O(n^3)$. 3Sum, 3Sum Closest, and 4Sum all work this way.

## What goes wrong

Four mistakes cause most failures in this family:

1. Forgetting to sort before using opposite-end pointers.
2. Off-by-one errors in window boundaries: decide whether the window is `[left, right]` or `[left, right)` and stick to it.
3. Not handling duplicates in 3Sum: always skip `arr[i] == arr[i-1]`, and skip repeated values after recording a triplet.
4. The wrong shrink condition: for "longest," shrink when invalid; for "shortest," shrink while valid.

Interviewers favor different variants. As a rough guide:

| Company | Favorite variant | Difficulty |
|---------|-----------------|------------|
| Google | sliding window with complex state | Hard |
| Meta | 3Sum, substring problems | Medium |
| Amazon | minimum window substring | Medium-Hard |
| NVIDIA | parallel-friendly window operations | Medium |
| Citadel | optimized streaming aggregation | Hard |

## In this chapter

Each solution is a JavaScript function; run it with Node by adding a small driver, or paste it into an online judge.

- [3Sum](3sum.js): sorts the array, returns early for fewer than three elements or an array of identical values, then fixes each element and sweeps the rest with two pointers. After recording a triplet it moves both pointers past repeated values, so each triplet appears once.
- [Container With Most Water](container-with-most-water.js): starts with the widest container and repeatedly moves the pointer at the shorter line inward, tracking the best `min(height) * width`. The comments record the reasoning that turns the problem from a sort into a greedy two-pointer sweep.
