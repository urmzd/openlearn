# Linked lists

Linked list problems are rarely about clever algorithms. They test **pointer manipulation** and **edge-case handling**: whether you can rewire `next` pointers without losing part of the list, and whether your code still works when the list is empty, has one node, or has the operation land on its head. The good news is that a few primitives cover almost everything. Dummy head nodes, fast and slow pointers, and in-place reversal, plus merging and digit-by-digit arithmetic, combine in twos and threes to solve nearly every linked list problem you will meet.

This chapter presents each primitive with a template, closes with an edge-case checklist to run against any solution, and then walks through a recursive solution to Add Two Numbers.

> **Learning objectives**
> - Use a dummy head node to remove special cases when the head can change.
> - Find the middle of a list, detect a cycle, and locate the cycle's start with fast and slow pointers.
> - Reverse a whole list and a sublist in place.
> - Merge sorted lists, and add numbers stored as digit lists with correct carry handling.

## Before you begin

The fast and slow pointer idea was introduced in [Two pointers and sliding window](../04-two-pointers-sliding-window/README.md). You should be comfortable with references (or pointers) and with recursion. The doubly linked list behind an LRU cache is built in full in [the intrusive list and LRU](../15-systems-data-structures/03-intrusive-list-and-lru.md) in this book's systems chapter.

## Dummy head nodes

Any problem where the head might change (insertion, deletion, merging) gets simpler with a dummy node in front of the head. Every real node then has a predecessor, so there is no "if head is null" or "if we are deleting the head" branch, and the answer is simply `dummy.next`:

```
dummy = ListNode(0)
dummy.next = head
prev = dummy
# ... operations ...
return dummy.next
```

Remove Nth Node From End of List, Merge Two Sorted Lists, and Partition List all use it.

## Fast and slow pointers

For cycle detection, finding the middle, or finding the $k$th node from the end, move two pointers at different speeds. Floyd's algorithm detects a cycle when the pointers meet, and then finds the cycle's start: reset one pointer to the head and advance both one step at a time; they meet again exactly at the start of the cycle.

```
slow, fast = head, head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
        # Cycle detected. Find start:
        slow = head
        while slow != fast:
            slow = slow.next
            fast = fast.next
        return slow  # cycle start
```

Without a cycle, the same loop leaves `slow` at the midpoint when `fast` runs off the end:

```
slow, fast = head, head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
# slow is at midpoint
```

Linked List Cycle, Linked List Cycle II, Middle of the Linked List, and Palindrome Linked List are the standard problems. For the $k$th node from the end, start `fast` $k$ nodes ahead and move both at the same speed.

## In-place reversal

Reversing a list walks it once, pointing each node back at its predecessor. Save `next` before overwriting it, or the rest of the list is lost:

```
prev = None
curr = head
while curr:
    nxt = curr.next
    curr.next = prev
    prev = curr
    curr = nxt
return prev  # new head
```

Reversing only the nodes between positions $m$ and $n$ combines reversal with a dummy head. Walk `pre` to the node before position $m$; then repeatedly take the node after `tail` and move it to the front of the sublist:

```
dummy = ListNode(0, head)
pre = dummy
for _ in range(m - 1):
    pre = pre.next
tail = pre.next
for _ in range(n - m):
    tmp = pre.next
    pre.next = tail.next
    tail.next = tail.next.next
    pre.next.next = tmp
```

Reverse Linked List, Reverse Linked List II, and Reverse Nodes in k-Group use these templates.

## Merging sorted lists

To combine two sorted lists, repeatedly attach the smaller head to a growing result behind a dummy node, and attach whatever remains when one list runs out:

```
dummy = ListNode(0)
curr = dummy
while l1 and l2:
    if l1.val <= l2.val:
        curr.next = l1
        l1 = l1.next
    else:
        curr.next = l2
        l2 = l2.next
    curr = curr.next
curr.next = l1 or l2
return dummy.next
```

For $k$ lists, keep the current heads in a min-heap of size $k$, for $O(N \log k)$ over $N$ total nodes. Merge Two Sorted Lists and Merge k Sorted Lists are the classic problems.

## Digit-by-digit arithmetic

When numbers are stored as linked lists of digits in reverse order, addition is the grade-school algorithm. Keep going while either list has digits *or* a carry remains, which handles lists of unequal length and a final carry in one condition:

```
dummy = ListNode(0)
curr = dummy
carry = 0
while l1 or l2 or carry:
    val = carry
    if l1: val += l1.val; l1 = l1.next
    if l2: val += l2.val; l2 = l2.next
    carry, digit = divmod(val, 10)
    curr.next = ListNode(digit)
    curr = curr.next
return dummy.next
```

Add Two Numbers and Add Two Numbers II (digits in forward order, usually solved by reversing first or using stacks) are the standard problems. The two things to manage carefully are the carry and unequal lengths.

## An edge-case checklist

Run every linked list solution against these inputs before you trust it:

- an empty list (`head` is null)
- a single node
- two nodes
- odd and even lengths
- a list with a cycle
- an operation on the head node
- an operation on the tail node

Interviewers favor different variants. As a rough guide:

| Company | Favorite variant | Difficulty |
|---------|-----------------|------------|
| Google | Reverse Nodes in k-Group | Hard |
| Meta | Merge k Sorted Lists | Medium-Hard |
| Amazon | LRU cache (doubly linked list + hash map) | Medium |
| Apple | in-place operations (memory) | Medium |
| Palantir | custom linked structures | Medium |

## Key terms

- **Dummy head node**: a placeholder node placed before a linked list's real head so that every real node has a predecessor; insertions, deletions, and merges then need no special case for the head, and the result is `dummy.next`.
- **Floyd's cycle detection / tortoise and hare**: detecting a cycle in a linked list by advancing one pointer one step and another two steps until they meet, then finding the cycle's start by restarting one pointer at the head and advancing both one step at a time until they meet again.
- **In-place list reversal**: reversing a linked list in one pass with $O(1)$ extra space by pointing each node's `next` back at its predecessor, saving the original `next` first so the rest of the list is not lost.
- **k-way merge**: merging $k$ sorted lists by keeping their current heads in a min-heap of size $k$ and repeatedly taking the smallest, in $O(N \log k)$ time for $N$ elements in total.

## In this chapter

- [Add Two Numbers](add-two-numbers.js): a recursive version of digit-by-digit addition. The `helper` function adds the current digits and the carry, appends a node for `sum % 10` to the tail, remembers the head on the first call, and recurses on both `next` pointers with the new carry, treating a missing node as $0$. It stops when the carry is zero and both lists are exhausted. The file is a solution function for Node (it begins with a `node` shebang and prints its intermediate values with `console.log`); add a driver that defines `ListNode` and builds the input lists to run it, or paste it into an online judge.
