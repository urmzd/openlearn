# 04-linked-lists

## Summary
- Contains: `add-two-numbers.js`.
- JavaScript solution for linked-list arithmetic.

## Key takeaways
- Manage carry carefully and handle unequal list lengths.
- Dummy-head nodes simplify list construction.

## How to run
- This is a solution function. Run with Node by adding a small driver, or paste into an online judge.

---

# Linked List Patterns

## Core Insight

Linked list problems test **pointer manipulation** and **edge case handling**. The key techniques are: dummy head nodes, fast/slow pointers, and in-place reversal. Almost every linked list problem combines 2-3 of these primitives.

## Pattern 1: Dummy Head Node

**When to use**: Any problem where the head might change (insertion, deletion, merging).

**Template**:
```
dummy = ListNode(0)
dummy.next = head
prev = dummy
# ... operations ...
return dummy.next
```

**Interview problems**: Remove Nth Node From End, Merge Two Sorted Lists, Partition List

**Why it matters**: Eliminates all "if head is null" special cases.

## Pattern 2: Fast/Slow Pointers

**When to use**: Cycle detection, finding middle, finding kth from end.

**Template (cycle detection — Floyd's)**:
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

**Template (find middle)**:
```
slow, fast = head, head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
# slow is at midpoint
```

**Interview problems**: Linked List Cycle, Linked List Cycle II, Middle of Linked List, Palindrome Linked List

## Pattern 3: In-Place Reversal

**When to use**: Reverse entire list or reverse a sublist between positions.

**Template (full reversal)**:
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

**Template (reverse between positions m and n)**:
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

**Interview problems**: Reverse Linked List, Reverse Linked List II, Reverse Nodes in K-Group

## Pattern 4: Merge Lists

**When to use**: Combining two or more sorted lists.

**Template**:
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

For k lists: use a min-heap of size k → O(N log k).

**Interview problems**: Merge Two Sorted Lists, Merge K Sorted Lists

## Pattern 5: Two-Number Arithmetic

**When to use**: Numbers stored as linked lists (digits in reverse order).

**Template**:
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

**Interview problems**: Add Two Numbers, Add Two Numbers II

## Company Targeting

| Company | Favorite Variant | Difficulty |
|---------|-----------------|------------|
| Google | Reverse in K-Groups | Hard |
| Meta | Merge K Sorted Lists | Medium-Hard |
| Amazon | LRU Cache (doubly-linked + hash) | Medium |
| Apple | In-place operations (memory) | Medium |
| Palantir | Custom linked structures | Medium |

## Edge Cases Checklist

- [ ] Empty list (head == null)
- [ ] Single node
- [ ] Two nodes
- [ ] Odd vs even length
- [ ] Cycle present
- [ ] Operation on head node
- [ ] Operation on tail node
