# The Online Assessment

The first technical gate in Anthropic's infrastructure loop is a 90-minute online assessment on CodeSignal. It looks like a familiar coding screen, but it is graded differently from most. You will typically get two problems, or a single problem with four progressive levels of increasing difficulty, and Python is the most commonly reported language, although others are accepted. What separates a pass from a fail is rarely the algorithm. It is whether the code you write in 90 minutes looks like code you would ship.

This section walks through what the graders look for, then works through the three problems candidates most often report: a multi-level LRU cache, a task management system with dependencies, and an in-memory database with TTLs and compression. For each one you will see a clean reference solution and the reasoning behind it, so you can explain your design choices rather than just reproduce them.

> **Learning objectives**
> - Describe the production-quality criteria the assessment rewards and how they differ from a typical algorithm screen.
> - Implement an LRU cache three ways: with `OrderedDict`, from scratch with a doubly linked list and hash map, and thread-safe with a lock.
> - Design a task scheduler with priorities, DAG dependencies, cascading cancellation, and cycle detection.
> - Build an in-memory key-value store with prefix scans, TTL expiry, and value compression.
> - Plan your 90 minutes so each level is complete and clean before you layer on the next.

## What the graders look for

This is not a typical leetcode assessment. Anthropic evaluates the code as a reviewer would evaluate a pull request. **Production quality** comes first: clean APIs, meaningful variable names, and docstrings where they help. **Thread safety** matters because many of the problems explicitly or implicitly require handling concurrent access. **Error handling** covers edge cases, input validation, and graceful failure modes. You should be ready to state the **time and space complexity** of each operation. And brief **comments** should explain non-obvious design decisions.

Hold those criteria in mind as you read the solutions below. Each one is deliberately plain: the interesting part is not cleverness but the fact that every operation is obviously correct and its cost is obvious too.

## The multi-level LRU cache

The most frequently reported problem is a least-recently-used cache, delivered in levels that each build on the last. An LRU cache holds at most `capacity` entries; reading or writing a key marks it as most recently used, and inserting past capacity evicts the least recently used entry.

At the first level you may use `collections.OrderedDict`, which keeps insertion order and can move a key to the end in $O(1)$. The front of the dictionary is therefore always the least recently used key.

```python
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: str) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

The second level asks you to build the same thing from scratch, without `OrderedDict`. The classic structure is a hash map from key to node, plus a doubly linked list that holds the recency order. Two dummy nodes, `head` and `tail`, remove every special case for an empty list: the least recently used node is always `head.next`, and new or touched nodes go just before `tail`. Both `get` and `put` are $O(1)$, because the map finds the node and the links let you unlink and relink it without walking the list.

```python
class Node:
    def __init__(self, key: str, value: str):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: dict[str, Node] = {}
        self.head = Node("", "")  # dummy head
        self.tail = Node("", "")  # dummy tail
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_end(self, node: Node) -> None:
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    def get(self, key: str) -> Optional[str]:
        if key not in self.cache:
            return None
        node = self.cache[key]
        self._remove(node)
        self._add_to_end(node)
        return node.value

    def put(self, key: str, value: str) -> None:
        if key in self.cache:
            self._remove(self.cache[key])
        node = Node(key, value)
        self._add_to_end(node)
        self.cache[key] = node
        if len(self.cache) > self.capacity:
            lru = self.head.next
            self._remove(lru)
            del self.cache[lru.key]
```

Notice why each node stores its own `key`: when you evict `head.next`, you need the key to delete the map entry too.

The third level adds thread safety. The simplest correct answer is a single `threading.Lock` held for the whole of each operation. That matters even for `get`, because `get` *mutates* the recency order; two threads interleaving `move_to_end` and `popitem` can corrupt the structure.

```python
import threading

class ThreadSafeLRUCache:
    def __init__(self, capacity: int):
        self.lock = threading.Lock()
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str) -> Optional[str]:
        with self.lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: str, value: str) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
```

The fourth level is open-ended: add comprehensive error handling, TTL support, or metrics. A sensible answer validates `capacity` at construction, stores an expiry time alongside each value and treats expired entries as misses, and counts hits, misses, and evictions under the same lock.

## The task management system

The second commonly reported problem is a task scheduler. It must support task creation with priorities (HIGH, MEDIUM, LOW); assignment of tasks to available workers; DAG dependencies, where task B cannot start until task A completes; cascading cancellation, where cancelling a task cancels everything that depends on it; and circular dependency detection, rejecting any submission that would create a cycle.

The data structures follow directly from the requirements. A priority queue (a min-heap) orders runnable tasks, which is why the priority enum below maps HIGH to the smallest value. An adjacency list represents the dependency graph. And a topological sort or depth-first search detects cycles. Cascading cancellation is a traversal of the same adjacency list from the cancelled task to all of its dependents.

```python
from enum import IntEnum
from collections import defaultdict

class Priority(IntEnum):
    HIGH = 0
    MEDIUM = 1
    LOW = 2

class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

def has_cycle(graph: dict, start: str, target: str) -> bool:
    """Check if adding edge start -> target creates a cycle."""
    visited = set()
    stack = [target]
    while stack:
        node = stack.pop()
        if node == start:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(graph.get(node, []))
    return False
```

The cycle check is the part worth being able to explain. Adding an edge `start -> target` creates a cycle exactly when `start` is already reachable from `target`. So the function runs an iterative DFS from `target` and reports a cycle if it ever reaches `start`. It costs $O(V + E)$ per check in the worst case, and the `visited` set keeps it from looping on diamonds in the graph.

## The in-memory database

The third reported problem is a key-value store with basic `SET`, `GET`, and `DELETE`; filtered scans that return all keys matching a prefix or pattern; time-to-live expiry; and compression of values on storage with decompression on retrieval.

The reference design stores each value as a tuple of compressed bytes and an optional absolute expiry timestamp. Expiry is **lazy**: `get` checks the timestamp and deletes the entry if it has expired, and `scan` skips expired entries. That keeps every write $O(1)$ without a background sweeper, at the cost of expired entries occupying memory until they are touched.

```python
import time
import zlib

class InMemoryDB:
    def __init__(self):
        self.store: dict[str, tuple[bytes, Optional[float]]] = {}

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        compressed = zlib.compress(value.encode())
        expiry = time.time() + ttl if ttl else None
        self.store[key] = (compressed, expiry)

    def get(self, key: str) -> Optional[str]:
        if key not in self.store:
            return None
        data, expiry = self.store[key]
        if expiry and time.time() > expiry:
            del self.store[key]
            return None
        return zlib.decompress(data).decode()

    def delete(self, key: str) -> bool:
        return self.store.pop(key, None) is not None

    def scan(self, prefix: str) -> list[str]:
        now = time.time()
        results = []
        for key, (_, expiry) in self.store.items():
            if key.startswith(prefix):
                if expiry is None or now <= expiry:
                    results.append(key)
        return results
```

The prefix scan is $O(n)$ over all keys. If an interviewer pushes on that, the follow-up answer is a sorted structure or a trie keyed by prefix.

> **Warning:** `if ttl` treats a TTL of `0` as "no expiry." If the problem statement defines zero as "expire immediately," test `ttl is not None` instead and say so in a comment. Small semantic choices like this are exactly what the graders read for.

## Using your 90 minutes

Ninety minutes goes fast when you are writing production-quality code, so practice with a timer. Start with the simple version and then layer: get Level 1 working cleanly before you add complexity, because a clean, complete Level 2 beats a broken Level 4. Write tests inline; even brief `assert` statements show a testing instinct. Name things well (`_remove_node`, not `_rm`; `is_expired`, not `chk`). And state complexity, with a comment like `# O(1) amortized` on any operation where it is not obvious.

## Where this leads

The thread-safety theme of the LRU cache continues in [the first live coding round](02-coding-round-1.md), which moves from locks to asyncio, and in [the second coding round](04-coding-round-2.md), which returns to concurrent data structures. The background on locks, the GIL, and asyncio is in [Concurrency patterns](06-concurrency-patterns.md).

## Key terms

- **LRU cache**: a bounded cache that evicts the least recently used entry when it is full.
- **Dummy (sentinel) node**: a placeholder node at the head or tail of a linked list that removes empty-list special cases.
- **DAG dependency**: a directed, acyclic "must finish before" relation between tasks.
- **Cascading cancellation**: cancelling a task together with every task that transitively depends on it.
- **Lazy expiry**: removing an expired entry only when it is next read or scanned, rather than with a background sweeper.

## Summary

- The assessment grades code like a pull request: clean APIs, thread safety, error handling, complexity, and comments.
- The LRU cache comes in levels: `OrderedDict`, then a hash map with a doubly linked list for $O(1)$ operations, then a lock around every operation, then TTLs, metrics, or error handling.
- The task scheduler needs a min-heap for priority, an adjacency list for dependencies, and a reachability check to reject cycles.
- The in-memory database stores compressed values with absolute expiry times and expires entries lazily.
- Finish each level cleanly before starting the next, and annotate complexity as you go.

## Review questions

1. Why does a thread-safe LRU cache need to lock `get`, not just `put`?
   <details><summary>Answer</summary><code>get</code> mutates the recency order by moving the key to the end. Concurrent <code>move_to_end</code> and <code>popitem</code> calls can corrupt the ordering or evict the wrong key, so every operation that touches the order must hold the lock.</details>
2. In the from-scratch LRU cache, why does each node store its key as well as its value?
   <details><summary>Answer</summary>On eviction you remove <code>head.next</code> from the list, and you also need to delete its entry from the hash map. The node's key is the only way to find that entry without scanning the map.</details>
3. How does `has_cycle(graph, start, target)` decide whether adding `start -> target` creates a cycle?
   <details><summary>Answer</summary>A cycle appears exactly when <code>start</code> is already reachable from <code>target</code>. The function runs an iterative DFS from <code>target</code> and returns true if it reaches <code>start</code>, using a visited set to avoid revisiting nodes.</details>
4. What is the tradeoff of lazy TTL expiry in the in-memory database?
   <details><summary>Answer</summary>Writes and reads stay simple and $O(1)$ with no background thread, but expired entries keep occupying memory until something reads or scans them. A periodic sweeper or an expiry heap would reclaim memory sooner at the cost of more machinery.</details>
5. You have 20 minutes left and Level 3 works. What should you do next?
   <details><summary>Answer</summary>Make sure what you have is clean, tested with a few asserts, and annotated with complexity, then add one Level 4 feature (for example, TTL support) carefully. A polished partial solution is graded better than a rushed, broken complete one.</details>
