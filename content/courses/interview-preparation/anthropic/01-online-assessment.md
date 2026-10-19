# Round 1: Online Assessment (CodeSignal)

## Format

- **Platform**: CodeSignal
- **Duration**: 90 minutes
- **Structure**: 2 problems (or 1 problem with 4 progressive levels of increasing difficulty)
- **Language**: Python preferred (most reported), but others accepted

## What They Look For

This is NOT a typical leetcode OA. Anthropic evaluates:

- **Production quality**: Clean APIs, meaningful variable names, docstrings where helpful
- **Thread safety**: Many problems explicitly or implicitly require concurrent access handling
- **Error handling**: Edge cases, input validation, graceful failure modes
- **Complexity analysis**: Be ready to state time/space complexity for each operation
- **Comments**: Brief comments explaining non-obvious design decisions

## Reported Problems

### 1. LRU Cache (Multi-Level)

**Level 1**: Implement an LRU cache using `collections.OrderedDict`.
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

**Level 2**: Implement from scratch using a doubly-linked list + hashmap (no OrderedDict).

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

**Level 3**: Add thread safety with `threading.Lock`.

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

**Level 4**: Add comprehensive error handling, TTL support, or metrics.

### 2. Task Management System

Build a task scheduler that supports:
- **Task creation** with priorities (HIGH, MEDIUM, LOW)
- **Worker assignment** -- tasks assigned to available workers
- **DAG dependencies** -- task B depends on task A completing first
- **Cascading cancellation** -- cancelling a task cancels all dependents
- **Circular dependency detection** -- reject task submissions that create cycles

Key data structures:
- Priority queue (min-heap) for scheduling
- Adjacency list for dependency graph
- Topological sort or DFS for cycle detection

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

### 3. In-Memory Database

Build a key-value store with:
- **Basic operations**: SET, GET, DELETE
- **Filtered scans**: Return all keys matching a prefix or pattern
- **TTL (Time-To-Live)**: Keys expire after a configurable duration
- **Compression/Decompression**: Compress values on storage, decompress on retrieval

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

## Preparation Tips

1. **Practice with a timer** -- 90 minutes goes fast when you're writing production-quality code
2. **Start with the simple version, then layer** -- Get Level 1 working cleanly before adding complexity
3. **Write tests inline** -- Even brief `assert` statements show testing instinct
4. **Name things well** -- `_remove_node` not `_rm`, `is_expired` not `chk`
5. **State complexity** -- Add a comment like `# O(1) amortized` for non-obvious operations
