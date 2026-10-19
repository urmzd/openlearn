# Round 4: Coding Round 2 (Live)

## Format

- **Duration**: 60 minutes
- **Setting**: Live video with shared editor
- **Focus**: Data structures, concurrency primitives, systems-level coding
- **Language**: Python preferred, but may use Java/C++ for certain problems

## Reported Problems

### 1. Stack Sampling Profiler -> Trace Events

Given periodic stack samples from a profiler, convert them into trace events (enter/exit) suitable for a flame graph or Chrome trace viewer.

#### Input

```python
# Each sample is a list of function names, bottom of stack first
samples = [
    ["main", "foo", "bar"],       # t=0
    ["main", "foo", "bar"],       # t=1
    ["main", "foo", "baz"],       # t=2
    ["main", "foo", "baz"],       # t=3
    ["main", "qux"],              # t=4
    ["main"],                     # t=5
]
```

#### Output

```python
# Trace events with enter (B) and exit (E) markers
[
    {"name": "main", "ph": "B", "ts": 0},
    {"name": "foo",  "ph": "B", "ts": 0},
    {"name": "bar",  "ph": "B", "ts": 0},
    {"name": "bar",  "ph": "E", "ts": 2},
    {"name": "baz",  "ph": "B", "ts": 2},
    {"name": "baz",  "ph": "E", "ts": 4},
    {"name": "foo",  "ph": "E", "ts": 4},
    {"name": "qux",  "ph": "B", "ts": 4},
    {"name": "qux",  "ph": "E", "ts": 5},
    {"name": "main", "ph": "E", "ts": 5},  # implicit at end
]
```

#### Solution

```python
def samples_to_trace_events(samples: list[list[str]]) -> list[dict]:
    events = []
    prev_sample = []

    for ts, sample in enumerate(samples):
        # Find the divergence point between previous and current sample
        common_depth = 0
        for i in range(min(len(prev_sample), len(sample))):
            if prev_sample[i] == sample[i]:
                common_depth = i + 1
            else:
                break

        # Exit events for functions no longer on stack (reverse order)
        for i in range(len(prev_sample) - 1, common_depth - 1, -1):
            events.append({"name": prev_sample[i], "ph": "E", "ts": ts})

        # Enter events for new functions on stack
        for i in range(common_depth, len(sample)):
            events.append({"name": sample[i], "ph": "B", "ts": ts})

        prev_sample = sample

    # Close remaining stack at end
    final_ts = len(samples)
    for i in range(len(prev_sample) - 1, -1, -1):
        events.append({"name": prev_sample[i], "ph": "E", "ts": final_ts})

    return events
```

#### Edge Case: Recursive Functions

When a function calls itself, matching by name alone breaks. Track by **position in the stack**, not just name.

```python
samples = [
    ["main", "fib", "fib", "fib"],  # t=0: fib calls itself 3 times
    ["main", "fib", "fib"],          # t=1: innermost fib returns
    ["main", "fib"],                 # t=2: middle fib returns
]
```

The divergence-point algorithm already handles this correctly because it compares position-by-position, not by searching for names.

---

### 2. Thread-Safe Min-Heap

Implement a concurrent priority queue that supports:
- `push(item, priority)` -- thread-safe insert
- `pop()` -- thread-safe extract-min, blocks if empty
- `peek()` -- non-blocking read of minimum element

```python
import threading
import heapq

class ThreadSafeMinHeap:
    def __init__(self):
        self._heap: list[tuple[int, int, any]] = []
        self._counter = 0  # tie-breaker for equal priorities
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def push(self, item, priority: int) -> None:
        with self._not_empty:
            heapq.heappush(self._heap, (priority, self._counter, item))
            self._counter += 1
            self._not_empty.notify()

    def pop(self, timeout: Optional[float] = None):
        """Block until an item is available, then return it."""
        with self._not_empty:
            while not self._heap:
                if not self._not_empty.wait(timeout=timeout):
                    raise TimeoutError("Pop timed out")
            _, _, item = heapq.heappop(self._heap)
            return item

    def peek(self):
        """Return the minimum element without removing it, or None if empty."""
        with self._lock:
            if self._heap:
                return self._heap[0][2]
            return None

    def __len__(self):
        with self._lock:
            return len(self._heap)
```

**Discussion points**:
- Why `Condition` over just `Lock`? -- Allows `pop()` to block efficiently until data arrives
- Why a counter tie-breaker? -- Without it, heapq breaks when priorities are equal and items aren't comparable
- `ConcurrentHashMap` vs `Hashtable` (Java context) -- ConcurrentHashMap uses segment-level locking (finer granularity), Hashtable locks entire structure

---

### 3. Top-K Frequent Queries from Concurrent Stream

Given a stream of queries arriving from multiple threads, maintain a real-time top-K frequency count.

```python
import threading
from collections import defaultdict
import heapq

class TopKTracker:
    def __init__(self, k: int):
        self.k = k
        self.counts: dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()

    def record(self, query: str) -> None:
        with self.lock:
            self.counts[query] += 1

    def top_k(self) -> list[tuple[str, int]]:
        with self.lock:
            # Use a min-heap of size K for efficiency
            if len(self.counts) <= self.k:
                return sorted(self.counts.items(), key=lambda x: -x[1])

            heap = []
            for query, count in self.counts.items():
                if len(heap) < self.k:
                    heapq.heappush(heap, (count, query))
                elif count > heap[0][0]:
                    heapq.heapreplace(heap, (count, query))

            return sorted([(q, c) for c, q in heap], key=lambda x: -x[1])
```

**Follow-up discussions**:
- Approximate counting with Count-Min Sketch for massive streams
- Sharding by query hash for parallelism
- Sliding window (time-decayed counts) vs. all-time counts

---

### 4. Trie with Read-Write Locks

Implement a concurrent trie supporting insert, search, and prefix search with read-write locking for better concurrent read performance.

```python
import threading

class TrieNode:
    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_end = False

class ConcurrentTrie:
    def __init__(self):
        self.root = TrieNode()
        self.lock = threading.RWLock()  # Python doesn't have RWLock natively

    # In practice, implement RWLock or use a readers-writer pattern:
    def __init__(self):
        self.root = TrieNode()
        self._readers = 0
        self._readers_lock = threading.Lock()
        self._write_lock = threading.Lock()

    def _acquire_read(self):
        with self._readers_lock:
            self._readers += 1
            if self._readers == 1:
                self._write_lock.acquire()

    def _release_read(self):
        with self._readers_lock:
            self._readers -= 1
            if self._readers == 0:
                self._write_lock.release()

    def insert(self, word: str) -> None:
        with self._write_lock:
            node = self.root
            for ch in word:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            node.is_end = True

    def search(self, word: str) -> bool:
        self._acquire_read()
        try:
            node = self.root
            for ch in word:
                if ch not in node.children:
                    return False
                node = node.children[ch]
            return node.is_end
        finally:
            self._release_read()

    def starts_with(self, prefix: str) -> list[str]:
        """Return all words with the given prefix."""
        self._acquire_read()
        try:
            node = self.root
            for ch in prefix:
                if ch not in node.children:
                    return []
                node = node.children[ch]
            # DFS to collect all words
            results = []
            self._collect(node, prefix, results)
            return results
        finally:
            self._release_read()

    def _collect(self, node: TrieNode, path: str, results: list[str]) -> None:
        if node.is_end:
            results.append(path)
        for ch, child in node.children.items():
            self._collect(child, path + ch, results)
```

---

### 5. Inverted Index

Build an inverted index for full-text search supporting AND/OR queries.

```python
from collections import defaultdict
import re

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.documents: dict[int, str] = {}

    def add_document(self, doc_id: int, text: str) -> None:
        self.documents[doc_id] = text
        tokens = self._tokenize(text)
        for token in tokens:
            self.index[token].add(doc_id)

    def search_and(self, terms: list[str]) -> set[int]:
        """Return doc IDs containing ALL terms."""
        if not terms:
            return set()
        result = self.index.get(terms[0].lower(), set()).copy()
        for term in terms[1:]:
            result &= self.index.get(term.lower(), set())
        return result

    def search_or(self, terms: list[str]) -> set[int]:
        """Return doc IDs containing ANY term."""
        result = set()
        for term in terms:
            result |= self.index.get(term.lower(), set())
        return result

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())
```

## Preparation Tips

1. **Practice diffing algorithms** -- The profiler problem tests your ability to diff two sequences and generate events. Draw it out.
2. **Know your concurrency primitives** -- `Lock`, `RLock`, `Condition`, `Semaphore`, `Event`, `Barrier`. Know when to use each.
3. **Implement data structures from scratch** -- Don't rely on library implementations. Know the internals of heaps, tries, and hash maps.
4. **Think about fairness** -- Reader-writer locks can starve writers. Discuss solutions (writer-preference, fair queuing).
5. **Test concurrent code mentally** -- Walk through scenarios with 2-3 threads interleaving. Find the race condition before the interviewer does.
