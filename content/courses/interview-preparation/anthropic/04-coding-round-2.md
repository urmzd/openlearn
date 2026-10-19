# The Second Coding Round: Data Structures and Concurrency Primitives

The second live coding round is another 60 minutes over video in a shared editor. Python is preferred, although Java or C++ may be used for some problems. Where [the first coding round](02-coding-round-1.md) centered on asyncio and I/O-bound concurrency, this one is about data structures, concurrency primitives, and systems-level coding: you will be asked to build a structure from scratch and then make it safe to share between threads.

This section works through the five problems candidates report. The first, converting stack samples into trace events, is a sequence-diffing problem and the one most worth practicing. The rest are concurrent data structures (a thread-safe min-heap, a top-K tracker over a concurrent stream, a trie with a readers-writer lock) and an inverted index. For each you will see the reference solution, why it is shaped the way it is, and the follow-up questions to expect.

> **Learning objectives**
> - Convert periodic stack samples into enter and exit trace events by diffing consecutive samples, including recursive stacks.
> - Build a blocking thread-safe priority queue with a `Condition`, and explain the tie-breaking counter.
> - Maintain a top-K frequency count over a concurrent stream with a bounded min-heap, and name approximate and windowed alternatives.
> - Implement a readers-writer lock from two mutexes and a counter, and explain why it can starve writers.
> - Build an inverted index that answers AND and OR queries with set operations.

## Stack samples to trace events

A sampling profiler records the call stack periodically. Your task is to turn those snapshots into trace events, enter (`B`) and exit (`E`) markers, suitable for a flame graph or the Chrome trace viewer. Each sample lists function names from the bottom of the stack first:

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

The convention is that a function is assumed to start at the first sample where it appears and to end at the first sample where it no longer appears. So `bar`, seen at $t = 0$ and $t = 1$, gets `B` at 0 and `E` at 2. Functions still on the stack after the last sample are closed at the timestamp one past it, `len(samples)`, which is 6 here. The expected output is:

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
    {"name": "main", "ph": "E", "ts": 6},  # implicit at end
]
```

### The divergence-point algorithm

The key observation is that two consecutive samples share a common prefix, the part of the stack that did not change, and differ only above it. So for each sample you find the **divergence point**: the length of the longest common prefix with the previous sample. Every frame of the previous sample above that point has exited, and you emit their exits from the top down so the events nest properly. Every frame of the current sample above that point has entered, and you emit their enters from the bottom up. After the last sample, you close whatever remains on the stack, again top down.

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

Trace it once on the example. At $t = 2$ the common prefix with `["main", "foo", "bar"]` is `["main", "foo"]`, depth 2, so `bar` exits and `baz` enters. At $t = 4$ the common prefix with `["main", "foo", "baz"]` is just `["main"]`, so `baz` and then `foo` exit, and `qux` enters. Each step costs time proportional to the two stacks' depths.

### Recursive functions

Recursion is the edge case interviewers ask about. When a function calls itself, matching frames by name breaks, because the same name appears at several depths. The fix is to track frames by **position in the stack**, not by name:

```python
samples = [
    ["main", "fib", "fib", "fib"],  # t=0: fib calls itself 3 times
    ["main", "fib", "fib"],          # t=1: innermost fib returns
    ["main", "fib"],                 # t=2: middle fib returns
]
```

The divergence-point algorithm already handles this correctly, because it compares position by position instead of searching for names. At $t = 1$ the common depth is 3, so exactly one `fib`, the one at index 3, exits.

> **Note:** Sampling has an inherent blind spot worth mentioning. If a function returns and is called again at the same depth between two samples, the stacks look identical and no exit or enter is emitted. A profiler can only report what its samples show.

## A thread-safe min-heap

The second problem is a concurrent priority queue with three operations: `push(item, priority)`, a thread-safe insert; `pop()`, a thread-safe extract-min that blocks while the heap is empty; and `peek()`, a non-blocking read of the minimum element.

The design wraps `heapq` with a lock, and builds a `Condition` on that same lock so that `pop` can wait efficiently for data instead of spinning. `push` notifies one waiter after inserting. `pop` waits in a `while` loop, not an `if`, because a woken thread must recheck the condition: another consumer may have taken the item first. Each heap entry is a tuple `(priority, counter, item)`.

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

Expect three discussion points. *Why a `Condition` rather than just a `Lock`?* It lets `pop()` block efficiently until data arrives, releasing the lock while it waits. *Why the counter?* Without a tie-breaker, two entries with equal priority are compared by their items, and `heapq` raises an error when the items are not comparable; the counter also makes equal-priority items come out in insertion order. And in a Java context you may be asked about *`ConcurrentHashMap` versus `Hashtable`*: `ConcurrentHashMap` uses finer-grained, segment-level locking, while `Hashtable` locks the entire structure on every operation.

## Top-K frequent queries from a concurrent stream

The third problem gives you a stream of queries arriving from multiple threads and asks you to maintain a real-time top-K frequency count.

Recording is a locked increment in a dictionary. Answering is the classic bounded-heap technique: keep a min-heap of at most K `(count, query)` pairs, and replace the root whenever you see a count larger than the smallest one in the heap. That costs $O(n \log K)$ for $n$ distinct queries instead of $O(n \log n)$ for a full sort. If there are no more than K distinct queries, a plain sort is simpler.

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

The lock around `counts[query] += 1` is necessary: the increment is a read-modify-write, and the GIL does not make it atomic. Note also that `top_k` holds the lock for its whole scan, which blocks writers; that is a reasonable first answer and a natural place for the follow-ups. Those follow-ups are approximate counting with a **Count-Min Sketch** for massive streams, **sharding by query hash** so independent counters can be updated in parallel, and **sliding windows** or time-decayed counts as opposed to all-time counts.

## A trie with a readers-writer lock

The fourth problem asks for a concurrent trie supporting insert, exact search, and prefix search, with read-write locking so that concurrent reads do not block one another.

Python's standard library has no readers-writer lock, so you build one. The classic construction uses a reader count protected by its own mutex, plus a write lock. The *first* reader to arrive acquires the write lock on behalf of all readers, and the *last* reader to leave releases it. Writers simply take the write lock. Many readers can therefore proceed together, while a writer excludes everyone. (A `threading.Lock` may be released by a different thread than the one that acquired it, which this construction relies on.)

```python
import threading

class TrieNode:
    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_end = False

class ConcurrentTrie:
    # Python doesn't have an RWLock natively, so build a readers-writer pattern:
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

The `try`/`finally` around each read guarantees the reader count is decremented even when the method returns early, which matters: a leaked reader would hold the write lock forever.

> **Warning:** This readers-preference lock can starve writers. As long as readers keep overlapping, the reader count never reaches zero and a writer never gets the lock. Be ready to discuss fixes: writer preference (block new readers once a writer is waiting) or fair, queue-ordered locking.

## An inverted index

The last reported problem is an inverted index for full-text search with AND and OR queries. Each token maps to the set of document IDs containing it, so AND is set intersection and OR is set union. Tokenization lowercases the text and extracts word characters, and queries are lowercased the same way, which keeps index-time and query-time analysis consistent.

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

Two details are easy to get wrong. `search_and` copies the first posting set before intersecting into it, because `&=` would otherwise mutate the index itself. And it uses `self.index.get` rather than indexing the `defaultdict`, so a query for an unknown term does not insert an empty entry. Production engines use sorted postings lists with skip pointers and rank results with BM25; that design is developed in [Search and Indexing](../../infrastructure/04-search-and-indexing/README.md) in *Infrastructure*.

## Preparing for the round

Practice diffing algorithms: the profiler problem tests your ability to diff two sequences and generate events, so draw it out. Know your concurrency primitives (`Lock`, `RLock`, `Condition`, `Semaphore`, `Event`, `Barrier`) and when to use each. Implement data structures from scratch rather than relying on library implementations; know the internals of heaps, tries, and hash maps. Think about fairness, since readers-writer locks can starve writers. And test concurrent code mentally: walk through scenarios with two or three threads interleaving, and find the race before the interviewer does.

## Where this leads

The primitives used here, and the reasoning about races and deadlocks, are developed in [Concurrency patterns](concepts/concurrency-patterns.md). The asyncio side of the same skills is in [the first coding round](02-coding-round-1.md).

## Key terms

- **Trace event**: a timestamped enter (`B`) or exit (`E`) marker for a function, as consumed by flame graphs and the Chrome trace viewer.
- **Divergence point**: the length of the longest common prefix of two consecutive stack samples.
- **Condition variable**: a primitive that lets a thread release a lock and sleep until another thread signals that state has changed.
- **Readers-writer lock**: a lock that admits many concurrent readers or one exclusive writer.
- **Writer starvation**: a writer waiting indefinitely because overlapping readers never release a readers-preference lock.
- **Count-Min Sketch**: a fixed-size probabilistic structure that estimates frequencies in a stream with bounded overcount.

## Summary

- Stack samples become trace events by diffing consecutive samples at their divergence point: exits top down, enters bottom up, and remaining frames closed at the end.
- Comparing frames by position, not name, makes recursion work.
- A blocking priority queue is `heapq` plus a lock and a `Condition`, with a counter to break priority ties.
- Top-K over a stream uses a locked counter and a size-K min-heap; scale with sketches, sharding, or windows.
- A readers-writer lock can be built from a reader count and two mutexes, and it can starve writers.

## Review questions

1. Using the divergence-point algorithm, what events are produced at $t = 4$ when the previous sample is `["main", "foo", "baz"]` and the current one is `["main", "qux"]`?
   <details><summary>Answer</summary>The common depth is 1 (<code>main</code>). Exits are emitted top down for <code>baz</code> then <code>foo</code>, each at ts 4, followed by an enter for <code>qux</code> at ts 4.</details>
2. Why does the profiler algorithm handle `["main", "fib", "fib", "fib"]` followed by `["main", "fib", "fib"]` correctly?
   <details><summary>Answer</summary>It compares frames position by position. The first three positions match, so the common depth is 3 and exactly one exit is emitted, for the <code>fib</code> at index 3. A name-based search would be confused by the repeated name.</details>
3. Why does `pop` in the thread-safe heap wait inside a `while` loop rather than an `if`?
   <details><summary>Answer</summary>When a waiting thread wakes, another consumer may already have taken the item, and condition variables can also wake spuriously. The loop rechecks that the heap is non-empty before popping.</details>
4. What goes wrong if the heap stores `(priority, item)` without the counter?
   <details><summary>Answer</summary>When two entries have equal priority, tuple comparison falls through to the items. If the items are not comparable (for example, dicts or custom objects), <code>heapq</code> raises a <code>TypeError</code>. The counter breaks ties and preserves insertion order.</details>
5. In the readers-writer trie, what happens if a reader returns early without releasing its read lock?
   <details><summary>Answer</summary>The reader count never returns to zero, so the write lock acquired by the first reader is never released and every subsequent insert blocks forever. That is why every read is wrapped in <code>try</code>/<code>finally</code>.</details>
6. Why does `search_and` call `.copy()` on the first posting set?
   <details><summary>Answer</summary>Intersecting with <code>&=</code> mutates the set in place. Without the copy, the query would shrink the index's own posting set for the first term, corrupting future results.</details>
