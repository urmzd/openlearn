# Concurrency Patterns

Concurrency appears in every round of the Anthropic infrastructure loop: thread-safe data structures in the [online assessment](01-online-assessment.md), asyncio in [the first coding round](02-coding-round-1.md), locks and condition variables in [the second coding round](04-coding-round-2.md), and consistency across replicas in [the system design round](03-system-design.md). This section gathers the concepts behind all of them in one place, so you can reason from principles rather than recall snippets.

It moves from the smallest scale to the largest. It starts inside one Python process, with the GIL and what it does and does not make safe, then covers asyncio's event loop and the patterns built on it, then the threading primitives and when each fits. From there it steps below locks to lock-free structures built on compare-and-swap, then out to many machines with Raft, simpler leader election, and replica synchronization. It closes with the three concurrency questions interviewers ask most often.

> **Learning objectives**
> - Choose among threading, asyncio, and multiprocessing for a workload, based on what the GIL allows.
> - Identify which Python operations are atomic under the GIL and which need a lock.
> - Use asyncio's gather, semaphore, queue, timeout, and executor patterns, and the threading Lock, RLock, Condition, Event, Barrier, and thread pool.
> - Explain compare-and-swap, the Treiber stack, the ABA problem, and when lock-free designs are worth their complexity.
> - Describe Raft leader election and log replication, lease-based election, and quorum arithmetic for replicas.

## The GIL

CPython's Global Interpreter Lock is a mutex that protects access to Python objects, allowing only one thread to execute Python bytecode at a time. Everything about choosing a concurrency model in Python follows from it, as the table shows.

| Workload Type | Threading | asyncio | multiprocessing |
|---------------|-----------|---------|-----------------|
| I/O-bound (network, disk) | Good (GIL released during I/O) | Best (single thread, many connections) | Overkill |
| CPU-bound (computation) | Bad (GIL limits to 1 core) | Bad (single thread) | Good (bypasses GIL) |
| Mixed I/O + CPU | Moderate | Use `run_in_executor` for CPU parts | Good |

Threads still help with I/O because the GIL is released while a thread waits. That happens in `socket.recv()` and `socket.send()`, in `file.read()` and `file.write()`, in `time.sleep()`, and inside C extensions that explicitly release it, such as numpy. In asyncio, the equivalent yield point is every `await` expression.

The GIL also determines what is atomic. Operations that compile to a single bytecode instruction are thread-safe because of it: `list.append(x)`, `dict[key] = value`, and `x = some_list[i]`. Operations that take several instructions are not, because another thread can run between them. `counter += 1` is a read, an increment, and a write. `if key not in dict: dict[key] = value` is a check followed by an act. And `list.sort()` modifies the list across multiple steps.

> **Warning:** The GIL protects the interpreter, not your invariants. Any read-modify-write or check-then-act sequence needs an explicit lock, even on built-in types.

## The asyncio event loop

asyncio takes the opposite approach to threads. A single thread runs all tasks, and a task gives up control only when it hits `await`, at which point the event loop can run another task.

```text
[Event Loop]
    |
    +-- [Task A] --await--> [I/O] --resume--> [Task A continues]
    |
    +-- [Task B] --await--> [I/O] --resume--> [Task B continues]
    |
    +-- [Task C] --running-->  (only one task runs at a time)
```

There is no parallelism, but there is excellent concurrency for I/O, and because switches happen only at `await`, code between two `await`s cannot be interleaved. Five patterns cover most interview problems.

To run many I/O operations concurrently, **gather** them. `return_exceptions=True` returns failures as values so that one failed request does not cancel the others.

```python
async def fetch_all(urls: list[str]) -> list[str]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

To bound how many run at once, hold a **semaphore** around each operation.

```python
sem = asyncio.Semaphore(10)  # max 10 concurrent

async def rate_limited_fetch(session, url):
    async with sem:
        async with session.get(url) as resp:
            return await resp.text()
```

To decouple producers from consumers, use an **`asyncio.Queue`**, with a sentinel value to signal the end of the stream.

```python
async def producer(queue: asyncio.Queue):
    for item in items:
        await queue.put(item)
    await queue.put(None)  # sentinel

async def consumer(queue: asyncio.Queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        await process(item)
```

To bound how long an operation may take, wrap it in a **timeout**.

```python
try:
    result = await asyncio.wait_for(slow_operation(), timeout=5.0)
except asyncio.TimeoutError:
    handle_timeout()
```

And to call CPU-bound or blocking library code without stalling every other task, **run it in an executor**.

```python
# For CPU-bound or blocking library calls
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(None, blocking_function, arg1, arg2)
```

## Threading primitives

When you do use threads, the standard library gives you six tools, each matched to a different coordination need.

A **Lock** (mutex) gives exclusive access to a shared resource. Use it whenever threads read and write the same state.

```python
lock = threading.Lock()

def thread_safe_increment():
    global shared_counter
    with lock:
        shared_counter += 1
```

An **RLock** (reentrant lock) can be acquired again by the thread that already holds it. Use it when a thread may need to re-acquire a lock it holds: in recursive functions, or when one locked method calls another method that also locks.

```python
rlock = threading.RLock()

def recursive_operation():
    with rlock:
        # Can acquire the same lock again in nested calls
        if condition:
            recursive_operation()
```

A **Condition** variable lets threads wait for a specific state change, such as a queue becoming non-empty or a buffer gaining space. `wait()` releases the lock and sleeps until notified; the `while` loop rechecks the state after waking.

```python
condition = threading.Condition()
queue = []

def producer():
    with condition:
        queue.append(item)
        condition.notify()  # Wake one waiting consumer

def consumer():
    with condition:
        while not queue:
            condition.wait()  # Release lock and sleep until notified
        return queue.pop(0)
```

An **Event** is a one-time signal that many threads can wait on, such as "initialization complete" or "shutdown requested."

```python
event = threading.Event()

def waiter():
    event.wait()  # Block until event is set
    do_work()

def trigger():
    event.set()  # Wake all waiters
```

A **Barrier** makes a fixed number of threads meet at a point before any of them continues, which suits parallel computations that proceed in phases.

```python
barrier = threading.Barrier(3)  # Wait for 3 threads

def worker():
    do_phase_one()
    barrier.wait()  # All 3 must reach here before any proceeds
    do_phase_two()
```

Finally, a **thread pool** manages a fixed set of worker threads for you. `as_completed` yields futures as they finish, and mapping each future back to its input lets you report errors per item.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(process, item): item for item in items}
    for future in as_completed(futures):
        try:
            result = future.result()
        except Exception as e:
            handle_error(futures[future], e)
```

## Lock-free data structures

Locks are not the only way to make shared state safe. Lock-free structures avoid them entirely, and the building block they rest on is **compare-and-swap (CAS)**, typically a single CPU instruction. CAS atomically checks whether a location still holds an expected value and, only if so, replaces it.

```python
# Pseudocode -- CAS is typically a CPU instruction
def compare_and_swap(location, expected, new_value) -> bool:
    """Atomically: if location == expected, set to new_value and return True."""
    if location.value == expected:
        location.value = new_value
        return True
    return False
```

The classic example is the **Treiber stack**. To push, read the current top, point the new node at it, and CAS the top from the old value to the new node; if another thread changed the top in the meantime, the CAS fails and you retry. Pop works the same way in reverse.

```python
# Conceptual implementation
class LockFreeStack:
    def __init__(self):
        self.top = AtomicReference(None)

    def push(self, value):
        new_node = Node(value)
        while True:
            old_top = self.top.get()
            new_node.next = old_top
            if self.top.compare_and_set(old_top, new_node):
                return

    def pop(self):
        while True:
            old_top = self.top.get()
            if old_top is None:
                return None
            new_top = old_top.next
            if self.top.compare_and_set(old_top, new_top):
                return old_top.value
```

CAS has a well-known trap, the **ABA problem**. If a value changes from A to B and back to A between a thread's read and its CAS, the CAS succeeds even though the structure changed underneath it; in the stack, the node that was A may have been popped, freed, and reused. There are three standard solutions. **Tagged pointers** pair the pointer with a version counter, so A-then-B-then-A is detected as a different version. **Hazard pointers** track which nodes each thread is currently accessing so they are not reclaimed. **Epoch-based reclamation** defers freeing memory until no thread can still be accessing it.

Lock-free designs are not automatically better. The table shows when they pay off.

| Scenario | Lock-Free? | Why |
|----------|------------|-----|
| Low contention | No | Locks are simpler and fast when uncontended |
| High contention, simple operations | Yes | Avoids lock convoy and priority inversion |
| Complex multi-step operations | No | Lock-free composition is extremely hard |
| Real-time / latency-critical | Yes | No risk of priority inversion or lock holder preemption |

## Distributed consistency

Across machines there is no shared memory to lock, so agreement has to be built from messages. That is the job of consensus.

### Raft

Raft provides distributed consensus, agreement on a sequence of values, across a cluster. Each node plays one of three roles. The **leader** handles all client requests and replicates them to followers. A **follower** is passive and receives log entries from the leader. A **candidate** requests votes during an election.

Leader election proceeds in five steps. A follower times out after hearing no heartbeat from the leader. It becomes a candidate, increments the term, and votes for itself. It requests votes from all its peers. It wins if it receives a majority. And as leader, it sends heartbeats to maintain its authority.

Log replication also takes five steps. The leader receives a client request, appends it to its local log, and sends `AppendEntries` to the followers. Once a majority acknowledges, the entry is committed. The leader notifies the followers of the commitment, and the entry is applied to each node's state machine.

Those mechanics yield Raft's safety guarantees: at most one leader per term; leaders never overwrite their own logs; and once a log entry is committed, it will be present in the logs of all future leaders. The KRaft controller in Kafka is a working example; see [Messaging and Distributed Queueing](../../../systems/05-messaging-and-queueing/README.md) in *Systems Engineering*.

### Simpler leader election

Not every system needs full Raft. Two simpler patterns are common. With a **lease**, the leader holds a time-limited lease (for example, 30 seconds) and must renew it before it expires; if the leader fails, the lease runs out and another node can claim it. This is simple, but clock skew between nodes can cause two of them to believe they hold the lease. With an **external coordination service** such as ZooKeeper or etcd, each candidate tries to create an ephemeral node; whoever creates it first is the leader. When the leader dies, its ephemeral node disappears and the watchers are notified.

### Replica synchronization

Once there is a leader, its writes must reach the replicas, and the strategy you choose trades consistency against latency and availability:

| Strategy | Consistency | Latency | Availability |
|----------|-------------|---------|-------------|
| Synchronous replication | Strong | High (wait for all replicas) | Lower (one slow replica slows all) |
| Asynchronous replication | Eventual | Low | High |
| Semi-synchronous | Tunable | Medium | Medium |
| Quorum (W + R > N) | Strong reads if quorum met | Medium | Medium |

Quorum arithmetic is worth working once. With $N = 5$ replicas, write to $W = 3$ and read from $R = 3$. Because $W + R = 6 > 5$, every read set overlaps every write set in at least one replica, so reads always see the latest write. The system can tolerate two replica failures for reads and two for writes.

## The questions interviewers ask

Three concurrency questions come up so often that you should have a prepared answer for each.

### "Make it thread-safe"

There is a ladder of four options. A **coarse-grained lock**, one `threading.Lock()` around all methods, is simple and correct but limits concurrency. **Fine-grained locks**, one per hash bucket or per list node, allow more concurrency at the cost of complexity. A **read-write lock** allows many concurrent readers and exclusive writers, which is good when reads far outnumber writes. And **lock-free** CAS-based operations give the best performance under contention but are the hardest to implement correctly. Start with the coarse lock, then optimize if the interviewer asks for better concurrency.

### "What could go wrong?"

Name the classic failure modes. A **race condition** is two threads doing read-modify-write on the same variable. A **deadlock** is thread A holding lock 1 and waiting for lock 2 while thread B holds lock 2 and waits for lock 1. A **livelock** is threads that keep retrying and interfering with each other without progress. **Starvation** is a thread that never gets access to a resource, such as a writer under a readers-preference lock. And **priority inversion** is a high-priority thread waiting on a low-priority thread that holds a lock.

### "How would you debug a deadlock?"

Get a thread dump (in Python, `faulthandler.dump_traceback()`; in Java, `jstack`). Identify which threads hold which locks. Draw the wait-for graph and find the cycle. Then fix it by enforcing a global lock ordering, so that every thread acquires locks in the same order and no cycle can form.

## Key terms

- **GIL**: CPython's Global Interpreter Lock, allowing one thread at a time to execute bytecode.
- **Atomic operation**: an operation no other thread can observe half-done; under the GIL, a single bytecode instruction.
- **Reentrant lock**: a lock its holder may acquire again without deadlocking itself.
- **Compare-and-swap (CAS)**: an atomic instruction that writes a new value only if the location still holds the expected one.
- **ABA problem**: a CAS succeeding wrongly because a value changed and changed back between the read and the swap.
- **Quorum**: a subset of replicas large enough that any read set and write set overlap ($W + R > N$).
- **Lock ordering**: a global order in which all threads acquire locks, which prevents deadlock cycles.

## Summary

- The GIL makes threads useful for I/O but not for CPU work; use processes for CPU-bound parallelism and asyncio for many connections.
- Single bytecode operations are atomic; read-modify-write and check-then-act are not.
- asyncio's core patterns are gather, semaphore, queue, timeout, and executor; threading's are Lock, RLock, Condition, Event, Barrier, and pools.
- Lock-free structures rest on CAS and must handle ABA; they pay off only under high contention or strict latency requirements.
- Raft elects one leader per term and commits entries on a majority; quorums with $W + R > N$ give strong reads.
- For "make it thread-safe," start with a coarse lock and refine; fix deadlocks with a global lock order.

## Review questions

1. Is `if key not in d: d[key] = value` safe across threads under the GIL? Why or why not?
   <details><summary>Answer</summary>No. The membership check and the assignment are separate operations, and another thread can insert the key between them. It is a check-then-act race and needs a lock (or <code>d.setdefault</code>, which is a single operation).</details>
2. You need to fetch 10,000 URLs and then run a CPU-heavy parse on each response. How would you structure the work in Python?
   <details><summary>Answer</summary>Use asyncio with a semaphore for the fetches, since they are I/O-bound, and hand each parse to a process pool through <code>run_in_executor</code>, since CPU work is serialized by the GIL in threads and coroutines.</details>
3. Why must a condition-variable consumer check its condition in a `while` loop?
   <details><summary>Answer</summary>After waking, the condition may no longer hold: another consumer may have taken the item, or the wakeup may be spurious. The loop rechecks the state before proceeding.</details>
4. Describe the ABA problem in the Treiber stack and one fix.
   <details><summary>Answer</summary>A thread reads top = A and is preempted. Others pop A, pop B, and push A back (possibly reused memory). The first thread's CAS from A to A.next succeeds, but A.next is now stale, corrupting the stack. Tagged pointers fix this by pairing the pointer with a version counter that changes on every update.</details>
5. With $N = 5$, is $W = 2, R = 3$ enough for strong reads? What about $W = 3, R = 3$?
   <details><summary>Answer</summary>$W = 2, R = 3$ gives $W + R = 5$, not greater than $N$, so a read might miss the latest write. $W = 3, R = 3$ gives $6 > 5$, so every read overlaps the latest write.</details>
6. How do you fix a deadlock between two threads that acquire locks 1 and 2 in opposite orders?
   <details><summary>Answer</summary>Impose a global lock ordering, for example always acquiring lock 1 before lock 2. With every thread following the same order, the wait-for graph cannot contain a cycle.</details>
