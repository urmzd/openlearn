# Concurrency Patterns: Deep Dive

Reference material for concurrency concepts that appear throughout Anthropic's interview process.

## Python GIL (Global Interpreter Lock)

The GIL is a mutex that protects access to Python objects, allowing only one thread to execute Python bytecode at a time.

### Implications

| Workload Type | Threading | asyncio | multiprocessing |
|---------------|-----------|---------|-----------------|
| I/O-bound (network, disk) | Good (GIL released during I/O) | Best (single thread, many connections) | Overkill |
| CPU-bound (computation) | Bad (GIL limits to 1 core) | Bad (single thread) | Good (bypasses GIL) |
| Mixed I/O + CPU | Moderate | Use `run_in_executor` for CPU parts | Good |

### When the GIL Is Released

- `socket.recv()`, `socket.send()`
- `file.read()`, `file.write()`
- `time.sleep()`
- C extensions that explicitly release it (numpy, etc.)
- `await` expressions in asyncio

### Atomic Operations Under GIL

These are thread-safe due to the GIL (single bytecode instruction):
- `list.append(x)`
- `dict[key] = value`
- `x = some_list[i]`

These are NOT thread-safe (multiple bytecode instructions):
- `counter += 1` (read, increment, write)
- `if key not in dict: dict[key] = value` (check-then-act)
- `list.sort()` (modifies list across multiple steps)

## asyncio Event Loop

### Core Model

```
[Event Loop]
    |
    +-- [Task A] --await--> [I/O] --resume--> [Task A continues]
    |
    +-- [Task B] --await--> [I/O] --resume--> [Task B continues]
    |
    +-- [Task C] --running-->  (only one task runs at a time)
```

A single thread runs all tasks. When a task hits `await`, it yields control to the event loop, which can run another task. No parallelism, but excellent concurrency for I/O.

### Key Patterns

#### Gather for Parallel I/O

```python
async def fetch_all(urls: list[str]) -> list[str]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

#### Semaphore for Rate Limiting

```python
sem = asyncio.Semaphore(10)  # max 10 concurrent

async def rate_limited_fetch(session, url):
    async with sem:
        async with session.get(url) as resp:
            return await resp.text()
```

#### Queue for Producer-Consumer

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

#### Timeout

```python
try:
    result = await asyncio.wait_for(slow_operation(), timeout=5.0)
except asyncio.TimeoutError:
    handle_timeout()
```

#### Run Blocking Code in Executor

```python
# For CPU-bound or blocking library calls
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(None, blocking_function, arg1, arg2)
```

## Threading Patterns

### Lock (Mutex)

```python
lock = threading.Lock()

def thread_safe_increment():
    with lock:
        shared_counter += 1
```

Use when: You need exclusive access to a shared resource.

### RLock (Reentrant Lock)

```python
rlock = threading.RLock()

def recursive_operation():
    with rlock:
        # Can acquire the same lock again in nested calls
        if condition:
            recursive_operation()
```

Use when: A thread might need to re-acquire a lock it already holds (recursive functions, calling other methods that also lock).

### Condition Variable

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

Use when: Threads need to wait for a specific state change (queue non-empty, buffer has space).

### Event

```python
event = threading.Event()

def waiter():
    event.wait()  # Block until event is set
    do_work()

def trigger():
    event.set()  # Wake all waiters
```

Use when: Multiple threads should wait for a one-time signal (initialization complete, shutdown requested).

### Barrier

```python
barrier = threading.Barrier(3)  # Wait for 3 threads

def worker():
    do_phase_one()
    barrier.wait()  # All 3 must reach here before any proceeds
    do_phase_two()
```

Use when: Multiple threads must synchronize at a specific point (parallel computation phases).

### Thread Pool

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

## Lock-Free Data Structures

### Compare-And-Swap (CAS)

The fundamental building block of lock-free programming:

```python
# Pseudocode -- CAS is typically a CPU instruction
def compare_and_swap(location, expected, new_value) -> bool:
    """Atomically: if location == expected, set to new_value and return True."""
    if location.value == expected:
        location.value = new_value
        return True
    return False
```

### Lock-Free Stack (Treiber Stack)

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

### ABA Problem

CAS can be tricked if a value changes from A -> B -> A between read and CAS. Solutions:
- **Tagged pointers**: Pair the pointer with a version counter
- **Hazard pointers**: Track which nodes are being accessed by threads
- **Epoch-based reclamation**: Defer memory reclamation until no thread can access it

### When to Use Lock-Free

| Scenario | Lock-Free? | Why |
|----------|------------|-----|
| Low contention | No | Locks are simpler and fast when uncontended |
| High contention, simple operations | Yes | Avoids lock convoy and priority inversion |
| Complex multi-step operations | No | Lock-free composition is extremely hard |
| Real-time / latency-critical | Yes | No risk of priority inversion or lock holder preemption |

## Distributed Consistency

### Raft Consensus

Raft provides distributed consensus (agreement on a sequence of values) across a cluster.

**Key roles**:
- **Leader**: Handles all client requests, replicates to followers
- **Follower**: Passive, receives log entries from leader
- **Candidate**: Requests votes during leader election

**Leader election**:
1. Follower times out (no heartbeat from leader)
2. Becomes candidate, increments term, votes for self
3. Requests votes from all peers
4. Wins with majority of votes
5. Sends heartbeats to maintain authority

**Log replication**:
1. Leader receives client request
2. Appends to local log, sends AppendEntries to followers
3. Once majority acknowledges, entry is committed
4. Leader notifies followers of commitment
5. Entry is applied to state machine

**Safety guarantees**:
- At most one leader per term
- Leaders never overwrite their logs
- If a log entry is committed, it will be present in all future leaders' logs

### Leader Election Patterns (Simpler)

For systems that don't need full Raft:

**Lease-based**:
- Leader holds a time-limited lease (e.g., 30 seconds)
- Must renew before expiry
- If leader fails, lease expires and another node can claim it
- Simple but clock skew can cause issues

**ZooKeeper / etcd**:
- Use an external coordination service
- Create an ephemeral node; whoever creates it first is leader
- When leader dies, ephemeral node disappears, watchers get notified

### Replica Synchronization

| Strategy | Consistency | Latency | Availability |
|----------|-------------|---------|-------------|
| Synchronous replication | Strong | High (wait for all replicas) | Lower (one slow replica slows all) |
| Asynchronous replication | Eventual | Low | High |
| Semi-synchronous | Tunable | Medium | Medium |
| Quorum (W + R > N) | Strong reads if quorum met | Medium | Medium |

**Quorum example** (N=5):
- Write to W=3 replicas, read from R=3 replicas
- W + R = 6 > 5, so reads always see the latest write
- Can tolerate 2 replica failures for reads, 2 for writes

## Common Interview Patterns

### "Make It Thread-Safe"

When asked to add thread safety to a data structure:

1. **Coarse-grained lock**: Add a single `threading.Lock()` around all methods. Simple, correct, but limits concurrency.
2. **Fine-grained locks**: Lock per bucket (hash map) or per node (linked list). Better concurrency, more complex.
3. **Read-write lock**: Multiple concurrent readers, exclusive writers. Good when reads >> writes.
4. **Lock-free**: CAS-based operations. Best performance under contention, hardest to implement correctly.

Start with option 1, then optimize if the interviewer asks for better concurrency.

### "What Could Go Wrong?"

Common concurrency bugs to mention:

- **Race condition**: Two threads read-modify-write the same variable
- **Deadlock**: Thread A holds lock 1 and waits for lock 2; Thread B holds lock 2 and waits for lock 1
- **Livelock**: Threads keep retrying and interfering with each other
- **Starvation**: A thread never gets access to a resource (writer starvation in RW locks)
- **Priority inversion**: High-priority thread waits for low-priority thread holding a lock

### "How Would You Debug a Deadlock?"

1. Get a thread dump (Python: `faulthandler.dump_traceback()`, Java: `jstack`)
2. Identify which threads hold which locks
3. Draw the wait-for graph
4. Find the cycle
5. Fix by enforcing a global lock ordering
