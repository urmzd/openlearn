# Collectives over processes: ring all-reduce

[Mixed precision, loss scaling, gradient accumulation, and activation checkpointing](01-mixed-precision-accumulation-and-checkpointing.md) made one process train the capstone on a laptop. Every larger run splits the batch over several workers instead: each computes the gradient of its own slice, and before anyone takes a step the gradients must be averaged across all of them. That averaging is an all-reduce, and it is the operation every data-parallel step waits on.

On a laptop the workers are processes on the same machine, which is enough to build the real algorithm. Processes cannot share Python objects, so every byte moves through an explicit channel, and you can count those bytes. This section builds the collectives (`all_reduce`, `reduce_scatter`, `all_gather`, `broadcast`, `barrier`) over pipes with the bandwidth-optimal ring, and the next section, [Data parallelism and ZeRO stages 1 to 3](03-ddp-and-zero.md), builds data parallelism and ZeRO on top. The section is optional: a laptop capstone needs mixed precision and accumulation, not data parallelism.

There is nothing to build first. The only background is the byte counting of [Counting: parameters, FLOPs, KV bytes, and memory plans](../../mathematics-for-computing/06-discrete-math-1/01-counting-params-flops-kv-bytes.md) in *Mathematics for Computing*: counting bytes per parameter is the same arithmetic as counting bytes per link. The solution is [`comm.py`](../../../../course/solutions/python/tinyllm/dist/comm.py) (`chunk_bounds`, `Comm` with `send`, `recv`, `reduce_scatter`, `all_gather`, `all_reduce`, `broadcast`, `barrier`, and `spawn`), and the contract is [`comm.pyi`](../../../../course/contracts/py/tinyllm/dist/comm.pyi).

> **Learning objectives**
> - Derive the ring reduce-scatter and all-gather index rules and trace them by hand on three ranks.
> - Show that a ring all-reduce sends $2(p-1)n\beta/p$ bytes per rank and explain why that is optimal.
> - Explain why a blocking ring deadlocks and how parity ordering breaks the cycle.
> - Make a multi-process run fail loudly when one rank crashes.

## Notation

| Symbol | Meaning | Type / shape |
|---|---|---|
| $p$ (`world`) | number of processes | `int` |
| $r$ (`rank`) | this process's number, $0 \le r < p$ | `int` |
| $x_r$ | rank $r$'s input array, $n$ entries | `float64[n]` |
| $\sum_r x_r$ | the elementwise sum all ranks want | `float64[n]` |
| $c_j$ | chunk $j$ of the flat array: entries $[a_j, b_j)$ from `chunk_bounds` | slice |
| right, left | ranks $(r+1) \bmod p$ and $(r-1) \bmod p$ | `int` |
| $\beta$ | bytes per entry (8 for float64) | `int` |

## Processes and channels

`spawn(fn, p)` starts $p$ processes with multiprocessing's "spawn" method (a fresh interpreter each, so `fn` must be importable by name) and connects every pair of ranks with a full-duplex pipe. `send(x, dst)` writes a copy of the array, and `recv(src)` blocks until the next array from `src` arrives; messages between two ranks arrive in order. A pipe holds only a few kilobytes, so a larger `send` blocks until the receiver reads. That one fact will matter a great deal when we get to deadlock.

## Why a ring

The obvious all-reduce sends every array to rank 0, adds, and sends the sum back. Rank 0 then receives $(p-1)n$ entries and sends $(p-1)n$, so its link carries traffic that grows with $p$ while the other links idle. The ring keeps every link equally busy.

The ring starts by splitting the flat array into $p$ contiguous chunks $c_0, \dots, c_{p-1}$, sized as `numpy.array_split` sizes them (the first $n \bmod p$ chunks one entry larger). Rank $r$ owns chunk $r$. ZeRO shards parameters by the same bounds in the next section, so the bounds must agree everywhere.

**Reduce-scatter.** In step $s = 0, \dots, p-2$, rank $r$ sends chunk $(r - s - 1) \bmod p$ to its right neighbour, receives chunk $(r - s - 2) \bmod p$ from its left neighbour, and adds what it received into its own copy of that chunk. Follow one chunk $c_j$: it starts at rank $j + 1$, which sends its $x_{j+1}$ part right; each rank adds its own part and passes the partial sum on; after $p - 1$ steps it reaches rank $j$ holding all $p$ contributions. So every rank ends holding its own chunk of the sum, $\big(\sum_r x_r\big)[c_r]$, having sent $p - 1$ chunks.

**All-gather.** Now each rank has one finished chunk and needs the others. In step $s$, rank $r$ sends chunk $(r - s) \bmod p$ right and receives chunk $(r - s - 1) \bmod p$ from the left. After $p - 1$ steps every rank has every chunk. All-reduce is reduce-scatter followed by all-gather.

## Counting bytes and rounding

Each half sends $p - 1$ chunks of about $n/p$ entries, so

$$\text{bytes sent per rank} = 2\,\frac{p-1}{p}\, n\, \beta,$$

which approaches $2n\beta$ and never grows with $p$: doubling the ranks halves each chunk. A lower bound says no all-reduce can do better (each rank must send at least $(p-1)/p$ of its data out and receive the same in), which is why NCCL uses rings for large messages.

The ring also changes the order of addition. The sum of chunk $c_j$ is accumulated in ring order starting at rank $j + 1$, not in rank order, and floating-point addition is not associative, so the result differs from `numpy.sum` in the last bits. But each chunk is reduced once and then copied, so all ranks hold identical bits, which is what keeps data-parallel replicas identical.

## Deadlock, broadcast, barrier, and failure

If every rank does `send` then `recv` in a step, and the message is larger than the pipe buffer, every rank blocks in `send` waiting for its right neighbour to `recv`, which is itself blocked in `send`: a cycle of waits, forever. Breaking the cycle needs one rank that receives first. The contract's rule is that even ranks send then receive, and odd ranks receive then send. With $p \ge 2$, rank 1 is always receiver-first, so the chain unwinds from there, for odd and even $p$ alike. MPI's alternatives are `MPI_Sendrecv` and non-blocking sends.

`broadcast(x, src)` passes $x$ around the ring from `src`: each rank receives from the left and forwards right, except the rank just before `src`, which would send it back to where it started and leave a message in that pipe that the next collective would read by mistake. A barrier makes every rank wait until all have arrived; it is multiprocessing's `Barrier`.

Finally, a rank that raises must stop the whole run with its traceback; otherwise its neighbours wait forever in a `recv`. `spawn` collects each rank's result or traceback, terminates the rest on the first error, and raises `TimeoutError` if the ranks do not finish in time.

## Working it by hand

Three ranks hold $x_0 = [1, 2, 3]$, $x_1 = [10, 20, 30]$, $x_2 = [100, 200, 300]$. With $n = 3$, each chunk is one entry: $c_0$ is entry 0, $c_1$ entry 1, $c_2$ entry 2. The table traces the two reduce-scatter (RS) steps and the two all-gather (AG) steps:

| step | rank 0 sends | rank 1 sends | rank 2 sends | rank 0 holds after | rank 1 holds after | rank 2 holds after |
|---|---|---|---|---|---|---|
| RS 0 | $c_2 = 3$ | $c_0 = 10$ | $c_1 = 200$ | $c_1$: $2 + 200 = 202$ | $c_2$: $30 + 3 = 33$ | $c_0$: $100 + 10 = 110$ |
| RS 1 | $c_1 = 202$ | $c_2 = 33$ | $c_0 = 110$ | $c_0$: $1 + 110 = 111$ | $c_1$: $20 + 202 = 222$ | $c_2$: $300 + 33 = 333$ |
| AG 0 | $c_0 = 111$ | $c_1 = 222$ | $c_2 = 333$ | gets $c_2 = 333$ | gets $c_0 = 111$ | gets $c_1 = 222$ |
| AG 1 | $c_2 = 333$ | $c_0 = 111$ | $c_1 = 222$ | gets $c_1 = 222$ | gets $c_2 = 333$ | gets $c_0 = 111$ |

In step RS 0, rank $r$ sends chunk $(r - 1) \bmod 3$ and adds what arrives into chunk $(r - 2) \bmod 3$: rank 0 receives rank 2's $c_1 = 200$ and adds its own 2. After reduce-scatter rank $r$ holds chunk $r$ of the sum: $[111]$, $[222]$, $[333]$. After all-gather every rank holds $[111, 222, 333]$. Each rank sent 4 entries of 8 bytes, 32 bytes, matching $2 \cdot \tfrac{2}{3} \cdot 3 \cdot 8 = 32$.

## Designing the API

```python
# python/tinyllm/dist/comm.py
def chunk_bounds(n: int, world: int) -> list[tuple[int, int]]    # numpy.array_split sizes
class Comm:
    rank: int; world: int; bytes_sent: int
    def send(self, x, dst: int) -> None; def recv(self, src: int) -> NDArray
    def reduce_scatter(self, x) -> NDArray          # chunk `rank` of the sum
    def all_gather(self, x) -> NDArray              # every rank's chunk, in rank order
    def all_reduce(self, x, op="sum") -> NDArray    # "sum" or "mean", x's shape
    def broadcast(self, x, src: int) -> NDArray
    def barrier(self) -> None
def spawn(fn, world: int, *args, timeout: float = 60.0) -> list    # fn(comm, *args) per rank
```

A worker is a module-level function `fn(comm, *args)`, and `spawn` returns the workers' return values by rank. Collectives must be called by every rank in the same order.

## Building it step by step

The code follows from six observations.

1. **Every rank must end with the same full sum, and no link may carry more than its share.** Gathering to rank 0 fails the second requirement (its link carries $(p-1)n$ entries each way). Splitting the array into $p$ chunks and making each rank responsible for reducing one chunk spreads the work: each chunk's sum is built by one walk around the ring.
2. **A chunk's walk visits every rank once.** Chunk $j$ must collect $p$ contributions and end at its owner $j$, so it starts at $j+1$ and takes $p-1$ hops. In step $s$ rank $r$ therefore forwards the partial sum it received in step $s-1$, which is chunk $(r - s - 1) \bmod p$, and receives chunk $(r - s - 2) \bmod p$. Check with $s = 0$: rank $j+1$ sends chunk $j$, as required.
3. **The all-gather is the same walk without the addition.** After reduce-scatter rank $r$ owns finished chunk $r$; in step $s$ it forwards chunk $(r - s) \bmod p$ (its own at $s = 0$, then whatever it just received) and stores what arrives as chunk $(r - s - 1) \bmod p$.
4. **Each step is "send right, receive left" on every rank at once, which is a cycle of waits.** Blocking pipes break it only if some rank receives first. Alternating by parity puts a receiver-first rank (rank 1) on every ring with $p \ge 2$.
5. **The chunking must be a shared, deterministic function.** ZeRO shards parameters by the same bounds, so `chunk_bounds` is one function with `array_split`'s sizes, not an inline computation.
6. **Failure must be loud.** A crashed rank leaves its neighbours blocked in `recv` forever, so the parent must collect results with a deadline and terminate everyone on the first error.

### Point to point

Point to point sends the dtype, shape, and raw bytes, so the receiver rebuilds exactly the array that was sent, and it counts bytes, not entries:

```python
def send(self, x: ArrayLike, dst: int) -> None:
    if dst == self.rank or not 0 <= dst < self.world:
        raise ValueError(f"rank {self.rank} cannot send to {dst} (world {self.world})")
    a = np.ascontiguousarray(x)
    self._conns[dst].send((a.dtype.str, a.shape, a.tobytes()))
    self.bytes_sent += a.nbytes          # bytes, so the 2(p-1)/p n beta claim is measurable

def recv(self, src: int) -> NDArray:
    dt, shape, raw = self._conns[src].recv()
    # frombuffer gives a read-only view of the message; copy so the receiver owns a writable array
    return np.frombuffer(raw, dtype=np.dtype(dt)).reshape(shape).copy()
```

### One ring step and the two halves

One ring step, in the deadlock-free order:

```python
def _exchange(self, out: NDArray) -> NDArray:
    right, left = (self.rank + 1) % self.world, (self.rank - 1) % self.world
    if self.rank % 2 == 0:        # even ranks: send, then receive
        self.send(out, right)
        return self.recv(left)
    got = self.recv(left)         # odd ranks: receive first, which breaks the cycle of blocked sends
    self.send(out, right)
    return got
```

The two halves are the index rules of observations 2 and 3, verbatim:

```python
def reduce_scatter(self, x: ArrayLike) -> NDArray:
    flat = np.array(x).reshape(-1)  # a copy: the caller's array is never written
    p, r = self.world, self.rank
    b = chunk_bounds(flat.size, p)
    for s in range(p - 1):          # p - 1 hops: a chunk visits every rank once
        si, ri = (r - s - 1) % p, (r - s - 2) % p
        got = self._exchange(flat[b[si][0] : b[si][1]])
        flat[b[ri][0] : b[ri][1]] += got   # add into the chunk that arrived, not the one sent
    return flat[b[r][0] : b[r][1]].copy()  # rank r owns chunk r of the sum

def all_gather(self, x: ArrayLike) -> NDArray:
    p, r = self.world, self.rank
    chunks: list = [None] * p
    chunks[r] = np.array(x).reshape(-1)
    for s in range(p - 1):
        si, ri = (r - s) % p, (r - s - 1) % p
        chunks[ri] = self._exchange(chunks[si])  # forward what you have, store what arrives
    return np.concatenate(chunks)                # rank order, whatever order chunks arrived in

def all_reduce(self, x: ArrayLike, op: str = "sum") -> NDArray:
    # ... op must be "sum" or "mean"
    a = np.asarray(x)
    out = self.all_gather(self.reduce_scatter(a)).reshape(a.shape)
    return out / self.world if op == "mean" else out
```

Because `all_gather` takes each rank's chunk as given, it works for unequal chunk sizes too: the sizes come from the arrays themselves, not from a formula.

### Broadcast and spawn

Broadcast walks the ring once from `src` and stops one rank short:

```python
def broadcast(self, x: ArrayLike, src: int) -> NDArray:
    p, r = self.world, self.rank
    if p == 1:
        return np.array(x)
    right, left = (r + 1) % p, (r - 1) % p
    data = np.array(x) if r == src else self.recv(left)
    if right != src:              # the source already has it; a send here would sit in the pipe
        self.send(data, right)    # and be read by the next collective as its first message
    return data
```

`spawn` connects every pair of ranks with a duplex pipe, starts each rank in `_run` (which puts `("ok", result)` or `("err", traceback)` on a shared queue), and collects results against one deadline:

```python
try:
    for _ in range(world):
        left = deadline - time.monotonic()
        try:
            rank, status, value = results.get(timeout=max(left, 0.01))
        except Exception:  # queue.Empty
            raise TimeoutError(f"spawn: ranks did not finish within {timeout} s (deadlock?)") from None
        if status == "err":
            raise RuntimeError(f"rank {rank} failed:\n{value}")   # first error stops the run
        out[rank] = value
finally:
    for p in procs:
        if p.is_alive():
            p.terminate()          # neighbours blocked in recv on a dead rank are killed, not left hanging
    # ... join
```

The full solution is [`comm.py`](../../../../course/solutions/python/tinyllm/dist/comm.py), and a model test suite is [`test_l11_2_mine.py`](../../../../course/solutions/learner-tests/L11.2/python/tests/l11-2-comm/test_l11_2_mine.py).

### Why this design

All-reduce takes $2(p-1)$ steps and sends $2(p-1)n\beta/p$ bytes per rank; the bandwidth term is optimal. The latency term grows with $p$, which is why production libraries switch to trees for small messages. Gather-to-root, by contrast, slows down linearly in $p$ because the root's link carries $(p-1)n$ entries each way; the ring's per-link traffic is constant.

Non-blocking sends (threads or `MPI_Isend`) would avoid the deadlock too, but they add buffering and a second concurrency mechanism. Parity ordering is three lines and works for every $p \ge 2$ with blocking pipes. Its cost is that in each step the odd ranks wait for the even ranks' sends, so a step takes roughly two message times instead of one.

`reduce_scatter` copies its input once and `recv` copies each message. That spends memory to guarantee that the caller's gradient is never mutated and that every received array is writable, both of which the next section relies on.

## What can go wrong

A correct solution must pass these checks:

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| hand example: ring all-reduce | unit | the three-rank trace: chunks, sums, 32 bytes per rank | the ring's indices are pinned to exact values |
| chunk bounds | unit | `numpy.array_split` sizes for $n < 23$, $p \le 5$ | ZeRO shards line up |
| all-reduce matches `numpy.sum` | differential | $p = 2, 3, 4$ on 35 entries, sum and mean, bitwise-equal ranks, input untouched | DDP replicas stay identical |
| bytes moved is $2(p-1)n\beta/p$ | property | exactly $2(p-1)n\beta/p$ for $p = 2, 4, 5$ | the bandwidth argument |
| the two halves on their own | unit | the halves on their own, unequal chunks | ZeRO stage 2 and 3 |
| broadcast from every source | unit | every source, bytes per rank, a clean pipe afterwards | DDP's initial weights |
| large messages do not deadlock | boundary | 1 MiB messages on a ring of 3 | real gradient sizes |
| point to point | unit | order, dtype, shape, a writable copy, bad ranks | the layer under everything |
| a rank's error propagates | boundary | a crash on rank 1 raises its traceback quickly | no silent hangs |
| barrier | unit | a late rank holds everyone | ordering side effects |

Most bugs in a ring are off-by-one errors in the indices or the step count. If you add the received chunk into the one you just sent, the sums come out wrong and different on every rank; `reduce_scatter` adds into chunk `ri = (r - s - 2) % p`, not the sent `si`. One step too few in reduce-scatter leaves partial sums for $p > 2$, and one step too few in all-gather leaves stale chunks on some ranks; both halves loop `range(p - 1)` times. If `"mean"` returns the sum, gradients come out $p$ times too large; `all_reduce` divides by `world` for `"mean"`.

> **Warning:** If every rank sends first, the ring deadlocks as soon as a message exceeds the pipe buffer. `_exchange` makes even ranks send first and odd ranks receive first.

The remaining pitfalls concern data handling and process management. Accumulating into the caller's array changes the caller's gradient under it; `reduce_scatter` works on `np.array(x)`, a copy. Chunks of the wrong sizes make ZeRO shards and collectives disagree; `chunk_bounds` gives the first `n % world` chunks one extra entry. Counting entries instead of bytes puts the bandwidth numbers off by $\beta$; the code uses `bytes_sent += a.nbytes`. Broadcasting back to the source leaves a stray message that corrupts the next collective; `broadcast` skips the send when `right == src`. Ignoring a rank's failure lets the run "finish" with a missing result; `spawn` raises `RuntimeError` with the rank's traceback. A barrier that does not wait lets side effects race; `barrier` waits on a shared `Barrier`. Returning a read-only view of the message leaves the receiver unable to write its own array; `recv` returns `np.frombuffer(...).copy()`.

## Where this leads

[Data parallelism and ZeRO stages 1 to 3](03-ddp-and-zero.md) runs every collective through `Comm`: DDP all-reduces gradients, and ZeRO uses reduce-scatter and all-gather on parameter chunks.

The production versions keep the same algorithms and add scale:

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Comm` over pipes | NCCL | rings and trees over NVLink and InfiniBand, chunked and pipelined so every link is busy at once | NCCL source, `src/collectives/` |
| `all_reduce` | `torch.distributed.all_reduce` | process groups, backends (NCCL, Gloo, MPI), async handles | `torch/distributed/distributed_c10d.py` |
| the ring | tree and recursive halving-doubling | lower latency for small messages ($\log p$ steps instead of $p$) | Thakur et al. (2005) |
| `spawn` | `torchrun` | rendezvous across machines, restarts, elastic world sizes | `torch/distributed/run.py` |

### Further reading

- Patarasuk and Yuan, ["Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations"](https://doi.org/10.1016/j.jpdc.2008.09.002) (JPDC, 2009).
- Thakur, Rabenseifner, and Gropp, "Optimization of Collective Communication Operations in MPICH" (IJHPCA, 2005).
- The NCCL documentation on collective operations.

## Key terms

- **collective**: an operation that every rank calls in the same order and that moves data among all of them.
- **all-reduce**: every rank ends with the elementwise reduction (sum or mean) of all ranks' arrays.
- **reduce-scatter**: every rank ends with one chunk of the reduction.
- **all-gather**: every rank ends with the concatenation of all ranks' chunks, in rank order.
- **ring**: a topology in which each rank sends only to its right neighbour and receives only from its left.
- **deadlock**: a cycle of blocked operations, each waiting on the next, so none can proceed.

## Summary

- An all-reduce is a reduce-scatter followed by an all-gather; on a ring each half takes $p - 1$ steps of one chunk of $n/p$ entries.
- Every rank sends $2(p-1)/p$ of the array whatever $p$ is, so adding ranks does not add traffic per link.
- The ring adds in its own order: the result equals `numpy.sum` to rounding, and every rank holds the same bits.
- A blocking send on a ring deadlocks once messages outgrow the pipe buffer; even ranks send first and odd ranks receive first.
- A crashed rank must stop the run: `spawn` collects results against a deadline and terminates every rank on the first error.

## Review questions

1. Why does rank $r$ add the received data into chunk $(r - s - 2) \bmod p$ and not into the chunk it just sent?

   <details><summary>Answer</summary>

   The left neighbour $r-1$ sent its chunk $((r-1) - s - 1) \bmod p = (r - s - 2) \bmod p$. That is the chunk that arrived, so that is where the partial sum continues. Adding into the sent chunk mixes partial sums of two different chunks.

   </details>

2. Why do all ranks hold identical bits even though the result differs from `numpy.sum`?

   <details><summary>Answer</summary>

   Each chunk is summed exactly once, by its walk around the ring, in a fixed order starting at rank $j+1$. The all-gather then only copies that result. The order differs from `numpy.sum`'s, so the last bits can differ, but no rank ever re-adds anything.

   </details>

3. Why does doubling $p$ not double the traffic per rank?

   <details><summary>Answer</summary>

   Each rank sends $2(p-1)$ chunks, but each chunk is $n/p$ entries, so the total is $2(p-1)n/p < 2n$. More ranks means more, smaller messages.

   </details>

4. What happens if every rank sends first and the message is 1 MiB?

   <details><summary>Answer</summary>

   A pipe buffers a few kilobytes, so every `send` blocks until the right neighbour reads. Every neighbour is itself blocked in `send`, so nobody reaches `recv`: a cycle of waits. `spawn`'s deadline turns that into a `TimeoutError` instead of a hang.

   </details>
