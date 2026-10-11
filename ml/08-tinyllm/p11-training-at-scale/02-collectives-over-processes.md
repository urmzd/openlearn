<!-- ol:module L11.2 -->
# Collectives over processes: ring all-reduce

## Overview

| | |
|---|---|
| **Module** | `L11.2` · side · Python · Pass 9 (optional) · 4 h |
| **You build** | `python/tinyllm/dist/comm.py`: `chunk_bounds`, `Comm` (`send`, `recv`, `reduce_scatter`, `all_gather`, `all_reduce`, `broadcast`, `barrier`), `spawn` |
| **Contract** | [`course/contracts/py/tinyllm/dist/comm.pyi`](../../../course/contracts/py/tinyllm/dist/comm.pyi) |
| **Tests** | `course/tests/L11.2/` (what they check: section 4) · your own tests in `python/tests/l11-2-comm/`, rung R5, graded by mutation (threshold 0.80, every pitfall mutant required) |
| **Needs** | nothing to build first · reading: `M05.1` counting bytes |
| **Used by** | `L11.3` DDP and ZeRO run every collective through `Comm` |
| **Milestone** | none; this optional side module is not part of `MS-L11` |
| **Optional depth** | Patarasuk and Yuan, ["Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations"](https://doi.org/10.1016/j.jpdc.2008.09.002) (JPDC, 2009); Thakur, Rabenseifner, and Gropp, "Optimization of Collective Communication Operations in MPICH" (IJHPCA, 2005); the NCCL documentation on collective operations |

## Key Takeaways

- An all-reduce is a reduce-scatter followed by an all-gather; on a ring each half takes $p - 1$ steps of one chunk of $n/p$ entries (`test_hand_example_ring_allreduce`, `test_reduce_scatter_and_all_gather`).
- Every rank sends $2(p-1)/p$ of the array whatever $p$ is, so adding ranks does not add traffic per link (`test_bytes_moved_is_2_p_minus_1_over_p`).
- The ring adds in its own order: the result equals `numpy.sum` to rounding, and every rank holds the same bits (`test_allreduce_matches_numpy_sum`).
- A blocking send on a ring deadlocks once messages outgrow the pipe buffer; even ranks send first and odd ranks receive first (`test_large_messages_do_not_deadlock`).

## How to work this chapter

```bash
ol start L11.2              # stubs comm.py into your repo
ol tests L11.2              # read the test catalog first
ol check L11.2              # course tests, then your tests graded by mutation
ol mutate L11.2             # the full mutation grade of your tests
ol diff  L11.2              # after passing: your code against the reference
```

---

## 1. Why now

`L11.1` made one process train the capstone on a laptop. Every larger run splits the batch over several workers instead: each computes the gradient of its own slice, and before anyone takes a step the gradients must be averaged across all of them. That averaging is an all-reduce, and it is the operation every data-parallel step waits on. On a laptop the workers are processes on the same machine, which is enough to build the real algorithm: processes cannot share Python objects, so every byte moves through an explicit channel, and you can count those bytes. This optional module builds the collectives (`all_reduce`, `reduce_scatter`, `all_gather`, `broadcast`, `barrier`) over pipes with the bandwidth-optimal ring, and `L11.3` builds data parallelism and ZeRO on top.

## 2. Principles

| Symbol | Meaning | Type / shape |
|---|---|---|
| $p$ (`world`) | number of processes | `int` |
| $r$ (`rank`) | this process's number, $0 \le r < p$ | `int` |
| $x_r$ | rank $r$'s input array, $n$ entries | `float64[n]` |
| $\sum_r x_r$ | the elementwise sum all ranks want | `float64[n]` |
| $c_j$ | chunk $j$ of the flat array: entries $[a_j, b_j)$ from `chunk_bounds` | slice |
| right, left | ranks $(r+1) \bmod p$ and $(r-1) \bmod p$ | `int` |
| $\beta$ | bytes per entry (8 for float64) | `int` |

**Processes and channels.** `spawn(fn, p)` starts $p$ processes with multiprocessing's "spawn" method (a fresh interpreter each, so `fn` must be importable by name) and connects every pair of ranks with a full-duplex pipe. `send(x, dst)` writes a copy of the array, `recv(src)` blocks until the next array from `src` arrives; messages between two ranks arrive in order. A pipe holds only a few kilobytes: a larger `send` blocks until the receiver reads.

**Why not gather to one rank.** The obvious all-reduce sends every array to rank 0, adds, and sends the sum back: rank 0 receives $(p-1)n$ entries and sends $(p-1)n$, so its link carries traffic that grows with $p$, while the other links idle. The ring keeps every link equally busy.

**Chunks.** Split the flat array into $p$ contiguous chunks $c_0, \dots, c_{p-1}$, sizes as `numpy.array_split` (the first $n \bmod p$ chunks one entry larger). Rank $r$ owns chunk $r$; ZeRO (`L11.3`) shards parameters by the same bounds, so they must agree everywhere.

**Reduce-scatter on a ring.** In step $s = 0, \dots, p-2$, rank $r$ sends chunk $(r - s - 1) \bmod p$ to its right neighbour, receives chunk $(r - s - 2) \bmod p$ from its left neighbour, and adds what it received into its own copy of that chunk. Follow one chunk $c_j$: it starts at rank $j + 1$, which sends its $x_{j+1}$ part right; each rank adds its own part and passes the partial sum on; after $p - 1$ steps it reaches rank $j$ holding all $p$ contributions. So every rank ends holding its own chunk of the sum, $\big(\sum_r x_r\big)[c_r]$, having sent $p - 1$ chunks.

**All-gather on a ring.** Now each rank has one finished chunk and needs the others. In step $s$, rank $r$ sends chunk $(r - s) \bmod p$ right and receives chunk $(r - s - 1) \bmod p$ from the left. After $p - 1$ steps every rank has every chunk. All-reduce is reduce-scatter followed by all-gather.

**Bytes.** Each half sends $p - 1$ chunks of about $n/p$ entries, so

$$\text{bytes sent per rank} = 2\,\frac{p-1}{p}\, n\, \beta,$$

which approaches $2n\beta$ and never grows with $p$: doubling the ranks halves each chunk. A lower bound says no all-reduce can do better (each rank must send at least $(p-1)/p$ of its data out and receive the same in), which is why NCCL uses rings for large messages.

**Rounding.** The sum of chunk $c_j$ is accumulated in ring order starting at rank $j + 1$, not in rank order: floating-point addition is not associative, so the result differs from `numpy.sum` in the last bits. But each chunk is reduced once and then copied, so all ranks hold identical bits, which is what keeps data-parallel replicas identical.

**Deadlock.** If every rank does `send` then `recv` in a step, and the message is larger than the pipe buffer, every rank blocks in `send` waiting for its right neighbour to `recv`, which is itself blocked in `send`: a cycle of waits, forever. Breaking the cycle needs one rank that receives first. The contract's rule: even ranks send then receive, odd ranks receive then send. With $p \ge 2$ rank 1 is always a receiver-first, so the chain unwinds from there, for odd and even $p$ alike. MPI calls the alternatives `MPI_Sendrecv` and non-blocking sends.

**Broadcast and barrier.** `broadcast(x, src)` passes $x$ around the ring from `src`: each rank receives from the left and forwards right, except the rank just before `src`, which would send it back to where it started (and leave a message in that pipe that the next collective would read by mistake). A barrier makes every rank wait until all have arrived; it is multiprocessing's `Barrier`.

**Failure.** A rank that raises must stop the whole run with its traceback; otherwise its neighbours wait forever in a `recv`. `spawn` collects each rank's result or traceback, terminates the rest on the first error, and raises `TimeoutError` if the ranks do not finish in time.

## 3. Worked example by hand

Three ranks hold $x_0 = [1, 2, 3]$, $x_1 = [10, 20, 30]$, $x_2 = [100, 200, 300]$; $n = 3$, so each chunk is one entry: $c_0 = $ entry 0, $c_1 = $ entry 1, $c_2 = $ entry 2.

| step | rank 0 sends | rank 1 sends | rank 2 sends | rank 0 holds after | rank 1 holds after | rank 2 holds after |
|---|---|---|---|---|---|---|
| RS 0 | $c_2 = 3$ | $c_0 = 10$ | $c_1 = 200$ | $c_1$: $2 + 200 = 202$ | $c_2$: $30 + 3 = 33$ | $c_0$: $100 + 10 = 110$ |
| RS 1 | $c_1 = 202$ | $c_2 = 33$ | $c_0 = 110$ | $c_0$: $1 + 110 = 111$ | $c_1$: $20 + 202 = 222$ | $c_2$: $300 + 33 = 333$ |
| AG 0 | $c_0 = 111$ | $c_1 = 222$ | $c_2 = 333$ | gets $c_2 = 333$ | gets $c_0 = 111$ | gets $c_1 = 222$ |
| AG 1 | $c_2 = 333$ | $c_0 = 111$ | $c_1 = 222$ | gets $c_1 = 222$ | gets $c_2 = 333$ | gets $c_0 = 111$ |

In step RS 0, rank $r$ sends chunk $(r - 1) \bmod 3$ and adds what arrives into chunk $(r - 2) \bmod 3$: rank 0 receives rank 2's $c_1 = 200$ and adds its own 2. After reduce-scatter rank $r$ holds chunk $r$ of the sum: $[111]$, $[222]$, $[333]$; after all-gather every rank holds $[111, 222, 333]$. Each rank sent 4 entries of 8 bytes, 32 bytes, and $2 \cdot \tfrac{2}{3} \cdot 3 \cdot 8 = 32$. This is `test_hand_example_ring_allreduce`.

## 4. The interface

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

A worker is a module-level function `fn(comm, *args)`; `spawn` returns the workers' return values by rank. Collectives must be called by every rank in the same order.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_ring_allreduce` | unit | section 3: chunks, sums, 16 and 32 bytes | you and the test agree on the ring |
| `test_chunk_bounds` | unit | `numpy.array_split` sizes for $n < 23$, $p \le 5$ | ZeRO shards line up (`L11.3`) |
| `test_allreduce_matches_numpy_sum` | differential | $p = 2, 3, 4$ on 35 entries, sum and mean, bitwise-equal ranks, input untouched | DDP replicas stay identical |
| `test_bytes_moved_is_2_p_minus_1_over_p` | property | exactly $2(p-1)n\beta/p$ for $p = 2, 4, 5$ | the bandwidth argument |
| `test_reduce_scatter_and_all_gather` | unit | the halves on their own, unequal chunks | ZeRO stage 2 and 3 |
| `test_broadcast_from_every_src` | unit | every source, bytes per rank, a clean pipe afterwards | DDP's initial weights |
| `test_large_messages_do_not_deadlock` | boundary | 1 MiB messages on a ring of 3 | real gradient sizes |
| `test_send_recv_point_to_point` | unit | order, dtype, shape, a writable copy, bad ranks | the layer under everything |
| `test_rank_error_propagates` | boundary | a crash on rank 1 raises its traceback quickly | no silent hangs |
| `test_barrier` | unit | a late rank holds everyone | ordering side effects |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| 1. adding the received chunk into the one you just sent | wrong sums, different on every rank | `test_hand_example_ring_allreduce` (mutant `s01`) |
| 2. one step too few in reduce-scatter | partial sums for $p > 2$ | `test_allreduce_matches_numpy_sum` (mutant `s02`) |
| 3. one step too few in all-gather | stale chunks on some ranks | `test_reduce_scatter_and_all_gather` (mutant `s03`) |
| 4. `"mean"` returning the sum | gradients $p$ times too large | `test_allreduce_matches_numpy_sum` (mutant `s04`) |
| 5. every rank sends first | a deadlock as soon as a message exceeds the pipe buffer | `test_large_messages_do_not_deadlock` (mutant `s05`) |
| 6. accumulating into the caller's array | the caller's gradient changes under it | `test_allreduce_matches_numpy_sum` (mutant `s06`) |
| 7. chunks of the wrong sizes | ZeRO shards and collectives disagree | `test_chunk_bounds` (mutant `s07`) |
| 8. counting entries instead of bytes | the bandwidth numbers are off by $\beta$ | `test_bytes_moved_is_2_p_minus_1_over_p` (mutant `s08`) |
| 9. broadcasting back to the source | a stray message corrupts the next collective | `test_broadcast_from_every_src` (mutant `s09`) |
| 10. ignoring a rank's failure | the run "finishes" with a missing result | `test_rank_error_propagates` (mutant `s10`) |
| 11. a barrier that does not wait | side effects race | `test_barrier` (mutant `s11`) |
| 12. returning a read-only view of the message | the receiver cannot write its own array | `test_send_recv_point_to_point` (mutant `s12`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `M05.1` | counting bytes per parameter is the same arithmetic as counting bytes per link |
| Forward | `L11.3` | DDP all-reduces gradients; ZeRO uses reduce-scatter and all-gather on parameter chunks |

This module is optional (D20): a laptop capstone needs `L11.1`, not data parallelism. Without it, `L11.3` cannot run.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `Comm` over pipes | NCCL | rings and trees over NVLink and InfiniBand, chunked and pipelined so every link is busy at once | NCCL source, `src/collectives/` |
| `all_reduce` | `torch.distributed.all_reduce` | process groups, backends (NCCL, Gloo, MPI), async handles | `torch/distributed/distributed_c10d.py` |
| the ring | tree and recursive halving-doubling | lower latency for small messages ($\log p$ steps instead of $p$) | Thakur et al. (2005) |
| `spawn` | `torchrun` | rendezvous across machines, restarts, elastic world sizes | `torch/distributed/run.py` |
