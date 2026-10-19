# Paged KV cache in Python

## Overview

| | |
|---|---|
| **Module** | `L8.3` · side · Python · Pass 6 · 2 h reading (optional) |
| **Solution** | [`paged.py`](../../../../course/solutions/python/tinyllm/infer/paged.py): `PagedKVCache` (`add_seq`, `fork`, `append`, `block_table`, `seq_len`, `gather`, `free`, `stats`, `num_free_blocks`, `close`) and `OutOfBlocks`; the `PagedHook` adapter in [`cli_infer.py`](../../../../course/solutions/entry/python/tinyllm/cli_infer.py); model property tests in [`test_paged.py`](../../../../course/solutions/learner-tests/L8.3/python/tests/l8-3-paged/test_paged.py) |
| **Contract** | [`course/contracts/py/tinyllm/infer/paged.pyi`](../../../../course/contracts/py/tinyllm/infer/paged.pyi); the implementation owns its NumPy blocks |
| **Needs** | `L8.2` [the contiguous KV cache](02-kv-cache-and-generate.md) (the oracle) |
| **Used by** | no module: the Rust engine (`L10.2`, `L10.4`) reimplements paging and reads this module as its behavioral reference; the system's `generate --cache paged` runs on it and gives the same greedy text as the contiguous cache |
| **Optional depth** | Kwon et al., [*Efficient Memory Management for LLM Serving with PagedAttention*](https://arxiv.org/abs/2309.06180), section 4; Silberschatz et al., *Operating System Concepts*, chapter 9 (paging) |

## Key Takeaways

- A paged cache is a block table per sequence: position $p$ lives in slot $p \bmod B$ of block `table[p // B]`, so memory grows one block at a time.
- The cache stores float16 K and V values in NumPy blocks and gathers them into contiguous arrays.
- A fork copies a table and takes references; the first write into a shared block copies that block alone.
- Paged and contiguous float16 caches hold the same numbers, so a decoder gives the same logits and the same greedy tokens from either.
- An append that cannot get its blocks raises `OutOfBlocks` before anything visible changes, so the scheduler can preempt and retry.

## How to read this chapter

Sections 1 to 3 derive block tables, per-layer lengths, copy on write, and failure without side effects, and trace five appends and a fork by hand. Section 4 fixes the interface and the cases a correct solution must handle. Section 5 walks through the solution in the order it is easiest to understand: allocation, `append` one token at a time, chunks that cross block boundaries, `fork` with copy on write, then `free` and `OutOfBlocks`. Try to predict each step of the solution before reading it.

---

## 1. Why now

`L8.2`'s `KVCache` preallocates `max_len` positions per sequence and layer. With a 2,048-token context and SmolLM2's 30 layers, 3 KV heads, and 64 dimensions, that is $2048 \cdot 30 \cdot 2 \cdot 3 \cdot 64 \cdot 2 = 47$ MB per sequence in float16, whether the sequence ends after 20 tokens or 2,000; and beam search with 4 beams copies the whole prompt 4 times. This module stores the cache in fixed-size blocks in Python, with reference counts and copy on write. This module matches the contiguous cache, so `generate` can switch to paged memory and the Rust engine (`L10.4`) has a clear behavioral reference. It is optional: no later Python module calls it, because the engine reimplements paging in Rust. Read it to see block tables and copy on write in a few hundred lines before meeting them in `L10.2` and `L10.4`.

## 2. Principles

| Symbol | Meaning | Type |
|---|---|---|
| $B$ | `block_size`, positions per block | `int` |
| $L$, $H$, $D$ | layers, KV heads, head dimension | `int` |
| $\ell_l$ | `seq_len(seq, l)`: positions layer $l$ holds | `int` |
| table | the sequence's block ids in position order | `list[int]` |
| $T$ | positions appended by one call | `int` |

**Block tables.** Position $p$ of a sequence lives in slot $p \bmod B$ of block `table[p // B]`. A sequence of $n$ positions needs $\lceil n / B \rceil$ blocks; the last one is usually partial. Appending $T$ positions starting at $\ell$ touches blocks $\lfloor \ell / B \rfloor$ through $\lfloor (\ell + T - 1) / B \rfloor$ and allocates the ones past the end of the table.

**Layers append separately.** A forward pass appends layer 0's keys and values, then layer 1's, and so on, so mid-pass the layers hold different lengths. Each block holds all layers, while each layer has its own length. A page table may contain enough blocks for the longest layer; unused slots remain zero until that layer appends there.

**The block view.** Each page stores a float16 K slab and a float16 V slab. Appends copy slices into the requested layer and slots; gather copies those slots in sequence order. Values round to float16 on write, so gather matches a contiguous float16 cache exactly.

**Copy on write.** `fork(parent, child)` copies the table and increments every block reference. Before `append` writes into a shared block, it copies that block for the writer. The parent and child then diverge in their last block only; every full block of the shared prompt stays shared.

**Failure without side effects.** `append` checks capacity before allocating or changing the table, copies shared blocks it will touch, then writes and advances the length. If the cache has too few free blocks, it raises `OutOfBlocks`, and `gather` and `seq_len` return what they did before.

## 3. Worked example by hand

$B = 2$, one layer, one head, $D = 2$. Append five tokens one at a time, with K of token $t$ = `[t + 0.5, -t]` and V = `-K`:

| After token | Length | Table | Where the token went | Pool (free, used) of 4 |
|---|---|---|---|---|
| 0 | 1 | `[b0]` | `b0` slot 0 | (3, 1) |
| 1 | 2 | `[b0]` | `b0` slot 1 | (3, 1) |
| 2 | 3 | `[b0, b1]` | `b1` slot 0 | (2, 2) |
| 3 | 4 | `[b0, b1]` | `b1` slot 1 | (2, 2) |
| 4 | 5 | `[b0, b1, b2]` | `b2` slot 0 | (1, 3) |

`gather(0, 0)` gives K positions `0.5 1.5 2.5 3.5 4.5` in the first dimension. A fork now shares `b0`, `b1`, `b2` (each at refcount 2, still 3 used); the child's sixth token goes into `b2` slot 1, so `b2` is copied first (4 used), and the parent's view is unchanged.

## 4. The interface

```python
class OutOfBlocks(RuntimeError): ...

class PagedKVCache:
    def __init__(self, num_blocks: int, block_size: int, n_layers: int, n_kv_heads: int, d_head: int): ...
    def add_seq(self, seq_id: int) -> None
    def fork(self, parent: int, child: int) -> None
    def append(self, seq_id: int, layer: int, k, v) -> None        # k, v: [n_kv_heads, T, d_head]
    def block_table(self, seq_id: int) -> NDArray                  # int32
    def seq_len(self, seq_id: int, layer: int = 0) -> int
    def gather(self, seq_id: int, layer: int) -> tuple[NDArray, NDArray]   # float16 [H, len, D]
    def free(self, seq_id: int) -> None
    def stats(self) -> dict[str, int]                              # free, used, cached, evictions
    def num_free_blocks(self) -> int
    def close(self) -> None                                        # also __exit__ and garbage collection
```

Shapes have no batch axis: one sequence per call. The Python allocator owns its blocks and page tables; it does not expose a native pointer.

### What a correct solution must handle

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| the hand example's block table | unit | section 3: a block only when a position needs it; the gathered token values; allocation counts | the solution and the reader agree on the mapping |
| gather keeps head, token, dimension order | unit | gathered arrays use `[heads, tokens, dimensions]` order | attention receives the expected layout |
| gather matches a contiguous float16 cache | differential | chunks of 1 to 7 tokens, 3 layers, 2 interleaved sequences equal a contiguous float16 cache exactly | positions in order across block boundaries |
| layers keep independent lengths | unit | each layer can append independently while sharing the sequence block table | transformer layers advance in order |
| a fork shares until a write | unit | a fork allocates nothing; the first write into the shared partial block copies it; full blocks stay shared | beam search and parallel sampling |
| free restores the pool | unit | a forked block returns only when both holders free it; freeing all restores every block | KV usage back to baseline after each request |
| out of blocks changes nothing | fault | a multi-block append that cannot be satisfied leaves length, values, and allocation counts as they were | the scheduler preempts and retries safely |
| bad arguments | boundary | bad shapes, empty chunks, unknown or reused ids, bad layers, sizes below 1 | engine bugs fail loudly |
| paged matches contiguous decoding | differential | a 2-layer toy decoder: logits within 1e-5 and the same greedy ids from the paged cache and from `L8.2`'s `KVCache(dtype=float16)`, through a prompt and two forked branches | the equivalence `generate --cache paged` relies on |
| random operations conserve blocks | property | $10^4$ seeded add, fork, append, free operations against a numpy oracle; allocated and free block counts add up after each; freeing all restores every block | no refcount drifts or leaks over a long run |

### Model property tests

[`test_paged.py`](../../../../course/solutions/learner-tests/L8.3/python/tests/l8-3-paged/test_paged.py) imports only the contract (`tinyllm.infer.paged`) and turns properties stated in prose into Hypothesis tests: "gather after any sequence of appends equals the concatenation of what was appended, in float16", "a fork then an append to the child never changes the parent", "freeing every sequence returns every block", "an `OutOfBlocks` append changes nothing". Properties suit a data structure like this one: the bugs that matter (a missing reference, a shared block written in place) appear only after particular sequences of operations, which random sequences find and hand-written examples miss.

## 5. The solution, step by step

### 5.1 From first principles

1. Memory should grow with what a sequence actually holds. So storage is a pool of fixed-size blocks, and each sequence owns a **table** of block ids in position order; position $p$ lives at slot $p \bmod B$ of block `table[p // B]`.
2. One block holds all layers for its $B$ positions, so one table serves every layer. Layers append one after another within a forward pass, so each layer keeps its own length, and `gather` reads exactly that many positions.
3. A block's layout is `[n_layers, n_kv_heads, block_size, d_head]`. Then one layer's slice is `[H, B, D]`, and gathering in table order concatenates along the token axis to `[H, len, D]`, the layout attention expects.
4. Store float16 and return float16: the cache then holds exactly the numbers a contiguous float16 cache holds, which is the equivalence the decoder needs.
5. Sharing (forks for beam search or parallel samples) is reference counting. A fork copies the table and increments each block's count; a block is free when its count reaches 0.
6. A shared block must never be written in place. Before an append writes into a block with refcount above 1, it gives the writer its own copy (copy on write). Only blocks the write touches are copied, so full prompt blocks stay shared.
7. A failed append must leave no trace, so the scheduler can preempt and retry. Count everything the append needs (new blocks plus copies) and compare with the free list **before** allocating, copying, or touching a table or length.

### 5.2 The code, annotated

**State and allocation.**

```python
@dataclass
class _Block:
    k: NDArray
    v: NDArray
    refs: int = 1

    # in __init__:
        self._blocks = [None] * self.num_blocks           # None = not allocated
        self._free = list(range(self.num_blocks - 1, -1, -1))   # pop() hands out b0 first
        self._tables = {}                                 # seq_id -> [block ids]
        self._lens = {}                                   # seq_id -> [length per layer]

    def _alloc_id(self):
        block_id = self._free.pop()
        shape = (self.n_layers, self.n_kv_heads, self.block_size, self.d_head)   # step 3
        self._blocks[block_id] = _Block(
            np.zeros(shape, dtype=np.float16), np.zeros(shape, dtype=np.float16)
        )
        return block_id

    def _unref(self, block_id):
        block = self._block(block_id)
        block.refs -= 1
        if block.refs == 0:
            self._blocks[block_id] = None                 # drop the arrays
            self._free.append(block_id)                   # back to the pool
```

The free list is a stack initialised in reverse, so blocks come out as `b0, b1, b2, ...`, matching section 3's table.

**Sequences, forks, and frees.**

```python
    def add_seq(self, seq_id):
        # ... unknown id required
        self._tables[seq_id] = []                         # no block until a position needs one
        self._lens[seq_id] = [0] * self.n_layers

    def fork(self, parent, child):
        table = self._table(parent)
        # ... child must be new
        for block_id in table:
            block = self._block(block_id)
            block.refs += 1                               # share, do not copy (step 5)
        self._tables[child] = list(table)                 # a copy of the table, not of the blocks
        self._lens[child] = list(self._lens[parent])

    def free(self, seq_id):
        table = self._table(seq_id)
        for block_id in table:
            self._unref(block_id)                         # every block, not just the first
        del self._tables[seq_id]
        del self._lens[seq_id]
```

**Append: plan, check, then act.**

```python
    def append(self, seq_id, layer, k, v):
        table = self._table(seq_id)
        layer = self._layer(layer)
        k16 = np.asarray(k, dtype=np.float16)             # round on the way in (step 4)
        v16 = np.asarray(v, dtype=np.float16)
        want = (self.n_kv_heads, k16.shape[1] if k16.ndim == 3 else -1, self.d_head)
        if k16.ndim != 3 or k16.shape != want or v16.shape != want or want[1] < 1:
            raise ValueError(...)                         # also rejects an empty chunk
        start = self._lens[seq_id][layer]
        end = start + want[1]
        first, last = start // self.block_size, (end - 1) // self.block_size   # blocks this write touches
        # existing blocks in range that someone else also holds: they need a private copy
        shared = [
            i for i in range(first, min(last + 1, len(table)))
            if self._block(table[i]).refs > 1
        ]
        new_count = max(0, last + 1 - len(table))         # blocks past the end of the table
        if len(self._free) < len(shared) + new_count:
            raise OutOfBlocks("the KV cache has no free block left")   # nothing has changed yet
```

`last` uses `end - 1`, the last position written, so a write that ends exactly on a block boundary does not allocate the next block. With $B = 2$, token 1 (`start = 1, end = 2`) gives `first = last = 0`: no new block, as in section 3's second row.

```python
        # Reserve every required block before changing references or tables.
        replacements = {}
        for i in shared:
            old_id = table[i]
            old = self._block(old_id)
            new_id = self._alloc_id()
            clone = self._block(new_id)
            clone.k[...] = old.k                          # copy every layer: other layers may
            clone.v[...] = old.v                          # already have data in this block
            replacements[i] = new_id
        allocated = [self._alloc_id() for _ in range(new_count)]
        for i, new_id in replacements.items():
            self._unref(table[i])                         # the writer lets go of the shared block
            table[i] = new_id
        table.extend(allocated)
```

Because the capacity check already passed, every `_alloc_id` here succeeds, so there is no half-allocated state to undo.

```python
        pos = start
        while pos < end:
            block_index, slot = divmod(pos, self.block_size)
            count = min(self.block_size - slot, end - pos)   # up to the end of this block
            src = slice(pos - start, pos - start + count)
            block = self._block(table[block_index])
            block.k[layer, :, slot : slot + count, :] = k16[:, src, :]
            block.v[layer, :, slot : slot + count, :] = v16[:, src, :]   # V into the V slab
            pos += count
        self._lens[seq_id][layer] = end                   # the length moves last
```

The loop writes a chunk in block-sized pieces, starting at the chunk's slot within its first block. Section 3's fork: the child appends token 5 at `start = 5`; block index 2 has refcount 2, so it is in `shared`, a fourth block is allocated and filled with `b2`'s contents, the child's table points at it, `b2`'s refcount drops to 1, and the write lands in the copy's slot 1. The parent still reads the original `b2`.

**Gather.**

```python
    def gather(self, seq_id, layer):
        table = self._table(seq_id)
        layer = self._layer(layer)
        n = self._lens[seq_id][layer]                     # this layer's own length (step 2)
        k = np.empty((self.n_kv_heads, n, self.d_head), dtype=np.float16)
        v = np.empty_like(k)
        for block_index in range((n + self.block_size - 1) // self.block_size):
            start = block_index * self.block_size
            count = min(self.block_size, n - start)       # the last block may be partial
            block = self._block(table[block_index])
            k[:, start : start + count] = block.k[layer, :, :count]
            v[:, start : start + count] = block.v[layer, :, :count]
        return k, v
```

`close` frees every sequence and is also called by `__exit__` and `__del__`, so a cache used in a `with` block always returns its blocks.

**Plugging into `generate`.** `L8.2`'s `generate` takes any object with `update`, `seq_len`, and `positions`. The entry point's `PagedHook` wraps one sequence of the paged cache in that interface:

```python
class PagedHook:
    def __init__(self, paged, n_layers):
        self.p, self.n_layers = paged, n_layers
        self.p.add_seq(0)

    def update(self, layer, k_new, v_new):
        self.p.append(0, layer, np.asarray(k_new)[0], np.asarray(v_new)[0])   # drop the batch axis
        k, v = self.p.gather(0, layer)
        return k[None].astype(np.float32), v[None].astype(np.float32)        # stored values, as L8.2 returns

    def seq_len(self, layer=None):
        if layer is not None:
            return self.p.seq_len(0, layer)
        return min(self.p.seq_len(0, i) for i in range(self.n_layers))    # committed length, as in L8.2
    # positions and mask: L8.2's definitions over seq_len()
```

Because both caches store float16 and return the stored values, `generate --cache paged` and `generate --cache contiguous --kv-dtype f16` attend over identical keys and values.

Full solution: [`paged.py`](../../../../course/solutions/python/tinyllm/infer/paged.py).

### 5.3 Why this solution

- **Memory.** A sequence of $n$ positions holds $\lceil n/B \rceil$ blocks, wasting at most $B - 1$ slots, against `max_len` slots for the contiguous cache. A fork costs nothing until a write, and then one block.
- **Cost.** `append` is $O(T)$ copying plus one block copy per shared block touched (at most one for decode). `gather` copies the whole sequence, $O(n)$ per layer per step: acceptable for a reference, and the reason real engines (vLLM, FlashInfer, `L10.4`) instead let the attention kernel read blocks through the table directly.
- **One block for all layers.** A block per layer would need $L$ tables or $L$ times the bookkeeping; one table per sequence matches how vLLM shares block ids across layers.
- **Check before acting.** Rolling back a half-done append (undoing copies and allocations) is error-prone. Counting first makes failure a pure early return.
- **Reverse free stack.** Any order is correct; handing out low ids first makes tables predictable and section 3 reproducible.

### 5.4 Explain it back

1. Why does `append` compute the last touched block from `end - 1` rather than `end`?
   <details><summary>Answer</summary>`end` is one past the last position written. A chunk that ends exactly at a block boundary writes nothing into the next block, so allocating it from `end // B` would waste a block per sequence.</details>
2. Why does a copy-on-write clone copy every layer of the block, not only the layer being written?
   <details><summary>Answer</summary>The block holds all layers' data for its positions, and other layers may already have written there (layers append one after another). The writer's table now points at the clone for all layers, so the clone must carry everything.</details>
3. Why is the capacity check done before any allocation?
   <details><summary>Answer</summary>So `OutOfBlocks` leaves lengths, tables, refcounts, and the free list exactly as they were. The scheduler can then preempt a sequence and retry the same append without repairing state.</details>
4. Why must `gather` use the requested layer's length and not the longest layer's?
   <details><summary>Answer</summary>Mid-pass, earlier layers hold more positions than later ones. The table may already include a block for the longest layer; reading that many positions for a later layer would return zero slots it never wrote.</details>
5. What happens to a block shared by a parent and child when only the parent is freed?
   <details><summary>Answer</summary>`free` calls `_unref`, which lowers the count from 2 to 1; the block stays allocated for the child. It returns to the pool only when the child frees it too.</details>

## 6. Pitfalls

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Allocating a block when a write ends exactly on a block boundary | one block per sequence wasted; tables longer than $\lceil n / B \rceil$ | `last = (end - 1) // block_size` |
| Gathering whole blocks | the partial last block returns stale positions | `count = min(block_size, n - start)` |
| Viewing a block as `[B][H][D]` | dimensions land in the wrong slots | blocks are `[layers, heads, block_size, d_head]` and are sliced as `[layer, :, slots, :]` |
| Writing K into the V slab | attention mixes the wrong values | `block.k` gets `k16`, `block.v` gets `v16` |
| A chunk ignoring its starting slot | the write runs past the block's slab | `divmod(pos, block_size)` gives the slot; `count` stops at the block's end |
| Forking without taking a reference on each shared block | the parent's free returns blocks the child still reads | `fork` increments `refs` for every block in the table |
| Writing into a shared block in place | the parent sees the child's tokens | blocks with `refs > 1` in the write's range are cloned first (step 6) |
| Freeing the table without releasing blocks | blocks leak on every request | `free` calls `_unref` on every block |
| Releasing only the first block | leaks all but one block per request | `free` loops over the whole table |
| Moving the length before allocating | a failed append still grows the sequence | the length is set last, after the writes |
| Allocating new blocks one at a time | a failed multi-block append keeps what it took | one capacity check for copies plus new blocks before any allocation (step 7) |
| Gathering every layer to the longest layer's length | a layer that has not been written yet returns positions it never stored | `gather` reads `self._lens[seq_id][layer]` |
| Accepting an empty chunk | silent no-ops hide engine bugs | `want[1] < 1` raises `ValueError` |

## 7. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L8.2` | the contiguous float16 cache is the oracle; `generate(cache="paged")` takes this cache |

## Going further

| Solution piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `PagedKVCache` | vLLM's `KVCacheManager` | block tables on the GPU, prefix-cache hits at allocation, sliding-window and hybrid layouts | [`vllm/v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py) |
| `fork` with copy on write | vLLM v0 `BlockSpaceManager.fork` | the same refcounted fork for beam search | [`vllm/core/block_manager.py`](https://github.com/vllm-project/vllm/blob/v0.6.6/vllm/core/block_manager.py) |
| float16 blocks | FlashInfer paged KV | page tables consumed directly by fused attention kernels | [FlashInfer](https://github.com/flashinfer-ai/flashinfer) `include/flashinfer/page.cuh` |
