<!-- ol:module L8.3 -->
# Paged KV cache in Python

## Overview

| | |
|---|---|
| **Module** | `L8.3` · side · Python · Pass 6 · 4 to 6 h (optional) |
| **You build** | `python/tinyllm/infer/paged.py`: `PagedKVCache` (`add_seq`, `fork`, `append`, `block_table`, `seq_len`, `gather`, `free`, `stats`, `num_free_blocks`, `close`) and `OutOfBlocks` |
| **Contract** | [`course/contracts/py/tinyllm/infer/paged.pyi`](../../../course/contracts/py/tinyllm/infer/paged.pyi); the implementation owns its NumPy blocks |
| **Tests** | `course/tests/L8.3/test_paged.py`, 10 tests (what they check: section 4) · your own tests in `python/tests/l8-3-paged/`, rung R4, graded by mutation (threshold 0.80) |
| **Needs** | `L8.2` [the contiguous KV cache](02-kv-cache-and-generate.md) (the oracle) |
| **Used by** | no module: the Rust engine (`L10.2`, `L10.4`) reimplements paging and reads this module as its behavioral reference; your `generate --cache paged` can run on it |
| **Milestone** | none (optional; `MS-L8` checks the contiguous cache) |
| **Optional depth** | Kwon et al., [*Efficient Memory Management for LLM Serving with PagedAttention*](https://arxiv.org/abs/2309.06180), section 4; Silberschatz et al., *Operating System Concepts*, chapter 9 (paging) |

## Key Takeaways

- A paged cache is a block table per sequence: position $p$ lives in slot $p \bmod B$ of block `table[p // B]`, so memory grows one block at a time (`test_hand_example_block_table`).
- The cache stores float16 K and V values in NumPy blocks and gathers them into contiguous arrays (`test_gather_preserves_head_token_dimension_order`).
- A fork copies a table and takes references; the first write into a shared block copies that block alone (`test_fork_shares_until_a_write`).
- Paged and contiguous float16 caches hold the same numbers, so a decoder gives the same logits and the same greedy tokens from either (`test_paged_matches_contiguous_decoding`).
- An append that cannot get its blocks raises `OutOfBlocks` before anything visible changes, so the scheduler can preempt and retry (`test_out_of_blocks_changes_nothing`).

## How to work this chapter

```bash
ol start L8.3               # stubs python/tinyllm/infer/paged.py
ol tests L8.3               # read the test catalog first
ol check L8.3               # exit code is the verdict; then grades your tests by mutation
ol check L8.3 --ref-deps    # only if L8.2 is not passing yet
ol diff  L8.3               # after passing: your code against the reference
```

Start with `add_seq`, `append` for one token at a time, and `gather`; then multi-token chunks that cross block boundaries; then `fork` with copy on write; then `free` and `OutOfBlocks`.

---

## 1. Why now

`L8.2`'s `KVCache` preallocates `max_len` positions per sequence and layer. With a 2,048-token context and SmolLM2's 30 layers, 3 KV heads, and 64 dimensions, that is $2048 \cdot 30 \cdot 2 \cdot 3 \cdot 64 \cdot 2 = 47$ MB per sequence in float16, whether the sequence ends after 20 tokens or 2,000; and beam search with 4 beams copies the whole prompt 4 times. You now build fixed-size blocks in Python, with reference counts and copy on write. This module matches the contiguous cache, so `generate` can switch to paged memory and the Rust engine (`L10.4`) has a clear behavioral reference. It is optional: no later Python module calls it, because the engine reimplements paging in Rust. Do it to see block tables and copy on write in a few hundred lines before you meet them in `L10.2` and `L10.4`.

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

`gather(0, 0)` gives K positions `0.5 1.5 2.5 3.5 4.5` in the first dimension. This is `test_hand_example_block_table`. A fork now shares `b0`, `b1`, `b2` (each at refcount 2, still 3 used); the child's sixth token goes into `b2` slot 1, so `b2` is copied first (4 used), and the parent's view is unchanged.

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

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `test_hand_example_block_table` | unit | section 3: a block only when a position needs it; the gathered token values; allocation counts | you and the tests agree on the mapping |
| `test_gather_preserves_head_token_dimension_order` | unit | gathered arrays use `[heads, tokens, dimensions]` order | attention receives the expected layout |
| `test_gather_matches_contiguous_float16` | differential | chunks of 1 to 7 tokens, 3 layers, 2 interleaved sequences equal a contiguous float16 cache exactly | positions in order across block boundaries |
| `test_layers_keep_independent_lengths` | unit | each layer can append independently while sharing the sequence block table | transformer layers advance in order |
| `test_fork_shares_until_a_write` | unit | a fork allocates nothing; the first write into the shared partial block copies it; full blocks stay shared | beam search and parallel sampling |
| `test_free_restores_the_pool` | unit | a forked block returns only when both holders free it; freeing all restores every block | KV usage back to baseline after each request |
| `test_out_of_blocks_changes_nothing` | fault | a multi-block append that cannot be satisfied leaves length, values, and allocation counts as they were | the scheduler preempts and retries safely |
| `test_bad_arguments` | boundary | bad shapes, empty chunks, unknown or reused ids, bad layers, sizes below 1 | engine bugs fail loudly |
| `test_paged_matches_contiguous_decoding` | differential | a 2-layer toy decoder: logits within 1e-5 and the same greedy ids from the paged cache and from your `L8.2` `KVCache(dtype=float16)`, through a prompt and two forked branches | the equivalence `generate --cache paged` relies on |
| `test_random_ops_conserve_blocks` | property | $10^4$ seeded add, fork, append, free operations against a numpy oracle; allocated and free block counts add up after each; freeing all restores every block | no refcount drifts or leaks over a long run |

**Your tests (rung R4).** Write `python/tests/l8-3-paged/test_*.py`, importing only names from `contracts/py` (`tinyllm.infer.paged`), with properties in prose turned into code (Hypothesis is allowed): "gather after any sequence of appends equals the concatenation of what was appended, in float16", "a fork then an append to the child never changes the parent", "freeing every sequence returns every block", "an `OutOfBlocks` append changes nothing". The planted bugs "a fork copies the table without references" (`s06`) and "a shared block is written in place" (`s07`) are required; overall 80%.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Allocating a block when a write ends exactly on a block boundary | one block per sequence wasted; tables longer than $\lceil n / B \rceil$ | `test_hand_example_block_table` (mutant `s01`) |
| Gathering whole blocks | the partial last block returns stale positions | `test_hand_example_block_table` (mutant `s02`) |
| Viewing a block as `[B][H][D]` | dimensions land in the wrong slots | `test_gather_preserves_head_token_dimension_order` (mutant `s03`) |
| Writing K into the V slab | attention mixes the wrong values | `test_gather_matches_contiguous_float16` (mutant `s04`) |
| A chunk ignoring its starting slot | the write runs past the block's slab | `test_gather_matches_contiguous_float16` (mutant `s05`) |
| Forking without taking a reference on each shared block | the parent's free returns blocks the child still reads | `test_fork_shares_until_a_write`, `test_random_ops_conserve_blocks` (mutant `s06`) |
| Writing into a shared block in place | the parent sees the child's tokens | `test_fork_shares_until_a_write`, `test_paged_matches_contiguous_decoding` (mutant `s07`) |
| Freeing the table without releasing blocks | blocks leak on every request | `test_free_restores_the_pool` (mutant `s08`) |
| Releasing only the first block | leaks all but one block per request | `test_free_restores_the_pool` (mutant `s09`) |
| Moving the length before allocating | a failed append still grows the sequence | `test_out_of_blocks_changes_nothing` (mutant `s10`) |
| Allocating new blocks one at a time | a failed multi-block append keeps what it took | `test_out_of_blocks_changes_nothing` (mutant `s11`) |
| Gathering every layer to the longest layer's length | a layer that has not been written yet returns positions it never stored | `test_layers_keep_independent_lengths` (mutant `s12`) |
| Accepting an empty chunk | silent no-ops hide engine bugs | `test_bad_arguments` (mutant `s13`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `L8.2` | the contiguous float16 cache is the oracle; `generate(cache="paged")` takes this cache |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `PagedKVCache` | vLLM's `KVCacheManager` | block tables on the GPU, prefix-cache hits at allocation, sliding-window and hybrid layouts | [`vllm/v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py) |
| `fork` with copy on write | vLLM v0 `BlockSpaceManager.fork` | the same refcounted fork for beam search | [`vllm/core/block_manager.py`](https://github.com/vllm-project/vllm/blob/v0.6.6/vllm/core/block_manager.py) |
| float16 blocks | FlashInfer paged KV | page tables consumed directly by fused attention kernels | [FlashInfer](https://github.com/flashinfer-ai/flashinfer) `include/flashinfer/page.cuh` |
