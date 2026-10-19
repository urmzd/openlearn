# Systems Data Structures

## Overview

- **Primary references**: Cormen, Leiserson, Rivest, and Stein, *Introduction to Algorithms* (4th ed.), chapters 6 (heaps) and 11 (hash tables); Abseil, [Swiss Tables design notes](https://abseil.io/about/design/swisstables) (free)
- **Supplementary**: Celis, *Robin Hood Hashing* (1986); Bloom, [*Space/Time Trade-offs in Hash Coding with Allowable Errors*](https://dl.acm.org/doi/10.1145/362686.362692) (1970); Mirrokni, Thorup, and Zadimoghaddam, [*Consistent Hashing with Bounded Loads*](https://arxiv.org/abs/1608.01350) (free); Zheng et al., [*SGLang: Efficient Execution of Structured Language Model Programs*](https://arxiv.org/abs/2312.07104) (RadixAttention, free); the [practice C drills](../../../../practice/build/systems/c/) 01 to 04
- **Prerequisites**: the course primers for [C](../../software-craftsmanship/12-language-and-tool-primers/03-c.md), [Rust](../../software-craftsmanship/12-language-and-tool-primers/04-rust.md), and [Go](../../software-craftsmanship/12-language-and-tool-primers/06-go.md); [Discrete Math 2](../../math/06-discrete-math-2/) for load factors and hashing probability
- **Estimated time**: 1 to 2 weeks spread over course Passes 3, 6, and 7 (each structure sits right before its first caller)

## Key Takeaways

- **Every structure here has a caller in the system you build.** The hash maps index KV blocks and tokenizer merges, the heaps pick top-k tokens and the next request, the radix tree finds shared prompt prefixes, the Bloom filter screens duplicate paragraphs, and the ring routes requests to replicas.
- **Open addressing is a probe-length problem.** Linear probing, Robin Hood, and Swiss tables differ in how they bound the worst probe, not the average one; the load factor decides when that bound breaks.
- **The same idea appears in three languages**: C for the runtime (`tl_vec`, `tl_map`, the LRU, `tl_topk_f32`), Rust for the tokenizer and engine, Go for the gateway. The interface is the contract; the language is a detail.
- **Probabilistic structures trade certainty for space**: a Bloom filter never misses a member and is wrong about non-members at a rate you choose.

## How to Study

Implement each structure from its chapter before reading the reference design (Abseil's notes for Swiss tables, SGLang's paper for the radix tree). The interview-pattern chapters in this track ([01 Arrays & Hashing](../01-arrays-hashing/), [05 Trees](../05-trees/), [15 Probabilistic Structures](../15-probabilistic-structures/)) are the warm-ups. In the course, `ds.05` is the first hash-table chapter and `ds.06` the first heap chapter; the C versions in Pass 6 build on them.

---

# Concepts & Techniques

## Core Insight

A data structure in a production system is chosen by its worst case under the system's access pattern, not by its textbook complexity. The KV block pool needs a hash index that never stalls on a rehash and an LRU that evicts in O(1); the tokenizer needs a hash map tuned for small integer keys; the router needs a hash that moves few keys when a replica joins. Each chapter starts from the call site and derives the structure it needs.

## 1. Hash tables: Robin Hood and Swiss

**Key ideas**:
- **Robin Hood** (`ds.05`, Rust): on insert, a key that has travelled further from its home slot steals the slot; deletion shifts back instead of leaving tombstones. Called by the `L1.5` tokenizer for vocab and merge ranks.
- **Swiss table** (`ds.02`, C): control bytes with 7 bits of the hash, matched 8 or 16 at a time (SWAR), load factor 7/8. Builds on the practice `c/02` linear-probing baseline; called by the `rt.04` prefix-hash index and KV-transfer dedup.

## 2. Arrays, lists, and LRU

**Key ideas**:
- **`tl_vec`** (`ds.01`): a type-erased growable array; the block tables of `rt.04`.
- **Intrusive list and LRU** (`ds.03`): O(1) move-to-front and evict through embedded links; evictable cached KV blocks.

## 3. Heaps and top-k

**Key ideas**:
- **Lazy-deletion heap** (`ds.06`, Rust): generation counters invalidate stale entries; the BPE merge queue and the engine's waiting queue.
- **Top-k** (`ds.04`, optional C): a standalone size-$k$ min-heap over logits with ties to the lower index; parity uses fixture files from the Python reference.

## 4. Prefixes, membership, and placement

**Key ideas**:
- **Radix tree over token ids** (`ds.07`): `match_prefix` returns the longest cached prefix; locked nodes are never evicted. The `L8.4` prefix cache.
- **Bloom filter** (`ds.08`): $m = -n \ln p / (\ln 2)^2$ bits and $k = (m/n) \ln 2$ hashes for $n$ items at false-positive rate $p$. The `data.03` exact-dedup screen.
- **Consistent hashing with bounded loads** (`ds.09`, Go): virtual nodes on a ring, and no replica takes more than $\lceil c \cdot \text{avg} \rceil$ keys. Gateway affinity routing in `gw.05`.

## Course modules

| Module | Topic | Kind | Pass |
|---|---|---|---|
| `ds.05` | Robin Hood hash map (backward-shift delete); **the first hash-table chapter** (hashing, probing, load factor from first principles) | build | 3 |
| `ds.06` | Binary heap with lazy deletion (generation counters); **the first heap chapter** | build | 3 |
| `ds.08` | Bloom filter | build | 3 |
| `ds.01` | Growable array `tl_vec` (type-erased) | build | 6 |
| `ds.02` | Swiss table `tl_map` (u64 to u64); practice `c/02` linear probing is the worked baseline; builds on the ds.05 chapter (the first hash-table chapter) | build | 6 |
| `ds.03` | Intrusive list + LRU | build | 6 |
| `ds.04` | Binary heap top-k `tl_topk_f32` (ties: lower index wins); builds on the ds.06 chapter (the first heap chapter) | build | 6 |
| `ds.07` | Radix tree over token ids with index-linked LRU leaf list | build | 6 |
| `ds.09` | Consistent hash ring with bounded loads | build | 7 |

The engine flag `--prefix-cache=hash|radix` gives both `ds.02` (hash index in C) and `ds.07` (radix tree in Rust) a production call site; `L10.4` benchmarks one against the other.

## Chapters

| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `ds.01` | [Growable array tl_vec (type-erased, optional C)](01-growable-array.md) | side | 6 |
| 2 | `ds.02` | [Swiss table tl_map (u64 to u64, optional C)](02-swiss-table.md) | side | 6 |
| 3 | `ds.03` | [Intrusive list + LRU (optional C)](03-intrusive-list-and-lru.md) | side | 6 |
| 4 | `ds.04` | [Binary heap top-k in C (optional)](04-binary-heap-top-k.md) | side | 6 |
| 5 | `ds.05` | [Robin Hood hash map with backward-shift deletion](05-robin-hood-hash-map.md) | build | 3 |
| 6 | `ds.06` | [Binary heap with lazy deletion](06-binary-heap-lazy-deletion.md) | build | 3 |
| 7 | `ds.07` | [Radix tree over token ids with index-linked LRU leaf list](07-radix-tree.md) | build | 6 |
| 8 | `ds.08` | [Bloom filter](08-bloom-filter.md) | side | 3 |
| 9 | `ds.09` | [Consistent hash ring with bounded loads](09-consistent-hash-ring.md) | build | 7 |

## Connections to Other Tracks

| Track | Connection |
|---|---|
| [Arrays & Hashing](../01-arrays-hashing/) | the interview-pattern warm-up for the hash tables |
| [Probabilistic Structures](../15-probabilistic-structures/) | `bloom.py` is the worked example for `ds.08` |
| [tinyllm Part 8](../../../courses/build-an-llm/p08-inference/) | the KV block pool (`rt.04`) and the prefix cache (`L8.4`) |
| [Gateway](../../ai-platform-engineering/12-gateway/) | affinity routing over the bounded-load ring |
| [Corpus Pipeline](../../data-engineering/05-corpus-pipeline/) | the Bloom screen in exact dedup |

## Company Relevance

| Company | Practice |
|---|---|
| Google (Abseil) | Swiss tables as the default C++ hash map |
| SGLang, vLLM | radix and hash prefix caches over paged KV blocks |
| Akamai, Vimeo | consistent hashing, and the bounded-load variant in production load balancers |
