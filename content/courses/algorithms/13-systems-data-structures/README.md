# Systems Data Structures

A data structure in a production system is chosen by its worst case under the system's access pattern, not by its textbook complexity. The KV block pool needs a hash index that never stalls on a rehash and an LRU that evicts in $O(1)$; the tokenizer needs a hash map tuned for small integer keys and a priority queue it can delete from; the router needs a hash that moves few keys when a replica joins. This chapter builds each of those structures from its call site, deriving what it must do before showing how it does it.

Every structure here has a caller in the system the books build. The hash maps index KV blocks and tokenizer merges, the heaps pick top-$k$ tokens and the next request, the radix tree finds shared prompt prefixes, the Bloom filter screens duplicate paragraphs, and the ring routes requests to replicas. The same ideas appear in three languages: C for the runtime (`tl_vec`, `tl_map`, the LRU, `tl_topk_f32`), Rust for the tokenizer and engine, and Go for the gateway. The interface is the contract; the language is a detail.

Two themes run through the chapter. First, open addressing is a probe-length problem: linear probing, Robin Hood, and Swiss tables differ in how they bound the worst probe, not the average one, and the load factor decides when that bound breaks. Second, probabilistic structures trade certainty for space: a Bloom filter never misses a member and is wrong about non-members at a rate you choose, and a consistent hash ring trades exact balance for minimal movement, then wins balance back with a load cap.

> **Learning objectives**
> - Build open-addressing hash tables (Robin Hood with backward-shift deletion, and a Swiss table with SWAR control-byte matching) and explain how each bounds probe length at load 7/8.
> - Build binary heaps for a stable priority queue with $O(1)$ lazy deletion and for allocation-free top-$k$ selection.
> - Build allocation-free and failure-atomic containers in C: a growable array and an intrusive LRU list.
> - Build a radix tree over token ids with an index-linked LRU of unlocked leaves for prefix caching.
> - Size a Bloom filter from $n$ and $p$, and route keys with a consistent hash ring under bounded loads.

## Before you begin

You should be comfortable with the language primers in *Programming Languages*: [C](../../programming-languages/02-language-and-tool-primers/03-c.md), [Rust](../../programming-languages/02-language-and-tool-primers/04-rust.md), and [Go](../../programming-languages/02-language-and-tool-primers/06-go.md). Load factors and hashing probability come from [Discrete Math 2](../../mathematics-for-computing/07-discrete-math-2/) in *Mathematics for Computing*. The interview-pattern chapters of this book, [Arrays and Hashing](../01-arrays-hashing/), [Trees](../05-trees/), and [Probabilistic Structures](../12-probabilistic-structures/), are good warm-ups; `bloom.py` in the last of these is the worked example behind the Bloom filter section. The [practice C drills](../../../../practice/build/systems/c/) 01 to 04 build linear probing and an intrusive list as baselines.

Try to implement each structure from its section before reading the reference design: Abseil's notes for Swiss tables, SGLang's paper for the radix tree.

## In this chapter

The sections are numbered by where they sit in the C runtime, but two of them are foundations for the rest. [A Robin Hood hash map](05-robin-hood-hash-map.md) is the chapter's first hash-table section, developing hashing, probing, and the load factor from first principles, and [A binary heap with lazy deletion](06-binary-heap-lazy-deletion.md) is its first heap section. If these ideas are new to you, read those two first; the Swiss table and C top-$k$ build on them.

1. [A growable array](01-growable-array.md). The type-erased C array `tl_vec`, with geometric growth for $O(1)$ amortized appends and a failure-atomic growth order; it holds each sequence's block table in the KV pool.
2. [A Swiss table](02-swiss-table.md). The C map `tl_map` from u64 to u64: 7-bit fingerprints in control bytes matched 8 at a time with SWAR, load 7/8, and tombstones only where a probe may have passed. It is the KV pool's prefix-hash index.
3. [An intrusive list and an LRU](03-intrusive-list-and-lru.md). Links embedded in each element and a sentinel head give $O(1)$ touch and evict with no allocation, for the pool's evictable cached blocks.
4. [Heap top-k in C](04-binary-heap-top-k.md). An optional standalone C unit: a size-$k$ min-heap over logits in the caller's buffers, with ties to the lower index and parity with the Python sampler through shared fixtures.
5. [A Robin Hood hash map](05-robin-hood-hash-map.md). The Rust map behind the tokenizer's vocabulary and merge table: Fx hashing from the top bits, Robin Hood insertion, early-exit lookup, and backward-shift deletion.
6. [A binary heap with lazy deletion](06-binary-heap-lazy-deletion.md). A stable Rust heap with $O(n)$ heapify, plus handles and generation counters for $O(1)$ removal; the BPE merge queue and the engine's waiting queue.
7. [A radix tree over token ids](07-radix-tree.md). A compressed trie in granules of one KV block, with `match_prefix` returning the longest cached prefix, locks that pin running requests, and an index-linked LRU of leaves. It is the radix prefix cache.
8. [A Bloom filter](08-bloom-filter.md). $m = -n \ln p / (\ln 2)^2$ bits and $k = (m/n) \ln 2$ hashes, double hashing from FNV-1a and SplitMix64, and a portable byte format shared with the corpus pipeline's exact-dedup screen.
9. [A consistent hash ring with bounded loads](09-consistent-hash-ring.md). A Go ring with virtual nodes where no replica takes more than $\lceil c \cdot \text{avg} \rceil$ keys; it drives the gateway's affinity routing.

The engine flag `--prefix-cache=hash|radix` gives both the Swiss table (hash index in C) and the radix tree (in Rust) a production call site, and [Block manager with prefix cache](../../build-an-llm/11-serving/05-block-manager-and-prefix-cache.md) in *Build Your Own LLM System* benchmarks one against the other.

## Connections to other books

These structures are used throughout the other books:

- The KV block pool and the prefix cache in [Inference](../../build-an-llm/09-inference/) in *Build Your Own LLM System* use the growable array, the Swiss table, the LRU, and the radix tree.
- The [Gateway](../../ai-platform-engineering/12-gateway/) chapter of *AI Platform Engineering* uses the bounded-load ring for affinity routing.
- The [Corpus Pipeline](../../data-engineering/05-corpus-pipeline/) chapter of *Data Engineering* uses the Bloom screen in exact dedup.

The same designs run in production: Google's Abseil makes Swiss tables the default C++ hash map; SGLang and vLLM keep radix and hash prefix caches over paged KV blocks; and Akamai and Vimeo use consistent hashing, including the bounded-load variant, in production load balancers.

## Further reading

- Cormen, Leiserson, Rivest, and Stein, *Introduction to Algorithms* (4th ed.), chapters 6 (heaps) and 11 (hash tables).
- Abseil, [Swiss Tables design notes](https://abseil.io/about/design/swisstables) (free).
- Celis, *Robin Hood Hashing* (1986).
- Bloom, [*Space/Time Trade-offs in Hash Coding with Allowable Errors*](https://dl.acm.org/doi/10.1145/362686.362692) (1970).
- Mirrokni, Thorup, and Zadimoghaddam, [*Consistent Hashing with Bounded Loads*](https://arxiv.org/abs/1608.01350) (free).
- Zheng et al., [*SGLang: Efficient Execution of Structured Language Model Programs*](https://arxiv.org/abs/2312.07104) (RadixAttention, free).
