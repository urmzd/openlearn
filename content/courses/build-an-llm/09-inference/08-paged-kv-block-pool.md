# A paged KV block pool in C

[The contiguous KV cache](02-kv-cache-and-generate.md) gave every sequence a K and V array of `max_len` positions per layer. That wastes memory twice: a sequence that stops after 40 tokens still holds `max_len` positions, and two requests that start with the same 500-token system prompt each store it. A serving engine runs hundreds of sequences of unknown length, forks sequences for beam search and parallel sampling, and sees the same prompt prefixes over and over. The fix is the one operating systems use for processes: fixed-size pages and a table per sequence.

This optional section studies a block pool in C as a standalone exercise. It goes beyond [the Python paged cache](03-paged-kv-cache.md) in three ways: blocks have three states (free, used, cached) so that prefixes survive their requests; a full block is named by a chained hash so two engines can find each other's blocks; and blocks travel between engines in a checksummed wire format (format v1). The Python paged cache and the Rust Candle engine implement storage independently; shared fixture files compare their serialized behavior. In the kernels part of the book, [paged attention for decode](../10-kernels/06-paged-attention-decode.md) uses this pool in its standalone C exercise.

The pool builds on [C runtime support and the arena allocator](../10-kernels/01-arena-allocator-with-marks.md) for allocation and errors, and on three structures from *Algorithms*: [the growable array `tl_vec`](../../algorithms/13-systems-data-structures/01-growable-array.md) (the free list), [the Swiss table](../../algorithms/13-systems-data-structures/02-swiss-table.md) (the prefix index), and [the intrusive LRU](../../algorithms/13-systems-data-structures/03-intrusive-list-and-lru.md) (cached blocks). FNV-1a comes from [modular arithmetic, hashing, and PCG32](../../mathematics-for-computing/07-discrete-math-2/03-modular-arithmetic-hashing-and-pcg32.md) in *Mathematics for Computing*. The solution is [`kv_pool.c`](../../../../course/solutions/c/src/runtime/kv_pool.c); the contract is [`kv_pool.h`](../../../../course/contracts/c/include/tinyllm/kv_pool.h), with the hash and the bytes in [`formats/kv-block.md`](../../../../course/contracts/formats/kv-block.md) and the ABI rules in [`c/ABI.md`](../../../../course/contracts/c/ABI.md).

> **Learning objectives**
> - Keep every block in exactly one of three states, with `free + used + cached == n_blocks` after every call.
> - Compute the chained FNV-1a block hash by hand and explain why it covers the whole prefix.
> - Implement copy on write, all-or-nothing allocation, and LRU eviction of cached blocks.
> - Write and validate the export envelope, byte for byte, including the CRC-32C trailer.

## Blocks and states

The section's quantities:

| Symbol | Meaning | Type |
|---|---|---|
| $N$ | `n_blocks`, blocks in the pool | `uint32_t` |
| $B$ | `block_tokens`, token positions per block (the engine uses 16) | `uint32_t` |
| $L$, $H$, $D$ | layers, KV heads, head dimension | `uint32_t` |
| $P$ | payload bytes of one block, $L \cdot 2 \cdot H \cdot B \cdot D \cdot 2$ (f16) | `size_t` |
| ref | holders of a block (sequences, a lookup hit) | `uint32_t` |
| fill | positions of the block that hold data, $0..B$ | `uint32_t` |
| $h_i$ | the hash of a sequence's $i$-th full block | `uint64_t` |

The pool allocates one slab of $N \cdot P$ bytes at create and never allocates again (until destroy). A sequence is a list of block ids, so memory is reserved a block at a time and never fragments. Inside block `id`, the K or V values of one layer are a slab `[H][B][D]` of f16, at offset `id * P + (layer * 2 + is_v) * H * B * D * 2`: exactly the order of the export payload, so `tl_kv_block_ptr` is arithmetic and export copies rows.

Every block is in exactly one state:

| State | ref | Registered | Where | Next state |
|---|---|---|---|---|
| free | 0 | no | the free list (a `tl_vec` stack of ids) | used (alloc) |
| used | at least 1 | maybe | held by sequences | free or cached when ref reaches 0 |
| cached | 0 | yes | the prefix index and the LRU list | used (lookup hit) or free (evicted by alloc) |

`tl_kv_alloc(n)` takes $n$ blocks, all or nothing: free ones first; when the free list runs dry it evicts cached blocks oldest first (unregistering each and counting `evictions`); if free + cached $< n$ it takes nothing and returns `TL_EFULL`. `tl_kv_unref` at ref 0 caches a registered block and frees any other. A second unref of a block at ref 0 is `TL_EINVAL` and changes nothing: a refcount below zero would put one block on the free list twice.

## Copy on write

A fork shares every block of its parent: `tl_kv_ref` on each. Before a sequence writes into a block it calls `tl_kv_cow(id, &out)`. With ref 1 it is the only holder and writes in place (`out == id`). With ref above 1 the pool allocates a block, copies the contents and the fill, drops the writer's reference to the original (which the others keep), and returns the copy. Only the partially filled last block of a fork is ever copied; full prefix blocks stay shared forever.

## The chained block hash and the prefix index

Reuse across requests needs a name for a block's contents. A full block's hash is FNV-1a 64 over 8 little-endian bytes of the previous block's hash (0 for block 0) followed by 4 little-endian bytes per token id:

$$h_i = \text{fnv1a64}\big(\text{le}_{64}(h_{i-1}) \,\|\, \text{le}_{32}(t_{iB}) \,\|\, \cdots \,\|\, \text{le}_{32}(t_{iB+B-1})\big), \qquad h_{-1} = 0,$$

with a result of 0 replaced by 1, because 0 means "no hash" (the partial tail). Chaining makes $h_i$ name the whole prefix up to block $i$, not just its $B$ tokens, so equal hashes mean equal prefixes.

`tl_kv_register(id, h)` puts $h \to$ id in the prefix index (a `tl_map`, created for $N$ keys so it never grows); `tl_kv_lookup(h)` returns the id and takes a reference, reviving a cached block. Only full blocks are registered, because a partial block's contents still change. A second block with an already registered hash gets `TL_EBUSY` and stays a private copy. A miss is `TL_ENOTFOUND` and leaves the error slot alone: it is the normal answer for a new prompt.

## The export envelope

Blocks also travel between engines, for example from a prefill engine to a decode engine. `tl_kv_export` writes the blocks of a sequence as:

- a 28-byte header (`TLKV`, version 1, dtype 1 = f16, $n$, $B$, $L$, $H$, $D$, all little-endian);
- per block, its hash (0 if unregistered), its fill as `n_tokens`, and its payload with every position at or past the fill written as zeros;
- the CRC-32C (Castagnoli, reflected polynomial `0x82F63B78`, initial value and final XOR `0xFFFFFFFF`) of everything before it.

Zeroing past the fill makes the envelope a pure function of the block's data, so two implementations write the same bytes, and one flipped bit fails the checksum.

`tl_kv_import` refuses, allocating nothing, any envelope that is too short, has a bad magic, a version other than 1 or a dtype other than f16 (`TL_EFORMAT`), a length that does not match its header, a CRC mismatch, an `n_tokens` outside $1..B$, or a hashed block that is partial (`TL_EFORMAT`), or dimensions that differ from the pool's (`TL_ESHAPE`). It registers each imported full block unless the hash is already held, in which case the caller keeps a private copy.

## Working it by hand

Use the example of `formats/kv-block.md`: $B = 2$, $L = H = 1$, $D = 2$, one sequence with tokens 1 2 3 4 5.

**Hashes.** Block 0 hashes the 16 bytes `00 00 00 00 00 00 00 00 01 00 00 00 02 00 00 00`: $h_0 =$ `0xA91AB0C1027B9366`. Block 1 hashes $h_0$'s 8 bytes, least significant first (`66 93 7b 02 c1 b0 1a a9`), then `03 00 00 00 04 00 00 00`: $h_1 =$ `0x03DF829571605C60`. Token 5 is in a partial block: no hash.

**One block on the wire.** Block 0 holds K = `[[1, 2], [3, 4]]` and V = `[[0.5, -1], [0, 0.25]]` (rows are positions). In f16, 1.0 is `0x3C00` (sign 0, exponent 15 = bias, mantissa 0), stored least significant byte first as `00 3c`. $P = 1 \cdot 2 \cdot 1 \cdot 2 \cdot 2 \cdot 2 = 16$, so the envelope is $28 + 12 + 16 + 4 = 60$ bytes:

```
54 4c 4b 56 01 00 01 00 01 00 00 00 02 00 00 00     magic, version 1, dtype 1 (f16), n_blocks 1, block_tokens 2
01 00 00 00 01 00 00 00 02 00 00 00                 n_layers 1, n_kv_heads 1, head_dim 2
66 93 7b 02 c1 b0 1a a9 02 00 00 00                 block_hash 0xA91AB0C1027B9366, n_tokens 2
00 3c 00 40 00 42 00 44                             K: 1.0 2.0 3.0 4.0
00 38 00 bc 00 00 00 34                             V: 0.5 -1.0 0.0 0.25
1c 54 0b 4f                                         crc32c 0x4F0B541C
```

**States through a fork.** Take a pool of 3 blocks. Alloc 2 for the sequence (free 1, used 2), register block 0 under $h_0$, and fork (both blocks at ref 2). The child writes position 3 into block 1: `tl_kv_cow` copies it (free 0, used 3). The parent finishes: block 0 drops to ref 1, and block 1 (its own now) to 0 and, unregistered, back to free. The child finishes: block 0 reaches ref 0 and, registered, becomes cached (free 2, used 0, cached 1). A new prompt starting with 1 2 looks up $h_0$ and gets block 0 back without recomputing it.

## Designing the API

```c
/* tinyllm/kv_pool.h (rt.04), abridged; the header documents every status */
typedef struct { uint32_t n_blocks, block_tokens, n_layers, n_kv_heads, head_dim;
                 int32_t dtype; uint32_t format; } tl_kv_cfg;        /* TL_F16, format 1 */
typedef struct { uint32_t free, used, cached, evictions; } tl_kv_stats;
tl_status tl_kv_pool_create(const tl_kv_cfg *cfg, tl_kv_pool **out);
tl_status tl_kv_alloc(tl_kv_pool *p, uint32_t n, uint32_t *ids);     /* all or nothing; TL_EFULL */
void      tl_kv_ref(tl_kv_pool *p, uint32_t id);
tl_status tl_kv_unref(tl_kv_pool *p, uint32_t id);                   /* double unref: TL_EINVAL */
tl_status tl_kv_cow(tl_kv_pool *p, uint32_t id, uint32_t *out);
tl_status tl_kv_set_fill(tl_kv_pool *p, uint32_t id, uint32_t n_tokens);
uint64_t  tl_kv_block_hash(uint64_t parent, const uint32_t *toks, uint32_t n);
tl_status tl_kv_register(tl_kv_pool *p, uint32_t id, uint64_t hash); /* full blocks; TL_EBUSY */
tl_status tl_kv_lookup(tl_kv_pool *p, uint64_t hash, uint32_t *id);  /* TL_ENOTFOUND, slot untouched */
void     *tl_kv_block_ptr(const tl_kv_pool *p, uint32_t id, uint32_t layer, int is_v);
tl_status tl_kv_export(const tl_kv_pool *p, const uint32_t *ids, uint32_t n,
                       void *buf, size_t cap, size_t *written);       /* TL_EFULL sets *written */
tl_status tl_kv_import(tl_kv_pool *p, const void *buf, size_t len, uint32_t *ids_out);
uint32_t  tl_crc32c(const void *data, size_t n, uint32_t crc);
```

Two choices the header leaves open are fixed here. Export refuses a block with fill 0 (`TL_EINVAL`), because reader rule 5 would refuse the envelope. And the free list is a `tl_vec` of ids reserved for $N$ at create, popped by reading the last id and shortening `len`, so pushing a freed id never reallocates.

## Building it step by step

The solution comes in three layers, each complete before the next: the pool itself (create, alloc, ref, unref, cow, block_ptr, stats), then hashing, the prefix index, and the LRU, then export and import.

### The reasoning

1. **Memory must not fragment and must not grow at run time.** So the pool takes one slab of $N \cdot P$ bytes at create, and a sequence owns block ids, not memory. Allocation is popping an id; nothing calls `tl_alloc` after create.
2. **Sharing needs a count of holders.** A block reachable from several sequences (a fork, a prefix hit) can only be reclaimed when the last holder lets go: a refcount per block. The three states follow from "ref is zero or not" crossed with "registered or not".
3. **Reuse needs a name for a block's contents that two engines agree on.** The name must cover the whole prefix, otherwise two prompts that differ early would collide later: chain the parent's hash into each block's hash. Only full blocks are named, because a partial block's contents still change.
4. **Cached blocks are reclaimable, so reclaim the least useful first.** Keeping them on an intrusive LRU list makes "touch", "remove", and "pop oldest" $O(1)$ without allocating.
5. **Admission must be all or nothing.** The scheduler asks for the blocks a request needs; taking some and failing would leak them. Check `free + cached >= n` before taking anything.
6. **Writes into shared blocks must not be visible to the other holders**: copy on write at ref above 1, in place at ref 1.
7. **The wire format must be a pure function of the data.** Zero every position past the fill (stale bytes from an earlier owner would differ between engines), checksum everything, and validate every field before allocating, so a bad envelope changes nothing.

### State

Each block's record holds its state and an embedded LRU node; the pool holds the counts that make `free + used + cached == n_blocks` cheap to report:

```c
typedef struct {
    uint32_t ref;        /* holders; 0 for free and cached blocks */
    uint32_t fill;       /* token positions holding data, 0..block_tokens */
    uint64_t hash;       /* registered hash; 0 when not registered */
    tl_list_node lru;    /* linked into pool->lru while cached */
    uint32_t id;
} kv_block;

struct tl_kv_pool {
    tl_kv_cfg cfg;
    size_t elem, row_bytes, slab_bytes, block_bytes;  /* 2; D*2; H*B*row; L*2*slab */
    unsigned char *data;   /* the one slab */
    kv_block *blocks;
    tl_vec free_ids;       /* ids of free blocks (uint32_t); the last is taken first */
    uint32_t n_used, n_cached, evictions;
    tl_map *index;         /* hash -> id of every registered block */
    tl_lru lru;            /* cached blocks, oldest first */
};
```

Create reserves everything up front (the free list for $N$ ids, the index for $N$ keys) and pushes ids in reverse so they come out `0, 1, 2, ...`; any failing allocation frees what was taken and returns `TL_ENOMEM`. Format 2 and dtypes other than f16 are `TL_EUNSUPPORTED`; zero dimensions and a size that overflows are `TL_EINVAL`.

### The chained hash

This is step 3: FNV-1a 64 from the offset basis, over the parent's 8 bytes then each token's 4, all little-endian.

```c
uint64_t tl_kv_block_hash(uint64_t parent, const uint32_t *toks, uint32_t n) {
    // ... NULL check ...
    unsigned char b[8];
    put_u64(b, parent);                                        /* le64(h_{i-1}) */
    uint64_t h = kv_fnv1a64(b, 8, UINT64_C(0xCBF29CE484222325)); /* offset basis, not 0 */
    for (uint32_t i = 0; i < n; i++) {
        put_u32(b, toks[i]);                                   /* 4 bytes per id */
        h = kv_fnv1a64(b, 4, h);                               /* continue the same hash */
    }
    return h == 0 ? 1 : h; /* 0 means "no hash" (the partial tail block) */
}
```

Writing bytes explicitly (`put_u64`, `put_u32`) rather than hashing the integers' memory makes the result independent of the host's byte order, which is what lets the Rust engine compute the same $h_0 =$ `0xA91AB0C1027B9366`.

### Allocation

Steps 4 and 5:

```c
tl_status tl_kv_alloc(tl_kv_pool *p, uint32_t n, uint32_t *ids) {
    // ... NULL checks ...
    if ((uint64_t)p->free_ids.len + p->n_cached < n) { /* all or nothing */
        tl_set_last_error("tl_kv_alloc: not enough free or cached blocks");
        return TL_EFULL;
    }
    for (uint32_t i = 0; i < n; i++) {
        kv_block *b;
        if (p->free_ids.len > 0) { /* pop: read the last id, shorten len */
            b = &p->blocks[((const uint32_t *)p->free_ids.data)[p->free_ids.len - 1]];
            p->free_ids.len--;
        } else { /* reclaim the least recently used cached block */
            tl_list_node *old = tl_lru_pop_oldest(&p->lru);
            b = TL_CONTAINER_OF(old, kv_block, lru);
            uncache(p, b); /* already unlinked; drop it from the index */
            p->evictions++;
        }
        b->ref = 1;
        b->fill = 0;
        p->n_used++;
        ids[i] = b->id;
    }
    return TL_OK;
}
```

Free blocks are taken before cached ones because a cached block may still produce a prefix hit. `uncache` removes the block from the LRU (a no-op after `pop_oldest`), deletes its hash from the index, and clears `hash`, so a reclaimed block can never be found by an old lookup.

### Reference counting and lookup

This is step 2. `used_block` returns NULL for an id out of range or at ref 0, which makes a double unref and a `ref` on a free block errors that change nothing:

```c
tl_status tl_kv_unref(tl_kv_pool *p, uint32_t id) {
    kv_block *b = used_block(p, id, "tl_kv_unref: id out of range or refcount already 0 (double unref)");
    if (b == NULL) return TL_EINVAL;     /* never below zero */
    if (--b->ref > 0) return TL_OK;
    p->n_used--;
    if (b->hash != 0) { /* registered: keep the contents for a later lookup */
        tl_lru_touch(&p->lru, &b->lru);  /* newest end of the LRU */
        p->n_cached++;
    } else {
        b->fill = 0;
        tl_vec_push(&p->free_ids, &id); /* never grows: cap is n_blocks */
    }
    return TL_OK;
}

tl_status tl_kv_lookup(tl_kv_pool *p, uint64_t hash, uint32_t *id) {
    // ... NULL checks ...
    uint64_t v;
    if (hash == 0 || !tl_map_get(p->index, hash, &v)) return TL_ENOTFOUND; /* a miss is not an error */
    kv_block *b = &p->blocks[v];
    if (b->ref == 0) { /* cached: back to used */
        tl_lru_remove(&p->lru, &b->lru);  /* or the next alloc could evict it */
        p->n_cached--;
        p->n_used++;
    }
    b->ref++;
    *id = (uint32_t)v;
    return TL_OK;
}
```

A miss returns before touching the error slot: it is the normal answer for a new prompt, and overwriting the slot would destroy the caller's last real error message.

> **Warning:** The subtlest bug in this pool is a lookup hit that leaves the revived block on the eviction list. The next allocation that runs out of free blocks would pop it and hand it to another request while the first one is reading it. `tl_kv_lookup` calls `tl_lru_remove` for exactly this reason.

### Copy on write

Step 6:

```c
tl_status tl_kv_cow(tl_kv_pool *p, uint32_t id, uint32_t *out) {
    kv_block *b = used_block(p, id, "tl_kv_cow: block is not in use");
    // ... NULL checks ...
    if (b->ref == 1) { /* the only holder may write in place */
        *out = id;
        return TL_OK;
    }
    uint32_t nid;
    tl_status st = tl_kv_alloc(p, 1, &nid);
    if (st != TL_OK) return st;          /* a full pool: nothing changed */
    memcpy(p->data + (size_t)nid * p->block_bytes, p->data + (size_t)id * p->block_bytes, p->block_bytes);
    p->blocks[nid].fill = b->fill;
    b->ref--; /* still >= 1: the other holders keep the original */
    *out = nid;
    return TL_OK;
}
```

Registration refuses a hash of 0, a block that is not full, and a block already registered under another hash; a hash another block holds is `TL_EBUSY` (the first holder stays reachable). `tl_kv_set_fill` refuses a registered block, so a named block is immutable.

The slab layout is arithmetic, `[n_layers][2][n_kv_heads][block_tokens][head_dim]`:

```c
size_t off = (size_t)id * p->block_bytes + ((size_t)layer * 2 + (is_v ? 1 : 0)) * p->slab_bytes;
```

### Export

Step 7. The header is written field by field little-endian, then per block the hash, the fill, and the payload with positions past the fill as zeros, then the CRC-32C of everything before it:

```c
for (uint32_t i = 0; i < n; i++) {
    const kv_block *b = &p->blocks[ids[i]];
    put_u64(o + at, b->hash);           /* 0 if unregistered */
    put_u32(o + at + 8, b->fill);       /* n_tokens */
    at += 12;
    const unsigned char *src = p->data + (size_t)ids[i] * p->block_bytes;
    size_t rows = (size_t)p->cfg.n_layers * 2 * p->cfg.n_kv_heads; /* (layer, K/V, head) */
    for (size_t r = 0; r < rows; r++)
        for (uint32_t t = 0; t < B; t++) {
            const unsigned char *s = src + (r * B + t) * p->row_bytes;
            for (uint32_t d = 0; d < D; d++) {
                uint16_t v;
                memcpy(&v, s + (size_t)d * 2, 2);     /* in memory: host-order f16 bits */
                put_u16(o + at, t < b->fill ? v : 0); /* past the fill: zero bytes */
                at += 2;
            }
        }
}
put_u32(o + at, tl_crc32c(o, at, 0));   /* header and every block */
```

Before writing, export checks every block is in use and has a non-zero fill, and sets `*written` to the size needed before returning `TL_EFULL` for a short buffer, so a caller can size its buffer with one failed call (or `tl_kv_export_bytes`).

The CRC is the bitwise reflected form. `~crc` on entry and exit makes a fresh checksum start at `0xFFFFFFFF` and lets a caller continue from a previous result:

```c
uint32_t tl_crc32c(const void *data, size_t n, uint32_t crc) {
    const unsigned char *p = (const unsigned char *)data;
    uint32_t c = ~crc; /* a fresh checksum starts from 0xFFFFFFFF */
    for (size_t i = 0; i < n; i++) {
        c ^= p[i];
        for (int k = 0; k < 8; k++) c = (c >> 1) ^ (0x82F63B78u & (0u - (c & 1u)));  /* branch-free */
    }
    return ~c;
}
```

### Import

Import validates in the order that keeps every check meaningful: length and magic; version and dtype (`TL_EFORMAT`); total length against the header's $n$ and dimensions (with overflow guards); the CRC; then the pool's shape (`TL_ESHAPE`); then every record's `n_tokens` in $1..B$ and "hashed implies full", all before `tl_kv_alloc`. Only then does it allocate, copy each payload, set the fill, and register a full block whose hash is not already held (otherwise it stays a private copy). If registering fails it unrefs every block it took.

The full solution is [`kv_pool.c`](../../../../course/solutions/c/src/runtime/kv_pool.c); the contract is [`kv_pool.h`](../../../../course/contracts/c/include/tinyllm/kv_pool.h).

### Why this design

The operations are cheap. `alloc`, `ref`, `unref`, `lookup`, and `register` are $O(1)$ amortised per block (stack pop, refcount, one hash-map operation, one list operation). `cow` is one block copy, $O(P)$. Export and import are $O(nP)$ plus one CRC pass; the bitwise CRC is about 8 operations per bit, fine for a reference and replaceable by a table or the SSE4.2 `crc32` instruction without changing a byte.

Consider the alternatives. Contiguous per-sequence buffers (`malloc` per sequence) waste `max_len - len` positions each and cannot share prefixes; per-block `malloc` fragments and makes allocation fallible at run time. One slab plus ids has neither problem. Freeing registered blocks at ref 0 would be simpler, but no prefix would ever be reused across requests; the cached state is what makes a shared system prompt free after the first request. Hashing token blocks independently, without the chain, would let two prompts that differ in block 0 but agree in block 1 share block 1, whose keys were computed under different attention context.

Two tradeoffs are accepted. Eviction is per block, LRU, with no notion of which prefix a block belongs to; [the radix prefix cache](04-radix-prefix-cache.md) adds that structure. And the pool is not thread-safe; the engine serialises access per step.

## What can go wrong

A correct solution must handle every case below.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| The hand-example hash chain | unit, golden | $h_0$, $h_1$; the same tokens under another parent hash differently | the block manager and disaggregated engines find each other's blocks only through these values |
| The hand-example envelope | unit, golden | the 60-byte envelope, nothing written past it | the PushKv payload, compared byte for byte across implementations |
| CRC-32C check value and continuation | unit | `crc32c("123456789") = 0xE3069283`; continuing from a previous result | the envelope trailer and the gRPC chunk checksum |
| Alloc is all or nothing | unit, boundary | asking for more than free + cached takes nothing | admission by free blocks in continuous batching |
| Unref frees or caches | unit | ref 0 caches a registered block, frees others; the three counts add up | prefix reuse without leaking memory |
| A double unref changes nothing | fault | a second unref and a bad id are `TL_EINVAL`; ref on a free block sets the error slot | one block on the free list twice is shared memory nobody knows about |
| Lookup revives, and misses quietly | unit | a hit returns the block used again; a miss is `TL_ENOTFOUND` with the error slot unchanged; the contents are intact | prefix hits for every new prompt |
| Eviction takes the least recently used | unit | a hit moves a cached block to the newest end; two allocations evict the two oldest; `evictions` counts | the cache keeps the prefixes in use |
| A revived block is never evicted | unit | after a lookup hit, the next allocation reclaims another cached block | a running request's prefix stays put |
| CoW copies only when shared | unit | ref 1 writes in place; ref 2 copies contents and fill; the original is unchanged | forks for beam search and parallel sampling |
| CoW in a full pool | boundary | a copy that cannot allocate changes nothing | preemption instead of corruption |
| Only full blocks are registered | unit, boundary | partial, hash 0, and unheld blocks refused; a duplicate hash is `TL_EBUSY`; a registered block is immutable | a partial block's hash would name contents that still change |
| The `block_ptr` layout | unit, boundary | every (block, layer, K/V) slab distinct and exactly $H B D$ f16 long; out of range gives NULL | paged attention and the Python paged cache index slabs `[H][B][D]` |
| Export zeroes past the fill and reports its size | unit, boundary | stale data past the fill exported as zeros; a short buffer gets `TL_EFULL` and the size needed | envelopes compare byte for byte |
| Export, import, export | unit | export, import into another pool, export again: same bytes; full blocks arrive registered, the tail does not | disaggregated prefill and decode |
| Import keeps the first holder of a hash | unit | an imported block whose hash is held stays a private copy | KV-transfer dedup |
| Every bit flip is refused | fault | all 480 single-bit flips of the worked envelope are refused; nothing allocated | a corrupted transfer never decodes |
| Import reader rules | boundary | version 2, a wrong dtype, bad `n_tokens`, a hashed partial block (each with a valid CRC) are `TL_EFORMAT`; another shape is `TL_ESHAPE` | format v2 arrives only through its migration |
| Create rules and allocation failure | boundary, fault | format 2 and other dtypes `TL_EUNSUPPORTED`; zero dimensions `TL_EINVAL`; every failing allocation of create `TL_ENOMEM` with no leak (under a counting allocator) | the engine starts or fails cleanly |
| Random operations keep the invariants | property | 40 seeded runs of alloc, ref, unref, cow, register, lookup, export, and import against a model of the refcounts; held blocks keep their contents | the three promises under any interleaving |

The cases worth being able to state in your own words are the hash chain by hand, alloc all or nothing, unref caching registered blocks, a refused double unref, a lookup hit that is not evicted, CoW copying shared blocks, partial blocks not being registered, and the round trip and a bit flip.

The pitfalls are many, because a byte-exact format and a refcounted pool each have their own ways to fail. This table groups them by symptom.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Hashing a block without its parent | two prompts that differ early share later blocks | the parent's 8 bytes are hashed first |
| Hashing token ids with the wrong width | hashes differ from the Rust engine's | `put_u32` per token: 4 little-endian bytes |
| Starting FNV from 0 instead of the offset basis | every hash differs from the format's | starts from `0xCBF29CE484222325` |
| Header fields out of order | another engine reads a nonsense shape | writes each field at its fixed offset |
| Exporting hash 0 for registered blocks | the receiver cannot deduplicate | writes `b->hash` for every block |
| Checksumming only the payload | header corruption passes | CRC over `o[0..at)`, header included |
| CRC without the initial inversion | the check value is wrong, envelopes disagree | `~crc` on entry and on exit |
| Taking blocks one at a time and failing halfway | blocks leak on every refused admission | checks `free + cached >= n` before taking any |
| Freeing registered blocks at ref 0 | no prefix is ever reused | `unref` caches a block with a non-zero hash |
| Letting a refcount go below zero | a block handed to two sequences | `used_block` refuses ref 0 with `TL_EINVAL` |
| `tl_kv_ref` reviving a block nobody holds | a free block counts as used and leaks | `ref` goes through `used_block` too; only `lookup` revives |
| Forgetting to unlink a revived block from the LRU | the next allocation evicts a block a request is reading | `lookup` calls `tl_lru_remove` on a cached block |
| Setting the error slot on a miss | the caller's earlier error message is lost | a miss returns `TL_ENOTFOUND` without `tl_set_last_error` |
| Evicting the newest cached block | the hot system prompt is recomputed | `tl_lru_pop_oldest`; `unref` touches to the newest end |
| Not counting evictions | `/metrics` shows no pressure | `evictions++` on every reclaim |
| Copy on write keeping the writer's reference | the original never returns to the pool | `b->ref--` after the copy |
| Writing a shared block in place | the parent sees the child's tokens | copies whenever `ref > 1` |
| Registering a partial block | a prefix hit returns half-written KV | `register` requires `fill == block_tokens` |
| A duplicate hash replacing the first holder | the first block stays cached but unreachable | returns `TL_EBUSY` when the index already holds the hash |
| K and V sharing one slab | every V overwrites a K | offset `(layer * 2 + is_v) * slab_bytes` |
| Exporting stale data past the fill | two engines write different bytes for the same block | writes 0 for `t >= fill` |
| Not reporting the size needed | the caller cannot size its buffer | sets `*written = need` before `TL_EFULL` |
| Import dropping the fill | the decode engine sees empty blocks | sets `fill` from each record |
| Import never registering | transferred prefixes are never hit again | registers each full block with a hash |
| Import fighting the first holder of a hash | a valid envelope fails with `TL_EBUSY` | skips registration when the hash is held |
| Skipping the CRC | corrupted KV decodes | compares the trailer before allocating |
| Reading any version | a v2 envelope's fp8 bytes read as f16 | accepts version 1 with dtype f16 only |
| Ignoring the receiving pool's shape | a payload laid out for other dimensions is copied in | `TL_ESHAPE` when any dimension differs |
| Accepting format 2 before its migration | a pool that cannot read what it was asked to hold | create returns `TL_EUNSUPPORTED` |
| Leaking on a failed create | live allocations after a failed create | frees every partial allocation before `TL_ENOMEM` |
| Envelope version field 0 | every reader refuses it | writes `TL_KV_FORMAT_V1` |
| Losing a freed id | the pool shrinks by one block per request | `unref` pushes every freed id; the free list never reallocates |

## Where this leads

[The Python paged cache](03-paged-kv-cache.md) implements its own allocation and block operations; fixture files can compare its behavior with this pool's. [Paged attention for decode](../10-kernels/06-paged-attention-decode.md) reads K and V slabs through `tl_kv_block_ptr`. In the serving engine, [the block manager](../11-serving/05-block-manager-and-prefix-cache.md) with `--prefix-cache=hash` uses register and lookup, and [disaggregated prefill and decode](../11-serving/07-disaggregated-prefill-decode.md) moves blocks between engines as export envelopes. The fp8 format v2 arrives through [the KV format v1 to v2 migration](../../software-craftsmanship/07-maintenance/01-kv-format-v1-to-v2.md) in *Software Craftsmanship*.

Production systems build the same pieces at scale:

- vLLM's `BlockPool` and `KVCacheManager` ([`vllm/v1/core/block_pool.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py)) are a block pool with refcounts and CoW, adding block hashes with extra keys (LoRA, multimodal) and per-request block tables on the GPU.
- vLLM's `hash_block_tokens` ([`vllm/v1/core/kv_cache_utils.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py)) is the same parent-chained hash, in Python over tuples.
- NVIDIA Dynamo and [LMCache](https://github.com/LMCache/LMCache) KV transfer play the role of the export envelope, with NIXL/RDMA transfer of KV blocks between GPUs and nodes.
- llama.cpp's KV cache (`src/llama-kv-cache.cpp`) goes beyond f16 blocks with quantized KV cache types (`q8_0`, `q4_0`) selected per run.

### Further reading

- Kwon et al., [*Efficient Memory Management for LLM Serving with PagedAttention*](https://arxiv.org/abs/2309.06180) (SOSP 2023), sections 4.1 to 4.4.

## Key terms

- **Block pool**: one slab of fixed-size KV blocks handed out by id.
- **Free, used, cached**: the three block states; cached blocks have ref 0 but stay registered for reuse.
- **Chained block hash**: FNV-1a over the parent block's hash and the block's token ids, naming the whole prefix.
- **Prefix index**: the map from block hash to block id.
- **Copy on write**: copying a shared block before a holder writes into it.
- **Export envelope**: the format v1 wire encoding of a sequence's blocks, with a CRC-32C trailer.
- **CRC-32C**: the Castagnoli cyclic redundancy check, reflected polynomial `0x82F63B78`.

## Summary

- The KV cache becomes fixed-size blocks in one slab; a sequence is a list of block ids, so memory is reserved a block at a time and never fragments.
- A block is in exactly one of three states, free, used, or cached, and `free + used + cached == n_blocks` after every call.
- A full block is named by a chained FNV-1a hash of its parent's hash and its token ids, so equal hashes mean equal prefixes; only full blocks are hashed, registered, or deduplicated.
- Shared blocks are copied only when someone writes: refcount 1 writes in place, refcount above 1 copies.
- The export envelope is a pure function of the blocks' contents, checksummed with CRC-32C, so one flipped bit is refused and two implementations produce the same bytes.

## Review questions

1. Why does `tl_kv_alloc` check `free + cached >= n` before taking any block?
   <details><summary>Answer</summary>Taking blocks one at a time and failing halfway would leave some blocks marked used with nobody holding them. The upfront check makes the call all or nothing, so a refused admission changes nothing.</details>
2. What goes wrong if `tl_kv_lookup` revives a cached block without unlinking it from the LRU?
   <details><summary>Answer</summary>It is still at its place on the eviction list, so the next allocation that runs out of free blocks can pop it, unregister it, and hand it to another request while the request that looked it up is reading it.</details>
3. Why is the tail block's hash 0, and why is a computed hash of 0 replaced by 1?
   <details><summary>Answer</summary>A partial block's contents still grow, so it must never be found by a lookup; 0 is reserved for "not named". A real hash that happened to be 0 would then be indistinguishable from "no hash", so it is mapped to 1.</details>
4. Why must export zero the positions past the fill?
   <details><summary>Answer</summary>Those bytes are whatever the block's previous owner left there. Two engines exporting the same logical block would write different bytes and different CRCs, so envelopes could not be compared byte for byte or deduplicated.</details>
5. Why does import check the CRC before the shape?
   <details><summary>Answer</summary>A corrupted envelope should be reported as corrupted (`TL_EFORMAT`), not as a mismatch with the pool. Checking the CRC first ensures `TL_ESHAPE` means "a valid envelope for other dimensions", which is a configuration error the caller can act on.</details>
