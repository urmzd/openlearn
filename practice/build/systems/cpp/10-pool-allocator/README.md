# C++ 10: Pool allocator

**Concepts:** custom allocators, intrusive free lists, alignment
**Difficulty:** ⭐⭐⭐⭐

A fixed-size-block allocator, plus the `std::allocator`-shaped adaptor that
lets `std::list` and `std::map` use it. The solution is the header
[`pool.hpp`](pool.hpp); [`main.cpp`](main.cpp) is the test suite.

## The problem

| Member | Does |
|--------|------|
| `Pool(block_size, block_align, blocks_per_chunk)` | Configure; allocates nothing yet |
| `Pool::allocate()` | Pop a block off the free list, growing by a chunk if empty |
| `Pool::deallocate(p)` | Push it back |
| `Pool::release()` | Return every chunk to the system |
| `PoolAllocator<T>` | The standard-library adaptor over a `Pool*` |

Constraints: blocks never overlap, are reused after free, honour the requested
alignment, and are at least pointer-sized; the adaptor must work with
`std::list` and `std::map`, and fall back to the global allocator for requests
the pool cannot serve.

## Build and run

From this directory:

```bash
c++ -std=c++20 -Wall -Wextra -O2 main.cpp -o pool && ./pool
# ok  cpp/10-pool-allocator  1060 checks passed
```

## What to notice

**Handling one size collapses the entire problem.** A general allocator must
search for a fit, split blocks, coalesce neighbours on free, and store a header
per block so `free` knows the size. A pool does none of that: every block is
interchangeable, so allocation is "pop the head" and deallocation is "push the
head". Two pointer writes, no search, no metadata.

**The free list is intrusive, and that is why the overhead is zero.** The
"next" pointer lives *inside* the free block's own storage, because a free
block's contents are by definition not being used for anything else. This is
the same idea as the [C intrusive list](../../c/03-intrusive-list/), applied to
memory that is not currently an object.

**Which is why blocks are at least pointer-sized.** Ask for a pool of 1-byte
blocks and the free list has nowhere to store its link. The constructor rounds
up, and the test asserts it.

**An allocator is a handle, not an owner.** Putting the `Pool` *inside*
`PoolAllocator` is the mistake almost everyone makes first. Containers copy
their allocator freely, rebind it to their internal node type, and compare
copies for equality, so an allocator must be cheap to copy and two copies must
be interchangeable. `test_allocator_is_a_handle` frees through a different copy
than it allocated from.

**Rebinding is why a `std::list<int>` allocator is never asked for `int`s.**
The container needs nodes, which hold your value *plus* two pointers, so it
rebinds `PoolAllocator<int>` to `PoolAllocator<ListNode<int>>` via the
templated converting constructor. You cannot know the node size portably, which
is why the tests size the pool generously and rely on the fallback.

**Falling back is not an admission of defeat.** `std::vector` asks for one
allocation of *n* elements, which a fixed-size pool fundamentally cannot serve.
Returning a too-small block would be memory corruption; forwarding to
`::operator new` is correct and keeps the allocator usable with any container.

**Building the free list backwards is a deliberate small thing.** Threading a
new chunk from the last block to the first leaves the list in ascending address
order, so consuming blocks in sequence walks memory forwards.

## The solution, step by step

### From first principles

1. **Fixed-size blocks make every free block interchangeable.** So the set of
   free blocks needs no order and no sizes: a stack (LIFO list) is enough, and
   push and pop are O(1).
2. **A free block's memory is unused, so the list can live in it.** The first
   `sizeof(void*)` bytes of each free block hold the pointer to the next free
   block. No side structure, no per-block header.
3. **Therefore every block must hold a pointer and be suitably aligned.** Block
   size is rounded up to at least `sizeof(void*)` and to a multiple of the
   alignment, and the alignment to at least `alignof(void*)`. Rounding the size
   to a multiple of the alignment keeps every block in a chunk aligned, not just
   the first.
4. **Get memory from the system in chunks.** When the free list is empty, one
   aligned `operator new` provides `blocks_per_chunk` blocks, which are threaded
   onto the list. The chunk pointers are remembered so `release()` can free them.
5. **The container adaptor holds a pointer to the pool.** Copies, rebinds, and
   comparisons then all refer to the same pool, which is what the standard
   allocator requirements need.
6. **Serve only what fits.** A request for one object no bigger than a block
   comes from the pool; anything else goes to the global allocator, and
   deallocation makes the same decision from the same inputs.

### The code, annotated

Construction rounds the block geometry:

```cpp
Pool(std::size_t block_size, std::size_t block_align, std::size_t blocks_per_chunk = 64)
    // At least a pointer, rounded to a multiple of the (at least pointer)
    // alignment, so every block in a chunk stays aligned.
    : block_size_(round_up(block_size < sizeof(void*) ? sizeof(void*) : block_size,
                           block_align < alignof(void*) ? alignof(void*) : block_align)),
      block_align_(block_align < alignof(void*) ? alignof(void*) : block_align),
      blocks_per_chunk_(blocks_per_chunk ? blocks_per_chunk : 1) {}

static std::size_t round_up(std::size_t n, std::size_t align) {
  return (n + align - 1) & ~(align - 1);   // align must be a power of two
}
```

Allocation and deallocation are a stack pop and push on the intrusive list:

```cpp
void* allocate() {
  if (!free_list_) grow();

  // The "next" pointer is stored in the free block itself.
  void* block = free_list_;
  free_list_ = *reinterpret_cast<void**>(block);
  ++live_;
  return block;
}

void deallocate(void* p) noexcept {
  if (!p) return;
  // Overwrite the first bytes of the freed block with the old head.
  *reinterpret_cast<void**>(p) = free_list_;
  free_list_ = p;
  --live_;
}
```

Growing by a chunk:

```cpp
void grow() {
  std::size_t bytes = block_size_ * blocks_per_chunk_;
  void* chunk = ::operator new(bytes, std::align_val_t{block_align_});
  chunks_.push_back(chunk);     // remembered for release()

  // Thread backwards so the list ends up in ascending address order.
  auto* base = static_cast<std::byte*>(chunk);
  for (std::size_t i = blocks_per_chunk_; i-- > 0;) {
    void* block = base + i * block_size_;
    *reinterpret_cast<void**>(block) = free_list_;
    free_list_ = block;
  }
}

void release() noexcept {
  for (void* chunk : chunks_) {
    ::operator delete(chunk, std::align_val_t{block_align_});  // matching aligned delete
  }
  chunks_.clear();
  free_list_ = nullptr;   // every block pointer just became dangling
  live_ = 0;
}
```

The adaptor:

```cpp
template <typename T>
class PoolAllocator {
 public:
  using value_type = T;

  explicit PoolAllocator(Pool* pool) noexcept : pool_(pool) {}

  // Rebinding: a PoolAllocator<int> converts to PoolAllocator<Node>.
  template <typename U>
  PoolAllocator(const PoolAllocator<U>& other) noexcept : pool_(other.pool()) {}

  T* allocate(std::size_t n) {
    // One object that fits: pool. Anything else: global aligned new.
    if (n != 1 || sizeof(T) > pool_->block_size()) {
      return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{alignof(T)}));
    }
    return static_cast<T*>(pool_->allocate());
  }

  void deallocate(T* p, std::size_t n) noexcept {
    // Same test as allocate, so each pointer goes back where it came from.
    if (n != 1 || sizeof(T) > pool_->block_size()) {
      ::operator delete(static_cast<void*>(p), std::align_val_t{alignof(T)});
      return;
    }
    pool_->deallocate(p);
  }

  // Interchangeable iff they share a pool.
  template <typename U>
  bool operator==(const PoolAllocator<U>& o) const noexcept { return pool_ == o.pool(); }
  // ...
 private:
  Pool* pool_;
};
```

Full solution: [`pool.hpp`](pool.hpp).

### Why this solution

- **Cost.** `allocate` and `deallocate` are O(1), a few instructions, plus one
  system allocation per `blocks_per_chunk` blocks. Zero bytes of overhead per
  block beyond rounding.
- **Versus `malloc` per node.** A general allocator has to find a block of the
  right size and record its size; for millions of identical nodes that work is
  wasted, and the nodes end up scattered.
- **Versus an arena** ([C 08](../../c/08-arena-allocator/)). An arena serves any
  size but frees only all at once; a pool frees individually but only one size.
- **Accepted trade.** The pool is not thread-safe, never returns chunks to the
  system before `release()`, and has no use-after-free detection. The adaptor
  also does not check `alignof(T)` against the pool's block alignment: a type
  stricter than the pool's alignment would get an under-aligned block, so the
  pool must be configured for the strictest type it will serve.

### Explain it back

1. Where is the free list stored, and why does that require blocks to be at
   least pointer-sized?

   <details><summary>Answer</summary>

   In the free blocks themselves: the first bytes of each free block hold the
   pointer to the next. A block smaller than a pointer could not hold that link.

   </details>

2. Why is the block size rounded up to a multiple of the alignment?

   <details><summary>Answer</summary>

   Blocks are laid out back to back from an aligned chunk start, at offsets
   `i * block_size`. Only if `block_size` is a multiple of the alignment is every
   such offset, and so every block, aligned.

   </details>

3. Why does `PoolAllocator` hold a `Pool*` rather than a `Pool`?

   <details><summary>Answer</summary>

   Containers copy and rebind their allocator and may free memory through a
   different copy than the one that allocated it. A pointer makes every copy
   refer to the same pool, so any copy can free any block. An embedded pool would
   give each copy its own free list, and freeing through another copy would
   corrupt it.

   </details>

4. Why must `deallocate` repeat the exact test that `allocate` used?

   <details><summary>Answer</summary>

   It is the only record of where a pointer came from: there is no per-block
   header. Since the test depends only on `n` and `sizeof(T)`, which the
   container passes back unchanged, it reaches the same answer and returns the
   memory to the right allocator.

   </details>

## Extending it

Add a debug mode that poisons freed blocks with a recognisable byte pattern and
checks it on allocation, which turns use-after-free from silently plausible
data into an immediate assertion. Then try making the pool thread-safe with a
lock-free free list using `compare_exchange` and meet the ABA problem, which the
[Rust lock-free stack exercise](../../rust/) covers from the other direction.
