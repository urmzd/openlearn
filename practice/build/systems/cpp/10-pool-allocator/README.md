# Memory pool allocator

A `std::list` or `std::map` allocates one node per element, every node the same size, and frees them one at a time. A general-purpose allocator handles that workload badly: it searches for a fit it does not need to search for, records a size it already knows, and scatters the nodes across the heap. A pool allocator is built for exactly this shape. It hands out fixed-size blocks from a free list, in O(1), with zero bytes of overhead per block. This section builds one, plus the `std::allocator`-shaped adaptor that lets `std::list` and `std::map` use it.

Two lessons from earlier sections meet here. The free list is intrusive, like the C [intrusive linked list](../../c/03-intrusive-list/README.md), but applied to memory that is not currently an object. And alignment is handled with the same power-of-two rounding as the C [arena allocator](../../c/08-arena-allocator/README.md). What is new is the standard-library allocator model: an allocator is a handle, not an owner, containers rebind it to types you never see, and it must fall back gracefully for requests it cannot serve. The solution is the header [`pool.hpp`](pool.hpp); [`main.cpp`](main.cpp) is the test suite.

> **Learning objectives**
> - Implement a fixed-size block allocator with an intrusive free list and O(1) allocate and free.
> - Round block size and alignment so every block in a chunk is aligned and can hold a pointer.
> - Write a standard-library allocator adaptor that is a cheap, comparable handle and supports rebinding.
> - Route requests the pool cannot serve to the global allocator, and free each pointer back where it came from.

## The interface

| Member | Does |
|--------|------|
| `Pool(block_size, block_align, blocks_per_chunk)` | Configure; allocates nothing yet |
| `Pool::allocate()` | Pop a block off the free list, growing by a chunk if empty |
| `Pool::deallocate(p)` | Push it back |
| `Pool::release()` | Return every chunk to the system |
| `PoolAllocator<T>` | The standard-library adaptor over a `Pool*` |

The constraints: blocks never overlap, are reused after free, honour the requested alignment, and are at least pointer-sized. The adaptor must work with `std::list` and `std::map`, and fall back to the global allocator for requests the pool cannot serve.

To build and run the tests from this directory:

```bash
c++ -std=c++20 -Wall -Wextra -O2 main.cpp -o pool && ./pool
# ok  cpp/10-pool-allocator  1060 checks passed
```

## One size collapses the problem

Handling one size collapses the entire problem. A general allocator must search for a fit, split blocks, coalesce neighbours on free, and store a header per block so `free` knows the size. A pool does none of that: every block is interchangeable, so allocation is "pop the head" and deallocation is "push the head". Two pointer writes, no search, no metadata.

The free list is intrusive, and that is why the overhead is zero. The "next" pointer lives *inside* the free block's own storage, because a free block's contents are by definition not being used for anything else. That is also why blocks are at least pointer-sized: ask for a pool of 1-byte blocks and the free list has nowhere to store its link. The constructor rounds up, and the test asserts it.

Building the free list backwards is a deliberate small choice. Threading a new chunk from the last block to the first leaves the list in ascending address order, so consuming blocks in sequence walks memory forwards.

## Allocators in the standard library

An allocator is a handle, not an owner. Containers copy their allocator freely, rebind it to their internal node type, and compare copies for equality, so an allocator must be cheap to copy and two copies must be interchangeable. `test_allocator_is_a_handle` frees through a different copy than it allocated from.

Rebinding is why a `std::list<int>` allocator is never asked for `int`s. The container needs nodes, which hold your value *plus* two pointers, so it rebinds `PoolAllocator<int>` to `PoolAllocator<ListNode<int>>` via the templated converting constructor. You cannot know the node size portably, which is why the tests size the pool generously and rely on the fallback.

Falling back is not an admission of defeat. `std::vector` asks for one allocation of $n$ elements, which a fixed-size pool fundamentally cannot serve. Returning a too-small block would be memory corruption; forwarding to `::operator new` is correct and keeps the allocator usable with any container.

## Deriving the design

Six observations build the allocator.

Fixed-size blocks make every free block interchangeable. So the set of free blocks needs no order and no sizes: a stack (LIFO list) is enough, and push and pop are O(1).

A free block's memory is unused, so the list can live in it. The first `sizeof(void*)` bytes of each free block hold the pointer to the next free block. There is no side structure and no per-block header.

Therefore every block must hold a pointer and be suitably aligned. Block size is rounded up to at least `sizeof(void*)` and to a multiple of the alignment, and the alignment to at least `alignof(void*)`. Rounding the size to a multiple of the alignment keeps every block in a chunk aligned, not just the first.

Get memory from the system in chunks. When the free list is empty, one aligned `operator new` provides `blocks_per_chunk` blocks, which are threaded onto the list. The chunk pointers are remembered so `release()` can free them.

The container adaptor holds a pointer to the pool. Copies, rebinds, and comparisons then all refer to the same pool, which is what the standard allocator requirements need.

Serve only what fits. A request for one object no bigger than a block comes from the pool; anything else goes to the global allocator, and deallocation makes the same decision from the same inputs.

## Building it step by step

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

The full solution is in [`pool.hpp`](pool.hpp).

## Why this design

`allocate` and `deallocate` are O(1), a few instructions, plus one system allocation per `blocks_per_chunk` blocks. There are zero bytes of overhead per block beyond rounding.

Against `malloc` per node, the pool skips work that is wasted for identical nodes: a general allocator has to find a block of the right size and record its size, and the nodes end up scattered. Against an [arena](../../c/08-arena-allocator/README.md), the choice turns on lifetimes: an arena serves any size but frees only all at once, while a pool frees individually but serves only one size.

## What can go wrong

The most common first mistake is putting the `Pool` *inside* `PoolAllocator`. Each copy then has its own free list, and a container that frees through a different copy than it allocated from corrupts it.

| Mistake | Consequence |
|---------|-------------|
| Embedding the `Pool` in the allocator | Copies have separate free lists; freeing through another copy corrupts them |
| Blocks smaller than a pointer | Nowhere to store the free-list link |
| Not rounding block size to the alignment | Only the first block in each chunk is aligned |
| Serving an $n$-element request from the pool | A too-small block and memory corruption |
| `deallocate` using a different test than `allocate` | Pointers are returned to the wrong allocator |
| Using blocks after `release()` | Every block pointer dangles |

> **Warning:** The adaptor does not check `alignof(T)` against the pool's block alignment. A type stricter than the pool's alignment would get an under-aligned block, so the pool must be configured for the strictest type it will serve. The pool is also not thread-safe, never returns chunks to the system before `release()`, and has no use-after-free detection; these are accepted trades.

## Where this leads

Add a debug mode that poisons freed blocks with a recognisable byte pattern and checks it on allocation, which turns use-after-free from silently plausible data into an immediate assertion. Then try making the pool thread-safe with a lock-free free list using `compare_exchange`, and meet the ABA problem, which the lock-free stack brief in the [Rust](../../rust/README.md) chapter covers from the other direction. The [Vector](../01-vector/README.md) section is a good place to plug a custom allocator in once you add an `Allocator` parameter.

## Key terms

- **Pool allocator**: an allocator that serves fixed-size blocks from a free list.
- **Intrusive free list**: a list of free blocks whose links are stored inside the free blocks themselves.
- **Chunk**: a large allocation from the system, carved into many blocks.
- **Rebinding**: converting an allocator for `T` into one for another type, such as a container's internal node.
- **Allocator equality**: two allocators compare equal when memory from one can be freed through the other.

## Summary

- Every block is the same size, so the free list is a LIFO stack and allocate and free are a pop and a push.
- The next pointer lives in the free block, so blocks are at least pointer-sized and there is no per-block header.
- Block size is rounded to a multiple of an alignment of at least `alignof(void*)`, keeping every block aligned.
- `PoolAllocator` holds a `Pool*`, supports rebinding, and compares equal when two copies share a pool.
- Requests that are not a single object that fits go to the global allocator, and `deallocate` repeats the same test.

## Review questions

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
