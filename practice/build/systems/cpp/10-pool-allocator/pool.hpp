// pool.hpp - a fixed-size-block pool allocator, plus the std::allocator
// adaptor that lets standard containers use it.
//
// You write the marked regions. main.cpp tests them.
//
// A general allocator has to handle any size, which means searching, splitting,
// coalescing, and per-block headers. A pool handles exactly one size, which
// collapses all of that into two pointer writes: allocation pops the free list,
// deallocation pushes it back. That is why pools sit under node-based
// containers, particle systems, and network packet buffers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

class Pool {
 public:
  // block_size is rounded up so every block can hold a free-list pointer and
  // satisfy the requested alignment.
  Pool(std::size_t block_size, std::size_t block_align, std::size_t blocks_per_chunk = 64)
      : block_size_(round_up(block_size < sizeof(void*) ? sizeof(void*) : block_size,
                             block_align < alignof(void*) ? alignof(void*) : block_align)),
        block_align_(block_align < alignof(void*) ? alignof(void*) : block_align),
        blocks_per_chunk_(blocks_per_chunk ? blocks_per_chunk : 1) {}

  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;

  ~Pool() { release(); }

  void* allocate() {
    if (!free_list_) grow();

    // Pop the head. The free list is INTRUSIVE: the "next" pointer lives in
    // the free block's own storage, so the bookkeeping costs zero extra
    // memory. A block is either in use or holding a pointer, never both.
    void* block = free_list_;
    free_list_ = *reinterpret_cast<void**>(block);
    ++live_;
    return block;
  }

  void deallocate(void* p) noexcept {
    if (!p) return;
    // Push onto the head. No searching, no coalescing, no size lookup:
    // every block is interchangeable, which is the whole point.
    *reinterpret_cast<void**>(p) = free_list_;
    free_list_ = p;
    --live_;
  }

  // Return every chunk to the system. Any outstanding pointer dangles
  // afterwards, so this is only safe once nothing is live.
  void release() noexcept {
    for (void* chunk : chunks_) {
      ::operator delete(chunk, std::align_val_t{block_align_});
    }
    chunks_.clear();
    free_list_ = nullptr;
    live_ = 0;
  }

  std::size_t block_size() const noexcept { return block_size_; }
  std::size_t live_blocks() const noexcept { return live_; }
  std::size_t chunk_count() const noexcept { return chunks_.size(); }
  std::size_t capacity() const noexcept { return chunks_.size() * blocks_per_chunk_; }

 private:
  static std::size_t round_up(std::size_t n, std::size_t align) {
    return (n + align - 1) & ~(align - 1);
  }

  void grow() {
    std::size_t bytes = block_size_ * blocks_per_chunk_;
    void* chunk = ::operator new(bytes, std::align_val_t{block_align_});
    chunks_.push_back(chunk);

    // Thread every block in the new chunk onto the free list. Building the
    // links backwards leaves the list in address order, which is friendlier
    // to the prefetcher when blocks are consumed in sequence.
    auto* base = static_cast<std::byte*>(chunk);
    for (std::size_t i = blocks_per_chunk_; i-- > 0;) {
      void* block = base + i * block_size_;
      *reinterpret_cast<void**>(block) = free_list_;
      free_list_ = block;
    }
  }

  void* free_list_ = nullptr;
  std::vector<void*> chunks_;
  std::size_t block_size_;
  std::size_t block_align_;
  std::size_t blocks_per_chunk_;
  std::size_t live_ = 0;
};

// ---- the standard-library adaptor -----------------------------------------
//
// An Allocator in the C++ sense is a handle, not an owner: it must be
// copyable, cheap to copy, and two copies must be interchangeable. So this
// holds a POINTER to the pool rather than the pool itself. Getting that
// backwards is the classic mistake, because a container is free to copy its
// allocator whenever it likes.

template <typename T>
class PoolAllocator {
 public:
  using value_type = T;

  explicit PoolAllocator(Pool* pool) noexcept : pool_(pool) {}

  // Required so a container can rebind to its internal node type. This is why
  // a std::list<int> allocator is never actually asked for ints.
  template <typename U>
  PoolAllocator(const PoolAllocator<U>& other) noexcept : pool_(other.pool()) {}

  T* allocate(std::size_t n) {
    // A pool serves exactly one size. Anything else falls back to the global
    // allocator rather than silently handing back a block that is too small.
    if (n != 1 || sizeof(T) > pool_->block_size()) {
      return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{alignof(T)}));
    }
    return static_cast<T*>(pool_->allocate());
  }

  void deallocate(T* p, std::size_t n) noexcept {
    if (n != 1 || sizeof(T) > pool_->block_size()) {
      ::operator delete(static_cast<void*>(p), std::align_val_t{alignof(T)});
      return;
    }
    pool_->deallocate(p);
  }

  Pool* pool() const noexcept { return pool_; }

  // Two allocators are equal when memory from one can be freed by the other,
  // which here means pointing at the same pool. Containers rely on this when
  // deciding whether a move can steal the buffer.
  template <typename U>
  bool operator==(const PoolAllocator<U>& o) const noexcept {
    return pool_ == o.pool();
  }
  template <typename U>
  bool operator!=(const PoolAllocator<U>& o) const noexcept {
    return !(*this == o);
  }

 private:
  Pool* pool_;
};
