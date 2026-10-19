// main.cpp - the test suite for the reference implementation. Exits 0 when
// every check holds.
#include "pool.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <map>
#include <set>
#include <vector>

static int checks = 0;
#define CHECK(cond, what)                                                     \
  do {                                                                        \
    ++checks;                                                                 \
    if (!(cond)) {                                                            \
      std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, (what));   \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

struct Node {
  int value;
  double weight;
};

static bool is_aligned(const void* p, std::size_t a) {
  return reinterpret_cast<std::uintptr_t>(p) % a == 0;
}

static void test_allocate_and_deallocate() {
  Pool pool(sizeof(Node), alignof(Node), 8);
  CHECK(pool.live_blocks() == 0, "a fresh pool has nothing live");
  CHECK(pool.chunk_count() == 0, "and has allocated nothing");

  void* a = pool.allocate();
  CHECK(a != nullptr, "allocation succeeds");
  CHECK(pool.live_blocks() == 1, "one block live");
  CHECK(pool.chunk_count() == 1, "the first allocation created a chunk");

  void* b = pool.allocate();
  CHECK(b != a, "a second allocation is a different block");
  CHECK(pool.live_blocks() == 2, "two live");

  pool.deallocate(a);
  CHECK(pool.live_blocks() == 1, "deallocation drops the count");

  pool.deallocate(nullptr);
  CHECK(pool.live_blocks() == 1, "deallocating nullptr is harmless");
}

// Freed blocks must come back. A pool that leaked its free list would keep
// growing chunks instead.
static void test_blocks_are_reused() {
  Pool pool(sizeof(Node), alignof(Node), 4);

  void* first = pool.allocate();
  pool.deallocate(first);
  void* again = pool.allocate();
  CHECK(again == first, "a freed block is handed straight back");
  CHECK(pool.chunk_count() == 1, "without allocating a new chunk");
  pool.deallocate(again);

  // Cycle far more times than the pool has capacity for.
  for (int i = 0; i < 10000; ++i) {
    void* p = pool.allocate();
    pool.deallocate(p);
  }
  CHECK(pool.chunk_count() == 1, "10000 alloc/free cycles reused one chunk");
  CHECK(pool.live_blocks() == 0, "and left nothing live");
}

static void test_blocks_never_overlap() {
  Pool pool(sizeof(Node), alignof(Node), 8);
  std::vector<Node*> ptrs;

  // Allocate well past one chunk, so several chunks are in play.
  for (int i = 0; i < 100; ++i) {
    Node* p = static_cast<Node*>(pool.allocate());
    p->value = i;
    p->weight = i * 1.5;
    ptrs.push_back(p);
  }
  CHECK(pool.live_blocks() == 100, "all live");
  CHECK(pool.chunk_count() > 1, "the test really did span several chunks");

  // Every pointer distinct.
  std::set<void*> unique(ptrs.begin(), ptrs.end());
  CHECK(unique.size() == 100, "every block has a distinct address");

  // And nothing was overwritten by a neighbour.
  for (int i = 0; i < 100; ++i) {
    CHECK(ptrs[static_cast<std::size_t>(i)]->value == i, "block contents intact");
    CHECK(ptrs[static_cast<std::size_t>(i)]->weight == i * 1.5, "including the second field");
  }

  for (Node* p : ptrs) pool.deallocate(p);
  CHECK(pool.live_blocks() == 0, "all returned");
}

// Freeing in a scrambled order must be fine: the free list does not care.
static void test_out_of_order_free() {
  Pool pool(sizeof(Node), alignof(Node), 8);
  std::vector<void*> ptrs;
  for (int i = 0; i < 64; ++i) ptrs.push_back(pool.allocate());

  // Free the odd indices, then the even ones.
  for (std::size_t i = 1; i < ptrs.size(); i += 2) pool.deallocate(ptrs[i]);
  CHECK(pool.live_blocks() == 32, "half returned");
  for (std::size_t i = 0; i < ptrs.size(); i += 2) pool.deallocate(ptrs[i]);
  CHECK(pool.live_blocks() == 0, "all returned");

  // Everything must be reusable afterwards, with no duplicates.
  std::set<void*> reissued;
  for (int i = 0; i < 64; ++i) reissued.insert(pool.allocate());
  CHECK(reissued.size() == 64, "every freed block came back exactly once");
  CHECK(pool.chunk_count() <= 8, "and no extra chunks were needed");
}

static void test_alignment() {
  // An over-aligned block, beyond max_align_t.
  Pool pool(64, 64, 4);
  for (int i = 0; i < 16; ++i) {
    void* p = pool.allocate();
    CHECK(is_aligned(p, 64), "every block honours a 64-byte alignment");
  }

  // A block smaller than a pointer must still be usable, because the free
  // list needs to store a pointer inside it.
  Pool tiny(1, 1, 4);
  CHECK(tiny.block_size() >= sizeof(void*),
        "blocks are at least pointer-sized, so the free list fits");
  void* a = tiny.allocate();
  void* b = tiny.allocate();
  CHECK(a != b, "tiny blocks are still distinct");
  tiny.deallocate(a);
  tiny.deallocate(b);
}

// The container tests. std::list and std::map are node-based, so every insert
// is one fixed-size allocation, which is exactly what a pool is for.
static void test_with_std_list() {
  // A list node is bigger than its value type: it also holds two pointers.
  // Sizing the pool generously and letting the allocator fall back is the
  // honest way to handle not knowing the node layout.
  Pool pool(64, alignof(std::max_align_t), 32);
  {
    std::list<int, PoolAllocator<int>> lst{PoolAllocator<int>(&pool)};
    for (int i = 0; i < 500; ++i) lst.push_back(i);
    CHECK(lst.size() == 500, "the list filled");
    CHECK(pool.live_blocks() > 0, "and it used the pool");

    int expect = 0;
    for (int v : lst) CHECK(v == expect++, "values are intact and in order");

    lst.remove_if([](int v) { return v % 2 == 0; });
    CHECK(lst.size() == 250, "removal worked");
  }
  CHECK(pool.live_blocks() == 0, "the list returned every block on destruction");
}

static void test_with_std_map() {
  Pool pool(96, alignof(std::max_align_t), 32);
  {
    using Alloc = PoolAllocator<std::pair<const int, int>>;
    std::map<int, int, std::less<int>, Alloc> m{std::less<int>(), Alloc(&pool)};
    for (int i = 0; i < 300; ++i) m[i] = i * 2;
    CHECK(m.size() == 300, "the map filled");

    for (int i = 0; i < 300; ++i) CHECK(m[i] == i * 2, "values are correct");

    for (int i = 0; i < 300; i += 2) m.erase(i);
    CHECK(m.size() == 150, "erasure worked");
  }
  CHECK(pool.live_blocks() == 0, "the map returned every block");
}

// A vector allocates ONE block of n elements, which a pool cannot serve. The
// adaptor must fall back rather than hand back a too-small block.
static void test_fallback_for_multi_element_allocation() {
  Pool pool(sizeof(int), alignof(int), 8);
  {
    std::vector<int, PoolAllocator<int>> v{PoolAllocator<int>(&pool)};
    for (int i = 0; i < 1000; ++i) v.push_back(i);
    CHECK(v.size() == 1000, "the vector filled despite the pool being unsuitable");
    CHECK(v[999] == 999, "with correct contents");
    CHECK(v[0] == 0, "throughout");
  }
  CHECK(pool.live_blocks() == 0, "nothing was left live");
}

// An allocator is a handle: copies must be interchangeable, and memory taken
// through one must be returnable through another.
static void test_allocator_is_a_handle() {
  Pool pool(sizeof(Node), alignof(Node), 8);
  PoolAllocator<Node> a(&pool);
  PoolAllocator<Node> b = a;

  CHECK(a == b, "copies of an allocator compare equal");

  Node* p = a.allocate(1);
  b.deallocate(p, 1);  // freed through the copy
  CHECK(pool.live_blocks() == 0, "memory from one copy is freeable by another");

  // Rebinding to another type must keep pointing at the same pool.
  PoolAllocator<int> rebound(a);
  CHECK(rebound.pool() == a.pool(), "rebinding preserves the pool");
  CHECK(rebound == a, "and compares equal across types");

  Pool other(sizeof(Node), alignof(Node), 8);
  PoolAllocator<Node> c(&other);
  CHECK(!(a == c), "allocators on different pools are not equal");
}

static void test_release() {
  Pool pool(sizeof(Node), alignof(Node), 8);
  for (int i = 0; i < 50; ++i) (void)pool.allocate();
  CHECK(pool.chunk_count() > 1, "several chunks in use");

  pool.release();
  CHECK(pool.chunk_count() == 0, "release returned every chunk");
  CHECK(pool.live_blocks() == 0, "and reset the live count");

  // The pool must still work afterwards.
  void* p = pool.allocate();
  CHECK(p != nullptr, "the pool is usable after release");
  CHECK(pool.chunk_count() == 1, "and allocated a fresh chunk");
}

int main() {
  test_allocate_and_deallocate();
  test_blocks_are_reused();
  test_blocks_never_overlap();
  test_out_of_order_free();
  test_alignment();
  test_with_std_list();
  test_with_std_map();
  test_fallback_for_multi_element_allocation();
  test_allocator_is_a_handle();
  test_release();
  std::printf("ok  cpp/10-pool-allocator  %d checks passed\n", checks);
  return 0;
}
