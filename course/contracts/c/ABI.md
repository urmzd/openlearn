# The C interfaces of the optional C modules

`include/tinyllm.h` is a C-only umbrella header over one header per unit in `include/tinyllm/`. One module owns one header section and one source file, which is what lets the course swap a single unit for a stub or a reference. Every C module is optional depth: each one builds its own test binary, and its numerical parity with the Python reference is checked through fixture files the Python reference writes. Python, Rust, and Go never load or link this code (D8): there is no shared library ABI, no `ctypes`, and no Rust binding.

## Headers

| Header | Unit | Owner |
|---|---|---|
| [`tinyllm/abi.h`](include/tinyllm/abi.h) | `c/src/runtime/abi.c` | `rt.02` (status codes, the error slot, the allocator hook) |
| [`tinyllm/arena.h`](include/tinyllm/arena.h) | `c/src/runtime/arena.c` | `rt.02` |
| [`tinyllm/pool.h`](include/tinyllm/pool.h) | `c/src/runtime/pool.c` | `rt.03` |
| [`tinyllm/kv_pool.h`](include/tinyllm/kv_pool.h) | `c/src/runtime/kv_pool.c` | `rt.04` |
| [`tinyllm/kv_pool_v2.h`](include/tinyllm/kv_pool_v2.h) | `c/src/runtime/kv_pool.c` | an optional C mirror of KV format v2 (`craft.13` owns the format in Rust); not in `tinyllm.h` (D13) |
| [`tinyllm/ds.h`](include/tinyllm/ds.h) | `c/src/ds/vec.c`, `swiss.c`, `list.c`, `lru.c` | `ds.01`, `ds.02`, `ds.03` |
| [`tinyllm/topk.h`](include/tinyllm/topk.h) | `c/src/ds/topk.c` | `ds.04` |
| [`tinyllm/numerics.h`](include/tinyllm/numerics.h) | `c/src/numerics/lowp.c`, `rsqrt.c`, `expf.c` | `M09.7`, `M09.5`, `M09.6` |
| [`tinyllm/matmul.h`](include/tinyllm/matmul.h) | `c/src/kernels/matmul.c` | `L9.1` |
| [`tinyllm/softmax.h`](include/tinyllm/softmax.h) | `c/src/kernels/softmax.c` | `L9.2` |
| [`tinyllm/attention.h`](include/tinyllm/attention.h) | `c/src/kernels/flash_attn.c`, `paged_attn.c` | `L9.3`, `L9.4` |
| [`tinyllm/qmatmul.h`](include/tinyllm/qmatmul.h) | `c/src/kernels/qmatmul.c` | `L9.5` |
| [`tinyllm/elementwise.h`](include/tinyllm/elementwise.h) | `c/src/kernels/elementwise.c` | `L9.6` |

A header that names several units is split by function: each unit defines exactly the functions of its section. Units not started yet are linked as stubs (rule 11), so including the whole umbrella is always safe.

### Struct layouts (64-bit targets)

| Struct | Size | Fields (offset) |
|---|---|---|
| `tl_allocator` | 24 | `alloc` 0, `free` 8, `user` 16 |
| `tl_arena_mark` | 16 | `offset` 0, `block` 8 |
| `tl_arena_stats` | 32 | `bytes_used` 0, `bytes_reserved` 8, `high_water` 16, `n_blocks` 24 |
| `tl_kv_cfg` | 28 | `n_blocks` 0, `block_tokens` 4, `n_layers` 8, `n_kv_heads` 12, `head_dim` 16, `dtype` 20, `format` 24 |
| `tl_kv_stats` | 16 | `free` 0, `used` 4, `cached` 8, `evictions` 12 |
| `tl_vec` | 32 | `data` 0, `len` 8, `cap` 16, `elem` 24 |
| `tl_list_node` | 16 | `prev` 0, `next` 8 |
| `tl_lru` | 24 | `head` 0, `len` 16 |

The layouts are fixed so that a C test can write a struct's bytes into a fixture and compare them; no other language reads these structs in memory.

## Rules

1. **The caller owns every buffer.** No function allocates except a `*_create` constructor, and constructors allocate only through the allocator hook (`tl_alloc`).
2. **Constructors** return a `tl_status` and write the object through an out parameter: `tl_status tl_x_create(..., tl_x **out)`. Each has a matching `tl_x_destroy`, which accepts `NULL`.
3. **Callbacks stay inside C.** The only function pointer in the interfaces is `tl_parallel_for`'s range function (rt.03).
4. **Shapes and strides are `int64_t`.** Matrices are row-major with an explicit leading dimension (the distance in elements between two rows).
5. **Enums are `int32_t`.** `tl_status` and `tl_dtype` are `int32_t` with named constants, and struct fields that carry them are `int32_t`, so a fixture file stores them at a fixed width.
6. **Errors.** A function that fails returns a positive `tl_status` (or `NULL`) and first sets the thread's error slot with `tl_set_last_error`. Read it with `tl_last_error()` before the next `tl_` call on that thread.
7. **Exports.** Every public function is named `tl_`; helpers are `static`.
8. **Layout.** Every public struct's size and field offsets match the table above; a harness self-test compiles the headers and checks them with `sizeof` and `offsetof`.
9. **Thread safety.** Kernels are reentrant. Stateful objects (arena, KV pool, map, LRU) are not thread-safe: the caller serializes every call on one object. The error slot is `_Thread_local`. Threads use pthreads only (`<threads.h>` is missing on macOS).
10. **Batch invariance** (from L9.1, L9.3, L9.4). Each output element's reduction uses one fixed order, independent of the batch size, of the row's position in the batch, and of the tile. Row `i` of a product computed with `M = 1` equals row `i` computed with `M = 37` bit for bit.
11. **Stubs.** The harness links a stub object for every unit not yet started: each function sets the error slot and returns `TL_EUNSUPPORTED` (or the type's zero value), so a test binary always links and only the stubbed call fails.
12. **Versioning.** `tl_abi_version()` returns `TL_ABI_VERSION`, the version of these headers. A C test that reads a fixture checks it; changing it is a migration.
13. **Standard.** Every header compiles on its own under `cc -std=c11 -pedantic -Wall -Werror`.

## Building

The system's `c/Makefile` may produce `c/build/libtinyllm.a` for local experiments; `SANITIZE=1` adds `-fsanitize=address,undefined`. Each module's C test binary is compiled from its own objects: an ASan and UBSan build with the counting allocator, and a ThreadSanitizer build for modules whose tests declare `sanitize = ["thread"]`. Results reach other languages only as fixture files and process output.
