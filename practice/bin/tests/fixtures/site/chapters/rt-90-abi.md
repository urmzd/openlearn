<!-- ol:module rt.90 -->
# Demo ABI: status codes, error slot, allocator hook

## Overview

| | |
|---|---|
| **Module** | `rt.90` · side · C · Pass 1 · 20 min |
| **You build** | `c/src/runtime/abi.c`: `tl_abi_version`, `tl_status_str`, `tl_set_allocator`, `tl_alloc`, `tl_free` |
| **Contract** | [`contracts/c/include/tinyllm/abi.h`](../course/contracts/c/include/tinyllm/abi.h) |
| **Tests** | `course/tests/rt.90/` (what they check: section 4) |
| **Needs** | nothing |
| **Used by** | `rt.91` |
| **Optional depth** | none: this is a harness fixture |

## Key Takeaways

- A status is an `int32_t`, never a C enum type, because a Rust `repr(C)` enum holding an unknown value is undefined behavior.

## How to work this chapter

```bash
ol start rt.90
ol tests rt.90
ol check rt.90
```

---

## 1. Why now

Nothing in C can report a failure yet: every later unit needs a status code, a place to put a message, and an allocator the tests can count.

## 2. Principles

A status is an `int32_t`, never a C enum type, because a Rust `repr(C)` enum holding an unknown value is undefined behavior. Allocation goes through one hook so a test can count live blocks and fail the n-th allocation.

## 3. Worked example by hand

`tl_alloc(10, 64)` rounds 10 up to 64 bytes and returns a pointer whose address mod 64 is 0; with the counting allocator installed, the live count goes from 0 to 1, and `tl_free` brings it back to 0 (`alloc_goes_through_the_hook`).

## 4. The interface

```c
uint32_t tl_abi_version(void);
const char *tl_status_str(tl_status s);
tl_status tl_set_allocator(const tl_allocator *a);
void *tl_alloc(size_t n, size_t align);
void tl_free(void *p);
```

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `alloc_goes_through_the_hook` | fault | the counting allocator only sees allocations that use tl_alloc, and 64-byte alignment is what the kernels rely on | rt.91 |
| `abi_version_matches_header` | unit | a C consumer can compare the library version with the header's version before using the interface | rt.91 |
| `status_strings` | unit | every status has a readable name and an unknown code still gets one | rt.91 |
| `set_allocator_rejects_half_a_hook` | boundary | a hook with alloc but no free would leak every block it hands out | rt.91 |

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| accepting an allocator with no free function | every block leaks | `set_allocator_rejects_half_a_hook` |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Forward | `rt.91` | demo float sum and owned buffer |

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| demo abi: status codes, error slot, allocator hook | the real course module this fixture stands in for | everything | `course/DESIGN.md` |
