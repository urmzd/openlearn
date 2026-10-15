<!-- ol:module ds.02 -->
# Swiss table tl_map (u64 to u64, optional C)

## Overview

| | |
|---|---|
| **Module** | `ds.02` · side · C · Pass 6 · 5 to 7 h |
| **You build** | `c/src/ds/swiss.c`: `tl_map_create`, `tl_map_put`, `tl_map_get`, `tl_map_del`, `tl_map_len`, `tl_map_destroy` |
| **Contract** | the ds.02 section of [`tinyllm/ds.h`](../../../../course/contracts/c/include/tinyllm/ds.h); rules in [`c/ABI.md`](../../../../course/contracts/c/ABI.md) |
| **Tests** | `course/tests/ds.02/test_swiss.c`, 11 tests under ASan and UBSan with the counting allocator (what they check: section 4) · your own tests in `c/tests/ds02-swiss/`, rung R2, graded by mutation (threshold 0.60) · bench `ol bench ds.02` |
| **Needs** | `rt.02` (C allocation and error support) · reading: `ds.05` [Robin Hood hashing](05-robin-hood-hash-map.md) (the first hash-table chapter) · `S-M06a` [load factor and expected probe length](../../math/06-discrete-math-2/90-problem-set-a.md) |
| **Used by** | `rt.04` keeps its prefix index (block hash to block id) in a `tl_map` |
| **Milestone** | `MS-L9`, the optional standalone C module group |
| **Optional depth** | Abseil, [Swiss Tables design notes](https://abseil.io/about/design/swisstables); Matt Kulukundis, *Designing a Fast, Efficient, Cache-friendly Hash Table, Step by Step* (CppCon 2017); Knuth, *The Art of Computer Programming*, vol. 3, section 6.4 |

## Key Takeaways

- A Swiss table keeps one control byte per slot (7 bits of the key's hash, or a marker for EMPTY and DELETED) and compares 16 of them with a few word operations, so a probe reads a key only when its 7-bit fingerprint matched (`keys_sharing_a_control_byte_are_told_apart`).
- One group of 16 slots holds 14 keys at the maximum load of 7/8; the 15th key is the one that needs a bigger table (`hand_example_fourteen_fit_the_fifteenth_grows`).
- A delete leaves a tombstone only when a probe may have walked past the slot's group; in a group that still has an EMPTY slot it writes EMPTY, so churn at a steady size never allocates (`churn_in_a_group_with_room_never_allocates`, `probe_continues_past_a_full_group`).
- Tombstones count toward the load; when they fill the table it is rebuilt at the same size, and it doubles only for live keys (`tombstone_churn_rehashes_without_growing`).
- A rebuild allocates the new arrays before touching the old ones, so a failed growth leaves every key in place (`failed_growth_keeps_the_old_table`).

## How to work this chapter

```bash
ol start ds.02              # stubs c/src/ds/swiss.c (every body returns TL_EUNSUPPORTED or 0)
ol tests ds.02              # read the test catalog first
ol check ds.02              # exit code is the verdict; then grades your tests by mutation
ol bench ds.02              # local: lookups against the linear-probing baseline
ol diff  ds.02              # after passing: your code against the reference
```

Start with `cap_for` and `tl_map_create` on the worked sizes of section 3, then `find` with a plain byte loop over the group (no word tricks yet), `tl_map_put`, and `tl_map_del`. Once `ol check ds.02` passes, replace the byte loop with the SWAR match of section 2.4 and run it again.

---

## 1. Why now

The KV block pool you build next (`rt.04`) must answer one question on every prompt: "is the block whose tokens hash to `h` already in memory, and which block is it?" The answer is a map from a 64-bit block hash to a 32-bit block id, consulted for every full block of every prompt, mostly with hashes it does not hold (a new prompt), and updated every time a block is cached or evicted. Practice drill `c/02` gave you a linear-probing map: correct, but a miss walks the whole run of occupied slots to the next empty one, and with deletes its tombstones make every run longer until a rebuild. This chapter keeps the drill's open addressing and changes how a probe looks at slots, so a miss costs one 16-byte scan and the table holds 7/8 of its slots without slowing down.

## 2. Principles

### 2.1 Hash, fingerprint, home group

| Symbol | Meaning | Type |
|---|---|---|
| $k$ | a key; every value, 0 and $2^{64}-1$ included | `uint64_t` |
| $h = \text{mix}(k)$ | the SplitMix64 finalizer of $k$ | `uint64_t` |
| $H_2 = h \bmod 2^7$ | the key's 7-bit fingerprint, stored in its control byte | 0 to 127 |
| $H_1 = \lfloor h / 2^7 \rfloor$ | the rest of the hash; picks the home group | `uint64_t` |
| $G$ | number of groups, a power of two; the table has $16G$ slots | `size_t` |
| $n$, $u$ | live keys; used slots (live keys plus tombstones) | `size_t` |

The integer keys here are block hashes and ids, but a map must not trust its callers to spread their keys, so the table mixes every key first: $\text{mix}(x)$ is `x ^= x >> 30; x *= 0xBF58476D1CE4E5B9; x ^= x >> 27; x *= 0x94D049BB133111EB; x ^= x >> 31`, in which every input bit reaches every output bit. The slots form groups of 16; a key's **home group** is $H_1 \bmod G$. Each slot has a **control byte**:

| Control byte | Meaning |
|---|---|
| `0x80` EMPTY | never used since the last rebuild: a probe for any key may stop here |
| `0xFE` DELETED | a tombstone: a removed key that a probe must walk past |
| `0x00` to `0x7F` FULL | the slot holds a key whose fingerprint $H_2$ is this byte |

Keys and values live in a separate array of `{key, value}` pairs, so the table never reserves a key value to mean "empty": the control byte says it.

### 2.2 The probe

A lookup visits groups home, home + 1, home + 3, home + 6, ... (step $i$ adds $i(i+1)/2$, modulo $G$), which reaches every group exactly once when $G$ is a power of two. In each group it finds the slots whose control byte equals $H_2$ and compares only those keys. If the group has any EMPTY slot, the key is absent: an insert would have put it in that slot or earlier. Otherwise it moves to the next group. With 7 bits of fingerprint, a group of 16 has on average $16/128$ false candidates, so almost every key read is the right one.

### 2.3 Load, tombstones, and rebuilds

Like practice `c/02`, deleting a key cannot simply write EMPTY: a key that was inserted past this slot's group (because the group was full) would become unreachable. The table writes DELETED instead. But if the slot's group **still has an EMPTY slot**, no probe has ever continued past this group (a probe continues only through a group with no EMPTY slot, and a group never regains an EMPTY slot until a rebuild), so the delete can safely write EMPTY.

Tombstones occupy slots, so the load the table controls is $u$, not $n$. An insert that would turn an EMPTY slot FULL when $u = \frac78 \cdot 16G$ first rebuilds the table into `cap_for(n + 1)` slots: the smallest $16 \cdot 2^j$ with $n + 1 \le \frac78 \cdot 16 \cdot 2^j$, but never fewer than `tl_map_create` gave. A rebuild drops every tombstone. When tombstones caused the pressure the table is rebuilt **at the same size**; it doubles only when the live keys need it. `tl_map_create(hint)` sizes the table the same way, so `hint` keys fit without any allocation, and `hint = 0` allocates no slots until the first key.

A rebuild allocates the new control bytes and slots first. If the allocator hook fails, the call returns `TL_ENOMEM` and the old table is untouched.

### 2.4 Matching 8 control bytes at once (SWAR)

Load 8 control bytes as one 64-bit word $w$ (byte $i$ in bits $8i$ to $8i+7$), and let $L = \texttt{0x0101010101010101}$, $M = \texttt{0x8080808080808080}$.

- **Fingerprint match.** $x = w \oplus (L \cdot H_2)$ turns every byte equal to $H_2$ into 0. Then $(x - L) \mathbin{\&} \lnot x \mathbin{\&} M$ has bit 7 of byte $i$ set where byte $i$ of $x$ is 0: subtracting 1 from a zero byte borrows and sets its top bit, and $\lnot x$ keeps that top bit only for bytes whose own top bit was clear. A borrow can also report a FULL byte equal to $H_2 \oplus 1$ right above a match; that is harmless because every candidate's key is compared. EMPTY and DELETED bytes are never reported (their top bit is set, so $\lnot x$ clears it).
- **EMPTY match.** $w \mathbin{\&} \lnot(w \ll 6) \mathbin{\&} M$: bit 7 of a byte is set where the byte has bit 7 set and bit 1 clear, which among control bytes is EMPTY (`1000 0000`) only; DELETED is `1111 1110`.
- **Free slots** (EMPTY or DELETED) are simply $w \mathbin{\&} M$.

Each set bit $8i + 7$ names slot $i$; the lowest one is found with a count of trailing zeros, divided by 8. A group is two such words.

## 3. Worked example by hand

**Sizes.** One group is 16 slots, and $\frac78 \cdot 16 = 14$. So `tl_map_create(14)` makes 16 slots, 14 keys go in with every allocation failing, and the 15th key needs a rebuild into 32 slots: it gets `TL_ENOMEM` while allocation is failing, and the 14 keys are all still there. `tl_map_create(15)` makes 32 slots ($\frac78 \cdot 32 = 28$), so 28 keys fit and the 29th grows. This is `hand_example_fourteen_fit_the_fifteenth_grows`.

**A fingerprint.** For key 42, $h = \text{mix}(42) = \texttt{0xA759EA27D4727622}$: $H_2 = \texttt{0x22}$ and the low bit of $H_1$ is 0, so in a 2-group table its home is group 0.

**One SWAR match.** Take 8 control bytes (byte 0 first) `2A 80 13 2A FE 05 2A 80` and look for $H_2 = \texttt{0x2A}$:

| Step | Bytes 0 to 7 |
|---|---|
| $w$ | `2A 80 13 2A FE 05 2A 80` |
| $x = w \oplus (L \cdot \texttt{2A})$ | `00 AA 39 00 D4 2F 00 AA` |
| $x - L$ | `FF A8 38 FF D2 2E FF A8` (the borrow out of byte 0 makes byte 1 `A8`, not `A9`) |
| $\lnot x$ | `FF 55 C6 FF 2B D0 FF 55` |
| $(x - L) \mathbin{\&} \lnot x \mathbin{\&} M$ | `80 00 00 80 00 00 80 00`: slots 0, 3, 6 are candidates |
| $w \mathbin{\&} \lnot(w \ll 6) \mathbin{\&} M$ | `00 80 00 00 00 00 00 80`: slots 1 and 7 are EMPTY |

So a lookup compares three keys here, and if none matches, the EMPTY at slot 1 ends the probe: the key is absent, and the DELETED byte at slot 4 did not stop it.

## 4. The interface

```c
/* tinyllm/ds.h (ds.02) */
typedef struct tl_map tl_map;                         /* opaque */
tl_status tl_map_create(size_t hint, tl_map **out);    /* TL_ENOMEM, TL_EINVAL */
tl_status tl_map_put(tl_map *m, uint64_t k, uint64_t v);  /* insert or overwrite; a failed growth is TL_ENOMEM, table unchanged */
int       tl_map_get(const tl_map *m, uint64_t k, uint64_t *v);  /* 1 found (v may be NULL), 0 not */
int       tl_map_del(tl_map *m, uint64_t k);           /* 1 removed, 0 absent; never allocates */
size_t    tl_map_len(const tl_map *m);
void      tl_map_destroy(tl_map *m);                   /* NULL does nothing */
```

Every allocation goes through `tl_alloc` (rt.01), so the counting allocator of `ol_test.h` sees it and can fail it. The chapter fixes what the header leaves open: the hash (section 2.1), the sizing rule of `tl_map_create` and of a rebuild (section 2.3), and the EMPTY-in-group rule for deletes. A different hash still passes every test; the tests pick their keys with this one to force the hard cases.

### What the tests check

| Test | KIND | Checks | Why it matters downstream |
|---|---|---|---|
| `hand_example_fourteen_fit_the_fifteenth_grows` | unit, boundary | section 3: 14 keys in 16 slots with allocation failing, the 15th is `TL_ENOMEM`; 28 fit in 32 | rt.04 sizes the index for `n_blocks` once and must never allocate afterwards |
| `put_get_overwrite_delete` | unit | overwrite keeps `len`; `get` with a NULL value pointer; delete twice | the map semantics before any probing detail |
| `every_u64_is_a_key` | boundary, regression | keys 0, `UINT64_MAX`, `0x80`, `0xFE` | block hashes are any u64 but 0; nothing is reserved |
| `keys_sharing_a_control_byte_are_told_apart` | unit | 14 keys with one $H_2$ in one group; a 15th with the same $H_2$ is absent | a fingerprint match is only a candidate |
| `probe_continues_past_a_full_group` | unit | 20 keys homed in group 0 of 32 slots; deletes in the full group keep the spilled keys reachable | tombstones where a probe may have passed |
| `reput_after_a_tombstone_overwrites` | unit | a spilled key put again after a delete in its home group overwrites, never duplicates | an insert searches the whole probe before reusing a tombstone |
| `churn_in_a_group_with_room_never_allocates` | fault | 10,000 delete-and-insert rounds at 13 keys in 16 slots with allocation failing | the EMPTY-in-group rule keeps $u$ from creeping up |
| `tombstone_churn_rehashes_without_growing` | property | 20,000 rounds at 28 keys in 32 slots: the largest block ever requested stays the first one | rebuilds drop tombstones at the same size |
| `failed_growth_keeps_the_old_table` | fault | failing each allocation of a rebuild in turn: `TL_ENOMEM`, every key intact, no leak | the contract's "old table valid and unchanged" |
| `create_and_null_arguments` | boundary, fault | `TL_EINVAL` for a NULL out; create under allocation failure; NULL map calls | cleanup paths in rt.04 need no special cases |
| `differential_against_linear_probing` | differential | $10^6$ seeded puts, gets, deletes agree with the practice `c/02` baseline on every result | the Swiss table is a map |

**Your tests (rung R2).** Write C files under `c/tests/ds02-swiss/` (including only `tinyllm.h`, `tinyllm/*.h`, and `ol_*.h`, with one `main` returning `OL_RUN_ALL()`) with these tests, bodies yours: `fourteen_fit_then_grow`, `overwrite_keeps_len`, `shared_fingerprint_keys`, `spilled_keys_survive_deletes`, `failed_growth_keeps_keys`, `matches_a_simple_model`. `ol check ds.02` runs them against the reference with one planted bug at a time; at least 60% of the planted bugs must make one of them fail.

**Bench (local).** `ol bench ds.02` times lookups of both tables at load 3/4 (3,071 keys in 4,096 slots). The budget is on misses, the prefix index's common case: `miss_speedup >= 1.3`. The reference measures about 2 to 3 for misses and about 1.05 for hits; the chapter does not ask hits to be faster, because a hit in either table is one compare.

## 5. Pitfalls

| Pitfall | Symptom | Caught by |
|---|---|---|
| Writing EMPTY on every delete | keys that spilled past a full group vanish | `probe_continues_past_a_full_group`, `differential_against_linear_probing` (mutant `s01`) |
| Trusting a fingerprint match without comparing the key | `get` returns another key's value | `keys_sharing_a_control_byte_are_told_apart` (mutant `s02`) |
| Ending a probe at a tombstone | spilled keys are "absent" after a delete | `probe_continues_past_a_full_group` (mutant `s03`) |
| Growing at load 7/8 instead of above it | the 14th key allocates; the index of rt.04 grows when it must not | `hand_example_fourteen_fit_the_fifteenth_grows` (mutant `s04`) |
| Sizing `tl_map_create` one key short | `create(14)` gives 32 slots | `hand_example_fourteen_fit_the_fifteenth_grows` (mutant `s05`) |
| Forgetting the mask on the probe step | reads past the control bytes (ASan) | `differential_against_linear_probing` (mutant `s06`) |
| Freeing the old arrays before the new ones exist | a failed growth loses every key | `failed_growth_keeps_the_old_table` (mutant `s07`) |
| Reusing the first tombstone without finishing the search | the key exists twice; a delete leaves a stale copy | `reput_after_a_tombstone_overwrites` (mutant `s08`) |
| Counting an overwrite as a new key | `len` drifts upward | `put_get_overwrite_delete` (mutant `s09`) |
| Checking only the first candidate of a group | keys sharing a fingerprint are lost | `keys_sharing_a_control_byte_are_told_apart` (mutant `s10`) |
| Doubling on every rebuild | a table with constant live keys grows without bound | `tombstone_churn_rehashes_without_growing` (mutant `s13`) |
| Leaking the struct when the slot allocation fails | the counting allocator reports a live block | `create_and_null_arguments` (mutant `s14`) |

## 6. Where it's used next

| Direction | Module | How it uses this |
|---|---|---|
| Back | `rt.02` | allocation hooks and error reporting for standalone C modules |
| Back | `ds.05` | open addressing, the load factor, and probe lengths, first met in Rust |
| Forward | `rt.04` | the prefix index: `tl_kv_register` puts (hash, id), `tl_kv_lookup` gets, eviction deletes |
| Forward | `L10.4` | `--prefix-cache=hash` in the engine is your table behind rt.04, benchmarked against the radix tree of `ds.07` |

If you skip this module, `ol check rt.04` stops with `needs ds.02: build it, or pass --ref-deps`.

## Going further

| Your piece | Production equivalent | What it adds | Where to look |
|---|---|---|---|
| `tl_map` | [Abseil `flat_hash_map`](https://github.com/abseil/abseil-cpp/blob/master/absl/container/internal/raw_hash_set.h) | SSE2 and NEON group matches, a sentinel byte so groups may start at any slot, in-place tombstone cleanup | `raw_hash_set.h`: `Group`, `DropDeletesWithoutResize` |
| `tl_map` | [hashbrown](https://github.com/rust-lang/hashbrown) (Rust's `HashMap`) | the same design in Rust, generic over keys | `src/raw/mod.rs` |
| SWAR match | [Folly F14](https://github.com/facebook/folly/blob/main/folly/container/detail/F14Table.h) | 14-slot chunks with an overflow count instead of tombstones | `F14Table.h` |
