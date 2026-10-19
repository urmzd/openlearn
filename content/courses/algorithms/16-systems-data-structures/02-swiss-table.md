# A Swiss table

The [Paged KV block pool](../../build-an-llm/p08-inference/08-paged-kv-block-pool.md) in *Build an LLM* must answer one question on every prompt: "is the block whose tokens hash to `h` already in memory, and which block is it?" The answer is a map from a 64-bit block hash to a 32-bit block id. It is consulted for every full block of every prompt, mostly with hashes it does not hold (a new prompt), and updated every time a block is cached or evicted. So the case that matters most is the miss.

The practice C drill `c/02` built a linear-probing map. It is correct, but a miss walks the whole run of occupied slots to the next empty one, and with deletes its tombstones make every run longer until a rebuild. This section keeps the drill's open addressing and changes how a probe looks at slots, so that a miss costs one 16-byte scan and the table holds 7/8 of its slots without slowing down. The result is a **Swiss table**, the design behind Abseil's `flat_hash_map` and Rust's `HashMap`.

This section builds on [Robin Hood hashing](05-robin-hood-hash-map.md), the chapter's first hash-table section, where hashing, probing, and the load factor are developed from first principles, and on the [load factor and expected probe length problems](../../mathematics-for-computing/06-discrete-math-2/90-problem-set-a.md) in *Mathematics for Computing*. The C allocation hooks come from [C runtime support and arena allocator with marks](../../build-an-llm/p09-kernels/02-arena-allocator-with-marks.md). The interface is the `tl_map` section of [`tinyllm/ds.h`](../../../../course/contracts/c/include/tinyllm/ds.h), under the rules in [`c/ABI.md`](../../../../course/contracts/c/ABI.md), and the full solution is [`swiss.c`](../../../../course/solutions/c/src/ds/swiss.c).

> **Learning objectives**
> - Explain how a 7-bit fingerprint in a control byte lets a probe skip almost every key comparison.
> - Match 8 control bytes at once with SWAR word arithmetic, and work one match by hand.
> - State exactly when a delete may write EMPTY and when it must leave a tombstone.
> - Size the table so that `hint` keys fit without allocation, and rebuild at the same size when tombstones, not live keys, cause the pressure.
> - Make a failed growth leave every key in place.

## Hash, fingerprint, and home group

The notation for this section:

| Symbol | Meaning | Type |
|---|---|---|
| $k$ | a key; every value, 0 and $2^{64}-1$ included | `uint64_t` |
| $h = \text{mix}(k)$ | the SplitMix64 finalizer of $k$ | `uint64_t` |
| $H_2 = h \bmod 2^7$ | the key's 7-bit fingerprint, stored in its control byte | 0 to 127 |
| $H_1 = \lfloor h / 2^7 \rfloor$ | the rest of the hash; picks the home group | `uint64_t` |
| $G$ | number of groups, a power of two; the table has $16G$ slots | `size_t` |
| $n$, $u$ | live keys; used slots (live keys plus tombstones) | `size_t` |

The integer keys here are block hashes and ids, but a map must not trust its callers to spread their keys, so the table mixes every key first. $\text{mix}(x)$ is `x ^= x >> 30; x *= 0xBF58476D1CE4E5B9; x ^= x >> 27; x *= 0x94D049BB133111EB; x ^= x >> 31`, in which every input bit reaches every output bit.

The slots form groups of 16, and a key's **home group** is $H_1 \bmod G$. Each slot also has a **control byte** that records its state:

| Control byte | Meaning |
|---|---|
| `0x80` EMPTY | never used since the last rebuild: a probe for any key may stop here |
| `0xFE` DELETED | a tombstone: a removed key that a probe must walk past |
| `0x00` to `0x7F` FULL | the slot holds a key whose fingerprint $H_2$ is this byte |

Keys and values live in a separate array of `{key, value}` pairs, so the table never reserves a key value to mean "empty": the control byte says it.

### The probe

A lookup visits groups home, home + 1, home + 3, home + 6, and so on (step $i$ adds $i(i+1)/2$, modulo $G$), which reaches every group exactly once when $G$ is a power of two. In each group it finds the slots whose control byte equals $H_2$ and compares only those keys. If the group has any EMPTY slot, the key is absent: an insert would have put it in that slot or earlier. Otherwise it moves to the next group. With 7 bits of fingerprint, a group of 16 has on average $16/128$ false candidates, so almost every key read is the right one.

## Load, tombstones, and rebuilds

The probe's early exit at an EMPTY slot is what makes misses cheap, and deletes threaten it. As in the linear-probing drill, deleting a key cannot simply write EMPTY: a key that was inserted past this slot's group (because the group was full) would become unreachable. The table writes DELETED instead.

There is an important exception. If the slot's group **still has an EMPTY slot**, no probe has ever continued past this group. A probe continues only through a group with no EMPTY slot, and a group never regains an EMPTY slot until a rebuild. So in that case the delete can safely write EMPTY.

Tombstones occupy slots, so the load the table controls is $u$, not $n$. An insert that would turn an EMPTY slot FULL when $u = \frac78 \cdot 16G$ first rebuilds the table into `cap_for(n + 1)` slots: the smallest $16 \cdot 2^j$ with $n + 1 \le \frac78 \cdot 16 \cdot 2^j$, but never fewer than `tl_map_create` gave. A rebuild drops every tombstone. When tombstones caused the pressure, the table is rebuilt **at the same size**; it doubles only when the live keys need it. `tl_map_create(hint)` sizes the table the same way, so `hint` keys fit without any allocation, and `hint = 0` allocates no slots until the first key.

A rebuild allocates the new control bytes and slots first. If the allocator hook fails, the call returns `TL_ENOMEM` and the old table is untouched.

## Matching 8 control bytes at once

The fingerprint filter only pays off if checking 16 control bytes is cheap. The trick is SWAR, "SIMD within a register." Load 8 control bytes as one 64-bit word $w$ (byte $i$ in bits $8i$ to $8i+7$), and let $L = \texttt{0x0101010101010101}$, $M = \texttt{0x8080808080808080}$. Three masks answer every question the probe asks.

The **fingerprint match** starts with $x = w \oplus (L \cdot H_2)$, which turns every byte equal to $H_2$ into 0. Then $(x - L) \mathbin{\&} \lnot x \mathbin{\&} M$ has bit 7 of byte $i$ set where byte $i$ of $x$ is 0: subtracting 1 from a zero byte borrows and sets its top bit, and $\lnot x$ keeps that top bit only for bytes whose own top bit was clear. A borrow can also report a FULL byte equal to $H_2 \oplus 1$ right above a match; that is harmless because every candidate's key is compared. EMPTY and DELETED bytes are never reported (their top bit is set, so $\lnot x$ clears it).

The **EMPTY match** is $w \mathbin{\&} \lnot(w \ll 6) \mathbin{\&} M$: bit 7 of a byte is set where the byte has bit 7 set and bit 1 clear, which among control bytes is EMPTY (`1000 0000`) only; DELETED is `1111 1110`.

**Free slots** (EMPTY or DELETED) are simply $w \mathbin{\&} M$.

Each set bit $8i + 7$ names slot $i$; the lowest one is found with a count of trailing zeros, divided by 8. A group is two such words.

## Working it by hand

Start with sizes. One group is 16 slots, and $\frac78 \cdot 16 = 14$. So `tl_map_create(14)` makes 16 slots, 14 keys go in even with every allocation failing, and the 15th key needs a rebuild into 32 slots: it gets `TL_ENOMEM` while allocation is failing, and the 14 keys are all still there. `tl_map_create(15)` makes 32 slots ($\frac78 \cdot 32 = 28$), so 28 keys fit and the 29th grows.

Next, a fingerprint. For key 42, $h = \text{mix}(42) = \texttt{0xA759EA27D4727622}$: $H_2 = \texttt{0x22}$ and the low bit of $H_1$ is 0, so in a 2-group table its home is group 0.

Finally, one SWAR match. Take 8 control bytes (byte 0 first) `2A 80 13 2A FE 05 2A 80` and look for $H_2 = \texttt{0x2A}$. The table shows each intermediate word.

| Step | Bytes 0 to 7 |
|---|---|
| $w$ | `2A 80 13 2A FE 05 2A 80` |
| $x = w \oplus (L \cdot \texttt{2A})$ | `00 AA 39 00 D4 2F 00 AA` |
| $x - L$ | `FF A8 38 FF D2 2E FF A8` (the borrow out of byte 0 makes byte 1 `A8`, not `A9`) |
| $\lnot x$ | `FF 55 C6 FF 2B D0 FF 55` |
| $(x - L) \mathbin{\&} \lnot x \mathbin{\&} M$ | `80 00 00 80 00 00 80 00`: slots 0, 3, 6 are candidates |
| $w \mathbin{\&} \lnot(w \ll 6) \mathbin{\&} M$ | `00 80 00 00 00 00 00 80`: slots 1 and 7 are EMPTY |

So a lookup compares three keys here, and if none matches, the EMPTY at slot 1 ends the probe: the key is absent, and the DELETED byte at slot 4 did not stop it.

## The interface

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

Every allocation goes through `tl_alloc`, so a counting allocator can observe every block and fail any one of them on demand. This section fixes what the header leaves open: the hash, the sizing rule of `tl_map_create` and of a rebuild, and the EMPTY-in-group rule for deletes. A different hash would still be a correct map; this one is fixed so that the hard cases (keys sharing a fingerprint, keys spilling out of one group) can be constructed on purpose in tests.

## Building it step by step

Read the solution in the order it was designed: sizing, then the probe with a plain byte loop in mind, then the SWAR word tricks that replace that loop. Eight observations lead there.

1. **The map must accept every u64 as a key.** So no key value can mean "empty". Slot state has to live somewhere else: a separate byte per slot. That byte is the control byte.
2. **A byte has 8 bits; the state needs only 3 values (EMPTY, DELETED, FULL).** Spending the 7 spare bits on a piece of the hash ($H_2$) turns the state byte into a filter: a probe can reject a slot without touching its key. With 7 bits, a non-matching FULL slot passes the filter with probability $1/128$.
3. **The filter is only worth it if checking it is cheaper than checking keys.** Control bytes sit contiguously, so 16 of them are two 64-bit words, and a few ALU operations test all 8 bytes of a word at once. One group check costs about as much as one key compare in linear probing.
4. **Probing must therefore move a group at a time.** The home group comes from bits of the hash not used by $H_2$ ($H_1 = h \gg 7$), so the fingerprint and the group are independent. Triangular steps over a power-of-two group count visit every group exactly once, so a probe of $G$ steps is guaranteed to have seen the whole table.
5. **A miss must stop early.** An insert always takes the first free slot on the key's probe, so if a group on the probe still has an EMPTY slot, the key would have been placed there or earlier: absent. That is why a miss is usually one group check.
6. **Deletes must not break step 5.** Writing EMPTY over a key that a later key's probe walked past would end that probe too early. Hence the tombstone. But a probe walks past a group only if the group had no EMPTY slot at the time, and a group gains EMPTY slots only at a rebuild; so if the group has an EMPTY slot now, no probe ever passed it and EMPTY is safe. In the worked example, 14 keys in a 16-slot table always leave 2 EMPTY slots in the one group, so every delete writes EMPTY and $u$ never grows.
7. **Tombstones are slots a miss cannot stop at,** so the load limit must count them: $u$, not $n$. When $u$ hits $\frac78$ of the slots, a rebuild is needed, but its size depends only on $n$ (`cap_for(n + 1)`). If tombstones caused the pressure, that is the same size, which keeps a steady-size workload from growing forever.
8. **A rebuild that fails must change nothing,** so it allocates both new arrays first and only then moves keys and frees the old arrays. Moving keys cannot fail: the new table has room by construction.

### Hashing and sizing

`cap_for` is the one function both `tl_map_create` and a rebuild call, so "create(hint) holds hint keys without allocating" and "a rebuild sizes for the live keys" are the same rule.

```c
#define GROUP 16
#define CTRL_EMPTY 0x80u      /* 1000 0000 */
#define CTRL_DELETED 0xFEu    /* 1111 1110: bit 1 set tells it apart from EMPTY */
#define LSB 0x0101010101010101ull
#define MSB 0x8080808080808080ull

typedef struct {
    uint64_t key, val; /* side by side: a hit reads one cache line */
} slot;

struct tl_map {
    uint8_t *ctrl;  /* cap control bytes */
    slot *slots;    /* cap slots */
    size_t cap;     /* 0 or 16 * 2^k */
    size_t cap_min; /* the capacity create() sized for its hint: never shrink below it */
    size_t len;     /* live keys (n) */
    size_t used;    /* live keys + tombstones (u): the quantity the load limit controls */
};

/* Smallest legal capacity holding n keys at load <= 7/8; 0 for n == 0;
 * SIZE_MAX when it would overflow. */
static size_t cap_for(size_t n) {
    if (n == 0) return 0;               /* create(0) allocates no slots */
    size_t cap = GROUP;
    while (n > cap / 8 * 7) {           /* 16 -> 14 keys, 32 -> 28, ... */
        if (cap > SIZE_MAX / 2 / (1 + sizeof(slot))) return SIZE_MAX;  /* ctrl + slots bytes would overflow */
        cap *= 2;
    }
    return cap;
}
```

### The SWAR matchers

These are the formulas from the matching discussion, verbatim. `load_group` byte-swaps on a big-endian host so that "byte $i$ of the group is byte $i \bmod 8$ of word $i / 8$, least significant first" holds everywhere, which is what `lowest_byte` assumes.

```c
/* Bit 7 of byte i is set when byte i of w equals b (b < 0x80). A FULL byte
 * equal to b ^ 1 right above a match can also be reported (a borrow), so
 * callers always compare the key itself. EMPTY and DELETED are never
 * reported: their bit 7 is set, so ~x clears it. */
static uint64_t match_byte(uint64_t w, uint8_t b) {
    uint64_t x = w ^ (LSB * b);          /* matching bytes become 0x00 */
    return (x - LSB) & ~x & MSB;         /* zero bytes borrow into bit 7 */
}

/* EMPTY is the only control byte with bit 7 set and bit 1 clear. */
static uint64_t match_empty(uint64_t w) {
    return w & ~(w << 6) & MSB;          /* move bit 1 up to bit 7 and mask it out */
}

/* The lowest set bit of a match mask is bit 8i + 7, so ctz / 8 = i. */
static int lowest_byte(uint64_t m) {
    return __builtin_ctzll(m) >> 3;      /* (a portable byte loop is used without GCC/Clang) */
}

static size_t probe_group(size_t home, size_t i, size_t ngroups) {
    return (home + i * (i + 1) / 2) & (ngroups - 1);   /* triangular step, masked: never past the end */
}
```

### Lookup

Inside a group every candidate is checked (`mm &= mm - 1` clears the lowest set bit), and only after both halves are checked does an EMPTY byte end the probe: the key may sit in the same group as the EMPTY slot.

```c
/* Slot of k, or SIZE_MAX when absent. */
static size_t find(const tl_map *m, uint64_t k, uint64_t h) {
    if (m->cap == 0) return SIZE_MAX;
    size_t ngroups = m->cap / GROUP;
    size_t home = (size_t)(h >> 7) & (ngroups - 1);   /* H1 picks the group */
    uint8_t h2 = (uint8_t)(h & 0x7F);                 /* H2 is the fingerprint */
    for (size_t i = 0; i < ngroups; i++) {            /* G steps visit every group once */
        size_t g = probe_group(home, i, ngroups);
        uint64_t w[2];
        load_group(m->ctrl, g, w);
        for (int half = 0; half < 2; half++) {
            for (uint64_t mm = match_byte(w[half], h2); mm; mm &= mm - 1) {
                size_t s = g * GROUP + (size_t)half * 8 + (size_t)lowest_byte(mm);
                if (m->slots[s].key == k) return s;   /* a fingerprint is only a candidate */
            }
        }
        if (match_empty(w[0]) | match_empty(w[1])) return SIZE_MAX; /* the probe ends here */
    }
    return SIZE_MAX;
}

/* First EMPTY or DELETED slot on k's probe (k known absent). */
static size_t find_free(const tl_map *m, uint64_t h) {
    /* ... same probe ... */
        for (int half = 0; half < 2; half++) {
            uint64_t free_ = w[half] & MSB; /* EMPTY or DELETED: bit 7 set */
            if (free_) return g * GROUP + (size_t)half * 8 + (size_t)lowest_byte(free_);
        }
    /* ... */
}
```

### Rebuild

Both arrays are allocated before `m` is touched. After that point nothing can fail, because `new_cap` holds every live key below 7/8 and `find_free` always finds a slot.

```c
static tl_status rehash(tl_map *m, size_t new_cap) {
    if (new_cap == SIZE_MAX) { /* ... */ return TL_ENOMEM; }
    uint8_t *ctrl = tl_alloc(new_cap, 64);
    slot *slots = tl_alloc(new_cap * sizeof(slot), 64);
    if (ctrl == NULL || slots == NULL) {
        tl_free(ctrl);                       /* free whichever one succeeded */
        tl_free(slots);
        tl_set_last_error("tl_map_put: out of memory growing the table");
        return TL_ENOMEM;                    /* m untouched */
    }
    memset(ctrl, CTRL_EMPTY, new_cap);
    tl_map old = *m;                         /* keep the old arrays until the move is done */
    m->ctrl = ctrl;
    m->slots = slots;
    m->cap = new_cap;
    m->used = 0;                             /* tombstones are not copied */
    for (size_t s = 0; s < old.cap; s++) {
        if (old.ctrl[s] & 0x80u) continue;   /* EMPTY or DELETED */
        uint64_t h = mix64(old.slots[s].key);
        size_t t = find_free(m, h);          /* keys are distinct: no find() needed */
        m->ctrl[t] = (uint8_t)(h & 0x7F);
        m->slots[t] = old.slots[s];
        m->used++;
    }
    tl_free(old.ctrl);
    tl_free(old.slots);
    return TL_OK;
}
```

`tl_map_create` uses the same `rehash` to allocate its first arrays, and frees the struct if that fails, so a failed create leaks nothing.

### Insert

The order matters: search the whole probe first (an existing key may lie past a tombstone), then pick the first free slot. Only turning an EMPTY slot FULL raises $u$, so reusing a tombstone never triggers growth.

```c
tl_status tl_map_put(tl_map *m, uint64_t k, uint64_t v) {
    if (m == NULL) { /* ... */ return TL_EINVAL; }
    uint64_t h = mix64(k);
    size_t s = find(m, k, h);
    if (s != SIZE_MAX) {                     /* present: overwrite in place, len unchanged */
        m->slots[s].val = v;
        return TL_OK;
    }
    s = find_free(m, h);
    if (s == SIZE_MAX || (m->ctrl[s] == CTRL_EMPTY && m->used + 1 > m->cap / 8 * 7)) {
        size_t want = cap_for(m->len + 1);   /* sized by live keys: same size if tombstones caused it */
        tl_status st = rehash(m, want > m->cap_min ? want : m->cap_min);
        if (st != TL_OK) return st;          /* old table intact */
        s = find_free(m, h);                 /* slots moved: probe again */
    }
    if (m->ctrl[s] == CTRL_EMPTY) m->used++; /* a reused tombstone is already counted */
    m->ctrl[s] = (uint8_t)(h & 0x7F);
    m->slots[s] = (slot){k, v};
    m->len++;
    return TL_OK;
}
```

### Delete

Delete is the EMPTY-in-group rule of step 6, applied to the group that holds the slot.

```c
int tl_map_del(tl_map *m, uint64_t k) {
    if (m == NULL) return 0;
    size_t s = find(m, k, mix64(k));
    if (s == SIZE_MAX) return 0;
    uint64_t w[2];
    load_group(m->ctrl, s / GROUP, w);
    if (match_empty(w[0]) | match_empty(w[1])) {
        m->ctrl[s] = CTRL_EMPTY;   /* no probe ever passed this group */
        m->used--;
    } else {
        m->ctrl[s] = CTRL_DELETED; /* probes may pass: leave a tombstone */
    }
    m->len--;
    return 1;
}
```

The full solution is [`swiss.c`](../../../../course/solutions/c/src/ds/swiss.c).

### Why this design

`get`, `put`, and `del` are expected $O(1)$: at load up to 7/8, most probes end in the home group, and each group step is two word loads and a few ALU operations. Space is 17 bytes per slot (one control byte plus a 16-byte pair), at least $8/7$ slots per key, plus the rebuild's temporary second copy. A rebuild is $O(\text{cap})$, amortized to $O(1)$ per insert because capacity doubles.

It helps to compare the design with its neighbors. Linear probing, as in the `c/02` drill, reads one key per slot, and a miss walks the whole occupied run; at high load runs grow long and tombstones lengthen them further. The Swiss table reads 16 control bytes per step and a key only on a fingerprint hit, so it can run at 7/8 load where linear probing would already be slow. [Robin Hood hashing](05-robin-hood-hash-map.md) bounds probe-length variance by moving keys on insert and backward-shifting on delete, so it needs no tombstones, but it still compares keys slot by slot. The Swiss table instead accepts tombstones and pays for them with occasional same-size rebuilds. A chained table would allocate per key, breaking the KV pool's rule that its index never allocates after sizing, and would chase pointers on every lookup.

Abseil compares 16 bytes in one SSE2 or NEON instruction. Two 64-bit words do the same in portable C at a small cost, with no intrinsics or per-architecture code. Two smaller tradeoffs are accepted on purpose. A false candidate from the borrow in `match_byte` costs one extra key compare and is never wrong. Groups are aligned to multiples of 16 rather than starting at any slot, which keeps the probe simple at the cost of slightly worse spreading than Abseil's sentinel-byte scheme.

On performance: at load 3/4 (3,071 keys in 4,096 slots) the solution answers misses about 2 to 3 times faster than the linear-probing baseline and hits at about the same speed (about 1.05 times). Misses are the prefix index's common case; a hit in either table is one key compare, so there is nothing to win there.

## Testing it and what can go wrong

A correct table must handle the cases below; each is a property the KV pool relies on.

| Case | Kind of check | Checks | Why it matters downstream |
|---|---|---|---|
| Fourteen fit, the fifteenth grows | unit, boundary | 14 keys in 16 slots with allocation failing, the 15th is `TL_ENOMEM`; 28 fit in 32 | the KV pool sizes the index for `n_blocks` once and must never allocate afterwards |
| Put, get, overwrite, delete | unit | overwrite keeps `len`; `get` with a NULL value pointer; delete twice | the map semantics before any probing detail |
| Every u64 is a key | boundary, regression | keys 0, `UINT64_MAX`, `0x80`, `0xFE` | block hashes are any u64 but 0; nothing is reserved |
| Keys sharing a control byte are told apart | unit | 14 keys with one $H_2$ in one group; a 15th with the same $H_2$ is absent | a fingerprint match is only a candidate |
| The probe continues past a full group | unit | 20 keys homed in group 0 of 32 slots; deletes in the full group keep the spilled keys reachable | tombstones where a probe may have passed |
| Re-put after a tombstone overwrites | unit | a spilled key put again after a delete in its home group overwrites, never duplicates | an insert searches the whole probe before reusing a tombstone |
| Churn in a group with room never allocates | fault | 10,000 delete-and-insert rounds at 13 keys in 16 slots with allocation failing | the EMPTY-in-group rule keeps $u$ from creeping up |
| Tombstone churn rehashes without growing | property | 20,000 rounds at 28 keys in 32 slots: the largest block ever requested stays the first one | rebuilds drop tombstones at the same size |
| A failed growth keeps the old table | fault | failing each allocation of a rebuild in turn: `TL_ENOMEM`, every key intact, no leak | the contract's "old table valid and unchanged" |
| Create and NULL arguments | boundary, fault | `TL_EINVAL` for a NULL out; create under allocation failure; NULL map calls | cleanup paths in the KV pool need no special cases |
| Agrees with linear probing | differential | $10^6$ seeded puts, gets, deletes agree with the `c/02` baseline on every result | the Swiss table is a map |

A compact model test suite for these cases (fourteen fit then grow, overwrite keeps `len`, shared fingerprints, spilled keys survive deletes, failed growth keeps keys, and a property test against a simple array model) is in [`test_swiss.c`](../../../../course/solutions/learner-tests/ds.02/c/tests/ds02-swiss/test_swiss.c).

The mistakes people make writing a Swiss table are remarkably consistent. Each one below has a symptom you can recognize and a line in the solution that prevents it.

| Pitfall | Symptom | How the solution avoids it |
|---|---|---|
| Writing EMPTY on every delete | keys that spilled past a full group vanish | `tl_map_del` writes EMPTY only when the slot's group still has an EMPTY slot (step 6) |
| Trusting a fingerprint match without comparing the key | `get` returns another key's value | `find` compares `slots[s].key == k` for every candidate |
| Ending a probe at a tombstone | spilled keys are "absent" after a delete | `find` stops only on `match_empty`, which never matches DELETED |
| Growing at load 7/8 instead of above it | the 14th key allocates; the KV pool's index grows when it must not | the test is `used + 1 > cap / 8 * 7`, strictly greater |
| Sizing `tl_map_create` one key short | `create(14)` gives 32 slots | `cap_for` loops while `n > cap / 8 * 7`, so 14 keys fit in 16 slots |
| Forgetting the mask on the probe step | reads past the control bytes (ASan) | `probe_group` masks with `ngroups - 1` |
| Freeing the old arrays before the new ones exist | a failed growth loses every key | `rehash` allocates both arrays first and returns before touching `m` on failure |
| Reusing the first tombstone without finishing the search | the key exists twice; a delete leaves a stale copy | `tl_map_put` calls `find` over the whole probe before `find_free` |
| Counting an overwrite as a new key | `len` drifts upward | the overwrite branch returns before `m->len++` |
| Checking only the first candidate of a group | keys sharing a fingerprint are lost | `find` loops over every set bit with `mm &= mm - 1` |
| Doubling on every rebuild | a table with constant live keys grows without bound | the rebuild size is `cap_for(len + 1)`, clamped to `cap_min` |
| Leaking the struct when the slot allocation fails | the allocator reports a live block | `tl_map_create` frees `m` when its first `rehash` fails |

> **Warning:** The two most dangerous mistakes are the ones that never crash: writing EMPTY on every delete, and reusing a tombstone before finishing the search. Both pass small tests and silently lose or duplicate keys once a group fills.

## Where this leads

The direct caller is the prefix index of the [Paged KV block pool](../../build-an-llm/p08-inference/08-paged-kv-block-pool.md): `tl_kv_register` puts (hash, id), `tl_kv_lookup` gets, and eviction deletes. In the engine, `--prefix-cache=hash` is this table behind that pool, and [Block manager with prefix cache](../../build-an-llm/p10-serving/04-block-manager-and-prefix-cache.md) benchmarks it against the [radix tree](07-radix-tree.md) of this chapter.

Production tables extend the same design:

- [Abseil `flat_hash_map`](https://github.com/abseil/abseil-cpp/blob/master/absl/container/internal/raw_hash_set.h) adds SSE2 and NEON group matches, a sentinel byte so groups may start at any slot, and in-place tombstone cleanup (`raw_hash_set.h`: `Group`, `DropDeletesWithoutResize`).
- [hashbrown](https://github.com/rust-lang/hashbrown), Rust's `HashMap`, is the same design in Rust, generic over keys (`src/raw/mod.rs`).
- [Folly F14](https://github.com/facebook/folly/blob/main/folly/container/detail/F14Table.h) uses 14-slot chunks with an overflow count instead of tombstones (`F14Table.h`).

### Further reading

- Abseil, [Swiss Tables design notes](https://abseil.io/about/design/swisstables).
- Matt Kulukundis, *Designing a Fast, Efficient, Cache-friendly Hash Table, Step by Step* (CppCon 2017).
- Knuth, *The Art of Computer Programming*, vol. 3, section 6.4.

## Key terms

- **Control byte**: one byte per slot holding EMPTY, DELETED, or the key's 7-bit fingerprint.
- **Fingerprint ($H_2$)**: the low 7 bits of the mixed hash, used to reject slots without reading their keys.
- **Home group**: the group of 16 slots where a key's probe starts, chosen by $H_1 \bmod G$.
- **Tombstone**: a DELETED control byte that a probe must walk past.
- **SWAR**: SIMD within a register; testing all bytes of a 64-bit word with a few integer operations.
- **Same-size rebuild**: a rehash that drops tombstones without growing, because the live keys still fit.

## Summary

- A Swiss table keeps one control byte per slot (7 bits of the key's hash, or a marker for EMPTY and DELETED) and compares 16 of them with a few word operations, so a probe reads a key only when its fingerprint matched.
- One group of 16 slots holds 14 keys at the maximum load of 7/8; the 15th key is the one that needs a bigger table.
- A delete leaves a tombstone only when a probe may have walked past the slot's group; in a group that still has an EMPTY slot it writes EMPTY, so churn at a steady size never allocates.
- Tombstones count toward the load; when they fill the table it is rebuilt at the same size, and it doubles only for live keys.
- A rebuild allocates the new arrays before touching the old ones, so a failed growth leaves every key in place.

## Review questions

1. Why can a delete write EMPTY when its group still has an EMPTY slot, but not otherwise?
   <details><summary>Answer</summary>A probe moves past a group only when that group has no EMPTY slot. A group gains EMPTY slots only at a rebuild (which resets every probe). So if the group has an EMPTY slot now, it had one ever since the last rebuild, and no key's probe ever continued past it: no key lives beyond it because of it, and writing EMPTY cannot cut any probe short. If the group is full, some key may have spilled past it, and EMPTY would end that key's probe early.</details>
2. `tl_map_put` calls `find` before `find_free`. What breaks if it used the first free slot it met and only then checked for the key?
   <details><summary>Answer</summary>The key may already sit further along its probe, past a tombstone left by an earlier delete. Writing it into the tombstone would store it twice; `len` would be wrong, and a later delete would remove only the first copy, leaving a stale value visible.</details>
3. Why is the growth test `used + 1 > cap / 8 * 7` applied only when the free slot is EMPTY?
   <details><summary>Answer</summary>Reusing a DELETED slot does not change $u$: the slot was already counted as used. Only EMPTY to FULL raises $u$, and $u$ is the quantity that must stay at or below 7/8 so that every probe meets an EMPTY slot quickly. Growing on a tombstone reuse would allocate when the load has not changed, which would break the steady-size churn case.</details>
4. Why is a rebuild sized by `cap_for(len + 1)` and not by doubling `cap`?
   <details><summary>Answer</summary>When the pressure came from tombstones, the live keys still fit in the current size; dropping the tombstones is enough. Doubling would make a table with a constant number of live keys grow without bound under delete-and-insert churn. Clamping to `cap_min` keeps the table at least as big as `tl_map_create` promised.</details>
5. `match_byte` can report a byte that does not equal $H_2$. Why is that acceptable, and why can it never report EMPTY or DELETED?
   <details><summary>Answer</summary>The borrow from a zero byte can propagate into the byte above it, so a FULL byte equal to $H_2 \oplus 1$ just above a real match can be flagged. `find` always compares the key, so a false candidate costs one compare and never a wrong answer. EMPTY and DELETED both have bit 7 set, and $H_2 < \texttt{0x80}$ keeps bit 7 of $x$ set for them, so $\lnot x$ clears bit 7 and the mask drops them.</details>
