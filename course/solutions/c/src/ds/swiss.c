/* c/src/ds/swiss.c (ds.02): a Swiss table from u64 to u64.
 * Contract: tinyllm/ds.h (ds.02 section). The worked baseline is the
 * linear-probing map of practice c/02; this file keeps its open addressing
 * and tombstones and changes how a probe looks at slots.
 *
 * Layout. cap slots, cap = 16 * 2^k (or 0 before the first key). Slots are
 * grouped in runs of 16. Each slot has one control byte:
 *
 *   EMPTY    0x80  never used since the last rehash
 *   DELETED  0xFE  a tombstone: a removed key that probes must walk past
 *   FULL     0x00 .. 0x7F, the 7 low bits of the key's hash (H2)
 *
 * A key's hash h = mix64(key) (the SplitMix64 finalizer) splits in two:
 * H1 = h >> 7 picks the home group, H2 = h & 0x7F goes in the control byte.
 * A probe visits groups home, home + 1, home + 3, home + 6, ... (triangular
 * steps, mod the group count), and inside a group compares all 16 control
 * bytes with H2 at once: two 64-bit words and a few bit operations (SWAR,
 * "SIMD within a register"). Only slots whose byte equals H2 have their key
 * compared, so a probe almost never reads a key that does not match.
 *
 * Load. used = live keys + tombstones. An insert that would turn an EMPTY
 * slot FULL while used == 7/8 * cap first rehashes into cap_for(len + 1)
 * slots (never fewer than the capacity create() gave): tombstones are
 * dropped, and the table doubles only when the live keys need it.
 *
 * Delete. A removed slot becomes EMPTY when its group still has an EMPTY
 * slot (no probe ever continued past such a group), else DELETED.
 */
#include <stdint.h>
#include <string.h>

#include "tinyllm/abi.h"
#include "tinyllm/ds.h"

#define GROUP 16
#define CTRL_EMPTY 0x80u
#define CTRL_DELETED 0xFEu
#define LSB 0x0101010101010101ull
#define MSB 0x8080808080808080ull

typedef struct {
    uint64_t key, val; /* side by side: a hit reads one cache line */
} slot;

struct tl_map {
    uint8_t *ctrl; /* cap control bytes */
    slot *slots;   /* cap slots */
    size_t cap;     /* 0 or 16 * 2^k */
    size_t cap_min; /* the capacity create() sized for its hint */
    size_t len;     /* live keys */
    size_t used;    /* live keys + tombstones */
};

/* The SplitMix64 finalizer: every input bit reaches every output bit. */
static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return x;
}

/* Smallest legal capacity holding n keys at load <= 7/8; 0 for n == 0;
 * SIZE_MAX when it would overflow. */
static size_t cap_for(size_t n) {
    if (n == 0) return 0;
    size_t cap = GROUP;
    while (n > cap / 8 * 7) {
        if (cap > SIZE_MAX / 2 / (1 + sizeof(slot))) return SIZE_MAX;
        cap *= 2;
    }
    return cap;
}

/* The 16 control bytes of group g as two words; byte i of the group is
 * byte i % 8 of word i / 8, counting from the least significant byte, on
 * any host (memcpy, then a byte swap on a big-endian host). */
static void load_group(const uint8_t *ctrl, size_t g, uint64_t w[2]) {
    memcpy(w, ctrl + g * GROUP, GROUP);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    w[0] = __builtin_bswap64(w[0]);
    w[1] = __builtin_bswap64(w[1]);
#endif
}

/* Bit 7 of byte i is set when byte i of w equals b (b < 0x80). A FULL
 * byte equal to b ^ 1 right above a match can also be reported (a borrow
 * from the subtraction), so callers always compare the key itself. EMPTY
 * and DELETED bytes are never reported: their bit 7 is set, so ~x clears
 * it. */
static uint64_t match_byte(uint64_t w, uint8_t b) {
    uint64_t x = w ^ (LSB * b);
    return (x - LSB) & ~x & MSB;
}

/* Bit 7 of byte i is set when byte i of w is EMPTY (0x80): bit 7 set and
 * bit 1 clear. DELETED (0xFE) has bit 1 set; FULL has bit 7 clear. */
static uint64_t match_empty(uint64_t w) {
    return w & ~(w << 6) & MSB;
}

/* Index of the lowest reported byte of a non-zero match mask: its lowest
 * set bit is bit 8i + 7. */
static int lowest_byte(uint64_t m) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(m) >> 3;
#else
    int i = 0;
    while (!(m & 0x80u)) {
        m >>= 8;
        i++;
    }
    return i;
#endif
}

/* The group visited at step i of a probe that starts at group home. */
static size_t probe_group(size_t home, size_t i, size_t ngroups) {
    return (home + i * (i + 1) / 2) & (ngroups - 1);
}

/* Slot of k, or SIZE_MAX when absent. */
static size_t find(const tl_map *m, uint64_t k, uint64_t h) {
    if (m->cap == 0) return SIZE_MAX;
    size_t ngroups = m->cap / GROUP;
    size_t home = (size_t)(h >> 7) & (ngroups - 1);
    uint8_t h2 = (uint8_t)(h & 0x7F);
    for (size_t i = 0; i < ngroups; i++) {
        size_t g = probe_group(home, i, ngroups);
        uint64_t w[2];
        load_group(m->ctrl, g, w);
        for (int half = 0; half < 2; half++) {
            for (uint64_t mm = match_byte(w[half], h2); mm; mm &= mm - 1) {
                size_t s = g * GROUP + (size_t)half * 8 + (size_t)lowest_byte(mm);
                if (m->slots[s].key == k) return s;
            }
        }
        if (match_empty(w[0]) | match_empty(w[1])) return SIZE_MAX; /* the probe ends here */
    }
    return SIZE_MAX;
}

/* First EMPTY or DELETED slot on k's probe (k known absent), or SIZE_MAX
 * when every slot is FULL. */
static size_t find_free(const tl_map *m, uint64_t h) {
    if (m->cap == 0) return SIZE_MAX;
    size_t ngroups = m->cap / GROUP;
    size_t home = (size_t)(h >> 7) & (ngroups - 1);
    for (size_t i = 0; i < ngroups; i++) {
        size_t g = probe_group(home, i, ngroups);
        uint64_t w[2];
        load_group(m->ctrl, g, w);
        for (int half = 0; half < 2; half++) {
            uint64_t free_ = w[half] & MSB; /* EMPTY or DELETED: bit 7 set */
            if (free_) return g * GROUP + (size_t)half * 8 + (size_t)lowest_byte(free_);
        }
    }
    return SIZE_MAX;
}

/* Rebuild into new_cap slots. Allocates the new arrays first, so a failure
 * leaves m exactly as it was. */
static tl_status rehash(tl_map *m, size_t new_cap) {
    if (new_cap == SIZE_MAX) {
        tl_set_last_error("tl_map_put: table size overflows");
        return TL_ENOMEM;
    }
    uint8_t *ctrl = tl_alloc(new_cap, 64);
    slot *slots = tl_alloc(new_cap * sizeof(slot), 64);
    if (ctrl == NULL || slots == NULL) {
        tl_free(ctrl);
        tl_free(slots);
        tl_set_last_error("tl_map_put: out of memory growing the table");
        return TL_ENOMEM;
    }
    memset(ctrl, CTRL_EMPTY, new_cap);
    tl_map old = *m;
    m->ctrl = ctrl;
    m->slots = slots;
    m->cap = new_cap;
    m->used = 0;
    for (size_t s = 0; s < old.cap; s++) {
        if (old.ctrl[s] & 0x80u) continue; /* EMPTY or DELETED */
        uint64_t h = mix64(old.slots[s].key);
        size_t t = find_free(m, h);
        m->ctrl[t] = (uint8_t)(h & 0x7F);
        m->slots[t] = old.slots[s];
        m->used++;
    }
    tl_free(old.ctrl);
    tl_free(old.slots);
    return TL_OK;
}

tl_status tl_map_create(size_t hint, tl_map **out) {
    if (out == NULL) {
        tl_set_last_error("tl_map_create: out is NULL");
        return TL_EINVAL;
    }
    *out = NULL;
    size_t cap = cap_for(hint);
    if (cap == SIZE_MAX) {
        tl_set_last_error("tl_map_create: hint too large");
        return TL_EINVAL;
    }
    tl_map *m = tl_alloc(sizeof *m, _Alignof(tl_map));
    if (m == NULL) {
        tl_set_last_error("tl_map_create: out of memory");
        return TL_ENOMEM;
    }
    memset(m, 0, sizeof *m);
    m->cap_min = cap;
    if (cap > 0 && rehash(m, cap) != TL_OK) {
        tl_free(m);
        tl_set_last_error("tl_map_create: out of memory");
        return TL_ENOMEM;
    }
    *out = m;
    return TL_OK;
}

tl_status tl_map_put(tl_map *m, uint64_t k, uint64_t v) {
    if (m == NULL) {
        tl_set_last_error("tl_map_put: map is NULL");
        return TL_EINVAL;
    }
    uint64_t h = mix64(k);
    size_t s = find(m, k, h);
    if (s != SIZE_MAX) { /* present: overwrite in place */
        m->slots[s].val = v;
        return TL_OK;
    }
    s = find_free(m, h);
    if (s == SIZE_MAX || (m->ctrl[s] == CTRL_EMPTY && m->used + 1 > m->cap / 8 * 7)) {
        size_t want = cap_for(m->len + 1);
        tl_status st = rehash(m, want > m->cap_min ? want : m->cap_min);
        if (st != TL_OK) return st;
        s = find_free(m, h);
    }
    if (m->ctrl[s] == CTRL_EMPTY) m->used++; /* a reused tombstone is already counted */
    m->ctrl[s] = (uint8_t)(h & 0x7F);
    m->slots[s] = (slot){k, v};
    m->len++;
    return TL_OK;
}

int tl_map_get(const tl_map *m, uint64_t k, uint64_t *v) {
    if (m == NULL) return 0;
    size_t s = find(m, k, mix64(k));
    if (s == SIZE_MAX) return 0;
    if (v != NULL) *v = m->slots[s].val;
    return 1;
}

int tl_map_del(tl_map *m, uint64_t k) {
    if (m == NULL) return 0;
    size_t s = find(m, k, mix64(k));
    if (s == SIZE_MAX) return 0;
    uint64_t w[2];
    load_group(m->ctrl, s / GROUP, w);
    if (match_empty(w[0]) | match_empty(w[1])) {
        m->ctrl[s] = CTRL_EMPTY; /* no probe ever passed this group */
        m->used--;
    } else {
        m->ctrl[s] = CTRL_DELETED; /* probes may pass: leave a tombstone */
    }
    m->len--;
    return 1;
}

size_t tl_map_len(const tl_map *m) {
    return m == NULL ? 0 : m->len;
}

void tl_map_destroy(tl_map *m) {
    if (m == NULL) return;
    tl_free(m->ctrl);
    tl_free(m->slots);
    tl_free(m);
}
