/* ol_prop.h: seeded property tests for C (course/DESIGN.md 5.9).
 *
 * Frozen helper, used with ol_test.h. A property is a function of a
 * generator and a size; OL_CHECK_PROP runs it for `cases` seeds derived from
 * OL_SEED (the harness's single seed) and, on the first failure, shrinks by
 * halving the size while the property still fails at that seed, then reports
 * the smallest failing size and the seed that reproduces it.
 *
 *   static int sum_is_order_free(ol_gen *g, int size) {
 *       float x[64]; int n = size % 64;
 *       for (int i = 0; i < n; i++) x[i] = ol_gen_f32(g, -1, 1);
 *       ...
 *       return ok;                          // nonzero: the property holds
 *   }
 *
 *   OL_TEST(sum_property) {
 *       // WHY: ...  KIND: property
 *       OL_CHECK_PROP(sum_is_order_free, 200, 64);   // cases, max size
 *   }
 *
 * The generator is SplitMix64 (no global state): every case is reproducible
 * from the printed seed alone.
 */
#ifndef OL_PROP_H
#define OL_PROP_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "ol_test.h"

typedef struct {
    uint64_t s;
} ol_gen;

static inline uint64_t ol_gen_u64(ol_gen *g) {
    uint64_t z = (g->s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
static inline uint32_t ol_gen_u32(ol_gen *g) { return (uint32_t)(ol_gen_u64(g) >> 32); }
/* Uniform integer in [lo, hi]. */
static inline int64_t ol_gen_int(ol_gen *g, int64_t lo, int64_t hi) {
    uint64_t span = (uint64_t)(hi - lo) + 1u;
    return lo + (int64_t)(span ? ol_gen_u64(g) % span : ol_gen_u64(g));
}
/* Uniform double in [lo, hi), 53 random bits. */
static inline double ol_gen_f64(ol_gen *g, double lo, double hi) {
    return lo + (hi - lo) * ((double)(ol_gen_u64(g) >> 11) * (1.0 / 9007199254740992.0));
}
static inline float ol_gen_f32(ol_gen *g, float lo, float hi) { return (float)ol_gen_f64(g, lo, hi); }

static inline uint64_t ol_prop_base_seed(void) {
    const char *s = getenv("OL_SEED");
    return s ? (uint64_t)strtoull(s, NULL, 10) : 0u;
}

/* Returns 1 when the property held for every case, else 0 (after printing). */
static inline int ol_prop_run(int (*prop)(ol_gen *, int), const char *name, int cases, int max_size) {
    uint64_t base = ol_prop_base_seed() * 0x100000001B3ull + 0xcbf29ce484222325ull;
    for (int c = 0; c < cases; c++) {
        uint64_t seed = base + (uint64_t)c * 0x9E3779B97F4A7C15ull;
        int size = max_size > 0 ? (int)(1 + (uint64_t)c * (uint64_t)max_size / (uint64_t)(cases > 1 ? cases - 1 : 1)) : 0;
        if (size > max_size) size = max_size;
        ol_gen g = {seed};
        if (prop(&g, size)) continue;
        int smallest = size;
        for (int s = size / 2; s >= 0; s /= 2) {
            ol_gen h = {seed};
            if (prop(&h, s)) break;
            smallest = s;
            if (s == 0) break;
        }
        printf("    property %s failed: case %d, seed %llu, smallest failing size %d\n", name, c,
               (unsigned long long)seed, smallest);
        return 0;
    }
    return 1;
}

#define OL_CHECK_PROP(prop, cases, max_size)                                                       \
    do {                                                                                           \
        if (!ol_prop_run((prop), #prop, (cases), (max_size))) {                                    \
            ol__fail(__FILE__, __LINE__, "OL_CHECK_PROP(" #prop ")");                              \
            return;                                                                                \
        }                                                                                          \
    } while (0)

#endif /* OL_PROP_H */
