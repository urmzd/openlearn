/* ds.02 bench (`ol bench ds.02`): tl_map against the practice c/02
 * linear-probing baseline (_baseline_lp.h) at the same load, 3/4, the
 * highest the baseline allows: 3,071 keys, which both tables hold in 4,096
 * slots (the size of an engine's prefix index).
 *
 * Metrics (ns per lookup and the baseline / tl_map ratios):
 *   hit_speedup   lookups of present keys
 *   miss_speedup  lookups of absent keys: the common case for the prefix
 *                 index, where most block hashes of a new prompt miss
 * The budget is on miss_speedup (DEVIATIONS B82-04): a miss in linear
 * probing walks the whole run to an empty slot, while the Swiss table
 * stops at the first group with an EMPTY control byte after one 16-byte
 * scan. Hits cost about the same in both (one compare each).
 */
#include "ol_bench.h" /* first: it sets the POSIX feature macros clock_gettime needs */
#include <stdint.h>
#include <stdlib.h>

#include "tinyllm.h"
#include "../_baseline_lp.h"

#define N_KEYS 3071u
#define N_PROBES (1u << 16)

static uint64_t *keys, *order, *absent;
static tl_map *swiss;
static lp_map base;
static volatile uint64_t sink;

static void swiss_hits(void *ctx) {
    (void)ctx;
    uint64_t s = 0, v;
    for (uint32_t i = 0; i < N_PROBES; i++)
        if (tl_map_get(swiss, keys[order[i]], &v)) s += v;
    sink = s;
}

static void base_hits(void *ctx) {
    (void)ctx;
    uint64_t s = 0, v;
    for (uint32_t i = 0; i < N_PROBES; i++)
        if (lp_get(&base, keys[order[i]], &v)) s += v;
    sink = s;
}

static void swiss_misses(void *ctx) {
    (void)ctx;
    uint64_t s = 0, v;
    for (uint32_t i = 0; i < N_PROBES; i++) s += (uint64_t)tl_map_get(swiss, absent[i], &v);
    sink = s;
}

static void base_misses(void *ctx) {
    (void)ctx;
    uint64_t s = 0, v;
    for (uint32_t i = 0; i < N_PROBES; i++) s += (uint64_t)lp_get(&base, absent[i], &v);
    sink = s;
}

int main(void) {
    keys = malloc(sizeof *keys * N_KEYS);
    order = malloc(sizeof *order * N_PROBES);
    absent = malloc(sizeof *absent * N_PROBES);
    if (!keys || !order || !absent) return 1;
    uint64_t x = 0;
    for (uint32_t i = 0; i < N_KEYS; i++) keys[i] = lp_mix64(x += 0x9E3779B97F4A7C15ull) & ~1ull; /* even */
    for (uint32_t i = 0; i < N_PROBES; i++) {
        order[i] = lp_mix64(i) % N_KEYS;
        absent[i] = lp_mix64(i + 0x5555) | 1; /* odd: never a key */
    }
    if (tl_map_create(N_KEYS, &swiss) != TL_OK) return 1;
    lp_init(&base);
    for (uint32_t i = 0; i < N_KEYS; i++) {
        if (tl_map_put(swiss, keys[i], i) != TL_OK) return 1;
        if (lp_put(&base, keys[i], i) != 0) return 1;
    }
    double sh = ol_bench_best(swiss_hits, NULL, 5, 0.3) / N_PROBES;
    double bh = ol_bench_best(base_hits, NULL, 5, 0.3) / N_PROBES;
    double sm = ol_bench_best(swiss_misses, NULL, 5, 0.3) / N_PROBES;
    double bm = ol_bench_best(base_misses, NULL, 5, 0.3) / N_PROBES;
    ol_bench_metric("swiss_hit_ns", sh * 1e9);
    ol_bench_metric("baseline_hit_ns", bh * 1e9);
    ol_bench_metric("swiss_miss_ns", sm * 1e9);
    ol_bench_metric("baseline_miss_ns", bm * 1e9);
    ol_bench_metric("hit_speedup", bh / sh);
    ol_bench_metric("miss_speedup", bm / sm);
    tl_map_destroy(swiss);
    lp_free(&base);
    free(keys);
    free(order);
    free(absent);
    return ol_bench_done();
}
