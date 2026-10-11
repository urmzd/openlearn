/* ol_invariance.h: bitwise batch and chunk invariance checks for C tests
 * (course/DESIGN.md 2.4, 5.11; L9.1 matmul, L9.2 softmax, L9.3 attention).
 *
 * Frozen helper, used with ol_test.h. A batch-invariant kernel gives each
 * row the same bits alone or inside any batch at any position; a
 * chunk-invariant one gives the same bits for every tile or chunk size.
 *
 *   // rows: computes `m` output rows of `out_len` floats from `m` input rows
 *   static void rows(void *ctx, const float *x, int64_t m, float *out) { ... }
 *
 *   OL_TEST(matmul_is_batch_invariant) {
 *       // WHY: ...  KIND: property
 *       OL_BATCH_INVARIANT(rows, &w, x, 16, K, N);   // 16 input rows of K, outputs of N
 *   }
 *
 * OL_BITS_EQ_F32(a, b, n) compares two float arrays bit for bit and reports
 * the first differing index with both values and the ULP distance. */
#ifndef OL_INVARIANCE_H
#define OL_INVARIANCE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ol_test.h"

static inline uint32_t ol__f32_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    return u;
}

/* -1 when equal, else the first differing index. */
static inline int64_t ol_bits_diff_f32(const float *a, const float *b, int64_t n) {
    for (int64_t i = 0; i < n; i++)
        if (ol__f32_bits(a[i]) != ol__f32_bits(b[i])) return i;
    return -1;
}

static inline int ol__bits_eq_f32(const float *a, const float *b, int64_t n, const char *what, const char *file, int line) {
    int64_t i = ol_bits_diff_f32(a, b, n);
    if (i < 0) return 1;
    int64_t ua = (int64_t)(int32_t)ol__f32_bits(a[i]), ub = (int64_t)(int32_t)ol__f32_bits(b[i]);
    char buf[512];
    snprintf(buf, sizeof buf, "%s: element %lld differs in its bits: %.9g vs %.9g (%lld ulp)", what, (long long)i,
             (double)a[i], (double)b[i], (long long)(ua > ub ? ua - ub : ub - ua));
    return ol__fail(file, line, buf);
}

#define OL_BITS_EQ_F32(a, b, n)                                                                    \
    do {                                                                                           \
        if (!ol__bits_eq_f32((a), (b), (n), "OL_BITS_EQ_F32(" #a ", " #b ")", __FILE__, __LINE__)) \
            return;                                                                                \
    } while (0)

typedef void (*ol_rows_fn)(void *ctx, const float *x, int64_t m, float *out);

/* Every row alone vs inside batches of 1..rows at every offset. 1 when invariant. */
static inline int ol__batch_invariant(ol_rows_fn fn, void *ctx, const float *x, int64_t rows, int64_t in_len,
                                      int64_t out_len, const char *file, int line) {
    float *alone = calloc((size_t)(rows * out_len + 1), sizeof *alone);
    float *batch = calloc((size_t)(rows * out_len + 1), sizeof *batch);
    if (!alone || !batch) {
        free(alone), free(batch);
        return ol__fail(file, line, "OL_BATCH_INVARIANT: out of memory");
    }
    for (int64_t i = 0; i < rows; i++) fn(ctx, x + i * in_len, 1, alone + i * out_len);
    int ok = 1;
    for (int64_t m = 1; m <= rows && ok; m++)
        for (int64_t off = 0; off + m <= rows && ok; off++) {
            fn(ctx, x + off * in_len, m, batch);
            for (int64_t j = 0; j < m && ok; j++) {
                char what[160];
                snprintf(what, sizeof what, "row %lld inside a batch of %lld at position %lld", (long long)(off + j),
                         (long long)m, (long long)j);
                ok = ol__bits_eq_f32(batch + j * out_len, alone + (off + j) * out_len, out_len, what, file, line);
            }
        }
    free(alone), free(batch);
    return ok;
}

#define OL_BATCH_INVARIANT(fn, ctx, x, rows, in_len, out_len)                                      \
    do {                                                                                           \
        if (!ol__batch_invariant((fn), (ctx), (x), (rows), (in_len), (out_len), __FILE__, __LINE__)) \
            return;                                                                                \
    } while (0)

typedef void (*ol_chunk_fn)(void *ctx, int64_t chunk, float *out);

/* fn(ctx, chunk, out) for every chunk size in chunks[0..n) gives the bits of chunks[0]. */
static inline int ol__chunk_invariant(ol_chunk_fn fn, void *ctx, const int64_t *chunks, int n, int64_t out_len,
                                      const char *file, int line) {
    float *want = calloc((size_t)out_len + 1, sizeof *want), *got = calloc((size_t)out_len + 1, sizeof *got);
    if (!want || !got) {
        free(want), free(got);
        return ol__fail(file, line, "OL_CHUNK_INVARIANT: out of memory");
    }
    fn(ctx, chunks[0], want);
    int ok = 1;
    for (int i = 1; i < n && ok; i++) {
        fn(ctx, chunks[i], got);
        char what[96];
        snprintf(what, sizeof what, "chunk size %lld vs %lld", (long long)chunks[i], (long long)chunks[0]);
        ok = ol__bits_eq_f32(got, want, out_len, what, file, line);
    }
    free(want), free(got);
    return ok;
}

#define OL_CHUNK_INVARIANT(fn, ctx, chunks, n, out_len)                                            \
    do {                                                                                           \
        if (!ol__chunk_invariant((fn), (ctx), (chunks), (n), (out_len), __FILE__, __LINE__)) return; \
    } while (0)

#endif /* OL_INVARIANCE_H */
