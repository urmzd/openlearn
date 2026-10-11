/* Parity driver for `kv.wire.v1` (rt.04, C): one JSON case per stdin line
 * ({B, L, H, D, tokens, values}), one {"hex": envelope} per stdout line.
 *
 * The case is a sequence: tokens cut into blocks of B, and values its K and
 * V as f16 bit patterns, row-major [position][layer][K, V][head][dim]. The
 * driver builds the blocks the way an engine would (tl_kv_alloc, writes
 * through tl_kv_block_ptr, tl_kv_set_fill, tl_kv_register under the chained
 * tl_kv_block_hash for every full block), exports them in order, imports
 * that envelope into a second pool (tl_kv_import), and prints the second
 * pool's export: the bytes compare with the Python transcription of
 * formats/kv-block.md (and the Rust writer of L10.6) only if both the
 * writer and the reader of rt.04 are exact. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "tinyllm.h"
#include "ol_parity.h"

#define MAXV (1 << 16)

int main(void) {
    static char line[1 << 20];
    static double tok_d[MAXV], val_d[MAXV];
    while (ssp_line(line, sizeof line)) {
        uint32_t B = (uint32_t)ssp_int(line, "B", 0), L = (uint32_t)ssp_int(line, "L", 0);
        uint32_t H = (uint32_t)ssp_int(line, "H", 0), D = (uint32_t)ssp_int(line, "D", 0);
        int64_t n = ssp_doubles(line, "tokens", tok_d, MAXV);
        int64_t nv = ssp_doubles(line, "values", val_d, MAXV);
        if (B == 0 || n <= 0 || n > MAXV || nv != n * L * 2 * H * D) {
            fprintf(stderr, "kv_wire_v1: bad case\n");
            return 2;
        }
        uint32_t nb = (uint32_t)((n + B - 1) / B);
        tl_kv_cfg cfg = {nb, B, L, H, D, TL_F16, TL_KV_FORMAT_V1};
        tl_kv_pool *p = NULL;
        uint32_t *ids = malloc(sizeof(uint32_t) * nb);
        uint32_t *toks = malloc(sizeof(uint32_t) * (size_t)n);
        if (!ids || !toks || tl_kv_pool_create(&cfg, &p) != TL_OK || tl_kv_alloc(p, nb, ids) != TL_OK) {
            fprintf(stderr, "kv_wire_v1: %s\n", tl_last_error());
            return 2;
        }
        for (int64_t i = 0; i < n; i++) toks[i] = (uint32_t)tok_d[i];
        for (int64_t pos = 0; pos < n; pos++)
            for (uint32_t l = 0; l < L; l++)
                for (int kv = 0; kv < 2; kv++) {
                    uint16_t *slab = tl_kv_block_ptr(p, ids[pos / B], l, kv);
                    for (uint32_t h = 0; h < H; h++)
                        for (uint32_t d = 0; d < D; d++) {
                            size_t src = ((((size_t)pos * L + l) * 2 + (size_t)kv) * H + h) * D + d;
                            slab[((size_t)h * B + (size_t)(pos % B)) * D + d] = (uint16_t)val_d[src];
                        }
                }
        uint64_t parent = 0;
        for (uint32_t b = 0; b < nb; b++) {
            uint32_t fill = (uint32_t)((int64_t)(b + 1) * B <= n ? B : n - (int64_t)b * B);
            tl_kv_set_fill(p, ids[b], fill);
            if (fill == B) {
                parent = tl_kv_block_hash(parent, toks + (size_t)b * B, B);
                tl_kv_register(p, ids[b], parent);
            }
        }
        size_t need = tl_kv_export_bytes(p, nb), written = 0;
        unsigned char *first = malloc(need), *buf = malloc(need);
        tl_kv_pool *q = NULL;
        if (!first || !buf || tl_kv_export(p, ids, nb, first, need, &written) != TL_OK ||
            tl_kv_pool_create(&cfg, &q) != TL_OK || tl_kv_import(q, first, written, ids) != TL_OK ||
            tl_kv_export(q, ids, nb, buf, need, &written) != TL_OK) {
            fprintf(stderr, "kv_wire_v1: export/import: %s\n", tl_last_error());
            return 2;
        }
        printf("{\"hex\":\"");
        for (size_t i = 0; i < written; i++) printf("%02x", buf[i]);
        printf("\"}\n");
        fflush(stdout);
        free(first);
        free(buf);
        free(ids);
        free(toks);
        tl_kv_pool_destroy(q);
        tl_kv_pool_destroy(p);
    }
    return 0;
}
