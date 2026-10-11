/* Parity driver for `quant.int4`, C side (L9.5). One case per stdin line,
 * {"rows", "cols", "group", "packed", "scales_f16", ...}. The C library
 * consumes the int4 layout and cannot produce it, so the driver decodes:
 * it dequantizes every weight with tl_matmul_q4_f32 on identity
 * activations (x = I_cols, so y[i][r] = q(r, i) * s exactly), recovers
 * q = y / s (exact; q = 0 where s = 0), and packs q again by
 * formats/safetensors.md. It prints {"packed": [...], "scales_f16": [...]},
 * the same shape as the Python (L8.5) driver, so a kernel that reads a
 * nibble, a sign, or a scale differently from L8.5's encoder fails. */
#include <stdlib.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_parity.h"

int main(void) {
    static char line[1 << 20];
    static double pd[4096], sd[4096];
    while (ssp_line(line, sizeof line)) {
        int64_t n = ssp_int(line, "rows", -1), k = ssp_int(line, "cols", -1), g = ssp_int(line, "group", -1);
        if (n <= 0 || k <= 0 || g <= 0 || k % g != 0 || n * k / 2 > 4096) return 2;
        if (ssp_doubles(line, "packed", pd, 4096) != n * k / 2) return 2;
        if (ssp_doubles(line, "scales_f16", sd, 4096) != n * (k / g)) return 2;
        uint8_t *q = malloc((size_t)(n * k / 2));
        uint16_t *s = malloc(sizeof *s * (size_t)(n * (k / g)));
        float *x = calloc((size_t)(k * k), sizeof *x), *y = malloc(sizeof *y * (size_t)(k * n));
        if (!q || !s || !x || !y) return 3;
        for (int64_t i = 0; i < n * k / 2; i++) q[i] = (uint8_t)pd[i];
        for (int64_t i = 0; i < n * (k / g); i++) s[i] = (uint16_t)sd[i];
        for (int64_t i = 0; i < k; i++) x[i * k + i] = 1.0f;
        tl_status st = tl_matmul_q4_f32(x, q, s, y, k, n, k, g, NULL);
        if (st != TL_OK) {
            fprintf(stderr, "tl_matmul_q4_f32: %s: %s\n", tl_status_str(st), tl_last_error());
            return 1;
        }
        printf("{\"packed\": [");
        for (int64_t r = 0; r < n; r++)
            for (int64_t b = 0; b < k / 2; b++) {
                int nib[2];
                for (int h = 0; h < 2; h++) {
                    int64_t i = 2 * b + h;
                    float sc = tl_f16_to_f32(s[r * (k / g) + i / g]);
                    int v = sc == 0.0f ? 0 : (int)(y[i * n + r] / sc);
                    nib[h] = v & 0xF;
                }
                printf(r || b ? ",%d" : "%d", nib[0] | (nib[1] << 4));
            }
        printf("], \"scales_f16\": [");
        for (int64_t i = 0; i < n * (k / g); i++) printf(i ? ",%u" : "%u", (unsigned)s[i]);
        puts("]}");
        fflush(stdout);
        free(q), free(s), free(x), free(y);
    }
    return 0;
}
