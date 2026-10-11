/* Parity driver for `matmul`: tl_matmul_f32 with alpha 1, beta 0, packed
 * row-major operands (lda = K, ldb = N, ldc = N), no transpose, serial. */
#include <stdlib.h>

#include "tinyllm.h"
#include "ol_parity.h"

int main(void) {
    static char line[1 << 20];
    while (ssp_line(line, sizeof line)) {
        int64_t m = ssp_int(line, "m", -1), k = ssp_int(line, "k", -1), n = ssp_int(line, "n", -1);
        if (m < 0 || k < 0 || n < 0) return 2;
        float *a = calloc((size_t)(m * k + 1), sizeof *a);
        float *b = calloc((size_t)(k * n + 1), sizeof *b);
        float *c = calloc((size_t)(m * n + 1), sizeof *c);
        if (!a || !b || !c) return 3;
        if (ssp_floats(line, "a", a, m * k) != m * k || ssp_floats(line, "b", b, k * n) != k * n) return 2;
        tl_status st = tl_matmul_f32(a, b, c, m, n, k, k, n, n, 1.0f, 0.0f, 0, NULL);
        if (st != TL_OK) {
            fprintf(stderr, "tl_matmul_f32: %s: %s\n", tl_status_str(st), tl_last_error());
            return 1;
        }
        ssp_print_floats(c, m * n);
        free(a);
        free(b);
        free(c);
    }
    return 0;
}
