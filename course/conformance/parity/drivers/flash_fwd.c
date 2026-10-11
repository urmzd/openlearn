/* Parity driver for `flash.fwd` (L9.3): one case per stdin line,
 * {"B", "H", "Hkv", "Tq", "Tk", "D", "scale", "q_offset", "causal",
 * "window", "q", "k", "v"}; prints {"o": [B*H*Tq*D], "lse": [B*H*Tq]}
 * from tl_flash_attn_fwd_f32 with the default tiles, a private arena, and
 * no pool. */
#include <stdlib.h>

#include "tinyllm.h"
#include "ol_parity.h"

static void print_list(const char *key, const float *x, int64_t n, const char *end) {
    printf("\"%s\": [", key);
    for (int64_t i = 0; i < n; i++) printf(i ? ",%.9g" : "%.9g", (double)x[i]);
    printf("]%s", end);
}

int main(void) {
    static char line[1 << 20];
    while (ssp_line(line, sizeof line)) {
        int64_t B = ssp_int(line, "B", -1), H = ssp_int(line, "H", -1), Hkv = ssp_int(line, "Hkv", -1);
        int64_t Tq = ssp_int(line, "Tq", -1), Tk = ssp_int(line, "Tk", -1), D = ssp_int(line, "D", -1);
        int64_t q_offset = ssp_int(line, "q_offset", 0), window = ssp_int(line, "window", 0);
        int causal = (int)ssp_int(line, "causal", 1);
        float scale = (float)ssp_double(line, "scale", 1.0);
        if (B < 1 || H < 1 || Hkv < 1 || Tq < 1 || Tk < 1 || D < 1) return 2;
        int64_t nq = B * H * Tq * D, nk = B * Hkv * Tk * D;
        float *q = malloc(sizeof *q * (size_t)nq), *k = malloc(sizeof *k * (size_t)nk);
        float *v = malloc(sizeof *v * (size_t)nk), *o = malloc(sizeof *o * (size_t)nq);
        float *lse = malloc(sizeof *lse * (size_t)(B * H * Tq));
        if (!q || !k || !v || !o || !lse) return 3;
        if (ssp_floats(line, "q", q, nq) != nq || ssp_floats(line, "k", k, nk) != nk || ssp_floats(line, "v", v, nk) != nk)
            return 2;
        tl_status st = tl_flash_attn_fwd_f32(q, k, v, o, lse, B, H, Hkv, Tq, Tk, D, scale, q_offset, causal, window,
                                             NULL, 0, 0, NULL, NULL);
        if (st != TL_OK) {
            fprintf(stderr, "tl_flash_attn_fwd_f32: %s: %s\n", tl_status_str(st), tl_last_error());
            return 1;
        }
        putchar('{');
        print_list("o", o, nq, ", ");
        print_list("lse", lse, B * H * Tq, "");
        puts("}");
        fflush(stdout);
        free(q), free(k), free(v), free(o), free(lse);
    }
    return 0;
}
