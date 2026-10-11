/* Parity driver for `quant.fp8`, C side (the optional M09.7 mirror). One case
 * per stdin line, {"fmt": 4 | 5, "x_bits": [...], "codes": [...]}: encodes
 * each float32 (given as its bit pattern) with tl_f32_to_e4m3 or
 * tl_f32_to_e5m2 and decodes each code with the matching *_to_f32. Prints
 * {"codes": [...], "values_bits": [...]} with -1 for a NaN value, the same
 * shape as the Python (M09.4) driver. */
#include <math.h>
#include <string.h>

#include "tinyllm.h"
#include "ol_parity.h"

int main(void) {
    static char line[1 << 20];
    static double xd[4096], cd[4096];
    while (ssp_line(line, sizeof line)) {
        int64_t fmt = ssp_int(line, "fmt", -1);
        int64_t nx = ssp_doubles(line, "x_bits", xd, 4096);
        int64_t nc = ssp_doubles(line, "codes", cd, 4096);
        if ((fmt != 4 && fmt != 5) || nx < 0 || nc < 0 || nx > 4096 || nc > 4096) return 2;
        printf("{\"codes\": [");
        for (int64_t i = 0; i < nx; i++) {
            uint32_t b = (uint32_t)xd[i];
            float x;
            memcpy(&x, &b, sizeof x);
            unsigned c = fmt == 4 ? tl_f32_to_e4m3(x) : tl_f32_to_e5m2(x);
            printf(i ? ",%u" : "%u", c);
        }
        printf("], \"values_bits\": [");
        for (int64_t i = 0; i < nc; i++) {
            uint8_t c = (uint8_t)cd[i];
            float v = fmt == 4 ? tl_e4m3_to_f32(c) : tl_e5m2_to_f32(c);
            uint32_t b;
            memcpy(&b, &v, sizeof b);
            if (isnan(v)) printf(i ? ",-1" : "-1");
            else printf(i ? ",%u" : "%u", (unsigned)b);
        }
        puts("]}");
        fflush(stdout);
    }
    return 0;
}
