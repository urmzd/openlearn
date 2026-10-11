/* Parity driver: tl_demo_sum_f32 (rt.91). */
#include "tinyllm.h"
#include "ol_parity.h"

int main(void) {
    static char line[1 << 16];
    while (ssp_line(line, sizeof line)) {
        float xs[256];
        int64_t n = ssp_floats(line, "xs", xs, 256);
        float y = 0.0f;
        if (n < 0 || tl_demo_sum_f32(xs, n, &y) != TL_OK) return 1;
        printf("%.9g\n", (double)y);
        fflush(stdout);
    }
    return 0;
}
