/* Parity driver for `softmax.online` (L9.2): one case per stdin line,
 * {"rows", "cols", "online", "x"}; prints the rows x cols result as one flat
 * JSON array. online 0 runs tl_softmax_f32, 1 runs tl_softmax_online_f32. */
#include <stdlib.h>

#include "tinyllm.h"
#include "ol_parity.h"

int main(void) {
    static char line[1 << 20];
    while (ssp_line(line, sizeof line)) {
        int64_t rows = ssp_int(line, "rows", -1), cols = ssp_int(line, "cols", -1);
        int64_t online = ssp_int(line, "online", 0);
        if (rows < 0 || cols < 0) return 2;
        float *x = calloc((size_t)(rows * cols + 1), sizeof *x);
        float *y = calloc((size_t)(rows * cols + 1), sizeof *y);
        if (!x || !y) return 3;
        if (ssp_floats(line, "x", x, rows * cols) != rows * cols) return 2;
        tl_status st = online ? tl_softmax_online_f32(x, y, rows, cols) : tl_softmax_f32(x, y, rows, cols);
        if (st != TL_OK) {
            fprintf(stderr, "softmax: %s: %s\n", tl_status_str(st), tl_last_error());
            return 1;
        }
        ssp_print_floats(y, rows * cols);
        free(x);
        free(y);
    }
    return 0;
}
