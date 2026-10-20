/* conv2d.c — the 2D convolution (cross-correlation) at the heart of every CNN
 * (README §4), in plain C with no libraries.
 *
 * Math:  S(i,j) = Σ_u Σ_v X(i+u, j+v) · K(u,v) + b
 * Knobs: stride, "valid" (no) padding. Output size for HxW input, kxk kernel:
 *        out = floor((H - k) / stride) + 1.
 *
 * This is the reference semantics; production kernels (im2col+GEMM, Winograd,
 * FFT, cuDNN) compute the SAME sum far faster — see conv2d.cu for the GPU map.
 *
 * Build & run:
 *   cc -O2 -o conv2d conv2d.c && ./conv2d
 */
#include <stdio.h>

#define H 5
#define W 5
#define K 3
#define STRIDE 1
#define OH (((H) - (K)) / (STRIDE) + 1)
#define OW (((W) - (K)) / (STRIDE) + 1)

static void conv2d(const double in[H][W], const double ker[K][K], double bias,
                   double out[OH][OW]) {
    for (int i = 0; i < OH; i++) {
        for (int j = 0; j < OW; j++) {
            double acc = bias;
            for (int u = 0; u < K; u++)
                for (int v = 0; v < K; v++)
                    acc += in[i * STRIDE + u][j * STRIDE + v] * ker[u][v];
            out[i][j] = acc;
        }
    }
}

int main(void) {
    /* A 5x5 image with a bright vertical edge in the middle columns. */
    double img[H][W] = {
        {0, 0, 1, 1, 1}, {0, 0, 1, 1, 1}, {0, 0, 1, 1, 1},
        {0, 0, 1, 1, 1}, {0, 0, 1, 1, 1},
    };
    /* Sobel-style vertical-edge detector: fires on left-to-right intensity jump. */
    double sobel_x[K][K] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

    double out[OH][OW];
    conv2d(img, sobel_x, 0.0, out);

    printf("vertical-edge feature map (%dx%d):\n", OH, OW);
    for (int i = 0; i < OH; i++) {
        for (int j = 0; j < OW; j++) printf("%6.1f", out[i][j]);
        putchar('\n');
    }
    /* The nonzero column marks where the edge is — the filter is translation-
     * equivariant: it would fire at the same offset wherever the edge moved. */
    return 0;
}
