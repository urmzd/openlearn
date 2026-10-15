/* backprop_xor.c — a 2-layer MLP trained by backpropagation, from scratch,
 * solving the XOR that defeated the single perceptron (README §2-§3).
 *
 * Network:  x(2) -> [W1,b1] -> tanh -> h(2) -> [W2,b2] -> sigmoid -> y(1)
 * Loss:     binary cross-entropy.
 *
 * This is the whole training algorithm with no libraries. The backward pass is
 * the four chain-rule lines from README §2, specialized to this graph:
 *     dL/dz2 = (y - t)                       (sigmoid + BCE collapse to this)
 *     dL/dW2 = dL/dz2 · hᵀ ,  dL/db2 = dL/dz2
 *     dL/dh  = W2ᵀ · dL/dz2
 *     dL/dz1 = dL/dh ⊙ (1 - tanh²(z1))       (tanh' = 1 - tanh²)
 *     dL/dW1 = dL/dz1 · xᵀ ,  dL/db1 = dL/dz1
 *
 * Build & run:
 *   cc -O2 -o backprop_xor backprop_xor.c -lm && ./backprop_xor
 */
#include <math.h>
#include <stdio.h>

#define NIN 2   /* inputs       */
#define NH  4   /* hidden units (2 can solve XOR but stalls at the symmetric
                 * saddle for many inits; 4 + broken symmetry is reliable) */
#define NS  4   /* samples      */

static double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }

/* Tiny deterministic LCG so the run is reproducible without a fixed (and
 * possibly symmetric) hand init. Returns a weight in [-0.8, 0.8]. */
static unsigned long g_seed = 1234567u;
static double rand_w(void) {
    g_seed = g_seed * 1103515245u + 12345u;
    double u = ((g_seed >> 16) & 0x7fff) / 32767.0;  /* [0,1] */
    return 1.6 * u - 0.8;                              /* [-0.8, 0.8] */
}

int main(void) {
    double X[NS][NIN] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double T[NS]      = {0, 1, 1, 0};                 /* XOR targets */

    double W1[NH][NIN], b1[NH], W2[NH], b2 = 0.0;
    for (int j = 0; j < NH; j++) {
        for (int i = 0; i < NIN; i++) W1[j][i] = rand_w();
        b1[j] = 0.0;
        W2[j] = rand_w();
    }

    const double eta = 0.5;
    const int epochs = 20000;

    for (int e = 1; e <= epochs; e++) {
        double loss = 0.0;
        for (int s = 0; s < NS; s++) {
            /* ---- forward ---- */
            double z1[NH], h[NH];
            for (int j = 0; j < NH; j++) {
                double z = b1[j];
                for (int i = 0; i < NIN; i++) z += W1[j][i] * X[s][i];
                z1[j] = z;
                h[j]  = tanh(z);
            }
            double z2 = b2;
            for (int j = 0; j < NH; j++) z2 += W2[j] * h[j];
            double y = sigmoid(z2);

            loss += -(T[s] * log(y + 1e-12) + (1 - T[s]) * log(1 - y + 1e-12));

            /* ---- backward (chain rule) ---- */
            double dz2 = y - T[s];                    /* sigmoid + BCE */
            double dh[NH];
            for (int j = 0; j < NH; j++) dh[j] = W2[j] * dz2;

            /* ---- update output layer ---- */
            for (int j = 0; j < NH; j++) W2[j] -= eta * dz2 * h[j];
            b2 -= eta * dz2;

            /* ---- update hidden layer ---- */
            for (int j = 0; j < NH; j++) {
                double dz1 = dh[j] * (1.0 - h[j] * h[j]);   /* tanh' */
                for (int i = 0; i < NIN; i++) W1[j][i] -= eta * dz1 * X[s][i];
                b1[j] -= eta * dz1;
            }
        }
        if (e % 4000 == 0) printf("epoch %5d  loss %.5f\n", e, loss / NS);
    }

    /* ---- evaluate ---- */
    printf("\nlearned XOR:\n");
    for (int s = 0; s < NS; s++) {
        double h[NH];
        for (int j = 0; j < NH; j++) {
            double z = b1[j];
            for (int i = 0; i < NIN; i++) z += W1[j][i] * X[s][i];
            h[j] = tanh(z);
        }
        double z2 = b2;
        for (int j = 0; j < NH; j++) z2 += W2[j] * h[j];
        double y = sigmoid(z2);
        printf("  (%g,%g) -> %.3f  (target %g)\n", X[s][0], X[s][1], y, T[s]);
    }
    return 0;
}
