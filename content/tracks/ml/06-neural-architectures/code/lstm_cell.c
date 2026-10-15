/* lstm_cell.c — one LSTM timestep, all six equations, from scratch (README §7).
 *
 *   f_t = σ(W_f·[h,x] + b_f)          forget gate
 *   i_t = σ(W_i·[h,x] + b_i)          input gate
 *   g_t = tanh(W_g·[h,x] + b_g)       candidate
 *   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t   cell state  (ADDITIVE — the key)
 *   o_t = σ(W_o·[h,x] + b_o)          output gate
 *   h_t = o_t ⊙ tanh(c_t)             hidden state
 *
 * Why this beats a vanilla RNN: when f_t≈1, i_t≈0 the cell passes through
 * unchanged (∂c_t/∂c_{t-1}≈1), so the gradient rides the cell-state "conveyor
 * belt" instead of through a repeated matrix multiply that vanishes. Same
 * additive-path trick as ResNet's skip connection (README §4-§7).
 *
 * Build & run:
 *   cc -O2 -o lstm_cell lstm_cell.c -lm && ./lstm_cell
 */
#include <math.h>
#include <stdio.h>

#define NH 3   /* hidden/cell size */
#define NX 2   /* input size       */
#define NZ (NH + NX)  /* concatenated [h_{t-1}, x_t] */

static double sigmoid(double z) { return 1.0 / (1.0 + exp(-z)); }

/* gate = act(W·z + b), W is [NH][NZ]. act=1 -> sigmoid, act=0 -> tanh. */
static void gate(const double W[NH][NZ], const double b[NH], const double z[NZ],
                 int sig, double out[NH]) {
    for (int k = 0; k < NH; k++) {
        double s = b[k];
        for (int j = 0; j < NZ; j++) s += W[k][j] * z[j];
        out[k] = sig ? sigmoid(s) : tanh(s);
    }
}

int main(void) {
    /* Toy weights (normally learned). Forget-gate bias starts positive — the
     * standard init that biases the cell toward REMEMBERING early in training. */
    double Wf[NH][NZ] = {{.1,.2,.0,.1,.0},{.0,.1,.2,.0,.1},{.1,.0,.1,.2,.0}};
    double Wi[NH][NZ] = {{.2,.1,.1,.0,.1},{.1,.2,.0,.1,.0},{.0,.1,.2,.1,.1}};
    double Wg[NH][NZ] = {{.1,.0,.2,.1,.1},{.2,.1,.1,.0,.2},{.1,.1,.0,.2,.1}};
    double Wo[NH][NZ] = {{.0,.2,.1,.1,.0},{.1,.0,.2,.1,.1},{.2,.1,.1,.0,.1}};
    double bf[NH] = {1.0, 1.0, 1.0};   /* forget-gate bias = 1 */
    double bi[NH] = {0,0,0}, bg[NH] = {0,0,0}, bo[NH] = {0,0,0};

    double h[NH] = {0, 0, 0};          /* h_{t-1} */
    double c[NH] = {0, 0, 0};          /* c_{t-1} */
    double x[NX] = {0.5, -0.5};        /* x_t     */

    double z[NZ];                      /* [h_{t-1}, x_t] */
    for (int k = 0; k < NH; k++) z[k] = h[k];
    for (int k = 0; k < NX; k++) z[NH + k] = x[k];

    double f[NH], i[NH], g[NH], o[NH];
    gate(Wf, bf, z, 1, f);
    gate(Wi, bi, z, 1, i);
    gate(Wg, bg, z, 0, g);
    gate(Wo, bo, z, 1, o);

    double c_new[NH], h_new[NH];
    for (int k = 0; k < NH; k++) {
        c_new[k] = f[k] * c[k] + i[k] * g[k];     /* additive cell update */
        h_new[k] = o[k] * tanh(c_new[k]);
    }

    printf("forget gate f_t : "); for (int k=0;k<NH;k++) printf("%.3f ", f[k]); putchar('\n');
    printf("input  gate i_t : "); for (int k=0;k<NH;k++) printf("%.3f ", i[k]); putchar('\n');
    printf("cell state  c_t : "); for (int k=0;k<NH;k++) printf("%.3f ", c_new[k]); putchar('\n');
    printf("hidden      h_t : "); for (int k=0;k<NH;k++) printf("%.3f ", h_new[k]); putchar('\n');
    return 0;
}
