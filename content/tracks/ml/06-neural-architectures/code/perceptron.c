/* perceptron.c — Rosenblatt's 1958 perceptron and the XOR wall (README §3).
 *
 * One neuron, step activation, and the perceptron learning rule:
 *     prediction:  y = step(w·x + b)
 *     update:      w ← w + η (target − y) x ,   b ← b + η (target − y)
 *
 * The perceptron convergence theorem guarantees this finds a separating line
 * IF one exists. AND and OR are linearly separable, so it converges. XOR is
 * NOT linearly separable (Minsky & Papert, 1969), so it never converges — the
 * exact result that triggered the first AI winter. Fix: hidden layers +
 * backprop, see backprop_xor.c.
 *
 * Build & run:
 *   cc -O2 -o perceptron perceptron.c && ./perceptron
 */
#include <stdio.h>

#define N 4   /* training examples */
#define D 2   /* input dimension   */

static int step(double z) { return z >= 0.0 ? 1 : 0; }

/* Train one perceptron on a 2-input boolean function. Returns epochs until
 * convergence, or -1 if it didn't converge within `max_epochs`. */
static int train(const double X[N][D], const int t[N], const char *name) {
    double w[D] = {0.0, 0.0}, b = 0.0;
    const double eta = 0.1;
    const int max_epochs = 100;

    for (int epoch = 1; epoch <= max_epochs; epoch++) {
        int errors = 0;
        for (int i = 0; i < N; i++) {
            double z = b;
            for (int j = 0; j < D; j++) z += w[j] * X[i][j];
            int y = step(z);
            int err = t[i] - y;            /* −1, 0, or +1 */
            if (err != 0) {
                errors++;
                for (int j = 0; j < D; j++) w[j] += eta * err * X[i][j];
                b += eta * err;
            }
        }
        if (errors == 0) {
            printf("%-4s converged in %2d epochs  (w=[%.2f %.2f] b=%.2f)\n",
                   name, epoch, w[0], w[1], b);
            return epoch;
        }
    }
    printf("%-4s DID NOT converge in %d epochs — not linearly separable\n",
           name, max_epochs);
    return -1;
}

int main(void) {
    double X[N][D] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    int AND[N] = {0, 0, 0, 1};
    int OR[N]  = {0, 1, 1, 1};
    int XOR[N] = {0, 1, 1, 0};

    train(X, AND, "AND");
    train(X, OR,  "OR");
    train(X, XOR, "XOR");   /* the wall */
    return 0;
}
