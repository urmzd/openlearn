/* primers/lang.03/vec.c: the growable array (lang.03 primer exercise). */
#include "vec.h"

#include <stdint.h> /* SIZE_MAX */
#include <stdlib.h>

void vec_init(Vec *v) {
    v->data = NULL;
    v->len = 0;
    v->cap = 0;
}

int vec_reserve(Vec *v, size_t n) {
    if (n <= v->cap) return 0;
    /* Double, starting at 4, until the request fits. */
    size_t cap = v->cap ? v->cap : 4;
    while (cap < n) {
        if (cap > SIZE_MAX / 2) { /* doubling would wrap around */
            cap = n;
            break;
        }
        cap *= 2;
    }
    if (cap > SIZE_MAX / sizeof(int)) return -1; /* cap * sizeof(int) would wrap */
    /* Keep the old pointer until realloc succeeds: `v->data = realloc(...)`
     * loses (and leaks) the buffer when realloc returns NULL. */
    int *grown = realloc(v->data, cap * sizeof(int));
    if (grown == NULL) return -1;
    v->data = grown;
    v->cap = cap;
    return 0;
}

int vec_push(Vec *v, int x) {
    if (v->len == v->cap && vec_reserve(v, v->len + 1) != 0) return -1;
    v->data[v->len++] = x;
    return 0;
}

int vec_pop(Vec *v, int *out) {
    if (v->len == 0) return -1;
    *out = v->data[--v->len];
    return 0;
}

int vec_get(const Vec *v, size_t i) {
    return v->data[i];
}

void vec_free(Vec *v) {
    free(v->data); /* free(NULL) is defined to do nothing */
    v->data = NULL;
    v->len = 0;
    v->cap = 0;
}
