/* vec.c - the part you write.
 *
 * Everything between SOLUTION-BEGIN and SOLUTION-END is stripped out when
 * `ol start build c 01` clones this into your scratchpad.
 */
#include "vec.h"

#include <stdint.h> /* SIZE_MAX */
#include <stdlib.h>

void vec_init(Vec *v) {
  /* SOLUTION-BEGIN */
  v->data = NULL;
  v->len = 0;
  v->cap = 0;
  /* SOLUTION-END */
}

int vec_reserve(Vec *v, size_t n) {
  /* SOLUTION-BEGIN */
  if (n <= v->cap) return 0;

  /* Double, but never less than the caller asked for, and start at 4 rather
   * than 1 so the first few pushes do not each cause a realloc. */
  size_t cap = v->cap ? v->cap : 4;
  while (cap < n) {
    /* Overflow check before doubling. On a 64-bit host this is unreachable in
     * practice, but "unreachable in practice" is how you get a heap overflow
     * on a 32-bit build. */
    if (cap > SIZE_MAX / 2) { cap = n; break; }
    cap *= 2;
  }
  if (cap > SIZE_MAX / sizeof(int)) return -1;

  int *grown = realloc(v->data, cap * sizeof(int));
  /* Assign only after realloc succeeds. Writing v->data = realloc(v->data,...)
   * directly leaks the old buffer on failure, which is the single most common
   * bug in hand-rolled dynamic arrays. */
  if (!grown) return -1;
  v->data = grown;
  v->cap = cap;
  return 0;
  /* SOLUTION-END */
}

int vec_push(Vec *v, int x) {
  /* SOLUTION-BEGIN */
  if (v->len == v->cap && vec_reserve(v, v->len + 1) != 0) return -1;
  v->data[v->len++] = x;
  return 0;
  /* SOLUTION-END */
}

int vec_pop(Vec *v, int *out) {
  /* SOLUTION-BEGIN */
  if (v->len == 0) return -1;
  *out = v->data[--v->len];
  return 0;
  /* SOLUTION-END */
}

int vec_get(const Vec *v, size_t i) {
  /* SOLUTION-BEGIN */
  return v->data[i];
  /* SOLUTION-END */
}

void vec_free(Vec *v) {
  /* SOLUTION-BEGIN */
  free(v->data);
  v->data = NULL;
  v->len = 0;
  v->cap = 0;
  /* SOLUTION-END */
}
