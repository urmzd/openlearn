/* vec.h - a growable array of int.
 *
 * The interface. vec.c implements it and main.c tests it.
 */
#ifndef VEC_H
#define VEC_H

#include <stddef.h>

typedef struct {
  int *data;
  size_t len;
  size_t cap;
} Vec;

/* Zero-capacity vector. Allocates nothing: a vector you never push to must
 * never call malloc. */
void vec_init(Vec *v);

/* Append x, growing the buffer if needed. 0 on success, -1 if allocation
 * failed (in which case v is unchanged and still valid). */
int vec_push(Vec *v, int x);

/* Remove the last element into *out. -1 if the vector is empty. */
int vec_pop(Vec *v, int *out);

/* Element at i. Reading past len is undefined behaviour, on purpose: bounds
 * checking every access is not what this type is for. */
int vec_get(const Vec *v, size_t i);

/* Ensure capacity for at least n elements without changing len. 0 on success,
 * -1 on allocation failure. */
int vec_reserve(Vec *v, size_t n);

/* Release the buffer and return v to the zero state, so calling it twice is
 * safe. */
void vec_free(Vec *v);

#endif /* VEC_H */
