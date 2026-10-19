/* intern.h - a string interning table.
 *
 * The interface. intern.c implements it and main.c tests it.
 *
 * Interning means: the table owns exactly one copy of each distinct string,
 * and hands back the same pointer every time you ask for it. Two interned
 * strings are equal if and only if their pointers are equal, which turns
 * strcmp into a pointer comparison everywhere downstream.
 */
#ifndef INTERN_H
#define INTERN_H

#include <stddef.h>

/* An interned string. `str` is NUL-terminated and owned by the table; `len`
 * is cached because the table already knows it and callers usually want it.
 *
 * The pointer stays valid until intern_free, INCLUDING across growth of the
 * table. That is the property that makes pointer equality safe, and it is the
 * main constraint on how you implement this. */
typedef struct {
  const char *str;
  size_t len;
} Interned;

typedef struct InternTable InternTable;

/* Create an empty table. NULL on allocation failure. */
InternTable *intern_new(void);

/* Intern a NUL-terminated string. Returns a stable Interned, or NULL on
 * allocation failure. Interning the same text twice returns the identical
 * pointer. */
const Interned *intern(InternTable *t, const char *s);

/* Intern a string that may contain embedded NULs, or that is not
 * NUL-terminated in the caller's buffer. */
const Interned *intern_n(InternTable *t, const char *s, size_t len);

/* Look up without interning: NULL when the string has never been interned. */
const Interned *intern_find(const InternTable *t, const char *s);

/* How many distinct strings the table holds. */
size_t intern_count(const InternTable *t);

/* Free the table and every string it owns. Every Interned handed out is
 * dangling afterwards. */
void intern_free(InternTable *t);

#endif /* INTERN_H */
