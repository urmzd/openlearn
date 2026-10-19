/* lang.02 primer: the two-file C program your Makefile builds (given in the chapter). */
#ifndef GREET_H
#define GREET_H

#include <stddef.h>

/* Write "hello, <name>" into buf (at most cap bytes, NUL-terminated).
   Returns the length it wanted to write, as snprintf does. */
int greet(char *buf, size_t cap, const char *name);

#endif
