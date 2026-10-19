/* lang.02 primer: given in the chapter; the Makefile builds it. */
#include <stdio.h>

#include "greet.h"

int greet(char *buf, size_t cap, const char *name) {
    return snprintf(buf, cap, "hello, %s", name);
}
